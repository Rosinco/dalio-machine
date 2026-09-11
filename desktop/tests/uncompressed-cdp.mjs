import WebSocket from 'ws';
import { transportDiagnostic } from './transport-diagnostics.mjs';

const limit = 256 * 1024 * 1024;

// Public Playwright ConnectOverCDPTransport, with a separate WebSocket client so
// the test can explicitly decline compression. No Playwright internals or app
// bridge hooks are used. The stock compressed client remains an explicit native
// runner option for reproducing the earlier malformed-protocol failures.
export async function uncompressedCdp(endpoint, diagnostic = () => {}, timeout = 1500) {
  const base = new URL(endpoint);
  if (base.protocol !== 'http:' || base.hostname !== '127.0.0.1' || !base.port || base.username || base.password) throw new Error('CDP discovery must use the isolated loopback endpoint');
  const response = await fetch(new URL('/json/version', base), { redirect: 'error', signal: AbortSignal.timeout(timeout) });
  if (!response.ok) throw new Error(`CDP discovery HTTP ${response.status}`);
  const info = await response.json();
  const address = new URL(info.webSocketDebuggerUrl);
  if (address.protocol !== 'ws:' || address.hostname !== base.hostname || address.port !== base.port || address.username || address.password || !address.pathname.startsWith('/devtools/browser/') || address.search || address.hash) throw new Error('CDP WebSocket must belong to the isolated loopback browser');

  const socket = new WebSocket(address, [], { perMessageDeflate: false, maxPayload: limit, handshakeTimeout: timeout, followRedirects: false });
  let receive, disconnected, closed = false, closeReason, queuedBytes = 0;
  const pending = [];
  const reportFailure = (raw, error, kind) => {
    // JSON.parse errors can quote input contents in newer runtimes. Retain only
    // their class and numeric offset, plus the existing bounded envelope digest.
    const position = String(error?.message).match(/position (\d+)/)?.[1];
    const reason = `${kind}: ${error?.name ?? 'Error'}${position ? ` at position ${position}` : ''}`;
    diagnostic(transportDiagnostic(`<closing ws> eventData=${raw} e=${reason}`).detail);
    socket.close(1002, 'Invalid CDP message');
  };
  const deliver = (raw, binary) => {
    if (closed || !receive) return;
    if (binary) {
      diagnostic({ stage: 'transport_websocket_error', reason: 'Unexpected binary CDP message' });
      socket.close(1002, 'Expected text CDP');
      return;
    }
    let message;
    try { message = JSON.parse(raw); }
    catch (error) { reportFailure(raw, error, 'Malformed JSON'); return; }
    if (!message || typeof message !== 'object' || Array.isArray(message)) {
      reportFailure(raw, new TypeError('CDP envelope must be an object'), 'Invalid CDP envelope');
      return;
    }
    try { Promise.resolve(receive(message)).catch(error => reportFailure(raw, error, 'CDP callback failed')); }
    catch (error) { reportFailure(raw, error, 'CDP callback failed'); }
  };
  socket.on('message', (data, binary) => {
    const raw = data.toString('utf8');
    if (!receive) {
      queuedBytes += Buffer.byteLength(raw);
      if (queuedBytes > limit) {
        diagnostic({ stage: 'transport_websocket_error', reason: 'CDP startup queue exceeded 256 MiB' });
        socket.close(1009, 'Startup queue limit');
      } else pending.push([raw, binary]);
      return;
    }
    // Match Playwright's normal transport scheduling while retaining ordering.
    setImmediate(() => deliver(raw, binary));
  });
  socket.on('error', error => {
    const detail = transportDiagnostic(`<ws error> ${error.message}`).detail;
    diagnostic(detail);
  });
  socket.on('close', (code, reason) => {
    closed = true;
    pending.length = 0;
    queuedBytes = 0;
    closeReason = `WebSocket closed (${code})`;
    diagnostic(transportDiagnostic(`<ws disconnected> code=${code} reason=${reason.toString('utf8')}`).detail);
    disconnected?.(closeReason);
  });
  const transport = {
    get onmessage() { return receive; },
    set onmessage(callback) {
      receive = callback;
      if (callback) {
        for (const [raw, binary] of pending.splice(0)) setImmediate(() => deliver(raw, binary));
        queuedBytes = 0;
      }
    },
    get onclose() { return disconnected; },
    set onclose(callback) { disconnected = callback; if (closed) callback?.(closeReason); },
    send(message) {
      if (socket.readyState !== WebSocket.OPEN) throw new Error('CDP WebSocket is not open');
      socket.send(JSON.stringify(message));
    },
    close() { socket.close(1000, 'Test connection closed'); },
  };
  await new Promise((resolve, reject) => {
    socket.once('open', () => {
      if (socket.extensions !== '') {
        socket.close(1002, 'Unexpected compression');
        reject(new Error('Uncompressed CDP experiment negotiated an extension'));
        return;
      }
      diagnostic({ stage: 'transport_websocket_opened', transport: 'ws-public-custom', compressionRequested: false, negotiatedExtensions: socket.extensions });
      resolve();
    });
    socket.once('error', reject);
    socket.once('close', () => reject(new Error('CDP WebSocket closed before connection completed')));
  });
  return transport;
}
