import { afterEach, expect, it } from 'vitest';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { WebSocketServer } from 'ws';
import { uncompressedCdp } from './uncompressed-cdp.mjs';

const cleanup = [];
afterEach(async () => { for (const finish of cleanup.splice(0).reverse()) await finish(); });

async function endpoint() {
  const server = createServer((request, response) => {
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ webSocketDebuggerUrl: `ws://127.0.0.1:${server.address().port}/devtools/browser/test` }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const ws = new WebSocketServer({ server, perMessageDeflate: true });
  const connected = once(ws, 'connection');
  cleanup.push(async () => {
    for (const socket of ws.clients) socket.terminate();
    server.closeAllConnections();
    await Promise.all([new Promise(resolve => ws.close(resolve)), new Promise(resolve => server.close(resolve))]);
  });
  return { url: `http://127.0.0.1:${server.address().port}`, connected, ws };
}

it('declines offered compression and passes complete large CDP objects in both directions, in order', async () => {
  const fixture = await endpoint(), events = [];
  const transport = await uncompressedCdp(fixture.url, event => events.push(event));
  const [peer, request] = await fixture.connected;
  expect(request.headers['sec-websocket-extensions']).toBeUndefined();
  expect(peer.extensions).toBe('');
  expect(events).toContainEqual({ stage: 'transport_websocket_opened', transport: 'ws-public-custom', compressionRequested: false, negotiatedExtensions: '' });
  const outgoing = { id: 15, method: 'Runtime.evaluate', params: { expression: '1 + 1' } };
  const requestReceived = once(peer, 'message');
  transport.send(outgoing);
  expect(JSON.parse((await requestReceived)[0].toString())).toEqual(outgoing);
  const incoming = [{ id: 15, result: { value: 'å'.repeat(2400000) } }, { method: 'Network.loadingFinished', params: { requestId: 'fixture' } }];
  const delivered = [];
  const all = new Promise(resolve => { transport.onmessage = message => { delivered.push(message); if (delivered.length === incoming.length) resolve(); }; });
  for (const message of incoming) peer.send(JSON.stringify(message));
  await all;
  expect(delivered).toEqual(incoming);
  const closed = new Promise(resolve => { transport.onclose = resolve; });
  transport.close();
  expect(await closed).toBe('WebSocket closed (1000)');
  expect(events.filter(event => event.stage === 'transport_websocket_closed')).toHaveLength(1);
});

it('retains early events until Playwright attaches handlers and reports peer closure once', async () => {
  const fixture = await endpoint(), events = [];
  fixture.ws.on('connection', peer => peer.send(JSON.stringify({ method: 'Target.targetCreated', params: { targetId: 'early' } })));
  const transport = await uncompressedCdp(fixture.url, event => events.push(event));
  const [peer] = await fixture.connected;
  await new Promise(resolve => setTimeout(resolve, 20));
  const received = new Promise(resolve => { transport.onmessage = resolve; });
  expect(await received).toEqual({ method: 'Target.targetCreated', params: { targetId: 'early' } });
  let count = 0;
  const closed = new Promise(resolve => { transport.onclose = reason => { count++; resolve(reason); }; });
  peer.close(1001, 'fixture complete');
  expect(await closed).toBe('WebSocket closed (1001)');
  transport.close();
  expect(count).toBe(1);
  expect(() => transport.send({ id: 16, method: 'Browser.getVersion' })).toThrow('not open');
  expect(events.at(-1)).toEqual({ stage: 'transport_websocket_closed', code: 1001, reason: 'fixture complete' });
});

it('closes on malformed JSON without exposing request bodies or parser snippets', async () => {
  const fixture = await endpoint(), events = [];
  const transport = await uncompressedCdp(fixture.url, event => events.push(event));
  const [peer] = await fixture.connected;
  const received = [];
  transport.onmessage = message => received.push(message);
  const closed = new Promise(resolve => { transport.onclose = resolve; });
  const raw = '{"method":"Network.requestWillBeSent","params" {"postData":"SECRET BODY"}}';
  peer.send(raw);
  expect(await closed).toBe('WebSocket closed (1002)');
  expect(received).toEqual([]);
  expect(events.find(event => event.stage === 'transport_payload_redacted')).toMatchObject({ method: 'Network.requestWillBeSent', payload_characters: raw.length });
  expect(JSON.stringify(events)).not.toContain('SECRET BODY');
  expect(JSON.stringify(events)).not.toContain('postData');
});

it.each(['binary', 'null', 'async rejection'])('notifies closure for %s without an unhandled exception', async kind => {
  const fixture = await endpoint(), events = [];
  const transport = await uncompressedCdp(fixture.url, event => events.push(event));
  const [peer] = await fixture.connected;
  transport.onmessage = async () => { throw new Error('PRIVATE callback contents'); };
  // Deliberately attach onclose after the socket has already closed.
  const peerClosed = once(peer, 'close');
  peer.send(kind === 'binary' ? Buffer.from('PRIVATE binary contents') : kind === 'null' ? 'null' : '{"id":1,"result":{}}');
  await peerClosed;
  await new Promise(resolve => setImmediate(resolve));
  expect(await new Promise(resolve => { transport.onclose = resolve; })).toBe('WebSocket closed (1002)');
  expect(JSON.stringify(events)).not.toContain('PRIVATE');
});

it('rejects discovery endpoints outside the isolated loopback before connecting', async () => {
  await expect(uncompressedCdp('https://example.invalid:99')).rejects.toThrow('isolated loopback');
  const server = createServer((request, response) => response.end(JSON.stringify({ webSocketDebuggerUrl: 'ws://example.invalid:99/devtools/browser/other' })));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  cleanup.push(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  await expect(uncompressedCdp(`http://127.0.0.1:${server.address().port}`)).rejects.toThrow('isolated loopback browser');
});
