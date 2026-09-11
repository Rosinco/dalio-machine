import { createHash } from 'node:crypto';

const clean = text => text.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '').replace(/(?:https?|wss?):\/\/[^\s"'\\]+/g, '[URL redacted]').slice(0, 1000);

export function transportDiagnostic(text) {
  const payloadStart = text.indexOf('eventData=');
  if (payloadStart >= 0 && text.includes('<closing ws>')) {
    const errorStart = text.lastIndexOf(' e=');
    const raw = text.slice(payloadStart + 'eventData='.length, errorStart > payloadStart ? errorStart : undefined);
    const error = errorStart > payloadStart ? clean(text.slice(errorStart + 3)).trim() : 'Transport exception; error suffix unavailable';
    const first = raw.slice(0, 100);
    const method = first.match(/"method"\s*:\s*"([A-Za-z][A-Za-z0-9]*\.[A-Za-z][A-Za-z0-9]*)"/)?.[1] ?? null;
    // Preserve the outer envelope's syntax, including a malformed separator,
    // while stopping at params/result/error before any content values appear.
    const content = first.match(/"(?:params|result|error)"\s*[:{,]?/);
    let prefix = content ? first.slice(0, content.index + content[0].length) : first;
    prefix = clean(prefix).replace(/"((?:\\.|[^"\\])*)"/g, (token, value, offset, source) => ['method', 'id', 'sessionId', 'params', 'result', 'error'].includes(value) || value === method && /"method"\s*:\s*$/.test(source.slice(0, offset)) || /^[a-fA-F0-9-]{8,64}$/.test(value) && /"sessionId"\s*:\s*$/.test(source.slice(0, offset)) ? token : '"[redacted]"');
    // An unfinished string at the boundary could contain a URL or content.
    if ((prefix.match(/(?<!\\)"/g)?.length ?? 0) % 2) prefix = prefix.slice(0, prefix.lastIndexOf('"')) + '"[truncated]"';
    const position = error.match(/position (\d+)/);
    const detail = { stage: 'transport_payload_redacted', method, envelope_prefix: prefix + (content ? '[content redacted]' : '[truncated at 100 characters]'), payload_characters: raw.length, payload_sha256: createHash('sha256').update(raw).digest('hex'), parse_error_position: position ? Number(position[1]) : null, reason: error };
    return { detail, text: `${text.slice(0, payloadStart)}eventData=[transport payload redacted] e=${error}\n` };
  }
  if (text.includes('<ws disconnected>')) {
    const match = clean(text).match(/code=(\d+) reason=(.*)/);
    return { detail: { stage: 'transport_websocket_closed', code: match ? Number(match[1]) : null, reason: match?.[2]?.trim() ?? 'Not supplied' }, text: clean(text) + '\n' };
  }
  if (text.includes('<ws error>')) return { detail: { stage: 'transport_websocket_error', reason: clean(text).trim() }, text: clean(text) + '\n' };
  return null;
}
