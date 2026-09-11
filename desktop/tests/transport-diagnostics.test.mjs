import { expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { transportDiagnostic } from './transport-diagnostics.mjs';

it('identifies malformed outer method and separator without retaining bulk parameters', () => {
  const raw = '{"method":"Network.requestWillBeSent","params" {"url":"https://private.example/secret","postData":"PRIVATE BODY"}}';
  const result = transportDiagnostic(`pw:browser <closing ws> Closing websocket due to malformed JSON. eventData=${raw} e=Expected ':' after property name in JSON at position 45`);
  expect(result.detail.method).toBe('Network.requestWillBeSent');
  expect(result.detail.envelope_prefix).toBe('{"method":"Network.requestWillBeSent","params" {[content redacted]');
  expect(result.detail.parse_error_position).toBe(45);
  expect(result.detail.payload_characters).toBe(raw.length);
  expect(result.detail.payload_sha256).toBe(createHash('sha256').update(raw).digest('hex'));
  expect(JSON.stringify(result)).not.toMatch(/PRIVATE BODY|private.example|secret/);
});

it('retains response metadata while redacting result and URL-bearing unknown values', () => {
  const result = transportDiagnostic('pw:browser <closing ws> failed onmessage callback. eventData={"id":15,"result":{"payload":"SECRET FINANCIAL DATA"}} e=callback failed');
  expect(result.detail.envelope_prefix).toBe('{"id":15,"result":[content redacted]');
  expect(result.detail.method).toBeNull();
  expect(JSON.stringify(result)).not.toContain('SECRET');
  const unknown = transportDiagnostic('pw:browser <closing ws> malformed JSON. eventData={"unknown":"https://private.example/secret","weird":"PERSONAL DATA"} e=bad JSON');
  expect(JSON.stringify(unknown)).not.toMatch(/private.example|PERSONAL DATA/);
});

it('captures safe WebSocket close/error reasons and ignores unrelated stderr', () => {
  expect(transportDiagnostic('pw:browser <ws disconnected> ws://127.0.0.1:123/test code=1006 reason=')).toMatchObject({ detail: { stage: 'transport_websocket_closed', code: 1006, reason: '' } });
  expect(transportDiagnostic('pw:browser <ws error> ws://127.0.0.1:123/test error Max payload size exceeded').detail.reason).toContain('Max payload size exceeded');
  expect(transportDiagnostic('ordinary test output')).toBeNull();
});
