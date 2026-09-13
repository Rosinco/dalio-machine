#!/usr/bin/env node
/** Bind a tested 0.17 installation to its exact source and offline evidence. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = 'test-results/research-gauge-release-verification-0.17.0.json';
const read = path => readFileSync(resolve(root, path));
const json = path => JSON.parse(read(path).toString('utf8').replace(/^\uFEFF/, ''));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const identity = path => ({ path, bytes: statSync(resolve(root, path)).size, sha256: sha(read(path)) });
function tree(path) {
  return readdirSync(resolve(root, path), { withFileTypes: true }).flatMap(entry => {
    if (entry.name === '__pycache__') return [];
    assert.ok(!entry.isSymbolicLink(), 'Release sources cannot use unbound symlinks');
    const next = `${path}/${entry.name}`;
    return entry.isDirectory() ? tree(next) : entry.isFile() ? [next] : [];
  }).sort();
}
function passingReport(path, native, expected) {
  const r = json(path);
  assert.ok(Array.isArray(r.checks) && (expected ? r.checks.length === expected : r.checks.length >= 10), `${path}: missing checks`);
  assert.equal(new Set(r.checks).size, r.checks.length);
  assert.ok(r.checks.every(c => typeof c === 'string' && c.trim()));
  assert.deepEqual(r[native ? 'externalRequests' : 'external'], []);
  assert.deepEqual(r[native ? 'runtimeErrors' : 'errors'], []);
  if (native) assert.equal(r.status, 'PASS');
  return r;
}

const manifest = json('src/data/research-gauge-manifest.json');
assert.equal(manifest.artifact.path, 'data/research-gauge/gauge.bin');
const compressed = read(`public/${manifest.artifact.path}`), raw = gunzipSync(compressed);
assert.equal(sha(compressed), manifest.artifact.sha256); assert.equal(compressed.length, manifest.artifact.bytes);
assert.equal(sha(raw), manifest.artifact.uncompressedSha256); assert.equal(raw.length, manifest.artifact.uncompressedBytes);
assert.equal(JSON.parse(raw).rows.length, 19140);
assert.equal(sha(read(`dist/${manifest.artifact.path}`)), manifest.artifact.sha256);
const auditPath = 'test-results/research-gauge-export-2026-09-13/receipt.json', audit = json(auditPath);
assert.equal(audit.status, 'passed'); assert.deepEqual(audit.manifest, manifest);
for (const input of audit.inputs) assert.deepEqual(identity(input.path), input, `Stale export source: ${input.path}`);
for (const key of ['listings', 'independentMid', 'independentLow', 'frozenMidReconciliations', 'nonmutatedHistories']) assert.equal(audit.counts[key], 19140);
assert.equal(audit.counts.reverseEquations, 13882);
assert.ok(audit.priorLedger); assert.deepEqual(identity(audit.priorLedger.path), audit.priorLedger);

const browser = passingReport('test-results/research-gauge-browser-report.json', false, 12);
const native = passingReport('test-results/windows-native-research-gauge-report.json', true, 12);
const valuation = passingReport('test-results/valuation-browser-report.json', false, 55);
const nativeValuation = passingReport('test-results/windows-native-valuation-report.json', true, 58);
for (const r of [browser, native]) {
  assert.equal(r.listings, 19140); assert.equal(r.reverseChecks, 13882);
  assert.equal(r.pack, manifest.financialPackId); assert.equal(r.taxonomy, manifest.taxonomySha256);
  assert.equal(r.artifactSha256, manifest.artifact.sha256);
}
const unitLog = read('test-results/unit-tests-0.17.0.log').toString();
assert.match(unitLog, /Test Files\s+28 passed \(28\)/); assert.match(unitLog, /Tests\s+236 passed \(236\)/);
assert.ok(!/\bFAIL\b/.test(unitLog));
assert.match(read('test-results/frontend-build-0.17.0.log').toString(), /built in/);
assert.match(read('test-results/windows-build-0.17.0.log').toString(), /Finished `release` profile/);
for (const path of ['package.json', 'src-tauri/tauri.conf.json']) assert.equal(json(path).version, '0.17.0');
assert.match(read('src-tauri/Cargo.toml').toString(), /version = "0\.17\.0"/);
const executable = identity('src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
for (const r of [native, nativeValuation]) assert.equal(r.executableSha256, executable.sha256);
const installation = json('test-results/windows-installation-0.17.0.json');
assert.equal(installation.status, 'passed'); assert.equal(installation.version, '0.17.0');
assert.equal(installation.sha256, executable.sha256); assert.equal(installation.previousVersionRetained, '0.16.0');
assert.equal(installation.nativeChecks, native.checks.length);
assert.ok(installation.processIds.length > 0 && installation.documents.includes('decisions\\0040-universe-research-screen.md'));
const target = /^([A-Za-z]):\\(.*)$/.exec(installation.executable);
assert.ok(target && !target[2].split('\\').includes('..'));
assert.equal(sha(readFileSync(`/mnt/${target[1].toLowerCase()}/${target[2].replaceAll('\\', '/')}`)), executable.sha256);

const files = [...new Set([
  ...tree('src'), ...tree('tests'), ...tree('scripts'),
  'package.json', 'package-lock.json', 'src-tauri/Cargo.toml', 'src-tauri/Cargo.lock', 'src-tauri/tauri.conf.json',
  'README.md', 'HANDOFF.md', '../README.md', '../decisions/0040-universe-research-screen.md', '../decisions/README.md', '../CLAUDE.md', '../project_context.md',
  `public/${manifest.artifact.path}`, ...tree('dist'), executable.path, auditPath,
  'test-results/unit-tests-0.17.0.log', 'test-results/frontend-build-0.17.0.log', 'test-results/windows-build-0.17.0.log',
  'test-results/research-gauge-browser-report.json', 'test-results/windows-native-research-gauge-report.json',
  'test-results/valuation-browser-report.json', 'test-results/windows-native-valuation-report.json', 'test-results/windows-installation-0.17.0.json',
  'test-results/research-gauge-browser-overview-timing.json', 'test-results/research-gauge-browser-final-0.17.0.log',
  'test-results/windows-research-gauge-test-0.17.0.log', 'test-results/windows-valuation-test-0.17.0.log',
  'test-results/verify-windows-installation-0.17.0.ps1',
])].sort().map(identity);
const result = { status: 'passed', version: '0.17.0', counts: { listings: 19140, unit: 236, researchBrowser: browser.checks.length, researchNative: native.checks.length, valuationBrowser: valuation.checks.length, valuationNative: nativeValuation.checks.length }, manifest, executable, installation, files };
assert.ok(process.argv.slice(2).every(arg => arg === '--check'), 'Supported argument: --check');
if (process.argv.includes('--check')) assert.deepEqual(json(output), result, 'Release evidence changed after verification.');
else writeFileSync(resolve(root, output), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ status: 'passed', receipt: output, sha256: sha(read(output)), counts: result.counts }));
