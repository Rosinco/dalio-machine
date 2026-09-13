#!/usr/bin/env node
// Local, release-specific closeout. Run only after the 0.16.0 installation verifier.
// node scripts/verify-purchase-release.mjs [--check] [--replace]
// --check performs all gates without writing; --replace explicitly refreshes a receipt.
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { gunzipSync } from 'node:zlib';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const version = '0.16.0';
const expected = Object.freeze({ unit: 217, unitFiles: 26, browser: 55, native: 58, purchase: 10, universe: 19140 });
const outputPath = 'test-results/purchase-release-verification-0.16.0.json';
const evidencePaths = Object.freeze({
  unit: 'test-results/unit-tests-0.16.0.log',
  frontendBuild: 'test-results/frontend-build-0.16.0.log',
  nativeBuild: 'test-results/windows-build-0.16.0.log',
  browser: 'test-results/valuation-browser-report.json',
  browserLog: 'test-results/valuation-browser-test-0.16.0.log',
  purchase: 'test-results/purchase-range-browser-report.json',
  purchaseLog: 'test-results/purchase-range-browser.log',
  native: 'test-results/windows-native-valuation-report.json',
  nativeLog: 'test-results/windows-valuation-test-0.16.0.log',
  installation: 'test-results/windows-installation-0.16.0.json',
  installationFinalizationLog: 'test-results/windows-installation-finalization-0.16.0.log',
  audit: 'test-results/purchase-range-2026-09-13/universe-audit.json',
});
const packNames = [
  'a0c53dad1a0a726ee955d1e3a47f1d011b272b610e22652cef30e1bdc1c6d005.sqlite',
  '1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a.sqlite',
];
const installationScripts = [
  'test-results/verify-windows-installation-0.16.0.ps1',
  'test-results/finalize-purchase-installation-0.16.0.ps1',
];
const installedDocuments = [
  'README.md', 'HANDOFF.md', 'docs/standard-company-valuation.md',
  'docs/company-valuation-framework.md', 'decisions/0037-empirical-company-cash-flow-starters.md',
  'docs/cash-flow-universe-setup-proposal-2026-09-12.md',
  'decisions/0038-separate-sustainable-terminal-cash.md', 'docs/cash-component-audit-2026-09-13.md',
  'docs/cash-flow-midline-challengers-2026-09-13.md', 'decisions/0039-editable-purchase-price-range.md',
];
const text = path => readFileSync(resolve(project, path), 'utf8').replace(/^\uFEFF/, '');
const json = path => JSON.parse(text(path));
const stamp = path => statSync(resolve(project, path)).mtimeMs;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const normalizedPath = path => path.replaceAll('\\', '/');

function filesIn(path) {
  return readdirSync(resolve(project, path), { withFileTypes: true }).flatMap(entry => {
    if (entry.name === '__pycache__' || entry.name === '.DS_Store') return [];
    const child = `${path}/${entry.name}`;
    assert.ok(!entry.isSymbolicLink(), `Unbound symlink: ${child}`);
    return entry.isDirectory() ? filesIn(child) : entry.isFile() ? [child] : [];
  }).sort();
}

function identity(path) {
  const absolute = resolve(project, path), before = statSync(absolute);
  assert.ok(before.isFile(), `Expected a file: ${path}`);
  const bytes = readFileSync(absolute), after = statSync(absolute);
  assert.equal(after.size, before.size, `File changed while hashing: ${path}`);
  assert.equal(after.mtimeMs, before.mtimeMs, `File changed while hashing: ${path}`);
  return { path: normalizedPath(relative(project, absolute)), sha256: sha256(bytes), bytes: bytes.length };
}

function auditIdentity(entry) {
  assert.ok(entry && typeof entry.path === 'string' && !isAbsolute(entry.path), 'Audit identity needs a relative path');
  assert.ok(!normalizedPath(entry.path).split('/').includes('..'), `Audit path escapes desktop: ${entry.path}`);
  assert.match(entry.sha256, /^[a-f0-9]{64}$/);
  const actual = identity(entry.path);
  assert.equal(actual.sha256, entry.sha256, `Stale or altered audit input: ${entry.path}; rerun the universe audit`);
  assert.equal(actual.bytes, entry.bytes, `Audit input size differs: ${entry.path}`);
  return actual;
}

function freshAfter(evidence, inputs) {
  const time = stamp(evidence);
  for (const input of inputs) {
    assert.ok(stamp(input) <= time, `Stale evidence: ${evidence} predates ${input}`);
  }
}

function cleanReport(report, count, native = false) {
  assert.ok(Array.isArray(report.checks) && report.checks.length === count, `Expected ${count} passing checks`);
  assert.equal(new Set(report.checks).size, count, 'Duplicate check descriptions cannot count as separate evidence');
  assert.ok(report.checks.every(check => typeof check === 'string' && check.trim()), 'Empty check description');
  assert.deepEqual(report[native ? 'externalRequests' : 'external'], [], 'External requests were recorded or omitted');
  assert.deepEqual(report[native ? 'runtimeErrors' : 'errors'], [], 'Runtime errors were recorded or omitted');
}

function sameChecksInLog(path, checks) {
  const log = text(path);
  assert.ok(!/^Browser:|\b(?:AssertionError|TimeoutError|Unhandled Rejection|Uncaught Exception)\b/m.test(log), `Browser run log contains an error: ${path}`);
  const candidates = log.split(/\r?\n/).flatMap(line => {
    try { const value = JSON.parse(line); return Array.isArray(value.checks) ? [value] : []; } catch { return []; }
  });
  assert.equal(candidates.length, 1, `Expected exactly one completed check result in ${path}`);
  assert.deepEqual(candidates[0].checks, checks, `Report and run log disagree: ${path}`);
}

function localWindowsPath(path) {
  assert.equal(typeof path, 'string');
  const match = /^([A-Za-z]):\\(.*)$/.exec(path);
  assert.ok(match, `Expected a Windows drive path: ${path}`);
  assert.ok(!match[2].split('\\').includes('..'), 'Windows evidence path escapes its folder');
  return process.platform === 'win32' ? path : `/mnt/${match[1].toLowerCase()}/${normalizedPath(match[2])}`;
}

export async function verifyPurchaseRelease() {
  // JSON receipts have no original source manifest. Freshness uses local modification
  // times; this writer then binds current bytes. It does not invent a retroactive
  // cryptographic source identity for the browser or unit test runs.
  const packageJson = json('package.json'), lock = json('package-lock.json');
  assert.equal(packageJson.version, version);
  assert.equal(lock.version, version);
  assert.equal(lock.packages[''].version, version);
  assert.equal(json('src-tauri/tauri.conf.json').version, version);
  assert.match(text('src-tauri/Cargo.toml'), /\[package\][\s\S]*?name = "macro-atlas"\s+version = "0\.16\.0"/);
  assert.match(text('src-tauri/Cargo.lock'), /\[\[package\]\]\s+name = "macro-atlas"\s+version = "0\.16\.0"/);

  const configs = ['package.json', 'package-lock.json', 'index.html', 'tsconfig.json', 'vite.config.ts', 'vitest.config.ts'];
  const appSources = filesIn('src');
  const tests = filesIn('tests');
  const browserTests = tests.filter(path => path.endsWith('.mjs') && !path.endsWith('.test.mjs'));
  const unitInputs = [...appSources, ...tests.filter(path => path.endsWith('.test.mjs') || path.includes('/fixtures/')), ...configs];
  const nativeSources = ['src-tauri/Cargo.toml', 'src-tauri/Cargo.lock', 'src-tauri/build.rs', 'src-tauri/tauri.conf.json',
    ...filesIn('src-tauri/src'), ...filesIn('src-tauri/capabilities'), ...filesIn('src-tauri/icons')];
  const sourcePaths = [...new Set([...appSources, ...tests, ...filesIn('scripts'), ...nativeSources, ...configs])].sort();
  const frontendPaths = filesIn('dist'), publicPaths = filesIn('public');
  assert.ok(frontendPaths.includes('dist/index.html') && frontendPaths.some(path => /^dist\/assets\/index-.*\.js$/.test(path)), 'Current frontend build is missing');
  const sourceFiles = sourcePaths.map(identity), frontendFiles = frontendPaths.map(identity), publicFiles = publicPaths.map(identity);

  const unitLog = text(evidencePaths.unit).replace(/\u001b\[[0-9;]*m/g, '');
  assert.match(unitLog, /^> macro-atlas@0\.16\.0 test$/m);
  assert.match(unitLog, /^\s*Test Files\s+26 passed \(26\)\s*$/m);
  assert.match(unitLog, /^\s*Tests\s+217 passed \(217\)\s*$/m);
  assert.equal((unitLog.match(/^\s*Tests\s+/gm) ?? []).length, 1, 'Ambiguous multiple unit runs');
  assert.ok(!/\b(?:FAIL|Unhandled Errors?|Unhandled Rejection|Uncaught Exception)\b|\d+ failed|\d+ skipped|\d+ todo/i.test(unitLog), 'Unit log contains failures, skips or runtime errors');
  freshAfter(evidencePaths.unit, unitInputs);

  const frontendLog = text(evidencePaths.frontendBuild);
  assert.match(frontendLog, /^> macro-atlas@0\.16\.0 build$/m);
  assert.match(frontendLog, /✓ built in /);
  for (const path of frontendPaths.filter(path => path.startsWith('dist/assets/'))) {
    assert.ok(frontendLog.includes(path), `Current frontend asset missing from successful build log: ${path}`);
  }
  freshAfter('dist/index.html', [...appSources.filter(path => !path.includes('.test.')), ...configs.filter(path => path !== 'vitest.config.ts'), ...publicPaths]);
  freshAfter(evidencePaths.frontendBuild, frontendPaths);

  const browser = json(evidencePaths.browser), purchase = json(evidencePaths.purchase), native = json(evidencePaths.native);
  cleanReport(browser, expected.browser);
  cleanReport(purchase, expected.purchase);
  assert.equal(purchase.status, 'passed');
  cleanReport(native, expected.native, true);
  assert.equal(native.status, 'PASS');
  assert.equal(native.scope, 'valuation-only');
  assert.deepEqual(native.checks.slice(0, expected.browser), browser.checks, 'Native and integrated browser scenarios differ');
  assert.deepEqual(browser.checks.slice(-expected.purchase), purchase.checks, 'Integrated and standalone purchase scenarios differ');
  for (const [receipt, log] of [[evidencePaths.browser, evidencePaths.browserLog], [evidencePaths.purchase, evidencePaths.purchaseLog]]) {
    freshAfter(receipt, [...browserTests, ...frontendPaths, ...appSources, ...configs]);
    freshAfter(log, [receipt]);
    sameChecksInLog(log, json(receipt).checks);
  }

  const executable = identity('src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
  assert.equal(executable.sha256, native.executableSha256, 'Current executable differs from native-tested executable');
  assert.match(text(evidencePaths.nativeBuild), /Compiling macro-atlas v0\.16\.0/);
  assert.match(text(evidencePaths.nativeBuild), /Finished `release` profile/);
  freshAfter(executable.path, [...frontendPaths, ...nativeSources]);
  freshAfter(evidencePaths.nativeBuild, [executable.path]);
  freshAfter(evidencePaths.native, [executable.path, ...browserTests, 'scripts/test-windows.ps1']);
  freshAfter(evidencePaths.nativeLog, [evidencePaths.native]);
  const diagnosticsFiles = readdirSync(resolve(project, 'test-results')).filter(name => /^windows-native-diagnostics-[^/]+\.json$/.test(name)).map(name => `test-results/${name}`);
  const diagnostics = diagnosticsFiles.map(path => ({ path, value: json(path) }))
    .filter(item => item.value.scope === 'valuation-only').sort((a, b) => stamp(b.path) - stamp(a.path))[0];
  assert.ok(diagnostics, 'Native transport diagnostics missing');
  assert.equal(diagnostics.value.status, 'PASS', 'Latest native valuation run failed');
  assert.equal(diagnostics.value.executableSha256, executable.sha256);
  assert.deepEqual(diagnostics.value.externalRequests, []);
  assert.deepEqual(diagnostics.value.runtimeErrors, []);
  assert.ok(!diagnostics.value.failure, 'Native diagnostics include a failure');
  const eventsPath = `test-results/windows-native-events-${diagnostics.value.runId}.jsonl`;
  const events = text(eventsPath).trim().split(/\r?\n/).map(line => JSON.parse(line));
  assert.deepEqual(events, diagnostics.value.events, 'Native event ledger and diagnostics differ');
  assert.ok(!events.some(event => ['test_failure', 'page_error'].includes(event.stage)), 'Native runtime event failure');
  assert.ok(text(evidencePaths.nativeLog).includes(`windows-native-diagnostics-${diagnostics.value.runId}.json`), 'Native log is from a different diagnostics run');
  assert.ok(text(evidencePaths.nativeLog).includes(executable.sha256), 'Native log omits tested executable hash');
  freshAfter(diagnostics.path, [evidencePaths.native]);

  const installation = json(evidencePaths.installation);
  assert.deepEqual(json(evidencePaths.installationFinalizationLog), installation, 'Installation finalization log and current receipt differ');
  freshAfter(evidencePaths.installationFinalizationLog, [evidencePaths.installation]);
  assert.equal(installation.status, 'passed');
  assert.equal(installation.version, version);
  assert.match(installation.fileVersion, /^0\.16\.0(?:\.0)?$/);
  assert.match(installation.productVersion, /^0\.16\.0(?:\.0)?$/);
  assert.equal(installation.sha256, executable.sha256);
  assert.equal(installation.nativeScope, 'valuation-only');
  assert.equal(installation.nativeChecks, expected.native);
  assert.equal(installation.nativeReceipt, 'windows-native-valuation-report.json');
  assert.equal(installation.shortcutTarget, installation.executable);
  assert.match(installation.executable, /\\MacroAtlas\\0\.16\.0\\Macro Atlas\.exe$/);
  assert.equal(installation.workingDirectory, installation.executable.slice(0, -'\\Macro Atlas.exe'.length));
  assert.ok(Array.isArray(installation.processIds) && installation.processIds.length > 0 && installation.processIds.every(Number.isInteger), 'Missing installed process evidence');
  assert.ok(Number.isFinite(Date.parse(installation.verifiedAt)) && Date.parse(installation.verifiedAt) <= Date.now() + 1000, 'Invalid installation verification date');
  assert.ok(Date.parse(installation.verifiedAt) >= stamp(evidencePaths.native), 'Installation verification predates native test');
  const installationFolder = localWindowsPath(installation.workingDirectory);
  const installedExecutable = identity(localWindowsPath(installation.executable));
  assert.equal(installedExecutable.sha256, executable.sha256, 'Installed executable changed after verification');
  const shortcut = identity(localWindowsPath(installation.shortcut));
  assert.equal(installation.previousVersionRetained, '0.15.0');
  assert.ok(existsSync(resolve(installationFolder, '../0.15.0/Macro Atlas.exe')), 'Previous installed version missing');
  assert.deepEqual(installation.financialPacks.map(pack => pack.name).sort(), [...packNames].sort());
  const financialPacks = packNames.map(name => {
    const source = identity(`financial-data/${name}`), installed = identity(resolve(installationFolder, 'financial-data', name));
    assert.equal(source.sha256, name.slice(0, -'.sqlite'.length));
    assert.equal(installed.sha256, source.sha256, `Installed source pack differs: ${name}`);
    assert.equal(installation.financialPacks.find(pack => pack.name === name).sha256, source.sha256);
    return { source, installed };
  });
  assert.deepEqual(installation.documents.map(normalizedPath).sort(), [...installedDocuments].sort());
  const installedDocumentFiles = installedDocuments.map(path => {
    const sourcePath = /^(docs|decisions)\//.test(path) ? `../${path}` : path;
    const source = identity(sourcePath), installed = identity(resolve(installationFolder, path));
    if (path === 'README.md') {
      assert.equal(text(installed.path), text(sourcePath).replaceAll('(../docs/', '(docs/').replaceAll('(../decisions/', '(decisions/'), 'Installed README content/links differ');
    } else assert.equal(installed.sha256, source.sha256, `Installed document differs: ${path}`);
    return { source, installed };
  });
  freshAfter(evidencePaths.installation, [evidencePaths.native, ...installationScripts, ...installedDocumentFiles.map(item => item.source.path)]);

  const audit = json(evidencePaths.audit), counts = audit.counts;
  assert.equal(audit.status, 'passed');
  assert.equal(audit.modelAsOf, '2026-09-13');
  assert.equal(audit.policy.marginOfSafetyPercent, 30);
  assert.equal(audit.policy.referenceScenario, 'mid');
  assert.equal(audit.policy.unit, 'equity');
  for (const name of ['listings', 'exactFreshStarterEconomics', 'exactPriorValuationResults', 'unchangedDrafts']) assert.equal(counts[name], expected.universe, name);
  for (const name of ['exactPriorScenarioResults', 'unpricedCeilingComparisons', 'clearedCandidateComparisons']) assert.equal(counts[name], expected.universe * 3, name);
  assert.equal(counts.availableIntrinsicScenarios + counts.unavailableIntrinsicScenarios, expected.universe * 3);
  assert.equal(counts.verifiedExplicitCandidateNPVs, counts.availableIntrinsicScenarios);
  assert.equal(counts.verifiedCashOnlyCeilings, counts.availableIntrinsicScenarios);
  assert.equal(counts.positivePurchaseCeilings, counts.positiveIntrinsicScenarios);
  assert.ok(Array.isArray(audit.inputs) && audit.inputs.length >= 13, 'Missing universe source manifest');
  const auditFiles = [...audit.inputs, audit.runtime, audit.ledger].map(auditIdentity);
  assert.equal(new Set(audit.inputs.map(input => input.path)).size, audit.inputs.length, 'Duplicate universe input identities');
  for (const required of ['src/purchaseRange.ts', 'src/valuation.ts', 'src/starterValuations.ts', 'src/terminalValue.ts', 'src/cashUncertainty.ts', 'src/data/cash-uncertainty-2026-09-12.json', 'scripts/audit-purchase-ranges.mjs', `financial-data/${packNames[1]}`]) {
    assert.ok(audit.inputs.some(input => input.path === required), `Universe audit did not bind ${required}`);
  }
  assert.ok(Array.isArray(audit.execution.dependencies) && audit.execution.dependencies.length > 0);
  for (const path of audit.execution.dependencies) assert.ok(audit.inputs.some(input => input.path === path), `Unbound audit dependency ${path}`);
  const rows = gunzipSync(readFileSync(resolve(project, audit.ledger.path))).toString('utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(rows.length, expected.universe, 'Universe ledger row count differs');
  assert.equal(new Set(rows.map(row => row.company)).size, expected.universe, 'Universe ledger has duplicate companies');
  assert.ok(rows.every(row => row.asOf === audit.modelAsOf), 'Universe ledger date differs');

  const documentPaths = ['README.md', 'HANDOFF.md', '../README.md', '../CLAUDE.md', '../project_context.md', ...filesIn('../docs'), ...filesIn('../decisions')];
  const documentFiles = [...new Set(documentPaths)].sort().map(identity);
  const receiptPaths = [...Object.values(evidencePaths), ...installationScripts, diagnostics.path, eventsPath];
  const receipts = [...new Set(receiptPaths)].sort().map(identity);
  // Catch modifications while this verification hashes large financial packs.
  for (const bound of [...sourceFiles, ...frontendFiles, ...publicFiles, ...documentFiles, ...receipts, ...auditFiles, executable, installedExecutable,
    shortcut, ...financialPacks.flatMap(pack => [pack.source, pack.installed]), ...installedDocumentFiles.flatMap(doc => [doc.source, doc.installed])]) {
    assert.deepEqual(identity(bound.path), bound, `Evidence changed during verification: ${bound.path}`);
  }
  const git = async args => (await promisify(execFile)('git', ['-C', project, ...args], { encoding: 'utf8' })).stdout.trimEnd();
  const [head, branch, dirtyStatus] = await Promise.all([git(['rev-parse', 'HEAD']), git(['branch', '--show-current']), git(['status', '--porcelain=v1', '--untracked-files=all'])]);
  return {
    status: 'passed', version, verifiedAt: new Date().toISOString(),
    scope: 'Editable purchase-price implementation, arithmetic preservation and installed release verification. No forecast accuracy, investment performance or probabilistic purchase range claim.',
    checks: expected,
    repository: { head, branch, dirtyStatus, note: 'Informational local repository state; neither a commit, push nor clean-tree claim.' },
    evidenceBasis: 'Unit/browser freshness is checked against current local modification times, then current bytes are hashed. Native and installed executable identities are directly reconciled. Universe inputs retain their original audit hashes. This receipt does not retroactively prove the original unit/browser source identity or guarantee a process remains running after installation verification.',
    executable, installedExecutable, installation: identity(evidencePaths.installation), shortcut,
    financialPacks, installedDocumentFiles, modelAudit: counts,
    sourceFiles, frontendFiles, publicFiles, documentFiles, receipts, auditFiles,
  };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    const args = process.argv.slice(2);
    assert.ok(args.every(arg => ['--check', '--replace'].includes(arg)), 'Usage: node scripts/verify-purchase-release.mjs [--check] [--replace]');
    const checkOnly = args.includes('--check');
    assert.ok(checkOnly || args.includes('--replace') || !existsSync(resolve(project, outputPath)), 'Receipt already exists; use --check or explicitly --replace');
    const receipt = await verifyPurchaseRelease();
    if (!checkOnly) writeFileSync(resolve(project, outputPath), JSON.stringify(receipt, null, 2) + '\n', { flag: args.includes('--replace') ? 'w' : 'wx' });
    console.log(JSON.stringify({ status: receipt.status, version, checks: receipt.checks, executableSha256: receipt.executable.sha256, output: checkOnly ? null : outputPath }, null, 2));
  } catch (error) {
    console.error(`Purchase release verification failed: ${error.message}`);
    process.exitCode = 1;
  }
}
