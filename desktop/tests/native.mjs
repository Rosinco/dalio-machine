import { countryEvidenceFlows } from './country-evidence-flows.mjs';
import { transportDiagnostic } from './transport-diagnostics.mjs';
import { uncompressedCdp } from './uncompressed-cdp.mjs';
import { comparisonFlows, restoreComparison, restoreMarketComparison } from './comparison-flows.mjs';
import { financialFlows } from './financial-flows.mjs';
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { readFile, writeFile, mkdir, mkdtemp, rm, copyFile, readdir, stat } from 'node:fs/promises';
import { appendFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { researchFlows } from './research-flows.mjs';
import { businessFlows } from './business-flows.mjs';
import { taxonomyFlows } from './taxonomy-flows.mjs';
import { listingFlows } from './listing-flows.mjs';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const financialOnly = process.argv.includes('--financial-only');
const noCdpCompression = !process.argv.includes('--compressed-cdp');
const executable = process.argv.slice(2).find(arg => !arg.startsWith('--')) || resolve(project, 'src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
const resultFolder = resolve(project, 'test-results');
await mkdir(resultFolder, { recursive: true });
const started = Date.now(), runId = new Date(started).toISOString().replaceAll(':', '-');
const diagnosticsPath = resolve(resultFolder, `windows-native-diagnostics-${runId}.json`);
const eventsPath = resolve(resultFolder, `windows-native-events-${runId}.jsonl`);
const events = [];
let failed = false, failure, intentionalStop = false;
const diagnostic = detail => {
  const event = { at: new Date().toISOString(), elapsed_ms: Date.now() - started, ...detail };
  events.push(event);
  appendFileSync(eventsPath, JSON.stringify(event) + '\n');
  console.log('Native diagnostic', JSON.stringify(event));
};
// pw:browser usually logs only transport events, but Playwright's malformed-JSON
// and onmessage exception branches interpolate the entire eventData payload.
// Remove that field before stderr reaches PowerShell or any retained test log.
const stderrWrite = process.stderr.write.bind(process.stderr);
process.stderr.write = (chunk, ...args) => {
  const text = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
  const transport = transportDiagnostic(text);
  if (transport) {
    diagnostic(transport.detail);
    return stderrWrite(transport.text, ...args);
  }
  return stderrWrite(chunk, ...args);
};
diagnostic({ stage: 'runner_environment', platform: process.platform, arch: process.arch, versions: process.versions, cdpTransport: noCdpCompression ? 'uncompressed-public-custom' : 'playwright-default' });
// Isolate both the WebView preferences and the native research archive.
const profile = await mkdtemp(resolve(tmpdir(), 'macro-atlas-test-'));
const archive = resolve(profile, 'research');
// Exercise local Windows storage, as installed. Cross-WSL filesystem reads have
// very different latency and do not represent the portable application's runtime.
const financialFolder = resolve(profile, 'included-financials');
let app, browser, page;
const runtimeErrors = [], externalRequests = [];
async function startApp() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  app = spawn(executable, [], { stdio: 'ignore', env: { ...process.env, ATLAS_RESEARCH_DIR: archive, ATLAS_FINANCIALS_DIR: financialFolder, WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}` } });
  const pid = app.pid;
  intentionalStop = false;
  diagnostic({ stage: 'app_spawned', pid });
  app.on('exit', (code, signal) => diagnostic({ stage: 'app_exit', pid, code, signal, intentionalStop }));
  let launchError;
  app.on('error', error => { launchError = error; });
  const deadline = Date.now() + 25000;
  browser = undefined;
  while (!browser && Date.now() < deadline) {
    if (launchError) throw launchError;
    if (app.exitCode !== null) throw new Error(`App exited: ${app.exitCode}`);
    let transport;
    try {
      const endpoint = `http://127.0.0.1:${port}`;
      if (noCdpCompression) transport = await uncompressedCdp(endpoint, diagnostic);
      browser = await chromium.connectOverCDP(transport ?? endpoint, { timeout: 1500, isLocal: true });
    } catch {
      transport?.close();
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }
  assert.ok(browser, 'Windows app must expose its test page');
  diagnostic({ stage: 'cdp_connected', pid, browserVersion: browser.version() });
  browser.on('disconnected', () => diagnostic({ stage: 'cdp_disconnected', pid, intentionalStop }));
  page = browser.contexts()[0].pages()[0];
  page.on('pageerror', error => { runtimeErrors.push(error.message); diagnostic({ stage: 'page_error', pid, message: error.message }); });
  page.on('crash', () => diagnostic({ stage: 'page_crash', pid, intentionalStop }));
  page.on('close', () => diagnostic({ stage: 'page_closed', pid, intentionalStop }));
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !/^https?:\/\/(tauri|ipc)\.localhost([/:]|$)/.test(request.url())) externalRequests.push(request.url());
  });
  await page.waitForURL('http://tauri.localhost/');
  return app.pid;
}
async function stopApp() {
  intentionalStop = true;
  diagnostic({ stage: 'intentional_test_stop', pid: app?.pid });
  await browser?.close().catch(() => {});
  browser = undefined;
  if (app && app.exitCode === null) {
    const stopped = new Promise(resolve => app.once('exit', resolve));
    app.kill();
    await Promise.race([stopped, new Promise((_, reject) => setTimeout(() => reject(new Error('Test app did not exit')), 10000))]);
  }
}
try {
  await mkdir(financialFolder);
  for (const name of await readdir(resolve(project, 'financial-data'))) {
    if (/^[a-f0-9]{64}\.sqlite$/.test(name)) await copyFile(resolve(project, 'financial-data', name), resolve(financialFolder, name));
  }
  console.log('Financial pack copied to local Windows test storage');
  console.log(`Windows app started for testing: ${await startApp()}`);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  if (financialOnly) {
    // Use the real included default release and full native companion/importer.
    // Skip unrelated research, directory and branch loops, never shrink the pack.
    const catalogue = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
    await page.locator(`[data-active-release="${catalogue.default_id}"] [data-country="SE"][data-ready="true"]`).waitFor();
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
    await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
    diagnostic({ stage: 'focused_financial_flow_started', release: catalogue.default_id });
    const financial = await financialFlows(page, project, { native: true, archive, diagnostic });
    const firstPid = app.pid;
    await stopApp();
    await startApp();
    assert.notEqual(app.pid, firstPid);
    await page.locator(`[data-active-release="${catalogue.default_id}"] [data-financial-history="102"][data-financial-pack="${financial.index.id}"]`).waitFor();
    assert.deepEqual(externalRequests, []); assert.deepEqual(runtimeErrors, []);
    const report = { status: 'PASS', scope: 'financial-only', checks: [...financial.checks, 'Full imported financial pack survives native process restart'], pack: financial.index.id, bytes: financial.index.bytes, taxonomy: financial.index.taxonomy_sha256, externalRequests, runtimeErrors, executableSha256: createHash('sha256').update(await readFile(executable)).digest('hex') };
    await writeFile(resolve(resultFolder, 'windows-native-financial-report.json'), JSON.stringify(report, null, 2));
    diagnostic({ stage: 'focused_financial_flow_passed', pack: report.pack, bytes: report.bytes });
    console.log(JSON.stringify(report, null, 2));
  } else {
  const expression = await readFile(resolve(project, 'tests/native-smoke.js'), 'utf8');
  const report = await page.evaluate(expression);
  const research = await researchFlows(page, project);
  report.checks.push(...research.checks);
  assert.equal(await readFile(resolve(archive, `${research.older.id}.atlas.json`), 'utf8'), research.original);
  const countryEvidence = await countryEvidenceFlows(page, project);
  report.checks.push(...countryEvidence.checks);
  report.countryEvidencePack = countryEvidence.pack;
  const business = await businessFlows(page, project);
  report.checks.push(...business.checks);
  assert.equal(await readFile(resolve(archive, `${business.current.id}.atlas.json`), 'utf8'), business.original);
  const taxonomy = await taxonomyFlows(page, project);
  report.checks.push(...taxonomy.checks);
  const listings = await listingFlows(page, project);
  report.checks.push(...listings.checks); report.listingFlowsMs = listings.duration_ms;
  const financial = await financialFlows(page, project, { native: true, archive, diagnostic });
  report.checks.push(...financial.checks); report.financialCompanySwitchMs = financial.timings; report.financialExport = financial.exportedPath;
  const comparison = await comparisonFlows(page, project);
  report.checks.push(...comparison.checks); report.branchForestryMs = comparison.forestryMs; report.branchMiningMs = comparison.miningMs;
  const firstPid = app.pid;
  await stopApp();
  console.log(`Windows app restarted for persistence testing: ${await startApp()}`);
  assert.notEqual(app.pid, firstPid);
  await page.locator(`[data-active-release="${research.current.id}"] [data-company="102"][data-business-ready="true"]`).waitFor();
  await page.locator(`[data-financial-history="102"][data-financial-pack="${financial.index.id}"]`).waitFor();
  report.checks.push('Imported company financial pack survives native process restart');
  await restoreMarketComparison(page);
  report.checks.push('SEK market-cap metrics, bubble size, all-currency filter and Swedish notes survive native process restart');
  await restoreComparison(page);
  await page.screenshot({ path: resolve(resultFolder, 'windows-branch-comparison.png') });
  report.checks.push('Saved branch comparison and Swedish research notes survive native process restart');
  await page.getByLabel('Open financials for Holmen', { exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
  await page.screenshot({ path: resolve(resultFolder, 'windows-holmen.png') });
  await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
  await page.getByLabel('Open data library').click();
  await page.locator(`[data-release-id="${research.older.id}"][data-storage="imported"]`).waitFor();
  await page.locator(`[data-release-id="${research.current.id}"][data-storage="imported"]`).waitFor();
  assert.equal(await readFile(resolve(archive, `${taxonomy.legacyId}.atlas.json`), 'utf8'), taxonomy.legacyText);
  report.checks.push('V1, V2 and V3 imported packages survive native process restart, byte-for-byte');
  await page.locator(`[data-active-release="${research.current.id}"]`).waitFor();
  await page.getByRole('button', { name: 'Save a copy of active release', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.library-message')?.textContent.startsWith('Saved to '));
  report.researchExport = (await page.locator('.library-message').innerText()).replace(/^Saved to /, '').trim();
  const exported = JSON.parse(await readFile(report.researchExport, 'utf8'));
  assert.equal(exported.fundamentals.sha256, research.current.fundamentals_sha256);
  assert.equal(exported.liquidity.sha256, research.current.liquidity_sha256);
  assert.equal(exported.business.sha256, research.current.business_sha256);
  assert.equal(exported.taxonomy.sha256, research.current.taxonomy_sha256);
  report.checks.push('Native portable research export to Downloads');
  await page.keyboard.press('Escape');
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
  await page.getByLabel('Export selected history as CSV').click();
  await page.waitForFunction(() => document.querySelector('.toast')?.textContent.startsWith('Saved to '));
  const exportPath = (await page.locator('.toast').innerText()).replace(/^Saved to /, '').trim();
  const csv = await readFile(exportPath, 'utf8');
  assert.ok(csv.includes('source_snapshot_sha256') && csv.includes('Sweden'), 'Native export must contain the selected country and source reference');
  report.checks.push('Native CSV export to Downloads');
  report.csvExport = exportPath;
  await page.getByLabel('Dismiss message').click();
  await page.screenshot({ path: resolve(resultFolder, 'windows-native.png') });
  assert.deepEqual(externalRequests, [], 'Native app must load its assets locally');
  assert.deepEqual(runtimeErrors, [], 'Native app must have no runtime errors');
  Object.assign(report, { externalRequests, runtimeErrors, executableSha256: createHash('sha256').update(await readFile(executable)).digest('hex') });
  await writeFile(resolve(resultFolder, 'windows-native-report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  }
} catch (error) {
  failed = true;
  failure = { message: String(error), appExitCode: app?.exitCode, appSignal: app?.signalCode, pageClosed: page?.isClosed(), browserConnected: browser?.isConnected(), lastEvent: events.at(-1) };
  diagnostic({ stage: 'test_failure', ...failure });
  if (page) {
    await page.screenshot({ path: resolve(resultFolder, 'windows-native-failure.png') }).catch(() => {});
    console.error(await page.locator('body').innerText().catch(() => 'Page text unavailable'));
  }
  console.error('Native failure state', { appExitCode: app?.exitCode, appSignal: app?.signalCode, pageClosed: page?.isClosed(), browserConnected: browser?.isConnected() });
  console.error(error);
  process.exitCode = 1;
} finally {
  await stopApp().catch(error => console.error(error));
  await new Promise(resolve => setTimeout(resolve, 500));
  const financialFiles = [];
  const uploadFolder = resolve(archive, 'financial-packs');
  for (const name of await readdir(uploadFolder).catch(() => [])) {
    if (!/^(?:upload-)?[a-f0-9]{64}\.(?:part|sqlite)$/.test(name)) continue;
    const metadata = await stat(resolve(uploadFolder, name));
    financialFiles.push({ file: `financial-packs/${name}`, bytes: metadata.size, modified_at: metadata.mtime.toISOString() });
  }
  await writeFile(diagnosticsPath, JSON.stringify({ runId, scope: financialOnly ? 'financial-only' : 'full', status: failed ? 'FAIL' : 'PASS', executableSha256: createHash('sha256').update(await readFile(executable)).digest('hex'), failure, isolatedProfile: failed ? profile : null, financialFiles, events, runtimeErrors, externalRequests }, null, 2));
  console.log(`Native diagnostics: ${diagnosticsPath}`);
  if (failed) console.warn(`Preserved failed isolated test profile: ${profile}`);
  else await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => console.warn(`Test profile still in use: ${profile}`));
}
