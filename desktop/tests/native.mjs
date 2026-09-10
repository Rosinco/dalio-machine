import { financialFlows } from './financial-flows.mjs';
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { readFile, writeFile, mkdir, mkdtemp, rm, copyFile, readdir } from 'node:fs/promises';
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
const executable = process.argv[2] || resolve(project, 'src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
const resultFolder = resolve(project, 'test-results');
await mkdir(resultFolder, { recursive: true });
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
  let launchError;
  app.on('error', error => { launchError = error; });
  const deadline = Date.now() + 25000;
  browser = undefined;
  while (!browser && Date.now() < deadline) {
    if (launchError) throw launchError;
    if (app.exitCode !== null) throw new Error(`App exited: ${app.exitCode}`);
    try { browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 1500 }); }
    catch { await new Promise(resolve => setTimeout(resolve, 200)); }
  }
  assert.ok(browser, 'Windows app must expose its test page');
  page = browser.contexts()[0].pages()[0];
  page.on('pageerror', error => runtimeErrors.push(error.message));
  page.on('request', request => {
    if (/^https?:/.test(request.url()) && !/^https?:\/\/(tauri|ipc)\.localhost([/:]|$)/.test(request.url())) externalRequests.push(request.url());
  });
  await page.waitForURL('http://tauri.localhost/');
  return app.pid;
}
async function stopApp() {
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
  const expression = await readFile(resolve(project, 'tests/native-smoke.js'), 'utf8');
  const report = await page.evaluate(expression);
  const research = await researchFlows(page, project);
  report.checks.push(...research.checks);
  assert.equal(await readFile(resolve(archive, `${research.older.id}.atlas.json`), 'utf8'), research.original);
  const business = await businessFlows(page, project);
  report.checks.push(...business.checks);
  assert.equal(await readFile(resolve(archive, `${business.current.id}.atlas.json`), 'utf8'), business.original);
  const taxonomy = await taxonomyFlows(page, project);
  report.checks.push(...taxonomy.checks);
  const listings = await listingFlows(page, project);
  report.checks.push(...listings.checks); report.listingFlowsMs = listings.duration_ms;
  const financial = await financialFlows(page, project, { native: true, archive });
  report.checks.push(...financial.checks); report.financialCompanySwitchMs = financial.timings; report.financialExport = financial.exportedPath;
  const firstPid = app.pid;
  await stopApp();
  console.log(`Windows app restarted for persistence testing: ${await startApp()}`);
  assert.notEqual(app.pid, firstPid);
  await page.locator(`[data-active-release="${research.current.id}"] [data-company="102"][data-business-ready="true"]`).waitFor();
  await page.locator(`[data-financial-history="102"][data-financial-pack="${financial.index.id}"]`).waitFor();
  report.checks.push('Imported company financial pack survives native process restart');
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
} catch (error) {
  if (page) {
    await page.screenshot({ path: resolve(resultFolder, 'windows-native-failure.png') }).catch(() => {});
    console.error(await page.locator('body').innerText().catch(() => 'Page text unavailable'));
  }
  console.error(error);
  process.exitCode = 1;
} finally {
  await stopApp().catch(error => console.error(error));
  await new Promise(resolve => setTimeout(resolve, 500));
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => console.warn(`Test profile still in use: ${profile}`));
}
