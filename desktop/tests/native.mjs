import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const executable = process.argv[2] || resolve(project, 'src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
const resultFolder = resolve(project, 'test-results');
await mkdir(resultFolder, { recursive: true });
const server = createServer();
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;
await new Promise(resolve => server.close(resolve));
// Keep the test browser and its preferences separate from an open Atlas window.
const profile = await mkdtemp(resolve(tmpdir(), 'macro-atlas-test-'));
const app = spawn(executable, [], { stdio: 'ignore', env: { ...process.env, WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port}` } });
let launchError;
app.on('error', error => { launchError = error; });
let browser, page;
const runtimeErrors = [], externalRequests = [];
try {
  console.log(`Windows app started for testing: ${app.pid}`);
  const deadline = Date.now() + 25000;
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
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  const expression = await readFile(resolve(project, 'tests/native-smoke.js'), 'utf8');
  const report = await page.evaluate(expression);
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
  await browser?.close().catch(() => {});
  if (app.exitCode === null) app.kill();
  await new Promise(resolve => setTimeout(resolve, 500));
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 }).catch(() => console.warn(`Test profile still in use: ${profile}`));
}
