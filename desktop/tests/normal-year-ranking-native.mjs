import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { readFile, writeFile, mkdir, mkdtemp, copyFile, readdir, rm } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { uncompressedCdp } from './uncompressed-cdp.mjs';
import { normalYearRankingFlows, assertNormalYearRankingRestart } from './normal-year-ranking-flows.mjs';

assert.equal(process.platform, 'win32', 'This runner requires Windows Node');
const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const executable = process.argv[2] || resolve(project, 'src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe');
const profile = await mkdtemp(resolve(tmpdir(), 'macro-atlas-normal-ranking-'));
const financialFolder = resolve(profile, 'included-financials');
const externalRequests = [], runtimeErrors = [];
let app, browser, page, passed = false;
async function start() {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  app = spawn(executable, [], { stdio: 'ignore', env: { ...process.env, ATLAS_RESEARCH_DIR: resolve(profile, 'research'), ATLAS_FINANCIALS_DIR: financialFolder, WEBVIEW2_USER_DATA_FOLDER: profile, WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-address=127.0.0.1 --remote-debugging-port=${port}` } });
  let launchError;
  app.on('error', error => { launchError = error; });
  const deadline = Date.now() + 30000;
  while (!browser && Date.now() < deadline) {
    if (launchError) throw launchError;
    if (app.exitCode !== null) throw new Error(`Test app exited: ${app.exitCode}`);
    let transport;
    try {
      transport = await uncompressedCdp(`http://127.0.0.1:${port}`, () => {}, 1500);
      browser = await chromium.connectOverCDP(transport, { timeout: 1500, isLocal: true });
    } catch { transport?.close(); await new Promise(resolve => setTimeout(resolve, 200)); }
  }
  assert.ok(browser, 'Isolated native app must expose its test page');
  page = browser.contexts()[0].pages()[0];
  page.on('pageerror', error => runtimeErrors.push(error.message));
  page.on('request', request => { if (/^https?:/.test(request.url()) && !/^https?:\/\/(tauri|ipc)\.localhost([/:]|$)/.test(request.url())) externalRequests.push(request.url()); });
  await page.waitForURL('http://tauri.localhost/');
  return app.pid;
}
async function stop() {
  await browser?.close().catch(() => {}); browser = undefined;
  if (app && app.exitCode === null) {
    const exited = new Promise(resolve => app.once('exit', resolve));
    app.kill();
    await Promise.race([exited, new Promise((_, reject) => setTimeout(() => reject(new Error('Isolated test app did not exit')), 10000))]);
  }
}
try {
  await mkdir(financialFolder);
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  for (const name of await readdir(resolve(project, 'financial-data'))) if (/^[a-f0-9]{64}\.sqlite$/.test(name)) await copyFile(resolve(project, 'financial-data', name), resolve(financialFolder, name));
  const firstPid = await start();
  const result = await normalYearRankingFlows(page, project, { native: true });
  await stop();
  assert.notEqual(await start(), firstPid);
  await assertNormalYearRankingRestart(page, result.restart);
  result.checks.push('A full native process restart retains the five-year 60/40 ranking saved views, exact scores and rows and untouched authored records');
  assert.deepEqual(externalRequests, []); assert.deepEqual(runtimeErrors, []);
  const report = { ...result, status: 'PASS', scope: 'Isolated Windows profile and native archives', externalRequests, runtimeErrors, executableSha256: createHash('sha256').update(await readFile(executable)).digest('hex') };
  await writeFile(resolve(project, 'test-results/normal-year-five-year-ranking-native-report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, checks: report.checks, cohortSize: result.cohortSize, rankedCount: result.rankedCount, unrankedCount: result.unrankedCount, executableSha256: report.executableSha256 }));
  passed = true;
} catch (error) {
  const stem = resolve(project, 'test-results', `normal-year-five-year-ranking-native-failed-${Date.now()}`);
  await page?.screenshot({ path: `${stem}.png` }).catch(() => {});
  await writeFile(`${stem}.json`, JSON.stringify({ status: 'FAIL', profile, error: error.stack || String(error), externalRequests, runtimeErrors, executableSha256: createHash('sha256').update(await readFile(executable)).digest('hex') }, null, 2));
  throw error;
} finally {
  await stop();
  if (passed) await rm(profile, { recursive: true, force: true }).catch(() => {});
}
