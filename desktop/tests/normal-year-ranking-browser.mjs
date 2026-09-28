import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { normalYearRankingFlows } from './normal-year-ranking-flows.mjs';

const base = process.env.ATLAS_URL || 'http://127.0.0.1:1420';
await mkdir('test-results', { recursive: true });
const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let page;
try {
  const context = await browser.newContext({ viewport: { width: 1500, height: 960 } });
  const externalRequests = [], runtimeErrors = [];
  const csp = JSON.parse(await readFile('src-tauri/tauri.conf.json', 'utf8')).app.security.csp;
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (url === `${base}/`) {
      const response = await route.fetch();
      return route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': csp } });
    }
    if (url.startsWith(`${base}/`) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
    externalRequests.push(url); return route.abort();
  });
  page = await context.newPage();
  page.on('pageerror', error => runtimeErrors.push(error.message));
  await page.goto(base);
  const result = await normalYearRankingFlows(page, process.cwd());
  assert.deepEqual(externalRequests, []); assert.deepEqual(runtimeErrors, []);
  const report = { ...result, status: 'PASS', scope: 'Isolated production-CSP browser profile', externalRequests, runtimeErrors };
  await writeFile('test-results/normal-year-five-year-ranking-browser-report.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ status: report.status, checks: report.checks, cohortSize: result.cohortSize, rankedCount: result.rankedCount, unrankedCount: result.unrankedCount }));
} catch (error) {
  const stem = `test-results/normal-year-five-year-ranking-browser-failed-${Date.now()}`;
  await page?.screenshot({ path: `${stem}.png` }).catch(() => {});
  await writeFile(`${stem}.json`, JSON.stringify({ status: 'FAIL', error: error.stack || String(error) }, null, 2));
  throw error;
} finally { await browser.close(); }
