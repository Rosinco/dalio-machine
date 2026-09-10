import { comparisonFlows } from './comparison-flows.mjs';
import { financialFlows } from './financial-flows.mjs';
import { chromium } from 'playwright';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { researchFlows } from './research-flows.mjs';
import { businessFlows } from './business-flows.mjs';
import { taxonomyFlows } from './taxonomy-flows.mjs';
import { listingFlows } from './listing-flows.mjs';

const base = process.env.ATLAS_URL || 'http://127.0.0.1:1420';
await mkdir('test-results', { recursive: true });
const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let page;
try {
const context = await browser.newContext({ viewport: { width: 1500, height: 960 } });
const external = [], errors = [], timing = {};
const csp = JSON.parse(await readFile('src-tauri/tauri.conf.json', 'utf8')).app.security.csp;
await context.route('**/*', async route => {
  const url = route.request().url();
  if (url === `${base}/`) {
    const response = await route.fetch();
    return route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': csp } });
  }
  if (url.startsWith(base) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
  external.push(url); return route.abort();
});
page = await context.newPage();
page.on('pageerror', e => errors.push(e.message));
page.on('console', e => { if (e.type() === 'error') console.error('Browser:', e.text()); });
const start = performance.now();
await page.goto(base);
await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
await page.locator('[data-map-ready="true"]').waitFor();
if (process.argv.includes('--comparison-only')) {
  console.log(JSON.stringify(await comparisonFlows(page, process.cwd())));
  assert.deepEqual(external, []); assert.deepEqual(errors, []);
  await browser.close(); process.exit(0);
}
if (process.argv.includes('--financial-only')) {
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await page.locator('[data-business-ready="true"]').waitFor();
  console.log(JSON.stringify(await financialFlows(page, process.cwd()), (key, value) => key === 'index' ? undefined : value));
  await browser.close(); process.exit(0);
}
await page.waitForFunction(() => document.querySelector('.maplibregl-canvas')?.width > 0);
await page.waitForTimeout(800);
timing.startup_ms = Math.round(performance.now() - start);
await page.screenshot({ path: 'test-results/world-sweden.png' });
assert.equal(await page.locator('.map-error').count(), 0);
await page.getByLabel('Search countries').fill('United States');
await page.getByLabel('Search countries').press('Enter');
await page.locator('[data-country="US"][data-ready="true"]').waitFor();
// A real click on the central Swedish polygon at the fixed test viewport.
await page.locator('.map-canvas').click({ position: { x: 532, y: 228 } });
await page.screenshot({ path: 'test-results/map-click.png' });
await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
await page.getByLabel('Comparison country').selectOption('DE');
await page.waitForFunction(() => document.querySelector('.chart-key')?.textContent.includes('Germany'));
await page.getByLabel('Map category').selectOption('promises');
assert.match(await page.locator('.legend').innerText(), /Relative fundamentals/);
assert.equal(await page.locator('.legend-scale i').first().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(207, 87, 87)');
assert.equal(await page.locator('.legend-scale i').last().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(59, 148, 108)');
await page.getByLabel('History & outlook', { exact: true }).click();
await page.getByLabel('Map historical indicator').selectOption('gov_debt_pct_gdp');
await page.waitForFunction(() => !document.querySelector('.map-footnote')?.textContent.includes('Loading'));
await page.getByLabel('Historical year').fill('2005');
assert.match(await page.locator('.time-control').innerText(), /2005/);
assert.match(await page.locator('.metric-readout').innerText(), /Historical observation · 2005/);
assert.match(await page.locator('.legend').innerText(), /Lower values = stronger/);
assert.equal(await page.locator('.legend-scale i').first().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(59, 148, 108)');
assert.equal(await page.locator('.legend-scale i').last().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(207, 87, 87)');
await page.screenshot({ path: 'test-results/history.png' });
await page.getByLabel('Map historical indicator').selectOption('gdp_pc_ppp');
assert.match(await page.locator('.legend').innerText(), /Higher values = stronger/);
assert.equal(await page.locator('.legend-scale i').last().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(59, 148, 108)');
await page.getByLabel('Map historical indicator').selectOption('gdp_usd');
assert.match(await page.locator('.legend').innerText(), /no good\/bad rating/);
assert.equal(await page.locator('.legend-scale i').last().evaluate(el => getComputedStyle(el).backgroundColor), 'rgb(36, 87, 130)');
await page.getByLabel('Map historical indicator').selectOption('gov_debt_pct_gdp');
await page.getByRole('tab', { name: 'Indicators', exact: true }).click();
assert.match(await page.locator('.indicator-row').first().innerText(), /WORLD BANK/);
await page.getByLabel('Trade connections', { exact: true }).click();
assert.equal(await page.locator('.legend').getAttribute('data-colour-direction'), 'neutral');
assert.equal(await page.locator('.trade-partners button').count(), 6);
assert.match(await page.locator('.trade-partners').innerText(), /Other destinations/);
assert.doesNotMatch(await page.locator('.trade-partners').innerText(), /Euro area/);
await page.screenshot({ path: 'test-results/trade.png' });
await page.getByLabel('Search countries').fill('United States');
const countryStart = performance.now();
await page.getByLabel('Search countries').press('Enter');
await page.locator('[data-country="US"][data-ready="true"]').waitFor();
timing.country_switch_ms = Math.round(performance.now() - countryStart);
await page.getByLabel('Fundamentals', { exact: true }).click();
await page.locator('.flow .react-flow__node').first().waitFor();
await page.locator('.flow').scrollIntoViewIfNeeded();
await page.screenshot({ path: 'test-results/pressure-flow.png' });
await page.getByRole('tab', { name: 'Evidence', exact: true }).click();
assert.match(await page.locator('.release-details').innerText(), /42e9a3af48f9/);
await page.getByLabel('Open data library').click();
await page.getByRole('dialog').waitFor();
await page.keyboard.press('Escape');
await page.getByLabel('Search countries').fill('Sweden');
await page.getByLabel('Search countries').press('Enter');
await page.getByRole('tab', { name: 'Overview', exact: true }).click();
await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
const downloadPromise = page.waitForEvent('download');
await page.getByLabel('Export selected history as CSV').click();
const download = await downloadPromise;
assert.match(download.suggestedFilename(), /Macro-Atlas-SE-gov_debt_pct_gdp.csv/);
await download.saveAs('test-results/history-export.csv');
const research = await researchFlows(page, process.cwd());
const business = await businessFlows(page, process.cwd());
const taxonomy = await taxonomyFlows(page, process.cwd());
const listings = await listingFlows(page, process.cwd());
taxonomy.checks.push(...listings.checks); timing.listing_flows_ms = listings.duration_ms;
const financial = await financialFlows(page, process.cwd());
taxonomy.checks.push(...financial.checks); timing.financial_company_switch_ms = financial.timings;
const comparison = await comparisonFlows(page, process.cwd());
taxonomy.checks.push(...comparison.checks); timing.branch_forestry_ms = comparison.forestryMs; timing.branch_mining_ms = comparison.miningMs;
await page.setViewportSize({ width: 1100, height: 760 });
await page.screenshot({ path: 'test-results/company-compact.png' });
assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
await page.getByLabel('Open data library').click();
if (await page.locator('.app').getAttribute('data-active-release') !== research.current.id) await page.getByLabel(`Use release ${research.current.as_of}`, { exact: true }).click();
await page.locator(`[data-active-release="${research.current.id}"]`).waitFor();
await page.keyboard.press('Escape');
await page.getByRole('tab', { name: 'Liquidity', exact: true }).click();
await page.setViewportSize({ width: 1100, height: 760 });
await page.screenshot({ path: 'test-results/compact.png' });
assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
assert.deepEqual(external, [], 'The offline application must not request any external resources');
assert.deepEqual(errors, [], 'The browser must not report runtime errors');
await writeFile('test-results/browser-report.json', JSON.stringify({ timing, externalRequests: external, runtimeErrors: errors, checks: ['native content security policy', 'initial Sweden', 'map click', 'comparison', 'category change', 'history mode/year', 'indicator evidence', 'trade denominator', 'country search', 'lazy flow diagram', 'evidence manifest', 'library', 'CSV export', 'compact viewport', ...research.checks, ...business.checks, ...taxonomy.checks] }, null, 2));
console.log(JSON.stringify({ status: 'PASS', timing, researchChecks: research.checks, businessChecks: business.checks, taxonomyChecks: taxonomy.checks, externalRequests: external.length, runtimeErrors: errors.length }));
await browser.close();
} catch (error) {
  if (page) {
    await page.screenshot({ path: 'test-results/browser-failure.png' }).catch(() => {});
    console.error(await page.locator('body').innerText().catch(() => 'Page unavailable'));
  }
  throw error;
} finally { await browser.close(); }
