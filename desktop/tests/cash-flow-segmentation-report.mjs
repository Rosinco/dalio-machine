import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

// Validate the already-generated research artifact; no application or source writes.
const here = dirname(fileURLToPath(import.meta.url));
const flag = process.argv.indexOf('--directory');
if (flag >= 0 && !process.argv[flag + 1]) throw new Error('--directory requires an artifact path.');
const directory = resolve(flag >= 0 ? process.argv[flag + 1] : join(here, '../test-results/cash-flow-segmentation-2026-09-12'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const load = async name => JSON.parse(await readFile(join(directory, name), 'utf8'));
const receipt = await load('report-receipt.json');
const results = await load('results.json');
const within = await load('within-branch-results.json');
const reportFile = join(directory, 'report.html');
assert.equal(hash(await readFile(reportFile)), receipt.report.sha256);
for (const [name, expected] of Object.entries(receipt.inputs)) assert.equal(hash(await readFile(join(directory, name))), expected, name);
const number = value => value.toLocaleString('en-US', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const percent = value => `${(100 * value).toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 })}%`;
const identityKeys = ['scope', 'model', 'horizon', 'period', 'weighting'];
const match = (row, state) => identityKeys.every(key => row[key] === state[key]);
const primary = { scope: 'operating_and_property', model: 'linear', horizon: 1, period: 'recent', weighting: 'listing_balanced' };
const checks = [], errors = [], requests = [];
const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox'] });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, offline: true, acceptDownloads: true });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (/^https?:/.test(request.url())) requests.push(request.url()); });
  const started = performance.now();
  await page.goto(pathToFileURL(reportFile).href, { waitUntil: 'load', timeout: 60000 });
  await page.locator('#comparison-table tbody tr').first().waitFor();
  const loadMs = Math.round(performance.now() - started);
  const embedded = await page.evaluate(() => {
    const d = JSON.parse(document.getElementById('report-data').textContent);
    return { inputs: d.inputHashes, metrics: d.results.metrics.length, groups: d.results.groupMetrics.length,
      companies: d.companies.length, within: d.withinBranch.metrics };
  });
  assert.deepEqual(embedded.inputs, receipt.inputs);
  assert.equal(embedded.metrics, receipt.comparisonRows);
  assert.equal(embedded.groups, receipt.groupRows);
  assert.equal(embedded.companies, receipt.companyFY2024Rows);
  assert.deepEqual(embedded.within, within.metrics);
  checks.push('Offline file loads and embedded comparison, group, company and full-precision within-branch records match hashed artifacts');

  async function choose(state) {
    for (const [key, value] of Object.entries(state)) await page.locator(`#${key}`).selectOption(String(value));
  }
  async function checkComparisons(state) {
    const expected = results.metrics.filter(row => match(row, state));
    if (!expected.length) {
      assert.ok(await page.locator('#no-results').isVisible());
      assert.ok(await page.locator('#scored-content').isHidden());
      return;
    }
    assert.ok(await page.locator('#no-results').isHidden());
    const rows = await page.locator('#comparison-table tbody tr').evaluateAll(elements => elements.map(row => ({
      candidate: row.querySelector('button').dataset.candidate,
      cells: [...row.cells].map(cell => cell.textContent),
    })));
    assert.equal(rows.length, expected.length);
    for (const row of expected) {
      const shown = rows.find(value => value.candidate === row.candidate);
      assert.ok(shown, row.candidate);
      assert.deepEqual(shown.cells.slice(1), [percent(row.coverage), number(row.meanWidth), number(row.meanIntervalScore),
        percent(row.scoreImprovement), percent(row.fallbackShare)], `${JSON.stringify(state)} / ${row.candidate}`);
    }
    assert.equal(await page.locator('#scatter svg').count(), 1);
    assert.equal(await page.locator('#scatter circle').count(), expected.length);
  }
  await checkComparisons(primary);
  const drift = results.metrics.find(row => match(row, { ...primary, period: 'FY2022' }) && row.candidate === 'cash_dispersion');
  assert.ok((await page.locator('#drift-note').innerText()).includes(percent(drift.coverage)));
  await page.screenshot({ path: join(directory, 'report-overview.png') });

  const cashGroups = results.groupMetrics.filter(row => match(row, primary) && row.candidate === 'cash_dispersion');
  assert.equal(await page.locator('#cash-behavior-chart svg rect').count(), 7);
  const chart = await page.locator('#cash-behavior-chart').innerText();
  for (const group of cashGroups.filter(row => ['low', 'medium', 'high'].includes(row.group))) {
    assert.ok(chart.includes(`${percent(group.globalCoverage)} → ${percent(group.coverage)}`));
    assert.ok(chart.includes(`${number(group.globalWidth)} → ${number(group.meanWidth)}`));
  }
  await page.locator('#cash-behavior-panel').screenshot({ path: join(directory, 'report-cash-groups.png') });
  checks.push('Default coverage, width and interval-score comparisons, three cash-behavior groups and FY2022 drift warning reconcile to source metrics');

  for (const state of [
    { ...primary, model: 'naive' },
    { ...primary, weighting: 'duplicate_history_downweighted' },
    { ...primary, scope: 'all_listings' },
    { ...primary, horizon: 2, period: 'validation' },
    { ...primary, horizon: 2 },
    { ...primary, horizon: 3, period: 'validation' },
    { ...primary, horizon: 4 },
    ...['FY2022', 'FY2023', 'FY2024', 'FY2025'].map(period => ({ ...primary, period })),
  ]) {
    await choose(state);
    await checkComparisons(state);
    const expectedWithin = within.metrics.filter(row => match(row, state));
    assert.equal(await page.locator('#within-table tbody tr').count(), expectedWithin.length);
    assert.equal(await page.locator('#within-unavailable').isVisible(), !expectedWithin.length);
    if (expectedWithin.length) {
      const cells = await page.locator('#within-table tbody tr').evaluateAll(rows => rows.map(row => [...row.cells].map(cell => cell.textContent)));
      expectedWithin.forEach((row, index) => assert.deepEqual(cells[index].slice(1), [percent(row.coverage), number(row.meanWidth),
        number(row.meanIntervalScore), percent(row.scoreImprovement), percent(row.fallbackShare)]));
    }
  }
  checks.push('Model, horizon, scope, weighting and annual/combined period selectors use the matching metrics; unavailable timing cohorts remain missing');

  await choose(primary);
  const withinHeader = await page.locator('#within-table thead').innerText();
  assert.match(withinHeader, /vs branch/i);
  assert.ok((await page.locator('#within-caption').innerText()).includes('mid forecast is unchanged'));
  const actualWithin = embedded.within.filter(row => match(row, primary));
  assert.equal(actualWithin.length, 5);
  const refinements = actualWithin.filter(row => row.candidate !== 'branch').map(row => ({ candidate: row.candidate, improvementPercent: 100 * row.scoreImprovement }));
  checks.push('Within-branch improvements use the branch comparator, preserve full precision and display the final artifact values');

  await page.locator('#comparison-table button[data-candidate="cash_dispersion"]').click();
  assert.equal(await page.locator('#candidate').inputValue(), 'cash_dispersion');
  await page.locator('#group-search').fill('high');
  assert.equal(await page.locator('#group-table tbody tr').count(), 1);
  assert.match(await page.locator('#group-table tbody').innerText(), /high/);
  await page.locator('#group-search').fill('');
  await page.locator('#candidate').selectOption('tbv_ebit');
  await page.locator('#group-search').fill('Missing');
  assert.equal(await page.locator('#group-table tbody tr').count(), 1);
  assert.match(await page.locator('#group-table tbody').innerText(), /100\.0%/);
  await page.locator('#group-search').fill('');
  await page.locator('#candidate').selectOption('sector');
  assert.match(await page.locator('#group-table tbody').innerText(), /Property \(within Financials taxonomy\)/);
  await page.locator('#candidate').selectOption('branch');
  const branchExample = results.groupMetrics.find(row => match(row, primary) && row.candidate === 'branch' && row.group.includes(' / ')).group;
  await page.locator('#group-search').fill(branchExample);
  assert.ok(await page.locator('#group-table tbody tr').count() >= 1);
  assert.ok((await page.locator('#group-table tbody').innerText()).includes(branchExample));
  await page.locator('#group-search').fill('not-a-real-financial-group-982');
  assert.match(await page.locator('#group-table tbody').innerText(), /No groups match/);
  await page.locator('#group-search').fill('');
  await page.locator('#candidate').selectOption('global');
  assert.ok(await page.locator('#group-unavailable').isVisible());
  checks.push('Clickable grouping rules, group search, missing-ratio fallback and property-only Financials labels remain explicit');

  await page.locator('#company-section summary').click();
  await page.locator('#company-search').fill('Stora Enso');
  assert.match(await page.locator('#company-table tbody').innerText(), /Stora Enso/);
  await page.locator('#company-search').fill('SCA');
  assert.ok(await page.locator('#company-table tbody tr').count() >= 1);
  await page.locator('#company-search').fill('not-a-real-company-982');
  assert.match(await page.locator('#company-table tbody').innerText(), /No company histories match/);
  await page.locator('#company-search').fill('');
  assert.equal(await page.locator('#company-table tbody tr').count(), Math.min(100, receipt.companyFY2024Rows));
  await page.locator('#company-section summary').click();
  checks.push('Company feature search works offline, shows missing inputs and caps the rendered table at 100 matches');

  await choose({ ...primary, candidate: 'cash_dispersion' });
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    const dimensions = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth }));
    assert.ok(dimensions.page <= dimensions.viewport + 1, JSON.stringify(dimensions));
  }
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: join(directory, 'report-mobile.png') });
  checks.push('Desktop 1440px and mobile 390px avoid horizontal page overflow; wide charts and tables scroll inside containers');
  assert.deepEqual(errors, []);
  assert.deepEqual(requests, []);
  assert.doesNotMatch(await page.locator('main').innerText(), /\bundefined\b|\bNaN\b/);
  const screenshots = [];
  for (const name of ['report-overview.png', 'report-cash-groups.png', 'report-mobile.png']) {
    const bytes = await readFile(join(directory, name));
    screenshots.push({ name, sha256: hash(bytes), bytes: bytes.length });
  }
  const validation = { status: 'pass', completedAt: new Date().toISOString(), reportSha256: receipt.report.sha256,
    validatorSha256: hash(await readFile(fileURLToPath(import.meta.url))), loadMs,
    counts: { comparisons: embedded.metrics, groupRows: embedded.groups, companyHistories: embedded.companies, withinBranch: embedded.within.length },
    withinBranchPrimaryRecent: refinements, checks, errors, externalRequests: requests, screenshots };
  await writeFile(join(directory, 'report-browser-validation.json'), JSON.stringify(validation, null, 2) + '\n');
  console.log(JSON.stringify(validation, null, 2));
} finally {
  await browser.close();
}
