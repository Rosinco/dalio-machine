import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

// This checks a saved audit artifact, so it neither starts nor rebuilds the app.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const directoryFlag = process.argv.indexOf('--directory');
if (directoryFlag >= 0 && !process.argv[directoryFlag + 1]) throw new Error('--directory requires an audit output path.');
const dir = resolve(directoryFlag >= 0 ? process.argv[directoryFlag + 1] : join(root, 'test-results/cash-flow-backtest-2026-09-12'));
const file = join(dir, 'report.html');
const report = JSON.parse(await readFile(join(dir, 'report-receipt.json'), 'utf8'));
const common = JSON.parse(await readFile(join(dir, 'matched-window-common-scale.json'), 'utf8'));
assert.equal(createHash('sha256').update(await readFile(file)).digest('hex'), report.sha256);
const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox'] });
const checks = [], errors = [], external = [];
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, offline: true, acceptDownloads: true });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (/^https?:/.test(request.url())) external.push(request.url()); });
  const start = performance.now();
  await page.goto(pathToFileURL(file).href, { waitUntil: 'load', timeout: 60000 });
  await page.locator('#company-table tr').first().waitFor();
  const loadMs = Math.round(performance.now() - start);
  const totals = await page.evaluate(() => ({ listings: D.companies.length, rows: D.rows.length, pilot: D.pilotRows.length }));
  assert.deepEqual(totals, { listings: report.listings, rows: report.retrospectiveRows, pilot: report.frozenPilotListings });
  checks.push('Standalone file loads with networking disabled and matching embedded record counts');
  assert.equal(await page.locator('#horizon-table tr').count(), 5);
  assert.ok((await page.locator('#cards').innerText()).includes(report.listings.toLocaleString('en-GB')));
  await page.screenshot({ path: join(dir, 'report-overview.png') });

  await page.getByText('Compare simpler forecasts and learned ranges', { exact: true }).click();
  for (const cohort of ['five', 'matched', 'source']) {
    await page.locator('#comparison-cohort').selectOption(cohort);
    await page.locator('#comparison-horizon').selectOption('1');
    assert.equal(await page.locator('#models tr').count(), cohort === 'matched' ? 8 : 4);
    if (cohort === 'matched') {
      const labels = { naive: 'Last reported cash', flat: 'Weighted flat cash', damped: 'Half-strength trend', linear: 'Current weighted trend' };
      const rows = await page.locator('#models tr').evaluateAll(rows => rows.map(r => Array.from(r.cells).map(c => c.textContent)));
      const number = value => Number(value).toLocaleString('en-GB', { maximumFractionDigits: 2 });
      for (const expected of common.metrics.filter(r => r.horizon === 1)) {
        const row = rows.find(row => row[0] === `${labels[expected.model]} · ${expected.window}y history`);
        assert.ok(row, expected.model);
        assert.equal(row[2], number(expected.medianNormalizedAbsoluteError));
        assert.equal(row[4], number(expected.meanNormalizedWidth));
        assert.equal(row[5], number(expected.meanNormalizedIntervalScore));
      }
    }
    const expected = await page.evaluate(cohort => {
      const rows = cohort === 'matched' ? D.matched : cohort === 'source' ? D.sameSource : D.metrics;
      const m = rows.find(r => r.model === 'linear' && r.window === 5 && r.horizon === 1);
      return { count: fmt(m.calibratedCount, 0), raw: pct(m.rawOnCalibratedCoverage), learned: pct(m.calibratedCoverage) };
    }, cohort);
    const cells = await page.locator('#learned tr').evaluateAll(rows => rows.map(r => Array.from(r.cells).map(c => c.textContent)));
    assert.equal(cells[0][1], expected.count); assert.equal(cells[1][1], expected.count);
    assert.equal(cells[0][2], expected.raw); assert.equal(cells[1][2], expected.learned);
  }
  await page.locator('#comparison-horizon').selectOption('5');
  assert.match(await page.locator('#learned').innerText(), /Unavailable/);
  checks.push('All three comparison cohorts select their own matching learned/default rows; matched model values equal the independent common-scale artifact; horizon five stays unavailable');

  for (const id of ['696', '197', '3']) {
    await page.locator(`[data-case="${id}"]`).click();
    const expected = await page.evaluate(id => { const r = D.rows.find(r => r[0] === id && r[1] === 2024 && r[3] === 1); return { mid: fmt(r[4]), actual: fmt(r[5]), state: status(r), name: names.get(id).name }; }, id);
    assert.match(await page.locator('#detail-title').innerText(), new RegExp(expected.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    const text = await page.locator('#detail-text').innerText();
    assert.ok(text.includes(expected.mid)); assert.ok(text.includes(expected.actual)); assert.ok(text.includes(expected.state));
    assert.ok(await page.locator('#chart circle').count() >= 1);
    assert.equal(await page.locator('#chart polygon').count(), 1);
    await page.locator('#detail').screenshot({ path: join(dir, `report-case-${id}.png`) });
  }
  checks.push('Stora, SCA and ABB cases show the selected stored forecast, actual, outcome and signed fan');

  await page.locator('#search').fill('Stora Enso');
  await page.locator('#horizon').selectOption('1');
  await page.locator('#outcome').selectOption('above');
  assert.ok(await page.locator('#company-table button[data-row]').count() > 0);
  assert.ok((await page.locator('#company-table .pill').allTextContents()).every(value => value === 'above'));
  await page.locator('#diagnostic').selectOption('forecast_actual_sign_flip');
  const exportRows = await page.evaluate(() => filtered.length);
  assert.ok(exportRows > 0);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadPromise;
  const exportPath = join(dir, 'report-filtered-forecasts.csv');
  await download.saveAs(exportPath);
  const csv = await readFile(exportPath, 'utf8');
  assert.equal(csv.split(/\r?\n/).filter(Boolean).length, exportRows + 1);
  assert.match(csv, /original_sensitivity/); assert.doesNotMatch(csv, /undefined|NaN/);
  checks.push('Company, horizon, outcome and diagnostic filters work; CSV download contains every filtered row offline');

  await page.locator('#search').fill('');
  await page.locator('#diagnostic').selectOption('all');
  await page.locator('#horizon').selectOption('5');
  await page.locator('#band').selectOption('learned');
  await page.locator('#outcome').selectOption('missing');
  assert.ok((await page.locator('#matches').innerText()).includes(`${report.listings.toLocaleString('en-GB')} listings with no eligible learned-range result`));
  assert.ok(await page.locator('#export').isDisabled());
  assert.ok(await page.locator('#diagnostic').isDisabled());
  assert.match(await page.locator('#company-table').innerText(), /not classified as a miss/);
  await page.locator('#outcome').selectOption('all');
  assert.ok(await page.locator('#export').isEnabled());
  assert.ok((await page.locator('#company-table .pill').allTextContents()).every(value => value === 'unavailable'));
  checks.push('Unavailable learned results remain missing; export is disabled for the missing-listing view and restored afterwards');

  await page.locator('#pilot-search').fill('Stora Enso');
  assert.match(await page.locator('#pilot-table').innerText(), /Stora Enso/);
  await page.locator('#pilot-search').fill('');
  await page.locator('#pilot-status').selectOption('outcome_unavailable');
  assert.ok(await page.locator('#pilot-table tr').count() > 0);
  assert.doesNotMatch(await page.locator('#pilot-table').innerText(), /NaN|undefined/);
  checks.push('Frozen-cohort search and unavailable-outcome filter work without inventing zero outcomes');

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    const sizes = await page.evaluate(() => ({ body: document.documentElement.scrollWidth, width: innerWidth }));
    assert.ok(sizes.body <= sizes.width + 1, JSON.stringify(sizes));
  }
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({ path: join(dir, 'report-mobile.png') });
  checks.push('Desktop and narrow viewport avoid horizontal page overflow; wide tables scroll inside their containers');
  assert.deepEqual(external, []); assert.deepEqual(errors, []);
  const result = { status: 'pass', reportSha256: report.sha256, loadMs, totals, checks, external, errors };
  await writeFile(join(dir, 'report-browser-validation.json'), JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
} finally { await browser.close(); }
