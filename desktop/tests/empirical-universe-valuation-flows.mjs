import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const method = 'empirical-cash-starter-v3';
const calibrationId = 'cash-uncertainty-2026-09-12-v1';
const chartLabel = 'Cash flow over time: history and forecast scenarios';
const savedTitle = 'TEST FIXTURE — empirical Stora crisis and deliberate gap';
const savedNote = 'Independent crisis assumptions; retain signed cash and original baseline.';
const cash = [705, -181, -561, 840, 1027];
const scale = cash.reduce((sum, amount) => sum + Math.abs(amount), 0) / 5;
const mean = cash.reduce((sum, amount) => sum + amount, 0) / 5;
const dispersion = Math.sqrt(cash.reduce((sum, amount) => sum + (amount - mean) ** 2, 0) / 5) / scale;
const group = dispersion < .25 ? 'low' : dispersion < .75 ? 'medium' : 'high';
const near = (actual, expected, label = 'amount') => assert.ok(Math.abs(Number(actual) - expected) <= Math.max(1e-6, Math.abs(expected) * 1e-10), `${label}: ${actual} != ${expected}`);

async function openCompany(page, name, id) {
  if (await page.locator(`[data-valuation-company="${id}"]`).count()) return;
  await page.getByLabel('Search companies or countries').fill(name);
  await page.locator(`[data-search-listing="${id}"]`).click();
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function reopen(page, id = '696') {
  await page.reload();
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function draftState(page, id = '696') {
  return page.evaluate(id => {
    const items = Object.keys(localStorage).filter(key => key.startsWith(`macro-atlas-valuation-draft-v1:${id}:`))
      .map(key => ({ key, saved: JSON.parse(localStorage.getItem(key)) }));
    return items.sort((a, b) => a.saved.created.localeCompare(b.saved.created)).at(-1) ?? null;
  }, id);
}

async function revisions(page) {
  return page.evaluate(() => JSON.parse(localStorage.getItem('macro-atlas-valuations-v1') ?? '{"items":[]}').items);
}

async function setFixture(page, fixture) {
  await page.evaluate(fixture => {
    const key = Object.keys(localStorage).filter(key => key.startsWith('macro-atlas-valuation-draft-v1:696:')).at(-1);
    const saved = JSON.parse(localStorage.getItem(key));
    saved.draft = fixture;
    localStorage.setItem(key, JSON.stringify(saved));
  }, fixture);
  await reopen(page);
}

async function inputs(page) {
  if (await page.getByLabel('Mid year 1 cash payment', { exact: true }).count() === 0) {
    await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  }
}

async function settings(page) {
  const panel = page.locator('.valuation-history-settings');
  if (await panel.getAttribute('open') === null) await panel.locator('summary').click();
  return panel;
}

async function defaults(page) {
  await settings(page);
  await page.getByRole('button', { name: 'Use empirical defaults', exact: true }).click();
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await page.locator(`[data-valuation-starter="${method}"]`).waitFor();
  await inputs(page);
}

async function annualRanges(page) {
  const chart = page.locator('.cash-flow-forecast');
  const table = chart.locator('.cash-flow-range-table');
  if (await table.getAttribute('open') === null) await table.locator('summary').click();
  return chart.locator('[data-year]');
}

function csvRecords(csv) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const c = csv[i];
    if (c === '"') {
      if (quoted && csv[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted;
    } else if (!quoted && c === ',') { row.push(cell); cell = ''; }
    else if (!quoted && c === '\r' && csv[i + 1] === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; i++; }
    else cell += c;
  }
  row.push(cell); rows.push(row);
  const [header, ...data] = rows;
  return data.filter(values => values.length === header.length).map(values => Object.fromEntries(header.map((key, i) => [key, values[i]])));
}

async function exportRows(page, project, native, suffix) {
  let csv;
  if (native) {
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const status = page.getByRole('status').filter({ hasText: 'Saved to ' });
    await status.waitFor();
    csv = await readFile((await status.innerText()).replace(/^Saved to /, ''), 'utf8');
  } else {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const download = await pending, path = resolve(project, `test-results/empirical-valuation-${suffix}.csv`);
    await download.saveAs(path); csv = await readFile(path, 'utf8');
  }
  return csvRecords(csv);
}

async function assertDefault(page, calibration) {
  await page.locator(`[data-valuation-starter="${method}"][data-valuation-ready="true"]`).waitFor();
  await inputs(page);
  const d = (await draftState(page)).saved.draft;
  assert.equal(d.starterOrigin.projection, 'latest');
  assert.equal(d.starterOrigin.historyYears, 5);
  assert.equal(d.starterOrigin.rangeMode, 'historical');
  assert.equal(d.starterOrigin.calibrationId, calibrationId);
  assert.deepEqual(calibration.covidTreatment.omittedTargetYears, []);
  assert.equal(calibration.covidTreatment.historicalShocksRemoved, false);
  for (let year = 1; year <= 10; year++) {
    const factor = calibration.calibrationByModelHorizon.naive[String(Math.min(year, 4))].cashDispersion[group].factor;
    const half = scale * (factor + Math.max(0, year - 4) * .1);
    near(d.scenarios.mid.cashFlows[year - 1], 705, `mid ${year}`);
    near(d.scenarios.low.cashFlows[year - 1], 705 - half, `low ${year}`);
    near(d.scenarios.high.cashFlows[year - 1], 705 + half, `high ${year}`);
  }
  const rows = await annualRanges(page);
  assert.equal(await rows.count(), 10);
  for (let year = 1; year <= 10; year++) assert.equal(await page.locator(`.cash-flow-forecast [data-year="${year}"]`).getAttribute('data-range-kind'), year <= 4 ? 'historical' : 'assumed-tail');
  const chart = page.locator('.cash-flow-forecast');
  assert.equal(await chart.getAttribute('data-range-group'), group);
  assert.equal(await chart.getAttribute('data-calibration-id'), calibrationId);
  await page.getByRole('img', { name: chartLabel, exact: true }).locator('canvas').waitFor();
  for (const label of ['Discounted cash flow: low, mid and high scenarios', 'Cumulative NPV: low, mid and high scenarios']) {
    await page.getByRole('img', { name: label, exact: true }).locator('canvas').waitFor();
  }
}

async function legacyMigrations(page, project, checks) {
  const heading = await page.locator('.valuation-heading h1').innerText();
  const v1 = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-weighted-cash-v1.json'), 'utf8'));
  const v2 = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-weighted-cash-v2.json'), 'utf8'));
  v1.title = `${heading} — weighted cash starter`; v2.title = `${heading} — weighted cash trend starter`;
  for (const fixture of [v1, v2]) {
    fixture.investment = 2500;
    await setFixture(page, fixture);
    await page.locator(`[data-valuation-starter="${method}"]`).waitFor();
    near((await draftState(page)).saved.draft.investment, 2500);
    const backup = (await revisions(page)).find(row => JSON.stringify(row.draft) === JSON.stringify(fixture));
    assert.ok(backup, 'An exact prior draft must be saved before upgrade');
    await page.locator('.valuation-saved button').filter({ hasText: fixture.title }).first().click();
    await reopen(page);
    assert.equal((await draftState(page)).saved.draft.starterOrigin.id, fixture.starterOrigin.id);
    assert.equal((await draftState(page)).saved.draft.researchAutofillDisabled, true);
  }
  checks.push('Untouched v1/v2 defaults upgrade with exact saved backups and preserved investment; explicitly restored backups do not migrate again');

  for (const fixture of [v1, v2]) {
    const edited = structuredClone(fixture);
    edited.title = `TEST CUSTOM ${fixture.starterOrigin.id}`;
    edited.notes.macro = 'Preserve my company-specific assumption';
    edited.scenarios.mid.cashFlows[0] = null;
    edited.scenarios.high.cashFlows[0] = 1234;
    await setFixture(page, edited);
    assert.deepEqual((await draftState(page)).saved.draft, edited);
    await reopen(page);
    assert.deepEqual((await draftState(page)).saved.draft, edited);
  }
  checks.push('Custom v1/v2 cash, notes, titles and deliberate blanks survive restart without changing their model');

  await page.addInitScript(() => {
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === 'macro-atlas-valuations-v1' && localStorage.getItem('atlas-test-revision-failure') === '1') throw new DOMException('Test revision quota failure', 'QuotaExceededError');
      if (key.startsWith('macro-atlas-valuation-draft-v1:696:') && localStorage.getItem('atlas-test-replacement-failure') === '1' && JSON.parse(value).draft.starterOrigin?.id === 'empirical-cash-starter-v3') throw new DOMException('Test replacement quota failure', 'QuotaExceededError');
      return original.call(this, key, value);
    };
  });
  await page.evaluate(() => localStorage.setItem('atlas-test-revision-failure', '1'));
  await setFixture(page, v2);
  assert.equal((await draftState(page)).saved.draft.starterOrigin.id, 'weighted-cash-starter-v2');
  assert.deepEqual((await draftState(page)).saved.draft, v2);
  assert.match(await page.locator('.valuation-workspace').innerText(), /preserv|backup|saved|quota/i);
  await page.evaluate(() => localStorage.removeItem('atlas-test-revision-failure'));
  await reopen(page);
  await page.locator(`[data-valuation-starter="${method}"]`).waitFor();
  checks.push('A failed backup write retains the original legacy draft; migration succeeds only after storage can preserve it');

  await page.evaluate(() => localStorage.setItem('atlas-test-replacement-failure', '1'));
  await setFixture(page, v2);
  assert.deepEqual((await draftState(page)).saved.draft, v2);
  assert.equal(await page.locator('.valuation-workspace').getAttribute('data-valuation-starter'), 'weighted-cash-starter-v2');
  assert.ok((await revisions(page)).some(row => JSON.stringify(row.draft) === JSON.stringify(v2)));
  await page.evaluate(() => localStorage.removeItem('atlas-test-replacement-failure'));
  await reopen(page);
  await page.locator(`[data-valuation-starter="${method}"]`).waitFor();
  checks.push('A failed replacement write also retains the active legacy draft after preserving its backup');
}

async function assertPreserved(page) {
  await page.locator(`[data-valuation-company="696"][data-valuation-starter="${method}"][data-valuation-ready="false"]`).waitFor();
  const d = (await draftState(page)).saved.draft;
  assert.equal(d.title, savedTitle);
  assert.equal(d.scenarios.mid.cashFlows[0], null);
  assert.equal(d.scenarios.high.cashFlows[0], 1234);
  assert.equal(d.crisis.enabled, true);
  assert.equal(d.crisis.shockPercent, 50);
  assert.equal(d.crisis.startYear, 2);
  assert.equal(d.crisis.durationYears, 2);
  assert.equal(d.crisis.recoveryYears, 2);
  assert.equal(d.crisis.extraAnnualCashCost, 20);
  assert.equal(d.crisis.discountRate, 12);
  assert.equal(d.crisis.terminalEquity, 500);
  assert.equal(d.crisis.rationale, savedNote);
  assert.equal(await page.getByRole('button', { name: 'Export calculations', exact: true }).isDisabled(), true);
}

export async function restoreEmpiricalUniverseValuation(page) {
  await openCompany(page, 'Stora Enso R', '696');
  await assertPreserved(page);
  await openCompany(page, 'Holmen', '102');
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
}

export async function empiricalUniverseValuationFlows(page, project, { native = false } = {}) {
  const checks = [], calibration = JSON.parse(await readFile(resolve(project, 'src/data/cash-uncertainty-2026-09-12.json'), 'utf8'));
  await openCompany(page, 'Stora Enso R', '696');
  await assertDefault(page, calibration);
  const original = (await draftState(page)).saved.draft;
  const originalRows = await exportRows(page, project, native, 'original');
  assert.equal(originalRows.filter(row => ['low', 'mid', 'high'].includes(row.scenario)).length, 33);
  assert.ok(originalRows.every(row => row.starter_method === method));
  assert.ok(originalRows.every(row => Object.entries(row).some(([key, value]) => key.endsWith('calibration_id') && value === calibrationId)));
  await page.getByLabel('Equity market value', { exact: true }).fill('');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  await page.getByRole('img', { name: chartLabel, exact: true }).locator('canvas').waitFor();
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  await page.getByLabel('Equity market value', { exact: true }).fill(String(original.marketCap));
  await page.locator('[data-valuation-ready="true"]').waitFor();
  checks.push('Default Stora uses independently calculated latest-cash mids and model-specific historical Year1–4 ranges with explicitly assumed Year5–10 tails; cash chart survives missing price and CSV retains calibration identity');

  await legacyMigrations(page, project, checks);
  for (const change of ['weights', 'window', 'flat']) {
    await defaults(page); await settings(page);
    if (change === 'weights') {
      await page.getByLabel('Year 1 weight', { exact: true }).fill('35');
      await page.getByLabel('Year 2 weight', { exact: true }).fill('20');
    } else if (change === 'window') await page.getByLabel('Historical years', { exact: true }).selectOption('10');
    else await page.getByLabel('Projection', { exact: true }).selectOption('flat');
    await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
    await annualRanges(page);
    for (let year = 1; year <= 10; year++) assert.equal(await page.locator(`.cash-flow-forecast [data-year="${year}"]`).getAttribute('data-range-kind'), 'percentage');
  }
  checks.push('Custom weights, ten-year windows and unsupported flat-weighted mids expose percentage assumptions instead of borrowing empirical labels');

  await defaults(page); await settings(page);
  await page.getByLabel('Projection', { exact: true }).selectOption('trend');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await annualRanges(page);
  const trend = (await draftState(page)).saved.draft;
  const weights = [.3, .25, .2, .15, .1], x = [0, -1, -2, -3, -4];
  const xMean = x.reduce((sum, value, i) => sum + value * weights[i], 0);
  const yMean = cash.reduce((sum, value, i) => sum + value * weights[i], 0);
  const slope = x.reduce((sum, value, i) => sum + weights[i] * (value - xMean) * (cash[i] - yMean), 0) / x.reduce((sum, value, i) => sum + weights[i] * (value - xMean) ** 2, 0);
  const intercept = yMean - slope * xMean;
  for (let year = 1; year <= 10; year++) {
    const factor = calibration.calibrationByModelHorizon.linear[String(Math.min(year, 4))].cashDispersion[group].factor;
    const half = scale * (factor + Math.max(0, year - 4) * .1);
    near(trend.scenarios.mid.cashFlows[year - 1], intercept + slope * year, `trend mid ${year}`);
    near(trend.scenarios.low.cashFlows[year - 1], intercept + slope * year - half, `trend low ${year}`);
    near(trend.scenarios.high.cashFlows[year - 1], intercept + slope * year + half, `trend high ${year}`);
    assert.equal(await page.locator(`.cash-flow-forecast [data-year="${year}"]`).getAttribute('data-range-kind'), year <= 4 ? 'historical' : 'assumed-tail');
  }
  checks.push('The supported weighted historical trend matches an independent regression and uses its own linear-model calibration factors');

  await defaults(page); await settings(page);
  await page.getByLabel('Later-year widening (% of cash scale)', { exact: true }).fill('15');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  const widerTail = (await draftState(page)).saved.draft;
  for (let year = 1; year <= 10; year++) {
    const factor = calibration.calibrationByModelHorizon.naive[String(Math.min(year, 4))].cashDispersion[group].factor;
    near(widerTail.scenarios.high.cashFlows[year - 1], 705 + scale * (factor + Math.max(0, year - 4) * .15));
  }
  await defaults(page); await settings(page);
  await page.getByLabel('Uncertainty range', { exact: true }).selectOption('percentage');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await annualRanges(page);
  const assumed = (await draftState(page)).saved.draft;
  for (let year = 1; year <= 10; year++) {
    near(assumed.scenarios.low.cashFlows[year - 1], 705 - 705 * year / 10);
    near(assumed.scenarios.high.cashFlows[year - 1], 705 + 705 * year / 10);
    assert.equal(await page.locator(`.cash-flow-forecast [data-year="${year}"]`).getAttribute('data-range-kind'), 'percentage');
  }
  checks.push('Later-year widening is editable in absolute historical cash-scale units; explicit percentage mode uses only the stated assumed sensitivities');

  await defaults(page);
  const beforeCrisis = structuredClone((await draftState(page)).saved.draft.scenarios);
  await page.getByLabel('Enable crisis scenario', { exact: true }).check();
  for (const [label, value] of [['Crisis cash reduction (%)', 50], ['Crisis start year', 2], ['Crisis duration (years)', 2], ['Crisis recovery (years)', 2], ['Crisis extra annual cash cost', 20], ['Crisis required return', 12], ['Crisis final equity sale', 500]]) {
    await page.getByLabel(label, { exact: true }).fill(String(value));
  }
  await page.getByLabel('Crisis assumptions and evidence', { exact: true }).fill(savedNote);
  assert.deepEqual((await draftState(page)).saved.draft.scenarios, beforeCrisis);
  const crisisRows = (await exportRows(page, project, native, 'crisis')).filter(row => row.scenario === 'crisis');
  assert.equal(crisisRows.length, 11);
  const expectedCash = [705, 332.5, 332.5, 518.75, 705, 705, 705, 705, 705, 705];
  let pv = 0;
  for (let year = 1; year <= 10; year++) {
    const row = crisisRows.find(row => Number(row.year) === year);
    near(row.equity_cash_payment_m, expectedCash[year - 1], `crisis cash ${year}`);
    const discounted = expectedCash[year - 1] / 1.12 ** year;
    near(row.discounted_payment_m, discounted, `crisis discounted ${year}`);
    pv += discounted;
    near(row.cumulative_npv_m, pv - original.marketCap, `crisis NPV ${year}`);
    assert.ok(Object.entries(row).filter(([key]) => /(?:scenario|crisis)_probability/.test(key)).every(([, value]) => value === '' || /unassigned|not assigned/i.test(value)));
  }
  near(crisisRows.find(row => row.year === '10').cumulative_npv_with_sale_m, pv + 500 / 1.12 ** 10 - original.marketCap);
  assert.match(await page.locator('[data-crisis-enabled="true"]').innerText(), /probability|probabilities/i);
  for (const viewport of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.locator('.valuation-crisis').screenshot({ path: resolve(project, `test-results/empirical-crisis-${viewport.width}.png`) });
    await page.locator('.cash-flow-forecast').screenshot({ path: resolve(project, `test-results/empirical-cash-range-${viewport.width}.png`) });
  }
  await page.setViewportSize({ width: 1500, height: 960 });
  checks.push('A separate editable crisis path applies signed shock and fading extra costs, exports independently reconciled DCF/NPV, assigns no probability and leaves the original three scenarios unchanged');

  await page.getByLabel('Valuation study title').fill(savedTitle);
  await page.getByLabel('High year 1 cash payment', { exact: true }).fill('1234');
  await page.getByLabel('Mid year 1 cash payment', { exact: true }).fill('');
  await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
  await reopen(page); await assertPreserved(page);
  checks.push('Crisis settings, custom cash, notes and deliberate gaps survive reload and remain available for full-process restart verification');

  for (const [name, id] of [['Nordea Bank', '159'], ['Fairfax Financial Holdings Ltd', '14473']]) {
    await openCompany(page, name, id);
    await page.locator('[data-valuation-ready="false"]').waitFor();
    const manual = (await draftState(page, id)).saved.draft;
    assert.ok(Object.values(manual.scenarios).every(s => s.cashFlows.every(value => value === null)));
    assert.match(await page.locator('.valuation-workspace').innerText(), /bank|insur|capital|financial/i);
    await inputs(page);
    for (const label of ['Low', 'Mid', 'High']) {
      await page.getByLabel(`${label} required return`, { exact: true }).fill('10');
      await page.getByLabel(`${label} final equity sale`, { exact: true }).fill('0');
      for (let year = 1; year <= 10; year++) await page.getByLabel(`${label} year ${year} cash payment`, { exact: true }).fill('100');
    }
    if (!(await draftState(page, id)).saved.draft.marketCap) {
      await page.getByLabel('Equity market value', { exact: true }).fill('1000');
      await page.getByLabel('Valuation price date', { exact: true }).fill(manual.valuationDate);
    }
    await page.locator('[data-valuation-ready="true"]').waitFor();
    await page.getByRole('img', { name: 'Cumulative NPV: low, mid and high scenarios', exact: true }).locator('canvas').waitFor();
  }
  checks.push('Banks and insurers retain manual capital/distribution workflows; entered forecasts still produce the common charts');

  await openCompany(page, 'Atrium Ljungberg', '20');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  const negative = (await draftState(page, '20')).saved.draft;
  assert.ok(negative.scenarios.mid.cashFlows.every(value => value === -1410));
  assert.equal(negative.scenarios.mid.terminalEquity, 0);
  assert.ok(negative.scenarios.low.cashFlows.every((value, index) => value <= negative.scenarios.mid.cashFlows[index]));
  await openCompany(page, 'ABB', '3');
  await annualRanges(page);
  assert.equal(await page.locator('.cash-flow-forecast [data-range-kind="historical"]').count(), 0, 'Native calibration does not validate archived quote-currency histories');
  await openCompany(page, 'Logistea A', '167');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  checks.push('Negative property cash retains ordered signed ranges and zero assumed terminal sale; quote-currency and missing-history cases receive no unsupported historical-range label');

  for (const [name, id, study] of [['Holmen', '102', 'holmen-2026-09-11-v1'], ['SCA', '197', 'sca-2026-09-12-v1']]) {
    const prior = await draftState(page, id);
    await openCompany(page, name, id);
    assert.equal(await page.locator('.valuation-workspace').getAttribute('data-valuation-study'), study);
    if (prior) assert.deepEqual((await draftState(page, id)).saved.draft, prior.saved.draft);
  }
  checks.push('Reviewed Holmen/SCA studies and existing user revisions retain their original data and calculations');

  await openCompany(page, 'Stora Enso R', '696'); await assertPreserved(page);
  for (const viewport of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
    await page.screenshot({ path: resolve(project, `test-results/empirical-valuation-${viewport.width}.png`), fullPage: true });
  }
  await page.setViewportSize({ width: 1500, height: 960 });
  await restoreEmpiricalUniverseValuation(page);
  checks.push('Valuation and crisis controls fit desktop and mobile viewports without horizontal page overflow');
  return { checks };
}

async function standalone() {
  const { chromium } = await import('playwright');
  const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const base = process.env.ATLAS_URL ?? 'http://127.0.0.1:1420';
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const external = [], errors = [];
  const context = await browser.newContext({ viewport: { width: 1500, height: 960 }, acceptDownloads: true });
  const csp = JSON.parse(await readFile(resolve(project, 'src-tauri/tauri.conf.json'), 'utf8')).app.security.csp;
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (url === `${base}/`) {
      const response = await route.fetch();
      return route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': csp } });
    }
    if (url.startsWith(base) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
    external.push(url); return route.abort();
  });
  const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base);
    await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
    await page.locator('[data-business-ready="true"]').waitFor();
    const result = await empiricalUniverseValuationFlows(page, project);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await writeFile(resolve(project, 'test-results/empirical-valuation-browser-report.json'), JSON.stringify({ status: 'passed', ...result, external, errors }, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } catch (error) {
    await page.screenshot({ path: resolve(project, 'test-results/empirical-valuation-failure.png'), fullPage: true }).catch(() => {});
    console.error(await page.locator('body').innerText().catch(() => 'Page unavailable'));
    throw error;
  } finally { await browser.close(); }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await standalone();
