import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { selectObservatory, openListPanel } from './workspace-navigation.mjs';

const key = 'macro-atlas-company-lists-v2';
const kpis = { score: 'normal_quality_value_score', discountRank: 'normal_discount_rank', quality: 'normal_quality_rank' };
const rawColumns = { discount: 'normal_npv_5y_percent', roce: 'normal_roce_median', marginFloor: 'normal_ebit_margin_min', cfoGrowth: 'normal_cfo_growth', netDebtEbitda: 'net_debt_ebitda' };
const fiveYearKpi = 'normal_npv_5y_percent';
const secondarySort = view => [{ columnId: view.columns.find(column => column.kpiId === fiveYearKpi).id, direction: 'desc' }];
const protectedValues = {
  'macro-atlas-valuation-draft-v1:ranking-test': JSON.stringify({ version: 1, title: 'TEST FIXTURE — keep authored forecasts', cash: [75, null, -20], purchaseMargin: 30 }),
  'macro-atlas-valuations-v1': JSON.stringify({ version: 1, items: [{ id: 'ranking-test-revision', notes: 'TEST FIXTURE — prior revision' }] }),
  'macro-atlas-company-notes-v1:ranking-test': 'TEST FIXTURE — prior notes stay unchanged',
};
const list = page => page.locator('[data-company-list-ready="true"]');
const preferences = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
const bytes = page => page.evaluate(key => localStorage.getItem(key), key);
const protectedStorage = page => page.evaluate(keys => Object.fromEntries(keys.map(key => [key, localStorage.getItem(key)])), Object.keys(protectedValues));
const readJson = async path => JSON.parse(await readFile(path, 'utf8'));
const near = (actual, expected, message) => {
  if (expected === null) assert.equal(actual, null, message);
  else assert.ok(typeof actual === 'number' && Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-8 * Math.max(1, Math.abs(expected)), `${message}: ${actual} != ${expected}`);
};

function audit() {
  window.__rankingProtectedWrites = [];
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith('macro-atlas-valuation') || String(args[0]).startsWith('macro-atlas-company-notes'))) window.__rankingProtectedWrites.push({ method, key: args[0] ?? null });
      return original.apply(this, args);
    };
  }
}
async function openLists(page) {
  await selectObservatory(page, 'companies');
  await page.getByLabel('Company lists', { exact: true }).click();
  await list(page).waitFor();
}
async function waitMatches(page, expected) {
  await page.waitForFunction(expected => Number(document.querySelector('[data-company-list-ready="true"]')?.getAttribute('data-company-list-matches')) === expected, expected);
  await page.locator('.company-list-status').filter({ hasText: /Opening .*selected KPI datasets/ }).waitFor({ state: 'detached' });
}
async function assertPolicy(page, cohortSize) {
  const policy = page.locator('[data-company-list-quality-value-policy]');
  await page.locator('[data-company-list-quality-value-policy][data-quality-value-ready="true"]').waitFor();
  assert.equal(Number(await policy.getAttribute('data-quality-value-peers')), cohortSize);
  const text = await policy.innerText();
  assert.match(text, /60%[\s\S]*five.year[\s\S]*40%[\s\S]*quality/i);
  assert.match(text, /years? 1[–-]5|five.year/i);
  const fiveYearPolicy = await page.locator('[data-company-list-five-year-valuation-policy]').innerText();
  assert.match(fiveYearPolicy, /years 1[–-]5/);
  assert.match(fiveYearPolicy, /no terminal|without terminal|terminal.*0/i);
  assert.match(text, /fixed/i);
  assert.match(text, /missing|unranked/i);
  assert.match(await page.locator('[data-company-list-normal-year-policy]').innerText(), /2020–2023/);
}
async function tableValues(page) {
  return page.locator('[data-company-listing]').evaluateAll((rows, { kpis, rawColumns }) => rows.map(row => {
    const value = selector => { const raw = row.querySelector(selector)?.getAttribute('data-company-list-value'); return raw === null || raw === undefined || raw === '' ? null : Number(raw); };
    return { id: row.getAttribute('data-company-listing'), legacyHalf: value('[data-company-list-column="normal_npv_half"]'), ...Object.fromEntries(Object.entries(kpis).map(([name, kpi]) => [name, value(`[data-company-list-kpi="${kpi}"]`)])), raw: Object.fromEntries(Object.entries(rawColumns).map(([name, id]) => [name, value(name === 'discount' ? `[data-company-list-kpi="${id}"]` : `[data-company-list-column="${id}"]`)])) };
  }), { kpis, rawColumns });
}
async function allRows(page, expected) {
  await page.getByLabel('Company list rows per page', { exact: true }).selectOption('250');
  await page.waitForFunction(expected => document.querySelectorAll('[data-company-listing]').length === expected, expected);
  return tableValues(page);
}
async function addRanking(page) {
  await page.getByLabel('Company list rows per page', { exact: true }).selectOption('50');
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const picker = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  for (const kpi of [...Object.values(kpis), fiveYearKpi]) {
    await picker.getByLabel('Search KPIs', { exact: true }).fill('');
    await picker.locator(`[data-company-list-kpi-option="${kpi}"]`).click();
    assert.equal(await picker.getByLabel('KPI time period', { exact: true }).inputValue(), 'latest');
    assert.equal(await picker.getByLabel('KPI calculation', { exact: true }).inputValue(), 'latest');
    await picker.getByRole('button', { name: 'Add column', exact: true }).click();
  }
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  const columns = (await preferences(page)).columns;
  const combined = columns.find(column => column.kpiId === kpis.score);
  const fiveYear = columns.find(column => column.kpiId === fiveYearKpi);
  assert.ok(fiveYear);
  assert.ok(combined);
  await openListPanel(page, 'tools');
  const sortControls = page.locator('.company-list-sort-controls');
  if (await sortControls.getAttribute('open') === null) await sortControls.locator('summary').click();
  await page.getByLabel('Sort priority 1', { exact: true }).selectOption(combined.id);
  await page.getByLabel('Sort direction 1', { exact: true }).selectOption('desc');
  await page.getByRole('button', { name: 'Add sort priority', exact: true }).click();
  await page.getByLabel('Sort priority 2', { exact: true }).selectOption(fiveYear.id);
  await page.getByLabel('Sort direction 2', { exact: true }).selectOption('desc');
  await page.getByRole('button', { name: 'Close list tools', exact: true }).click();
  assert.equal(await page.locator(`[data-company-list-header="${combined.id}"]`).getAttribute('aria-sort'), 'descending');
  assert.deepEqual((await preferences(page)).features.secondarySorts, secondarySort({ columns }));
  return columns;
}
async function saveView(page, name) {
  await page.getByRole('button', { name: 'Save view', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Save company list view', exact: true });
  await dialog.getByLabel('View name', { exact: true }).fill(name);
  await dialog.getByRole('button', { name: 'Save current view', exact: true }).click();
  await page.waitForFunction(({ key, name }) => JSON.parse(localStorage.getItem(key)).savedViews.some(view => view.name === name), { key, name });
  return (await preferences(page)).savedViews.find(view => view.name === name);
}
async function assertPreserved(page, originalViews, features) {
  const current = await preferences(page);
  for (const view of originalViews) {
    assert.deepEqual(current.savedViews.find(saved => saved.id === view.id), view);
    assert.deepEqual(current.features.viewFeatures[view.id], features.viewFeatures[view.id]);
  }
  assert.deepEqual(current.features.watchlists, features.watchlists);
  assert.deepEqual(current.features.comparisonIds, features.comparisonIds);
  assert.deepEqual(await protectedStorage(page), protectedValues);
  assert.deepEqual(await page.evaluate(() => window.__rankingProtectedWrites), []);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}

export async function normalYearRankingFlows(page, project, { native = false } = {}) {
  const checks = [];
  const startedAt = Date.now();
  const timings = [];
  const stage = name => { const elapsedMs = Date.now() - startedAt; const previous = timings.at(-1)?.elapsedMs ?? 0; timings.push({ name, elapsedMs, stageMs: elapsedMs - previous }); console.log(JSON.stringify(timings.at(-1))); };
  page.setDefaultTimeout(60000);
  const expectedPath = resolve(project, 'tests/fixtures/normal-year-five-year-ranking.json');
  const expected = await readJson(expectedPath);
  assert.equal(expected.policyId, 'normal-quality-watch-npv5-60-40-v2');
  assert.deepEqual(expected.assumptions.discountYears, [1, 2, 3, 4, 5]);
  assert.equal(expected.assumptions.discountRatePercent, 10);
  assert.equal(expected.assumptions.terminalValue, 0);
  assert.equal(expected.assumptions.npvWeight, .6);
  assert.equal(expected.assumptions.qualityWeight, .4);
  const priorExpected = await readJson(resolve(project, 'tests/fixtures/normal-year-ranking.json'));
  const [strictRecord, watchRecord] = await Promise.all(['normal-years-quality-value-2026-09-28.json', 'normal-years-quality-watch-2026-09-28.json'].map(name => readJson(resolve(project, 'research/screens', name))));
  const priorRanking = await readJson(resolve(project, 'research/screens/normal-years-ranking-verification-2026-09-29.json'));
  const watch = { ...watchRecord.view, id: 'fixture-unranked-watch', name: 'TEST historical 24-column watch' };
  const strict = { ...strictRecord.view, id: 'fixture-unranked-strict', name: 'TEST historical 24-column discount' };
  const originalViews = [...priorRanking.savedViews, strict, watch];
  assert.equal(expected.qualityCount, 248); assert.equal(expected.cohortSize, 180); assert.equal(expected.unrankedCount, 68); assert.equal(expected.strictCount, 0);
  await openLists(page);
  stage('Initial company lists ready');
  const features = { version: 1, watchlists: [{ id: 'default', name: 'Watchlist', listingIds: ['102', '999999999'] }], activeWatchlistId: 'default', comparisonIds: ['102'], secondarySorts: [], density: 'comfortable', viewFeatures: { ...Object.fromEntries(originalViews.map(view => [view.id, { secondarySorts: [], density: 'comfortable', activeWatchlistId: 'default' }])), ...priorRanking.savedViewFeatures } };
  await page.evaluate(({ key, watch, originalViews, features, protectedValues }) => {
    localStorage.setItem(key, JSON.stringify({ version: 2, columns: watch.columns, filters: watch.filters, sort: watch.sort, features, watchlistIds: features.watchlists[0].listingIds, savedViews: originalViews }));
    for (const [key, value] of Object.entries(protectedValues)) localStorage.setItem(key, value);
  }, { key, watch, originalViews, features, protectedValues });
  await page.context().addInitScript(audit);
  await page.reload(); await openLists(page); await waitMatches(page, expected.qualityCount);
  const oldIds = (await allRows(page, expected.qualityCount)).map(row => row.id);
  assert.deepEqual(oldIds, watchRecord.independentEvidence.ids);
  stage('Original 248 rows verified');
  const rankedColumns = await addRanking(page);
  stage('Three rank columns, raw five-year NPV and two sort priorities added');
  assert.deepEqual(rankedColumns.slice(0, watch.columns.length), watch.columns);
  assert.deepEqual((await preferences(page)).filters, watch.filters);
  await assertPolicy(page, expected.cohortSize); await waitMatches(page, expected.qualityCount);
  const rows = await allRows(page, expected.qualityCount);
  assert.deepEqual(rows.map(row => row.id), expected.orderedIds);
  for (const row of rows) {
    const calculation = expected.byId[row.id];
    for (const field of Object.keys(kpis)) near(row[field], calculation[field], `${row.id} ${field}`);
    for (const field of Object.keys(rawColumns)) near(row.raw[field], calculation.raw[field], `${row.id} raw ${field}`);
    near(row.legacyHalf, priorExpected.byId[row.id].raw.discount, `${row.id} unchanged ten-year half-terminal NPV`);
    if (row.score !== null) {
      near(row.score, .6 * row.discountRank + .4 * row.quality, `${row.id} 60/40 sum`);
      assert.ok(row.score >= 0 && row.score <= 100);
      assert.ok(row.raw.discount < row.legacyHalf, 'Five-year cash-only value excludes later cash and terminal value');
    }
  }
  assert.deepEqual(rows.filter(row => row.score === null).map(row => row.id), expected.unrankedIds);
  assert.ok(rows.slice(expected.cohortSize).every(row => row.score === null), 'Every unranked row follows all scored rows');
  checks.push('All 248 watch listings retain their filters and raw normal-year inputs; raw five-year cash-only NPV, three 60/40 ranking columns and exact ordering match independent Python arithmetic, with 68 unranked rows last');
  stage('All independent rank and raw cells verified');

  const leader = rows[0];
  await page.locator(`[data-company-listing="${leader.id}"] [data-company-list-kpi="${kpis.score}"]`).getByRole('button').click();
  const details = page.getByRole('dialog', { name: 'KPI value details', exact: true });
  const detailText = await details.innerText();
  for (const pattern of [/60\/40|60%/, /five.year|years? 1[–-]5/i, /no terminal|without terminal|terminal.*0/i, /capital|ROCE/i, /margin/i, /CFO|operating cash/i, /debt/i, /180/, /2020–2023/]) assert.match(detailText, pattern);
  const componentValues = detailText.match(/Component rank points: capital return ([^;]+); minimum margin ([^;]+); CFO growth ([^;]+); debt ([^.]+(?:\.\d+)?)/);
  assert.ok(componentValues, 'Details must expose each quality component rank');
  for (const [index, field] of ['roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda'].entries()) near(Number(componentValues[index + 1]), expected.byId[leader.id].components[field], `${leader.id} detail component ${field}`);
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  await page.locator(`[data-company-listing="${leader.id}"] [data-company-list-kpi="${fiveYearKpi}"]`).getByRole('button').click();
  const fiveYearDetail = await details.innerText();
  for (const pattern of [/five.year|years? 1[–-]5/i, /no terminal|without terminal|terminal.*0/i, /10%/, /2020–2023/]) assert.match(fiveYearDetail, pattern);
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  await page.locator(`[data-company-listing="${expected.unrankedIds[0]}"] [data-company-list-kpi="${kpis.score}"]`).getByRole('button').click();
  assert.match(await details.innerText(), /missing|unavailable|unranked/i);
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  checks.push('The five-year cash-only 60/40 policy, fixed 180-listing peer count, four component explanations and an unavailable-score reason are visible');
  stage('Ranked and missing component dialogs verified');

  await page.getByLabel('Company list rows per page', { exact: true }).selectOption('50');
  await page.getByLabel('Company list search', { exact: true }).fill(leader.id);
  await page.locator(`[data-company-listing="${leader.id}"]`).waitFor();
  await assertPolicy(page, expected.cohortSize);
  const filteredLeader = (await tableValues(page)).find(row => row.id === leader.id);
  assert.deepEqual(filteredLeader, leader);
  await page.getByLabel('Company list search', { exact: true }).fill('');
  await waitMatches(page, expected.qualityCount);
  const rankHeader = page.locator(`[data-company-list-header][data-company-list-kpi="${kpis.score}"]`);
  assert.equal(await rankHeader.locator('.company-list-column-range-footer small').innerText(), 'number');
  assert.equal(await page.locator(`[data-company-list-header][data-company-list-kpi="${fiveYearKpi}"] .company-list-column-range-footer small`).innerText(), '%');
  await rankHeader.getByRole('textbox', { name: /^Min / }).fill(String(leader.score));
  await waitMatches(page, rows.filter(row => row.score !== null && row.score >= leader.score).length);
  await assertPolicy(page, expected.cohortSize);
  assert.deepEqual((await tableValues(page)).find(row => row.id === leader.id), leader);
  await rankHeader.getByRole('button', { name: /^Clear range for / }).click();
  await waitMatches(page, expected.qualityCount);
  assert.deepEqual((await allRows(page, expected.qualityCount)).map(row => row.id), expected.orderedIds);
  checks.push('Search and numeric rank bounds leave scores, components, raw inputs and fixed peer count unchanged; rank bounds use number units');
  stage('Query and numeric bounds verified');

  await page.getByLabel('Company list rows per page', { exact: true }).selectOption('50');
  const rankedWatch = await saveView(page, 'TEST — normal years NPV5 60/40 watch');
  await assertPreserved(page, originalViews, features);
  await page.getByLabel('Saved company list view', { exact: true }).selectOption(strict.id);
  await waitMatches(page, expected.strictCount);
  const strictColumns = await addRanking(page);
  assert.deepEqual(strictColumns.slice(0, strict.columns.length), strict.columns);
  assert.deepEqual((await preferences(page)).filters, strict.filters);
  await assertPolicy(page, expected.cohortSize); await waitMatches(page, expected.strictCount);
  const rankedStrict = await saveView(page, 'TEST — normal years NPV5 60/40 discount');
  await assertPreserved(page, originalViews, features);
  checks.push('Both saved normal-year screens accept the new 60/40 primary rank and five-year NPV tie-breaker without changing their original bounds; the zero-match discount screen keeps the same fixed peers');
  stage('Both ranking views saved and previous views preserved');

  await page.getByLabel('Saved company list view', { exact: true }).selectOption(watch.id);
  await waitMatches(page, expected.qualityCount);
  assert.deepEqual((await allRows(page, expected.qualityCount)).map(row => row.id), oldIds);
  assert.equal(await page.locator('[data-company-list-quality-value-policy]').count(), 0);
  await page.getByLabel('Saved company list view', { exact: true }).selectOption(rankedWatch.id);
  await waitMatches(page, expected.qualityCount); await assertPolicy(page, expected.cohortSize);
  const beforeReload = await bytes(page);
  await assertPreserved(page, originalViews, features);
  await page.reload(); await openLists(page); await waitMatches(page, expected.qualityCount);
  assert.equal(await bytes(page), beforeReload);
  assert.deepEqual(await allRows(page, expected.qualityCount), rows);
  await assertPolicy(page, expected.cohortSize); await assertPreserved(page, originalViews, features);
  const state = await preferences(page);
  for (const saved of [rankedWatch, rankedStrict]) assert.deepEqual(state.features.viewFeatures[saved.id].secondarySorts, secondarySort(saved));
  checks.push('Saved ranked views, primary and secondary ordering, previous 24-column and 50/50 view definitions, unknown watchlist IDs and authored draft/revision/notebook bytes survive reload without protected writes');
  stage('Reload and exact full-table persistence verified');
  const screenshot = resolve(project, `test-results/normal-year-five-year-ranking-${native ? 'native' : 'browser'}.png`);
  await page.screenshot({ path: screenshot });
  return { checks, screenshot, timings, durationMs: Date.now() - startedAt, expectedPath, referenceIdentity: expected.referenceIdentity, cohortSize: expected.cohortSize, rankedCount: expected.rankedIds.length, unrankedCount: expected.unrankedCount, restart: { preferences: beforeReload, protectedValues, count: expected.qualityCount, rows, cohortSize: expected.cohortSize, rankedWatch, rankedStrict } };
}

export async function assertNormalYearRankingRestart(page, state) {
  page.setDefaultTimeout(60000);
  await openLists(page); await waitMatches(page, state.count);
  assert.equal(await bytes(page), state.preferences);
  assert.deepEqual(await protectedStorage(page), state.protectedValues);
  await assertPolicy(page, state.cohortSize);
  assert.deepEqual(await allRows(page, state.count), state.rows);
  const saved = await preferences(page);
  for (const view of [state.rankedWatch, state.rankedStrict]) {
    assert.deepEqual(saved.savedViews.find(item => item.id === view.id), view);
    assert.deepEqual(saved.features.viewFeatures[view.id].secondarySorts, secondarySort(view));
  }
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
