import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { selectObservatory, openListPanel } from './workspace-navigation.mjs';

const key = 'macro-atlas-company-lists-v2';
const windowId = 'normal_2020_2023:5';
const protectedValues = {
  'macro-atlas-valuation-draft-v1:normal-year-test': JSON.stringify({ version: 1, title: 'TEST FIXTURE — keep deliberate blanks', cash: [100, null, -15], notes: 'Authored cash assumptions' }),
  'macro-atlas-valuations-v1': JSON.stringify({ version: 1, items: [{ id: 'normal-year-test-revision', notes: 'TEST FIXTURE — original revision' }] }),
  'macro-atlas-company-notes-v1:normal-year-test': 'TEST FIXTURE — keep original company notes',
};
const list = page => page.locator('[data-company-list-ready="true"]');
const preferences = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
const bytes = page => page.evaluate(key => localStorage.getItem(key), key);
const matches = async page => Number(await list(page).getAttribute('data-company-list-matches'));
const visibleIds = page => page.locator('[data-company-listing]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-company-listing')));
const protectedStorage = page => page.evaluate(keys => Object.fromEntries(keys.map(key => [key, localStorage.getItem(key)])), Object.keys(protectedValues));

function audit() {
  window.__normalYearProtectedWrites = [];
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith('macro-atlas-valuation') || String(args[0]).startsWith('macro-atlas-company-notes'))) window.__normalYearProtectedWrites.push({ method, key: args[0] ?? null });
      return original.apply(this, args);
    };
  }
}

async function openLists(page) {
  await selectObservatory(page, 'companies');
  await page.getByLabel('Company lists', { exact: true }).click();
  await list(page).waitFor();
}

async function assertPolicy(page) {
  const policy = page.locator('[data-company-list-normal-year-policy]');
  assert.equal(await policy.isVisible(), true);
  assert.match(await policy.innerText(), /2020–2023/);
  assert.match(await policy.innerText(), /fiscal end years/);
  assert.match(await policy.innerText(), /both weak and strong years/);
  const valuation = page.locator('[data-company-list-normal-valuation-policy]');
  assert.equal(await valuation.isVisible(), true);
  for (const text of [/median provider FCF/, /10 years/, /10% discount rate/, /0% terminal growth/, /dated saved equity price/]) assert.match(await valuation.innerText(), text);
  assert.doesNotMatch(await page.locator('.company-list-table thead').innerText(), /normal_2020_2023/);
}

export async function normalYearScreenFlows(page, project, { native = false } = {}) {
  const checks = [];
  page.setDefaultTimeout(60000);
  await openLists(page);
  const oldColumns = [
    { id: 'old-margin', kpiId: 'ebit_margin', window: '5', calculation: 'median', range: { min: '10', max: '' } },
    { id: 'old-price-date', kpiId: 'price_date', window: 'latest', calculation: 'latest' },
  ];
  const filters = { query: '', sectorId: 'all', branchId: 'all', country: 'all', route: 'all', readiness: 'all', presence: 'all', watchlistOnly: false, preset: 'all', numericRules: [] };
  const oldView = { id: 'old-view', name: 'TEST — original all-year screen', columns: oldColumns, filters, sort: { columnId: 'name', direction: 'asc' } };
  const features = { version: 1, watchlists: [{ id: 'default', name: 'Watchlist', listingIds: ['102', '999999999'] }], activeWatchlistId: 'default', comparisonIds: ['102'], secondarySorts: [], density: 'comfortable', viewFeatures: { 'old-view': { secondarySorts: [], density: 'comfortable', activeWatchlistId: 'default' } } };
  await page.evaluate(({ key, oldView, features, protectedValues }) => {
    localStorage.setItem(key, JSON.stringify({ version: 2, columns: oldView.columns, filters: oldView.filters, sort: oldView.sort, features, watchlistIds: features.watchlists[0].listingIds, savedViews: [oldView] }));
    for (const [key, value] of Object.entries(protectedValues)) localStorage.setItem(key, value);
  }, { key, oldView, features, protectedValues });
  await page.context().addInitScript(audit);
  await page.reload(); await openLists(page);
  const oldMatches = await matches(page);
  const originalPreferences = await preferences(page);
  assert.deepEqual(originalPreferences.savedViews, [oldView]);
  assert.deepEqual(await protectedStorage(page), protectedValues);
  checks.push('An existing all-year saved view, watchlist IDs, comparison selection and authored records load without migration writes');

  await openListPanel(page, 'columns');
  await page.getByLabel('Column preset', { exact: true }).selectOption('normal_years');
  await assertPolicy(page);
  const columns = (await preferences(page)).columns;
  assert.equal(columns.filter(column => column.kpiId === 'normal_npv_percent').length, 3);
  for (const kpiId of ['normal_roce', 'normal_rota', 'tangible_assets_revenue', 'ebit_margin', 'revenue']) assert.equal(columns.find(column => column.kpiId === kpiId)?.window, windowId);
  assert.equal((await preferences(page)).sort.columnId, columns.find(column => column.kpiId === 'normal_npv_percent' && column.calculation === 'terminal_50').id);
  checks.push('The normal-year column set exposes separate full, half-terminal and cash-only surplus plus readable excluded-year quality measures');

  await openListPanel(page, 'filters');
  await page.getByLabel('Company list preset', { exact: true }).selectOption('normal_quality');
  assert.match(await page.locator('[data-company-list-preset-note]').innerText(), /outside fiscal end years 2020–2023/);
  assert.match(await page.locator('[data-company-list-preset-note]').innerText(), /known publication dates/);
  assert.match(await page.locator('[data-company-list-preset-note]').innerText(), /within 550 days/);
  assert.match(await page.locator('[data-company-list-preset-note]').innerText(), /Add quality and valuation ranges separately/);
  await page.getByRole('button', { name: 'Close list filters', exact: true }).click();
  const eligibleCount = await matches(page);
  assert.ok(eligibleCount > 0 && eligibleCount < Number(await list(page).getAttribute('data-company-list-total')));
  const half = columns.find(column => column.kpiId === 'normal_npv_percent' && column.calculation === 'terminal_50');
  const header = page.locator(`[data-company-list-header="${half.id}"]`);
  await header.getByRole('textbox').first().fill('25');
  const filteredCount = await matches(page);
  assert.ok(filteredCount > 0 && filteredCount < eligibleCount, 'A positive normal-year surplus condition narrows eligible companies');
  const visibleValues = await page.locator(`tbody [data-company-list-column="${half.id}"]`).evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('data-company-list-value'))));
  assert.ok(visibleValues.length > 0 && visibleValues.every(value => Number.isFinite(value) && value >= 25));
  checks.push('The exception-year starting filter and inclusive half-terminal surplus range combine while all visible values meet the bound');

  await page.locator('tbody [data-company-list-kpi="ebit_margin"]').first().getByRole('button').click();
  const valueDetails = page.getByRole('dialog', { name: 'KPI value details', exact: true });
  assert.match(await valueDetails.innerText(), /Selected:/);
  assert.match(await valueDetails.innerText(), /Excluded raw observations/);
  assert.match(await valueDetails.innerText(), /published/);
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  checks.push('Opening a normal-year value exposes the selected report dates and excluded raw observations for stress review');

  const margin = columns.find(column => column.kpiId === 'ebit_margin');
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const picker = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  await picker.locator(`[data-company-list-selected-column="${margin.id}"]`).getByRole('button', { name: /^Edit / }).click();
  assert.equal(await picker.getByLabel('KPI time period', { exact: true }).inputValue(), windowId);
  assert.match(await picker.getByLabel('KPI time period', { exact: true }).locator('option:checked').innerText(), /excluding 2020–2023/);
  await page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
  checks.push('The KPI picker restores the explicit exception-year period with a readable label');

  await page.getByRole('button', { name: 'Save view', exact: true }).click();
  const save = page.getByRole('dialog', { name: 'Save company list view', exact: true });
  await save.getByLabel('View name', { exact: true }).fill('TEST — quality and value excluding 2020–2023');
  await save.getByRole('button', { name: 'Save current view', exact: true }).click();
  const saved = await preferences(page), newView = saved.savedViews.find(view => view.id !== oldView.id);
  assert.ok(newView); assert.equal(newView.filters.preset, 'normal_quality');
  assert.deepEqual(saved.savedViews.find(view => view.id === oldView.id), oldView);
  assert.deepEqual(saved.features.watchlists, features.watchlists);
  assert.deepEqual(saved.features.comparisonIds, features.comparisonIds);
  assert.deepEqual(saved.features.viewFeatures[oldView.id], features.viewFeatures[oldView.id]);
  await page.getByLabel('Saved company list view', { exact: true }).selectOption(oldView.id);
  assert.equal(await matches(page), oldMatches);
  assert.equal(await page.locator('[data-company-list-normal-year-policy]').count(), 0);
  assert.deepEqual((await preferences(page)).columns, oldColumns);
  await page.getByLabel('Saved company list view', { exact: true }).selectOption(newView.id);
  await assertPolicy(page);
  const beforeReload = await bytes(page), ids = await visibleIds(page);
  assert.deepEqual(await protectedStorage(page), protectedValues);
  assert.deepEqual(await page.evaluate(() => window.__normalYearProtectedWrites), [], 'Screen edits must not even rewrite identical authored data');
  await page.reload(); await openLists(page);
  assert.equal(await bytes(page), beforeReload);
  assert.equal(await matches(page), filteredCount);
  assert.deepEqual(await visibleIds(page), ids);
  await assertPolicy(page);
  assert.deepEqual(await protectedStorage(page), protectedValues);
  assert.deepEqual(await page.evaluate(() => window.__normalYearProtectedWrites), []);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
  checks.push('Saving and reloading retain the exact exception policy, bounds and results; the old view still has its original all-year matches and all authored records are byte-identical');

  const screenshot = resolve(project, `test-results/normal-year-screen-${native ? 'native' : 'browser'}.png`);
  await page.screenshot({ path: screenshot, fullPage: true });
  return { checks, screenshot, eligibleCount, filteredCount, restart: { preferences: beforeReload, protectedValues, count: filteredCount, ids } };
}

export async function assertNormalYearScreenRestart(page, state) {
  page.setDefaultTimeout(60000);
  await openLists(page);
  assert.equal(await bytes(page), state.preferences);
  assert.deepEqual(await protectedStorage(page), state.protectedValues);
  assert.equal(await matches(page), state.count);
  assert.deepEqual(await visibleIds(page), state.ids);
  await assertPolicy(page);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
