import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const preferencesKey = 'macro-atlas-company-lists-v2';
const draftPrefix = 'macro-atlas-valuation-draft-v1:';
const revisionKey = 'macro-atlas-valuations-v1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const near = (actual, expected, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= Math.max(1e-8, Math.abs(expected) * 1e-10), `${label}: ${actual} != ${expected}`);
const alphabetical = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const byName = (a, b) => alphabetical.compare(a.name, b.name) || Number(a.id) - Number(b.id);
const positiveFive = series => series.values.length === 5 && series.values.every(value => typeof value === 'number' && value > 0);
const consistent = row => row.route === 'operating' && row.readiness === 'history_available' && !row.classificationConflict && positiveFive(row.annual.cash) && positiveFive(row.annual.ebit);
// Deliberately independent of companyListModel and the precomputed cash factor.
const candidate = row => consistent(row) && row.valuation.status === 'positive-priced' && row.valuation.value > 0 && row.valuation.candidateEquity > 0 && row.valuation.candidateEquity <= .7 * row.valuation.value;

async function fixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  const bytes = await readFile(resolve(project, 'public', manifest.artifact.path));
  assert.equal(bytes.length, manifest.artifact.bytes); assert.equal(hash(bytes), manifest.artifact.sha256);
  const raw = gunzipSync(bytes);
  assert.equal(raw.length, manifest.artifact.uncompressedBytes); assert.equal(hash(raw), manifest.artifact.uncompressedSha256);
  const artifact = JSON.parse(raw.toString('utf8'));
  const envelope = JSON.parse(await readFile(resolve(project, 'public/data/research.atlas.json'), 'utf8'));
  assert.equal(hash(envelope.taxonomy.content), manifest.taxonomySha256);
  const taxonomy = JSON.parse(envelope.taxonomy.content);
  const rows = artifact.rows, byId = Object.fromEntries(rows.map(row => [row.id, row]));
  assert.equal(rows.length, 19140); assert.equal(Object.keys(byId).length, rows.length);
  assert.deepEqual(Object.keys(byId).sort(), Object.keys(taxonomy.catalogue.listings).sort());
  return { manifest, artifact, rows, byId, candidates: rows.filter(candidate).sort(byName), consistent: rows.filter(consistent).sort(byName) };
}

async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort().map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}

async function seedAuthoredWork(page, project, manifest) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  draft.title = 'TEST FIXTURE — company lists preserve authored assumptions and deliberate blanks';
  delete draft.starterOrigin; delete draft.researchOrigin;
  draft.researchAutofillDisabled = true; draft.scenarios.mid.cashFlows[1] = null;
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const saved = { format: 'macro-atlas-valuation', version: 1, id: 'company-list-preserved-draft', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: manifest.financialPackId, taxonomy: manifest.taxonomySha256, draft };
  const key = `${draftPrefix}${saved.company}:${saved.release}:${saved.financial}:${saved.taxonomy}`;
  await page.evaluate(({ key, saved, revisionKey }) => {
    localStorage.setItem(key, JSON.stringify(saved));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...saved, id: 'company-list-preserved-revision' }] }));
  }, { key, saved, revisionKey });
  return protectedStorage(page);
}

function auditScript({ draftPrefix, revisionKey }) {
  if (window.__companyListAudit) return;
  const audit = { writes: [], invokes: [], restore: [] };
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith(draftPrefix) || args[0] === revisionKey)) audit.writes.push({ method, key: args[0] ?? null });
      return original.apply(this, args);
    };
    audit.restore.push(() => { Storage.prototype[method] = original; });
  }
  const original = window.fetch;
  const watched = function (input, ...rest) {
    const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
    if (url.hostname === 'ipc.localhost') audit.invokes.push(decodeURIComponent(url.pathname.slice(1)));
    return original.call(this, input, ...rest);
  };
  window.fetch = watched;
  if (window.fetch !== watched) throw new Error('Company-list native request audit could not be installed');
  audit.restore.push(() => { window.fetch = original; }); window.__companyListAudit = audit;
}

async function assertReadOnly(page, expected) {
  assert.deepEqual(await protectedStorage(page), expected, 'Company lists preserve every authored draft and revision byte');
  const audit = await page.evaluate(() => ({ writes: window.__companyListAudit?.writes, invokes: window.__companyListAudit?.invokes }));
  assert.ok(audit.writes && audit.invokes, 'Read-only audit must actually be installed');
  assert.deepEqual(audit.writes, [], 'List interactions must never attempt valuation writes, including same-value writes');
  assert.deepEqual(audit.invokes.filter(command => /^(financial_(annual|begin|append|finish|cancel|export)|research_(import|export|save|delete))$/.test(command)), [], 'Lists neither bulk-load annual histories nor mutate native archives');
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
  return audit;
}

const list = page => page.locator('[data-company-list-ready="true"]');
const row = (page, id) => page.locator(`[data-company-listing="${id}"]`);
const rows = page => page.locator('[data-company-listing]');
const rowIds = page => rows(page).evaluateAll(nodes => nodes.map(node => node.getAttribute('data-company-listing')));
const matches = async page => Number(await list(page).getAttribute('data-company-list-matches'));

async function openLists(page) {
  await page.getByLabel('Company lists', { exact: true }).click();
  await list(page).waitFor();
}

async function openHolmen(page) {
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await page.getByLabel('Search companies or countries', { exact: true }).fill('Holmen');
  await page.locator('[data-search-listing="102"]').click();
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
}

async function readPreferences(page) { return page.evaluate(key => localStorage.getItem(key), preferencesKey); }

async function verifyCell(page, id, kpi, expected, currency = null) {
  const cell = row(page, id).locator(`[data-company-list-kpi="${kpi}"]`);
  assert.equal(await cell.count(), 1, `Exactly one ${kpi} cell for ${id}`);
  const actual = await cell.getAttribute('data-company-list-value'), text = await cell.innerText();
  if (expected === null) { assert.equal(actual, ''); assert.match(text, /—|unavailable/i); }
  else {
    near(Number(actual), expected, `${id} ${kpi}`);
    assert.match(text, /\d/, `${id} ${kpi} contains a rendered value, never Pro`);
    assert.doesNotMatch(text, /^\s*Pro\s*$/i);
    if (expected < -.01) assert.match(text, /[-−]/, 'Signed values retain the minus sign');
  }
  if (currency && expected !== null) assert.ok(text.includes(currency), `${id} ${kpi} displays its own currency`);
  return cell;
}

async function collectAllPages(page) {
  const ids = [];
  for (;;) {
    ids.push(...await rowIds(page));
    const next = page.getByLabel('Next company list page', { exact: true });
    if (await next.isDisabled()) break;
    await next.click();
  }
  assert.equal(new Set(ids).size, ids.length, 'Paged results never repeat a listing');
  return ids;
}

async function chooseKpi(page, kpiId, { window = 'latest', calculation = 'latest' } = {}) {
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  await dialog.waitFor();
  await dialog.locator(`[data-company-list-kpi-option="${kpiId}"]`).click();
  await dialog.getByLabel('KPI time period', { exact: true }).selectOption(window);
  await dialog.getByLabel('KPI calculation', { exact: true }).selectOption(calculation);
  await dialog.getByRole('button', { name: 'Add column', exact: true }).click();
  if (await dialog.isVisible()) await page.keyboard.press('Escape');
  await dialog.waitFor({ state: 'hidden' });
  const state = JSON.parse(await readPreferences(page));
  return state.columns.findLast(column => column.kpiId === kpiId && column.window === window && column.calculation === calculation);
}

async function search(page, text) { await page.getByLabel('Company list search', { exact: true }).fill(text); }

async function star(page, item) {
  await search(page, item.name);
  await row(page, item.id).getByRole('button', { name: `Add ${item.name} to watchlist`, exact: true }).click();
}

async function verifyPreferenceFailures(page, expectedStorage) {
  const preserved = await readPreferences(page);
  for (const invalid of ['{company-list-corrupt', JSON.stringify({ version: 999, columns: ['future-opaque'], watchlistIds: ['102'] })]) {
    await assertReadOnly(page, expectedStorage);
    await page.evaluate(({ key, invalid }) => localStorage.setItem(key, invalid), { key: preferencesKey, invalid });
    await page.reload(); await openLists(page);
    await page.locator('[data-company-list-storage-error]').waitFor();
    assert.equal(await readPreferences(page), invalid, 'Opening an unreadable or future preference format never overwrites it');
    assert.equal(await matches(page), 19140, 'Unreadable preferences have a usable complete-universe fallback');
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    await page.keyboard.press('Escape');
    assert.equal(await readPreferences(page), invalid, 'Inspecting a picker is not a preference mutation');
    await assertReadOnly(page, expectedStorage);
  }
  await page.evaluate(({ key, preserved }) => localStorage.setItem(key, preserved), { key: preferencesKey, preserved });
  await page.reload(); await openLists(page);
  await page.evaluate(key => {
    const original = Storage.prototype.setItem, probe = { attempts: 0 };
    window.__companyListQuotaProbe = probe;
    window.__companyListQuotaRestore = () => { Storage.prototype.setItem = original; };
    const fail = function (keyToWrite, value) {
      if (this === localStorage && keyToWrite === key) { probe.attempts++; throw new DOMException('Company-list test quota exceeded', 'QuotaExceededError'); }
      return original.call(this, keyToWrite, value);
    };
    Storage.prototype.setItem = fail;
    if (Storage.prototype.setItem !== fail) throw new Error('Real Storage write failure could not be injected');
  }, preferencesKey);
  try {
    await search(page, 'unsaved list change');
    await page.locator('[data-company-list-storage-error]').waitFor();
    assert.ok(await page.evaluate(() => window.__companyListQuotaProbe.attempts > 0), 'Quota test must reach the real preference write');
    assert.equal(await readPreferences(page), preserved, 'Failed preference save retains previous stored bytes');
    await assertReadOnly(page, expectedStorage);
  } finally {
    await page.evaluate(() => { window.__companyListQuotaRestore?.(); delete window.__companyListQuotaRestore; delete window.__companyListQuotaProbe; });
    await page.reload(); await openLists(page);
  }
}

async function verifySourceMismatch(page, expectedStorage, native) {
  const badPack = 'f'.repeat(64);
  if (native) await page.evaluate(badPack => {
    const original = window.fetch, probe = { responses: 0 };
    window.__companyListMismatchProbe = probe;
    window.__companyListMismatchRestore = () => { window.fetch = original; };
    const watched = async function (input, ...rest) {
      const response = await original.call(this, input, ...rest);
      const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
      if (url.hostname !== 'ipc.localhost' || url.pathname !== '/financial_index') return response;
      const value = await response.clone().json();
      if (value?.format !== 'macro-atlas-financials') return response;
      probe.responses++;
      return new Response(JSON.stringify({ ...value, id: badPack }), { status: response.status, statusText: response.statusText, headers: response.headers });
    };
    window.fetch = watched;
    if (window.fetch !== watched) throw new Error('Real native financial-index injection could not be installed');
  }, badPack);
  else await page.route('**/api/financials/index?*', async route => {
    const response = await route.fetch(), value = await response.json();
    return route.fulfill({ response, json: value ? { ...value, id: badPack } : value });
  });
  const resetIndex = async () => {
    await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
    await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  };
  try {
    await resetIndex();
    if (native) await page.waitForFunction(() => window.__companyListMismatchProbe?.responses > 0);
    await page.getByLabel('Company lists', { exact: true }).click();
    await page.locator('[data-company-list-ready] [role="alert"]').waitFor();
    assert.equal(await rows(page).count(), 0, 'A cached gauge from another financial pack cannot leak company results');
    await assertReadOnly(page, expectedStorage);
  } finally {
    if (native) await page.evaluate(() => { window.__companyListMismatchRestore?.(); delete window.__companyListMismatchRestore; delete window.__companyListMismatchProbe; });
    else await page.unroute('**/api/financials/index?*');
    await resetIndex(); await openLists(page);
  }
}

export async function companyListFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), checks = [], screenshots = [], viewport = page.viewportSize();
  const checked = message => { checks.push(message); console.log(`Company lists: ${message}`); };
  await openHolmen(page);
  await page.evaluate(key => localStorage.removeItem(key), preferencesKey);
  const expectedStorage = await seedAuthoredWork(page, project, data.manifest);
  await page.addInitScript(auditScript, { draftPrefix, revisionKey });
  await page.evaluate(auditScript, { draftPrefix, revisionKey });
  try {
    await openLists(page);
    assert.equal(Number(await list(page).getAttribute('data-company-list-total')), 19140);
    assert.equal(await matches(page), 19140);
    const sorted = [...data.rows].sort(byName);
    assert.deepEqual(await rowIds(page), sorted.slice(0, 50).map(item => item.id));
    await page.getByLabel('Next company list page', { exact: true }).click();
    assert.deepEqual(await rowIds(page), sorted.slice(50, 100).map(item => item.id));
    await page.getByLabel('Previous company list page', { exact: true }).click();
    checked('All 19,140 exact source listing IDs remain available with distinct, bounded 50-row pages');

    await page.getByLabel('Company list preset', { exact: true }).selectOption('cash_consistency');
    assert.equal(await matches(page), data.consistent.length);
    await page.getByLabel('Company list preset', { exact: true }).selectOption('cash_and_margin');
    assert.equal(await matches(page), data.candidates.length);
    assert.deepEqual(await collectAllPages(page), data.candidates.map(item => item.id), 'Every candidate independently meets five positive FCF/EBIT observations, eligible operating evidence and price <=70% of Mid');
    await page.getByLabel('Company list preset', { exact: true }).selectOption('all');
    checked(`Both declared candidate presets reconcile independently; all ${data.candidates.length} cash-and-margin listing IDs match without an opaque score`);

    await search(page, 'Holmen');
    const holmen = data.byId['102'];
    await verifyCell(page, '102', 'ebit_margin', holmen.annual.margins.values[0]);
    await verifyCell(page, '102', 'cash_factor_30', holmen.valuation.candidateEquity / (.7 * holmen.valuation.value));
    await verifyCell(page, '102', 'terminal_share', 100 * holmen.valuation.terminalPV / holmen.valuation.value);
    const price = await verifyCell(page, '102', 'stock_close', holmen.valuation.priceBasis.close, holmen.valuation.priceBasis.currency);
    assert.equal(await price.getAttribute('data-company-list-date'), holmen.valuation.priceDate);
    const fcfColumn = await chooseKpi(page, 'fcf', { window: '3', calculation: 'median' });
    assert.ok(fcfColumn);
    const cash = holmen.annual.cash.values.slice(0, 3).sort((a, b) => a - b);
    await verifyCell(page, '102', 'fcf', cash[1], holmen.annual.currency);
    checked('Visible KPI cells show actual dated cash, margin, price and starter values; three-report median uses the selected signed observations');

    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    let dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 960 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth + 1);
      assert.equal(await dialog.isVisible(), true);
      const path = resolve(project, `test-results/company-list-${native ? 'native' : 'browser'}-picker-${width}.png`);
      await page.screenshot({ path, fullPage: true }); screenshots.push(path);
    }
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    await dialog.getByRole('button', { name: 'Close company list dialog', exact: true }).focus();
    await page.keyboard.press('Shift+Tab');
    assert.equal(await dialog.getByRole('button', { name: 'Done', exact: true }).evaluate(node => document.activeElement === node), true, 'Keyboard focus wraps inside the KPI picker');
    await page.keyboard.press('Tab');
    assert.equal(await dialog.getByRole('button', { name: 'Close company list dialog', exact: true }).evaluate(node => document.activeElement === node), true);
    const before = JSON.parse(await readPreferences(page)).columns.map(column => column.id);
    const selected = dialog.locator(`[data-company-list-selected-column="${fcfColumn.id}"]`);
    await selected.getByRole('button', { name: / left$/ }).click();
    const reordered = JSON.parse(await readPreferences(page)).columns.map(column => column.id);
    const at = before.indexOf(fcfColumn.id), expectedOrder = [...before];
    [expectedOrder[at - 1], expectedOrder[at]] = [expectedOrder[at], expectedOrder[at - 1]];
    assert.deepEqual(reordered, expectedOrder, 'KPI move changes exactly the requested adjacent columns');
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).evaluate(node => document.activeElement === node), true, 'Escape restores focus to the opening control');
    const temporary = await chooseKpi(page, 'net_debt');
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    await dialog.locator(`[data-company-list-selected-column="${temporary.id}"]`).getByRole('button', { name: /^Remove / }).click();
    await page.keyboard.press('Escape');
    assert.deepEqual(JSON.parse(await readPreferences(page)).columns.map(column => column.id), expectedOrder, 'Removing a temporary KPI preserves every other selected column');
    checked('KPI picker adds period calculations, moves and removes columns; Escape closes the dialog and returns keyboard focus');

    const watched = [holmen, data.byId['13'], data.byId['167'], data.byId['20720']];
    for (const item of watched) await star(page, item);
    await search(page, '');
    await page.getByRole('button', { name: 'Watchlist', exact: true }).click();
    assert.equal(await matches(page), watched.length);
    const expectedMargin = item => item.annual.margins.values[0] ?? null;
    for (const item of watched) await verifyCell(page, item.id, 'ebit_margin', expectedMargin(item));
    await verifyCell(page, '13', 'fcf', [...data.byId['13'].annual.cash.values.slice(0, 3)].sort((a, b) => a - b)[1], data.byId['13'].annual.currency);
    await verifyCell(page, '167', 'fcf', null);
    for (const direction of ['asc', 'desc']) {
      await page.locator('th[data-company-list-column="ebit_margin"]').getByRole('button').click();
      const ordered = [...watched].sort((a, b) => {
        const av = expectedMargin(a), bv = expectedMargin(b);
        return av === null || bv === null ? av === bv ? byName(a, b) : av === null ? 1 : -1 : (direction === 'asc' ? av - bv : bv - av) || byName(a, b);
      });
      assert.deepEqual(await rowIds(page), ordered.map(item => item.id));
    }
    checked('Personal watchlist stores explicit listing choices; signed and zero KPIs remain numeric and missing values sort last in both directions');

    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    await page.getByLabel('Add numeric KPI filter', { exact: true }).click();
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption('ebit_margin');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('0');
    for (const operator of ['gte', 'lte']) {
      await page.getByLabel('Operator for condition 1', { exact: true }).selectOption(operator);
      const expected = watched.filter(item => expectedMargin(item) !== null && (operator === 'gte' ? expectedMargin(item) >= 0 : expectedMargin(item) <= 0)).map(item => item.id);
      assert.deepEqual(new Set(await rowIds(page)), new Set(expected));
      assert.ok(expected.includes('20720'), 'Exact zero satisfies both inclusive inequality boundaries');
      assert.ok(!expected.includes('167'), 'Unavailable values never satisfy a zero boundary');
    }
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(fcfColumn.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('gte');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('-1000000');
    await page.getByLabel('Currency for condition 1', { exact: true }).selectOption('SEK');
    assert.deepEqual(new Set(await rowIds(page)), new Set(['102', '13']), 'A SEK cash condition excludes the USD company and missing cash');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('-40');
    assert.deepEqual(await rowIds(page), ['102']);
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    await dialog.locator(`[data-company-list-selected-column="${fcfColumn.id}"]`).getByRole('button', { name: /^Edit / }).click();
    await dialog.getByLabel('KPI time period', { exact: true }).selectOption('5');
    await dialog.getByLabel('KPI calculation', { exact: true }).selectOption('max');
    await dialog.getByRole('button', { name: 'Update column', exact: true }).click();
    await page.keyboard.press('Escape');
    assert.match(await page.getByLabel('KPI for condition 1', { exact: true }).locator('option:checked').innerText(), /3.*median.*retained/i, 'An edited display column cannot relabel the retained condition as a different calculation');
    assert.deepEqual(await rowIds(page), ['102'], 'Retained three-report median condition remains in effect after the displayed column changes to five-report maximum');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('');
    assert.equal(await matches(page), 0, 'A deliberately cleared threshold stays incomplete rather than becoming zero');
    await assertReadOnly(page, expectedStorage);
    await page.reload(); await openLists(page);
    assert.equal(await matches(page), 0, 'A cleared threshold remains fail-closed after restart');
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    assert.equal(await page.getByLabel('Value for condition 1', { exact: true }).inputValue(), '');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('-40');
    assert.deepEqual(await rowIds(page), ['102']);
    await page.getByLabel('Currency for condition 1', { exact: true }).selectOption('');
    assert.equal(await matches(page), 0, 'An amount threshold without a chosen currency cannot compare unrelated monetary units');
    await assertReadOnly(page, expectedStorage);
    await page.reload(); await openLists(page);
    assert.equal(await matches(page), 0, 'Restart never silently drops an incomplete currency condition and broadens the list');
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    await page.getByLabel('Remove condition 1', { exact: true }).click();
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    assert.equal(await matches(page), watched.length);
    checked('Numeric filters include exact signed/zero boundaries, exclude missing values and require the selected monetary currency');

    const starsBeforeBranch = JSON.parse(await readPreferences(page)).watchlistIds;
    await page.getByRole('button', { name: 'All listings', exact: true }).click();
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    await page.getByLabel('Company list sector', { exact: true }).selectOption('1');
    assert.equal(await matches(page), data.rows.filter(item => item.sectorId === '1').length);
    assert.equal(JSON.parse(await readPreferences(page)).filters.sectorId, '1', 'The incompatible sector must actually be saved before branch entry');
    await page.getByLabel('Observatory', { exact: true }).selectOption('sectors');
    await page.getByLabel('Branch company lists', { exact: true }).click();
    await list(page).waitFor();
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    assert.equal(await page.getByLabel('Company list sector', { exact: true }).inputValue(), 'all', 'Explicit branch entry clears the incompatible saved sector');
    const branchSelect = page.getByLabel('Company list branch', { exact: true });
    assert.equal(await branchSelect.inputValue(), holmen.branchId);
    assert.equal(await branchSelect.locator(`option[value="${holmen.branchId}"]`).count(), 1, 'The active branch has an actual dropdown option');
    assert.ok((await branchSelect.locator('option:checked').innerText()).includes(holmen.branchName));
    const branchIds = data.rows.filter(item => item.branchId === holmen.branchId).map(item => item.id);
    assert.equal(await matches(page), branchIds.length);
    assert.deepEqual(new Set(await collectAllPages(page)), new Set(branchIds), 'Branch results contain exactly the source branch despite the saved Finance sector');
    assert.deepEqual(JSON.parse(await readPreferences(page)).watchlistIds, starsBeforeBranch, 'Branch navigation preserves every personal star');
    await assertReadOnly(page, expectedStorage);
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
    await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
    await openLists(page);
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    await page.getByLabel('Company list sector', { exact: true }).selectOption('all');
    await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
    await page.getByRole('button', { name: 'Watchlist', exact: true }).click();
    assert.equal(await matches(page), watched.length);
    assert.deepEqual(JSON.parse(await readPreferences(page)).watchlistIds, starsBeforeBranch);
    checked('Branch Lists clears an incompatible saved sector, exposes the selected branch, reconciles exact branch members and preserves personal stars');

    // A same-company navigation needs an explicit view change, since companyId
    // itself is unchanged when the current profile is opened from the table.
    await row(page, '102').getByRole('button', { name: 'Open profile for Holmen', exact: true }).click();
    await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
    await openLists(page);
    assert.equal(await matches(page), watched.length);
    checked('Opening the already-selected company profile works and returning to Lists retains the working columns and watchlist');

    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 960 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth + 1);
      const scroll = page.getByRole('region', { name: 'Company list results', exact: true });
      assert.equal(await scroll.isVisible(), true);
      if (width === 390) {
        const dimensions = await scroll.evaluate(node => { node.scrollLeft = node.scrollWidth; return { width: node.clientWidth, scrollWidth: node.scrollWidth, scrollLeft: node.scrollLeft }; });
        assert.ok(dimensions.scrollWidth > dimensions.width && dimensions.scrollLeft > 0, 'Narrow view scrolls within the table instead of overflowing the page');
      }
      const path = resolve(project, `test-results/company-list-${native ? 'native' : 'browser'}-${width}.png`);
      await page.screenshot({ path, fullPage: true }); screenshots.push(path);
    }
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    checked('Dense KPI table fits 1500px and 390px screens with usable internal horizontal scrolling');

    await page.getByRole('button', { name: 'Save view', exact: true }).click();
    const save = page.getByRole('dialog', { name: 'Save company list view', exact: true });
    await save.getByLabel('View name', { exact: true }).fill('Cash ideas — acceptance');
    await save.getByRole('button', { name: 'Save current view', exact: true }).click();
    const savedState = JSON.parse(await readPreferences(page)), savedView = savedState.savedViews.find(view => view.name === 'Cash ideas — acceptance');
    assert.ok(savedView); assert.ok(savedView.filters.watchlistOnly);
    await page.getByRole('button', { name: 'All listings', exact: true }).click();
    await page.getByLabel('Saved company list view', { exact: true }).selectOption(savedView.id);
    assert.equal(await matches(page), watched.length);
    assert.deepEqual(JSON.parse(await readPreferences(page)).columns, savedView.columns);
    await assertReadOnly(page, expectedStorage);
    await page.reload(); await openLists(page);
    assert.equal(await matches(page), watched.length);
    assert.deepEqual(JSON.parse(await readPreferences(page)).columns, savedView.columns);
    await page.evaluate(key => { const state = JSON.parse(localStorage.getItem(key)); state.watchlistIds.push('999999999'); state.features.watchlists.find(list => list.id === 'default').listingIds.push('999999999'); localStorage.setItem(key, JSON.stringify(state)); }, preferencesKey);
    await assertReadOnly(page, expectedStorage);
    await page.reload(); await openLists(page);
    assert.equal(await matches(page), watched.length);
    assert.ok(JSON.parse(await readPreferences(page)).watchlistIds.includes('999999999'), 'A saved listing absent from this pack is retained without fabricating a displayed row');
    checked('Named views restore selected columns, sorting and filters; personal membership and view settings survive browser reload');

    await verifyPreferenceFailures(page, expectedStorage);
    checked('Corrupt and future preference formats are not overwritten by browsing; real storage-quota failure is visible and preserves saved bytes');
    await verifySourceMismatch(page, expectedStorage, native);
    checked('A mismatched active financial pack withholds cached company results and recovers with the matching source');

    const finalAudit = await assertReadOnly(page, expectedStorage);
    if (native) assert.ok(finalAudit.invokes.includes('financial_index'), 'Native audit observes real financial-index requests');
    const preferences = await readPreferences(page);
    checked('Every tested list action preserves authored draft, deliberate blank, purchase settings and revision bytes without attempted valuation writes');
    return { checks, listings: data.rows.length, candidateListings: data.candidates.length, consistentCashListings: data.consistent.length, pack: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, artifactSha256: data.manifest.artifact.sha256, screenshots, protectedStorage: expectedStorage, preferences, watchlistIds: watched.map(item => item.id), savedViewId: savedView.id };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/company-list-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
  }
}

export async function assertCompanyListRestart(page, result) {
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await page.locator('[data-business-ready="true"]').waitFor();
  assert.deepEqual(await protectedStorage(page), result.protectedStorage, 'Full process restart preserves the authored valuation and revision bytes');
  assert.equal(await readPreferences(page), result.preferences, 'Full process restart preserves exact saved list preferences');
  await openLists(page);
  assert.equal(Number(await list(page).getAttribute('data-company-list-total')), result.listings);
  assert.equal(await matches(page), result.watchlistIds.length);
  assert.deepEqual(new Set(await rowIds(page)), new Set(result.watchlistIds));
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
