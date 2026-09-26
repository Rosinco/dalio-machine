import { selectObservatory, openListPanel } from './workspace-navigation.mjs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, unlink } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const key = 'macro-atlas-company-lists-v2', oldKey = 'macro-atlas-company-lists-v1';
const draftPrefix = 'macro-atlas-valuation-draft-v1:', revisionKey = 'macro-atlas-valuations-v1';
const finite = value => typeof value === 'number' && Number.isFinite(value);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const collator = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const list = page => page.locator('[data-company-list-ready="true"]');
const row = (page, id) => page.locator(`[data-company-listing="${id}"]`);
const cellSelector = (id, column) => `[data-company-listing="${id}"] [data-company-list-column="${column.id}"]`;
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
const savedBytes = page => page.evaluate(key => localStorage.getItem(key), key);
const rowIds = page => page.locator('[data-company-listing]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-company-listing')));
const closeDialog = page => page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
const search = (page, value) => page.getByLabel('Company list search', { exact: true }).fill(value);
const near = (actual, expected, message) => assert.ok(finite(actual) && Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${message}: ${actual} ≈ ${expected}`);

async function fixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  const { version } = JSON.parse(await readFile(resolve(project, 'package.json'), 'utf8'));
  const bytes = await readFile(resolve(project, 'public', manifest.artifact.path));
  assert.equal(hash(bytes), manifest.artifact.sha256); assert.equal(bytes.length, manifest.artifact.bytes);
  const raw = gunzipSync(bytes); assert.equal(hash(raw), manifest.artifact.uncompressedSha256); assert.equal(raw.length, manifest.artifact.uncompressedBytes);
  const artifact = JSON.parse(raw); assert.equal(artifact.rows.length, manifest.rows);
  return { manifest, version, rows: artifact.rows, byId: new Map(artifact.rows.map(row => [row.id, row])) };
}

function auditScript({ draftPrefix, revisionKey }) {
  if (window.__rangeListAudit) return;
  const audit = { protectedWrites: [], nativeCommands: [] };
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith(draftPrefix) || args[0] === revisionKey)) audit.protectedWrites.push({ method, key: args[0] ?? null });
      return original.apply(this, args);
    };
  }
  const original = window.fetch;
  window.fetch = function (input, ...rest) {
    const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
    if (url.hostname === 'ipc.localhost') audit.nativeCommands.push(decodeURIComponent(url.pathname.slice(1)));
    return original.call(this, input, ...rest);
  };
  window.__rangeListAudit = audit;
}
async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort().map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}
async function seed(page, project, data) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  delete draft.starterOrigin; delete draft.researchOrigin; draft.researchAutofillDisabled = true; draft.scenarios.mid.cashFlows[1] = null;
  draft.title = 'TEST FIXTURE — header ranges preserve authored cash and deliberate blanks';
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const valuation = { format: 'macro-atlas-valuation', version: 1, id: 'range-authored', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, draft };
  const draftKey = `${draftPrefix}${valuation.company}:${valuation.release}:${valuation.financial}:${valuation.taxonomy}`;
  await page.evaluate(({ key, oldKey, draftKey, valuation, revisionKey }) => {
    localStorage.removeItem(key); localStorage.removeItem(oldKey); localStorage.setItem(draftKey, JSON.stringify(valuation));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...valuation, id: 'range-revision' }] }));
  }, { key, oldKey, draftKey, valuation, revisionKey });
  const columns = ['country', 'price_date', 'annual_periods', 'ebit_margin', 'fcf', 'stock_close'].map(kpiId => ({ id: kpiId, kpiId, window: 'latest', calculation: 'latest' }));
  const filters = { query: '', sectorId: 'all', branchId: 'all', country: 'all', route: 'all', readiness: 'all', presence: 'all', watchlistOnly: false, preset: 'all', numericRules: [] };
  const features = { version: 1, watchlists: [{ id: 'default', name: 'Watchlist', listingIds: [] }], activeWatchlistId: 'default', comparisonIds: [], secondarySorts: [], density: 'comfortable', viewFeatures: {} };
  await page.evaluate(({ key, columns, filters, features }) => localStorage.setItem(key, JSON.stringify({ version: 2, columns, filters, features, sort: { columnId: 'name', direction: 'asc' }, watchlistIds: [], savedViews: [] })), { key, columns, filters, features });
  return protectedStorage(page);
}
async function openLists(page) {
  await selectObservatory(page, 'companies');
  await page.getByLabel('Company lists', { exact: true }).click(); await list(page).waitFor();
}
async function waitMatches(page, count) {
  await page.waitForFunction(count => Number(document.querySelector('[data-company-list-ready="true"]')?.getAttribute('data-company-list-matches')) === count, count);
}
async function assertReadOnly(page, protectedValues) {
  assert.deepEqual(await protectedStorage(page), protectedValues);
  const audit = await page.evaluate(() => window.__rangeListAudit); assert.ok(audit);
  assert.deepEqual(audit.protectedWrites, [], 'No attempted draft or revision writes, including identical writes');
  assert.deepEqual(audit.nativeCommands.filter(command => /^(financial_(annual|begin|append|finish|cancel|export)|research_(import|export|save|delete))$/.test(command)), []);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
async function ensureFilters(page) {
  if (!await page.getByLabel('Company list filters', { exact: true }).isVisible()) await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
}
async function saveView(page, name) {
  await page.getByRole('button', { name: 'Save view', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Save company list view', exact: true });
  await dialog.getByLabel('View name', { exact: true }).fill(name); await dialog.getByRole('button', { name: 'Save current view', exact: true }).click();
  const view = (await saved(page)).savedViews.find(view => view.name === name); assert.ok(view); return view;
}
function parseCsv(text) {
  const rows = [], fields = []; let field = '', quoted = false; text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) { if (char === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (char === '"') quoted = false; else field += char; }
    else if (char === '"') quoted = true;
    else if (char === ',') { fields.push(field); field = ''; }
    else if (char === '\n') { fields.push(field.replace(/\r$/, '')); rows.push([...fields]); fields.length = 0; field = ''; }
    else field += char;
  }
  assert.equal(quoted, false); if (field || fields.length) { fields.push(field); rows.push([...fields]); } return rows;
}
async function captureCsv(page, native) {
  await page.evaluate(native => {
    if (native) {
      const original = window.fetch;
      window.fetch = async function (input, ...rest) {
        const response = await original.call(this, input, ...rest);
        const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
        if (url.hostname === 'ipc.localhost' && url.pathname === '/export_csv') window.__rangeListCsv = { path: await response.clone().json(), status: response.status };
        return response;
      };
      window.__restoreRangeListCsv = () => { window.fetch = original; };
    } else {
      const blobs = new Map(), create = URL.createObjectURL, click = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = function (blob) { const url = create.call(this, blob); blobs.set(url, blob); return url; };
      HTMLAnchorElement.prototype.click = function () {
        if (this.download?.startsWith('Macro-Atlas-Companies-') && blobs.has(this.href)) {
          const result = { filename: this.download, text: null }; window.__rangeListCsv = result;
          blobs.get(this.href).text().then(text => { result.text = text; }); return;
        }
        return click.call(this);
      };
      window.__restoreRangeListCsv = () => { URL.createObjectURL = create; HTMLAnchorElement.prototype.click = click; };
    }
  }, native);
  try {
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    await page.getByRole('button', { name: 'Export company list CSV', exact: true }).click();
    await page.waitForFunction(native => native ? typeof window.__rangeListCsv?.path === 'string' : typeof window.__rangeListCsv?.text === 'string', native);
    const result = await page.evaluate(() => window.__rangeListCsv);
    if (!native) return result;
    assert.equal(result.status, 200); assert.match(basename(result.path), /^Macro-Atlas-Companies-\d{4}-\d{2}-\d{2}-\d+\.csv$/);
    const bytes = await readFile(result.path); assert.ok(bytes.toString('utf8').includes('"Listing ID","Company"'));
    // Native export uses create_new; remove only this test's own returned file.
    await unlink(result.path);
    return { filename: basename(result.path), text: bytes.toString('utf8'), path: result.path, bytes: bytes.length, sha256: hash(bytes), removedAfterVerification: true };
  } finally { await page.evaluate(() => { window.__restoreRangeListCsv?.(); delete window.__restoreRangeListCsv; }); }
}

const range = (page, column) => page.locator(`[data-company-list-range="${column.id}"]`);
const minInput = (page, column) => range(page, column).getByRole('textbox', { name: /^Min / });
const maxInput = (page, column) => range(page, column).getByRole('textbox', { name: /^Max / });
async function setRange(page, column, min = '', max = '', currency) {
  await minInput(page, column).fill(min);
  await maxInput(page, column).fill(max);
  if (currency !== undefined) await range(page, column).getByRole('combobox', { name: /^Currency for / }).selectOption(currency);
}
async function assertMatches(page, expected) {
  await waitMatches(page, expected.length);
  const ordered = [...expected].sort((a, b) => collator.compare(a.name, b.name) || collator.compare(a.id, b.id));
  assert.deepEqual(await rowIds(page), ordered.slice(0, 50).map(row => row.id));
}
async function clearFilters(page) {
  await ensureFilters(page);
  await page.getByLabel('Company list filters', { exact: true }).getByRole('button', { name: 'Clear list filters', exact: true }).click();
}
async function assertBlankRanges(page) {
  const bounds = await page.locator('[data-company-list-range] input').evaluateAll(inputs => inputs.map(input => input.value));
  assert.ok(bounds.length > 0); assert.ok(bounds.every(value => value === ''));
  assert.ok((await saved(page)).columns.every(column => !column.range || !column.range.min && !column.range.max));
}

// Read the downloaded annual observation and its documented availability boundary.
// Bounds and expected membership are calculated here without importing app helpers.
function latestAnnual(row, metric) {
  const period = row.annual.periods[0];
  if (!period || !row.annual.currency || period.currency !== row.annual.currency) return null;
  const days = (Date.parse(period.end) - Date.parse(period.start)) / 86400000 + 1;
  if (!finite(days) || days < 330 || days > 400) return null;
  const value = metric === 'ebit_margin' ? row.annual.margins.values[0] : row.annual.cash.values[0];
  return finite(value) ? value : null;
}
const inBounds = (value, min, max) => finite(value) && value >= min && value <= max;

async function providerFixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/expanded-kpi-manifest.json'), 'utf8'));
  const read = async descriptor => {
    const bytes = await readFile(resolve(project, 'public', descriptor.path));
    assert.equal(hash(bytes), descriptor.sha256); assert.equal(bytes.length, descriptor.bytes);
    const raw = gunzipSync(bytes); assert.equal(hash(raw), descriptor.uncompressedSha256); assert.equal(raw.length, descriptor.uncompressedBytes);
    return JSON.parse(raw);
  };
  const variant = manifest.variants.find(v => v.metricId === 'provider_2' && v.calcGroup === 'last' && v.calculation === 'latest');
  assert.ok(variant); assert.equal(variant.unit, 'multiple');
  const descriptor = manifest.shards.find(shard => shard.id === variant.shard), index = await read(manifest.index), shard = await read(descriptor);
  assert.equal(shard.variantIds[variant.offset], variant.id);
  return { manifest, variant, descriptor, values: new Map(index.ids.map((id, i) => [id, shard.values[variant.offset][i]])) };
}
async function addProviderColumn(page, variant) {
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  await dialog.getByRole('button', { name: 'All KPIs', exact: true }).click();
  await dialog.locator(`[data-company-list-kpi-option="${variant.metricId}"]`).click();
  await dialog.getByLabel('KPI time period', { exact: true }).selectOption(`provider:${variant.source}:${variant.calcGroup}`);
  await dialog.getByLabel('KPI calculation', { exact: true }).selectOption(`provider:${variant.calculation}`);
  await dialog.getByRole('button', { name: 'Add column', exact: true }).click(); await closeDialog(page);
  const column = (await saved(page)).columns.findLast(column => column.kpiId === variant.metricId);
  assert.ok(column); return column;
}

async function providerAvailability(page, data, provider, column, protectedValues) {
  const marker = 'macro-atlas-range-provider-probe';
  await page.addInitScript(({ marker, path }) => {
    const mode = sessionStorage.getItem(marker); if (!mode) return;
    sessionStorage.removeItem(marker);
    const original = window.fetch, probe = { mode, responses: 0 };
    window.__rangeProviderProbe = probe;
    window.fetch = async function (input, ...rest) {
      const response = await original.call(this, input, ...rest);
      const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
      if (!url.pathname.endsWith(`/${path}`)) return response;
      probe.responses++;
      if (mode === 'hold') {
        await new Promise(resolve => { window.__releaseRangeProvider = resolve; });
        return response;
      }
      const bytes = new Uint8Array(await response.clone().arrayBuffer());
      if (!bytes.length) throw new Error('Expected a non-empty source shard');
      bytes[0] ^= 1;
      const headers = new Headers(response.headers); headers.delete('content-encoding'); headers.set('content-length', String(bytes.length));
      return new Response(bytes, { status: response.status, statusText: response.statusText, headers });
    };
  }, { marker, path: provider.descriptor.path });
  const expected = data.rows.filter(row => inBounds(provider.values.get(row.id), 0, 20));
  await page.evaluate(marker => sessionStorage.setItem(marker, 'hold'), marker);
  await page.reload(); await openLists(page);
  await page.waitForFunction(() => window.__rangeProviderProbe?.responses > 0);
  await waitMatches(page, 0);
  assert.ok(await minInput(page, column).isVisible());
  assert.equal(await page.getByRole('button', { name: 'Export company list CSV', exact: true }).isDisabled(), true);
  await page.evaluate(() => window.__releaseRangeProvider());
  await assertMatches(page, expected);
  await page.evaluate(marker => sessionStorage.setItem(marker, 'corrupt'), marker);
  await page.reload(); await openLists(page);
  await page.waitForFunction(() => window.__rangeProviderProbe?.responses > 0);
  await page.getByText(/selected KPI datasets could not be opened/).waitFor();
  await waitMatches(page, 0); assert.ok(await minInput(page, column).isVisible());
  await setRange(page, column);
  await assertMatches(page, data.rows);
  await search(page, data.byId.get('102').name);
  assert.equal(await page.locator(cellSelector('102', column)).getAttribute('data-company-list-value'), '');
  assert.equal(Number(await row(page, '102').locator('[data-company-list-kpi="annual_periods"]').getAttribute('data-company-list-value')), data.byId.get('102').annual.periods.length);
  await search(page, ''); await setRange(page, column, '0', '20');
  await assertReadOnly(page, protectedValues);
  await page.reload(); await openLists(page); await assertMatches(page, expected);
  return { path: provider.descriptor.path, expectedMatches: expected.length };
}

export async function companyListRangeFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), provider = await providerFixture(project), checks = [], screenshots = [];
  const viewport = page.viewportSize(), checked = message => { checks.push(message); console.log(`Company list ranges: ${message}`); };
  const protectedValues = await seed(page, project, data);
  await page.addInitScript(auditScript, { draftPrefix, revisionKey }); await page.reload(); await openLists(page);
  try {
    assert.equal(await page.locator('.rail-version').innerText(), `V${data.version}`);
    let preferences = await saved(page);
    const getColumn = metric => preferences.columns.find(column => column.kpiId === metric);
    const periods = getColumn('annual_periods'), margin = getColumn('ebit_margin'), cash = getColumn('fcf'), price = getColumn('stock_close');
    await assertMatches(page, data.rows);
    for (const column of preferences.columns) {
      const header = page.locator(`th[data-company-list-column="${column.id}"]`);
      if (['country', 'price_date'].includes(column.kpiId)) { assert.equal(await range(page, column).count(), 0); continue; }
      assert.equal(await header.locator('[data-company-list-range]').count(), 1);
      const label = (await header.getByRole('button', { name: /^Sort by / }).getAttribute('aria-label')).slice('Sort by '.length);
      assert.equal(await minInput(page, column).getAttribute('aria-label'), `Min ${label}`);
      assert.equal(await maxInput(page, column).getAttribute('aria-label'), `Max ${label}`);
    }
    await setRange(page, periods, '0', '0');
    const noHistory = data.rows.filter(row => row.annual.periods.length === 0); assert.ok(noHistory.length > 0);
    await assertMatches(page, noHistory); await setRange(page, periods);
    const negative = data.rows.find(row => inBounds(latestAnnual(row, 'ebit_margin'), -20, -1)); assert.ok(negative);
    const exact = latestAnnual(negative, 'ebit_margin');
    await setRange(page, margin, String(exact), String(exact));
    await assertMatches(page, data.rows.filter(row => latestAnnual(row, 'ebit_margin') === exact));
    await setRange(page, margin, '-20,5', '0');
    await assertMatches(page, data.rows.filter(row => inBounds(latestAnnual(row, 'ebit_margin'), -20.5, 0)));
    assert.equal((await saved(page)).columns.find(column => column.id === margin.id).range.min, '-20,5');
    checked('Every numeric header has accessible Min/Max controls; text and dates do not; zero, negative equal endpoints and decimal commas filter inclusively');

    await setRange(page, margin, 'not-a-number', '0'); await waitMatches(page, 0);
    assert.ok(await minInput(page, margin).isVisible()); assert.ok(await range(page, margin).locator('[aria-invalid="true"]').count() > 0);
    assert.match(await range(page, margin).innerText(), /valid minimum|number|invalid/i);
    const invalidBytes = await savedBytes(page); await page.reload(); await openLists(page); await waitMatches(page, 0);
    assert.equal(await savedBytes(page), invalidBytes); assert.equal(await minInput(page, margin).inputValue(), 'not-a-number');
    await setRange(page, margin, '10', '5'); await waitMatches(page, 0);
    assert.ok(await range(page, margin).locator('[aria-invalid="true"]').count() > 0);
    await setRange(page, margin, '', '20,5');
    await assertMatches(page, data.rows.filter(row => inBounds(latestAnnual(row, 'ebit_margin'), -Infinity, 20.5)));
    await setRange(page, margin);
    checked('Invalid text and reversed ranges fail closed with visible errors, preserve editable headers at zero results, survive reload, and recover with an unbounded minimum');

    await setRange(page, cash, '-100', '100', ''); await waitMatches(page, 0);
    assert.ok(await range(page, cash).getByRole('combobox', { name: /^Currency for / }).isVisible());
    await setRange(page, cash, '-100', '100', 'SEK');
    const cashMatches = data.rows.filter(row => row.annual.currency === 'SEK' && inBounds(latestAnnual(row, 'fcf'), -100, 100));
    assert.ok(cashMatches.length > 0); await assertMatches(page, cashMatches);
    await setRange(page, price, '0', '100', 'SEK');
    const currencyMatches = cashMatches.filter(row => row.valuation.priceDate && row.valuation.priceBasis?.currency === 'SEK' && inBounds(row.valuation.priceBasis.close, 0, 100));
    assert.ok(currencyMatches.length > 0); await assertMatches(page, currencyMatches);
    assert.ok((await saved(page)).columns.filter(column => ['fcf', 'stock_close'].includes(column.kpiId)).every(column => column.range.currency === 'SEK'));
    await clearFilters(page); await assertMatches(page, data.rows); await assertBlankRanges(page);
    checked('Money and share-price ranges require an explicit currency, exclude other currencies and missing amounts, combine together, and clear through the existing reset action');

    await setRange(page, periods, '3', '5'); await setRange(page, margin, '0', '20,5');
    let expected = data.rows.filter(row => row.annual.periods.length >= 3 && row.annual.periods.length <= 5 && inBounds(latestAnnual(row, 'ebit_margin'), 0, 20.5));
    await assertMatches(page, expected);
    await ensureFilters(page); await page.getByRole('button', { name: 'Add numeric KPI filter', exact: true }).click();
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(periods.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('gte');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('5');
    expected = expected.filter(row => row.annual.periods.length >= 5); assert.ok(expected.length > 50);
    await assertMatches(page, expected);
    const view = await saveView(page, 'TEST — Five reports and margin range');
    const csv = await captureCsv(page, native), csvRows = parseCsv(csv.text);
    assert.equal(csvRows.length - 1, expected.length);
    assert.deepEqual(new Set(csvRows.slice(1).map(row => row[0])), new Set(expected.map(row => row.id)));
    assert.ok(await page.locator('[data-company-listing]').count() <= 50);
    await clearFilters(page); await assertMatches(page, data.rows); await assertBlankRanges(page);
    await page.getByLabel('Saved company list view', { exact: true }).selectOption(view.id); await assertMatches(page, expected);
    assert.deepEqual((await saved(page)).columns, view.columns);
    checked('Column ranges AND with other ranges and existing numeric conditions; saved views restore raw bounds and CSV exports every matching row beyond the first page');

    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: 960 });
      await minInput(page, margin).scrollIntoViewIfNeeded(); assert.ok(await minInput(page, margin).isVisible());
      const path = resolve(project, `test-results/company-list-range-${native ? 'native' : 'browser'}-${width}.png`);
      await page.screenshot({ path, fullPage: true }); screenshots.push(path);
    }
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    await clearFilters(page);
    const peColumn = await addProviderColumn(page, provider.variant);
    await setRange(page, peColumn, '0', '20');
    const providerExpected = data.rows.filter(row => inBounds(provider.values.get(row.id), 0, 20));
    assert.ok(providerExpected.length > 0 && providerExpected.length < data.rows.length);
    await assertMatches(page, providerExpected);
    const availability = await providerAvailability(page, data, provider, peColumn, protectedValues);
    checked('Provider KPI ranges match verified saved values; loading and corrupt shards match no active range, headers remain editable, core data survives, and clean reload recovers');

    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    let dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    await dialog.locator(`[data-company-list-selected-column="${peColumn.id}"]`).getByRole('button', { name: /^Edit / }).click();
    const historical = provider.manifest.variants.find(v => v.metricId === 'provider_2' && v.calcGroup === '5year' && v.calculation === 'mean'); assert.ok(historical);
    await dialog.getByLabel('KPI time period', { exact: true }).selectOption(`provider:${historical.source}:5year`);
    await dialog.getByLabel('KPI calculation', { exact: true }).selectOption('provider:mean');
    await dialog.getByRole('button', { name: 'Update column', exact: true }).click(); await closeDialog(page);
    await assertMatches(page, data.rows); assert.equal(await minInput(page, peColumn).inputValue(), ''); assert.equal(await maxInput(page, peColumn).inputValue(), '');
    await setRange(page, peColumn, '0', '20');
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click(); dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    await dialog.locator(`[data-company-list-selected-column="${peColumn.id}"]`).getByRole('button', { name: /^Remove / }).click(); await closeDialog(page);
    await assertMatches(page, data.rows); assert.equal(await range(page, peColumn).count(), 0); assert.equal((await saved(page)).columns.some(column => column.id === peColumn.id), false);
    await setRange(page, periods, '5', '5');
    await openListPanel(page, 'columns');
    await page.getByLabel('Column preset', { exact: true }).selectOption('terminal_sensitivity');
    await waitMatches(page, data.rows.length); await assertBlankRanges(page);
    assert.equal(await range(page, periods).count(), 0);
    assert.equal((await saved(page)).columns.filter(column => column.kpiId === 'valuation_attractiveness').length, 3);
    checked('Changing a provider period/calculation clears its old range; removing a column removes its filter; applying a preset discards attached ranges and preserves independent duplicate-KPI headers');

    await page.getByLabel('Saved company list view', { exact: true }).selectOption(view.id); await assertMatches(page, expected);
    const beforeReload = await savedBytes(page); await assertReadOnly(page, protectedValues);
    await page.reload(); await openLists(page); await assertMatches(page, expected);
    assert.equal(await savedBytes(page), beforeReload); await assertReadOnly(page, protectedValues);
    assert.equal(await minInput(page, periods).inputValue(), '3'); assert.equal(await maxInput(page, margin).inputValue(), '20,5');
    checked('Reload preserves exact range preferences, combined matches and saved views, with no attempted writes to authored valuation drafts or saved revisions');
    const restart = { version: data.version, preferences: beforeReload, protectedStorage: protectedValues, matches: expected.length, ids: await rowIds(page), bounds: [{ column: periods, min: '3', max: '5' }, { column: margin, min: '0', max: '20,5' }] };
    return { checks, screenshots, listings: data.rows.length, exportRows: csvRows.length - 1, availability, ...(native ? { nativeCsvDelivery: { path: csv.path, bytes: csv.bytes, sha256: csv.sha256, removedAfterVerification: true } } : {}), restart };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/company-list-range-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {}); throw error;
  } finally { await page.setViewportSize(viewport ?? { width: 1500, height: 960 }); }
}

export async function assertCompanyListRangeRestart(page, result) {
  await openLists(page); await waitMatches(page, result.matches);
  assert.equal(await page.locator('.rail-version').innerText(), `V${result.version}`);
  assert.equal(await savedBytes(page), result.preferences); assert.deepEqual(await protectedStorage(page), result.protectedStorage);
  assert.deepEqual(await rowIds(page), result.ids);
  for (const bound of result.bounds) {
    assert.equal(await minInput(page, bound.column).inputValue(), bound.min);
    assert.equal(await maxInput(page, bound.column).inputValue(), bound.max);
  }
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
