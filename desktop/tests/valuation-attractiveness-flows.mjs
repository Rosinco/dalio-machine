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

// Deliberately independent of application modules: source components, positive dated
// saved price and the written eligibility rules determine the expected observations.
function expected(row, metric, credit = 1) {
  const v = row.valuation;
  if (['financial', 'unclassified'].includes(row.route) || v.status === 'manual-financial' || row.classificationConflict || ['reconcile_data', 'no_history'].includes(row.readiness) || !v.priceDate || !v.priceBasis?.sourceId?.trim() || !v.priceBasis.sourceAsOf || v.priceDate > v.priceBasis.sourceAsOf || !/^[A-Z]{3}$/.test(v.currency) || !finite(v.candidateEquity) || v.candidateEquity <= 0 || !finite(v.value) || !finite(v.cashPV) || !finite(v.terminalPV) || v.terminalPV < 0) return null;
  assert.ok(Math.abs(v.value - v.cashPV - v.terminalPV) <= 1e-9 * Math.max(1, Math.abs(v.value)), 'Source DCF components reconcile');
  const price = v.candidateEquity;
  if (metric === 'valuation_attractiveness') return 100 * ((v.cashPV + credit * v.terminalPV) / price - 1);
  if (metric === 'mid_npv_percent') return 100 * (v.value / price - 1);
  if (metric === 'dcf_price_ratio') return v.value / price;
  if (metric === 'cash_price_coverage') return 100 * v.cashPV / price;
  if (metric === 'low_npv_percent') return finite(v.lowValue) ? 100 * (v.lowValue / price - 1) : null;
  throw new Error(`Unsupported independent expected metric ${metric}`);
}
function ordered(rows, credit) {
  return [...rows].sort((a, b) => {
    const av = expected(a, 'valuation_attractiveness', credit), bv = expected(b, 'valuation_attractiveness', credit);
    if (av === null && bv !== null) return 1;
    if (av !== null && bv === null) return -1;
    return (av === null ? 0 : bv - av) || collator.compare(a.name, b.name) || collator.compare(a.id, b.id);
  });
}
function auditScript({ draftPrefix, revisionKey }) {
  if (window.__attractivenessAudit) return;
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
  window.__attractivenessAudit = audit;
}
async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort().map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}
async function seed(page, project, data) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  delete draft.starterOrigin; delete draft.researchOrigin; draft.researchAutofillDisabled = true; draft.scenarios.mid.cashFlows[1] = null;
  draft.title = 'TEST FIXTURE — list ranking preserves authored cash and deliberate blanks';
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const valuation = { format: 'macro-atlas-valuation', version: 1, id: 'attractiveness-authored', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, draft };
  const draftKey = `${draftPrefix}${valuation.company}:${valuation.release}:${valuation.financial}:${valuation.taxonomy}`;
  await page.evaluate(({ key, oldKey, draftKey, valuation, revisionKey }) => {
    localStorage.removeItem(key); localStorage.removeItem(oldKey); localStorage.setItem(draftKey, JSON.stringify(valuation));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...valuation, id: 'attractiveness-revision' }] }));
  }, { key, oldKey, draftKey, valuation, revisionKey });
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
  const audit = await page.evaluate(() => window.__attractivenessAudit); assert.ok(audit);
  assert.deepEqual(audit.protectedWrites, [], 'No attempted draft or revision writes, including identical writes');
  assert.deepEqual(audit.nativeCommands.filter(command => /^(financial_(annual|begin|append|finish|cancel|export)|research_(import|export|save|delete))$/.test(command)), []);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
async function ensureFilters(page) {
  if (!await page.getByLabel('Company list filters', { exact: true }).isVisible()) await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
}
async function chooseCredit(page, column, percent) {
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  if (column) await dialog.locator(`[data-company-list-selected-column="${column.id}"]`).getByRole('button', { name: /^Edit / }).click();
  else { await dialog.getByRole('button', { name: 'All KPIs', exact: true }).click(); await dialog.locator('[data-company-list-kpi-option="valuation_attractiveness"]').click(); }
  assert.deepEqual(await dialog.getByLabel('KPI calculation', { exact: true }).locator('option').evaluateAll(nodes => nodes.map(node => node.value)).then(values => values.sort()), ['terminal_0', 'terminal_100', 'terminal_25', 'terminal_50', 'terminal_75']);
  await dialog.getByLabel('KPI calculation', { exact: true }).selectOption(`terminal_${percent}`);
  await dialog.getByRole('button', { name: column ? 'Update column' : 'Add column', exact: true }).click(); await closeDialog(page);
  const result = (await saved(page)).columns.find(item => column ? item.id === column.id : item.kpiId === 'valuation_attractiveness' && item.calculation === `terminal_${percent}`);
  assert.ok(result); assert.equal(result.calculation, `terminal_${percent}`); return result;
}
async function verifyCell(page, record, column, value) {
  const selector = cellSelector(record.id, column);
  await page.waitForFunction(selector => document.querySelector(selector) !== null, selector);
  const cell = page.locator(selector), raw = await cell.getAttribute('data-company-list-value');
  if (value === null) { assert.equal(raw, ''); assert.match(await cell.innerText(), /—/); }
  else { near(Number(raw), value, `${record.name}/${column.kpiId}/${column.calculation}`); assert.match(await cell.innerText(), /\d/); }
  assert.equal(await cell.getAttribute('data-company-list-currency'), '', 'Price-normalized fields do not split ranking into currency groups');
  if (value !== null) assert.equal(await cell.getAttribute('data-company-list-date'), record.valuation.priceDate);
}
async function verifyPriceDates(page, data) {
  for (const id of await rowIds(page)) {
    const record = data.byId.get(id), label = row(page, id).locator('[data-company-list-candidate-price-date]');
    await label.waitFor(); assert.equal(await label.getAttribute('data-company-list-candidate-price-date'), record.valuation.priceDate ?? '');
    assert.ok((await label.innerText()).includes(record.valuation.priceDate ?? 'unavailable'));
  }
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
        if (url.hostname === 'ipc.localhost' && url.pathname === '/export_csv') window.__attractivenessCsv = { path: await response.clone().json(), status: response.status };
        return response;
      };
      window.__restoreAttractivenessCsv = () => { window.fetch = original; };
    } else {
      const blobs = new Map(), create = URL.createObjectURL, click = HTMLAnchorElement.prototype.click;
      URL.createObjectURL = function (blob) { const url = create.call(this, blob); blobs.set(url, blob); return url; };
      HTMLAnchorElement.prototype.click = function () {
        if (this.download?.startsWith('Macro-Atlas-Companies-') && blobs.has(this.href)) {
          const result = { filename: this.download, text: null }; window.__attractivenessCsv = result;
          blobs.get(this.href).text().then(text => { result.text = text; }); return;
        }
        return click.call(this);
      };
      window.__restoreAttractivenessCsv = () => { URL.createObjectURL = create; HTMLAnchorElement.prototype.click = click; };
    }
  }, native);
  try {
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    await page.getByRole('button', { name: 'Export company list CSV', exact: true }).click();
    await page.waitForFunction(native => native ? typeof window.__attractivenessCsv?.path === 'string' : typeof window.__attractivenessCsv?.text === 'string', native);
    const result = await page.evaluate(() => window.__attractivenessCsv);
    if (!native) return result;
    assert.equal(result.status, 200); assert.match(basename(result.path), /^Macro-Atlas-Companies-\d{4}-\d{2}-\d{2}-\d+\.csv$/);
    const bytes = await readFile(result.path); assert.ok(bytes.toString('utf8').includes('"Listing ID","Company"'));
    // Native export uses create_new; remove only this test's own returned file.
    await unlink(result.path);
    return { filename: basename(result.path), text: bytes.toString('utf8'), path: result.path, bytes: bytes.length, sha256: hash(bytes), removedAfterVerification: true };
  } finally { await page.evaluate(() => { window.__restoreAttractivenessCsv?.(); delete window.__restoreAttractivenessCsv; }); }
}

export async function valuationAttractivenessFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), checks = [], screenshots = [], viewport = page.viewportSize();
  const checked = message => { checks.push(message); console.log(`Valuation attractiveness: ${message}`); };
  assert.equal(await page.locator('.rail-version').innerText(), `V${data.version}`, 'Macro sidebar reports the packaged application version');
  const protectedValues = await seed(page, project, data);
  await page.addInitScript(auditScript, { draftPrefix, revisionKey }); await page.reload(); await openLists(page);
  try {
    assert.equal(await page.locator('.rail-version').innerText(), `V${data.version}`, 'Company sidebar reports the packaged application version');
    await openListPanel(page, 'columns');
    await page.getByLabel('Column preset', { exact: true }).selectOption('terminal_sensitivity'); await waitMatches(page, data.rows.length);
    const sensitivityColumns = (await saved(page)).columns.filter(column => column.kpiId === 'valuation_attractiveness');
    assert.deepEqual(sensitivityColumns.map(column => column.calculation), ['terminal_100', 'terminal_50', 'terminal_0']);
    assert.deepEqual((await saved(page)).sort, { columnId: sensitivityColumns[0].id, direction: 'desc' });
    assert.deepEqual(await rowIds(page), ordered(data.rows, 1).slice(0, 50).map(row => row.id));
    await search(page, data.byId.get('102').name);
    for (const selected of sensitivityColumns) await verifyCell(page, data.byId.get('102'), selected, expected(data.byId.get('102'), selected.kpiId, Number(selected.calculation.slice('terminal_'.length)) / 100));
    await search(page, '');
    checked('Both sidebar versions match the package; Terminal sensitivities displays 100%, 50% and 0% credit with independent calculations and explicit full-credit sorting');

    await openListPanel(page, 'columns');
    await page.getByLabel('Column preset', { exact: true }).selectOption('valuation_rank'); await waitMatches(page, data.rows.length);
    let preferences = await saved(page), column = preferences.columns.find(column => column.kpiId === 'valuation_attractiveness');
    assert.ok(column); assert.equal(column.calculation, 'terminal_100');
    assert.deepEqual(preferences.sort, { columnId: column.id, direction: 'desc' });
    assert.deepEqual(await rowIds(page), ordered(data.rows, 1).slice(0, 50).map(row => row.id));
    assert.ok(new Set(ordered(data.rows, 1).slice(0, 50).map(row => row.valuation.currency)).size > 1);
    for (const metric of ['mid_npv_percent', 'dcf_price_ratio', 'cash_price_coverage', 'low_npv_percent']) assert.ok(preferences.columns.some(column => column.kpiId === metric), `Preset includes ${metric}`);
    checked('The preset ranks all saved listings by price-normalized attractiveness, with independently verified descending order across currencies');

    const holmen = data.byId.get('102'); await search(page, holmen.name);
    for (const selected of preferences.columns.filter(item => ['valuation_attractiveness', 'mid_npv_percent', 'dcf_price_ratio', 'cash_price_coverage', 'low_npv_percent'].includes(item.kpiId))) await verifyCell(page, holmen, selected, expected(holmen, selected.kpiId));
    near(expected(holmen, 'valuation_attractiveness'), expected(holmen, 'mid_npv_percent'), 'Full terminal credit is Mid NPV divided by saved price');
    await page.locator(cellSelector(holmen.id, column)).getByRole('button').click();
    let dialog = page.getByRole('dialog', { name: 'KPI value details', exact: true });
    assert.ok((await dialog.innerText()).includes(holmen.valuation.priceDate));
    assert.match(await dialog.innerText(), /selected terminal credit \/ 100/, 'Displayed formula converts the selected percentage into its fractional contribution'); await closeDialog(page);
    const lowColumn = preferences.columns.find(item => item.kpiId === 'low_npv_percent');
    await page.locator(cellSelector(holmen.id, lowColumn)).getByRole('button').click(); dialog = page.getByRole('dialog', { name: 'KPI value details', exact: true });
    assert.ok((await dialog.innerText()).includes(`Low scenario DCF ${new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(holmen.valuation.lowValue)} ${holmen.valuation.currency} m`), 'Low NPV details expose their own scenario value for reconstruction'); await closeDialog(page);
    const negative = data.rows.find(record => expected(record, 'valuation_attractiveness') < -100); assert.ok(negative);
    await search(page, negative.name); await verifyCell(page, negative, column, expected(negative, 'valuation_attractiveness'));
    for (const missing of [data.rows.find(record => record.route === 'financial'), data.rows.find(record => record.readiness === 'reconcile_data')]) {
      assert.ok(missing); await search(page, missing.name); await verifyCell(page, missing, column, null);
    }
    checked('DCF, Mid/Low NPV and annual cash coverage reconcile to real saved components; signed negative values remain numeric while financial and unresolved rows remain unavailable');

    await search(page, ''); column = await chooseCredit(page, column, 50);
    assert.deepEqual(await rowIds(page), ordered(data.rows, .5).slice(0, 50).map(row => row.id));
    await search(page, holmen.name); await verifyCell(page, holmen, column, expected(holmen, column.kpiId, .5)); await search(page, '');
    await ensureFilters(page); await page.getByRole('button', { name: 'Add numeric KPI filter', exact: true }).click();
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(column.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('gte'); await page.getByLabel('Value for condition 1', { exact: true }).fill('0');
    const selected = data.rows.filter(record => expected(record, 'valuation_attractiveness', .5) !== null && expected(record, 'valuation_attractiveness', .5) >= 0);
    await waitMatches(page, selected.length); const halfView = await saveView(page, '50% terminal screen');
    column = await chooseCredit(page, column, 0); await waitMatches(page, selected.length);
    assert.equal((await saved(page)).filters.numericRules[0].column.calculation, 'terminal_50', 'A filter keeps the calculation the user selected');
    assert.deepEqual(await rowIds(page), ordered(selected, 0).slice(0, 50).map(row => row.id));
    assert.equal((await saved(page)).filters.numericRules[0].currency, undefined);
    checked('Changing terminal credit from 100% to 50% to 0% recomputes and reorders values; a saved 50% numeric condition retains its original formula');

    const fullColumn = await chooseCredit(page, null, 100), halfColumn = await chooseCredit(page, null, 50);
    preferences = await saved(page);
    assert.deepEqual(preferences.columns.filter(item => item.kpiId === 'valuation_attractiveness').map(item => item.calculation).sort(), ['terminal_0', 'terminal_100', 'terminal_50']);
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click(); dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    for (const selectedColumn of preferences.columns.filter(item => ['price_date', 'stock_close', 'saved_equity_price'].includes(item.kpiId))) await dialog.locator(`[data-company-list-selected-column="${selectedColumn.id}"]`).getByRole('button', { name: /^Remove / }).click();
    await closeDialog(page); await verifyPriceDates(page, data);
    for (const [selectedColumn, percent] of [[column, 0], [halfColumn, 50], [fullColumn, 100]]) {
      const header = page.locator(`thead [data-company-list-column="${selectedColumn.id}"]`);
      assert.match(await header.getByRole('button', { name: /^Sort by / }).getAttribute('aria-label'), new RegExp(`Count ${percent}% of terminal value`, 'i'));
      assert.match(await header.innerText(), new RegExp(percent === 0 ? 'Cash-only surplus' : percent === 50 ? 'Half terminal surplus' : 'Full DCF surplus'));
    }
    checked('Separate columns retain 0%, 50% and 100% terminal assumptions, and exact saved quote dates remain visible after all dedicated price columns are removed');

    await page.getByLabel('Company list country', { exact: true }).selectOption('SE');
    const exportRows = selected.filter(record => record.country === 'SE'); assert.ok(exportRows.length > 50); await waitMatches(page, exportRows.length);
    const csv = await captureCsv(page, native), records = parseCsv(csv.text); preferences = await saved(page);
    assert.equal(records.length, exportRows.length + 1); assert.deepEqual(new Set(records.slice(1).map(record => record[0])), new Set(exportRows.map(row => row.id)));
    assert.equal(records[0].length, 6 + preferences.columns.length * 5);
    for (const [selectedColumn, credit] of [[column, 0], [halfColumn, .5], [fullColumn, 1]]) {
      const offset = 6 + preferences.columns.findIndex(item => item.id === selectedColumn.id) * 5;
      assert.ok(records[0][offset].includes(selectedColumn.calculation)); assert.match(records[0][offset], new RegExp(`Count ${credit * 100}% of terminal value`, 'i'));
      for (const record of records.slice(1)) { const source = data.byId.get(record[0]); near(Number(record[offset]), expected(source, 'valuation_attractiveness', credit), `CSV ${record[0]}/${credit}`); assert.equal(record[offset + 1], 'percent'); assert.equal(record[offset + 2], ''); assert.equal(record[offset + 3], source.valuation.priceDate); assert.equal(record[offset + 4], 'available'); }
    }
    checked(`CSV exports every ${exportRows.length} matching Swedish listing with all three exact calculation identities, percentage units and saved quote dates`);

    const sensitivityView = await saveView(page, 'Terminal sensitivity comparison');
    await page.getByRole('button', { name: 'Clear list filters', exact: true }).click(); await ensureFilters(page);
    await page.getByRole('button', { name: 'Add numeric KPI filter', exact: true }).click(); await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(column.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('missing');
    const missingCount = data.rows.filter(record => expected(record, 'valuation_attractiveness', 0) === null).length; await waitMatches(page, missingCount);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('present'); await waitMatches(page, data.rows.length - missingCount);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('lt'); await page.getByLabel('Value for condition 1', { exact: true }).fill('-100');
    await waitMatches(page, data.rows.filter(record => expected(record, 'valuation_attractiveness', 0) !== null && expected(record, 'valuation_attractiveness', 0) < -100).length);
    checked('Missing, present and below-minus-100% filters reconcile with the full source universe without treating negative projections as missing');

    await page.getByLabel('Saved company list view', { exact: true }).selectOption(halfView.id); await waitMatches(page, selected.length);
    assert.deepEqual((await saved(page)).columns, halfView.columns); assert.equal((await saved(page)).columns.find(item => item.kpiId === 'valuation_attractiveness').calculation, 'terminal_50');
    await page.getByLabel('Saved company list view', { exact: true }).selectOption(sensitivityView.id); await waitMatches(page, exportRows.length);
    assert.deepEqual((await saved(page)).columns, sensitivityView.columns); await assertReadOnly(page, protectedValues);
    const before = await savedBytes(page); await page.reload(); await openLists(page); await waitMatches(page, exportRows.length);
    assert.equal(await savedBytes(page), before); await assertReadOnly(page, protectedValues); await verifyPriceDates(page, data);
    const ids = ordered(exportRows, 0).slice(0, 50).map(record => record.id); assert.deepEqual(await rowIds(page), ids);
    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 960 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth + 1);
      const path = resolve(project, `test-results/valuation-attractiveness-${native ? 'native' : 'browser'}-${width}.png`); await page.screenshot({ path, fullPage: true }); screenshots.push(path);
      if (width < 500) {
        await page.getByRole('region', { name: 'Company list results', exact: true }).scrollIntoViewIfNeeded();
        const tablePath = resolve(project, `test-results/valuation-attractiveness-${native ? 'native' : 'browser'}-${width}-table.png`);
        await page.screenshot({ path: tablePath }); screenshots.push(tablePath);
      }
    }
    checked('Saved views restore exact terminal assumptions and conditions; reload preserves authored draft and revision bytes, and the table fits wide and narrow screens');
    const restart = { version: data.version, preferences: before, protectedStorage: protectedValues, ids, matches: exportRows.length, priceDates: Object.fromEntries(ids.map(id => [id, data.byId.get(id).valuation.priceDate])), columns: (await saved(page)).columns.filter(item => item.kpiId === 'valuation_attractiveness') };
    return { checks, screenshots, version: data.version, listings: data.rows.length, missingCount, exportRows: exportRows.length, artifactSha256: data.manifest.artifact.sha256, ...(native ? { nativeCsvDelivery: { path: csv.path, bytes: csv.bytes, sha256: csv.sha256, removedAfterVerification: true } } : {}), restart };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/valuation-attractiveness-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {}); throw error;
  } finally { await page.setViewportSize(viewport ?? { width: 1500, height: 960 }); }
}

export async function assertValuationAttractivenessRestart(page, result) {
  await openLists(page); await waitMatches(page, result.matches);
  assert.equal(await page.locator('.rail-version').innerText(), `V${result.version}`);
  assert.equal(await savedBytes(page), result.preferences); assert.deepEqual(await protectedStorage(page), result.protectedStorage);
  assert.deepEqual(await rowIds(page), result.ids); assert.deepEqual((await saved(page)).columns.filter(item => item.kpiId === 'valuation_attractiveness'), result.columns);
  assert.deepEqual(result.columns.map(column => column.calculation).sort(), ['terminal_0', 'terminal_100', 'terminal_50']);
  assert.equal((await saved(page)).filters.numericRules[0].column.calculation, 'terminal_50');
  for (const [id, date] of Object.entries(result.priceDates)) assert.equal(await row(page, id).locator('[data-company-list-candidate-price-date]').getAttribute('data-company-list-candidate-price-date'), date ?? '');
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
