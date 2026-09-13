import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, unlink } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const oldKey = 'macro-atlas-company-lists-v1', key = 'macro-atlas-company-lists-v2';
const draftPrefix = 'macro-atlas-valuation-draft-v1:', revisionKey = 'macro-atlas-valuations-v1';
const finite = value => typeof value === 'number' && Number.isFinite(value);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const collator = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const compareName = (a, b) => collator.compare(a.name, b.name) || collator.compare(a.id, b.id);
const list = page => page.locator('[data-company-list-ready="true"]');
const row = (page, id) => page.locator(`[data-company-listing="${id}"]`);
const rowIds = page => page.locator('[data-company-listing]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-company-listing')));
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
const savedBytes = page => page.evaluate(key => localStorage.getItem(key), key);
const count = async page => Number(await list(page).getAttribute('data-company-list-matches'));
const search = (page, value) => page.getByLabel('Company list search', { exact: true }).fill(value);
const closeDialog = page => page.getByRole('button', { name: 'Close company list dialog', exact: true }).click();
const columnSelector = (id, col) => `[data-company-listing="${id}"] [data-company-list-column="${col.id}"]`;

async function readArtifact(project, descriptor) {
  const bytes = await readFile(resolve(project, 'public', descriptor.path));
  assert.equal(bytes.length, descriptor.bytes); assert.equal(hash(bytes), descriptor.sha256);
  const raw = gunzipSync(bytes); assert.equal(raw.length, descriptor.uncompressedBytes); assert.equal(hash(raw), descriptor.uncompressedSha256);
  return JSON.parse(raw);
}
async function fixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/expanded-kpi-manifest.json'), 'utf8'));
  const gaugeManifest = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  assert.ok(manifest.metrics.length >= 200); assert.ok(manifest.variants.length >= 3000);
  assert.equal(manifest.financialPackId, gaugeManifest.financialPackId); assert.equal(manifest.taxonomySha256, gaugeManifest.taxonomySha256);
  const index = await readArtifact(project, manifest.index), gauge = await readArtifact(project, gaugeManifest.artifact);
  assert.equal(index.ids.length, 19140); assert.deepEqual(new Set(index.ids), new Set(gauge.rows.map(row => row.id)));
  const positions = new Map(index.ids.map((id, position) => [id, position])), shards = new Map();
  const variant = (metricId, group = 'last', calculation = 'latest') => {
    const result = manifest.variants.find(v => v.metricId === metricId && v.calcGroup === group && v.calculation === calculation);
    assert.ok(result, `Verified provider choice ${metricId}/${group}/${calculation}`); return result;
  };
  const values = async variant => {
    const descriptor = manifest.shards.find(shard => shard.id === variant.shard); assert.ok(descriptor);
    if (!shards.has(descriptor.id)) shards.set(descriptor.id, await readArtifact(project, descriptor));
    const shard = shards.get(descriptor.id); assert.equal(shard.variantIds[variant.offset], variant.id);
    const observations = shard.values[variant.offset]; assert.equal(observations.length, index.ids.length);
    return new Map(index.ids.map(id => [id, observations[positions.get(id)]]));
  };
  return { manifest, gaugeManifest, rows: gauge.rows, byId: new Map(gauge.rows.map(row => [row.id, row])), variant, values };
}

function auditScript({ draftPrefix, revisionKey }) {
  if (window.__expandedListAudit) return;
  const audit = { protectedWrites: [], assetRequests: [], nativeCommands: [] };
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith(draftPrefix) || args[0] === revisionKey)) audit.protectedWrites.push({ method, key: args[0] ?? null });
      return original.apply(this, args);
    };
  }
  const original = window.fetch;
  const watched = function (input, ...rest) {
    const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
    if (url.pathname.includes('/data/expanded-kpis/')) audit.assetRequests.push(url.pathname);
    if (url.hostname === 'ipc.localhost') audit.nativeCommands.push(decodeURIComponent(url.pathname.slice(1)));
    return original.call(this, input, ...rest);
  };
  window.fetch = watched; if (window.fetch !== watched) throw new Error('Expanded list source request audit not installed');
  window.__expandedListAudit = audit;
}
async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort().map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}
async function seed(page, project, data) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  delete draft.starterOrigin; delete draft.researchOrigin; draft.researchAutofillDisabled = true; draft.scenarios.mid.cashFlows[1] = null;
  draft.title = 'TEST FIXTURE — expanded lists retain authored cash and deliberate blanks';
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const valuation = { format: 'macro-atlas-valuation', version: 1, id: 'expanded-list-authored', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: data.gaugeManifest.financialPackId, taxonomy: data.gaugeManifest.taxonomySha256, draft };
  const draftKey = `${draftPrefix}${valuation.company}:${valuation.release}:${valuation.financial}:${valuation.taxonomy}`;
  const columns = ['country', 'branch', 'stock_close', 'fcf', 'ebit_margin', 'cash_factor_30', 'terminal_share'].map(kpiId => ({ id: `legacy-${kpiId}`, kpiId, window: 'latest', calculation: 'latest' }));
  const filters = { query: '', sectorId: 'all', branchId: 'all', country: 'all', route: 'all', readiness: 'all', presence: 'all', watchlistOnly: false, preset: 'all', numericRules: [] };
  const legacy = JSON.stringify({ version: 1, columns, filters, sort: { columnId: 'name', direction: 'asc' }, watchlistIds: ['102', '99999999999999999999'], savedViews: [{ id: 'legacy-view', name: 'Legacy view', columns, filters, sort: { columnId: 'name', direction: 'asc' } }] });
  await page.evaluate(({ oldKey, key, legacy, draftKey, valuation, revisionKey }) => {
    localStorage.removeItem(key); localStorage.setItem(oldKey, legacy); localStorage.setItem(draftKey, JSON.stringify(valuation));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...valuation, id: 'expanded-list-revision' }] }));
  }, { oldKey, key, legacy, draftKey, valuation, revisionKey });
  return { legacy, protectedStorage: await protectedStorage(page), columns };
}
async function openLists(page) {
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await page.locator('[data-business-ready="true"]').waitFor();
  await page.getByLabel('Company lists', { exact: true }).click(); await list(page).waitFor();
}
async function assertReadOnly(page, expected) {
  assert.deepEqual(await protectedStorage(page), expected);
  const audit = await page.evaluate(() => window.__expandedListAudit); assert.ok(audit);
  assert.deepEqual(audit.protectedWrites, [], 'No attempted valuation or revision writes, including identical writes');
  assert.deepEqual(audit.nativeCommands.filter(command => /^(financial_(annual|begin|append|finish|cancel|export)|research_(import|export|save|delete))$/.test(command)), [], 'Company lists do not load annual histories or mutate native archives');
  assert.equal(await page.locator('.valuation-workspace').count(), 0); return audit;
}
async function waitMatches(page, expected) {
  await page.waitForFunction(expected => Number(document.querySelector('[data-company-list-ready="true"]')?.getAttribute('data-company-list-matches')) === expected, expected);
}
async function chooseProvider(page, variant) {
  await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
  await dialog.getByLabel('Search KPIs', { exact: true }).fill('');
  await dialog.getByRole('button', { name: /^All KPIs/ }).click();
  await dialog.locator(`[data-company-list-kpi-option="${variant.metricId}"]`).click();
  await dialog.getByLabel('KPI time period', { exact: true }).selectOption(`provider:${variant.source}:${variant.calcGroup}`);
  await dialog.getByLabel('KPI calculation', { exact: true }).selectOption(`provider:${variant.calculation}`);
  await dialog.getByRole('button', { name: 'Add column', exact: true }).click(); await closeDialog(page);
  const preferences = await saved(page), column = preferences.columns.findLast(column => column.kpiId === variant.metricId && column.window === `provider:${variant.source}:${variant.calcGroup}` && column.calculation === `provider:${variant.calculation}`);
  assert.ok(column); return column;
}
async function verifyProviderCell(page, id, column, expected, snapshot) {
  await page.waitForFunction(({ selector, expected }) => document.querySelector(selector)?.getAttribute('data-company-list-value') === String(expected), { selector: columnSelector(id, column), expected });
  const cell = page.locator(columnSelector(id, column));
  assert.match(await cell.innerText(), /\d/); assert.doesNotMatch(await cell.innerText(), /^\s*Pro\s*$/i);
  assert.equal(await cell.getAttribute('data-company-list-date'), '', 'Provider snapshot is not assigned an invented report or quote date');
  await cell.getByRole('button').click();
  const dialog = page.getByRole('dialog', { name: 'KPI value details', exact: true });
  assert.ok((await dialog.innerText()).includes(snapshot)); assert.match(await dialog.innerText(), /provider|Börsdata/);
  assert.match(await dialog.innerText(), /not.*(?:underlying|report|quote)|Not available/i);
  await closeDialog(page);
}
async function ensureFilters(page) {
  const panel = page.getByLabel('Company list filters', { exact: true });
  if (!await panel.isVisible()) await page.getByRole('button', { name: 'Show list filters', exact: true }).click();
}
async function ensureSorts(page) {
  const details = page.locator('.company-list-sort-controls');
  if (!await details.evaluate(node => node.open)) await details.locator('summary').click();
}

function parseCsv(text) {
  const rows = [], fields = []; let field = '', quoted = false;
  text = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) { if (char === '"' && text[i + 1] === '"') { field += '"'; i++; } else if (char === '"') quoted = false; else field += char; }
    else if (char === '"') quoted = true;
    else if (char === ',') { fields.push(field); field = ''; }
    else if (char === '\n') { fields.push(field.replace(/\r$/, '')); rows.push([...fields]); fields.length = 0; field = ''; }
    else field += char;
  }
  assert.equal(quoted, false); if (field || fields.length) { fields.push(field); rows.push([...fields]); }
  return rows;
}
async function captureCsv(page, native) {
  if (native) {
    await page.evaluate(() => {
      const original = window.fetch;
      const capture = async function (input, ...rest) {
        const response = await original.call(this, input, ...rest);
        const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
        if (url.hostname === 'ipc.localhost' && url.pathname === '/export_csv') window.__expandedNativeCsv = { path: await response.clone().json(), status: response.status };
        return response;
      };
      window.fetch = capture; if (window.fetch !== capture) throw new Error('Real native CSV response capture not installed');
      window.__restoreExpandedCsv = () => { window.fetch = original; };
    });
    try {
      await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
      await page.getByRole('button', { name: 'Export company list CSV', exact: true }).click();
      await page.waitForFunction(() => window.__expandedNativeCsv !== undefined);
      const result = await page.evaluate(() => window.__expandedNativeCsv);
      assert.equal(result.status, 200); assert.equal(typeof result.path, 'string');
      const filename = basename(result.path); assert.match(filename, /^Macro-Atlas-Companies-\d{4}-\d{2}-\d{2}-\d+\.csv$/);
      const bytes = await readFile(result.path), text = bytes.toString('utf8'); assert.ok(text.includes('"Listing ID","Company"'));
      // The native command uses create_new and a timestamp; remove only the file returned by this test's actual export.
      await unlink(result.path);
      return { filename, text, deliveredPath: result.path, bytes: bytes.length, sha256: hash(bytes) };
    } finally { await page.evaluate(() => { window.__restoreExpandedCsv?.(); delete window.__restoreExpandedCsv; }); }
  }
  await page.evaluate(() => {
    const blobs = new Map(), create = URL.createObjectURL, click = HTMLAnchorElement.prototype.click;
    URL.createObjectURL = function (blob) { const url = create.call(this, blob); blobs.set(url, blob); return url; };
    const capture = function () {
      if (this.download?.startsWith('Macro-Atlas-Companies-') && blobs.has(this.href)) {
        const result = { filename: this.download, text: null }; window.__expandedCsv = result;
        blobs.get(this.href).text().then(text => { result.text = text; }); return;
      }
      return click.call(this);
    };
    HTMLAnchorElement.prototype.click = capture;
    if (HTMLAnchorElement.prototype.click !== capture) throw new Error('CSV download capture not installed');
    window.__restoreExpandedCsv = () => { URL.createObjectURL = create; HTMLAnchorElement.prototype.click = click; };
  });
  try {
    const button = page.getByRole('button', { name: 'Export company list CSV', exact: true });
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    await button.click(); await page.waitForFunction(() => typeof window.__expandedCsv?.text === 'string');
    return await page.evaluate(() => window.__expandedCsv);
  } finally { await page.evaluate(() => { window.__restoreExpandedCsv?.(); delete window.__restoreExpandedCsv; }); }
}

async function verifyCorruptProviderShard(page, data, pe, peColumn, protectedValues) {
  const preferences = await savedBytes(page), descriptor = data.manifest.shards.find(shard => shard.id === pe.shard);
  assert.ok(descriptor);
  const marker = 'macro-atlas-expanded-kpi-fault-once';
  await page.addInitScript(({ marker, path }) => {
    if (sessionStorage.getItem(marker) !== 'armed') return;
    sessionStorage.removeItem(marker);
    const original = window.fetch, probe = { responses: 0, originalLength: null, corruptedLength: null };
    const corrupt = async function (input, ...rest) {
      const response = await original.call(this, input, ...rest);
      const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
      if (!url.pathname.endsWith(`/${path}`)) return response;
      const bytes = new Uint8Array(await response.clone().arrayBuffer());
      if (!bytes.length) throw new Error('Cannot exercise checksum failure with empty source bytes');
      probe.responses++; probe.originalLength = bytes.length; bytes[0] ^= 1; probe.corruptedLength = bytes.length;
      const headers = new Headers(response.headers); headers.delete('content-encoding'); headers.set('content-length', String(bytes.length));
      return new Response(bytes, { status: response.status, statusText: response.statusText, headers });
    };
    window.fetch = corrupt; if (window.fetch !== corrupt) throw new Error('Actual provider checksum injection not installed');
    window.__expandedKpiFault = probe;
  }, { marker, path: descriptor.path });
  await page.evaluate(marker => sessionStorage.setItem(marker, 'armed'), marker);
  try {
    // A fresh document clears the module cache, so the selected source request must really occur.
    await page.reload(); await openLists(page);
    await page.waitForFunction(() => window.__expandedKpiFault?.responses > 0);
    const probe = await page.evaluate(() => window.__expandedKpiFault);
    assert.equal(probe.originalLength, descriptor.bytes); assert.equal(probe.corruptedLength, descriptor.bytes);
    await page.waitForFunction(selector => /checksum/i.test(document.querySelector(selector)?.getAttribute('title') ?? ''), columnSelector('102', peColumn));
    const failed = page.locator(columnSelector('102', peColumn));
    assert.equal(await failed.getAttribute('data-company-list-value'), '');
    assert.match(await failed.innerText(), /—/);
    await failed.getByRole('button').click();
    assert.match(await page.getByRole('dialog', { name: 'KPI value details', exact: true }).innerText(), /checksum/i); await closeDialog(page);
    const core = row(page, '102').locator('[data-company-list-kpi="cash_factor_30"]');
    assert.equal(Number(await core.getAttribute('data-company-list-value')), data.byId.get('102').valuation.reverseCashFactor30, 'A failed additional source leaves the original cash-flow context intact');
    await ensureFilters(page);
    await page.getByRole('button', { name: 'Add numeric KPI filter', exact: true }).click();
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(peColumn.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('missing');
    await waitMatches(page, 0);
    await assertReadOnly(page, protectedValues);
    return { path: descriptor.path, checkedBytes: probe.originalLength, injectedResponses: probe.responses };
  } finally {
    await page.evaluate(({ key, preferences, marker }) => { localStorage.setItem(key, preferences); sessionStorage.removeItem(marker); }, { key, preferences, marker });
    await page.reload(); await openLists(page);
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    assert.equal(await savedBytes(page), preferences); await assertReadOnly(page, protectedValues);
  }
}

export async function expandedCompanyListFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), checks = [], screenshots = [], viewport = page.viewportSize();
  const checked = message => { checks.push(message); console.log(`Expanded company lists: ${message}`); };
  const pe = data.variant('provider_2'), peHistory = data.variant('provider_2', '5year', 'mean');
  const peValues = await data.values(pe), historyValues = await data.values(peHistory);
  const seedState = await seed(page, project, data);
  await page.addInitScript(auditScript, { draftPrefix, revisionKey });
  await page.reload(); await openLists(page);
  try {
    assert.equal(await savedBytes(page), null, 'Opening legacy lists does not eagerly write a replacement');
    assert.equal(await page.evaluate(oldKey => localStorage.getItem(oldKey), oldKey), seedState.legacy);
    assert.equal(await count(page), 19140);
    assert.deepEqual((await page.evaluate(() => window.__expandedListAudit.assetRequests)).filter(path => /shard-/.test(path)), [], 'Core-only lists do not load provider shards');
    await page.getByRole('button', { name: 'Choose KPI columns', exact: true }).click();
    let dialog = page.getByRole('dialog', { name: 'Choose KPI columns', exact: true });
    const visibleMetricIds = await dialog.locator('[data-company-list-kpi-option]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-company-list-kpi-option')).filter(id => id.startsWith('provider_')));
    const availableMetrics = data.manifest.metrics.filter(metric => data.manifest.variants.some(variant => variant.metricId === metric.id && variant.availableCount > 0)).map(metric => metric.id);
    assert.deepEqual(new Set(visibleMetricIds), new Set(availableMetrics), 'Default picker exposes metrics with actual saved values');
    await dialog.getByLabel('Only KPIs with saved values', { exact: true }).uncheck();
    assert.ok(await dialog.locator('[data-company-list-kpi-option]').count() >= 241);
    await dialog.locator('[data-company-list-kpi-option="provider_2"]').click();
    const options = await dialog.getByLabel('KPI time period', { exact: true }).locator('option').evaluateAll(nodes => nodes.map(node => node.value));
    for (const years of [1, 3, 5, 7, 10, 15]) assert.ok(options.some(option => option.endsWith(`:${years}year`)), `P/E exposes ${years}-year provider choice`);
    await closeDialog(page);
    assert.equal(await savedBytes(page), null); assert.equal((await page.evaluate(() => window.__expandedListAudit.assetRequests)).filter(path => /shard-/.test(path)).length, 0);
    checked('Legacy lists open without writes; the expanded catalogue and all six year windows can be inspected without loading value shards');

    const peColumn = await chooseProvider(page, pe);
    let preferences = await saved(page); assert.equal(preferences.version, 2);
    assert.deepEqual(preferences.features.watchlists.find(list => list.id === 'default').listingIds, ['102', '99999999999999999999']);
    assert.equal(preferences.savedViews[0].id, 'legacy-view');
    assert.equal(await page.evaluate(oldKey => localStorage.getItem(oldKey), oldKey), seedState.legacy);
    await search(page, 'Holmen'); assert.ok(finite(peValues.get('102')));
    await verifyProviderCell(page, '102', peColumn, peValues.get('102'), data.manifest.snapshot);
    const historyColumn = await chooseProvider(page, peHistory);
    await verifyProviderCell(page, '102', historyColumn, historyValues.get('102'), data.manifest.snapshot);
    const holmen = data.byId.get('102');
    assert.equal(Number(await page.locator(columnSelector('102', seedState.columns.find(column => column.kpiId === 'cash_factor_30'))).getAttribute('data-company-list-value')), holmen.valuation.reverseCashFactor30);
    checked('The first explicit edit migrates to a separate v2 key; latest and five-year provider P/E match exact source values while frozen valuation context remains unchanged');

    await page.getByRole('button', { name: 'Manage watchlists', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Manage watchlists', exact: true });
    await dialog.getByLabel('Watchlist name', { exact: true }).fill('Potential compounders');
    await dialog.getByRole('button', { name: 'Create watchlist', exact: true }).click();
    preferences = await saved(page); const namedId = preferences.features.activeWatchlistId; assert.notEqual(namedId, 'default');
    await closeDialog(page);
    const watched = ['102', '2', '34'].map(id => data.byId.get(id)); assert.ok(watched.every(Boolean));
    for (const item of watched) { await search(page, item.name); await row(page, item.id).getByRole('button', { name: `Add ${item.name} to watchlist`, exact: true }).click(); }
    await search(page, ''); await page.getByRole('button', { name: 'Watchlist', exact: true }).click(); await waitMatches(page, watched.length);
    assert.deepEqual(new Set(await rowIds(page)), new Set(watched.map(item => item.id)));
    await page.getByRole('button', { name: 'Manage watchlists', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Manage watchlists', exact: true });
    await dialog.getByRole('button', { name: 'Rename watchlist Potential compounders', exact: true }).click();
    await dialog.getByLabel('Watchlist name', { exact: true }).fill('Deep dive candidates');
    await dialog.getByRole('button', { name: 'Rename', exact: true }).click(); await closeDialog(page);
    preferences = await saved(page);
    assert.deepEqual(preferences.watchlistIds, ['102', '99999999999999999999']);
    assert.equal(preferences.features.watchlists.find(item => item.id === namedId).name, 'Deep dive candidates');
    await page.getByLabel('Active watchlist', { exact: true }).selectOption('default'); await waitMatches(page, 1);
    assert.deepEqual(await rowIds(page), ['102']);
    await page.getByLabel('Active watchlist', { exact: true }).selectOption(namedId); await waitMatches(page, 3);
    checked('Named watchlists create, rename and retain separate memberships; legacy stars and out-of-pack IDs remain intact');

    for (const item of watched) await row(page, item.id).getByLabel(`Compare ${item.name}`, { exact: true }).check();
    await page.getByRole('button', { name: 'Compare selected companies', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Compare companies', exact: true });
    for (const item of watched) {
      const expected = await page.locator(columnSelector(item.id, peColumn)).getByRole('button').innerText();
      assert.equal(await dialog.locator(`[data-company-compare-listing="${item.id}"][data-company-compare-kpi="provider_2"]`).first().locator('strong').innerText(), expected);
    }
    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: width < 500 ? 844 : 960 });
      await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth + 1);
      const path = resolve(project, `test-results/expanded-company-list-${native ? 'native' : 'browser'}-compare-${width}.png`);
      await page.screenshot({ path, fullPage: true }); screenshots.push(path);
    }
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    await dialog.getByRole('button', { name: `Remove ${watched[2].name} from comparison`, exact: true }).click();
    assert.deepEqual((await saved(page)).features.comparisonIds, watched.slice(0, 2).map(item => item.id)); await closeDialog(page);
    checked('Comparison renders the same selected source observations side by side, supports removal, and fits desktop and narrow screens');

    await ensureSorts(page);
    await page.getByLabel('Sort priority 1', { exact: true }).selectOption('legacy-country');
    await page.getByRole('button', { name: 'Add sort priority', exact: true }).click();
    await page.getByLabel('Sort priority 2', { exact: true }).selectOption(peColumn.id);
    await page.getByLabel('Sort direction 2', { exact: true }).selectOption('desc');
    await page.getByRole('button', { name: 'Add sort priority', exact: true }).click();
    await page.getByLabel('Sort priority 3', { exact: true }).selectOption('name');
    assert.equal(await page.getByRole('button', { name: 'Add sort priority', exact: true }).isDisabled(), true);
    const expectedWatched = [...watched].sort((a, b) => collator.compare(a.country ?? '', b.country ?? '') || (peValues.get(b.id) ?? -Infinity) - (peValues.get(a.id) ?? -Infinity) || compareName(a, b));
    assert.deepEqual(await rowIds(page), expectedWatched.map(item => item.id));
    await page.getByLabel('Table density', { exact: true }).selectOption('compact');
    await page.getByRole('button', { name: 'Save view', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Save company list view', exact: true });
    await dialog.getByLabel('View name', { exact: true }).fill('Expanded acceptance view');
    await dialog.getByRole('button', { name: 'Save current view', exact: true }).click();
    preferences = await saved(page); const view = preferences.savedViews.find(view => view.name === 'Expanded acceptance view'); assert.ok(view);
    assert.deepEqual(preferences.features.viewFeatures[view.id], { activeWatchlistId: namedId, density: 'compact', secondarySorts: [{ columnId: peColumn.id, direction: 'desc' }, { columnId: 'name', direction: 'asc' }] });
    checked('Three sorting priorities reconcile independently; a named view saves column choices, active list, density and all sort directions');

    await page.getByRole('button', { name: 'All listings', exact: true }).click(); await ensureFilters(page);
    await page.getByRole('button', { name: 'Add numeric KPI filter', exact: true }).click();
    await page.getByLabel('KPI for condition 1', { exact: true }).selectOption(peColumn.id);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('between');
    await page.getByLabel('Value for condition 1', { exact: true }).fill('10'); await page.getByLabel('Upper value for condition 1', { exact: true }).fill('20');
    const bounded = data.rows.filter(item => finite(peValues.get(item.id)) && peValues.get(item.id) >= 10 && peValues.get(item.id) <= 20);
    await waitMatches(page, bounded.length);
    await page.getByLabel('Upper value for condition 1', { exact: true }).fill(''); await waitMatches(page, 0);
    assert.equal((await saved(page)).filters.numericRules[0].valueTo, null);
    await assertReadOnly(page, seedState.protectedStorage); await page.reload(); await openLists(page); await ensureFilters(page);
    assert.equal(await page.getByLabel('Upper value for condition 1', { exact: true }).inputValue(), ''); await waitMatches(page, 0);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('missing');
    const missing = data.rows.filter(item => !finite(peValues.get(item.id))); await waitMatches(page, missing.length);
    await page.getByLabel('Operator for condition 1', { exact: true }).selectOption('present');
    const present = data.rows.filter(item => finite(peValues.get(item.id))); await waitMatches(page, present.length);
    checked('Between and missing/present filters reconcile with every provider observation; a blank upper bound survives reload and matches no companies');

    await page.getByLabel('Company list country', { exact: true }).selectOption('SE');
    const exportRows = present.filter(item => item.country === 'SE'); assert.ok(exportRows.length > 50); await waitMatches(page, exportRows.length);
    const csv = await captureCsv(page, native), csvRows = parseCsv(csv.text), exportPreferences = await saved(page);
    assert.equal(csvRows.length, exportRows.length + 1); assert.deepEqual(new Set(csvRows.slice(1).map(record => record[0])), new Set(exportRows.map(item => item.id)));
    assert.equal(csvRows[0].length, 6 + exportPreferences.columns.length * 5); assert.ok(csvRows.every(record => record.length === csvRows[0].length));
    const peOffset = 6 + exportPreferences.columns.findIndex(column => column.id === peColumn.id) * 5;
    assert.ok(csvRows[0][peOffset].includes(data.manifest.snapshot)); assert.ok(csvRows[0][peOffset].includes(`provider:${pe.source}:${pe.calcGroup}`));
    for (const record of csvRows.slice(1)) { assert.equal(Number(record[peOffset]), peValues.get(record[0])); assert.equal(record[peOffset + 3], '', 'CSV does not substitute snapshot date for report date'); }
    assert.match(csv.filename, /^Macro-Atlas-Companies-.*\.csv$/);
    checked(`CSV exports all ${exportRows.length} filtered Swedish listings beyond the visible page, with exact values, units, separate dates and source explanations`);

    await page.getByRole('button', { name: 'Clear list filters', exact: true }).click();
    await page.getByLabel('Column preset', { exact: true }).selectOption('valuation');
    preferences = await saved(page); assert.ok(preferences.columns.some(column => column.kpiId === 'provider_2')); assert.ok(preferences.columns.some(column => column.kpiId === 'provider_4')); assert.ok(preferences.columns.some(column => column.kpiId === 'cash_factor_30'));
    await page.getByLabel('Table density', { exact: true }).selectOption('comfortable');
    await page.getByLabel('Active watchlist', { exact: true }).selectOption('default');
    await page.getByLabel('Saved company list view', { exact: true }).selectOption(view.id);
    await waitMatches(page, watched.length);
    assert.equal(await page.getByLabel('Active watchlist', { exact: true }).inputValue(), namedId);
    assert.equal(await page.getByLabel('Table density', { exact: true }).inputValue(), 'compact');
    assert.deepEqual((await saved(page)).columns, view.columns);
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    assert.deepEqual(await rowIds(page), expectedWatched.map(item => item.id));
    checked('The valuation column preset adds supported choices; restoring a saved view restores its original columns, named list and complete sort setup');

    await page.getByRole('button', { name: 'Manage watchlists', exact: true }).click();
    dialog = page.getByRole('dialog', { name: 'Manage watchlists', exact: true });
    await dialog.getByLabel('Watchlist name', { exact: true }).fill('Temporary list');
    await dialog.getByRole('button', { name: 'Create watchlist', exact: true }).click();
    await dialog.getByRole('button', { name: 'Delete watchlist Temporary list', exact: true }).click();
    assert.equal((await saved(page)).features.activeWatchlistId, 'default');
    assert.equal((await saved(page)).features.watchlists.some(item => item.name === 'Temporary list'), false); await closeDialog(page);
    await page.getByLabel('Saved company list view', { exact: true }).selectOption('');
    await page.getByLabel('Saved company list view', { exact: true }).selectOption(view.id); await waitMatches(page, watched.length);
    assert.equal(await page.evaluate(oldKey => localStorage.getItem(oldKey), oldKey), seedState.legacy);
    await assertReadOnly(page, seedState.protectedStorage);
    const beforeReload = await savedBytes(page); await page.reload(); await openLists(page); await waitMatches(page, watched.length);
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
    assert.equal(await savedBytes(page), beforeReload); assert.deepEqual(await rowIds(page), expectedWatched.map(item => item.id));
    assert.equal(await page.evaluate(oldKey => localStorage.getItem(oldKey), oldKey), seedState.legacy);
    const audit = await assertReadOnly(page, seedState.protectedStorage);
    const requested = [...new Set(audit.assetRequests.filter(path => /shard-/.test(path)))];
    assert.ok(requested.length > 0 && requested.length < data.manifest.shards.length, 'Restart fetches selected source shards, not the entire catalogue');
    checked('Deleting a named list repairs its active reference; reload preserves exact v2 preferences, legacy bytes and authored valuation/revision bytes');

    const corruptShard = await verifyCorruptProviderShard(page, data, pe, peColumn, seedState.protectedStorage);
    await waitMatches(page, watched.length); assert.deepEqual(await rowIds(page), expectedWatched.map(item => item.id));
    checked('A real corrupted provider response fails checksum validation; failed data matches no missing-value filter, core values remain visible, and a clean reload recovers');
    const restart = { protectedStorage: seedState.protectedStorage, preferences: await savedBytes(page), legacy: seedState.legacy, ids: expectedWatched.map(item => item.id), activeWatchlistId: namedId, comparisonIds: watched.slice(0, 2).map(item => item.id), savedViewId: view.id };
    return { checks, screenshots, listings: data.rows.length, providerMetrics: data.manifest.metrics.length, providerVariants: data.manifest.variants.length, exportRows: exportRows.length, corruptShard, ...(native ? { nativeCsvDelivery: { path: csv.deliveredPath, bytes: csv.bytes, sha256: csv.sha256, removedAfterVerification: true } } : {}), pack: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, selectedShardsOnRestart: requested, restart };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/expanded-company-list-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally { await page.setViewportSize(viewport ?? { width: 1500, height: 960 }); }
}

export async function assertExpandedCompanyListRestart(page, result) {
  await openLists(page); await waitMatches(page, result.ids.length);
  await page.waitForFunction(() => !document.querySelector('button[aria-label="Export company list CSV"]')?.disabled);
  assert.equal(await savedBytes(page), result.preferences); assert.equal(await page.evaluate(oldKey => localStorage.getItem(oldKey), oldKey), result.legacy);
  assert.deepEqual(await protectedStorage(page), result.protectedStorage); assert.deepEqual(await rowIds(page), result.ids);
  assert.equal(await page.getByLabel('Active watchlist', { exact: true }).inputValue(), result.activeWatchlistId);
  assert.equal(await page.getByLabel('Table density', { exact: true }).inputValue(), 'compact');
  assert.deepEqual((await saved(page)).features.comparisonIds, result.comparisonIds);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}
