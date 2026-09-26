import { selectObservatory, openListPanel, closeListPanel } from './workspace-navigation.mjs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

const key = 'macro-atlas-company-lists-v2', oldKey = 'macro-atlas-company-lists-v1';
const draftPrefix = 'macro-atlas-valuation-draft-v1:', revisionKey = 'macro-atlas-valuations-v1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const list = page => page.locator('[data-company-list-ready="true"]');
const row = (page, id) => page.locator(`[data-company-listing="${id}"]`);
const saved = page => page.evaluate(key => JSON.parse(localStorage.getItem(key)), key);
const savedBytes = page => page.evaluate(key => localStorage.getItem(key), key);
const search = (page, value) => page.getByLabel('Company list search', { exact: true }).fill(value);
const searchValue = page => page.getByLabel('Company list search', { exact: true }).inputValue();

async function fixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  const { version } = JSON.parse(await readFile(resolve(project, 'package.json'), 'utf8'));
  const bytes = await readFile(resolve(project, 'public', manifest.artifact.path));
  assert.equal(hash(bytes), manifest.artifact.sha256); assert.equal(bytes.length, manifest.artifact.bytes);
  const raw = gunzipSync(bytes); assert.equal(hash(raw), manifest.artifact.uncompressedSha256); assert.equal(raw.length, manifest.artifact.uncompressedBytes);
  const artifact = JSON.parse(raw); assert.equal(artifact.rows.length, manifest.rows);
  return { manifest, version, rows: artifact.rows, byId: new Map(artifact.rows.map(row => [row.id, row])) };
}

async function capexFixture(project, id, financialManifest) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/expanded-kpi-manifest.json'), 'utf8'));
  assert.equal(manifest.financialPackId, financialManifest.financialPackId); assert.equal(manifest.taxonomySha256, financialManifest.taxonomySha256);
  const read = async descriptor => {
    const bytes = await readFile(resolve(project, 'public', descriptor.path));
    assert.equal(bytes.length, descriptor.bytes); assert.equal(hash(bytes), descriptor.sha256);
    const raw = gunzipSync(bytes); assert.equal(raw.length, descriptor.uncompressedBytes); assert.equal(hash(raw), descriptor.uncompressedSha256);
    return JSON.parse(raw);
  };
  const index = await read(manifest.index), position = index.ids.indexOf(id); assert.ok(position >= 0);
  const values = [];
  for (const [group, calculation] of [['last', 'latest'], ['5year', 'mean']]) {
    const variant = manifest.variants.find(value => value.metricId === 'provider_64' && value.source === 'screener' && value.calcGroup === group && value.calculation === calculation); assert.ok(variant);
    const shard = await read(manifest.shards.find(value => value.id === variant.shard)); assert.equal(shard.variantIds[variant.offset], variant.id);
    values.push({ window: `provider:screener:${group}`, value: shard.values[variant.offset][position] });
  }
  return { values, snapshot: manifest.snapshot };
}

function auditScript({ draftPrefix, revisionKey }) {
  if (window.__usabilityAudit) return;
  const audit = { protectedWrites: [], notebookWrites: [], nativeCommands: [] };
  for (const method of ['setItem', 'removeItem', 'clear']) {
    const original = Storage.prototype[method];
    Storage.prototype[method] = function (...args) {
      if (this === localStorage && (method === 'clear' || String(args[0]).startsWith(draftPrefix) || args[0] === revisionKey)) audit.protectedWrites.push({ method, key: args[0] ?? null });
      if (this === localStorage && String(args[0]).startsWith('macro-atlas-company-notes-v1:')) audit.notebookWrites.push({ method, key: args[0] });
      return original.apply(this, args);
    };
  }
  const original = window.fetch;
  window.fetch = function (input, ...rest) {
    const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
    if (url.hostname === 'ipc.localhost') audit.nativeCommands.push(decodeURIComponent(url.pathname.slice(1)));
    return original.call(this, input, ...rest);
  };
  window.__usabilityAudit = audit;
}
async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort().map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}
function authoredValuations(records) {
  return Object.fromEntries(Object.entries(records).map(([key, raw]) => {
    if (!key.startsWith(draftPrefix)) return [key, raw];
    const { id: _mountEnvelopeId, created: _mountTimestamp, ...authored } = JSON.parse(raw);
    return [key, authored];
  }));
}
async function seed(page, project, data) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  delete draft.starterOrigin; delete draft.researchOrigin; draft.researchAutofillDisabled = true; draft.scenarios.mid.cashFlows[1] = null;
  draft.title = 'TEST FIXTURE — guidance preserves authored assumptions and deliberate blanks';
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const valuation = { format: 'macro-atlas-valuation', version: 1, id: 'usability-authored', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, draft };
  const draftKey = `${draftPrefix}${valuation.company}:${valuation.release}:${valuation.financial}:${valuation.taxonomy}`;
  await page.evaluate(({ key, oldKey, draftKey, valuation, revisionKey }) => {
    localStorage.removeItem(key); localStorage.removeItem(oldKey); localStorage.setItem(draftKey, JSON.stringify(valuation));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...valuation, id: 'usability-revision' }] }));
  }, { key, oldKey, draftKey, valuation, revisionKey });
  const columns = ['country', 'price_date', 'annual_periods', 'ebit_margin', 'fcf', 'stock_close'].map(kpiId => ({ id: kpiId, kpiId, window: 'latest', calculation: 'latest' }));
  const filters = { query: '', sectorId: 'all', branchId: 'all', country: 'all', route: 'all', readiness: 'all', presence: 'all', watchlistOnly: false, preset: 'all', numericRules: [] };
  const features = { version: 1, watchlists: [{ id: 'default', name: 'Watchlist', listingIds: [] }], activeWatchlistId: 'default', comparisonIds: [], secondarySorts: [], density: 'comfortable', viewFeatures: {} };
  await page.evaluate(({ key, columns, filters, features }) => localStorage.setItem(key, JSON.stringify({ version: 2, columns, filters, features, sort: { columnId: 'name', direction: 'asc' }, watchlistIds: [], savedViews: [] })), { key, columns, filters, features });
  return protectedStorage(page);
}
async function waitMatches(page, count) {
  await page.waitForFunction(count => Number(document.querySelector('[data-company-list-ready="true"]')?.getAttribute('data-company-list-matches')) === count, count);
}
async function assertReadOnly(page, protectedValues) {
  assert.deepEqual(await protectedStorage(page), protectedValues);
  const audit = await page.evaluate(() => window.__usabilityAudit); assert.ok(audit);
  assert.deepEqual(audit.protectedWrites, [], 'No attempted draft or revision writes, including identical writes');
  assert.deepEqual(audit.nativeCommands.filter(command => /^research_(import|export|save|delete)$/.test(command)), []);
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
}

const trigger = page => page.getByRole('button', { name: 'Open getting started guide', exact: true });
const guide = page => page.getByRole('dialog', { name: 'What would you like to understand?', exact: true });
const bound = (page, column, name = 'Min') => page.locator(`[data-company-list-range="${column.id}"]`).getByRole('textbox', { name: new RegExp(`^${name} `) });
const explain = (page, column) => page.locator(`th[data-company-list-column="${column.id}"]`).getByRole('button', { name: /^Explain / });
async function screenshot(page, project, name, screenshots, locator) {
  const path = resolve(project, `test-results/usability-${name}.png`);
  if (locator) await locator.screenshot({ path }); else await page.screenshot({ path, fullPage: !name.includes('-opening-') });
  screenshots.push(path);
}

// Check actual rendered text against its composited background. These samples
// cover navigation, explanation copy, headings and bound labels; this is not a
// whole-application accessibility certification.
async function readableText(page, selectors) {
  const results = await page.evaluate(selectors => {
    const rgba = text => {
      const values = text.match(/[\d.]+/g)?.map(Number);
      if (!values || values.length < 3) throw new Error(`Unsupported computed color ${text}`);
      return [values[0] / 255, values[1] / 255, values[2] / 255, values[3] ?? 1];
    };
    const over = (front, back) => [0, 1, 2].map(index => front[index] * front[3] + back[index] * (1 - front[3]));
    const luminance = rgb => rgb.map(channel => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    return selectors.map(selector => {
      const node = document.querySelector(selector); if (!node) return { selector, missing: true };
      const ancestors = []; for (let element = node; element; element = element.parentElement) ancestors.unshift(element);
      let background = [1, 1, 1];
      for (const ancestor of ancestors) background = over(rgba(getComputedStyle(ancestor).backgroundColor), background);
      const style = getComputedStyle(node), foreground = over(rgba(style.color), background), light = [luminance(foreground), luminance(background)].sort((a, b) => a - b);
      return { selector, size: parseFloat(style.fontSize), contrast: (light[1] + 0.05) / (light[0] + 0.05), color: style.color, background, text: node.textContent.trim().slice(0, 90) };
    });
  }, selectors);
  for (const result of results) {
    assert.equal(result.missing, undefined, `Readability sample exists: ${result.selector}`);
  }
  return results;
}

// A locator click scrolls controls into view automatically. Measure the opening
// viewport explicitly so a long setup form cannot pass merely because its table
// or chart exists far below the fold.
async function openingGeometry(page, selector) {
  // Resizing the viewport changes CSS before the chart's ResizeObserver has
  // resized its canvas. Measure the rendered chart after those widths agree.
  if (selector.includes('canvas')) await page.waitForFunction(selector => {
    const canvases = [...document.querySelectorAll(selector)];
    return canvases.length > 0 && canvases.every(canvas => {
      const host = canvas.closest('[role="img"]');
      return host && Math.abs(canvas.getBoundingClientRect().width - host.getBoundingClientRect().width) <= 1;
    });
  }, selector);
  return page.locator(selector).evaluateAll(nodes => nodes.map(node => {
    const box = node.getBoundingClientRect();
    let top = Math.max(0, box.top), bottom = Math.min(innerHeight, box.bottom);
    let left = Math.max(0, box.left), right = Math.min(innerWidth, box.right);
    for (let parent = node.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent), rect = parent.getBoundingClientRect();
      if (/(hidden|auto|scroll|clip)/.test(style.overflowY)) { top = Math.max(top, rect.top); bottom = Math.min(bottom, rect.bottom); }
      if (/(hidden|auto|scroll|clip)/.test(style.overflowX)) { left = Math.max(left, rect.left); right = Math.min(right, rect.right); }
    }
    return { top: box.top, bottom: box.bottom, height: box.height, width: box.width, visibleHeight: Math.max(0, bottom - top), visibleWidth: Math.max(0, right - left) };
  }));
}

async function assertSectionReached(page, selector) {
  // Section links participate in Atlas history, so verify their actual scroll
  // destination instead of requiring an unrelated browser hash-history entry.
  await page.waitForFunction(selector => {
    const node = document.querySelector(selector); if (!node) return false;
    const box = node.getBoundingClientRect();
    let top = 0, bottom = innerHeight;
    for (let parent = node.parentElement; parent; parent = parent.parentElement) {
      if (/(hidden|auto|scroll|clip)/.test(getComputedStyle(parent).overflowY)) {
        const rect = parent.getBoundingClientRect(); top = Math.max(top, rect.top); bottom = Math.min(bottom, rect.bottom);
      }
    }
    return box.top >= top - 2 && box.top < bottom - 20;
  }, selector, { timeout: 4500 });
}

async function listOpening(page, project, native, screenshots) {
  const observations = [];
  for (const width of [1500, 390]) {
    const height = width === 390 ? 844 : 960;
    await page.setViewportSize({ width, height });
    await selectObservatory(page, 'companies'); await list(page).waitFor();
    for (const name of ['Show list filters', 'Show column sets', 'Show list tools']) assert.equal(await page.getByRole('button', { name, exact: true }).getAttribute('aria-expanded'), 'false', `${name} is optional on opening`);
    await page.locator('[data-company-listing]').first().waitFor();
    const rows = await openingGeometry(page, '[data-company-listing]');
    const fullyVisible = rows.filter(row => row.height > 0 && row.visibleHeight >= row.height - 1 && row.visibleWidth >= 150).length;
    await screenshot(page, project, `${native ? 'native' : 'browser'}-opening-lists-${width}`, screenshots);
    assert.ok(fullyVisible >= (width === 390 ? 1 : 3), `${width}px Lists opens with real company rows, observed ${fullyVisible}`);
    if (width === 1500) assert.ok(rows[0].top < height * .65, `${width}px first row starts promptly, y=${rows[0].top}`);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), true, 'The page fits the viewport; the table owns its horizontal scroll');
    observations.push({ width, height, firstRowTop: rows[0].top, fullyVisibleRows: fullyVisible });
  }
  await page.setViewportSize({ width: 1500, height: 960 });
  return observations;
}

export async function usabilityFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), checks = [], screenshots = [], samples = [], viewport = page.viewportSize();
  const checked = message => { checks.push(message); console.log(`Usability: ${message}`); };
  assert.equal(await page.locator('.app').getAttribute('data-active-observatory'), 'companies', 'A fresh profile starts with company discovery');
  await list(page).waitFor();
  const openingLists = await listOpening(page, project, native, screenshots);
  checked('A fresh profile opens Lists with real company rows already in the desktop and narrow viewport; advanced panels remain optional');
  const protectedValues = await seed(page, project, data);
  await page.addInitScript(auditScript, { draftPrefix, revisionKey }); await page.reload();
  try {
    await trigger(page).click(); await guide(page).waitFor();
    assert.match(await guide(page).innerText(), /key performance indicator/i);
    assert.match(await guide(page).innerText(), /watchlist|saved views/i);
    await guide(page).getByText('How to read the numbers and charts', { exact: true }).click();
    assert.match(await guide(page).innerText(), /DCF is discounted cash flow/);
    assert.match(await guide(page).innerText(), /Terminal value.*already included in DCF/);
    samples.push(...await readableText(page, ['.atlas-guide > p', '.atlas-guide-steps li']));
    await screenshot(page, project, `${native ? 'native' : 'browser'}-guide`, screenshots);
    await page.keyboard.press('Escape'); await guide(page).waitFor({ state: 'hidden' });
    await page.waitForFunction(() => document.activeElement?.getAttribute('aria-label') === 'Open getting started guide');
    await trigger(page).click(); await guide(page).getByRole('button', { name: /^Find companies to research/ }).click();
    await list(page).waitFor(); assert.equal(await page.getByLabel('Observatory', { exact: true }).inputValue(), 'companies');
    assert.equal(await guide(page).count(), 0);
    checked('Start here explains the workflow, units and DCF/NPV; Escape returns focus and Find companies opens Lists directly');

    await openListPanel(page, 'columns');
    await page.getByLabel('Column preset', { exact: true }).selectOption('terminal_sensitivity'); await waitMatches(page, data.rows.length);
    await closeListPanel(page, 'columns');
    let preferences = await saved(page);
    const column = preferences.columns.find(column => column.kpiId === 'valuation_attractiveness' && column.calculation === 'terminal_50'); assert.ok(column);
    const frozenBytes = await savedBytes(page);
    await explain(page, column).click();
    let dialog = page.getByRole('dialog', { name: 'KPI explained', exact: true }); await dialog.waitFor();
    const about = dialog.getByRole('region', { name: 'About Valuation attractiveness', exact: true });
    // An explicitly named section may map to region, while definition text remains
    // available to a reader without depending on a particular semantic wrapper.
    assert.match(await dialog.innerText(), /What this measures/);
    assert.match(await dialog.innerText(), /How to read it/);
    assert.match(await dialog.innerText(), /Count 50% of terminal value/);
    assert.match(await dialog.innerText(), /Units[\s\S]*(percent|%)/i);
    assert.match(await dialog.innerText(), /Period[\s\S]*Calculation/);
    if (await about.count()) assert.ok(await about.isVisible());
    await page.keyboard.press('Escape'); await dialog.waitFor({ state: 'hidden' });
    assert.equal(await savedBytes(page), frozenBytes, 'Reading a KPI explanation does not change the selected variant or filters');
    await bound(page, column).fill('9000000000000000'); await waitMatches(page, 0);
    await explain(page, column).click(); dialog = page.getByRole('dialog', { name: 'KPI explained', exact: true }); await dialog.waitFor();
    assert.match(await dialog.innerText(), /Count 50% of terminal value/);
    await page.keyboard.press('Escape'); await dialog.waitFor({ state: 'hidden' }); await bound(page, column).fill('0');
    await page.waitForFunction(() => Number(document.querySelector('[data-company-list-ready="true"]')?.getAttribute('data-company-list-matches')) > 0);
    samples.push(...await readableText(page, ['.company-list-heading p', '.company-list-column-name', '[data-company-list-range] label span', '.company-list-range-note span']));
    checked('KPI header help explains the exact 50% terminal variant, definition and units without changing settings, including when no company matches');

    // Use a known, source-verified company that survives the selected range.
    const candidate = data.byId.get('102'); assert.ok(candidate);
    await bound(page, column).fill('-100');
    await search(page, candidate.name);
    await row(page, candidate.id).waitFor();
    const beforeCompany = await savedBytes(page);
    await row(page, candidate.id).getByRole('button', { name: `Open profile for ${candidate.name}`, exact: true }).click();
    const workspace = page.locator('[data-observatory="companies"][data-business-view="financials"]'); await workspace.waitFor();
    assert.equal(await workspace.getAttribute('data-company-map-visible'), 'false');
    assert.match(await page.getByLabel('Company financials', { exact: true }).innerText(), /Financials/);
    assert.match(await page.getByLabel('Company valuation', { exact: true }).innerText(), /Valuation/);
    await page.getByRole('button', { name: 'Show listing map', exact: true }).click();
    await page.waitForFunction(() => document.querySelector('[data-observatory="companies"]')?.getAttribute('data-company-map-visible') === 'true');
    await page.getByRole('button', { name: 'Hide listing map', exact: true }).click();
    await page.getByRole('button', { name: 'Back to Lists', exact: true }).click(); await list(page).waitFor();
    assert.equal(await savedBytes(page), beforeCompany); assert.equal(await bound(page, column).inputValue(), '-100');
    await assertReadOnly(page, protectedValues);
    for (const domain of ['sectors', 'macro', 'companies']) await selectObservatory(page, domain);
    await list(page).waitFor(); assert.equal(await savedBytes(page), beforeCompany); assert.equal(await searchValue(page), candidate.name);
    checked('Financials and Valuation have a selected-company context; listing map is optional, and returning through Lists or broader domains restores exact ranges and search');

    await search(page, '');
    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: 960 });
      await bound(page, column).click({ timeout: 15000 });
      assert.equal(await bound(page, column).evaluate(node => document.activeElement === node), true);
      assert.equal(await bound(page, column).evaluate(node => { const rect = node.getBoundingClientRect(); return document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2) === node; }), true, 'The range input is not hidden underneath a pinned company column');
      await screenshot(page, project, `${native ? 'native' : 'browser'}-lists-${width}`, screenshots);
      await screenshot(page, project, `${native ? 'native' : 'browser'}-table-${width}`, screenshots, page.getByRole('region', { name: 'Company list results', exact: true }));
    }
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    checked('Desktop and narrow Lists permit a real click on Min, restore keyboard focus and leave the input centre unobscured');

    await search(page, candidate.name); await row(page, candidate.id).getByRole('button', { name: `Open profile for ${candidate.name}`, exact: true }).click();
    await page.locator('#company-business-overview').waitFor();
    assert.deepEqual(await page.locator('[data-business-figure-group]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-business-figure-group'))), ['performance', 'cash', 'capital']);
    assert.deepEqual(await page.locator('[data-business-chart]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-business-chart'))), ['revenue', 'profitability', 'operating-cash', 'investing-cash', 'assets', 'financing']);
    assert.match(await page.locator('[data-business-capex-note]').innerText(), /Maintenance and growth capex are not separated/);
    assert.match(await page.locator('[data-business-capex-note]').innerText(), /acquisitions.*financial investments.*asset sales/);
    assert.match(await page.locator('[data-business-capex-note]').innerText(), /compare its definition with the separately saved capex/);
    const capex = await capexFixture(project, candidate.id, data.manifest);
    const capexPanel = page.locator(`[data-provider-capex="${candidate.id}"][data-provider-capex-ready="true"]`); await capexPanel.waitFor();
    for (const observation of capex.values) {
      assert.equal(await capexPanel.locator(`[data-capex-variant="${observation.window}"]`).getAttribute('data-capex-value'), observation.value === null ? '' : String(observation.value));
    }
    assert.match(await capexPanel.innerText(), /underlying fiscal dates are unavailable/);
    assert.match(await capexPanel.innerText(), /do not follow the annual\/quarterly selector/);
    assert.ok((await capexPanel.innerText()).includes(capex.snapshot));
    const financialOpening = [];
    for (const width of [1500, 390]) {
      await page.setViewportSize({ width, height: width === 390 ? 844 : 960 });
      const chart = page.locator('[data-business-chart="revenue"] canvas'); await chart.waitFor();
      const [geometry] = await openingGeometry(page, '[data-business-chart="revenue"] canvas');
      financialOpening.push({ viewportWidth: width, ...geometry });
      await screenshot(page, project, `${native ? 'native' : 'browser'}-opening-business-${width}`, screenshots);
    }
    await page.setViewportSize({ width: 1500, height: 960 });
    await page.getByRole('button', { name: 'Quarterly', exact: true }).click();
    assert.match(await page.locator('[data-business-chart="revenue"]').getByRole('img').getAttribute('aria-label'), /quarterly business overview$/);
    for (const observation of capex.values) assert.equal(await capexPanel.locator(`[data-capex-variant="${observation.window}"]`).getAttribute('data-capex-value'), observation.value === null ? '' : String(observation.value), 'Changing statement frequency does not relabel or recalculate independent provider capex');
    await page.getByRole('button', { name: 'Annual', exact: true }).click();
    checked('Business overview orders revenue/profitability, operating/investing cash, and assets/financing charts; independently verified provider capex remains separate from dated investing cash');
    await page.getByLabel('Company financial chart', { exact: true }).waitFor();
    const chart = page.getByLabel('Company financial chart', { exact: true }), financialNav = page.getByRole('navigation', { name: 'Within company financial history', exact: true });
    assert.equal(await chart.getAttribute('aria-describedby'), 'financial-metric-explanation');
    await chart.selectOption('free_cash_flow'); assert.match(await page.locator('#financial-metric-explanation').innerText(), /free cash flow|FCF/i);
    for (const [name, target] of [['Financial trends', '#company-financial-chart'], ['Report details', '#company-financial-statements'], ['Saved market values', '#company-market-history']]) {
      const link = financialNav.getByRole('link', { name, exact: true }); assert.equal(await link.getAttribute('href'), target);
      await link.click(); await assertSectionReached(page, target); assert.equal(await page.locator(target).count(), 1);
    }
    await assertReadOnly(page, protectedValues);
    checked('Financial history explains the selected measure and provides working links to trends, report details and dated market values without changing saved valuations');

    const notebookKey = id => `macro-atlas-company-notes-v1:${id}`;
    const firstNote = 'TEST FIXTURE — Revenue quality matters more than a one-year rebound.';
    const nextNote = 'TEST FIXTURE — Reconcile investing cash with acquisitions and actual capex.';
    await page.getByLabel('Company research', { exact: true }).click();
    const noteLabels = ['1. Understand the business', '2. Explain the financial history', '3. Challenge the assumptions', '4. Decide what to investigate next'];
    for (const label of noteLabels) assert.equal(await page.getByLabel(label, { exact: true }).inputValue(), '');
    assert.deepEqual(await page.evaluate(() => window.__usabilityAudit.notebookWrites), [], 'Merely opening the notebook does not create or replace saved notes');
    await page.getByLabel(noteLabels[0], { exact: true }).fill(firstNote);
    await page.getByLabel(noteLabels[3], { exact: true }).fill(nextNote);
    const holmenNotes = await page.evaluate(key => localStorage.getItem(key), notebookKey(candidate.id)); assert.ok(holmenNotes);
    const other = data.byId.get('696'); assert.ok(other);
    await page.getByLabel('Search companies or countries', { exact: true }).fill(other.name); await page.locator(`[data-search-listing="${other.id}"]`).click();
    await page.getByLabel('Company research', { exact: true }).click();
    for (const label of noteLabels) assert.equal(await page.getByLabel(label, { exact: true }).inputValue(), '', 'A different company does not inherit the first notebook');
    await page.getByLabel(noteLabels[0], { exact: true }).fill('TEST FIXTURE — Separate company notebook.');
    assert.equal(await page.evaluate(key => localStorage.getItem(key), notebookKey(candidate.id)), holmenNotes);
    await page.getByLabel('Search companies or countries', { exact: true }).fill(candidate.name); await page.locator(`[data-search-listing="${candidate.id}"]`).click();
    await page.getByLabel('Company research', { exact: true }).click();
    await page.reload(); await page.locator('[data-business-view="research"]').waitFor();
    assert.equal(await page.getByLabel(noteLabels[0], { exact: true }).inputValue(), firstNote);
    assert.equal(await page.getByLabel(noteLabels[3], { exact: true }).inputValue(), nextNote);
    assert.equal(await page.evaluate(key => localStorage.getItem(key), notebookKey(candidate.id)), holmenNotes);
    await assertReadOnly(page, protectedValues);
    checked('The four research steps save per-company notes only on explicit edits, isolate another company, and restore the research screen and exact notes after reload without changing valuation drafts');

    await page.getByLabel('Company valuation', { exact: true }).click();
    await page.locator(`[data-valuation-company="${candidate.id}"]`).waitFor();
    await page.locator('.cash-flow-forecast canvas').waitFor();
    const valuationOpening = await openingGeometry(page, '.cash-flow-forecast canvas');
    assert.equal(await page.getByRole('button', { name: 'Back to Lists', exact: true }).isVisible(), true);
    const reading = page.getByRole('region', { name: 'How to use this valuation', exact: true });
    await reading.getByText('DCF, NPV and terminal value explained', { exact: true }).click();
    assert.match(await reading.innerText(), /discounted cash flow/); assert.match(await reading.innerText(), /net present value/);
    assert.match(await reading.innerText(), /100.*70/); assert.match(await reading.innerText(), /once/);
    const presentationBaseline = await protectedStorage(page);
    const scenarios = page.getByRole('tab', { name: 'Value, price & payback', exact: true }), evidence = page.getByRole('tab', { name: 'Business, capital & evidence', exact: true });
    await scenarios.focus(); await page.keyboard.press('ArrowRight');
    assert.equal(await evidence.getAttribute('aria-selected'), 'true'); assert.equal(await evidence.getAttribute('tabindex'), '0');
    assert.equal(await page.getByRole('tabpanel', { name: 'Business, capital & evidence', exact: true }).isVisible(), true);
    await page.keyboard.press('Home'); assert.equal(await scenarios.getAttribute('aria-selected'), 'true');
    assert.equal(await page.getByRole('tabpanel', { name: 'Value, price & payback', exact: true }).isVisible(), true);
    const valuationNav = page.getByRole('navigation', { name: 'Within this valuation', exact: true });
    for (const [name, target] of [['Cash forecast', '#valuation-cash'], ['Value today', '#valuation-results'], ['Terminal assumptions', '#valuation-terminal'], ['Purchase price', '#valuation-purchase'], ['Saved revisions', '#valuation-saved']]) {
      const link = valuationNav.getByRole('link', { name, exact: true }); assert.equal(await link.getAttribute('href'), target);
      await link.click(); await assertSectionReached(page, target); assert.equal(await page.locator(target).count(), 1);
    }
    assert.deepEqual(await protectedStorage(page), presentationBaseline, 'Definitions, keyboard tabs and section links only change presentation');
    await screenshot(page, project, `${native ? 'native' : 'browser'}-valuation`, screenshots);
    checked('Valuation defines cash, DCF, NPV, terminal value and margin of safety; accessible keyboard tabs and section links work without changing the working assumptions');
    await page.waitForFunction(() => JSON.parse(localStorage.getItem('atlas.preferences') ?? '{}').companyView === 'valuation');
    await page.reload(); await page.locator(`[data-valuation-company="${candidate.id}"]`).waitFor();
    assert.deepEqual(authoredValuations(await protectedStorage(page)), authoredValuations(presentationBaseline), 'Reopening preserves every authored draft/basis field and exact revision bytes; existing autosave renews only the envelope ID and timestamp');
    await page.getByRole('button', { name: 'Back to Lists', exact: true }).click(); await list(page).waitFor();
    assert.equal(await bound(page, column).inputValue(), '-100'); assert.equal(await searchValue(page), candidate.name);
    checked('Valuation is restored on reload and its direct Back to Lists returns to the same candidate and range');
    for (const geometry of financialOpening) assert.ok(geometry.visibleHeight >= (geometry.viewportWidth === 390 ? 70 : 120), `${geometry.viewportWidth}px financial overview opens on a real revenue chart: ${JSON.stringify(geometry)}`);
    assert.ok(valuationOpening[0]?.visibleHeight >= 120, `Valuation opens with a real cash chart: ${JSON.stringify(valuationOpening)}`);
    for (const sample of samples) {
      assert.ok(sample.size >= 12, `Readable text at least 12px: ${JSON.stringify(sample)}`);
      assert.ok(sample.contrast >= 4.5, `Normal text contrast at least 4.5:1: ${JSON.stringify(sample)}`);
    }
    return { checks, screenshots, openingLists, financialOpening, valuationOpening, readabilitySamples: samples, listings: data.rows.length, version: data.version, protectedValuationsUnchangedUntilValuationOpened: true, valuationPresentationPreserved: true };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/usability-${native ? 'native' : 'browser'}-failure-viewport.png`) }).catch(() => {});
    await page.screenshot({ path: resolve(project, `test-results/usability-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {}); throw error;
  } finally { await page.setViewportSize(viewport ?? { width: 1500, height: 960 }); }
}
