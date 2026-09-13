import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const keys = ['low', 'mid', 'high'];
const names = { low: 'Low', mid: 'Mid', high: 'High' };
const prefix = 'macro-atlas-valuation-draft-v1:';
const revisionKey = 'macro-atlas-valuations-v1';
const preferenceKey = 'atlas.preferences';
// Generated once from the preserved pre-terminal runtime and validated source
// pack, never from the implementation under test. Native runs read only JSON.
const oldV3Provenance = {
  fixture: 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json',
  fixtureSha256: 'ea49c698bcf2c8642edced3b333928b5700f8c2dff410ac570aba3c6548d144b',
  runtime: 'test-results/empirical-cash-starter-2026-09-13/starter-runtime.executed.mjs',
  runtimeSha256: '69b2e991851a3554583005125f7616cc53cb19b30c6be1168e4d63c2d8162af9',
  sourcePackSha256: '1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a',
  taxonomySha256: 'cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff',
  companyPayloadSha256: 'd84d07d6f315697dde1697d2584512327e7fcc52c45bcb86f8369682ebf07872',
  asOf: '2026-09-13'
};
const near = (actual, expected, label) => assert.ok(Number.isFinite(Number(actual)) && Math.abs(Number(actual) - expected) <= Math.max(1e-7, Math.abs(expected) * 1e-10), `${label}: ${actual} != ${expected}`);

async function openCompany(page, name, id) {
  if (await page.locator(`[data-valuation-company="${id}"]`).count()) return;
  await page.getByLabel('Search companies or countries').fill(name);
  await page.locator(`[data-search-listing="${id}"]`).click();
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function reopen(page, id) {
  await page.reload();
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function savedDraft(page, id = '696') {
  return page.evaluate(({ id, prefix }) => {
    const key = Object.keys(localStorage).find(key => key.startsWith(`${prefix}${id}:`));
    return key ? { key, saved: JSON.parse(localStorage.getItem(key)) } : null;
  }, { id, prefix });
}

async function installFixture(page, id, draft) {
  const stored = await savedDraft(page, id);
  assert.ok(stored);
  await page.evaluate(({ key, saved, draft }) => localStorage.setItem(key, JSON.stringify({ ...saved, draft })), { ...stored, draft });
  await reopen(page, id);
}

async function inputs(page) {
  if (await page.getByLabel('Mid required return', { exact: true }).count() === 0) await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
}

async function chartRows(page, kind) {
  return page.locator(`section[data-valuation-chart="${kind}"] [data-range-year]`).evaluateAll(rows => rows.map(row => ({
    year: Number(row.getAttribute('data-range-year')),
    basis: row.querySelector('[data-range-basis]')?.getAttribute('data-range-basis'),
    sale: row.getAttribute('data-includes-sale') === 'true',
    ...Object.fromEntries([...row.querySelectorAll('[data-range-value]')].map(cell => [cell.getAttribute('data-range-value'), Number(cell.getAttribute('data-value'))]))
  })));
}

async function cashBasis(page) {
  return page.locator('.cash-flow-forecast [data-year]').evaluateAll(rows => rows.map(row => ({
    year: Number(row.getAttribute('data-year')), kind: row.getAttribute('data-range-kind')
  })));
}

const terminalInputs = draft => Object.fromEntries(keys.map(key => [key, structuredClone(draft.scenarios[key].terminalCash)]));
const annualCash = draft => Object.fromEntries(keys.map(key => [key, structuredClone(draft.scenarios[key].cashFlows)]));
function effectiveSale(scenario) {
  return scenario.terminalCash ? Math.max(0, scenario.terminalCash.cashFlow) / ((scenario.discountRate - scenario.terminalCash.growthRate) / 100) : scenario.terminalEquity;
}

function independent(draft, sale) {
  const scale = draft.investment / draft.marketCap;
  return Object.fromEntries(keys.map(key => {
    const scenario = draft.scenarios[key], rate = 1 + scenario.discountRate / 100;
    const dcf = scenario.cashFlows.slice(0, draft.years).map((cash, index) => cash / rate ** (index + 1) * scale);
    const npv = [-draft.investment];
    for (const cash of dcf) npv.push(npv.at(-1) + cash);
    if (sale) npv[draft.years] += effectiveSale(scenario) / rate ** draft.years * scale;
    return [key, { dcf, npv }];
  }));
}

async function verifyPaths(page, draft, sale = false) {
  const expected = independent(draft, sale);
  for (const kind of ['dcf', 'npv']) {
    const rows = await chartRows(page, kind);
    assert.equal(rows.length, draft.years + Number(kind === 'npv'));
    rows.forEach((row, i) => {
      const values = keys.map(key => expected[key][kind][i]);
      for (const [key, value] of Object.entries({ low: values[0], mid: values[1], high: values[2], min: Math.min(...values), max: Math.max(...values) })) near(row[key], value, `${kind} year ${row.year} ${key}`);
    });
  }
}

function csvRecords(csv) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const character = csv[i];
    if (character === '"') {
      if (quoted && csv[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted;
    } else if (!quoted && character === ',') { row.push(cell); cell = ''; }
    else if (!quoted && character === '\r' && csv[i + 1] === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; i++; }
    else cell += character;
  }
  row.push(cell); rows.push(row);
  const [header, ...data] = rows;
  return data.filter(values => values.length === header.length).map(values => Object.fromEntries(header.map((key, index) => [key, values[index]])));
}

async function exportRows(page, project, native) {
  let csv;
  if (native) {
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const status = page.getByRole('status').filter({ hasText: 'Saved to ' }); await status.waitFor();
    csv = await readFile((await status.innerText()).replace(/^Saved to /, ''), 'utf8');
  } else {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const download = await pending, path = resolve(project, 'test-results/terminal-valuation-export.csv');
    await download.saveAs(path); csv = await readFile(path, 'utf8');
  }
  return csvRecords(csv);
}

export async function terminalValuationFlows(page, project, { native = false } = {}) {
  const checks = [], viewport = page.viewportSize();
  const original = await page.evaluate(({ prefix, revisionKey, preferenceKey }) => ({
    drafts: Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(prefix)).map(key => [key, localStorage.getItem(key)])),
    revisions: localStorage.getItem(revisionKey), preferences: localStorage.getItem(preferenceKey),
    valuationOpen: !!document.querySelector('.valuation-workspace')
  }), { prefix, revisionKey, preferenceKey });
  const originalCompany = JSON.parse(original.preferences).companyId;
  try {
    await openCompany(page, 'Stora Enso R', '696');
    await page.getByRole('button', { name: 'Company profile', exact: true }).click();
    await page.evaluate(prefix => Object.keys(localStorage).filter(key => key.startsWith(`${prefix}696:`)).forEach(key => localStorage.removeItem(key)), prefix);
    await reopen(page, '696');
    await page.locator('[data-valuation-company="696"][data-valuation-ready="true"]').waitFor();
    await inputs(page);
    const standard = (await savedDraft(page)).saved.draft;
    assert.equal(standard.starterOrigin.id, 'empirical-cash-starter-v3');
    assert.equal(standard.starterOrigin.terminalMethod, 'historical-median-v1');
    // Median of the preserved signed annual history [705,-181,-561,840,1027].
    for (const [key, cash] of [['low', 564], ['mid', 705], ['high', 846]]) {
      near(standard.scenarios[key].terminalCash.cashFlow, cash, `${key} signed-history median sensitivity`);
      assert.equal(standard.scenarios[key].terminalCash.growthRate, 0);
      near(effectiveSale(standard.scenarios[key]), cash / .1, `${key} independent terminal sale`);
      assert.equal(await page.getByLabel(`${names[key]} terminal value method`, { exact: true }).inputValue(), 'sustainable');
    }
    const sourceBasis = await cashBasis(page);
    assert.equal(sourceBasis.filter(row => row.kind === 'historical').length, 4);
    assert.equal(sourceBasis.filter(row => row.kind === 'assumed-tail').length, 6);
    checks.push('New Stora defaults retain the v3 cash calibration and seed a separate signed-history median terminal cash with explicit ±20% sensitivities and zero mature growth');

    const oldBytes = await readFile(resolve(project, oldV3Provenance.fixture));
    assert.equal(createHash('sha256').update(oldBytes).digest('hex'), oldV3Provenance.fixtureSha256);
    const oldV3 = JSON.parse(oldBytes); oldV3.investment = 2500;
    await installFixture(page, '696', oldV3);
    await page.waitForFunction(() => {
      const key = Object.keys(localStorage).find(key => key.startsWith('macro-atlas-valuation-draft-v1:696:'));
      return key && JSON.parse(localStorage.getItem(key)).draft.starterOrigin?.terminalMethod === 'historical-median-v1';
    }, undefined, { timeout: 30_000 });
    near((await savedDraft(page)).saved.draft.investment, 2500, 'Untouched old-v3 migration retains investment');
    const backups = await page.evaluate(key => JSON.parse(localStorage.getItem(key)).items, revisionKey);
    assert.ok(backups.some(revision => JSON.stringify(revision.draft) === JSON.stringify(oldV3)), 'Migration first saves the complete original old-v3 draft');
    await page.locator('.valuation-saved button').filter({ hasText: oldV3.title }).first().click();
    await reopen(page, '696');
    const restoredOld = (await savedDraft(page)).saved.draft;
    assert.equal(restoredOld.starterOrigin.terminalMethod, undefined);
    assert.deepEqual(restoredOld, { ...oldV3, researchAutofillDisabled: true });
    checks.push('The exact frozen pre-terminal v3 draft upgrades only after a complete saved backup, retains investment, and an explicitly restored old revision survives reload without upgrading again');

    await inputs(page);
    await page.getByLabel('Low final equity sale', { exact: true }).fill('4321');
    await page.getByLabel('Mid terminal value method', { exact: true }).selectOption('sustainable');
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('555');
    await page.getByLabel('Mid mature cash growth', { exact: true }).fill('3');
    await page.getByLabel('Mid required return', { exact: true }).fill('12');
    const oldWithExplicitTerminal = (await savedDraft(page)).saved.draft;
    assert.equal(oldWithExplicitTerminal.starterOrigin.terminalMethod, undefined);
    assert.equal(oldWithExplicitTerminal.scenarios.low.terminalCash, undefined);
    assert.equal(oldWithExplicitTerminal.scenarios.high.terminalCash, undefined);
    const oldSettings = page.locator('.valuation-history-settings');
    if (await oldSettings.getAttribute('open') === null) await oldSettings.locator('summary').click();
    await page.getByLabel('Later-year widening (% of cash scale)', { exact: true }).fill('20');
    await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
    const changedOldHistory = (await savedDraft(page)).saved.draft;
    assert.notEqual(changedOldHistory.scenarios.high.cashFlows[9], oldWithExplicitTerminal.scenarios.high.cashFlows[9]);
    for (const key of keys) {
      assert.deepEqual(changedOldHistory.scenarios[key].terminalCash, oldWithExplicitTerminal.scenarios[key].terminalCash);
      assert.equal(changedOldHistory.scenarios[key].terminalEquity, oldWithExplicitTerminal.scenarios[key].terminalEquity);
      assert.equal(changedOldHistory.scenarios[key].discountRate, oldWithExplicitTerminal.scenarios[key].discountRate);
    }
    checks.push('An explicitly restored old-v3 study can adopt sustainable terminal cash without the median-origin flag; applying annual history settings preserves that cash/growth/return and the other scenarios’ manual sales');
    await installFixture(page, '696', standard); await inputs(page);

    const settings = page.locator('.valuation-history-settings');
    if (await settings.getAttribute('open') === null) await settings.locator('summary').click();
    await page.getByLabel('Later-year widening (% of cash scale)', { exact: true }).fill('25');
    await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
    await inputs(page);
    const wider = (await savedDraft(page)).saved.draft;
    assert.notEqual(wider.scenarios.high.cashFlows[9], standard.scenarios.high.cashFlows[9]);
    assert.deepEqual(terminalInputs(wider), terminalInputs(standard));
    for (const key of keys) near(effectiveSale(wider.scenarios[key]), effectiveSale(standard.scenarios[key]), `${key} sale independent of Year 10 widening`);
    const widerCash = annualCash(wider), widerDcf = await chartRows(page, 'dcf');
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('500');
    await page.getByLabel('Mid mature cash growth', { exact: true }).fill('2');
    const terminalEdited = (await savedDraft(page)).saved.draft;
    assert.deepEqual(annualCash(terminalEdited), widerCash);
    assert.deepEqual(await cashBasis(page), sourceBasis);
    assert.deepEqual(await chartRows(page, 'dcf'), widerDcf);
    await page.getByLabel('Mid required return', { exact: true }).fill('12');
    const rateEdited = (await savedDraft(page)).saved.draft;
    near(effectiveSale(rateEdited.scenarios.mid), 5000, 'Terminal denominator follows the current required return');
    assert.deepEqual(annualCash(rateEdited), widerCash);
    assert.deepEqual(await cashBasis(page), sourceBasis);
    await verifyPaths(page, rateEdited);
    checks.push('Annual tail widening retains independent terminal inputs; terminal cash/growth edits leave annual cash and calibrated labels intact, while required-return edits recalculate both discounting and terminal value');

    await openCompany(page, 'Nordea Bank', '159');
    const fixture = structuredClone((await savedDraft(page, '159')).saved.draft);
    delete fixture.researchOrigin; delete fixture.starterOrigin; delete fixture.crisis;
    Object.assign(fixture, { title: 'TEST FIXTURE — separate sustainable terminal cash', currency: 'SEK', valuationDate: '2026-09-13', priceDate: '2026-09-12', priceSource: 'Independent hypothetical sustainable cash test; not a company valuation.', marketCap: 1000, investment: 2500, years: 3, researchAutofillDisabled: true });
    for (const key of keys) Object.assign(fixture.scenarios[key], { cashFlows: [100, 200, 300], discountRate: 25,
      terminalEquity: 987654, terminalCash: { cashFlow: 500, growthRate: 5 }, recoveryEquity: null, recoveryYear: null,
      rationale: 'First post-horizon cash is 500 after reinvestment. The stale manual cache must not affect this calculation.' });
    await installFixture(page, '159', fixture);
    await page.locator('[data-valuation-company="159"][data-valuation-ready="true"]').waitFor();
    await inputs(page);
    await verifyPaths(page, fixture);
    const withoutSale = await chartRows(page, 'npv'), dcf = await chartRows(page, 'dcf');
    await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).check();
    await verifyPaths(page, fixture, true);
    const withSale = await chartRows(page, 'npv');
    near(withSale.at(-1).mid, 1604, 'Independent terminal sale 2500 / 1.25^3 plus cash PV 361.6, scaled to investment, less purchase');
    assert.deepEqual(withSale.slice(0, -1), withoutSale.slice(0, -1));
    assert.deepEqual(await chartRows(page, 'dcf'), dcf);
    const exported = await exportRows(page, project, native);
    for (const key of keys) {
      const row = exported.find(row => row.scenario === key && row.year === '3');
      near(row.final_equity_sale_m, 2500, `${key} CSV resolves the terminal model instead of a stale cache`);
      near(row.terminal_cash_after_reinvestment_m, 500, `${key} CSV first post-horizon cash`);
      near(row.terminal_growth_pct, 5, `${key} CSV growth`);
      near(row.cumulative_npv_with_sale_m, 641.6, `${key} CSV company-level NPV`);
      assert.match(row.terminal_method, /sustainable/);
    }
    checks.push('An independent three-year case reconciles terminal value, final-only cumulative NPV and CSV from first post-horizon cash without multiplying growth twice or using a stale manual-sale cache');

    await page.getByLabel('Mid final equity sale', { exact: true }).fill('400');
    const manual = (await savedDraft(page, '159')).saved.draft;
    assert.equal(manual.scenarios.mid.terminalCash, undefined);
    assert.equal(manual.scenarios.mid.terminalEquity, 400);
    assert.equal(await page.getByLabel('Mid terminal value method', { exact: true }).inputValue(), 'manual');
    await verifyPaths(page, manual, true);
    await reopen(page, '159'); await inputs(page);
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, manual);
    await page.getByLabel('Mid terminal value method', { exact: true }).selectOption('sustainable');
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('500');
    await page.getByLabel('Mid mature cash growth', { exact: true }).fill('25');
    await page.locator('[data-valuation-company="159"][data-valuation-ready="false"]').waitFor();
    assert.equal(await page.locator('.valuation-charts').count(), 0);
    const invalidGrowth = (await savedDraft(page, '159')).saved.draft;
    await reopen(page, '159'); await inputs(page);
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, invalidGrowth);
    assert.equal(await page.getByLabel('Mid mature cash growth', { exact: true }).inputValue(), '25');
    await page.getByLabel('Mid mature cash growth', { exact: true }).fill('5');
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('');
    await page.locator('[data-valuation-ready="false"]').waitFor();
    const cleared = (await savedDraft(page, '159')).saved.draft;
    assert.equal(cleared.scenarios.mid.terminalCash.cashFlow, null);
    await reopen(page, '159'); await inputs(page);
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, cleared);
    assert.equal(await page.getByLabel('Mid sustainable terminal cash', { exact: true }).inputValue(), '');
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('500');
    await page.locator('[data-valuation-ready="true"]').waitFor();
    checks.push('A manual sale edit removes sustainable mode; growth equal to return and deliberately cleared terminal cash fail closed and survive reload without refilling saved assumptions');

    const beforeCrisis = structuredClone((await savedDraft(page, '159')).saved.draft.scenarios);
    await page.getByLabel('Enable crisis scenario', { exact: true }).check();
    for (const [label, value] of [['Crisis cash reduction (%)', 0], ['Crisis start year', 1], ['Crisis duration (years)', 1], ['Crisis recovery (years)', 0], ['Crisis extra annual cash cost', 0], ['Crisis required return', 25], ['Crisis final equity sale', 0]]) await page.getByLabel(label, { exact: true }).fill(String(value));
    const crisisExport = (await exportRows(page, project, native)).filter(row => row.scenario === 'crisis');
    const crisisFinal = crisisExport.find(row => row.year === '3');
    near(crisisFinal.final_equity_sale_m, 0, 'Crisis retains its own explicit zero sale');
    near(crisisFinal.cumulative_npv_with_sale_m, -638.4, 'Crisis does not inherit the sustainable mid terminal model');
    assert.deepEqual((await savedDraft(page, '159')).saved.draft.scenarios, beforeCrisis);
    const preserved = (await savedDraft(page, '159')).saved.draft;
    await reopen(page, '159'); await inputs(page);
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, preserved);
    checks.push('The optional crisis keeps its own manual sale and independently reconciled NPV without inheriting or changing the sustainable low/mid/high terminal models; all settings persist');

    for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(size);
      const control = page.getByLabel('Mid sustainable terminal cash', { exact: true });
      await control.scrollIntoViewIfNeeded();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      for (const key of keys) for (const label of ['terminal value method', 'sustainable terminal cash', 'mature cash growth', 'final equity sale']) {
        const box = await page.getByLabel(`${names[key]} ${label}`, { exact: true }).boundingBox();
        assert.ok(box && box.x >= 0 && box.x + box.width <= size.width + 1, `${key} ${label} fits ${size.width}px`);
      }
      await page.screenshot({ path: resolve(project, `test-results/terminal-valuation-${native ? 'native' : 'browser'}-${size.width}.png`) });
    }
    checks.push('Sustainable terminal cash, growth, method and manual-sale controls fit desktop and mobile widths');
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/terminal-valuation-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    if (await page.locator('.valuation-workspace').count()) await page.getByRole('button', { name: 'Company profile', exact: true }).click();
    await page.evaluate(({ original, prefix, revisionKey, preferenceKey }) => {
      const touched = key => ['159', '696'].some(id => key.startsWith(`${prefix}${id}:`));
      for (const key of Object.keys(localStorage).filter(touched)) if (!(key in original.drafts)) localStorage.removeItem(key);
      for (const [key, raw] of Object.entries(original.drafts)) if (touched(key)) localStorage.setItem(key, raw);
      for (const [key, raw] of [[revisionKey, original.revisions], [preferenceKey, original.preferences]]) {
        if (raw === null) localStorage.removeItem(key); else localStorage.setItem(key, raw);
      }
    }, { original, prefix, revisionKey, preferenceKey });
    await page.reload();
    await page.locator(`[data-company="${originalCompany}"][data-business-ready="true"]`).waitFor();
    if (original.valuationOpen) {
      await page.getByLabel('Company valuation', { exact: true }).click();
      await page.locator(`[data-valuation-company="${originalCompany}"]`).waitFor();
      await page.evaluate(({ drafts, prefix, originalCompany }) => {
        for (const [key, raw] of Object.entries(drafts)) if (key.startsWith(`${prefix}${originalCompany}:`)) localStorage.setItem(key, raw);
      }, { drafts: original.drafts, prefix, originalCompany });
    }
    if (viewport) await page.setViewportSize(viewport);
    const restored = await page.evaluate(({ prefix, revisionKey, preferenceKey }) => ({
      drafts: Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(prefix)).map(key => [key, localStorage.getItem(key)])),
      revisions: localStorage.getItem(revisionKey), preferences: localStorage.getItem(preferenceKey),
      valuationOpen: !!document.querySelector('.valuation-workspace')
    }), { prefix, revisionKey, preferenceKey });
    assert.deepEqual(restored, original, 'Restore every original draft, saved revision, company and preference after transient terminal tests');
  }
  checks.push('Temporary terminal fixtures and their revisions are removed while all previous drafts, reviewed studies, crisis settings and selections are restored byte for byte');
  return { checks, oldV3Provenance };
}

/** Leave one complete terminal draft in the isolated native profile for a real
 * process-restart check. Return to Holmen profile for the existing restart flow. */
export async function seedTerminalRestart(page) {
  await openCompany(page, 'Nordea Bank', '159'); await inputs(page);
  const title = 'TEST FIXTURE — native sustainable terminal restart';
  await page.getByLabel('Valuation study title', { exact: true }).fill(title);
  await page.getByLabel('Mid terminal value method', { exact: true }).selectOption('sustainable');
  await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('555');
  await page.getByLabel('Mid mature cash growth', { exact: true }).fill('3');
  await page.getByLabel('Mid required return', { exact: true }).fill('12');
  await page.waitForFunction(({ prefix, title }) => {
    const key = Object.keys(localStorage).find(key => key.startsWith(`${prefix}159:`));
    if (!key) return false;
    const draft = JSON.parse(localStorage.getItem(key)).draft;
    return draft.title === title && draft.scenarios.mid.terminalCash?.cashFlow === 555
      && draft.scenarios.mid.terminalCash.growthRate === 3 && draft.scenarios.mid.discountRate === 12;
  }, { prefix, title }, { timeout: 30_000 });
  await page.locator('[data-valuation-company="159"][data-valuation-ready="true"]').waitFor();
  const { key, saved } = await savedDraft(page, '159');
  const expected = { key, draft: saved.draft, basis: Object.fromEntries(['company', 'release', 'financial', 'taxonomy'].map(name => [name, saved[name]])) };
  await verifyPaths(page, expected.draft);
  await openCompany(page, 'Holmen', '102');
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
  return expected;
}

export async function assertTerminalRestart(page, expected) {
  await openCompany(page, 'Nordea Bank', '159');
  await page.locator('[data-valuation-company="159"][data-valuation-ready="true"]').waitFor();
  const { key, saved } = await savedDraft(page, '159');
  assert.equal(key, expected.key);
  assert.deepEqual(Object.fromEntries(['company', 'release', 'financial', 'taxonomy'].map(name => [name, saved[name]])), expected.basis);
  assert.equal(JSON.stringify(saved.draft), JSON.stringify(expected.draft), 'Native restart retains the exact terminal draft and its company/data basis');
  await inputs(page);
  near(await page.getByLabel('Mid final equity sale', { exact: true }).inputValue(), 555 / .09, 'Native restart resolves the saved sustainable cash/growth/return');
  await verifyPaths(page, expected.draft);
  await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).check();
  await verifyPaths(page, expected.draft, true);
  await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).uncheck();
  assert.equal(JSON.stringify((await savedDraft(page, '159')).saved.draft), JSON.stringify(expected.draft));
  await openCompany(page, 'Holmen', '102');
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
}

async function standalone() {
  const { chromium } = await import('playwright');
  const project = resolve(dirname(fileURLToPath(import.meta.url)), '..'), base = process.env.ATLAS_URL ?? 'http://127.0.0.1:1420';
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const external = [], errors = [], context = await browser.newContext({ viewport: { width: 1500, height: 960 }, acceptDownloads: true });
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
    await page.goto(base); await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
    await page.locator('[data-business-ready="true"]').waitFor();
    const result = await terminalValuationFlows(page, project);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await writeFile(resolve(project, 'test-results/terminal-valuation-browser-report.json'), JSON.stringify({ status: 'passed', ...result, external, errors }, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await standalone();
