import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const keys = ['low', 'mid', 'high'];
const prefix = 'macro-atlas-valuation-draft-v1:';
const revisionKey = 'macro-atlas-valuations-v1';
const preferenceKey = 'atlas.preferences';
const near = (actual, expected, label) => assert.ok(actual !== null && actual !== '' && Number.isFinite(Number(actual)) && Math.abs(Number(actual) - expected) <= Math.max(1e-7, Math.abs(expected) * 1e-10), `${label}: ${actual} != ${expected}`);

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

async function savedDraft(page, id = '159') {
  return page.evaluate(({ id, prefix }) => {
    const key = Object.keys(localStorage).find(key => key.startsWith(`${prefix}${id}:`));
    return key ? { key, saved: JSON.parse(localStorage.getItem(key)) } : null;
  }, { id, prefix });
}

async function installFixture(page, draft, id = '159') {
  const stored = await savedDraft(page, id); assert.ok(stored);
  await page.evaluate(({ key, saved, draft }) => localStorage.setItem(key, JSON.stringify({ ...saved, draft })), { ...stored, draft });
  await reopen(page, id);
  await page.locator('[data-purchase-range="true"]').waitFor();
}

async function inputs(page) {
  if (await page.getByLabel('Mid required return', { exact: true }).count() === 0) await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
}

function hypothetical(original) {
  const draft = structuredClone(original);
  delete draft.researchOrigin; delete draft.starterOrigin; delete draft.crisis; delete draft.purchaseRange;
  Object.assign(draft, { title: 'TEST FIXTURE — purchase-price arithmetic', currency: 'SEK', valuationDate: '2026-09-13', priceDate: '2026-09-12', priceSource: 'Independent hypothetical whole-equity purchase test; not a company valuation.', marketCap: 1000, investment: 2500, years: 3, researchAutofillDisabled: true });
  for (const [key, terminalCash] of [['low', 250], ['mid', 500], ['high', 750]]) Object.assign(draft.scenarios[key], {
    cashFlows: [100, 200, 300], discountRate: 25, terminalEquity: 987654,
    terminalCash: { cashFlow: terminalCash, growthRate: 5 }, recoveryEquity: null, recoveryYear: null,
    rationale: 'First post-horizon equity cash is already after reinvestment; stale sale cache must be ignored.'
  });
  return draft;
}

function independentlyValue(draft, key) {
  const scenario = draft.scenarios[key], rate = 1 + scenario.discountRate / 100;
  const cashPV = scenario.cashFlows.slice(0, draft.years).reduce((sum, cash, i) => sum + cash / rate ** (i + 1), 0);
  const sale = scenario.terminalCash ? Math.max(0, scenario.terminalCash.cashFlow) / ((scenario.discountRate - scenario.terminalCash.growthRate) / 100) : scenario.terminalEquity;
  const terminalPV = sale / rate ** draft.years;
  return { cashPV, terminalPV, value: cashPV + terminalPV };
}

async function purchaseRows(page) {
  return page.locator('[data-purchase-range="true"] [data-purchase-scenario]').evaluateAll(rows => Object.fromEntries(rows.map(row => {
    const number = name => { const raw = row.getAttribute(name); return raw === null || raw === '' ? null : Number(raw); };
    return [row.getAttribute('data-purchase-scenario'), {
      value: number('data-equity-value'), ceiling: number('data-price-ceiling'), cashOnly: number('data-cash-only-ceiling'),
      terminalPV: number('data-terminal-pv'), candidateNPV: number('data-candidate-npv')
    }];
  })));
}

async function verifyPurchases(page, draft, margin, candidate) {
  const rows = await purchaseRows(page); assert.deepEqual(Object.keys(rows).sort(), [...keys].sort());
  for (const key of keys) {
    const expected = independentlyValue(draft, key), row = rows[key];
    near(row.value, expected.value, `${key} whole-equity present value`);
    near(row.terminalPV, expected.terminalPV, `${key} terminal dependence`);
    if (margin === null || margin >= 100) { assert.equal(row.ceiling, null); assert.equal(row.cashOnly, null); }
    else {
      if (expected.value <= 0) assert.equal(row.ceiling, null);
      else near(row.ceiling, expected.value * (1 - margin / 100), `${key} scenario price ceiling`);
      if (expected.cashPV <= 0) assert.equal(row.cashOnly, null);
      else near(row.cashOnly, expected.cashPV * (1 - margin / 100), `${key} cash-only ceiling`);
    }
    if (candidate === null) assert.equal(row.candidateNPV, null);
    else near(row.candidateNPV, expected.value - candidate, `${key} candidate NPV`);
  }
  return rows;
}

function csvRecords(csv) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const c = csv[i];
    if (c === '"') { if (quoted && csv[i + 1] === '"') { cell += '"'; i++; } else quoted = !quoted; }
    else if (!quoted && c === ',') { row.push(cell); cell = ''; }
    else if (!quoted && c === '\r' && csv[i + 1] === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; i++; }
    else if (!quoted && c === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += c;
  }
  row.push(cell); rows.push(row);
  const [header, ...data] = rows;
  return data.filter(values => values.length === header.length).map(values => Object.fromEntries(header.map((key, i) => [key, values[i]])));
}

async function exportPurchase(page, project, native) {
  let csv;
  if (native) {
    await page.getByRole('button', { name: 'Export purchase range', exact: true }).click();
    const status = page.getByRole('status').filter({ hasText: 'Saved to ' }); await status.waitFor();
    csv = await readFile((await status.innerText()).replace(/^Saved to /, ''), 'utf8');
  } else {
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export purchase range', exact: true }).click();
    const download = await pending, path = resolve(project, 'test-results/purchase-range-export.csv');
    await download.saveAs(path); csv = await readFile(path, 'utf8');
  }
  return csvRecords(csv);
}

async function snapshot(page) {
  return page.evaluate(({ prefix, revisionKey, preferenceKey }) => ({
    drafts: Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(prefix)).map(key => [key, localStorage.getItem(key)])),
    revisions: localStorage.getItem(revisionKey), preferences: localStorage.getItem(preferenceKey),
    valuationOpen: !!document.querySelector('.valuation-workspace')
  }), { prefix, revisionKey, preferenceKey });
}

export async function purchaseRangeFlows(page, project, { native = false } = {}) {
  const checks = [], viewport = page.viewportSize(), original = await snapshot(page);
  const originalCompany = JSON.parse(original.preferences).companyId;
  try {
    await openCompany(page, 'Nordea Bank', '159');
    const fixture = hypothetical((await savedDraft(page)).saved.draft);
    await installFixture(page, fixture);
    const card = page.locator('[data-purchase-range="true"]');
    assert.equal(await page.getByLabel('Purchase margin of safety (%)', { exact: true }).inputValue(), '30');
    assert.equal(await page.getByLabel('Purchase reference scenario', { exact: true }).inputValue(), 'mid');
    assert.equal(await page.getByLabel('Purchase price units', { exact: true }).inputValue(), 'equity');
    const originalRows = await verifyPurchases(page, fixture, 30, 1000);
    near(originalRows.mid.value, 1641.6, 'Independent cash PV361.6 plus terminal PV1280');
    near(originalRows.mid.ceiling, 1149.12, 'Mid 30% margin price ceiling');
    near(originalRows.mid.cashOnly, 253.12, 'Cash-only comparison');
    near(await card.getAttribute('data-reference-ceiling'), 1149.12, 'Default selected ceiling');
    assert.equal(await card.getAttribute('data-qualifying-range'), 'true');
    assert.equal(await card.getAttribute('data-purchase-qualifies'), 'true');
    await card.getByRole('img', { name: 'NPV across purchase prices', exact: true }).locator('canvas').waitFor();
    assert.deepEqual((await savedDraft(page)).saved.draft, fixture, 'Derived defaults do not create or alter saved purchase settings');
    checks.push('Derived Mid/30% defaults leave the saved draft untouched and reconcile whole-equity DCF, terminal dependence, cash-only ceiling and candidate NPV independently');

    const rescaled = structuredClone(fixture); rescaled.investment = 9000; rescaled.marketCap = 2000;
    await installFixture(page, rescaled); await verifyPurchases(page, rescaled, 30, 2000);
    near((await purchaseRows(page)).mid.ceiling, 1149.12, 'Investment and current market price do not determine intrinsic ceiling');
    rescaled.marketCap = null; rescaled.priceDate = ''; rescaled.priceSource = '';
    await installFixture(page, rescaled); await verifyPurchases(page, rescaled, 30, null);
    near(await card.getAttribute('data-reference-ceiling'), 1149.12, 'Price target still exists without a saved market quote');
    assert.equal(await card.getAttribute('data-purchase-qualifies'), '');
    checks.push('Whole-equity ceilings are independent of investment and saved market price; complete cash assumptions produce targets even without a dated market quote');

    await installFixture(page, fixture);
    await page.getByLabel('Proposed purchase price', { exact: true }).fill('1500');
    let edited = (await savedDraft(page)).saved.draft;
    assert.equal(edited.purchaseRange.candidateEquity, 1500);
    assert.deepEqual({ ...edited, purchaseRange: undefined }, { ...fixture, purchaseRange: undefined });
    await verifyPurchases(page, edited, 30, 1500);
    assert.equal(await card.getAttribute('data-purchase-qualifies'), 'false');
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('0');
    near(await card.getAttribute('data-reference-ceiling'), 1641.6, 'Zero margin coincides with break-even');
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('100');
    assert.equal(await card.getAttribute('data-qualifying-range'), 'false');
    assert.equal(await card.getAttribute('data-reference-ceiling'), '', '100% margin leaves no qualifying positive price');
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('');
    await verifyPurchases(page, fixture, null, 1500);
    assert.equal(await card.getAttribute('data-qualifying-range'), '');
    await reopen(page, '159');
    assert.equal(await page.getByLabel('Purchase margin of safety (%)', { exact: true }).inputValue(), '');
    assert.equal((await savedDraft(page)).saved.draft.purchaseRange.marginOfSafetyPercent, null);
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('30');
    await page.getByLabel('Proposed purchase price', { exact: true }).fill('');
    await reopen(page, '159');
    assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).inputValue(), '');
    assert.equal((await savedDraft(page)).saved.draft.purchaseRange.candidateEquity, null);
    await page.getByRole('button', { name: 'Use saved purchase price', exact: true }).click();
    assert.equal((await savedDraft(page)).saved.draft.purchaseRange.candidateEquity, undefined);
    await verifyPurchases(page, fixture, 30, 1000);
    checks.push('Candidate prices change only purchase settings; zero/100% margins and deliberately cleared policy or candidate fields remain explicit and persist, with a separate return-to-saved-price action');

    await inputs(page);
    const beforeToggle = await purchaseRows(page);
    await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).check();
    assert.deepEqual(await purchaseRows(page), beforeToggle);
    await page.getByLabel('Mid net equity recovery', { exact: true }).fill('999999');
    await page.getByLabel('Mid recovery payment year', { exact: true }).fill('1');
    await page.getByLabel('Enable crisis scenario', { exact: true }).check();
    assert.deepEqual(await purchaseRows(page), beforeToggle);
    await page.getByLabel('Mid sustainable terminal cash', { exact: true }).fill('600');
    const newTerminal = (await savedDraft(page)).saved.draft;
    await verifyPurchases(page, newTerminal, 30, 1000);
    near((await purchaseRows(page)).mid.value, 1897.6, 'Terminal edit contributes exactly one extra discounted sale');
    checks.push('Terminal inputs update purchase values once; the NPV display toggle, separate liquidation recovery and optional crisis never alter the three purchase ceilings');

    const missing = structuredClone(fixture); missing.scenarios.mid.cashFlows[1] = null;
    await installFixture(page, missing);
    let rows = await purchaseRows(page);
    assert.equal(rows.mid.value, null); near(rows.low.value, 1001.6, 'Available low value survives missing Mid');
    assert.equal(await card.getAttribute('data-qualifying-range'), '');
    assert.equal(await card.getAttribute('data-reference-ceiling'), '');
    const partialExport = await exportPurchase(page, project, native);
    assert.equal(partialExport.length, 3);
    assert.equal(partialExport.find(row => row.scenario === 'mid').equity_value_m, '');
    assert.equal(partialExport.find(row => row.scenario === 'mid').purchase_ceiling_m, '');
    near(partialExport.find(row => row.scenario === 'low').equity_value_m, 1001.6, 'Partial export retains independently available scenarios');
    await page.getByLabel('Purchase reference scenario', { exact: true }).selectOption('low');
    near(await card.getAttribute('data-reference-ceiling'), 701.12, 'Explicit valid reference can be selected');
    const signed = structuredClone(fixture); signed.years = 1;
    for (const [key, value] of [['low', 200], ['mid', -100], ['high', 50]]) Object.assign(signed.scenarios[key], { cashFlows: [value], discountRate: 0, terminalEquity: 0, terminalCash: undefined });
    await installFixture(page, signed); rows = await purchaseRows(page);
    near(rows.low.value, 200, 'Named Low can be the highest value'); near(rows.mid.value, -100, 'Negative present value stays signed'); near(rows.high.value, 50, 'Named High is not relabelled');
    assert.equal(await card.getAttribute('data-qualifying-range'), 'false');
    await page.getByLabel('Purchase reference scenario', { exact: true }).selectOption('low');
    near(await card.getAttribute('data-reference-ceiling'), 140, 'Selected reference follows its actual value');
    assert.equal(await card.getAttribute('data-qualifying-range'), 'true');
    checks.push('Missing reference cash remains unavailable, while negative values and reversed scenario rankings retain their labels and never invent a positive purchase floor');

    await installFixture(page, fixture);
    await page.getByLabel('Purchase price units', { exact: true }).selectOption('share');
    await page.getByLabel('Purchase share count (millions)', { exact: true }).fill('2');
    await page.getByLabel('Purchase share basis date', { exact: true }).fill('2026-09-12');
    await page.getByLabel('Purchase share basis source', { exact: true }).fill('Hypothetical reviewed ownership: two million equal economic shares; SEK basis.');
    await page.getByLabel('Proposed purchase price', { exact: true }).fill('300');
    const perShare = (await savedDraft(page)).saved.draft;
    assert.equal(perShare.purchaseRange.unit, 'share');
    near(perShare.purchaseRange.shareBasis.sharesMillions, 2, 'Saved share count is in millions');
    near(perShare.purchaseRange.candidateEquity, 600, '300 per share converts to 600 million total equity');
    await verifyPurchases(page, fixture, 30, 600);
    assert.match(await card.locator('[data-purchase-scenario="mid"]').innerText(), /574\.56/);
    const exported = await exportPurchase(page, project, native);
    assert.equal(exported.length, 3);
    for (const key of keys) {
      const row = exported.find(row => row.scenario === key), expected = independentlyValue(fixture, key);
      near(row.equity_value_m, expected.value, `${key} CSV total equity value`);
      near(row.purchase_ceiling_m, expected.value * .7, `${key} CSV ceiling`);
      near(row.cash_only_ceiling_m, expected.cashPV * .7, `${key} CSV cash-only ceiling`);
      near(row.terminal_pv_m, expected.terminalPV, `${key} CSV terminal dependence`);
      near(row.candidate_equity_m, 600, `${key} CSV candidate equity`);
      near(row.npv_at_candidate_m, expected.value - 600, `${key} CSV candidate NPV`);
      near(row.price_ceiling_display, expected.value * .7 / 2, `${key} CSV ceiling per share`);
      near(row.npv_at_candidate_display, (expected.value - 600) / 2, `${key} CSV NPV per share`);
      assert.equal(row.display_units, 'SEK / share'); near(row.shares_m, 2, `${key} CSV denominator`);
    }
    await reopen(page, '159');
    assert.deepEqual((await savedDraft(page)).saved.draft, perShare);
    assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).inputValue(), '300');
    checks.push('A dated sourced two-million-share basis converts both purchase price and NPV consistently, persists the canonical total-equity candidate, and exports independently reconciled whole-equity values');

    const mismatched = structuredClone(perShare); mismatched.purchaseRange.shareBasis.currency = 'EUR';
    await installFixture(page, mismatched);
    assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).isDisabled(), true);
    await page.getByLabel('Purchase price units', { exact: true }).selectOption('equity');
    assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).isDisabled(), false);
    await verifyPurchases(page, fixture, 30, 600);
    const tinyShares = structuredClone(perShare); tinyShares.purchaseRange.shareBasis.sharesMillions = 1e-308;
    await installFixture(page, tinyShares);
    assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).isDisabled(), true);
    assert.equal(await card.getByRole('img', { name: 'NPV across purchase prices', exact: true }).count(), 0);
    assert.match(await card.innerText(), /supported.*range/i, 'Nonfinite share conversions fail closed with an explicit numerical-range message');
    await installFixture(page, perShare);
    const title = 'TEST FIXTURE — retained purchase policy and share basis';
    await inputs(page); await page.getByLabel('Valuation study title', { exact: true }).fill(title);
    await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('45');
    await page.locator('.valuation-saved button').filter({ hasText: title }).first().click();
    await reopen(page, '159');
    assert.deepEqual((await savedDraft(page)).saved.draft.purchaseRange, perShare.purchaseRange);
    checks.push('An incompatible currency disables per-share entry while total-equity analysis remains usable; saved revisions restore the complete purchase policy and share basis');

    for (const size of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(size);
      await card.evaluate(el => el.scrollIntoView({ block: 'start', inline: 'nearest' }));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const chart = card.getByRole('img', { name: 'NPV across purchase prices', exact: true });
      await chart.locator('canvas').waitFor();
      assert.ok(await chart.locator('canvas').first().evaluate(el => el.width > 100 && el.height > 100));
      // The desktop workspace has its own scroll container. Capture the full
      // card in a tall viewport at the tested width; locator screenshots can
      // otherwise resize the viewport and reset that container's scroll offset.
      if (size.width > 900) {
        const height = Math.ceil((await card.boundingBox()).height) + 200;
        if (height > size.height) await page.setViewportSize({ width: size.width, height });
        await card.evaluate(el => el.scrollIntoView({ block: 'start', inline: 'nearest' }));
        const visible = await card.boundingBox();
        assert.ok(visible.y >= 0 && visible.y + visible.height <= page.viewportSize().height + 1, `The purchase card fits the screenshot viewport: ${JSON.stringify(visible)}`);
      }
      await card.screenshot({ path: resolve(project, `test-results/purchase-range-${native ? 'native' : 'browser'}-${size.width}.png`) });
    }
    checks.push('The purchase-price chart and controls render at desktop and mobile widths without horizontal page overflow');

    if (viewport) await page.setViewportSize(viewport);
    await openCompany(page, 'SCA', '197');
    await page.getByRole('button', { name: 'Start from researched assumptions', exact: true }).click();
    const researched = (await savedDraft(page, '197')).saved.draft;
    assert.equal(researched.researchOrigin.id, 'sca-2026-09-12-v1');
    await page.getByLabel('Purchase price units', { exact: true }).selectOption('share');
    await page.getByRole('button', { name: 'Use saved share reference', exact: true }).click();
    const reviewedBasis = (await savedDraft(page, '197')).saved.draft.purchaseRange.shareBasis;
    near(reviewedBasis.sharesMillions, 702.342489, 'Reviewed SCA outstanding A+B shares, excluding treasury');
    assert.equal(reviewedBasis.date, '2026-06-30'); assert.equal(reviewedBasis.currency, 'SEK');
    assert.match(reviewedBasis.source, /half-year report.*page 17/i);
    assert.match(reviewedBasis.source, /Equal economic rights and unchanged shares are assumptions/);
    near(await page.getByLabel('Proposed purchase price', { exact: true }).inputValue(), 110.3, 'Reviewed per-share price reconciles its exact saved equity denominator');
    await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('35');
    await page.getByLabel('Purchase reference scenario', { exact: true }).selectOption('low');
    await page.getByLabel('Proposed purchase price', { exact: true }).fill('100');
    const purchasePolicy = (await savedDraft(page, '197')).saved.draft.purchaseRange;
    near(purchasePolicy.candidateEquity, 70234.2489, 'Reviewed shares convert a manually proposed share price to total equity');
    assert.deepEqual((await savedDraft(page, '197')).saved.draft.scenarios, researched.scenarios, 'Purchase settings do not edit the researched cash, terminal or recovery model');
    await reopen(page, '197');
    assert.deepEqual((await savedDraft(page, '197')).saved.draft.purchaseRange, purchasePolicy);
    const history = page.locator('.valuation-history-settings');
    if (!await history.evaluate(el => el.open)) await history.locator('summary').click();
    await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
    assert.deepEqual((await savedDraft(page, '197')).saved.draft.purchaseRange, purchasePolicy, 'Applying a historical baseline preserves the authored purchase policy and denominator');
    await page.getByRole('button', { name: 'Start from researched assumptions', exact: true }).click();
    assert.deepEqual((await savedDraft(page, '197')).saved.draft.purchaseRange, purchasePolicy, 'A deliberate researched reset preserves the separate purchase policy');
    assert.deepEqual((await savedDraft(page, '197')).saved.draft.scenarios, researched.scenarios);
    checks.push('The reviewed SCA ownership denominator reconciles the saved share price; purchase settings persist independently through reload, historical baseline application and researched resets');
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/purchase-range-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    if (await page.locator('.valuation-workspace').count()) await page.getByRole('button', { name: 'Company profile', exact: true }).click();
    await page.evaluate(({ original, prefix, revisionKey, preferenceKey }) => {
      for (const key of Object.keys(localStorage).filter(key => key.startsWith(prefix))) if (!(key in original.drafts)) localStorage.removeItem(key);
      for (const [key, raw] of Object.entries(original.drafts)) localStorage.setItem(key, raw);
      for (const [key, raw] of [[revisionKey, original.revisions], [preferenceKey, original.preferences]]) {
        if (raw === null) localStorage.removeItem(key); else localStorage.setItem(key, raw);
      }
    }, { original, prefix, revisionKey, preferenceKey });
    await page.reload(); await page.locator(`[data-company="${originalCompany}"][data-business-ready="true"]`).waitFor();
    if (original.valuationOpen) {
      await page.getByLabel('Company valuation', { exact: true }).click();
      await page.locator(`[data-valuation-company="${originalCompany}"]`).waitFor();
      await page.evaluate(({ drafts, prefix, id }) => {
        for (const [key, raw] of Object.entries(drafts)) if (key.startsWith(`${prefix}${id}:`)) localStorage.setItem(key, raw);
      }, { drafts: original.drafts, prefix, id: originalCompany });
    }
    if (viewport) await page.setViewportSize(viewport);
    assert.deepEqual(await snapshot(page), original, 'All prior drafts, revisions, company selection and preferences are restored exactly');
  }
  checks.push('All temporary purchase fixtures and revisions are removed while prior forecasts, terminal models, crisis settings, studies and company selection are restored byte for byte');
  return { checks };
}

export async function seedPurchaseRestart(page) {
  await openCompany(page, 'Fairfax Financial Holdings Ltd', '14473');
  const fixture = hypothetical((await savedDraft(page, '14473')).saved.draft);
  fixture.title = 'TEST FIXTURE — purchase policy survives native restart';
  await installFixture(page, fixture, '14473');
  await page.getByLabel('Purchase margin of safety (%)', { exact: true }).fill('40');
  await page.getByLabel('Purchase reference scenario', { exact: true }).selectOption('high');
  await page.getByLabel('Purchase price units', { exact: true }).selectOption('equity');
  await page.getByLabel('Proposed purchase price', { exact: true }).fill('2468');
  await page.waitForFunction(({ prefix }) => {
    const key = Object.keys(localStorage).find(key => key.startsWith(`${prefix}14473:`));
    return key && JSON.parse(localStorage.getItem(key)).draft.purchaseRange?.candidateEquity === 2468;
  }, { prefix });
  const { key, saved } = await savedDraft(page, '14473');
  const expected = { key, draft: saved.draft, basis: Object.fromEntries(['company', 'release', 'financial', 'taxonomy'].map(name => [name, saved[name]])) };
  await openCompany(page, 'Holmen', '102'); await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
  return expected;
}

export async function assertPurchaseRestart(page, expected) {
  await openCompany(page, 'Fairfax Financial Holdings Ltd', '14473');
  const { key, saved } = await savedDraft(page, '14473');
  assert.equal(key, expected.key);
  assert.deepEqual(Object.fromEntries(['company', 'release', 'financial', 'taxonomy'].map(name => [name, saved[name]])), expected.basis);
  assert.equal(JSON.stringify(saved.draft), JSON.stringify(expected.draft), 'Native restart retains exact purchase settings and underlying forecast');
  assert.equal(await page.getByLabel('Purchase margin of safety (%)', { exact: true }).inputValue(), '40');
  assert.equal(await page.getByLabel('Purchase reference scenario', { exact: true }).inputValue(), 'high');
  assert.equal(await page.getByLabel('Proposed purchase price', { exact: true }).inputValue(), '2468');
  await verifyPurchases(page, expected.draft, 40, 2468);
  await openCompany(page, 'Holmen', '102'); await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
}

async function standalone() {
  const { chromium } = await import('playwright');
  const project = resolve(dirname(fileURLToPath(import.meta.url)), '..'), base = process.env.ATLAS_URL ?? 'http://127.0.0.1:1420';
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const external = [], errors = [], context = await browser.newContext({ viewport: { width: 1500, height: 960 }, acceptDownloads: true });
  context.setDefaultTimeout(30000);
  const csp = JSON.parse(await readFile(resolve(project, 'src-tauri/tauri.conf.json'), 'utf8')).app.security.csp;
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (url === `${base}/`) { const response = await route.fetch(); return route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': csp } }); }
    if (url.startsWith(base) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
    external.push(url); return route.abort();
  });
  const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base); await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    await page.getByLabel('Observatory', { exact: true }).selectOption('companies'); await page.locator('[data-business-ready="true"]').waitFor();
    const result = await purchaseRangeFlows(page, project); assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await writeFile(resolve(project, 'test-results/purchase-range-browser-report.json'), JSON.stringify({ status: 'passed', ...result, external, errors }, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await standalone();
