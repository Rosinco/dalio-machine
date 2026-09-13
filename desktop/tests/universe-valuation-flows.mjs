import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const method = 'weighted-cash-starter-v2';
const savedTitle = 'TEST FIXTURE — Stora weighted history';
const legacyTitle = 'LEGACY V1 — Stora preserved assumptions';
const savedWeights = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
const rangeLabel = 'First-year range (%)';
const wideningLabel = 'Annual widening (percentage points)';
const cashChartLabel = 'Cash flow over time: history and forecast scenarios';

const near = (actual, expected) => assert.ok(Math.abs(Number(actual) - expected) < 0.000001, `${actual} must equal ${expected} within display input precision`);

async function assertCashChart(page, historyCount, forecastCount) {
  const chart = page.locator('.cash-flow-forecast');
  await chart.waitFor();
  assert.equal(await chart.getAttribute('data-history-count'), String(historyCount));
  assert.equal(await chart.getAttribute('data-forecast-count'), String(forecastCount));
  assert.equal(await chart.getByRole('heading', { name: 'Cash flow over time', exact: true }).count(), 1);
  if (historyCount || forecastCount) {
    const canvas = chart.getByRole('img', { name: cashChartLabel, exact: true }).locator('canvas').first();
    await canvas.waitFor();
    assert.ok(await canvas.evaluate(element => element.width > 100 && element.height > 100));
  } else assert.equal(await chart.locator('canvas').count(), 0);
}

// Read exported records independently of optional column order and quoted commas.
function csvRecords(csv) {
  const rows = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < csv.length; i++) {
    const character = csv[i];
    if (character === '"') {
      if (quoted && csv[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && character === ',') { row.push(cell); cell = ''; }
    else if (!quoted && character === '\r' && csv[i + 1] === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; i++; }
    else cell += character;
  }
  row.push(cell); rows.push(row);
  const [header, ...records] = rows;
  return records.map(values => Object.fromEntries(header.map((column, index) => [column, values[index]])));
}

async function openHistorySettings(page) {
  const settings = page.locator('.valuation-history-settings');
  if (await settings.getAttribute('open') === null) await settings.getByText('Adjust history weights & range', { exact: true }).click();
}

async function openCompany(page, name, id) {
  await page.getByLabel('Search companies or countries').fill(name);
  await page.locator(`[data-search-listing="${id}"]`).click();
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function exportCalculations(page, project, native) {
  if (native) {
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const status = page.getByRole('status').filter({ hasText: 'Saved to ' });
    await status.waitFor();
    return readFile((await status.innerText()).replace(/^Saved to /, ''), 'utf8');
  }
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
  const download = await downloadPromise, path = resolve(project, 'test-results/universe-valuation-export.csv');
  await download.saveAs(path);
  return readFile(path, 'utf8');
}

async function restoreHolmenProfile(page) {
  await openCompany(page, 'Holmen', '102');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.equal(await page.locator('[data-scenario="mid"] [data-result="value"]').innerText(), '1,228.91 SEK');
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
}

async function assertPreservedStarter(page) {
  await page.locator(`[data-valuation-company="696"][data-valuation-starter="${method}"][data-valuation-ready="false"]`).waitFor();
  assert.equal(await page.getByLabel('Valuation study title').inputValue(), savedTitle);
  assert.equal(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), '');
  assert.equal(await page.getByLabel('High year 1 cash payment', { exact: true }).inputValue(), '700');
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  await assertCashChart(page, 10, 9);
  assert.match(await page.locator('.valuation-saved').innerText(), /TEST FIXTURE — Stora weighted history/);
  assert.ok((await page.locator('.valuation-saved').innerText()).includes(legacyTitle));
  await openHistorySettings(page);
  assert.equal(await page.getByLabel('Historical years', { exact: true }).inputValue(), '10');
  assert.equal(await page.getByLabel('Projection', { exact: true }).locator('option:checked').innerText(), 'Flat weighted average');
  for (const [index, weight] of savedWeights.entries()) {
    assert.equal(await page.getByLabel(`Year ${index + 1} weight`, { exact: true }).inputValue(), String(weight));
  }
  assert.equal(await page.getByLabel(rangeLabel, { exact: true }).inputValue(), '20');
  assert.equal(await page.getByLabel(wideningLabel, { exact: true }).inputValue(), '10');
}

async function preserveLegacyStarter(page) {
  // Emulate a 0.12 draft using the old persisted origin schema. All company/data
  // versions remain those of this isolated test profile; no user profile is read.
  await page.evaluate(({ legacyTitle }) => {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('macro-atlas-valuation-draft-v1:696:'));
    const [key, stored] = keys.map(key => [key, JSON.parse(localStorage.getItem(key))]).sort((a, b) => a[1].created.localeCompare(b[1].created)).at(-1);
    const d = stored.draft;
    d.title = legacyTitle;
    d.investment = 1000;
    delete d.researchAutofillDisabled;
    d.starterOrigin = { id: 'weighted-cash-starter-v1', asOf: d.valuationDate, weights: [30, 25, 20, 15, 10], spreadPercent: 20 };
    for (const [scenario, payment] of [['low', 226.2], ['mid', 282.75], ['high', 339.3]]) {
      d.scenarios[scenario] = { cashFlows: Array(10).fill(payment), discountRate: 10, terminalEquity: payment / .1, recoveryEquity: null, recoveryYear: null, rationale: 'Preserved version-one flat weighted cash assumptions.' };
    }
    localStorage.setItem(key, JSON.stringify(stored));
  }, { legacyTitle });
  await page.reload();
  await page.locator('[data-company="696"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('[data-valuation-starter="weighted-cash-starter-v1"][data-valuation-ready="true"]').waitFor();
  assert.equal(await page.locator('[data-scenario="mid"] [data-result="value"]').innerText(), '338.73 EUR');
  await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Study revision saved' }).waitFor();
  await page.reload();
  await page.locator('[data-company="696"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('[data-valuation-starter="weighted-cash-starter-v1"][data-valuation-ready="true"]').waitFor();
  assert.ok((await page.locator('.valuation-saved').innerText()).includes(legacyTitle));
  await openHistorySettings(page);
  assert.equal(await page.getByLabel('Historical years', { exact: true }).inputValue(), '5');
  assert.equal(await page.getByLabel('Projection', { exact: true }).locator('option:checked').innerText(), 'Flat weighted average');
  assert.equal(await page.getByLabel(rangeLabel, { exact: true }).inputValue(), '20');
  assert.equal(await page.getByLabel(wideningLabel, { exact: true }).inputValue(), '0');
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  for (const year of [1, 10]) assert.equal(await page.getByLabel(`Mid year ${year} cash payment`, { exact: true }).inputValue(), '282.75');
}

async function upgradeUntouchedLegacy(page, project) {
  const fixture = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-weighted-cash-v1.json'), 'utf8'));
  // The curated profile uses "Stora Enso" while the directory calls the same
  // listing "Stora Enso R". Apply only the legacy default title template.
  fixture.title = `${await page.locator('.valuation-heading h1').innerText()} — weighted cash starter`;
  fixture.investment = 2500;
  await page.evaluate(fixture => {
    const keys = Object.keys(localStorage).filter(key => key.startsWith('macro-atlas-valuation-draft-v1:696:'));
    const [key, stored] = keys.map(key => [key, JSON.parse(localStorage.getItem(key))]).sort((a, b) => a[1].created.localeCompare(b[1].created)).at(-1);
    stored.draft = fixture;
    localStorage.setItem(key, JSON.stringify(stored));
  }, fixture);
  await page.reload();
  await page.locator('[data-company="696"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-starter="${method}"][data-valuation-ready="true"]`).waitFor();
  assert.ok((await page.locator('.valuation-saved').innerText()).includes(fixture.title));
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  assert.equal(await page.getByLabel('Valuation investment amount', { exact: true }).inputValue(), '2500');
  near(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), 147);
  await page.locator('.valuation-saved button').filter({ hasText: fixture.title }).first().click();
  await page.locator('[data-valuation-starter="weighted-cash-starter-v1"][data-valuation-ready="true"]').waitFor();
  await page.reload();
  await page.locator('[data-company="696"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('[data-valuation-starter="weighted-cash-starter-v1"][data-valuation-ready="true"]').waitFor();
  assert.equal(await page.locator('[data-scenario="mid"] [data-result="value"]').innerText(), '846.83 EUR');
}

// Optional additional full-process check for the native runner.
export async function restoreUniverseValuation(page) {
  await openCompany(page, 'Stora Enso R', '696');
  await assertPreservedStarter(page);
  await restoreHolmenProfile(page);
}

export async function universeValuationFlows(page, project, { native = false } = {}) {
  const checks = [];
  await openCompany(page, 'Stora Enso R', '696');
  await page.locator(`[data-valuation-kind="starter"][data-valuation-starter="${method}"][data-valuation-ready="true"]`).waitFor();
  assert.equal(await page.locator('.valuation-workspace').getAttribute('data-valuation-study'), '');
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).count(), 0);
  assert.match(await page.locator('.valuation-starter-banner').innerText(), /historical|cash.flow/i);
  assert.match(await page.locator('.valuation-price-summary').innerText(), /EUR[\s\S]*2026-02-04/);
  for (const [key, value] of [['low', '-100.4 EUR'], ['mid', '-40.7 EUR'], ['high', '18.99 EUR']]) {
    assert.equal(await page.locator(`[data-scenario="${key}"] [data-result="value"]`).innerText(), value);
  }
  for (const label of ['Discounted cash flow: low, mid and high scenarios', 'Cumulative NPV: low, mid and high scenarios']) {
    await page.getByRole('img', { name: label, exact: true }).locator('canvas').waitFor();
  }
  assert.equal(await page.locator('[data-recovery]').count(), 0);
  await assertCashChart(page, 5, 10);
  await page.screenshot({ path: resolve(project, 'test-results/stora-automatic-history-valuation.png') });
  const cashChart = page.locator('.cash-flow-forecast');
  await cashChart.getByRole('heading', { name: 'Cash flow over time', exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: resolve(project, 'test-results/stora-history-forecast-fan.png') });
  await cashChart.getByRole('img', { name: cashChartLabel, exact: true }).screenshot({ path: resolve(project, 'test-results/stora-cash-flow-fan.png') });
  await cashChart.getByText('Inspect annual cash-flow ranges', { exact: true }).click();
  assert.match(await cashChart.locator('[data-year="1"]').innerText(), /132\.3[\s\S]*147[\s\S]*161\.7/);
  assert.match(await cashChart.locator('[data-year="2"]').innerText(), /74\.16[\s\S]*92\.7[\s\S]*111\.24/);
  assert.match(await cashChart.locator('[data-year="3"]').innerText(), /26\.88[\s\S]*38\.4[\s\S]*49\.92/);
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  await page.getByLabel('Equity market value', { exact: true }).fill('');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  await assertCashChart(page, 5, 10);
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  assert.equal(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), '147');
  await page.getByLabel('Equity market value', { exact: true }).fill('8347.331');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  await page.getByRole('button', { name: 'Hide input forms', exact: true }).click();
  await page.locator('.valuation-charts').scrollIntoViewIfNeeded();
  await page.screenshot({ path: resolve(project, 'test-results/stora-history-charts.png') });
  await page.getByRole('button', { name: 'Inspect starter assumptions', exact: true }).click();
  const evidence = page.locator('.valuation-starter-evidence');
  await evidence.waitFor();
  assert.match(await evidence.innerText(), /282\.75 EUR m/);
  assert.match(await evidence.innerText(), /lease principal[\s\S]*acquisitions[\s\S]*financing/i);
  assert.match(await evidence.innerText(), /sensitivity[\s\S]*(?:statistical confidence|confidence interval)/);
  const annual = evidence.locator('table').first();
  assert.equal(await annual.locator('tbody tr').count(), 5);
  assert.match(await annual.getByRole('row', { name: /^FY 2023 / }).innerText(), /-561/);
  await evidence.getByText('Saved files and provenance', { exact: true }).click();
  assert.match(await evidence.locator('.valuation-starter-provenance').innerText(), /all_yearly_reports\.parquet[\s\S]*SHA-256 [a-f0-9]{64}/);
  await page.setViewportSize({ width: 1100, height: 760 });
  await evidence.getByRole('heading', { name: 'How the standard historical baseline was built', exact: true }).scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
  await page.screenshot({ path: resolve(project, 'test-results/stora-history-evidence-compact.png') });
  await page.setViewportSize({ width: 1500, height: 960 });
  checks.push('Stora plots five actual cash-flow years and the independent weighted-trend forecast with widening 10/20/30 percent ranges, even when price is cleared');

  await upgradeUntouchedLegacy(page, project);
  checks.push('Untouched version-one defaults upgrade with an exact saved backup and preserved investment; explicitly restoring that backup disables repeat migration');
  await preserveLegacyStarter(page);
  checks.push('Version-one flat starter drafts and saved revisions survive reload with their original constant range and no automatic trend migration');

  const revisions = await page.locator('.valuation-saved > div').count();
  await openHistorySettings(page);
  await page.getByLabel('Year 1 weight', { exact: true }).fill('0');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await page.getByRole('alert').filter({ hasText: 'add to 100%' }).waitFor();
  assert.equal(await page.locator('.valuation-saved > div').count(), revisions);
  await page.getByLabel('Historical years', { exact: true }).selectOption('10');
  await page.getByLabel('Projection', { exact: true }).selectOption({ label: 'Weighted historical trend' });
  await page.getByLabel(rangeLabel, { exact: true }).fill('10');
  await page.getByLabel(wideningLabel, { exact: true }).fill('10');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await page.locator(`[data-valuation-starter="${method}"][data-valuation-ready="true"]`).waitFor();
  assert.equal(await page.locator('.valuation-saved > div').count(), revisions + 1);
  await assertCashChart(page, 10, 10);
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  near(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), 122.645771144279);
  near(await page.getByLabel('Mid year 10 cash payment', { exact: true }).inputValue(), -468.348530076888);
  checks.push('Selecting ten historical years fits the independent ten-year weighted line and retains the preceding version-one study');

  await page.getByLabel('Projection', { exact: true }).selectOption({ label: 'Flat weighted average' });
  for (const [index, weight] of savedWeights.entries()) await page.getByLabel(`Year ${index + 1} weight`, { exact: true }).fill(String(weight));
  await page.getByLabel(rangeLabel, { exact: true }).fill('20');
  await page.getByLabel(wideningLabel, { exact: true }).fill('10');
  await page.getByRole('button', { name: 'Apply history assumptions', exact: true }).click();
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.equal(await page.locator('.valuation-saved > div').count(), revisions + 2);
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  assert.equal(await page.getByLabel('Valuation currency', { exact: true }).inputValue(), 'EUR');
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '8347.331');
  // Reversed ten-year weights produce 595.74 EUR m. A 20% first-year
  // range widening by 10 percentage points reaches 110% in year ten.
  for (const [year, amounts] of [[1, [476.592, 595.74, 714.888]], [3, [357.444, 595.74, 834.036]], [10, [-59.574, 595.74, 1251.054]]]) {
    for (const [index, name] of ['Low', 'Mid', 'High'].entries()) near(await page.getByLabel(`${name} year ${year} cash payment`, { exact: true }).inputValue(), amounts[index]);
  }
  const csv = await exportCalculations(page, project, native);
  assert.equal(csv.split('\r\n').length, 34);
  for (const row of csvRecords(csv)) {
    assert.equal(row.company_id, '696');
    assert.equal(row.starter_method, method);
    assert.equal(row.starter_weights_newest_first, savedWeights.join('/'));
    assert.equal(row.starter_spread_pct, '20');
    assert.equal(row.starter_history_years, '10');
    assert.equal(row.starter_projection, 'flat');
    assert.equal(row.starter_spread_step_pct, '10');
    assert.equal(row.edited_from_starter, 'false');
  }
  checks.push('Editable ten-year weights, flat alternative and widening beyond 100 percent regenerate exact signed ranges while preserving drafts and CSV provenance');

  await page.getByLabel('Valuation study title').fill(savedTitle);
  await page.getByLabel('High year 1 cash payment', { exact: true }).fill('700');
  await page.getByLabel('Mid year 1 cash payment', { exact: true }).fill('');
  await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Study revision saved' }).waitFor();

  await openCompany(page, 'Atrium Ljungberg', '20');
  await page.locator('[data-valuation-kind="starter"][data-valuation-ready="true"]').waitFor();
  for (const [key, value] of [['low', '79.23 SEK'], ['mid', '669.58 SEK'], ['high', '1,259.93 SEK']]) {
    assert.equal(await page.locator(`[data-scenario="${key}"] [data-result="value"]`).innerText(), value);
  }
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  for (const [name, first, second] of [['Low', -55, 176.08], ['Mid', -50, 220.1], ['High', -45, 264.12]]) {
    near(await page.getByLabel(`${name} year 1 cash payment`, { exact: true }).inputValue(), first);
    near(await page.getByLabel(`${name} year 2 cash payment`, { exact: true }).inputValue(), second);
  }
  await assertCashChart(page, 5, 10);
  assert.equal(await page.locator('.valuation-charts').count(), 1);
  assert.equal(await page.locator('[data-recovery]').count(), 0);
  await page.getByRole('img', { name: cashChartLabel, exact: true }).screenshot({ path: resolve(project, 'test-results/atrium-cash-flow-fan.png') });
  await page.screenshot({ path: resolve(project, 'test-results/negative-cash-history-valuation.png') });
  checks.push('A fitted trend crossing from negative to positive cash retains correctly ordered signed scenario ranges and a separate missing recovery case');

  await openCompany(page, 'ABB', '3');
  await page.locator('[data-valuation-kind="starter"][data-valuation-ready="true"]').waitFor();
  await page.getByRole('button', { name: 'Inspect starter assumptions', exact: true }).click();
  const quotedEvidence = page.locator('.valuation-starter-evidence');
  assert.match(await quotedEvidence.innerText(), /33,439\.25 SEK m/);
  assert.match(await quotedEvidence.innerText(), /archived quote currency/);
  assert.match(await quotedEvidence.innerText(), /not constant-FX/);
  assert.match(await quotedEvidence.locator('table').first().getByRole('row', { name: /^FY 2025 / }).innerText(), /28,418\.54 SEK[\s\S]*9\.2268[\s\S]*3,080 USD[\s\S]*28,418\.54/);
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  assert.equal(await page.getByLabel('Valuation currency', { exact: true }).inputValue(), 'SEK');
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '1393136.6');
  near(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), 29866.834185714);
  await assertCashChart(page, 5, 10);
  await page.getByRole('img', { name: cashChartLabel, exact: true }).screenshot({ path: resolve(project, 'test-results/abb-cash-flow-fan.png') });
  checks.push('ABB uses documented archived SEK quote amounts for its USD-reporting history, exposes report conversion and avoids an invented current FX rate');

  await openCompany(page, 'Logistea A', '167');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '');
  assert.equal(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), '');
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  assert.equal(await page.locator('.valuation-chart-placeholders').count(), 1);
  await assertCashChart(page, 0, 0);
  assert.equal(await page.getByRole('button', { name: 'Export calculations', exact: true }).isDisabled(), true);
  assert.match(await page.locator('.valuation-starter-banner').innerText(), /missing|unavailable|no usable|no saved/i);
  checks.push('Listings without reports retain unavailable inputs and chart placeholders without borrowing a previous company forecast');

  await openCompany(page, 'Stora Enso R', '696');
  await assertPreservedStarter(page);
  await page.reload();
  await page.locator('[data-company="696"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await assertPreservedStarter(page);
  await restoreHolmenProfile(page);
  checks.push('Ten-year settings, projection choice, widening, edited payments and deliberate gaps survive switching and reload, preserving reviewed and legacy studies');
  return { checks };
}
