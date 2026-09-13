import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const studyId = 'sca-2026-09-12-v1';
const savedTitle = 'TEST FIXTURE — SCA preserved edits';
const savedNote = 'SCA test note — svensk skog; review verified export exposure.';

async function openCompany(page, name, id) {
  await page.getByLabel('Search companies or countries').fill(name);
  await page.getByLabel('Search companies or countries').press('Enter');
  await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator(`[data-valuation-company="${id}"]`).waitFor();
}

async function exportCalculations(page, project, native, suffix) {
  if (native) {
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const status = page.getByRole('status').filter({ hasText: 'Saved to ' });
    await status.waitFor();
    return readFile((await status.innerText()).replace(/^Saved to /, ''), 'utf8');
  }
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
  const download = await downloadPromise, path = resolve(project, `test-results/sca-valuation-${suffix}.csv`);
  await download.saveAs(path);
  return readFile(path, 'utf8');
}

function assertExport(csv, edited) {
  const [header, ...rows] = csv.split('\r\n');
  assert.match(header, /"starting_valuation_study","starting_study_date","edited_from_starting_study"(?:,|$)/);
  assert.equal(rows.length, 63, 'Twenty forecast years plus year zero for all three scenarios');
  for (const row of rows) {
    assert.ok(row.startsWith('"197",'), 'SCA export must retain its own company ID');
    assert.match(row, /,"2026-09-12","2026-08-07","SEK",77468\.3765367,/);
    assert.match(row, new RegExp(`,"${studyId}","2026-09-12","${edited}"(?:,|$)`), 'Every annual row must retain the SCA study version and edit state');
    assert.doesNotMatch(row, /Holmen/i);
  }
}

async function assertPreservedScaDraft(page) {
  await page.locator(`[data-valuation-company="197"][data-valuation-study="${studyId}"][data-valuation-ready="false"]`).waitFor();
  assert.equal(await page.getByLabel('Valuation study title').inputValue(), savedTitle);
  assert.equal(await page.getByLabel('Valuation investment amount').inputValue(), '2500');
  assert.equal(await page.getByLabel('High year 1 cash payment', { exact: true }).inputValue(), '4321');
  assert.equal(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), '');
  assert.equal(await page.getByLabel('Low net equity recovery', { exact: true }).inputValue(), '');
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  assert.equal(await page.getByRole('button', { name: 'Export calculations', exact: true }).isDisabled(), true);
  assert.match(await page.locator('.valuation-research-banner').innerText(), /edited SCA/i);
  assert.match(await page.locator('.valuation-saved').innerText(), /TEST FIXTURE — SCA preserved edits/);
  assert.doesNotMatch(await page.locator('.valuation-saved').innerText(), /Swedish scenario study|PREVIOUS USER DRAFT/);
  await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).click();
  assert.equal(await page.getByLabel('Available surplus cash', { exact: true }).inputValue(), '321');
  assert.equal(await page.getByLabel('Macro and branch evidence → company exposure → forecast assumption').inputValue(), savedNote);
  assert.doesNotMatch(await page.locator('.valuation-workspace').innerText(), /Holmen/i);
  await page.getByRole('tab', { name: 'Value, price & payback', exact: true }).click();
  await page.getByRole('img', { name: 'Cash flow over time: history and forecast scenarios', exact: true }).locator('canvas').waitFor();
  assert.equal(await page.locator('.cash-flow-forecast').getAttribute('data-forecast-count'), '19');
}

async function restoreHolmenProfile(page) {
  await openCompany(page, 'Holmen', '102');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.equal(await page.locator('[data-scenario="mid"] [data-result="value"]').innerText(), '1,228.91 SEK');
  assert.match(await page.locator('.valuation-saved').innerText(), /TEST FIXTURE — Swedish scenario study/);
  assert.doesNotMatch(await page.locator('.valuation-saved').innerText(), /SCA preserved edits/);
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"]').waitFor();
}

// The native runner calls this after terminating and relaunching the actual app
// with the same isolated profile, beyond the reload checks in the shared flow.
export async function restoreScaValuation(page) {
  await openCompany(page, 'SCA', '197');
  await assertPreservedScaDraft(page);
  await restoreHolmenProfile(page);
}

export async function scaValuationFlows(page, project, { native = false } = {}) {
  const checks = [];
  await openCompany(page, 'SCA', '197');
  await page.locator(`[data-valuation-ready="true"][data-valuation-study="${studyId}"]`).waitFor();
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).count(), 0);
  assert.match(await page.locator('.valuation-research-banner').innerText(), /SCA.*scenarios are ready/);
  assert.match(await page.locator('.valuation-price-summary').innerText(), /2026-08-07/);
  assert.match(await page.locator('.valuation-price-summary').innerText(), /2026-09-12/);
  assert.match(await page.locator('.valuation-price-summary').innerText(), /702\.342489/);
  assert.doesNotMatch(await page.locator('.valuation-workspace').innerText(), /Holmen/i);
  await page.getByRole('img', { name: 'Cash flow over time: history and forecast scenarios', exact: true }).locator('canvas').waitFor();
  assert.equal(await page.locator('.cash-flow-forecast').getAttribute('data-forecast-count'), '20');
  for (const [key, value, recovery, year] of [['low', '84.83 SEK', 9828, 5], ['mid', '210.28 SEK', 39435.7, 3], ['high', '427.77 SEK', 68351.45, 2]]) {
    const result = page.locator(`[data-scenario="${key}"]`), name = key[0].toUpperCase() + key.slice(1);
    assert.equal(await result.locator('[data-result="value"]').innerText(), value);
    assert.equal(await result.locator('[data-result="payback"]').innerText(), 'Not reached in 20 years');
    assert.equal(await result.locator('[data-result="discounted-payback"]').innerText(), 'Not reached in 20 years');
    assert.ok(Math.abs(Number(await page.getByLabel(`${name} net equity recovery`, { exact: true }).inputValue()) - recovery) < 0.000001);
    assert.equal(await page.getByLabel(`${name} recovery payment year`, { exact: true }).inputValue(), String(year));
  }
  for (const label of ['Discounted cash flow: low, mid and high scenarios', 'Cumulative NPV: low, mid and high scenarios']) {
    const chart = page.getByRole('img', { name: label, exact: true });
    await chart.locator('canvas').waitFor();
    assert.ok(await chart.locator('canvas').first().evaluate(el => el.width > 100 && el.height > 100));
  }
  await page.screenshot({ path: resolve(project, 'test-results/sca-automatic-valuation.png') });
  checks.push('SCA opens directly with its own reviewed study, three complete scenarios and DCF/NPV charts');

  await page.getByRole('button', { name: 'Inspect sources and calculations', exact: true }).click();
  await page.locator(`[data-research-study="${studyId}"]`).waitFor();
  const evidence = page.locator('.valuation-research-evidence');
  assert.doesNotMatch(await evidence.innerText(), /Holmen/i);
  assert.match(await evidence.innerText(), /SCA/);
  assert.match(await evidence.innerText(), /normalized reference/);
  assert.match(await evidence.innerText(), /902 SEK m/);
  assert.equal(await page.getByLabel('Tangible book equity', { exact: true }).inputValue(), '99902');
  assert.equal(await page.getByLabel('Gross corporate debt', { exact: true }).inputValue(), '15536');
  for (const kind of ['source', 'assumption', 'calculation']) {
    assert.ok(await evidence.locator(`[data-evidence-kind="${kind}"]`).count() > 0);
  }
  for (const date of ['2026-07-22', '2026-03-04', '2026-08-10']) {
    assert.ok((await evidence.locator('.valuation-source-list').innerText()).includes(date));
  }
  assert.ok(await evidence.locator('.valuation-source-list a[href*="sca.com/"]').count() > 0);
  await page.getByText('Inspect the separate breakup calculation', { exact: true }).click();
  const recovery = page.locator('.valuation-recovery-bridge');
  assert.equal(await recovery.getByRole('row', { name: /^Net common-equity recovery / }).isVisible(), true);
  assert.match(await recovery.getByRole('row', { name: /^Net common-equity recovery / }).innerText(), /9,828[\s\S]*39,435\.7[\s\S]*68,351\.45/);
  assert.match(await recovery.getByRole('row', { name: /^All prior claims / }).innerText(), /46,490/);
  assert.match(await recovery.innerText(), /Additional costs and taxes/);
  assert.match(await recovery.innerText(), /Additional cash burn/);
  await page.screenshot({ path: resolve(project, 'test-results/sca-research-evidence.png') });
  checks.push('SCA source dates, cash bridge, capital and separate recovery calculations are inspectable without Holmen text');

  await page.getByRole('tab', { name: 'Value, price & payback', exact: true }).click();
  assertExport(await exportCalculations(page, project, native, 'researched'), false);
  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  await page.getByLabel('Valuation study title').fill(savedTitle);
  await page.getByLabel('Valuation investment amount').fill('2500');
  await page.getByLabel('High year 1 cash payment', { exact: true }).fill('4321');
  assertExport(await exportCalculations(page, project, native, 'edited'), true);
  checks.push('SCA CSV export preserves its company/study ID and distinguishes reviewed defaults from user edits');

  await page.getByLabel('Mid year 1 cash payment', { exact: true }).fill('');
  await page.getByLabel('Low net equity recovery', { exact: true }).fill('');
  await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).click();
  await page.getByLabel('Available surplus cash', { exact: true }).fill('321');
  await page.getByLabel('Macro and branch evidence → company exposure → forecast assumption').fill(savedNote);
  await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Study revision saved' }).waitFor();
  await restoreHolmenProfile(page);
  await openCompany(page, 'SCA', '197');
  await assertPreservedScaDraft(page);
  await page.reload();
  await page.locator('[data-company="197"][data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await assertPreservedScaDraft(page);
  await restoreHolmenProfile(page);
  checks.push('SCA edits, cleared cash/recovery inputs and saved revisions survive company switches and reload without changing Holmen');
  return { checks };
}
