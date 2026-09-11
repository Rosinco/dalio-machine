import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export async function valuationFlows(page, project, { native = false } = {}) {
  const checks = [];
  const openCompany = async name => {
    await page.getByLabel('Search companies or countries').fill(name);
    await page.getByLabel('Search companies or countries').press('Enter');
    await page.locator('[data-business-ready="true"]').waitFor();
    await page.getByLabel('Company valuation', { exact: true }).click();
    await page.locator('.valuation-workspace').waitFor();
  };
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await page.locator('[data-business-ready="true"]').waitFor();
  await openCompany('Holmen');
  assert.equal(await page.locator('.valuation-workspace').getAttribute('data-valuation-company'), '102');
  await page.locator('[data-valuation-ready="true"][data-valuation-study="holmen-2026-09-11-v1"]').waitFor();
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).count(), 0);
  assert.equal(await page.locator('[data-scenario="mid"] [data-result="value"]').innerText(), '401.38 SEK');
  assert.equal(await page.locator('[data-scenario="high"] [data-result="payback"]').innerText(), 'Year 16');
  assert.match(await page.locator('.valuation-price-summary').innerText(), /2026-08-07/);
  await page.screenshot({ path: resolve(project, 'test-results/holmen-automatic-valuation.png') });
  await page.locator('.valuation-charts').scrollIntoViewIfNeeded();
  await page.getByRole('img', { name: 'Discounted cash flow: low, mid and high scenarios', exact: true }).locator('canvas').waitFor();
  await page.screenshot({ path: resolve(project, 'test-results/holmen-automatic-charts.png') });
  checks.push('Holmen opens with sourced price, three complete researched scenarios, charts and payback without entering inputs');

  await page.getByRole('button', { name: 'Inspect sources and calculations', exact: true }).click();
  await page.locator('[data-research-study="holmen-2026-09-11-v1"]').waitFor();
  assert.match(await page.locator('.valuation-research-evidence').innerText(), /1,243 SEK m/);
  assert.equal(await page.getByLabel('Tangible book equity', { exact: true }).inputValue(), '53700');
  assert.equal(await page.getByLabel('Gross corporate debt', { exact: true }).inputValue(), '6917');
  checks.push('Cash normalization, capital, source dates and analyst recovery assumptions are inspectable offline');

  // Simulate a user's old autosaved draft containing only a selected price.
  // This isolated test profile must migrate once and retain the original work.
  await page.evaluate(() => {
    const key = Object.keys(localStorage).find(k => k.startsWith('macro-atlas-valuation-draft-v1:102:'));
    const previous = JSON.parse(localStorage.getItem(key));
    const d = previous.draft; delete d.researchOrigin;
    d.title = 'PREVIOUS USER DRAFT'; d.marketCap = 58000; d.priceSource = 'Previously selected price';
    d.years = 10;
    for (const s of Object.values(d.scenarios)) Object.assign(s, { cashFlows: Array(10).fill(null), discountRate: null, terminalEquity: 0, recoveryEquity: null, recoveryYear: null, rationale: '' });
    for (const k of Object.keys(d.capital)) d.capital[k] = null;
    for (const k of Object.keys(d.notes)) d.notes[k] = '';
    localStorage.setItem(key, JSON.stringify(previous));
  });
  await page.reload();
  await page.locator('[data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.match(await page.locator('.valuation-saved').innerText(), /PREVIOUS USER DRAFT/);
  await page.locator('.valuation-saved').getByRole('button', { name: /PREVIOUS USER DRAFT/ }).click();
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '58000');
  await page.reload();
  await page.locator('[data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '58000');
  assert.equal(await page.locator('[data-valuation-ready="false"]').count(), 1);
  await page.getByRole('button', { name: 'Start from researched assumptions', exact: true }).click();
  await page.locator('[data-valuation-ready="true"]').waitFor();
  checks.push('Old empty forecasts migrate automatically with a saved backup; explicitly restored drafts are not overwritten after reload');

  await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
  await page.getByLabel('Mid year 1 cash payment', { exact: true }).fill('');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  await page.reload();
  await page.locator('[data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  assert.equal(await page.getByLabel('Mid year 1 cash payment', { exact: true }).inputValue(), '');
  checks.push('Clearing a researched input persists; automatic defaults never refill a user edit');

  await page.getByLabel('Valuation study title').fill('TEST FIXTURE — Swedish scenario study');
  await page.getByLabel('Valuation date', { exact: true }).fill('2026-09-11');
  await page.getByLabel('Valuation price date').fill('2026-09-11');
  await page.getByLabel('Equity market value', { exact: true }).fill('1000');
  await page.getByLabel('Valuation price source').fill('Hypothetical test inputs, not a Holmen valuation.');
  await page.getByLabel('Valuation forecast years').fill('10');
  await page.getByText('Generate an editable cash-flow path', { exact: true }).click();
  for (const [key, name, amount] of [['low', 'Low', 120], ['mid', 'Mid', 200], ['high', 'High', 350]]) {
    await page.getByLabel(`${name} required return`, { exact: true }).fill('10');
    await page.getByLabel(`${name} final equity sale`, { exact: true }).fill('0');
    await page.getByLabel('Generate valuation scenario').selectOption(key);
    await page.getByLabel('Generated first cash payment').fill(String(amount));
    await page.getByLabel('Generated cash payment growth').fill('0');
    await page.getByRole('button', { name: `Fill ${name} path`, exact: true }).click();
  }
  await page.locator('[data-valuation-ready="true"]').waitFor();
  const text = (key, field) => page.locator(`[data-scenario="${key}"] [data-result="${field}"]`).innerText();
  assert.equal(await text('mid', 'value'), '1,228.91 SEK');
  assert.equal(await text('mid', 'npv'), '228.91 SEK');
  assert.equal(await text('low', 'payback'), 'Year 9');
  assert.equal(await text('low', 'discounted-payback'), 'Not reached in 10 years');
  assert.equal(await text('mid', 'payback'), 'Year 5');
  assert.equal(await text('mid', 'discounted-payback'), 'Year 8');
  assert.equal(await text('high', 'payback'), 'Year 3');
  assert.equal(await text('high', 'discounted-payback'), 'Year 4');
  checks.push('Independent constant-annuity values and all six cash/disc payback results match');

  await page.locator('.valuation-charts').scrollIntoViewIfNeeded();
  for (const label of ['Discounted cash flow: low, mid and high scenarios', 'Cumulative NPV: low, mid and high scenarios']) {
    const chart = page.getByRole('img', { name: label, exact: true });
    await chart.locator('canvas').waitFor();
    assert.ok(await chart.locator('canvas').first().evaluate(el => el.width > 100 && el.height > 100));
  }
  await page.screenshot({ path: resolve(project, 'test-results/valuation-charts.png') });
  checks.push('DCF and cumulative NPV render with three scenario paths and a range envelope');

  await page.getByLabel('High final equity sale', { exact: true }).fill('1000');
  assert.equal(await text('high', 'payback'), 'Year 3');
  await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).check();
  assert.match(await page.locator('.valuation-charts').innerText(), /Includes the assumed equity sale/);
  await page.getByLabel('Low net equity recovery').fill('0');
  await page.getByLabel('Low recovery payment year').fill('2');
  assert.match(await page.locator('[data-recovery="low"]').innerText(), /Present value: 0 SEK[\s\S]*NPV: -1,000 SEK/);
  assert.equal(await text('mid', 'value'), '1,228.91 SEK');
  checks.push('Final sale and zero recovery are explicit and never added twice or used for cash-only payback');

  await page.getByLabel('Mid year 2 cash payment', { exact: true }).fill('');
  await page.locator('[data-valuation-ready="false"]').waitFor();
  assert.equal(await page.locator('.valuation-charts').count(), 0);
  await page.getByLabel('Mid year 2 cash payment', { exact: true }).fill('200');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).click();
  await page.getByLabel('Gross corporate debt', { exact: true }).fill('500');
  await page.getByLabel('Available surplus cash', { exact: true }).fill('100');
  await page.getByLabel('Macro and branch evidence → company exposure → forecast assumption').fill('Svenska räntor; verified customer exposure is required.');
  await page.getByRole('button', { name: 'Save study revision', exact: true }).click();
  await page.getByRole('status').filter({ hasText: 'Study revision saved' }).waitFor();
  checks.push('Missing inputs remove charts; capital and Swedish evidence notes persist with the study');

  await page.reload();
  await page.locator('[data-business-ready="true"]').waitFor();
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.equal(await text('mid', 'value'), '1,228.91 SEK');
  assert.match(await page.locator('.valuation-saved').innerText(), /TEST FIXTURE/);
  await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).click();
  assert.match(await page.getByLabel('Macro and branch evidence → company exposure → forecast assumption').inputValue(), /Svenska räntor/);
  await page.getByRole('tab', { name: 'Value, price & payback', exact: true }).click();
  checks.push('Draft and saved revision survive reload with original company, price and data versions');

  let exported;
  if (native) {
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    await page.getByRole('status').filter({ hasText: 'Saved to ' }).waitFor();
    const path = (await page.getByRole('status').innerText()).replace(/^Saved to /, '');
    exported = await readFile(path, 'utf8');
  } else {
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export calculations', exact: true }).click();
    const download = await downloadPromise, path = resolve(project, 'test-results/valuation-export.csv');
    await download.saveAs(path); exported = await readFile(path, 'utf8');
  }
  assert.equal(exported.split('\r\n').length, 34);
  assert.match(exported, /cumulative_npv_with_sale_m/); assert.match(exported, /Hypothetical test inputs/);
  checks.push('CSV export retains the three annual paths, price, discount rates and source versions');

  await page.setViewportSize({ width: 1100, height: 760 });
  await page.locator('.valuation-charts').scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
  await page.screenshot({ path: resolve(project, 'test-results/valuation-compact.png') });
  await page.setViewportSize({ width: 1500, height: 960 });
  await openCompany('Stora');
  assert.equal(await page.getByLabel('Equity market value', { exact: true }).inputValue(), '');
  await openCompany('Holmen');
  await page.locator('[data-valuation-ready="true"]').waitFor();
  assert.equal(await text('mid', 'value'), '1,228.91 SEK');
  await page.getByRole('button', { name: 'Company profile', exact: true }).click();
  await page.locator('[data-business-ready="true"]').waitFor();
  checks.push('Company switching isolates drafts, restores prior work and fits the compact viewport');
  return { checks };
}
