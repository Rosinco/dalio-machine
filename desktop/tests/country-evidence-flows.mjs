import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Shared by the browser runner and the Windows WebView2 verification runner.
export async function countryEvidenceFlows(page, project) {
  const root = resolve(project, 'public/data');
  const index = JSON.parse(await readFile(resolve(root, 'country-evidence/index.json'), 'utf8'));
  const catalogue = JSON.parse(await readFile(resolve(root, 'catalog.json'), 'utf8'));
  const document = async code => JSON.parse(await readFile(resolve(root, 'country-evidence', index.countries[code].file), 'utf8'));
  const checks = [];
  await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
  async function select(query, code) {
    await page.getByLabel('Search countries').fill(query);
    await page.getByLabel('Search countries').press('Enter');
    const tab = page.getByRole('tab', { name: 'Assessments', exact: true });
    if (await tab.count()) await tab.click();
    await page.locator(`[data-country-evidence="${code}"][data-evidence-ready="true"]`).waitFor();
    return page.locator(`[data-country-evidence="${code}"]`);
  }
  const se = await document('SE');
  let panel = await select('Sweden', 'SE');
  assert.equal(await page.getByLabel('Comparison country', { exact: true }).count(), 0);
  assert.equal(await panel.locator('[data-signal]').count(), 5);
  assert.match(await panel.locator('.ce-intro').innerText(), new RegExp(se.monitoring.as_known_at.replaceAll('+', '\\+')));
  const production = se.monitoring.profile.signals.find(s => s.indicator === 'industrial_production');
  assert.ok((await panel.locator('[data-signal="industrial_production"]').innerText()).includes(production.comparison.window));
  assert.ok((await panel.locator('[data-signal="industrial_production"]').innerText()).includes(String(production.latest.value)));
  assert.equal(await panel.locator('[data-signal="industrial_production"] .ce-change').evaluate(e => getComputedStyle(e).color), 'rgb(67, 95, 118)');
  await panel.getByRole('button', { name: 'Inspect native observations', exact: true }).first().click();
  assert.ok((await panel.locator('.ce-history').first().innerText()).includes(production.latest.period));
  await panel.getByLabel('Country evidence annual indicator').selectOption('gov_debt_pct_gdp');
  await panel.locator('[data-history-indicator="gov_debt_pct_gdp"]').waitFor();
  await panel.locator('[data-scenario="demand_shortfall"] summary').click();
  assert.match(await panel.locator('[data-scenario="demand_shortfall"]').innerText(), /Company evidence still required/);
  await page.screenshot({ path: resolve(project, 'test-results/country-evidence-sweden.png') });
  checks.push('Sweden five neutral signals, exact comparison window and source history', 'dated IMF history and conditional company requirements');

  for (const code of ['FI', 'NO', 'BE']) {
    panel = await select(index.countries[code].name, code);
    assert.equal(await panel.locator('[data-signal]').count(), index.countries[code].signals);
    if (!index.countries[code].signals) assert.match(await panel.innerText(), /Monitoring has not been collected/);
  }
  checks.push('Finland, Norway and Belgium selectable without a legacy scored profile');
  panel = await select('GB', 'UK');
  assert.match(await panel.innerText(), /United Kingdom/);
  checks.push('GB listing alias opens UK macro evidence');

  panel = await select('Sweden', 'SE');
  const before = await panel.locator('.ce-intro').innerText();
  const active = await page.locator('.app').getAttribute('data-active-release');
  const other = catalogue.releases.find(r => r.id !== active);
  assert.ok(other, 'Two saved research releases are required for independent date check');
  await page.getByLabel('Open data library').click();
  await page.getByLabel(`Use release ${other.as_of}`, { exact: true }).click();
  await page.locator(`[data-active-release="${other.id}"]`).waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await panel.locator('.ce-intro').innerText(), before);
  assert.equal(await panel.getAttribute('data-evidence-pack'), index.id);
  await page.getByLabel('Open data library').click();
  await page.getByLabel(`Use release ${catalogue.releases.find(r => r.id === active).as_of}`, { exact: true }).click();
  await page.locator(`[data-active-release="${active}"]`).waitFor();
  await page.keyboard.press('Escape');
  checks.push('Research release switching preserves independent evidence dates and identity');

  // National expansion: the structural US lending gap must remain unfilled.
  if (index.countries.US.signals) {
    const us = await document('US');
    panel = await select('United States', 'US');
    const gaps = us.monitoring.profile.signals.filter(s => !s.latest);
    for (const gap of gaps) {
      await panel.locator(`[data-signal="${gap.indicator}"]`).click();
      assert.match(await panel.locator('.ce-signal-detail').innerText(), /No eligible source history/);
      await panel.getByText('Source documentation for this gap', { exact: true }).click();
      assert.match(await panel.locator('.ce-signal-detail').innerText(), /federalreserve.gov\/releases\/e2/);
    }
    assert.ok(gaps.length > 0, 'US new-loan-rate structural gap remains explicit');
    checks.push('US structural gap has no fallback or proxy');
    panel = await select('Canada', 'CA');
    const ca = (await document('CA')).monitoring.profile.signals.find(s => s.indicator === 'industrial_production');
    const card = panel.locator('[data-signal="industrial_production"]');
    assert.ok((await card.innerText()).includes(ca.latest.unit));
    assert.ok((await card.innerText()).includes(ca.latest.period));
    assert.match(await card.innerText(), /millions of chained 2017 Canadian dollars/);
    panel = await select('Germany', 'DE');
    await panel.locator('[data-signal="corporate_new_lending_rate"]').click();
    assert.match(await panel.locator('[data-signal="corporate_new_lending_rate"]').innerText(), /Native P · provisional/);
    await panel.getByRole('button', { name: 'Inspect native observations', exact: true }).first().click();
    assert.match(await panel.locator('.ce-history').first().innerText(), /Native P · provisional/);
    checks.push('Canada real-volume unit and native period retained', 'Germany provisional loan-rate flag visible in card and observations');
  }
  await page.setViewportSize({ width: 1100, height: 760 });
  panel = await select('Finland', 'FI');
  await page.screenshot({ path: resolve(project, 'test-results/country-evidence-compact.png') });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
  await page.setViewportSize({ width: 1500, height: 960 });
  await select('Sweden', 'SE');
  await page.getByRole('tab', { name: 'Overview', exact: true }).click();
  assert.equal(await page.getByLabel('Comparison country', { exact: true }).count(), 1);
  checks.push('Compact country evidence layout fits viewport');
  return { checks, countries: Object.keys(index.countries).length, signals: Object.values(index.countries).reduce((n, c) => n + c.signals, 0), pack: index.id };
}
