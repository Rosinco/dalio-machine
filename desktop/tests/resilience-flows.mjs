import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { selectObservatory } from './workspace-navigation.mjs';

const companyKey = 'macro-atlas-resilience-review-v1:102';
const portfolioKey = 'macro-atlas-portfolio-stress-v1';
const company = page => page.locator('[data-resilience-company="102"]');
const portfolio = page => page.locator('[data-portfolio-stress="true"]');
const fill = (scope, label, value) => scope.getByLabel(label, { exact: typeof label === 'string' }).fill(String(value));
const openReviews = async page => {
  for (const name of ['Survival and permanent-loss review', 'Shared-shock portfolio review']) {
    const summary = page.getByText(name, { exact: true });
    await summary.waitFor();
    if (await summary.locator('..').getAttribute('open') === null) await summary.click();
  }
};
const research = async page => {
  await page.getByLabel('Company research', { exact: true }).click();
  await openReviews(page);
  await company(page).waitFor();
};
const valuation = async page => {
  await page.getByLabel('Company valuation', { exact: true }).click();
  await page.locator('.valuation-workspace').waitFor();
};
async function protectedStorage(page) {
  return page.evaluate(() => Object.fromEntries(Object.keys(localStorage).filter(k => k.startsWith('macro-atlas-valuation') || k.startsWith('macro-atlas-company-notes')).sort().map(k => {
    const raw = localStorage.getItem(k);
    if (!k.startsWith('macro-atlas-valuation-draft')) return [k, raw];
    const { id: _id, created: _created, ...authored } = JSON.parse(raw);
    return [k, authored];
  })));
}

export async function resilienceFlows(page, project) {
  const checks = [];
  await selectObservatory(page, 'companies');
  await page.getByLabel('Company financials', { exact: true }).click();
  await page.getByLabel('Search companies or countries').fill('Holmen');
  await page.getByLabel('Search companies or countries').press('Enter');
  await page.locator('[data-business-ready="true"]').waitFor();
  await valuation(page);
  const original = await protectedStorage(page);
  assert.equal(await page.locator('[data-review-status="unassessed"]').count(), 1);
  await page.getByRole('button', { name: 'Review survival & permanent loss', exact: true }).click();
  await openReviews(page);
  await company(page).waitFor();
  assert.deepEqual(await page.evaluate(([a, b]) => [localStorage.getItem(a), localStorage.getItem(b)], [companyKey, portfolioKey]), [null, null]);
  assert.deepEqual(await protectedStorage(page), original);
  checks.push('Opening company and portfolio reviews writes neither review records nor valuation/notebook data');

  const c = company(page);
  for (const [label, value] of [
    ['Resilience review currency', 'SEK'], ['Liquidity evidence date', '2026-09-26'],
    ['Liquidity source evidence', 'TEST FIXTURE: report cash and debt note, page 12; all amounts illustrative.'],
    ['Combined shock assumptions', 'TEST FIXTURE: demand contraction and closed refinancing market.'],
    ['Funding availability and covenant constraints', 'No undrawn facility assumed; covenant terms require explicit review.'],
    ['Liquidity opening date', '2026-09-26'], ['Opening unrestricted cash', 100], ['Minimum operating liquidity', 20],
    ['Existing ownership after financing and rescue', 'Existing shareholders retain 100%; no rescue funding assumed.'],
    ['Permanent-impairment assumptions', 'Permanent closure; zero proceeds to the existing equity claim.'],
    ['Impairment evidence date', '2026-09-26'], ['Impairment source evidence', 'TEST FIXTURE: hypothetical closure assumptions, not issuer research.'],
    ['Annual cash to the existing claim', '0,0,0'], ['Impairment required annual return (%)', 10],
    ['Final net proceeds to the existing claim', 0], ['Optional purchase price of the existing claim', 100],
    ['Impairment purchase price date', '2026-09-26'], ['Purchase price source and ownership basis', 'Illustrative price for the whole existing equity claim.'],
    ['Conclusion, unresolved questions and monitoring evidence', 'Conditional scenario only. No probability or loss bound; review changed financing conditions.'],
  ]) await fill(c, label, value);
  for (let i = 1; i <= 4; i++) {
    await fill(c, `Period ${i} net operating cash`, i === 4 ? 200 : 0);
    await fill(c, `Period ${i} principal due`, i === 2 ? 90 : 0);
    await fill(c, new RegExp(`^Period ${i} .*funding`), 0);
    await fill(c, `Period ${i} extra shock drain`, 0);
  }
  await c.getByLabel('Resilience review conclusion').selectOption('reviewed-for-stated-scenario');
  assert.equal(await c.locator('[data-resilience-status="material-failure"]').count(), 1);
  assert.match(await c.locator('[data-liquidity-results]').innerText(), /Period 2/);
  assert.match(await c.locator('[data-impairment-value]').innerText(), /^0 SEK/);
  assert.match(await c.locator('[data-impairment-npv]').innerText(), /-100 SEK/);
  await c.getByRole('button', { name: 'Save resilience review', exact: true }).click();
  await page.reload(); await openReviews(page); await company(page).waitFor();
  assert.equal(await c.locator('[data-resilience-status="material-failure"]').count(), 1);
  checks.push('A liquidity breach remains material despite later recovery and an authored reviewed conclusion; zero impairment recovery persists');

  await fill(c, 'Period 2 principal due', 0);
  await fill(c, 'Period 4 net operating cash', 0);
  for (let i = 1; i <= 4; i++) await fill(c, `Period ${i} extra shock drain`, 10);
  assert.match(await c.locator('[data-resilience-extra-drain]').innerText(), /^10 SEK/);
  await fill(c, 'Combined shock assumptions', 'TEST FIXTURE: revised shock retained across navigation before saving.');
  await valuation(page);
  assert.equal(await page.locator('[data-review-status="material-failure"]').count(), 1, 'Value must read the saved review, not unsaved edits');
  await research(page);
  assert.match(await c.getByLabel('Combined shock assumptions').inputValue(), /retained across navigation/);
  assert.match(await c.locator('[data-resilience-extra-drain]').innerText(), /^10 SEK/);
  await c.getByRole('button', { name: 'Save resilience review', exact: true }).click();
  await valuation(page);
  assert.equal(await page.locator('[data-review-status="reviewed-for-stated-scenario"]').count(), 1);
  assert.deepEqual(await protectedStorage(page), original);
  checks.push('Reverse stress finds 10 million extra drain per quarter; unsaved edits survive navigation; saved status reaches Value without altering DCF');

  await research(page);
  const p = portfolio(page);
  for (const [label, value] of [
    ['Portfolio starting-weight date', '2026-09-26'], ['Portfolio tolerable loss (%)', 30],
    ['Portfolio exposure basis and source', 'TEST FIXTURE: two disjoint positions, same starting capital.'],
    ['Portfolio shared shock', 'Common lender withdraws uncommitted refinancing during demand collapse.'],
    ['Portfolio exposure evidence and sources', 'Hypothetical exposures for this test; not real holdings.'],
    ['Portfolio stress assumptions', 'Assume each entered position loses all its initial value.'],
    ['Portfolio position 1 name', 'Company A'], ['Portfolio position 1 driver', 'Shared funding shock'],
    ['Portfolio position 1 weight (%)', 20], ['Portfolio position 1 loss (%)', 100],
  ]) await fill(p, label, value);
  await p.getByRole('button', { name: 'Add exposure row', exact: true }).click();
  for (const [label, value] of [['Portfolio position 2 name', 'Company B'], ['Portfolio position 2 driver', 'Shared funding shock'], ['Portfolio position 2 weight (%)', 20], ['Portfolio position 2 loss (%)', 100]]) await fill(p, label, value);
  assert.equal(await p.locator('[data-loss-percent="40"]').count(), 1);
  assert.match(await p.locator('[data-portfolio-stress-result]').innerText(), /10 percentage points above/);
  await valuation(page); await research(page);
  assert.equal(await p.locator('[data-loss-percent="40"]').count(), 1);
  await p.getByRole('button', { name: 'Save portfolio stress', exact: true }).click();
  await page.reload(); await openReviews(page); await company(page).waitFor();
  assert.equal(await p.locator('[data-loss-percent="40"]').count(), 1);
  assert.match(await p.innerText(), /unmodeled portfolio exposure/i);
  checks.push('Two 20% wipeouts produce 40% simultaneous portfolio loss with unmodeled residual disclosed; unsaved navigation and saved reload retain inputs');

  await fill(p, 'Portfolio position 2 weight (%)', 90);
  assert.equal(await p.locator('[data-portfolio-stress-result]').count(), 0);
  await fill(p, 'Portfolio position 2 weight (%)', '');
  assert.equal(await p.locator('[data-portfolio-stress-result]').count(), 0);
  await fill(p, 'Portfolio position 2 weight (%)', 20);
  checks.push('Overallocated and missing portfolio weights withhold results');

  await page.setViewportSize({ width: 420, height: 900 });
  await c.getByRole('heading', { name: 'Survival and permanent loss', exact: true }).scrollIntoViewIfNeeded();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1), false, 'Mobile page must not overflow horizontally');
  await page.screenshot({ path: resolve(project, 'test-results/resilience-mobile.png') });
  await p.locator('[data-portfolio-stress-result]').scrollIntoViewIfNeeded();
  await page.screenshot({ path: resolve(project, 'test-results/portfolio-stress-mobile.png') });
  await page.setViewportSize({ width: 1500, height: 960 });
  checks.push('Company and portfolio forms remain within a 420px mobile viewport');

  const savedPortfolio = await page.evaluate(key => localStorage.getItem(key), portfolioKey);
  await page.evaluate(key => {
    const original = Storage.prototype.setItem;
    window.__restoreStressStorage = () => { Storage.prototype.setItem = original; };
    Storage.prototype.setItem = function(k, v) { if (k === key) throw new Error('TEST quota exhausted'); return original.call(this, k, v); };
  }, portfolioKey);
  await fill(p, 'Portfolio tolerable loss (%)', 25);
  await p.getByRole('button', { name: 'Save portfolio stress', exact: true }).click();
  assert.match(await p.getByRole('alert').innerText(), /Could not save/);
  assert.equal(await page.evaluate(key => localStorage.getItem(key), portfolioKey), savedPortfolio);
  await page.evaluate(() => window.__restoreStressStorage());
  checks.push('Failed storage writes stay visible and preserve the prior portfolio record');

  await page.evaluate(key => {
    const record = JSON.parse(localStorage.getItem(key)); record.releaseId = 'earlier-source';
    localStorage.setItem(key, JSON.stringify(record));
  }, companyKey);
  await page.reload(); await openReviews(page); await company(page).waitFor();
  assert.equal(await c.locator('[data-resilience-status="stale"]').count(), 1);
  await c.getByRole('button', { name: 'Start reassessment using current sources', exact: true }).click();
  assert.equal(await c.getByLabel('Opening unrestricted cash').inputValue(), '100');
  assert.equal(await c.getByLabel('Resilience review conclusion').inputValue(), 'unassessed');
  assert.equal(await page.evaluate(key => JSON.parse(localStorage.getItem(key)).releaseId, companyKey), 'earlier-source');
  checks.push('Source mismatch marks the review stale; explicit reassessment preserves amounts and resets conclusion without an implicit save');

  const unreadableCompany = '{keep original company review', unsupportedPortfolio = JSON.stringify({ version: 999, preserve: 'future authored data' });
  await page.evaluate(({ companyKey, portfolioKey, unreadableCompany, unsupportedPortfolio }) => {
    localStorage.setItem(companyKey, unreadableCompany); localStorage.setItem(portfolioKey, unsupportedPortfolio);
  }, { companyKey, portfolioKey, unreadableCompany, unsupportedPortfolio });
  await page.reload(); await openReviews(page); await company(page).waitFor();
  assert.equal(await c.getByLabel('Opening unrestricted cash').isDisabled(), true);
  assert.equal(await p.getByLabel('Portfolio tolerable loss (%)').isDisabled(), true);
  assert.deepEqual(await page.evaluate(([a, b]) => [localStorage.getItem(a), localStorage.getItem(b)], [companyKey, portfolioKey]), [unreadableCompany, unsupportedPortfolio]);
  assert.deepEqual(await protectedStorage(page), original);
  checks.push('Malformed/future records are preserved and blocked from editing; unrelated valuation and notebook data remain unchanged');
  return { checks, screenshots: ['resilience-mobile.png', 'portfolio-stress-mobile.png'], scope: 'Isolated browser profile; no installed-app data accessed' };
}
