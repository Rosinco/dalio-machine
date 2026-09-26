import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { selectObservatory, openListPanel, closeListPanel } from './workspace-navigation.mjs';

const listKey = 'macro-atlas-company-lists-v2';
const revisionKey = 'macro-atlas-valuations-v1';
const draftPrefix = 'macro-atlas-valuation-draft-v1:';
const notePrefix = 'macro-atlas-company-notes-v1:';
const back = page => page.getByRole('button', { name: 'Back', exact: true });
const forward = page => page.getByRole('button', { name: 'Forward', exact: true });
const list = page => page.locator('[data-company-list-ready="true"]');
const note = page => page.getByLabel('1. Understand the business', { exact: true });
const listBytes = page => page.evaluate(key => localStorage.getItem(key), listKey);

async function company(page, id, view) {
  await page.locator(`[data-observatory="companies"][data-business-view="${view}"]`).waitFor();
  // Valuation has its own ready marker rather than the Financials sidebar.
  if (view === 'valuation') await page.locator(`.valuation-workspace[data-valuation-company="${id}"]`).waitFor();
  else await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
}

async function workBytes(page) {
  return page.evaluate(({ revisionKey, draftPrefix, notePrefix }) => Object.fromEntries(Object.keys(localStorage)
    .filter(key => key === revisionKey || key.startsWith(draftPrefix) || key.startsWith(notePrefix))
    .sort().map(key => [key, localStorage.getItem(key)])), { revisionKey, draftPrefix, notePrefix });
}

// Valuation already renews only these two envelope fields on mount. Navigation
// must preserve every authored field, source binding, deliberate blank and
// revision; it must not behave like undo for research work.
function authoredWork(records) {
  return Object.fromEntries(Object.entries(records).map(([key, raw]) => {
    if (!key.startsWith(draftPrefix)) return [key, raw];
    const { id: _mountId, created: _mountTime, ...authored } = JSON.parse(raw);
    return [key, authored];
  }));
}

function assertOriginalFields(actual, expected, path = 'authored valuation') {
  if (expected && typeof expected === 'object' && !Array.isArray(expected)) {
    for (const [key, value] of Object.entries(expected)) assertOriginalFields(actual?.[key], value, `${path}.${key}`);
  } else assert.deepEqual(actual, expected, `${path} remains unchanged`);
}

async function seedDraft(page, project) {
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const gauge = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  delete draft.starterOrigin; delete draft.researchOrigin;
  draft.researchAutofillDisabled = true;
  draft.title = 'TEST FIXTURE — navigation retains authored cash and deliberate gaps';
  draft.scenarios.mid.cashFlows[1] = null;
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const valuation = { format: 'macro-atlas-valuation', version: 1, id: 'navigation-authored', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: gauge.financialPackId, taxonomy: gauge.taxonomySha256, draft };
  const key = `${draftPrefix}${valuation.company}:${valuation.release}:${valuation.financial}:${valuation.taxonomy}`;
  await page.evaluate(({ key, valuation, revisionKey }) => {
    localStorage.setItem(key, JSON.stringify(valuation));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...valuation, id: 'navigation-revision' }] }));
  }, { key, valuation, revisionKey });
  return authoredWork(await workBytes(page));
}

async function titleContains(button, direction, words) {
  const title = await button.getAttribute('title');
  assert.ok(title?.startsWith(`${direction} to `), `History title identifies a destination: ${title}`);
  for (const word of words) assert.ok(title.toLowerCase().includes(word.toLowerCase()), `${title} includes ${word}`);
}

async function scrollFinancials(page) {
  const state = await page.locator('#company-business-overview').evaluate(node => {
    for (let element = node.parentElement; element; element = element.parentElement) {
      if (/(auto|scroll)/.test(getComputedStyle(element).overflowY) && element.scrollHeight - element.clientHeight > 500) {
        const selector = ['business-content', 'business-stage-content', 'business-sidebar'].find(name => element.classList.contains(name));
        if (!selector) continue;
        element.scrollTop = 420;
        return { selector: `.${selector}`, top: element.scrollTop };
      }
    }
    window.scrollTo(0, 420);
    return { selector: 'window', top: scrollY };
  });
  assert.ok(state.top >= 300, `Financial history has been genuinely scrolled: ${JSON.stringify(state)}`);
  return state;
}

async function assertScroll(page, expected) {
  await page.waitForFunction(({ selector, top }) => Math.abs((selector === 'window' ? scrollY : document.querySelector(selector)?.scrollTop ?? -999) - top) <= 2, expected, { timeout: 4500 });
}

async function assertSectionReached(page, selector) {
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

async function branchState(page) {
  return {
    branch: await page.getByLabel('Comparison branch', { exact: true }).inputValue(),
    metric: await page.getByLabel('Branch comparison metric', { exact: true }).inputValue(),
    size: await page.getByLabel('Branch bubble size', { exact: true }).inputValue(),
    currency: await page.getByLabel('Comparison reporting currency', { exact: true }).inputValue(),
    year: await page.getByLabel('Comparison selected year', { exact: true }).inputValue(),
    focus: await page.locator('[data-comparison-focus]').getAttribute('data-comparison-focus'),
    notes: await page.getByLabel('Comparison research notes', { exact: true }).inputValue(),
    title: await page.getByLabel('Saved comparison name', { exact: true }).inputValue(),
    query: await page.getByLabel('Find branch comparison listings', { exact: true }).inputValue(),
    selected: await page.locator('[data-comparison-listing]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-comparison-listing'))),
  };
}

async function financialState(page) {
  return {
    quarterly: await page.getByRole('button', { name: 'Quarterly', exact: true }).getAttribute('aria-pressed'),
    metric: await page.getByLabel('Company financial chart', { exact: true }).inputValue(),
    statement: await page.getByRole('group', { name: 'Financial statement', exact: true }).getByRole('button', { name: 'Cash flow', exact: true }).getAttribute('aria-pressed'),
    period: await page.getByLabel('Statement period', { exact: true }).inputValue(),
  };
}

export async function assertNavigationRestart(page, expected) {
  await company(page, '102', 'research');
  await note(page).waitFor();
  assert.equal(await note(page).inputValue(), expected.note);
  assert.equal(await listBytes(page), expected.lists);
  assert.deepEqual(await workBytes(page), expected.work, 'Reopening Research preserves exact list, notebook and valuation bytes');
  assert.equal(await back(page).isDisabled(), true, 'Back starts empty for each application session');
  assert.equal(await forward(page).isDisabled(), true, 'Forward starts empty for each application session');
}

export async function navigationFlows(page, project, { native = false } = {}) {
  const checks = [], screenshots = [];
  const originalViewport = page.viewportSize() ?? await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  const checked = text => { checks.push(text); console.log(`Navigation: ${text}`); };
  const image = async name => {
    const path = resolve(project, `test-results/navigation-${native ? 'native' : 'browser'}-${name}.png`);
    await page.screenshot({ path }); screenshots.push(path);
  };
  const { version } = JSON.parse(await readFile(resolve(project, 'package.json'), 'utf8'));
  try {
    await list(page).waitFor();
    assert.equal(await back(page).isDisabled(), true); assert.equal(await forward(page).isDisabled(), true);
    const initialAuthored = await seedDraft(page, project);
    await openListPanel(page, 'filters');
    await page.getByLabel('Company list preset', { exact: true }).selectOption('all');
    await closeListPanel(page, 'filters');
    const bounds = page.locator('[data-company-list-range="ebit_margin"]');
    await bounds.getByRole('textbox', { name: /^Min / }).fill('0');
    await bounds.getByRole('textbox', { name: /^Max / }).fill('100');
    await page.getByLabel('Company list search', { exact: true }).fill('Holmen');
    const holmenRow = page.locator('[data-company-listing="102"]'); await holmenRow.waitFor();
    const filteredLists = await listBytes(page);
    assert.equal(await back(page).isDisabled(), true, 'Editing a list does not create fictitious visited screens');
    checked('A fresh session has disabled Back/Forward; changing list filters does not create navigation history');

    await holmenRow.getByRole('button', { name: 'Open profile for Holmen', exact: true }).click();
    await company(page, '102', 'financials');
    await titleContains(back(page), 'Back', ['Lists']);
    await page.getByRole('button', { name: 'Quarterly', exact: true }).click();
    await page.getByLabel('Company financial chart', { exact: true }).selectOption('free_cash_flow');
    await page.getByRole('group', { name: 'Financial statement', exact: true }).getByRole('button', { name: 'Cash flow', exact: true }).click();
    await page.getByLabel('Statement period', { exact: true }).selectOption({ index: 1 });
    const financialControls = await financialState(page);
    const financialScroll = await scrollFinancials(page);
    await page.getByLabel('Company valuation', { exact: true }).click(); await company(page, '102', 'valuation');
    await titleContains(back(page), 'Back', ['Holmen', 'Financials']);
    // The older fixture may gain current-format default fields on its first
    // mount. Every original authored field must survive that initialization.
    assertOriginalFields(authoredWork(await workBytes(page)), initialAuthored);
    await page.getByLabel('Company research', { exact: true }).click(); await company(page, '102', 'research');
    const researchNote = 'TEST FIXTURE — retain this company thesis when moving back. Förvärv och kassaflöden.';
    await note(page).fill(researchNote);
    const authored = authoredWork(await workBytes(page));
    await titleContains(back(page), 'Back', ['Holmen', 'Valuation']);
    await back(page).click(); await company(page, '102', 'valuation');
    await page.keyboard.press('Alt+ArrowLeft'); await company(page, '102', 'financials');
    assert.deepEqual(await financialState(page), financialControls);
    await assertScroll(page, financialScroll);
    await back(page).click(); await list(page).waitFor();
    assert.equal(await back(page).isDisabled(), true);
    assert.equal(await listBytes(page), filteredLists);
    assert.equal(await page.getByLabel('Company list search', { exact: true }).inputValue(), 'Holmen');
    assert.equal(await bounds.getByRole('textbox', { name: /^Min / }).inputValue(), '0');
    assert.equal(await bounds.getByRole('textbox', { name: /^Max / }).inputValue(), '100');
    assert.deepEqual(authoredWork(await workBytes(page)), authored);
    checked('Lists → Financials → Valuation → Research returns one real screen at a time, restoring the quarterly measure, statement period, scroll and exact filtered Lists while retaining authored work');

    await titleContains(forward(page), 'Forward', ['Holmen', 'Financials']);
    await page.keyboard.press('Alt+ArrowRight'); await company(page, '102', 'financials');
    await assertScroll(page, financialScroll);
    await forward(page).click(); await company(page, '102', 'valuation');
    await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).click();
    assert.equal(await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).getAttribute('aria-selected'), 'true');
    await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).focus();
    await page.keyboard.press('Alt+ArrowLeft');
    assert.equal(await page.getByRole('tab', { name: 'Value, price & payback', exact: true }).getAttribute('aria-selected'), 'true');
    await page.getByRole('tab', { name: 'Value, price & payback', exact: true }).focus();
    await page.keyboard.press('Alt+ArrowRight');
    assert.equal(await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).getAttribute('aria-selected'), 'true');
    await page.getByLabel('Company financials', { exact: true }).click(); await company(page, '102', 'financials');
    await page.getByLabel('Search companies or countries', { exact: true }).fill('Holmen');
    await page.locator('[data-search-listing="102"]').click(); await company(page, '102', 'financials');
    await back(page).click(); await company(page, '102', 'valuation');
    assert.equal(await page.getByRole('tab', { name: 'Business, capital & evidence', exact: true }).getAttribute('aria-selected'), 'true', 'Reselecting the same company Financials does not create an invisible duplicate route');
    await back(page).click();
    await back(page).click(); await company(page, '102', 'financials');
    await page.getByLabel('Company peers', { exact: true }).click(); await company(page, '102', 'peers');
    assert.equal(await forward(page).isDisabled(), true, 'A new destination replaces the abandoned forward path');
    checked('Alt+Left/Right, visible Back/Forward and Valuation tabs share the same history; opening Peers after going back clears the old forward path');

    await page.getByLabel('Peer comparison metric', { exact: true }).selectOption('return_on_capital');
    await page.locator('[data-peer="197"] button').click();
    await page.locator('[data-company="197"][data-business-ready="true"]').waitFor();
    await titleContains(back(page), 'Back', ['Holmen', 'Peers']);
    await back(page).click(); await company(page, '102', 'peers');
    assert.equal(await page.getByLabel('Peer comparison metric', { exact: true }).inputValue(), 'return_on_capital');
    assert.equal(await page.locator('[data-peer="102"]').getAttribute('class'), 'selected');
    checked('Opening SCA from Holmen Peers and going Back restores Holmen, its Peers screen and the chosen comparison measure');

    await page.getByLabel('Company macro context', { exact: true }).click(); await company(page, '102', 'context');
    await page.getByRole('button', { name: 'Open Sweden in Macro', exact: true }).click();
    await page.locator('[data-active-observatory="macro"] [data-country="SE"]').waitFor();
    await titleContains(back(page), 'Back', ['Holmen']);
    await back(page).click(); await company(page, '102', 'context');
    assert.match(await page.locator('.company-location').innerText(), /Holmen/);
    checked('Company macro context → Macro → Back restores the same company context, rather than opening Lists or a different company');

    await page.getByRole('button', { name: 'Open Sweden in Macro', exact: true }).click();
    await page.locator('[data-active-observatory="macro"] [data-country="SE"]').waitFor();
    await page.getByLabel('History & outlook', { exact: true }).click();
    await page.getByLabel('Map historical indicator', { exact: true }).selectOption('gov_debt_pct_gdp');
    await page.waitForFunction(() => !document.querySelector('.map-footnote')?.textContent.includes('Loading'));
    await page.getByLabel('Historical year', { exact: true }).fill('2005');
    await page.getByRole('tab', { name: 'Indicators', exact: true }).click();
    await page.getByLabel('Search countries', { exact: true }).fill('Sweden');
    await page.getByLabel('Search countries', { exact: true }).press('Enter');
    await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    await page.getByLabel('Search countries', { exact: true }).fill('Germany');
    await page.getByLabel('Search countries', { exact: true }).press('Enter');
    await page.locator('[data-country="DE"][data-ready="true"]').waitFor();
    await back(page).click(); await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    assert.equal(await page.getByLabel('History & outlook', { exact: true }).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByLabel('Map historical indicator', { exact: true }).inputValue(), 'gov_debt_pct_gdp');
    assert.equal(await page.getByLabel('Historical year', { exact: true }).inputValue(), '2005');
    assert.equal(await page.getByRole('tab', { name: 'Indicators', exact: true }).getAttribute('aria-selected'), 'true');
    await back(page).click();
    assert.equal(await page.getByRole('tab', { name: 'Overview', exact: true }).getAttribute('aria-selected'), 'true');
    await back(page).click();
    assert.equal(await page.getByLabel('Fundamentals', { exact: true }).getAttribute('aria-pressed'), 'true');
    await back(page).click(); await company(page, '102', 'context');
    checked('Macro country, tab and mode history returns through the real prior screens with indicator/year intact; reselecting the same country adds no duplicate');

    await selectObservatory(page, 'sectors');
    await page.getByLabel('Search sectors and branches', { exact: true }).fill('Forest');
    const sector = await page.getByLabel('Filter sectors', { exact: true }).locator('option').filter({ hasText: 'Materials' }).getAttribute('value');
    assert.ok(sector); await page.getByLabel('Filter sectors', { exact: true }).selectOption(sector);
    await page.getByLabel('Filter branch coverage', { exact: true }).selectOption('histories');
    await page.getByRole('button', { name: 'Explore Forest & Wood Products', exact: true }).click();
    await page.locator('[data-observatory="sectors"][data-business-view="overview"] [data-branch="21"]').waitFor();
    await back(page).click();
    await page.locator('[data-observatory="sectors"][data-business-view="browse"]').waitFor();
    assert.equal(await page.getByLabel('Search sectors and branches', { exact: true }).inputValue(), 'Forest');
    assert.equal(await page.getByLabel('Filter sectors', { exact: true }).inputValue(), sector);
    assert.equal(await page.getByLabel('Filter branch coverage', { exact: true }).inputValue(), 'histories');
    // Forestry has exactly one fifty-row page in the saved directory. Use the
    // real, larger mining directory to exercise pagination without fake rows.
    await page.getByLabel('Search sectors and branches', { exact: true }).fill('Mining');
    await page.getByRole('button', { name: 'Explore Mining', exact: true }).click();
    await page.getByLabel('Branch companies', { exact: true }).click();
    await page.getByLabel('Show all listing countries', { exact: true }).check();
    assert.equal(await page.getByLabel('Next company page', { exact: true }).isDisabled(), false, 'The full mining directory has another page to exercise');
    await page.getByLabel('Next company page', { exact: true }).click();
    const pagination = await page.locator('.listing-pagination [role="status"]').innerText();
    assert.match(pagination, /^51–/);
    const listing = page.locator('.company-list [data-listing]').first();
    const listingId = await listing.getAttribute('data-listing'); assert.ok(listingId);
    await listing.click(); await company(page, listingId, 'financials');
    await back(page).click();
    await page.locator('[data-observatory="sectors"][data-business-view="companies"]').waitFor();
    assert.equal(await page.getByLabel('Show all listing countries', { exact: true }).isChecked(), true);
    assert.equal(await page.locator('.listing-pagination [role="status"]').innerText(), pagination);
    await page.getByLabel('Browse sectors and branches', { exact: true }).click();
    await page.getByLabel('Search sectors and branches', { exact: true }).fill('Forest');
    await page.getByRole('button', { name: 'Explore Forest & Wood Products', exact: true }).click();
    await page.getByLabel('Branch companies', { exact: true }).click();
    await page.getByLabel('Filter company listings', { exact: true }).fill('Holmen');
    await page.locator('.company-list [data-listing="102"]').click(); await company(page, '102', 'financials');
    await back(page).click();
    await page.locator('[data-observatory="sectors"][data-business-view="companies"]').waitFor();
    assert.equal(await page.getByLabel('Filter company listings', { exact: true }).inputValue(), 'Holmen');
    checked('Industry directory filters and branch-company country scope, page and search survive real drill-down and Back journeys');
    await page.getByLabel('Branch comparison', { exact: true }).click();
    await page.getByLabel('Comparison branch', { exact: true }).selectOption('21');
    await page.locator('[data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
    await page.getByLabel('Branch comparison metric', { exact: true }).selectOption('free_cash_flow');
    await page.getByLabel('Branch bubble size', { exact: true }).selectOption('total_assets');
    await page.getByLabel('Comparison reporting currency', { exact: true }).selectOption('SEK');
    await page.getByLabel('Comparison selected year', { exact: true }).fill('2024');
    await page.getByLabel('Focus Holmen', { exact: true }).click();
    await page.getByLabel('Comparison research notes', { exact: true }).fill('TEST FIXTURE — unfinished branch comparison; keep these notes.');
    await page.getByLabel('Saved comparison name', { exact: true }).fill('TEST FIXTURE — comparison still in progress');
    await page.getByLabel('Find branch comparison listings', { exact: true }).fill('SCA');
    const branch = await branchState(page);
    await page.getByLabel('Open financials for Holmen', { exact: true }).click(); await company(page, '102', 'financials');
    await titleContains(back(page), 'Back', ['comparison']);
    await back(page).click();
    await page.locator('[data-observatory="sectors"][data-business-view="compare"] [data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
    assert.deepEqual(await branchState(page), branch);
    assert.deepEqual(authoredWork(await workBytes(page)), authored);
    assert.equal(await listBytes(page), filteredLists);
    checked('A company opened from a branch comparison returns to that branch, metric, currency, year, focus and unfinished comparison notes');
    await image('restored-comparison');

    await page.getByLabel('Open financials for Holmen', { exact: true }).click(); await company(page, '102', 'financials');
    const chartLink = page.getByRole('navigation', { name: 'Within company financial history', exact: true }).getByRole('link', { name: 'Financial trends', exact: true });
    await chartLink.click(); await assertSectionReached(page, '#company-financial-chart');
    const contentBox = await page.locator('.business-content').boundingBox(); assert.ok(contentBox);
    // ECharts intentionally consumes wheel events over its canvas for zoom.
    // Scroll from container padding so this is a real page/pane gesture.
    await page.mouse.move(contentBox.x + 3, contentBox.y + 100); await page.mouse.wheel(0, -1200);
    await page.waitForFunction(() => document.querySelector('#company-financial-chart').getBoundingClientRect().top > innerHeight);
    await chartLink.click(); await assertSectionReached(page, '#company-financial-chart');
    // Repeating the same section must scroll again without creating a duplicate
    // history destination that traps the user on Back.
    await back(page).click();
    await company(page, '102', 'financials');
    await back(page).click();
    await page.locator('[data-observatory="sectors"][data-business-view="compare"] [data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
    assert.deepEqual(await branchState(page), branch);
    await page.getByLabel('Open financials for Holmen', { exact: true }).click(); await company(page, '102', 'financials');
    await page.getByLabel('Company research', { exact: true }).click(); await company(page, '102', 'research');
    assert.equal(await note(page).inputValue(), researchNote);
    const restart = { lists: await listBytes(page), work: await workBytes(page), note: researchNote };
    await page.reload();
    await assertNavigationRestart(page, restart);
    checked('Reload starts a fresh navigation session at the persisted Research screen while preserving exact notes, revisions, valuation drafts and Lists settings');
    await image('restored-research');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByLabel('Company financials', { exact: true }).click(); await company(page, '102', 'financials');
    await page.mouse.move(10, 500); await page.mouse.wheel(0, 420);
    await page.waitForFunction(() => scrollY >= 300);
    assert.equal(await back(page).isVisible(), true); assert.equal(await forward(page).isVisible(), true);
    assert.equal(await back(page).evaluate(node => {
      const rect = node.getBoundingClientRect();
      return rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight;
    }), true, 'Back remains visible in the narrow topbar after scrolling down the page');
    await image('narrow-scrolled-back');
    await back(page).click(); await company(page, '102', 'research');
    assert.equal(await note(page).inputValue(), researchNote);
    assert.deepEqual(await workBytes(page), restart.work);
    await image('narrow-back');
    checked('After scrolling a narrow window, Back and Forward remain visible and a real Back click returns to the unchanged company notebook');
    await page.setViewportSize(originalViewport);
    return { version, checks, screenshots, financialScroll, branch, restart };
  } catch (error) {
    await image('failure').catch(() => {});
    throw error;
  }
}
