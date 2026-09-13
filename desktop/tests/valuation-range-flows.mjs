import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const keys = ['low', 'mid', 'high'];
const draftPrefix = 'macro-atlas-valuation-draft-v1:';
const revisionKey = 'macro-atlas-valuations-v1';
const preferenceKey = 'atlas.preferences';
const chartLabels = { dcf: 'Discounted cash flow: low, mid and high scenarios', npv: 'Cumulative NPV: low, mid and high scenarios' };
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

async function savedDraft(page, id) {
  return page.evaluate(({ id, draftPrefix }) => {
    const key = Object.keys(localStorage).find(key => key.startsWith(`${draftPrefix}${id}:`));
    return key ? { key, saved: JSON.parse(localStorage.getItem(key)) } : null;
  }, { id, draftPrefix });
}

async function fixture(page, id, draft) {
  const saved = await savedDraft(page, id);
  assert.ok(saved, `Company ${id} has a local draft`);
  await page.evaluate(({ key, saved, draft }) => localStorage.setItem(key, JSON.stringify({ ...saved, draft })), { ...saved, draft });
  await reopen(page, id);
  await page.locator(`[data-valuation-company="${id}"][data-valuation-ready="true"]`).waitFor();
  assert.deepEqual((await savedDraft(page, id)).saved.draft, draft);
}

function modelPaths(draft, sale = false) {
  // Independent scalar arithmetic: do not import chart or valuation implementation.
  const scale = draft.investment / draft.marketCap;
  return Object.fromEntries(keys.map(key => {
    const scenario = draft.scenarios[key], rate = 1 + scenario.discountRate / 100;
    const dcf = scenario.cashFlows.slice(0, draft.years).map((cash, index) => cash / rate ** (index + 1) * scale);
    const npv = [-draft.investment];
    for (const payment of dcf) npv.push(npv.at(-1) + payment);
    if (sale) npv[draft.years] += scenario.terminalEquity / rate ** draft.years * scale;
    return [key, { dcf, npv }];
  }));
}

async function rangeTable(page, kind) {
  const card = page.locator(`section[data-valuation-chart="${kind}"]`);
  const summary = card.getByText(kind === 'dcf' ? 'Inspect annual DCF ranges' : 'Inspect cumulative NPV ranges', { exact: true });
  const details = summary.locator('..');
  if (await details.getAttribute('open') === null) await summary.click();
  return card;
}

async function verifyRanges(page, draft, sale = false) {
  const expected = modelPaths(draft, sale), rows = {};
  for (const kind of ['dcf', 'npv']) {
    const card = await rangeTable(page, kind);
    const first = kind === 'dcf' ? 1 : 0;
    assert.equal(await card.locator('[data-range-year]').count(), draft.years + (kind === 'npv' ? 1 : 0));
    rows[kind] = [];
    for (let year = first; year <= draft.years; year++) {
      const row = card.locator(`[data-range-year="${year}"]`), index = year - first;
      const values = keys.map(key => expected[key][kind][index]);
      const wanted = { low: values[0], mid: values[1], high: values[2], min: Math.min(...values), max: Math.max(...values) };
      const actual = {};
      for (const [name, value] of Object.entries(wanted)) {
        const cell = row.locator(`[data-range-value="${name}"]`);
        const raw = await cell.getAttribute('data-value');
        assert.notEqual(raw, null, `${kind} year ${year} ${name} exposes its full numeric value`);
        near(raw, value, `${kind} year ${year} ${name}`);
        assert.match(await cell.innerText(), new RegExp(draft.currency));
        actual[name] = Number(raw);
      }
      rows[kind].push(actual);
    }
  }
  return rows;
}

async function basis(page, kind, year) {
  const row = page.locator(`section[data-valuation-chart="${kind}"] [data-range-year="${year}"]`);
  const value = await row.getAttribute('data-range-basis');
  if (value !== null) return value;
  return row.locator('[data-range-basis]').getAttribute('data-range-basis');
}

function manualFixture(original) {
  const draft = structuredClone(original);
  delete draft.researchOrigin; delete draft.starterOrigin; delete draft.crisis;
  Object.assign(draft, { title: 'TEST FIXTURE — widening annual DCF and cumulative NPV', currency: 'SEK', valuationDate: '2026-09-13', priceDate: '2026-09-12', priceSource: 'Independent hypothetical range test; not a company valuation.', marketCap: 1000, investment: 2500, years: 5, researchAutofillDisabled: true });
  const mid = [100, 120, 140, 160, 180];
  for (const key of keys) Object.assign(draft.scenarios[key], {
    cashFlows: mid.map((cash, i) => cash * (1 + (key === 'low' ? -1 : key === 'high' ? 1 : 0) * (i + 1) / 10)),
    discountRate: 10, terminalEquity: 0, recoveryEquity: null, recoveryYear: null, rationale: 'Explicit 10 percentage point annual cash sensitivity.'
  });
  return draft;
}

export async function valuationRangeFlows(page, project, { native = false } = {}) {
  const checks = [], originalViewport = page.viewportSize();
  const original = await page.evaluate(({ draftPrefix, revisionKey, preferenceKey }) => ({
    drafts: Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix)).map(key => [key, localStorage.getItem(key)])),
    revisions: localStorage.getItem(revisionKey), preferences: localStorage.getItem(preferenceKey),
    valuationOpen: !!document.querySelector('.valuation-workspace')
  }), { draftPrefix, revisionKey, preferenceKey });
  const originalCompany = JSON.parse(original.preferences).companyId;
  const touched = ['159', '696'];
  try {
    await openCompany(page, 'Nordea Bank', '159');
    const increasing = manualFixture((await savedDraft(page, '159')).saved.draft);
    await fixture(page, '159', increasing);
    const growingRows = await verifyRanges(page, increasing);
    for (let i = 1; i < increasing.years; i++) {
      assert.ok(growingRows.dcf[i].max - growingRows.dcf[i].min > growingRows.dcf[i - 1].max - growingRows.dcf[i - 1].min);
      assert.ok(growingRows.npv[i + 1].max - growingRows.npv[i + 1].min > growingRows.npv[i].max - growingRows.npv[i].min);
    }
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, increasing);
    checks.push('Independent growing ±10/20/30/40/50% cash ranges produce exact investment-scaled annual DCF and whole-path cumulative NPV, starting at the negative investment');

    // Close both tables before measuring either card: desktop cards share a grid
    // row, so the other card's expanded table otherwise stretches its screenshot.
    for (const kind of ['dcf', 'npv']) {
      const details = page.locator(`section[data-valuation-chart="${kind}"] details`);
      if (await details.getAttribute('open') !== null) await details.locator('summary').click();
    }
    for (const viewport of [{ width: 1440, height: 960 }, { width: 390, height: 844 }]) {
      await page.setViewportSize(viewport);
      for (const kind of ['dcf', 'npv']) {
        const card = page.locator(`section[data-valuation-chart="${kind}"]`);
        await card.evaluate(el => el.scrollIntoView({ block: 'start', inline: 'nearest' }));
        const canvas = card.getByRole('img', { name: chartLabels[kind], exact: true }).locator('canvas').first();
        await canvas.waitFor();
        assert.ok(await canvas.evaluate(el => el.width > 100 && el.height > 100));
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), false);
        const clippedBadges = await card.locator('.valuation-range-key > span').evaluateAll(badges => badges.flatMap(badge => {
          const outer = badge.getBoundingClientRect();
          return [...badge.querySelectorAll('strong, small')].filter(label => {
            const inner = label.getBoundingClientRect();
            return inner.left < outer.left - 1 || inner.right > outer.right + 1 || inner.top < outer.top - 1 || inner.bottom > outer.bottom + 1;
          }).map(label => label.textContent);
        }));
        assert.deepEqual(clippedBadges, [], `${kind} checkpoint amounts and basis labels fit their badges`);
        await card.screenshot({ path: resolve(project, `test-results/valuation-${kind}-range-${native ? 'native' : 'browser'}-${viewport.width}.png`) });
      }
    }
    await page.setViewportSize(originalViewport ?? { width: 1500, height: 960 });
    checks.push('The actual DCF and NPV cards render their annual ranges at desktop and 390px mobile widths without page overflow');

    const crossing = structuredClone(increasing);
    crossing.years = 3;
    for (const [key, cashFlows, discountRate, terminalEquity] of [
      ['low', [120, -80, 220], 0, 60], ['mid', [100, 80, 100], 10, 90], ['high', [80, 200, -40], 20, 120]
    ]) Object.assign(crossing.scenarios[key], { cashFlows, discountRate, terminalEquity });
    await fixture(page, '159', crossing);
    const withoutSale = await verifyRanges(page, crossing);
    near(withoutSale.dcf[0].max, withoutSale.dcf[0].low, 'Named low is the first-year numerical maximum');
    near(withoutSale.dcf[1].min, -200, 'Negative discounted cash is retained');
    near(withoutSale.npv[2].min, -2400, 'Whole low path is the second-year NPV minimum');
    const stitchedAnnualMin = -crossing.investment + withoutSale.dcf.reduce((sum, row) => sum + row.min, 0);
    assert.ok(Math.abs(withoutSale.npv[3].min - stitchedAnnualMin) > 100, 'NPV cannot combine different scenarios into an invented minimum path');
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, crossing);
    checks.push('Signed cash and differing required returns preserve named scenario crossings; cumulative NPV bounds follow complete scenarios rather than summing each year’s extrema');

    await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).check();
    const withSale = await verifyRanges(page, crossing, true);
    for (let year = 0; year <= crossing.years; year++) {
      const row = page.locator(`section[data-valuation-chart="npv"] [data-range-year="${year}"]`);
      assert.equal(await row.getAttribute('data-includes-sale'), String(year === crossing.years));
    }
    assert.match(await page.locator(`section[data-valuation-chart="npv"] [data-range-year="${crossing.years}"] [data-range-basis]`).innerText(), /assumed sale/);
    assert.deepEqual(withSale.dcf, withoutSale.dcf);
    assert.deepEqual(withSale.npv.slice(0, -1), withoutSale.npv.slice(0, -1));
    assert.deepEqual((await savedDraft(page, '159')).saved.draft, crossing);
    await page.getByRole('checkbox', { name: 'Include final sale', exact: true }).uncheck();
    assert.deepEqual(await verifyRanges(page, crossing), withoutSale);
    checks.push('The sale toggle adds each scenario’s discounted sale only at the final NPV year, leaving annual DCF, earlier NPV and the saved draft unchanged');

    await openCompany(page, 'Stora Enso R', '696');
    // Regenerate an ordinary starter in this isolated test profile without adding
    // a revision or overwriting the empirical flow’s deliberate-gap/crisis draft.
    await page.getByRole('button', { name: 'Company profile', exact: true }).click();
    await page.evaluate(({ draftPrefix }) => Object.keys(localStorage).filter(key => key.startsWith(`${draftPrefix}696:`)).forEach(key => localStorage.removeItem(key)), { draftPrefix });
    await reopen(page, '696');
    await page.locator('[data-valuation-company="696"][data-valuation-ready="true"][data-valuation-starter="empirical-cash-starter-v3"]').waitFor();
    const standard = (await savedDraft(page, '696')).saved.draft;
    await verifyRanges(page, standard);
    for (let year = 1; year <= standard.years; year++) {
      assert.equal(await basis(page, 'dcf', year), year <= 4 ? 'historical' : 'assumed-tail');
      assert.equal(await basis(page, 'npv', year), year <= 4 ? 'historical' : 'historical-and-assumed');
    }
    assert.deepEqual((await savedDraft(page, '696')).saved.draft, standard);
    checks.push('Unedited Stora DCF and NPV tables distinguish historical-error cash ranges through Year 4 from the assumed later-year widening');

    await page.getByRole('button', { name: 'Edit price & assumptions', exact: true }).click();
    await page.getByLabel('Mid year 1 cash payment', { exact: true }).fill(String(standard.scenarios.mid.cashFlows[0] + 25));
    const edited = (await savedDraft(page, '696')).saved.draft;
    await verifyRanges(page, edited);
    for (const kind of ['dcf', 'npv']) for (let year = 1; year <= standard.years; year++) {
      assert.equal(await basis(page, kind, year), 'edited');
    }
    assert.deepEqual((await savedDraft(page, '696')).saved.draft, edited);
    checks.push('Editing the cash assumptions removes the original historical calibration labels from the DCF and NPV ranges');
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/valuation-range-${native ? 'native' : 'browser'}-fixture-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    // Unmount the edited fixture before restoring exact bytes, so autosave cannot
    // replace the originals. Never clear global storage or delete saved studies.
    if (await page.locator('.valuation-workspace').count()) await page.getByRole('button', { name: 'Company profile', exact: true }).click();
    await page.evaluate(({ original, touched, draftPrefix, preferenceKey }) => {
      const touchedKey = key => touched.some(id => key.startsWith(`${draftPrefix}${id}:`));
      for (const key of Object.keys(localStorage).filter(touchedKey)) if (!(key in original.drafts)) localStorage.removeItem(key);
      for (const [key, value] of Object.entries(original.drafts)) if (touchedKey(key)) localStorage.setItem(key, value);
      if (original.preferences === null) localStorage.removeItem(preferenceKey); else localStorage.setItem(preferenceKey, original.preferences);
    }, { original, touched, draftPrefix, preferenceKey });
    await page.reload();
    await page.locator(`[data-company="${originalCompany}"][data-business-ready="true"]`).waitFor();
    if (original.valuationOpen) {
      await page.getByLabel('Company valuation', { exact: true }).click();
      await page.locator(`[data-valuation-company="${originalCompany}"]`).waitFor();
      // Opening a valuation updates autosave metadata but not its draft; preserve
      // that original metadata as well after the normal mount effect has settled.
      await page.evaluate(({ drafts, draftPrefix, originalCompany }) => {
        for (const [key, value] of Object.entries(drafts)) if (key.startsWith(`${draftPrefix}${originalCompany}:`)) localStorage.setItem(key, value);
      }, { drafts: original.drafts, draftPrefix, originalCompany });
    }
    if (originalViewport) await page.setViewportSize(originalViewport);
    const restored = await page.evaluate(({ draftPrefix, revisionKey, preferenceKey }) => ({
      drafts: Object.fromEntries(Object.keys(localStorage).filter(key => key.startsWith(draftPrefix)).map(key => [key, localStorage.getItem(key)])),
      revisions: localStorage.getItem(revisionKey), preferences: localStorage.getItem(preferenceKey),
      valuationOpen: !!document.querySelector('.valuation-workspace')
    }), { draftPrefix, revisionKey, preferenceKey });
    assert.deepEqual(restored, original, 'Every prior draft, saved revision, preference and company view is restored exactly');
  }
  checks.push('All temporary range fixtures are removed and original drafts, crisis settings, deliberate blanks, revisions and selected company are restored byte for byte');
  return { checks };
}

async function standalone() {
  const { chromium } = await import('playwright');
  const project = resolve(dirname(fileURLToPath(import.meta.url)), '..'), base = process.env.ATLAS_URL ?? 'http://127.0.0.1:1420';
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const external = [], errors = [], context = await browser.newContext({ viewport: { width: 1500, height: 960 } });
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
    const result = await valuationRangeFlows(page, project);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await writeFile(resolve(project, 'test-results/valuation-range-browser-report.json'), JSON.stringify({ status: 'passed', ...result, external, errors }, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } catch (error) {
    await page.screenshot({ path: resolve(project, 'test-results/valuation-range-failure.png'), fullPage: true }).catch(() => {});
    throw error;
  } finally { await browser.close(); }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await standalone();
