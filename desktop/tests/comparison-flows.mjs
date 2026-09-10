import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function annualRows(page, pack, ids) {
  return page.evaluate(async ({ pack, ids }) => {
    if (window.__TAURI_INTERNALS__) return window.__TAURI_INTERNALS__.invoke('financial_annual', { pack, ids });
    const response = await fetch(`/api/financials/annual?${new URLSearchParams({ pack, ids: ids.join(',') })}`);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }, { pack, ids });
}
export async function restoreComparison(page, title = 'Skog – jämförelse med anteckningar') {
  await page.getByLabel('Observatory', { exact: true }).selectOption('sectors');
  await page.getByLabel('Branch comparison', { exact: true }).click();
  await page.getByLabel('Comparison branch', { exact: true }).selectOption('21');
  await page.locator('[data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
  await page.locator('.saved-comparisons button').filter({ hasText: title }).click();
  assert.equal(await page.getByLabel('Comparison research notes').inputValue(), 'Årsredovisningar: jämför skogsmark, förvärv och kassaflöden.');
  assert.equal(await page.getByLabel('Branch comparison metric').inputValue(), 'free_cash_flow');
  assert.equal(await page.getByLabel('Comparison reporting currency').inputValue(), 'SEK');
  assert.equal(await page.getByLabel('Branch bubble size').inputValue(), 'total_assets');
  assert.equal(await page.getByLabel('Comparison selected year').inputValue(), '2024');
  assert.equal(await page.locator('[data-comparison-focus]').getAttribute('data-comparison-focus'), '102');
}
export async function comparisonFlows(page, project) {
  const envelope = JSON.parse(await readFile(resolve(project, 'public/data/research.atlas.json'), 'utf8'));
  const taxonomy = JSON.parse(envelope.taxonomy.content), catalogue = taxonomy.catalogue;
  const catalog = JSON.parse(await readFile(resolve(project, 'financial-data/catalog.json'), 'utf8'));
  const pack = catalog.packs[0].id;
  const branchIds = Object.values(catalogue.listings).filter(c => (taxonomy.classifications[c.id]?.branch_id ?? c.branch_id) === '21').map(c => c.id);
  await page.getByLabel('Observatory', { exact: true }).selectOption('sectors');
  await page.getByLabel('Branch comparison', { exact: true }).click();
  await page.getByLabel('Comparison branch', { exact: true }).selectOption('21');
  const started = performance.now();
  await page.locator('[data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
  const forestryMs = Math.round(performance.now() - started);
  assert.equal(await page.getByLabel('Branch bubble size').inputValue(), 'equal');
  assert.match(await page.locator('.bubble-history').innerText(), /do not represent market cap/);
  assert.ok(Number(await page.locator('[data-bubble-points]').getAttribute('data-bubble-points')) > 0);
  const raw = [];
  for (let i = 0; i < branchIds.length; i += 32) raw.push(...(await annualRows(page, pack, branchIds.slice(i, i + 32))).companies);
  assert.ok(raw.every(c => !('quarterly' in c)));
  await assert.rejects(annualRows(page, pack, ['102', '102']));
  // Independently recover the nominal SEK FCF from compact source rows (column 12).
  await page.getByLabel('Branch comparison metric').selectOption('free_cash_flow');
  await page.getByLabel('Comparison reporting currency').selectOption('SEK');
  await page.getByLabel('Branch bubble size').selectOption('total_assets');
  await page.getByLabel('Comparison selected year').fill('2024');
  const comparable = raw.flatMap(c => c.annual.filter(r => r[0] === 2024 && r[5] === 'SEK' && r[6] > 0 && r[12] !== null && (Date.parse(r[3]) - Date.parse(r[2])) / 86400000 + 1 >= 330 && (Date.parse(r[3]) - Date.parse(r[2])) / 86400000 + 1 <= 400).map(r => r[12] / r[6])).sort((a, b) => a - b);
  const middle = (comparable[(comparable.length - 1) >> 1] + comparable[comparable.length >> 1]) / 2;
  assert.equal(Number(await page.locator('[data-benchmark-n]').getAttribute('data-benchmark-n')), comparable.length);
  assert.match(await page.locator('.comparison-readout').innerText(), new RegExp(new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(middle).replaceAll('.', '\\.')));
  assert.equal(Number(await page.locator('[data-comparison-listing="102"] [data-comparison-value]').getAttribute('data-comparison-value')), raw.find(c => c.id === '102').annual.find(r => r[0] === 2024)[12]);
  await page.getByLabel('Focus Holmen', { exact: true }).click();
  const focus = await page.locator('[data-comparison-focus]').getAttribute('data-comparison-focus');
  assert.equal(focus, '102');
  // Find an actual SCA bubble by its legend colour, then click the rendered canvas.
  // This tests the chart's event wiring without a test-only application API.
  const hit = await page.evaluate(() => {
    const dom = document.querySelector('[aria-label="Branch bubble history"]');
    const rgb = getComputedStyle(document.querySelector('[aria-label="Focus SCA"] i')).backgroundColor.match(/\d+/g).map(Number);
    for (const [index, canvas] of [...dom.querySelectorAll('canvas')].entries()) {
      const context = canvas.getContext('2d'), pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      for (let y = 30; y < canvas.height - 30; y++) for (let x = 60; x < canvas.width - 30; x++) {
        const at = (y * canvas.width + x) * 4;
        if (rgb.every((v, i) => Math.abs(v - pixels[at + i]) <= 1) && pixels[at + 3] === 255) return { index, x: x * canvas.clientWidth / canvas.width, y: y * canvas.clientHeight / canvas.height };
      }
    }
    return null;
  });
  assert.ok(hit, 'A coloured SCA bubble must be visible');
  await page.getByLabel('Branch bubble history', { exact: true }).locator('canvas').nth(hit.index).click({ position: { x: hit.x, y: hit.y } });
  assert.equal(await page.locator('[data-comparison-focus]').getAttribute('data-comparison-focus'), '197');
  assert.equal(await page.locator('[data-comparison-listing="197"]').getAttribute('class'), 'focused');
  await page.getByLabel('Focus Holmen', { exact: true }).click();
  const title = 'Skog – jämförelse med anteckningar';
  await page.getByLabel('Saved comparison name').fill(title);
  await page.getByLabel('Comparison research notes').fill('Årsredovisningar: jämför skogsmark, förvärv och kassaflöden.');
  await page.getByRole('button', { name: 'Save new comparison', exact: true }).click();
  await page.locator('.comparison-save-status').filter({ hasText: 'Saved' }).waitFor();
  await page.screenshot({ path: resolve(project, 'test-results/branch-comparison.png') });
  const savedId = await page.locator('[data-saved-comparison]').first().getAttribute('data-saved-comparison');
  await page.evaluate(() => {
    const key = 'macro-atlas-branch-comparisons-v1', data = JSON.parse(localStorage.getItem(key));
    data.items.push({ ...data.items[0], id: 'other-version', title: 'Different data version', financial: 'd'.repeat(64) });
    localStorage.setItem(key, JSON.stringify(data));
  });
  // Filters apply to the cohort without silently removing a chosen listing.
  await page.getByLabel('Comparison listing country').selectOption('FI');
  assert.equal(await page.locator('[data-comparison-listing="102"] [data-comparison-value]').getAttribute('data-comparison-value'), '');
  assert.match(await page.locator('[data-comparison-listing="102"]').innerText(), /Outside current directory filters/);
  await page.reload();
  await restoreComparison(page);
  assert.equal(await page.locator(`[data-saved-comparison="${savedId}"]`).count(), 1);
  assert.equal(await page.locator('.saved-comparisons button').filter({ hasText: 'Different data version' }).isDisabled(), true);
  await page.getByLabel('Play comparison years').click();
  await page.waitForFunction(() => document.querySelector('[data-comparison-year]')?.dataset.comparisonYear === '2025');
  await page.getByLabel('Comparison selected year').fill('2024');
  // Largest branch: load only bounded annual batches, then cancel a branch switch.
  const largeStarted = performance.now();
  await page.getByLabel('Comparison branch').selectOption('16');
  await page.locator('[data-comparison-branch="16"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
  const miningMs = Math.round(performance.now() - largeStarted);
  assert.equal(await page.locator('.saved-comparisons button').filter({ hasText: title }).count(), 0);
  await page.getByLabel('Comparison branch').selectOption('90');
  await page.getByLabel('Comparison branch').selectOption('21');
  await page.locator('[data-comparison-branch="21"][data-comparison-ready="true"]').waitFor({ timeout: 60000 });
  await restoreComparison(page);
  const originalViewport = page.viewportSize() ?? await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  await page.setViewportSize({ width: 1100, height: 760 });
  await page.screenshot({ path: resolve(project, 'test-results/branch-comparison-compact.png') });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.equal(await page.locator('.branch-explorer').evaluate(el => el.scrollWidth > el.clientWidth), false);
  await page.setViewportSize(originalViewport);
  await page.getByLabel('Open financials for Holmen', { exact: true }).click();
  await page.locator('[data-company="102"][data-business-ready="true"] [data-financial-history="102"]').waitFor();
  return { savedId, forestryMs, miningMs, checks: ['Branch bubbles and whole-cohort medians reconcile to annual source rows and selected reporting currency', 'Missing data and out-of-scope selections remain explicit; market cap, ROIC and CAPEX are not substituted', 'Saved comparison preserves Unicode notes, fiscal years, metrics, filters, focus and exact data version across reload', 'Linked year playback, branch cancellation and opening the correct company financial history work offline', 'Largest branch loads annual histories in bounded batches without rendering the full company directory'] };
}
