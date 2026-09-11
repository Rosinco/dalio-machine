import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

async function financialRead(page, operation, args) {
  return page.evaluate(async ({ operation, args }) => {
    if (window.__TAURI_INTERNALS__) return window.__TAURI_INTERNALS__.invoke(`financial_${operation}`, args);
    const response = await fetch(`/api/financials/${operation}?${new URLSearchParams(args)}`);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  }, { operation, args });
}
export async function financialFlows(page, project, { native = false, archive, diagnostic } = {}) {
  const envelope = JSON.parse(await readFile(resolve(project, 'public/data/research.atlas.json'), 'utf8'));
  const taxonomy = JSON.parse(envelope.taxonomy.content);
  const index = await financialRead(page, 'index', { taxonomy: envelope.taxonomy.sha256 });
  assert.equal(index.summary.with_reports, 18943); assert.equal(index.summary.listings - index.summary.with_reports, 197);
  assert.equal(index.summary.annual, 268884); assert.equal(index.summary.quarterly, 603720); assert.equal(index.summary.withheld, 1174);
  assert.equal(index.version, 2);
  assert.deepEqual(index.market.summary, { count: 268884, local: 159486, sek: 159486, flagged: 38648, listings: 19140, with_local: 16895, with_sek: 16895 });
  assert.equal(await financialRead(page, 'index', { taxonomy: 'b'.repeat(64) }), null);
  const search = page.getByLabel('Search companies or countries', { exact: true });
  const timings = [];
  const choose = async id => {
    const start = performance.now();
    await search.fill(taxonomy.catalogue.listings[id].name);
    await page.locator(`[data-search-listing="${id}"]`).click();
    await page.locator(`[data-company="${id}"][data-business-ready="true"] [data-financial-history="${id}"]`).waitFor();
    timings.push(Math.round(performance.now() - start));
  };
  await choose('20');
  assert.match(await page.locator('[data-coverage="annual"]').innerText(), /21[\s\S]*2005–2025/);
  assert.equal(await page.locator('[data-financial="revenues"] strong').innerText(), '3,446');
  assert.match(await page.locator('.financial-statements').innerText(), /FY 2025[\s\S]*SEK million/);
  await page.getByRole('button', { name: 'Balance sheet', exact: true }).click();
  assert.equal(await page.locator('[data-statement-field]').count(), 12);
  assert.ok(await page.locator('[data-statement-field="total_assets"] td').innerText() !== '—');
  await page.getByLabel('Statement period', { exact: true }).selectOption('2005-5');
  assert.match(await page.locator('.financial-statements').innerText(), /saved 2025-06-21/);
  await page.screenshot({ path: resolve(project, 'test-results/company-balance-sheet.png') });
  await page.getByRole('button', { name: 'Cash flow', exact: true }).click();
  assert.equal(await page.locator('[data-statement-field]').count(), 5);
  assert.match(await page.locator('.financial-statements').innerText(), /not owner earnings/);
  await page.getByRole('button', { name: 'Quarterly', exact: true }).click();
  assert.match(await page.getByLabel('Statement period', { exact: true }).innerText(), /Q2 2026/);
  // This foreign listing's stored amounts are scaled into trading currency.
  await choose('55268');
  const raw = await financialRead(page, 'company', { pack: index.id, id: '55268' });
  const latest = raw.annual.at(-1);
  assert.equal(latest[5], 'KZT'); assert.equal(latest[6], 0.00196885);
  assert.equal(await page.locator('[data-financial="revenues"] strong').innerText(), new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(latest[8] / latest[6]));
  assert.match(await page.locator('[data-financial="revenues"]').innerText(), /KZT million/);
  await choose('1429');
  assert.match(await page.locator('[data-coverage="quarterly"]').innerText(), /1 missing quarters/);
  assert.match(await page.locator('.financial-quality-note').innerText(), /1 source row withheld/);
  await page.locator('.withheld-reports summary').click();
  assert.match(await page.locator('.withheld-reports').innerText(), /2699-03-22[\s\S]*after snapshot/);
  await page.getByRole('button', { name: 'Quarterly', exact: true }).click();
  assert.equal(await page.getByLabel('Statement period').locator('option[value="2026-3"]').count(), 0);
  await choose('167');
  assert.equal(await page.locator('[data-financial]').count(), 0);
  assert.equal(await page.getByRole('heading', { name: 'No usable saved reports', exact: true }).count(), 1);
  assert.match(await page.locator('[data-coverage="annual"]').innerText(), /No saved periods/);
  const mixed = Object.keys(index.companies).find(id => index.companies[id].currencies.length > 1);
  assert.ok(mixed); await choose(mixed);
  const currencies = index.companies[mixed].currencies;
  await page.getByLabel('Chart reporting currency', { exact: true }).selectOption(currencies[0]);
  assert.match(await page.locator('.company-financial-history').innerText(), /one reporting currency at a time/);
  await choose('102');
  const market = page.locator('[data-market-history="102"]');
  await market.locator('.market-table summary').click();
  const holmen2024 = market.locator('[data-market-year="2024"]');
  assert.equal(Number(await holmen2024.locator('[data-market-value]').getAttribute('data-market-value')), 66283.6272);
  assert.match(await holmen2024.innerText(), /420\.4 SEK[\s\S]*2025-01-31[\s\S]*157\.668/);
  await market.getByRole('heading', { name: 'Derived market cap', exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: resolve(project, 'test-results/company-market-history.png') });
  for (const [id, currency, expected, method] of [['674', 'EUR', 33788.734703999995, 'Observed direct rate'], ['21847', 'PLN', 848.3365660539805, 'Observed USD cross-rate']]) {
    await choose(id);
    const company = await financialRead(page, 'company', { pack: index.id, id });
    const row = company.market.at(-1), view = page.locator(`[data-market-history="${id}"]`);
    assert.equal(row.sek, expected); assert.equal(row.currency, currency);
    await view.locator('.market-table summary').click();
    assert.match(await view.locator(`[data-market-year="${row.year}"]`).innerText(), new RegExp(method));
    await page.getByLabel('Market-cap currency', { exact: true }).selectOption(currency);
    assert.equal(Number(await view.locator(`[data-market-year="${row.year}"] [data-market-value]`).getAttribute('data-market-value')), row.local);
    await page.getByLabel('Market-cap currency', { exact: true }).selectOption('SEK');
    assert.equal(Number(await view.locator(`[data-market-year="${row.year}"] [data-market-value]`).getAttribute('data-market-value')), expected);
  }
  await choose('608');
  await page.locator('.market-table summary').click();
  assert.equal(await page.locator('[data-market-year="2024"] [data-market-value]').getAttribute('data-market-value'), '');
  assert.match(await page.locator('[data-market-year="2024"]').innerText(), /basis requires review/);
  await choose('167');
  assert.equal(await page.locator('[data-market-latest]').count(), 0);
  assert.match(await page.locator('[data-market-history="167"]').innerText(), /No saved annual reports/);
  await choose('102');
  await page.getByLabel('Open data library').click();
  const library = page.getByLabel('Company financial history library', { exact: true });
  await page.locator(`[data-library-financial-pack="${index.id}"]`).waitFor();
  assert.match(await library.innerText(), /18,943 listings with reports[\s\S]*197 without reports[\s\S]*1,174 source rows withheld/);
  assert.match(await page.locator('[data-market-library]').innerText(), /16,895[\s\S]*159,486/);
  let exportedPath;
  if (native) {
    diagnostic?.({ stage: 'financial_export_started', pack: index.id, bytes: index.bytes, taxonomy: index.taxonomy_sha256 });
    await page.getByRole('button', { name: 'Save financial history pack', exact: true }).click();
    await library.locator('.library-message').waitFor();
    exportedPath = (await library.locator('.library-message').innerText()).replace(/^Saved /, '');
    assert.equal((await stat(exportedPath)).size, index.bytes);
    diagnostic?.({ stage: 'financial_export_completed', bytes: index.bytes });
    const input = page.getByLabel('Financial history file', { exact: true });
    await input.setInputFiles({ name: 'invalid.sqlite', mimeType: 'application/octet-stream', buffer: Buffer.from('junk') });
    await library.getByRole('alert').waitFor();
    assert.equal(await library.getAttribute('data-library-financial-pack'), index.id);
    diagnostic?.({ stage: 'invalid_financial_import_rejected' });
    const prefix = 'ATLAS_FINANCIAL_DIAGNOSTIC ';
    const onConsole = message => {
      if (!message.text().startsWith(prefix)) return;
      try { diagnostic?.(JSON.parse(message.text().slice(prefix.length))); } catch { /* Preserve importer behavior if logging fails. */ }
    };
    if (diagnostic) {
      page.on('console', onConsole);
      await page.evaluate(prefix => {
        const library = document.querySelector('[aria-label="Company financial history library"]');
        let previous = -1, previousStage = '';
        const log = (stage, detail = {}) => {
          const memory = performance.memory;
          const heap = memory && (detail.percent === undefined || detail.percent % 10 === 0) ? { usedJSHeapSize: memory.usedJSHeapSize, totalJSHeapSize: memory.totalJSHeapSize, jsHeapSizeLimit: memory.jsHeapSizeLimit } : undefined;
          console.debug(prefix + JSON.stringify({ stage, observed_at: new Date().toISOString(), ...detail, heap }));
        };
        // Observe only rendered progress; Tauri's fixed invoke descriptor and
        // importer remain untouched. The UI rounds percentages, so 100% is a
        // displayed validation phase, not proof that the last chunk was copied.
        const observe = () => {
          const progress = library.querySelector('progress[aria-label="Financial import progress"]');
          if (progress && progress.value !== previous) {
            previous = progress.value;
            log('financial_ui_copy_progress', { percent: previous });
          }
          const message = library.querySelector('.library-message')?.textContent ?? '';
          const error = library.querySelector('[role="alert"]')?.textContent ?? '';
          const stage = message.includes('Financial history saved ·') ? 'financial_ui_import_completed' : error ? 'financial_ui_import_error' : previous >= 100 ? 'financial_ui_validation_visible' : progress ? 'financial_ui_copy_visible' : '';
          if (stage && stage !== previousStage) {
            previousStage = stage;
            log(stage, error ? { message: error } : {});
          }
        };
        const selected = event => {
          if (event.target.matches('input[aria-label="Financial history file"]')) log('financial_dom_file_selected', { bytes: event.target.files?.[0]?.size ?? null });
        };
        const observer = new MutationObserver(observe);
        observer.observe(library, { subtree: true, childList: true, attributes: true, attributeFilter: ['value'] });
        library.addEventListener('change', selected);
        window.__atlasRestoreFinancialDiagnostic = () => { observer.disconnect(); library.removeEventListener('change', selected); delete window.__atlasRestoreFinancialDiagnostic; };
      }, prefix);
    }
    // Native runner declares isLocal:true because Node and WebView share the
    // Windows filesystem. Select the complete file by path using Playwright's
    // public API; no transferred buffer or second manual CDP session is needed.
    try {
      diagnostic?.({ stage: 'financial_native_file_selection_started', bytes: index.bytes, selectionMethod: 'playwright_local_path' });
      await input.setInputFiles(exportedPath);
      diagnostic?.({ stage: 'financial_native_file_selection_completed' });
      await library.getByText(/Financial history saved ·/).waitFor({ timeout: 120000 });
      diagnostic?.({ stage: 'financial_import_success_message' });
    } finally {
      if (diagnostic) {
        await page.evaluate(() => window.__atlasRestoreFinancialDiagnostic?.()).catch(() => {});
        page.off('console', onConsole);
      }
    }
    assert.equal(createHash('sha256').update(await readFile(resolve(archive, 'financial-packs', `${index.id}.sqlite`))).digest('hex'), index.id);
  } else {
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save financial history pack', exact: true }).click();
    exportedPath = resolve(project, 'test-results/exported-financials.sqlite');
    await (await download).saveAs(exportedPath);
  }
  assert.equal(createHash('sha256').update(await readFile(exportedPath)).digest('hex'), index.id);
  await page.keyboard.press('Escape');
  await page.reload();
  await page.locator(`[data-financial-history="102"][data-financial-pack="${index.id}"]`).waitFor();
  return { index, exportedPath, timings, checks: ['159,486 dated valuations cover 16,895 listings; all 19,140 retain explicit market coverage', 'Holmen prices and shares reconcile; EUR direct and PLN cross-rate valuations switch correctly between local currency and SEK', 'Review flags leave valuations blank and no-report listings clear prior market history', '872,604 reports reconcile to coverage for 18,943 listings; 197 report gaps stay visible', 'Income statement, balance sheet and cash flow use selected fiscal period and dated sources', 'Quarterly views exclude annual-only return proxy and invalid future-dated source rows', 'Foreign listing monetary amounts recover KZT reporting currency using saved FX', 'Empty histories clear previous company figures; multiple reporting currencies are selectable', 'Financial packs attach only to their exact company directory', native ? 'Native financial export/import is byte-identical, rejects invalid files, and survives reload' : 'Offline browser financial export is byte-identical to the indexed local pack'] };
}
