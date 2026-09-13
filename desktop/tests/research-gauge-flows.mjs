import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';

const draftPrefix = 'macro-atlas-valuation-draft-v1:';
const revisionKey = 'macro-atlas-valuations-v1';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const near = (actual, expected, label) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= Math.max(1e-8, Math.abs(expected) * 1e-10), `${label}: ${actual} != ${expected}`);

async function protectedStorage(page) {
  return page.evaluate(({ draftPrefix, revisionKey }) => Object.fromEntries(Object.keys(localStorage)
    .filter(key => key.startsWith(draftPrefix) || key === revisionKey).sort()
    .map(key => [key, localStorage.getItem(key)])), { draftPrefix, revisionKey });
}

// Observe attempted writes, including a same-value write which a before/after
// snapshot alone would miss. The wrapper changes no application return values.
async function watchReadOnly(page) {
  await page.evaluate(({ draftPrefix, revisionKey }) => {
    if (window.__researchGaugeAudit) return;
    const audit = { writes: [], invokes: [], restore: [] };
    for (const method of ['setItem', 'removeItem', 'clear']) {
      const original = Storage.prototype[method];
      Storage.prototype[method] = function (...args) {
        if (this === localStorage && (method === 'clear' || String(args[0]).startsWith(draftPrefix) || args[0] === revisionKey)) audit.writes.push({ method, key: args[0] ?? null });
        return original.apply(this, args);
      };
      audit.restore.push(() => { Storage.prototype[method] = original; });
    }
    if (window.__TAURI_INTERNALS__?.invoke) {
      // Tauri defines invoke as non-writable. Observe its real custom-protocol
      // fetch instead; assigning invoke would silently leave a vacuous audit.
      const original = window.fetch;
      const watched = function (input, ...rest) {
        const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
        if (url.hostname === 'ipc.localhost') audit.invokes.push({ command: decodeURIComponent(url.pathname.slice(1)) });
        return original.call(this, input, ...rest);
      };
      window.fetch = watched;
      if (window.fetch !== watched) throw new Error('Native command observation could not be installed');
      audit.restore.push(() => { window.fetch = original; });
    }
    window.__researchGaugeAudit = audit;
  }, { draftPrefix, revisionKey });
}

async function assertReadOnly(page, expected) {
  assert.deepEqual(await protectedStorage(page), expected, 'Browsing research retains every saved valuation draft and revision byte');
  const audit = await page.evaluate(() => ({ writes: window.__researchGaugeAudit?.writes ?? [], invokes: window.__researchGaugeAudit?.invokes ?? [] }));
  assert.deepEqual(audit.writes, [], 'Browsing research must not attempt to save, remove, clear, migrate or back up valuations');
  assert.deepEqual(audit.invokes.filter(item => item.command === 'financial_annual'), [], 'Research screening must not fetch annual histories in batches');
  assert.deepEqual(audit.invokes.filter(item => /^(financial_(begin|append|finish|cancel|export)|research_(import|export|save|delete))$/.test(item.command)), [], 'Research browsing must not invoke native archive or financial mutations');
  assert.equal(await page.locator('.valuation-workspace').count(), 0, 'A read-only card must not mount the editable valuation workspace');
  return audit;
}

async function openCompany(page, name, id, { wait = true } = {}) {
  await page.getByLabel('Search companies or countries', { exact: true }).fill(name);
  await page.locator(`[data-search-listing="${id}"]`).click();
  if (wait) await page.locator(`[data-company="${id}"][data-business-ready="true"]`).waitFor();
}

async function openScreen(page) {
  await page.getByLabel('Universe research screen', { exact: true }).click();
  await page.locator('[data-research-gauge-ready="true"]').waitFor();
}

async function rowIds(page) {
  return page.locator('[data-research-listing]').evaluateAll(rows => rows.map(row => row.getAttribute('data-research-listing')));
}

async function matches(page) {
  return Number(await page.locator('[data-research-gauge-ready="true"]').getAttribute('data-research-gauge-matches'));
}

async function fixture(project) {
  const manifest = JSON.parse(await readFile(resolve(project, 'src/data/research-gauge-manifest.json'), 'utf8'));
  const compressed = await readFile(resolve(project, 'public', manifest.artifact.path.replace(/^\//, '')));
  assert.equal(compressed.length, manifest.artifact.bytes);
  assert.equal(hash(compressed), manifest.artifact.sha256);
  const expanded = gunzipSync(compressed);
  assert.equal(expanded.length, manifest.artifact.uncompressedBytes);
  assert.equal(hash(expanded), manifest.artifact.uncompressedSha256);
  const artifact = JSON.parse(expanded.toString('utf8'));
  const envelope = JSON.parse(await readFile(resolve(project, 'public/data/research.atlas.json'), 'utf8'));
  assert.equal(hash(envelope.taxonomy.content), manifest.taxonomySha256);
  const taxonomy = JSON.parse(envelope.taxonomy.content), listings = taxonomy.catalogue.listings;
  const rows = artifact.rows, byId = Object.fromEntries(rows.map(row => [row.id, row]));
  assert.equal(rows.length, 19140);
  assert.equal(Object.keys(byId).length, rows.length, 'No duplicate listing can displace a missing listing');
  assert.deepEqual(Object.keys(byId).sort(), Object.keys(listings).sort(), 'Gauge retains the exact downloaded directory, including missing/manual listings');
  assert.equal(rows.filter(row => row.presence === 'latest').length, 17593);
  assert.equal(rows.filter(row => row.presence === 'older').length, 1547);
  const lanes = { positivePriced: 0, positiveUnpriced: 0, nonpositive: 0, financial: 0, otherMissing: 0 };
  let reverseChecks = 0;
  for (const row of rows) {
    const classification = taxonomy.classifications[row.id], v = row.valuation;
    const route = !classification.sector_id || !classification.branch_id ? 'unclassified' : ['75', '76'].includes(classification.branch_id) ? 'property' : classification.sector_id === '1' ? 'financial' : 'operating';
    assert.equal(row.route, route, `${row.id} business route`);
    assert.equal(row.branchId, classification.branch_id);
    assert.equal(row.presence, listings[row.id].source_as_of === taxonomy.catalogue.as_of ? 'latest' : 'older');
    assert.equal(row.annual.periods.length, row.annual.cash.values.length);
    assert.ok(row.annual.periods.length <= 5);
    for (const key of ['cash', 'operatingCash', 'ebit', 'revenue', 'margins']) {
      const series = row.annual[key], values = series.values.filter(value => value !== null);
      assert.equal(series.count, values.length, `${row.id} ${key} actual denominator`);
      assert.equal(series.positive, values.filter(value => value > 0).length);
      assert.equal(series.negative, values.filter(value => value < 0).length);
      assert.equal(series.zero, values.filter(value => value === 0).length);
    }
    if (v.value !== null) {
      near(v.value, v.cashPV + v.terminalPV, `${row.id} DCF cash plus terminal once`);
      if (v.value > 0) {
        near(v.ceiling, 0.7 * v.value, `${row.id} declared 30% starter policy`);
        near(v.terminalShare, v.terminalPV / v.value, `${row.id} terminal share`);
        if (v.candidateEquity !== null) {
          lanes.positivePriced++;
          near(v.reverseCashFactor, v.candidateEquity / v.value, `${row.id} required whole-cash level`);
          near(v.reverseCashFactor30, v.candidateEquity / (0.7 * v.value), `${row.id} required whole-cash level at 30% margin`);
          reverseChecks += 2;
        } else {
          lanes.positiveUnpriced++; assert.equal(v.reverseCashFactor, null); assert.equal(v.reverseCashFactor30, null);
        }
      } else {
        lanes.nonpositive++; assert.equal(v.ceiling, null); assert.equal(v.reverseCashFactor, null); assert.equal(v.reverseCashFactor30, null);
      }
    } else {
      lanes[row.route === 'financial' ? 'financial' : 'otherMissing']++;
      assert.equal(v.ceiling, null); assert.equal(v.reverseCashFactor, null); assert.equal(v.reverseCashFactor30, null);
    }
    if (row.route === 'financial') assert.equal(v.value, null, `${row.id} needs a reviewed financial-capital model`);
    if (v.lowValue !== null && v.candidateEquity !== null) near(v.lowNPV, v.lowValue - v.candidateEquity, `${row.id} Low NPV subtracts saved price once`);
    else assert.equal(v.lowNPV, null);
  }
  assert.deepEqual(lanes, { positivePriced: 6941, positiveUnpriced: 489, nonpositive: 6749, financial: 2731, otherMissing: 2230 });
  assert.equal(reverseChecks, 13882);
  return { manifest, artifact, taxonomy, rows, byId, lanes, reverseChecks, compressed };
}

async function seedAuthoredWork(page, project, manifest) {
  const original = await protectedStorage(page);
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const draft = JSON.parse(await readFile(resolve(project, 'tests/fixtures/stora-empirical-cash-v3-pre-terminal.json'), 'utf8'));
  draft.title = 'TEST FIXTURE — retained authored assumptions and deliberate blank';
  delete draft.starterOrigin; delete draft.researchOrigin;
  draft.researchAutofillDisabled = true;
  draft.scenarios.mid.cashFlows[1] = null;
  draft.purchaseRange = { marginOfSafetyPercent: 40, referenceScenario: 'high', unit: 'equity', candidateEquity: 2468 };
  const saved = { format: 'macro-atlas-valuation', version: 1, id: 'research-gauge-preserved-draft', created: '2026-09-13T00:00:00.000Z', company: '102', release: catalog.default_id, financial: manifest.financialPackId, taxonomy: manifest.taxonomySha256, draft };
  const key = `${draftPrefix}${saved.company}:${saved.release}:${saved.financial}:${saved.taxonomy}`;
  await page.evaluate(({ key, saved, revisionKey }) => {
    localStorage.setItem(key, JSON.stringify(saved));
    localStorage.setItem(revisionKey, JSON.stringify({ version: 1, items: [{ ...saved, id: 'research-gauge-preserved-revision' }] }));
  }, { key, saved, revisionKey });
  return { original, seeded: await protectedStorage(page) };
}

async function restoreAuthoredWork(page, original) {
  await page.evaluate(({ original, draftPrefix, revisionKey }) => {
    window.__researchGaugeAudit?.restore.forEach(restore => restore());
    delete window.__researchGaugeAudit;
    for (const key of Object.keys(localStorage)) if (key.startsWith(draftPrefix) || key === revisionKey) localStorage.removeItem(key);
    for (const [key, value] of Object.entries(original)) localStorage.setItem(key, value);
  }, { original, draftPrefix, revisionKey });
}

async function fitViewport(page, width, label) {
  await page.setViewportSize({ width, height: width < 500 ? 844 : 960 });
  await page.waitForFunction(() => document.documentElement.scrollWidth <= window.innerWidth + 1);
  assert.ok(await page.locator('[data-research-gauge-ready="true"], [data-research-card]').first().isVisible(), `${label} is usable at ${width}px`);
}

async function verifyEvidence(locator, row) {
  const v = row.valuation;
  for (const [attribute, expected] of [['data-research-value', v.value], ['data-research-price', v.candidateEquity], ['data-research-reverse', v.reverseCashFactor], ['data-research-reverse30', v.reverseCashFactor30]]) {
    const actual = await locator.locator(`[${attribute}]`).getAttribute(attribute);
    if (expected === null) assert.equal(actual, '', `${row.id} ${attribute} must remain missing`);
    else near(Number(actual), expected, `${row.id} ${attribute} uses the standard source snapshot`);
  }
  const priceDate = await locator.locator('[data-research-price-date]').innerText();
  assert.ok(priceDate.includes(v.priceDate ?? 'Unavailable'));
  if (v.priceAgeDays !== null) assert.ok(priceDate.includes(v.priceAgeDays.toLocaleString('en-US')));
  const text = await locator.innerText();
  assert.match(text, /reviewed study or your edited working valuation may differ/i);
  assert.match(text, /whole-equity/i);
  assert.match(text, /not expected growth/i);
  assert.match(text, /not.*probability/i);
  if (v.hasSignedCash) assert.match(text, /scal.*negative cash/i);
}

async function verifyCard(page, row, manifest) {
  const card = page.locator(`[data-research-card="${row.id}"][data-research-card-ready="true"]`);
  await card.waitFor();
  assert.equal(await card.getAttribute('data-research-readiness'), row.readiness);
  assert.equal(await card.getAttribute('data-research-route'), row.route);
  const details = card.locator('details').first();
  if (await details.getAttribute('open') === null) await details.locator('summary').first().click();
  const evidence = card.locator(`[data-research-evidence="${row.id}"]`);
  await verifyEvidence(evidence, row);
  const source = evidence.locator('details.research-gauge-method');
  if (await source.getAttribute('open') === null) await source.locator('summary').click();
  const text = await source.innerText();
  assert.ok(text.includes(row.sourceCompanySha256));
  assert.ok(text.includes(manifest.financialPackId));
  assert.ok(text.includes(manifest.taxonomySha256));
  if (row.presence === 'older') assert.match(text, /older download.*does not establish listing status/i);
  if (row.annual.latest) assert.ok(text.includes(row.annual.latest.end));
  return card;
}

async function switchRelease(page, date) {
  await page.getByLabel('Open data library', { exact: true }).click();
  await page.getByLabel(`Use release ${date}`, { exact: true }).click();
  await page.keyboard.press('Escape');
}

async function failClosedFlows(page, project, data, expectedStorage, native) {
  const { manifest, compressed } = data;
  const asset = `**${manifest.artifact.path.startsWith('/') ? manifest.artifact.path : '/' + manifest.artifact.path}`;
  const damaged = Buffer.from(compressed); damaged[damaged.length - 1] ^= 1;
  await page.route(asset, route => route.fulfill({ status: 200, contentType: 'application/gzip', body: damaged }));
  try {
    await page.reload();
    await page.locator('[data-research-card] [role="alert"]').waitFor();
    assert.equal(await page.locator('[data-research-card-ready="true"]').count(), 0);
    await page.getByLabel('Universe research screen', { exact: true }).click();
    await page.locator('[data-research-gauge-ready] [role="alert"]').waitFor();
    assert.equal(await page.locator('[data-research-listing]').count(), 0);
    assert.deepEqual(await protectedStorage(page), expectedStorage);
  } finally {
    await page.unroute(asset);
    await page.reload();
  }
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
  await watchReadOnly(page);
  const catalog = JSON.parse(await readFile(resolve(project, 'public/data/catalog.json'), 'utf8'));
  const current = catalog.releases.find(release => release.id === catalog.default_id);
  const older = catalog.releases.find(release => release.id !== catalog.default_id);
  assert.ok(older);
  const badPack = 'f'.repeat(64);
  if (native) await page.evaluate(badPack => {
    const original = window.fetch, probe = { responses: 0 };
    window.__researchGaugeMismatchProbe = probe;
    window.__researchGaugeIndexRestore = () => { window.fetch = original; };
    const intercepted = async function (input, ...rest) {
      const response = await original.call(this, input, ...rest);
      const url = new URL(typeof input === 'string' || input instanceof URL ? String(input) : input.url, location.href);
      if (url.hostname !== 'ipc.localhost' || url.pathname !== '/financial_index') return response;
      const result = await response.clone().json();
      if (result?.format !== 'macro-atlas-financials') return response;
      probe.responses++;
      return new Response(JSON.stringify({ ...result, id: badPack }), { status: response.status, statusText: response.statusText, headers: response.headers });
    };
    window.fetch = intercepted;
    if (window.fetch !== intercepted) throw new Error('Native financial-index response injection could not be installed');
  }, badPack);
  else await page.route('**/api/financials/index?*', async route => {
    const response = await route.fetch(), result = await response.json();
    return route.fulfill({ response, json: result ? { ...result, id: badPack } : result });
  });
  try {
    if (native) {
      // Leaving Companies disables the index resource. Await the real Macro
      // view before returning, so a new native index read is deterministic.
      await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
      await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
      await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
      await page.waitForFunction(() => window.__researchGaugeMismatchProbe?.responses > 0);
    } else {
      await switchRelease(page, older.as_of);
      await switchRelease(page, current.as_of);
    }
    await page.locator('[data-research-card] [role="alert"]').waitFor();
    assert.equal(await page.locator('[data-research-card-ready="true"]').count(), 0);
    await page.getByLabel('Universe research screen', { exact: true }).click();
    await page.locator('[data-research-gauge-ready] [role="alert"]').waitFor();
    assert.equal(await page.locator('[data-research-listing]').count(), 0);
    await assertReadOnly(page, expectedStorage);
  } finally {
    if (native) {
      await page.evaluate(() => { window.__researchGaugeIndexRestore?.(); delete window.__researchGaugeIndexRestore; delete window.__researchGaugeMismatchProbe; });
      await page.getByLabel('Observatory', { exact: true }).selectOption('macro');
      await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
      await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
    } else {
      await page.unroute('**/api/financials/index?*');
      await switchRelease(page, older.as_of);
      await switchRelease(page, current.as_of);
    }
  }
  await page.getByLabel('Company financials', { exact: true }).click();
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
}

export async function researchGaugeFlows(page, project, { native = false } = {}) {
  const data = await fixture(project), checks = [], viewport = page.viewportSize(), requests = [], assetResponses = [];
  const checked = message => { checks.push(message); console.log(`Research gauge: ${message}`); };
  const recordRequest = request => { if (request.url().includes('/api/financials/')) requests.push(request.url()); };
  const recordResponse = response => { if (response.url().endsWith('/' + data.manifest.artifact.path)) assetResponses.push({ status: response.status(), contentEncoding: response.headers()['content-encoding'] ?? null }); };
  page.on('request', recordRequest);
  page.on('response', recordResponse);
  await page.getByLabel('Observatory', { exact: true }).selectOption('companies');
  await openCompany(page, 'Holmen', '102');
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
  const work = await seedAuthoredWork(page, project, data.manifest);
  await watchReadOnly(page);
  try {
    checked('All 19,140 unique listing IDs, source hashes, directory presence, historical denominators and five valuation readiness lanes reconcile with 13,882 independent reverse-cash equations');
    await verifyCard(page, data.byId['102'], data.manifest);
    await assertReadOnly(page, work.seeded);
    checked('Company profile shows a dated standard starter and source-bound research evidence while preserving authored drafts, deliberate blanks, purchase settings and revision bytes');
    await openScreen(page);
    const screen = page.locator('[data-research-gauge-ready="true"]');
    assert.equal(Number(await screen.getAttribute('data-research-gauge-total')), 19140);
    assert.equal(await matches(page), 19140);
    assert.equal(await page.getByLabel('Show starter valuation context', { exact: true }).isChecked(), false);
    assert.equal(await page.locator('[data-research-row-valuation]').count(), 0);
    const alphabetical = [...data.rows].sort((a, b) => a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }) || Number(a.id) - Number(b.id));
    assert.deepEqual(await rowIds(page), alphabetical.slice(0, 50).map(row => row.id));
    assert.equal(await page.getByLabel('Previous research page', { exact: true }).isDisabled(), true);
    await page.getByLabel('Next research page', { exact: true }).click();
    assert.deepEqual(await rowIds(page), alphabetical.slice(50, 100).map(row => row.id));
    await page.getByLabel('Previous research page', { exact: true }).click();
    assert.deepEqual(await rowIds(page), alphabetical.slice(0, 50).map(row => row.id));
    checked('Universe opens with historical evidence, alphabetical order, all missing/manual listings and bounded distinct 50-row pages; starter valuation is optional');

    for (const route of ['operating', 'property', 'financial', 'unclassified']) {
      await page.getByLabel('Research business route', { exact: true }).selectOption(route);
      assert.equal(await matches(page), data.rows.filter(row => row.route === route).length);
      for (const id of await rowIds(page)) assert.equal(data.byId[id].route, route);
    }
    await page.getByLabel('Research business route', { exact: true }).selectOption('all');
    for (const readiness of ['history_available', 'limited_history', 'reconcile_data', 'no_history']) {
      await page.getByLabel('Research annual coverage', { exact: true }).selectOption(readiness);
      assert.equal(await matches(page), data.rows.filter(row => row.readiness === readiness).length);
      for (const id of await rowIds(page)) assert.equal(data.byId[id].readiness, readiness);
    }
    await page.getByLabel('Research annual coverage', { exact: true }).selectOption('all');
    for (const presence of ['latest', 'older']) {
      await page.getByLabel('Research snapshot coverage', { exact: true }).selectOption(presence);
      assert.equal(await matches(page), data.rows.filter(row => row.presence === presence).length);
    }
    await page.getByLabel('Research snapshot coverage', { exact: true }).selectOption('all');
    for (const country of ['SE', 'US']) {
      await page.getByLabel('Research listing country', { exact: true }).selectOption(country);
      assert.equal(await matches(page), data.rows.filter(row => row.country === country).length);
    }
    await page.getByLabel('Research listing country', { exact: true }).selectOption('all');
    await page.getByLabel('Research sector', { exact: true }).selectOption('1');
    assert.equal(await matches(page), data.rows.filter(row => row.sectorId === '1').length);
    await page.getByLabel('Research business route', { exact: true }).selectOption('property');
    assert.equal(await matches(page), data.rows.filter(row => row.sectorId === '1' && row.route === 'property').length);
    await page.getByLabel('Research sector', { exact: true }).selectOption('all');
    await page.getByLabel('Research business route', { exact: true }).selectOption('all');
    await page.getByLabel('Research branch', { exact: true }).selectOption('21');
    assert.equal(await matches(page), data.rows.filter(row => row.branchId === '21').length);
    await page.getByLabel('Research snapshot coverage', { exact: true }).selectOption('older');
    assert.equal(await matches(page), data.rows.filter(row => row.branchId === '21' && row.presence === 'older').length);
    await page.getByRole('button', { name: 'Clear all research filters', exact: true }).click();
    checked('Business route, annual evidence, country, branch and newest/older directory filters reconcile independently and combine without dropping unknown or specialist cases');

    const lenses = {
      positive_cash: row => row.annual.cash.count === 5 && row.annual.cash.positive === 5,
      negative_cash: row => row.annual.cash.negative > 0,
      negative_operating_cash: row => row.annual.operatingCash.latest !== null && row.annual.operatingCash.latest < 0,
    };
    for (const [lens, predicate] of Object.entries(lenses)) {
      await page.getByLabel('Research historical lens', { exact: true }).selectOption(lens);
      assert.equal(await matches(page), data.rows.filter(predicate).length);
      for (const id of await rowIds(page)) assert.ok(predicate(data.byId[id]));
    }
    await page.getByRole('button', { name: 'Clear all research filters', exact: true }).click();
    await page.getByLabel('Research listing search', { exact: true }).fill('zzzz research gauge no listing 999999');
    assert.equal(await matches(page), 0);
    assert.deepEqual(await rowIds(page), []);
    assert.match(await screen.innerText(), /Missing observations are not treated as zeros/);
    assert.equal(await page.getByLabel('Next research page', { exact: true }).isDisabled(), true);
    await page.getByLabel('Research listing search', { exact: true }).fill(data.byId['102'].isin);
    assert.deepEqual(await rowIds(page), data.rows.filter(row => `${row.name} ${row.ticker ?? ''} ${row.isin ?? ''} ${row.id}`.toLowerCase().includes(data.byId['102'].isin.toLowerCase())).sort((a, b) => a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }) || Number(a.id) - Number(b.id)).map(row => row.id));
    checked('Historical lenses use actual cash observations independently of valuation; ISIN search, zero matches and missing denominators stay explicit');

    await page.getByLabel('Show starter valuation context', { exact: true }).check();
    await page.locator('[data-research-listing="102"]').getByLabel('Inspect research evidence for Holmen', { exact: true }).click();
    await verifyEvidence(page.locator('[data-research-inspector="102"]'), data.byId['102']);
    const method = page.locator('[data-research-method]');
    await method.locator('summary').click();
    assert.match(await method.innerText(), /550-day cutoff.*not a validated investment threshold/);
    assert.match(await method.innerText(), /not independent companies/);
    assert.match(await method.innerText(), /Crisis years remain/);
    await assertReadOnly(page, work.seeded);
    checked('Optional price context reconciles dated equity price, terminal value, Low NPV and reverse cash levels; policy, signed cash, crisis history and coverage rules remain inspectable');

    for (const width of [1500, 390]) {
      await fitViewport(page, width, 'Research table and open evidence');
      await page.screenshot({ path: resolve(project, `test-results/research-gauge-${native ? 'native' : 'browser'}-table-${width}.png`), fullPage: true });
    }
    await page.locator('[data-research-listing="102"]').getByLabel('Open profile for Holmen', { exact: true }).click();
    await verifyCard(page, data.byId['102'], data.manifest);
    await fitViewport(page, 390, 'Company research card');
    await page.locator('[data-research-card="102"]').screenshot({ path: resolve(project, `test-results/research-gauge-${native ? 'native' : 'browser'}-card-390.png`) });
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
    checked('Research table, evidence inspector and company card fit desktop and 390-pixel viewports with internal table scrolling');

    await page.getByLabel('Observatory', { exact: true }).selectOption('sectors');
    await page.getByLabel('Branch research screen', { exact: true }).click();
    await page.locator('[data-research-gauge-ready="true"]').waitFor();
    assert.equal(await page.getByLabel('Research branch', { exact: true }).inputValue(), data.byId['102'].branchId);
    assert.equal(await matches(page), data.rows.filter(row => row.branchId === data.byId['102'].branchId).length);
    await page.getByLabel('Research listing search', { exact: true }).fill('Holmen');
    await page.locator('[data-research-listing="102"]').getByLabel('Open profile for Holmen', { exact: true }).click();
    await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
    checked('Branch research opens with the selected branch and can return to the currently selected company profile');

    const specialRows = [data.byId['159'], data.byId['167'], data.rows.find(row => row.valuation.value !== null && row.valuation.value <= 0 && row.annual.periods.length > 0)];
    for (const row of specialRows) {
      assert.ok(row);
      await openCompany(page, row.name, row.id);
      const card = await verifyCard(page, row, data.manifest);
      if (row.route === 'financial') assert.match(await card.innerText(), /regulatory capital and distributions/);
      if (!row.annual.periods.length) assert.match(await card.innerText(), /No comparable annual window/);
      if (row.valuation.value !== null && row.valuation.value <= 0) assert.match(await card.innerText(), /nonpositive/);
    }
    checked('Financial-business, no-history older listing and nonpositive-value examples keep appropriate research questions and missing valuation outputs');

    if (!native) {
      let releaseResponse, requested, timer;
      const released = new Promise(resolve => { releaseResponse = resolve; });
      const started = new Promise(resolve => { requested = resolve; });
      const route = '**/api/financials/company?*id=159*';
      await page.route(route, async route => { requested(); await released; await route.continue(); });
      try {
        await openCompany(page, 'Nordea Bank', '159', { wait: false });
        await Promise.race([started, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Previous-company history request was not intercepted')), 30000); })]);
        clearTimeout(timer);
        await openCompany(page, 'SCA', '197');
        await page.locator('[data-research-card="197"][data-research-card-ready="true"]').waitFor();
        releaseResponse();
        await page.waitForLoadState('networkidle');
        assert.equal(await page.locator('[data-research-card="159"]').count(), 0);
        assert.equal(await page.locator('[data-research-card="197"][data-research-card-ready="true"]').count(), 1);
        await assertReadOnly(page, work.seeded);
      } finally { clearTimeout(timer); releaseResponse(); await page.unroute(route); }
      checked('A delayed previous-company history response cannot replace the selected company research card');
    }
    await openCompany(page, 'Holmen', '102');
    await assertReadOnly(page, work.seeded);
    await failClosedFlows(page, project, data, work.seeded, native);
    checked('Corrupt compressed research data and an active financial-pack identity mismatch withhold both profile and universe results and recover after the valid source returns');
    const finalAudit = await assertReadOnly(page, work.seeded);
    if (native) assert.ok(finalAudit.invokes.some(item => item.command === 'financial_index'), 'Native audit must observe actual commands, not merely an empty event list');
    assert.deepEqual(requests.filter(url => url.includes('/api/financials/annual?')), [], 'Research screen never requests bulk annual histories');
    assert.ok(assetResponses.length > 0, 'Acceptance must exercise the real local research asset transport');
    assert.ok(assetResponses.every(response => response.contentEncoding === null), 'Opaque compressed payload must reach the checksum validator without server precompressed-file Content-Encoding');
    checked('Browsing, filters, evidence, company switches and source failures do not mount valuations, mutate draft/revision storage or request bulk annual histories');
    return { checks, listings: data.rows.length, pack: data.manifest.financialPackId, taxonomy: data.manifest.taxonomySha256, artifactSha256: data.manifest.artifact.sha256, valuationLanes: data.lanes, reverseChecks: data.reverseChecks, protectedStorage: work.original };
  } catch (error) {
    await page.screenshot({ path: resolve(project, `test-results/research-gauge-${native ? 'native' : 'browser'}-failure.png`), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    page.off('request', recordRequest);
    page.off('response', recordResponse);
    await restoreAuthoredWork(page, work.original);
    await page.setViewportSize(viewport ?? { width: 1500, height: 960 });
  }
}

export async function assertResearchGaugeRestart(page, result) {
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
  assert.deepEqual(await protectedStorage(page), result.protectedStorage, 'Process restart retains original draft/revision bytes after read-only research browsing');
  assert.equal(await page.locator('.valuation-workspace').count(), 0);
  await openScreen(page);
  assert.equal(await matches(page), result.listings);
  assert.equal(await page.getByLabel('Show starter valuation context', { exact: true }).isChecked(), false);
  await page.getByRole('button', { name: 'Back to explorer', exact: true }).click();
  await page.locator('[data-research-card="102"][data-research-card-ready="true"]').waitFor();
}

async function standalone() {
  const project = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  if (process.argv.includes('--artifact-only')) {
    const data = await fixture(project);
    console.log(JSON.stringify({ status: 'passed', listings: data.rows.length, lanes: data.lanes, reverseChecks: data.reverseChecks }));
    return;
  }
  const { chromium } = await import('playwright');
  const base = process.env.ATLAS_URL ?? 'http://127.0.0.1:1420';
  await mkdir(resolve(project, 'test-results'), { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/google/chrome/chrome', headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const external = [], errors = [], context = await browser.newContext({ viewport: { width: 1500, height: 960 } });
  context.setDefaultTimeout(30000);
  const csp = JSON.parse(await readFile(resolve(project, 'src-tauri/tauri.conf.json'), 'utf8')).app.security.csp;
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (url === `${base}/`) { const response = await route.fetch(); return route.fulfill({ response, headers: { ...response.headers(), 'content-security-policy': csp } }); }
    if (url.startsWith(base) || url.startsWith('blob:') || url.startsWith('data:')) return route.continue();
    external.push(url); return route.abort();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base);
    await page.locator('[data-country="SE"][data-ready="true"]').waitFor();
    const result = await researchGaugeFlows(page, project);
    assert.deepEqual(external, []); assert.deepEqual(errors, []);
    await writeFile(resolve(project, 'test-results/research-gauge-browser-report.json'), JSON.stringify({ status: 'passed', ...result, external, errors }, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await standalone();
