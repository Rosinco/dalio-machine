#!/usr/bin/env node
/** Deterministic derived snapshot. Reads only the pinned source pack; never opens an app profile. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, relative, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import { build } from 'esbuild';

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const checkOnly = process.argv.includes('--check');
assert.ok(process.argv.slice(2).every(arg => arg === '--check'), 'Supported argument: --check');
const asOf = '2026-09-13', scratch = mkdtempSync(resolve(tmpdir(), 'atlas-research-gauge-'));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const identity = path => ({ path: relative(desktop, path), sha256: sha(readFileSync(path)), bytes: statSync(path).size });
const calibrationPath = resolve(desktop, 'src/data/cash-uncertainty-2026-09-12.json');
const calibration = JSON.parse(readFileSync(calibrationPath));
const packPath = resolve(desktop, calibration.provenance.sourcePack.path);
const taxonomyPath = resolve(desktop, 'public/data/taxonomy.json');
const businessPath = resolve(desktop, 'public/data/business-index.json');
// An opaque extension prevents static servers from applying Content-Encoding:gzip
// and making the browser decompress before the compressed-byte integrity check.
const artifactPath = resolve(desktop, 'public/data/research-gauge/gauge.bin');
const manifestPath = resolve(desktop, 'src/data/research-gauge-manifest.json');
const receiptPath = resolve(desktop, 'test-results/research-gauge-export-2026-09-13/receipt.json');
const finite = x => typeof x === 'number' && Number.isFinite(x);
const amount = x => finite(x) && Math.abs(x) <= 1e12;
const close = (a, b, message) => b === null ? assert.equal(a, null, message) : assert.ok(finite(a) && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${message}: ${a} != ${b}`);

// Independent arithmetic; uses inputs, never valuation/terminal/purchase functions.
function scalar(draft, key) {
  const scenario = draft.scenarios[key], rate = scenario.discountRate, cash = scenario.cashFlows.slice(0, draft.years);
  if (!amount(rate) || rate <= 0 || rate > 100 || cash.length !== draft.years || !cash.every(amount)) return null;
  let sale = scenario.terminalEquity;
  if (scenario.terminalCash) {
    const c = scenario.terminalCash.cashFlow, g = scenario.terminalCash.growthRate;
    if (!amount(c) || !amount(g) || g <= -100 || g >= rate) return null;
    sale = Math.max(0, c) / ((rate - g) / 100);
  }
  if (!amount(sale) || sale < 0) return null;
  const cashPV = cash.reduce((sum, c, i) => sum + c / (1 + rate / 100) ** (i + 1), 0);
  const terminalPV = sale / (1 + rate / 100) ** draft.years;
  return { value: cashPV + terminalPV, cashPV, terminalPV };
}

let db;
try {
  const pack = identity(packPath), taxonomyFile = identity(taxonomyPath);
  assert.equal(pack.sha256, calibration.provenance.sourcePack.sha256, 'Source pack differs from retained calibration');
  assert.equal(taxonomyFile.sha256, calibration.taxonomySha256, 'Taxonomy differs from retained calibration');
  const runtimePath = resolve(scratch, 'runtime.mjs');
  const bundleOptions = { absWorkingDir: desktop, stdin: { contents: [
    "export {buildResearchGaugeRow} from './src/researchGaugeModel';",
    "export {buildStarterValuation} from './src/starterValuations';",
    "export {decodeFinancialCompany,validateFinancialIndex} from './src/financialData';",
    "export {companyEntries} from './src/listingCatalogue';",
  ].join('\n'), resolveDir: desktop }, outfile: runtimePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent', metafile: true };
  const discovery = await build({ ...bundleOptions, write: false });
  const inputs = [...new Set([packPath, taxonomyPath, businessPath, calibrationPath, fileURLToPath(import.meta.url), ...Object.keys(discovery.metafile.inputs).filter(p => p !== '<stdin>').map(p => resolve(desktop, p))])].sort();
  const before = inputs.map(identity);
  await build(bundleOptions);
  const runtime = await import(pathToFileURL(runtimePath));
  const taxonomy = JSON.parse(readFileSync(taxonomyPath)), business = JSON.parse(readFileSync(businessPath));
  db = new DatabaseSync(packPath, { readOnly: true }); db.exec('PRAGMA query_only = ON');
  const index = { ...JSON.parse(gunzipSync(db.prepare("SELECT payload FROM metadata WHERE key='index'").get().payload)), id: pack.sha256, bytes: pack.bytes };
  runtime.validateFinancialIndex(index, taxonomyFile.sha256);
  const entries = runtime.companyEntries(business, taxonomy, index).sort((a, b) => Number(a.id) - Number(b.id));
  assert.equal(entries.length, 19140); assert.equal(new Set(entries.map(e => e.id)).size, 19140);
  assert.deepEqual(entries.map(e => e.id), Object.keys(index.companies).sort((a, b) => Number(a) - Number(b)));
  assert.equal(db.prepare('SELECT COUNT(*) n FROM companies').get().n, entries.length);
  const priorPath = resolve(desktop, 'test-results/purchase-range-2026-09-13/purchase-ledger.jsonl.gz');
  const prior = existsSync(priorPath) ? new Map(gunzipSync(readFileSync(priorPath)).toString().trim().split('\n').map(line => { const r = JSON.parse(line); return [r.company, r]; })) : null;
  const counts = { listings: 0, independentMid: 0, independentLow: 0, reverseEquations: 0, frozenMidReconciliations: 0, nonmutatedHistories: 0, route: {}, readiness: {}, valuation: {}, quarterRevenue: 0 };
  const bump = (dict, key) => { dict[key] = (dict[key] ?? 0) + 1; };
  const rows = [], select = db.prepare('SELECT payload,sha256 FROM companies WHERE id=?');
  for (const entry of entries) {
    const packed = select.get(entry.id), payload = gunzipSync(packed.payload);
    assert.equal(sha(payload), packed.sha256); assert.equal(packed.sha256, index.companies[entry.id].sha256);
    const history = runtime.decodeFinancialCompany(JSON.parse(payload), index, entry.id), snapshot = JSON.stringify(history);
    const row = runtime.buildResearchGaugeRow(entry, history, index, taxonomy, asOf);
    assert.equal(JSON.stringify(history), snapshot, `${entry.id}: history mutated`); counts.nonmutatedHistories++;
    const draft = runtime.buildStarterValuation({ ...entry, ...taxonomy.classifications[entry.id] }, history, index, asOf).draft;
    const mid = scalar(draft, 'mid'), low = scalar(draft, 'low');
    for (const key of ['value', 'cashPV', 'terminalPV']) close(row.valuation[key], mid?.[key] ?? null, `${entry.id}: independent Mid ${key}`);
    close(row.valuation.lowValue, low?.value ?? null, `${entry.id}: independent Low value`);
    close(row.valuation.lowNPV, low && row.valuation.candidateEquity !== null ? low.value - row.valuation.candidateEquity : null, `${entry.id}: independent Low NPV`);
    close(row.valuation.ceiling, mid && mid.value > 0 ? .7 * mid.value : null, `${entry.id}: 30% policy`);
    counts.independentMid++; counts.independentLow++;
    if (row.valuation.reverseCashFactor !== null) {
      for (const [factor, margin] of [[row.valuation.reverseCashFactor, 1], [row.valuation.reverseCashFactor30, .7]]) {
        const scaled = structuredClone(draft), s = scaled.scenarios.mid;
        s.cashFlows = s.cashFlows.map(c => c * factor);
        if (s.terminalCash) s.terminalCash.cashFlow *= factor; else s.terminalEquity *= factor;
        // Scaling can exceed the application input cap; validate the mathematical identity directly as well.
        const independent = scalar(scaled, 'mid');
        close(independent ? margin * independent.value : margin * factor * mid.value, row.valuation.candidateEquity, `${entry.id}: reverse equation`);
        counts.reverseEquations++;
      }
    }
    if (prior) {
      const old = prior.get(entry.id); assert.ok(old, `${entry.id}: missing frozen valuation`);
      assert.equal(old.sourceCompanySha256, row.sourceCompanySha256);
      close(row.valuation.value, old.scenarios.mid.value, `${entry.id}: frozen Mid economics`);
      assert.equal(row.valuation.candidateEquity, old.candidateEquity); counts.frozenMidReconciliations++;
    }
    for (const stat of ['cash', 'operatingCash', 'ebit', 'revenue', 'margins']) {
      const series = row.annual[stat]; assert.equal(series.values.length, row.annual.periods.length);
      assert.equal(series.count, series.values.filter(finite).length); assert.equal(series.count, series.positive + series.negative + series.zero);
      assert.equal(series.latest, series.values[0] ?? null);
    }
    assert.ok(row.annual.periods.every(p => p.currency === row.annual.currency));
    if (row.route === 'financial') assert.equal(row.valuation.value, null);
    counts.listings++; bump(counts.route, row.route); bump(counts.readiness, row.readiness); bump(counts.valuation, row.valuation.status);
    counts.quarterRevenue += Number(row.quarter.revenueChangePercent !== null); rows.push(row);
  }
  const artifact = { format: 'macro-atlas-research-gauge', version: 1, asOf, financialPackId: pack.sha256, taxonomySha256: taxonomyFile.sha256, calibrationId: calibration.id, model: 'research-gauge-v1', rows };
  const bytes = Buffer.from(JSON.stringify(artifact)), zipped = gzipSync(bytes, { level: 9 });
  assert.ok(bytes.length <= 96 * 1024 * 1024, 'Research gauge uncompressed budget exceeded');
  assert.ok(zipped.length <= 12 * 1024 * 1024, 'Research gauge compressed budget exceeded');
  const manifest = { format: 'macro-atlas-research-gauge-manifest', version: 1, asOf, model: 'research-gauge-v1', financialPackId: pack.sha256, financialPackBytes: pack.bytes, taxonomySha256: taxonomyFile.sha256, calibrationId: calibration.id, rows: rows.length,
    artifact: { path: 'data/research-gauge/gauge.bin', sha256: sha(zipped), bytes: zipped.length, uncompressedSha256: sha(bytes), uncompressedBytes: bytes.length } };
  const manifestBytes = Buffer.from(JSON.stringify(manifest, null, 2) + '\n');
  assert.deepEqual(inputs.map(identity), before, 'Source inputs changed during export');
  if (checkOnly) {
    assert.deepEqual(readFileSync(artifactPath), zipped, 'Generated artifact is stale');
    assert.deepEqual(readFileSync(manifestPath), manifestBytes, 'Generated manifest is stale');
  } else {
    mkdirSync(dirname(artifactPath), { recursive: true }); writeFileSync(artifactPath, zipped);
    writeFileSync(manifestPath, manifestBytes);
    // This exporter owns the superseded generated transport asset only.
    rmSync(resolve(desktop, 'public/data/research-gauge/gauge.json.gz'), { force: true });
    mkdirSync(dirname(receiptPath), { recursive: true });
    writeFileSync(receiptPath, JSON.stringify({ status: 'passed', asOf, inputs: before, manifest, counts, priorLedger: prior ? identity(priorPath) : null,
      notes: ['Generic unedited starters only; no app profile or personal studies opened.', 'Counts describe listings, not distinct issuers.', 'Derived cash/asset ratios do not establish shareholder cash or investment quality.', 'No whole-universe financial history is fetched at runtime.'] }, null, 2) + '\n');
  }
  console.log(JSON.stringify({ status: checkOnly ? 'verified-identical' : 'exported', manifest, counts }, null, 2));
} finally { db?.close(); rmSync(scratch, { recursive: true, force: true }); }
