#!/usr/bin/env node
/** Independent whole-directory purchase arithmetic and retained-0.15 valuation replay. No app profiles or source writes. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gunzipSync, gzipSync } from 'node:zlib';
import { build, version as esbuildVersion } from 'esbuild';

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const allowedOutput = resolve(desktop, 'test-results/purchase-range-2026-09-13');
const outputFlag = process.argv.indexOf('--output');
if (outputFlag >= 0) assert.ok(process.argv[outputFlag + 1], '--output needs a directory');
const out = outputFlag >= 0 ? resolve(desktop, process.argv[outputFlag + 1]) : allowedOutput;
assert.ok(out === allowedOutput || out.startsWith(`${allowedOutput}${sep}`), 'Outputs must remain inside this audit directory');
assert.ok(!existsSync(out) || readdirSync(out).length === 0, 'Refuse to overwrite retained audit artifacts; use a new --output subdirectory');
mkdirSync(out, { recursive: true });
const started = Date.now(), asOf = '2026-09-13';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const identity = path => ({ path: relative(desktop, path), sha256: sha(readFileSync(path)), bytes: statSync(path).size });
const finite = value => typeof value === 'number' && Number.isFinite(value);
const amount = value => finite(value) && Math.abs(value) <= 1e12;
const day = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && value >= '1900-01-01' && value <= '2300-12-31'
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
const close = (actual, expected, label) => {
  if (expected === null) assert.equal(actual, null, label);
  else assert.ok(finite(actual) && finite(expected) && Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
};
const keys = ['low', 'mid', 'high'];

// This calculation uses draft cash inputs directly and imports no valuation or terminal helpers.
function scalarCashValue(draft, key) {
  if (!day(draft.valuationDate) || !/^[A-Z]{3}$/.test(draft.currency)
    || !Number.isInteger(draft.years) || draft.years < 1 || draft.years > 50) return null;
  const scenario = draft.scenarios[key], rate = scenario.discountRate;
  if (!amount(rate) || rate < 0 || rate > 100 || scenario.cashFlows.length < draft.years) return null;
  const cash = Array.from({ length: draft.years }, (_, index) => scenario.cashFlows[index]);
  if (!cash.every(amount)) return null;
  let terminalSale;
  if (scenario.terminalCash) {
    const { cashFlow, growthRate } = scenario.terminalCash;
    if (!amount(cashFlow) || !amount(growthRate) || rate <= 0 || growthRate <= -100 || growthRate >= rate) return null;
    terminalSale = Math.max(0, cashFlow) / ((rate - growthRate) / 100);
  } else terminalSale = scenario.terminalEquity;
  if (!amount(terminalSale) || terminalSale < 0) return null;
  const factor = 1 + rate / 100;
  let cashPV = 0;
  for (let year = 1; year <= draft.years; year++) cashPV += cash[year - 1] / factor ** year;
  const terminalPV = terminalSale / factor ** draft.years, value = cashPV + terminalPV;
  if (![cashPV, terminalPV, value].every(finite)) return null;
  return { value, cashPV, terminalPV, ceiling: value > 0 ? .7 * value : null,
    cashOnlyCeiling: cashPV > 0 ? .7 * cashPV : null, terminalShare: value > 0 ? terminalPV / value : null };
}

const calibrationPath = resolve(desktop, 'src/data/cash-uncertainty-2026-09-12.json');
const calibration = JSON.parse(readFileSync(calibrationPath));
const packPath = resolve(desktop, calibration.provenance.sourcePack.path);
const taxonomyPath = resolve(desktop, 'public/data/taxonomy.json');
const businessPath = resolve(desktop, 'public/data/business-index.json');
const priorReceiptPath = resolve(desktop, 'test-results/terminal-starters-2026-09-13/universe-audit.json');
const priorReceipt = JSON.parse(readFileSync(priorReceiptPath));
const priorRuntimePath = resolve(desktop, 'test-results/terminal-starters-2026-09-13/runtime.mjs');
const decoderPath = resolve(desktop, 'test-results/empirical-cash-starter-2026-09-13/starter-runtime.executed.mjs');
assert.equal(priorReceipt.status, 'passed');
assert.equal(identity(priorRuntimePath).sha256, priorReceipt.runtime.sha256);
assert.equal(priorReceipt.runtime.sha256, '93d04b843b24809b12d9d4cacb87103b881fed4fb9e11073f7ff1e2c960cc5ab');
assert.equal(identity(decoderPath).sha256, priorReceipt.inputs.find(input => input.path === relative(desktop, decoderPath))?.sha256);
assert.equal(identity(decoderPath).sha256, '69b2e991851a3554583005125f7616cc53cb19b30c6be1168e4d63c2d8162af9');
assert.equal(identity(packPath).sha256, calibration.provenance.sourcePack.sha256);
assert.equal(identity(taxonomyPath).sha256, calibration.taxonomySha256);
for (const path of [packPath, taxonomyPath, businessPath, calibrationPath])
  assert.equal(identity(path).sha256, priorReceipt.inputs.find(input => input.path === relative(desktop, path))?.sha256);

const runtimePath = resolve(out, 'purchase-runtime.executed.mjs');
const buildOptions = {
  absWorkingDir: desktop,
  stdin: { contents: [
    "export {buildStarterValuation} from './src/starterValuations';",
    "export {calculateValuation,calculateCashValue} from './src/valuation';",
    "export {calculatePurchaseRange,defaultPurchaseRangeSettings} from './src/purchaseRange';",
  ].join('\n'), resolveDir: desktop },
  outfile: runtimePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent', metafile: true,
};
// Discover the exact dependency set before the executed build, then bind its bytes.
const discovery = await build({ ...buildOptions, write: false });
const sourcePaths = Object.keys(discovery.metafile.inputs).filter(path => path !== '<stdin>').map(path => resolve(desktop, path));
const inputPaths = [...new Set([packPath, taxonomyPath, businessPath, calibrationPath, priorReceiptPath, priorRuntimePath, decoderPath, fileURLToPath(import.meta.url), ...sourcePaths])].sort();
const before = inputPaths.map(identity);
const built = await build(buildOptions);
assert.deepEqual(Object.keys(built.metafile.inputs).sort(), Object.keys(discovery.metafile.inputs).sort());
assert.deepEqual(inputPaths.map(identity), before, 'Inputs changed while bundling the executed runtime');
const current = await import(pathToFileURL(runtimePath));
const prior = await import(pathToFileURL(priorRuntimePath));
const decoder = await import(pathToFileURL(decoderPath));
assert.deepEqual(current.defaultPurchaseRangeSettings, { marginOfSafetyPercent: 30, referenceScenario: 'mid', unit: 'equity' });
const defaultsBefore = JSON.stringify(current.defaultPurchaseRangeSettings);
const explicitPolicy = Object.freeze({ ...current.defaultPurchaseRangeSettings, candidateEquity: 1234.5 });
const clearedPolicy = Object.freeze({ ...current.defaultPurchaseRangeSettings, candidateEquity: null });
const policiesBefore = JSON.stringify([explicitPolicy, clearedPolicy]);

const taxonomy = JSON.parse(readFileSync(taxonomyPath)), business = JSON.parse(readFileSync(businessPath));
const db = new DatabaseSync(packPath, { readOnly: true });
db.exec('PRAGMA query_only = ON');
const packIdentity = identity(packPath);
const index = { ...JSON.parse(gunzipSync(db.prepare("SELECT payload FROM metadata WHERE key='index'").get().payload)), id: packIdentity.sha256, bytes: packIdentity.bytes };
decoder.validateFinancialIndex(index, calibration.taxonomySha256);
const entries = decoder.companyEntries(business, taxonomy, index);
assert.equal(entries.length, 19140);
assert.equal(new Set(entries.map(entry => entry.id)).size, entries.length);
assert.equal(db.prepare('SELECT COUNT(*) AS n FROM companies').get().n, entries.length);
const select = db.prepare('SELECT payload,sha256 FROM companies WHERE id=?');
const counts = {
  listings: 0, exactFreshStarterEconomics: 0, exactPriorValuationResults: 0, exactPriorScenarioResults: 0,
  priorReadyValuations: 0, priorReadyScenarios: 0, availableIntrinsicScenarios: 0, unavailableIntrinsicScenarios: 0,
  positiveIntrinsicScenarios: 0, zeroIntrinsicScenarios: 0, negativeIntrinsicScenarios: 0,
  positivePurchaseCeilings: 0, completePositiveCeilingRanges: 0, selectedPositiveCeilings: 0,
  selectedNonpositiveValues: 0, selectedUnavailableValues: 0, validDatedCandidates: 0, missingDatedCandidates: 0,
  selectedQualifies: 0, selectedDoesNotQualify: 0, selectedQualificationUnavailable: 0,
  verifiedDefaultCandidateNPVs: 0, verifiedExplicitCandidateNPVs: 0, verifiedCashOnlyCeilings: 0,
  terminalContributionAbove100Percent: 0, listingsWithCashValueButNoDatedCandidate: 0,
  unpricedCeilingComparisons: 0, clearedCandidateComparisons: 0, unchangedDrafts: 0,
  decodedAnnualRows: 0, decodedQuarterlyRows: 0, decodedWithheldRows: 0,
};
const scenarioErrors = {}, examples = {}, ledger = [];
const sample = (name, value) => { if (!(name in examples)) examples[name] = value; };
let activeListing = null;
try {
  for (const entry of entries) {
    activeListing = entry.id;
    const row = select.get(entry.id);
    assert.ok(row, `${entry.id}: missing company payload`);
    const bytes = gunzipSync(row.payload);
    assert.equal(sha(bytes), row.sha256);
    assert.equal(row.sha256, index.companies[entry.id].sha256);
    const history = decoder.decodeFinancialCompany(JSON.parse(bytes), index, entry.id);
    counts.decodedAnnualRows += history.annual.length; counts.decodedQuarterlyRows += history.quarterly.length; counts.decodedWithheldRows += history.withheld.length;
    const effective = { ...entry, ...taxonomy.classifications[entry.id] };
    const draft = current.buildStarterValuation(effective, history, index, asOf).draft;
    const oldDraft = prior.buildStarterValuation(effective, history, index, asOf).draft;
    // Optional purchase controls are additive and cannot change existing starter economics.
    const { purchaseRange: _newPolicy, ...economicDraft } = draft;
    const { purchaseRange: _oldPolicy, ...oldEconomicDraft } = oldDraft;
    assert.deepEqual(economicDraft, oldEconomicDraft, `${entry.id}: fresh starter economics`);
    counts.exactFreshStarterEconomics++;
    const snapshot = JSON.stringify(draft);
    const currentValuation = current.calculateValuation(draft), oldValuation = prior.calculateValuation(draft);
    assert.deepEqual(currentValuation, oldValuation, `${entry.id}: exact pre-refactor valuation replay`);
    counts.exactPriorValuationResults++;
    counts.exactPriorScenarioResults += 3;
    counts.priorReadyValuations += Number(oldValuation.ready);
    counts.priorReadyScenarios += keys.filter(key => oldValuation.scenarios[key].error === null).length;
    const purchase = current.calculatePurchaseRange(draft, current.defaultPurchaseRangeSettings);
    assert.deepEqual(current.calculatePurchaseRange(draft), purchase, `${entry.id}: fresh default policy`);
    const explicit = current.calculatePurchaseRange(draft, explicitPolicy);
    const cleared = current.calculatePurchaseRange(draft, clearedPolicy);
    const unpricedDraft = { ...draft, marketCap: null, investment: null, priceDate: '', priceSource: '' };
    const unpriced = current.calculatePurchaseRange(unpricedDraft, current.defaultPurchaseRangeSettings);
    const candidate = amount(draft.marketCap) && draft.marketCap > 0 && day(draft.priceDate) && draft.priceDate <= draft.valuationDate && draft.priceSource.trim()
      ? draft.marketCap : null;
    assert.equal(purchase.marginOfSafetyPercent, 30);
    assert.equal(purchase.referenceScenario, 'mid');
    assert.equal(purchase.marginError, null); assert.equal(purchase.referenceError, null);
    assert.equal(purchase.candidateEquity, candidate);
    assert.equal(purchase.candidateError === null, candidate !== null);
    assert.equal(cleared.candidateEquity, null); assert.equal(cleared.qualifies, null);
    assert.equal(unpriced.candidateEquity, null); assert.equal(unpriced.qualifies, null);
    if (candidate === null) counts.missingDatedCandidates++; else counts.validDatedCandidates++;
    const expectedByKey = {};
    for (const key of keys) {
      const expected = scalarCashValue(draft, key), result = purchase.scenarios[key];
      expectedByKey[key] = expected;
      assert.equal(result.error === null, expected !== null, `${entry.id}/${key}: intrinsic availability`);
      for (const variant of [cleared, unpriced, explicit]) {
        for (const field of ['value', 'cashPV', 'terminalPV', 'ceiling', 'cashOnlyCeiling', 'terminalShare', 'error'])
          assert.equal(variant.scenarios[key][field], result[field], `${entry.id}/${key}: value independent of price/stake (${field})`);
      }
      assert.equal(cleared.scenarios[key].npvAtCandidate, null);
      assert.equal(unpriced.scenarios[key].npvAtCandidate, null);
      counts.unpricedCeilingComparisons++; counts.clearedCandidateComparisons++;
      if (expected === null) {
        counts.unavailableIntrinsicScenarios++;
        scenarioErrors[result.error] = (scenarioErrors[result.error] ?? 0) + 1;
        for (const field of ['value', 'cashPV', 'terminalPV', 'ceiling', 'cashOnlyCeiling', 'npvAtCandidate', 'terminalShare']) assert.equal(result[field], null);
        assert.equal(explicit.scenarios[key].npvAtCandidate, null);
        continue;
      }
      counts.availableIntrinsicScenarios++;
      if (expected.value > 0) counts.positiveIntrinsicScenarios++;
      else if (expected.value === 0) counts.zeroIntrinsicScenarios++;
      else counts.negativeIntrinsicScenarios++;
      for (const field of ['value', 'cashPV', 'terminalPV', 'ceiling', 'cashOnlyCeiling', 'terminalShare']) close(result[field], expected[field], `${entry.id}/${key}: independent ${field}`);
      close(result.npvAtCandidate, candidate === null ? null : expected.value - candidate, `${entry.id}/${key}: default candidate NPV`);
      close(explicit.scenarios[key].npvAtCandidate, expected.value - explicitPolicy.candidateEquity, `${entry.id}/${key}: explicit candidate NPV`);
      counts.verifiedExplicitCandidateNPVs++; counts.verifiedCashOnlyCeilings++;
      if (candidate !== null) counts.verifiedDefaultCandidateNPVs++;
      if (result.ceiling !== null) counts.positivePurchaseCeilings++;
      if (result.terminalShare > 1) counts.terminalContributionAbove100Percent++;
    }
    const ceilings = keys.map(key => expectedByKey[key]?.ceiling ?? null);
    const expectedRange = ceilings.every(finite) ? { min: Math.min(...ceilings), max: Math.max(...ceilings) } : null;
    if (expectedRange === null) assert.equal(purchase.scenarioCeilingRange, null);
    else {
      close(purchase.scenarioCeilingRange.min, expectedRange.min, 'Complete range minimum');
      close(purchase.scenarioCeilingRange.max, expectedRange.max, 'Complete range maximum');
      counts.completePositiveCeilingRanges++;
    }
    const selected = expectedByKey.mid;
    assert.equal(purchase.selected, purchase.scenarios.mid);
    close(purchase.selectedCeiling, selected?.ceiling ?? null, 'Selected ceiling');
    const expectedQualifies = !selected || candidate === null ? null : selected.ceiling === null ? false : candidate <= selected.ceiling;
    assert.equal(purchase.qualifies, expectedQualifies, `${entry.id}: selected policy qualification`);
    if (purchase.qualifies === null) counts.selectedQualificationUnavailable++;
    else if (purchase.qualifies) counts.selectedQualifies++;
    else counts.selectedDoesNotQualify++;
    if (selected === null) counts.selectedUnavailableValues++;
    else if (selected.ceiling === null) counts.selectedNonpositiveValues++;
    else counts.selectedPositiveCeilings++;
    if (candidate === null && keys.some(key => expectedByKey[key] !== null)) counts.listingsWithCashValueButNoDatedCandidate++;
    assert.equal(JSON.stringify(draft), snapshot, `${entry.id}: purchase/default calculations mutated the draft`);
    counts.unchangedDrafts++; counts.listings++;
    const record = { company: entry.id, name: entry.display_name, currency: draft.currency, asOf,
      sourceCompanySha256: row.sha256, draftSha256: sha(snapshot), priorValuationReady: oldValuation.ready,
      candidateEquity: candidate, priceDate: draft.priceDate, selectedCeiling: purchase.selectedCeiling,
      qualifies: purchase.qualifies, scenarioCeilingRange: purchase.scenarioCeilingRange, scenarios: purchase.scenarios };
    ledger.push(JSON.stringify(record));
    if (['102', '197', '696', '161'].includes(entry.id)) examples[entry.id] = record;
    if (candidate === null && selected?.ceiling !== null && selected) sample('ceilingWithoutDatedQuote', record);
    if (selected?.value < 0) sample('negativeSelectedValue', record);
    if (purchase.scenarioCeilingRange) sample('completePositiveRange', record);
  }
  assert.equal(counts.listings, 19140);
  assert.equal(counts.availableIntrinsicScenarios + counts.unavailableIntrinsicScenarios, 3 * counts.listings);
  assert.equal(counts.positiveIntrinsicScenarios + counts.zeroIntrinsicScenarios + counts.negativeIntrinsicScenarios, counts.availableIntrinsicScenarios);
  assert.equal(counts.selectedPositiveCeilings + counts.selectedNonpositiveValues + counts.selectedUnavailableValues, counts.listings);
  assert.equal(counts.priorReadyValuations, priorReceipt.counts.readyDCFNPV);
  assert.equal(counts.priorReadyScenarios, priorReceipt.counts.independentlyReconciledScenarios);
  assert.equal(JSON.stringify(current.defaultPurchaseRangeSettings), defaultsBefore);
  assert.equal(JSON.stringify([explicitPolicy, clearedPolicy]), policiesBefore);
  assert.deepEqual(inputPaths.map(identity), before, 'Bound inputs changed during the audit');
  const ledgerPath = resolve(out, 'purchase-ledger.jsonl.gz');
  writeFileSync(ledgerPath, gzipSync(ledger.join('\n') + '\n', { mtime: 0 }));
  const receipt = {
    status: 'passed', capturedAt: new Date().toISOString(), modelAsOf: asOf, elapsedMs: Date.now() - started,
    scope: 'All 19,140 frozen listings; newly generated generic starters. No normal app profile or reviewed user draft was opened.',
    policy: { marginOfSafetyPercent: 30, referenceScenario: 'mid', unit: 'equity', explicitAuditCandidateEquity: explicitPolicy.candidateEquity },
    limits: [
      'Implementation and preservation audit, not forecast accuracy, investment performance or a guarantee that a purchase threshold is attractive.',
      'The range is numeric only when all three positive scenario ceilings exist; negative/missing values never become a zero-price floor.',
      'Automatic candidate comparisons use saved dated prices, not live prices. Explicit audit prices are synthetic arithmetic inputs.',
      'Generic provider cash, terminal assumptions and annual uncertainty calibration retain their existing limitations.',
      'Per-share ownership validation and UI rendering are covered separately; this audit uses canonical equity currency millions.',
    ],
    counts, scenarioErrors, examples, inputs: before,
    execution: { node: process.version, esbuild: esbuildVersion, dependencies: Object.keys(built.metafile.inputs).filter(path => path !== '<stdin>').sort() },
    runtime: identity(runtimePath), ledger: identity(ledgerPath),
  };
  writeFileSync(resolve(out, 'universe-audit.json'), JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ status: receipt.status, counts, elapsedMs: receipt.elapsedMs, ledger: receipt.ledger }, null, 2));
} catch (error) {
  writeFileSync(resolve(out, 'failure.json'), JSON.stringify({ status: 'failed', activeListing, error: String(error?.stack ?? error), counts, capturedAt: new Date().toISOString(), inputs: before }, null, 2) + '\n');
  throw error;
} finally {
  db.close();
}
