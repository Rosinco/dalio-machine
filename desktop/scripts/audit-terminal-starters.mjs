#!/usr/bin/env node
// Compare every listing with the retained pre-terminal runtime. No app profile writes.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { gunzipSync, gzipSync } from 'node:zlib';
import { build } from 'esbuild';

const desktop = resolve(fileURLToPath(new URL('..', import.meta.url)));
const out = resolve(desktop, 'test-results/terminal-starters-2026-09-13');
mkdirSync(out, { recursive: true });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const identity = path => ({ path: relative(desktop, path), sha256: sha(readFileSync(path)), bytes: statSync(path).size });
const calibrationPath = resolve(desktop, 'src/data/cash-uncertainty-2026-09-12.json');
const calibration = JSON.parse(readFileSync(calibrationPath));
const pack = resolve(desktop, calibration.provenance.sourcePack.path);
const taxonomyPath = resolve(desktop, 'public/data/taxonomy.json');
const businessIndexPath = resolve(desktop, 'public/data/business-index.json');
const frozenPath = resolve(desktop, 'test-results/empirical-cash-starter-2026-09-13/starter-runtime.executed.mjs');
const previousReceiptPath = resolve(desktop, 'test-results/empirical-cash-starter-2026-09-13/universe-runtime-audit.json');
const inputs = [pack, taxonomyPath, businessIndexPath, calibrationPath, frozenPath, previousReceiptPath,
  ...['starterValuations.ts', 'valuation.ts', 'terminalValue.ts', 'cashComponents.ts', 'cashUncertainty.ts', 'financialData.ts', 'marketData.ts', 'starterMigration.ts', 'savedValuations.ts'].map(f => resolve(desktop, 'src', f))];
const before = inputs.map(identity);
assert.equal(before[0].sha256, calibration.provenance.sourcePack.sha256);
assert.equal(before[1].sha256, calibration.taxonomySha256);
const previousReceipt = JSON.parse(readFileSync(previousReceiptPath));
assert.equal(previousReceipt.status, 'passed');
assert.equal(before[4].sha256, previousReceipt.outputs.find(output => output.path === before[4].path)?.sha256);
assert.equal(before[4].sha256, '69b2e991851a3554583005125f7616cc53cb19b30c6be1168e4d63c2d8162af9');
const runtimePath = resolve(out, 'runtime.mjs');
await build({ stdin: { contents: "export {buildStarterValuation, settingsForStarterOrigin} from './src/starterValuations'; export {calculateValuation} from './src/valuation'; export {validateValuation} from './src/savedValuations'; export {isUntouchedLegacyStarter} from './src/starterMigration';", resolveDir: desktop }, outfile: runtimePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent' });
const current = await import(pathToFileURL(runtimePath));
const old = await import(pathToFileURL(frozenPath));
const taxonomy = JSON.parse(readFileSync(taxonomyPath)), businessIndex = JSON.parse(readFileSync(businessIndexPath));
const db = new DatabaseSync(pack, { readOnly: true }); db.exec('PRAGMA query_only = ON');
const index = { ...JSON.parse(gunzipSync(db.prepare("SELECT payload FROM metadata WHERE key='index'").get().payload)), id: before[0].sha256, bytes: before[0].bytes };
const entries = old.companyEntries(businessIndex, taxonomy, index);
const select = db.prepare('SELECT payload,sha256 FROM companies WHERE id=?');
const close = (a, b, name) => assert.ok(Number.isFinite(a) && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${name}: ${a} != ${b}`);
const counts = { listings: 0, unchangedAnnualPaths: 0, exactOldDraftReplay: 0, exactCurrentReplay: 0, untouchedOldMigration: 0, terminalReady: 0, terminalUnavailable: 0, nonpositiveMedian: 0, readyDCFNPV: 0, independentlyReconciledScenarios: 0, terminalChangedListings: 0 };
const ledger = [], examples = {};
for (const entry of entries) {
  const row = select.get(entry.id), bytes = gunzipSync(row.payload);
  assert.equal(sha(bytes), row.sha256); assert.equal(row.sha256, index.companies[entry.id].sha256);
  const history = old.decodeFinancialCompany(JSON.parse(bytes), index, entry.id);
  const effective = { ...entry, ...taxonomy.classifications[entry.id] };
  const prior = old.buildStarterValuation(effective, history, index, '2026-09-13');
  const built = current.buildStarterValuation(effective, history, index, '2026-09-13'), d = built.draft;
  const replay = current.buildStarterValuation(effective, history, index, '2026-09-13', current.settingsForStarterOrigin(prior.draft.starterOrigin));
  assert.deepEqual(replay.draft, prior.draft, `${entry.id} old-v3 exact reproduction`); counts.exactOldDraftReplay++;
  assert.deepEqual(current.buildStarterValuation(effective, history, index, '2026-09-13', current.settingsForStarterOrigin(d.starterOrigin)).draft, d); counts.exactCurrentReplay++;
  assert.equal(current.isUntouchedLegacyStarter(prior.draft, replay.draft), true); counts.untouchedOldMigration++;
  current.validateValuation({ format: 'macro-atlas-valuation', version: 1, id: `audit-${entry.id}`, company: entry.id, created: '2026-09-13T00:00:00.000Z', release: 'a'.repeat(64), financial: index.id, taxonomy: calibration.taxonomySha256, draft: d });
  assert.deepEqual(built.evidence.forecast, prior.evidence.forecast); counts.unchangedAnnualPaths++;
  const cash = built.evidence.annual.map(a => a.cashFlow);
  const sorted = [...cash].sort((a,b) => a-b), n = sorted.length;
  const median = built.evidence.manualCashRequired || n < 3 || cash.some(c => c === null) ? null : n % 2 ? sorted[(n-1)/2] : (sorted[n/2-1]+sorted[n/2])/2;
  if (median === null) counts.terminalUnavailable++; else { counts.terminalReady++; if (median <= 0) counts.nonpositiveMedian++; }
  let changed = false;
  const result = current.calculateValuation(d);
  for (const [key, coefficient] of [['low', -.2], ['mid', 0], ['high', .2]]) {
    const s = d.scenarios[key], expectedCash = median === null ? null : median + Math.abs(median)*coefficient;
    assert.deepEqual(s.cashFlows, prior.draft.scenarios[key].cashFlows);
    if (expectedCash === null) { assert.equal(s.terminalCash.cashFlow, null); assert.equal(s.terminalEquity, null); }
    else { close(s.terminalCash.cashFlow, expectedCash, 'Terminal seed'); close(s.terminalEquity, Math.max(0, expectedCash)/.1, 'Terminal sale'); }
    if (s.terminalEquity !== prior.draft.scenarios[key].terminalEquity) changed = true;
    if (!result.scenarios[key].error) {
      const r = result.scenarios[key], discounted = s.cashFlows.map((c,i) => c/1.1**(i+1));
      const cashPV = discounted.reduce((a,b) => a+b, 0), terminalPV = Math.max(0, expectedCash)/.1/1.1**10;
      close(r.value, cashPV+terminalPV, 'DCF'); close(r.npv, cashPV+terminalPV-d.marketCap, 'NPV');
      let cumulative = -d.marketCap;
      for (let year=0; year<=10; year++) {
        if (year) cumulative += discounted[year-1];
        close(r.cumulativeNPV[year], cumulative, 'Cumulative NPV');
        close(r.cumulativeNPVWithSale[year], cumulative+(year===10 ? terminalPV : 0), 'Cumulative NPV with sale');
      }
      counts.independentlyReconciledScenarios++;
    }
  }
  if (changed) counts.terminalChangedListings++;
  if (result.ready) counts.readyDCFNPV++;
  counts.listings++;
  const frozen = { company: entry.id, name: entry.display_name, asOf: d.valuationDate, timeBasis: 'Full model years after valuation date; fiscal outcome mapping needs separate review', currency: d.currency, sector: effective.sector_id, branch: effective.branch_id, sourceCompanySha256: row.sha256, origin: d.starterOrigin, annualInputs: built.evidence.annual, cashDefinition: 'unreviewed-provider-free_cash_flow', marketCap: d.marketCap, priceDate: d.priceDate, years: d.years, scenarios: d.scenarios, ready: result.ready };
  ledger.push(JSON.stringify(frozen));
  if (['696', '102', '161'].includes(entry.id)) examples[entry.id] = { name: entry.display_name, median, oldSales: Object.fromEntries(Object.entries(prior.draft.scenarios).map(([k,s]) => [k,s.terminalEquity])), newSales: Object.fromEntries(Object.entries(d.scenarios).map(([k,s]) => [k,s.terminalEquity])) };
}
db.close();
assert.equal(counts.listings, 19140); assert.deepEqual(inputs.map(identity), before);
const ledgerPath = resolve(out, 'frozen-starters.jsonl.gz'); writeFileSync(ledgerPath, gzipSync(ledger.join('\n')+'\n'));
const receipt = { status: 'passed', capturedAt: new Date().toISOString(), modelAsOf: '2026-09-13', scope: 'Generic baselines for all saved listings; reviewed/default user drafts are not overwritten', limits: ['Implementation audit, not forecast validation', 'Median and 20% terminal range are assumptions', 'Annual calibration and source proxy definitions unchanged', 'Future outcome mapping must respect fiscal dates, vintage changes and departed listings'], counts, examples, inputs: before, runtime: identity(runtimePath), ledger: identity(ledgerPath), script: identity(fileURLToPath(import.meta.url)) };
writeFileSync(resolve(out, 'universe-audit.json'), JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({ status: receipt.status, counts, examples, ledger: receipt.ledger }));
