#!/usr/bin/env node
/** Read-only source audit. Run from desktop; never opens the application profile. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, statSync, copyFileSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { gunzipSync, gzipSync } from 'node:zlib';
import { build } from 'esbuild';

const desktop = resolve(fileURLToPath(new URL('..', import.meta.url)));
const out = resolve(desktop, 'test-results/empirical-cash-starter-2026-09-13');
const asOf = '2026-09-13', started = Date.now();
mkdirSync(out, { recursive: true });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const file = name => ({ path: relative(desktop, name), sha256: sha(readFileSync(name)), bytes: statSync(name).size });
const calibrationPath = resolve(desktop, 'src/data/cash-uncertainty-2026-09-12.json');
const calibration = JSON.parse(readFileSync(calibrationPath));
const pack = resolve(desktop, calibration.provenance.sourcePack.path);
const taxonomyPath = resolve(desktop, 'public/data/taxonomy.json');
const businessPath = resolve(desktop, 'public/data/research.atlas.json');
const businessIndexPath = resolve(desktop, 'public/data/business-index.json');
const sources = [pack, taxonomyPath, businessPath, businessIndexPath, calibrationPath,
  ...['starterValuations.ts', 'cashUncertainty.ts', 'financialData.ts', 'marketData.ts', 'valuation.ts', 'listingCatalogue.ts', 'taxonomy.ts', 'business.ts', 'starterMigration.ts'].map(x => resolve(desktop, 'src', x))];
const before = sources.map(file);
assert.equal(before[0].sha256, calibration.provenance.sourcePack.sha256);
assert.equal(before[1].sha256, calibration.taxonomySha256);
assert.deepEqual(calibration.covidTreatment.omittedTargetYears, []);
assert.equal(calibration.covidTreatment.historicalShocksRemoved, false);

const runtimePath = resolve(out, 'starter-runtime.executed.mjs');
await build({ stdin: { contents: [
  "export {buildStarterValuation,defaultStarterSettings} from './src/starterValuations';",
  "export {decodeFinancialCompany,validateFinancialIndex} from './src/financialData';",
  "export {companyEntries} from './src/listingCatalogue';",
  "export {validateTaxonomy} from './src/taxonomy';",
  "export {validateBusiness,businessIndex} from './src/business';",
  "export {calculateValuation} from './src/valuation';",
  "export {isUntouchedLegacyStarter} from './src/starterMigration';",
].join('\n'), resolveDir: desktop }, outfile: runtimePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent' });
const runtime = await import(pathToFileURL(runtimePath));
const legacyPath = resolve(out, 'starter-v2-runtime.executed.mjs');
// Preserve the pre-edit runtime as part of this receipt, rather than relying on /tmp later.
copyFileSync('/tmp/macro-atlas-starter-v2-audit-module.mjs', legacyPath);
const legacy = await import(pathToFileURL(legacyPath));
const taxonomy = JSON.parse(readFileSync(taxonomyPath)), researchPackage = JSON.parse(readFileSync(businessPath));
const business = JSON.parse(researchPackage.business.content);
assert.equal(sha(researchPackage.business.content), researchPackage.business.sha256);
runtime.validateBusiness(business);
runtime.validateTaxonomy(taxonomy, business, researchPackage.business.sha256);
const businessIndex = JSON.parse(readFileSync(businessIndexPath));
assert.deepEqual(businessIndex, runtime.businessIndex(business));
const db = new DatabaseSync(pack, { readOnly: true });
db.exec('PRAGMA query_only = ON');
const index = { ...JSON.parse(gunzipSync(db.prepare("SELECT payload FROM metadata WHERE key='index'").get().payload)), id: before[0].sha256, bytes: before[0].bytes };
runtime.validateFinancialIndex(index, before[1].sha256);
const entries = runtime.companyEntries(businessIndex, taxonomy, index);
assert.equal(entries.length, 19140);
assert.equal(db.prepare('SELECT COUNT(*) n FROM companies').get().n, entries.length);
const select = db.prepare('SELECT payload,sha256 FROM companies WHERE id=?');
const counts = { listings: 0, effectiveClassificationChanges: 0, manualFinancials: 0, propertyExceptions: 0,
  historical: 0, percentage: 0, unavailable: 0, allCashYearsAvailable: 0, valuationReady: 0, missingMarketValue: 0,
  negativeMid: 0, zeroMid: 0, zeroMidHistorical: 0, quoteBasis: 0, historyShort: 0, noEligibleAnnualCash: 0,
  legacyV1FullDraftEqual: 0, legacyV2FullDraftEqual: 0, untouchedLegacyDefaultsRecognized: 0,
  historicalYearCellsVerified: 0, assumedTailYearCellsVerified: 0, percentageYearCellsVerified: 0,
  decodedAnnualRows: 0, decodedQuarterlyRows: 0, decodedWithheldRows: 0, nativeCashCellsVerified: 0, dcfNpvScenariosVerified: 0 };
const groups = {}, reasons = {}, examples = {}, anchors = [], ledger = [];
const baseSettings = { historyYears: 5, weights: [30,25,20,15,10], spreadPercent: 10, spreadStepPercent: 10, projection: 'trend', method: 'weighted-cash-starter-v2' };
const close = (actual, expected, label) => {
  if (expected === null) assert.equal(actual, null, label);
  else assert.ok(typeof actual === 'number' && Number.isFinite(actual) && Math.abs(actual - expected) <= 1e-9 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
};
const bump = (obj, key) => { obj[key] = (obj[key] ?? 0) + 1; };
const sample = (name, value) => { if (!(name in examples)) examples[name] = value; };
const independentClass = values => {
  if (values.length !== 5 || values.some(x => !Number.isFinite(x))) return { scale: null, dispersion: null, group: null };
  const scale = values.reduce((s,x) => s + Math.abs(x), 0) / 5;
  if (!scale) return { scale: 0, dispersion: null, group: null };
  const mean = values.reduce((s,x) => s + x, 0) / 5;
  const dispersion = Math.sqrt(values.reduce((s,x) => s + (x - mean) ** 2, 0) / 5) / scale;
  return { scale, dispersion, group: dispersion < .25 ? 'low' : dispersion < .75 ? 'medium' : 'high' };
};
const independentFactor = (model, year, group) => {
  const cell = calibration.calibrationByModelHorizon[model][year];
  const supported = x => x?.supported && Number.isFinite(x.factor) && x.factor >= 0 && x.count >= calibration.minimumCalibrationRows && x.listings >= calibration.minimumCalibrationListings && x.histories >= calibration.minimumCalibrationHistories;
  const chosenGroup = supported(cell.cashDispersion[group]) ? group : 'global';
  const record = chosenGroup === 'global' ? cell.global : cell.cashDispersion[group];
  assert.ok(supported(record));
  return { ...record, group: chosenGroup };
};
let activeId = null, stora = null;
try {
  for (const entry of entries) {
    activeId = entry.id;
    const row = select.get(entry.id), bytes = gunzipSync(row.payload);
    assert.equal(sha(bytes), row.sha256, `${entry.id}: compressed payload identity`);
    assert.equal(row.sha256, index.companies[entry.id].sha256);
    const history = runtime.decodeFinancialCompany(JSON.parse(bytes), index, entry.id);
    const effective = { ...entry, ...taxonomy.classifications[entry.id] };
    const built = runtime.buildStarterValuation(effective, history, index, asOf);
    const { evidence: e, draft } = built, result = runtime.calculateValuation(draft);
    const manual = effective.sector_id === '1' && !['75','76'].includes(effective.branch_id);
    counts.listings++;
    counts.decodedAnnualRows += history.annual.length; counts.decodedQuarterlyRows += history.quarterly.length; counts.decodedWithheldRows += history.withheld.length;
    counts.effectiveClassificationChanges += Number(entry.sector_id !== effective.sector_id || entry.branch_id !== effective.branch_id);
    counts.manualFinancials += Number(manual);
    counts.propertyExceptions += Number(effective.sector_id === '1' && !manual);
    counts[e.uncertainty.status]++;
    counts.valuationReady += Number(result.ready);
    counts.missingMarketValue += Number(draft.marketCap === null);
    counts.allCashYearsAvailable += Number(['low','mid','high'].every(k => draft.scenarios[k].cashFlows.every(Number.isFinite)));
    counts.quoteBasis += Number(e.amountBasis === 'quote');
    counts.historyShort += Number(e.annual.length > 0 && e.annual.length < 5);
    counts.noEligibleAnnualCash += Number(e.annual.length === 0);
    if (e.uncertainty.status === 'historical') bump(groups, e.uncertainty.group);
    bump(reasons, e.uncertainty.reason ?? 'Eligible source-bound historical ranges');
    assert.equal(draft.starterOrigin.id, 'empirical-cash-starter-v3');
    assert.equal(draft.starterOrigin.asOf, asOf);
    assert.equal(draft.starterOrigin.calibrationId, calibration.id);
    assert.equal(e.manualCashRequired, manual);
    assert.equal(draft.crisis, undefined, 'Crisis is initially disabled');
    assert.equal(e.projection, 'latest');
    assert.equal(e.historyYears, 5);
    for (const item of e.annual) {
      const source = history.annual.find(x => x.year === item.year && x.source_id === item.sourceId);
      assert.ok(source && (source.report_date === null || source.report_date <= asOf) && source.end <= asOf);
      assert.ok(source.currency_ratio > 0);
      const nativeCash = source.raw.free_cash_flow / source.currency_ratio;
      close(item.reportedCashFlow, nativeCash, `${entry.id} FY${item.year} native conversion`);
      close(item.cashFlow, e.amountBasis === 'reporting' ? nativeCash : source.raw.free_cash_flow, 'Cash basis');
      counts.nativeCashCellsVerified++;
    }
    const classification = independentClass(e.annual.map(x => x.rawCashFlow / x.currencyRatio));
    close(e.uncertainty.scale, classification.scale, 'History scale');
    close(e.uncertainty.dispersion, classification.dispersion, 'Signed cash dispersion');
    assert.equal(e.uncertainty.group, classification.group);
    const mid = manual ? null : e.annual[0]?.cashFlow ?? null;
    close(e.anchor, mid, 'Latest default anchor');
    counts.negativeMid += Number(mid !== null && mid < 0);
    counts.zeroMid += Number(mid === 0);
    counts.zeroMidHistorical += Number(mid === 0 && e.uncertainty.status === 'historical');
    if (manual) {
      assert.equal(e.uncertainty.status, 'unavailable');
      assert.equal(result.ready, false);
      for (const s of Object.values(draft.scenarios)) { assert.ok(s.cashFlows.every(x => x === null)); assert.equal(s.terminalEquity, null); }
    }
    if (e.uncertainty.status === 'historical') {
      assert.equal(e.amountBasis, 'reporting'); assert.equal(e.annual.length, 5);
      assert.ok(/^(?:[1-9]|10)$/.test(effective.sector_id) && !manual && classification.scale > 0);
      assert.equal(e.uncertainty.targetCoverage, .8);
      assert.equal(e.uncertainty.calibrationId, calibration.id);
      assert.ok(e.annual[0].year > calibration.calibrationLastTargetYear && e.annual[0].reportDate >= calibration.calibrationCutoff);
      for (const [i,input] of e.annual.entries()) {
        assert.ok(calibration.sourceFiles.some(s => s.id === input.sourceId && s.sha256 === input.sourceHash && s.path === input.sourcePath && s.as_of === input.sourceAsOf));
        assert.match(input.reportDate ?? '', /^\d{4}-\d{2}-\d{2}$/);
        assert.ok(input.reportDate >= input.end && input.reportDate <= e.annual[0].reportDate);
        const duration = (Date.parse(input.end)-Date.parse(input.start))/86400000+1;
        assert.ok(duration >= 330 && duration <= 400);
        assert.equal(input.reportingCurrency,e.annual[0].reportingCurrency);
        if(i) {
          const gap = (Date.parse(e.annual[i-1].start)-Date.parse(input.end))/86400000;
          assert.ok(gap >= 1 && gap <= 35);
          assert.equal(input.year,e.annual[i-1].year-1);
        }
        assert.ok(!history.withheld.some(x => x.period === 5 && x.year === input.year));
      }
    }
    for (let i = 0; i < 10; i++) {
      const f = e.forecast[i], year = i + 1;
      close(f.mid, mid, 'Latest signed cash stays flat');
      const factor = e.uncertainty.status === 'historical' ? independentFactor('naive', Math.min(year,4), classification.group) : null;
      const halfWidth = mid === null ? null : factor ? factor.factor * classification.scale + Math.max(0,year-4) * .1 * classification.scale : mid === 0 ? null : Math.abs(mid) * (10 + 10*i) / 100;
      close(f.halfWidth, halfWidth, 'Independent absolute half-width');
      close(f.low, halfWidth === null ? null : mid - halfWidth, 'Low cash');
      close(f.high, halfWidth === null ? null : mid + halfWidth, 'High cash');
      assert.equal(f.rangeBasis, halfWidth === null ? 'unavailable' : factor ? year <= 4 ? 'historical' : 'assumed-tail' : 'percentage');
      if (f.rangeBasis === 'historical') {
        counts.historicalYearCellsVerified++;
        close(f.factor, factor.factor, 'Exact saved factor');
        assert.equal(f.calibrationGroup, factor.group);
        assert.deepEqual(f.support, { count:factor.count, listings:factor.listings, histories:factor.histories });
      } else if (f.rangeBasis === 'assumed-tail') { counts.assumedTailYearCellsVerified++; assert.equal(f.factor, null); assert.equal(f.support, null); }
      else if (f.rangeBasis === 'percentage') counts.percentageYearCellsVerified++;
      for (const key of ['low','mid','high']) close(draft.scenarios[key].cashFlows[i], f[key], 'Chart/draft same cash');
    }
    for (const key of ['low','mid','high']) {
      const scenario = draft.scenarios[key], lastCash = scenario.cashFlows.at(-1);
      close(scenario.terminalEquity,lastCash === null ? null : Math.max(0,lastCash)/.1,'Default terminal sale assumption');
      assert.equal(scenario.discountRate,10);
      if(result.ready) {
        const calculated = result.scenarios[key];
        let cashPV = 0, cumulativeCash = 0;
        close(calculated.cumulativeNPV[0],-draft.marketCap,'Year-zero NPV');
        for(const [i,cash] of scenario.cashFlows.entries()) {
          const discounted = cash/(1.1**(i+1));
          cashPV += discounted; cumulativeCash += cash;
          close(calculated.discounted[i],discounted,'Annual discounted cash');
          close(calculated.cumulativeCash[i+1],cumulativeCash,'Cumulative nominal cash');
          close(calculated.cumulativeNPV[i+1],cashPV-draft.marketCap,'Cumulative operating-cash NPV');
        }
        const terminalPV = scenario.terminalEquity/(1.1**10), value = cashPV+terminalPV;
        close(calculated.cashPV,cashPV,'Cash present value');
        close(calculated.terminalPV,terminalPV,'Terminal present value');
        close(calculated.value,value,'DCF value');
        close(calculated.npv,value-draft.marketCap,'DCF NPV');
        close(calculated.cumulativeNPVWithSale[10],value-draft.marketCap,'Cumulative NPV including terminal');
        counts.dcfNpvScenariosVerified++;
      }
    }
    for (const version of [1,2]) {
      const settings = version === 1 ? { ...baseSettings, method:'weighted-cash-starter-v1',projection:'flat',spreadPercent:20,spreadStepPercent:0 } : baseSettings;
      const old = legacy.buildStarterValuation(entry, history, index, asOf, settings);
      const reproduced = runtime.buildStarterValuation(entry, history, index, asOf, settings);
      assert.deepEqual(reproduced.draft, old.draft, `${entry.id}: full v${version} draft parity`);
      assert.deepEqual(e.annual,old.evidence.annual,'Original annual source selection retained');
      assert.equal(e.amountBasis,old.evidence.amountBasis,'Original source currency basis retained');
      assert.deepEqual(e.price,old.evidence.price,'Original dated price and ownership lineage retained');
      counts[`legacyV${version}FullDraftEqual`]++;
      assert.equal(runtime.isUntouchedLegacyStarter(old.draft, reproduced.draft), true);
      counts.untouchedLegacyDefaultsRecognized++;
    }
    const item = { id:entry.id, name:entry.display_name, sourceSector:entry.sector_id, sourceBranch:entry.branch_id, sector:effective.sector_id, branch:effective.branch_id,
      currency:e.currency, amountBasis:e.amountBasis, annualYears:e.annual.map(x => x.year), signedCash:e.annual.map(x => x.cashFlow), signedNativeCash:e.annual.map(x => x.reportedCashFlow),
      mid, scale:classification.scale, dispersion:classification.dispersion, group:classification.group, status:e.uncertainty.status, reason:e.uncertainty.reason, manual, ready:result.ready,
      marketCap:draft.marketCap, priceDate:draft.priceDate, forecast:e.forecast.filter(x => [1,4,5,10].includes(x.year)), companyPayloadSha256:row.sha256 };
    ledger.push(item);
    sample(e.uncertainty.status, item);
    if (e.amountBasis === 'quote') sample('quoteCurrency', item);
    if (e.annual.length > 0 && e.annual.length < 5) sample('shortHistory', item);
    if (mid === 0) sample(e.uncertainty.status === 'historical' ? 'zeroMidHistorical' : 'zeroMidFallback', item);
    if (mid !== null && mid < 0) sample('negativeMid', item);
    if (manual) sample('manualFinancials', item);
    if (['3','102','197','696'].includes(entry.id)) anchors.push(item);
    if (entry.id === '696') stora = {entry:effective,history,built};
    if (counts.listings % 4000 === 0) process.stdout.write(`Audited ${counts.listings}/${entries.length} listings\n`);
  }
  assert.ok(stora);
  // Explicit guard fixtures change only in-memory references, never archived source bytes.
  const guards = [
    ['unknown sector', {entry:{...stora.entry,sector_id:null,branch_id:null}}],
    ['unrecognized sector', {entry:{...stora.entry,sector_id:'999',branch_id:null}}],
    ['changed pack identity', {index:{...index,id:'0'.repeat(64)}}],
    ['changed taxonomy identity', {index:{...index,taxonomy_sha256:'0'.repeat(64)}}],
    ['changed annual source identity', {index:{...index,sources:index.sources.map(s => ({...s,sha256:'0'.repeat(64)}))}}],
    ['unavailable calibration version', {settings:{...runtime.defaultStarterSettings,calibrationId:'unavailable-version'}}],
    ['custom weights', {settings:{...runtime.defaultStarterSettings,weights:[20,20,20,20,20]}}],
    ['weighted flat model', {settings:{...runtime.defaultStarterSettings,projection:'flat'}}],
    ['ten-year history', {settings:{...runtime.defaultStarterSettings,historyYears:10,weights:[19,17,15,13,11,9,7,5,3,1]}}],
  ].map(([name,changed]) => {
    const built = runtime.buildStarterValuation(changed.entry ?? stora.entry, stora.history, changed.index ?? index, asOf, changed.settings);
    assert.equal(built.evidence.uncertainty.status, 'percentage', name);
    assert.ok(built.evidence.forecast.every(x => x.rangeBasis === 'percentage'));
    return {name,status:built.evidence.uncertainty.status,reason:built.evidence.uncertainty.reason};
  });
  // The optional weighted trend must use linear-model factors and its own fitted midline.
  const trend = runtime.buildStarterValuation(stora.entry, stora.history, index, asOf, {...runtime.defaultStarterSettings,projection:'trend'});
  const v2trend = legacy.buildStarterValuation(stora.entry, stora.history,index,asOf,baseSettings);
  assert.equal(trend.evidence.uncertainty.status,'historical');
  for(const f of trend.evidence.forecast) {
    close(f.mid,v2trend.evidence.forecast[f.year-1].mid,'Optional historical trend mid unchanged');
    const q = independentFactor('linear',Math.min(f.year,4),trend.evidence.uncertainty.group);
    close(f.halfWidth,q.factor*trend.evidence.uncertainty.scale+Math.max(0,f.year-4)*.1*trend.evidence.uncertainty.scale,'Optional trend model-specific width');
  }
  assert.equal(counts.listings,counts.historical+counts.percentage+counts.unavailable);
  assert.equal(counts.historical,Object.values(groups).reduce((a,b)=>a+b,0));
  assert.equal(counts.historicalYearCellsVerified,counts.historical*4);
  assert.equal(counts.assumedTailYearCellsVerified,counts.historical*6);
  assert.equal(counts.dcfNpvScenariosVerified,counts.valuationReady*3);
  assert.equal(counts.decodedAnnualRows,index.summary.annual);
  assert.equal(counts.decodedQuarterlyRows,index.summary.quarterly);
  assert.equal(counts.decodedWithheldRows,index.summary.withheld);
  db.close();
  assert.deepEqual(sources.map(file),before,'Source files unchanged through audit');
  const ledgerPath = resolve(out,'universe-starters.jsonl.gz');
  writeFileSync(ledgerPath,gzipSync(ledger.map(x=>JSON.stringify(x)).join('\n')+'\n'));
  const receipt = {status:'passed',asOf,createdAt:new Date().toISOString(),elapsedSeconds:(Date.now()-started)/1000,
    scope:'Every saved listing, using the application company directory, effective taxonomy, read-only validated financial pack and production starter/valuation runtime. Generic starter generation only; reviewed studies and user drafts retain application precedence, verified separately by browser E2E.',
    method:'Independent signed raw/ratio cash scale, population dispersion, exact immutable naive factors/support, source/timing eligibility, absolute tail arithmetic, latest-cash midlines, chart/draft agreement and DCF/NPV arithmetic for every ready path; full pre-edit v1/v2 draft object comparisons and source/price selection parity for every listing.',
    limitations:['Listing counts are not unique issuers.','This is implementation/source integrity verification, not a new forecast accuracy experiment.','No user profile or saved draft storage was opened. Backup ordering and reviewed-study precedence are browser-E2E responsibilities.','Historical ranges retain their 80% research target; Years 5–10 are explicit assumptions.','Synthetic guard fixtures only alter in-memory lineage/settings; real history, zero, negative, short and quote examples come from the retained pack.'],
    calibration:{id:calibration.id,sha256:before[4].sha256,targetCoverage:calibration.targetCoverage,covidTreatment:calibration.covidTreatment},
    counts,groups,reasons,anchors,examples,guards,optionalTrendAnchor:{id:'696',forecast:trend.evidence.forecast},
    inputs:before,outputs:[file(runtimePath),file(legacyPath),file(ledgerPath)],script:file(fileURLToPath(import.meta.url))};
  writeFileSync(resolve(out,'universe-runtime-audit.json'),JSON.stringify(receipt,null,2)+'\n');
  process.stdout.write(JSON.stringify({status:receipt.status,elapsedSeconds:receipt.elapsedSeconds,counts,groups,receipt:'test-results/empirical-cash-starter-2026-09-13/universe-runtime-audit.json'},null,2)+'\n');
} catch(error) {
  try { db.close(); } catch {}
  writeFileSync(resolve(out,'universe-runtime-audit-failure.json'),JSON.stringify({status:'failed',id:activeId,message:error.message,stack:error.stack,counts},null,2)+'\n');
  throw error;
}
