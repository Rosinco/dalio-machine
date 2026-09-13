// Offline research only. The source pack is a retrospective merged vintage, not
// a point-in-time archive. This script does not change the app or its defaults.
import { createHash } from 'node:crypto';
import { closeSync, createWriteStream, mkdirSync, openSync, readFileSync, readSync, statSync, writeFileSync } from 'node:fs';
import { once } from 'node:events';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createGzip, gunzipSync } from 'node:zlib';
import { DatabaseSync } from 'node:sqlite';

export const PROTOCOL = Object.freeze({
  id: 'retrospective-native-cash-backtest-v1', vintage: 'retrospective-current-vintage-not-point-in-time',
  windows: [5, 10], horizons: [1, 2, 3, 4, 5], models: ['naive', 'flat', 'damped', 'linear'],
  calibrationLastTargetYear: 2020, validationTargetYears: [2021, 2022, 2023], testTargetYears: [2024, 2025],
  calibrationFreezeDate: '2021-06-30', calibrationFirstApplicableOriginYear: 2021,
  calibrationCoverageTarget: 0.8, calibrationMinimum: 30, defaultSpreadPerHorizon: 0.1,
  currencyBasis: 'Native reporting currency: saved raw free_cash_flow divided by its own positive currency_ratio. No price or FX-matched-market eligibility.',
  dampedDefinition: 'Weighted fitted intercept + 0.5 × weighted fitted slope × horizon; intercept is unchanged.',
  normalization: 'Unweighted mean absolute training cash; zero scale is missing, never floored. MASE uses unweighted mean absolute adjacent training differences.',
  calibration: 'Nearest-rank 80th percentile of absolute residual / training mean absolute cash, separately by window/model/horizon. Only target FY≤2020 with known publication≤2021-06-30. Both calibration and application require all training publications known and≤origin publication, which must precede target fiscal end. Apply only to origins FY≥2021 with origin publication≥freeze date.',
  intervalScore: 'Width + (2 / 0.2) × distance outside the interval, divided by training mean absolute cash. Nominal-80% interval score is used for comparison; the fixed sensitivity has no established probability level.',
  comparison: 'Models share identical eligible folds within each window. Compare history lengths only in the separately reported common ten-year-eligible cohort. No model is selected or deployed from these results.',
  dependence: 'The observation unit is a listing/origin/horizon. Share classes, overlapping training histories, repeated targets across horizons and firms are dependent. Counts are descriptive, not independent trials.',
  limitations: [
    'Latest retained rows may contain later revisions; publication metadata does not restore the historical source vintage.',
    'The current listing directory can omit former or delisted issuers. Duplicate share classes are not deduplicated into issuers.',
    'Annual model steps begin after the fitted fiscal origin; no interim cash bridge or investment-return test is implied.',
    'Saved provider FCF is an unreviewed proxy, not verified distributable equity cash or universally comparable FCFF, especially for financial companies.',
    'Reporting currencies must remain identical through training and outcome. This omits the application’s historical quote-currency fallback.',
    'Recent test has at most two target fiscal years per company/horizon; company hits and misses are descriptions, not stable accuracy classes.',
    'Audit flags identify numerical patterns and review hypotheses; they do not establish business causes.',
    'Calibrated bands are a research comparison with an assumed 80% target, not a probability or coverage guarantee.',
  ],
});
export const weightsFor = n => n === 5 ? [30, 25, 20, 15, 10] : n === 10 ? [19, 17, 15, 13, 11, 9, 7, 5, 3, 1] : (() => { throw new Error('History window must be five or ten years.'); })();
const finite = x => typeof x === 'number' && Number.isFinite(x);
const day = x => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x;
const days = (later, earlier) => (Date.parse(later) - Date.parse(earlier)) / 86400000;
const mean = xs => xs.length ? xs.reduce((sum, x) => sum + x, 0) / xs.length : null;
const median = xs => { if (!xs.length) return null; const sorted = [...xs].sort((a, b) => a - b), i = Math.floor(sorted.length / 2); return sorted.length % 2 ? sorted[i] : (sorted[i - 1] + sorted[i]) / 2; };
const normalize = (amount, scale) => finite(amount) && finite(scale) && scale > 0 ? amount / scale : null;
export const splitForYear = year => year <= 2020 ? 'calibration' : year <= 2023 ? 'validation' : year <= 2025 ? 'test' : null;

export function decodeAnnual(rows, index) {
  const cashColumn = index.columns.indexOf('free_cash_flow');
  if (cashColumn < 0) throw new Error('Source pack has no saved free_cash_flow column.');
  const sources = new Map(index.sources.map(s => [s.id, s]));
  return rows.map(r => {
    const [year, period, start, end, reportDate, currency, currencyRatio, sourceId] = r;
    if (!Array.isArray(r) || r.length !== 8 + index.columns.length || !sources.has(sourceId)) throw new Error('Malformed or unbound annual source row.');
    const rawCash = r[8 + cashColumn];
    const cash = finite(rawCash) && finite(currencyRatio) && currencyRatio > 0 ? rawCash / currencyRatio : null;
    return { year, period, start, end, reportDate, currency, currencyRatio, rawCash, cash: finite(cash) ? cash : null,
      rawAmounts: r.slice(8), sourceId, sourceAsOf: sources.get(sourceId).as_of };
  });
}

export function annualIssue(row) {
  if (!row) return 'missing';
  if (row.period !== 5 || !Number.isSafeInteger(row.year) || !day(row.start) || !day(row.end)) return 'invalid_period';
  const length = days(row.end, row.start) + 1;
  if (length < 330 || length > 400) return 'not_full_annual';
  if (!/^[A-Z]{3}$/.test(row.currency ?? '')) return 'currency_missing';
  if (!finite(row.currencyRatio) || row.currencyRatio <= 0) return 'currency_ratio_missing';
  if (row.rawAmounts.some(x => x === 0) && row.rawAmounts.every(x => x === 0 || x === null)) return 'all_amounts_zero';
  if (!finite(row.cash)) return 'cash_missing';
  return null;
}

export function fitWeighted(training, window) {
  if (training.length !== window) throw new Error('Backtests require the complete requested training window.');
  const weights = weightsFor(window), latest = training[0], weightSum = weights.reduce((a, b) => a + b, 0);
  const xMean = training.reduce((a, r, i) => a + (r.year - latest.year) * weights[i], 0) / weightSum;
  const weightedMean = training.reduce((a, r, i) => a + r.cash * weights[i], 0) / weightSum;
  const covariance = training.reduce((a, r, i) => a + weights[i] * (r.year - latest.year - xMean) * (r.cash - weightedMean), 0);
  const variance = training.reduce((a, r, i) => a + weights[i] * (r.year - latest.year - xMean) ** 2, 0);
  const slope = variance === 0 ? 0 : covariance / variance, intercept = weightedMean - slope * xMean;
  const meanAbs = mean(training.map(r => Math.abs(r.cash)));
  const meanAbsDiff = mean(training.slice(1).map((r, i) => Math.abs(training[i].cash - r.cash)));
  const unweightedMean = mean(training.map(r => r.cash));
  const volatility = normalize(Math.sqrt(mean(training.map(r => (r.cash - unweightedMean) ** 2))), meanAbs);
  return { weights, latest: latest.cash, mean: weightedMean, slope, intercept, xMean, meanAbs, meanAbsDiff, volatility };
}

export function predict(fit, model, horizon) {
  if (model === 'naive') return fit.latest;
  if (model === 'flat') return fit.mean;
  if (model === 'damped') return fit.intercept + 0.5 * fit.slope * horizon;
  if (model === 'linear') return fit.intercept + fit.slope * horizon;
  throw new Error(`Unsupported forecast model: ${model}`);
}

export function publicationEligibility(fold) {
  const origin = fold.training[0];
  if (!day(origin.reportDate)) return 'origin_publication_missing';
  if (fold.training.some(r => !day(r.reportDate))) return 'training_publication_missing';
  if (fold.training.some(r => r.reportDate < r.end)) return 'training_publication_before_period_end';
  if (day(fold.target.reportDate) && fold.target.reportDate < fold.target.end) return 'target_publication_before_period_end';
  if (fold.training.some(r => r.reportDate > origin.reportDate)) return 'training_publication_after_origin';
  if (origin.reportDate >= fold.target.end) return 'origin_publication_not_before_target_end';
  return null;
}
export function calibratedEligibility(fold) {
  if (fold.originYear < PROTOCOL.calibrationFirstApplicableOriginYear) return 'origin_before_freeze_cohort';
  const issue = publicationEligibility(fold);
  if (issue) return issue;
  if (fold.training[0].reportDate < PROTOCOL.calibrationFreezeDate) return 'origin_publication_before_freeze';
  return null;
}

export function buildCompanyFolds(company, options = {}) {
  const windows = options.windows ?? PROTOCOL.windows, horizons = options.horizons ?? PROTOCOL.horizons, maxTargetYear = options.maxTargetYear ?? 2025;
  const byYear = new Map(), duplicates = new Set();
  for (const r of company.annual) { if (byYear.has(r.year)) duplicates.add(r.year); byYear.set(r.year, r); }
  const withheld = new Set((company.withheld ?? []).filter(r => r.period === 5).map(r => r.year));
  const origins = [...new Set([...byYear.keys(), ...withheld])].filter(y => y < maxTargetYear).sort((a, b) => a - b);
  const folds = [], exclusions = [];
  const consecutive = (older, newer) => newer.year === older.year + 1 && days(newer.start, older.end) >= 1 && days(newer.start, older.end) <= 35;
  const issueFor = year => duplicates.has(year) ? 'duplicate_year' : withheld.has(year) ? 'withheld' : annualIssue(byYear.get(year));
  for (const originYear of origins) for (const window of windows) {
    const training = []; let trainingIssue = null;
    for (let offset = 0; offset < window; offset++) {
      const year = originYear - offset, row = byYear.get(year), issue = issueFor(year);
      if (issue) { trainingIssue = `training_${issue}`; break; }
      if (training.length && (!consecutive(row, training.at(-1)) || row.currency !== training[0].currency)) { trainingIssue = row.currency !== training[0].currency ? 'training_currency_change' : 'training_gap_or_overlap'; break; }
      training.push(row);
    }
    const fit = trainingIssue ? null : fitWeighted(training, window);
    for (const horizon of horizons) {
      const targetYear = originYear + horizon;
      if (targetYear > maxTargetYear) continue;
      const split = splitForYear(targetYear);
      if (!split) continue;
      const identity = { companyId: company.id, window, horizon, originYear, targetYear, split };
      if (trainingIssue) { exclusions.push({ ...identity, stage: 'training', reason: trainingIssue }); continue; }
      let targetIssue = null, previous = training[0];
      for (let step = 1; step <= horizon; step++) {
        const year = originYear + step, next = byYear.get(year), issue = issueFor(year);
        if (issue) { targetIssue = `${step === horizon ? 'target' : 'intermediate'}_${issue}`; break; }
        if (!consecutive(previous, next) || next.currency !== training[0].currency) { targetIssue = next.currency !== training[0].currency ? 'target_currency_change' : 'target_gap_or_overlap'; break; }
        previous = next;
      }
      if (targetIssue) { exclusions.push({ ...identity, stage: 'target', reason: targetIssue }); continue; }
      const target = byYear.get(targetYear);
      const trainingFlags = [];
      if (fit.volatility !== null && fit.volatility > 1) trainingFlags.push('high_history_volatility');
      if (new Set(training.map(r => Math.sign(r.cash))).size > 1) trainingFlags.push('training_sign_change');
      if (Math.sign(fit.latest) !== Math.sign(fit.mean)) trainingFlags.push('latest_vs_weighted_mean_sign_change');
      if (company.sectorId === '1' || company.sector === 'Financials') trainingFlags.push('financial_sector_proxy');
      if (company.market?.some(m => m.year === originYear && m.source_id === training[0].sourceId && m.currency && m.currency !== training[0].currency)) trainingFlags.push('origin_quote_currency_differs');
      if (training.some(r => !day(r.reportDate)) || !day(target.reportDate)) trainingFlags.push('publication_metadata_missing');
      if (training.some(r => r.sourceAsOf > training[0].end)) trainingFlags.push('source_vintage_after_origin');
      if (new Set([...training, target].map(r => r.sourceId)).size > 1) trainingFlags.push('mixed_source_vintages');
      const fold = { ...identity, currency: target.currency, training, target, fit, trainingFlags, matchedTenYear: false };
      fold.publicationUnavailable = publicationEligibility(fold);
      fold.calibrationUnavailable = calibratedEligibility(fold);
      folds.push(fold);
    }
  }
  const tenKeys = new Set(folds.filter(f => f.window === 10).map(f => `${f.originYear}/${f.targetYear}`));
  folds.forEach(f => { f.matchedTenYear = tenKeys.has(`${f.originYear}/${f.targetYear}`); });
  return { folds, exclusions };
}

export function intervalMetrics(actual, low, high, scale, alpha = 0.2) {
  const outcome = actual < low ? 'below' : actual > high ? 'above' : 'inside';
  const width = high - low;
  const intervalScore = width + 2 / alpha * (actual < low ? low - actual : actual > high ? actual - high : 0);
  return { outcome, width, normalizedWidth: normalize(width, scale), normalizedIntervalScore: normalize(intervalScore, scale) };
}

export function scoreForecast(fold, model, factor) {
  const prediction = predict(fold.fit, model, fold.horizon), actual = fold.target.cash;
  const halfWidth = Math.abs(prediction) * PROTOCOL.defaultSpreadPerHorizon * fold.horizon;
  const low = prediction - halfWidth, high = prediction + halfWidth, error = prediction - actual;
  const meanAbs = fold.fit.meanAbs, meanAbsDiff = fold.fit.meanAbsDiff;
  const calibratedHalfWidth = factor !== null && finite(factor) && meanAbs > 0 && !fold.calibrationUnavailable ? factor * meanAbs : null;
  const calibrated = calibratedHalfWidth === null ? null : { low: prediction - calibratedHalfWidth, high: prediction + calibratedHalfWidth,
    ...intervalMetrics(actual, prediction - calibratedHalfWidth, prediction + calibratedHalfWidth, meanAbs) };
  const flags = [...fold.trainingFlags];
  if (halfWidth === 0) flags.push('zero_width_collapse');
  if (prediction === 0 || meanAbs > 0 && Math.abs(prediction) <= 0.1 * meanAbs) flags.push('near_zero_mid');
  if (Math.sign(actual) !== Math.sign(prediction)) flags.push('forecast_actual_sign_flip');
  if (fold.fit.slope !== 0 && Math.sign(actual - fold.fit.latest) !== 0 && Math.sign(actual - fold.fit.latest) !== Math.sign(fold.fit.slope)) flags.push('actual_change_opposes_trend');
  return { companyId: fold.companyId, window: fold.window, model, horizon: fold.horizon, originYear: fold.originYear,
    targetYear: fold.targetYear, targetReportDate: fold.target.reportDate, split: fold.split, currency: fold.currency,
    prediction, actual, low, high, error, absoluteError: Math.abs(error), meanAbs, meanAbsDiff,
    normalizedAbsoluteError: normalize(Math.abs(error), meanAbs), normalizedBias: normalize(error, meanAbs), mase: normalize(Math.abs(error), meanAbsDiff),
    ...intervalMetrics(actual, low, high, meanAbs), calibrated, calibrationUnavailable: fold.calibrationUnavailable, publicationUnavailable: fold.publicationUnavailable, flags };
}

const calibrationKey = r => `${r.window}/${r.model}/${r.horizon}`;
export function calibrationFactors(scores, { minimum = PROTOCOL.calibrationMinimum } = {}) {
  const groups = new Map();
  for (const row of scores) {
    if (row.targetYear > PROTOCOL.calibrationLastTargetYear) continue;
    const key = calibrationKey(row);
    if (!groups.has(key)) groups.set(key, { window: row.window, model: row.model, horizon: row.horizon, residuals: [], zeroScale: 0, publicationExcluded: 0 });
    const group = groups.get(key);
    if (!day(row.targetReportDate) || row.targetReportDate > PROTOCOL.calibrationFreezeDate || row.publicationUnavailable !== null) { group.publicationExcluded++; continue; }
    const value = normalize(row.absoluteError, row.meanAbs);
    if (value === null) group.zeroScale++; else group.residuals.push(value);
  }
  return [...groups.values()].map(({ residuals, ...group }) => {
    residuals.sort((a, b) => a - b);
    return { ...group, count: residuals.length, minimum, coverageTarget: PROTOCOL.calibrationCoverageTarget,
      factor: residuals.length >= minimum ? residuals[Math.ceil(PROTOCOL.calibrationCoverageTarget * residuals.length) - 1] : null };
  });
}

class ScoreSummary {
  constructor() {
    this.n = 0; this.below = 0; this.inside = 0; this.above = 0; this.ids = new Set(); this.years = new Set();
    this.normalizedErrors = []; this.mases = []; this.biasSum = 0; this.widthSum = 0; this.scoreSum = 0;
    this.currencies = {}; this.calibratedCount = 0; this.calibratedBelow = 0; this.calibratedInside = 0; this.calibratedAbove = 0;
    this.calibratedWidthSum = 0; this.calibratedScoreSum = 0; this.flags = {}; this.calibrationUnavailable = {};
    this.calibratedIds = new Set(); this.calibratedYears = new Set(); this.rawOnCalibratedBelow = 0; this.rawOnCalibratedInside = 0; this.rawOnCalibratedAbove = 0;
    this.rawOnCalibratedWidthSum = 0; this.rawOnCalibratedScoreSum = 0;
  }
  add(row) {
    this.n++; this[row.outcome]++; this.ids.add(row.companyId); this.years.add(row.targetYear);
    if (row.normalizedAbsoluteError !== null) { this.normalizedErrors.push(row.normalizedAbsoluteError); this.biasSum += row.normalizedBias; this.widthSum += row.normalizedWidth; this.scoreSum += row.normalizedIntervalScore; }
    if (row.mase !== null) this.mases.push(row.mase);
    const currency = this.currencies[row.currency] ??= { count: 0, absoluteErrorSum: 0, biasSum: 0 };
    currency.count++; currency.absoluteErrorSum += row.absoluteError; currency.biasSum += row.error;
    if (row.calibrated) {
      this.calibratedCount++; this[`calibrated${row.calibrated.outcome[0].toUpperCase()}${row.calibrated.outcome.slice(1)}`]++;
      this.calibratedWidthSum += row.calibrated.normalizedWidth; this.calibratedScoreSum += row.calibrated.normalizedIntervalScore;
      this.calibratedIds.add(row.companyId); this.calibratedYears.add(row.targetYear);
      this[`rawOnCalibrated${row.outcome[0].toUpperCase()}${row.outcome.slice(1)}`]++;
      this.rawOnCalibratedWidthSum += row.normalizedWidth; this.rawOnCalibratedScoreSum += row.normalizedIntervalScore;
    } else { const reason = row.calibrationUnavailable ?? (row.meanAbs > 0 ? 'calibration_factor_unavailable' : 'zero_training_scale'); this.calibrationUnavailable[reason] = (this.calibrationUnavailable[reason] ?? 0) + 1; }
    for (const flag of row.flags) this.flags[flag] = (this.flags[flag] ?? 0) + 1;
  }
  result() {
    const normalizedCount = this.normalizedErrors.length;
    return { count: this.n, companies: this.ids.size, targetYears: [...this.years].sort((a, b) => a - b), below: this.below, inside: this.inside, above: this.above,
      coverage: this.n ? this.inside / this.n : null, normalizedCount, medianNormalizedAbsoluteError: median(this.normalizedErrors),
      meanNormalizedAbsoluteError: mean(this.normalizedErrors), meanNormalizedBias: normalizedCount ? this.biasSum / normalizedCount : null,
      meanNormalizedWidth: normalizedCount ? this.widthSum / normalizedCount : null, meanNormalizedIntervalScore: normalizedCount ? this.scoreSum / normalizedCount : null,
      maseCount: this.mases.length, medianMase: median(this.mases),
      maeByCurrency: Object.fromEntries(Object.entries(this.currencies).map(([code, x]) => [code, { count: x.count, mae: x.absoluteErrorSum / x.count, meanBias: x.biasSum / x.count }])),
      calibratedCount: this.calibratedCount, calibratedBelow: this.calibratedBelow, calibratedInside: this.calibratedInside, calibratedAbove: this.calibratedAbove,
      calibratedCoverage: this.calibratedCount ? this.calibratedInside / this.calibratedCount : null,
      calibratedMeanNormalizedWidth: this.calibratedCount ? this.calibratedWidthSum / this.calibratedCount : null,
      calibratedMeanNormalizedIntervalScore: this.calibratedCount ? this.calibratedScoreSum / this.calibratedCount : null,
      calibratedCompanies: this.calibratedIds.size, calibratedTargetYears: [...this.calibratedYears].sort((a, b) => a - b),
      rawOnCalibratedCount: this.calibratedCount, rawOnCalibratedBelow: this.rawOnCalibratedBelow, rawOnCalibratedInside: this.rawOnCalibratedInside, rawOnCalibratedAbove: this.rawOnCalibratedAbove,
      rawOnCalibratedCoverage: this.calibratedCount ? this.rawOnCalibratedInside / this.calibratedCount : null,
      rawOnCalibratedMeanNormalizedWidth: this.calibratedCount ? this.rawOnCalibratedWidthSum / this.calibratedCount : null,
      rawOnCalibratedMeanNormalizedIntervalScore: this.calibratedCount ? this.rawOnCalibratedScoreSum / this.calibratedCount : null,
      calibrationUnavailable: this.calibrationUnavailable, auditFlags: this.flags };
  }
}
export function summarizeScores(scores) { const summary = new ScoreSummary(); scores.forEach(row => summary.add(row)); return summary.result(); }

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
function fileHash(path) {
  const file = openSync(path, 'r'), hash = createHash('sha256'), buffer = Buffer.alloc(65536);
  try { let n; while ((n = readSync(file, buffer, 0, buffer.length, null))) hash.update(buffer.subarray(0, n)); } finally { closeSync(file); }
  return hash.digest('hex');
}
function openPack(path, taxonomyPath) {
  const sha256 = fileHash(path);
  if (/^[a-f0-9]{64}\.sqlite$/.test(basename(path)) && basename(path) !== `${sha256}.sqlite`) throw new Error('Financial pack checksum mismatch.');
  const db = new DatabaseSync(path, { readOnly: true, enableDoubleQuotedStringLiterals: false, allowExtension: false });
  db.exec('PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;');
  const index = JSON.parse(gunzipSync(db.prepare('SELECT payload FROM metadata WHERE key=?').get('index').payload));
  if (index.format !== 'macro-atlas-financials' || ![1, 2].includes(index.version)) throw new Error('Unsupported financial pack.');
  const taxonomyBytes = readFileSync(taxonomyPath), taxonomyHash = digest(taxonomyBytes);
  if (taxonomyHash !== index.taxonomy_sha256) throw new Error('Taxonomy does not match the selected financial pack.');
  const taxonomy = JSON.parse(taxonomyBytes), query = db.prepare('SELECT payload, sha256 FROM companies WHERE id=?');
  return { db, index, taxonomy, sha256, taxonomyHash,
    company(id) {
      const row = query.get(id), bytes = gunzipSync(row.payload, { maxOutputLength: 2_000_000 });
      if (digest(bytes) !== row.sha256 || row.sha256 !== index.companies[id].sha256) throw new Error(`Company ${id} source checksum mismatch.`);
      const packed = JSON.parse(bytes);
      if (packed.id !== id) throw new Error('Company source identity mismatch.');
      const entry = taxonomy.catalogue?.listings[id] ?? {}, classification = taxonomy.classifications[id] ?? {};
      const sectorId = classification.sector_id ?? entry.sector_id ?? null, branchId = classification.branch_id ?? entry.branch_id ?? null;
      return { id, name: entry.name ?? id, isin: entry.isin ?? null, sectorId, sector: taxonomy.sectors[sectorId]?.name_en ?? null,
        branchId, branch: taxonomy.branches[branchId]?.name_en ?? null, annual: decodeAnnual(packed.annual, index),
        withheld: packed.withheld, market: packed.market ?? [], sourceHash: row.sha256 };
    } };
}

export const FORECAST_COLUMNS = ['company_id', 'company_name', 'isin', 'sector', 'branch', 'currency', 'window', 'model', 'horizon', 'split', 'origin_year', 'origin_end', 'origin_publication', 'target_year', 'target_start', 'target_end', 'target_publication', 'train_years', 'train_end_dates', 'train_publications', 'train_source_ids', 'train_source_as_of', 'train_currency_ratios', 'train_cash', 'target_source_id', 'target_source_as_of', 'target_currency_ratio', 'latest_cash', 'weighted_mean', 'intercept', 'slope', 'training_mean_absolute_cash', 'training_mean_absolute_difference', 'history_volatility', 'prediction', 'actual', 'lower', 'upper', 'outcome', 'absolute_error', 'normalized_absolute_error', 'normalized_bias', 'mase', 'normalized_width', 'normalized_interval_score', 'calibrated_factor', 'calibrated_lower', 'calibrated_upper', 'calibrated_outcome', 'calibrated_normalized_width', 'calibrated_normalized_interval_score', 'calibration_unavailable', 'matched_ten_year', 'audit_flags', 'company_source_sha256'];
const csvCell = x => { const text = x === null || x === undefined ? '' : typeof x === 'object' ? JSON.stringify(x) : String(x); return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; };
const csvLine = xs => `${xs.map(csvCell).join(',')}\n`;
function gzipWriter(path) {
  const gzip = createGzip({ level: 6 }), destination = createWriteStream(path);
  gzip.pipe(destination);
  return { async write(text) { if (!gzip.write(text)) await once(gzip, 'drain'); }, async close() { gzip.end(); await once(destination, 'finish'); } };
}
function addSummary(groups, key, score, dimensions) {
  if (!groups.has(key)) groups.set(key, { dimensions, summary: new ScoreSummary() });
  groups.get(key).summary.add(score);
}
function groupDimensions(row) { return { split: row.split, window: row.window, model: row.model, horizon: row.horizon }; }
const groupKey = row => `${row.split}/${row.window}/${row.model}/${row.horizon}`;
const summarizeGroups = groups => [...groups.values()].map(({ dimensions, summary }) => ({ ...dimensions, ...summary.result() }));
function forecastCells(company, fold, row, factor) {
  const train = fold.training;
  return [company.id, company.name, company.isin, company.sector, company.branch, fold.currency, fold.window, row.model, fold.horizon, fold.split,
    fold.originYear, train[0].end, train[0].reportDate, fold.targetYear, fold.target.start, fold.target.end, fold.target.reportDate,
    train.map(r => r.year).join('|'), train.map(r => r.end).join('|'), train.map(r => r.reportDate ?? '').join('|'), train.map(r => r.sourceId).join('|'), train.map(r => r.sourceAsOf).join('|'), train.map(r => r.currencyRatio).join('|'), train.map(r => r.cash).join('|'),
    fold.target.sourceId, fold.target.sourceAsOf, fold.target.currencyRatio, fold.fit.latest, fold.fit.mean, fold.fit.intercept, fold.fit.slope, fold.fit.meanAbs, fold.fit.meanAbsDiff, fold.fit.volatility,
    row.prediction, row.actual, row.low, row.high, row.outcome, row.absoluteError, row.normalizedAbsoluteError, row.normalizedBias, row.mase, row.normalizedWidth, row.normalizedIntervalScore,
    row.calibrated ? factor : null, row.calibrated?.low, row.calibrated?.high, row.calibrated?.outcome, row.calibrated?.normalizedWidth, row.calibrated?.normalizedIntervalScore,
    row.calibrated ? null : row.calibrationUnavailable ?? (row.meanAbs > 0 ? 'calibration_factor_unavailable' : 'zero_training_scale'), fold.matchedTenYear, row.flags.join('|'), company.sourceHash];
}

export async function runBacktest({ packPath, taxonomyPath, outputDir, companyIds = null }) {
  const startedAt = new Date().toISOString(), pack = openPack(packPath, taxonomyPath);
  const ids = (companyIds ?? Object.keys(pack.index.companies)).sort((a, b) => Number(a) - Number(b));
  if (ids.some(id => !pack.index.companies[id])) throw new Error('Requested company is not in this financial pack.');
  mkdirSync(outputDir, { recursive: true });
  try {
    // Calibration is a separate first pass: no validation/test residual enters it.
    function* calibrationRows() {
      for (const id of ids) {
        const { folds } = buildCompanyFolds(pack.company(id), { maxTargetYear: 2020 });
        for (const fold of folds) for (const model of PROTOCOL.models) {
          const prediction = predict(fold.fit, model, fold.horizon);
          yield { window: fold.window, model, horizon: fold.horizon, targetYear: fold.targetYear, targetReportDate: fold.target.reportDate, publicationUnavailable: fold.publicationUnavailable, absoluteError: Math.abs(prediction - fold.target.cash), meanAbs: fold.fit.meanAbs };
        }
      }
    }
    const calibration = calibrationFactors(calibrationRows());
    const factors = new Map(calibration.map(r => [calibrationKey(r), r.factor]));
    writeFileSync(join(outputDir, 'calibration.json'), JSON.stringify({ protocol: PROTOCOL, factors: calibration }, null, 2) + '\n');
    const forecasts = gzipWriter(join(outputDir, 'forecasts.csv.gz')), aggregates = gzipWriter(join(outputDir, 'company-aggregate.csv.gz'));
    await forecasts.write(csvLine(FORECAST_COLUMNS));
    const aggregateColumns = ['company_id', 'company_name', 'isin', 'sector', 'branch', 'split', 'window', 'model', 'horizon', ...Object.keys(new ScoreSummary().result())];
    await aggregates.write(csvLine(aggregateColumns));
    const companyFile = createWriteStream(join(outputDir, 'company-summary.json'));
    companyFile.write('{"model":"linear","window":5,"unit":"listing, split and horizon; targets are not independent","companies":[\n');
    let companyWritten = false, forecastRows = 0, readyCompanies = 0;
    const groups = new Map(), matchedGroups = new Map(), sameSourceGroups = new Map(), flagGroups = new Map(), exclusions = new Map(), coverage = new Map();
    for (let index = 0; index < ids.length; index++) {
      const company = pack.company(ids[index]), { folds, exclusions: rejected } = buildCompanyFolds(company), companyGroups = new Map();
      if (folds.length) readyCompanies++;
      for (const rejectedFold of rejected) {
        const key = `${rejectedFold.split}/${rejectedFold.window}/${rejectedFold.horizon}/${rejectedFold.stage}/${rejectedFold.reason}`;
        if (!exclusions.has(key)) exclusions.set(key, { split: rejectedFold.split, window: rejectedFold.window, horizon: rejectedFold.horizon, stage: rejectedFold.stage, reason: rejectedFold.reason, count: 0, ids: new Set() });
        const tally = exclusions.get(key); tally.count++; tally.ids.add(company.id);
      }
      for (const fold of [...folds, ...rejected]) {
        const key = `${fold.split}/${fold.window}/${fold.horizon}`;
        if (!coverage.has(key)) coverage.set(key, { split: fold.split, window: fold.window, horizon: fold.horizon, candidateOrigins: 0, eligible: 0, trainingUnavailable: 0, targetUnavailable: 0, ids: new Set(), eligibleIds: new Set() });
        const c = coverage.get(key); c.candidateOrigins++; c.ids.add(company.id);
        if (!fold.stage) { c.eligible++; c.eligibleIds.add(company.id); } else c[fold.stage === 'training' ? 'trainingUnavailable' : 'targetUnavailable']++;
      }
      let text = '';
      for (const fold of folds) for (const model of PROTOCOL.models) {
        const factor = factors.get(calibrationKey({ ...fold, model })) ?? null, score = scoreForecast(fold, model, factor), key = groupKey(score), dimensions = groupDimensions(score);
        addSummary(groups, key, score, dimensions); addSummary(companyGroups, key, score, dimensions);
        if (fold.matchedTenYear) addSummary(matchedGroups, key, score, dimensions);
        if (!fold.trainingFlags.includes('mixed_source_vintages')) addSummary(sameSourceGroups, key, score, dimensions);
        if (model === 'linear' && fold.window === 5 && fold.split === 'test') for (const flag of score.flags.filter(f => f !== 'source_vintage_after_origin')) addSummary(flagGroups, `${fold.horizon}/${flag}`, score, { ...dimensions, flag });
        text += csvLine(forecastCells(company, fold, score, factor)); forecastRows++;
      }
      if (text) await forecasts.write(text);
      const metrics = summarizeGroups(companyGroups);
      if (metrics.length) await aggregates.write(metrics.map(row => csvLine([company.id, company.name, company.isin, company.sector, company.branch, ...aggregateColumns.slice(5).map(key => row[key])])).join(''));
      const payload = { id: company.id, name: company.name, isin: company.isin, sector: company.sector, branch: company.branch,
        sourceHash: company.sourceHash, annualCount: company.annual.length, eligibleFoldCount: folds.length,
        metrics: metrics.filter(r => r.window === 5 && r.model === 'linear') };
      if (!companyFile.write(`${companyWritten ? ',\n' : ''}${JSON.stringify(payload)}`)) await once(companyFile, 'drain');
      companyWritten = true;
      if ((index + 1) % 2000 === 0) process.stderr.write(`Backtest ${index + 1}/${ids.length} listings; ${forecastRows} forecast rows\n`);
    }
    await forecasts.close(); await aggregates.close(); companyFile.end('\n]}\n'); await once(companyFile, 'finish');
    const summary = { protocol: PROTOCOL, startedAt, completedAt: new Date().toISOString(),
      inputs: { financialPack: { path: resolve(packPath), sha256: pack.sha256, bytes: statSync(packPath).size, asOf: pack.index.as_of },
        taxonomy: { path: resolve(taxonomyPath), sha256: pack.taxonomyHash }, sources: pack.index.sources,
        script: { path: fileURLToPath(import.meta.url), sha256: fileHash(fileURLToPath(import.meta.url)) } },
      listings: ids.length, listingsWithEligibleFolds: readyCompanies, listingsWithoutEligibleFolds: ids.length - readyCompanies, forecastRows,
      candidateDefinition: 'Each annual or withheld origin year recorded for a current listing, requested window and horizon with target fiscal year≤2025. Missing origin years before/within history are not separate candidates; required window and target gaps remain exclusions. Listings with no recorded annual origin have no candidate folds.',
      coverage: [...coverage.values()].map(({ ids, eligibleIds, ...r }) => ({ ...r, companies: ids.size, eligibleCompanies: eligibleIds.size })),
      exclusions: [...exclusions.values()].map(({ ids, ...r }) => ({ ...r, companies: ids.size })),
      calibration, metrics: summarizeGroups(groups), matchedTenYearMetrics: summarizeGroups(matchedGroups), sameSourceMetrics: summarizeGroups(sameSourceGroups), defaultTestAuditFlagMetrics: summarizeGroups(flagGroups),
      artifacts: ['calibration.json', 'forecasts.csv.gz', 'company-aggregate.csv.gz', 'company-summary.json'] };
    writeFileSync(join(outputDir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
    const receipts = summary.artifacts.concat('summary.json').map(name => ({ name, bytes: statSync(join(outputDir, name)).size, sha256: fileHash(join(outputDir, name)) }));
    writeFileSync(join(outputDir, 'receipt.json'), JSON.stringify({ protocolId: PROTOCOL.id, completedAt: summary.completedAt, sourcePackSha256: pack.sha256, scriptSha256: summary.inputs.script.sha256, artifacts: receipts }, null, 2) + '\n');
    return summary;
  } finally { pack.db.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), get = name => { const i = args.indexOf(name); return i < 0 ? null : args[i + 1]; };
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const catalog = JSON.parse(readFileSync(join(root, 'financial-data/catalog.json')));
  const packPath = get('--pack') ?? join(root, 'financial-data', `${catalog.packs[0].id}.sqlite`);
  const taxonomyPath = get('--taxonomy') ?? join(root, 'public/data/taxonomy.json');
  const outputDir = get('--output') ?? join(root, 'test-results/cash-flow-backtest-2026-09-12');
  const companyIds = get('--companies')?.split(',') ?? null;
  const result = await runBacktest({ packPath, taxonomyPath, outputDir, companyIds });
  process.stdout.write(JSON.stringify({ outputDir, listings: result.listings, forecastRows: result.forecastRows, sourcePackSha256: result.inputs.financialPack.sha256 }) + '\n');
}
