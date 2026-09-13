import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { gzipSync, gunzipSync } from 'node:zlib';
import { annualIssue, buildCompanyFolds, calibrationFactors, decodeAnnual, fitWeighted, intervalMetrics, predict, runBacktest, scoreForecast, splitForYear, summarizeScores } from '../scripts/cash-flow-backtest.mjs';

const row = (year, cash, overrides = {}) => ({ year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, reportDate: `${year + 1}-02-01`, currency: 'EUR', currencyRatio: 2, cash, rawCash: cash * 2, rawAmounts: [cash * 2, 10], sourceId: 'saved-2026', sourceAsOf: '2026-08-10', ...overrides });
const company = (annual, overrides = {}) => ({ id: '1', name: 'Fixture', annual, withheld: [], market: [], ...overrides });
const options = { windows: [5], horizons: [1], maxTargetYear: 2025 };
const fixture = (start = 2015, end = 2025, f = year => (year - start + 1) * 10) => Array.from({ length: end - start + 1 }, (_, i) => row(start + i, f(start + i)));
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} != ${expected}`);

test('split is determined exclusively by target fiscal year', () => {
  assert.equal(splitForYear(2020), 'calibration');
  assert.equal(splitForYear(2021), 'validation');
  assert.equal(splitForYear(2023), 'validation');
  assert.equal(splitForYear(2024), 'test');
  assert.equal(splitForYear(2025), 'test');
  assert.equal(splitForYear(2026), null);
});

test('independent five-year weighted trend and competing models', () => {
  const fit = fitWeighted(fixture(2015, 2019).reverse(), 5);
  close(fit.mean, 35); close(fit.intercept, 50); close(fit.slope, 10);
  close(fit.meanAbs, 30); close(fit.meanAbsDiff, 10);
  close(predict(fit, 'linear', 2), 70);
  close(predict(fit, 'damped', 2), 60);
  close(predict(fit, 'flat', 2), 35);
  close(predict(fit, 'naive', 2), 50);
});

test('uneven weighted least squares does not anchor the fitted line to latest actual', () => {
  const fit = fitWeighted([4, 5, 1, 2, 0].map((cash, i) => row(2019 - i, cash)), 5);
  close(fit.mean, 2.95); close(fit.slope, 15 / 14); close(fit.intercept, 319 / 70);
  close(predict(fit, 'linear', 1), 197 / 35);
});

test('ten-year weights fit a signed linear history without losing negative values', () => {
  const fit = fitWeighted(fixture(2000, 2009, year => -100 + 2 * (year - 2000)).reverse(), 10);
  close(fit.slope, 2); close(fit.intercept, -82); close(predict(fit, 'linear', 5), -72);
  assert.deepEqual(fit.weights, [19, 17, 15, 13, 11, 9, 7, 5, 3, 1]);
});

test('native values use each row ratio and reject placeholders, missing and malformed periods', () => {
  const index = { columns: ['free_cash_flow', 'revenues'], sources: [{ id: 's', as_of: '2026-08-10' }] };
  const decoded = decodeAnnual([[2024, 5, '2024-01-01', '2024-12-31', '2025-01-30', 'USD', 4, 's', -200, 800]], index)[0];
  close(decoded.cash, -50); assert.equal(decoded.currency, 'USD');
  assert.equal(annualIssue(row(2024, 0)), null);
  assert.equal(annualIssue(row(2024, 0, { rawAmounts: [0, null] })), 'all_amounts_zero');
  assert.equal(annualIssue(row(2024, null)), 'cash_missing');
  assert.equal(annualIssue(row(2024, 1, { currencyRatio: 0 })), 'currency_ratio_missing');
  assert.equal(annualIssue(row(2024, 1, { start: '2024-03-01' })), 'not_full_annual');
});

test('rolling fit includes strictly prior years and a full window', () => {
  const result = buildCompanyFolds(company(fixture()), options);
  const fold = result.folds.find(f => f.targetYear === 2024);
  assert.deepEqual(fold.training.map(r => r.year), [2023, 2022, 2021, 2020, 2019]);
  assert.ok(fold.training.every(r => r.end < fold.target.start));
  assert.equal(result.folds[0].targetYear, 2020);
  assert.ok(result.exclusions.some(r => r.reason === 'training_missing'));
});

test('future poisoning cannot alter earlier fit, prediction, scale, or flags', () => {
  const original = company(fixture());
  const poisoned = company([...fixture().map(r => r.year > 2023 ? { ...r, cash: 1e12 } : r), row(2030, -1e15)]);
  const a = buildCompanyFolds(original, options).folds.find(f => f.targetYear === 2024);
  const b = buildCompanyFolds(poisoned, options).folds.find(f => f.targetYear === 2024);
  assert.deepEqual(a.fit, b.fit); assert.deepEqual(a.trainingFlags, b.trainingFlags);
  assert.equal(predict(a.fit, 'linear', 1), predict(b.fit, 'linear', 1));
  assert.notEqual(a.target.cash, b.target.cash);
});

test('target missing, short, currency change and withheld years are excluded instead of zero-filled', () => {
  for (const change of [
    rs => rs.filter(r => r.year !== 2024),
    rs => rs.map(r => r.year === 2024 ? { ...r, start: '2024-04-01' } : r),
    rs => rs.map(r => r.year === 2024 ? { ...r, currency: 'USD' } : r),
  ]) {
    const result = buildCompanyFolds(company(change(fixture())), options);
    assert.ok(!result.folds.some(f => f.targetYear === 2024));
    assert.ok(result.exclusions.some(f => f.targetYear === 2024 && f.stage === 'target'));
  }
  const result = buildCompanyFolds(company(fixture(), { withheld: [{ year: 2024, period: 5, reason: 'invalid' }] }), options);
  assert.ok(result.exclusions.some(f => f.targetYear === 2024 && f.reason === 'target_withheld'));
});

test('training gaps and overlap reject the fold, including intermediate target years', () => {
  const gap = company(fixture().filter(r => r.year !== 2022));
  assert.ok(!buildCompanyFolds(gap, options).folds.some(f => f.targetYear === 2024));
  const overlap = company(fixture().map(r => r.year === 2023 ? { ...r, start: '2022-12-15', end: '2023-12-14' } : r));
  assert.ok(!buildCompanyFolds(overlap, options).folds.some(f => f.targetYear === 2024));
  const oneDayOverlap = company(fixture().map(r => r.year === 2023 ? { ...r, start: '2022-12-31' } : r));
  assert.ok(!buildCompanyFolds(oneDayOverlap, options).folds.some(f => f.targetYear === 2024));
  const intermediate = buildCompanyFolds(company(fixture().filter(r => r.year !== 2024)), { ...options, horizons: [2] });
  assert.ok(!intermediate.folds.some(f => f.originYear === 2023 && f.targetYear === 2025));
});

test('duplicate year is rejected rather than choosing a favorable source row', () => {
  const result = buildCompanyFolds(company([...fixture(), row(2022, 1e9)]), options);
  assert.ok(!result.folds.some(f => f.targetYear === 2024));
  assert.ok(result.exclusions.some(f => f.reason.includes('duplicate')));
});

test('negative and zero predictions retain honest default sensitivity boundaries', () => {
  const fit = fitWeighted(fixture(2015, 2019, () => -10).reverse(), 5);
  const fold = { fit, target: row(2020, -12), horizon: 1, targetYear: 2020, window: 5, split: 'calibration', trainingFlags: [] };
  const scored = scoreForecast(fold, 'linear', null);
  assert.equal(scored.low, -11); assert.equal(scored.high, -9); assert.equal(scored.outcome, 'below');
  assert.ok(!scored.flags.includes('actual_change_opposes_trend'));
  const zero = scoreForecast({ ...fold, fit: { ...fit, intercept: 0, slope: 0 } }, 'linear', null);
  assert.equal(zero.low, 0); assert.equal(zero.high, 0); assert.ok(zero.flags.includes('near_zero_mid'));
});

test('interval score penalizes width and misses, normalizers never use actual outcomes', () => {
  const score = intervalMetrics(15, 8, 12, 10);
  assert.equal(score.outcome, 'above'); close(score.normalizedWidth, 0.4); close(score.normalizedIntervalScore, 3.4);
  assert.equal(intervalMetrics(2, 0, 4, 0).normalizedIntervalScore, null);
});

test('calibration uses only fixed earlier target years and nearest-rank 80th percentile', () => {
  const scores = [1, 2, 3, 4, 5].map(error => ({ window: 5, model: 'linear', horizon: 1, targetYear: 2020, targetReportDate: '2021-02-01', publicationUnavailable: null, absoluteError: error, meanAbs: 1 }));
  const factors = calibrationFactors([...scores, { ...scores[0], targetYear: 2021, absoluteError: 1e15 }, { ...scores[0], targetYear: 2025, absoluteError: 1e20 }], { minimum: 1 });
  assert.equal(factors[0].factor, 4); assert.equal(factors[0].count, 5);
  assert.equal(calibrationFactors(scores, { minimum: 6 })[0].factor, null);
  const zero = calibrationFactors([{ ...scores[0], meanAbs: 0 }], { minimum: 1 });
  assert.equal(zero[0].factor, null); assert.equal(zero[0].zeroScale, 1);
});

test('poisoning all validation and test values cannot tune calibrated factor', () => {
  const scoresFor = rows => buildCompanyFolds(company(rows), options).folds.map(f => scoreForecast(f, 'linear', null));
  const a = calibrationFactors(scoresFor(fixture(2005)), { minimum: 1 });
  const b = calibrationFactors(scoresFor(fixture(2005).map(r => r.year >= 2021 ? { ...r, cash: 1e20 } : r)), { minimum: 1 });
  assert.deepEqual(a, b);
});

test('calibration cannot use future publication, unknown publication, or late training metadata', () => {
  const base = { window: 5, model: 'linear', horizon: 1, targetYear: 2020, targetReportDate: '2021-06-30', publicationUnavailable: null, absoluteError: 1, meanAbs: 1 };
  const rows = [base, ...['2021-07-01', null, '2021-02-30'].map(targetReportDate => ({ ...base, targetReportDate, absoluteError: 1e10 })), { ...base, publicationUnavailable: 'training_publication_after_origin', absoluteError: 1e15 }];
  const calibrated = calibrationFactors(rows, { minimum: 1 })[0];
  assert.equal(calibrated.factor, 1); assert.equal(calibrated.count, 1); assert.equal(calibrated.publicationExcluded, 4);
});

test('information-time gating withholds old-origin and all recent-test fifth-horizon learned intervals', () => {
  const result = buildCompanyFolds(company(fixture(2000)), { windows: [5], horizons: [1, 2, 3, 4, 5] });
  const old = result.folds.find(f => f.originYear === 2020 && f.targetYear === 2025);
  assert.equal(scoreForecast(old, 'linear', 2).calibrated, null);
  assert.ok(result.folds.filter(f => f.split === 'test' && f.horizon === 5).every(f => scoreForecast(f, 'linear', 2).calibrated === null));
  const allowed = result.folds.find(f => f.originYear === 2021 && f.targetYear === 2025);
  assert.ok(scoreForecast(allowed, 'linear', 2).calibrated);
});

test('late older training publications and missing origin dates only suppress calibrated scoring', () => {
  for (const change of [
    rs => rs.map(r => r.year === 2022 ? { ...r, reportDate: '2025-02-01' } : r),
    rs => rs.map(r => r.year === 2023 ? { ...r, reportDate: null } : r),
    rs => rs.map(r => r.year === 2023 ? { ...r, reportDate: '2024-12-31' } : r),
    rs => rs.map(r => r.year === 2023 ? { ...r, reportDate: '2023-02-01' } : r),
    rs => rs.map(r => r.year === 2024 ? { ...r, reportDate: '2024-02-01' } : r),
  ]) {
    const fold = buildCompanyFolds(company(change(fixture())), options).folds.find(f => f.targetYear === 2024);
    assert.ok(fold); assert.equal(scoreForecast(fold, 'linear', 2).calibrated, null);
    assert.ok(['below', 'inside', 'above'].includes(scoreForecast(fold, 'linear', 2).outcome));
  }
});

test('all model comparisons use identical fold keys and ten-year matched cohorts are marked', () => {
  const result = buildCompanyFolds(company(fixture(2000)), { windows: [5, 10], horizons: [1, 2], maxTargetYear: 2025 });
  for (const ten of result.folds.filter(f => f.window === 10)) {
    const five = result.folds.find(f => f.window === 5 && f.originYear === ten.originYear && f.targetYear === ten.targetYear);
    assert.ok(five?.matchedTenYear);
    for (const model of ['naive', 'flat', 'damped', 'linear']) assert.ok(Number.isFinite(scoreForecast(ten, model, null).prediction));
  }
});

test('summary keeps currency errors separate and distinct companies and target years visible', () => {
  const base = { companyId: '1', targetYear: 2024, currency: 'EUR', prediction: 5, actual: 3, absoluteError: 2, error: 2, normalizedAbsoluteError: 0.2, normalizedBias: 0.2, mase: null, outcome: 'inside', normalizedWidth: 0.4, normalizedIntervalScore: 0.4, calibrated: null, flags: [] };
  const result = summarizeScores([base, { ...base, companyId: '2', targetYear: 2025, currency: 'USD', absoluteError: 100, error: -100 }]);
  assert.equal(result.companies, 2); assert.deepEqual(result.targetYears, [2024, 2025]);
  assert.equal(result.maeByCurrency.EUR.mae, 2); assert.equal(result.maeByCurrency.USD.mae, 100);
  assert.equal(result.inside, 2); assert.equal(result.maseCount, 0);
});

test('default and learned interval comparison uses the identical calibrated cohort', () => {
  const result = buildCompanyFolds(company(fixture(2000)), { windows: [5], horizons: [4] });
  const rows = result.folds.filter(f => f.split === 'test').map(f => scoreForecast(f, 'linear', 2));
  const summary = summarizeScores(rows);
  assert.equal(summary.count, 2); assert.equal(summary.calibratedCount, 1);
  assert.equal(summary.rawOnCalibratedCount, 1); assert.deepEqual(summary.calibratedTargetYears, [2025]);
  const eligible = rows.find(r => r.calibrated);
  close(summary.rawOnCalibratedMeanNormalizedWidth, eligible.normalizedWidth);
  close(summary.rawOnCalibratedMeanNormalizedIntervalScore, eligible.normalizedIntervalScore);
});

test('source-vintage changes remain flagged separately from native currency matching', () => {
  const rows = fixture().map(r => r.year === 2020 ? { ...r, sourceId: 'older-vintage', sourceAsOf: '2025-06-21' } : r);
  const fold = buildCompanyFolds(company(rows), options).folds.find(f => f.targetYear === 2024);
  assert.ok(fold.trainingFlags.includes('mixed_source_vintages'));
  assert.equal(fold.currency, 'EUR');
});

test('source-bound SQLite integration writes reconciled compressed tables and verifies payload hashes', async () => {
  const root = mkdtempSync(join(tmpdir(), 'atlas-backtest-'));
  const sha = bytes => createHash('sha256').update(bytes).digest('hex');
  try {
    const taxonomy = Buffer.from(JSON.stringify({ catalogue: { listings: { '1': { name: 'Fixture', isin: 'XX123' } } }, classifications: {}, sectors: {}, branches: {} }));
    const payload = Buffer.from(JSON.stringify({ id: '1', annual: fixture(2000).map(r => [r.year, 5, r.start, r.end, r.reportDate, r.currency, 2, 'saved-2026', r.rawCash, 100]), quarterly: [], withheld: [] }));
    const index = { format: 'macro-atlas-financials', version: 2, as_of: '2026-08-10', taxonomy_sha256: sha(taxonomy), columns: ['free_cash_flow', 'revenues'],
      sources: [{ id: 'saved-2026', as_of: '2026-08-10' }], companies: { '1': { sha256: sha(payload) } } };
    const packPath = join(root, 'fixture.sqlite'), taxonomyPath = join(root, 'taxonomy.json');
    writeFileSync(taxonomyPath, taxonomy);
    const db = new DatabaseSync(packPath);
    db.exec('CREATE TABLE metadata(key TEXT, payload BLOB); CREATE TABLE companies(id TEXT, payload BLOB, sha256 TEXT)');
    db.prepare('INSERT INTO metadata VALUES (?,?)').run('index', gzipSync(JSON.stringify(index)));
    db.prepare('INSERT INTO companies VALUES (?,?,?)').run('1', gzipSync(payload), sha(payload)); db.close();
    const result = await runBacktest({ packPath, taxonomyPath, outputDir: join(root, 'out') });
    assert.equal(result.listings, 1); assert.equal(result.inputs.financialPack.sha256, sha(readFileSync(packPath)));
    const csv = gunzipSync(readFileSync(join(root, 'out/forecasts.csv.gz'))).toString().trim().split('\n');
    assert.equal(csv.length - 1, result.forecastRows);
    assert.equal(result.metrics.reduce((sum, r) => sum + r.count, 0), result.forecastRows);
    const testFifth = result.metrics.find(r => r.model === 'linear' && r.window === 5 && r.horizon === 5 && r.split === 'test');
    assert.equal(testFifth.calibratedCount, 0);
    const companySummary = JSON.parse(readFileSync(join(root, 'out/company-summary.json')));
    assert.equal(companySummary.companies[0].id, '1'); assert.ok(companySummary.companies[0].metrics.every(r => r.model === 'linear' && r.window === 5));
    const corrupt = new DatabaseSync(packPath); corrupt.prepare('UPDATE companies SET sha256=?').run('0'.repeat(64)); corrupt.close();
    await assert.rejects(runBacktest({ packPath, taxonomyPath, outputDir: join(root, 'bad') }), /checksum mismatch/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
