/** Independently check an exported Python pilot with centered-covariance WLS. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const arg = name => { const i = process.argv.indexOf(name); return i < 0 ? undefined : process.argv[i + 1]; };
if (process.argv.includes('--help')) {
  console.log('node desktop/scripts/validate-frozen-vintage-cash-pilot.mjs --output <artifact directory> [--prefix frozen-2025-pilot]');
  process.exit(0);
}
const output = arg('--output');
if (!output) throw new Error('Pass the artifact directory with --output.');
const prefix = arg('--prefix') ?? 'frozen-2025-pilot';
if (!/^[a-zA-Z0-9-]+$/.test(prefix)) throw new Error('Invalid artifact prefix.');
const root = path.resolve(output);
const load = suffix => JSON.parse(fs.readFileSync(path.join(root, `${prefix}-${suffix}.json`), 'utf8'));
const ledger = load('companies'), summary = load('summary');
const errors = [], ids = new Set();
const near = (a, b) => typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= Math.max(1e-7, Math.abs(b) * 1e-10);
const fail = (id, reason) => { errors.push({ id, reason }); };
const count = { listings: ledger.companies.length, trainingEligible: 0, evaluated: 0, hits: 0, below: 0, above: 0, missingOutcomes: 0, trainingUnavailable: 0, departed: 0 };
for (const r of ledger.companies) {
  if (ids.has(r.id)) fail(r.id, 'Duplicate listing');
  ids.add(r.id);
  if (r.in_later_instrument_directory === false) count.departed++;
  if (r.status === 'training_unavailable') { count.trainingUnavailable++; continue; }
  if (!['outcome_unavailable', 'evaluated'].includes(r.status)) fail(r.id, 'Invalid row status');
  count.trainingEligible++;
  const w = [.3, .25, .2, .15, .1], x = [0, -1, -2, -3, -4], y = r.training_cash_native;
  if (y.length !== 5 || r.training_years.some((year, i) => year !== r.latest_training_year - i) || r.training_publication_dates.some(day => !day || day > ledger.origin)) fail(r.id, 'Invalid training window or publication cutoff');
  if (r.training_ratios.some(ratio => !Number.isFinite(ratio) || ratio <= 0)) fail(r.id, 'Invalid saved currency conversion');
  const meanX = x.reduce((sum, value, i) => sum + w[i] * value, 0);
  const meanY = y.reduce((sum, value, i) => sum + w[i] * value, 0);
  const slope = y.reduce((sum, value, i) => sum + w[i] * (value - meanY) * (x[i] - meanX), 0) / x.reduce((sum, value, i) => sum + w[i] * (value - meanX) ** 2, 0);
  const intercept = meanY - slope * meanX, mid = intercept + slope;
  if (!near(meanY, r.weighted_mean) || !near(slope, r.slope) || !near(intercept, r.intercept) || !near(mid, r.mid) || !near(mid - Math.abs(mid) * .1, r.low) || !near(mid + Math.abs(mid) * .1, r.high)) fail(r.id, 'Independent regression or signed-band mismatch');
  if (r.status === 'outcome_unavailable') {
    count.missingOutcomes++;
    if (r.actual !== null || r.hit !== null || !r.reason) fail(r.id, 'Missing outcome was scored or reason omitted');
    continue;
  }
  count.evaluated++;
  if (!(r.target_end > ledger.origin && r.target_end <= ledger.outcomeSnapshot && r.target_publication > ledger.origin && r.target_publication <= ledger.outcomeSnapshot)) fail(r.id, 'Target date is not a forward observed outcome');
  if (r.target_year !== r.latest_training_year + 1) fail(r.id, 'Target is not the next fiscal year');
  if (r.identity_isin_changed) fail(r.id, 'Changed listing identity was scored');
  if (!near(r.actual_raw / r.actual_ratio, r.actual)) fail(r.id, 'Outcome native currency conversion mismatch');
  const tolerance = Math.max(1e-8, Math.abs(r.mid) * 1e-10);
  const hit = r.actual >= r.low - tolerance && r.actual <= r.high + tolerance;
  const direction = hit ? null : r.actual < r.low ? 'below' : 'above';
  if (hit !== r.hit || direction !== r.miss_direction) fail(r.id, 'Independent coverage classification mismatch');
  if (!near(r.actual - r.mid, r.actual_minus_mid)) fail(r.id, 'Error amount mismatch');
  count.hits += Number(hit);
  if (direction) count[direction]++;
}
const expected = summary.overall;
for (const [key, expectedKey] of [['listings', 'frozenListings'], ['trainingEligible', 'trainingEligible'], ['evaluated', 'evaluated'], ['hits', 'hits'], ['below', 'below'], ['above', 'above'], ['missingOutcomes', 'unavailableOutcomeAfterTraining'], ['departed', 'absentFromLaterDirectory']]) {
  if (count[key] !== expected[expectedKey]) fail('all', `Summary ${key} does not reconcile`);
}
if (count.listings !== count.trainingUnavailable + count.missingOutcomes + count.evaluated) fail('all', 'Status ledger does not retain the entire cohort');
if (!near(100 * count.hits / count.evaluated, expected.coveragePercent)) fail('all', 'Summary coverage does not reconcile');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const receipt = {
  format: 'macro-atlas-frozen-vintage-pilot-validation', version: 1,
  method: 'Independent Node centered-covariance regression, signed bands, publication cutoff, target dates, saved currency conversions, missing-outcome handling and summary reconciliation against the Python normal-equation pilot. Source row selection remains independently documented in the generation protocol.',
  origin: ledger.origin, outcomeSnapshot: ledger.outcomeSnapshot, counts: count, errors,
  hashes: { companies: hash(path.join(root, `${prefix}-companies.json`)), summary: hash(path.join(root, `${prefix}-summary.json`)), validator: hash(fileURLToPath(import.meta.url)) },
};
fs.writeFileSync(path.join(root, `${prefix}-validation.json`), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ counts: count, errors }, null, 2));
if (errors.length) process.exitCode = 1;
