import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, relative, isAbsolute } from 'node:path';
import { build } from 'esbuild';

// Consume the fresh output from build-normal-year-ranking.py; never write fixtures.
const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
assert.equal(process.argv.length, 3, 'Provide the generator output directory.');
const output = resolve(process.argv[2]);
const outputRelative = relative(resolve(desktop, 'test-results'), output);
assert.ok(outputRelative && outputRelative !== '..' && !outputRelative.startsWith('../') && !outputRelative.startsWith('..\\') && !isAbsolute(outputRelative), 'Output must be under desktop/test-results.');
const read = path => JSON.parse(readFileSync(path));
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const expected = read(resolve(output, 'expected-ranking.json'));
const modulePath = resolve(output, 'model-module.mjs');
await build({ absWorkingDir: desktop, stdin: { contents: [
  "export * from './src/companyListModel';", "export * from './src/companyListRanges';",
  "export * from './src/companyListFeatures';", "export * from './src/normalQualityValueModel';",
  "export * from './src/normalYearScreen';",
].join('\n'), resolveDir: desktop }, outfile: modulePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent' });
const m = await import(pathToFileURL(modulePath));
const gm = read(resolve(desktop, 'src/data/research-gauge-manifest.json'));
const km = read(resolve(desktop, 'src/data/expanded-kpi-manifest.json'));
function decode(d) {
  const packed = readFileSync(resolve(desktop, 'public', d.path)), raw = gunzipSync(packed);
  assert.equal(packed.length, d.bytes); assert.equal(sha(packed), d.sha256);
  assert.equal(raw.length, d.uncompressedBytes); assert.equal(sha(raw), d.uncompressedSha256);
  return JSON.parse(raw);
}
const rows = decode(gm.artifact).rows, index = decode(km.index);
index.positions = new Map(index.ids.map((id, i) => [id, i]));
const expanded = { index, variants: new Map(), loading: new Set(), errors: new Map(), error: '', ready: true };
const view = expected.referenceInputs.qualityView;
for (const column of view.columns.filter(column => column.kpiId.startsWith('provider_'))) {
  const variant = km.variants.find(v => v.metricId === column.kpiId && `provider:${v.source}:${v.calcGroup}` === column.window && `provider:${v.calculation}` === column.calculation);
  const shard = km.shards.find(s => s.id === variant.shard);
  expanded.variants.set(variant.id, { variant, values: decode(shard).values[variant.offset] });
}
const ranking = m.buildNormalQualityValueContext(rows, expanded);
assert.equal(ranking.status, 'available'); assert.equal(ranking.eligibleCount, 248); assert.equal(ranking.cohortSize, 180);
const close = (a, b, label) => b === null ? assert.equal(a, null, label) : assert.ok(typeof a === 'number' && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b)), `${label}: ${a} vs ${b}`);
let checkedValues = 0;
for (const [id, independent] of Object.entries(expected.byId)) {
  const actual = ranking.byId.get(id), observation = ranking.observations.get(id);
  assert.ok(actual && observation.eligible);
  for (const key of ['score', 'quality', 'discountRank']) { close(actual[key], independent[key], `${id} ${key}`); checkedValues++; }
  for (const key of ['discount', 'roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda']) { close(observation[key], independent.raw[key], `${id} raw ${key}`); checkedValues++; }
  if (independent.components === null) assert.equal(actual.components, null);
  else for (const key of ['roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda']) { close(actual.components[key], independent.components[key], `${id} component ${key}`); checkedValues++; }
}
assert.deepEqual([...ranking.observations.values()].filter(row => row.eligible).map(row => row.id).sort(), Object.keys(expected.byId).sort());
const rankColumns = ['normal_quality_value_score', 'normal_discount_rank', 'normal_quality_rank'].map(kpiId => ({ id: kpiId, kpiId, window: 'latest', calculation: 'latest' }));
const npv5Column = { id: 'normal_npv_5y_percent', kpiId: 'normal_npv_5y_percent', window: 'latest', calculation: 'latest' };
const halfColumn = view.columns.find(column => column.id === 'normal_npv_half');
const rankedView = { ...view, columns: [rankColumns[0], npv5Column, rankColumns[2], rankColumns[1], halfColumn, ...view.columns.filter(column => column.id !== halfColumn.id)], sort: { columnId: 'normal_quality_value_score', direction: 'desc' } };
const preferences = { ...m.defaultCompanyListPreferences(), ...rankedView };
preferences.features.secondarySorts = [{ columnId: 'normal_npv_5y_percent', direction: 'desc' }];
const parsed = m.parseCompanyListPreferences(preferences);
assert.deepEqual(parsed.filters, view.filters); assert.equal(parsed.columns.length, 28);
for (const oldColumn of view.columns) assert.deepEqual(parsed.columns.find(column => column.id === oldColumn.id), oldColumn);
assert.deepEqual(parsed.columns, rankedView.columns); assert.deepEqual(parsed.features.secondarySorts, preferences.features.secondarySorts);
const matches = (row, saved) => m.matchesCompanyListFilters(row, saved.filters, new Set(), expanded, ranking) && saved.columns.every(column => m.matchesCompanyListRange(m.companyListCell(row, column, expanded, ranking), column.range));
const watch = rows.filter(row => matches(row, rankedView));
assert.equal(watch.length, 248);
const legacyEvidence = read(resolve(output, 'legacy-metrics.json'));
const legacyById = new Map(legacyEvidence.rows.map(row => [row.id, row]));
let legacyMetricsChecked = 0;
for (const row of watch) for (const column of view.columns) {
  const independent = legacyById.get(row.id).metrics[column.id];
  if (independent === undefined) continue;
  close(m.companyListCell(row, column, expanded, ranking).value, independent, `${row.id} existing ${column.id}`);
  legacyMetricsChecked++;
}
let fiveYearValuesChecked = 0;
for (const row of watch) {
  const cash = legacyById.get(row.id).metrics.normal_fcf_median;
  const cashPV = Array.from({ length: 5 }, (_, i) => cash / 1.1 ** (i + 1)).reduce((sum, cash) => sum + cash, 0);
  const actual = m.calculateNormalYearFiveYearValuation(row);
  close(actual.normalCash, cash, `${row.id} five-year cash`);
  close(actual.cashPV, cashPV, `${row.id} five-year cash PV`);
  close(actual.value, cashPV, `${row.id} five-year value`);
  assert.equal(actual.terminalPV, 0);
  close(actual.surplusPercent, expected.byId[row.id].raw.discount, `${row.id} five-year NPV/price`);
  close(m.companyListCell(row, npv5Column, expanded, ranking).value, expected.byId[row.id].raw.discount, `${row.id} five-year NPV cell`);
  fiveYearValuesChecked += 6;
}
const strictView = read(resolve(desktop, 'research/screens/normal-years-quality-value-2026-09-28.json')).view;
assert.equal(rows.filter(row => matches(row, strictView)).length, 0);
const sorted = m.sortCompanyListRows(watch, parsed.columns, [parsed.sort, ...parsed.features.secondarySorts], (row, column) => m.companyListCell(row, column, expanded, ranking));
const differing = sorted.filter((row, i) => row.id !== expected.orderedIds[i]);
if (differing.length) writeFileSync(resolve(output, 'sort-diagnostic.json'), JSON.stringify(differing.map(row => ({ id: row.id, actual: ranking.byId.get(row.id), expected: expected.byId[row.id] })), null, 2) + '\n');
assert.deepEqual(sorted.map(row => row.id), expected.orderedIds);
for (const row of sorted) for (const [kpiId, key] of [['normal_quality_value_score', 'score'], ['normal_discount_rank', 'discountRank'], ['normal_quality_rank', 'quality']]) {
  const cell = m.companyListCell(row, { id: kpiId, kpiId, window: 'latest', calculation: 'latest' }, expanded, ranking);
  close(cell.value, expected.byId[row.id][key], `${row.id} displayed ${key}`);
  assert.equal(cell.status, expected.byId[row.id][key] === null ? 'missing' : 'available'); checkedValues++;
}
const subset = watch.filter(row => row.country === sorted[0].country);
for (const row of subset) close(m.companyListCell(row, rankColumns[0], expanded, ranking).value, expected.byId[row.id].score, `${row.id} filtered score`);
const receipt = { status: 'passed', sourceRows: rows.length, eligibleCount: watch.length, cohortSize: ranking.cohortSize, unrankedCount: 68, strictCount: 0, checkedValues, legacyMetricsChecked, fiveYearValuesChecked,
  orderedIds: sorted.map(row => row.id), originalQualityFiltersPreserved: true, originalColumnsAndRangesPreserved: true, displayedCellsMatchIndependent: true, subsetScoresUnchanged: true,
  expectedSha256: sha(readFileSync(resolve(output, 'expected-ranking.json'))), moduleSha256: sha(readFileSync(modulePath)) };
writeFileSync(resolve(output, 'model-verification.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ ...receipt, orderedIds: undefined }));
