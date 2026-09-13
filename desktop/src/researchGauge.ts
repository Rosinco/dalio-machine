import type { FinancialIndex } from './financialData';
import type { ResearchGaugeArtifact, ResearchGaugeManifest, ResearchGaugePeriod, ResearchGaugeRow, ResearchGaugeSeries } from './researchGaugeModel';

function fail(ok: unknown, message = 'The saved research screen is invalid.'): asserts ok { if (!ok) throw new Error(message); }
const object = (v: any) => v !== null && typeof v === 'object' && !Array.isArray(v);
const text = (v: any) => typeof v === 'string' && v.length <= 2000;
const hash = (v: any) => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const finite = (v: any): v is number => typeof v === 'number' && Number.isFinite(v);
const numeric = (v: any) => v === null || finite(v);
const integer = (v: any, max: number) => Number.isSafeInteger(v) && v >= 0 && v <= max;
// Report calendars repeat across the universe. Parse each distinct date once.
const dates = new Map<string, number>();
function timestamp(v: string): number {
  if (dates.has(v)) return dates.get(v)!;
  const parsed = Date.parse(v);
  const result = Number.isFinite(parsed) && new Date(parsed).toISOString().slice(0, 10) === v ? parsed : NaN;
  if (dates.size >= 10000) dates.clear();
  dates.set(v, result); return result;
}
const day = (v: any): v is string => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v) && Number.isFinite(timestamp(v));
const close = (a: number | null, b: number | null) => b === null ? a === null : finite(a) && finite(b) && Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
const differenceDays = (a: string, b: string) => (timestamp(a) - timestamp(b)) / 86400000;

export function validateResearchGaugeManifest(v: any): ResearchGaugeManifest {
  fail(object(v) && v.format === 'macro-atlas-research-gauge-manifest' && v.version === 1 && v.model === 'research-gauge-v1', 'Unsupported research screen format.');
  fail(day(v.asOf) && hash(v.financialPackId) && hash(v.taxonomySha256) && text(v.calibrationId) && v.calibrationId.length > 0);
  fail(integer(v.financialPackBytes, 512 * 1024 * 1024) && v.financialPackBytes > 0 && integer(v.rows, 100000) && v.rows > 0);
  const a = v.artifact;
  fail(object(a) && a.path === 'data/research-gauge/gauge.bin' && hash(a.sha256) && hash(a.uncompressedSha256));
  fail(integer(a.bytes, 32 * 1024 * 1024) && a.bytes > 0 && integer(a.uncompressedBytes, 128 * 1024 * 1024) && a.uncompressedBytes > 0, 'Research screen exceeds its supported size.');
  return v;
}

export function validateResearchGaugeBinding(manifest: ResearchGaugeManifest, financial: FinancialIndex, taxonomy: string | null | undefined) {
  fail(financial.id === manifest.financialPackId && financial.bytes === manifest.financialPackBytes && taxonomy === manifest.taxonomySha256 && financial.taxonomy_sha256 === taxonomy,
    'This research screen belongs to a different financial pack or company directory. Select its matching saved release.');
  fail(Object.keys(financial.companies).length === manifest.rows && financial.summary.listings === manifest.rows && financial.as_of <= manifest.asOf,
    'Research screen coverage does not match the selected financial pack.');
}

function validatePeriod(v: ResearchGaugePeriod | null, asOf: string, annual: boolean, financial: FinancialIndex) {
  if (v === null) return;
  fail(object(v) && integer(v.year, 2300) && v.year >= 2000 && (annual ? v.period === 5 : integer(v.period, 4) && v.period > 0));
  fail(day(v.start) && day(v.end) && v.start <= v.end && day(v.sourceAsOf) && v.end <= v.sourceAsOf && v.sourceAsOf <= asOf);
  fail(v.published === null || day(v.published) && v.published >= v.end && v.published <= v.sourceAsOf);
  fail(typeof v.currency === 'string' && /^[A-Z]{3}$/.test(v.currency));
  const source = financial.sources.find(s => s.id === v.sourceId);
  fail(source?.as_of === v.sourceAsOf && source.frequency === (annual ? 'annual' : 'quarterly'), 'Research observations have an unexpected source.');
}

function validateSeries(s: ResearchGaugeSeries, periods: number) {
  fail(object(s) && Array.isArray(s.values) && s.values.length === periods && s.values.every(numeric));
  fail(['count', 'positive', 'negative', 'zero'].every(k => integer(s[k as keyof ResearchGaugeSeries], periods)));
  fail(s.count === s.values.filter(finite).length && s.positive === s.values.filter(x => x !== null && x > 0).length && s.negative === s.values.filter(x => x !== null && x < 0).length && s.zero === s.values.filter(x => x === 0).length);
  fail(s.latest === (s.values[0] ?? null) && numeric(s.median) && numeric(s.dispersion) && (s.dispersion === null || s.dispersion >= 0));
  const sorted = s.values.filter(finite).sort((a, b) => a - b);
  fail(close(s.median, sorted.length ? (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2 : null));
}

function validateRow(row: ResearchGaugeRow, manifest: ResearchGaugeManifest, financial: FinancialIndex) {
  fail(object(row) && typeof row.id === 'string' && /^[1-9][0-9]{0,9}$/.test(row.id) && row.sourceCompanySha256 === financial.companies[row.id]?.sha256,
    'A research row does not match its company source.');
  fail(text(row.name) && row.name.length > 0 && ['ticker', 'isin', 'country', 'sectorId', 'sectorName', 'branchId', 'branchName'].every(k => row[k as keyof ResearchGaugeRow] === null || text(row[k as keyof ResearchGaugeRow])));
  fail(day(row.sourceAsOf) && row.sourceAsOf <= manifest.asOf && ['latest', 'older'].includes(row.presence));
  fail(['operating', 'property', 'financial', 'unclassified'].includes(row.route) && typeof row.classificationConflict === 'boolean' && ['history_available', 'limited_history', 'reconcile_data', 'no_history'].includes(row.readiness));
  fail(Array.isArray(row.issues) && row.issues.length <= 50 && row.issues.every(text));
  const a = row.annual;
  fail(object(a) && (a.currency === null || typeof a.currency === 'string' && /^[A-Z]{3}$/.test(a.currency)) && Array.isArray(a.periods) && a.periods.length <= 5 && (a.reason === null || text(a.reason)));
  validatePeriod(a.latest, manifest.asOf, true, financial);
  for (const [i, period] of a.periods.entries()) {
    validatePeriod(period, manifest.asOf, true, financial);
    fail(period.currency === a.currency && differenceDays(period.end, period.start) >= 329 && differenceDays(period.end, period.start) <= 399);
    if (i > 0) fail(period.year === a.periods[i - 1].year - 1 && differenceDays(a.periods[i - 1].start, period.end) >= 1 && differenceDays(a.periods[i - 1].start, period.end) <= 35);
  }
  if (a.periods.length) fail(a.latest?.end === a.periods[0].end && a.latest?.sourceId === a.periods[0].sourceId);
  fail(object(a.excluded) && ['outsideWindow', 'placeholder', 'missingPublication', 'withheld'].every(k => integer(a.excluded[k as keyof typeof a.excluded], 10000)));
  for (const key of ['cash', 'operatingCash', 'ebit', 'revenue', 'margins'] as const) validateSeries(a[key], a.periods.length);
  fail(object(a.latestValues) && ['revenues', 'ebit', 'cash', 'operatingCash', 'financingCash', 'cashBalance', 'netDebt', 'equity', 'assets', 'netDebtToAssetsPercent', 'equityToAssetsPercent', 'tangibleAssetsToRevenue', 'intangibleAssetsToAssetsPercent', 'cashComponentDifference'].every(k => numeric(a.latestValues[k as keyof typeof a.latestValues])));
  fail(a.latestValues.cash === a.cash.latest && a.latestValues.operatingCash === a.operatingCash.latest && a.latestValues.ebit === a.ebit.latest && a.latestValues.revenues === a.revenue.latest);
  const q = row.quarter;
  fail(object(q) && [q.revenueChangePercent, q.ebitMarginChangePoints, q.cashChange].every(numeric) && (q.reason === null || text(q.reason)));
  validatePeriod(q.latest, manifest.asOf, false, financial); validatePeriod(q.comparison, manifest.asOf, false, financial);
  if ([q.revenueChangePercent, q.ebitMarginChangePoints, q.cashChange].some(v => v !== null)) fail(q.latest && q.comparison && q.latest.currency === q.comparison.currency && q.latest.published && q.comparison.published && q.latest.period === q.comparison.period && q.latest.year === q.comparison.year + 1);
  const v = row.valuation;
  fail(object(v) && typeof v.currency === 'string' && /^[A-Z]{3}$/.test(v.currency) && ['historical', 'percentage', 'unavailable'].includes(v.rangeStatus));
  fail(['value', 'cashPV', 'terminalPV', 'terminalShare', 'candidateEquity', 'ceiling', 'lowValue', 'lowNPV', 'reverseCashFactor', 'reverseCashFactor30'].every(k => numeric(v[k as keyof typeof v])));
  fail(typeof v.hasSignedCash === 'boolean' && integer(v.annualHistoryCount, 10000) && (v.reason === null || text(v.reason)));
  fail(v.candidateEquity === null ? v.priceDate === null && v.priceAgeDays === null : v.candidateEquity > 0 && day(v.priceDate) && v.priceDate <= manifest.asOf && v.priceAgeDays === differenceDays(manifest.asOf, v.priceDate));
  if (v.priceBasis !== null) {
    const p = v.priceBasis;
    fail(object(p) && financial.sources.some(s => s.id === p.sourceId && s.frequency === 'annual' && s.as_of === p.sourceAsOf) && day(p.reportEnd) && p.reportEnd <= p.sourceAsOf && (p.reportDate === null || day(p.reportDate) && p.reportDate >= p.reportEnd && p.reportDate <= p.sourceAsOf));
    fail([p.shares, p.close, p.fxRate].every(numeric) && (p.currency === null || /^[A-Z]{3}$/.test(p.currency)) && ['local', 'sek'].includes(p.method) && (p.fxDate === null || day(p.fxDate) && p.fxDate <= p.sourceAsOf));
  }
  if (v.value !== null) fail(v.cashPV !== null && v.terminalPV !== null && v.terminalPV >= 0 && close(v.value, v.cashPV + v.terminalPV));
  else fail(v.cashPV === null && v.terminalPV === null, 'Incomplete research valuations must retain missing components.');
  fail(close(v.ceiling, v.value !== null && v.value > 0 ? .7 * v.value : null));
  fail(close(v.terminalShare, v.value !== null && v.value > 0 && v.terminalPV !== null ? v.terminalPV / v.value : null));
  fail(close(v.reverseCashFactor, v.value !== null && v.value > 0 && v.candidateEquity !== null ? v.candidateEquity / v.value : null));
  fail(close(v.reverseCashFactor30, v.ceiling !== null && v.candidateEquity !== null ? v.candidateEquity / v.ceiling : null));
  fail(close(v.lowNPV, v.lowValue !== null && v.candidateEquity !== null ? v.lowValue - v.candidateEquity : null));
  const status = row.route === 'financial' ? 'manual-financial' : v.value === null ? 'missing' : v.value <= 0 ? 'nonpositive' : v.candidateEquity === null ? 'positive-unpriced' : 'positive-priced';
  fail(v.status === status && (row.route !== 'financial' || [v.value, v.cashPV, v.terminalPV, v.lowValue, v.lowNPV].every(value => value === null)), 'Research valuation readiness does not reconcile.');
}

export function validateResearchGaugeArtifact(v: any, manifest: ResearchGaugeManifest, financial: FinancialIndex, taxonomy: string | null | undefined): ResearchGaugeArtifact {
  validateResearchGaugeBinding(manifest, financial, taxonomy);
  fail(object(v) && v.format === 'macro-atlas-research-gauge' && v.version === 1 && v.model === manifest.model && v.asOf === manifest.asOf && v.financialPackId === manifest.financialPackId && v.taxonomySha256 === manifest.taxonomySha256 && v.calibrationId === manifest.calibrationId, 'Research screen metadata does not match its pinned source.');
  fail(Array.isArray(v.rows) && v.rows.length === manifest.rows, 'Research screen listing coverage is incomplete.');
  const ids = new Set<string>();
  for (const row of v.rows) { validateRow(row, manifest, financial); fail(!ids.has(row.id), 'Research screen has duplicate listings.'); ids.add(row.id); }
  fail(Object.keys(financial.companies).every(id => ids.has(id)), 'Research screen omits a saved listing.');
  return v;
}

export async function researchGaugeHash(bytes: Uint8Array) {
  const digest = await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), x => x.toString(16).padStart(2, '0')).join('');
}

/** Bound both network input and decompressed output before JSON parsing. */
export async function readResearchGaugeBytes(stream: ReadableStream<Uint8Array> | null, expected: number): Promise<Uint8Array> {
  fail(stream, 'The saved research screen is empty.');
  const reader = stream.getReader(), chunks: Uint8Array[] = []; let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      size += value.byteLength;
      fail(size <= expected, 'Research screen exceeds its pinned size.'); chunks.push(value);
    }
    fail(size === expected, 'Research screen is incomplete.');
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return bytes;
}

export async function decodeResearchGauge(bytes: Uint8Array, manifest: ResearchGaugeManifest, financial: FinancialIndex, taxonomy: string | null | undefined): Promise<ResearchGaugeArtifact> {
  validateResearchGaugeBinding(manifest, financial, taxonomy);
  fail(bytes.byteLength === manifest.artifact.bytes && await researchGaugeHash(bytes) === manifest.artifact.sha256, 'Research screen checksum failed. Its calculations have been withheld.');
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'));
  const raw = await readResearchGaugeBytes(stream, manifest.artifact.uncompressedBytes);
  fail(await researchGaugeHash(raw) === manifest.artifact.uncompressedSha256, 'Decompressed research screen checksum failed.');
  return validateResearchGaugeArtifact(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw)), manifest, financial, taxonomy);
}
