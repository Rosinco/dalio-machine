import pin from './data/expanded-kpi-manifest.json';
import type { FinancialIndex } from './financialData';
import type { CompanyKpi, CompanyKpiUnit, CompanyListCell, CompanyListColumn } from './companyListModel';
import type { ResearchGaugeRow } from './researchGaugeModel';
import { readResearchGaugeBytes, researchGaugeHash } from './researchGauge';

export type ProviderUnit = 'number' | 'percent' | 'multiple' | 'count' | 'millions' | 'per_share' | 'price' | 'date' | 'text';
export type ProviderCurrency = 'report' | 'quote' | 'SEK' | 'USD' | 'none' | 'unverified';
export type ExpandedArtifact = { id?: string | number; path: string; sha256: string; bytes: number; uncompressedSha256: string; uncompressedBytes: number; variantIds?: string[] };
export type ExpandedKpiVariant = { id: string; metricId: string; calcGroup: string; calculation: string; source: string; label: string; unit: ProviderUnit; currencyBasis: ProviderCurrency; availableCount: number; conflictCount: number; shard: string | number; offset: number; notes: string[] };
export type ExpandedKpiMetric = { id: string; providerKpiId: number; label: string; category: string; description: string; unit: ProviderUnit; currencyBasis: ProviderCurrency; variants: string[] };
export type ExpandedKpiManifest = { format: 'macro-atlas-expanded-kpi-manifest'; version: 1; snapshot: string; financialPackId: string; taxonomySha256: string; rows: number; index: ExpandedArtifact; metrics: ExpandedKpiMetric[]; variants: ExpandedKpiVariant[]; shards: ExpandedArtifact[]; notes?: string[] };
export type ExpandedKpiIndex = { ids: string[]; reportCurrencies: (string | null)[]; quoteCurrencies: (string | null)[]; positions: ReadonlyMap<string, number> };
export type ExpandedLoadedVariant = { variant: ExpandedKpiVariant; values: (number | string | null)[] };
export type ExpandedKpiContext = { index: ExpandedKpiIndex | null; variants: ReadonlyMap<string, ExpandedLoadedVariant>; loading: ReadonlySet<string>; errors: ReadonlyMap<string, string>; error: string; ready: boolean };
export const EXPANDED_KPI_MANIFEST = pin as unknown as ExpandedKpiManifest;
const byVariant = new Map(EXPANDED_KPI_MANIFEST.variants.map(v => [v.id, v]));
const byMetric = new Map(EXPANDED_KPI_MANIFEST.metrics.map(v => [v.id, v]));
const byColumn = new Map(EXPANDED_KPI_MANIFEST.variants.map(v => [`${v.metricId}|provider:${v.source}:${v.calcGroup}|provider:${v.calculation}`, v]));
const units: ProviderUnit[] = ['number', 'percent', 'multiple', 'count', 'millions', 'per_share', 'price', 'date', 'text'];
const currencyBases: ProviderCurrency[] = ['report', 'quote', 'SEK', 'USD', 'none', 'unverified'];
function fail(condition: unknown, message = 'The additional KPI data is invalid.'): asserts condition { if (!condition) throw new Error(message); }
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown, max = 2000): value is string => typeof value === 'string' && value.length <= max;
const hash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const integer = (value: unknown, max: number): value is number => Number.isSafeInteger(value) && Number(value) >= 0 && Number(value) <= max;
const day = (value: unknown): value is string => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;
const currency = (value: unknown) => value === null || typeof value === 'string' && /^[A-Z]{3}$/.test(value);
const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const smallFormatter = new Intl.NumberFormat('en-US', { maximumSignificantDigits: 3 });

export function expandedUnit(unit: ProviderUnit, basis?: ProviderCurrency): CompanyKpiUnit {
  return unit === 'millions' ? basis === 'none' ? 'shares_millions' : 'money' : unit === 'per_share' ? 'price' : unit;
}
export function expandedVariant(column: CompanyListColumn): ExpandedKpiVariant | undefined {
  return byColumn.get(`${column.kpiId}|${column.window}|${column.calculation}`);
}
export function expandedVariantById(id: string) { return byVariant.get(id); }
export function expandedShard(variant: ExpandedKpiVariant) {
  return EXPANDED_KPI_MANIFEST.shards.find(s => s.id === variant.shard || s.path === variant.shard);
}
export function providerWindowLabel(variant: ExpandedKpiVariant): string {
  const group = variant.calcGroup;
  if (group.toLowerCase() === 'last') return 'Latest provider snapshot';
  if (/^\d+year$/.test(group)) return `${Number.parseInt(group)} year${group.startsWith('1year') ? '' : 's'} · provider`;
  if (/^Year\d+$/.test(group)) return `Annual history · position ${group.slice(4)}`;
  if (/^RQ\d+$/.test(group)) return `R12 history · position ${group.slice(2)}`;
  if (/^Q\d+$/.test(group)) return `Quarter history · position ${group.slice(1)}`;
  return `${group.replace(/(\d)([a-z])/g, '$1 $2')} · provider`;
}
export function providerCalculationLabel(variant: ExpandedKpiVariant): string {
  const labels: Record<string, string> = { latest: 'Latest', default: 'Saved value', mean: 'Average', high: 'Highest', low: 'Lowest', sum: 'Sum', cagr: 'Annual growth', return: 'Return', History: 'Historical observation', year: 'Full year', quarter: 'Quarter growth', r12: 'Trailing 12 months', sek: 'SEK', usd: 'USD' };
  return labels[variant.calculation] ?? variant.calculation;
}
export const EXPANDED_KPIS: CompanyKpi[] = EXPANDED_KPI_MANIFEST.metrics.map(metric => {
  const variants = metric.variants.map(id => byVariant.get(id)).filter((v): v is ExpandedKpiVariant => !!v);
  const aliases: Record<number, string[]> = { 6: ['EPS'], 8: ['BVPS'], 33: ['ROE'], 34: ['ROA'], 35: ['ROTA'], 36: ['ROC', 'ROCE'], 37: ['ROIC'], 55: ['EBIT'], 54: ['EBITDA'], 42: ['leverage'] };
  return { id: metric.id, label: metric.providerKpiId === 158 ? 'Moving-average comparison' : metric.label, searchTerms: aliases[metric.providerKpiId], category: metric.category, description: metric.description,
    formula: `Börsdata KPI ${metric.providerKpiId}; provider calculation as downloaded. Underlying report and quote dates may be unavailable.`, unit: expandedUnit(metric.unit, metric.currencyBasis),
    windows: [...new Set(variants.map(v => `provider:${v.source}:${v.calcGroup}` as const))], calculations: [...new Set(variants.map(v => `provider:${v.calculation}` as const))],
    supportedVariants: variants.map(v => ({ window: `provider:${v.source}:${v.calcGroup}` as const, calculation: `provider:${v.calculation}` as const, windowLabel: providerWindowLabel(v), calculationLabel: providerCalculationLabel(v) })),
    coverage: Math.max(0, ...variants.map(v => v.availableCount)), source: 'Börsdata provider snapshot', snapshot: EXPANDED_KPI_MANIFEST.snapshot };
});

function validateDescriptor(value: ExpandedArtifact) {
  fail(object(value) && text(value.path) && /^data\/expanded-kpis\/[A-Za-z0-9_-]+\.bin$/.test(value.path));
  fail(hash(value.sha256) && hash(value.uncompressedSha256) && integer(value.bytes, 8 * 1024 * 1024) && value.bytes > 0 && integer(value.uncompressedBytes, 32 * 1024 * 1024) && value.uncompressedBytes > 0);
}
export function validateExpandedManifest(value: ExpandedKpiManifest) {
  fail(object(value) && value.format === 'macro-atlas-expanded-kpi-manifest' && value.version === 1 && day(value.snapshot) && hash(value.financialPackId) && hash(value.taxonomySha256));
  fail(integer(value.rows, 100000) && value.rows > 0 && Array.isArray(value.metrics) && value.metrics.length > 0 && value.metrics.length <= 500 && Array.isArray(value.variants) && value.variants.length <= 10000 && Array.isArray(value.shards) && value.shards.length <= 2000);
  validateDescriptor(value.index);
  const metrics = new Map<string, ExpandedKpiMetric>(), variants = new Map<string, ExpandedKpiVariant>(), shards = new Map<string | number, ExpandedArtifact>();
  for (const metric of value.metrics) {
    fail(object(metric) && /^provider_[1-9]\d*$/.test(metric.id) && integer(metric.providerKpiId, 10000) && metric.id === `provider_${metric.providerKpiId}` && !metrics.has(metric.id));
    fail(text(metric.label) && metric.label.length > 0 && text(metric.category) && text(metric.description, 12000) && units.includes(metric.unit) && currencyBases.includes(metric.currencyBasis) && Array.isArray(metric.variants) && metric.variants.length <= 1000);
    metrics.set(metric.id, metric);
  }
  let bytes = 0;
  for (const shard of value.shards) {
    validateDescriptor(shard); fail(shard.id !== undefined && !shards.has(shard.id) && Array.isArray(shard.variantIds) && shard.variantIds.length > 0 && shard.variantIds.length <= 16);
    bytes += shard.bytes; shards.set(shard.id, shard);
  }
  fail(bytes <= 300 * 1024 * 1024);
  for (const variant of value.variants) {
    const shard = shards.get(variant.shard);
    fail(object(variant) && text(variant.id, 200) && !variants.has(variant.id) && metrics.has(variant.metricId) && [variant.calcGroup, variant.calculation, variant.source, variant.label].every(v => text(v)));
    fail(units.includes(variant.unit) && currencyBases.includes(variant.currencyBasis) && integer(variant.availableCount, value.rows) && integer(variant.conflictCount, value.rows));
    fail(shard && integer(variant.offset, 15) && shard.variantIds?.[variant.offset] === variant.id && Array.isArray(variant.notes) && variant.notes.every(n => text(n, 12000)));
    variants.set(variant.id, variant);
  }
  fail([...metrics.values()].every(m => new Set(m.variants).size === m.variants.length && m.variants.every(id => variants.get(id)?.metricId === m.id)));
  fail([...shards.values()].every(s => s.variantIds?.every((id, offset) => variants.get(id)?.shard === s.id && variants.get(id)?.offset === offset)));
  return value;
}
export function validateExpandedBinding(manifest: ExpandedKpiManifest, financial: FinancialIndex | null, taxonomy: string | null | undefined) {
  fail(financial && financial.id === manifest.financialPackId && financial.taxonomy_sha256 === manifest.taxonomySha256 && taxonomy === manifest.taxonomySha256 && Object.keys(financial.companies).length === manifest.rows,
    'The additional KPIs belong to a different financial pack or company directory. Their values have been withheld.');
}
export function validateExpandedIndex(value: any, manifest: ExpandedKpiManifest, financial: FinancialIndex): ExpandedKpiIndex {
  fail(object(value) && value.format === 'macro-atlas-expanded-kpi-index' && value.version === 1 && value.snapshot === manifest.snapshot && Array.isArray(value.ids) && value.ids.length === manifest.rows && Array.isArray(value.reportCurrencies) && value.reportCurrencies.length === manifest.rows && Array.isArray(value.quoteCurrencies) && value.quoteCurrencies.length === manifest.rows);
  const ids = value.ids as string[];
  fail(ids.every((id, i) => typeof id === 'string' && /^[1-9][0-9]{0,9}$/.test(id) && !!financial.companies[id] && (!i || Number(ids[i - 1]) < Number(id))), 'Additional KPI listing identities do not match the selected directory.');
  fail(value.reportCurrencies.every(currency) && value.quoteCurrencies.every(currency));
  return { ids, reportCurrencies: value.reportCurrencies as (string | null)[], quoteCurrencies: value.quoteCurrencies as (string | null)[], positions: new Map(ids.map((id, i) => [id, i])) };
}
export function validateExpandedShard(value: any, descriptor: ExpandedArtifact, manifest: ExpandedKpiManifest): ExpandedLoadedVariant[] {
  fail(object(value) && value.format === 'macro-atlas-expanded-kpi-shard' && value.version === 1 && value.snapshot === manifest.snapshot && Array.isArray(value.variantIds) && Array.isArray(value.values) && JSON.stringify(value.variantIds) === JSON.stringify(descriptor.variantIds) && value.values.length === descriptor.variantIds?.length);
  const variants = new Map(manifest.variants.map(v => [v.id, v]));
  return value.values.map((values: unknown, i: number) => {
    const variant = variants.get((value.variantIds as string[])[i])!; fail(variant && Array.isArray(values) && values.length === manifest.rows);
    let available = 0;
    for (const value of values) { fail(value === null || (variant.unit === 'date' || variant.unit === 'text' ? text(value, 12000) : finite(value))); if (value !== null) available++; }
    fail(available === variant.availableCount, 'Additional KPI coverage does not reconcile with its manifest.');
    return { variant, values: values as (number | string | null)[] };
  });
}
export async function decodeExpandedArtifact(bytes: Uint8Array, descriptor: ExpandedArtifact): Promise<unknown> {
  validateDescriptor(descriptor);
  fail(bytes.byteLength === descriptor.bytes && await researchGaugeHash(bytes) === descriptor.sha256, 'Additional KPI checksum failed. Values have been withheld.');
  const stream = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(new DecompressionStream('gzip'));
  const raw = await readResearchGaugeBytes(stream, descriptor.uncompressedBytes);
  fail(await researchGaugeHash(raw) === descriptor.uncompressedSha256, 'Decompressed KPI checksum failed.');
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw));
}
export function expandedKpiCell(row: ResearchGaugeRow, column: CompanyListColumn, context?: ExpandedKpiContext): CompanyListCell {
  const variant = expandedVariant(column), metric = byMetric.get(column.kpiId), unit = expandedUnit(variant?.unit ?? metric?.unit ?? 'number', variant?.currencyBasis ?? metric?.currencyBasis);
  const base = `${metric?.description ?? 'Provider KPI.'} Saved provider snapshot ${EXPANDED_KPI_MANIFEST.snapshot}. ${variant?.label ?? 'Unsupported selection'}. This is a downloaded provider result, separate from the standard DCF starter. ${variant?.notes.join(' ') ?? ''}`;
  const missing = (reason: string, status: CompanyListCell['status'] = 'missing'): CompanyListCell => ({ value: null, display: status === 'loading' ? '…' : '—', unit, currency: null, date: null, detail: `${base} ${reason}`, status });
  if (!variant) return missing('This period/calculation combination is unsupported.');
  if (context?.error) return missing(context.error, 'error');
  if (context?.errors.has(variant.id)) return missing(context.errors.get(variant.id)!, 'error');
  const loaded = context?.variants.get(variant.id), position = context?.index?.positions.get(row.id);
  if (!loaded || !context?.index) return missing('Loading the selected KPI values.', 'loading');
  if (variant.currencyBasis === 'unverified') return missing('This monetary field has no established currency or scale. Its values are withheld from display, comparisons and numeric thresholds.');
  if (position === undefined) return missing('This listing has no corresponding downloaded provider identity.');
  const value = loaded.values[position];
  if (value === null || value === undefined) return missing('No usable value in this snapshot. Missing observations, conflicting records and documented data defects are withheld; they are not zeros.');
  const basis = variant.currencyBasis;
  const currency = basis === 'report' ? context.index.reportCurrencies[position] : basis === 'quote' ? context.index.quoteCurrencies[position] : basis === 'SEK' || basis === 'USD' ? basis : null;
  if ((unit === 'money' || unit === 'price') && !currency) return missing('The monetary currency is not established for this listing.');
  const number = typeof value === 'number' ? (value !== 0 && Math.abs(value) < .01 ? smallFormatter : formatter).format(value) : value;
  const display = unit === 'shares_millions' ? `${number} m shares` : unit === 'money' ? `${number} ${currency} m` : unit === 'price' ? `${number} ${currency}` : unit === 'percent' ? `${number}%` : unit === 'multiple' ? `${number}×` : String(number);
  return { value, display, unit, currency, date: null, status: 'available', detail: `${base} Exact provider value: ${value}${currency ? ` ${currency}${unit === 'money' ? ' million' : ' per share'}` : ''}. Snapshot date is not an underlying report or quote date. API key: ${metric?.providerKpiId}/${variant.calcGroup}/${variant.calculation} (${variant.source}).` };
}
