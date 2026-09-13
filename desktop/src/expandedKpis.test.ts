import { describe, expect, it } from 'vitest';
import type { FinancialIndex } from './financialData';
import type { ResearchGaugeRow } from './researchGaugeModel';
import { companyListUnit, companyListVariants, COMPANY_KPIS, defaultCompanyListPreferences, parseCompanyListPreferences } from './companyListModel';
import { decodeExpandedArtifact, expandedKpiCell, expandedVariant, EXPANDED_KPI_MANIFEST, validateExpandedBinding, validateExpandedIndex, validateExpandedManifest, validateExpandedShard } from './expandedKpis';
import type { ExpandedKpiManifest, ExpandedKpiContext } from './expandedKpis';
import { researchGaugeHash } from './researchGauge';

const hash = 'a'.repeat(64), taxonomy = 'b'.repeat(64);
function fixture() {
  const descriptor = { path: 'data/expanded-kpis/example.bin', sha256: hash, bytes: 10, uncompressedSha256: hash, uncompressedBytes: 20 };
  const variant = { id: 'provider_6_last_latest_screener', metricId: 'provider_6', calcGroup: 'last', calculation: 'latest', source: 'screener', label: 'EPS latest', unit: 'per_share' as const, currencyBasis: 'report' as const, availableCount: 1, conflictCount: 0, shard: 0, offset: 0, notes: [] };
  const manifest: ExpandedKpiManifest = { format: 'macro-atlas-expanded-kpi-manifest', version: 1, snapshot: '2026-08-10', financialPackId: hash, taxonomySha256: taxonomy, rows: 2, index: descriptor, metrics: [{ id: 'provider_6', providerKpiId: 6, label: 'EPS', category: 'Ratios', description: 'Downloaded EPS', unit: 'per_share', currencyBasis: 'report', variants: [variant.id] }], variants: [variant], shards: [{ ...descriptor, id: 0, variantIds: [variant.id] }] };
  const financial = { id: hash, taxonomy_sha256: taxonomy, companies: { '1': {}, '2': {} } } as unknown as FinancialIndex;
  const index = { format: 'macro-atlas-expanded-kpi-index', version: 1, snapshot: manifest.snapshot, ids: ['1', '2'], reportCurrencies: ['JPY', null], quoteCurrencies: ['EUR', 'SEK'] };
  const shard = { format: 'macro-atlas-expanded-kpi-shard', version: 1, snapshot: manifest.snapshot, variantIds: [variant.id], values: [[412.69, null]] };
  return { manifest, financial, index, shard, variant };
}
describe('expanded KPI integrity and definitions', () => {
  it('accepts the exported catalogue and exact variant pairs, never an invented combination', () => {
    expect(validateExpandedManifest(EXPANDED_KPI_MANIFEST).metrics.length).toBe(210);
    const metric = COMPANY_KPIS.find(kpi => kpi.id === 'provider_151')!;
    const variants = companyListVariants(metric);
    const price = { id: 'price', kpiId: metric.id, ...variants.find(v => v.window.endsWith(':last') && v.calculation === 'provider:default')! };
    const performance = { id: 'performance', kpiId: metric.id, ...variants.find(v => v.window.endsWith(':1year') && v.calculation === 'provider:return')! };
    const shares = COMPANY_KPIS.find(kpi => kpi.id === 'provider_61')!;
    const shareColumn = { id: 'shares', kpiId: shares.id, ...companyListVariants(shares).find(v => v.window.endsWith(':last') && v.calculation === 'provider:latest')! };
    expect(companyListUnit(shareColumn)).toBe('shares_millions');
    expect(companyListUnit(price)).toBe('price'); expect(companyListUnit(performance)).toBe('percent');
    expect(expandedVariant({ ...price, calculation: 'provider:imaginary' })).toBeUndefined();
    const preferences = parseCompanyListPreferences({ ...defaultCompanyListPreferences(), columns: [price, performance] });
    expect(preferences.columns.map(c => c.id)).toEqual(['price', 'performance']);
  });
  it('rejects different source bindings and incomplete or duplicate listing identities', () => {
    const f = fixture(); validateExpandedBinding(f.manifest, f.financial, taxonomy);
    expect(() => validateExpandedBinding(f.manifest, { ...f.financial, id: taxonomy }, taxonomy)).toThrow(/different/);
    expect(() => validateExpandedBinding(f.manifest, f.financial, hash)).toThrow(/different/);
    expect(validateExpandedIndex(f.index, f.manifest, f.financial).positions.get('2')).toBe(1);
    for (const ids of [['1'], ['1', '1'], ['2', '1'], ['1', '3']]) expect(() => validateExpandedIndex({ ...f.index, ids }, f.manifest, f.financial)).toThrow();
    expect(() => validateExpandedIndex({ ...f.index, snapshot: '2026-08-11' }, f.manifest, f.financial)).toThrow();
    expect(() => validateExpandedIndex({ ...f.index, reportCurrencies: ['?', null] }, f.manifest, f.financial)).toThrow();
  });
  it('rejects unbounded, mislinked or duplicate artifact descriptions', () => {
    for (const mutate of [
      (m: ExpandedKpiManifest) => { m.index.path = '../outside.bin'; },
      (m: ExpandedKpiManifest) => { m.index.bytes = 1e9; },
      (m: ExpandedKpiManifest) => { m.variants[0].offset = 1; },
      (m: ExpandedKpiManifest) => { m.metrics.push(m.metrics[0]); },
      (m: ExpandedKpiManifest) => { m.shards.push(m.shards[0]); },
    ]) { const f = fixture(); mutate(f.manifest); expect(() => validateExpandedManifest(f.manifest)).toThrow(); }
  });
  it('reconciles dimensions, numeric types, source snapshot and observed coverage', () => {
    const f = fixture();
    expect(validateExpandedShard(f.shard, f.manifest.shards[0], f.manifest)[0].values).toEqual([412.69, null]);
    for (const values of [[[412.69]], [[412.69, 0]], [['412.69', null]], [[Infinity, null]]]) expect(() => validateExpandedShard({ ...f.shard, values }, f.manifest.shards[0], f.manifest)).toThrow();
    expect(() => validateExpandedShard({ ...f.shard, snapshot: '2026-08-11' }, f.manifest.shards[0], f.manifest)).toThrow();
  });
  it('verifies compressed and decompressed content, including the exact decompression limit', async () => {
    const raw = new TextEncoder().encode('{"verified":true}'), bytes = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
    const descriptor = { ...fixture().manifest.index, bytes: bytes.length, sha256: await researchGaugeHash(bytes), uncompressedBytes: raw.length, uncompressedSha256: await researchGaugeHash(raw) };
    expect(await decodeExpandedArtifact(bytes, descriptor)).toEqual({ verified: true });
    await expect(decodeExpandedArtifact(bytes, { ...descriptor, sha256: hash })).rejects.toThrow(/checksum/);
    await expect(decodeExpandedArtifact(bytes, { ...descriptor, uncompressedBytes: raw.length - 1 })).rejects.toThrow();
    await expect(decodeExpandedArtifact(bytes, { ...descriptor, uncompressedSha256: hash })).rejects.toThrow(/checksum/);
  });
  it('keeps report and quote currencies distinct and never invents an observation date', () => {
    const f = fixture(), column = { id: 'eps', kpiId: 'provider_6', window: 'provider:screener:last' as const, calculation: 'provider:latest' as const };
    const index = validateExpandedIndex(f.index, f.manifest, f.financial);
    const context: ExpandedKpiContext = { index, variants: new Map([[f.variant.id, { variant: f.variant, values: [412.69, null] }]]), loading: new Set(), errors: new Map(), error: '', ready: true };
    const row = { id: '1' } as ResearchGaugeRow;
    expect(expandedKpiCell(row, column, context)).toMatchObject({ value: 412.69, currency: 'JPY', date: null, status: 'available' });
    expect(expandedKpiCell({ id: '2' } as ResearchGaugeRow, column, context)).toMatchObject({ value: null, status: 'missing' });
    expect(expandedKpiCell(row, column)).toMatchObject({ value: null, status: 'loading' });
    expect(expandedKpiCell(row, column, { ...context, error: 'Checksum failed' })).toMatchObject({ value: null, status: 'error' });
    expect(expandedKpiCell(row, column, { ...context, errors: new Map([[f.variant.id, 'Shard failed']]) }).detail).toContain('Shard failed');
    index.reportCurrencies[0] = null;
    expect(expandedKpiCell(row, column, context).value).toBeNull();
  });
  it('migrates old preferences without resurrecting deleted default watchlist members in v2', () => {
    const migrated = parseCompanyListPreferences({ ...defaultCompanyListPreferences(), version: 1, watchlistIds: ['1', '2'] });
    expect(migrated.version).toBe(2); expect(migrated.features.watchlists[0].listingIds).toEqual(['1', '2']);
    migrated.features.watchlists[0].listingIds = ['2'];
    expect(parseCompanyListPreferences(migrated).watchlistIds).toEqual(['2']);
  });
});
