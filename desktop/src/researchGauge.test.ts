import { describe, expect, it } from 'vitest';
import type { FinancialIndex } from './financialData';
import type { ResearchGaugeArtifact, ResearchGaugeManifest, ResearchGaugeRow } from './researchGaugeModel';
import { decodeResearchGauge, readResearchGaugeBytes, researchGaugeHash, validateResearchGaugeArtifact, validateResearchGaugeManifest } from './researchGauge';

const pack = 'a'.repeat(64), taxonomy = 'b'.repeat(64), companyHash = 'c'.repeat(64);
function fixture() {
  const series = () => ({ values: [], count: 0, positive: 0, negative: 0, zero: 0, latest: null, median: null, dispersion: null });
  const row: ResearchGaugeRow = {
    id: '1', name: 'Missing history example', ticker: null, isin: null, country: 'SE', sectorId: '2', sectorName: 'Example', branchId: '3', branchName: 'Example', sourceAsOf: '2026-08-10', presence: 'latest', sourceCompanySha256: companyHash,
    route: 'operating', classificationConflict: false, readiness: 'no_history', issues: ['No retained annual report is available.'],
    annual: { currency: null, latest: null, periods: [], excluded: { outsideWindow: 0, placeholder: 0, missingPublication: 0, withheld: 0 }, cash: series(), operatingCash: series(), ebit: series(), revenue: series(), margins: series(),
      latestValues: { revenues: null, ebit: null, cash: null, operatingCash: null, financingCash: null, cashBalance: null, netDebt: null, equity: null, assets: null, netDebtToAssetsPercent: null, equityToAssetsPercent: null, tangibleAssetsToRevenue: null, intangibleAssetsToAssetsPercent: null, cashComponentDifference: null }, reason: 'No retained annual report is available.' },
    quarter: { latest: null, comparison: null, revenueChangePercent: null, ebitMarginChangePoints: null, cashChange: null, reason: 'No quarters.' },
    valuation: { status: 'missing', currency: 'SEK', value: null, cashPV: null, terminalPV: null, terminalShare: null, candidateEquity: null, priceDate: null, priceAgeDays: null, priceBasis: null, ceiling: null, lowValue: null, lowNPV: null, reverseCashFactor: null, reverseCashFactor30: null, hasSignedCash: false, annualHistoryCount: 0, rangeStatus: 'unavailable', reason: 'Missing annual cash.' },
  };
  const data: ResearchGaugeArtifact = { format: 'macro-atlas-research-gauge', version: 1, model: 'research-gauge-v1', asOf: '2026-09-13', financialPackId: pack, taxonomySha256: taxonomy, calibrationId: 'example-calibration', rows: [row] };
  const manifest: ResearchGaugeManifest = { format: 'macro-atlas-research-gauge-manifest', version: 1, model: data.model, asOf: data.asOf, financialPackId: pack, financialPackBytes: 1000, taxonomySha256: taxonomy, calibrationId: data.calibrationId, rows: 1, artifact: { path: 'data/research-gauge/gauge.bin', sha256: pack, bytes: 1, uncompressedSha256: pack, uncompressedBytes: 1 } };
  const financial = { id: pack, bytes: 1000, taxonomy_sha256: taxonomy, as_of: '2026-08-10', companies: { '1': { sha256: companyHash } }, summary: { listings: 1 }, sources: [] } as unknown as FinancialIndex;
  return { data, row, manifest, financial };
}

describe('source-bound research screen', () => {
  it('keeps a listing with no history and no valuation visible as missing', () => {
    const f = fixture();
    expect(validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy).rows[0].valuation.value).toBeNull();
  });
  it('rejects different packs, taxonomy, company sources and incomplete coverage', () => {
    const f = fixture();
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, { ...f.financial, id: companyHash }, taxonomy)).toThrow(/different/);
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, companyHash)).toThrow(/different/);
    f.row.sourceCompanySha256 = pack;
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow(/company source/);
    f.data.rows = [];
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow(/coverage/);
  });
  it('rejects duplicate rows even if their total count is correct', () => {
    const f = fixture(); f.manifest.rows = 2; f.financial.summary.listings = 2; f.financial.companies['2'] = f.financial.companies['1']; f.data.rows.push(structuredClone(f.row));
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow(/duplicate/);
  });
  it('preserves signed funding, terminal contribution above 100 percent and independently checks reverse factors', () => {
    const f = fixture();
    Object.assign(f.row.valuation, { status: 'positive-priced', value: 100, cashPV: -20, terminalPV: 120, terminalShare: 1.2, candidateEquity: 140, priceDate: '2026-09-03', priceAgeDays: 10, ceiling: 70, lowValue: -50, lowNPV: -190, reverseCashFactor: 1.4, reverseCashFactor30: 2, hasSignedCash: true });
    expect(validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toBe(f.data);
    f.row.valuation.reverseCashFactor30 = 1.4;
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow();
  });
  it('does not turn a nonpositive valuation into a positive ceiling or automatic financial-business forecast', () => {
    const f = fixture(); Object.assign(f.row.valuation, { status: 'nonpositive', value: 0, cashPV: -50, terminalPV: 50 });
    expect(validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toBe(f.data);
    f.row.valuation.ceiling = 0;
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow();
    f.row.valuation.ceiling = null; f.row.route = 'financial'; f.row.valuation.status = 'manual-financial';
    expect(() => validateResearchGaugeArtifact(f.data, f.manifest, f.financial, taxonomy)).toThrow(/readiness/);
  });
  it('rejects unsafe paths, oversized assets and malformed manifests', () => {
    const f = fixture(); expect(validateResearchGaugeManifest(f.manifest)).toBe(f.manifest);
    for (let repeat = 0; repeat < 2; repeat++) expect(() => validateResearchGaugeManifest({ ...f.manifest, asOf: '2026-02-30' })).toThrow();
    f.manifest.artifact.path = 'https://example.test/gauge.json.gz'; expect(() => validateResearchGaugeManifest(f.manifest)).toThrow();
    f.manifest.artifact.path = 'data/research-gauge/gauge.bin'; f.manifest.artifact.uncompressedBytes = 129 * 1024 * 1024; expect(() => validateResearchGaugeManifest(f.manifest)).toThrow(/size/);
  });
  it('verifies compressed and decompressed hashes before accepting a complete artifact', async () => {
    const f = fixture(), raw = new TextEncoder().encode(JSON.stringify(f.data));
    const compressed = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
    Object.assign(f.manifest.artifact, { bytes: compressed.length, sha256: await researchGaugeHash(compressed), uncompressedBytes: raw.length, uncompressedSha256: await researchGaugeHash(raw) });
    expect(await decodeResearchGauge(compressed, f.manifest, f.financial, taxonomy)).toEqual(f.data);
    const corrupt = compressed.slice(); corrupt[20] ^= 1;
    await expect(decodeResearchGauge(corrupt, f.manifest, f.financial, taxonomy)).rejects.toThrow(/checksum/);
    f.manifest.artifact.uncompressedSha256 = pack;
    await expect(decodeResearchGauge(compressed, f.manifest, f.financial, taxonomy)).rejects.toThrow(/checksum/);
  });
  it('rejects truncated and oversized streams before parsing', async () => {
    await expect(readResearchGaugeBytes(new Blob(['abc']).stream(), 2)).rejects.toThrow(/size/);
    await expect(readResearchGaugeBytes(new Blob(['abc']).stream(), 4)).rejects.toThrow(/incomplete/);
  });
});
