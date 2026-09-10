import { describe, expect, it } from 'vitest';
import { decodePackage, digest, previousRelease } from './research';
import type { ResearchRelease } from './types';
import businessFixture from '../tests/fixtures/business.json';
import { businessIndex, financialSeries } from './business';
import taxonomyFixture from '../tests/fixtures/taxonomy.json';
import { directoryTotals, findBranches, fold, validateTaxonomy } from './taxonomy';

const categories = ['real_stuff', 'production', 'exchange', 'promises', 'enforcer'];
function fixture() {
  return {
    version: 1, as_of: '2021-01-01', generated_at: '2021-01-01T12:00:00+00:00', categories, ranking_population: ['SE'], trade: [],
    indicators: categories.map(name => ({ name, category: name, label: name, unit: '%', scored: true, higher_is_better: false, uncertainty: 'A', description: 'Synthetic package test', cadence: 'A', sources: ['TEST'], forward: false })),
    countries: { SE: { name: 'Test Sweden', iso3: 'SWE', currency: 'SEK', on_map: true, data_quality: { flag: 'test', note: null }, pressures: [],
      categories: Object.fromEntries(categories.map(k => [k, { score: 50, n_available: 1, n_total: 1 }])),
      indicators: Object.fromEntries(categories.map(k => [k, { value: 0, pct: 50, date: null, source: null, is_forecast: false }])),
      history: { promises: [{ year: 2020, value: null, is_forecast: false }, { year: 2021, value: 0, is_forecast: true }] },
    } },
  };
}
async function envelope(raw = fixture(), liquidity: unknown = null) {
  const content = JSON.stringify(raw);
  const liquidContent = JSON.stringify(liquidity);
  return { format: 'macro-atlas-research', schema_version: 1, fundamentals: { source_file: 'synthetic.json', sha256: await digest(content), content }, liquidity: liquidity ? { source_file: 'liquidity.json', content: liquidContent, sha256: await digest(liquidContent) } : null };
}
describe('portable research files', () => {
  it('preserves source hashes, zero, null and forecast boundaries', async () => {
    const payload = await envelope();
    const decoded = await decodePackage(JSON.stringify(payload));
    expect(decoded.release.fundamentals_sha256).toBe(await digest(payload.fundamentals.content));
    expect(decoded.fundamentals.countries.SE.history).toEqual(fixture().countries.SE.history);
    expect(decoded.fundamentals.countries.SE.indicators.promises.value).toBe(0);
    expect(decoded.liquidity).toBeNull();
  });
  it('rejects damaged content and unsupported schema before import', async () => {
    const payload = await envelope(); payload.fundamentals.content = '{}';
    await expect(decodePackage(JSON.stringify(payload))).rejects.toThrow(/checksum/);
    await expect(decodePackage(JSON.stringify({ ...await envelope(), schema_version: 4 }))).rejects.toThrow(/version/);
  });
  it('rejects invalid scores, dates and duplicate history years despite valid checksums', async () => {
    const invalidScore = fixture(); invalidScore.countries.SE.categories.promises.score = 150;
    await expect(decodePackage(JSON.stringify(await envelope(invalidScore)))).rejects.toThrow(/scores/);
    const invalidDate = fixture(); invalidDate.as_of = '2021-02-30';
    await expect(decodePackage(JSON.stringify(await envelope(invalidDate)))).rejects.toThrow(/date/);
    const duplicate = fixture(); duplicate.countries.SE.history.promises[1].year = 2020;
    await expect(decodePackage(JSON.stringify(await envelope(duplicate)))).rejects.toThrow(/duplicate/);
  });
  it('rejects incomplete liquidity reports instead of accepting broken panels', async () => {
    await expect(decodePackage(JSON.stringify(await envelope(fixture(), { version: 1, methodology_version: 'liquidity-diagnostics-v1', as_of: '2021-01-01', as_known_at: '2021-01-01T12:00:00Z', snapshot_sha256: 'x', methodology_sha256: 'y', broad_money: [], offshore_credit: [], input_releases: [], coverage: {} }))))
      .rejects.toThrow(/panel/);
  });
  it('selects an earlier distinct fundamentals vintage, skipping liquidity-only updates', () => {
    const release = (id: string, generated_at: string, hash: string) => ({ id, generated_at, fundamentals_sha256: hash }) as ResearchRelease;
    const old = release('old', '2021-01-01T12:00:00Z', 'old-fund');
    const same = release('same', '2021-01-03T12:00:00Z', 'current-fund');
    const current = release('current', '2021-01-04T12:00:00+00:00', 'current-fund');
    expect(previousRelease([current, same, old], current)?.id).toBe('old');
    expect(previousRelease([current, same], current)).toBeUndefined();
  });
});

describe('taxonomy directory and v3 identity', () => {
  async function directoryPackage(raw = structuredClone(taxonomyFixture)) {
    const businessContent = JSON.stringify(businessFixture);
    const business = { source_file: 'business.json', content: businessContent, sha256: await digest(businessContent) };
    const content = JSON.stringify({ ...raw, business_sha256: business.sha256 });
    return { ...await envelope(), schema_version: 3, business, taxonomy: { source_file: 'taxonomy.json', content, sha256: await digest(content) } };
  }
  it('includes the directory hash in identity and retains the source classification', async () => {
    const packageValue = await directoryPackage();
    const decoded = await decodePackage(JSON.stringify(packageValue));
    expect(decoded.release.id).toBe(await digest(`macro-atlas-research-v3\n${packageValue.fundamentals.sha256}\n\n${packageValue.business.sha256}\n${packageValue.taxonomy.sha256}`));
    expect(decoded.taxonomy?.classifications['102'].source_branch_id).toBe('21');
    expect(decoded.release.branch_count).toBe(2);
    expect(decoded.release.sector_count).toBe(2);
  });
  it('does not accept an unhashed directory in v2 or a mismatched business binding', async () => {
    await expect(decodePackage(JSON.stringify({ ...await directoryPackage(), schema_version: 2 }))).rejects.toThrow(/version/i);
    const packageValue = await directoryPackage();
    const raw = JSON.parse(packageValue.taxonomy.content); raw.business_sha256 = 'b'.repeat(64);
    packageValue.taxonomy.content = JSON.stringify(raw); packageValue.taxonomy.sha256 = await digest(packageValue.taxonomy.content);
    await expect(decodePackage(JSON.stringify(packageValue))).rejects.toThrow(/business document/);
    packageValue.taxonomy.content = 'null'; packageValue.taxonomy.sha256 = await digest('null');
    await expect(decodePackage(JSON.stringify(packageValue))).rejects.toThrow(/taxonomy/i);
  });
  it('searches bilingual names and counts shared dossiers once without inferring completion', () => {
    validateTaxonomy(taxonomyFixture, businessFixture, 'a'.repeat(64));
    const index = businessIndex(businessFixture);
    expect(findBranches(taxonomyFixture, 'skogs', 'all', 'all', index).map(b => b.id)).toEqual(['21']);
    expect(findBranches(taxonomyFixture, 'construction', '5', 'all', index).map(b => b.id)).toEqual(['31']);
    expect(findBranches(taxonomyFixture, '', 'all', 'profiles', index).map(b => b.id)).toEqual(['21']);
    expect(fold('Hälsovård')).toBe('halsovard');
    expect(directoryTotals(taxonomyFixture)).toEqual({ sectors: 2, branches: 2, studies: 1, dives: 1 });
  });
  it('moves displayed profiles only when a correction matches its source and target', () => {
    const raw: any = structuredClone(taxonomyFixture);
    const c = raw.classifications['102'];
    c.correction = { company_id: '102', expected_sector_id: '7', expected_branch_id: '21', branch_id: '31', reason: 'Synthetic correction', source: 'Synthetic filing', reviewed_at: '2026-09-10' };
    c.status = 'corrected'; c.branch_id = '31'; c.sector_id = '5';
    validateTaxonomy(raw, businessFixture, 'a'.repeat(64));
    expect(findBranches(raw, '', 'all', 'profiles', businessIndex(businessFixture)).map(b => b.id)).toEqual(['31']);
    c.branch_id = '21';
    expect(() => validateTaxonomy(raw, businessFixture, 'a'.repeat(64))).toThrow(/effective classification/);
  });
  it('requires typed shared-branch IDs and real inventory timestamps', () => {
    const raw: any = structuredClone(taxonomyFixture);
    raw.shared_studies.shared.branch_ids = [21, 31];
    expect(() => validateTaxonomy(raw, businessFixture, 'a'.repeat(64))).toThrow();
    for (const exported_at of ['2026-02-30T12:00:00Z', '2026-09-10T24:00:00Z', '1999-01-01T00:00:00Z', '2026-09-10T12:00:00junkZ']) {
      expect(() => validateTaxonomy({ ...taxonomyFixture, exported_at }, businessFixture, 'a'.repeat(64))).toThrow();
    }
    const standalone: any = { ...taxonomyFixture, business_sha256: null, classification_as_of: null, classifications: {} };
    validateTaxonomy(standalone, null, null);
    delete (standalone as Record<string, unknown>).business_sha256;
    expect(() => validateTaxonomy(standalone, null, null)).toThrow(/does not match/);
  });
});

describe('business research packages', () => {
  async function companyPackage(raw = structuredClone(businessFixture)) {
    const content = JSON.stringify(raw);
    return { ...await envelope(), schema_version: 2, business: { source_file: 'business.json', sha256: await digest(content), content } };
  }
  it('hashes all documents, keeps v1 identity and projects a small company catalogue', async () => {
    const legacy = await envelope();
    const v1 = await decodePackage(JSON.stringify(legacy));
    expect(v1.release.id).toBe(await digest(`macro-atlas-research-v1\n${legacy.fundamentals.sha256}\n`));
    expect(v1.business).toBeNull();
    const payload = await companyPackage();
    const decoded = await decodePackage(JSON.stringify(payload));
    expect(decoded.release.id).toBe(await digest(`macro-atlas-research-v2\n${payload.fundamentals.sha256}\n\n${payload.business.sha256}`));
    expect(decoded.release.company_count).toBe(1);
    const index = businessIndex(decoded.business!);
    expect(index.common_year).toBe(2020);
    expect(index.companies['102'].latest_annual?.values.cash_flow_from_operating_activities).toBe(0);
    expect(index.companies['102']).not.toHaveProperty('annual');
    expect(index).not.toHaveProperty('research');
  });
  it('rejects business checksum damage, duplicate periods and a v1 document collision', async () => {
    const damaged = await companyPackage(); damaged.business.content = '{}';
    await expect(decodePackage(JSON.stringify(damaged))).rejects.toThrow(/checksum/);
    const raw = structuredClone(businessFixture); raw.companies['102'].annual.push(raw.companies['102'].annual[0]);
    await expect(decodePackage(JSON.stringify(await companyPackage(raw)))).rejects.toThrow(/Duplicate/);
    await expect(decodePackage(JSON.stringify({ ...await companyPackage(), schema_version: 1 }))).rejects.toThrow(/version/i);
  });
  it('rejects missing financial fields and keeps gaps and changed currency out of amount charts', async () => {
    const decoded = await decodePackage(JSON.stringify(await companyPackage()));
    const first = decoded.business!.companies['102'].annual[0];
    const third = { ...first, year: 2022, currency: 'EUR' };
    expect(financialSeries([first, third], 'revenues', 'SEK').values).toEqual([100, null, null]);
    expect(financialSeries([first, third], 'operating_margin', 'SEK').values).toEqual([10, null, 10]);
    const raw = structuredClone(businessFixture); delete (raw.companies['102'].annual[0].values as Record<string, unknown>).revenues;
    await expect(decodePackage(JSON.stringify(await companyPackage(raw)))).rejects.toThrow(/business/);
  });
});
