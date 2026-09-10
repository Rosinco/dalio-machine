import { describe, expect, it } from 'vitest';
import fixture from '../tests/fixtures/taxonomy-listings.json';
import business from '../tests/fixtures/business.json';
import { businessIndex } from './business';
import { companyEntries, countBranches, presenceMatches } from './listingCatalogue';
import { findBranches, validateTaxonomy } from './taxonomy';

const validate = (raw: unknown) => validateTaxonomy(raw, business, 'a'.repeat(64));
describe('downloaded company directory', () => {
  it('keeps all listing IDs, missing metadata and older rows without inventing financial profiles', () => {
    validateTaxonomy(fixture, business, 'a'.repeat(64));
    const index = businessIndex(business), entries = companyEntries(index, fixture);
    expect(entries).toHaveLength(6);
    expect(entries.filter(e => e.profile).map(e => e.id)).toEqual(['102']);
    expect(entries.find(e => e.id === '204')).toMatchObject({ listing_country: null, branch_id: null, display_name: 'Listing 204', profile: null });
    expect(entries.filter(e => e.isin === 'FI0000000001')).toHaveLength(6);
    expect(entries.filter(e => presenceMatches(e, 'older', fixture.catalogue.as_of)).map(e => e.id)).toEqual(['203']);
    expect(entries.filter(e => presenceMatches(e, 'latest', fixture.catalogue.as_of))).toHaveLength(5);
    expect(entries.find(e => e.id === '201')?.search).toContain('aland');
    const counts = countBranches(entries, fixture);
    expect(counts).toEqual({ '21': { listings: 2, profiles: 1 }, '31': { listings: 3, profiles: 0 } });
    expect(findBranches(fixture, '', 'all', 'listings', index, counts)).toHaveLength(2);
    expect(findBranches(fixture, '', 'all', 'profiles', index, counts).map(b => b.id)).toEqual(['21']);
    const olderCounts = countBranches(entries.filter(e => presenceMatches(e, 'older', fixture.catalogue.as_of)), fixture);
    expect(findBranches(fixture, '', 'all', 'profiles', index, olderCounts)).toHaveLength(0);
  });
  it('rejects missing listings, fabricated countries, wrong counts and classification drift despite valid JSON', () => {
    const mutations: ((raw: any) => void)[] = [
      raw => { delete raw.catalogue.listings['102']; },
      raw => { delete raw.classifications['201']; },
      raw => { raw.catalogue.listings['204'].listing_country = 'SE'; },
      raw => { raw.catalogue.snapshots[1].company_count++; raw.catalogue.snapshots[1].instrument_count++; },
      raw => { raw.catalogue.listings['201'].source_as_of = '2026-08-09'; },
      raw => { raw.classifications['205'].source_sector_id = '5'; },
      raw => { raw.catalogue.included_types[0] = '0'; },
      raw => { raw.catalogue.listings['204'].listing_date = '1799-01-01'; },
      raw => { raw.catalogue.listings['204'].listing_date = '2026-02-30'; },
      raw => { raw.version = 1; },
    ];
    for (const mutate of mutations) { const raw: any = structuredClone(fixture); mutate(raw); expect(() => validate(raw)).toThrow(); }
    for (const exported_at of ['2026-02-30T12:00:00Z', '1999-01-01T00:00:00Z', '2026-09-10T12:00:00junkZ', '2026-09-10T24:00:00Z']) {
      const raw = structuredClone(fixture); raw.catalogue.exported_at = exported_at; expect(() => validate(raw)).toThrow();
    }
  });
  it('applies reviewed corrections to directory-only listings while retaining conflicting source IDs', () => {
    const raw: any = structuredClone(fixture), c = raw.classifications['205'];
    c.correction = { company_id: '205', expected_sector_id: '3', expected_branch_id: '31', branch_id: '21', reason: 'Synthetic review', source: 'Test filing', reviewed_at: '2026-09-10' };
    c.branch_id = '21'; c.sector_id = '7'; c.status = 'corrected';
    validate(raw);
    expect(c.source_sector_id).toBe('3');
    c.correction.expected_sector_id = '5';
    expect(() => validate(raw)).toThrow();
    c.branch_id = '31'; c.sector_id = '5'; c.status = 'needs_review';
    validate(raw);
  });
});
