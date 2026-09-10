import { describe, expect, it } from 'vitest';
import { decodePackage, digest, previousRelease } from './research';
import type { ResearchRelease } from './types';

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
    await expect(decodePackage(JSON.stringify({ ...await envelope(), schema_version: 2 }))).rejects.toThrow(/version/);
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
