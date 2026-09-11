import { expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { decodeCountryEvidence, validateEvidenceIndex } from '../src/countryEvidence';

it.skipIf(!existsSync(resolve('public/data/country-evidence/index.json')))('verifies every bundled country hash, source identity and history, rejecting altered bytes', async () => {
    const root = resolve('public/data/country-evidence');
    const index = await validateEvidenceIndex(JSON.parse(readFileSync(resolve(root, 'index.json'), 'utf8')));
    expect(Object.keys(index.countries)).toHaveLength(19);
    for (const code of Object.keys(index.countries)) {
      const text = readFileSync(resolve(root, index.countries[code].file), 'utf8');
      const country = await decodeCountryEvidence(text, index, code);
      expect(country.assessment.histories).toHaveLength(8);
      expect(country.monitoring?.profile.signals.length ?? 0).toBe(index.countries[code].signals);
      await expect(decodeCountryEvidence(text + ' ', index, code)).rejects.toThrow(/checksum/);
    }
    const changed = structuredClone(index); changed.countries.SE.name = 'Altered';
    await expect(validateEvidenceIndex(changed)).rejects.toThrow(/checksum/);
  });
