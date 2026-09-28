import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, relative, isAbsolute } from 'node:path';
import { build } from 'esbuild';

// Consume the fresh output from build-normal-year-ranking.py; never write fixtures.
const desktop = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
assert.equal(process.argv.length, 3, 'Provide the generator output directory.');
const output = resolve(process.argv[2]);
const outputRelative = relative(resolve(desktop, 'test-results'), output);
assert.ok(outputRelative && outputRelative !== '..' && !outputRelative.startsWith('../') && !outputRelative.startsWith('..\\') && !isAbsolute(outputRelative), 'Output must be under desktop/test-results.');
const expected = JSON.parse(readFileSync(resolve(output, 'expected-ranking.json')));
const inputs = JSON.parse(readFileSync(resolve(output, 'ranking-inputs.json')));
const modulePath = resolve(output, 'ranking-module.mjs');
await build({ entryPoints: [resolve(desktop, 'src/normalQualityValueRanking.ts')], outfile: modulePath, bundle: true, platform: 'node', format: 'esm', logLevel: 'silent' });
const { rankNormalQualityValue } = await import(pathToFileURL(modulePath));
const result = rankNormalQualityValue(inputs);
assert.equal(result.cohortSize, expected.cohortSize);
let checkedValues = 0;
for (const input of inputs) {
  const actual = result.byId.get(input.id), independent = expected.byId[input.id];
  assert.ok(actual);
  for (const key of ['score', 'quality', 'discountRank']) {
    if (independent[key] === null) assert.equal(actual[key], null);
    else assert.ok(Math.abs(actual[key] - independent[key]) <= 1e-12, `${input.id} ${key}: ${actual[key]} vs ${independent[key]}`);
    checkedValues++;
  }
  if (independent.components === null) assert.equal(actual.components, null);
  else for (const key of ['roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda']) {
    assert.ok(Math.abs(actual.components[key] - independent.components[key]) <= 1e-12, `${input.id} ${key}`);
    checkedValues++;
  }
}
assert.deepEqual([...rankNormalQualityValue([...inputs].reverse()).byId].sort(), [...result.byId].sort(), 'Input ordering changed ranks');
const source = readFileSync(resolve(desktop, 'src/normalQualityValueRanking.ts'));
const receipt = { status: 'passed', listings: inputs.length, cohortSize: result.cohortSize, checkedValues, inputOrderInvariant: true,
  independentExpectedSha256: createHash('sha256').update(readFileSync(resolve(output, 'expected-ranking.json'))).digest('hex'),
  moduleSourceSha256: createHash('sha256').update(source).digest('hex') };
writeFileSync(resolve(output, 'ranking-module-verification.json'), JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify(receipt));
