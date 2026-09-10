import { describe, expect, it } from 'vitest';
import { explainScore, comparableScores } from './scores';
import type { AtlasIndex, Country } from './types';

const index = { ranking_population: ['SE','DE'], indicators: ['a','b','c','d'].map(name => ({ name, category: 'promises', scored: true, higher_is_better: false, unit: '%' })) } as AtlasIndex;
const country = (scores: (number | null)[], reported: number | null) => ({ categories: { promises: { score: reported, n_available: scores.filter(x => x !== null).length, n_total: 4 } }, indicators: Object.fromEntries(scores.map((pct,i) => [['a','b','c','d'][i], { pct, value: pct === null ? null : 10 }])) }) as Country;
describe('score explanations', () => {
  it('explains equal weighting of available percentiles without treating missing as zero', () => {
    const explanation = explainScore(index, country([90,10,null,null],50), 'promises');
    expect(explanation.mean).toBe(50); expect(explanation.consistent).toBe(true);
    expect(explanation.rows.map(r => r.weight)).toEqual([.5,.5,null,null]);
    expect(explanation.rows.map(r => r.contribution)).toEqual([45,5,null,null]);
  });
  it('does not manufacture a score below the half-coverage threshold', () => {
    const explanation = explainScore(index, country([90,null,null,null],null), 'promises');
    expect(explanation.mean).toBeNull(); expect(explanation.minimum).toBe(2);
    expect(explanation.rows[0].contribution).toBeNull();
  });
  it('flags a saved score that does not reconcile to its component percentiles', () => {
    expect(explainScore(index,country([90,10,null,null],80),'promises').consistent).toBe(false);
  });
  it('compares only matching populations, units and indicator directions', () => {
    expect(comparableScores(index,{...index,ranking_population:['DE','SE']},'promises')).toBe(true);
    expect(comparableScores(index,{...index,ranking_population:['SE','US']},'promises')).toBe(false);
    expect(comparableScores(index,{...index,indicators:index.indicators.map(i=>({...i,higher_is_better:true}))},'promises')).toBe(false);
  });
});
