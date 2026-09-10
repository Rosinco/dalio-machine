import type { AtlasIndex, Category, Country } from './types';
import { finite } from './model';

export function explainScore(index: AtlasIndex, country: Country, category: Category) {
  const indicators = index.indicators.filter(i => i.scored && i.category === category);
  const available = indicators.filter(i => finite(country.indicators[i.name]?.pct)).length;
  const minimum = Math.ceil(indicators.length / 2);
  const enough = available > 0 && available >= minimum;
  const rows = indicators.map(meta => {
    const cell = country.indicators[meta.name];
    const weight = enough && finite(cell?.pct) ? 1 / available : null;
    return { meta, cell, weight, contribution: weight !== null ? cell.pct! * weight : null };
  });
  const mean = enough ? rows.reduce((total,row) => total + (row.contribution ?? 0),0) : null;
  const reported = country.categories[category]?.score;
  const consistent = mean === null ? !finite(reported) : finite(reported) && Math.abs(mean - reported) < 0.00001;
  return { rows, available, minimum, mean, reported, consistent };
}
export function comparableScores(current: AtlasIndex, previous: AtlasIndex | null, category: Category): boolean {
  if (!previous) return false;
  const catalogue = (x: AtlasIndex) => x.indicators.filter(i=>i.scored && i.category === category).map(i=>[i.name,i.unit,i.higher_is_better]).sort((a,b)=>String(a[0]).localeCompare(String(b[0])));
  return JSON.stringify([...current.ranking_population].sort()) === JSON.stringify([...previous.ranking_population].sort()) && JSON.stringify(catalogue(current)) === JSON.stringify(catalogue(previous));
}
