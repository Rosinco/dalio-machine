import type { Country, Point, Trade } from './types';

export const categories = {
  real_stuff: { label: 'Resources & people', short: 'Resources', description: 'Energy dependence and demographics' },
  production: { label: 'Production & growth', short: 'Production', description: 'Output, innovation and productive capacity' },
  exchange: { label: 'Trade & external balance', short: 'Trade', description: 'Trade reach, external balances and reserves' },
  promises: { label: 'Debt & fiscal position', short: 'Debt', description: 'Debt, budgets and debt service' },
  enforcer: { label: 'Institutions & stability', short: 'Institutions', description: 'Institutions, stability and state capacity' },
};
export const palette = ['#d5b483', '#b5c5ad', '#78aa99', '#438576', '#205d54'];
export const missingColor = '#e2e5de';
export const finite = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
export function quintile(score: number | null | undefined) {
  return finite(score) ? Math.max(0, Math.min(4, Math.floor(score / 20))) : null;
}
export function format(value: number | null | undefined, digits = 1): string {
  return finite(value) ? new Intl.NumberFormat('en-GB', { maximumFractionDigits: digits }).format(value) : 'Not available';
}
export function atYear(history: Point[] = [], year: number): Point | undefined {
  // History is a calendar-year view of the saved vintage. Never carry forward or insert forecasts.
  return history.find(p => p.year === year && !p.is_forecast && finite(p.value));
}
export function historyLines(points: Point[], startYear: number) {
  const selected = points.filter(p => p.year >= startYear).sort((a, b) => a.year - b.year);
  const historical = selected.filter(p => !p.is_forecast);
  const forecast = selected.filter(p => p.is_forecast);
  const fill = (items: Point[]): [number, number | null][] => {
    if (!items.length) return [];
    const byYear = new Map(items.map(p => [p.year, p.value]));
    return Array.from({ length: items.at(-1)!.year - items[0].year + 1 }, (_, i) => {
      const year = items[0].year + i; const value = byYear.get(year);
      return [year, finite(value) ? value : null];
    });
  };
  const last = historical.at(-1);
  const join = last && forecast.length && forecast[0].year === last.year + 1 ? [last] : [];
  return { historical: fill(historical), forecast: fill([...join, ...forecast]) };
}
export function latestTrade(rows: Trade[], reporter: string): Trade[] {
  const selected = rows.filter(r => r.iso2 === reporter && r.partner !== 'EU');
  const year = Math.max(...selected.map(r => r.year));
  return selected.filter(r => r.year === year && finite(r.x_share) && r.x_share >= 0)
    .sort((a, b) => (b.x_share ?? 0) - (a.x_share ?? 0));
}
export function tradeSlices(rows: Trade[], names: Record<string, Country>) {
  const top = rows.slice(0, 5).map(r => ({ name: names[r.partner]?.name ?? r.partner, value: r.x_share!, code: r.partner }));
  const total = top.reduce((s, r) => s + r.value, 0);
  // Percent shares can only form a whole when they fit the published denominator.
  if (total > 100.05) return [];
  return [...top, ...(total < 99.95 ? [{ name: 'Other destinations', value: 100 - total, code: '' }] : [])];
}
export function sourceUrl(source: string, indicator?: string): string | null {
  const wb: Record<string, string> = {
    energy_net_imports_pct: 'EG.IMP.CONS.ZS', old_age_dependency: 'SP.POP.DPND.OL',
    gdp_pc_ppp: 'NY.GDP.PCAP.PP.KD', rd_pct_gdp: 'GB.XPD.RSDV.GD.ZS',
    reserves_months_imports: 'FI.RES.TOTL.MO', rule_of_law: 'RL.EST', political_stability: 'PV.EST',
  };
  if (source === 'WORLD_BANK' && indicator && wb[indicator]) return `https://data.worldbank.org/indicator/${wb[indicator]}`;
  return ({ WORLD_BANK: 'https://data.worldbank.org/', WORLD_BANK_WGI: 'https://www.worldbank.org/en/publication/worldwide-governance-indicators', IMF_WEO: 'https://www.imf.org/en/Publications/WEO/weo-database', IMF_WEO_FCST: 'https://www.imf.org/en/Publications/WEO/weo-database', BIS: 'https://data.bis.org/', BIS_TC: 'https://data.bis.org/', BIS_DSR: 'https://data.bis.org/', OEC_ECI: 'https://oec.world/en/rankings/eci/hs6/hs92', SIPRI: 'https://www.sipri.org/databases/milex' } as Record<string, string>)[source] ?? null;
}
