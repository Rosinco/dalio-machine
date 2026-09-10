import type { FinancialIndex, SourcedReport } from './financialData';

export type MarketCoverage = { count: number; local: number; sek: number; flagged: number };
type FileSource = { path: string; bytes: number; sha256: string };
export type MarketSource = { id: string; as_of: string; instruments: FileSource; prices: FileSource; fx_pairs: Record<string, string> };
export type MarketMetadata = { version: number; method: string; base_pack: string; entry_days: number; fx_days: number; sources: MarketSource[]; basis_code: FileSource; summary: MarketCoverage & { listings: number; with_local: number; with_sek: number } };
export type MarketReport = { year: number; source_id: string; currency: string | null; shares: number | null; price: number | null; price_date: string | null; local: number | null; sek: number | null; fx_rate: number | null; fx_date: string | null; fx_method: 'identity' | 'direct' | 'usd_cross' | null; fx_instruments: string[]; flags: string[] };
export const marketFlags: Record<string, string> = {
  missing_publication: 'Report publication date unavailable', missing_shares: 'Positive reported share count unavailable',
  missing_price: 'No valid close within 30 days after publication', missing_currency: 'Listing currency unavailable',
  short_period: 'Financial period outside 330–400 days', share_basis: 'Share-count basis requires review',
  scale_suspect: 'Valuation scale requires review', receipt_basis: 'Preference, receipt or unit basis requires review',
  missing_fx: 'No observed FX rate within 7 days before the price',
};
export const marketDefinition = 'Derived listing valuation: reported shares (millions) × the first saved close on or within 30 days after report publication. The fiscal-year label and valuation date differ. Quality flags withhold uncertain values. SEK conversion uses observed FX at or up to 7 days before the price. Each listing is separate; this is not a sum across share classes or receipts.';
const quality = ['short_period', 'share_basis', 'scale_suspect', 'receipt_basis'];
const check = (v: unknown, message = 'Invalid market-cap history.') => { if (!v) throw new Error(message); };
const object = (x: any) => x !== null && typeof x === 'object' && !Array.isArray(x);
const hash = (x: any) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const id = (x: any) => typeof x === 'string' && /^[1-9][0-9]{0,9}$/.test(x);
const ccy = (x: any) => typeof x === 'string' && /^[A-Z]{3}$/.test(x);
const positive = (x: any) => typeof x === 'number' && Number.isFinite(x) && x > 0;
const integer = (x: any) => Number.isSafeInteger(x) && x >= 0 && x <= 10000000;
const day = (x: any): x is string => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) && x >= '1800-01-01' && x <= '2300-12-31' && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x;
const days = (a: string, b: string) => (Date.parse(a) - Date.parse(b)) / 86400000;
const near = (a: any, b: number | null) => b === null ? a === null : positive(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(b));
function file(v: any) { check(object(v) && hash(v.sha256) && Number.isSafeInteger(v.bytes) && v.bytes > 0 && v.bytes <= 10_000_000_000 && typeof v.path === 'string' && v.path.length <= 2000 && !/[\\:]/.test(v.path) && v.path.split('/').every((s: string) => s && s !== '.' && s !== '..')); }
export function marketCoverage(rows: MarketReport[]): MarketCoverage { return { count: rows.length, local: rows.filter(r => r.local !== null).length, sek: rows.filter(r => r.sek !== null).length, flagged: rows.filter(r => r.flags.some(f => quality.includes(f))).length }; }
export function marketReason(row: MarketReport | null | undefined, sek = true) { return row ? row.flags.filter(f => sek || f !== 'missing_fx').map(f => marketFlags[f]).join('; ') : 'Market history is not included in this saved pack'; }

export function validateMarketIndex(index: FinancialIndex) {
  const m: any = index.market;
  if (index.version === 1) { check(m === undefined && Object.values(index.companies).every(c => c.market === undefined), 'Market data requires financial pack version 2.'); return; }
  check(object(m) && m.version === 1 && m.method === 'reported-shares-publication-close-v1' && hash(m.base_pack) && m.entry_days === 30 && m.fx_days === 7, 'Unsupported market-cap method.');
  check(Array.isArray(m.sources) && m.sources.length > 0 && m.sources.length <= 1000);
  const sources = new Set();
  for (const s of m.sources) {
    check(object(s) && typeof s.id === 'string' && !sources.has(s.id) && index.sources.some(r => r.id === s.id && r.frequency === 'annual' && r.as_of === s.as_of) && day(s.as_of));
    sources.add(s.id); file(s.instruments); file(s.prices);
    check(object(s.fx_pairs) && Object.keys(s.fx_pairs).length <= 256 && Object.entries(s.fx_pairs).every(([k, v]) => id(k) && typeof v === 'string' && /^[A-Z]{3}\/[A-Z]{3}$/.test(v)));
  }
  check(index.sources.filter(s => s.frequency === 'annual').every(s => sources.has(s.id)));
  file(m.basis_code);
  const totals = { count: 0, local: 0, sek: 0, flagged: 0, listings: Object.keys(index.companies).length, with_local: 0, with_sek: 0 };
  for (const c of Object.values(index.companies)) {
    const v: any = c.market;
    check(object(v) && ['count', 'local', 'sek', 'flagged'].every(k => integer(v[k])) && v.count === c.annual.count && v.local <= v.count && v.sek <= v.local && v.flagged <= v.count - v.local);
    for (const k of ['count', 'local', 'sek', 'flagged'] as const) totals[k] += v[k];
    totals.with_local += Number(v.local > 0); totals.with_sek += Number(v.sek > 0);
  }
  check(object(m.summary) && Object.entries(totals).every(([k, n]) => m.summary[k] === n), 'Market-cap coverage does not reconcile.');
}

export function decodeMarketRows(raw: any, index: FinancialIndex, id: string, annual: SourcedReport[]): MarketReport[] {
  if (index.version === 1) { check(raw === undefined || raw === null, 'Market history requires a version 2 pack.'); return []; }
  check(Array.isArray(raw) && raw.length === annual.length && raw.length <= 2000 && index.market);
  const sources = new Map(index.market!.sources.map(s => [s.id, s]));
  raw.forEach((r: any, i: number) => {
    const a = annual[i], source = sources.get(r?.source_id);
    check(object(r) && r.year === a.year && r.source_id === a.source_id && source, 'Market history does not match the annual report.');
    check(r.currency === null || ccy(r.currency));
    for (const k of ['shares', 'price', 'local', 'sek', 'fx_rate']) check(r[k] === null || positive(r[k]));
    check(r.price === null ? r.price_date === null : r.price < 1e10 && day(r.price_date) && a.report_date !== null && r.price_date >= a.report_date && r.price_date <= source!.as_of && days(r.price_date, a.report_date) <= 30, 'Market price is outside the publication window.');
    check(Array.isArray(r.flags) && r.flags.every((f: any) => typeof f === 'string' && Object.hasOwn(marketFlags, f)) && new Set(r.flags).size === r.flags.length);
    const has = (f: string) => r.flags.includes(f);
    check(has('missing_publication') === (a.report_date === null) && has('missing_shares') === (r.shares === null) && has('missing_price') === (r.price === null) && has('missing_currency') === (r.currency === null));
    check(has('short_period') === (days(a.end, a.start) + 1 < 330 || days(a.end, a.start) + 1 > 400));
    const candidate = r.shares !== null && r.price !== null ? r.shares * r.price : null;
    const suspect = candidate !== null && (!positive(candidate) || positive(a.raw.profit_to_equity_holders) && candidate / a.raw.profit_to_equity_holders! < 1 || positive(a.raw.total_equity) && candidate / a.raw.total_equity! < .05);
    check(has('scale_suspect') === suspect, 'Market valuation scale flags disagree with report amounts.');
    const local = r.flags.some((f: string) => f !== 'missing_fx') ? null : candidate;
    check(near(r.local, local), 'Market value does not equal shares × price.');
    check(Array.isArray(r.fx_instruments) && r.fx_instruments.every((id: any) => typeof id === 'string') && new Set(r.fx_instruments).size === r.fx_instruments.length);
    if (r.fx_rate === null) check(r.fx_date === null && r.fx_method === null && r.fx_instruments.length === 0 && has('missing_fx'));
    else {
      check(!has('missing_fx') && r.currency !== null && day(r.fx_date) && r.price_date !== null && r.fx_date <= r.price_date && days(r.price_date, r.fx_date) <= 7, 'FX date is unavailable, stale or later than the price.');
      const pairs = r.fx_instruments.map((k: string) => source!.fx_pairs[k]);
      if (r.fx_method === 'identity') check(r.currency === 'SEK' && r.fx_rate === 1 && r.fx_date === r.price_date && pairs.length === 0);
      else if (r.fx_method === 'direct') check(pairs.length === 1 && pairs[0] === `${r.currency}/SEK`);
      else check(r.fx_method === 'usd_cross' && pairs.length === 2 && pairs[0] === 'USD/SEK' && pairs[1] === `USD/${r.currency}`);
    }
    check(near(r.sek, r.local !== null && r.fx_rate !== null ? r.local * r.fx_rate : null), 'SEK market value does not match the dated conversion.');
  });
  const c = marketCoverage(raw), expected = index.companies[id].market!;
  check(Object.entries(c).every(([k, v]) => expected[k as keyof MarketCoverage] === v), 'Market rows disagree with coverage.');
  return raw;
}
