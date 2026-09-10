import type { BusinessDocument, BusinessIndex, CompanySummary } from './business';
import type { Branch, Classification, FileRecord, Taxonomy } from './taxonomy';

export type Listing = { id: string; name: string | null; ticker: string | null; isin: string | null; country_id: string; listing_country: string | null; sector_id: string | null; branch_id: string | null; instrument_type: number; source_as_of: string; listing_date: string | null; stock_currency: string | null; report_currency: string | null };
export type ListingCatalogue = { version: number; as_of: string; exported_at: string; included_types: number[];
  countries: Record<string, { id: string; name: string; name_en: string; iso2: string | null }>;
  snapshots: { as_of: string; instruments: FileRecord; countries: FileRecord; instrument_count: number; company_count: number; excluded_count: number }[];
  listings: Record<string, Listing>;
};
export type CompanyEntry = Listing & { display_name: string; profile: CompanySummary | null; search: string };
export type Presence = 'all' | 'latest' | 'older';
export type BranchCounts = Record<string, { listings: number; profiles: number }>;
export const normalizeSearch = (s: string) => s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
export function listingCountries(index: BusinessIndex | null, taxonomy: Taxonomy | null) {
  return { ...index?.countries, ...Object.fromEntries(Object.values(taxonomy?.catalogue?.countries ?? {}).filter(c => c.iso2).map(c => [c.iso2!, c.name_en])) };
}
export function companyEntry(id: string, index: BusinessIndex | null, taxonomy: Taxonomy | null): CompanyEntry | undefined {
  const profile = index?.companies[id] ?? null, listing = taxonomy?.catalogue?.listings[id];
  if (!listing && !profile) return undefined;
  const raw: Listing = listing ?? { ...profile!, country_id: '', instrument_type: 0, source_as_of: index!.as_of, listing_date: null };
  const display_name = profile?.name ?? raw.name ?? `Listing ${id}`;
  return { ...raw, display_name, profile, search: normalizeSearch(`${display_name} ${raw.name ?? ''} ${raw.ticker ?? ''} ${raw.isin ?? ''} ${raw.id}`) };
}
export function companyEntries(index: BusinessIndex | null, taxonomy: Taxonomy | null): CompanyEntry[] {
  return Object.keys(taxonomy?.catalogue?.listings ?? index?.companies ?? {}).map(id => companyEntry(id, index, taxonomy)!)
    .sort((a, b) => Number(!!b.profile) - Number(!!a.profile) || a.display_name.localeCompare(b.display_name) || Number(a.id) - Number(b.id));
}
export function presenceMatches(entry: CompanyEntry, presence: Presence, latest: string) { return presence === 'all' || (entry.source_as_of === latest) === (presence === 'latest'); }
export function countBranches(entries: CompanyEntry[], taxonomy: Taxonomy | null): BranchCounts {
  const counts: BranchCounts = {};
  for (const e of entries) { const branch = taxonomy?.classifications[e.id] ? taxonomy.classifications[e.id].branch_id : e.branch_id; if (!branch) continue; const n = counts[branch] ??= { listings: 0, profiles: 0 }; n.listings++; if (e.profile) n.profiles++; }
  return counts;
}

export function expectedClassification(sector: string | null, branch: string | null, correction: Classification['correction'], branches: Record<string, Branch>) {
  let target = branch && branches[branch] ? branch : null;
  let status: Classification['status'] = target === null ? 'unclassified' : branches[target].sector_id === sector ? 'source' : 'sector_mismatch';
  if (correction) {
    if (branch === correction.branch_id && sector === branches[branch!]?.sector_id) status = 'aligned';
    else if (sector === correction.expected_sector_id && branch === correction.expected_branch_id) { status = 'corrected'; target = correction.branch_id; }
    else status = 'needs_review';
  }
  return { status, branch_id: target, sector_id: target ? branches[target].sector_id : null };
}

export function validateCatalogue(raw: any, business: BusinessDocument | null, inventoryDate: string): asserts raw is ListingCatalogue {
  const check = (ok: unknown, message = 'Invalid company listing catalogue.') => { if (!ok) throw new Error(message); };
  const object = (x: any) => !!x && typeof x === 'object' && !Array.isArray(x);
  const text = (x: any) => typeof x === 'string' && new TextEncoder().encode(x).length <= 2000;
  const id = (x: any) => typeof x === 'string' && /^[1-9][0-9]{0,9}$/.test(x);
  const date = (x: any) => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) && +x.slice(0, 4) >= 1800 && +x.slice(0, 4) <= 2300 && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x;
  const file = (x: any) => check(object(x) && text(x.path) && x.path && !/[\\:]/.test(x.path) && x.path.split('/').every((p: string) => p && p !== '.' && p !== '..') && /^[a-f0-9]{64}$/.test(x.sha256) && Number.isSafeInteger(x.bytes) && x.bytes >= 0);
  check(raw?.version === 1 && date(raw.as_of) && raw.as_of <= inventoryDate && typeof raw.exported_at === 'string' && /^20\d{2}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|\+00:00)$/.test(raw.exported_at) && date(raw.exported_at.slice(0, 10)));
  check(Array.isArray(raw.included_types) && raw.included_types.every(Number.isInteger) && [...raw.included_types].sort((a: number, b: number) => a - b).join(',') === '0,1,3,8,9,10');
  check(object(raw.countries) && Object.keys(raw.countries).length <= 300);
  const codes = new Set();
  for (const [key, c] of Object.entries(raw.countries) as [string, any][]) { check(id(key) && object(c) && c.id === key && text(c.name) && c.name && text(c.name_en) && c.name_en && (c.iso2 === null || /^[A-Z]{2}$/.test(c.iso2) && !codes.has(c.iso2))); if (c.iso2) codes.add(c.iso2); }
  check(Array.isArray(raw.snapshots) && raw.snapshots.length > 0 && raw.snapshots.length <= 1000);
  const snapshots = new Map<string, any>(); let previous = '';
  for (const s of raw.snapshots) { check(object(s) && date(s.as_of) && s.as_of > previous && s.as_of <= raw.as_of); previous = s.as_of; for (const k of ['instrument_count', 'company_count', 'excluded_count']) check(Number.isSafeInteger(s[k]) && s[k] >= 0); check(s.company_count + s.excluded_count === s.instrument_count); file(s.instruments); file(s.countries); snapshots.set(s.as_of, s); }
  check(previous === raw.as_of && object(raw.listings) && Object.keys(raw.listings).length <= 100000);
  const counts = new Map<string, number>();
  for (const [key, row] of Object.entries(raw.listings) as [string, any][]) {
    check(id(key) && object(row) && row.id === key && id(row.country_id) && (row.sector_id === null || id(row.sector_id)) && (row.branch_id === null || id(row.branch_id)) && [0, 1, 3, 8, 9, 10].includes(row.instrument_type) && snapshots.has(row.source_as_of));
    for (const k of ['name', 'ticker', 'isin', 'stock_currency', 'report_currency']) check(row[k] === null || text(row[k]));
    check(row.listing_country === (raw.countries[row.country_id]?.iso2 ?? null) && (row.listing_date === null || date(row.listing_date)));
    counts.set(row.source_as_of, (counts.get(row.source_as_of) ?? 0) + 1);
  }
  check([...snapshots].every(([date, s]) => (counts.get(date) ?? 0) <= s.company_count) && (counts.get(raw.as_of) ?? 0) === snapshots.get(raw.as_of).company_count, 'Company catalogue counts do not reconcile with its source snapshots.');
  for (const c of Object.values(business?.companies ?? {})) check(raw.listings[c.id], 'A financial profile is missing from the company catalogue.');
}
