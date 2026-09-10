import type { BusinessDocument, BusinessIndex, CompanySummary } from './business';

export type FileRecord = { path: string; sha256: string; bytes: number };
export type DiveRecord = { folder: string; label: string; documents: FileRecord[] };
export type Sector = { id: string; name_sv: string; name_en: string; slug: string };
export type Branch = { id: string; sector_id: string; name_sv: string; name_en: string; study_path: string; study_status: 'scaffold' | 'graduated'; corpus_status: 'none' | 'graduated'; overview: FileRecord; deep_dives: DiveRecord[]; shared_study_ids: string[] };
export type Correction = { company_id: string; expected_sector_id: string; expected_branch_id: string; branch_id: string; reason: string; source: string; reviewed_at: string };
export type Classification = { company_id: string; source_sector_id: string; source_branch_id: string; sector_id: string; branch_id: string; status: 'source' | 'corrected' | 'aligned' | 'needs_review'; correction: Correction | null };
export type SharedStudy = { id: string; path: string; branch_ids: string[]; overview: FileRecord; deep_dives: DiveRecord[] };
export type Taxonomy = {
  version: number; as_of: string; exported_at: string; business_sha256: string | null; classification_as_of: string | null;
  sectors: Record<string, Sector>; branches: Record<string, Branch>; classifications: Record<string, Classification>;
  shared_studies: Record<string, SharedStudy>; unmapped_deep_dives: DiveRecord[]; sources: FileRecord[]; corrections_sha256: string; notes: string[];
};
export const fold = (text: string) => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
export const companyBranch = (company: CompanySummary, taxonomy: Taxonomy | null) => taxonomy?.classifications[company.id]?.branch_id ?? company.branch_id;
export const branchCompanies = (index: BusinessIndex | null, taxonomy: Taxonomy | null, branch: string) => Object.values(index?.companies ?? {}).filter(c => companyBranch(c, taxonomy) === branch);
export function branchDives(taxonomy: Taxonomy, branch: Branch) {
  return [...new Map([...branch.deep_dives, ...branch.shared_study_ids.flatMap(id => taxonomy.shared_studies[id].deep_dives)].map(d => [d.folder, d])).values()];
}
export function directoryTotals(taxonomy: Taxonomy) {
  const dives = [...Object.values(taxonomy.branches).flatMap(b => b.deep_dives), ...Object.values(taxonomy.shared_studies).flatMap(s => s.deep_dives), ...taxonomy.unmapped_deep_dives];
  return { sectors: Object.keys(taxonomy.sectors).length, branches: Object.keys(taxonomy.branches).length, studies: Object.values(taxonomy.branches).filter(b => b.study_status === 'graduated').length, dives: new Set(dives.map(d => d.folder)).size };
}
export function findBranches(taxonomy: Taxonomy, query: string, sector: string, coverage: string, index: BusinessIndex | null) {
  const term = fold(query);
  return Object.values(taxonomy.branches).filter(b => (sector === 'all' || b.sector_id === sector)
    && fold(`${b.id} ${b.name_sv} ${b.name_en} ${taxonomy.sectors[b.sector_id].name_sv} ${taxonomy.sectors[b.sector_id].name_en}`).includes(term)
    && (coverage === 'all' || coverage === 'profiles' && branchCompanies(index, taxonomy, b.id).length > 0 || coverage === 'studies' && b.study_status === 'graduated' || coverage === 'dives' && branchDives(taxonomy, b).length > 0))
    .sort((a, b) => a.name_en.localeCompare(b.name_en));
}

export function validateTaxonomy(raw: any, business: BusinessDocument | null, businessHash: string | null): asserts raw is Taxonomy {
  const check = (ok: unknown, message = 'The taxonomy directory is invalid.') => { if (!ok) throw new Error(message); };
  const obj = (x: any) => !!x && typeof x === 'object' && !Array.isArray(x);
  const text = (x: any) => typeof x === 'string' && x.length > 0 && new TextEncoder().encode(x).length <= 20000;
  const id = (x: any) => typeof x === 'string' && /^[1-9][0-9]{0,9}$/.test(x);
  const hash = (x: any) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
  const path = (x: any) => text(x) && !/[\\:]/.test(x) && x.split('/').every((p: string) => p && p !== '.' && p !== '..');
  const date = (x: any) => typeof x === 'string' && /^20\d{2}-\d{2}-\d{2}$/.test(x) && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x;
  const list = (x: any, max = 1000): any[] => { check(Array.isArray(x) && x.length <= max); return x; };
  const objects = (x: any, max: number): [string, any][] => { check(obj(x) && Object.keys(x).length <= max); return Object.entries(x); };
  const file = (x: any) => check(obj(x) && path(x.path) && hash(x.sha256) && Number.isSafeInteger(x.bytes) && x.bytes >= 0, 'Invalid taxonomy source file.');
  const dives = (rows: any) => { const seen = new Set(); for (const d of list(rows, 10000)) { check(obj(d) && path(d.folder) && text(d.label) && !seen.has(d.folder)); seen.add(d.folder); const docs = list(d.documents); check(docs.length > 0); const names = new Set(); for (const f of docs) { file(f); check(f.path.startsWith(`${d.folder}/`) && !names.has(f.path)); names.add(f.path); } } };
  check(raw?.version === 1 && date(raw.as_of), 'Unsupported taxonomy version or inventory date.');
  check(typeof raw.exported_at === 'string' && /^20\d{2}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|\+00:00)$/.test(raw.exported_at) && date(raw.exported_at.slice(0, 10)) && Number.isFinite(Date.parse(raw.exported_at)));
  const sectors = objects(raw.sectors, 100); check(sectors.length > 0);
  for (const [key, s] of sectors) check(obj(s) && id(key) && s.id === key && text(s.name_sv) && text(s.name_en) && typeof s.slug === 'string' && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(s.slug));
  const branches = objects(raw.branches, 1000); check(branches.length > 0);
  for (const [key, b] of branches) { check(obj(b) && id(key) && b.id === key && id(b.sector_id) && raw.sectors[b.sector_id] && text(b.name_sv) && text(b.name_en) && path(b.study_path) && ['scaffold', 'graduated'].includes(b.study_status) && ['none', 'graduated'].includes(b.corpus_status)); file(b.overview); check(b.overview.path === `${b.study_path}/README.md`); dives(b.deep_dives); check(list(b.shared_study_ids).every(text)); }
  const groups = objects(raw.shared_studies, 1000);
  for (const [key, g] of groups) { check(obj(g) && /^[a-z][a-z0-9_]{0,80}$/.test(key) && g.id === key && path(g.path)); file(g.overview); check(g.overview.path.startsWith(`studies/sectors/${g.path}/`)); const members = list(g.branch_ids); check(members.length >= 2 && new Set(members).size === members.length && members.every((bid: string) => id(bid) && raw.branches[bid]?.shared_study_ids.includes(key))); dives(g.deep_dives); }
  for (const [key, b] of branches) check(new Set(b.shared_study_ids).size === b.shared_study_ids.length && b.shared_study_ids.every((group: string) => raw.shared_studies[group]?.branch_ids.includes(key)), 'Invalid shared taxonomy study membership.');
  dives(raw.unmapped_deep_dives); for (const f of list(raw.sources)) file(f); check(list(raw.notes).every(text) && hash(raw.corrections_sha256));
  check(raw.business_sha256 === businessHash && raw.classification_as_of === (business?.as_of ?? null), 'Taxonomy does not match the selected business document.');
  const assignments = objects(raw.classifications, 300);
  check(assignments.length === Object.keys(business?.companies ?? {}).length, 'Taxonomy company coverage does not match the business document.');
  for (const [key, c] of assignments) {
    const original = business?.companies[key];
    check(obj(c) && id(key) && c.company_id === key && original && c.source_sector_id === original.sector_id && c.source_branch_id === original.branch_id && raw.branches[c.source_branch_id]?.sector_id === c.source_sector_id, 'Taxonomy original classification does not match the business document.');
    let status = 'source', target = c.source_branch_id;
    if (c.correction !== null) {
      const r = c.correction;
      check(obj(r) && r.company_id === key && id(r.expected_branch_id) && id(r.expected_sector_id) && raw.branches[r.expected_branch_id]?.sector_id === r.expected_sector_id && id(r.branch_id) && raw.branches[r.branch_id] && text(r.reason) && r.reason.trim() && text(r.source) && r.source.trim() && date(r.reviewed_at) && r.reviewed_at <= raw.as_of, 'Invalid reviewed taxonomy correction.');
      if (c.source_branch_id === r.branch_id) status = 'aligned';
      else if (c.source_branch_id === r.expected_branch_id && c.source_sector_id === r.expected_sector_id) { status = 'corrected'; target = r.branch_id; }
      else status = 'needs_review';
    }
    check(c.status === status && c.branch_id === target && c.sector_id === raw.branches[target]?.sector_id, 'Taxonomy effective classification disagrees with its correction.');
  }
}
