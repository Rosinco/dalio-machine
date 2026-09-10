import { branchMetrics, needsCurrency, type BranchSettings } from './branchComparison';

export type SavedComparison = { id: string; title: string; notes: string; created: string; release: string; financial: string; taxonomy: string; branch: string; settings: BranchSettings };
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem'>;
const key = 'macro-atlas-branch-comparisons-v1';
const check = (ok: unknown) => { if (!ok) throw new Error('Saved comparison is invalid or uses an unsupported format. Existing saved work has been retained.'); };
const hash = (x: unknown) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const text = (x: unknown, max: number) => typeof x === 'string' && x.length <= max;
function validate(v: any): asserts v is SavedComparison {
  check(v && typeof v === 'object' && !Array.isArray(v));
  check(typeof v.id === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(v.id) && text(v.title, 120) && v.title.trim() && text(v.notes, 10000));
  check(typeof v.created === 'string' && /^\d{4}-\d{2}-\d{2}T.+Z$/.test(v.created) && Number.isFinite(Date.parse(v.created)));
  check(hash(v.release) && hash(v.financial) && hash(v.taxonomy) && typeof v.branch === 'string' && /^(?:[1-9][0-9]{0,9}|unassigned)$/.test(v.branch));
  const s = v.settings;
  check(s && typeof s === 'object' && Object.hasOwn(branchMetrics, s.metric) && ['equal', 'revenues', 'total_assets', 'market_cap'].includes(s.size));
  check(typeof s.currency === 'string' && /^(?:all|[A-Z]{3})$/.test(s.currency) && (!needsCurrency(s) || s.currency !== 'all'));
  check(typeof s.country === 'string' && /^(?:all|[A-Z]{2})$/.test(s.country) && ['all', 'latest', 'older'].includes(s.presence));
  check(Number.isInteger(s.month) && s.month >= 0 && s.month <= 12);
  check(Number.isInteger(s.from) && Number.isInteger(s.to) && s.from >= 2000 && s.to <= 2300 && s.to >= s.from && s.to - s.from <= 60 && Number.isInteger(s.year) && s.year >= s.from && s.year <= s.to);
  check(Array.isArray(s.selected) && s.selected.length <= 8 && new Set(s.selected).size === s.selected.length && s.selected.every((id: any) => typeof id === 'string' && /^[1-9][0-9]{0,9}$/.test(id)));
  check(s.focus === null || s.selected.includes(s.focus));
}
export function loadComparisons(storage: Storage): SavedComparison[] {
  const raw = storage.getItem(key); if (raw === null) return [];
  check(raw.length <= 2_000_000);
  const data = JSON.parse(raw);
  check(data?.version === 1 && Array.isArray(data.items) && data.items.length <= 100);
  data.items.forEach(validate);
  check(new Set(data.items.map((v: SavedComparison) => v.id)).size === data.items.length);
  return data.items;
}
export function saveComparison(storage: Storage, view: SavedComparison) {
  validate(view);
  const existing = loadComparisons(storage);
  check(existing.length < 100 && !existing.some(v => v.id === view.id));
  const raw = JSON.stringify({ version: 1, items: [view, ...existing] }); check(raw.length <= 2_000_000);
  storage.setItem(key, raw);
}
export function removeComparison(storage: Storage, id: string) {
  const items = loadComparisons(storage).filter(v => v.id !== id);
  storage.setItem(key, JSON.stringify({ version: 1, items }));
}
export function compatibleComparison(view: SavedComparison, release: string, financial: string, taxonomy: string, branchIds: string[]) {
  const ids = new Set(branchIds);
  return view.release === release && view.financial === financial && view.taxonomy === taxonomy && view.settings.selected.every(id => ids.has(id));
}
