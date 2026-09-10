import type { AtlasIndex, Country, HistoryPanel, ResearchCatalogue, ResearchDocument, ResearchPackage, ResearchRelease } from './types';
import type { LiquidityReport } from './liquidity';

export const MAX_PACKAGE_BYTES = 32 * 1024 * 1024;
const native = () => '__TAURI_INTERNALS__' in window;
async function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return (await import('@tauri-apps/api/core')).invoke<T>(command, args);
}
async function json<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error('The saved research could not be opened.');
  return response.json();
}
export async function digest(text: string): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(hash)].map(b => b.toString(16).padStart(2, '0')).join('');
}
const requireField = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };
const record = (v: unknown): v is Record<string, any> => !!v && typeof v === 'object' && !Array.isArray(v);
const numberOrNull = (v: unknown) => v === null || typeof v === 'number' && Number.isFinite(v);
const score = (v: unknown) => v === null || typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 100;
const utcTimestamp = (v: unknown) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)$/.test(v) && Number.isFinite(Date.parse(v));
const validDate = (v: unknown) => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v) && Number.isFinite(Date.parse(v)) && new Date(v).toISOString().slice(0, 10) === v && Number(v.slice(0, 4)) >= 1800 && Number(v.slice(0, 4)) <= 2300;
const categoryNames = ['real_stuff', 'production', 'exchange', 'promises', 'enforcer'];

function validateFundamentals(raw: any) {
  requireField(raw?.version === 1, 'This fundamentals format requires a different Atlas version.');
  requireField(validDate(raw.as_of) && utcTimestamp(raw.generated_at), 'The release date is invalid.');
  requireField(Array.isArray(raw.categories) && raw.categories.length === 5 && categoryNames.every(k => raw.categories.includes(k)), 'The five fundamentals categories are required.');
  requireField(record(raw.countries) && raw.countries.SE && Object.keys(raw.countries).length <= 300, 'The package must include a country panel with Sweden.');
  requireField(Array.isArray(raw.indicators) && raw.indicators.length > 0 && raw.indicators.length <= 500, 'The indicator catalogue is invalid.');
  const names = new Set();
  for (const i of raw.indicators) {
    requireField(typeof i.name === 'string' && !names.has(i.name) && categoryNames.includes(i.category), 'Indicator names and categories must be valid.'); names.add(i.name);
    requireField(['label', 'unit', 'description', 'uncertainty', 'cadence'].every(k => typeof i[k] === 'string') && typeof i.scored === 'boolean' && typeof i.higher_is_better === 'boolean' && typeof i.forward === 'boolean' && Array.isArray(i.sources), 'Indicator metadata is incomplete.');
  }
  requireField(Array.isArray(raw.ranking_population) && raw.ranking_population.length && raw.ranking_population.every((k: string) => raw.countries[k]), 'The ranking population is invalid.');
  requireField(new Set(raw.ranking_population).size === raw.ranking_population.length, 'Ranking countries must be unique.');
  for (const [code, c] of Object.entries(raw.countries) as [string, any][]) {
    requireField(/^[A-Z]{2}$/.test(code) && typeof c.name === 'string' && typeof c.iso3 === 'string' && (c.currency == null || typeof c.currency === 'string') && typeof c.on_map === 'boolean' && typeof c.data_quality?.flag === 'string', 'Country metadata is invalid.');
    for (const key of categoryNames) requireField(record(c.categories?.[key]) && score(c.categories[key].score) && Number.isInteger(c.categories[key].n_available) && c.categories[key].n_available >= 0 && Number.isInteger(c.categories[key].n_total) && c.categories[key].n_total <= 500 && c.categories[key].n_available <= c.categories[key].n_total, 'Category scores or coverage are invalid.');
    requireField(record(c.indicators) && record(c.history) && Array.isArray(c.pressures), 'The country profile is incomplete.');
    requireField(c.data_quality.note == null || typeof c.data_quality.note === 'string', 'Data-quality notes are invalid.');
    if (c.cycle != null) requireField(record(c.cycle) && ['long_term_label', 'short_term_label'].every(k => typeof c.cycle[k] === 'string') && ['long_term_confidence', 'short_term_confidence'].every(k => numberOrNull(c.cycle[k])), 'Cycle metadata is invalid.');
    for (const cell of Object.values(c.indicators) as any[]) {
      requireField(record(cell) && numberOrNull(cell.value) && score(cell.pct) && typeof cell.is_forecast === 'boolean', 'An indicator observation is invalid.');
      requireField(['source', 'date', 'trend', 'uncertainty'].every(k => cell[k] == null || typeof cell[k] === 'string'), 'Observation metadata is invalid.');
    }
    for (const p of c.pressures) requireField(record(p) && ['title','constraint','rule_id','uncertainty'].every(k => typeof p[k] === 'string') && Array.isArray(p.forced_options) && p.forced_options.every((v: unknown) => typeof v === 'string') && Array.isArray(p.spillovers) && p.spillovers.every((v: any) => record(v) && ['target','text','channel'].every(k => typeof v[k] === 'string')), 'Pressure metadata is invalid.');
    for (const points of Object.values(c.history) as any[]) {
      requireField(Array.isArray(points) && points.length <= 10000, 'An annual history is invalid.');
      const years = new Set();
      for (const p of points) { requireField(Number.isInteger(p.year) && p.year >= 1800 && p.year <= 2300 && !years.has(p.year) && numberOrNull(p.value) && typeof p.is_forecast === 'boolean', 'A history has duplicate years or invalid values.'); years.add(p.year); }
    }
  }
  requireField(Array.isArray(raw.trade) && raw.trade.length <= 100000 && raw.trade.every((r: any) => record(r) && /^[A-Z]{2}$/.test(r.iso2) && /^[A-Z]{2}$/.test(r.partner) && Number.isInteger(r.year) && ['x_share','m_share','x_usd','m_usd'].every(k => numberOrNull(r[k]))), 'The trade panel is invalid.');
}

function validateLiquidity(raw: any) {
  const strings = (v: any, keys: string[]) => requireField(record(v) && keys.every(k => typeof v[k] === 'string'), 'Research text metadata is invalid.');
  const measures = (v: any, keys: string[]) => requireField(record(v) && keys.every(k => v[k] == null || numberOrNull(v[k])), 'A liquidity measure is invalid.');
  const list = (v: any): any[] => { requireField(Array.isArray(v) && v.length <= 20000, 'A liquidity panel is invalid.'); return v; };
  const optionalText = (v: any, keys: string[]) => requireField(record(v) && keys.every(k => v[k] == null || typeof v[k] === 'string'), 'Liquidity metadata is invalid.');
  const trace = (v: any) => {
    strings(v, ['availability_status']);
    requireField(list(v.input_release_ids).every(Number.isSafeInteger), 'Source references are invalid.');
    if (v.input_statuses != null) requireField(list(v.input_statuses).every(x => typeof x === 'string'), 'Observation statuses are invalid.');
    optionalText(v, ['interpretation_limit']);
  };
  requireField(raw?.version === 1 && raw.methodology_version === 'liquidity-diagnostics-v1', 'This liquidity methodology requires a different Atlas version.');
  requireField(validDate(raw.as_of) && utcTimestamp(raw.as_known_at) && (raw.complete_snapshot_available_at == null || utcTimestamp(raw.complete_snapshot_available_at)), 'Liquidity release metadata is invalid.');
  strings(raw, ['snapshot_sha256', 'methodology_sha256']);
  for (const key of ['broad_money','offshore_credit','central_bank_divergence','input_releases','horizons','interpretation_limits']) list(raw[key]);
  for (const key of ['money_summary','mmf','repo','coverage','formulas','evidence_integrity']) requireField(record(raw[key]), 'A liquidity panel is invalid.');
  for (const row of [...raw.broad_money, ...raw.offshore_credit]) {
    trace(row); strings(row, ['currency','title','unit']); optionalText(row, ['period','movement']);
    measures(row, ['annual_log_growth_pct','acceleration_3m_pp','acceleration_1q_pp','latest_value']);
    const periods = new Set();
    for (const p of row.history == null ? [] : list(row.history)) {
      requireField(record(p) && validDate(p.date) && !periods.has(p.date.slice(0,7)) && numberOrNull(p.annual_log_growth_pct), 'A liquidity history has duplicate periods or invalid values.'); periods.add(p.date.slice(0,7));
    }
  }
  for (const row of raw.broad_money) strings(row, ['country']);
  for (const row of raw.central_bank_divergence) { trace(row); strings(row, ['country','currency']); optionalText(row, ['period']); measures(row, ['money_annual_log_growth_pct','central_bank_assets_annual_log_growth_pct','money_minus_assets_growth_gap_pp']); }
  for (const row of raw.input_releases) { requireField(Number.isSafeInteger(row.release_id), 'Source reference is invalid.'); strings(row, ['source_family','partition_key','content_sha256','available_at']); optionalText(row, ['published_at','source_url']); }
  for (const row of raw.horizons) strings(row, ['horizon','supported_context']);
  requireField(Object.values(raw.formulas).every(v => typeof v === 'string'), 'Calculation rules are invalid.');
  for (const row of [...Object.values(raw.coverage), raw.money_summary] as any[]) requireField(Number.isInteger(row.ready) && Number.isInteger(row.expected) && row.ready >= 0 && row.ready <= row.expected, 'Coverage is invalid.');
  strings(raw.money_summary, ['interpretation_limit']); optionalText(raw.money_summary, ['common_period']); measures(raw.money_summary, ['median_annual_log_growth_pct','positive_growth_breadth','accelerating_breadth']);
  trace(raw.mmf); optionalText(raw.mmf, ['period']); measures(raw.mmf, ['mmf_annual_log_growth_pct','mmf_minus_m2_growth_gap_pp','mmf_assets_to_m2_scale_pct']);
  for (const row of list(raw.mmf.asset_allocation)) { strings(row, ['title']); measures(row, ['share_of_total_pct']); }
  const ratios = raw.mmf.published_repo_counterparty_categories;
  strings(ratios, ['availability_status','interpretation_limit']); optionalText(ratios, ['period']);
  for (const row of list(ratios.ratios)) { strings(row, ['title']); measures(row, ['share_of_repo_pct']); }
  trace(raw.repo); optionalText(raw.repo, ['period_date']); measures(raw.repo, ['effr_pct','fragmentation_5d_median_bp','fragmentation_robust_z']);
  for (const row of list(raw.repo.venues)) { strings(row, ['venue','status']); measures(row, ['rate_pct','effr_premium_5d_median_bp']); }
  for (const row of list(raw.repo.volume_context)) { strings(row, ['title','unit','measure_kind']); optionalText(row, ['latest_date','status']); measures(row, ['latest_value']); }
}

export async function decodePackage(text: string): Promise<{ payload: ResearchPackage; release: ResearchRelease; fundamentals: AtlasIndex; liquidity: LiquidityReport | null }> {
  requireField(new TextEncoder().encode(text).length <= MAX_PACKAGE_BYTES, 'Research files must be smaller than 32 MB.');
  let payload: ResearchPackage;
  try { payload = JSON.parse(text); } catch { throw new Error('Choose a valid Macro Atlas research file (.atlas.json).'); }
  requireField(payload?.format === 'macro-atlas-research' && payload.schema_version === 1, 'This research package requires a different Atlas version.');
  const read = async (doc: ResearchDocument) => {
    requireField(doc && typeof doc.content === 'string' && typeof doc.source_file === 'string' && !/[\\/]/.test(doc.source_file), 'The source document is invalid.');
    requireField(/^[a-f0-9]{64}$/.test(doc.sha256) && await digest(doc.content) === doc.sha256, "The research file’s checksum does not match its contents.");
    return JSON.parse(doc.content);
  };
  const fundamentals = await read(payload.fundamentals); validateFundamentals(fundamentals);
  const liquidity = payload.liquidity ? await read(payload.liquidity) as LiquidityReport : null;
  if (liquidity) validateLiquidity(liquidity);
  const release: ResearchRelease = {
    id: await digest(`macro-atlas-research-v1\n${payload.fundamentals.sha256}\n${payload.liquidity?.sha256 ?? ''}`),
    as_of: fundamentals.as_of, generated_at: fundamentals.generated_at, fundamentals_sha256: payload.fundamentals.sha256,
    liquidity_as_of: liquidity?.as_of ?? null, liquidity_sha256: payload.liquidity?.sha256 ?? null,
    country_count: Object.keys(fundamentals.countries).length, indicator_count: fundamentals.indicators.length, storage: 'imported',
  };
  return { payload, release, fundamentals, liquidity };
}

function database(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('macro-atlas-research-v1', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('packages', { keyPath: 'id' });
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  });
}
async function browserRead(id?: string): Promise<any> {
  const db = await database();
  try { return await new Promise((resolve, reject) => {
    const tx = db.transaction('packages', 'readonly'); const request = id ? tx.objectStore('packages').get(id) : tx.objectStore('packages').getAll();
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  }); } finally { db.close(); }
}
export async function listResearch(): Promise<{ catalogue: ResearchCatalogue; releases: ResearchRelease[]; unreadable: number }> {
  const catalogue = await json<ResearchCatalogue>('./data/catalog.json');
  let saved: ResearchRelease[] = [], unreadable = 0;
  if (native()) { const result = await invoke<{ releases: ResearchRelease[]; unreadable: number }>('research_list'); saved = result.releases; unreadable = result.unreadable; }
  else for (const entry of await browserRead()) { try { saved.push((await decodePackage(entry.text)).release); } catch { unreadable++; } }
  const merged = new Map(catalogue.releases.map(r => [r.id, r]));
  for (const release of saved) merged.set(release.id, { ...release, storage: 'imported' });
  return { catalogue, releases: [...merged.values()].sort((a,b) => Date.parse(b.generated_at) - Date.parse(a.generated_at) || b.id.localeCompare(a.id)), unreadable };
}
export async function inspectResearch(text: string): Promise<ResearchRelease> {
  return native() ? { ...await invoke<ResearchRelease>('research_inspect', { contents: text }), storage: 'imported' } : (await decodePackage(text)).release;
}
export async function importResearch(text: string): Promise<ResearchRelease> {
  if (native()) return { ...await invoke<ResearchRelease>('research_import', { contents: text }), storage: 'imported' };
  const { release } = await decodePackage(text); const db = await database();
  try { await new Promise<void>((resolve, reject) => {
    const tx = db.transaction('packages', 'readwrite'); const store = tx.objectStore('packages'); const existing = store.get(release.id);
    existing.onsuccess = () => { if (!existing.result) store.add({ id: release.id, text }); };
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error);
  }); } finally { db.close(); }
  return release;
}
export async function packageText(release: ResearchRelease): Promise<string> {
  if (release.storage === 'included') { const response = await fetch(release.package_url!); if (!response.ok) throw new Error('The included research file could not be opened.'); return response.text(); }
  if (native()) return invoke<string>('research_package', { id: release.id });
  const saved = await browserRead(release.id); if (!saved) throw new Error('This saved research file is unavailable.'); return saved.text;
}
export async function resource<T>(release: ResearchRelease, key: string, signal?: AbortSignal): Promise<T> {
  if (release.storage === 'included') {
    if (key === 'liquidity' && !release.liquidity_sha256) return null as T;
    const file = key.startsWith('country:') ? `countries/${key.slice(8)}.json` : `${key}.json`;
    return json<T>(`${release.base}/${file}`, signal);
  }
  if (native()) return invoke<T>('research_resource', { id: release.id, resource: key });
  const decoded = await decodePackage(await packageText(release));
  if (key === 'liquidity') return decoded.liquidity as T;
  if (key.startsWith('country:')) return decoded.fundamentals.countries[key.slice(8)] as T;
  if (key === 'history') return Object.fromEntries(Object.entries(decoded.fundamentals.countries).map(([k,c]) => [k,c.history])) as T;
  const index = { ...decoded.fundamentals, countries: Object.fromEntries(Object.entries(decoded.fundamentals.countries).map(([k,c]) => { const { history: _history, ...summary } = c; return [k,summary]; })), manifest: { sha256: release.fundamentals_sha256, source_file: decoded.payload.fundamentals.source_file, source_bytes: new TextEncoder().encode(decoded.payload.fundamentals.content).length, country_files: {} } };
  return index as T;
}
export const countryResource = (release: ResearchRelease, code: string, signal?: AbortSignal) => resource<Country>(release, `country:${code}`, signal);
export const historyResource = (release: ResearchRelease, signal?: AbortSignal) => resource<HistoryPanel>(release, 'history', signal);
export async function exportResearch(release: ResearchRelease): Promise<string> {
  const text = await packageText(release);
  if (native()) return `Saved to ${await invoke<string>('research_export', { contents: text })}`;
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `Macro-Atlas-Research-${release.as_of}-${release.id.slice(0,12)}.atlas.json`; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000); return 'Research file exported.';
}

export function previousRelease(releases: ResearchRelease[], selected: ResearchRelease): ResearchRelease | undefined {
  return releases.filter(r => Date.parse(r.generated_at) < Date.parse(selected.generated_at) && r.fundamentals_sha256 !== selected.fundamentals_sha256)
    .sort((a,b) => Date.parse(b.generated_at) - Date.parse(a.generated_at))[0];
}
