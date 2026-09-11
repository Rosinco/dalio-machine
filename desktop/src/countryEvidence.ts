import { digest } from './research';

export type EvidencePoint = { value: number | null; status: string; unit?: string; year?: number; date?: string; period?: string; period_start?: string; period_end?: string; evidence_ref?: string; source?: string; source_locator?: string; [key: string]: any };
export type EvidenceHistory = { country: string; indicator: string; label?: string; definition?: string; series_id: string; source: string; source_url: string; unit: string; frequency: string; available_at: string; observations: EvidencePoint[]; [key: string]: any };
export type EvidenceEnvelope = { snapshot_sha256: string; as_of: string; as_known_at: string; methodology: Record<string, any>; profile: Record<string, any>; histories: EvidenceHistory[]; citations: Record<string, EvidencePoint>; baseline_year?: number; horizon_end_year?: number; input_gaps?: Record<string, any>[]; remaining_gaps?: string[] };
export type CountryEvidence = { version: 1; country: string; name: string; listing_iso2: string; assessment: EvidenceEnvelope; monitoring: EvidenceEnvelope | null };
export type EvidenceEntry = { name: string; listing_iso2: string; assessment_as_of: string; monitoring_as_of: string | null; signals: number; sha256: string; bytes: number; file: string };
export type EvidenceIndex = { format: 'macro-atlas-country-evidence-v1'; version: 1; id: string; sources: { kind: string; snapshot_sha256: string; file_sha256: string; as_of: string; as_known_at: string; file: string }[]; countries: Record<string, EvidenceEntry> };
export const evidenceCode = (code: string) => code === 'GB' ? 'UK' : code;
const requireField = (ok: unknown, message: string) => { if (!ok) throw new Error(message); };
const hash = (value: unknown) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const object = (value: any) => value !== null && typeof value === 'object' && !Array.isArray(value);
const clock = (value: unknown) => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}T.*(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value));
const number = (value: unknown) => value === null || typeof value === 'number' && Number.isFinite(value);
export const evidenceDate = (point?: EvidencePoint | null) => point?.period ?? (point?.period_start && point.period_end ? point.period_start === point.period_end ? point.period_end : `${point.period_start} – ${point.period_end}` : point?.year?.toString() ?? point?.date ?? 'Not available');
export const evidenceValue = (point?: EvidencePoint | null, unit?: string) => {
  if (point?.value == null) return 'Not available';
  const nativeUnit = unit ?? point.unit ?? '';
  const displayedUnit = nativeUnit === 'SEK_bn' ? 'billion SEK' : nativeUnit;
  return `${point.value.toLocaleString('en-GB', nativeUnit === 'SEK' ? { maximumFractionDigits: 2 } : { maximumSignificantDigits: 6 })} ${displayedUnit}`.trim();
};
export const evidenceLabel = (value: string) => value.replaceAll('_', ' ');
export const nativeStatus = (point?: EvidencePoint | null, source?: string) => {
  const flag = point?.native_status;
  if (flag == null || flag === '') return '';
  if (flag === 'P' && source?.includes('BUNDESBANK')) return 'Native P · provisional';
  return `Native status: ${typeof flag === 'string' ? flag : JSON.stringify(flag)}`;
};
export function safeSource(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  try { const url = new URL(value); return url.protocol === 'https:' && !url.username && !url.password; } catch { return false; }
}
function canonical(value: any): string { return JSON.stringify(value, (_key, item) => object(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : item); }
export async function validateEvidenceIndex(raw: any): Promise<EvidenceIndex> {
  requireField(raw?.format === 'macro-atlas-country-evidence-v1' && raw.version === 1 && hash(raw.id) && object(raw.countries), 'Unsupported country evidence pack.');
  const keys = Object.keys(raw.countries);
  requireField(keys.length > 0 && keys.length <= 250 && Array.isArray(raw.sources) && raw.sources.length <= 50, 'Invalid country evidence catalogue.');
  for (const code of keys) {
    const entry = raw.countries[code];
    requireField(/^[A-Z]{2}$/.test(code) && typeof entry.name === 'string' && hash(entry.sha256) && Number.isSafeInteger(entry.bytes) && entry.bytes > 0 && entry.bytes <= 32 * 1024 * 1024 && entry.file === `${raw.id}/countries/${code}.json`, 'Invalid evidence country file.');
  }
  for (const source of raw.sources) requireField(hash(source.snapshot_sha256) && hash(source.file_sha256) && clock(source.as_known_at), 'Invalid evidence source identity or cutoff.');
  const base = structuredClone(raw); delete base.id;
  for (const entry of Object.values(base.countries) as any[]) delete entry.file;
  requireField(await digest(canonical(base)) === raw.id, 'Country evidence index checksum failed.');
  return raw;
}
export function validateCountryEvidence(raw: any, country: string): CountryEvidence {
  requireField(raw?.version === 1 && raw.country === country && typeof raw.name === 'string', 'Country evidence identity mismatch.');
  for (const envelope of [raw.assessment, raw.monitoring].filter(Boolean)) {
    requireField(hash(envelope.snapshot_sha256) && clock(envelope.as_known_at) && typeof envelope.as_of === 'string' && envelope.as_of <= envelope.as_known_at.slice(0, 10) && envelope.profile?.country === country && object(envelope.citations) && Array.isArray(envelope.histories), 'Country evidence dates or profile are invalid.');
    const refs = envelope.citations;
    const visit = (value: any) => {
      if (Array.isArray(value)) { for (const child of value) visit(child); }
      else if (object(value)) {
        if (value.evidence_ref) {
          const cited = refs[value.evidence_ref]; requireField(cited?.country === country, 'Unresolved or wrong-country evidence reference.');
          for (const key of ['value', 'unit', 'year', 'date', 'period', 'status']) if (key in value) requireField(value[key] === cited[key], 'Evidence point differs from its citation.');
        }
        for (const [key, child] of Object.entries(value)) {
          if (key.endsWith('evidence_refs')) requireField(Array.isArray(child) && child.every(ref => refs[ref]?.country === country), 'Unresolved country evidence references.');
          else if (key !== 'evidence_ref') visit(child);
        }
      }
    };
    visit(envelope.profile);
    for (const series of envelope.histories) {
      requireField(series.country === country && typeof series.indicator === 'string' && typeof series.unit === 'string' && clock(series.available_at) && Date.parse(series.available_at) <= Date.parse(envelope.as_known_at) && Array.isArray(series.observations) && series.observations.length <= 100000, 'Native evidence history is invalid.');
      const periods = new Set();
      for (const row of series.observations) {
        const period = row.period ?? row.date;
        requireField(typeof period === 'string' && !periods.has(period) && number(row.value) && typeof row.status === 'string', 'Native history has duplicate periods or invalid values.'); periods.add(period);
      }
    }
  }
  requireField(raw.assessment && object(raw.assessment.profile.baseline) && Array.isArray(raw.assessment.profile.projections) && Array.isArray(raw.assessment.profile.scenarios), 'Annual assessment is incomplete.');
  requireField(raw.monitoring === null || Array.isArray(raw.monitoring.profile.signals), 'Monitoring profile is incomplete.');
  return raw;
}
export async function decodeCountryEvidence(text: string, index: EvidenceIndex, country: string): Promise<CountryEvidence> {
  const entry = index.countries[country];
  requireField(entry && new TextEncoder().encode(text).byteLength === entry.bytes && await digest(text) === entry.sha256, 'Country evidence file checksum failed.');
  return validateCountryEvidence(JSON.parse(text), country);
}
export function evidenceLines(series: EvidenceHistory, startYear = 1990) {
  const rows = series.observations.filter(row => Number((row.period ?? row.date ?? String(row.year)).slice(0, 4)) >= startYear);
  const byPeriod = new Map(rows.map(row => [series.frequency === 'annual' ? String(row.year) : row.period ?? row.date!, row]));
  let labels = [...byPeriod.keys()].sort();
  if (labels.length && series.frequency === 'annual') labels = Array.from({ length: Number(labels.at(-1)) - Number(labels[0]) + 1 }, (_, i) => String(Number(labels[0]) + i));
  if (labels.length && series.frequency === 'monthly') {
    const first = labels[0], last = labels.at(-1)!;
    const number = (p: string) => Number(p.slice(0, 4)) * 12 + Number(p.slice(5, 7)) - 1;
    labels = Array.from({ length: number(last) - number(first) + 1 }, (_, i) => { const n = number(first) + i; return `${Math.floor(n / 12)}-${String(n % 12 + 1).padStart(2, '0')}`; });
  }
  const forecast = (row?: EvidencePoint) => !!row && (/forecast/i.test(row.status) || /_FCST$/.test(row.source ?? ''));
  return { labels, historical: labels.map(label => { const row = byPeriod.get(label); return forecast(row) ? null : row?.value ?? null; }), forecast: labels.map(label => { const row = byPeriod.get(label); return forecast(row) ? row?.value ?? null : null; }) };
}
