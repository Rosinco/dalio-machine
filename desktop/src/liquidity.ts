export type LiquidityPoint = { date: string; period: string; annual_log_growth_pct: number | null; acceleration_3m_pp?: number | null; acceleration_1q_pp?: number | null };
export type Trace = { availability_status: string; input_release_ids: number[]; input_series_ids?: string[]; input_statuses?: string[]; formula_ids?: string[]; interpretation_limit?: string };
export type MoneyReading = Trace & { country: string; currency: string; title: string; unit: string; period: string; period_date: string; latest_value: number | null; annual_log_growth_pct: number | null; acceleration_3m_pp: number | null; movement: string; history?: LiquidityPoint[] };
export type OffshoreReading = Trace & { currency: string; title: string; unit: string; period: string; latest_value: number | null; annual_log_growth_pct: number | null; acceleration_1q_pp: number | null; movement: string; history?: LiquidityPoint[] };
export type InputRelease = { release_id: number; source_family: string; partition_key: string; content_sha256: string; available_at: string; published_at: string | null; source_url: string | null; artifact_manifest_status: string; artifacts: { role: string; sha256: string; path: string }[] };
export type LiquidityReport = {
  version: number; as_of: string; as_known_at: string; methodology_version: string; methodology_sha256: string; snapshot_sha256: string;
  complete_snapshot_available_at: string | null; earliest_input_available_at: string | null;
  history_mode: string; history_basis: string; broad_money: MoneyReading[];
  central_bank_divergence: (Trace & {country: string; currency: string; period: string; money_annual_log_growth_pct: number | null; central_bank_assets_annual_log_growth_pct: number | null; money_minus_assets_growth_gap_pp: number | null; gap_change_3m_pp: number | null})[];
  offshore_credit: OffshoreReading[];
  money_summary: { ready: number; expected: number; common_period: string | null; median_annual_log_growth_pct: number | null; positive_growth_breadth: number | null; accelerating_breadth: number | null; interpretation_limit: string };
  mmf: Trace & { period: string; mmf_annual_log_growth_pct: number | null; mmf_minus_m2_growth_gap_pp: number | null; mmf_assets_to_m2_scale_pct: number | null; asset_allocation: { title: string; share_of_total_pct: number | null }[]; published_repo_counterparty_categories: {period: string; availability_status: string; ratios: {title: string;share_of_repo_pct:number|null}[];interpretation_limit: string} };
  repo: Trace & { period_date: string; effr_pct: number | null; fragmentation_5d_median_bp: number | null; fragmentation_robust_z: number | null; maximum_effr_premium_5d_median_bp: number | null; venues: {venue: string;rate_pct: number | null;effr_premium_5d_median_bp: number | null;status: string}[]; volume_context: {title:string;latest_value:number|null;unit:string;measure_kind:string;latest_date:string;status:string}[] };
  coverage: Record<string,{ready:number;expected:number}>; formulas: Record<string,string>;
  evidence_integrity: Record<string, number>; input_releases: InputRelease[];
  horizons: {horizon:string;supported_context:string}[]; interpretation_limits: string[];
};
export function countryMoney(report: LiquidityReport, code: string, currency: string) {
  const direct = report.broad_money.find(r=>r.country===code);
  if (direct) return { reading: direct, scope: code === 'EU' ? 'currency-area' : 'country' };
  if (currency === 'EUR') return { reading: report.broad_money.find(r=>r.country==='EU'), scope:'currency-area' };
  return { reading: undefined, scope: 'unavailable' };
}
export function datedHistory(points: LiquidityPoint[], cadence: 'monthly' | 'quarterly') {
  const ordered = [...points].sort((a,b)=>a.date.localeCompare(b.date));
  if (!ordered.length) return { periods: [] as string[], values: [] as (number | null)[] };
  const month = (date: string) => Number(date.slice(0,4))*12 + Number(date.slice(5,7))-1;
  const valuesByMonth = new Map(ordered.map(p=>[month(p.date),p.annual_log_growth_pct]));
  const periods: string[] = [], values: (number | null)[] = [];
  for (let i=month(ordered[0].date);i<=month(ordered.at(-1)!.date);i+=cadence==='monthly'?1:3) {
    periods.push(`${Math.floor(i/12)}-${String(i%12+1).padStart(2,'0')}`);
    const value = valuesByMonth.get(i); values.push(typeof value === 'number' && Number.isFinite(value)?value:null);
  }
  return { periods, values };
}
