export type Mode = 'fundamentals' | 'history' | 'trade';
export type Category = 'real_stuff' | 'production' | 'exchange' | 'promises' | 'enforcer';
export type Indicator = {
  name: string; category: Category; label: string; unit: string;
  uncertainty: string; higher_is_better: boolean; description: string;
  cadence: string; sources: string[]; scored: boolean; forward: boolean;
};
export type Cell = {
  value: number | null; date: string | null; source: string | null; pct: number | null;
  trend: string | null; uncertainty: string; is_forecast: boolean;
  lag_value?: number | null; trend_5y?: number | null; se?: number | null;
};

export type ResearchRelease = {
  id: string; as_of: string; generated_at: string; fundamentals_sha256: string;
  liquidity_as_of: string | null; liquidity_sha256: string | null;
  business_as_of?: string | null; business_sha256?: string | null; company_count?: number;
  taxonomy_as_of?: string | null; taxonomy_sha256?: string | null; sector_count?: number; branch_count?: number;
  country_count: number; indicator_count: number;
  storage: 'included' | 'imported'; base?: string; package_url?: string;
};
export type ResearchCatalogue = { version: number; default_id: string; releases: ResearchRelease[] };
export type ResearchDocument = { source_file: string; sha256: string; content: string };
export type ResearchPackage = { format: string; schema_version: number; fundamentals: ResearchDocument; liquidity: ResearchDocument | null; business?: ResearchDocument | null; taxonomy?: ResearchDocument | null };
export type HistoryPanel = Record<string, Record<string, Point[]>>;
export type Point = { year: number; value: number | null; is_forecast: boolean };
export type Pressure = {
  rule_id: string; title: string; constraint: string; forced_options: string[];
  spillovers: { target: string; text: string; channel: string }[];
  confidence: number; uncertainty: string;
};
export type Country = {
  name: string; iso3: string; on_map: boolean; currency: string;
  data_quality: { flag: string; note: string | null };
  categories: Record<Category, { score: number | null; n_available: number; n_total: number }>;
  indicators: Record<string, Cell>;
  history?: Record<string, Point[]>;
  pressures: Pressure[];
  cycle?: { long_term_label: string; long_term_confidence: number; short_term_label: string; short_term_confidence: number };
};
export type Trade = { iso2: string; partner: string; year: number; x_share: number | null; m_share: number | null; x_usd: number | null; m_usd: number | null };
export type AtlasIndex = {
  version: number; as_of: string; generated_at: string; ranking_population: string[];
  categories: Category[]; indicators: Indicator[]; countries: Record<string, Country>;
  trade: Trade[]; manifest: { sha256: string; source_file: string; source_bytes: number; country_files: Record<string, string> };
};
