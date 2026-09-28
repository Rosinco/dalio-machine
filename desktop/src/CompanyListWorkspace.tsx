import { useEffect, useMemo, useRef, useState } from 'react';
import { useScreenState } from './NavigationContext';
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, ArrowUpDown, Check, ChevronDown, Columns3, Download, GitCompareArrows, Info, Plus, Save, Search, Settings2, SlidersHorizontal, Star, Trash2, X } from 'lucide-react';
import { COMPANY_KPIS, COMPANY_LIST_MAX_COLUMNS, COMPANY_LIST_STORAGE_KEY, LEGACY_COMPANY_LIST_STORAGE_KEY, VALUATION_RANKING_KPI_IDS, columnLabel, companyListCell, companyListVariants, companyListWindowLabel, companyListCalculationLabel, companyListUnit, defaultCompanyListPreferences, matchesCompanyListFilters, parseCompanyListPreferences } from './companyListModel';
import type { CompanyKpi, CompanyListColumn, CompanyListFilters, CompanyListNumericRule, CompanyListPreferences } from './companyListModel';
import { activeCompanyWatchlist, companyListCsv, defaultCompanyListFeatures, sortCompanyListRows, toggleCompanyComparison, updateCompanyWatchlist } from './companyListFeatures';
import type { CompanyListFeatures } from './companyListFeatures';
import type { FinancialIndex } from './financialData';
import { useExpandedKpis } from './useExpandedKpis';
import { expandedVariant, EXPANDED_KPI_MANIFEST } from './expandedKpis';
import { CompanyListComparison, CompanyListValueDetails } from './CompanyListDetails';
import { researchReadinessLabels, researchRouteLabels } from './ResearchGaugeCard';
import type { ResearchGaugeArtifact, ResearchGaugeRow } from './researchGaugeModel';
import { CompanyKpiExplanation } from './CompanyKpiExplanation';
import { CompanyListColumnRange } from './CompanyListColumnRange';
import { companyListRangeActive, companyListRangeError, matchesCompanyListRange } from './companyListRanges';
import type { CompanyListRange, CompanyListRangeCell } from './companyListRanges';
import { buildNormalQualityValueContext } from './normalQualityValueModel';
import { NORMAL_RANKING_KPIS, NORMAL_RANKING_DEPENDENCIES, NORMAL_RANKING_POLICY_ID, NORMAL_RANKING_REFERENCE } from './normalQualityValuePolicy';
import './companyList.css';

const count = (value: number) => value.toLocaleString('en-US');
const alphabetical = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const countryNames = new Intl.DisplayNames(['en'], { type: 'region' });
const kpiById = new Map(COMPANY_KPIS.map(kpi => [kpi.id, kpi]));
const categories = [...new Set(COMPANY_KPIS.map(kpi => kpi.category))];
const hasSavedValues = (kpi: CompanyKpi) => !kpi.id.startsWith('provider_') || (kpi.coverage ?? 0) > 0;
const usableKpiCount = COMPANY_KPIS.filter(hasSavedValues).length;
const presets = { all: 'All observations', cash_consistency: 'Consistent cash + EBIT', cash_and_margin: 'Cash + 30% margin', normal_quality: 'Comparable history excluding 2020–2023' };
const normalYearWindow = 'normal_2020_2023:5' as const;
const normalYearColumn = (column: CompanyListColumn) => column.window === normalYearWindow || column.kpiId.startsWith('normal_');
const monetary = (column: CompanyListColumn) => ['money', 'price'].includes(companyListUnit(column));
const numeric = (column: CompanyListColumn) => !['text', 'date'].includes(companyListUnit(column));
const calculationsFor = (kpi: CompanyKpi, window: CompanyListColumn['window']) => companyListVariants(kpi).filter(value => value.window === window).map(value => value.calculation);
const sameColumn = (a: CompanyListColumn, b: CompanyListColumn) => a.id === b.id && a.kpiId === b.kpiId && a.window === b.window && a.calculation === b.calculation;
const calculationLabel = companyListCalculationLabel;
const headingLabel = (column: CompanyListColumn) => column.kpiId === 'normal_npv_percent'
  ? ({ terminal_100: 'Normal-year full surplus', terminal_50: 'Normal-year half terminal', terminal_0: 'Normal-year cash-only surplus' } as Record<string, string>)[column.calculation] ?? 'Normal-year surplus'
  : column.kpiId === 'valuation_attractiveness'
  ? ({ terminal_100: 'Full DCF surplus', terminal_75: '75% terminal surplus', terminal_50: 'Half terminal surplus', terminal_25: '25% terminal surplus', terminal_0: 'Cash-only surplus' } as Record<string, string>)[column.calculation] ?? 'Valuation surplus'
  : kpiById.get(column.kpiId)?.label;
const headingContext = (column: CompanyListColumn) => {
  if (NORMAL_RANKING_KPIS.has(column.kpiId)) return '0–100 rank points · fixed reference';
  if (column.kpiId === 'valuation_attractiveness') return 'Saved starter scenario';
  if (column.kpiId === 'normal_npv_percent') return '10-year model · 2020–2023 excluded';
  if (column.kpiId === 'normal_npv_5y_percent') return 'Years 1–5 · no terminal value';
  if (column.kpiId.startsWith('provider_')) return `${companyListWindowLabel(column).replace('Latest provider snapshot', 'Latest saved').replace(' · provider', '')} · ${calculationLabel(column)}`;
  if (column.window !== 'latest') return `${companyListWindowLabel(column)}${column.calculation === 'latest' ? '' : ` · ${calculationLabel(column)}`}`;
  return companyListWindowLabel(column).replace('Latest annual report', 'Latest annual').replace('Standard starter snapshot', 'Saved starter').replace('Saved quarter comparison', 'Quarter vs prior year');
};

const unitLabel = (column: CompanyListColumn) => ({ money: 'currency millions', price: 'currency per share', percent: 'percent', points: 'percentage points', multiple: 'multiple', count: 'count', number: 'number', shares_millions: 'million shares', date: 'date', text: 'text' })[companyListUnit(column)];
const id = () => `list-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
const columnPresetDefinitions: { id: string; label: string; metricIds: string[] }[] = [
  { id: 'normal_quality_value', label: 'Five-year NPV + quality · 60/40', metricIds: ['normal_quality_value_score', 'normal_npv_5y_percent', 'normal_discount_rank', 'normal_quality_rank', 'normal_npv_percent', 'normal_roce', 'ebit_margin', 'cfo', 'provider_42', 'price_date'] },
  { id: 'normal_years', label: 'Quality & value excluding 2020–2023', metricIds: ['normal_npv_percent', 'normal_roce', 'normal_rota', 'tangible_assets_revenue', 'ebit_margin', 'fcf_margin', 'positive_fcf', 'positive_ebit', 'revenue', 'price_date', 'country', 'branch'] },
  { id: 'valuation_rank', label: 'Valuation attractiveness', metricIds: ['valuation_attractiveness', 'mid_npv_percent', 'dcf_price_ratio', 'cash_price_coverage', 'terminal_share', 'low_npv_percent', 'cash_factor_30', 'price_date', 'coverage'] },
  { id: 'terminal_sensitivity', label: 'Terminal sensitivities', metricIds: ['valuation_attractiveness', 'cash_price_coverage', 'terminal_share', 'low_npv_percent', 'price_date', 'coverage'] },
  { id: 'valuation', label: 'Valuation', metricIds: ['provider_2', 'provider_4', 'provider_3', 'provider_10', 'provider_11', 'provider_13', 'provider_50', 'cash_factor_30', 'terminal_share'] },
  { id: 'dividends', label: 'Dividends', metricIds: ['provider_1', 'provider_7', 'provider_20', 'provider_26', 'provider_66', 'provider_148', 'positive_fcf', 'fcf_margin'] },
  { id: 'quality', label: 'Profitability & returns', metricIds: ['provider_33', 'provider_34', 'provider_36', 'provider_37', 'provider_29', 'provider_32', 'provider_38', 'positive_fcf'] },
  { id: 'strength', label: 'Financial strength', metricIds: ['provider_42', 'provider_44', 'provider_39', 'provider_40', 'provider_41', 'provider_46', 'provider_60', 'cash_balance'] },
  { id: 'price_ownership', label: 'Price & insider activity', metricIds: ['provider_151', 'provider_152', 'provider_50', 'provider_159', 'provider_311', 'provider_110'] },
];
function presetColumns(presetId: string): CompanyListColumn[] {
  const definition = columnPresetDefinitions.find(item => item.id === presetId);
  if (!definition) return [];
  const metrics = definition.metricIds.filter(metric => kpiById.has(metric) && hasSavedValues(kpiById.get(metric)!));
  const selected = new Set<string>(['valuation_rank', 'terminal_sensitivity'].includes(presetId) ? [...metrics, 'branch', 'country'] : ['country', 'branch', ...metrics]);
  return [...selected].flatMap(kpiId => {
    if (presetId === 'normal_years' || presetId === 'normal_quality_value') {
      if (kpiId === 'normal_npv_percent') return ([100, 50, 0] as const).map(credit => ({ id: `preset-${presetId}-${kpiId}-${credit}`, kpiId, window: 'latest' as const, calculation: `terminal_${credit}` as const }));
      if (['normal_roce', 'normal_rota', 'tangible_assets_revenue', 'ebit_margin', 'fcf_margin'].includes(kpiId)) return { id: `preset-${presetId}-${kpiId}`, kpiId, window: normalYearWindow, calculation: presetId === 'normal_quality_value' && kpiId === 'ebit_margin' ? 'min' as const : 'median' as const };
      if (['positive_fcf', 'positive_ebit', 'revenue'].includes(kpiId)) return { id: `preset-${presetId}-${kpiId}`, kpiId, window: normalYearWindow, calculation: kpiId === 'revenue' ? 'growth' as const : 'latest' as const };
      if (kpiId === 'cfo') return { id: `preset-${presetId}-${kpiId}`, kpiId, window: normalYearWindow, calculation: 'growth' as const };
    }
    if (presetId === 'terminal_sensitivity' && kpiId === 'valuation_attractiveness') return ([100, 50, 0] as const).map(credit => ({ id: `preset-${presetId}-${kpiId}-${credit}`, kpiId, window: 'latest' as const, calculation: `terminal_${credit}` as const }));
    const metric = kpiById.get(kpiId)!;
    const variants = companyListVariants(metric).filter(variant => !kpiId.startsWith('provider_') || (expandedVariant({ id: 'preset', kpiId, ...variant })?.availableCount ?? 0) > 0);
    const specific = ['positive_fcf', 'positive_ebit'].includes(kpiId) ? variants.find(variant => variant.window === '5') : kpiId === 'provider_152' ? variants.find(variant => variant.window.endsWith(':1year')) : kpiId === 'provider_311' ? variants.find(variant => variant.window.endsWith(':30day')) : kpiId === 'provider_110' ? variants.find(variant => variant.window.endsWith(':1year') && variant.calculation === 'provider:BuyersDiffSellers') : undefined;
    const preferred = specific ?? variants.find(variant => variant.window.endsWith(':last') && variant.calculation === 'provider:latest') ?? variants.find(variant => variant.window.endsWith(':last') && variant.calculation === 'provider:default') ?? variants.find(variant => variant.window === 'latest') ?? variants[0];
    return { id: `preset-${presetId}-${kpiId}`, kpiId, window: preferred.window, calculation: preferred.calculation };
  }).slice(0, COMPANY_LIST_MAX_COLUMNS);
}
function countryLabel(value: string) {
  if (value === 'unassigned') return 'Country unavailable';
  try { return `${countryNames.of(value) ?? value} (${value})`; } catch { return value; }
}
function readPreferences(): { preferences: CompanyListPreferences; error: string } {
  try {
    const raw = localStorage.getItem(COMPANY_LIST_STORAGE_KEY) ?? localStorage.getItem(LEGACY_COMPANY_LIST_STORAGE_KEY);
    if (!raw) return { preferences: defaultCompanyListPreferences(), error: '' };
    if (raw.length > 16_000_000) return { preferences: defaultCompanyListPreferences(), error: 'The saved list settings exceed the supported size. Defaults are shown. Saved settings change only when you edit this view.' };
    try {
      const parsed = JSON.parse(raw);
      if (parsed?.version !== 1 && parsed?.version !== 2) return { preferences: defaultCompanyListPreferences(), error: 'The saved list settings use an unsupported format. Defaults are shown. Saved settings change only when you edit this view.' };
      const preferences = parseCompanyListPreferences(parsed);
      const repaired = !Array.isArray(parsed.columns) || parsed.columns.length !== preferences.columns.length || parsed.columns.some((column: CompanyListColumn, index: number) => !column || !sameColumn(column, preferences.columns[index]))
        || !Array.isArray(parsed.watchlistIds) || parsed.watchlistIds.length !== preferences.watchlistIds.length || !Array.isArray(parsed.savedViews) || parsed.savedViews.length !== preferences.savedViews.length
        || !Array.isArray(parsed.filters?.numericRules) || parsed.filters.numericRules.length !== preferences.filters.numericRules.length;
      return { preferences, error: repaired ? 'Some saved list settings could not be restored. Available settings are shown. Saved settings change only when you edit this view.' : '' };
    } catch { return { preferences: defaultCompanyListPreferences(), error: 'The saved list settings could not be read. Defaults are shown. Saved settings change only when you edit this view.' }; }
  } catch { return { preferences: defaultCompanyListPreferences(), error: 'Local list storage is unavailable. You can use this list, but changes may not survive a restart.' }; }
}

export default function CompanyListWorkspace({ data, ready, error, financial, taxonomySha256, initialBranchId, onCompany, onBack }: {
  data: ResearchGaugeArtifact | null; ready: boolean; error: string; financial?: FinancialIndex | null; taxonomySha256?: string | null; initialBranchId?: string; onCompany: (id: string) => void; onBack: () => void;
}) {
  const [initial] = useState(readPreferences);
  const [preferences, setPreferences] = useState(initial.preferences), [storageError, setStorageError] = useState(initial.error);
  const [saveEpoch, setSaveEpoch] = useState(0), [page, setPage] = useScreenState('lists-page', 0), [pageSize, setPageSize] = useScreenState('lists-page-size', 50);
  const [showFilters, setShowFilters] = useScreenState('lists-show-filters', false), [modal, setModal] = useState<'columns' | 'save' | 'watchlists' | 'compare' | 'cell' | 'help' | null>(null);
  const [toolPanel, setToolPanel] = useScreenState<'columns' | 'tools' | null>('lists-tool-panel', null);
  const [pickerQuery, setPickerQuery] = useScreenState('lists-picker-query', ''), [category, setCategory] = useScreenState('lists-picker-category', 'All KPIs');
  const [onlyWithValues, setOnlyWithValues] = useScreenState('lists-picker-with-values', true);
  const [pickerKpi, setPickerKpi] = useScreenState('lists-picker-kpi', COMPANY_KPIS[0].id), [pickerWindow, setPickerWindow] = useScreenState<CompanyListColumn['window']>('lists-picker-window', 'latest');
  const [pickerCalculation, setPickerCalculation] = useScreenState<CompanyListColumn['calculation']>('lists-picker-calculation', 'latest'), [editingColumn, setEditingColumn] = useState<string | null>(null);
  const [pickerNotice, setPickerNotice] = useState(''), [status, setStatus] = useState(''), [viewName, setViewName] = useState(''), [selectedView, setSelectedView] = useScreenState('lists-selected-view', '');
  const [ruleDrafts, setRuleDrafts] = useState<Record<number, string>>({});
  const [upperDrafts, setUpperDrafts] = useState<Record<number, string>>({});
  const [watchlistName, setWatchlistName] = useState(''), [watchlistEditId, setWatchlistEditId] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false), [exportError, setExportError] = useState('');
  const [cellSelection, setCellSelection] = useState<{ row: ResearchGaugeRow; column: CompanyListColumn } | null>(null);
  const [helpColumn, setHelpColumn] = useState<CompanyListColumn | null>(null);
  const [columnPreset, setColumnPreset] = useScreenState('lists-column-preset', '');
  const [appliedBranchScope, setAppliedBranchScope] = useScreenState<string | null>('lists-applied-branch-scope', null);
  const dialog = useRef<HTMLDivElement>(null), returnFocus = useRef<HTMLElement | null>(null);
  const openModal = (value: Exclude<typeof modal, null>) => { returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null; setModal(value); };
  const available = ready && !error ? data : null;
  const rows = available?.rows ?? [];
  const { columns, filters, sort } = preferences;
  const rankingColumns = columns.filter(column => VALUATION_RANKING_KPI_IDS.has(column.kpiId));
  const hasValuationRanking = rankingColumns.length > 0 || filters.numericRules.some(rule => VALUATION_RANKING_KPI_IDS.has(rule.column.kpiId));
  const hasStarterValuationRanking = rankingColumns.some(column => !column.kpiId.startsWith('normal_')) || filters.numericRules.some(rule => !rule.column.kpiId.startsWith('normal_') && VALUATION_RANKING_KPI_IDS.has(rule.column.kpiId));
  const features = useMemo(() => preferences.features ?? defaultCompanyListFeatures(preferences.watchlistIds), [preferences.features, preferences.watchlistIds]);
  const requestedColumns = useMemo(() => [...columns, ...filters.numericRules.map(rule => rule.column)], [columns, filters.numericRules]);
  const hasQualityRanking = requestedColumns.some(column => NORMAL_RANKING_KPIS.has(column.kpiId));
  const expandedColumns = useMemo(() => hasQualityRanking ? [...requestedColumns, ...NORMAL_RANKING_DEPENDENCIES] : requestedColumns, [requestedColumns, hasQualityRanking]);
  const hasNormalYears = filters.preset === 'normal_quality' || requestedColumns.some(normalYearColumn);
  const hasFiveYearValuation = hasQualityRanking || requestedColumns.some(column => column.kpiId === 'normal_npv_5y_percent');
  const hasNormalValuation = requestedColumns.some(column => column.kpiId === 'normal_npv_percent' || column.kpiId === 'normal_cash_pv' || column.kpiId === 'normal_terminal_pv');
  const expanded = useExpandedKpis(financial ?? null, taxonomySha256, expandedColumns, !!available);
  const qualityRanking = useMemo(() => hasQualityRanking && available ? buildNormalQualityValueContext(rows, expanded) : undefined, [hasQualityRanking, available, rows, expanded]);
  const evaluate = (row: ResearchGaugeRow, column: CompanyListColumn) => companyListCell(row, column, expanded, qualityRanking);
  const activeList = activeCompanyWatchlist(features);
  const selectedSavedView = preferences.savedViews.find(view => view.id === selectedView);
  const savedViewSettings = selectedSavedView ? features.viewFeatures[selectedSavedView.id] : undefined;
  const savedViewEdited = !!selectedSavedView && (JSON.stringify([columns, filters, sort]) !== JSON.stringify([selectedSavedView.columns, selectedSavedView.filters, selectedSavedView.sort])
    || !!savedViewSettings && JSON.stringify([features.secondarySorts, features.density, features.activeWatchlistId]) !== JSON.stringify([savedViewSettings.secondarySorts, savedViewSettings.density, savedViewSettings.activeWatchlistId]));
  const watchlist = useMemo(() => new Set(activeList.listingIds), [activeList.listingIds]);
  const comparisons = useMemo(() => new Set(features.comparisonIds), [features.comparisonIds]);
  const comparisonRows = useMemo(() => features.comparisonIds.map(listingId => rows.find(row => row.id === listingId)).filter((row): row is ResearchGaugeRow => !!row), [features.comparisonIds, rows]);
  const visibleWatchlistCount = useMemo(() => rows.filter(row => watchlist.has(row.id)).length, [rows, watchlist]);
  const countries = useMemo(() => [...new Set(rows.map(row => row.country ?? 'unassigned'))].sort(), [rows]);
  const sectors = useMemo(() => [...new Map(rows.map(row => [row.sectorId ?? 'unassigned', row.sectorName ?? 'Unassigned sector'])).entries()].sort((a, b) => alphabetical.compare(a[1], b[1])), [rows]);
  const branches = useMemo(() => [...new Map(rows.filter(row => filters.sectorId === 'all' || (row.sectorId ?? 'unassigned') === filters.sectorId).map(row => [row.branchId ?? 'unassigned', row.branchName ?? 'Unassigned branch'])).entries()].sort((a, b) => alphabetical.compare(a[1], b[1])), [rows, filters.sectorId]);
  const currencies = useMemo(() => [...new Set([...rows.flatMap(row => [row.annual.currency, row.valuation.currency, row.valuation.priceBasis?.currency]), ...(expanded.index?.reportCurrencies ?? []), ...(expanded.index?.quoteCurrencies ?? [])].filter((value): value is string => !!value))].sort(), [rows, expanded.index]);
  const sortColumn = columns.find(column => column.id === sort.columnId);
  const rangeColumns = useMemo(() => columns.filter(column => numeric(column) && companyListRangeActive(column.range)), [columns]);
  const baseMatches = useMemo(() => rows.filter(row => matchesCompanyListFilters(row, filters, watchlist, expanded, qualityRanking)), [rows, filters, watchlist, expanded, qualityRanking]);
  const rangeSelection = JSON.stringify(rangeColumns.map(({ id, kpiId, window, calculation }) => [id, kpiId, window, calculation]));
  // Bound edits reuse observations; changing source data or selected variants discards the cache.
  const rangeCells = useMemo(() => new Map<string, Map<string, CompanyListRangeCell>>(), [rows, expanded, rangeSelection, qualityRanking]);
  const matches = useMemo(() => {
    if (rangeColumns.some(column => companyListRangeError(column.range, companyListUnit(column)))) return [];
    const result = baseMatches.filter(row => rangeColumns.every(column => {
      let cells = rangeCells.get(column.id);
      if (!cells) { cells = new Map(); rangeCells.set(column.id, cells); }
      let observation = cells.get(row.id);
      if (!observation) {
        const { value, unit, currency, status } = companyListCell(row, column, expanded, qualityRanking);
        observation = { value, unit, currency, status }; cells.set(row.id, observation);
      }
      return matchesCompanyListRange(observation, column.range);
    }));
    return sortCompanyListRows(result, columns, [sort, ...features.secondarySorts], (row, column) => companyListCell(row, column, expanded, qualityRanking));
  }, [baseMatches, rangeCells, columns, sort, features.secondarySorts, expanded, rangeColumns, qualityRanking]);
  const pages = Math.max(1, Math.ceil(matches.length / pageSize)), selectedPage = Math.min(page, pages - 1);
  const visibleRows = matches.slice(selectedPage * pageSize, (selectedPage + 1) * pageSize);
  const currentKpi = kpiById.get(pickerKpi) ?? COMPANY_KPIS[0];
  const pickerColumn = { id: 'picker', kpiId: currentKpi.id, window: pickerWindow, calculation: pickerCalculation };
  const pickerVariant = expandedVariant(pickerColumn);
  const pickerMetrics = useMemo(() => COMPANY_KPIS.filter(kpi => (!onlyWithValues || hasSavedValues(kpi)) && (category === 'All KPIs' || kpi.category === category) && `${kpi.id} ${kpi.label} ${kpi.description} ${kpi.formula} ${kpi.category} ${kpi.searchTerms?.join(' ') ?? ''}`.toLowerCase().includes(pickerQuery.trim().toLowerCase())), [category, pickerQuery, onlyWithValues]);
  const numericColumns = columns.filter(numeric);
  const filterCount = [filters.country, filters.sectorId, filters.branchId, filters.route, filters.readiness, filters.presence].filter(value => value !== 'all').length + filters.numericRules.length + rangeColumns.length;
  const sortDescription = sortColumn ? `${columnLabel(sortColumn)} · ${sort.direction === 'asc' ? 'low to high' : 'high to low'}${monetary(sortColumn) ? ' within each currency; currencies A–Z' : ''}` : `Company name · ${sort.direction === 'asc' ? 'A–Z' : 'Z–A'}`;
  const change = (update: (value: CompanyListPreferences) => CompanyListPreferences) => { setPreferences(update); setSaveEpoch(value => value + 1); setStatus(''); };
  const changeFeatures = (update: (value: CompanyListFeatures) => CompanyListFeatures) => change(value => {
    const next = update(value.features ?? defaultCompanyListFeatures(value.watchlistIds));
    return { ...value, features: next, watchlistIds: next.watchlists.find(list => list.id === 'default')?.listingIds ?? value.watchlistIds };
  });
  const changeFilters = (patch: Partial<CompanyListFilters>) => change(value => ({ ...value, filters: { ...value.filters, ...patch } }));
  const changeRange = (columnId: string, range: CompanyListRange | undefined) => change(value => ({ ...value, columns: value.columns.map(column => {
    if (column.id !== columnId) return column;
    const { range: previousRange, ...identity } = column;
    return range ? { ...identity, range } : identity;
  }) }));
  useEffect(() => {
    if (initialBranchId && appliedBranchScope !== initialBranchId) {
      setAppliedBranchScope(initialBranchId);
      setPreferences(value => ({ ...value, filters: { ...value.filters, sectorId: 'all', branchId: initialBranchId } }));
    }
  }, [initialBranchId, appliedBranchScope, setAppliedBranchScope]);
  useEffect(() => {
    if (selectedView && !preferences.savedViews.some(view => view.id === selectedView)) setSelectedView('');
    if (columnPreset && JSON.stringify(columns.map(({ id, kpiId, window, calculation }) => ({ id, kpiId, window, calculation }))) !== JSON.stringify(presetColumns(columnPreset))) setColumnPreset('');
  }, [columns, preferences.savedViews, selectedView, columnPreset, setSelectedView, setColumnPreset]);
  const pageCriteria = JSON.stringify([filters, rangeColumns, sort, features.secondarySorts, features.activeWatchlistId, pageSize]);
  const previousCriteria = useRef(pageCriteria);
  // A restored page survives remount/data loading; actual screening edits restart pagination.
  useEffect(() => { if (previousCriteria.current !== pageCriteria) { previousCriteria.current = pageCriteria; setPage(0); } }, [pageCriteria, setPage]);
  useEffect(() => {
    if (!saveEpoch) return;
    try { localStorage.setItem(COMPANY_LIST_STORAGE_KEY, JSON.stringify(preferences)); setStorageError(''); }
    catch { setStorageError('Your current list is available in this session, but its changes could not be saved locally.'); }
  }, [preferences, saveEpoch]);
  useEffect(() => {
    if (!modal) return;
    const previous = returnFocus.current;
    const root = dialog.current;
    (root?.querySelector<HTMLElement>('[data-dialog-autofocus]') ?? root)?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setModal(null); }
      if (event.key !== 'Tab' || !root) return;
      const focusable = [...root.querySelectorAll<HTMLElement>('button:not([disabled]),input:not([disabled]),select:not([disabled]),[tabindex="0"]')].filter(element => element.getClientRects().length > 0);
      const first = focusable[0], last = focusable.at(-1);
      if (!first) { event.preventDefault(); root.focus(); }
      else if (event.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || !root.contains(document.activeElement))) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey, true);
    return () => { document.removeEventListener('keydown', onKey, true); previous?.focus(); };
  }, [modal]);
  const chooseKpi = (kpiId: string) => {
    const kpi = kpiById.get(kpiId)!;
    setPickerKpi(kpiId); setEditingColumn(null); setPickerNotice('');
    const preferred = companyListVariants(kpi).find(variant => variant.window.endsWith(':last') && ['provider:latest', 'provider:default'].includes(variant.calculation));
    const nextWindow = kpi.windows.includes(pickerWindow) ? pickerWindow : preferred?.window ?? kpi.windows[0], calculations = calculationsFor(kpi, nextWindow);
    setPickerWindow(nextWindow);
    setPickerCalculation(calculations.includes(pickerCalculation) ? pickerCalculation : preferred && preferred.window === nextWindow ? preferred.calculation : calculations[0]);
  };
  const editColumn = (column: CompanyListColumn) => { setPickerKpi(column.kpiId); setPickerWindow(column.window); setPickerCalculation(column.calculation); setEditingColumn(column.id); setPickerNotice(''); };
  const addColumn = () => {
    setColumnPreset('');
    const duplicate = columns.find(column => column.id !== editingColumn && column.kpiId === pickerKpi && column.window === pickerWindow && column.calculation === pickerCalculation);
    if (duplicate) { setPickerNotice('This KPI and calculation are already in the list.'); return; }
    if (!editingColumn && columns.length >= COMPANY_LIST_MAX_COLUMNS) { setPickerNotice(`This view supports up to ${COMPANY_LIST_MAX_COLUMNS} KPI columns. Remove one to add another.`); return; }
    const column: CompanyListColumn = { id: editingColumn ?? id(), kpiId: pickerKpi, window: pickerWindow, calculation: pickerCalculation };
    change(value => ({ ...value, columns: editingColumn ? value.columns.map(existing => existing.id === editingColumn ? sameColumn(existing, column) ? { ...column, ...(existing.range ? { range: existing.range } : {}) } : column : existing) : [...value.columns, column] }));
    setPickerNotice(`${columnLabel(column)} ${editingColumn ? 'updated' : 'added'}.`); setEditingColumn(null);
  };
  const removeColumn = (columnId: string) => {
    setColumnPreset('');
    if (columns.length <= 1) { setPickerNotice('Keep at least one KPI column in the list.'); return; }
    change(value => ({ ...value, columns: value.columns.filter(column => column.id !== columnId), sort: value.sort.columnId === columnId ? { columnId: 'name', direction: 'asc' } : value.sort }));
    changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.filter(item => item.columnId !== columnId) }));
    if (editingColumn === columnId) setEditingColumn(null);
  };
  const moveColumn = (columnId: string, delta: number) => { setColumnPreset(''); change(value => {
    const next = [...value.columns], from = next.findIndex(column => column.id === columnId), to = from + delta;
    if (from < 0 || to < 0 || to >= next.length) return value;
    [next[from], next[to]] = [next[to], next[from]];
    return { ...value, columns: next };
  }); };
  const toggleWatchlist = (row: ResearchGaugeRow) => {
    changeFeatures(value => updateCompanyWatchlist(value, value.activeWatchlistId, row.id, !watchlist.has(row.id)));
  };
  const setSort = (columnId: string) => {
    change(value => ({ ...value, sort: { columnId, direction: value.sort.columnId === columnId && value.sort.direction === 'asc' ? 'desc' : 'asc' } }));
    changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.filter(item => item.columnId !== columnId) }));
  };
  const resetFilters = () => { setRuleDrafts({}); setUpperDrafts({}); change(value => ({ ...value, columns: value.columns.map(({ range, ...column }) => column), filters: { ...defaultCompanyListPreferences().filters, watchlistOnly: value.filters.watchlistOnly } })); };
  const saveView = () => {
    const name = viewName.trim().slice(0, 80);
    if (!name) return;
    const existing = preferences.savedViews.find(view => view.name.toLowerCase() === name.toLowerCase());
    if (!existing && preferences.savedViews.length >= 20) { setStatus('You can save up to 20 views. Delete a saved view to make room.'); return; }
    const view = { id: existing?.id ?? id(), name, columns, filters, sort };
    change(value => ({ ...value, savedViews: existing ? value.savedViews.map(item => item.id === existing.id ? view : item) : [...value.savedViews, view] }));
    changeFeatures(value => ({ ...value, viewFeatures: { ...value.viewFeatures, [view.id]: { secondarySorts: value.secondarySorts, density: value.density, activeWatchlistId: value.activeWatchlistId } } }));
    setSelectedView(view.id); setStatus(`“${name}” is available in Saved view.`); setModal(null);
  };
  const loadView = (viewId: string) => {
    setSelectedView(viewId); setColumnPreset('');
    setRuleDrafts({}); setUpperDrafts({});
    const view = preferences.savedViews.find(item => item.id === viewId);
    if (view) {
      change(value => ({ ...value, columns: view.columns, filters: view.filters, sort: view.sort }));
      changeFeatures(value => ({ ...value, ...(value.viewFeatures[viewId] ?? { secondarySorts: [] }) }));
      setStatus(`Loaded “${view.name}”.`);
    }
  };
  const updateRule = (index: number, patch: Partial<CompanyListNumericRule>) => change(value => ({ ...value, filters: { ...value.filters, numericRules: value.filters.numericRules.map((rule, position) => position === index ? { ...rule, ...patch } : rule) } }));
  const addRule = () => {
    const column = numericColumns.find(item => !monetary(item)) ?? numericColumns[0];
    if (column) changeFilters({ numericRules: [...filters.numericRules, { column, operator: 'gte', value: 0, ...(monetary(column) ? { currency: currencies[0] ?? '' } : {}) }] });
  };
  const saveWatchlist = () => {
    const name = watchlistName.trim().slice(0, 80);
    if (!name) return;
    if (features.watchlists.some(item => item.id !== watchlistEditId && item.name.toLowerCase() === name.toLowerCase())) { setStatus('Choose a different watchlist name.'); return; }
    if (!watchlistEditId && features.watchlists.length >= 20) { setStatus('You can keep up to 20 named watchlists.'); return; }
    const nextId = watchlistEditId ?? id();
    changeFeatures(value => ({ ...value, activeWatchlistId: nextId, watchlists: watchlistEditId ? value.watchlists.map(item => item.id === watchlistEditId ? { ...item, name } : item) : [...value.watchlists, { id: nextId, name, listingIds: [] }] }));
    setWatchlistName(''); setWatchlistEditId(null); setStatus(`“${name}” is selected. Use stars to add companies.`);
  };
  const exportCsv = async () => {
    if (exportBusy) return;
    setExportBusy(true); setExportError(''); setStatus('Preparing all matching listings for export…');
    try {
      await new Promise<void>(resolve => window.setTimeout(resolve, 0));
      const csv = companyListCsv(matches, columns, evaluate, columnLabel, column => {
        const metric = kpiById.get(column.kpiId);
        if (NORMAL_RANKING_KPIS.has(column.kpiId)) return `${NORMAL_RANKING_POLICY_ID}; 60% five-year cash-only NPV + 40% quality; ${qualityRanking?.cohortSize ?? 0} fixed priced reference listings; snapshot ${available?.asOf ?? 'unavailable'}; ${NORMAL_RANKING_REFERENCE}`;
        return metric?.source ? `${metric.source}; snapshot ${metric.snapshot ?? 'unavailable'}` : `Atlas saved financial/research data; snapshot ${available?.asOf ?? 'unavailable'}`;
      });
      const filename = `Macro-Atlas-Companies-${available?.asOf ?? 'saved'}.csv`;
      if ('__TAURI_INTERNALS__' in window) {
        const { invoke } = await import('@tauri-apps/api/core');
        const destination = await invoke<string>('export_csv', { filename, contents: csv });
        setStatus(`Exported all ${count(matches.length)} matching listings. Saved to ${destination}`);
      } else {
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
        const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
        setStatus(`Exported all ${count(matches.length)} matching listings with dates, units and source details.`);
      }
    } catch (error) {
      setStatus(''); setExportError(`The company list could not be exported. ${error instanceof Error ? error.message : String(error)}`);
    } finally { setExportBusy(false); }
  };
  const applyColumnPreset = (presetId: string) => {
    setColumnPreset(presetId); const next = presetColumns(presetId); if (!next.length) return;
    const ranking = presetId === 'normal_quality_value' ? next.find(column => column.kpiId === 'normal_quality_value_score') : presetId === 'normal_years' ? next.find(column => column.kpiId === 'normal_npv_percent' && column.calculation === 'terminal_50') : ['valuation_rank', 'terminal_sensitivity'].includes(presetId) ? next.find(column => column.kpiId === 'valuation_attractiveness') : null;
    change(value => ({ ...value, columns: next, sort: { columnId: ranking?.id ?? 'name', direction: ranking ? 'desc' : 'asc' } }));
    changeFeatures(value => ({ ...value, secondarySorts: [] }));
    setStatus(ranking ? `${presetId === 'normal_quality_value' ? 'Quality/value rank' : 'Valuation attractiveness'} sorted high to low. Column ranges cleared; other filters and watchlists are kept.` : 'Column preset applied. Column ranges cleared; other filters and watchlists are kept.');
  };
  const toggleComparison = (listingId: string) => changeFeatures(value => toggleCompanyComparison(value, listingId));
  const openCell = (row: ResearchGaugeRow, column: CompanyListColumn) => { setCellSelection({ row, column }); openModal('cell'); };
  const sortIcon = (columnId: string) => sort.columnId !== columnId ? <ArrowUpDown size={12} /> : sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />;

  const toggleToolPanel = (panel: 'columns' | 'tools') => { setToolPanel(current => current === panel ? null : panel); setShowFilters(false); };
  const removeRule = (index: number) => { setRuleDrafts({}); setUpperDrafts({}); changeFilters({ numericRules: filters.numericRules.filter((_, position) => position !== index) }); };
  const activeFilters: { key: string; label: string; detail?: string; invalid?: boolean; remove: () => void }[] = [];
  if (filters.query.trim()) activeFilters.push({ key: 'query', label: `Search: ${filters.query}`, remove: () => changeFilters({ query: '' }) });
  if (filters.watchlistOnly) activeFilters.push({ key: 'watchlist', label: `Watchlist: ${activeList.name}`, remove: () => changeFilters({ watchlistOnly: false }) });
  if (filters.preset !== 'all') activeFilters.push({ key: 'preset', label: presets[filters.preset], remove: () => changeFilters({ preset: 'all' }) });
  if (filters.country !== 'all') activeFilters.push({ key: 'country', label: countryLabel(filters.country), remove: () => changeFilters({ country: 'all' }) });
  if (filters.sectorId !== 'all') activeFilters.push({ key: 'sector', label: `Sector: ${sectors.find(([value]) => value === filters.sectorId)?.[1] ?? filters.sectorId}`, remove: () => changeFilters({ sectorId: 'all' }) });
  if (filters.branchId !== 'all') activeFilters.push({ key: 'branch', label: `Industry: ${branches.find(([value]) => value === filters.branchId)?.[1] ?? filters.branchId}`, remove: () => changeFilters({ branchId: 'all' }) });
  if (filters.readiness !== 'all') activeFilters.push({ key: 'readiness', label: researchReadinessLabels[filters.readiness], remove: () => changeFilters({ readiness: 'all' }) });
  if (filters.route !== 'all') activeFilters.push({ key: 'route', label: researchRouteLabels[filters.route], remove: () => changeFilters({ route: 'all' }) });
  if (filters.presence !== 'all') activeFilters.push({ key: 'presence', label: filters.presence === 'latest' ? 'Newest directory' : 'Older directory only', remove: () => changeFilters({ presence: 'all' }) });
  for (const column of rangeColumns) {
    const range = column.range!, min = range.min.trim(), max = range.max.trim();
    const bounds = min && max ? `${min} to ${max}` : min ? `≥ ${min}` : `≤ ${max}`;
    const issue = companyListRangeError(range, companyListUnit(column));
    activeFilters.push({ key: `range-${column.id}`, label: `${headingLabel(column)}: ${bounds}${range.currency ? ` ${range.currency}` : ''}`, detail: `${columnLabel(column)} · ${unitLabel(column)}${issue ? `. ${issue}` : ''}`, invalid: !!issue, remove: () => changeRange(column.id, undefined) });
  }
  for (const [index, rule] of filters.numericRules.entries()) {
    const operator = { gte: '≥', lte: '≤', gt: '>', lt: '<', eq: '=', between: 'between', present: 'has a value', missing: 'unavailable' }[rule.operator];
    const bounded = !['present', 'missing'].includes(rule.operator);
    const bounds = bounded ? ` ${ruleDrafts[index] ?? rule.value ?? '…'}${rule.operator === 'between' ? ` and ${upperDrafts[index] ?? rule.valueTo ?? '…'}` : ''}${rule.currency ? ` ${rule.currency}` : ''}` : '';
    activeFilters.push({ key: `condition-${index}`, label: `${headingLabel(rule.column)} ${operator}${bounds}`, detail: `Condition ${index + 1}: ${columnLabel(rule.column)} · ${unitLabel(rule.column)}`, invalid: bounded && (rule.value == null || rule.operator === 'between' && (rule.valueTo == null || rule.valueTo < rule.value)), remove: () => removeRule(index) });
  }
  const shortSort = `${sortColumn ? headingLabel(sortColumn) : 'Company name'} · ${sort.direction === 'asc' ? sortColumn ? 'low to high' : 'A–Z' : sortColumn ? 'high to low' : 'Z–A'}${features.secondarySorts.length ? ` + ${features.secondarySorts.length} more` : ''}${sortColumn && monetary(sortColumn) ? ' · within each currency' : ''}`;

  return <main className="company-list-workspace" aria-label="Company lists workspace" data-company-list-ready={!!available} data-company-list-total={rows.length} data-company-list-matches={matches.length} data-density={features.density}>
    <div inert={modal ? true : undefined}>
      <header className="company-list-heading">
        <div className="company-list-heading-main"><h1>Company lists</h1>{available && <p role="status"><strong>{count(matches.length)}</strong> matching{filters.watchlistOnly ? ' watchlist' : ''} listings <span>of {count(rows.length)} saved</span></p>}</div>
        <div className="company-list-heading-side">{available && <span className="company-list-snapshot">Research snapshot <strong>{available.asOf}</strong></span>}{initialBranchId && <button className="company-list-back" onClick={onBack}><ArrowLeft size={14} />Branch overview</button>}</div>
      </header>
      {storageError && <p className="company-list-notice" role="status" data-company-list-storage-error>{storageError}</p>}
      {!ready ? <div className="company-list-loading" role="status">Opening the saved company universe…</div> : error ? <div className="company-list-loading" role="alert">{error}</div> : !available ? <div className="company-list-loading">Choose the matching financial pack and company directory in the Library to open company lists.</div> : <>
        <section className="company-list-panel" aria-label="Customizable company list">
          <div className="company-list-controls">
            <div className="company-list-primary-toolbar">
              <label className="company-list-main-search"><span>Find a company</span><span className="company-list-search"><Search size={17} /><input aria-label="Company list search" placeholder="Company name, ticker or ISIN" value={filters.query} onChange={event => changeFilters({ query: event.target.value })} /></span></label>
              <div className="company-list-tabs" aria-label="Company list scope"><button aria-label="All listings" aria-pressed={!filters.watchlistOnly} onClick={() => changeFilters({ watchlistOnly: false })}>All listings</button><button aria-label="Watchlist" aria-pressed={filters.watchlistOnly} onClick={() => changeFilters({ watchlistOnly: true })}><Star size={13} />Watchlist <span>{count(visibleWatchlistCount)}</span></button></div>
              <label className="company-list-watchlist-control">Watchlist<select aria-label="Active watchlist" value={features.activeWatchlistId} onChange={event => changeFeatures(value => ({ ...value, activeWatchlistId: event.target.value }))}>{features.watchlists.map(item => <option key={item.id} value={item.id}>{item.name} · {count(item.listingIds.length)}</option>)}</select></label>
              <label className="company-list-current-view"><span>Saved view{savedViewEdited && <span className="company-list-view-edited"> · edited</span>}</span><select aria-label="Saved company list view" value={selectedView} onChange={event => loadView(event.target.value)}><option value="">Current view</option>{preferences.savedViews.map(view => <option key={view.id} value={view.id}>{view.name}</option>)}</select></label>
              <button className="company-list-button company-list-button-primary company-list-columns-action" aria-label="Choose KPI columns" onClick={() => { setPickerNotice(''); openModal('columns'); }}><Columns3 size={15} />Columns <span>{columns.length}</span></button>
            </div>
            <div className="company-list-actionbar">
              <div className="company-list-action-group"><button className="company-list-button" aria-label="Show list filters" aria-controls="company-list-filters" aria-expanded={showFilters} onClick={() => { setShowFilters(!showFilters); setToolPanel(null); }}><SlidersHorizontal size={15} />Screening filters{filterCount + Number(filters.preset !== 'all') > 0 && ` · ${filterCount + Number(filters.preset !== 'all')}`}<ChevronDown size={13} /></button><button className="company-list-button" aria-label="Show column sets" aria-controls="company-list-column-sets" aria-expanded={toolPanel === 'columns'} onClick={() => toggleToolPanel('columns')}><Columns3 size={14} />Column sets<ChevronDown size={13} /></button><button className="company-list-button" aria-label="Show list tools" aria-controls="company-list-tools" aria-expanded={toolPanel === 'tools'} title="Sorting, density, watchlist management and help" onClick={() => toggleToolPanel('tools')}><Settings2 size={14} />View tools<ChevronDown size={13} /></button></div>
              <div className="company-list-action-group company-list-result-actions"><button className="company-list-button" aria-label="Compare selected companies" disabled={comparisonRows.length < 2} onClick={() => openModal('compare')}><GitCompareArrows size={14} />Compare · {comparisonRows.length}/8</button>{comparisons.size > 0 && <button className="company-list-text-button" onClick={() => changeFeatures(value => ({ ...value, comparisonIds: [] }))}>Clear selection</button>}<button className="company-list-button" aria-label="Save view" onClick={() => { setViewName(preferences.savedViews.find(view => view.id === selectedView)?.name ?? ''); openModal('save'); }}><Save size={14} />Save view</button><button className="company-list-button" aria-label="Export company list CSV" aria-busy={exportBusy} disabled={exportBusy || !matches.length || expanded.loading.size > 0} onClick={exportCsv}><Download size={14} />{exportBusy ? 'Exporting…' : 'Export CSV'}</button></div>
            </div>
            {toolPanel === 'columns' && <section id="company-list-column-sets" className="company-list-disclosure" aria-label="Company list column sets"><div className="company-list-disclosure-heading"><div><h2>Choose a starting column set</h2><p>A set replaces the visible KPIs and clears their Min/Max ranges. Your other screening filters and watchlists stay active.</p></div><button className="company-list-icon" aria-label="Close column sets" onClick={() => setToolPanel(null)}><X size={16} /></button></div><div className="company-list-set-selector"><label>Column set<select aria-label="Column preset" value={columnPreset} onChange={event => { applyColumnPreset(event.target.value); setToolPanel(null); }}><option value="">Custom columns</option>{columnPresetDefinitions.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><p>Then use <strong>Columns</strong> to add, change or reorder individual measures.</p></div><div className="company-list-set-options">{[{ id: 'terminal_sensitivity', title: 'Compare cash scenarios', detail: 'Full DCF, half terminal and cash-only surplus side by side.' }, { id: 'quality', title: 'Understand profitability', detail: 'Margins, returns and the consistency of past cash flow.' }, { id: 'strength', title: 'Examine the balance sheet', detail: 'Debt, cash and other financial-strength measures.' }].map(item => <button key={item.id} className="company-list-set-card" onClick={() => { applyColumnPreset(item.id); setToolPanel(null); }}><strong>{item.title}</strong><span>{item.detail}</span><span className="company-list-set-link">Use this set <ArrowRight size={13} /></span></button>)}</div></section>}
            {toolPanel === 'tools' && <section id="company-list-tools" className="company-list-disclosure" aria-label="Company list view tools"><div className="company-list-disclosure-heading"><div><h2>View tools</h2><p>Adjust the presentation and manage your saved research lists.</p></div><button className="company-list-icon" aria-label="Close list tools" onClick={() => setToolPanel(null)}><X size={16} /></button></div><div className="company-list-tool-options"><label>Row spacing<select aria-label="Table density" value={features.density} onChange={event => changeFeatures(value => ({ ...value, density: event.target.value as CompanyListFeatures['density'] }))}><option value="compact">Compact</option><option value="comfortable">Comfortable</option></select></label><button className="company-list-button" aria-label="Manage watchlists" onClick={() => { setWatchlistName(''); setWatchlistEditId(null); openModal('watchlists'); }}><Star size={14} />Manage watchlists</button>{selectedView && <button className="company-list-button" aria-label="Delete selected saved view" onClick={() => { change(value => ({ ...value, savedViews: value.savedViews.filter(view => view.id !== selectedView) })); changeFeatures(value => ({ ...value, viewFeatures: Object.fromEntries(Object.entries(value.viewFeatures).filter(([key]) => key !== selectedView)) })); setSelectedView(''); setStatus('Saved view deleted. Your current columns and watchlist are kept.'); }}><Trash2 size={14} />Delete selected view</button>}</div>
            <details className="company-list-sort-controls"><summary>Sort priorities · {1 + features.secondarySorts.length} of 3</summary><div className="company-list-sort-rules">
              {[sort, ...features.secondarySorts].map((entry, index) => <div key={index} className="company-list-sort-rule"><span>{index === 0 ? 'First' : 'Then'}</span><select aria-label={`Sort priority ${index + 1}`} value={entry.columnId} onChange={event => { if (index === 0) setSort(event.target.value); else changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.map((item, position) => position === index - 1 ? { ...item, columnId: event.target.value } : item) })); }}><option value="name" disabled={[sort, ...features.secondarySorts].some((item, position) => position !== index && item.columnId === 'name')}>Company name</option>{columns.map(column => <option key={column.id} value={column.id} disabled={[sort, ...features.secondarySorts].some((item, position) => position !== index && item.columnId === column.id)}>{columnLabel(column)}</option>)}</select><select aria-label={`Sort direction ${index + 1}`} value={entry.direction} onChange={event => { const direction = event.target.value as 'asc' | 'desc'; if (index === 0) change(value => ({ ...value, sort: { ...value.sort, direction } })); else changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.map((item, position) => position === index - 1 ? { ...item, direction } : item) })); }}><option value="asc">Ascending</option><option value="desc">Descending</option></select>{index > 0 && <button className="company-list-icon" aria-label={`Remove sort priority ${index + 1}`} onClick={() => changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.filter((_, position) => position !== index - 1) }))}><X size={13} /></button>}</div>)}
              <button className="company-list-button" disabled={features.secondarySorts.length >= 2} onClick={() => { const next = columns.find(column => column.id !== sort.columnId && !features.secondarySorts.some(item => item.columnId === column.id)); if (next) changeFeatures(value => ({ ...value, secondarySorts: [...value.secondarySorts, { columnId: next.id, direction: 'asc' }] })); }}><Plus size={13} />Add sort priority</button>
            </div></details>
        <details className="company-list-introduction"><summary>How Lists works: from a screen to a company review</summary><ol><li><strong>Choose KPIs</strong> adds the measures you want. A <strong>Column set</strong> selects a ready-made group; a <strong>Starting filter</strong> applies stated cash-history criteria.</li><li>Enter Min and Max under a heading. Click its information button to understand the measure, units and calculation. Clear a bound to broaden the list.</li><li>Open a company name for financial charts, then move to Valuation. Use stars for your watchlist; Save view keeps the current columns and filters.</li></ol></details>
            </section>}
            {showFilters && <div id="company-list-filters" className="company-list-filter-panel" aria-label="Company list filters">
              <div className="company-list-disclosure-heading"><div><h2>Screening filters</h2><p>Start with a cash-history screen, narrow the universe, or add specific conditions. Every active filter must match.</p></div><button className="company-list-icon" aria-label="Close list filters" onClick={() => setShowFilters(false)}><X size={16} /></button></div>
              <div className="company-list-filter-grid">
              <label className="company-list-starting-filter">Starting filter<select aria-label="Company list preset" value={filters.preset} onChange={event => changeFilters({ preset: event.target.value as CompanyListFilters['preset'] })}>{Object.entries(presets).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Listing country<select aria-label="Company list country" value={filters.country} onChange={event => changeFilters({ country: event.target.value })}><option value="all">All countries</option>{countries.map(value => <option key={value} value={value}>{countryLabel(value)}</option>)}</select></label>
              <label>Sector<select aria-label="Company list sector" value={filters.sectorId} onChange={event => changeFilters({ sectorId: event.target.value, branchId: 'all' })}><option value="all">All sectors</option>{sectors.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Industry / branch<select aria-label="Company list branch" value={filters.branchId} onChange={event => changeFilters({ branchId: event.target.value })}><option value="all">All branches</option>{branches.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Annual evidence<select aria-label="Company list coverage" value={filters.readiness} onChange={event => changeFilters({ readiness: event.target.value as CompanyListFilters['readiness'] })}><option value="all">All coverage</option>{Object.entries(researchReadinessLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Business research route<select aria-label="Company list route" value={filters.route} onChange={event => changeFilters({ route: event.target.value as CompanyListFilters['route'] })}><option value="all">All business routes</option>{Object.entries(researchRouteLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Directory coverage<select aria-label="Company list presence" value={filters.presence} onChange={event => changeFilters({ presence: event.target.value as CompanyListFilters['presence'] })}><option value="all">All saved listings</option><option value="latest">Newest directory download</option><option value="older">Older download only</option></select></label>
            </div>{filters.preset !== 'all' && <p className="company-list-preset-note" data-company-list-preset-note><strong>{presets[filters.preset]}:</strong> {filters.preset === 'normal_quality' ? 'operating businesses in the newest directory, no classification conflict, and five comparable annual reports with known publication dates outside fiscal end years 2020–2023. The latest annual report must end after 2023 and within 550 days of the saved snapshot. Add quality and valuation ranges separately.' : 'operating businesses with five-period evidence, no classification conflict, and positive provider FCF and EBIT in all five annual periods.'}{filters.preset === 'cash_and_margin' && ' Also requires positive saved equity price and starter value, with price at or below 70% of Mid DCF value.'} These saved observations are a starting point for review.</p>}<div className="company-list-numeric-heading"><h3>Numeric conditions · all must match</h3><button className="company-list-button" aria-label="Add numeric KPI filter" disabled={filters.numericRules.length >= 12 || !numericColumns.length} onClick={addRule}><Plus size={13} />Add condition</button></div><div className="company-list-numeric-rules">{filters.numericRules.map((rule, index) => {
              const needsBound = rule.operator !== 'present' && rule.operator !== 'missing';
              const unit = companyListUnit(rule.column);
              return <div className="company-list-rule" key={index} data-company-list-rule={index} data-rule-between={rule.operator === 'between'}>
                <select aria-label={`KPI for condition ${index + 1}`} value={numericColumns.some(column => sameColumn(column, rule.column)) ? rule.column.id : `retained-${index}`} onChange={event => { const column = columns.find(item => item.id === event.target.value); if (column) updateRule(index, { column, currency: monetary(column) ? currencies[0] ?? '' : undefined }); }}>
                  {!numericColumns.some(column => sameColumn(column, rule.column)) && <option value={`retained-${index}`}>{columnLabel(rule.column)} · retained condition</option>}{numericColumns.map(column => <option key={column.id} value={column.id}>{columnLabel(column)}</option>)}
                </select>
                <select aria-label={`Operator for condition ${index + 1}`} value={rule.operator} onChange={event => updateRule(index, { operator: event.target.value as CompanyListNumericRule['operator'] })}>
                  <option value="gte">At least ≥</option><option value="lte">At most ≤</option><option value="gt">Above &gt;</option><option value="lt">Below &lt;</option><option value="eq">Equals =</option><option value="between">Between, inclusive</option><option value="present">Has a value</option><option value="missing">Value unavailable</option>
                </select>
                {needsBound ? <div className="company-list-rule-bounds"><input aria-label={`Value for condition ${index + 1}`} type="text" inputMode="decimal" aria-invalid={rule.value === null} value={ruleDrafts[index] ?? (rule.value === null ? '' : String(rule.value))} onChange={event => { const raw = event.target.value; setRuleDrafts(current => ({ ...current, [index]: raw })); const value = raw.trim() ? Number(raw) : NaN; updateRule(index, { value: Number.isFinite(value) ? value : null }); }} />{rule.operator === 'between' && <><span>to</span><input aria-label={`Upper value for condition ${index + 1}`} type="text" inputMode="decimal" aria-invalid={rule.valueTo == null || rule.value !== null && rule.valueTo < rule.value} value={upperDrafts[index] ?? (rule.valueTo == null ? '' : String(rule.valueTo))} onChange={event => { const raw = event.target.value; setUpperDrafts(current => ({ ...current, [index]: raw })); const value = raw.trim() ? Number(raw) : NaN; updateRule(index, { valueTo: Number.isFinite(value) ? value : null }); }} /></>}</div> : <span className="company-list-rule-unit">Observed data only</span>}
                {needsBound && monetary(rule.column) ? <select className="company-list-currency" aria-label={`Currency for condition ${index + 1}`} value={rule.currency ?? ''} onChange={event => updateRule(index, { currency: event.target.value })}><option value="">Select currency</option>{currencies.map(value => <option key={value}>{value}</option>)}</select> : <span className="company-list-rule-unit">{needsBound ? unit === 'percent' ? '%' : unit === 'multiple' ? '×' : unit === 'points' ? 'pp' : unit === 'shares_millions' ? 'million shares' : rule.column.kpiId === 'price_age' ? 'days' : 'count' : ''}</span>}
                <button className="company-list-icon" aria-label={`Remove condition ${index + 1}`} onClick={() => { setRuleDrafts({}); setUpperDrafts({}); changeFilters({ numericRules: filters.numericRules.filter((_, position) => position !== index) }); }}><X size={13} /></button>
              </div>;
            })}</div><div className="company-list-filter-actions"><p>{filters.numericRules.some(rule => !['present', 'missing'].includes(rule.operator) && (rule.value === null || rule.operator === 'between' && (rule.valueTo == null || rule.valueTo < rule.value))) ? 'Enter a number for each condition. An incomplete condition matches no listings.' : filters.numericRules.length ? 'Missing values match only “Value unavailable”. Loading or failed datasets match no condition. Monetary conditions use the displayed units and selected currency.' : 'Add a condition using one of your numeric KPI columns. Missing values stay unavailable.'}</p><button className="company-list-text-button" onClick={resetFilters}>Clear list filters</button></div></div>}
            {exportError && <p className="company-list-notice" role="alert">{exportError}</p>}
            {expanded.error && <p className="company-list-notice" role="status">{expanded.error}</p>}
            {expanded.errors.size > 0 && <p className="company-list-notice" role="status">{expanded.errors.size} selected KPI datasets could not be opened. Their values and filter matches remain unavailable; click a cell for details.</p>}
            {expanded.loading.size > 0 && <p className="company-list-status" role="status">Opening {expanded.loading.size} selected KPI datasets…</p>}
          </div>
          {activeFilters.length > 0 && <div className="company-list-active-filters" aria-label="Active company list filters"><span className="company-list-active-label">Active · {activeFilters.length}</span><div className="company-list-filter-chips">{activeFilters.map(filter => <button key={filter.key} className="company-list-filter-chip" data-filter-key={filter.key} data-invalid={filter.invalid || undefined} title={filter.detail ?? filter.label} aria-label={`Remove ${filter.key.startsWith('range-') ? 'range' : filter.key.startsWith('condition-') ? 'condition' : 'filter'}: ${filter.label}`} onClick={filter.remove}><span>{filter.label}</span>{filter.invalid && <Info size={13} aria-label="Needs attention" />}<X size={13} /></button>)}</div><div className="company-list-clear-filters">{rangeColumns.length > 0 && <button className="company-list-text-button" onClick={() => change(value => ({ ...value, columns: value.columns.map(({ range, ...column }) => column) }))}>Clear KPI ranges · {rangeColumns.length}</button>}{activeFilters.some(filter => filter.key !== 'watchlist') && <button className="company-list-text-button" aria-label="Clear active list filters" onClick={resetFilters}>Clear list filters</button>}</div></div>}
          {hasNormalYears && <div className="company-list-range-note" data-company-list-normal-year-policy><span><strong>Exception years: 2020–2023.</strong> These fiscal end years are excluded from the selected normal-year measures, including both weak and strong years. Five eligible annual reports are required; each value shows the periods used. Original history and other columns keep their stated periods.</span></div>}
          {hasNormalValuation && <div className="company-list-range-note" data-company-list-normal-valuation-policy><span><strong>Ten-year normal-year comparison:</strong> median provider FCF from the five selected reports, held flat for 10 years; 10% discount rate and 0% terminal growth. Full, half-terminal and cash-only surplus compare this separate proxy with the dated saved equity price. Provider FCF needs owner-cash reconciliation; authored valuations and starter calibration are unchanged.</span></div>}
          {hasFiveYearValuation && <div className="company-list-range-note" data-company-list-five-year-valuation-policy><span><strong>Five-year cash-only NPV:</strong> median FCF from the five normal reports, held flat for years 1–5 and discounted at 10%; no terminal value. NPV subtracts the dated saved equity price; NPV / price makes listings comparable. Negative NPV means these five discounted cash flows do not cover that price. Provider FCF still needs owner-cash reconciliation.</span></div>}
          {hasQualityRanking && <div className="company-list-range-note" data-company-list-quality-value-policy data-quality-value-ready={qualityRanking?.status === 'available'} data-quality-value-peers={qualityRanking?.cohortSize ?? 0}><span><strong>Ranking: 60% five-year NPV + 40% quality.</strong> Years 1–5 only, discounted at 10%, with no terminal value. {qualityRanking?.status === 'available' ? `${count(qualityRanking.cohortSize)} priced listings in the fixed quality reference. Searches and filters keep the same scores; missing inputs stay unranked and sort last. High rank does not establish undervaluation.` : qualityRanking?.reason ?? 'Opening the fixed reference…'}<details><summary>Ranking method and reference</summary><p>Value ranks the five-year cash-only NPV / saved price, with no terminal value. Quality equally weights median pre-tax capital return, minimum EBIT margin, CFO CAGR and lower net debt / EBITDA (net cash scores like zero debt). Each quality component contributes 10% of the total. Equal values share percentile ranks. These are 0–100 rank points, not expected returns or proof of pricing power.</p><p>With the same flat-cash assumption and discount rate for every listing, five-year NPV preserves the previous valuation order. The larger 60% valuation weight changes the combined ranking. Ten-year comparisons are shown separately and are not counted again.</p><p>{NORMAL_RANKING_REFERENCE}</p></details></span></div>}
          <div className="company-list-range-note company-list-result-bar"><span>Min / Max: blank means any. All active ranges must match.</span><details className="company-list-reading-guide"><summary><Info size={13} />Reading this list</summary><div><p>Bounds include the entered value and use each KPI’s displayed units. Missing values are excluded only when a range or condition requires a value. Click a heading’s information button for its definition, or a value for its source. Stars save companies; checkboxes select up to eight for comparison.</p>{hasStarterValuationRanking && <p data-company-list-valuation-ranking-note>Valuation surplus compares a saved scenario with its dated price. A 30% discount to value requires at least +42.86% surplus. Full DCF includes terminal value once and equals Mid NPV / price. The comparisons are starter scenarios, not expected annual returns or business-quality scores; reviewed and edited valuations remain in Valuation.</p>}{requestedColumns.some(column => column.kpiId.startsWith('provider_')) && <p data-company-list-provider-snapshot>Provider KPIs downloaded {EXPANDED_KPI_MANIFEST.snapshot}. Underlying report and quote dates may be unavailable. Starter valuations retain their own saved inputs.</p>}<p>Monetary sorting groups currencies A–Z, then sorts within each currency. Missing values appear last. Separate listings may represent the same company.</p></div></details><span className="company-list-sort-summary" data-company-list-sort-status title={`Sorted by ${sortDescription} · missing values last`}>Sorted by {shortSort}</span></div>
          {<div className="company-list-table-scroll" role="region" aria-label="Company list results" tabIndex={0}><table className="company-list-table"><colgroup><col style={{ width: 230 }} />{columns.map(column => <col key={column.id} style={{ width: companyListUnit(column) === 'text' ? 165 : 175 }} />)}</colgroup><thead><tr><th scope="col" aria-sort={sort.columnId === 'name' ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'}><button aria-label="Sort by company name" onClick={() => setSort('name')}><span className="company-list-column-name">Company {sortIcon('name')}</span><small>Ticker · country · Börsdata ID</small></button></th>{columns.map(column => <th key={column.id} scope="col" data-company-list-header={column.id} data-company-list-column={column.id} data-company-list-kpi={column.kpiId} aria-sort={sort.columnId === column.id ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'}><button aria-label={`Sort by ${columnLabel(column)}`} title={kpiById.get(column.kpiId)?.description} onClick={() => setSort(column.id)}><span className="company-list-column-name">{headingLabel(column)}{sortIcon(column.id)}</span><small title={`${companyListWindowLabel(column)} · ${calculationLabel(column)}`}>{headingContext(column)}</small></button><button type="button" className="company-list-header-help" aria-label={`Explain ${columnLabel(column)}`} title="Definition, calculation and units" onClick={() => { setHelpColumn(column); openModal('help'); }}><Info size={14} /></button><CompanyListColumnRange column={column} currencies={currencies} onChange={range => changeRange(column.id, range)} /></th>)}</tr></thead><tbody>{visibleRows.map(row => <tr key={row.id} data-company-listing={row.id}><td><div className="company-list-company-cell"><input className="company-list-compare-check" type="checkbox" aria-label={`Compare ${row.name}`} checked={comparisons.has(row.id)} disabled={!comparisons.has(row.id) && comparisons.size >= 8} onChange={() => toggleComparison(row.id)} /><button className="company-list-star" aria-label={`${watchlist.has(row.id) ? 'Remove' : 'Add'} ${row.name} ${watchlist.has(row.id) ? 'from' : 'to'} watchlist`} aria-pressed={watchlist.has(row.id)} onClick={() => toggleWatchlist(row)}><Star size={15} /></button><div className="company-list-company-text"><button className="company-list-company" title={row.name} aria-label={`Open profile for ${row.name}`} onClick={() => onCompany(row.id)}>{row.name}</button><small>{row.ticker ?? '—'} · {row.country ?? '—'} · {row.id}</small>{(filters.preset === 'cash_and_margin' || hasValuationRanking) && <small data-company-list-candidate-price-date={row.valuation.priceDate ?? ''}>Saved price {row.valuation.priceDate ?? 'unavailable'}</small>}</div></div></td>{columns.map(column => { const cell = evaluate(row, column); return <td key={column.id} title={`${cell.display}. ${cell.detail}`} onDoubleClick={() => openCell(row, column)} aria-label={`${columnLabel(column)}: ${cell.display}. ${cell.detail}`} data-company-list-kpi={column.kpiId} data-company-list-column={column.id} data-company-list-value={cell.value ?? ''} data-company-list-currency={cell.currency ?? ''} data-company-list-date={cell.date ?? ''} data-value-kind={typeof cell.value === 'string' ? 'text' : cell.value === null ? 'missing' : 'number'}><button type="button" className={`company-list-cell-value company-list-cell-inspect${cell.value === null ? ' company-list-cell-missing' : ''}`} onClick={() => openCell(row, column)} aria-label={cell.value === null ? `${kpiById.get(column.kpiId)?.label} unavailable: ${cell.detail}` : undefined}>{cell.display}</button>{cell.date && companyListUnit(column) !== 'date' && <small className="company-list-cell-date">{cell.date}</small>}</td>; })}</tr>)}{!visibleRows.length && <tr><td colSpan={columns.length + 1}><div className="company-list-empty"><strong>{filters.watchlistOnly && !visibleWatchlistCount ? 'Your watchlist is ready for companies.' : 'No listings match this view.'}</strong>{filters.watchlistOnly && !visibleWatchlistCount ? 'Use the star beside a company in All listings to keep it here.' : 'Adjust the conditions or clear your filters to broaden the list.'}<div><button className="company-list-button" onClick={filters.watchlistOnly && !visibleWatchlistCount ? () => changeFilters({ watchlistOnly: false }) : resetFilters}>{filters.watchlistOnly && !visibleWatchlistCount ? 'Browse all listings' : 'Clear list filters'}</button></div></div></td></tr>}</tbody></table></div>}
          <div className="company-list-pagination"><span>{matches.length ? `${count(selectedPage * pageSize + 1)}–${count(Math.min((selectedPage + 1) * pageSize, matches.length))}` : '0'} of {count(matches.length)}</span><div><select aria-label="Company list rows per page" value={pageSize} onChange={event => setPageSize(Number(event.target.value))}><option value={25}>25 rows</option><option value={50}>50 rows</option><option value={100}>100 rows</option><option value={250}>250 rows</option></select><button className="company-list-icon" aria-label="Previous company list page" disabled={!selectedPage} onClick={() => setPage(selectedPage - 1)}><ArrowLeft size={14} /></button><span>{selectedPage + 1} / {pages}</span><button className="company-list-icon" aria-label="Next company list page" disabled={selectedPage + 1 >= pages} onClick={() => setPage(selectedPage + 1)}><ArrowRight size={14} /></button></div></div>
        </section>
        <p className="company-list-status" role="status">{status}</p>
        <details className="company-list-method"><summary>About KPI periods, saved prices and candidate screens</summary><p>Each value uses the validated data currently bundled in Atlas. “Latest saved” means the latest eligible report or saved quote, with its own date. Annual windows require the requested comparable periods. An em dash means a value or valid comparison is unavailable; click the cell for its reason. Money is shown in millions of the stated currency, and prices per share use the stated quote currency.</p><p>Columns can be sorted and filtered. Monetary sorting groups currencies A–Z, then sorts within each currency; it does not compare converted amounts. Named views save columns, their Min/Max ranges, filters and sorting. Changing a column’s KPI, period or calculation clears its range. Removing a column or applying a column preset removes its attached range; separately added numeric conditions remain active. Stars add companies to the selected named watchlist. Saved views also retain the active watchlist, up to three sorting priorities and the table density. Separate listings or share classes may represent the same company.</p><p>The historical screens apply their written criteria only. Five-period evidence requires five valid annual observations for FCF, operating cash, EBIT and revenue, known publication dates, and an annual end within 550 days of the saved snapshot. These observations do not establish future cash or business quality. Companies with thinner history and financial businesses remain in All listings.</p><p>The optional normal-year measures select five eligible reports after excluding fiscal end years 2020–2023. Growth uses elapsed calendar time between the selected endpoints, including the excluded years. This symmetric exception policy removes strong and weak years alike; inspect the full history and capital-return definitions before judging quality. Normal-year screening values use a separate median provider-FCF assumption and leave the all-year starter and authored valuations intact.</p><p>Starter DCF, terminal value and NPV use one connected calculation. The saved 30% purchase ceiling is 0.70 × positive Mid equity value. It is separate from reviewed or edited company valuations. Saved whole-equity price uses reported shares and a dated close; the share basis remains an unreviewed proxy. Provider FCF needs reconciliation to sustainable owner cash during a deep dive.</p><p>Provider KPIs use their saved source dates, definitions and supported period/calculation pairs. A current or R12 provider snapshot is distinct from an annual-report calculation. Imported provider prices and valuation ratios do not update the frozen starter DCF, NPV or purchase ceiling. Click a value to inspect its source and calculation.</p></details>
      </>}
    </div>
    {modal && <div className="company-list-modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setModal(null); }}><div ref={dialog} className={`company-list-dialog${modal === 'save' || modal === 'cell' || modal === 'watchlists' || modal === 'help' ? ' company-list-dialog-save' : ''}`} role="dialog" aria-modal="true" aria-label={{ columns: 'Choose KPI columns', save: 'Save company list view', watchlists: 'Manage watchlists', compare: 'Compare companies', cell: 'KPI value details', help: 'KPI explained' }[modal]} tabIndex={-1}>
      <div className="company-list-dialog-header"><div><h2>{{ columns: 'Choose your KPI columns', save: 'Save this view', watchlists: 'Your watchlists', compare: 'Compare selected companies', cell: 'Value and source', help: 'KPI explained' }[modal]}</h2><p>{{ columns: `${usableKpiCount} KPIs and fields with saved values · ${COMPANY_KPIS.length} in the catalogue. Choose a period and calculation.`, save: 'Keep columns, filters, sorting and watchlist together.', watchlists: 'Keep separate lists for different research ideas.', compare: `${comparisonRows.length} selected listings · your current columns`, cell: 'Inspect the observation behind this number.', help: 'Understand the measure before using it to filter or compare.' }[modal]}</p></div><button className="company-list-icon" aria-label="Close company list dialog" onClick={() => setModal(null)}><X size={17} /></button></div>
      {modal === 'columns' ? <><div className="company-list-picker-search"><span className="company-list-search"><Search size={15} /><input data-dialog-autofocus aria-label="Search KPIs" placeholder="Search name, definition or formula" value={pickerQuery} onChange={event => setPickerQuery(event.target.value)} /></span><span className="company-list-picker-count">{count(pickerMetrics.length)} matches</span></div><label className="company-list-picker-availability"><input type="checkbox" aria-label="Only KPIs with saved values" checked={onlyWithValues} onChange={event => { setOnlyWithValues(event.target.checked); if (event.target.checked && category !== 'All KPIs' && !COMPANY_KPIS.some(kpi => kpi.category === category && hasSavedValues(kpi))) setCategory('All KPIs'); }} />With saved values<span>Turn off to inspect catalogue fields that have no usable saved values.</span></label><div className="company-list-picker-body"><nav className="company-list-categories" aria-label="KPI categories">{['All KPIs', ...categories.filter(value => !onlyWithValues || COMPANY_KPIS.some(kpi => kpi.category === value && hasSavedValues(kpi)))].map(value => <button key={value} aria-label={value} aria-pressed={category === value} onClick={() => setCategory(value)}><span>{value}</span><small>{COMPANY_KPIS.filter(kpi => (!onlyWithValues || hasSavedValues(kpi)) && (value === 'All KPIs' || kpi.category === value)).length}</small></button>)}</nav><div className="company-list-kpi-grid" aria-label="Available KPIs">{pickerMetrics.length ? pickerMetrics.map(kpi => <button key={kpi.id} aria-label={`Choose ${kpi.label}`} data-company-list-kpi-option={kpi.id} aria-pressed={pickerKpi === kpi.id} onClick={() => chooseKpi(kpi.id)}><span>{kpi.label}</span><small>{kpi.category} · {hasSavedValues(kpi) ? kpi.id.startsWith('provider_') ? 'Börsdata' : 'Atlas' : 'No usable saved values'}</small></button>) : <p>No supported KPIs match this search. Try another term or category.</p>}</div><section className="company-list-kpi-detail" aria-label="Selected KPI details"><h3>{currentKpi.label}</h3><p>{currentKpi.description}</p><p className="company-list-formula">{currentKpi.formula}</p>{pickerVariant && <p className="company-list-kpi-coverage" data-company-list-kpi-coverage><strong>{pickerVariant.availableCount ? `${count(pickerVariant.availableCount)} / ${count(EXPANDED_KPI_MANIFEST.rows)} listings` : 'No usable saved values'}</strong>{pickerVariant.availableCount > 0 && ' have this period and calculation.'}<br />Saved snapshot {EXPANDED_KPI_MANIFEST.snapshot}. Source: {currentKpi.source ?? 'Börsdata'}.<br />Display unit: {unitLabel(pickerColumn)}.{pickerVariant.notes.length > 0 && <span> {pickerVariant.notes.join(' ')}</span>}</p>}<div className="company-list-column-options"><label>Time period<select aria-label="KPI time period" value={pickerWindow} onChange={event => { const next = event.target.value as CompanyListColumn['window']; setPickerWindow(next); const calculations = calculationsFor(currentKpi, next); if (!calculations.includes(pickerCalculation)) setPickerCalculation(calculations[0]); }}>{currentKpi.windows.map(value => <option key={value} value={value}>{companyListWindowLabel({ id: 'picker', kpiId: currentKpi.id, window: value, calculation: calculationsFor(currentKpi, value)[0] })}</option>)}</select></label><label>Calculation<select aria-label="KPI calculation" value={pickerCalculation} onChange={event => setPickerCalculation(event.target.value as CompanyListColumn['calculation'])}>{calculationsFor(currentKpi, pickerWindow).map(value => <option key={value} value={value}>{companyListCalculationLabel({ id: 'picker', kpiId: currentKpi.id, window: pickerWindow, calculation: value })}</option>)}</select></label></div><button className="company-list-button company-list-button-primary" onClick={addColumn}>{editingColumn ? <Check size={14} /> : <Plus size={14} />}{editingColumn ? 'Update column' : 'Add column'}</button><p className="company-list-status" role="status">{pickerNotice}</p></section></div><div className="company-list-selected-columns"><div className="company-list-selected-heading"><strong>Your columns · {columns.length} / {COMPANY_LIST_MAX_COLUMNS}</strong><span>Edit a column or move it left and right.</span></div><div className="company-list-column-chips">{columns.map((column, index) => <div className="company-list-column-chip" key={column.id} data-company-list-selected-column={column.id} data-editing={editingColumn === column.id}><button aria-label={`Edit ${columnLabel(column)}`} onClick={() => editColumn(column)}>{kpiById.get(column.kpiId)?.label}<small>{companyListWindowLabel(column)} · {calculationLabel(column)}</small></button><button aria-label={`Move ${columnLabel(column)} left`} disabled={index === 0} onClick={() => moveColumn(column.id, -1)}><ArrowLeft size={12} /></button><button aria-label={`Move ${columnLabel(column)} right`} disabled={index === columns.length - 1} onClick={() => moveColumn(column.id, 1)}><ArrowRight size={12} /></button><button aria-label={`Remove ${columnLabel(column)}`} disabled={columns.length <= 1} onClick={() => removeColumn(column.id)}><X size={13} /></button></div>)}</div></div><div className="company-list-dialog-footer"><p>Changes save locally as you edit. Only KPIs supported by this Atlas data release are listed.</p><button className="company-list-button company-list-button-primary" onClick={() => setModal(null)}>Done</button></div></> : modal === 'save' ? <form className="company-list-save-form" onSubmit={event => { event.preventDefault(); saveView(); }}><label>View name<input data-dialog-autofocus aria-label="View name" maxLength={80} required value={viewName} onChange={event => setViewName(event.target.value)} placeholder="e.g. Nordic cash generators" /></label><p>{preferences.savedViews.some(view => view.name.toLowerCase() === viewName.trim().toLowerCase()) ? 'Saving with this name replaces that saved view.' : 'This saves your columns, filters, sorting, density and active watchlist.'}</p><p className="company-list-status" role="status">{status}</p><div className="company-list-buttons"><button className="company-list-button" type="button" onClick={() => setModal(null)}>Cancel</button><button className="company-list-button company-list-button-primary" aria-label="Save current view" type="submit" disabled={!viewName.trim()}><Save size={14} />Save current view</button></div></form> : modal === 'watchlists' ? <section className="company-list-watchlists">
        <div className="company-list-watchlist-items">{features.watchlists.map(item => <div key={item.id} data-company-watchlist={item.id}><button className="company-list-watchlist-select" aria-pressed={features.activeWatchlistId === item.id} onClick={() => changeFeatures(value => ({ ...value, activeWatchlistId: item.id }))}><Star size={14} /><span>{item.name}<small>{count(item.listingIds.length)} saved listings</small></span>{features.activeWatchlistId === item.id && <Check size={14} />}</button><button className="company-list-text-button" aria-label={`Rename watchlist ${item.name}`} onClick={() => { setWatchlistEditId(item.id); setWatchlistName(item.name); }}>Rename</button>{item.id !== 'default' && <button className="company-list-icon" aria-label={`Delete watchlist ${item.name}`} onClick={() => { changeFeatures(value => ({ ...value, activeWatchlistId: value.activeWatchlistId === item.id ? 'default' : value.activeWatchlistId, watchlists: value.watchlists.filter(list => list.id !== item.id), viewFeatures: Object.fromEntries(Object.entries(value.viewFeatures).map(([key, view]) => [key, view.activeWatchlistId === item.id ? { ...view, activeWatchlistId: 'default' } : view])) })); if (watchlistEditId === item.id) { setWatchlistEditId(null); setWatchlistName(''); } }}><Trash2 size={13} /></button>}</div>)}</div>
        <form onSubmit={event => { event.preventDefault(); saveWatchlist(); }}><label>{watchlistEditId ? 'Rename watchlist' : 'New watchlist'}<input data-dialog-autofocus aria-label="Watchlist name" maxLength={80} required value={watchlistName} onChange={event => setWatchlistName(event.target.value)} placeholder="e.g. Dividend research" /></label><button className="company-list-button company-list-button-primary" type="submit" disabled={!watchlistName.trim()}>{watchlistEditId ? 'Rename' : 'Create watchlist'}</button>{watchlistEditId && <button className="company-list-text-button" type="button" onClick={() => { setWatchlistEditId(null); setWatchlistName(''); }}>Cancel rename</button>}</form><p className="company-list-status" role="status">{status}</p>
      </section> : modal === 'compare' ? <CompanyListComparison rows={comparisonRows} columns={columns} cell={evaluate} onCompany={listingId => { setModal(null); onCompany(listingId); }} onRemove={toggleComparison} /> : modal === 'help' && helpColumn ? <CompanyKpiExplanation column={helpColumn} /> : cellSelection ? <CompanyListValueDetails row={cellSelection.row} column={cellSelection.column} cell={evaluate(cellSelection.row, cellSelection.column)} /> : null}
    </div></div>}
  </main>;
}
