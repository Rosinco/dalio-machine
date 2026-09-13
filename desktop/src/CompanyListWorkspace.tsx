import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, ArrowUpDown, Check, Columns3, Download, GitCompareArrows, Plus, Save, Search, SlidersHorizontal, Star, Trash2, X } from 'lucide-react';
import { COMPANY_KPIS, COMPANY_LIST_MAX_COLUMNS, COMPANY_LIST_STORAGE_KEY, LEGACY_COMPANY_LIST_STORAGE_KEY, columnLabel, companyListCell, companyListVariants, companyListWindowLabel, companyListCalculationLabel, companyListUnit, defaultCompanyListPreferences, matchesCompanyListFilters, parseCompanyListPreferences } from './companyListModel';
import type { CompanyKpi, CompanyListColumn, CompanyListFilters, CompanyListNumericRule, CompanyListPreferences } from './companyListModel';
import { activeCompanyWatchlist, companyListCsv, defaultCompanyListFeatures, sortCompanyListRows, toggleCompanyComparison, updateCompanyWatchlist } from './companyListFeatures';
import type { CompanyListFeatures } from './companyListFeatures';
import type { FinancialIndex } from './financialData';
import { useExpandedKpis } from './useExpandedKpis';
import { expandedVariant, EXPANDED_KPI_MANIFEST } from './expandedKpis';
import { CompanyListComparison, CompanyListValueDetails } from './CompanyListDetails';
import { researchReadinessLabels, researchRouteLabels } from './ResearchGaugeCard';
import type { ResearchGaugeArtifact, ResearchGaugeRow } from './researchGaugeModel';
import './companyList.css';

const count = (value: number) => value.toLocaleString('en-US');
const alphabetical = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const countryNames = new Intl.DisplayNames(['en'], { type: 'region' });
const kpiById = new Map(COMPANY_KPIS.map(kpi => [kpi.id, kpi]));
const categories = [...new Set(COMPANY_KPIS.map(kpi => kpi.category))];
const hasSavedValues = (kpi: CompanyKpi) => !kpi.id.startsWith('provider_') || (kpi.coverage ?? 0) > 0;
const usableKpiCount = COMPANY_KPIS.filter(hasSavedValues).length;
const presets = { all: 'All observations', cash_consistency: 'Consistent cash + EBIT', cash_and_margin: 'Cash + 30% margin' };
const monetary = (column: CompanyListColumn) => ['money', 'price'].includes(companyListUnit(column));
const numeric = (column: CompanyListColumn) => !['text', 'date'].includes(companyListUnit(column));
const calculationsFor = (kpi: CompanyKpi, window: CompanyListColumn['window']) => companyListVariants(kpi).filter(value => value.window === window).map(value => value.calculation);
const sameColumn = (a: CompanyListColumn, b: CompanyListColumn) => a.id === b.id && a.kpiId === b.kpiId && a.window === b.window && a.calculation === b.calculation;
const calculationLabel = companyListCalculationLabel;
const unitLabel = (column: CompanyListColumn) => ({ money: 'currency millions', price: 'currency per share', percent: 'percent', points: 'percentage points', multiple: 'multiple', count: 'count', number: 'number', shares_millions: 'million shares', date: 'date', text: 'text' })[companyListUnit(column)];
const id = () => `list-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 9)}`;
const columnPresetDefinitions: { id: string; label: string; metricIds: string[] }[] = [
  { id: 'valuation', label: 'Valuation', metricIds: ['provider_2', 'provider_4', 'provider_3', 'provider_10', 'provider_11', 'provider_13', 'provider_50', 'cash_factor_30', 'terminal_share'] },
  { id: 'dividends', label: 'Dividends', metricIds: ['provider_1', 'provider_7', 'provider_20', 'provider_26', 'provider_66', 'provider_148', 'positive_fcf', 'fcf_margin'] },
  { id: 'quality', label: 'Profitability & returns', metricIds: ['provider_33', 'provider_34', 'provider_36', 'provider_37', 'provider_29', 'provider_32', 'provider_38', 'positive_fcf'] },
  { id: 'strength', label: 'Financial strength', metricIds: ['provider_42', 'provider_44', 'provider_39', 'provider_40', 'provider_41', 'provider_46', 'provider_60', 'cash_balance'] },
  { id: 'price_ownership', label: 'Price & insider activity', metricIds: ['provider_151', 'provider_152', 'provider_50', 'provider_159', 'provider_311', 'provider_110'] },
];
function presetColumns(presetId: string): CompanyListColumn[] {
  const definition = columnPresetDefinitions.find(item => item.id === presetId);
  if (!definition) return [];
  const selected = new Set<string>(['country', 'branch', ...definition.metricIds.filter(metric => kpiById.has(metric) && hasSavedValues(kpiById.get(metric)!))]);
  return [...selected].map(kpiId => {
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
  const [saveEpoch, setSaveEpoch] = useState(0), [page, setPage] = useState(0), [pageSize, setPageSize] = useState(50);
  const [showFilters, setShowFilters] = useState(false), [modal, setModal] = useState<'columns' | 'save' | 'watchlists' | 'compare' | 'cell' | null>(null);
  const [pickerQuery, setPickerQuery] = useState(''), [category, setCategory] = useState('All KPIs');
  const [onlyWithValues, setOnlyWithValues] = useState(true);
  const [pickerKpi, setPickerKpi] = useState(COMPANY_KPIS[0].id), [pickerWindow, setPickerWindow] = useState<CompanyListColumn['window']>('latest');
  const [pickerCalculation, setPickerCalculation] = useState<CompanyListColumn['calculation']>('latest'), [editingColumn, setEditingColumn] = useState<string | null>(null);
  const [pickerNotice, setPickerNotice] = useState(''), [status, setStatus] = useState(''), [viewName, setViewName] = useState(''), [selectedView, setSelectedView] = useState('');
  const [ruleDrafts, setRuleDrafts] = useState<Record<number, string>>({});
  const [upperDrafts, setUpperDrafts] = useState<Record<number, string>>({});
  const [watchlistName, setWatchlistName] = useState(''), [watchlistEditId, setWatchlistEditId] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false), [exportError, setExportError] = useState('');
  const [cellSelection, setCellSelection] = useState<{ row: ResearchGaugeRow; column: CompanyListColumn } | null>(null);
  const [columnPreset, setColumnPreset] = useState('');
  const dialog = useRef<HTMLDivElement>(null), returnFocus = useRef<HTMLElement | null>(null);
  const openModal = (value: Exclude<typeof modal, null>) => { returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null; setModal(value); };
  const available = ready && !error ? data : null;
  const rows = available?.rows ?? [];
  const { columns, filters, sort } = preferences;
  const features = useMemo(() => preferences.features ?? defaultCompanyListFeatures(preferences.watchlistIds), [preferences.features, preferences.watchlistIds]);
  const requestedColumns = useMemo(() => [...columns, ...filters.numericRules.map(rule => rule.column)], [columns, filters.numericRules]);
  const expanded = useExpandedKpis(financial ?? null, taxonomySha256, requestedColumns, !!available);
  const evaluate = (row: ResearchGaugeRow, column: CompanyListColumn) => companyListCell(row, column, expanded);
  const activeList = activeCompanyWatchlist(features);
  const watchlist = useMemo(() => new Set(activeList.listingIds), [activeList.listingIds]);
  const comparisons = useMemo(() => new Set(features.comparisonIds), [features.comparisonIds]);
  const comparisonRows = useMemo(() => features.comparisonIds.map(listingId => rows.find(row => row.id === listingId)).filter((row): row is ResearchGaugeRow => !!row), [features.comparisonIds, rows]);
  const visibleWatchlistCount = useMemo(() => rows.filter(row => watchlist.has(row.id)).length, [rows, watchlist]);
  const countries = useMemo(() => [...new Set(rows.map(row => row.country ?? 'unassigned'))].sort(), [rows]);
  const sectors = useMemo(() => [...new Map(rows.map(row => [row.sectorId ?? 'unassigned', row.sectorName ?? 'Unassigned sector'])).entries()].sort((a, b) => alphabetical.compare(a[1], b[1])), [rows]);
  const branches = useMemo(() => [...new Map(rows.filter(row => filters.sectorId === 'all' || (row.sectorId ?? 'unassigned') === filters.sectorId).map(row => [row.branchId ?? 'unassigned', row.branchName ?? 'Unassigned branch'])).entries()].sort((a, b) => alphabetical.compare(a[1], b[1])), [rows, filters.sectorId]);
  const currencies = useMemo(() => [...new Set([...rows.flatMap(row => [row.annual.currency, row.valuation.currency, row.valuation.priceBasis?.currency]), ...(expanded.index?.reportCurrencies ?? []), ...(expanded.index?.quoteCurrencies ?? [])].filter((value): value is string => !!value))].sort(), [rows, expanded.index]);
  const sortColumn = columns.find(column => column.id === sort.columnId);
  const matches = useMemo(() => {
    const result = rows.filter(row => matchesCompanyListFilters(row, filters, watchlist, expanded));
    return sortCompanyListRows(result, columns, [sort, ...features.secondarySorts], (row, column) => companyListCell(row, column, expanded));
  }, [rows, filters, watchlist, columns, sort, features.secondarySorts, expanded]);
  const pages = Math.max(1, Math.ceil(matches.length / pageSize)), selectedPage = Math.min(page, pages - 1);
  const visibleRows = matches.slice(selectedPage * pageSize, (selectedPage + 1) * pageSize);
  const currentKpi = kpiById.get(pickerKpi) ?? COMPANY_KPIS[0];
  const pickerColumn = { id: 'picker', kpiId: currentKpi.id, window: pickerWindow, calculation: pickerCalculation };
  const pickerVariant = expandedVariant(pickerColumn);
  const pickerMetrics = useMemo(() => COMPANY_KPIS.filter(kpi => (!onlyWithValues || hasSavedValues(kpi)) && (category === 'All KPIs' || kpi.category === category) && `${kpi.id} ${kpi.label} ${kpi.description} ${kpi.formula} ${kpi.category} ${kpi.searchTerms?.join(' ') ?? ''}`.toLowerCase().includes(pickerQuery.trim().toLowerCase())), [category, pickerQuery, onlyWithValues]);
  const numericColumns = columns.filter(numeric);
  const filterCount = [filters.country, filters.sectorId, filters.branchId, filters.route, filters.readiness, filters.presence].filter(value => value !== 'all').length + filters.numericRules.length;
  const sortDescription = sortColumn ? `${columnLabel(sortColumn)} · ${sort.direction === 'asc' ? 'low to high' : 'high to low'}${monetary(sortColumn) ? ' within each currency; currencies A–Z' : ''}` : `Company name · ${sort.direction === 'asc' ? 'A–Z' : 'Z–A'}`;
  const change = (update: (value: CompanyListPreferences) => CompanyListPreferences) => { setPreferences(update); setSaveEpoch(value => value + 1); setStatus(''); };
  const changeFeatures = (update: (value: CompanyListFeatures) => CompanyListFeatures) => change(value => {
    const next = update(value.features ?? defaultCompanyListFeatures(value.watchlistIds));
    return { ...value, features: next, watchlistIds: next.watchlists.find(list => list.id === 'default')?.listingIds ?? value.watchlistIds };
  });
  const changeFilters = (patch: Partial<CompanyListFilters>) => change(value => ({ ...value, filters: { ...value.filters, ...patch } }));
  useEffect(() => { if (initialBranchId) setPreferences(value => ({ ...value, filters: { ...value.filters, sectorId: 'all', branchId: initialBranchId } })); }, [initialBranchId]);
  useEffect(() => { setPage(0); }, [filters, sort, features.secondarySorts, features.activeWatchlistId, pageSize, available]);
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
    change(value => ({ ...value, columns: editingColumn ? value.columns.map(existing => existing.id === editingColumn ? column : existing) : [...value.columns, column] }));
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
  const resetFilters = () => { setRuleDrafts({}); setUpperDrafts({}); change(value => ({ ...value, filters: { ...defaultCompanyListPreferences().filters, watchlistOnly: value.filters.watchlistOnly } })); };
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
    change(value => ({ ...value, columns: next, sort: { columnId: 'name', direction: 'asc' } }));
    changeFeatures(value => ({ ...value, secondarySorts: [] }));
    setStatus('Column preset applied. Your filters and watchlists are kept.');
  };
  const toggleComparison = (listingId: string) => changeFeatures(value => toggleCompanyComparison(value, listingId));
  const openCell = (row: ResearchGaugeRow, column: CompanyListColumn) => { setCellSelection({ row, column }); openModal('cell'); };
  const sortIcon = (columnId: string) => sort.columnId !== columnId ? <ArrowUpDown size={12} /> : sort.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />;

  return <main className="company-list-workspace" aria-label="Company lists workspace" data-company-list-ready={!!available} data-company-list-total={rows.length} data-company-list-matches={matches.length} data-density={features.density}>
    <div inert={modal ? true : undefined}>
      <header className="company-list-heading"><div><button className="company-list-back" onClick={onBack}><ArrowLeft size={14} />Back to explorer</button><div className="eyebrow">COMPANY OBSERVATORY / LISTS</div><h1>Company lists</h1><p>Choose your KPIs. Find businesses to investigate. Keep the ones that interest you.</p></div>{available && <div className="company-list-snapshot"><strong>{count(rows.length)} saved listings · {available.asOf}</strong>Known observation dates are shown. Click a value for its source.</div>}</header>
      {storageError && <p className="company-list-notice" role="status" data-company-list-storage-error>{storageError}</p>}
      {!ready ? <div className="company-list-loading" role="status">Opening the saved company universe…</div> : error ? <div className="company-list-loading" role="alert">{error}</div> : !available ? <div className="company-list-loading">Choose the matching financial pack and company directory in the Library to open company lists.</div> : <>
        <section className="company-list-panel" aria-label="Customizable company list">
          <div className="company-list-controls">
            <div className="company-list-toolbar"><div className="company-list-tabs" aria-label="Company list scope"><button aria-label="All listings" aria-pressed={!filters.watchlistOnly} onClick={() => changeFilters({ watchlistOnly: false })}>All listings <span>{count(rows.length)}</span></button><button aria-label="Watchlist" aria-pressed={filters.watchlistOnly} onClick={() => changeFilters({ watchlistOnly: true })}><Star size={13} />Watchlist <span>{count(visibleWatchlistCount)}</span></button></div><div className="company-list-buttons"><button className="company-list-button" aria-label="Show list filters" aria-expanded={showFilters} onClick={() => setShowFilters(!showFilters)}><SlidersHorizontal size={14} />Filters{filterCount > 0 && ` · ${filterCount}`}</button><button className="company-list-button" aria-label="Choose KPI columns" onClick={() => { setPickerNotice(''); openModal('columns'); }}><Columns3 size={14} />Choose KPIs · {columns.length}</button><button className="company-list-button" aria-label="Save view" onClick={() => { setViewName(preferences.savedViews.find(view => view.id === selectedView)?.name ?? ''); openModal('save'); }}><Save size={14} />Save view</button></div></div>
            <div className="company-list-viewbar"><span className="company-list-search"><Search size={15} /><input aria-label="Company list search" placeholder="Search name, ticker or ISIN" value={filters.query} onChange={event => changeFilters({ query: event.target.value })} /></span><label>Historical screen<select aria-label="Company list preset" value={filters.preset} onChange={event => changeFilters({ preset: event.target.value as CompanyListFilters['preset'] })}>{Object.entries(presets).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><div className="company-list-saved-control"><label>Saved view<select aria-label="Saved company list view" value={selectedView} onChange={event => loadView(event.target.value)}><option value="">Current view</option>{preferences.savedViews.map(view => <option key={view.id} value={view.id}>{view.name}</option>)}</select></label>{selectedView && <button className="company-list-icon" aria-label="Delete selected saved view" onClick={() => { change(value => ({ ...value, savedViews: value.savedViews.filter(view => view.id !== selectedView) })); changeFeatures(value => ({ ...value, viewFeatures: Object.fromEntries(Object.entries(value.viewFeatures).filter(([key]) => key !== selectedView)) })); setSelectedView(''); setStatus('Saved view deleted. Your current columns and watchlist are kept.'); }}><Trash2 size={14} /></button>}</div></div>
            <div className="company-list-extra-toolbar">
              <label>Watchlist<select aria-label="Active watchlist" value={features.activeWatchlistId} onChange={event => changeFeatures(value => ({ ...value, activeWatchlistId: event.target.value }))}>{features.watchlists.map(item => <option key={item.id} value={item.id}>{item.name} · {count(item.listingIds.length)}</option>)}</select></label>
              <button className="company-list-button" aria-label="Manage watchlists" onClick={() => { setWatchlistName(''); setWatchlistEditId(null); openModal('watchlists'); }}><Star size={13} />Manage lists</button>
              <label>Column preset<select aria-label="Column preset" value={columnPreset} onChange={event => applyColumnPreset(event.target.value)}><option value="">Custom columns</option>{columnPresetDefinitions.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              <label>Density<select aria-label="Table density" value={features.density} onChange={event => changeFeatures(value => ({ ...value, density: event.target.value as CompanyListFeatures['density'] }))}><option value="compact">Compact</option><option value="comfortable">Comfortable</option></select></label>
              <button className="company-list-button" aria-label="Compare selected companies" disabled={comparisonRows.length < 2} onClick={() => openModal('compare')}><GitCompareArrows size={14} />Compare · {comparisonRows.length}/8</button>
              {comparisons.size > 0 && <button className="company-list-text-button" onClick={() => changeFeatures(value => ({ ...value, comparisonIds: [] }))}>Clear selection</button>}
              <button className="company-list-button" aria-label="Export company list CSV" aria-busy={exportBusy} disabled={exportBusy || !matches.length || expanded.loading.size > 0} onClick={exportCsv}><Download size={14} />{exportBusy ? 'Exporting…' : 'Export CSV'}</button>
            </div>
            <details className="company-list-sort-controls"><summary>Sort priorities · {1 + features.secondarySorts.length} of 3</summary><div className="company-list-sort-rules">
              {[sort, ...features.secondarySorts].map((entry, index) => <div key={index} className="company-list-sort-rule"><span>{index === 0 ? 'First' : 'Then'}</span><select aria-label={`Sort priority ${index + 1}`} value={entry.columnId} onChange={event => { if (index === 0) setSort(event.target.value); else changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.map((item, position) => position === index - 1 ? { ...item, columnId: event.target.value } : item) })); }}><option value="name" disabled={[sort, ...features.secondarySorts].some((item, position) => position !== index && item.columnId === 'name')}>Company name</option>{columns.map(column => <option key={column.id} value={column.id} disabled={[sort, ...features.secondarySorts].some((item, position) => position !== index && item.columnId === column.id)}>{columnLabel(column)}</option>)}</select><select aria-label={`Sort direction ${index + 1}`} value={entry.direction} onChange={event => { const direction = event.target.value as 'asc' | 'desc'; if (index === 0) change(value => ({ ...value, sort: { ...value.sort, direction } })); else changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.map((item, position) => position === index - 1 ? { ...item, direction } : item) })); }}><option value="asc">Ascending</option><option value="desc">Descending</option></select>{index > 0 && <button className="company-list-icon" aria-label={`Remove sort priority ${index + 1}`} onClick={() => changeFeatures(value => ({ ...value, secondarySorts: value.secondarySorts.filter((_, position) => position !== index - 1) }))}><X size={13} /></button>}</div>)}
              <button className="company-list-button" disabled={features.secondarySorts.length >= 2} onClick={() => { const next = columns.find(column => column.id !== sort.columnId && !features.secondarySorts.some(item => item.columnId === column.id)); if (next) changeFeatures(value => ({ ...value, secondarySorts: [...value.secondarySorts, { columnId: next.id, direction: 'asc' }] })); }}><Plus size={13} />Add sort priority</button>
            </div></details>
            {exportError && <p className="company-list-notice" role="alert">{exportError}</p>}
            {requestedColumns.some(column => column.kpiId.startsWith('provider_')) && <p className="company-list-source-date" data-company-list-provider-snapshot>Provider KPIs downloaded {EXPANDED_KPI_MANIFEST.snapshot}. Underlying report and quote dates may be unavailable. Starter valuations retain their own saved inputs.</p>}
            {expanded.error && <p className="company-list-notice" role="status">{expanded.error}</p>}
            {expanded.errors.size > 0 && <p className="company-list-notice" role="status">{expanded.errors.size} selected KPI datasets could not be opened. Their values and filter matches remain unavailable; click a cell for details.</p>}
            {expanded.loading.size > 0 && <p className="company-list-status" role="status">Opening {expanded.loading.size} selected KPI datasets…</p>}
            {showFilters && <div className="company-list-filter-panel" aria-label="Company list filters"><div className="company-list-filter-grid">
              <label>Listing country<select aria-label="Company list country" value={filters.country} onChange={event => changeFilters({ country: event.target.value })}><option value="all">All countries</option>{countries.map(value => <option key={value} value={value}>{countryLabel(value)}</option>)}</select></label>
              <label>Sector<select aria-label="Company list sector" value={filters.sectorId} onChange={event => changeFilters({ sectorId: event.target.value, branchId: 'all' })}><option value="all">All sectors</option>{sectors.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Branch<select aria-label="Company list branch" value={filters.branchId} onChange={event => changeFilters({ branchId: event.target.value })}><option value="all">All branches</option>{branches.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Annual evidence<select aria-label="Company list coverage" value={filters.readiness} onChange={event => changeFilters({ readiness: event.target.value as CompanyListFilters['readiness'] })}><option value="all">All coverage</option>{Object.entries(researchReadinessLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Business research route<select aria-label="Company list route" value={filters.route} onChange={event => changeFilters({ route: event.target.value as CompanyListFilters['route'] })}><option value="all">All business routes</option>{Object.entries(researchRouteLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Directory coverage<select aria-label="Company list presence" value={filters.presence} onChange={event => changeFilters({ presence: event.target.value as CompanyListFilters['presence'] })}><option value="all">All saved listings</option><option value="latest">Newest directory download</option><option value="older">Older download only</option></select></label>
            </div><div className="company-list-numeric-heading"><h3>Numeric conditions · all must match</h3><button className="company-list-button" aria-label="Add numeric KPI filter" disabled={filters.numericRules.length >= 12 || !numericColumns.length} onClick={addRule}><Plus size={13} />Add condition</button></div><div className="company-list-numeric-rules">{filters.numericRules.map((rule, index) => {
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
          </div>
          {filters.preset !== 'all' && <p className="company-list-preset-note" data-company-list-preset-note><strong>{presets[filters.preset]}:</strong> operating businesses with five-period evidence, no classification conflict, and positive provider FCF and EBIT in all five annual periods.{filters.preset === 'cash_and_margin' && ' Also requires positive saved equity price and starter value, with price at or below 70% of Mid DCF value.'} A historical research screen; inspect dated prices and starter assumptions before a deep dive.</p>}
          <div className="company-list-caption"><span role="status"><strong>{count(matches.length)}</strong> matching {filters.watchlistOnly ? 'watchlist ' : ''}listings{filters.watchlistOnly && ` · ${count(activeList.listingIds.length)} saved stars${activeList.listingIds.length !== visibleWatchlistCount ? `, ${count(visibleWatchlistCount)} in this release` : ''}`}</span><span data-company-list-sort-status>Sorted by {sortDescription}  · missing values last{features.secondarySorts.length > 0 && ` · then ${features.secondarySorts.map(item => `${columns.find(column => column.id === item.columnId) ? columnLabel(columns.find(column => column.id === item.columnId)!) : 'Company name'} ${item.direction === 'asc' ? 'ascending' : 'descending'}`).join(', then ')}`}</span></div>
          {visibleRows.length ? <div className="company-list-table-scroll" role="region" aria-label="Company list results" tabIndex={0}><table className="company-list-table"><colgroup><col style={{ width: 230 }} />{columns.map(column => <col key={column.id} style={{ width: companyListUnit(column) === 'text' ? 165 : 145 }} />)}</colgroup><thead><tr><th scope="col" aria-sort={sort.columnId === 'name' ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'}><button aria-label="Sort by company name" onClick={() => setSort('name')}><span className="company-list-column-name">Company {sortIcon('name')}</span><small>Ticker · country · Börsdata ID</small></button></th>{columns.map(column => <th key={column.id} scope="col" data-company-list-header={column.id} data-company-list-column={column.id} data-company-list-kpi={column.kpiId} aria-sort={sort.columnId === column.id ? sort.direction === 'asc' ? 'ascending' : 'descending' : 'none'}><button aria-label={`Sort by ${columnLabel(column)}`} title={kpiById.get(column.kpiId)?.description} onClick={() => setSort(column.id)}><span className="company-list-column-name">{kpiById.get(column.kpiId)?.label}{sortIcon(column.id)}</span><small>{companyListWindowLabel(column)} · {calculationLabel(column)}</small>{column.kpiId.startsWith('provider_') && <small className="company-list-provider-date">Downloaded {EXPANDED_KPI_MANIFEST.snapshot}</small>}</button></th>)}</tr></thead><tbody>{visibleRows.map(row => <tr key={row.id} data-company-listing={row.id}><td><div className="company-list-company-cell"><input className="company-list-compare-check" type="checkbox" aria-label={`Compare ${row.name}`} checked={comparisons.has(row.id)} disabled={!comparisons.has(row.id) && comparisons.size >= 8} onChange={() => toggleComparison(row.id)} /><button className="company-list-star" aria-label={`${watchlist.has(row.id) ? 'Remove' : 'Add'} ${row.name} ${watchlist.has(row.id) ? 'from' : 'to'} watchlist`} aria-pressed={watchlist.has(row.id)} onClick={() => toggleWatchlist(row)}><Star size={15} /></button><div className="company-list-company-text"><button className="company-list-company" title={row.name} aria-label={`Open profile for ${row.name}`} onClick={() => onCompany(row.id)}>{row.name}</button><small>{row.ticker ?? '—'} · {row.country ?? '—'} · {row.id}</small>{filters.preset === 'cash_and_margin' && <small data-company-list-candidate-price-date={row.valuation.priceDate ?? ''}>Saved price {row.valuation.priceDate ?? 'unavailable'}</small>}</div></div></td>{columns.map(column => { const cell = evaluate(row, column); return <td key={column.id} title={`${cell.display}. ${cell.detail}`} onDoubleClick={() => openCell(row, column)} aria-label={`${columnLabel(column)}: ${cell.display}. ${cell.detail}`} data-company-list-kpi={column.kpiId} data-company-list-column={column.id} data-company-list-value={cell.value ?? ''} data-company-list-currency={cell.currency ?? ''} data-company-list-date={cell.date ?? ''} data-value-kind={typeof cell.value === 'string' ? 'text' : cell.value === null ? 'missing' : 'number'}><button type="button" className={`company-list-cell-value company-list-cell-inspect${cell.value === null ? ' company-list-cell-missing' : ''}`} onClick={() => openCell(row, column)} aria-label={cell.value === null ? `${kpiById.get(column.kpiId)?.label} unavailable: ${cell.detail}` : undefined}>{cell.display}</button>{cell.date && companyListUnit(column) !== 'date' && <small className="company-list-cell-date">{cell.date}</small>}</td>; })}</tr>)}</tbody></table></div> : <div className="company-list-empty"><strong>{filters.watchlistOnly && !visibleWatchlistCount ? 'Your watchlist is ready for companies.' : 'No listings match this view.'}</strong>{filters.watchlistOnly && !visibleWatchlistCount ? 'Use the star beside a company in All listings to keep it here.' : 'Adjust the conditions or clear your filters to broaden the list.'}<div><button className="company-list-button" onClick={filters.watchlistOnly && !visibleWatchlistCount ? () => changeFilters({ watchlistOnly: false }) : resetFilters}>{filters.watchlistOnly && !visibleWatchlistCount ? 'Browse all listings' : 'Clear list filters'}</button></div></div>}
          <div className="company-list-pagination"><span>{matches.length ? `${count(selectedPage * pageSize + 1)}–${count(Math.min((selectedPage + 1) * pageSize, matches.length))}` : '0'} of {count(matches.length)}</span><div><select aria-label="Company list rows per page" value={pageSize} onChange={event => setPageSize(Number(event.target.value))}><option value={25}>25 rows</option><option value={50}>50 rows</option><option value={100}>100 rows</option><option value={250}>250 rows</option></select><button className="company-list-icon" aria-label="Previous company list page" disabled={!selectedPage} onClick={() => setPage(selectedPage - 1)}><ArrowLeft size={14} /></button><span>{selectedPage + 1} / {pages}</span><button className="company-list-icon" aria-label="Next company list page" disabled={selectedPage + 1 >= pages} onClick={() => setPage(selectedPage + 1)}><ArrowRight size={14} /></button></div></div>
        </section>
        <p className="company-list-status" role="status">{status}</p>
        <details className="company-list-method"><summary>About KPI periods, saved prices and candidate screens</summary><p>Each value uses the validated data currently bundled in Atlas. “Latest saved” means the latest eligible report or saved quote, with its own date. Annual windows require the requested comparable periods. An em dash means a value or valid comparison is unavailable; click the cell for its reason. Money is shown in millions of the stated currency, and prices per share use the stated quote currency.</p><p>Columns can be sorted and filtered. Monetary sorting groups currencies A–Z, then sorts within each currency; it does not compare converted amounts. Named views save columns, filters and sorting. Stars add companies to the selected named watchlist. Saved views also retain the active watchlist, up to three sorting priorities and the table density. Separate listings or share classes may represent the same company.</p><p>The two historical screens apply the written criteria only. Five-period evidence requires five valid annual observations for FCF, operating cash, EBIT and revenue, known publication dates, and an annual end within 550 days of the saved snapshot. These observations do not establish future cash or business quality. Companies with thinner history and financial businesses remain in All listings.</p><p>Starter DCF, terminal value and NPV use one connected calculation. The saved 30% purchase ceiling is 0.70 × positive Mid equity value. It is separate from reviewed or edited company valuations. Saved whole-equity price uses reported shares and a dated close; the share basis remains an unreviewed proxy. Provider FCF needs reconciliation to sustainable owner cash during a deep dive.</p><p>Provider KPIs use their saved source dates, definitions and supported period/calculation pairs. A current or R12 provider snapshot is distinct from an annual-report calculation. Imported provider prices and valuation ratios do not update the frozen starter DCF, NPV or purchase ceiling. Click a value to inspect its source and calculation.</p></details>
      </>}
    </div>
    {modal && <div className="company-list-modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) setModal(null); }}><div ref={dialog} className={`company-list-dialog${modal === 'save' || modal === 'cell' || modal === 'watchlists' ? ' company-list-dialog-save' : ''}`} role="dialog" aria-modal="true" aria-label={{ columns: 'Choose KPI columns', save: 'Save company list view', watchlists: 'Manage watchlists', compare: 'Compare companies', cell: 'KPI value details' }[modal]} tabIndex={-1}>
      <div className="company-list-dialog-header"><div><h2>{{ columns: 'Choose your KPI columns', save: 'Save this view', watchlists: 'Your watchlists', compare: 'Compare selected companies', cell: 'Value and source' }[modal]}</h2><p>{{ columns: `${usableKpiCount} KPIs and fields with saved values · ${COMPANY_KPIS.length} in the catalogue. Choose a period and calculation.`, save: 'Keep columns, filters, sorting and watchlist together.', watchlists: 'Keep separate lists for different research ideas.', compare: `${comparisonRows.length} selected listings · your current columns`, cell: 'Inspect the observation behind this number.' }[modal]}</p></div><button className="company-list-icon" aria-label="Close company list dialog" onClick={() => setModal(null)}><X size={17} /></button></div>
      {modal === 'columns' ? <><div className="company-list-picker-search"><span className="company-list-search"><Search size={15} /><input data-dialog-autofocus aria-label="Search KPIs" placeholder="Search name, definition or formula" value={pickerQuery} onChange={event => setPickerQuery(event.target.value)} /></span><span className="company-list-picker-count">{count(pickerMetrics.length)} matches</span></div><label className="company-list-picker-availability"><input type="checkbox" aria-label="Only KPIs with saved values" checked={onlyWithValues} onChange={event => { setOnlyWithValues(event.target.checked); if (event.target.checked && category !== 'All KPIs' && !COMPANY_KPIS.some(kpi => kpi.category === category && hasSavedValues(kpi))) setCategory('All KPIs'); }} />With saved values<span>Turn off to inspect catalogue fields that have no usable saved values.</span></label><div className="company-list-picker-body"><nav className="company-list-categories" aria-label="KPI categories">{['All KPIs', ...categories.filter(value => !onlyWithValues || COMPANY_KPIS.some(kpi => kpi.category === value && hasSavedValues(kpi)))].map(value => <button key={value} aria-label={value} aria-pressed={category === value} onClick={() => setCategory(value)}><span>{value}</span><small>{COMPANY_KPIS.filter(kpi => (!onlyWithValues || hasSavedValues(kpi)) && (value === 'All KPIs' || kpi.category === value)).length}</small></button>)}</nav><div className="company-list-kpi-grid" aria-label="Available KPIs">{pickerMetrics.length ? pickerMetrics.map(kpi => <button key={kpi.id} aria-label={`Choose ${kpi.label}`} data-company-list-kpi-option={kpi.id} aria-pressed={pickerKpi === kpi.id} onClick={() => chooseKpi(kpi.id)}><span>{kpi.label}</span><small>{kpi.category} · {hasSavedValues(kpi) ? kpi.id.startsWith('provider_') ? 'Börsdata' : 'Atlas' : 'No usable saved values'}</small></button>) : <p>No supported KPIs match this search. Try another term or category.</p>}</div><section className="company-list-kpi-detail" aria-label="Selected KPI details"><h3>{currentKpi.label}</h3><p>{currentKpi.description}</p><p className="company-list-formula">{currentKpi.formula}</p>{pickerVariant && <p className="company-list-kpi-coverage" data-company-list-kpi-coverage><strong>{pickerVariant.availableCount ? `${count(pickerVariant.availableCount)} / ${count(EXPANDED_KPI_MANIFEST.rows)} listings` : 'No usable saved values'}</strong>{pickerVariant.availableCount > 0 && ' have this period and calculation.'}<br />Saved snapshot {EXPANDED_KPI_MANIFEST.snapshot}. Source: {currentKpi.source ?? 'Börsdata'}.<br />Display unit: {unitLabel(pickerColumn)}.{pickerVariant.notes.length > 0 && <span> {pickerVariant.notes.join(' ')}</span>}</p>}<div className="company-list-column-options"><label>Time period<select aria-label="KPI time period" value={pickerWindow} onChange={event => { const next = event.target.value as CompanyListColumn['window']; setPickerWindow(next); const calculations = calculationsFor(currentKpi, next); if (!calculations.includes(pickerCalculation)) setPickerCalculation(calculations[0]); }}>{currentKpi.windows.map(value => <option key={value} value={value}>{companyListWindowLabel({ id: 'picker', kpiId: currentKpi.id, window: value, calculation: calculationsFor(currentKpi, value)[0] })}</option>)}</select></label><label>Calculation<select aria-label="KPI calculation" value={pickerCalculation} onChange={event => setPickerCalculation(event.target.value as CompanyListColumn['calculation'])}>{calculationsFor(currentKpi, pickerWindow).map(value => <option key={value} value={value}>{companyListCalculationLabel({ id: 'picker', kpiId: currentKpi.id, window: pickerWindow, calculation: value })}</option>)}</select></label></div><button className="company-list-button company-list-button-primary" onClick={addColumn}>{editingColumn ? <Check size={14} /> : <Plus size={14} />}{editingColumn ? 'Update column' : 'Add column'}</button><p className="company-list-status" role="status">{pickerNotice}</p></section></div><div className="company-list-selected-columns"><div className="company-list-selected-heading"><strong>Your columns · {columns.length} / {COMPANY_LIST_MAX_COLUMNS}</strong><span>Edit a column or move it left and right.</span></div><div className="company-list-column-chips">{columns.map((column, index) => <div className="company-list-column-chip" key={column.id} data-company-list-selected-column={column.id} data-editing={editingColumn === column.id}><button aria-label={`Edit ${columnLabel(column)}`} onClick={() => editColumn(column)}>{kpiById.get(column.kpiId)?.label}<small>{companyListWindowLabel(column)} · {calculationLabel(column)}</small></button><button aria-label={`Move ${columnLabel(column)} left`} disabled={index === 0} onClick={() => moveColumn(column.id, -1)}><ArrowLeft size={12} /></button><button aria-label={`Move ${columnLabel(column)} right`} disabled={index === columns.length - 1} onClick={() => moveColumn(column.id, 1)}><ArrowRight size={12} /></button><button aria-label={`Remove ${columnLabel(column)}`} disabled={columns.length <= 1} onClick={() => removeColumn(column.id)}><X size={13} /></button></div>)}</div></div><div className="company-list-dialog-footer"><p>Changes save locally as you edit. Only KPIs supported by this Atlas data release are listed.</p><button className="company-list-button company-list-button-primary" onClick={() => setModal(null)}>Done</button></div></> : modal === 'save' ? <form className="company-list-save-form" onSubmit={event => { event.preventDefault(); saveView(); }}><label>View name<input data-dialog-autofocus aria-label="View name" maxLength={80} required value={viewName} onChange={event => setViewName(event.target.value)} placeholder="e.g. Nordic cash generators" /></label><p>{preferences.savedViews.some(view => view.name.toLowerCase() === viewName.trim().toLowerCase()) ? 'Saving with this name replaces that saved view.' : 'This saves your columns, filters, sorting, density and active watchlist.'}</p><p className="company-list-status" role="status">{status}</p><div className="company-list-buttons"><button className="company-list-button" type="button" onClick={() => setModal(null)}>Cancel</button><button className="company-list-button company-list-button-primary" aria-label="Save current view" type="submit" disabled={!viewName.trim()}><Save size={14} />Save current view</button></div></form> : modal === 'watchlists' ? <section className="company-list-watchlists">
        <div className="company-list-watchlist-items">{features.watchlists.map(item => <div key={item.id} data-company-watchlist={item.id}><button className="company-list-watchlist-select" aria-pressed={features.activeWatchlistId === item.id} onClick={() => changeFeatures(value => ({ ...value, activeWatchlistId: item.id }))}><Star size={14} /><span>{item.name}<small>{count(item.listingIds.length)} saved listings</small></span>{features.activeWatchlistId === item.id && <Check size={14} />}</button><button className="company-list-text-button" aria-label={`Rename watchlist ${item.name}`} onClick={() => { setWatchlistEditId(item.id); setWatchlistName(item.name); }}>Rename</button>{item.id !== 'default' && <button className="company-list-icon" aria-label={`Delete watchlist ${item.name}`} onClick={() => { changeFeatures(value => ({ ...value, activeWatchlistId: value.activeWatchlistId === item.id ? 'default' : value.activeWatchlistId, watchlists: value.watchlists.filter(list => list.id !== item.id), viewFeatures: Object.fromEntries(Object.entries(value.viewFeatures).map(([key, view]) => [key, view.activeWatchlistId === item.id ? { ...view, activeWatchlistId: 'default' } : view])) })); if (watchlistEditId === item.id) { setWatchlistEditId(null); setWatchlistName(''); } }}><Trash2 size={13} /></button>}</div>)}</div>
        <form onSubmit={event => { event.preventDefault(); saveWatchlist(); }}><label>{watchlistEditId ? 'Rename watchlist' : 'New watchlist'}<input data-dialog-autofocus aria-label="Watchlist name" maxLength={80} required value={watchlistName} onChange={event => setWatchlistName(event.target.value)} placeholder="e.g. Dividend research" /></label><button className="company-list-button company-list-button-primary" type="submit" disabled={!watchlistName.trim()}>{watchlistEditId ? 'Rename' : 'Create watchlist'}</button>{watchlistEditId && <button className="company-list-text-button" type="button" onClick={() => { setWatchlistEditId(null); setWatchlistName(''); }}>Cancel rename</button>}</form><p className="company-list-status" role="status">{status}</p>
      </section> : modal === 'compare' ? <CompanyListComparison rows={comparisonRows} columns={columns} cell={evaluate} onCompany={listingId => { setModal(null); onCompany(listingId); }} onRemove={toggleComparison} /> : cellSelection ? <CompanyListValueDetails row={cellSelection.row} column={cellSelection.column} cell={evaluate(cellSelection.row, cellSelection.column)} /> : null}
    </div></div>}
  </main>;
}
