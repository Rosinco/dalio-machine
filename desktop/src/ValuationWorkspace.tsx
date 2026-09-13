import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Download, Save } from 'lucide-react';
import type { CompanyEntry } from './listingCatalogue';
import type { FinancialCompany, FinancialIndex } from './financialData';
import type { ResearchRelease } from './types';
import { format } from './model';
import { blankValuation, calculateValuation, capitalLabels, generateCashFlows, scenarioColors, scenarioKeys, scenarioNames, type Payback, type Scenario, type ScenarioKey, type ValuationDraft } from './valuation';
import { compatibleValuation, loadValuationDraft, loadValuations, saveValuation, saveValuationDraft, type SavedValuation } from './savedValuations';
import ValuationCharts from './ValuationCharts';
import CashFlowForecastChart from './CashFlowForecastChart';
import CrisisScenarioPanel from './CrisisScenarioPanel';
import { buildCrisisScenario } from './crisisScenario';
import { isUntouchedLegacyStarter } from './starterMigration';
import ValuationResearchEvidence from './ValuationResearchEvidence';
import StarterValuationEvidence from './StarterValuationEvidence';
import { buildStarterValuation, defaultStarterWeights, settingsForStarterOrigin, type StarterSettings } from './starterValuations';
import { buildResearchedValuation, shouldStartResearchedStudy, type ResearchedStudy } from './researchedValuations';
import { resolveTerminalSale } from './terminalValue';
import TerminalValueInputs from './TerminalValueInputs';
import TerminalValueSummary from './TerminalValueSummary';
import PurchaseRangePanel from './PurchaseRangePanel';
import { getPurchaseShareReference } from './purchaseShareBasis';
import './valuation.css';

const noteLabels: Record<keyof ValuationDraft['notes'], string> = { business: 'Business, competition and moat', macro: 'Macro and branch evidence → company exposure → forecast assumption', financing: 'Capital dates, reinvestment, debt and refinancing', recovery: 'Recovery assets, claims, costs, timing and feasibility', decision: 'Investment thesis, realization path and invalidating evidence' };
const paybackLabel = (p: Payback, years: number) => p.year === null ? `Not reached in ${years} years` : `Year ${p.year}${p.reversed ? ' · reverses later' : ''}`;
function NumberInput({ label, value, onChange, min, max, step = 'any' }: { label: string; value: number | null; onChange: (n: number | null) => void; min?: number; max?: number; step?: string }) {
  return <input type="number" aria-label={label} value={value ?? ''} min={min} max={max} step={step} onChange={e => onChange(e.target.value === '' || !Number.isFinite(e.target.valueAsNumber) ? null : e.target.valueAsNumber)} />;
}
export default function ValuationWorkspace({ entry, classification, financial, history, release, researched, onProfile }: { entry: CompanyEntry; classification?: { sector_id: string | null; branch_id: string | null }; financial: FinancialIndex | null; history: FinancialCompany | null; release: ResearchRelease; researched: ResearchedStudy | null; onProfile: () => void }) {
  const valuationEntry = classification ? { ...entry, ...classification } : entry;
  const basis = { company: entry.id, release: release.id, financial: financial?.id ?? null, taxonomy: release.taxonomy_sha256 ?? null };
  const [starterDate] = useState(() => new Date().toISOString().slice(0, 10));
  const startingStudy = useMemo(() => researched ? buildResearchedValuation(researched).draft : null, [researched]);
  const standardStarter = useMemo(() => buildStarterValuation(valuationEntry, history, financial, starterDate), [entry.id, classification?.sector_id, classification?.branch_id, history, financial, starterDate]);
  const startingDraft = startingStudy ?? standardStarter.draft;
  const [initial] = useState(() => {
    try {
      const previous = loadValuationDraft(localStorage, basis);
      const prior = previous?.draft;
      const legacy = prior?.starterOrigin;
      // Only untouched defaults migrate automatically. Custom weights, edits,
      // restored revisions and deliberate blanks retain their chosen model.
      const upgradeStarter = legacy && isUntouchedLegacyStarter(prior, buildStarterValuation(legacy.id === 'empirical-cash-starter-v3' ? valuationEntry : entry, history, financial, legacy.asOf, settingsForStarterOrigin(legacy)).draft);
      if (upgradeStarter || shouldStartResearchedStudy(prior ?? null)) {
        const draft = structuredClone(upgradeStarter ? standardStarter.draft : startingDraft);
        if (previous?.draft.investment != null && previous.draft.investment > 0) draft.investment = previous.draft.investment;
        return { draft: previous?.draft ?? draft, replacement: previous ? draft : null, previous: previous ? { ...previous, id: crypto.randomUUID() } : null, error: '', loaded: !previous };
      }
      return { draft: previous?.draft ?? structuredClone(startingDraft), replacement: null, previous: null, error: '', loaded: false };
    }
    catch (e) { return { draft: blankValuation(entry.display_name), replacement: null, previous: null, error: `The previous draft could not open: ${String(e)}`, loaded: false }; }
  });
  const [d, setDraft] = useState<ValuationDraft>(initial.draft), [saved, setSaved] = useState<SavedValuation[]>([]);
  const [message, setMessage] = useState(initial.loaded ? `${initial.draft.researchOrigin ? 'Researched' : 'Standard historical cash-flow'} ${entry.display_name} scenarios loaded automatically.${initial.previous ? ' Your previous draft is retained in saved studies below.' : ''}` : ''), [saveError, setSaveError] = useState(initial.error), [autoSaved, setAutoSaved] = useState(false);
  const starter = useMemo(() => buildStarterValuation(d.starterOrigin && d.starterOrigin.id !== 'empirical-cash-starter-v3' ? entry : valuationEntry, history, financial, d.starterOrigin?.asOf ?? starterDate, settingsForStarterOrigin(d.starterOrigin)), [entry.id, classification?.sector_id, classification?.branch_id, history, financial, d.starterOrigin, starterDate]);
  const [historyWeights, setHistoryWeights] = useState<(number | null)[]>(settingsForStarterOrigin(initial.draft.starterOrigin).weights);
  const [historyYears, setHistoryYears] = useState<5 | 10>(settingsForStarterOrigin(initial.draft.starterOrigin).historyYears);
  const [projection, setProjection] = useState<StarterSettings['projection']>(settingsForStarterOrigin(initial.draft.starterOrigin).projection);
  const [spread, setSpread] = useState<number | null>(settingsForStarterOrigin(initial.draft.starterOrigin).spreadPercent);
  const [spreadStep, setSpreadStep] = useState<number | null>(settingsForStarterOrigin(initial.draft.starterOrigin).spreadStepPercent);
  const [rangeMode, setRangeMode] = useState<'historical' | 'percentage'>(settingsForStarterOrigin(initial.draft.starterOrigin).rangeMode ?? 'historical');
  const [tailWidening, setTailWidening] = useState<number | null>(settingsForStarterOrigin(initial.draft.starterOrigin).tailWideningPercent ?? 10);
  const [settingsError, setSettingsError] = useState('');
  useEffect(() => {
    const settings = settingsForStarterOrigin(d.starterOrigin);
    setHistoryWeights(settings.weights); setHistoryYears(settings.historyYears); setProjection(settings.projection); setSpread(settings.spreadPercent); setSpreadStep(settings.spreadStepPercent);
    setRangeMode(settings.rangeMode ?? 'historical'); setTailWidening(settings.tailWideningPercent ?? 10);
  }, [d.starterOrigin]);
  const [sale, setSale] = useState(false), [tab, setTab] = useState<'scenarios' | 'evidence'>('scenarios');
  const [showInputs, setShowInputs] = useState(!calculateValuation(initial.draft).ready);
  const [generator, setGenerator] = useState<{ scenario: ScenarioKey; first: number | null; growth: number | null }>({ scenario: 'mid', first: null, growth: null });
  const result = useMemo(() => calculateValuation(d), [d]);
  const makeStudy = (): SavedValuation => ({ format: 'macro-atlas-valuation', version: 1, id: crypto.randomUUID(), created: new Date().toISOString(), ...basis, draft: d });
  const migrationAttempted = useRef(false);
  useEffect(() => { try { setSaved(loadValuations(localStorage)); } catch (e) { setSaveError(String(e)); } }, []);
  useEffect(() => {
    if (initial.error) return;
    try {
      if (initial.previous && initial.replacement && !migrationAttempted.current) {
        migrationAttempted.current = true;
        if (!loadValuations(localStorage).some(v => v.id === initial.previous!.id)) saveValuation(localStorage, initial.previous);
        setSaved(loadValuations(localStorage));
        saveValuationDraft(localStorage, { ...makeStudy(), draft: initial.replacement });
        setDraft(initial.replacement); setShowInputs(!calculateValuation(initial.replacement).ready); setAutoSaved(true); setSaveError('');
        setMessage('Updated starter opened. Your preceding draft is retained as a saved revision.');
        return;
      }
      saveValuationDraft(localStorage, makeStudy()); setAutoSaved(true); setSaveError('');
    }
    catch (e) { setAutoSaved(false); setSaveError(`Draft not saved: ${String(e)}`); }
  }, [d]);
  const update = (patch: Partial<ValuationDraft>) => { setMessage(''); setAutoSaved(false); setDraft(previous => ({ ...previous, ...patch })); };
  const scenario = (key: ScenarioKey, patch: Partial<Scenario>) => {
    const next = { ...d.scenarios[key], ...patch };
    if (next.terminalCash) next.terminalEquity = resolveTerminalSale(next).value;
    update({ scenarios: { ...d.scenarios, [key]: next } });
  };
  const changeYears = (years: number | null) => {
    if (years === null || !Number.isInteger(years) || years < 1 || years > 50) return;
    update({ years, scenarios: Object.fromEntries(scenarioKeys.map(k => [k, { ...d.scenarios[k], cashFlows: Array.from({ length: Math.max(years, d.scenarios[k].cashFlows.length) }, (_, i) => d.scenarios[k].cashFlows[i] ?? null) }])) as ValuationDraft['scenarios'] });
  };
  const save = () => { try { saveValuation(localStorage, makeStudy()); setSaved(loadValuations(localStorage)); setSaveError(''); setMessage('Study revision saved with these company and data versions.'); } catch (e) { setSaveError(String(e)); } };
  const useStartingDraft = (source: ValuationDraft, status: string) => {
    try {
      saveValuation(localStorage, makeStudy()); setSaved(loadValuations(localStorage));
      const draft = structuredClone(source); if (d.investment != null && d.investment > 0) draft.investment = d.investment;
      if (d.purchaseRange) draft.purchaseRange = structuredClone(d.purchaseRange);
      setDraft(draft); setAutoSaved(false); setShowInputs(!calculateValuation(draft).ready); setTab('scenarios'); setSaveError('');
      setMessage(`${status} Your preceding working draft is retained as a saved revision.`);
    } catch (e) { setSaveError(`Could not preserve the current draft: ${String(e)}`); }
  };
  const applyHistoryAssumptions = () => {
    try {
      if (historyWeights.some(value => value === null) || spread === null || spreadStep === null || tailWidening === null) throw new Error('Complete the history weights, percentage sensitivities and later-year widening.');
      const source = buildStarterValuation(valuationEntry, history, financial, starterDate, { historyYears, projection, rangeMode, tailWideningPercent: tailWidening, weights: historyWeights as number[], spreadPercent: spread, spreadStepPercent: spreadStep });
      if (d.starterOrigin?.id === 'empirical-cash-starter-v3' && (d.starterOrigin.terminalMethod || scenarioKeys.some(k => d.scenarios[k].terminalCash))) {
        for (const key of scenarioKeys) {
          const { terminalCash, terminalEquity, discountRate } = d.scenarios[key];
          Object.assign(source.draft.scenarios[key], { terminalCash, terminalEquity, discountRate });
        }
      }
      setSettingsError(''); useStartingDraft(source.draft, 'Historical cash scenarios opened; existing separate terminal inputs and required returns were retained when present.');
    } catch (error) { setSettingsError(String(error instanceof Error ? error.message : error)); }
  };
  const comparison = (draft: ValuationDraft) => JSON.stringify({ ...draft, crisis: undefined, purchaseRange: undefined, investment: 1000, title: '', researchAutofillDisabled: false });
  const changedFromResearch = startingStudy ? comparison(d) !== comparison(startingStudy) : false;
  const changedFromStarter = comparison(d) !== comparison(starter.draft);
  const sameCash = (reference: ValuationDraft) => d.currency === reference.currency && d.valuationDate === reference.valuationDate && d.years === reference.years && scenarioKeys.every(k => JSON.stringify(d.scenarios[k].cashFlows.slice(0, d.years)) === JSON.stringify(reference.scenarios[k].cashFlows.slice(0, d.years)));
  const matchesStarterCash = !!d.starterOrigin && sameCash(starter.draft);
  const matchesResearchCash = !!d.researchOrigin && !!startingStudy && sameCash(startingStudy);
  const latestMarket = history?.market.filter(m => m.sek !== null && m.price_date).sort((a, b) => a.price_date!.localeCompare(b.price_date!)).at(-1);
  const latestReport = history?.annual.at(-1);
  const scale = d.marketCap && d.investment ? d.investment / d.marketCap : null;
  const money = (amount: number | null | undefined, stake = false) => amount == null || stake && scale === null ? 'Unavailable' : `${format(stake ? amount * scale! : amount, 2)} ${d.currency}${stake ? '' : ' m'}`;
  const useHistoricalPrice = () => { if (!latestMarket?.price_date || latestMarket.sek === null) return; update({ marketCap: latestMarket.sek, priceDate: latestMarket.price_date, priceSource: `Saved Börsdata publication-window valuation; ${latestMarket.source_id}; price ${latestMarket.price_date}; reported shares ${latestMarket.shares} million; close ${latestMarket.price} ${latestMarket.currency}; FX ${latestMarket.fx_rate} (${latestMarket.fx_date}). Verify issuer/share-class basis.` }); };
  const exportCSV = async () => {
    try {
      if (!result.ready) throw new Error('Complete all three scenarios before exporting calculations.');
      const rows: (string | number | null)[][] = [['company_id', 'study', 'valuation_date', 'price_date', 'currency', 'market_cap_m', 'investment', 'scenario', 'year', 'equity_cash_payment_m', 'discounted_payment_m', 'cumulative_npv_m', 'cumulative_npv_with_sale_m', 'required_equity_return_pct', 'final_equity_sale_m', 'research_version', 'financial_version', 'price_source', 'starter_method', 'starter_as_of', 'starter_weights_newest_first', 'starter_spread_pct', 'starter_history_years', 'starter_projection', 'starter_spread_step_pct', 'edited_from_starter', 'starting_valuation_study', 'starting_study_date', 'edited_from_starting_study']];
      for (const key of scenarioKeys) for (let i = 0; i <= d.years; i++) rows.push([entry.id, d.title, d.valuationDate, d.priceDate, d.currency, d.marketCap, d.investment, key, i, i ? d.scenarios[key].cashFlows[i - 1] : -d.marketCap!, i ? result.scenarios[key].discounted[i - 1] : -d.marketCap!, result.scenarios[key].cumulativeNPV[i], result.scenarios[key].cumulativeNPVWithSale[i], d.scenarios[key].discountRate, i === d.years ? resolveTerminalSale(d.scenarios[key]).value : 0, release.id, financial?.id ?? '', d.priceSource, d.starterOrigin?.id ?? '', d.starterOrigin?.asOf ?? '', d.starterOrigin?.weights.join('/') ?? '', d.starterOrigin?.spreadPercent ?? '', d.starterOrigin ? starter.evidence.historyYears : '', d.starterOrigin ? starter.evidence.projection : '', d.starterOrigin ? starter.evidence.spreadStepPercent : '', d.starterOrigin ? String(changedFromStarter) : '', d.researchOrigin?.id ?? '', d.researchOrigin?.asOf ?? '', d.researchOrigin ? String(changedFromResearch) : '']);
      rows[0].push('starter_range_mode', 'range_status', 'cash_variability_group', 'range_calibration_id', 'annual_range_basis', 'annual_half_width_m', 'later_year_widening_pct_cash_scale', 'crisis_shock_pct', 'crisis_start_year', 'crisis_duration_years', 'crisis_recovery_years', 'crisis_extra_annual_cash_cost_m', 'crisis_rationale', 'terminal_method', 'terminal_cash_after_reinvestment_m', 'terminal_growth_pct');
      for (let index = 1; index < rows.length; index++) {
        const year = Number(rows[index][8]), evidence = starter.evidence.forecast[year - 1], s = d.scenarios[rows[index][7] as ScenarioKey];
        rows[index].push(d.starterOrigin ? starter.evidence.rangeMode : '', d.starterOrigin ? !matchesStarterCash ? 'edited' : starter.evidence.uncertainty.status : '', d.starterOrigin ? starter.evidence.uncertainty.group ?? '' : '', d.starterOrigin ? starter.evidence.uncertainty.calibrationId ?? '' : '',
          year && d.starterOrigin ? !matchesStarterCash ? 'edited' : evidence?.rangeBasis ?? 'percentage' : '', year && d.starterOrigin && matchesStarterCash ? evidence?.halfWidth ?? '' : '', d.starterOrigin ? starter.evidence.tailWideningPercent : '', '', '', '', '', '', '', s.terminalCash ? 'sustainable-cash' : 'explicit-sale', s.terminalCash?.cashFlow ?? '', s.terminalCash?.growthRate ?? '');
      }
      if (d.crisis?.enabled) {
        const crisis = buildCrisisScenario(d);
        if (crisis.result.error) throw new Error(`Complete the crisis assumptions before exporting: ${crisis.result.error}`);
        for (let i = 0; i <= d.years; i++) {
          const row = [...rows[1]];
          row[7] = 'crisis'; row[8] = i; row[9] = i ? crisis.cashFlows[i - 1] : -d.marketCap!;
          row[10] = i ? crisis.result.discounted[i - 1] : -d.marketCap!; row[11] = crisis.result.cumulativeNPV[i]; row[12] = crisis.result.cumulativeNPVWithSale[i];
          row[13] = d.crisis.discountRate; row[14] = i === d.years ? d.crisis.terminalEquity : 0;
          for (const [name, value] of Object.entries({ starter_range_mode: '', range_status: 'crisis-assumption', cash_variability_group: '', range_calibration_id: '', annual_range_basis: i ? 'crisis-assumption' : '', annual_half_width_m: '', later_year_widening_pct_cash_scale: '', crisis_shock_pct: d.crisis.shockPercent, crisis_start_year: d.crisis.startYear, crisis_duration_years: d.crisis.durationYears, crisis_recovery_years: d.crisis.recoveryYears, crisis_extra_annual_cash_cost_m: d.crisis.extraAnnualCashCost, crisis_rationale: d.crisis.rationale, terminal_method: 'explicit-sale', terminal_cash_after_reinvestment_m: '', terminal_growth_pct: '' })) row[rows[0].indexOf(name)] = value;
          rows.push(row);
        }
      }
      const csv = rows.map(row => row.map(v => typeof v === 'number' ? String(v) : `"${String(v ?? '').replace(/^[=+@\-\t\r]/, "'$&").replaceAll('"', '""')}"`).join(',')).join('\r\n');
      const filename = `Macro-Atlas-Valuation-${entry.id}-${d.valuationDate}.csv`;
      if ('__TAURI_INTERNALS__' in window) { const { invoke } = await import('@tauri-apps/api/core'); setMessage(`Saved to ${await invoke<string>('export_csv', { filename, contents: csv })}`); }
      else { const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); setMessage('Forecast calculations exported as CSV.'); }
    } catch (e) { setSaveError(String(e)); }
  };
  return <main className="valuation-workspace" data-valuation-company={entry.id} data-valuation-kind={d.researchOrigin ? "reviewed" : d.starterOrigin ? "starter" : "manual"} data-valuation-starter={d.starterOrigin?.id ?? ""} data-valuation-ready={result.ready} data-valuation-study={d.researchOrigin?.id ?? ''}>
    <header className="valuation-heading"><div><button className="valuation-back" onClick={onProfile}><ArrowLeft size={15} />Company profile</button><div className="eyebrow">COMPANIES · VALUE & PRICE</div><h1>{entry.display_name}</h1><p>Three possible paths from your purchase price to shareholder cash.</p></div>
      <div className="valuation-actions"><button aria-expanded={showInputs} onClick={() => { setShowInputs(v => !v); setTab('scenarios'); }}>{showInputs ? 'Hide input forms' : 'Edit price & assumptions'}</button><button onClick={save}><Save size={15} />Save study revision</button><button onClick={exportCSV} disabled={!result.ready}><Download size={15} />Export calculations</button><small>{autoSaved ? 'Working draft saved on this device' : 'Working draft not saved'}</small></div></header>
    {saveError && <p role="alert" className="valuation-alert">{saveError}</p>}{message && <p role="status" className="valuation-status">{message}</p>}
    {researched && <section className="valuation-research-banner"><div><div className="eyebrow">RESEARCHED VALUATION · {researched.asOf}</div><strong>{d.researchOrigin?.id === researched.id ? changedFromResearch ? `Your edited ${entry.display_name} study` : `${entry.display_name} scenarios are ready` : `A researched ${entry.display_name} study is available`}</strong><p>Reported figures and analyst assumptions are included. <button onClick={() => setTab('evidence')}>Inspect sources and calculations</button></p></div><button className="valuation-research-reset" onClick={() => startingStudy && useStartingDraft(startingStudy, 'Researched scenarios opened.')}>Start from researched assumptions</button></section>}
    <section className="valuation-starter-banner">
      <div className="eyebrow">STANDARD CASH-FLOW MODEL</div>
      <strong>{d.researchOrigin ? 'Historical baseline available alongside the researched study' : d.starterOrigin ? changedFromStarter ? 'Your adjusted cash-flow model' : 'Historical cash-flow starter' : 'Standard starter scenarios available'}</strong>
      <p>Start from the latest annual cash flow, with ranges informed by historical cash variability where comparable evidence is available. Review the trend, later-year assumptions and a separate crisis case in this workspace.</p>
      {d.starterOrigin && d.starterOrigin.id !== 'empirical-cash-starter-v3' && <p>Your saved historical model is preserved. Choose <strong>Use empirical defaults</strong>, then apply, to open the new starter and retain this model as a revision.</p>}
      <button onClick={() => setTab('evidence')}>Inspect starter assumptions</button>
      {!d.researchOrigin && starter.issues.length > 0 && <ul className="valuation-starter-issues">{starter.issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul>}
      <details className="valuation-history-settings"><summary>Adjust history weights & range</summary>
        <p>Newest annual period first. Weights must total 100%. If fewer consecutive usable years exist, their included weights are normalized and disclosed in the source table.</p>
        <div className="valuation-trend-settings">
          <label>Historical years<select aria-label="Historical years" value={historyYears} onChange={e => { const years = Number(e.target.value) as 5 | 10; setHistoryYears(years); setHistoryWeights(defaultStarterWeights(years)); }}><option value={5}>5 years</option><option value={10}>10 years</option></select></label>
          <label>Projection<select aria-label="Projection" value={projection} onChange={e => setProjection(e.target.value as StarterSettings['projection'])}><option value="latest">Latest annual cash · flat baseline</option><option value="trend">Weighted historical trend</option><option value="flat">Flat weighted average</option></select></label>
          <label>Uncertainty range<select aria-label="Uncertainty range" value={rangeMode} onChange={e => setRangeMode(e.target.value as typeof rangeMode)}><option value="historical">Historical errors · 80% target</option><option value="percentage">Assumed percentage range</option></select></label>
          {rangeMode === 'historical' && <label>After year 4 · annual addition, % of historical cash scale<NumberInput label="Later-year widening (% of cash scale)" value={tailWidening} min={0} max={100} onChange={setTailWidening} /></label>}
          <label>{rangeMode === 'historical' ? 'Fallback first-year range · ±%' : 'First-year range · ±%'}<NumberInput label="First-year range (%)" value={spread} min={0} max={100} onChange={setSpread} /></label>
          <label>{rangeMode === 'historical' ? 'Fallback annual widening · percentage points' : 'Annual widening · percentage points'}<NumberInput label="Annual widening (percentage points)" value={spreadStep} min={0} max={100} onChange={setSpreadStep} /></label>
        </div>
        <div className="valuation-weight-grid">{historyWeights.map((weight, index) => <label key={index}>Year {index + 1} weight · %<NumberInput label={`Year ${index + 1} weight`} value={weight} min={0} max={100} onChange={value => setHistoryWeights(previous => previous.map((item, position) => position === index ? value : item))} /></label>)}
        </div>
        {rangeMode === 'historical' ? <p>Historical ranges require five comparable years, supported cash definitions and a tested projection. Year 1–4 widths use historical errors; later years add an explicitly assumed amount to the year-4 half-width. Other settings use the stated percentage fallback. The 80% target is exploratory and does not describe the probability of the full cash path or DCF.</p> : <p>{spread === null || spreadStep === null ? 'Complete the range settings.' : `Year 1: ±${spread}%; year 2: ±${spread + spreadStep}%; year 3: ±${spread + 2 * spreadStep}%.`} These are editable sensitivities, without an assigned confidence level.</p>}
        <p>History weights apply to the trend and weighted-average alternatives. COVID and rebound years are retained; document any normalization in a saved deep-dive revision.</p>
        <p>Total weight: {historyWeights.some(value => value === null) ? 'Incomplete' : `${historyWeights.reduce<number>((sum, value) => sum + value!, 0)}%`}. Applying these settings saves your current draft as a revision. Existing separate terminal inputs and required returns are retained for generic starters.</p>
        {settingsError && <p role="alert">{settingsError}</p>}
        <div className="valuation-history-actions"><button onClick={applyHistoryAssumptions}>Apply history assumptions</button><button onClick={() => { const settings = settingsForStarterOrigin(); setHistoryYears(settings.historyYears); setHistoryWeights(settings.weights); setProjection(settings.projection); setSpread(settings.spreadPercent); setSpreadStep(settings.spreadStepPercent); setRangeMode(settings.rangeMode ?? 'historical'); setTailWidening(settings.tailWideningPercent ?? 10); setSettingsError(''); }}>Use empirical defaults</button></div>
      </details>
    </section>
    {!showInputs && <section className="valuation-price-summary"><div><small>EQUITY PURCHASE BASIS</small><strong>{money(d.marketCap)}</strong><span>Price {d.priceDate} · valuation {d.valuationDate}</span></div><div><small>YOUR INVESTMENT</small><strong>{d.investment === null ? 'Unavailable' : `${format(d.investment, 0)} ${d.currency}`}</strong><span>{d.years} forecast years · editable scenarios</span></div><p>{d.priceSource}</p></section>}
    <div className="valuation-tabs" role="tablist" aria-label="Valuation sections"><button role="tab" aria-selected={tab === 'scenarios'} onClick={() => setTab('scenarios')}>Value, price & payback</button><button role="tab" aria-selected={tab === 'evidence'} onClick={() => setTab('evidence')}>Business, capital & evidence</button></div>
    {showInputs && <section className="valuation-card"><div className="valuation-basis">
      <label>Study title<input aria-label="Valuation study title" value={d.title} maxLength={160} onChange={e => update({ title: e.target.value })} /></label>
      <label>Valuation date<input aria-label="Valuation date" type="date" value={d.valuationDate} onChange={e => update({ valuationDate: e.target.value })} /></label>
      <label>Currency<input aria-label="Valuation currency" value={d.currency} maxLength={3} onChange={e => update({ currency: e.target.value.toUpperCase() })} /></label>
      <label>Equity market value · {d.currency} m<NumberInput label="Equity market value" value={d.marketCap} min={0} onChange={marketCap => update({ marketCap })} /></label>
      <label>Price date<input aria-label="Valuation price date" type="date" value={d.priceDate} onChange={e => update({ priceDate: e.target.value })} /></label>
      <label>Your investment · {d.currency}<NumberInput label="Valuation investment amount" value={d.investment} min={0} onChange={investment => update({ investment })} /></label>
      <label>Forecast years<NumberInput label="Valuation forecast years" value={d.years} min={1} max={50} step="1" onChange={changeYears} /></label>
    </div><label>Price source, share basis and currency conversion<textarea aria-label="Valuation price source" value={d.priceSource} maxLength={5000} onChange={e => update({ priceSource: e.target.value })} rows={2} /></label>
      {latestMarket && <p className="valuation-source">Saved reference: {format(latestMarket.sek, 2)} SEK m on {latestMarket.price_date}. This is a historical publication-window valuation. <button disabled={d.currency !== 'SEK'} onClick={useHistoricalPrice}>Use this dated SEK price</button></p>}
      {d.priceDate && d.priceDate < d.valuationDate && <p className="valuation-caution">The price predates this valuation. The comparison uses that dated price, not a live quote.</p>}
      <p className="valuation-method">Enter company amounts in millions of the selected currency. The model discounts <strong>forecast cash to common shareholders</strong>. Standard starters assume the historical cash-flow proxy is distributable; review reinvestment, interest, leases and other claims when refining it. Banks and insurers need capital-constrained equity-cash forecasts. Negative payments represent explicitly modeled shareholder funding, not a personal obligation.</p>
    </section>}
    {tab === 'scenarios' ? <>
      <CashFlowForecastChart draft={d} evidence={starter.evidence} matchesStarter={matchesStarterCash} />
      {showInputs && <section className="valuation-card"><h2>High, mid and low assumptions</h2><p>Enter each path, or generate a starting path and edit individual years. Required returns are equity discount rates. Payments occur at year-end.</p>
        <div className="valuation-scenario-grid">{scenarioKeys.map(key => <div className="valuation-scenario" key={key} style={{ borderTopColor: scenarioColors[key] }}><h3>{scenarioNames[key]}</h3><label>Required equity return · %<NumberInput label={`${scenarioNames[key]} required return`} value={d.scenarios[key].discountRate} min={0} max={100} onChange={discountRate => scenario(key, { discountRate })} /></label><TerminalValueInputs scenario={d.scenarios[key]} name={scenarioNames[key]} currency={d.currency} years={d.years} onChange={patch => scenario(key, patch)} /><label>Assumptions and evidence<textarea aria-label={`${scenarioNames[key]} assumptions`} value={d.scenarios[key].rationale} maxLength={5000} rows={3} onChange={e => scenario(key, { rationale: e.target.value })} /></label></div>)}</div>
        <details className="valuation-generator"><summary>Generate an editable cash-flow path</summary><div><label>Scenario<select aria-label="Generate valuation scenario" value={generator.scenario} onChange={e => setGenerator({ ...generator, scenario: e.target.value as ScenarioKey })}>{scenarioKeys.map(k => <option value={k} key={k}>{scenarioNames[k]}</option>)}</select></label><label>Year 1 payment · {d.currency} m<NumberInput label="Generated first cash payment" value={generator.first} onChange={first => setGenerator({ ...generator, first })} /></label><label>Annual change · %<NumberInput label="Generated cash payment growth" value={generator.growth} onChange={growth => setGenerator({ ...generator, growth })} /></label><button onClick={() => { try { if (generator.first === null || generator.growth === null) throw new Error('Enter a starting payment and growth assumption.'); scenario(generator.scenario, { cashFlows: generateCashFlows(generator.first, generator.growth, d.years) }); setMessage('Forecast path filled from your assumptions. Review the annual payments.'); } catch (e) { setSaveError(String(e)); } }}>Fill {scenarioNames[generator.scenario]} path</button></div><p>This replaces the selected path. Growth describes cash payments, after the investment required to support them.</p></details>
        <div className="table-scroll valuation-forecast-table"><table><thead><tr><th>Year after valuation</th>{scenarioKeys.map(k => <th key={k}>{scenarioNames[k]} payment · {d.currency} m</th>)}</tr></thead><tbody>{Array.from({ length: d.years }, (_, i) => <tr key={i}><th>Year {i + 1}</th>{scenarioKeys.map(k => <td key={k}><NumberInput label={`${scenarioNames[k]} year ${i + 1} cash payment`} value={d.scenarios[k].cashFlows[i] ?? null} onChange={n => { const cashFlows = [...d.scenarios[k].cashFlows]; cashFlows[i] = n; scenario(k, { cashFlows }); }} /></td>)}</tr>)}</tbody></table></div>
      </section>}
      <section className="valuation-card"><h2>What your {format(d.investment, 0)} {d.currency} could return</h2><p>Value includes the forecast payments and any final sale. Payback from cash payments is reported separately from recovery that depends on selling.</p>
        <div className="valuation-scenario-grid">{scenarioKeys.map(k => { const r = result.scenarios[k]; return <article className="valuation-result" data-scenario={k} key={k} style={{ borderTopColor: scenarioColors[k] }}><h3>{scenarioNames[k]} scenario</h3>{r.error ? <p className="valuation-incomplete">{r.error}</p> : <><small>ESTIMATED PRESENT VALUE</small><strong data-result="value">{money(r.value, true)}</strong><dl><div><dt>NPV after purchase price</dt><dd data-result="npv">{money(r.npv, true)}</dd></div><div><dt>Value / price</dt><dd>{format(r.valuePrice, 2)}×</dd></div><div><dt>Discount to estimated value</dt><dd>{r.discountToValue === null ? 'Unavailable' : `${format(r.discountToValue * 100, 1)}%`}</dd></div><div><dt>Cash-payment payback</dt><dd data-result="payback">{paybackLabel(r.payback, d.years)}</dd></div><div><dt>Discounted payback</dt><dd data-result="discounted-payback">{paybackLabel(r.discountedPayback, d.years)}</dd></div><div><dt>Payback including final sale</dt><dd>{paybackLabel(r.paybackWithSale, d.years)}</dd></div><div><dt>Discounted, including sale</dt><dd>{paybackLabel(r.discountedPaybackWithSale, d.years)}</dd></div></dl><p>Present value from final sale: {money(r.terminalPV, true)}{r.terminalShare !== null ? ` (${format(r.terminalShare * 100, 1)}% of estimated value)` : ''}.</p></>}</article>; })}</div>
        <p className="valuation-method">Payback is the first year-end cumulative cash covers the initial purchase. “Reverses later” means later funding takes it below cost again. No payback is extrapolated beyond {d.years} years, and the low case is not a guaranteed downside bound.</p>
      </section>
      <TerminalValueSummary draft={d} />
      <PurchaseRangePanel draft={d} companyId={entry.id} financialVersion={basis.financial} researchVersion={basis.release} shareReference={getPurchaseShareReference(entry.id, d, starter, researched)} onChange={purchaseRange => update({ purchaseRange })} />
      {result.crossing && <p className="valuation-caution">Some named scenario paths cross. Their original labels remain visible; the shaded range follows the actual minimum and maximum.</p>}
      {result.ready ? <ValuationCharts draft={d} results={result.scenarios} sale={sale} onSale={setSale} evidence={starter.evidence} matchesStarter={matchesStarterCash} matchesResearch={matchesResearchCash} /> : <div className="valuation-chart-placeholders">{["Discounted cash flow", "Cumulative NPV"].map(title => <section className="valuation-card" key={title}><h2>{title}</h2><p className="valuation-empty">Complete the dated price and all three cash-flow paths to draw this chart. Missing source inputs remain visible above; you can supply reviewed assumptions.</p></section>)}</div>}
      <section className="valuation-card"><h2>Liquidation or breakup recovery</h2><p>A separate alternative to continuing the business. Enter net proceeds for common shareholders after all prior claims, taxes, closure costs and cash burn. Support the estimates in Business, capital & evidence.</p><div className="valuation-scenario-grid">{scenarioKeys.map(k => { const r = result.scenarios[k]; return <div className="valuation-scenario" key={k} style={{ borderTopColor: scenarioColors[k] }}><h3>{scenarioNames[k]}</h3><label>Net equity recovery · {d.currency} m<NumberInput label={`${scenarioNames[k]} net equity recovery`} value={d.scenarios[k].recoveryEquity} min={0} onChange={recoveryEquity => scenario(k, { recoveryEquity })} /></label><label>Payment year · 0 = now<NumberInput label={`${scenarioNames[k]} recovery payment year`} value={d.scenarios[k].recoveryYear} min={0} max={50} step="1" onChange={recoveryYear => scenario(k, { recoveryYear })} /></label>{r.recoveryError && <p>{r.recoveryError}</p>}{r.recovery && <p data-recovery={k}>Present value: <strong>{money(r.recovery.value, true)}</strong><br />NPV: {money(r.recovery.npv, true)}<br />Payback: {r.recovery.payback === null ? 'Not reached by this recovery' : `Year ${r.recovery.payback}`}<br />Discounted: {r.recovery.discountedPayback === null ? 'Not reached by this recovery' : `Year ${r.recovery.discountedPayback}`}</p>}</div>; })}</div><p className="valuation-method">Recovery is not added to the going-concern value. It is not a guaranteed price floor; common shareholders may receive zero. This case assumes one net payment and uses each scenario's required equity return.</p></section>
      <CrisisScenarioPanel draft={d} onChange={crisis => update({ crisis })} />
    </> : <>
      {researched && <ValuationResearchEvidence study={researched} edited={changedFromResearch} />}
      <StarterValuationEvidence starter={starter} edited={changedFromStarter} />
      <section className="valuation-card"><h2>Tangible capital and financing</h2><p>Enter reconciled company amounts in {d.currency} millions. These explain the business economics and your ownership exposure; the equity cash-flow model already accounts for financing, so these amounts are not added to or deducted from its value again.</p>
        {latestReport && <p className="valuation-source">Available annual reference: {latestReport.start}–{latestReport.end}, reported {latestReport.report_date ?? 'date unavailable'}, {latestReport.currency} m, saved {latestReport.source_as_of}. Figures below require your reconciliation.</p>}
        <div className="valuation-capital-grid">{(Object.keys(capitalLabels) as (keyof typeof capitalLabels)[]).map(k => <label key={k}>{capitalLabels[k]} · {d.currency} m<NumberInput label={capitalLabels[k]} value={d.capital[k]} onChange={n => update({ capital: { ...d.capital, [k]: n } })} />{scale !== null && d.capital[k] !== null && <small>Your investment's exposure: {money(d.capital[k], true)}</small>}</label>)}</div>
        <p>Return on average tangible capital: <strong>{d.capital.averageTCE !== null && d.capital.averageTCE > 0 && d.capital.nopat !== null ? `${format(d.capital.nopat / d.capital.averageTCE * 100, 2)}%` : 'Unavailable'}</strong>. Match normalized annual NOPAT to the same period's average capital. A nonpositive denominator is unsuitable for this ratio.</p><p>Tangible book equity includes cash and is already net of liabilities. Explain required versus surplus cash, restricted balances, leases, maturities and other claims in the notes.</p>
      </section>
      <section className="valuation-card"><h2>Evidence behind the value</h2><p>Record source dates and locations, the company exposure, the transmission into each forecast and what would invalidate it. A listing country or branch assignment alone does not establish an exposure.</p>{(Object.keys(noteLabels) as (keyof ValuationDraft['notes'])[]).map(k => <label key={k}>{noteLabels[k]}<textarea aria-label={noteLabels[k]} value={d.notes[k]} maxLength={10000} rows={5} onChange={e => update({ notes: { ...d.notes, [k]: e.target.value } })} /></label>)}</section>
    </>}
    <section className="valuation-card"><h2>Saved studies for {entry.display_name}</h2><p>Working drafts reopen on this device. Save revisions to preserve earlier assumptions. Each revision retains the company, research release and financial-data versions used.</p><div className="valuation-saved">{saved.filter(v => v.company === entry.id).map(v => { const matches = compatibleValuation(v, basis.company, basis.release, basis.financial, basis.taxonomy); return <div key={v.id}><button disabled={!matches} onClick={() => { const draft = { ...structuredClone(v.draft), researchAutofillDisabled: true }; setDraft(draft); setShowInputs(!calculateValuation(draft).ready); setMessage('Opened saved assumptions with their original dates.'); }}><strong>{v.draft.title}</strong><span>{v.created.slice(0, 10)} · valuation {v.draft.valuationDate} · price {v.draft.priceDate || 'not entered'}</span></button>{!matches && <small>Different data version. Open the matching research and financial data to reuse this study.</small>}</div>; })}{!saved.some(v => v.company === entry.id) && <p>No saved revisions yet.</p>}</div></section>
    <details className="valuation-method"><summary>Model definitions and source versions</summary><p>This workspace uses an equity-distribution DCF: cash actually forecast to reach common shareholders, plus separately assumed net equity sale proceeds. Standard starters use a historical cash-flow proxy with explicit range assumptions. Historical 80%-target annual ranges do not establish probabilities for the full cash path or DCF. Reviewed studies refine those assumptions. This is not a verified FCFF calculation, and debt is not deducted twice. Use a justified final sale estimate and document its model. Nominal/real assumptions and all currencies must match.</p><p>NPV = present value of modeled cash receipts − equity price. Ordinary payback accumulates undiscounted payments; discounted payback accumulates present values. All payments are at year-end, so payback is reported in whole years without invented within-year timing. Scenario probabilities are not assigned.</p><p>Drafts and saved revisions are stored locally, separately from research packs. CSV exports contain the dated inputs and annual calculations. Review assumptions and evidence before treating a calculated value as an investment conclusion.</p><p className="valuation-hash">Listing {entry.id} · directory snapshot {entry.source_as_of}<br />Research {basis.release}<br />Financial data {basis.financial ?? 'Unavailable'}<br />Taxonomy {basis.taxonomy ?? 'Unavailable'}</p></details>
  </main>;
}
