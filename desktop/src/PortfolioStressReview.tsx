import { useState } from 'react';
import { blankPortfolioStress, blankPortfolioStressRow, calculatePortfolioStress, loadPortfolioStress, maxPortfolioStressRows, restorePortfolioStressSession, savePortfolioStress, type PortfolioStressDraft, type PortfolioStressRow, type PortfolioStressSession } from './portfolioStress';
import './portfolioStress.css';

// Navigation preserves unsaved work in this running app; only Save writes storage.
let transientPortfolioStress: PortfolioStressSession | null = null;

function initialState() {
  try {
    const saved = typeof window === 'undefined' ? { draft: blankPortfolioStress(), raw: null, savedAt: '' } : loadPortfolioStress(window.localStorage);
    const session = restorePortfolioStressSession(saved, transientPortfolioStress);
    return { ...session, blocked: '', storageMessage: session.storageChanged ? 'Saved portfolio stress changed in another view. Your unsaved session edits are retained. Copy them before reloading; Save will not replace the newer saved work.' : '' };
  } catch (error) {
    return { ...(transientPortfolioStress ?? { draft: blankPortfolioStress(), raw: null, savedAt: '', dirty: false }), blocked: `Portfolio stress could not be loaded. Existing saved work is preserved and editing is disabled. ${String(error)}`, storageMessage: '' };
  }
}

const percentage = (value: number) => `${value.toLocaleString('en-US', { maximumFractionDigits: 2 })}%`;

/** One manual portfolio worksheet, shared across company research views. */
export default function PortfolioStressReview() {
  const [initial] = useState(initialState);
  const [draft, setDraft] = useState(initial.draft);
  const [raw, setRaw] = useState(initial.raw);
  const [savedAt, setSavedAt] = useState(initial.savedAt);
  const [blocked, setBlocked] = useState(initial.blocked);
  const [dirty, setDirty] = useState(initial.dirty);
  const [saveError, setSaveError] = useState(initial.storageMessage);
  const [savedMessage, setSavedMessage] = useState('');
  const { errors, result } = calculatePortfolioStress(draft);

  const change = (patch: Partial<PortfolioStressDraft>) => {
    if (blocked) return;
    const nextDraft = { ...draft, ...patch };
    transientPortfolioStress = { draft: nextDraft, raw, savedAt, dirty: true };
    setDraft(nextDraft); setDirty(true); setSaveError(''); setSavedMessage('');
  };
  const changeRow = (id: string, patch: Partial<PortfolioStressRow>) => change({ rows: draft.rows.map(row => row.id === id ? { ...row, ...patch } : row) });
  const addRow = () => {
    let index = 1;
    while (draft.rows.some(row => row.id === `position-${index}`)) index++;
    change({ rows: [...draft.rows, blankPortfolioStressRow(`position-${index}`)] });
  };
  const save = () => {
    if (blocked) return;
    try {
      const timestamp = new Date().toISOString();
      const nextRaw = savePortfolioStress(window.localStorage, draft, raw, timestamp);
      transientPortfolioStress = { draft, raw: nextRaw, savedAt: timestamp, dirty: false };
      setRaw(nextRaw); setSavedAt(timestamp); setDirty(false); setSaveError(''); setSavedMessage('Portfolio stress saved locally.');
    } catch (error) {
      setSavedMessage(''); setSaveError(`Could not save portfolio stress. Your current edits remain visible. ${String(error)}`);
      try { loadPortfolioStress(window.localStorage); } catch (readError) {
        setBlocked(`Existing saved work cannot be safely read. It is preserved and editing is disabled. ${String(readError)}`);
      }
    }
  };

  return <section className="portfolio-stress" aria-labelledby="portfolio-stress-title" data-portfolio-stress="true">
    <div className="portfolio-stress-heading"><div><div className="eyebrow">PORTFOLIO EXPOSURES · MANUAL REVIEW</div><h2 id="portfolio-stress-title">Shared-shock portfolio stress</h2></div>
      <button type="button" onClick={save} disabled={Boolean(blocked) || !dirty}>Save portfolio stress</button>
    </div>
    <p>How much starting capital could be lost if several positions suffer the same shock? Enter your own dated weights and loss assumptions for one simultaneous scenario. This worksheet is shared across companies and saved separately on this device.</p>
    <p>No holdings are imported. The result has no assigned probability, does not estimate the worst possible loss, and does not change your position-sizing policy.</p>
    {blocked && <p className="portfolio-stress-error" role="alert">{blocked}</p>}
    <fieldset disabled={Boolean(blocked)} className="portfolio-stress-fields"><legend className="visually-hidden">Manual portfolio stress inputs</legend>
      <div className="portfolio-stress-grid">
        <label>Starting-weight date<input type="date" aria-label="Portfolio starting-weight date" value={draft.exposureDate} onChange={event => change({ exposureDate: event.target.value })} /></label>
        <label>Your tolerable loss · % of starting capital<input type="number" min="0" max="100" step="any" aria-label="Portfolio tolerable loss (%)" value={draft.tolerancePercent} onChange={event => change({ tolerancePercent: event.target.value })} /></label>
      </div>
      <label>Exposure basis and source<textarea rows={2} maxLength={10000} aria-label="Portfolio exposure basis and source" value={draft.exposureBasis} onChange={event => change({ exposureBasis: event.target.value })} /></label>
      <p className="portfolio-stress-help">Identify the account or portfolio, valuation date, source and ownership basis. Each weight is a percentage of the same starting portfolio capital.</p>
      <label>Shared shock or combined shock scenario<textarea rows={2} maxLength={10000} aria-label="Portfolio shared shock" value={draft.shock} onChange={event => change({ shock: event.target.value })} /></label>
      <div className="portfolio-stress-grid">
        <label>Exposure evidence and sources<textarea rows={3} maxLength={10000} aria-label="Portfolio exposure evidence and sources" value={draft.evidence} onChange={event => change({ evidence: event.target.value })} /></label>
        <label>Assumed severity and loss rationale<textarea rows={3} maxLength={10000} aria-label="Portfolio stress assumptions" value={draft.assumptions} onChange={event => change({ assumptions: event.target.value })} /></label>
      </div>
      <p className="portfolio-stress-help">Keep observed exposures separate from assumed outcomes. State when evidence is missing. Review common customers, lenders, jurisdictions and industry demand.</p>
      <div className="portfolio-stress-heading"><h3>Positions exposed to this scenario</h3><button type="button" onClick={addRow} disabled={draft.rows.length >= maxPortfolioStressRows}>Add exposure row</button></div>
      <p>Enter each position once, with its combined loss under this scenario. Do not enter the same weight again for each exposure driver. All entered rows must be complete and total weights cannot exceed 100%.</p>
      <div className="portfolio-stress-rows">{draft.rows.map((row, index) => <fieldset className="portfolio-stress-row" key={row.id}>
        <legend>Exposure {index + 1}</legend>
        <div className="portfolio-stress-row-grid">
          <label>Position name<input maxLength={200} aria-label={`Portfolio position ${index + 1} name`} value={row.name} onChange={event => changeRow(row.id, { name: event.target.value })} /></label>
          <label>Shared exposure driver<input maxLength={1000} aria-label={`Portfolio position ${index + 1} driver`} value={row.driver} onChange={event => changeRow(row.id, { driver: event.target.value })} /></label>
          <label>Starting weight · %<input type="number" min="0" max="100" step="any" aria-label={`Portfolio position ${index + 1} weight (%)`} value={row.weightPercent} onChange={event => changeRow(row.id, { weightPercent: event.target.value })} /></label>
          <label>Assumed position loss · %<input type="number" min="0" max="100" step="any" aria-label={`Portfolio position ${index + 1} loss (%)`} value={row.lossPercent} onChange={event => changeRow(row.id, { lossPercent: event.target.value })} /></label>
        </div>
        <label>Position notes · optional<textarea rows={2} maxLength={5000} aria-label={`Portfolio position ${index + 1} notes`} value={row.notes} onChange={event => changeRow(row.id, { notes: event.target.value })} /></label>
        <button type="button" className="portfolio-stress-remove" aria-label={`Remove portfolio exposure ${index + 1}`} onClick={() => change({ rows: draft.rows.filter(item => item.id !== row.id) })}>Remove exposure {index + 1}</button>
      </fieldset>)}</div>
      <label>Outside-model risks and missing exposures · optional<textarea rows={3} maxLength={10000} aria-label="Portfolio outside-model risks" value={draft.outsideModelRisks} onChange={event => change({ outsideModelRisks: event.target.value })} /></label>
    </fieldset>
    <div className="portfolio-stress-assumption"><strong>Residual-exposure assumption</strong><p>Any unentered weight is unmodeled portfolio exposure, not identified as cash. Remaining capital assumes that residual exposure stays flat. Further losses there would reduce the result.</p></div>
    {!blocked && result ? <div className="portfolio-stress-result" data-portfolio-stress-result="true" data-loss-percent={result.lossPercent}>
      <dl className="portfolio-stress-results-grid">
        <div><dt>Loss of starting capital</dt><dd>{percentage(result.lossPercent)}</dd></div>
        <div><dt>Remaining capital · residual held flat</dt><dd>{percentage(result.remainingCapitalPercent)}</dd></div>
        <div><dt>Entered starting weight</dt><dd>{percentage(result.enteredWeightPercent)}</dd></div>
        <div><dt>Unmodeled residual exposure</dt><dd>{percentage(result.residualWeightPercent)}</dd></div>
      </dl>
      <p>{result.toleranceComparison === 'equal' ? `The scenario loss equals your entered tolerance of ${percentage(result.tolerancePercent)}.` : `The scenario loss is ${Math.abs(result.toleranceDifferencePercent).toLocaleString('en-US', { maximumFractionDigits: 2 })} percentage points ${result.toleranceComparison} your entered tolerance of ${percentage(result.tolerancePercent)}.`} This comparison describes your inputs; it does not establish that the portfolio is safe.</p>
    </div> : !blocked && <div className="portfolio-stress-incomplete"><strong>Complete the exposure basis and all rows to calculate this scenario.</strong><details><summary>Inputs needing attention ({errors.length})</summary><ul>{errors.map(error => <li key={error}>{error}</li>)}</ul></details></div>}
    <p className="portfolio-stress-help">Calculation: sum of each starting weight × its assumed loss ÷ 100. Remaining capital = 100% − that loss. Losses occur together and are added against the same starting capital; they are not compounded. This loss-only worksheet covers 0–100% position losses. Borrowing, short positions or liabilities that could exceed invested capital require a separate analysis.</p>
    <p className="portfolio-stress-save-state">{blocked ? 'Editing is blocked. Saved bytes have not been replaced.' : <>{dirty ? 'Unsaved edits are retained while navigating this app. Save before reloading, restarting or closing.' : savedAt ? `Last saved ${new Date(savedAt).toLocaleString()}.` : 'No saved worksheet yet.'} Incomplete drafts can be saved.</>}</p>
    {savedMessage && <p role="status">{savedMessage}</p>}
    {saveError && <p className="portfolio-stress-error" role="alert">{saveError}</p>}
  </section>;
}
