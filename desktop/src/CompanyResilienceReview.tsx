import { useState } from 'react';
import { blankLiquidityRow, blankResilienceReview, calculateImpairment, calculateLiquidity, getReviewStatus, loadResilienceReview, resilienceReviewKey, saveResilienceReview, type ImpairmentAssumptions, type LiquidityRow, type ResilienceBasis, type ResilienceReview, type ReviewConclusion } from './resilienceReview';
import './resilienceReview.css';

type Props = ResilienceBasis;
// Authored work survives internal navigation without writing browser storage.
// A changed source basis stays inside the draft until explicit reassessment.
const unsavedReviews = new Map<string, { record: ResilienceReview; raw: string | null }>();
const amount = (value: number | null) => value === null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
function NumberField({ label, value, onChange, min }: { label: string; value: number | null; onChange: (v: number | null) => void; min?: number }) {
  return <label>{label}<input type="number" step="any" min={min} aria-label={label} value={value ?? ''} onChange={e => onChange(e.target.value === '' || !Number.isFinite(e.target.valueAsNumber) ? null : e.target.valueAsNumber)} /></label>;
}
function TextField({ label, value, onChange, help }: { label: string; value: string; onChange: (v: string) => void; help?: string }) {
  return <label>{label}<textarea aria-label={label} rows={3} maxLength={10000} value={value} onChange={e => onChange(e.target.value)} />{help && <span className="resilience-help">{help}</span>}</label>;
}

/** Its own company key keeps this review independent from notebook and Value drafts. */
export default function CompanyResilienceReview(props: Props) {
  return <ReviewEditor key={`${props.companyId}:${props.releaseId}:${props.financialId}:${props.sourceAsOf}`} {...props} />;
}

function ReviewEditor(props: Props) {
  const [initial] = useState(() => {
    const cached = unsavedReviews.get(props.companyId);
    try { const record = loadResilienceReview(localStorage, props.companyId); return { record: cached?.record ?? record ?? blankResilienceReview(props), raw: cached ? cached.raw : localStorage.getItem(resilienceReviewKey(props.companyId)), dirty: !!cached, error: '' }; }
    catch (e) { return { record: cached?.record ?? blankResilienceReview(props), raw: cached?.raw ?? null, dirty: !!cached, error: `${String(e)} No saved content has been replaced.` }; }
  });
  const [review, setReview] = useState<ResilienceReview>(initial.record);
  const [savedRaw, setSavedRaw] = useState(initial.raw);
  const [protectedError, setProtectedError] = useState(initial.error);
  const [dirty, setDirty] = useState(initial.dirty), [error, setError] = useState(''), [message, setMessage] = useState('');
  const update = (next: ResilienceReview) => { unsavedReviews.set(props.companyId, { record: next, raw: savedRaw }); setReview(next); setDirty(true); setMessage(''); setError(''); };
  const field = <K extends keyof ResilienceReview>(key: K, value: ResilienceReview[K]) => update({ ...review, [key]: value });
  const impairmentField = <K extends keyof ImpairmentAssumptions>(key: K, value: ImpairmentAssumptions[K]) => field('impairment', { ...review.impairment, [key]: value });
  const liquidity = calculateLiquidity(review.liquidity), impairment = calculateImpairment(review.impairment, review.liquidity.startDate), status = getReviewStatus(review, props);
  const units = /^[A-Z]{3}$/.test(review.currency) ? `${review.currency} million` : 'currency millions';
  const rowField = (index: number, key: keyof LiquidityRow, value: number | null) => field('liquidity', { ...review.liquidity, rows: review.liquidity.rows.map((row, i) => i === index ? { ...row, [key]: value } : row) });
  const save = () => {
    const next = { ...review, updatedAt: new Date().toISOString() };
    try { const raw = saveResilienceReview(localStorage, next, savedRaw); unsavedReviews.delete(props.companyId); setSavedRaw(raw); setReview(next); setDirty(false); setError(''); setMessage('Company resilience review saved locally.'); window.dispatchEvent(new Event('macro-atlas-resilience-saved')); }
    catch (e) { setError(`Could not save: ${String(e)} Your current edits remain on screen.`); if (String(e).includes('unreadable or unsupported')) setProtectedError(String(e)); }
  };
  const reload = () => {
    try { const record = loadResilienceReview(localStorage, props.companyId); const raw = localStorage.getItem(resilienceReviewKey(props.companyId)); unsavedReviews.delete(props.companyId); setReview(record ?? blankResilienceReview(props)); setSavedRaw(raw); setDirty(false); setError(''); setMessage('Saved review reloaded; unsaved edits discarded.'); }
    catch (e) { setError(`Could not reload: ${String(e)} Your current edits remain on screen.`); if (String(e).includes('unreadable or unsupported')) setProtectedError(String(e)); }
  };
  return <section className="company-resilience" aria-label="Company survival and permanent-loss review" data-resilience-company={props.companyId}>
    <div className="resilience-heading"><div><div className="eyebrow">COMPANY RESILIENCE</div><h2>Survival and permanent loss</h2></div><span className={`resilience-status resilience-status-${status.code}`} data-resilience-status={protectedError ? 'unavailable' : status.code}>{protectedError ? 'Saved review unavailable' : status.label}</span></div>
    <p>Review whether {props.companyName} can meet its obligations under combined adverse conditions, then value a separate permanent-impairment outcome. These are authored scenarios with no assigned probabilities.</p>
    <p className="resilience-help">Inputs use one currency in millions. Sources, debt maturities, covenants and rescue terms need manual review; this form does not fetch live evidence or interpret contracts.</p>
    {protectedError && <p role="alert" className="resilience-error">{protectedError} Editing is disabled to preserve the existing record.</p>}
    <fieldset disabled={!!protectedError} className="resilience-fieldset">
      {status.stale && <div className="resilience-caution"><p>{status.detail}</p><button type="button" onClick={() => update({ ...review, releaseId: props.releaseId, financialId: props.financialId, sourceAsOf: props.sourceAsOf, conclusion: 'unassessed' })}>Start reassessment using current sources</button><p className="resilience-help">Keeps all authored amounts and evidence, and resets the conclusion to unassessed. Save after reviewing them.</p></div>}
      <div className="resilience-fields">
        <label>Review currency<input aria-label="Resilience review currency" value={review.currency} maxLength={3} placeholder="SEK" onChange={e => field('currency', e.target.value.toUpperCase())} /></label>
        <label>Liquidity evidence date<input type="date" aria-label="Liquidity evidence date" value={review.evidenceDate} onChange={e => field('evidenceDate', e.target.value)} /></label>
      </div>
      <TextField label="Liquidity source evidence" value={review.evidenceSources} onChange={value => field('evidenceSources', value)} help="Name dated reports, debt maturity tables, facility terms and relevant pages or links. Keep observed facts separate from the scenario assumptions below." />
      <TextField label="Combined shock assumptions" value={review.shockNarrative} onChange={value => field('shockNarrative', value)} help="Describe adverse events occurring together, their duration and cash effects. Explain what is already included in operating cash so the extra drain does not count it again." />
      <TextField label="Funding availability and covenant constraints" value={review.covenantConstraints} onChange={value => field('covenantConstraints', value)} help="Record restrictions, covenant tests, maturities and why committed facilities remain drawable in this scenario. Uncommitted refinancing and hoped-for rescue funding are not available funding." />

      <h3>Liquidity through the shock</h3>
      <p>Operating cash is after mandatory spending, maintenance investment, interest, tax and leases, before debt principal and new borrowing. Enter principal, incremental funding drawn in each period from committed facilities available in this scenario, and any additional combined shock drain separately.</p>
      <p className="resilience-help">Funding is the new cash drawn during that period, not the total facility limit. Cumulative draws must respect remaining capacity and repayment terms. Do not repeat the same undrawn capacity across periods or include funds already in opening cash.</p>
      <div className="resilience-fields">
        <label>Equal periods<select aria-label="Liquidity period frequency" value={review.liquidity.frequency} onChange={e => field('liquidity', { ...review.liquidity, frequency: e.target.value as 'quarterly' | 'yearly' })}><option value="quarterly">Quarterly</option><option value="yearly">Yearly</option></select></label>
        <label>Opening date<input type="date" aria-label="Liquidity opening date" value={review.liquidity.startDate} onChange={e => field('liquidity', { ...review.liquidity, startDate: e.target.value })} /></label>
        <NumberField label="Opening unrestricted cash" value={review.liquidity.openingCash} min={0} onChange={value => field('liquidity', { ...review.liquidity, openingCash: value })} />
        <NumberField label="Minimum operating liquidity" value={review.liquidity.minimumCash} min={0} onChange={value => field('liquidity', { ...review.liquidity, minimumCash: value })} />
      </div>
      <p className="resilience-help">All amounts are {units}. Blanks mean unknown; enter zero explicitly where supported. Changing period length does not convert the entered amounts.</p>
      <div className="resilience-periods">{review.liquidity.rows.map((row, i) => <div className="resilience-period" key={i}>
        <h4>{review.liquidity.frequency === 'quarterly' ? 'Quarter' : 'Year'} {i + 1}</h4><div className="resilience-fields">
          <NumberField label={`Period ${i + 1} net operating cash`} value={row.operatingCash} onChange={v => rowField(i, 'operatingCash', v)} />
          <NumberField label={`Period ${i + 1} principal due`} value={row.principalDue} min={0} onChange={v => rowField(i, 'principalDue', v)} />
          <NumberField label={`Period ${i + 1} committed funding drawn`} value={row.committedFunding} min={0} onChange={v => rowField(i, 'committedFunding', v)} />
          <NumberField label={`Period ${i + 1} extra shock drain`} value={row.shockDrain} min={0} onChange={v => rowField(i, 'shockDrain', v)} />
        </div>{liquidity.rows[i] && <p className={liquidity.rows[i].headroom < 0 ? 'resilience-error' : 'resilience-period-result'}>Endpoint cash: <strong>{amount(liquidity.rows[i].cash)}</strong> · headroom above minimum: <strong>{amount(liquidity.rows[i].headroom)}</strong> {units}</p>}
      </div>)}</div>
      <div className="resilience-actions"><button type="button" disabled={review.liquidity.rows.length >= 40} onClick={() => field('liquidity', { ...review.liquidity, rows: [...review.liquidity.rows, blankLiquidityRow()] })}>Add period</button><button type="button" disabled={review.liquidity.rows.length <= 1} onClick={() => field('liquidity', { ...review.liquidity, rows: review.liquidity.rows.slice(0, -1) })}>Remove last period</button></div>
      {liquidity.error ? <p className="resilience-caution" data-liquidity-error>{liquidity.error}</p> : <div className="resilience-results" data-liquidity-results>
        <p>First endpoint breach<strong>{liquidity.firstBreach === null ? 'None in the stated path' : liquidity.firstBreach === 0 ? 'Already below minimum at opening' : `Period ${liquidity.firstBreach}`}</strong></p>
        <p>Additional constant drain reaching minimum liquidity<strong data-resilience-extra-drain>{amount(liquidity.additionalDrainPerPeriod)} {units} per {review.liquidity.frequency === 'quarterly' ? 'quarter' : 'year'}</strong></p>
      </div>}
      <p className="resilience-help">The extra-drain threshold is the smallest endpoint headroom divided by the number of elapsed periods; it is zero if opening liquidity is below the minimum or an endpoint has no remaining headroom. It applies at every period end, on top of entered shocks, with funding held fixed. This is an endpoint calculation: within-period cash shortages, covenant breaches and changes in funding access are not automated. A later inflow does not remove an earlier breach.</p>

      <h3>Separate permanent-impairment outcome</h3>
      <p className="resilience-help">Present values are at {review.liquidity.startDate || 'the opening date (not entered)'}. The first annual payment arrives one year after that date. A price from after the opening date cannot be used for this NPV.</p>
      <p>Enter annual cash reaching the <strong>whole existing shareholder claim</strong> after financing costs and any rescue dilution, in {units}. The remaining stake and final proceeds must refer to that same claim. Do not enter the whole recapitalized company's cash if new investors own part of it.</p>
      <TextField label="Existing ownership after financing and rescue" value={review.impairment.ownership} onChange={v => impairmentField('ownership', v)} help="Explain the existing shareholders' retained claim, dilution and any additional funding they must choose to supply. Amounts are after these effects; the form does not calculate dilution." />
      <TextField label="Permanent-impairment assumptions" value={review.impairment.assumptions} onChange={v => impairmentField('assumptions', v)} help="Describe lasting lost earnings, closure, restructuring, asset realization or zero recovery. No automatic return to the previous cash path is assumed." />
      <div className="resilience-fields"><label>Impairment evidence date<input type="date" aria-label="Impairment evidence date" value={review.impairment.evidenceDate} onChange={e => impairmentField('evidenceDate', e.target.value)} /></label></div>
      <TextField label="Impairment source evidence" value={review.impairment.evidenceSources} onChange={v => impairmentField('evidenceSources', v)} />
      <TextField label="Annual cash to the existing claim" value={review.impairment.annualCash} onChange={v => impairmentField('annualCash', v)} help="One explicit amount per year, separated by commas or newlines, using decimal points. All payments are at year-end from the opening date above. Negative amounts mean modeled additional funding, not an automatic personal obligation." />
      <div className="resilience-fields">
        <NumberField label="Impairment required annual return (%)" value={review.impairment.discountRate} min={0} onChange={v => impairmentField('discountRate', v)} />
        <NumberField label="Final net proceeds to the existing claim" value={review.impairment.finalProceeds} min={0} onChange={v => impairmentField('finalProceeds', v)} />
        <NumberField label="Optional purchase price of the existing claim" value={review.impairment.purchaseEquity} min={0} onChange={v => impairmentField('purchaseEquity', v)} />
        <label>Purchase price date<input type="date" aria-label="Impairment purchase price date" value={review.impairment.priceDate} onChange={e => impairmentField('priceDate', e.target.value)} /></label>
      </div>
      <TextField label="Purchase price source and ownership basis" value={review.impairment.priceSource} onChange={v => impairmentField('priceSource', v)} help="Required for NPV when a purchase amount is entered. Use the same currency and existing equity claim as the cash flows." />
      {impairment.error ? <p className="resilience-caution" data-impairment-error>{impairment.error}</p> : <div className="resilience-results" data-impairment-results>
        <p>Present value of annual cash<strong>{amount(impairment.cashPV)} {units}</strong></p><p>Present value of final proceeds<strong>{amount(impairment.finalPV)} {units}</strong></p><p>Separate impairment value<strong data-impairment-value>{amount(impairment.value)} {units}</strong></p><p>NPV at the entered price<strong data-impairment-npv>{impairment.npv === null ? 'Unavailable without a valid dated price' : `${amount(impairment.npv)} ${units}`}</strong></p>
      </div>}
      {impairment.priceError && <p className="resilience-caution">{impairment.priceError}</p>}
      <p className="resilience-help">Final net proceeds enter once at the end of the last cash-flow year, separately from that year's payment. Explicit zero recovery is allowed. This alternative is not added to the continuing-business DCF, and no probability or price floor is assigned.</p>

      <h3>Authored conclusion</h3>
      <label>Review conclusion<select aria-label="Resilience review conclusion" value={review.conclusion} onChange={e => field('conclusion', e.target.value as ReviewConclusion)}><option value="unassessed">Unassessed</option><option value="unresolved">Unresolved</option><option value="material-failure">Material failure identified</option><option value="reviewed-for-stated-scenario">Reviewed for the stated scenarios</option></select></label>
      <TextField label="Conclusion, unresolved questions and monitoring evidence" value={review.conclusionRationale} onChange={v => field('conclusionRationale', v)} help="Explain what could cause permanent loss, any unresolved financing or covenant issue, and what evidence would require reassessment. Leave the conclusion unresolved while material questions remain." />
      <p className="resilience-caution" data-resilience-detail><strong>{dirty ? 'Draft status' : 'Review status'}: {status.label}.</strong> {status.detail}</p>
      <div className="resilience-actions"><button type="button" onClick={save} disabled={!dirty}>Save resilience review</button><span role="status">{dirty ? 'Unsaved changes survive navigation in this session. Save before closing or reloading the app.' : message || (savedRaw ? 'Saved locally; no edits pending.' : 'No saved review yet.')}</span></div>
      {error && <p role="alert" className="resilience-error">{error}</p>}
      {dirty && error && <button type="button" onClick={reload}>Reload saved review and discard unsaved edits</button>}
    </fieldset>
    <details className="resilience-source-basis"><summary>Saved review source basis</summary><p>Company {review.companyId} · source date {review.sourceAsOf}<br />Research {review.releaseId}<br />Financial data {review.financialId ?? 'Unavailable'}<br />Last saved {review.updatedAt || 'Not saved'}</p><p>This local record is separate from the notebook, provider evidence and valuation drafts. Opening this panel does not save or migrate it.</p></details>
  </section>;
}
