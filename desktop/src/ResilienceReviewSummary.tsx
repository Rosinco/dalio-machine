import { getReviewStatus, loadResilienceReview } from './resilienceReview';

/** Read-only companion to the valuation; opening it never creates a review. */
export default function ResilienceReviewSummary({ companyId, releaseId, financialId, sourceAsOf, onReview }: {
  companyId: string; releaseId: string; financialId: string | null; sourceAsOf: string; onReview: () => void;
}) {
  let status;
  try { status = getReviewStatus(loadResilienceReview(localStorage, companyId), { releaseId, financialId, sourceAsOf }); }
  catch { status = { code: 'unavailable', label: 'Saved review unavailable', detail: 'The saved review could not be read and has been preserved. Resolve it before relying on a valuation.', reviewed: false }; }
  return <section className="valuation-card resilience-review-summary" aria-label="Survival review status" data-review-status={status.code}>
    <h3>Survival and permanent loss · {status.label}</h3>
    <p>{status.detail}</p>
    <p>DCF and NPV describe the entered cash assumptions. A price discount cannot resolve a funding shortfall or establish a loss limit. The portfolio review must also consider holdings exposed to the same shock.</p>
    <button onClick={onReview}>Review survival &amp; permanent loss</button>
  </section>;
}
