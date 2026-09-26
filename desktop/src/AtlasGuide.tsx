import { useEffect, useRef } from 'react';
import { ArrowRight, X } from 'lucide-react';
import type { Observatory } from './business';

export default function AtlasGuide({ onExplore, onClose }: { onExplore: (view: Observatory) => void; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.current?.showModal();
    return () => previous?.focus();
  }, []);
  return <dialog ref={dialog} className="atlas-guide" aria-labelledby="atlas-guide-title" onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) { const bounds = event.currentTarget.getBoundingClientRect(); if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onClose(); } }}>
    <div className="atlas-guide-heading"><div><span className="eyebrow">START HERE</span><h2 id="atlas-guide-title">What would you like to understand?</h2></div><button aria-label="Close getting started guide" onClick={onClose}><X size={20} /></button></div>
    <p>Atlas connects the economy, industries and companies using saved data. Start with a question, then follow the evidence.</p>
    <div className="atlas-guide-paths">
      <button onClick={() => onExplore('companies')}><strong>Find companies to research <ArrowRight size={16} /></strong><span>Open Lists. Choose useful KPIs, set ranges and keep a watchlist.</span></button>
      <button onClick={() => onExplore('sectors')}><strong>Understand an industry <ArrowRight size={16} /></strong><span>Browse sectors and branches, then compare their companies.</span></button>
      <button onClick={() => onExplore('macro')}><strong>Explore the economy <ArrowRight size={16} /></strong><span>Select a country. Read its indicators, history and trade relationships.</span></button>
    </div>
    <h3>A company research workflow</h3>
    <ol className="atlas-guide-steps">
      <li><strong>Find and narrow — Lists.</strong> Choose columns, then enter Min and Max. A KPI is a key performance indicator: one measure of a business or its shares. Click a heading’s information button for its definition.</li>
      <li><strong>Understand the business — Financials.</strong> Open a company’s name. Read sales and profitability first, then cash generation and investment spending, then assets and debt. The chart sections explain what the figures mean and which questions to ask.</li>
      <li><strong>Check survival — Research notes.</strong> Record available cash, debt payments and combined shocks in the survival review. Test what exhausts liquidity, then model a separate permanent-loss case for existing shareholders. Review shared portfolio exposures before accepting a price comparison.</li>
      <li><strong>Test the price — Valuation.</strong> Review cash assumptions, then compare the value of that cash with a dated share price. The starting scenarios need your review.</li>
      <li><strong>Keep the reasoning — Research notes.</strong> Record how the business works, explain the numbers and list what you still need to verify. Notes save on this device as you type. Available archived analyses remain below your notebook.</li>
    </ol>
    <details><summary>How to read the numbers and charts</summary><p>Check the unit and date first. “Millions” means a whole-company amount; “per share” means one share. A downloaded date tells you when data was saved, which can differ from the financial period or price date.</p><p>Read time charts from left to right. The vertical axis shows the selected measure. Gaps mean missing data. Forecast lines and ranges show assumptions about future outcomes; a wider range means greater modeled uncertainty. They are not guaranteed limits.</p><p>DCF is discounted cash flow: future cash expressed as a value today. Terminal value represents cash after the explicit forecast and is already included in DCF. NPV is that value minus the proposed purchase price. A margin of safety lowers the price you are willing to pay below the assumed value. Historical range coverage does not establish a probability for the whole DCF; modeled Low is not a loss limit. Leave catastrophe probabilities unset when evidence cannot support them.</p></details>
    <p className="atlas-guide-footnote">Use Back and Forward at the top left to retrace screens and chart sections while keeping your place. Alt+Left and Alt+Right are keyboard shortcuts. Back to Lists opens your company list directly. Your notes and valuation edits stay saved when you go back.</p>
    <p className="atlas-guide-footnote">Stars keep a watchlist and saved views keep columns and filters. Atlas reopens your last screen after a restart; Back and Forward begin a new session.</p>
  </dialog>;
}
