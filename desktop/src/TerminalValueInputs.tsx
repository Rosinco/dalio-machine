import { resolveTerminalSale } from './terminalValue';
import { format } from './model';
import type { Scenario } from './valuation';

export default function TerminalValueInputs({ scenario: s, name, currency, years, onChange }: { scenario: Scenario; name: string; currency: string; years: number; onChange: (patch: Partial<Scenario>) => void }) {
  const amount = (event: React.ChangeEvent<HTMLInputElement>) => event.target.value === '' || !Number.isFinite(event.target.valueAsNumber) ? null : event.target.valueAsNumber;
  const terminal = resolveTerminalSale(s);
  return <div className="valuation-terminal-inputs">
    <label>Terminal value method<select aria-label={`${name} terminal value method`} value={s.terminalCash ? 'sustainable' : 'manual'} onChange={e => onChange(e.target.value === 'manual'
      ? { terminalCash: undefined, terminalEquity: terminal.value }
      : { terminalCash: { cashFlow: null, growthRate: 0 } })}><option value="sustainable">Separate sustainable cash</option><option value="manual">Explicit sale proceeds</option></select></label>
    {s.terminalCash && <>
      <label>Year {years + 1} sustainable equity cash · {currency} m<input type="number" step="any" aria-label={`${name} sustainable terminal cash`} value={s.terminalCash.cashFlow ?? ''} onChange={e => onChange({ terminalCash: { ...s.terminalCash!, cashFlow: amount(e) } })} /></label>
      <small>First annual cash after the forecast, after all required reinvestment and financing. It stays separate when forecast payments or range widths change.</small>
      <label>Mature cash growth · %<input type="number" step="any" aria-label={`${name} mature cash growth`} value={s.terminalCash.growthRate ?? ''} onChange={e => onChange({ terminalCash: { ...s.terminalCash!, growthRate: amount(e) } })} /></label>
      <small>Growth must be sustainable, supported by reinvestment, and below the required return. Review any jump from the last forecast payment ({format(s.cashFlows[years - 1], 2)} {currency} m).</small>
      {s.terminalCash.cashFlow !== null && s.terminalCash.cashFlow <= 0 && <p className="valuation-caution">Nonpositive sustainable cash gives zero assumed sale proceeds. Review a turnaround or finite-life case; this does not estimate recovery.</p>}
    </>}
    <label>Final equity sale · {currency} m<input type="number" step="any" min={0} aria-label={`${name} final equity sale`} value={terminal.value ?? ''} onChange={e => onChange({ terminalCash: undefined, terminalEquity: amount(e) })} /></label>
    <small>{s.terminalCash ? 'Calculated from sustainable cash ÷ (required return − mature growth). Editing sale proceeds switches to an explicit sale assumption. ' : ''}Net sale at year {years}, in addition to that year's payment. Zero means no sale.</small>
  </div>;
}
