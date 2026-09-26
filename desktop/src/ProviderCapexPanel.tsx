import type { FinancialIndex } from './financialData';
import type { CompanyListColumn } from './companyListModel';
import { expandedKpiCell, EXPANDED_KPI_MANIFEST } from './expandedKpis';
import { useExpandedKpis } from './useExpandedKpis';

const latest: CompanyListColumn = { id: 'capex-latest', kpiId: 'provider_64', window: 'provider:screener:last', calculation: 'provider:latest' };
const average: CompanyListColumn = { id: 'capex-average', kpiId: 'provider_64', window: 'provider:screener:5year', calculation: 'provider:mean' };
const history: CompanyListColumn[] = Array.from({ length: 10 }, (_, i) => ({ id: `capex-history-${i + 1}`, kpiId: 'provider_64', window: `provider:screener_history:Year${i + 1}`, calculation: 'provider:History' }));
const columns = [latest, average, ...history];

export default function ProviderCapexPanel({ companyId, financial }: { companyId: string; financial: FinancialIndex }) {
  const context = useExpandedKpis(financial, financial.taxonomy_sha256, columns, true);
  const cell = (column: CompanyListColumn) => expandedKpiCell({ id: companyId }, column, context);
  const unavailable = columns.every(column => cell(column).status !== 'available');
  return <div className="provider-capex-panel" data-provider-capex={companyId} data-provider-capex-ready={context.ready}>
    <h4>Capital expenditure (capex)</h4>
    <p>Spending on long-term assets, such as equipment and buildings. Ask how much keeps the existing business running and how much expands it.</p>
    <div className="provider-capex-cards">{([{ column: latest, label: 'Saved provider capex' }, { column: average, label: 'Provider five-year average' }]).map(({ column, label }) => { const observation = cell(column); return <div key={column.id} data-capex-variant={column.window} data-capex-value={observation.value ?? ''}><small>{label}</small><strong>{observation.status === 'loading' ? 'Loading…' : observation.status === 'available' ? observation.display : 'Unavailable'}</strong><span>{column.id === latest.id ? 'Latest provider calculation · period not supplied' : 'Saved five-year average · exact years not supplied'}</span></div>; })}</div>
    <p className="provider-capex-source">Börsdata KPI snapshot {EXPANDED_KPI_MANIFEST.snapshot}. These separately downloaded KPIs do not follow the annual/quarterly selector above. Their underlying fiscal dates are unavailable, so they are not joined to the dated statement charts.</p>
    {context.error && <p role="alert">{context.error}</p>}
    {context.errors.size > 0 && <p role="alert">Some capex data could not be verified: {[...new Set(context.errors.values())].join(' ')}</p>}
    {context.ready && unavailable && !context.error && !context.errors.size && <p>No usable capex observation is available for this listing. Missing or conflicting records and amounts with an unverified currency remain unavailable.</p>}
    <details><summary>Inspect saved annual capex observations and definitions</summary><p>The provider supplies annual history positions 1–10 without fiscal dates. Keep these separate from dated report history until the periods and definitions are reconciled.</p><div className="table-scroll"><table aria-label="Saved annual capex observations"><thead><tr><th>Provider history position</th><th>Capex</th></tr></thead><tbody>{history.map((column, i) => <tr key={column.id}><th>{i + 1}</th><td title={cell(column).detail}>{cell(column).status === 'available' ? cell(column).display : cell(column).status === 'loading' ? 'Loading…' : 'Unavailable'}</td></tr>)}</tbody></table></div><p>Values retain the provider’s sign, currency and definition. Investing cash flow also contains acquisitions, financial investments and disposals; it is not interchangeable with capex. The saved data does not separate maintenance from growth expenditure.</p></details>
  </div>;
}
