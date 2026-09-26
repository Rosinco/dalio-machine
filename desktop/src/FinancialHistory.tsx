import { useScreenState } from './NavigationContext';
import { financialSeries, formulas, periodLabel, ratioLabels } from './business';
import { format, seriesColors } from './model';
import { hasFinancialHistory, statementLabels, type CompanyCoverage, type FinancialCompany, type FinancialIndex } from './financialData';
import type { CompanyEntry } from './listingCatalogue';
import FinancialChart from './FinancialChart';
import MarketHistory from './MarketHistory';
import ProviderCapexPanel from './ProviderCapexPanel';
import './financialHelp.css';

const count = (n: number) => n.toLocaleString('en-US');
const metricHelp: Record<string, string> = {
  revenues: 'Income from the company’s sales and other reported revenue, before expenses. Revenue growth is not the same as profit growth.',
  net_sales: 'Sales after the adjustments included in the provider’s definition. It is a separate measure from revenue; do not add the two.',
  gross_income: 'Revenue left after the direct cost of producing goods or services, before other operating expenses.',
  operating_income: 'Profit from operations before financing costs and tax, also called EBIT. Accounting profit can differ from cash generated.',
  profit_before_tax: 'Profit after operating and financing items, before tax.',
  profit_to_equity_holders: 'Net profit attributed to the company’s equity holders. It is an accounting result, not necessarily cash available for distribution.',
  cash_flow_from_operating_activities: 'Cash generated or used by day-to-day operations. Working-capital changes can make it differ substantially from profit.',
  cash_flow_from_investing_activities: 'Cash spent or received on investments, acquisitions and asset sales. A negative amount may reflect investment in the business.',
  cash_flow_from_financing_activities: 'Cash from or to lenders and owners, including financing transactions. Positive financing cash does not by itself mean the business generated cash.',
  cash_flow_for_the_year: 'The reported net cash movement during this period. Despite the source field name, quarterly observations cover a quarter.',
  free_cash_flow: 'Free cash flow (FCF) under Börsdata’s definition. Review which lease, interest and investment cash items are included before treating it as shareholder cash.',
  total_assets: 'The accounting value of resources held by the business at period end. Book values are not estimates of sale proceeds.',
  non_current_assets: 'Assets expected to serve the business beyond the near term, using the report’s classification.',
  current_assets: 'Assets classified as short term, including items such as cash, receivables and inventory. They are not all immediately available cash.',
  tangible_assets: 'Reported physical assets such as property, plant and equipment. Their book value does not establish maintenance investment or market value.',
  intangible_assets: 'Reported nonphysical assets, such as goodwill or acquired rights. Their accounting value does not alone establish a durable competitive advantage.',
  financial_assets: 'Reported financial holdings and claims. Their usefulness depends on ownership, restrictions and liquidity.',
  cash_and_equivalents: 'Cash and short-term cash equivalents at period end. Some balances may be restricted or needed to run the business.',
  total_equity: 'Accounting assets less liabilities. This is book equity, not the company’s stock-market value.',
  current_liabilities: 'Obligations classified as short term. Review when payments fall due and how they will be financed.',
  non_current_liabilities: 'Obligations classified as longer term, including items beyond borrowing.',
  total_liabilities_and_equity: 'The financing side of the balance sheet: liabilities plus equity. It normally reconciles to total assets.',
  net_debt: 'Provider-reported debt net of its included cash balances. Negative net debt can indicate net cash; check leases and the exact debt definition.',
  operating_margin: 'The percentage of each revenue unit left as operating profit. Compare the same business definition across years.',
  operating_cash_margin: 'Operating cash flow as a percentage of revenue. Working-capital movements can cause temporary swings.',
  return_on_capital: 'A rough annual comparison of operating profit with year-end equity plus net debt. It is not a reviewed return on invested capital (ROIC).',
  equity_ratio: 'Book equity as a percentage of total assets. It describes accounting financing, not the liquidity or value of those assets.',
  net_debt_to_equity: 'Net debt as a percentage of book equity. A small or negative equity denominator can make the ratio difficult to interpret.',
};
const statements = {
  income: { label: 'Income statement', fields: ['revenues', 'net_sales', 'gross_income', 'operating_income', 'profit_before_tax', 'profit_to_equity_holders'] },
  balance: { label: 'Balance sheet', fields: ['total_assets', 'non_current_assets', 'intangible_assets', 'tangible_assets', 'financial_assets', 'current_assets', 'cash_and_equivalents', 'total_equity', 'non_current_liabilities', 'current_liabilities', 'total_liabilities_and_equity', 'net_debt'] },
  cash: { label: 'Cash flow', fields: ['cash_flow_from_operating_activities', 'cash_flow_from_investing_activities', 'cash_flow_from_financing_activities', 'cash_flow_for_the_year', 'free_cash_flow'] },
};
export default function FinancialHistory({ entry, index, company }: { entry: CompanyEntry; index: FinancialIndex; company: FinancialCompany }) {
  const coverage = index.companies[entry.id];
  const stateKey = `financial-history:${index.id}`;
  const [frequency, setFrequency] = useScreenState<'annual' | 'quarterly'>(`${stateKey}:frequency`, coverage.annual.count ? 'annual' : 'quarterly');
  const [metric, setMetric] = useScreenState(`${stateKey}:metric`, 'revenues'), [statement, setStatement] = useScreenState<keyof typeof statements>(`${stateKey}:statement`, 'income');
  const [chosenCurrency, setChosenCurrency] = useScreenState(`${stateKey}:currency`, ''), [period, setPeriod] = useScreenState(`${stateKey}:period`, '');
  const reports = company[frequency], latest = reports.at(-1), annual = company.annual.at(-1), quarter = company.quarterly.at(-1);
  const currency = chosenCurrency || latest?.currency || coverage.currencies[0] || entry.report_currency || '';
  const selected = reports.find(r => `${r.year}-${r.period}` === period) ?? latest;
  const labels = { ...statementLabels, ...ratioLabels }, series = financialSeries(reports, metric, currency);
  const unit = metric in ratioLabels ? '%' : `${currency} million`;
  const overviewChart = (id: string, title: string, fields: string[], explanation: string, chartUnit = `${currency} million`) => {
    const colors = [seriesColors.selected, seriesColors.comparison, '#7d7794'];
    const chartSeries = fields.map((field, i) => ({ name: labels[field], values: financialSeries(reports, field, currency).values, color: colors[i % colors.length] }));
    return <figure className="business-figure" data-business-chart={id}><figcaption><h4>{title}</h4><span>{chartUnit}</span></figcaption><FinancialChart labels={financialSeries(reports, fields[0], currency).labels} series={chartSeries} label={`${entry.display_name} ${title} ${frequency} business overview`} unit={chartUnit} /><p>{explanation}</p></figure>;
  };
  return <div className="company-financial-history" data-financial-history={entry.id} data-financial-pack={index.id}>
    <nav className="financial-section-nav" aria-label="Within company financial history">{hasFinancialHistory(coverage) && <><a href="#company-business-overview">Business overview</a><a href="#company-cash-overview">Cash &amp; investment</a><a href="#company-capital-overview">Assets &amp; debt</a><a href="#company-financial-chart">Financial trends</a><a href="#company-financial-statements">Report details</a></>}<a href="#company-market-history">Saved market values</a></nav>
    {hasFinancialHistory(coverage) ? <>
      <section className="business-figures-overview" id="company-business-overview">
        <div className="section-title"><h3>Business overview</h3><div className="frequency-buttons" role="group" aria-label="Financial reporting frequency"><button aria-pressed={frequency === 'annual'} onClick={() => { setFrequency('annual'); setPeriod(''); }}>Annual</button><button aria-pressed={frequency === 'quarterly'} onClick={() => { setFrequency('quarterly'); setPeriod(''); }}>Quarterly</button></div></div>
        {entry.profile?.segments.length ? <details className="business-description" data-company-description-state="saved"><summary><strong>What the company does:</strong> {entry.profile.segments.join(' · ')}</summary><p>{entry.profile.overlap}</p><p>Business mix from saved research dated {entry.profile.context_as_of}; newer financial figures do not update this description. Source record: {entry.profile.context_source_id}.</p></details> : <p className="business-description-missing" data-company-description-state="missing">Business description not yet reviewed. Use Research to record products, customers and how this company earns money.</p>}
        <p className="business-note">{latest ? `${periodLabel(latest)} · ${latest.start} to ${latest.end} · ${latest.currency} million · saved ${latest.source_as_of}` : 'No saved reports for this frequency.'}</p>
        <div className="financial-cards">{['revenues', 'operating_income', 'free_cash_flow', 'net_debt'].map(key => <div key={key} data-financial={key}><small>{statementLabels[key]}</small><strong>{format(latest?.values[key], 0)}</strong><span>{latest?.currency ?? currency} million</span></div>)}</div>
        {coverage.currencies.length > 1 && <label className="statement-control">Chart reporting currency<select aria-label="Chart reporting currency" value={currency} onChange={e => setChosenCurrency(e.target.value)}>{coverage.currencies.map(c => <option key={c}>{c}</option>)}</select></label>}
        <div className="business-figure-group" data-business-figure-group="performance">
          <div className="business-figure-heading"><h3>Revenue &amp; profitability</h3><p>Revenue shows the scale of the business. Margins show how much of each sales unit becomes profit or operating cash.</p></div>
          <div className="business-figure-grid">
            {overviewChart('revenue', 'Revenue', ['revenues'], 'Sales before expenses. Growth can come from demand, pricing, acquisitions or currency changes; the chart alone does not identify the cause.')}
            {overviewChart('profitability', 'Profit & operating cash margins', ['operating_margin', 'operating_cash_margin'], 'Operating profit / revenue and operating cash / revenue. Cash margins can swing when customers, inventory or suppliers tie up or release cash.', '%')}
          </div>
        </div>
        <div className="business-figure-group" data-business-figure-group="cash" id="company-cash-overview">
          <div className="business-figure-heading"><h3>Cash generation &amp; investment</h3><p>Check whether operating profit turns into cash, then inspect the investment needed to sustain the business.</p></div>
          <div className="business-figure-grid">
            {overviewChart('operating-cash', 'Operating & free cash flow', ['cash_flow_from_operating_activities', 'free_cash_flow'], 'Operating cash is generated by daily activity. Provider free cash flow needs reconciliation of leases, interest and investment definitions before it becomes a shareholder-cash forecast.')}
            {overviewChart('investing-cash', 'Investing cash flow', ['cash_flow_from_investing_activities'], 'Negative amounts are net cash outflows; positive amounts are net inflows. Acquisitions and asset sales can make one year unusually large.')}
          </div>
          <ProviderCapexPanel companyId={entry.id} financial={index} />
          <p className="business-capex-note" data-business-capex-note><strong>Review what the investment achieves.</strong> Maintenance and growth capex are not separated here. Investing cash also includes acquisitions, financial investments and asset sales; compare its definition with the separately saved capex figures before refining Value.</p>
        </div>
        <div className="business-figure-group" data-business-figure-group="capital" id="company-capital-overview">
          <div className="business-figure-heading"><h3>Assets &amp; financing</h3><p>Understand what the business owns, how it is funded, and which obligations could restrict shareholder cash.</p></div>
          <div className="business-figure-grid">
            {overviewChart('assets', 'Asset base', ['total_assets', 'tangible_assets', 'intangible_assets'], 'Accounting amounts at period end. Total assets already includes its components; do not add these lines together. Physical and intangible book values are not sale valuations.')}
            {overviewChart('financing', 'Equity, net debt & cash', ['total_equity', 'net_debt', 'cash_and_equivalents'], 'Net debt already reflects the cash included in the provider’s definition. Negative net debt can indicate net cash. Review debt maturities, leases and restricted cash separately.')}
          </div>
        </div>
        <p className="business-figures-context">{frequency === 'annual' ? 'Full financial years' : 'Standalone quarters, not trailing 12 months'} · charts use {currency} millions unless labelled %. Gaps stay blank; balance-sheet figures are measured at period end.</p>
      </section>
      <section id="company-financial-chart"><div className="section-title"><h3>Financial history</h3></div>
        <label className="financial-chart-choice">Choose a financial measure<select className="metric-select" aria-label="Company financial chart" aria-describedby="financial-metric-explanation" value={metric} onChange={e => setMetric(e.target.value)}>{Object.entries(labels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
        <p className="financial-metric-help" id="financial-metric-explanation"><strong>{labels[metric]}.</strong> {metricHelp[metric] ?? 'Reported under the provider’s definition; inspect the financial statement and its source notes.'}</p>

        <FinancialChart labels={series.labels} series={[{ name: labels[metric], values: series.values, color: seriesColors.selected }]} label={`${entry.display_name} ${labels[metric]} ${frequency} history`} unit={unit} />
        <p className="chart-caption">Read periods from left to right; the vertical axis uses {unit}. {frequency === 'annual' ? 'FY means a full financial year.' : 'Q means a standalone quarter, not the trailing 12 months (R12).'} Revenue, profit and cash cover the period; balance-sheet amounts are measured at period end. Gaps mean unavailable observations, not zero.</p>
        <details className="financial-chart-method"><summary>Calculation, currency and chart controls</summary><p>{formulas[metric] ?? 'Nominal reported amounts, recovered using each row’s saved currency conversion.'} {frequency === 'quarterly' && 'The return-on-capital proxy is annual only.'} {coverage.currencies.length > 1 && 'Monetary charts show one reporting currency at a time; ratios are dimensionless.'} Use the slider below the chart to shorten the visible period. View financial figures gives the values and dates in a table.</p></details>
        <details className="financial-table"><summary>View financial figures</summary><div className="table-scroll"><table><thead><tr><th>Period</th><th>Currency</th><th>{labels[metric]}</th><th>Reported</th><th>Saved</th></tr></thead><tbody>{[...reports].reverse().map(r => <tr key={`${r.year}-${r.period}`}><td>{periodLabel(r)}</td><td>{metric in ratioLabels ? '%' : `${r.currency} m`}</td><td>{format(r.values[metric], 2)}</td><td>{r.report_date ?? 'Unknown'}</td><td>{r.source_as_of}</td></tr>)}</tbody></table></div></details>
      </section>
      <section className="financial-statements" id="company-financial-statements"><h3>Financial statements</h3><p className="business-note">Income statement: profit over the period. Balance sheet: assets and obligations at period end. Cash flow: cash generated, invested and financed during the period.</p><div className="statement-tabs" role="group" aria-label="Financial statement">{Object.entries(statements).map(([key, value]) => <button key={key} aria-pressed={statement === key} onClick={() => setStatement(key as keyof typeof statements)}>{value.label}</button>)}</div>
        <label className="statement-control">{frequency === 'annual' ? 'Financial year' : 'Financial quarter'}<select aria-label="Statement period" value={selected ? `${selected.year}-${selected.period}` : ''} onChange={e => setPeriod(e.target.value)} disabled={!reports.length}>{[...reports].reverse().map(r => <option key={`${r.year}-${r.period}`} value={`${r.year}-${r.period}`}>{periodLabel(r)} · {r.currency} · ends {r.end}</option>)}{!reports.length && <option value="">No saved periods</option>}</select></label>
        {selected ? <><p className="chart-caption">{selected.start} to {selected.end} · published {selected.report_date ?? 'date unavailable'} · saved {selected.source_as_of}. Amounts in {selected.currency} million.</p><table className="statement-table" aria-label={`${statements[statement].label} figures`}><tbody>{statements[statement].fields.map(key => <tr key={key} data-statement-field={key}><th>{statementLabels[key]}</th><td>{format(selected.values[key], 2)}</td></tr>)}</tbody></table>
          <p className="chart-caption">Source fields are shown as provided. Missing values are not zero. {statement === 'balance' && 'Asset categories can overlap; net debt is a supplementary measure. These figures do not identify individual properties or facilities.'}{statement === 'income' && 'Revenue and net sales are separate provider fields and must not be added together.'}{statement === 'cash' && 'Börsdata FCF uses the provider’s definition; lease, interest and investment classifications need review before treating it as owner earnings.'}</p>
          <details className="financial-table"><summary>Inspect saved amounts and conversion</summary><p className="business-note">The source stores monetary amounts after multiplying by its currency ratio. Atlas divides each amount by that same ratio to recover {selected.currency} millions. Saved ratio: {selected.currency_ratio ?? 'unavailable'}. A missing or non-positive ratio leaves all converted values unavailable.</p><div className="table-scroll"><table><thead><tr><th>Field</th><th>Saved amount</th><th>{selected.currency} million</th></tr></thead><tbody>{statements[statement].fields.map(key => <tr key={key}><td>{statementLabels[key]}</td><td>{format(selected.raw[key], 4)}</td><td>{format(selected.values[key], 4)}</td></tr>)}</tbody></table></div></details></> : <p className="empty">No saved statements for this frequency.</p>}
      </section>
      <section><h3>Profitability through the cycle</h3><FinancialChart labels={financialSeries(company.annual, 'operating_margin', currency).labels} series={['operating_margin', 'return_on_capital'].map((key, i) => ({ name: ratioLabels[key], values: financialSeries(company.annual, key, currency).values, color: i ? seriesColors.comparison : seriesColors.selected }))} label={`${entry.display_name} annual operating margin and return on capital proxy`} unit="%" /><p className="business-note">Reported profit may include revaluations or one-off transactions. The annual return proxy uses year-end equity plus net debt; it is not adjusted ROIC. These ratios have different relevance across branches, particularly financial businesses.</p></section>
    </> : <section><h3>No usable saved reports</h3><p className="business-note">No annual or quarterly reports could be shown for this listing from the two saved downloads. Independently saved capex KPIs can still be inspected below when available.</p><ProviderCapexPanel companyId={entry.id} financial={index} /></section>}
    <details className="financial-coverage-details"><summary>Report coverage &amp; source dates</summary>
      <section><div className="section-title"><h3>Saved report coverage</h3><span className="micro">{index.as_of}</span></div><CoverageDetails coverage={coverage} />
      <p className="chart-caption">Coverage describes saved reports. Missing periods and unavailable values stay blank; a download date is not a financial period.</p>
      {coverage.withheld > 0 && <p className="financial-quality-note">{count(coverage.withheld)} source {coverage.withheld === 1 ? 'row withheld' : 'rows withheld'} because the period, date or currency metadata could not be used. See the source notes below.</p>}
    </section>
      <section><div className="period-overview"><div><small>LATEST FULL YEAR</small><strong>{annual ? periodLabel(annual) : 'Unavailable'}</strong><span>Reported {annual?.report_date ?? 'date unavailable'}</span></div><div><small>LATEST SAVED QUARTER</small><strong>{quarter ? periodLabel(quarter) : 'Unavailable'}</strong><span>Reported {quarter?.report_date ?? 'date unavailable'}</span></div></div></section>
    </details>
    <div id="company-market-history"><MarketHistory company={company} index={index} name={entry.display_name} /></div>
    <section><h3>Sources and interpretation</h3><p className="business-note">Period end tells you when the activity occurred. Reported is the provider’s publication date and has not been independently verified. Saved tells you when the data was downloaded; it does not make an old report current.</p><p className="business-note">The newest saved row takes precedence for each fiscal period. Earlier periods can come from an older download and may use a different accounting or restatement basis. Invalid newer rows are withheld; an older value is not silently substituted.</p><p className="business-note">Amounts are nominal. Börsdata FCF needs reconciliation of lease, interest and investment cash classifications. Per-share figures are omitted here because the saved files can use different split-adjustment bases.</p>
      {company.withheld.length > 0 && <details className="withheld-reports"><summary>{count(company.withheld.length)} withheld source rows</summary><div className="table-scroll"><table><thead><tr><th>Fiscal period</th><th>Saved</th><th>Source dates</th><th>Reason</th></tr></thead><tbody>{company.withheld.map((r, i) => <tr key={i}><td>{r.period === 5 ? 'FY' : `Q${r.period}`} {r.year}</td><td>{index.sources.find(s => s.id === r.source_id)?.as_of}</td><td>{r.start} – {r.end}<br />Published {r.published}</td><td>{r.reason}</td></tr>)}</tbody></table></div></details>}
      <details className="business-sources"><summary>Financial source register</summary>{index.sources.map(source => <dl key={source.id}><dt>{source.frequency} · saved {source.as_of}</dt><dd>{source.path}</dd><dd className="hash">SHA-256 {source.sha256}</dd></dl>)}<p className="chart-caption">Financial pack {index.id.slice(0, 12)} · {count(index.summary.annual)} annual and {count(index.summary.quarterly)} quarterly reports. The library can save a separate copy for backup or another offline computer.</p></details>
    </section>
  </div>;
}
function CoverageDetails({ coverage }: { coverage: CompanyCoverage }) {
  return <div className="financial-coverage">{(['annual', 'quarterly'] as const).map(frequency => { const c = coverage[frequency]; return <div key={frequency} data-coverage={frequency}><small>{frequency.toUpperCase()} REPORTS</small><strong>{count(c.count)}</strong><span>{c.count ? `${c.first}–${c.last}${frequency === 'quarterly' ? ` · latest Q${c.last_period}` : ''}` : 'No saved periods'}</span><span>{c.gaps} missing {frequency === 'annual' ? 'years' : 'quarters'} within span</span>{c.end && <span>Latest period ends {c.end}</span>}{c.unavailable > 0 && <span>{c.unavailable} reports without usable amounts</span>}</div>; })}</div>;
}
