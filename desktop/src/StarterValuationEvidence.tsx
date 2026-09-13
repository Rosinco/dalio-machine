import { format } from './model';
import type { buildStarterValuation } from './starterValuations';
import { scenarioKeys, scenarioNames } from './valuation';
import { reconcileCashComponents } from './cashComponents';
import { resolveTerminalSale } from './terminalValue';

type Starter = ReturnType<typeof buildStarterValuation>;
const amount = (value: number | null | undefined, digits = 2) => value == null ? 'Unavailable' : format(value, digits);

export default function StarterValuationEvidence({ starter, edited }: { starter: Starter; edited: boolean }) {
  const { draft, evidence: e, issues } = starter;
  const current = e.method === 'empirical-cash-starter-v3';
  const sourceRows = e.annual.length ? e.annual : e.latestAnnual ? [e.latestAnnual] : [];
  const sources = [...new Map(sourceRows.map(row => [row.sourceId, row])).values()];
  return <section className="valuation-card valuation-starter-evidence" data-starter-evidence="true" data-range-mode={e.rangeMode} data-range-group={e.uncertainty.group ?? ''} data-calibration-id={e.uncertainty.calibrationId ?? ''}>
    <div className="eyebrow">STANDARD HISTORICAL BASELINE</div>
    <h2>How the standard historical baseline was built</h2>
    <p>This starter uses up to {e.historyYears} consecutive years of saved free cash flow. {e.projection === 'latest' ? 'The latest eligible amount is the provisional midline; the five-year history informs cash variability.' : 'Your weights determine the historical trend or weighted average.'} The saved field is a cash-flow proxy; company-specific distributable cash, financing and investment needs still require review.</p>
    {edited && <p>The working study differs from this standard historical baseline. These tables retain the original source figures and baseline calculations; the charts use the working study, including any researched refinements or your edits.</p>}
    {issues.length > 0 && <div className="valuation-starter-issues"><h3>Inputs needing attention</h3><ul>{issues.map((issue, index) => <li key={index}>{issue}</li>)}</ul></div>}

    <div className="valuation-evidence-block" data-evidence-kind="source">
      <p className="valuation-source"><strong>Source facts</strong></p>
      <h3>Saved annual free-cash-flow history</h3>
      <p>{e.explanation[0]}</p>
      <p>{e.explanation[1]}</p>
      {sourceRows.length ? <div className="table-scroll"><table><thead><tr><th>Financial period</th><th>Published / saved</th><th>Vendor cash flow</th><th>Currency ratio</th><th>Reporting cash flow</th><th>Amount used · {e.currency} m</th></tr></thead><tbody>{sourceRows.map(row => <tr key={`${row.sourceId}-${row.year}-${row.end}`}><th>FY {row.year}<br />{row.start}–{row.end}</th><td>{row.reportDate ?? 'Unavailable'}<br />{row.sourceAsOf}</td><td>{amount(row.rawCashFlow)} {row.quotedCurrency ?? 'unit unavailable'}</td><td>{amount(row.currencyRatio, 6)}</td><td>{amount(row.reportedCashFlow)} {row.reportingCurrency}</td><td>{amount(row.cashFlow)}</td></tr>)}</tbody></table></div> : <p>No usable annual free-cash-flow history is included for this listing.</p>}
      <h3>Cash components and definition checks</h3>
      <p>These saved amounts use the same annual report, source and currency basis as the historical cash above. Operating plus investing cash is a separate arithmetic measure; matching provider FCF does not verify cash distributable to shareholders.</p>
      <div className="table-scroll"><table aria-label="Annual cash component checks"><thead><tr><th>Period</th><th>Operating cash</th><th>Investing cash</th><th>Financing cash</th><th>Net cash for period</th><th>Provider FCF</th><th>Operating + investing</th><th>FCF difference</th><th>Net cash less component sum</th></tr></thead><tbody>{sourceRows.map(row => {
        const c = row.components, check = reconcileCashComponents(c);
        return <tr key={`${row.sourceId}-${row.year}`}><th>FY {row.year} · {e.currency} m{row.possiblePlaceholder && <small>Possible source placeholder; zero equality is not corroboration.</small>}</th><td>{amount(c.operating)}</td><td>{amount(c.investing)}</td><td>{amount(c.financing)}</td><td>{amount(c.netCash)}</td><td>{amount(c.providerFcf)}</td><td>{amount(check.operatingPlusInvesting)}</td><td>{amount(check.providerDifference)}<br />{row.possiblePlaceholder ? 'Possible placeholder' : check.providerComparison}</td><td>{amount(check.netDifference)}<br />{row.possiblePlaceholder ? 'Possible placeholder' : check.netComparison}</td></tr>;
      })}</tbody></table></div>
      <p>The saved data does not separate maintenance/growth capital expenditure, acquisitions/disposals, working-capital cash movements, leases, cash interest or cash taxes. Differences can reflect definitions, currency effects or other reconciling items and are not automatically treated as errors. No missing amount is filled with zero or inferred as maintenance investment. Arithmetic comparison tolerance: 0.000001 million + 0.000001% of the larger absolute amount.</p>
      <h3>Dated purchase price and ownership basis</h3>
      {e.price ? <>
        <p>FY {e.price.year} report: {e.price.reportStart}–{e.price.reportEnd}, published {e.price.reportDate ?? 'date unavailable'}. Saved price: {amount(e.price.close, 6)} {e.price.currency ?? 'currency unavailable'} on {e.price.priceDate}; reported shares: {amount(e.price.shares, 6)} million. This is a historical publication-window price.</p>
        <p>{e.price.method === 'local' ? `The purchase basis uses the listing currency, ${e.currency}, with no currency conversion.` : `SEK conversion uses ${amount(e.price.fxRate, 6)} SEK per ${e.price.currency ?? 'listing currency'} on ${e.price.fxDate ?? 'date unavailable'}.`} Equity purchase basis: <strong>{amount(e.price.marketCap)} {e.currency} m</strong>. Each listing is treated separately; review the share class, receipts and common-equity ownership basis before relying on the result.</p>
      </> : <p>A supported dated purchase price is unavailable. Enter a price with its share and currency basis to complete the comparison.</p>}
      {e.capitalDate && <p>Capital references use the report ending {e.capitalDate}. They do not establish proceeds available in a liquidation.</p>}
    </div>

    <div className="valuation-evidence-block" data-evidence-kind="calculation">
      <p className="valuation-source"><strong>Calculated from the stated inputs</strong></p>
      <h3>{e.projection === 'latest' ? 'Latest annual cash baseline' : 'Weighted cash-flow baseline'}</h3>
      <p>{e.anchorBasis}</p>
      {e.annual.length > 0 && <div className="table-scroll"><table><thead><tr><th>Financial year</th><th>Chosen weight</th><th>Effective weight</th><th>Cash-flow proxy · {e.currency} m</th></tr></thead><tbody>{e.annual.map(row => <tr key={`${row.sourceId}-${row.year}`}><th>FY {row.year}</th><td>{amount(row.weightPercent)}%</td><td>{amount(row.effectiveWeightPercent)}%</td><td>{amount(row.cashFlow)}</td></tr>)}</tbody></table></div>}
      <p>{e.projection === 'latest' ? 'Latest-cash baseline' : 'Weighted average'}: <strong>{amount(e.anchor)} {e.currency} m</strong>. {e.projection === 'latest' ? 'Weights apply to the alternative trend and weighted-average projections.' : 'Included weights are divided by their sum.'} A shorter usable history is disclosed rather than treated as {e.historyYears} observed years. Missing recent inputs leave the calculation unavailable.</p>
      {e.projection === 'trend' && <><h3>Weighted historical trend</h3><p>A straight line is fitted to the annual cash flows using the same weights. Time is measured relative to the latest financial year. Recent years influence both its starting level and its annual change.</p><p>Fitted latest level: <strong>{amount(e.trend.intercept)} {e.currency} m</strong>. Annual change: <strong>{amount(e.trend.slope)} {e.currency} m per year</strong>. Mid forecast in year t = fitted latest level + annual change × t. The line can rise, fall or cross zero; it is a historical extrapolation.</p></>}
      {current && <><h3>Historical cash variability</h3><p>Five-year population standard deviation divided by mean absolute cash: <strong>{amount(e.uncertainty.dispersion, 3)}</strong>. Group: <strong>{e.uncertainty.group ?? 'Unavailable'}</strong>. Low is below 0.25; medium is 0.25 to below 0.75; high is 0.75 or above. Signed losses and COVID/rebound years remain included. Dispersion includes growth and changes in scale; it is not a company-quality rating.</p><p>Mean absolute historical cash scale: <strong>{amount(e.uncertainty.scale)} {sourceRows[0]?.reportingCurrency ?? e.currency} m</strong>. Classification uses native reporting cash. A quote-currency forecast does not inherit native-currency error ranges.</p></>}
    </div>

    <div className="valuation-evidence-block" data-evidence-kind="assumption">
      <p className="valuation-source"><strong>Assumptions · editable sensitivity rules</strong></p>
      <h3>From the historical proxy to future payments</h3>
      {current ? <><p>{e.explanation[2]}</p>{e.uncertainty.reason && <p className="valuation-caution">{e.uncertainty.reason}</p>}
        <p>Historical range status: <strong>{e.uncertainty.status === 'historical' ? '80% target · exploratory historical errors' : e.uncertainty.status === 'percentage' ? 'Assumed percentage sensitivity' : 'Unavailable'}</strong>. Eligible year 1–4 factors were estimated from older forecast errors for operating/property businesses. Coverage varies by period and is best supported near term; this is not a guaranteed probability for this company, the whole cash path or DCF. Sector and branch supply context; asset intensity and debt are reviewed separately.</p>
        <div className="table-scroll"><table aria-label="Historical error factors and assumed ranges"><thead><tr><th>Forecast year</th><th>Basis</th><th>Half-width · {e.currency} m</th><th>Error factor</th><th>Calibration support</th></tr></thead><tbody>{e.forecast.map(row => <tr key={row.year} data-range-kind={row.rangeBasis}><td>Year {row.year}</td><td>{row.rangeBasis === 'historical' ? 'Historical errors' : row.rangeBasis === 'assumed-tail' ? 'Assumed later years' : row.rangeBasis === 'percentage' ? 'Assumed percentage' : 'Unavailable'}</td><td>{amount(row.halfWidth)}</td><td>{amount(row.factor, 4)}</td><td>{row.support ? `${row.support.count.toLocaleString()} forecasts / ${row.support.listings.toLocaleString()} listings / ${row.support.histories.toLocaleString()} histories · ${row.calibrationGroup}` : 'No empirical support assigned'}</td></tr>)}</tbody></table></div>
      </> : <p>{e.projection === 'trend' ? `The mid case projects the weighted historical trend for ${e.years} years.` : `The mid case holds the weighted average constant for ${e.years} years.`} The first-year range is ±{e.spreadPercent}%, widening by {e.spreadStepPercent} percentage points each year. Low and high subtract and add that year's percentage of the absolute mid amount. No statistical confidence level or scenario probability is assigned.</p>}
      <p>Positive modeled payments assume the proxy can be distributed after maintaining the business and meeting financing needs. Negative payments represent hypothetical shareholder funding. A negative source cash-flow figure alone does not create an obligation for shareholders to contribute.</p>
      {!current && <div className="table-scroll"><table><thead><tr><th>Starter rule</th>{scenarioKeys.map(key => <th key={key}>{scenarioNames[key]}</th>)}</tr></thead><tbody>
        <tr><th>Year-one range around mid</th><td>−{e.spreadPercent}% × |mid|</td><td>0</td><td>+{e.spreadPercent}% × |mid|</td></tr>
        <tr><th>Year-{e.years} range around mid</th><td>−{e.spreadPercent + (e.years - 1) * e.spreadStepPercent}% × |mid|</td><td>0</td><td>+{e.spreadPercent + (e.years - 1) * e.spreadStepPercent}% × |mid|</td></tr>
        <tr><th>Required equity return</th>{scenarioKeys.map(key => <td key={key}>{e.requiredReturn}%</td>)}</tr>
      </tbody></table></div>}
      {e.uncertainty.status === 'percentage' && <p>The percentage range can exceed 100% and cross zero. Its absolute cash width also depends on the mid forecast; it need not increase when the mid amount moves toward zero.</p>}
      <p>{current ? e.explanation[3] : `The forecast lasts ${e.years} years. A positive final-year payment is assumed to continue without growth after that horizon: final equity sale = final-year payment ÷ ${e.requiredReturn / 100}. No selling costs are assumed. A nonpositive final payment gives zero assumed sale proceeds; missing cash inputs leave it unavailable.`}</p>
      <p>No automatic recovery case is supplied. Assets, prior claims, taxes, sale costs and timing need a separate company review before net common-equity recovery can be estimated.</p>
    </div>

    <div className="valuation-evidence-block" data-evidence-kind="calculation">
      <p className="valuation-source"><strong>Calculated starter payments · {e.currency} m</strong></p>
      <div className="table-scroll"><table><thead><tr><th>Calculated amount</th>{scenarioKeys.map(key => <th key={key}>{scenarioNames[key]}</th>)}</tr></thead><tbody>
        <tr><th>Year-one payment</th>{scenarioKeys.map(key => <td key={key}>{amount(draft.scenarios[key].cashFlows[0])}</td>)}</tr>
        {e.terminal && <><tr><th>Historical signed median · unreviewed</th>{scenarioKeys.map(key => <td key={key}>{amount(e.terminal!.median)}</td>)}</tr><tr><th>First post-horizon cash · assumed ±20%</th>{scenarioKeys.map(key => <td key={key}>{amount(draft.scenarios[key].terminalCash?.cashFlow)}</td>)}</tr></>}
        <tr><th>Final equity sale in year {e.years}</th>{scenarioKeys.map(key => <td key={key}>{amount(resolveTerminalSale(draft.scenarios[key]).value)}</td>)}</tr>
      </tbody></table></div>
    </div>

    <details className="valuation-method valuation-starter-provenance"><summary>Saved files and provenance</summary>
      <p>Starter method: {e.method}. Prepared {e.asOf}; financial snapshot {e.snapshotDate ?? 'unavailable'}. Original report, price and saved-file dates remain separate from the selected macro release.</p>
      {current && <p>Requested calibration: {draft.starterOrigin?.id === 'empirical-cash-starter-v3' ? draft.starterOrigin.calibrationId : 'Unavailable'}. Historical ranges require matching saved financial and taxonomy identities. Calibration includes eligible outcomes through FY2020 with a nominal 30 June 2021 publication cutoff, using later retained data vintages. Source cash-definition revisions remain unresolved; no pandemic observations were automatically removed. Future validation is required.</p>}
      <ul className="valuation-source-list">{sources.map(source => <li key={source.sourceId}><strong>Annual accounts · {source.sourceAsOf}</strong><span>{source.sourceId}</span><span>{source.sourcePath ?? 'Source path unavailable'}</span><small>SHA-256 {source.sourceHash ?? 'unavailable'}</small></li>)}
        {e.price && <li><strong>Price and market basis · {e.price.sourceAsOf}</strong><span>{e.price.sourceId}</span><span>{e.price.sourcePath ?? 'Source path unavailable'}</span><small>SHA-256 {e.price.sourceHash ?? 'unavailable'}</small></li>}
      </ul>
    </details>
  </section>;
}
