import { useState } from 'react';
import type { FinancialCompany, FinancialIndex } from './financialData';
import { marketDefinition, marketReason } from './marketData';
import { format, seriesColors } from './model';
import FinancialChart from './FinancialChart';

export default function MarketHistory({ company, index, name }: { company: FinancialCompany; index: FinancialIndex; name: string }) {
  const [code, setCode] = useState('SEK');
  const rows = company.market, coverage = index.companies[company.id].market;
  const currencies = ['SEK', ...new Set(rows.map(r => r.currency).filter((s): s is string => !!s && s !== 'SEK'))];
  const currency = currencies.includes(code) ? code : 'SEK';
  const value = (r: typeof rows[number] | undefined) => !r ? null : currency === 'SEK' ? r.sek : r.currency === currency ? r.local : null;
  const years = rows.length ? Array.from({ length: rows.at(-1)!.year - rows[0].year + 1 }, (_, i) => rows[0].year + i) : [];
  const byYear = new Map(rows.map(r => [r.year, r])), reports = new Map(company.annual.map(r => [r.year, r]));
  const latest = rows.at(-1), sources = index.market?.sources.filter(s => rows.some(r => r.source_id === s.id)) ?? [];
  return <section className="market-history" data-market-history={company.id} data-market-available={coverage?.sek ?? 0}>
    <div className="section-title"><h3>Derived market cap</h3><span className="micro">{coverage?.sek ?? 0} SEK observations</span></div>
    {!index.market ? <p className="business-note">Market history is not included in this older financial pack.</p> : <>
      <p className="business-note">{coverage?.local ?? 0} of {coverage?.count ?? 0} annual observations have a usable listing valuation; {coverage?.flagged ?? 0} require review. Each listing retains its own history.</p>
      {rows.length ? <>
        <div className="financial-cards market-cards"><div><small>LATEST FISCAL OBSERVATION</small><strong>FY {latest!.year}</strong><span>Valued {latest!.price_date ?? 'date unavailable'}</span></div><div><small>DERIVED MARKET CAP</small><strong data-market-latest>{format(value(latest), 1)}</strong><span>{currency} million</span></div></div>
        <label className="statement-control">Valuation currency<select aria-label="Market-cap currency" value={currency} onChange={e => setCode(e.target.value)}>{currencies.map(c => <option key={c}>{c}</option>)}</select></label>
        <FinancialChart labels={years.map(String)} series={[{ name: 'Derived market cap', values: years.map(y => value(byYear.get(y))), color: seriesColors.selected }]} label={`${name} derived market-cap history`} unit={`${currency} million`} />
        <p className="chart-caption">Fiscal years on the X-axis; valuations use the dated close around report publication. Gaps include unavailable inputs and observations held back for review. The table gives the price and FX dates for every observation.</p>
        <details className="financial-table market-table"><summary>Valuation dates, inputs and quality notes</summary><div className="table-scroll"><table><thead><tr><th>FY</th><th>Value · {currency} m</th><th>Price / date</th><th>Reported shares · m</th><th>FX to SEK / date</th><th>Source / quality</th></tr></thead><tbody>{[...rows].reverse().map(r => <tr key={r.year} data-market-year={r.year}><td>{r.year}</td><td data-market-value={value(r) ?? ''}>{format(value(r), 2)}</td><td>{format(r.price, 4)} {r.currency ?? ''}<br />{r.price_date ?? 'Unavailable'}</td><td>{format(r.shares, 4)}<br /><small>Report ends {reports.get(r.year)?.end}</small></td><td>{format(r.fx_rate, 6)}<br />{r.fx_date ?? 'Unavailable'}<small>{r.fx_method === 'identity' ? 'SEK identity' : r.fx_method === 'usd_cross' ? 'Observed USD cross-rate' : r.fx_method === 'direct' ? 'Observed direct rate' : ''}</small></td><td>{sources.find(s => s.id === r.source_id)?.as_of}<br />{marketReason(r) || 'No detected issue in these checks'}</td></tr>)}</tbody></table></div></details>
      </> : <p className="empty">No saved annual reports are available to anchor this listing’s market-cap history.</p>}
      <details className="business-sources"><summary>Market-cap method and sources</summary><p className="business-note">{marketDefinition}</p><p className="business-note">Reported shares retain the provider’s adjustment basis. Preference, receipt and unit instruments require a reviewed share basis. The checks detect some scale and share-count breaks; they do not certify every corporate action. Earlier accounting figures may be restated.</p>{sources.map(s => <div className="source-item" key={s.id}><strong>Saved {s.as_of}</strong><p>Shares: matching annual report source. Prices and FX: {s.prices.path}</p><p>Listing currency and instrument type: {s.instruments.path}</p><p className="hash">Prices SHA-256 {s.prices.sha256}<br />Instruments SHA-256 {s.instruments.sha256}</p></div>)}</details>
    </>}
  </section>;
}
