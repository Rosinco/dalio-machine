import type { CompanyListCell, CompanyListColumn } from './companyListModel';
import { columnLabel, companyListWindowLabel, companyListCalculationLabel } from './companyListModel';
import type { ResearchGaugeRow } from './researchGaugeModel';

export function CompanyListComparison({ rows, columns, cell, onCompany, onRemove }: {
  rows: ResearchGaugeRow[];
  columns: CompanyListColumn[];
  cell: (row: ResearchGaugeRow, column: CompanyListColumn) => CompanyListCell;
  onCompany: (id: string) => void;
  onRemove: (id: string) => void;
}) {
  return <>
    <p className="company-list-comparison-note">Your selected KPIs appear side by side. Each listing keeps its own currency and observation dates.</p>
    <div className="company-list-comparison-scroll" role="region" aria-label="Selected company comparison" tabIndex={0}>
      <table className="company-list-comparison">
        <thead><tr><th scope="col">KPI</th>{rows.map(row => <th key={row.id} scope="col"><button className="company-list-company" onClick={() => onCompany(row.id)}>{row.name}</button><small>{row.ticker ?? '—'} · {row.country ?? '—'}</small><button className="company-list-text-button" aria-label={`Remove ${row.name} from comparison`} onClick={() => onRemove(row.id)}>Remove</button></th>)}</tr></thead>
        <tbody>{columns.map(column => <tr key={column.id}><th scope="row">{columnLabel(column)}<small>{companyListWindowLabel(column)} · {companyListCalculationLabel(column)}</small></th>{rows.map(row => {
          const value = cell(row, column);
          return <td key={row.id} title={value.detail} data-company-compare-listing={row.id} data-company-compare-kpi={column.kpiId}><strong>{value.display}</strong>{value.date && <small>{value.date}</small>}{value.value === null && <small>{value.detail}</small>}</td>;
        })}</tr>)}</tbody>
      </table>
    </div>
  </>;
}

export function CompanyListValueDetails({ row, column, cell }: { row: ResearchGaugeRow; column: CompanyListColumn; cell: CompanyListCell }) {
  return <section className="company-list-value-details" aria-label="KPI value details">
    <p>{row.name} · {row.ticker ?? row.id}</p>
    <h3>{columnLabel(column)}</h3>
    <div className="company-list-detail-value">{cell.display}</div>
    <dl>
      <div><dt>Period</dt><dd>{companyListWindowLabel(column)}</dd></div>
      <div><dt>Calculation</dt><dd>{companyListCalculationLabel(column)}</dd></div>
      <div><dt>Observation or snapshot date</dt><dd>{cell.date ?? 'Not available for this value'}</dd></div>
      {cell.currency && <div><dt>Currency</dt><dd>{cell.currency}</dd></div>}
      {typeof cell.value === 'number' && <div><dt>Saved numeric value</dt><dd>{String(cell.value)}</dd></div>}
    </dl>
    <p className="company-list-detail-source">{cell.detail}</p>
  </section>;
}
