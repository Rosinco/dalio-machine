import { X } from 'lucide-react';
import { columnLabel, companyListUnit } from './companyListModel';
import type { CompanyListColumn } from './companyListModel';
import { companyListRangeActive, companyListRangeError } from './companyListRanges';
import type { CompanyListRange } from './companyListRanges';

export function CompanyListColumnRange({ column, currencies, onChange }: {
  column: CompanyListColumn;
  currencies: readonly string[];
  onChange: (range: CompanyListRange | undefined) => void;
}) {
  const unit = companyListUnit(column);
  if (unit === 'text' || unit === 'date') return null;
  const range = column.range ?? { min: '', max: '' };
  const label = columnLabel(column);
  const active = companyListRangeActive(range);
  const error = companyListRangeError(range, unit);
  const descriptionId = `company-list-range-description-${column.id}`;
  const monetary = unit === 'money' || unit === 'price';
  const units = unit === 'money' ? 'millions' : unit === 'price' ? 'per share' : unit === 'percent' ? '%' : unit === 'points' ? 'percentage points' : unit === 'multiple' ? '×' : unit === 'shares_millions' ? 'million shares' : column.kpiId === 'price_age' ? 'days' : unit === 'count' ? 'count' : 'number';
  const options = [...new Set([...currencies, ...(range.currency ? [range.currency] : [])])].sort();
  return <div className="company-list-column-range" data-company-list-range={column.id} data-active={active}>
    <div className="company-list-column-range-bounds">
      {(['min', 'max'] as const).map(bound => <label key={bound}>
        <span>{bound === 'min' ? 'Min' : 'Max'}</span>
        <input type="text" inputMode="decimal" autoComplete="off" spellCheck={false} maxLength={80}
          aria-label={`${bound === 'min' ? 'Min' : 'Max'} ${label}`}
          aria-describedby={descriptionId} aria-invalid={!!error}
          placeholder="Any" value={range[bound]}
          onChange={event => onChange({ ...range, [bound]: event.target.value })} />
      </label>)}
    </div>
    {monetary && <select aria-label={`Currency for ${label}`} value={range.currency ?? ''}
      aria-describedby={descriptionId} aria-invalid={active && !range.currency}
      onChange={event => onChange({ ...range, currency: event.target.value || undefined })}>
      <option value="">Choose currency</option>
      {options.map(currency => <option key={currency}>{currency}</option>)}
    </select>}
    <div className="company-list-column-range-footer">
      <small id={descriptionId} className={error ? 'company-list-column-range-error' : ''} role={error ? 'status' : undefined}>{error || units}</small>
      {active && <button type="button" className="company-list-column-range-clear" aria-label={`Clear range for ${label}`}
        title="Clear this range" onClick={() => onChange(undefined)}><X size={12} /></button>}
    </div>
  </div>;
}
