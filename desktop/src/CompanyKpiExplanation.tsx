import { COMPANY_KPIS, columnLabel, companyListCalculationLabel, companyListUnit, companyListVariants, companyListWindowLabel } from './companyListModel';
import type { CompanyKpiUnit, CompanyListColumn } from './companyListModel';
import { expandedVariant } from './expandedKpis';

export type CompanyKpiExplanationProps = { column: CompanyListColumn; compact?: boolean };
type Explanation = { meaning: string; reading: string };
const catalogue = new Map(COMPANY_KPIS.map(kpi => [kpi.id, kpi]));

const concepts: Record<string, Explanation> = {
  revenue: {
    meaning: 'Sales recognised by the business during the report period, before subtracting its expenses.',
    reading: 'Revenue of 100 with expenses of 90 leaves 10 before any other charges in this simple example. Sales growth alone does not tell you whether profit or cash generation improved.',
  },
  book_equity: {
    meaning: 'The residual book value after reported liabilities are deducted from reported assets.',
    reading: 'Assets of 100 less liabilities of 60 leave book equity of 40. Book equity is an accounting amount; asset quality and the rights of different owners matter when assessing what shareholders own.',
  },
  total_assets: {
    meaning: 'The resources recognised on the balance sheet, such as cash, inventory, equipment and acquired rights.',
    reading: 'A reported asset amount of 100 combines assets valued under their accounting rules. Some may turn into cash quickly, while others may be hard to sell or require a write-down.',
  },
  tangible_assets: {
    meaning: 'Physical assets recorded in the accounts, such as buildings and equipment.',
    reading: 'Equipment bought for 10 with accumulated depreciation of 4 may have a book value of 6 before other adjustments. That figure does not establish its replacement cost or sale price.',
  },
  intangible_assets: {
    meaning: 'Recognised assets without a physical form, such as software or patent rights.',
    reading: 'An acquisition can create recorded intangible assets even when the operating business changes little. Internally developed brands may be absent from the balance sheet; check whether the saved definition includes goodwill.',
  },
  asset_intensity: {
    meaning: 'Reported tangible assets compared with one year of revenue: a way to describe how much physical capital the business uses.',
    reading: 'A ratio of 2× means 200 of tangible assets for 100 of annual revenue. Asset age, depreciation and leasing affect this comparison; it does not directly measure future maintenance spending.',
  },
  capex: {
    meaning: 'Capital expenditure: investment in assets used over several years, such as equipment, buildings or software.',
    reading: 'Total capex of 10 million represents 10 million of investment under the saved definition. Check the sign convention and what is included; this figure alone does not separate maintenance from expansion.',
  },
  gross_margin: {
    meaning: 'The share of sales left after the costs classified as the direct cost of those sales.',
    reading: 'Sales of 100 and cost of sales of 60 give a gross margin of 40%. The business still has other expenses to pay, and companies can classify costs differently.',
  },
  profit_margin: {
    meaning: 'The profit measure used by the provider as a share of revenue.',
    reading: 'A margin of 10% represents 10 of the defined profit for every 100 of revenue. Check the treatment of tax and unusual items before comparing this with operating margin or another profit definition.',
  },
  ebitda: {
    meaning: 'Earnings before interest, tax, depreciation and amortisation. Depreciation and amortisation allocate the cost of long-lived assets across periods.',
    reading: 'Operating profit of 10 plus depreciation and amortisation of 5 gives EBITDA of 15 in a simple calculation. Tax, investment, working capital and debt payments still affect the cash available.',
  },
  ebitda_margin: {
    meaning: 'EBITDA as a share of revenue, showing earnings before interest, tax, depreciation and amortisation for each unit of sales.',
    reading: 'EBITDA of 20 on revenue of 100 gives a 20% margin. Businesses that need substantial replacement investment can have much less cash available than this margin suggests.',
  },
  roa: {
    meaning: 'A measure of the earnings produced relative to the book value of assets used in the provider’s calculation.',
    reading: 'A 5% ratio represents 5 of the defined earnings for every 100 of assets. Check the earnings definition and asset valuation; depreciation and acquisitions can change the denominator.',
  },
  equity_ratio: {
    meaning: 'The share of reported assets financed by the equity used in the calculation.',
    reading: 'Equity of 40 against assets of 100 gives a 40% equity ratio. The remaining financing comes from liabilities, which can include supplier balances and other obligations as well as borrowings.',
  },
  current_ratio: {
    meaning: 'Current assets compared with current liabilities: the assets and obligations classified as short term in the accounts.',
    reading: 'A ratio of 1.5× means 150 of current assets for 100 of current liabilities. Inventory and customer receivables may take time to turn into cash, so the composition and payment dates matter.',
  },
  price_book: {
    meaning: 'The share price compared with the book value per share used by the provider.',
    reading: 'A P/B of 1.2× means a price of 120 for a book value of 100 on the same share basis. Asset write-downs, unrecorded assets and negative equity can make the ratio difficult to interpret.',
  },
  price_sales: {
    meaning: 'The share price compared with revenue per share.',
    reading: 'A P/S of 2× means a price of 2 for each 1 of sales per share used in the ratio. Two businesses with the same P/S can retain very different amounts of those sales as profit and cash.',
  },
  earnings_share: {
    meaning: 'The earnings attributed to one share under the provider’s calculation.',
    reading: 'Earnings of 100 divided across 50 shares give 2 per share in a simple example. Check the share-count basis, dilution and unusual earnings when comparing periods.',
  },
  dividend_payout: {
    meaning: 'The dividend compared with the earnings used by the provider.',
    reading: 'Dividends of 60 against earnings of 100 give a 60% payout ratio. This compares dividends with accounting earnings; small or negative earnings can make it hard to interpret, and cash coverage may differ.',
  },
  ebit_margin: {
    meaning: 'The share of sales left as operating profit, before interest and tax.',
    reading: 'A 10% operating margin means 10 of operating profit for every 100 of sales. Compare similar businesses and periods; operating profit is different from cash received.',
  },
  ebit: {
    meaning: 'Operating profit: revenue less operating expenses, before interest and tax.',
    reading: 'A negative amount means an operating loss for that period. A larger company can have more profit without having a higher profit margin.',
  },
  fcf: {
    meaning: 'Free cash flow reported by the data provider: a starting point for investigating cash available after investment.',
    reading: 'A positive amount shows cash left under that definition; a negative amount shows a funding need. Check investment, lease payments and unusual cash items before treating this as cash available to shareholders.',
  },
  cfo: {
    meaning: 'Cash generated or used by operating activities during the report period.',
    reading: 'Cash can differ from accounting profit when customers pay later, inventory changes or suppliers are paid. This amount is before cash classified as investing or financing.',
  },
  fcf_margin: {
    meaning: 'Provider free cash flow as a share of revenue.',
    reading: 'A 10% FCF margin means 10 of provider free cash flow for every 100 of revenue. The cash-flow definition still needs review; unusual cash items can change the ratio.',
  },
  pe: {
    meaning: 'The share price compared with earnings per share.',
    reading: 'A P/E of 20× means the price is 20 times the earnings used in the ratio. It does not promise repayment in 20 years. Losses or unusually small earnings can make the ratio difficult to interpret.',
  },
  dividend_yield: {
    meaning: 'Dividend per share compared with the share price used by the provider.',
    reading: 'A 5% yield corresponds to a dividend of 5 for a price of 100 under the selected definition. Future dividends and the share price can change.',
  },
  net_debt: {
    meaning: 'Debt after deducting cash included in the provider’s net-debt definition.',
    reading: 'A negative amount indicates net cash under that definition. Check which debts, leases and cash balances are included before comparing companies.',
  },
  debt_ebitda: {
    meaning: 'Net debt compared with earnings before interest, tax, depreciation and amortisation (EBITDA).',
    reading: 'A ratio of 2× means net debt is twice the EBITDA used in the calculation. This is not a two-year repayment forecast: tax, investment and other cash needs still matter.',
  },
  roe: {
    meaning: 'Profit compared with the book value of shareholders’ equity.',
    reading: 'A 15% return represents 15 of profit for every 100 of equity used in the ratio. Borrowing, buybacks and a small equity base can raise this figure; it is not the shareholder’s stock-market return.',
  },
  roic: {
    meaning: 'A measure of the earnings produced by the capital invested in the business.',
    reading: 'A 15% ratio represents 15 of the defined earnings for every 100 of capital used in the calculation. The treatment of earnings, cash, debt and goodwill matters; use the same definition when comparing companies.',
  },
  dcf: {
    meaning: 'Discounted cash flow (DCF) converts projected future cash into a value today. This starter value covers the whole equity of the company.',
    reading: 'The value includes both the explicit forecast cash and the terminal value once. Low and Mid are scenarios based on saved inputs; Low is not a guaranteed floor.',
  },
  npv: {
    meaning: 'Net present value (NPV) is the starter DCF value minus the dated price proxy for the whole equity.',
    reading: 'A positive NPV means the scenario value exceeds that saved price; a negative NPV means it falls short. It is an amount, not an annual return, and terminal value is already included through DCF.',
  },
  npv_percent: {
    meaning: 'The difference between scenario value and saved price, expressed as a percentage of that price.',
    reading: 'Value of 150 against a price of 100 gives a 50% surplus. The discount to value is instead 33.3%. A 30% discount to value requires a surplus of about 42.86%. This is not an expected annual return.',
  },
  terminal_pv: {
    meaning: 'The value assigned to cash beyond the explicit forecast years, discounted back to today.',
    reading: 'This is one component already included in DCF. It depends on long-term assumptions and should not be added to DCF again.',
  },
  terminal_share: {
    meaning: 'How much of the starter DCF value comes from the terminal value beyond the explicit forecast.',
    reading: 'A 40% share means 40 of every 100 of positive DCF value comes from the terminal component. Negative cash during the forecast can make this share exceed 100%.',
  },
};

// Provider IDs and labels checked against the bundled expanded-kpi-manifest.json.
// These explain the concepts; they do not replace the provider’s saved definitions.
const conceptIds: Record<string, string> = {
  revenue: 'revenue', provider_53: 'revenue', provider_5: 'revenue',
  equity: 'book_equity', provider_58: 'book_equity', provider_8: 'book_equity',
  assets: 'total_assets', provider_57: 'total_assets', provider_127: 'tangible_assets', provider_126: 'intangible_assets',
  tangible_assets_revenue: 'asset_intensity', provider_64: 'capex',
  provider_28: 'gross_margin', provider_30: 'profit_margin', provider_54: 'ebitda', provider_32: 'ebitda_margin',
  provider_34: 'roa', equity_assets: 'equity_ratio', provider_39: 'equity_ratio', provider_44: 'current_ratio',
  provider_4: 'price_book', provider_3: 'price_sales', provider_6: 'earnings_share', provider_20: 'dividend_payout',
  ebit_margin: 'ebit_margin', provider_29: 'ebit_margin', ebit: 'ebit', provider_55: 'ebit',
  fcf: 'fcf', provider_63: 'fcf', provider_23: 'fcf', cfo: 'cfo', provider_62: 'cfo', fcf_margin: 'fcf_margin', provider_24: 'fcf_margin', provider_31: 'fcf_margin',
  provider_2: 'pe', provider_1: 'dividend_yield', provider_148: 'dividend_yield',
  net_debt: 'net_debt', provider_60: 'net_debt', provider_42: 'debt_ebitda',
  provider_33: 'roe', provider_36: 'roic', provider_37: 'roic',
  mid_dcf: 'dcf', low_dcf: 'dcf', mid_npv: 'npv', low_npv: 'npv',
  mid_npv_percent: 'npv_percent', low_npv_percent: 'npv_percent',
  terminal_pv: 'terminal_pv', terminal_share: 'terminal_share',
};

const unitLabels: Record<CompanyKpiUnit, string> = {
  money: 'Currency millions', price: 'Currency per share', percent: 'Percent (%)', points: 'Percentage points (pp)',
  multiple: 'Multiple (×)', count: 'Count', number: 'Number', shares_millions: 'Million shares', date: 'Date', text: 'Text',
};

function unitGuide(unit: CompanyKpiUnit): string {
  if (unit === 'points') return 'Percentage points measure the difference between two percentages: a move from 10% to 12% is +2 percentage points.';
  if (unit === 'percent') return 'Enter percentage filters as shown: 10 means 10%, not 0.10. Whether a larger figure is useful depends on what the KPI measures.';
  if (unit === 'multiple') return 'A multiple compares two amounts: 2× means twice the denominator. Check the definition and whether that denominator is positive.';
  if (unit === 'money') return 'Amounts are in millions of the stated currency: 25 means 25 million. Monetary filters require a currency; sorting groups currencies before comparing amounts.';
  if (unit === 'price') return 'Amounts are per share in the stated currency. Monetary filters require a currency; amounts in different currencies are not directly comparable.';
  if (unit === 'shares_millions') return 'Share counts are in millions: 25 means 25 million shares. Different share classes or share-count dates may need reconciliation.';
  if (unit === 'count') return 'This counts observations or items, rather than measuring money or a percentage. Read the selected period and definition before setting a minimum or maximum.';
  if (unit === 'date') return 'This is a saved date. Its definition determines whether it describes a report, quote or snapshot.';
  if (unit === 'text') return 'This describes a company or its evidence. It is not a numeric score; use the relevant category filter when available.';
  return 'Read the selected unit and definition before comparing values. A larger number does not by itself mean a more attractive business.';
}

function explanationFor(column: CompanyListColumn): Explanation | undefined {
  if (column.kpiId === 'valuation_attractiveness') {
    const credit = Number(column.calculation.slice('terminal_'.length));
    return {
      meaning: `Scenario value above the saved equity price, counting ${credit}% of terminal value and all of the explicit forecast cash.`,
      reading: `${concepts.npv_percent.reading} ${credit === 100 ? 'This full-DCF comparison equals Mid NPV / price.' : credit === 0 ? 'Cash-only omits terminal value; it is not a liquidation valuation.' : 'The terminal reduction is a sensitivity assumption, not an estimated probability.'}`,
    };
  }
  if (column.kpiId === 'dcf_price_ratio') return {
    meaning: 'Starter DCF value for each unit of the dated saved equity price.',
    reading: 'A ratio of 1.5× means scenario value of 150 against a price of 100: the same comparison as 50% NPV / price. These produce the same ranking, so they are not independent signals.',
  };
  if (column.kpiId === 'cash_pv') return {
    meaning: 'The value today of cash projected during the explicit forecast years, before terminal value.',
    reading: 'Later cash is discounted to make it comparable with cash today. Negative forecast cash reduces this amount. DCF adds the terminal component once.',
  };
  if (column.kpiId === 'cash_price_coverage') return {
    meaning: 'How much of the saved equity price is covered by the present value of explicit forecast cash alone.',
    reading: 'Coverage of 80% means 80 of forecast cash value for a price of 100. The cash-only surplus is then −20%. Coverage omits terminal value and is not a forecast return.',
  };
  if (column.kpiId === 'positive_fcf' || column.kpiId === 'positive_ebit') return {
    meaning: `How many of the selected annual reports show strictly positive ${column.kpiId === 'positive_fcf' ? 'provider free cash flow' : 'operating profit (EBIT)'}.`,
    reading: 'For a five-report window, 4 / 5 means four positive observations; the fifth may be zero or negative. All requested observations must be available. The count describes past consistency, not future certainty.',
  };
  if (column.kpiId === 'quarter_margin_change') return {
    meaning: 'The change in operating margin from the same fiscal quarter one year earlier.',
    reading: 'A margin rising from 10% to 12% gives +2 percentage points here, rather than +20% relative growth.',
  };
  if (column.kpiId === 'price_age') return {
    meaning: 'The number of days from the saved quote to the fixed research snapshot.',
    reading: '30 means the quote was 30 days old when the snapshot was made. Its age does not update as today’s date changes.',
  };
  if (column.kpiId === 'cash_factor' || column.kpiId === 'cash_factor_30') return {
    meaning: `How much all projected cash would have to be scaled to ${column.kpiId === 'cash_factor_30' ? 'put the saved price 30% below scenario value' : 'make scenario value equal the saved price'}, holding the other assumptions fixed.`,
    reading: 'A factor of 1.2× means 20% more cash in every projected annual and terminal amount; 0.8× means 20% less. This is not an annual growth rate. Negative funding cash scales too.',
  };
  return concepts[conceptIds[column.kpiId]];
}

export function CompanyKpiExplanation({ column, compact = false }: CompanyKpiExplanationProps) {
  const metric = catalogue.get(column.kpiId);
  const supported = metric && companyListVariants(metric).some(variant => variant.window === column.window && variant.calculation === column.calculation);
  if (!metric || !supported) return <section className="company-kpi-explanation" aria-label="Unavailable KPI explanation"><h3>Unavailable KPI</h3><p>This exact KPI, period or calculation is unsupported. Choose a supported column to inspect its definition.</p></section>;

  const unit = companyListUnit(column), variant = expandedVariant(column);
  const explanation = explanationFor(column);
  const calculation = column.calculation.replace(/^provider:/, '');
  const growth = calculation === 'growth' || calculation === 'cagr';
  const growthReading = 'This is the annual growth of the underlying KPI over the selected window. For example, 100 growing to 121 over two years is 10% per year compounded, even if individual years differ. For a margin or ratio, this measures growth in the ratio itself.';
  const summary: Record<string, string> = {
    average: 'The selected column averages the annual observations.', mean: 'The selected column uses the provider’s average for the chosen period.',
    median: 'The selected column uses the middle annual observation after sorting.', min: 'The selected column uses the lowest annual observation.',
    max: 'The selected column uses the highest annual observation.', high: 'The selected column uses the provider’s highest observation.',
    low: 'The selected column uses the provider’s lowest observation.', sum: 'The selected column sums the observations; it is not an average or necessarily a compounded return.',
    quarter: 'The selected column is the provider’s quarter-growth comparison, not the underlying KPI level. Check its comparison period in the saved source.',
    psh: 'The selected column expresses the underlying amount per share.',
  };
  const reading = growth ? growthReading : calculation === 'quarter' ? summary.quarter : [summary[calculation], explanation?.reading && `${summary[calculation] ? 'For an individual underlying observation: ' : ''}${explanation.reading}`].filter(Boolean).join(' ');
  const scope = `${companyListWindowLabel(column)} · ${companyListCalculationLabel(column)}`;
  const basis = variant?.currencyBasis;
  const units = column.kpiId === 'price_age' ? 'Days at the fixed snapshot' : unitLabels[unit] + (['money', 'price'].includes(unit) && basis && basis !== 'none' ? ` · ${basis === 'report' ? 'reporting currency' : basis === 'quote' ? 'quote currency' : basis === 'unverified' ? 'currency basis unverified' : basis}` : '');
  const timing = metric.source
    ? `Saved source: ${metric.source}${metric.snapshot ? `. Downloaded ${metric.snapshot}` : ''}. The download date may differ from the underlying report or quote date. Inspect a value for the dates available for that company.`
    : metric.category === 'Starter valuation'
      ? 'This is the fixed starter snapshot. Each comparison keeps its saved inputs and dated price; edited or reviewed company valuations are separate in Valuation.'
      : 'Each company uses its own saved observation dates. A report period end, publication date and quote date are different dates; inspect a value for the available details.';

  return <section className="company-kpi-explanation" data-compact={compact} aria-label={`About ${metric.label}`}>
    <h3>What this measures</h3>
    <p>{explanation?.meaning ?? metric.description}</p>
    <div className="company-kpi-explanation-selection" aria-label="Selected KPI variant" title={columnLabel(column)}>
      <dl><div><dt>Period</dt><dd>{companyListWindowLabel(column)}</dd></div><div><dt>Calculation</dt><dd>{companyListCalculationLabel(column)}</dd></div><div><dt>Units</dt><dd>{units}</dd></div></dl>
    </div>
    <h3>How to read it</h3>
    {reading && <p>{reading}</p>}
    <p>{unitGuide(unit)}</p>
    {!compact && <details className="company-kpi-explanation-source"><summary>Definition and saved source</summary><p>{metric.description}</p><p><strong>{metric.source ? 'Saved provider definition' : 'Underlying KPI formula'}:</strong> {metric.formula}</p><p><strong>Selected variant:</strong> {scope}.</p><p>{timing}</p><p>An em dash means the value or comparison is unavailable. It is not zero; inspect the cell for its reason.</p></details>}
    {compact && <p className="company-kpi-explanation-date">{timing}</p>}
  </section>;
}

export default CompanyKpiExplanation;
