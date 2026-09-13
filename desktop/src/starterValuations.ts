import type { FinancialCompany, FinancialIndex, SourcedReport } from './financialData';
import type { CompanyEntry } from './listingCatalogue';
import type { MarketReport } from './marketData';
import { blankValuation, scenarioKeys, validAmount, validDay, type StarterOrigin, type ValuationDraft } from './valuation';
import { cashCalibration, cashCalibrationId, classifyCashHistory, historicalCashFactor, type CashDispersionGroup, type HistoricalCashFactor } from './cashUncertainty';
import { resolveTerminalSale, seedTerminalCash } from './terminalValue';
import type { CashComponents } from './cashComponents';

export type StarterSettings = { historyYears: 5 | 10; weights: number[]; spreadPercent: number; spreadStepPercent: number; projection: 'latest' | 'trend' | 'flat'; method?: StarterOrigin['id']; rangeMode?: 'historical' | 'percentage'; tailWideningPercent?: number; calibrationId?: string; terminalMethod?: 'historical-median-v1' | 'final-forecast' };
export const defaultStarterWeights = (historyYears: 5 | 10) => historyYears === 10 ? [19, 17, 15, 13, 11, 9, 7, 5, 3, 1] : [30, 25, 20, 15, 10];
export const defaultStarterSettings: StarterSettings = { historyYears: 5, weights: defaultStarterWeights(5), spreadPercent: 10, spreadStepPercent: 10, projection: 'latest', rangeMode: 'historical', tailWideningPercent: 10, terminalMethod: 'historical-median-v1' };
export function settingsForStarterOrigin(origin?: StarterOrigin): StarterSettings {
  if (!origin) return { ...defaultStarterSettings, weights: [...defaultStarterSettings.weights] };
  return { method: origin.id, historyYears: origin.id === 'weighted-cash-starter-v1' ? 5 : origin.historyYears, weights: [...origin.weights], spreadPercent: origin.spreadPercent,
    spreadStepPercent: origin.id === 'weighted-cash-starter-v1' ? 0 : origin.spreadStepPercent, projection: origin.id === 'weighted-cash-starter-v1' ? 'flat' : origin.projection,
    rangeMode: origin.id === 'empirical-cash-starter-v3' ? origin.rangeMode : 'percentage', tailWideningPercent: origin.id === 'empirical-cash-starter-v3' ? origin.tailWideningPercent : 10,
    ...(origin.id === 'empirical-cash-starter-v3' ? { calibrationId: origin.calibrationId } : {}),
    terminalMethod: origin.id === 'empirical-cash-starter-v3' ? origin.terminalMethod ?? 'final-forecast' : 'final-forecast' };
}
export type StarterAnnualInput = {
  year: number; start: string; end: string; reportDate: string | null; sourceId: string; sourceAsOf: string;
  sourcePath: string | null; sourceHash: string | null; reportingCurrency: string; quotedCurrency: string | null;
  rawCashFlow: number | null; reportedCashFlow: number | null; currencyRatio: number | null; cashFlow: number | null;
  weightPercent: number; effectiveWeightPercent: number;
  components: CashComponents;
  possiblePlaceholder: boolean;
};
export type StarterPrice = {
  year: number; sourceId: string; reportStart: string; reportEnd: string; reportDate: string | null; sourceAsOf: string;
  priceDate: string; currency: string | null; shares: number | null; close: number | null; marketCap: number;
  fxRate: number | null; fxDate: string | null; method: 'local' | 'sek'; sourcePath: string | null; sourceHash: string | null;
};
export type StarterEvidence = {
  method: string; asOf: string; snapshotDate: string | null; currency: string; amountBasis: 'reporting' | 'quote';
  annual: StarterAnnualInput[]; latestAnnual: StarterAnnualInput | null; anchor: number | null; anchorBasis: string;
  weights: number[]; weightTotal: number; includedWeightTotal: number; spreadPercent: number;
  historyYears: 5 | 10; projection: 'latest' | 'trend' | 'flat'; spreadStepPercent: number;
  rangeMode: 'historical' | 'percentage'; tailWideningPercent: number; manualCashRequired: boolean;
  uncertainty: { status: 'historical' | 'percentage' | 'unavailable'; reason: string | null; group: CashDispersionGroup | null; dispersion: number | null; scale: number | null; calibrationId: string | null; targetCoverage: number | null };
  trend: { slope: number | null; intercept: number | null; weightedMeanYear: number | null; usableTrendYears: number };
  forecast: { year: number; mid: number | null; low: number | null; high: number | null; spreadPercent: number; rangeBasis?: 'historical' | 'assumed-tail' | 'percentage' | 'unavailable'; halfWidth?: number | null; factor?: number | null; calibrationGroup?: CashDispersionGroup | 'global' | null; support?: HistoricalCashFactor['support'] | null }[];
  requiredReturn: number; years: number; price: StarterPrice | null; capitalDate: string | null; explanation: string[];
  terminal?: { method: 'historical-median-v1'; median: number | null; historyCount: number; spreadPercent: number };
};
type Entry = Pick<CompanyEntry, 'id' | 'display_name' | 'report_currency' | 'stock_currency' | 'source_as_of'> & Partial<Pick<CompanyEntry, 'sector_id' | 'branch_id'>>;
type FinancialContext = Pick<FinancialIndex, 'as_of' | 'sources' | 'market'> & Partial<Pick<FinancialIndex, 'id' | 'taxonomy_sha256'>>;
const days = (later: string, earlier: string) => (Date.parse(later) - Date.parse(earlier)) / 86400000;
const fullYear = (r: SourcedReport) => days(r.end, r.start) + 1 >= 330 && days(r.end, r.start) + 1 <= 400;
const currencyCode = (value: unknown): value is string => typeof value === 'string' && /^[A-Z]{3}$/.test(value);
const numeric = (value: unknown) => validAmount(value) ? value : null;
const positive = (value: unknown): value is number => validAmount(value) && value > 0;
const hasConversion = (r: SourcedReport) => positive(r.currency_ratio);
const emptyAmounts = (r: SourcedReport) => Object.values(r.raw).some(value => value === 0) && Object.values(r.raw).every(value => value === null || value === 0);

export function validateStarterSettings(settings: StarterSettings): void {
  if (![5, 10].includes(settings.historyYears)) throw new Error('Choose five or ten annual history years.');
  if (!Array.isArray(settings.weights) || settings.weights.length !== settings.historyYears || !settings.weights.every(value => Number.isFinite(value) && value >= 0 && value <= 100) || Math.abs(settings.weights.reduce((sum, value) => sum + value, 0) - 100) > 1e-6) throw new Error(`Enter ${settings.historyYears} annual weights from 0% to 100% that add to 100%.`);
  if (!Number.isFinite(settings.spreadPercent) || settings.spreadPercent < 0 || settings.spreadPercent > 100) throw new Error('Enter a sensitivity spread from 0% to 100%.');
  if (!Number.isFinite(settings.spreadStepPercent) || settings.spreadStepPercent < 0 || settings.spreadStepPercent > 100) throw new Error('Enter an annual sensitivity spread increment from 0 to 100 percentage points.');
  if (!['latest', 'trend', 'flat'].includes(settings.projection)) throw new Error('Choose a latest-cash, trend or weighted flat projection.');
  if (settings.method !== undefined && !['weighted-cash-starter-v1', 'weighted-cash-starter-v2', 'empirical-cash-starter-v3'].includes(settings.method)) throw new Error('Unsupported starter method.');
  if (settings.rangeMode !== undefined && !['historical', 'percentage'].includes(settings.rangeMode)) throw new Error('Choose historical-error ranges or percentage sensitivities.');
  if (settings.tailWideningPercent !== undefined && (!Number.isFinite(settings.tailWideningPercent) || settings.tailWideningPercent < 0 || settings.tailWideningPercent > 100)) throw new Error('Enter assumed tail widening from 0% to 100% of the historical cash scale per year.');
  if (settings.calibrationId !== undefined && (typeof settings.calibrationId !== 'string' || !/^[a-zA-Z0-9-]{1,150}$/.test(settings.calibrationId))) throw new Error('Enter a valid historical calibration identifier.');
  if (settings.terminalMethod !== undefined && !['historical-median-v1', 'final-forecast'].includes(settings.terminalMethod)) throw new Error('Choose a supported terminal cash method.');
  if ((settings.method === 'weighted-cash-starter-v1' || settings.method === 'weighted-cash-starter-v2') && (settings.projection === 'latest' || settings.rangeMode === 'historical')) throw new Error('Legacy starters retain their original projection and percentage sensitivities. Apply the new method to change these settings.');
  if (settings.method === 'weighted-cash-starter-v1' && (settings.historyYears !== 5 || settings.projection !== 'flat' || settings.spreadStepPercent !== 0)) throw new Error('Legacy starters retain five history years, a flat projection and a constant spread. Apply the new method to change these settings.');
}

// This is a user-editable sensitivity model. Source FCF remains an unreviewed
// proxy; no inferred FCFF bridge, cash-distribution policy or debt deduction.
function buildWeightedStarter(entry: Entry, history: FinancialCompany | null, financial: FinancialContext | null = null, selectedAsOf?: string, settings: StarterSettings = defaultStarterSettings): { draft: ValuationDraft; issues: string[]; evidence: StarterEvidence } {
  validateStarterSettings(settings);
  if (history && history.id !== entry.id) throw new Error('Financial history does not match the selected company.');
  const snapshotDate = financial?.as_of ?? [...history?.annual ?? [], ...history?.quarterly ?? []].map(r => r.source_as_of).filter(validDay).sort().at(-1) ?? null;
  const asOf = selectedAsOf ?? snapshotDate ?? entry.source_as_of;
  if (!validDay(asOf)) throw new Error('Enter a valid starter valuation date.');
  const issues: string[] = [], weights = [...settings.weights], { spreadPercent, spreadStepPercent, historyYears, projection } = settings;
  const method = settings.method ?? 'weighted-cash-starter-v2', legacy = method === 'weighted-cash-starter-v1';
  const reports = (history?.annual ?? []).filter(r => r.end <= asOf && r.source_as_of <= asOf && (r.report_date === null || r.report_date <= asOf)).sort((a, b) => b.end.localeCompare(a.end) || b.year - a.year);
  const latest = reports[0] ?? null;
  const marketFor = (r: SourcedReport) => history?.market.find(m => m.year === r.year && m.source_id === r.source_id) ?? null;
  const nativeCurrency = latest?.currency ?? (currencyCode(entry.report_currency) ? entry.report_currency : currencyCode(entry.stock_currency) ? entry.stock_currency : 'SEK');
  const priceFor = (currency: string): { row: MarketReport; report: SourcedReport; amount: number; method: 'local' | 'sek' } | null => {
    // A prior year's usable price must not bypass current share/scale flags or
    // silently price the current cash window using an older ownership claim.
    const choices = (history?.market ?? []).filter(m => latest && m.year === latest.year && m.source_id === latest.source_id && validDay(m.price_date) && m.price_date <= asOf);
    for (const row of choices) {
      const report = reports.find(r => r.year === row.year && r.source_id === row.source_id);
      if (!report || row.flags.some(flag => flag !== 'missing_fx')) continue;
      if (row.currency === currency && positive(row.local)) return { row, report, amount: row.local, method: 'local' };
      if (currency === 'SEK' && positive(row.sek) && positive(row.fx_rate) && validDay(row.fx_date) && row.fx_date <= row.price_date! && !row.flags.includes('missing_fx')) return { row, report, amount: row.sek, method: 'sek' };
    }
    return null;
  };
  let selectedPrice = priceFor(nativeCurrency), amountBasis: StarterEvidence['amountBasis'] = 'reporting', currency = nativeCurrency;
  const latestQuoteCurrency = latest ? marketFor(latest)?.currency : null;
  if (!selectedPrice && currencyCode(latestQuoteCurrency) && latestQuoteCurrency !== nativeCurrency) {
    // Source DATA.md documents raw monetary values in the archived listing's
    // quote currency. Each compared row must establish that currency separately.
    currency = latestQuoteCurrency; amountBasis = 'quote'; selectedPrice = priceFor(currency);
  }
  const amountFor = (r: SourcedReport, key: string): number | null => {
    if (!hasConversion(r) || r.currency !== nativeCurrency) return null;
    if (amountBasis === 'quote') return marketFor(r)?.currency === currency ? numeric(r.raw[key]) : null;
    return numeric(r.values[key]);
  };
  const inputFor = (r: SourcedReport, index: number): StarterAnnualInput => {
    const source = financial?.sources.find(source => source.id === r.source_id);
    return { year: r.year, start: r.start, end: r.end, reportDate: r.report_date, sourceId: r.source_id, sourceAsOf: r.source_as_of,
      sourcePath: source?.path ?? null, sourceHash: source?.sha256 ?? null, reportingCurrency: r.currency, quotedCurrency: marketFor(r)?.currency ?? null,
      rawCashFlow: numeric(r.raw.free_cash_flow), reportedCashFlow: hasConversion(r) ? numeric(r.values.free_cash_flow) : null, currencyRatio: r.currency_ratio,
      cashFlow: amountFor(r, 'free_cash_flow'), weightPercent: weights[index], effectiveWeightPercent: 0,
      components: { operating: amountFor(r, 'cash_flow_from_operating_activities'), investing: amountFor(r, 'cash_flow_from_investing_activities'), financing: amountFor(r, 'cash_flow_from_financing_activities'), netCash: amountFor(r, 'cash_flow_for_the_year'), providerFcf: amountFor(r, 'free_cash_flow') }, possiblePlaceholder: emptyAmounts(r) };
  };
  if (!latest) issues.push('No annual financial period is available by the selected valuation date. Cash-flow inputs remain unavailable.');
  if (snapshotDate && asOf > snapshotDate) issues.push(`Financial snapshot ${snapshotDate} is older than valuation ${asOf}; source amounts and dated prices have not been refreshed.`);
  if (!latest && !currencyCode(entry.report_currency)) issues.push(`Reporting currency is unavailable. ${currency} is an editable form currency; no company amounts have been inferred.`);
  let eligible = !!latest;
  if (latest && !fullYear(latest)) { eligible = false; issues.push(`Latest annual period ${latest.start}–${latest.end} is outside 330–400 days. An older full year is not substituted.`); }
  if (latest && days(asOf, latest.end) > 550) { eligible = false; issues.push(`Latest annual period ended ${latest.end}, more than 550 days (about 18 months) before ${asOf}. A stale cash proxy is not projected.`); }
  const newerWithheld = history?.withheld.find(r => {
    const sourceDate = financial?.sources.find(source => source.id === r.source_id)?.as_of ?? [...history.annual, ...history.quarterly].find(report => report.source_id === r.source_id)?.source_as_of;
    if (r.period !== 5 || sourceDate && sourceDate > asOf || validDay(r.end) && r.end > asOf || validDay(r.published) && r.published > asOf) return false;
    return !latest || (validDay(r.end) ? r.end > latest.end : r.year > latest.year);
  });
  if (newerWithheld) { eligible = false; issues.push(`A newer annual period (${newerWithheld.year}, ${newerWithheld.end || 'end unavailable'}) was withheld: ${newerWithheld.reason}. The older period is not substituted.`); }
  if (latest && emptyAmounts(latest)) { eligible = false; issues.push('All saved monetary fields in the latest annual report are zero or missing. This may be a source placeholder; no zero-business forecast is generated.'); }
  if (latest && amountFor(latest, 'free_cash_flow') === null) { eligible = false; issues.push('Latest annual provider cash-flow proxy, saved currency conversion or matching currency lineage is unavailable. Older cash flows are not substituted.'); }
  const annual: StarterAnnualInput[] = [];
  if (eligible) for (let i = 0; i < Math.min(historyYears, reports.length); i++) {
    const r = reports[i], previous = reports[i - 1];
    if (i && (r.year !== previous.year - 1 || days(previous.start, r.end) < 0 || days(previous.start, r.end) > 35)) { issues.push(`The annual history has a gap before ${previous.year}. Older nonconsecutive periods are excluded.`); break; }
    if (!fullYear(r) || r.currency !== nativeCurrency || emptyAmounts(r) || amountFor(r, 'free_cash_flow') === null) { issues.push(`The ${r.year} period is not a comparable full year with an available cash proxy and matching currency. The window stops here; no older value fills the gap.`); break; }
    annual.push(inputFor(r, i));
  }
  const includedWeightTotal = annual.reduce((sum, row) => sum + row.weightPercent, 0);
  annual.forEach(row => { row.effectiveWeightPercent = includedWeightTotal > 0 ? 100 * row.weightPercent / includedWeightTotal : 0; });
  const anchor = includedWeightTotal > 0 ? annual.reduce((sum, row) => sum + row.cashFlow! * row.weightPercent, 0) / includedWeightTotal : null;
  if (annual.length && annual.length < historyYears && includedWeightTotal > 0) issues.push(`Only ${annual.length} consecutive comparable annual periods are available. Included weights total ${includedWeightTotal}%; normalization divides by ${includedWeightTotal}, making their effective weights sum to 100%. Missing years are not zeros.`);
  if (annual.length && includedWeightTotal === 0) issues.push('The available periods have zero assigned weight. A weighted cash proxy is unavailable until at least one included period has a positive weight.');
  if (anchor !== null && annual[0] && Math.sign(annual[0].cashFlow!) !== Math.sign(anchor)) issues.push(`Latest annual cash proxy (${annual[0].cashFlow} ${currency} m) and weighted cash proxy (${anchor} ${currency} m) have different signs. The weighted sensitivity retains every included signed amount; it does not establish a recovery or deterioration forecast.`);
  const usableTrendYears = annual.filter(row => row.weightPercent > 0).length;
  const weightedMeanYear = anchor === null ? null : annual.reduce((sum, row) => sum + (row.year - latest!.year) * row.weightPercent, 0) / includedWeightTotal;
  const variance = weightedMeanYear === null ? null : annual.reduce((sum, row) => sum + row.weightPercent * (row.year - latest!.year - weightedMeanYear) ** 2, 0);
  const covariance = weightedMeanYear === null || anchor === null ? null : annual.reduce((sum, row) => sum + row.weightPercent * (row.year - latest!.year - weightedMeanYear) * (row.cashFlow! - anchor), 0);
  const slope = covariance === null || variance === null ? null : usableTrendYears < 2 || variance === 0 ? 0 : covariance / variance;
  const intercept = anchor === null || slope === null || weightedMeanYear === null ? null : anchor - slope * weightedMeanYear;
  const trend = { slope, intercept, weightedMeanYear, usableTrendYears };
  if (projection === 'trend' && usableTrendYears === 1) issues.push('Only one positive-weight annual observation is available. The trend slope is assumed zero; no change over time can be estimated from one observation.');
  let price: StarterPrice | null = null;
  if (selectedPrice) {
    const { row, report, amount, method } = selectedPrice, source = financial?.market?.sources.find(source => source.id === row.source_id);
    price = { year: row.year, sourceId: row.source_id, reportStart: report.start, reportEnd: report.end, reportDate: report.report_date, sourceAsOf: source?.as_of ?? report.source_as_of,
      priceDate: row.price_date!, currency: row.currency, shares: row.shares, close: row.price, marketCap: amount, fxRate: row.fx_rate, fxDate: row.fx_date, method,
      sourcePath: source?.prices.path ?? null, sourceHash: source?.prices.sha256 ?? null };
    if (days(asOf, price.priceDate) > 550) issues.push(`The retained price is historical (${price.priceDate}), over 550 days before valuation. Replace it with a reviewed dated price before using this comparison.`);
  } else issues.push(`No usable dated equity market value with matching report/source lineage is available in ${currency}. Price remains unavailable; no FX rate is invented.`);
  const requiredReturn = 10, years = 10;
  const forecast: StarterEvidence['forecast'] = Array.from({ length: years }, (_, i) => {
    const year = i + 1, spread = spreadPercent + spreadStepPercent * i;
    const mid = projection === 'flat' ? anchor : intercept === null || slope === null ? null : intercept + slope * year;
    return { year, mid, low: mid === null ? null : mid - Math.abs(mid) * spread / 100, high: mid === null ? null : mid + Math.abs(mid) * spread / 100, spreadPercent: spread };
  });
  const basisDescription = amountBasis === 'quote'
    ? `Saved monetary amounts are in ${currency}, the archived quote currency established separately for each annual source. Historical provider conversions remain embedded; these are not constant-FX cash flows and no current FX rate is inferred.`
    : `Amounts are in reporting currency ${currency}, recovered by dividing each saved monetary amount by its own positive saved currency ratio.`;
  const anchorBasis = anchor === null ? 'Weighted cash proxy unavailable.' : `Weighted provider cash-flow proxy = sum(cash proxy × entered annual weight) ÷ ${includedWeightTotal}. ${annual.length} annual periods are included, newest first. This is a sensitivity input, not verified distributable cash.`;
  const explanation = [basisDescription,
    'Saved provider FCF proxy; not verified cash to shareholders. Lease principal, interest classification, acquisitions, restricted or regulatory capital and financing require review. For banks and insurers, equity-capital requirements and actual distributions need specific review; this proxy is not FCFF. No missing FCF is inferred from operating or investing cash flow.',
    legacy ? `Project the signed weighted proxy flat for ${years} full years. Low = mid − abs(mid) × ${spreadPercent}%; high = mid + abs(mid) × ${spreadPercent}%. The spread is a user-chosen sensitivity, not a confidence interval or estimated probability.`
      : `${projection === 'trend' ? 'Fit a weighted linear trend to the included annual cash proxies, with the latest historical year at time zero. Project from the fitted intercept: mid at year t = intercept + slope × t.' : 'Hold the signed weighted cash proxy flat through the forecast.'} Project ${years} full years. Low/high = mid ± abs(mid) × [${spreadPercent}% + ${spreadStepPercent} percentage points × (year − 1)]. The spread is a user-chosen sensitivity, not a confidence interval or estimated probability. Bands are not capped at 100%; signed paths may cross zero.`,
    `Required nominal equity return is an editable ${requiredReturn}% assumption. Final equity sale at year ${years} assumes year ${years + 1} cash equals year ${years} cash, with zero mature growth and no selling costs, capitalized at ${requiredReturn}%. Nonpositive cash receives zero assumed sale proceeds; this is not a recovery appraisal.`,
    'Treating this proxy as cash available to common shareholders is an explicit modeling assumption. Negative payments model hypothetical additional shareholder funding; accounting or cash-flow losses do not establish a personal obligation. Recovery is unavailable, and debt or asset values are not added to or deducted from the proxy DCF.',
  ];
  const draft = blankValuation(entry.display_name, currency, asOf);
  draft.title = `${entry.display_name} — ${legacy ? 'weighted cash starter' : projection === 'trend' ? 'weighted cash trend starter' : 'weighted cash starter'}`;
  draft.starterOrigin = legacy ? { id: 'weighted-cash-starter-v1', asOf, weights: [...weights], spreadPercent }
    : { id: 'weighted-cash-starter-v2', asOf, historyYears, weights: [...weights], spreadPercent, spreadStepPercent, projection: projection as 'trend' | 'flat' };
  draft.years = years;
  draft.marketCap = price?.marketCap ?? null;
  draft.priceDate = price?.priceDate ?? '';
  draft.priceSource = price ? `Saved publication-window equity price: ${price.shares} million reported shares × ${price.close} ${price.currency} close on ${price.priceDate}; annual period ${price.reportStart}–${price.reportEnd}, published ${price.reportDate ?? 'date unavailable'}, source ${price.sourceId}, saved ${price.sourceAsOf}. ${price.method === 'sek' ? `Observed conversion to SEK: ${price.fxRate}, FX date ${price.fxDate}.` : `Uses the matching ${currency} local market value; no additional FX conversion.`} This listing/share basis requires review; it is not a sum across share classes or a live quote.` : `Historical equity price unavailable in ${currency}; enter a reviewed price and ownership basis.`;
  const tangibleEquity = latest && eligible && amountFor(latest, 'total_equity') !== null && amountFor(latest, 'intangible_assets') !== null ? amountFor(latest, 'total_equity')! - amountFor(latest, 'intangible_assets')! : null;
  draft.capital.tangibleEquity = tangibleEquity;
  for (const key of scenarioKeys) {
    const final = forecast.at(-1)![key];
    draft.scenarios[key] = { cashFlows: forecast.map(row => row[key]), discountRate: requiredReturn, terminalEquity: final === null ? null : Math.max(0, final) / (requiredReturn / 100), recoveryEquity: null, recoveryYear: null,
      rationale: legacy ? `${key === 'mid' ? 'Mid uses the weighted historical cash proxy.' : `${key === 'low' ? 'Low subtracts' : 'High adds'} ${spreadPercent}% of the absolute weighted cash proxy.`} Annual payments remain flat. ${explanation[1]} ${explanation[3]} ${explanation[4]} Starter method weighted-cash-starter-v1; source basis ${asOf}.`
        : `${key === 'mid' ? `Mid follows the ${projection === 'trend' ? 'weighted historical trend' : 'weighted historical cash proxy held flat'}.` : `${key === 'low' ? 'Low subtracts' : 'High adds'} the widening percentage of each year's absolute mid cash.`} ${explanation[2]} ${explanation[1]} ${explanation[3]} ${explanation[4]} Starter method ${method}; source basis ${asOf}.` };
  }
  draft.notes = {
    business: `Unreviewed weighted cash starter. ${anchorBasis} ${explanation[1]}`,
    macro: 'No macro forecast, listing-country score or industry multiplier is inserted. Review verified company exposure before adjusting this sensitivity.',
    financing: `Tangible equity is a nullable source proxy: latest reported total equity minus intangibles (${latest?.end ?? 'date unavailable'}). Common-equity and minority claims are not reconciled. Average tangible operating capital, NOPAT, gross debt and surplus cash remain unavailable. The pack supplies net debt, which is not substituted for gross debt. ${basisDescription}`,
    recovery: 'No reviewed asset, prior-claim, tax, cost or payment schedule is available. Every recovery input remains missing; no book-value floor is assumed.',
    decision: `${explanation[2]} ${explanation[3]} ${explanation[4]} ${issues.join(' ')}`,
  };
  return { draft, issues, evidence: { method, asOf, snapshotDate, currency, amountBasis, annual, latestAnnual: latest ? inputFor(latest, 0) : null, anchor, anchorBasis, weights, weightTotal: weights.reduce((sum, value) => sum + value, 0), includedWeightTotal, spreadPercent, spreadStepPercent, projection, historyYears, trend, forecast, requiredReturn, years, price, capitalDate: tangibleEquity === null ? null : latest!.end, explanation,
    rangeMode: 'percentage', tailWideningPercent: 10, manualCashRequired: false, uncertainty: { status: 'percentage', reason: null, group: null, dispersion: null, scale: null, calibrationId: null, targetCoverage: null } } };
}

export function buildStarterValuation(entry: Entry, history: FinancialCompany | null, financial: FinancialContext | null = null, selectedAsOf?: string, settings: StarterSettings = defaultStarterSettings): { draft: ValuationDraft; issues: string[]; evidence: StarterEvidence } {
  validateStarterSettings(settings);
  if (settings.method === 'weighted-cash-starter-v1' || settings.method === 'weighted-cash-starter-v2') return buildWeightedStarter(entry, history, financial, selectedAsOf, settings);
  // Keep the v1/v2 source/price engine and draft text intact for exact saved-draft
  // reproduction. Only this v3 wrapper supplies the new model and range policy.
  const built = buildWeightedStarter(entry, history, financial, selectedAsOf, { ...settings, method: 'weighted-cash-starter-v2', projection: settings.projection === 'latest' ? 'flat' : settings.projection, rangeMode: 'percentage' });
  const { draft, evidence, issues } = built;
  const rangeMode = settings.rangeMode ?? 'historical', tailWideningPercent = settings.tailWideningPercent ?? 10;
  const calibrationId = settings.calibrationId ?? cashCalibrationId, projection = settings.projection;
  const manualCashRequired = entry.sector_id === '1' && !['75', '76'].includes(entry.branch_id ?? '');
  const classification = classifyCashHistory(evidence.annual.map(row => row.reportedCashFlow));
  const latest = evidence.annual[0] ?? null;
  const defaultWeights = settings.historyYears === 5 && settings.weights.every((weight, index) => weight === defaultStarterWeights(5)[index]);
  let reason: string | null = null;
  if (manualCashRequired) reason = 'This financial business requires manually reviewed distributable-equity cash, regulatory capital and financing inputs. Generic provider FCF is retained as historical evidence and is not projected automatically.';
  else if (typeof entry.sector_id !== 'string' || !/^(?:[1-9]|10)$/.test(entry.sector_id)) reason = 'Company classification is unavailable or unrecognized; eligibility for the operating/property calibration population is unverified.';
  else if (settings.historyYears !== 5 || evidence.annual.length !== 5) reason = 'Historical-error ranges require exactly five consecutive comparable full annual reports. Partial or ten-year settings use explicit percentage sensitivities.';
  else if (evidence.amountBasis !== 'reporting') reason = 'The saved quote-currency basis differs from the native reporting currency used in calibration. These ranges are explicit percentage sensitivities.';
  else if (!defaultWeights || projection === 'flat') reason = 'Custom history weights or the weighted-flat model were not used for these calibrated factors. These ranges are explicit percentage sensitivities.';
  else if (classification.scale === null || classification.scale <= 0 || classification.group === null) reason = 'The historical cash scale is unavailable or zero; a historical-error range cannot be estimated. Missing bounds do not imply certain future cash.';
  else if (calibrationId !== cashCalibrationId) reason = 'The saved historical calibration version is unavailable. Its factors are not replaced with another version.';
  else if (!financial || financial.id !== cashCalibration.provenance.sourcePack.sha256 || financial.taxonomy_sha256 !== cashCalibration.taxonomySha256) reason = 'The financial pack or company taxonomy does not match this source-bound calibration. These ranges are explicit percentage sensitivities until that source basis is evaluated.';
  else if (evidence.annual.some(row => !cashCalibration.sourceFiles.some(source => source.id === row.sourceId && source.sha256 === row.sourceHash && source.path === row.sourcePath && source.as_of === row.sourceAsOf))) reason = 'Annual source identities do not match the retained calibration source files. These ranges are explicit percentage sensitivities.';
  else if (!latest || latest.year <= cashCalibration.calibrationLastTargetYear || !validDay(latest.reportDate) || latest.reportDate < cashCalibration.calibrationCutoff
    || evidence.annual.some(row => !validDay(row.reportDate) || row.reportDate < row.end || row.reportDate > latest.reportDate!)
    || evidence.annual.some((row, index) => index > 0 && days(evidence.annual[index - 1].start, row.end) < 1)
    || history?.withheld.some(row => row.period === 5 && evidence.annual.some(annual => annual.year === row.year))) reason = 'The five-year publication dates or consecutive nonoverlapping source periods do not satisfy the calibration timing rules. These ranges are explicit percentage sensitivities.';
  const model = projection === 'trend' ? 'linear' : 'naive';
  const factors = reason === null && rangeMode === 'historical' && classification.group
    ? Array.from({ length: cashCalibration.latestEmpiricalHorizon }, (_, index) => historicalCashFactor(model, index + 1, classification.group!)) : [];
  if (factors.some(factor => factor === null)) reason = 'A supported group or global historical factor is unavailable for this model and horizon. These ranges are explicit percentage sensitivities.';
  const empirical = rangeMode === 'historical' && reason === null && factors.length === 4;
  if (reason && (rangeMode === 'historical' || manualCashRequired)) issues.push(reason);
  if (classification.scale !== null && classification.scale > 0 && classification.scale < 1e-6) issues.push('The historical cash scale is very small. Normalized errors and automatic cash ranges need source-unit and cash-definition review.');
  if (evidence.annual.length && new Set(evidence.annual.map(row => Math.sign(row.cashFlow!))).size > 1) issues.push('Historical cash changes sign. The retained losses and rebounds are part of the source history; the starter does not infer a recovery probability.');
  const maximumEmpiricalYear = cashCalibration.latestEmpiricalHorizon;
  const tailBase = empirical ? factors.at(-1)!.factor * classification.scale! : null;
  const forecast: StarterEvidence['forecast'] = Array.from({ length: evidence.years }, (_, index) => {
    const year = index + 1, spreadPercent = settings.spreadPercent + settings.spreadStepPercent * index;
    const rawMid = manualCashRequired ? null : projection === 'latest' ? latest?.cashFlow ?? null : evidence.forecast[index].mid;
    const mid = validAmount(rawMid) ? rawMid : null;
    const fitted = empirical && year <= maximumEmpiricalYear ? factors[index] : null;
    const halfWidth = mid === null ? null : fitted ? fitted.factor * classification.scale!
      : empirical ? tailBase! + classification.scale! * tailWideningPercent / 100 * (year - maximumEmpiricalYear)
      : mid === 0 ? null : Math.abs(mid) * spreadPercent / 100;
    const available = halfWidth !== null && validAmount(halfWidth) && validAmount(mid! - halfWidth) && validAmount(mid! + halfWidth);
    return { year, mid, low: available ? mid! - halfWidth! : null, high: available ? mid! + halfWidth! : null, spreadPercent,
      rangeBasis: !available ? 'unavailable' : fitted ? 'historical' : empirical ? 'assumed-tail' : 'percentage',
      halfWidth: available ? halfWidth : null, factor: available ? fitted?.factor ?? null : null,
      calibrationGroup: available ? fitted?.group ?? null : null, support: available ? fitted?.support ?? null : null };
  });
  if (forecast.some(row => row.mid === 0 && row.low === null)) issues.push('A percentage of zero cash provides no useful uncertainty span. Low/high cash inputs remain missing for those years until an absolute cash scenario is entered.');
  const midDescription = manualCashRequired ? 'Enter a reviewed path for cash distributable to common equity, after required capital and funding needs.'
    : projection === 'latest' ? 'Hold the latest eligible signed annual provider cash proxy flat as the provisional midline. This is a benchmark, not an optimal ten-year forecast.'
    : projection === 'trend' ? 'Fit the recency-weighted historical cash trend and project mid cash from its fitted latest-year intercept and slope.'
    : 'Hold the signed recency-weighted historical cash proxy flat through the forecast.';
  const rangeDescription = empirical
    ? `Years 1–${maximumEmpiricalYear} use the ${classification.group} cash-dispersion group's model-specific historical-error factors, falling back to the matching global pool only if support is insufficient. Half-width = factor × mean absolute five-year cash (${classification.scale} ${evidence.currency} m). The 80% research target is not a company guarantee or an equal-tail probability.`
    : manualCashRequired ? 'Automatic cash ranges are unavailable until the company-specific equity cash model is supplied.'
    : `Low/high use explicit percentage sensitivities: mid ± abs(mid) × [${settings.spreadPercent}% + ${settings.spreadStepPercent} percentage points × (year − 1)]. These are chosen assumptions with no empirical coverage claim. ${rangeMode === 'historical' && reason ? reason : ''}`;
  const tailDescription = empirical ? `Years ${maximumEmpiricalYear + 1}–${evidence.years} are assumed extensions, not calibrated ranges: carry the Year ${maximumEmpiricalYear} absolute half-width forward and add ${tailWideningPercent}% of the historical cash scale for each additional year.` : '';
  const probabilityDescription = 'Annual cash ranges do not establish an 80% range for the joint cash path or DCF. Cross-year dependence, required return and terminal value remain separate assumptions. Raw COVID, rebound and other exceptional years are retained; normalization requires a sourced deep-dive revision.';
  const explanation = [evidence.explanation[0], evidence.explanation[1], `${midDescription} ${rangeDescription} ${tailDescription} ${probabilityDescription}`,
    manualCashRequired ? 'Terminal and recovery cash remain unavailable until reviewed assumptions are entered.' : evidence.explanation[3], evidence.explanation[4]];
  draft.title = `${entry.display_name} — ${manualCashRequired ? 'equity cash inputs required' : projection === 'latest' ? 'latest cash starter' : projection === 'trend' ? 'cash trend starter' : 'weighted cash starter'}`;
  draft.starterOrigin = { id: 'empirical-cash-starter-v3', asOf: evidence.asOf, historyYears: settings.historyYears, weights: [...settings.weights], spreadPercent: settings.spreadPercent,
    spreadStepPercent: settings.spreadStepPercent, projection, rangeMode, tailWideningPercent, calibrationId };
  for (const key of scenarioKeys) {
    const final = forecast.at(-1)![key];
    draft.scenarios[key] = { ...draft.scenarios[key], cashFlows: forecast.map(row => row[key]), terminalEquity: final === null ? null : Math.max(0, final) / (evidence.requiredReturn / 100),
      rationale: `${key === 'mid' ? midDescription : `${key === 'low' ? 'Low subtracts' : 'High adds'} the stated annual cash half-width around the midline.`} ${rangeDescription} ${tailDescription} ${probabilityDescription} ${explanation[1]} ${explanation[3]} ${explanation[4]} Starter method empirical-cash-starter-v3; calibration ${calibrationId}; source basis ${evidence.asOf}.` };
  }
  const anchor = manualCashRequired ? null : projection === 'latest' ? latest?.cashFlow ?? null : evidence.anchor;
  const anchorBasis = manualCashRequired ? 'Historical provider cash proxies remain in the source history for context. Manual, reviewed distributable-equity cash inputs are required after regulatory capital and funding needs; no automatic forecast midline is supplied.'
    : projection === 'latest' ? `Latest eligible annual provider cash proxy (${latest?.year ?? 'unavailable'}) supplies the provisional midline: ${anchor ?? 'unavailable'} ${evidence.currency} m. The five-year dispersion classification uses signed native cash and its unweighted mean absolute scale; historical weights do not turn losses into missing values.` : evidence.anchorBasis;
  draft.notes.business = `${manualCashRequired ? 'Manual equity cash model required.' : 'Unreviewed editable cash starter.'} ${anchorBasis} ${explanation[1]}`;
  draft.notes.decision = `${explanation[2]} ${explanation[3]} ${explanation[4]} ${issues.join(' ')}`;
  Object.assign(evidence, { method: 'empirical-cash-starter-v3', projection, rangeMode, tailWideningPercent, manualCashRequired, forecast, anchor, anchorBasis, explanation,
    uncertainty: { status: manualCashRequired || !forecast.some(row => row.low !== null) ? 'unavailable' : empirical ? 'historical' : 'percentage',
      reason: manualCashRequired || rangeMode === 'historical' ? reason : null, ...classification, calibrationId: empirical ? calibrationId : null, targetCoverage: empirical ? cashCalibration.targetCoverage : null } });
  if (settings.terminalMethod !== 'final-forecast') {
    const seed = seedTerminalCash(manualCashRequired ? [] : evidence.annual.map(row => row.cashFlow));
    draft.starterOrigin.terminalMethod = 'historical-median-v1';
    evidence.terminal = { method: 'historical-median-v1', median: seed.median, historyCount: evidence.annual.length, spreadPercent: 20 };
    const description = `Terminal cash is independent of annual forecast endpoints and their uncertainty widths. The signed median of ${evidence.annual.length} eligible annual provider cash proxies (${seed.median ?? 'unavailable'} ${evidence.currency} m) seeds first post-horizon equity cash. Low/high subtract/add an assumed 20% of its absolute amount, with no assigned probability. This is an unreviewed historical starting point, not an estimate of sustainable distributable cash. At least three consecutive eligible periods are required; raw shock years are retained. Required equity return starts at 10%, mature cash growth at 0%. Sale = max(0, first post-horizon cash after reinvestment) / (required return minus mature growth). Required investment, financing and any transition from the final forecast year need company review. Nonpositive terminal cash supplies zero assumed sale, not a recovery appraisal.`;
    if (seed.median === null) issues.push(manualCashRequired ? 'Sustainable terminal cash requires the reviewed equity cash model.' : 'Fewer than three comparable cash observations are available. Enter sustainable terminal cash or an explicit final sale.');
    else if (seed.low === null || seed.high === null) issues.push('A historical terminal-cash sensitivity exceeds the supported amount range. Review source units and enter an explicit terminal assumption.');
    for (const key of scenarioKeys) {
      const s = draft.scenarios[key];
      s.terminalCash = { cashFlow: seed[key], growthRate: 0 };
      s.terminalEquity = resolveTerminalSale(s).value;
      s.rationale = s.rationale.replace(explanation[3], description);
    }
    draft.notes.decision = draft.notes.decision.replace(explanation[3], description);
    explanation[3] = description;
  }
  return built;
}
