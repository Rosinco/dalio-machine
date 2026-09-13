import { describe, expect, it } from 'vitest';
import { buildStarterValuation, defaultStarterSettings, defaultStarterWeights, settingsForStarterOrigin, type StarterSettings } from './starterValuations';
import { calculateValuation } from './valuation';
import { decodeValuation, loadValuationDraft, saveValuationDraft, type SavedValuation } from './savedValuations';
import type { FinancialCompany, SourcedReport } from './financialData';
import type { MarketReport } from './marketData';
import { cashCalibration, cashCalibrationId, historicalCashFactor } from './cashUncertainty';

const entry = { id: '200', display_name: 'Example', report_currency: 'EUR', stock_currency: 'EUR', source_as_of: '2026-08-10' };
function report(year: number, cash: number | null, currency = 'EUR', ratio = 2): SourcedReport {
  const values = { profit_to_equity_holders: 999999, free_cash_flow: cash, total_equity: 100, intangible_assets: 20, net_debt: 15 };
  return { year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, report_date: `${year + 1}-02-15`, currency, currency_ratio: ratio, source_id: 'annual-2026-08-10', source_as_of: '2026-08-10', values, raw: Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value === null ? null : value * ratio])) };
}
function market(r: SourcedReport, currency = 'EUR'): MarketReport {
  return { year: r.year, source_id: r.source_id, currency, shares: 10, price: 100, price_date: r.report_date, local: 1000, sek: 11000, fx_rate: 11, fx_date: r.report_date, fx_method: 'direct', fx_instruments: ['1'], flags: [] };
}
function history(cash = [10, 20, 30, 40, 50]): FinancialCompany {
  const annual = cash.map((value, i) => report(2026 - cash.length + i, value));
  return { id: entry.id, annual, quarterly: [], withheld: [], market: annual.map(r => market(r)) };
}
const settings = { weights: [30, 25, 20, 15, 10], spreadPercent: 20 };

function legacyStarter(e: typeof entry, h: FinancialCompany | null, f: Parameters<typeof buildStarterValuation>[2] = null, asOf?: string, options: Pick<StarterSettings, 'weights' | 'spreadPercent'> = settings) {
  return buildStarterValuation(e, h, f, asOf, settingsForStarterOrigin({ id: 'weighted-cash-starter-v1', asOf: asOf ?? '2026-08-10', ...options }));
}

describe('weighted historical cash starter valuations', () => {
  it('preserves the complete existing v1 draft including provenance and descriptive strings', async () => {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(legacyStarter(entry, history()).draft)));
    expect(Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')).toBe('49a19e3c5e866c259e48e4e0a88786c15a1ee5737c588bef7db20e78be320969');
  });
  it('uses five signed cash-proxy years with explicit recent-year weights and flat ±20% scenarios', () => {
    const built = legacyStarter(entry, history()), { draft, evidence } = built;
    expect(evidence.annual.map(r => r.year)).toEqual([2025, 2024, 2023, 2022, 2021]);
    expect(evidence.anchor).toBe(35);
    expect(evidence.includedWeightTotal).toBe(100);
    expect(evidence.annual.map(r => r.effectiveWeightPercent)).toEqual([30, 25, 20, 15, 10]);
    expect(draft.scenarios.low.cashFlows).toEqual(Array(10).fill(28));
    expect(draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(35));
    expect(draft.scenarios.high.cashFlows).toEqual(Array(10).fill(42));
    expect(draft.scenarios.mid.terminalEquity).toBe(350);
    expect(calculateValuation(draft).scenarios.mid.value).toBeCloseTo(350);
    expect(draft.starterOrigin).toEqual({ id: 'weighted-cash-starter-v1', asOf: '2026-08-10', ...settings });
    expect(draft.researchOrigin).toBeUndefined();
    expect(draft.notes.decision).toMatch(/not a confidence interval/);
  });
  it('never converts accounting earnings into cash or inserts net debt, surplus cash or recovery assumptions', () => {
    const h = history(); h.annual[4].values.profit_to_equity_holders = 1e10;
    const { draft } = legacyStarter(entry, h);
    expect(draft.scenarios.mid.cashFlows[0]).toBe(35);
    expect(draft.capital).toEqual({ tangibleEquity: 80, averageTCE: null, nopat: null, grossDebt: null, surplusCash: null });
    for (const s of Object.values(draft.scenarios)) { expect(s.recoveryEquity).toBeNull(); expect(s.recoveryYear).toBeNull(); }
    expect(draft.notes.financing).toMatch(/minority/);
    expect(draft.notes.business).toMatch(/proxy/);
  });
  it('keeps losses signed, applies spread by magnitude and assumes no positive terminal sale for nonpositive payments', () => {
    const { draft, evidence } = legacyStarter(entry, history([-10, -20, -30, -40, -50]));
    expect(evidence.anchor).toBe(-35);
    expect(draft.scenarios.low.cashFlows[0]).toBe(-42);
    expect(draft.scenarios.mid.cashFlows[0]).toBe(-35);
    expect(draft.scenarios.high.cashFlows[0]).toBe(-28);
    expect(draft.scenarios.high.terminalEquity).toBe(0);
    expect(draft.scenarios.low.rationale).toMatch(/not establish a personal obligation/);
    expect(calculateValuation(draft).scenarios.low.value!).toBeLessThan(calculateValuation(draft).scenarios.high.value!);
    const zero = legacyStarter(entry, history([0, 0, 0, 0, 0]));
    expect(zero.draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(0));
    expect(zero.draft.scenarios.mid.terminalEquity).toBe(0);
    expect(calculateValuation(zero.draft).ready).toBe(true);
  });
  it('explicitly normalizes partial contiguous history without skipping missing or nonconsecutive years', () => {
    const h = history(); h.annual = h.annual.slice(2);
    let built = legacyStarter(entry, h);
    expect(built.evidence.includedWeightTotal).toBe(75);
    expect(built.evidence.anchor).toBeCloseTo((50 * 30 + 40 * 25 + 30 * 20) / 75);
    expect(built.issues.join(' ')).toMatch(/75.*normaliz/i);
    h.annual = [report(2019, 999999), report(2025, 50)];
    built = legacyStarter(entry, h);
    expect(built.evidence.annual.map(r => r.year)).toEqual([2025]);
    expect(built.evidence.anchor).toBe(50);
    expect(built.issues.join(' ')).toMatch(/gap/);
    h.annual = [report(2023, 999999), report(2024, null), report(2025, 50)];
    expect(legacyStarter(entry, h).evidence.anchor).toBe(50);
  });
  it('leaves forecasts missing for latest null, missing conversion, short, stale or newer withheld annual periods', () => {
    const cases = [
      (h: FinancialCompany) => { h.annual[4].values.free_cash_flow = null; h.annual[4].raw.free_cash_flow = null; },
      (h: FinancialCompany) => { h.annual[4].currency_ratio = null; h.annual[4].values.free_cash_flow = null; },
      (h: FinancialCompany) => { h.annual[4].start = '2025-09-01'; },
      (h: FinancialCompany) => { for (const key of Object.keys(h.annual[4].raw)) { h.annual[4].raw[key] = 0; h.annual[4].values[key] = 0; } },
      (h: FinancialCompany) => { h.annual.pop(); },
      (h: FinancialCompany) => { h.withheld.push({ year: 2026, period: 5, source_id: h.annual[4].source_id, start: '2025-06-01', end: '2026-05-31', published: '2026-07-01', reason: 'Unavailable source row' }); },
    ];
    for (const change of cases) {
      const h = history(); change(h); const built = legacyStarter(entry, h);
      expect(built.evidence.anchor).toBeNull();
      expect(built.draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(null));
      expect(built.draft.scenarios.mid.terminalEquity).toBeNull();
      expect(calculateValuation(built.draft).ready).toBe(false);
    }
  });
  it('uses source-proven quote-currency amounts without inventing current FX', () => {
    const h = history(); h.market.forEach(row => { row.currency = 'USD'; row.sek = null; row.fx_rate = null; row.fx_date = null; row.fx_method = null; row.flags = ['missing_fx']; });
    const built = legacyStarter(entry, h);
    expect(built.draft.currency).toBe('USD');
    expect(built.evidence.amountBasis).toBe('quote');
    expect(built.evidence.anchor).toBe(70);
    expect(built.draft.capital.tangibleEquity).toBe(160);
    expect(built.evidence.annual[0]).toMatchObject({ rawCashFlow: 100, reportedCashFlow: 50, cashFlow: 100, reportingCurrency: 'EUR', quotedCurrency: 'USD', currencyRatio: 2 });
    expect(built.draft.marketCap).toBe(1000);
    expect(built.draft.priceSource).toMatch(/no additional FX conversion/i);
    const broken = structuredClone(h); broken.market[3].source_id = 'wrong';
    expect(legacyStarter(entry, broken).evidence.annual).toHaveLength(1);
  });
  it('uses observed SEK price conversion only when compatible, and pairs price with its own annual source clock', () => {
    const h = history(); h.annual.forEach(row => { row.currency = 'SEK'; }); h.market.forEach(row => { row.currency = 'USD'; });
    const built = legacyStarter({ ...entry, report_currency: 'SEK' }, h);
    expect(built.draft.currency).toBe('SEK');
    expect(built.evidence.amountBasis).toBe('reporting');
    expect(built.draft.marketCap).toBe(11000);
    expect(built.evidence.price).toMatchObject({ year: 2025, sourceId: 'annual-2026-08-10', reportEnd: '2025-12-31', reportDate: '2026-02-15', priceDate: '2026-02-15', fxRate: 11, method: 'sek' });
    const unmatched = history(); unmatched.market.forEach(row => { row.source_id = 'wrong'; });
    expect(legacyStarter(entry, unmatched).draft.marketCap).toBeNull();
  });
  it('does not bypass the latest annual price or ownership flags with an older usable valuation', () => {
    for (const flag of ['share_basis', 'scale_suspect', 'missing_price']) {
      const h = history(), latest = h.market[4];
      latest.flags = [flag]; latest.local = null; latest.sek = null;
      if (flag === 'missing_price') { latest.price = null; latest.price_date = null; }
      expect(h.market[3].local).toBe(1000);
      const built = legacyStarter(entry, h);
      expect(built.evidence.anchor).toBe(35);
      expect(built.draft.marketCap).toBeNull();
      expect(built.evidence.price).toBeNull();
      expect(calculateValuation(built.draft).ready).toBe(false);
    }
  });
  it('validates editable settings before replacing a draft and returns fresh inputs', () => {
    const changed = { weights: [100, 0, 0, 0, 0], spreadPercent: 40 };
    const built = legacyStarter(entry, history(), null, undefined, changed);
    expect(built.evidence.anchor).toBe(50);
    expect(built.draft.scenarios.low.cashFlows[0]).toBe(30);
    expect(built.draft.scenarios.high.cashFlows[0]).toBe(70);
    built.draft.starterOrigin!.weights[0] = 0;
    expect(changed.weights[0]).toBe(100);
    for (const invalid of [{ weights: [30, 25, 20], spreadPercent: 20 }, { weights: [0, 0, 0, 0, 0], spreadPercent: 20 }, { weights: [30, 25, 20, 15, 10], spreadPercent: 101 }]) expect(() => legacyStarter(entry, history(), null, undefined, invalid)).toThrow(/weights|spread/i);
    expect(() => legacyStarter(entry, { ...history(), id: '999' })).toThrow(/company/);
  });
  it('retains edited starter settings and cleared values through autosave/import and rejects mixed or malformed provenance', () => {
    const data = new Map<string, string>(), storage = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
    const draft = legacyStarter(entry, history()).draft;
    draft.scenarios.mid.cashFlows[0] = null; draft.notes.macro = 'My macro note';
    const saved: SavedValuation = { format: 'macro-atlas-valuation', version: 1, id: 'starter-1', created: '2026-09-12T12:00:00.000Z', company: entry.id, release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), draft };
    saveValuationDraft(storage, saved);
    expect(loadValuationDraft(storage, saved)?.draft).toEqual(draft);
    expect(decodeValuation(JSON.stringify(saved)).draft.starterOrigin).toEqual(draft.starterOrigin);
    const mixed = structuredClone(saved); mixed.draft.researchOrigin = { id: 'reviewed', asOf: '2026-09-12' };
    expect(() => decodeValuation(JSON.stringify(mixed))).toThrow();
    const malformed = structuredClone(saved); malformed.draft.starterOrigin!.weights = [100];
    expect(() => decodeValuation(JSON.stringify(malformed))).toThrow();
  });
});

const operatingEntry = { ...entry, sector_id: '5', branch_id: '10' };
const boundFinancial: NonNullable<Parameters<typeof buildStarterValuation>[2]> = {
  id: cashCalibration.provenance.sourcePack.sha256, taxonomy_sha256: cashCalibration.taxonomySha256, as_of: '2026-08-10',
  sources: cashCalibration.sourceFiles.map(source => ({ ...source, frequency: 'annual', bytes: 1, rows: 1, outside_directory: 0, usable: 1, withheld: 0 })),
};
function boundHistory(cash = [10, 20, 30, 40, 50]) {
  const result = history(cash);
  result.annual.forEach(row => { row.source_id = '2026-08-10-annual'; });
  result.market.forEach(row => { row.source_id = '2026-08-10-annual'; });
  return result;
}

describe('empirical cash starter v3', () => {
  it('defaults to latest signed annual cash and model-specific historical ranges', () => {
    const { draft, evidence } = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial);
    expect(draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(50));
    expect(draft.starterOrigin).toMatchObject({ id: 'empirical-cash-starter-v3', projection: 'latest', rangeMode: 'historical', calibrationId: cashCalibrationId });
    expect(evidence.uncertainty).toMatchObject({ status: 'historical', scale: 30, group: 'medium', targetCoverage: .8 });
    expect(evidence.uncertainty.dispersion).toBeCloseTo(Math.sqrt(200) / 30);
    const factor = historicalCashFactor('naive', 1, 'medium')!;
    expect(evidence.forecast[0].low).toBeCloseTo(50 - 30 * factor.factor);
    expect(evidence.forecast[0].high).toBeCloseTo(50 + 30 * factor.factor);
    expect(evidence.forecast[0].rangeBasis).toBe('historical');
  });
  it('marks years five through ten as assumptions extended from year-four absolute width', () => {
    const { evidence } = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial);
    const fourth = evidence.forecast[3].halfWidth!;
    expect(evidence.forecast[4]).toMatchObject({ rangeBasis: 'assumed-tail', factor: null, support: null });
    expect(evidence.forecast[4].halfWidth).toBeCloseTo(fourth + 3);
    expect(evidence.forecast[9].halfWidth).toBeCloseTo(fourth + 18);
    const changed = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, tailWideningPercent: 20 });
    expect(changed.evidence.forecast[9].halfWidth).toBeCloseTo(fourth + 36);
    expect(changed.evidence.explanation.join(' ')).toMatch(/not.*DCF|DCF.*not/i);
  });
  it('keeps nonzero historical ranges when the weighted trend crosses zero', () => {
    const { evidence, draft } = buildStarterValuation(operatingEntry, boundHistory([50, 40, 30, 20, 10]), boundFinancial, undefined, { ...defaultStarterSettings, projection: 'trend' });
    expect(evidence.forecast[0].mid).toBe(0);
    expect(evidence.forecast[0].low!).toBeLessThan(0);
    expect(evidence.forecast[0].high!).toBeGreaterThan(0);
    expect(evidence.forecast[0].halfWidth).toBeCloseTo(30 * historicalCashFactor('linear', 1, 'medium')!.factor);
    expect(draft.scenarios.mid.terminalEquity).toBe(300);
    expect(draft.scenarios.mid.terminalCash).toEqual({ cashFlow: 30, growthRate: 0 });
  });
  it('keeps terminal cash independent of annual uncertainty and preserves the old v3 definition on replay', () => {
    const current = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial);
    const wider = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, tailWideningPercent: 99 });
    expect(current.draft.scenarios.high.terminalCash).toEqual(wider.draft.scenarios.high.terminalCash);
    expect(current.draft.scenarios.high.cashFlows[9]).not.toBe(wider.draft.scenarios.high.cashFlows[9]);
    const old = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, terminalMethod: 'final-forecast' });
    expect(old.draft.starterOrigin).not.toHaveProperty('terminalMethod');
    expect(old.draft.scenarios.high).not.toHaveProperty('terminalCash');
    expect(old.draft.scenarios.high.terminalEquity).toBeCloseTo(Math.max(0, old.draft.scenarios.high.cashFlows[9]!) / .1);
    expect(buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, old.draft.valuationDate, settingsForStarterOrigin(old.draft.starterOrigin)).draft).toEqual(old.draft);
    expect(current.evidence.forecast).toEqual(old.evidence.forecast);
  });
  it('uses explicit percentage fallback for untested settings, partial histories and unbound data', () => {
    for (const setting of [{ historyYears: 10 as const, weights: defaultStarterWeights(10) }, { weights: [100, 0, 0, 0, 0] }, { projection: 'flat' as const }, { calibrationId: 'unknown-calibration' }]) {
      const built = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, ...setting });
      expect(built.evidence.uncertainty.status).toBe('percentage');
      expect(built.evidence.forecast[0].rangeBasis).toBe('percentage');
      expect(built.evidence.uncertainty.targetCoverage).toBeNull();
      expect(built.evidence.uncertainty.reason).toBeTruthy();
    }
    for (const [e, h, f] of [[operatingEntry, boundHistory([50]), boundFinancial], [entry, boundHistory(), boundFinancial], [operatingEntry, boundHistory(), null], [operatingEntry, boundHistory(), { ...boundFinancial, taxonomy_sha256: 'wrong' }]] as const) {
      expect(buildStarterValuation(e, h, f).evidence.uncertainty.status).toBe('percentage');
    }
  });
  it('preserves quote-currency source lineage with a labelled sensitivity fallback', () => {
    const h = boundHistory();
    h.market.forEach(row => { row.currency = 'USD'; row.sek = null; row.flags = ['missing_fx']; });
    const built = buildStarterValuation(operatingEntry, h, boundFinancial);
    expect(built.evidence.amountBasis).toBe('quote');
    expect(built.draft.scenarios.mid.cashFlows[0]).toBe(100);
    expect(built.evidence.uncertainty.status).toBe('percentage');
    expect(built.evidence.uncertainty.reason).toMatch(/native|reporting currency/i);
  });
  it('requires manual cash for financials but retains property cash and historical evidence', () => {
    const financial = { ...operatingEntry, sector_id: '1', branch_id: '70' };
    const built = buildStarterValuation(financial, boundHistory(), boundFinancial);
    expect(built.evidence.manualCashRequired).toBe(true);
    expect(built.evidence.annual).toHaveLength(5);
    expect(built.evidence.annual[0].reportedCashFlow).toBe(50);
    expect(built.evidence.anchor).toBeNull();
    expect(built.evidence.anchorBasis).toMatch(/historical.*context|context.*historical/i);
    expect(built.evidence.anchorBasis).toMatch(/manual|reviewed/i);
    expect(built.evidence.anchorBasis).not.toMatch(/supplies the provisional midline/i);
    expect(built.draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(null));
    expect(built.draft.scenarios.mid.terminalEquity).toBeNull();
    expect(built.evidence.uncertainty.status).toBe('unavailable');
    for (const branch_id of ['75', '76']) expect(buildStarterValuation({ ...financial, branch_id }, boundHistory(), boundFinancial).evidence.manualCashRequired).toBe(false);
    expect(v2Starter(financial, boundHistory(), boundFinancial).draft.scenarios.mid.cashFlows[0]).toBe(60);
  });
  it('does not mistake all-zero scale or a zero percentage mid for certain future cash', () => {
    const zero = buildStarterValuation(operatingEntry, boundHistory([0, 0, 0, 0, 0]), boundFinancial);
    expect(zero.draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(0));
    expect(zero.draft.scenarios.low.cashFlows).toEqual(Array(10).fill(null));
    expect(zero.draft.scenarios.high.cashFlows).toEqual(Array(10).fill(null));
    const crossing = buildStarterValuation(operatingEntry, boundHistory([50, 40, 30, 20, 10]), boundFinancial, undefined, { ...defaultStarterSettings, projection: 'trend', rangeMode: 'percentage' });
    expect(crossing.evidence.forecast[0].mid).toBe(0);
    expect(crossing.evidence.forecast[0].low).toBeNull();
  });
  it('gates empirical ranges on valid publications, no overlap and exact source lineage', () => {
    for (const change of [(h: FinancialCompany) => { h.annual[0].report_date = null; }, (h: FinancialCompany) => { h.annual[1].start = h.annual[0].end; }, (h: FinancialCompany) => { h.annual[0].source_id = 'unknown'; }]) {
      const h = boundHistory(); change(h);
      expect(buildStarterValuation(operatingEntry, h, boundFinancial).evidence.uncertainty.status).toBe('percentage');
    }
  });
  it('round-trips v3 settings and ignores annual source values outside the valuation date', () => {
    const configured = { ...defaultStarterSettings, projection: 'trend' as const, tailWideningPercent: 25 };
    const original = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, '2026-08-10', configured);
    expect(buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, original.draft.starterOrigin!.asOf, settingsForStarterOrigin(original.draft.starterOrigin)).draft).toEqual(original.draft);
    const poisoned = boundHistory();
    poisoned.annual.push({ ...report(2027, -999999), source_id: '2026-08-10-annual' });
    expect(buildStarterValuation(operatingEntry, poisoned, boundFinancial, '2026-08-10', configured).draft).toEqual(original.draft);
    const wrongPack = buildStarterValuation(operatingEntry, boundHistory(), { ...boundFinancial, id: 'b'.repeat(64) });
    expect(wrongPack.evidence.uncertainty.status).toBe('percentage');
    expect(wrongPack.evidence.uncertainty.reason).toMatch(/financial pack/i);
  });
  it('rejects calibration identifiers that the saved-draft format cannot preserve', () => {
    for (const calibrationId of ['', 'a'.repeat(151), 'bad identifier', 'bad_identifier', 'bad/id', 'bad\nid']) {
      expect(() => buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, calibrationId })).toThrow(/calibration identifier/i);
    }
    const supportedFormat = buildStarterValuation(operatingEntry, boundHistory(), boundFinancial, undefined, { ...defaultStarterSettings, calibrationId: 'a'.repeat(150) });
    expect(supportedFormat.draft.starterOrigin?.id).toBe('empirical-cash-starter-v3');
    expect(supportedFormat.evidence.uncertainty.status).toBe('percentage');
  });
  it('does not admit unknown sector identifiers to the empirical population', () => {
    for (const sector_id of ['0', '11', '01', 'Financials', '5 ']) {
      const built = buildStarterValuation({ ...operatingEntry, sector_id }, boundHistory(), boundFinancial);
      expect(built.evidence.uncertainty.status).toBe('percentage');
      expect(built.evidence.uncertainty.targetCoverage).toBeNull();
      expect(built.evidence.uncertainty.reason).toMatch(/classification/i);
    }
  });
});

const v2Settings: StarterSettings = { historyYears: 5, weights: defaultStarterWeights(5), spreadPercent: 10, spreadStepPercent: 10, projection: 'trend' };
function v2Starter(e: typeof entry, h: FinancialCompany | null, f: Parameters<typeof buildStarterValuation>[2] = null, asOf?: string, options: StarterSettings = v2Settings) {
  return buildStarterValuation(e, h, f, asOf, { ...options, method: options.method ?? 'weighted-cash-starter-v2' });
}

describe('weighted cash trend and widening scenarios', () => {
  it('preserves the complete existing v2 draft including provenance and descriptive strings', async () => {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(v2Starter(entry, history()).draft)));
    expect(Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')).toBe('f9158efbec2f7bf5f9c12366a2e7d81c883386d1f72d033daece672f7a511321');
  });
  it('projects the fitted latest-year intercept and widens the annual spread', () => {
    const { draft, evidence } = v2Starter(entry, history());
    expect(evidence.anchor).toBe(35);
    expect(evidence.trend).toEqual({ slope: 10, intercept: 50, weightedMeanYear: -1.5, usableTrendYears: 5 });
    expect(evidence.forecast.slice(0, 3)).toEqual([
      { year: 1, mid: 60, low: 54, high: 66, spreadPercent: 10 },
      { year: 2, mid: 70, low: 56, high: 84, spreadPercent: 20 },
      { year: 3, mid: 80, low: 56, high: 104, spreadPercent: 30 },
    ]);
    expect(draft.scenarios.mid.cashFlows).toEqual([60, 70, 80, 90, 100, 110, 120, 130, 140, 150]);
    expect(draft.scenarios.mid.terminalEquity).toBe(1500);
    expect(draft.scenarios.low.terminalEquity).toBe(0);
    expect(calculateValuation(draft).scenarios.mid.value).toBeCloseTo(Array.from({ length: 10 }, (_, i) => (60 + 10 * i) / 1.1 ** (i + 1)).reduce((sum, value) => sum + value, 0) + 1500 / 1.1 ** 10);
    expect(draft.starterOrigin).toEqual({ id: 'weighted-cash-starter-v2', asOf: '2026-08-10', ...v2Settings });
  });
  it('fits weighted covariance for uneven observations rather than projecting from the latest actual value', () => {
    const { evidence } = v2Starter(entry, history([0, 2, 1, 5, 4]));
    // Independent moments: E[x] = -1.5, E[y] = 2.95,
    // E[x²] = 4, E[xy] = -2.55, covariance = 1.875, variance = 1.75.
    expect(evidence.anchor).toBeCloseTo(2.95);
    expect(evidence.trend.slope).toBeCloseTo(15 / 14);
    expect(evidence.trend.intercept).toBeCloseTo(319 / 70);
    expect(evidence.forecast[0].mid).toBeCloseTo(197 / 35);
    expect(evidence.forecast[0].mid).not.toBeCloseTo(4 + 15 / 14);
  });
  it('preserves signed losses and ordering when the trend crosses zero', () => {
    const { evidence, draft } = v2Starter(entry, history([50, 40, 30, 20, 10]));
    expect(evidence.trend.slope).toBe(-10);
    expect(evidence.forecast[0]).toEqual({ year: 1, mid: 0, low: 0, high: 0, spreadPercent: 10 });
    expect(evidence.forecast[1]).toEqual({ year: 2, mid: -10, low: -12, high: -8, spreadPercent: 20 });
    expect(evidence.forecast[9]).toEqual({ year: 10, mid: -90, low: -180, high: 0, spreadPercent: 100 });
    expect(draft.scenarios.mid.terminalEquity).toBe(0);
    for (const row of evidence.forecast) { expect(row.low!).toBeLessThanOrEqual(row.mid!); expect(row.mid!).toBeLessThanOrEqual(row.high!); }
  });
  it('allows widening beyond 100% without clamping a zero crossing', () => {
    const { evidence } = v2Starter(entry, history([10, 10, 10, 10, 10]), null, undefined, { ...v2Settings, spreadStepPercent: 20 });
    expect(evidence.forecast[6]).toEqual({ year: 7, mid: 10, low: -3, high: 23, spreadPercent: 130 });
    expect(evidence.forecast[9]).toEqual({ year: 10, mid: 10, low: -9, high: 29, spreadPercent: 190 });
  });
  it('uses a disclosed zero slope with only one positive-weight observation', () => {
    const { evidence, issues } = v2Starter(entry, history(), null, undefined, { ...v2Settings, weights: [0, 0, 100, 0, 0] });
    expect(evidence.trend).toEqual({ slope: 0, intercept: 30, weightedMeanYear: -2, usableTrendYears: 1 });
    expect(evidence.forecast[0].mid).toBe(30);
    expect(evidence.forecast[9].mid).toBe(30);
    expect(issues.join(' ')).toMatch(/one positive-weight|one positively weighted/i);
    const missing = v2Starter(entry, { ...history(), annual: [report(2025, 50)] }, null, undefined, { ...v2Settings, weights: [0, 0, 100, 0, 0] });
    expect(missing.evidence.trend.slope).toBeNull();
    expect(missing.evidence.forecast.every(row => row.mid === null && row.low === null && row.high === null)).toBe(true);
  });
  it('supports ten consecutive annual periods with separate default weights', () => {
    const tenWeights = defaultStarterWeights(10);
    expect(tenWeights).toEqual([19, 17, 15, 13, 11, 9, 7, 5, 3, 1]);
    const { evidence } = v2Starter(entry, history([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]), null, undefined, { ...v2Settings, historyYears: 10, weights: tenWeights });
    expect(evidence.annual.map(row => row.year)).toEqual([2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]);
    expect(evidence.trend.slope).toBeCloseTo(2);
    expect(evidence.trend.intercept).toBeCloseTo(20);
    expect(evidence.forecast[0].mid).toBeCloseTo(22);
    expect(evidence.forecast[9].mid).toBeCloseTo(40);
    tenWeights[0] = 0;
    expect(defaultStarterWeights(10)[0]).toBe(19);
    expect(defaultStarterWeights(5)).toEqual([30, 25, 20, 15, 10]);
  });
  it('retains flat projection as an explicit option with independently widening bands', () => {
    const { draft, evidence } = v2Starter(entry, history(), null, undefined, { ...v2Settings, projection: 'flat' });
    expect(draft.scenarios.mid.cashFlows).toEqual(Array(10).fill(35));
    expect(evidence.forecast[0].low).toBe(31.5);
    expect(evidence.forecast[9].low).toBe(0);
    expect(evidence.projection).toBe('flat');
  });
  it('preserves both origin versions through decode and reconstructs their original baseline settings', () => {
    for (const draft of [legacyStarter(entry, history()).draft, v2Starter(entry, history()).draft]) {
      const saved: SavedValuation = { format: 'macro-atlas-valuation', version: 1, id: 'starter-origin-test', created: '2026-09-12T12:00:00.000Z', company: entry.id, release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), draft };
      expect(decodeValuation(JSON.stringify(saved)).draft).toEqual(draft);
      expect(v2Starter(entry, history(), null, draft.starterOrigin!.asOf, settingsForStarterOrigin(draft.starterOrigin)).draft).toEqual(draft);
    }
    const legacySettings = settingsForStarterOrigin({ id: 'weighted-cash-starter-v1', asOf: '2026-08-10', weights: [30, 25, 20, 15, 10], spreadPercent: 20 });
    expect(legacySettings).toMatchObject({ method: 'weighted-cash-starter-v1', historyYears: 5, projection: 'flat', spreadPercent: 20, spreadStepPercent: 0 });
    expect(settingsForStarterOrigin()).toEqual(defaultStarterSettings);
    const draft = v2Starter(entry, history()).draft;
    const saved: SavedValuation = { format: 'macro-atlas-valuation', version: 1, id: 'malformed-v2', created: '2026-09-12T12:00:00.000Z', company: entry.id, release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), draft };
    const malformed = structuredClone(saved);
    if (malformed.draft.starterOrigin?.id === 'weighted-cash-starter-v2') malformed.draft.starterOrigin.spreadStepPercent = -10;
    expect(() => decodeValuation(JSON.stringify(malformed))).toThrow();
  });
});
