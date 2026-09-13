import { describe, expect, it } from 'vitest';
import type { CompanyListColumn } from './companyListModel';
import { companyListCell, companyListUnit } from './companyListModel';
import { companyListCsv, matchesNumericCondition } from './companyListFeatures';
import { EXPANDED_KPI_MANIFEST, expandedKpiCell } from './expandedKpis';
import type { ExpandedKpiContext, ExpandedKpiVariant } from './expandedKpis';
import type { ResearchGaugeRow } from './researchGaugeModel';

// These assertions encode source-audited meanings, not the metadata-classification implementation.
// Values are deliberately synthetic: raw arithmetic and saved-source identities are verified by the export audit.
const row = { id: '65260', name: 'Special-unit regression fixture', country: 'JP', sourceAsOf: '2026-08-10' } as ResearchGaugeRow;
const column = (variant: ExpandedKpiVariant): CompanyListColumn => ({ id: variant.id, kpiId: variant.metricId, window: `provider:${variant.source}:${variant.calcGroup}`, calculation: `provider:${variant.calculation}` });
function selected(kpiId: number, calculations: string[]) {
  const variants = EXPANDED_KPI_MANIFEST.variants.filter(variant => variant.metricId === `provider_${kpiId}` && calculations.includes(variant.calculation));
  for (const calculation of calculations) expect(variants.some(variant => variant.calculation === calculation), `Source choice ${kpiId}/${calculation} exists`).toBe(true);
  return variants;
}
function context(variant: ExpandedKpiVariant, value: number, quoteCurrency: string | null = 'EUR', reportCurrency: string | null = 'JPY'): ExpandedKpiContext {
  return { index: { ids: [row.id], reportCurrencies: [reportCurrency], quoteCurrencies: [quoteCurrency], positions: new Map([[row.id, 0]]) }, variants: new Map([[variant.id, { variant, values: [value] }]]), errors: new Map(), loading: new Set(), error: '', ready: true };
}

describe('source-audited technical KPI units', () => {
  it('treats high/low observations as quote prices, and pricehigh/pricelow/default as percentages in every saved period', () => {
    const prices = selected(153, ['high', 'low']), percentages = selected(153, ['pricehigh', 'pricelow', 'default']);
    expect(prices.length).toBeGreaterThan(2); expect(percentages.length).toBeGreaterThan(3);
    for (const variant of prices) {
      expect(variant).toMatchObject({ unit: 'price', currencyBasis: 'quote' });
      expect(companyListUnit(column(variant))).toBe('price');
      const cell = expandedKpiCell(row, column(variant), context(variant, 123.45));
      expect(cell).toMatchObject({ value: 123.45, unit: 'price', currency: 'EUR', display: '123.45 EUR', date: null, status: 'available' });
      expect(matchesNumericCondition(cell, { operator: 'gte', value: 100, currency: 'JPY' })).toBe(false);
      expect(matchesNumericCondition(cell, { operator: 'gte', value: 100, currency: 'EUR' })).toBe(true);
    }
    for (const variant of percentages) {
      expect(variant).toMatchObject({ unit: 'percent', currencyBasis: 'none' });
      expect(companyListUnit(column(variant))).toBe('percent');
      const cell = expandedKpiCell(row, column(variant), context(variant, -12.5));
      expect(cell).toMatchObject({ value: -12.5, unit: 'percent', currency: null, display: '-12.5%', date: null, status: 'available' });
      expect(matchesNumericCondition(cell, { operator: 'lt', value: 0 })).toBe(true);
    }
  });

  it('renders the moving-average mean and all four Bollinger outputs in quote currency, including band width', () => {
    const movingAverage = selected(157, ['mean']);
    expect(movingAverage.map(variant => variant.calcGroup)).toEqual(['150days']);
    const bands = selected(161, ['mean', 'over', 'under', 'diff']);
    expect(bands).toHaveLength(4);
    for (const variant of [...movingAverage, ...bands]) {
      expect(variant).toMatchObject({ unit: 'price', currencyBasis: 'quote' });
      expect(companyListCell(row, column(variant), context(variant, 25.75))).toMatchObject({ value: 25.75, display: '25.75 EUR', unit: 'price', currency: 'EUR' });
    }
  });

  it('distinguishes monetary turnover in quote-currency millions from mean traded shares', () => {
    const monetary = selected(313, ['mill']), shares = selected(313, ['mean']);
    expect(monetary.length).toBeGreaterThan(1); expect(shares).toHaveLength(monetary.length);
    for (const variant of monetary) {
      expect(variant).toMatchObject({ unit: 'millions', currencyBasis: 'quote' });
      expect(companyListUnit(column(variant))).toBe('money');
      const cell = companyListCell(row, column(variant), context(variant, 12.345));
      expect(cell).toMatchObject({ value: 12.345, unit: 'money', currency: 'EUR', display: '12.35 EUR m' });
      expect(cell.display).not.toContain('shares');
      expect(matchesNumericCondition(cell, { operator: 'between', value: 10, valueTo: 15 })).toBe(false);
      expect(matchesNumericCondition(cell, { operator: 'between', value: 10, valueTo: 15, currency: 'EUR' })).toBe(true);
    }
    for (const variant of shares) {
      expect(variant).toMatchObject({ unit: 'count', currencyBasis: 'none' });
      expect(companyListUnit(column(variant))).toBe('count');
      expect(companyListCell(row, column(variant), context(variant, 1234.25))).toMatchObject({ value: 1234.25, unit: 'count', currency: null, display: '1,234.25' });
    }
  });

  it('withholds quote-denominated prices and turnover when quote currency is absent, without borrowing reporting currency', () => {
    const variants = [...selected(153, ['high', 'low']), ...selected(157, ['mean']), ...selected(161, ['mean', 'over', 'under', 'diff']), ...selected(313, ['mill'])];
    for (const variant of variants) {
      const cell = expandedKpiCell(row, column(variant), context(variant, 123.45, null, 'JPY'));
      expect(cell).toMatchObject({ value: null, currency: null, display: '—', status: 'missing' });
      expect(cell.detail).toMatch(/currency.*not established/i);
    }
    for (const variant of [...selected(153, ['pricehigh', 'pricelow', 'default']), ...selected(313, ['mean'])]) expect(expandedKpiCell(row, column(variant), context(variant, 0, null, null))).toMatchObject({ value: 0, currency: null, status: 'available' });
  });

  it('retains percentage units for short-capital measures and the audited spread calculations', () => {
    for (const variant of [...selected(146, ['SumCapital', 'AvgCapital']), ...selected(321, ['diff']), ...selected(322, ['diff'])]) {
      expect(variant).toMatchObject({ unit: 'percent', currencyBasis: 'none' });
      expect(companyListCell(row, column(variant), context(variant, 2.25))).toMatchObject({ value: 2.25, unit: 'percent', currency: null, display: '2.25%' });
    }
  });
});

describe('unverified monetary provider fields', () => {
  const withheld = () => [...selected(110, ['ValueBuy', 'ValueSell', 'ValueNet']), ...selected(144, ['Amount']), ...selected(146, ['SumValue', 'AvgValue'])];

  it('publishes no available observations for all 29 variants with unestablished currency or scale', () => {
    const variants = withheld(); expect(variants).toHaveLength(29);
    for (const variant of variants) {
      expect(variant, variant.id).toMatchObject({ currencyBasis: 'unverified', availableCount: 0 });
      expect(variant.notes.length).toBeGreaterThan(0);
    }
    for (const variant of selected(110, ['ValueBuy', 'ValueSell', 'ValueNet'])) expect(variant.unit).toBe('number');
    for (const variant of selected(144, ['Amount'])) expect(variant.unit).toBe('number');
    for (const variant of selected(146, ['SumValue', 'AvgValue'])) expect(variant.unit).toBe('millions');
  });

  it('blocks finite synthetic numbers, including zero and negatives, independently of the nominal unit', () => {
    for (const variant of withheld()) for (const value of [0, -900001.2345, 900001.2345]) {
      const cell = companyListCell(row, column(variant), context(variant, value));
      expect(cell, `${variant.id}: ${value}`).toMatchObject({ value: null, display: '—', currency: null, status: 'missing' });
      expect(cell.detail).toMatch(/no established currency or scale/i);
      expect(matchesNumericCondition(cell, { operator: 'present', value: null })).toBe(false);
      for (const currency of [undefined, 'EUR', 'JPY']) expect(matchesNumericCondition(cell, { operator: 'gte', value: -1_000_000, currency })).toBe(false);
    }
  });

  it('retains loading and source-error status before classifying an unverified field as unavailable', () => {
    const variant = withheld()[0], selection = column(variant), loaded = context(variant, 900001.2345);
    expect(companyListCell(row, selection)).toMatchObject({ value: null, status: 'loading' });
    expect(companyListCell(row, selection, { ...loaded, index: null })).toMatchObject({ value: null, status: 'loading' });
    const bindingError = companyListCell(row, selection, { ...loaded, error: 'A different source binding was selected.' });
    expect(bindingError).toMatchObject({ value: null, status: 'error' });
    expect(bindingError.detail).toContain('different source binding');
    const failedShard = companyListCell(row, selection, { ...loaded, errors: new Map([[variant.id, 'Checksum failed for this shard.']]) });
    expect(failedShard).toMatchObject({ value: null, status: 'error' });
    expect(failedShard.detail).toContain('Checksum failed');
    for (const cell of [bindingError, failedShard]) expect(matchesNumericCondition(cell, { operator: 'missing', value: null })).toBe(false);
  });

  it('does not accept a caller-supplied currency override or export a withheld synthetic value', () => {
    for (const variant of withheld()) {
      const candidate: ExpandedKpiContext = { ...context(variant, 900001.2345), variants: new Map([[variant.id, { variant: { ...variant, currencyBasis: 'quote', unit: 'price' }, values: [900001.2345] }]]) };
      const cell = companyListCell(row, column(variant), candidate);
      expect(cell).toMatchObject({ value: null, currency: null, status: 'missing' });
      const csv = companyListCsv([row], [column(variant)], () => cell, () => 'Withheld monetary field', () => 'Provider snapshot 2026-08-10');
      expect(csv).not.toContain('900001.2345'); expect(csv).toContain('"missing"');
    }
  });
});
