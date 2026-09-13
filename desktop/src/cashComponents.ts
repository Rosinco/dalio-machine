import { validAmount } from './valuation';

export type CashComponents = { operating: number | null; investing: number | null; financing: number | null; netCash: number | null; providerFcf: number | null };
export function reconcileCashComponents(c: CashComponents) {
  const bounded = (v: number) => validAmount(v) ? v : null;
  const sum = (...values: (number | null)[]) => values.every(validAmount) ? bounded(values.reduce((a, b) => a + b, 0)) : null;
  const difference = (a: number | null, b: number | null) => validAmount(a) && validAmount(b) ? bounded(a - b) : null;
  const operatingPlusInvesting = sum(c.operating, c.investing), componentSum = sum(c.operating, c.investing, c.financing);
  const providerDifference = difference(c.providerFcf, operatingPlusInvesting), netDifference = difference(c.netCash, componentSum);
  const comparison = (difference: number | null, a: number | null, b: number | null) => difference === null ? 'unavailable' : Math.abs(difference) <= 1e-6 + 1e-8 * Math.max(Math.abs(a!), Math.abs(b!)) ? 'matches' : 'differs';
  return { operatingPlusInvesting, providerDifference, componentSum, netDifference,
    providerComparison: comparison(providerDifference, c.providerFcf, operatingPlusInvesting),
    netComparison: comparison(netDifference, c.netCash, componentSum) };
}
