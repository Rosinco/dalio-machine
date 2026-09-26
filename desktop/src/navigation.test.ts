import { describe, expect, it } from 'vitest';
import { createNavigation, moveNavigation, visitNavigation, type NavigationRoute } from './navigation';

const route: NavigationRoute = { observatory: 'companies', companyView: 'lists', sectorView: 'browse', companyId: '102', branchId: '21', code: 'SE', unknownName: '', mode: 'fundamentals', tab: 'overview', category: 'production', metric: 'gov_debt_pct_gdp', compare: '', year: 2025, startYear: 1990, valuationTab: 'scenarios', anchor: '' };
describe('analysis navigation history', () => {
  it('returns through exact company, section and source-screen positions without fabricating entries', () => {
    let state = createNavigation(route);
    expect(moveNavigation(state, -1, {}).past).toHaveLength(0);
    state = visitNavigation(state, { companyView: 'peers' }, { window: [0, 320] });
    state = visitNavigation(state, { companyId: '110', companyView: 'financials' }, { '.business-content': [0, 700] });
    state = moveNavigation(state, -1, {});
    expect(state.current.route).toMatchObject({ companyId: '102', companyView: 'peers' });
    expect(state.current.scroll['.business-content']).toEqual([0, 700]);
    state = moveNavigation(state, -1, {});
    expect(state.current.route.companyView).toBe('lists');
    expect(state.current.scroll.window).toEqual([0, 320]);
    expect(moveNavigation(state, 1, {}).current.route.companyView).toBe('peers');
  });
  it('does not duplicate a destination and discards forward history after a different navigation', () => {
    const initial = createNavigation(route);
    expect(visitNavigation(initial, {}, {})).toBe(initial);
    let state = visitNavigation(initial, { companyView: 'financials' }, {});
    state = visitNavigation(state, { companyView: 'valuation' }, {});
    state = moveNavigation(state, -1, {});
    expect(state.future).toHaveLength(1);
    state = visitNavigation(state, { companyView: 'research' }, {});
    expect(state.future).toHaveLength(0);
    expect(state.past.at(-1)?.route.companyView).toBe('financials');
  });
  it('retains branch and macro context and bounds session history', () => {
    let state = createNavigation({ ...route, observatory: 'sectors', sectorView: 'compare', branchId: '42' });
    state = visitNavigation(state, { observatory: 'companies', companyId: '110', companyView: 'context' }, {});
    state = visitNavigation(state, { observatory: 'macro', mode: 'history', code: 'FI', year: 2024 }, {});
    state = moveNavigation(state, -1, {});
    expect(state.current.route).toMatchObject({ observatory: 'companies', companyId: '110', companyView: 'context' });
    state = moveNavigation(state, -1, {});
    expect(state.current.route).toMatchObject({ observatory: 'sectors', sectorView: 'compare', branchId: '42' });
    for (let i = 0; i < 100; i++) state = visitNavigation(state, { companyId: String(i + 1) }, {});
    expect(state.past).toHaveLength(80);
  });
  it('revisits the same chart anchor without adding a duplicate Back entry', () => {
    const first = visitNavigation(createNavigation(route), { anchor: 'company-financial-statements' }, { window: [0, 100] });
    const repeated = visitNavigation(first, { anchor: 'company-financial-statements' }, { window: [0, 900] });
    expect(repeated.past).toEqual(first.past);
    expect(repeated.current.scroll).toEqual({});
    expect(repeated.sequence).toBe(first.sequence + 1);
  });
});
