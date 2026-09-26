import type { Category, Mode } from './types';
import type { Observatory } from './business';

export type CompanyView = 'financials' | 'peers' | 'context' | 'research' | 'valuation' | 'screen' | 'lists';
export type SectorView = 'browse' | 'overview' | 'companies' | 'compare' | 'context' | 'research' | 'screen' | 'lists';
export interface NavigationRoute {
  observatory: Observatory; companyView: CompanyView; sectorView: SectorView;
  companyId: string; branchId: string; code: string; unknownName: string;
  mode: Mode; tab: string; category: Category; metric: string; compare: string;
  year: number; startYear: number; valuationTab: 'scenarios' | 'evidence'; anchor: string;
}
export type ScrollPositions = Record<string, [number, number]>;
export interface NavigationEntry { route: NavigationRoute; scroll: ScrollPositions }
export interface NavigationHistory { past: NavigationEntry[]; current: NavigationEntry; future: NavigationEntry[]; sequence: number }
export const createNavigation = (route: NavigationRoute): NavigationHistory => ({ past: [], current: { route, scroll: {} }, future: [], sequence: 0 });
export function visitNavigation(state: NavigationHistory, patch: Partial<NavigationRoute>, scroll: ScrollPositions): NavigationHistory {
  const route = { ...state.current.route, anchor: '', ...patch };
  if (Object.keys(route).every(key => route[key as keyof NavigationRoute] === state.current.route[key as keyof NavigationRoute])) {
    return patch.anchor ? { ...state, current: { route, scroll: {} }, sequence: state.sequence + 1 } : state;
  }
  return { past: [...state.past, { ...state.current, scroll }].slice(-80), current: { route, scroll: {} }, future: [], sequence: state.sequence + 1 };
}
export function moveNavigation(state: NavigationHistory, direction: -1 | 1, scroll: ScrollPositions): NavigationHistory {
  const source = direction === -1 ? state.past : state.future;
  const next = direction === -1 ? source.at(-1) : source[0];
  if (!next) return state;
  const current = { ...state.current, scroll };
  return direction === -1
    ? { past: state.past.slice(0, -1), current: next, future: [current, ...state.future].slice(0, 80), sequence: state.sequence + 1 }
    : { past: [...state.past, current].slice(-80), current: next, future: state.future.slice(1), sequence: state.sequence + 1 };
}
export function navigationScope(route: NavigationRoute): string {
  return route.observatory === 'companies' ? `company:${['lists', 'screen'].includes(route.companyView) ? 'all' : route.companyId}:${route.companyView}`
    : route.observatory === 'sectors' ? `branch:${route.branchId}:${route.sectorView}` : `macro:${route.code}:${route.mode}:${route.tab}`;
}
