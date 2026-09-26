import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, type Dispatch, type SetStateAction } from 'react';
import { createNavigation, moveNavigation, visitNavigation, type NavigationHistory, type NavigationRoute, type ScrollPositions } from './navigation';

export function createScreenStore() {
  return { values: new Map<string, unknown>(), listeners: new Set<() => void>() };
}
export const NavigationContext = createContext<{ scope: string; store: ReturnType<typeof createScreenStore> } | null>(null);

export function ScreenPresentation() {
  const context = useContext(NavigationContext);
  useLayoutEffect(() => {
    if (!context) return;
    const { scope, store } = context;
    const seen = new WeakSet<HTMLDetailsElement>();
    const name = (element: HTMLDetailsElement) => `${scope}:disclosure:${element.id || `${element.className}:${element.querySelector('summary')?.textContent}`}`;
    const restore = () => document.querySelectorAll<HTMLDetailsElement>('.workspace details').forEach(element => {
      if (seen.has(element)) return;
      seen.add(element);
      const saved = store.values.get(name(element));
      if (typeof saved === 'boolean') element.open = saved;
    });
    const remember = (event: Event) => { if (event.target instanceof HTMLDetailsElement && event.target.closest('.workspace')) store.values.set(name(event.target), event.target.open); };
    restore();
    const observer = new MutationObserver(restore);
    observer.observe(document.querySelector('.app') ?? document.body, { childList: true, subtree: true });
    document.addEventListener('toggle', remember, true);
    return () => { observer.disconnect(); document.removeEventListener('toggle', remember, true); };
  }, [context?.scope, context?.store]);
  return null;
}

/** Presentation choices stay in this app session. This never saves or rolls back financial inputs. */
export function useScreenState<T>(key: string, initial: T | (() => T)): [T, Dispatch<SetStateAction<T>>] {
  const context = useContext(NavigationContext);
  const local = useRef<ReturnType<typeof createScreenStore> | null>(null);
  if (!local.current) local.current = createScreenStore();
  const store = context?.store ?? local.current;
  const name = `${context?.scope ?? 'local'}:${key}`;
  if (!store.values.has(name)) store.values.set(name, typeof initial === 'function' ? (initial as () => T)() : initial);
  const value = useSyncExternalStore(listener => { store.listeners.add(listener); return () => { store.listeners.delete(listener); }; }, () => store.values.get(name) as T, () => store.values.get(name) as T);
  const setValue: Dispatch<SetStateAction<T>> = next => {
    const before = store.values.get(name) as T;
    const after = typeof next === 'function' ? (next as (previous: T) => T)(before) : next;
    if (!Object.is(before, after)) { store.values.set(name, after); store.listeners.forEach(listener => listener()); }
  };
  return [value, setValue];
}

const scrolling = ['.business-content', '.valuation-workspace', '.business-stage-content', '.sidebar-content', '.company-list-workspace', '.company-list-table-scroll', '.branch-explorer', '.research-gauge-workspace'];
export function captureScroll(): ScrollPositions {
  const positions: ScrollPositions = { window: [window.scrollX, window.scrollY] };
  scrolling.forEach(selector => { const element = document.querySelector(selector); if (element) positions[selector] = [element.scrollLeft, element.scrollTop]; });
  return positions;
}

export function useAtlasNavigation(initial: NavigationRoute, onRoute: (route: NavigationRoute) => void) {
  const [history, render] = useState(() => createNavigation(initial));
  const current = useRef(history);
  const commit = (next: NavigationHistory) => { if (next === current.current) return; current.current = next; onRoute(next.current.route); render(next); };
  const visit = (patch: Partial<NavigationRoute>) => commit(visitNavigation(current.current, patch, captureScroll()));
  const replace = (patch: Partial<NavigationRoute>) => {
    const state = current.current;
    commit({ ...state, current: { ...state.current, route: { ...state.current.route, ...patch } } });
  };
  const move = (direction: -1 | 1) => commit(moveNavigation(current.current, direction, captureScroll()));
  const reset = () => commit({ ...createNavigation(current.current.current.route), sequence: current.current.sequence + 1 });
  useEffect(() => {
    const state = current.current;
    if (!state.sequence) return;
    let stopped = false, frame = 0;
    const positions = state.current.scroll, anchor = state.current.route.anchor;
    const finish = () => { stopped = true; cancelAnimationFrame(frame); observer.disconnect(); window.removeEventListener('wheel', finish); window.removeEventListener('touchstart', finish); window.removeEventListener('pointerdown', finish); window.removeEventListener('keydown', finish); clearTimeout(timeout); };
    const apply = () => {
      if (stopped) return;
      const target = anchor ? document.getElementById(anchor) : null;
      if (anchor && !Object.keys(positions).length && target) { target.scrollIntoView({ block: 'start' }); return; }
      scrolling.forEach(selector => { const element = document.querySelector(selector); const [left, top] = positions[selector] ?? [0, 0]; element?.scrollTo({ left, top, behavior: 'instant' }); });
      const [left, top] = positions.window ?? [0, 0]; window.scrollTo({ left, top, behavior: 'instant' });
    };
    const observer = new MutationObserver(() => { cancelAnimationFrame(frame); frame = requestAnimationFrame(apply); });
    observer.observe(document.querySelector('.app') ?? document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'open', 'data-business-ready'] });
    const timeout = setTimeout(finish, 3000);
    window.addEventListener('wheel', finish, { passive: true }); window.addEventListener('touchstart', finish, { passive: true }); window.addEventListener('pointerdown', finish); window.addEventListener('keydown', finish);
    frame = requestAnimationFrame(apply);
    return finish;
  }, [history.sequence]);
  return { history, route: history.current.route, visit, replace, move, reset };
}
