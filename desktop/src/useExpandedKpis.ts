import { useEffect, useMemo, useState } from 'react';
import type { FinancialIndex } from './financialData';
import type { CompanyListColumn } from './companyListModel';
import { decodeExpandedArtifact, EXPANDED_KPI_MANIFEST, expandedShard, expandedVariant, validateExpandedBinding, validateExpandedIndex, validateExpandedManifest, validateExpandedShard } from './expandedKpis';
import type { ExpandedArtifact, ExpandedKpiContext, ExpandedLoadedVariant } from './expandedKpis';
import { readResearchGaugeBytes } from './researchGauge';

const manifest = EXPANDED_KPI_MANIFEST;
const cache = new Map<string, { value: unknown; bytes: number }>();
const cacheLimit = 64 * 1024 * 1024;
function remember(descriptor: ExpandedArtifact, value: unknown) {
  cache.delete(descriptor.sha256);
  cache.set(descriptor.sha256, { value, bytes: descriptor.uncompressedBytes });
  let bytes = [...cache.values()].reduce((sum, item) => sum + item.bytes, 0);
  for (const [key, item] of cache) { if (bytes <= cacheLimit) break; cache.delete(key); bytes -= item.bytes; }
}
async function artifact(descriptor: ExpandedArtifact, signal: AbortSignal): Promise<unknown> {
  const saved = cache.get(descriptor.sha256);
  if (saved) { remember(descriptor, saved.value); return saved.value; }
  const response = await fetch(`/${descriptor.path}`, { signal });
  if (!response.ok) throw new Error('The selected bundled KPI values could not be opened.');
  const bytes = await readResearchGaugeBytes(response.body, descriptor.bytes);
  const value = await decodeExpandedArtifact(bytes, descriptor);
  if (!signal.aborted) remember(descriptor, value);
  return value;
}
const empty = (ids: string[] = []): ExpandedKpiContext => ({ index: null, variants: new Map(), loading: new Set(ids), errors: new Map(), error: '', ready: ids.length === 0 });

/** Only selected columns and conditions request shards. The frozen starter artifact is independent. */
export function useExpandedKpis(financial: FinancialIndex | null, taxonomy: string | null | undefined, columns: CompanyListColumn[], enabled: boolean): ExpandedKpiContext {
  const selection = [...new Set(columns.map(column => expandedVariant(column)?.id).filter((id): id is string => !!id))].sort().join('|');
  const ids = useMemo(() => selection ? selection.split('|') : [], [selection]);
  const key = `${enabled}:${financial?.id}:${taxonomy}:${selection}`;
  const [state, setState] = useState<{ key: string; financial: FinancialIndex | null; context: ExpandedKpiContext }>({ key: '', financial: null, context: empty() });
  useEffect(() => {
    if (!enabled || !ids.length) { setState({ key, financial, context: empty() }); return; }
    let active = true; const controller = new AbortController();
    setState({ key, financial, context: empty(ids) });
    const load = async () => {
      validateExpandedManifest(manifest);
      validateExpandedBinding(manifest, financial, taxonomy);
      const index = validateExpandedIndex(await artifact(manifest.index, controller.signal), manifest, financial!);
      const variants = new Map<string, ExpandedLoadedVariant>(), errors = new Map<string, string>(), loading = new Set(ids);
      const publish = () => { if (active) setState({ key, financial, context: { index, variants: new Map(variants), errors: new Map(errors), loading: new Set(loading), error: '', ready: loading.size === 0 } }); };
      publish();
      const selected = new Set(ids);
      const descriptors = [...new Map(columns.map(expandedVariant).filter(v => v && selected.has(v.id)).map(v => { const shard = expandedShard(v!)!; return [shard.sha256, shard] as const; })).values()];
      let cursor = 0;
      await Promise.all(Array.from({ length: Math.min(4, descriptors.length) }, async () => {
        while (active && cursor < descriptors.length) {
          const descriptor = descriptors[cursor++];
          const affected = descriptor.variantIds!.filter(id => selected.has(id));
          try {
            const decoded = validateExpandedShard(await artifact(descriptor, controller.signal), descriptor, manifest);
            for (const item of decoded) if (selected.has(item.variant.id)) variants.set(item.variant.id, item);
          } catch (error) {
            for (const id of affected) errors.set(id, error instanceof Error ? error.message : String(error));
          }
          for (const id of affected) loading.delete(id);
          publish();
        }
      }));
    };
    load().catch(error => { if (active) setState({ key, financial, context: { ...empty(), ready: true, error: error instanceof Error ? error.message : String(error) } }); });
    return () => { active = false; controller.abort(); };
    // Selection is the canonical identity of requested columns, independent of UI array instances.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, financial, taxonomy, enabled, ids]);
  return useMemo(() => state.key === key && state.financial === financial ? state.context : empty(enabled ? ids : []), [state, key, financial, enabled, ids]);
}
