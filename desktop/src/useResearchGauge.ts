import { useEffect, useState } from 'react';
import type { FinancialIndex } from './financialData';
import type { ResearchGaugeArtifact } from './researchGaugeModel';
import pin from './data/research-gauge-manifest.json';
import { decodeResearchGauge, readResearchGaugeBytes, validateResearchGaugeArtifact, validateResearchGaugeBinding, validateResearchGaugeManifest } from './researchGauge';

type State = { key: string; data: ResearchGaugeArtifact | null; ready: boolean; error: string };
let cache: { key: string; financial: FinancialIndex; data: ResearchGaugeArtifact } | undefined;

export function useResearchGauge(financial: FinancialIndex | null, taxonomy: string | null | undefined, enabled: boolean) {
  const key = enabled ? `${financial?.id ?? 'none'}:${taxonomy ?? 'none'}` : '';
  const [state, setState] = useState<State>({ key: '', data: null, ready: true, error: '' });
  useEffect(() => {
    if (!enabled) { setState({ key, data: null, ready: true, error: '' }); return; }
    let active = true; const controller = new AbortController();
    setState({ key, data: null, ready: false, error: '' });
    const load = async () => {
      if (!financial) throw new Error('A matching financial history pack is needed to open this research screen.');
      const manifest = validateResearchGaugeManifest(pin);
      validateResearchGaugeBinding(manifest, financial, taxonomy);
      if (cache?.key === key) {
        // A newly loaded index is checked again, even when it claims the same ID.
        // Returning to a view with the same already validated index needs no full
        // repeat of every observation's numeric and calendar checks.
        if (cache.financial !== financial) {
          validateResearchGaugeArtifact(cache.data, manifest, financial, taxonomy);
          cache.financial = financial;
        }
        return cache.data;
      }
      const response = await fetch(`/${manifest.artifact.path}`, { signal: controller.signal });
      if (!response.ok) throw new Error('The bundled research screen could not be opened.');
      const bytes = await readResearchGaugeBytes(response.body, manifest.artifact.bytes);
      const data = await decodeResearchGauge(bytes, manifest, financial, taxonomy);
      if (active) cache = { key, financial, data };
      return data;
    };
    load().then(data => { if (active) setState({ key, data, ready: true, error: '' }); }).catch(error => {
      if (active) setState({ key, data: null, ready: true, error: error instanceof Error ? error.message : String(error) });
    });
    return () => { active = false; controller.abort(); };
  }, [key, enabled, financial, taxonomy]);
  return state.key === key ? state : { key, data: null, ready: !key, error: '' };
}
