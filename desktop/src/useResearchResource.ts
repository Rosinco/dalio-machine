import { useEffect, useState } from 'react';
import { resource } from './research';
import type { ResearchRelease } from './types';

export function useReleaseResource<T>(release: ResearchRelease | undefined, name: string, enabled = true) {
  const key = release ? `${release.id}:${release.storage}:${name}` : '';
  const [result, setResult] = useState<{ key: string; data: T | null; error: string }>();
  useEffect(() => {
    if (!release || !enabled || result?.key === key) return;
    let cancelled = false;
    const abort = new AbortController();
    resource<T>(release, name, abort.signal).then(data => {
      if (!cancelled) setResult({ key, data, error: '' });
    }).catch(e => { if (!cancelled) setResult({ key, data: null, error: `The saved research could not be opened: ${String(e)}` }); });
    return () => { cancelled = true; abort.abort(); };
  }, [key, enabled, result?.key]);
  return { data: result?.key === key ? result.data : null, error: result?.key === key ? result.error : '', ready: !!key && result?.key === key };
}
