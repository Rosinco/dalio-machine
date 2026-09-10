import { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { decodeFinancialCompany, validateFinancialIndex, type FinancialCompany, type FinancialIndex } from './financialData';

async function read(command: string, args: Record<string, string>) {
  if (isTauri()) return invoke(command, args);
  const response = await fetch(`/api/financials/${command.replace('financial_', '')}?${new URLSearchParams(args)}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
type Resource<T> = { key: string; data: T | null; ready: boolean; error: string };
function useFinancialResource<T>(key: string, request: () => Promise<T | null>) {
  const [state, setState] = useState<Resource<T>>({ key: '', data: null, ready: true, error: '' });
  useEffect(() => {
    let active = true;
    if (!key) { setState({ key, data: null, ready: true, error: '' }); return; }
    setState({ key, data: null, ready: false, error: '' });
    request().then(data => { if (active) setState({ key, data, ready: true, error: '' }); }).catch(e => { if (active) setState({ key, data: null, ready: true, error: String(e instanceof Error ? e.message : e) }); });
    return () => { active = false; };
    // The explicit key binds every response to its requested directory and company.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state.key === key ? state : { key, data: null, ready: !key, error: '' };
}
export function useFinancialIndex(taxonomy: string | null | undefined, enabled: boolean, revision: number) {
  return useFinancialResource<FinancialIndex>(enabled && taxonomy ? `${taxonomy}:${revision}` : '', async () => {
    const raw = await read('financial_index', { taxonomy: taxonomy! });
    if (raw === null) return null;
    validateFinancialIndex(raw, taxonomy!); return raw;
  });
}
export function useFinancialCompany(index: FinancialIndex | null, id: string | undefined) {
  return useFinancialResource<FinancialCompany>(index && id && index.companies[id] ? `${index.id}:${id}` : '', async () => decodeFinancialCompany(await read('financial_company', { pack: index!.id, id: id! }), index!, id!));
}
export async function importFinancialFile(file: File, progress: (percent: number) => void): Promise<FinancialIndex> {
  if (!isTauri()) throw new Error('Import financial histories in the desktop application.');
  if (!file.size || file.size > 512 * 1024 * 1024) throw new Error('Financial packs must be between 1 byte and 512 MiB.');
  const token = await invoke<string>('financial_begin', { bytes: file.size });
  try {
    for (let offset = 0; offset < file.size; offset += 512 * 1024) {
      const data = new Uint8Array(await file.slice(offset, offset + 512 * 1024).arrayBuffer());
      await invoke('financial_append', { token, offset, data: Array.from(data) });
      progress(Math.round(100 * (offset + data.length) / file.size));
    }
    const raw = await invoke<FinancialIndex>('financial_finish', { token });
    validateFinancialIndex(raw, raw.taxonomy_sha256); return raw;
  } catch (e) { await invoke('financial_cancel', { token }).catch(() => {}); throw e; }
}
export async function exportFinancialFile(index: FinancialIndex) {
  if (isTauri()) return `Saved ${await invoke<string>('financial_export', { pack: index.id })}`;
  const a = document.createElement('a'); a.href = `/api/financials/export?pack=${index.id}`; a.download = ''; a.click();
  return 'Financial pack download started.';
}
