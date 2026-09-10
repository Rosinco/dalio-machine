import { useEffect, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { decodeFinancialCompany, decodeFinancialRows, validateFinancialIndex, type FinancialCompany, type FinancialIndex } from './financialData';
import { projectAnnual, type BranchData } from './branchComparison';
import { decodeMarketRows } from './marketData';

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
export function useBranchAnnual(index: FinancialIndex | null, ids: string[]) {
  const [state, setState] = useState<Resource<BranchData> & { loaded: number }>({ key: '', data: null, ready: false, error: '', loaded: 0 });
  const key = index ? `${index.id}:${ids.join(',')}` : '';
  useEffect(() => {
    let active = true;
    setState({ key, data: null, ready: !key, error: '', loaded: 0 });
    if (!index) return;
    const run = async () => {
      const data: BranchData = {};
      const available = ids.filter(id => index.companies[id]?.annual.count);
      const total = available.reduce((n, id) => n + index.companies[id].annual.count, 0);
      if (total > 100000) throw new Error('This branch exceeds the 100,000 annual-report limit for an interactive comparison.');
      for (let offset = 0; offset < available.length; offset += 32) {
        if (!active) return;
        const batch = available.slice(offset, offset + 32);
        const raw: any = isTauri() ? await invoke('financial_annual', { pack: index.id, ids: batch }) : await read('financial_annual', { pack: index.id, ids: batch.join(',') });
        if (!active) return;
        if (raw?.pack !== index.id || !Array.isArray(raw.companies) || raw.companies.length !== batch.length || raw.companies.some((c: any, i: number) => c?.id !== batch[i])) throw new Error('Annual histories do not match the requested comparison.');
        for (const c of raw.companies) {
          const reports = decodeFinancialRows(c.annual, index, c.id, 'annual');
          data[c.id] = projectAnnual(reports, decodeMarketRows(c.market, index, c.id, reports));
        }
        setState({ key, data: null, ready: false, error: '', loaded: Math.min(offset + 32, available.length) });
      }
      if (active) setState({ key, data, ready: true, error: '', loaded: available.length });
    };
    run().catch(e => { if (active) setState({ key, data: null, ready: true, error: String(e instanceof Error ? e.message : e), loaded: 0 }); });
    return () => { active = false; };
    // Pack and ordered IDs fully identify this request; cancelled branches never publish partial medians.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state.key === key ? state : { key, data: null, ready: !key, error: '', loaded: 0 };
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
