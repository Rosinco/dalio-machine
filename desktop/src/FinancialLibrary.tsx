import { useRef, useState } from 'react';
import { isTauri } from '@tauri-apps/api/core';
import { ArrowDownToLine, FileUp } from 'lucide-react';
import type { FinancialIndex } from './financialData';
import { exportFinancialFile, importFinancialFile } from './financialService';

export type FinancialLibraryProps = { financial: FinancialIndex | null; financialReady: boolean; financialError: string; onFinancialImported: () => void };
export default function FinancialLibrary({ financial, financialReady, financialError, onFinancialImported }: FinancialLibraryProps) {
  const [busy, setBusy] = useState(false), [progress, setProgress] = useState<number | null>(null), [message, setMessage] = useState(''), [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const run = async (action: () => Promise<void>) => { setBusy(true); setError(''); setMessage(''); try { await action(); } catch (e) { setError(String(e instanceof Error ? e.message : e)); } finally { setBusy(false); setProgress(null); } };
  const n = (v: number) => v.toLocaleString('en-US');
  return <section className="financial-pack-library" aria-label="Company financial history library" data-library-financial-pack={financial?.id ?? ''}>
    <h3>Company financial history</h3>
    {!financialReady ? <p role="status">Checking saved financial coverage…</p> : financialError ? <p role="alert">{financialError}</p> : financial ? <><p><strong>{n(financial.summary.with_reports)} listings with reports</strong> · {n(financial.summary.listings - financial.summary.with_reports)} without reports<br />{n(financial.summary.annual)} annual · {n(financial.summary.quarterly)} quarterly reports<br />{n(financial.summary.withheld)} source rows withheld · downloads through {financial.as_of}<br />{(financial.bytes / 1000000).toFixed(1)} MB saved locally</p><p className="hash">SHA-256 {financial.id}</p></> : <p>No financial history pack matches this release’s company directory. Included research profiles, if any, remain available.</p>}
    <p>Financial histories are stored separately from the research release. Save both files when moving this research to another computer. Atlas attaches histories only to their matching company directory.</p>
    <div className="library-actions"><button className="secondary" disabled={busy || !financial} onClick={() => void run(async () => setMessage(await exportFinancialFile(financial!)))}><ArrowDownToLine size={15} />Save financial history pack</button><button className="secondary" disabled={busy || !isTauri()} onClick={() => { if (input.current) input.current.value = ''; input.current?.click(); }}><FileUp size={15} />Import financial history pack</button><input className="visually-hidden" ref={input} type="file" accept=".sqlite" aria-label="Financial history file" onChange={e => { const file = e.target.files?.[0]; if (file) void run(async () => { setProgress(0); const pack = await importFinancialFile(file, setProgress); onFinancialImported(); setMessage(`Financial history saved · ${n(pack.summary.with_reports)} listings. It will appear with the matching research release.`); }); }} /></div>
    {!isTauri() && <p className="chart-caption">Financial pack import is available in the desktop application.</p>}
    {progress !== null && <><progress value={progress} max={100} aria-label="Financial import progress" /><p role="status">{progress < 100 ? `Copying financial history… ${progress}%` : 'Validating every company record…'}</p></>}
    {busy && progress === null && <p role="status">Saving financial history…</p>}{error && <p className="validation-error" role="alert">{error}</p>}{message && <p className="library-message" role="status">{message}</p>}
  </section>;
}
