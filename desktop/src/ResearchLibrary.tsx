import { useRef, useState } from 'react';
import { ArrowDownToLine, ArrowRight, Check, FileUp, X } from 'lucide-react';
import type { ResearchRelease } from './types';
import { exportResearch, importResearch, inspectResearch, MAX_PACKAGE_BYTES } from './research';

export default function ResearchLibrary({ releases, active, unreadable, onUse, onImported, onClose }: {
  releases: ResearchRelease[]; active: ResearchRelease; unreadable: number;
  onUse: (release: ResearchRelease) => Promise<void>; onImported: (release: ResearchRelease) => Promise<void>; onClose: () => void;
}) {
  const [candidate, setCandidate] = useState<{ text: string; release: ResearchRelease; filename: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const choose = async (file?: File) => {
    setCandidate(null); setError(''); setMessage(''); if (!file) return;
    setBusy(true);
    try {
      if (file.size > MAX_PACKAGE_BYTES) throw new Error('Research files must be smaller than 32 MB.');
      const text = await file.text(); const release = await inspectResearch(text);
      setCandidate({ text, release, filename: file.name });
    } catch (e) { setError(String(e instanceof Error ? e.message : e)); }
    finally { setBusy(false); }
  };
  const run = async (action: () => Promise<void>) => {
    setBusy(true); setError(''); setMessage('');
    try { await action(); } catch (e) { setError(String(e instanceof Error ? e.message : e)); }
    finally { setBusy(false); }
  };
  return <div className="modal-backdrop" onClick={onClose}>
    <div className="modal research-library" role="dialog" aria-modal="true" aria-labelledby="library-title" onClick={e => e.stopPropagation()}>
      <button className="modal-close" aria-label="Close data library" onClick={onClose}><X size={20} /></button>
      <div className="eyebrow">YOUR LOCAL RESEARCH LIBRARY</div><h2 id="library-title">Saved evidence.<br />Ready to explore.</h2>
      <p>Import a Dalio research file to update the map, scores and histories. Earlier releases stay available. Everything here works offline.</p>
      <div className="library-actions">
        <button className="primary" disabled={busy} onClick={() => { if (input.current) input.current.value = ''; input.current?.click(); }}><FileUp size={16} />Import research file</button>
        <input ref={input} type="file" accept=".json,.atlas.json" aria-label="Research file" className="visually-hidden" onChange={e => void choose(e.target.files?.[0])} />
        <button className="secondary" disabled={busy} onClick={() => void run(async () => setMessage(await exportResearch(active)))}><ArrowDownToLine size={15} />Save a copy of active release</button>
      </div>
      {error && <p className="validation-error" role="alert">{error}</p>}
      {message && <p className="library-message" role="status">{message}</p>}
      {busy && <p className="section-note" role="status">Opening local research…</p>}
      {unreadable > 0 && <p className="validation-error">{unreadable} saved file(s) could not be read. The other releases remain available.</p>}
      {candidate && <section className="import-preview" aria-label="Import preview">
        <div className="eyebrow">FILE CHECKED</div><h3>Research from {candidate.release.as_of}</h3>
        <p>{candidate.filename}</p><ReleaseDescription release={candidate.release} />
        <p className="chart-caption">File checksums and the supported data format passed validation. These checks confirm file integrity; they do not authenticate the publisher.</p>
        <button className="primary" disabled={busy} onClick={() => void run(async () => {
          const release = await importResearch(candidate.text); await onImported(release); setCandidate(null); setMessage('Research saved and opened.');
        })}>Import and use <ArrowRight size={15} /></button>
      </section>}
      <div className="section-title"><h3>Available releases</h3><span className="micro">{releases.length} saved</span></div>
      <div className="release-grid">{releases.map(release => <article key={release.id} className={`release-card ${active.id === release.id ? 'active' : ''}`} data-release-id={release.id} data-storage={release.storage}>
        <div className="release-card-heading"><strong>{release.as_of}</strong><span>{release.storage === 'included' ? 'Included' : 'Imported'}</span></div>
        <ReleaseDescription release={release} />
        <p className="release-generated">Generated {release.generated_at.replace('T', ' ').slice(0, 19)} UTC</p>
        <button className="secondary" disabled={busy || active.id === release.id} aria-label={`Use release ${release.as_of}`} onClick={() => void run(async () => { await onUse(release); setMessage(`Opened research from ${release.as_of}.`); })}>{active.id === release.id ? <><Check size={14} />Active release</> : <>Use this release <ArrowRight size={14} /></>}</button>
      </article>)}</div>
      <details className="library-help"><summary>How to update the research</summary><p>Export a Macro Atlas research file from the Dalio project, then choose “Import research file” here. The file contains the saved fundamentals and, when included, liquidity diagnostics. Importing works without rebuilding or reinstalling Atlas.</p><p>Imported releases are copied into your local application data. “Save a copy” exports a portable research file for backup or another computer. There is no automatic online refresh.</p></details>
      <p className="chart-caption">Macro Atlas 0.2.0 · Country boundaries: Natural Earth, public domain. Sector research, Börsdata company profiles and verified company locations are planned for later releases.</p>
    </div>
  </div>;
}

function ReleaseDescription({ release }: { release: ResearchRelease }) {
  return <p className="release-description">{release.country_count} economies · {release.indicator_count} indicators<br />{release.liquidity_as_of ? `Liquidity diagnostics · ${release.liquidity_as_of}` : 'Fundamentals, history and trade'}</p>;
}
