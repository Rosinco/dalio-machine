import { useState } from 'react';

const fields = [
  { id: 'business', title: '1. Understand the business', prompt: 'What does the company sell, who pays, and why do customers choose it?' },
  { id: 'economics', title: '2. Explain the financial history', prompt: 'What drives margins, cash generation and investment needs? Record the figures and periods that support your view.' },
  { id: 'risks', title: '3. Challenge the assumptions', prompt: 'What could weaken cash flow? Consider competition, cyclicality, debt, leases and assets that may need replacing.' },
  { id: 'next', title: '4. Decide what to investigate next', prompt: 'Which report, note or source would resolve the most important uncertainty before changing the valuation?' },
] as const;
type Notes = Record<typeof fields[number]['id'], string>;
type NotesRecord = { version: 1; companyId: string; notes: Notes; updated: string; releaseId: string; sourceAsOf: string };
const empty = (): Notes => ({ business: '', economics: '', risks: '', next: '' });

function openNotes(key: string, companyId: string): { saved: NotesRecord | null; error: string } {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return { saved: null, error: '' };
    const saved = JSON.parse(raw);
    if (saved?.version !== 1 || saved.companyId !== companyId || !saved.notes || fields.some(field => typeof saved.notes[field.id] !== 'string') || typeof saved.updated !== 'string' || !Number.isFinite(Date.parse(saved.updated)) || typeof saved.releaseId !== 'string' || typeof saved.sourceAsOf !== 'string') throw new Error('The saved notes use an unsupported format.');
    return { saved, error: '' };
  } catch (error) { return { saved: null, error: `Your saved notes could not open and have been preserved. ${String(error)}` }; }
}

export default function CompanyResearchNotes({ companyId, companyName, releaseId, sourceAsOf }: { companyId: string; companyName: string; releaseId: string; sourceAsOf: string }) {
  const key = `macro-atlas-company-notes-v1:${companyId}`;
  const [initial] = useState(() => openNotes(key, companyId));
  const [notes, setNotes] = useState<Notes>(initial.saved?.notes ?? empty);
  const [saved, setSaved] = useState(initial.saved);
  const [error, setError] = useState(initial.error);
  const write = (next: Notes) => {
    setNotes(next);
    const record: NotesRecord = { version: 1, companyId, notes: next, updated: new Date().toISOString(), releaseId, sourceAsOf };
    try { localStorage.setItem(key, JSON.stringify(record)); setSaved(record); setError(''); }
    catch { setError('These changes could not be saved. Keep this page open and copy your notes before leaving.'); }
  };
  return <section className="company-research-notes" aria-label={`${companyName} research notes`}>
    <div className="section-title"><h3>Your research notebook</h3><span role="status">{error ? 'Notes need attention' : saved ? `Saved on this device · ${new Date(saved.updated).toLocaleString()}` : 'Notes save on this device as you type'}</span></div>
    <p>Keep observations, assumptions and unanswered questions separate. Add report dates and source references so you can revisit your reasoning.</p>
    {error && <p role="alert">{error}</p>}
    <div className="company-notes-grid">{fields.map(field => <label key={field.id}>{field.title}<span>{field.prompt}</span><textarea aria-label={field.title} value={notes[field.id]} maxLength={20000} rows={5} disabled={!!initial.error} onChange={event => write({ ...notes, [field.id]: event.target.value })} /></label>)}</div>
    {saved && <p className="company-notes-source">Last edited with company data saved {saved.sourceAsOf}. Your notes are personal research; they do not mark a company as reviewed or replace its valuation assumptions.</p>}
  </section>;
}
