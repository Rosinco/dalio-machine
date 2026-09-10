import { hasFinancialHistory, type FinancialIndex } from './financialData';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, Search } from 'lucide-react';
import type { BusinessIndex } from './business';
import type { Taxonomy } from './taxonomy';
import { companyEntries, listingCountries, normalizeSearch, type CompanyEntry } from './listingCatalogue';

const count = (n: number) => n.toLocaleString('en-US');
export function DirectorySearch({ index, taxonomy, financial, onCompany, onCountry }: { index: BusinessIndex | null; taxonomy: Taxonomy | null; financial: FinancialIndex | null; onCompany: (id: string) => void; onCountry: (code: string, name?: string) => void }) {
  const [query, setQuery] = useState(''), [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const entries = useMemo(() => companyEntries(index, taxonomy, financial), [index, taxonomy, financial]);
  const names = useMemo(() => listingCountries(index, taxonomy), [index, taxonomy]);
  useEffect(() => {
    const dismiss = (e: MouseEvent) => { if (!container.current?.contains(e.target as Node)) setOpen(false); };
    const escape = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', dismiss); document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('mousedown', dismiss); document.removeEventListener('keydown', escape); };
  }, []);
  const term = normalizeSearch(query), matches = entries.filter(c => c.search.includes(term));
  const companies = matches.slice(0, 20);
  const countries = Object.entries(names).filter(([code, name]) => normalizeSearch(`${code} ${name}`).includes(term));
  const choose = (action: () => void) => { action(); setQuery(''); setOpen(false); };
  return <div className="search" ref={container}><Search size={16} /><input aria-label="Search companies or countries" placeholder="Find a company, ticker or country…" value={query} onFocus={() => setOpen(true)} onChange={e => { setQuery(e.target.value); setOpen(true); }} onKeyDown={e => { if (e.key === 'Enter') { if (companies[0]) choose(() => onCompany(companies[0].id)); else if (countries[0]) choose(() => onCountry(...countries[0])); } }} /><span className="search-hint">{count(entries.length)} listings</span>
    {open && <div className="search-results">{companies.map(c => <button key={c.id} data-search-listing={c.id} onClick={() => choose(() => onCompany(c.id))}><span className="country-code">{c.listing_country ?? '—'}</span><span>{c.display_name}<small className="search-listing-note">{c.ticker ?? `ID ${c.id}`} · {c.financial ? hasFinancialHistory(c.financial) ? 'Financial history' : 'No saved reports' : c.profile ? 'Financial profile' : 'Directory entry'}</small></span><span className="result-note">{c.source_as_of !== taxonomy?.catalogue?.as_of && taxonomy?.catalogue ? 'Older download' : ''}</span></button>)}{matches.length > 20 && <p>Showing 20 of {count(matches.length)} matching listings. Refine your search.</p>}{countries.map(([code, name]) => <button key={code} onClick={() => choose(() => onCountry(code, name))}><span className="country-code">{code}</span>{name}<span className="result-note">Listing country</span></button>)}{!companies.length && !countries.length && <p>No matching company or country in this release.</p>}</div>}
  </div>;
}

export function CompanyList({ companies, taxonomy, onCompany }: { companies: CompanyEntry[]; taxonomy: Taxonomy | null; onCompany: (id: string) => void }) {
  const [query, setQuery] = useState(''), [page, setPage] = useState(0);
  useEffect(() => { setPage(0); setQuery(''); }, [companies]);
  const matches = companies.filter(c => c.search.includes(normalizeSearch(query)));
  const pages = Math.max(1, Math.ceil(matches.length / 50)), selectedPage = Math.min(page, pages - 1);
  const rows = matches.slice(selectedPage * 50, (selectedPage + 1) * 50);
  if (!companies.length) return <p className="empty">No company listings match this selection.</p>;
  return <div className="listing-table"><label className="directory-search"><Search size={14} /><input aria-label="Filter company listings" placeholder="Filter these company listings…" value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} /></label><div className="listing-pagination"><span role="status">{matches.length ? `${count(selectedPage * 50 + 1)}–${count(Math.min((selectedPage + 1) * 50, matches.length))}` : '0'} of {count(matches.length)} listings</span><div><button aria-label="Previous company page" disabled={!selectedPage} onClick={() => setPage(selectedPage - 1)}><ArrowLeft size={14} /></button><span>{selectedPage + 1} / {pages}</span><button aria-label="Next company page" disabled={selectedPage + 1 >= pages} onClick={() => setPage(selectedPage + 1)}><ArrowRight size={14} /></button></div></div>
    <div className="company-list">{rows.map(c => <button key={c.id} data-listing={c.id} onClick={() => onCompany(c.id)} aria-label={`Open ${c.display_name}`}><span className="company-monogram">{c.display_name.slice(0, 1)}</span><span><strong>{c.display_name}</strong><small>{c.ticker ?? 'Ticker unavailable'} · {c.listing_country ?? 'Country unavailable'} · ID {c.id}</small><span>{c.financial ? hasFinancialHistory(c.financial) ? `${c.financial.annual.count} annual · ${c.financial.quarterly.count} quarterly reports` : 'No saved reports' : c.profile ? 'Financial profile available' : 'Company directory entry'}{taxonomy?.catalogue && c.source_as_of !== taxonomy.catalogue.as_of && ` · Older download ${c.source_as_of}`}{['sector_mismatch', 'unclassified', 'needs_review'].includes(taxonomy?.classifications[c.id]?.status ?? '') && ' · Classification needs review'}</span></span><ArrowRight size={15} /></button>)}</div>{!rows.length && <p className="empty">No companies match this search.</p>}
  </div>;
}

export function ListingDetails({ entry, taxonomy, onSector, onMacro }: { entry: CompanyEntry; taxonomy: Taxonomy; onSector: () => void; onMacro: () => void }) {
  const catalogue = taxonomy.catalogue!, source = catalogue.snapshots.find(s => s.as_of === entry.source_as_of);
  const country = catalogue.countries[entry.country_id], sector = entry.sector_id ? taxonomy.sectors[entry.sector_id] : null, branch = entry.branch_id ? taxonomy.branches[entry.branch_id] : null;
  return <div data-listing-detail={entry.id}><section><h3>Company listing</h3><p className="business-note">Saved Börsdata identity and classification. This listing can now be found through its country and branch.</p>
    {entry.source_as_of !== catalogue.as_of && <p className="older-listing-note">Older download only · last recorded {entry.source_as_of}. This ID is absent from the {catalogue.as_of} instrument list; that alone does not establish its listing status.</p>}
    <dl className="listing-identity"><dt>Source name</dt><dd>{entry.name ?? 'Unavailable'}</dd><dt>Ticker / instrument ID</dt><dd>{entry.ticker ?? 'Unavailable'} / {entry.id}</dd><dt>ISIN</dt><dd>{entry.isin ?? 'Unavailable'}</dd><dt>Börsdata country</dt><dd>{country ? `${country.name_en} (${country.name})` : `Unmapped country ID ${entry.country_id}`}</dd><dt>Börsdata sector</dt><dd>{sector ? `${sector.name_en} · ${sector.name_sv}` : entry.sector_id ? `Unknown sector ID ${entry.sector_id}` : 'Unassigned'}</dd><dt>Börsdata branch</dt><dd>{branch ? `${branch.name_en} · ${branch.name_sv}` : entry.branch_id ? `Unknown branch ID ${entry.branch_id}` : 'Unassigned'}</dd><dt>Trading / reporting currency</dt><dd>{entry.stock_currency ?? 'Unavailable'} / {entry.report_currency ?? 'Unavailable'}</dd><dt>Listing date in source</dt><dd>{entry.listing_date ?? 'Unavailable'}</dd><dt>Identity snapshot</dt><dd>{entry.source_as_of}</dd></dl><p className="chart-caption">Each Börsdata ID is a separate listing. Multiple share classes or listings can represent the same business. Listing country does not identify its headquarters or operating assets.</p></section>
    <section><h3>Financial history and analysis</h3><p className="business-note">{entry.financial ? hasFinancialHistory(entry.financial) ? 'Saved financial history is available above. Company-specific analysis can be added as research grows.' : 'No usable annual or quarterly reports were found in the saved downloads for this listing.' : 'Financial charts and company-specific analysis have not yet been packaged for this listing. Its directory entry is ready for those additions.'}</p><div className="context-actions"><button onClick={onSector}>Explore the branch <ArrowRight size={14} /></button><button onClick={onMacro}>Open country macro data <ArrowRight size={14} /></button></div></section>
    {source && <section><details className="business-sources"><summary>Saved identity sources</summary>{[source.instruments, source.countries].map(s => <dl key={s.path}><dt>{s.path}</dt><dd className="hash">SHA-256 {s.sha256}</dd></dl>)}</details></section>}
  </div>;
}
