import { useMemo, useState } from 'react';
import { ArrowRight, BookOpen, Search } from 'lucide-react';
import type { BusinessIndex, CompanySummary } from './business';
import { branchCompanies, branchDives, directoryTotals, findBranches, type Branch, type Taxonomy } from './taxonomy';

export function TaxonomyDirectory({ taxonomy, index, branchId, onBranch }: { taxonomy: Taxonomy; index: BusinessIndex | null; branchId: string; onBranch: (id: string) => void }) {
  const [query, setQuery] = useState(''), [sector, setSector] = useState('all'), [coverage, setCoverage] = useState('all');
  const totals = useMemo(() => directoryTotals(taxonomy), [taxonomy]);
  const branches = useMemo(() => findBranches(taxonomy, query, sector, coverage, index), [taxonomy, query, sector, coverage, index]);
  const sectors = Object.values(taxonomy.sectors).sort((a, b) => a.name_en.localeCompare(b.name_en));
  return <div className="taxonomy-directory" data-taxonomy-ready="true">
    <div className="eyebrow">BÖRSDATA DIRECTORY</div><h2>Find your next branch</h2><p className="business-note">Browse the saved sector and branch divisions, then follow the research into individual companies.</p>
    <div className="directory-totals" aria-label="Directory coverage"><div><strong>{totals.sectors}</strong><span>Sectors</span></div><div><strong>{totals.branches}</strong><span>Branches</span></div><div><strong>{totals.studies}</strong><span>Branches studied</span></div><div><strong>{totals.dives}</strong><span>Saved deep dives</span></div></div>
    <p className="chart-caption">Study coverage comes from the project registry. Deep dives count saved company folders, including shared studies; their completion and freshness are not inferred. Full text is included only for selected profiles.</p>
    <label className="directory-search"><Search size={16} /><input aria-label="Search sectors and branches" placeholder="Forest, Skogsbolag, Biotech…" value={query} onChange={e => setQuery(e.target.value)} /></label>
    <div className="directory-filters"><label>Sector<select aria-label="Filter sectors" value={sector} onChange={e => setSector(e.target.value)}><option value="all">All sectors</option>{sectors.map(s => <option key={s.id} value={s.id}>{s.name_en} · {s.name_sv}</option>)}</select></label><label>Research coverage<select aria-label="Filter branch coverage" value={coverage} onChange={e => setCoverage(e.target.value)}><option value="all">All branches</option><option value="profiles">Profiles in Atlas</option><option value="studies">Branch study completed</option><option value="dives">Saved deep dives</option></select></label></div>
    <div className="directory-result-count" role="status">{branches.length} of {totals.branches} branches <span>Inventory {taxonomy.as_of}</span></div>
    <div className="sector-tree">{sectors.map(s => {
      const children = branches.filter(b => b.sector_id === s.id);
      if (!children.length) return null;
      return <details className="sector-group" key={`${s.id}:${query}:${sector}:${coverage}`} open={!!query || sector !== 'all' || coverage !== 'all' || taxonomy.branches[branchId]?.sector_id === s.id}>
        <summary><span><strong>{s.name_en}</strong><small>{s.name_sv} · sector {s.id}</small></span><span className="sector-count">{children.length}</span></summary>
        {children.map(b => { const profiles = branchCompanies(index, taxonomy, b.id).length, dives = branchDives(taxonomy, b).length; return <button className="branch-choice" key={b.id} data-branch={b.id} aria-label={`Explore ${b.name_en}`} aria-pressed={b.id === branchId} onClick={() => onBranch(b.id)}><span><strong>{b.name_en}</strong><small>{b.name_sv} · branch {b.id}</small><span className="coverage-badges"><span className={b.study_status === 'graduated' ? 'covered' : ''}>{b.study_status === 'graduated' ? 'Branch study' : 'Study pending'}</span>{dives > 0 && <span>{dives} saved {dives === 1 ? 'deep dive' : 'deep dives'}</span>}{profiles > 0 && <span className="covered">{profiles} profiles in Atlas</span>}</span></span><ArrowRight size={15} /></button>; })}
      </details>;
    })}{!branches.length && <p className="empty">No branches match these filters.</p>}</div>
    <p className="chart-caption">English and Swedish names share stable Börsdata IDs. Coverage badges describe available research, not investment quality.</p>
    <details className="business-sources"><summary>Directory sources and interpretation</summary>{taxonomy.notes.map(note => <p key={note} className="business-note">{note}</p>)}{taxonomy.sources.map(s => <dl key={s.path}><dt>{s.path}</dt><dd className="hash">SHA-256 {s.sha256}</dd></dl>)}</details>
  </div>;
}

export function BranchCoverage({ taxonomy, branch, profiles }: { taxonomy: Taxonomy; branch: Branch; profiles: number }) {
  const dives = branchDives(taxonomy, branch);
  return <section className="branch-coverage"><h3>Research coverage</h3><div className="coverage-grid"><div><small>BRANCH STUDY</small><strong>{branch.study_status === 'graduated' ? 'Completed' : 'Pending'}</strong><span>Saved project status</span></div><div><small>COMPANY DEEP DIVES</small><strong>{dives.length}</strong><span>Saved folders in project</span></div><div><small>PROFILES IN ATLAS</small><strong>{profiles}</strong><span>Available to open offline</span></div></div><p className="business-note">{branch.study_status === 'graduated' ? 'The project records a completed branch study.' : 'This branch has a research scaffold; it is not a completed branch analysis.'} {branch.shared_study_ids.length > 0 && `Shared research also covers ${branch.shared_study_ids.flatMap(id => taxonomy.shared_studies[id].branch_ids.filter(b => b !== branch.id).map(b => taxonomy.branches[b].name_en)).join(', ')}.`}</p><p className="chart-caption">Inventory {taxonomy.as_of}. Saved deep dives may have different dates and review states. Counts do not imply a current forecast.</p></section>;
}

export function BranchInventory({ taxonomy, branch }: { taxonomy: Taxonomy; branch: Branch }) {
  const dives = branchDives(taxonomy, branch);
  return <section className="branch-inventory"><h3>Research in your project</h3><p className="business-note">This inventory records saved source locations. Only research explicitly included in this Atlas release can be read here.</p><details className="business-sources"><summary>Branch study sources</summary><dl><dt>{branch.study_status === 'graduated' ? 'Completed branch study' : 'Research scaffold'}</dt><dd>{branch.overview.path}</dd><dd className="hash">SHA-256 {branch.overview.sha256}</dd></dl>{branch.shared_study_ids.map(id => { const study = taxonomy.shared_studies[id]; return <dl key={id}><dt>Shared study · {study.branch_ids.map(b => taxonomy.branches[b].name_en).join(' / ')}</dt><dd>{study.overview.path}</dd><dd className="hash">SHA-256 {study.overview.sha256}</dd></dl>; })}</details>
    {dives.length > 0 ? <div className="dive-inventory">{dives.map(dive => <details key={dive.folder}><summary><BookOpen size={14} /><span>{dive.label}<small>{dive.documents.length} saved {dive.documents.length === 1 ? 'document' : 'documents'} · project inventory</small></span></summary>{dive.documents.map(d => <dl key={d.path}><dt>{d.path}</dt><dd className="hash">SHA-256 {d.sha256}</dd></dl>)}</details>)}</div> : <p className="empty">No saved company deep-dive folders are registered for this branch.</p>}
  </section>;
}

export function ClassificationDetails({ taxonomy, company }: { taxonomy: Taxonomy | null; company: CompanySummary }) {
  const c = taxonomy?.classifications[company.id];
  if (!taxonomy || !c) return null;
  const label = (branchId: string) => { const b = taxonomy.branches[branchId]; return `${taxonomy.sectors[b.sector_id].name_en} → ${b.name_en}`; };
  return <details className="classification-details"><summary>Classification · {c.status === 'source' ? 'Börsdata default' : c.status === 'corrected' ? 'Reviewed correction' : c.status === 'aligned' ? 'Source agrees with review' : 'Correction needs review'}</summary><dl><dt>Börsdata original · {taxonomy.classification_as_of}</dt><dd>{label(c.source_branch_id)} · sector {c.source_sector_id}, branch {c.source_branch_id}</dd><dt>Used in Atlas</dt><dd>{label(c.branch_id)} · sector {c.sector_id}, branch {c.branch_id}</dd></dl>{c.correction && <><p>{c.correction.reason}</p><p className="chart-caption">Reviewed {c.correction.reviewed_at} · {c.correction.source}</p>{c.status === 'needs_review' && <p className="business-note">The source classification changed after this correction was reviewed. Atlas uses the source assignment until the saved correction is reviewed again.</p>}</>}<p className="chart-caption">Reviewed corrections preserve the original assignment. Segment tags and the archived peer cohort are maintained separately.</p></details>;
}
