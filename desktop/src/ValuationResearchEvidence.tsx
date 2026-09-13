import { format } from './model';
import { buildResearchedValuation, type ResearchedStudy } from './researchedValuations';
import { type EvidenceBlock, type ReviewedStudy } from './researchedStudy';
import { scenarioKeys, scenarioNames } from './valuation';

const sourceAnchor = (study: ReviewedStudy, id: string) => `research-${encodeURIComponent(study.id)}-source-${encodeURIComponent(id)}`;
const classifications = { source: 'Source facts', assumption: 'Analyst assumptions', calculation: 'Calculated from the stated inputs' };
const valueText = (value: string | number | null, precision = 2) => value === null ? 'Unavailable' : typeof value === 'number' ? format(value, precision) : value;

function SourceLinks({ study, ids }: { study: ReviewedStudy; ids: string[] }) {
  return ids.length ? <span> · Sources: {ids.map((id, index) => <span key={id}>{index > 0 && '; '}<a href={`#${sourceAnchor(study, id)}`}>{study.sources.find(source => source.id === id)!.title}</a></span>)}</span> : null;
}

function Evidence({ study, block }: { study: ReviewedStudy; block: EvidenceBlock }) {
  return <div className="valuation-evidence-block" data-evidence-kind={block.classification}>
    <p className="valuation-source"><strong>{classifications[block.classification]}</strong><SourceLinks study={study} ids={block.sourceIds} /></p>
    {block.kind === 'paragraph' ? <p>{block.text}</p> : <div className="table-scroll"><table><thead><tr>{block.columns.map((column, index) => <th key={index}>{column}</th>)}</tr></thead><tbody>{block.rows.map((row, index) => <tr key={index}><th>{row.label}</th>{row.values.map((value, index) => <td key={index}>{valueText(value, block.precision)}</td>)}</tr>)}</tbody></table></div>}
  </div>;
}

export default function ValuationResearchEvidence({ study: input, edited }: { study: ResearchedStudy; edited: boolean }) {
  const { study, recovery } = buildResearchedValuation(input), schedule = study.recovery;
  const recoveryRows = [
    ['Estimated gross proceeds', 'gross'], ['All prior claims', 'claims'], ['Additional costs and taxes', 'costs'], ['Additional cash burn', 'cashBurn'],
    ['Funding shortfall before common equity', 'shortfall'], ['Net common-equity recovery', 'net'],
  ] as const;
  return <section className="valuation-card valuation-research-evidence" data-research-study={study.id}>
    <div className="eyebrow">RESEARCHED STARTING ASSUMPTIONS · {study.asOf}</div><h2>How the {study.name} scenarios were built</h2>
    <p>{edited ? 'Your working study differs from these starting assumptions. The tables below retain the original research calculations; current results use your edited fields.' : 'These source figures and analyst choices populate the working study.'} All amounts below are {study.currency} millions unless a different unit is shown.</p>
    {study.evidence.map((section, index) => <div key={index}><h3>{section.heading}</h3>{section.blocks.map((block, index) => <Evidence key={index} study={study} block={block} />)}</div>)}
    {schedule.status === 'unavailable' ? <div className="valuation-recovery-bridge"><h3>Separate recovery unavailable</h3><p>{schedule.reason}</p><p className="valuation-source"><SourceLinks study={study} ids={schedule.sourceIds} /></p></div> : <details className="valuation-recovery-bridge"><summary>Inspect the separate breakup calculation</summary>
      <p>{schedule.explanation}</p><p className="valuation-source"><SourceLinks study={study} ids={schedule.sourceIds} /></p>
      <div className="table-scroll"><table><thead><tr><th>Asset</th><th>Book value</th>{scenarioKeys.map(key => <th key={key}>{scenarioNames[key]} proceeds</th>)}</tr></thead><tbody>
        {schedule.assets.map((asset, index) => <tr key={index}><th>{asset.label}</th><td>{valueText(asset.book)}</td>{scenarioKeys.map(key => <td key={key}>{valueText(asset.proceeds[key])}</td>)}</tr>)}
        {recoveryRows.map(([label, field]) => <tr key={field}><th>{label}</th><td />{scenarioKeys.map(key => <td key={key}>{valueText(recovery![key][field])}</td>)}</tr>)}
        <tr><th>Realization year</th><td />{scenarioKeys.map(key => <td key={key}>{valueText(schedule.scenarios[key].year, 0)}</td>)}</tr>
      </tbody></table></div>
      <p>Net common-equity recovery = proceeds − prior claims − additional costs and taxes − additional cash burn, with zero equity recovery when claims exceed proceeds. Missing inputs leave recovery unavailable. This is a separate alternative to the continuing-business valuation.</p>
      <p>{schedule.limitations}</p>
    </details>}
    <h3>Sources and dates</h3>
    <ul className="valuation-source-list">{study.sources.map(source => <li key={source.id} id={sourceAnchor(study, source.id)}><strong>{source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a> : source.title}</strong><span>{source.date} · {source.location}</span>{source.path && <span>{source.path}</span>}{source.sha256 && <small>SHA-256 {source.sha256}</small>}</li>)}</ul>
    <p>Linked research: {study.deepDive.date} · {study.deepDive.path}. Prepared {study.asOf}; source dates remain independent of the selected macro release. Research version: {study.id}.</p>
  </section>;
}
