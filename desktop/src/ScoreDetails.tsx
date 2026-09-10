import { ArrowLeft, ArrowUpRight } from 'lucide-react';
import type { AtlasIndex, Category, Country, ResearchRelease } from './types';
import { assessmentPalette, categories, finite, format, quintile } from './model';
import { comparableScores, explainScore } from './scores';

export default function ScoreDetails({ index, country, category, previous, previousRelease, previousError, onBack, onIndicator }: {
  index: AtlasIndex; country: Country; category: Category; previous: AtlasIndex | null; previousRelease?: ResearchRelease; previousError: string;
  onBack: () => void; onIndicator: (name: string) => void;
}) {
  const explanation = explainScore(index,country,category);
  const priorCountry = previous && Object.values(previous.countries).find(c=>c.iso3 === country.iso3);
  const comparable = comparableScores(index,previous,category);
  const priorScore = priorCountry?.categories[category]?.score;
  const delta = comparable && finite(priorScore) && finite(explanation.reported) ? explanation.reported - priorScore : null;
  const band = quintile(explanation.reported);
  return <div className="score-details" data-score-category={category}>
    <button className="text-button" onClick={onBack}><ArrowLeft size={14} />Back to overview</button>
    <div className="eyebrow">WHY THIS SCORE</div><h3>{categories[category].label}</h3>
    <div className="score-headline"><strong style={{ color: band === null ? undefined : assessmentPalette[band] }}>{format(explanation.reported,2)}</strong><span>/ 100<small>Saved score · {index.as_of}</small></span></div>
    <p className="body-note">Each available indicator has equal weight. Raw values are ranked within the {index.ranking_population.length}-country comparison group; direction is adjusted so a higher percentile is stronger.{!country.on_map && " This regional aggregate is interpolated against that population and is not itself ranked."}</p>
    <div className="score-formula"><strong>{explanation.available} of {explanation.rows.length} indicators available</strong><span>At least {explanation.minimum} required. Missing indicators receive no weight.</span>{explanation.consistent && explanation.mean !== null && <code>Sum of percentiles ÷ {explanation.available} = {format(explanation.mean,2)}</code>}{!explanation.consistent && <span className="validation-error">The saved score does not reconcile to this breakdown. The original value is shown.</span>}</div>
    <section className="release-change"><h4>Since the previous release</h4>{!previousRelease ? <p className="section-note">No earlier fundamentals release is saved.</p> : previousError ? <p className="validation-error">The earlier release could not be opened. The current score remains available.</p> : !previous ? <p className="section-note">Opening {previousRelease.as_of}…</p> : <><div className="score-change"><strong className={delta === null || Math.abs(delta) < .005 ? 'neutral' : delta > 0 ? 'positive' : 'negative'}>{delta === null ? 'Not comparable' : Math.abs(delta) < .005 ? 'Unchanged' : `${delta > 0 ? '+' : ''}${format(delta,2)} points`}</strong><span>Compared with {previousRelease.as_of}</span></div><p className="chart-caption">{comparable ? 'A relative-score change can reflect both this economy and its peers. Raw observations and source dates are listed below.' : 'The country population or indicator definitions differ between releases.'}</p></>}</section>
    <h4>Indicator contributions</h4>
    {explanation.rows.map(row => {
      const old = comparable ? priorCountry?.indicators[row.meta.name] : undefined;
      const changed = old && (old.value !== row.cell?.value || old.date !== row.cell?.date || old.source !== row.cell?.source);
      const percentile = row.cell?.pct; const q = quintile(percentile);
      return <article className="score-indicator" key={row.meta.name}>
        <button className="score-indicator-title" onClick={()=>onIndicator(row.meta.name)}>{row.meta.label}<ArrowUpRight size={14} /></button>
        <div className="score-observation"><strong>{format(row.cell?.value,2)} <small>{row.meta.unit}</small></strong><span>{row.meta.higher_is_better ? 'Higher' : 'Lower'} values = stronger</span></div>
        <div className="contribution-grid"><div><small>STRENGTH PERCENTILE</small><strong>{format(percentile,1)}{finite(percentile) && ' / 100'}</strong><div className="strength-track"><i style={{ width:`${percentile ?? 0}%`,background:q === null?'#e2e5de':assessmentPalette[q] }} /></div></div><div><small>EFFECTIVE WEIGHT</small><strong>{row.weight === null ? '—' : `${format(row.weight*100,1)}%`}</strong></div><div><small>CONTRIBUTION</small><strong>{format(row.contribution,2)}{row.contribution !== null && ' pts'}</strong></div></div>
        <p className="evidence-line">{row.cell?.source?.replaceAll('_',' ') || 'Source unavailable'} · {row.cell?.date || 'Date unavailable'} · Tier {row.cell?.uncertainty || row.meta.uncertainty}{row.cell?.is_forecast && ' · forecast'}</p>
        {previous && comparable && old && <p className="change-note">{changed ? `Previous: ${format(old.value,2)} ${row.meta.unit} · ${old.date || 'date unavailable'}${old.source !== row.cell?.source ? ' · source changed' : ''}` : 'Observation unchanged since the previous release.'}</p>}
      </article>;
    })}
  </div>;
}
