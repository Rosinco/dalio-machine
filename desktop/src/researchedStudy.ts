import { capitalLabels, scenarioKeys, validAmount, validDay, type ScenarioKey, type ValuationDraft } from './valuation';

export type ResearchSource = { id: string; title: string; date: string; location: string; url?: string; path?: string; sha256?: string };
export type EvidenceBlock = { classification: 'source' | 'assumption' | 'calculation'; sourceIds: string[] } & (
  { kind: 'paragraph'; text: string } |
  { kind: 'table'; columns: string[]; rows: { label: string; values: (string | number | null)[] }[]; precision?: number }
);
export type EvidenceSection = { heading: string; blocks: EvidenceBlock[] };
export type RecoverySchedule = {
  status: 'available'; explanation: string; sourceIds: string[];
  assets: { label: string; book: number | null; proceeds: Record<ScenarioKey, number | null> }[];
  scenarios: Record<ScenarioKey, { claims: number | null; costs: number | null; cashBurn: number | null; year: number | null }>;
  limitations: string;
};
// Company-specific accounting and forecast calculations belong in the reviewed
// source study. This contract transports their explicit results; it does not
// interpret vendor FCF, choose financial periods, or infer shareholder payments.
export type ReviewedStudy = {
  version: 2; id: string; company: string; isin: string; name: string; asOf: string; currency: string;
  deepDive: { date: string; path: string; sha256: string }; sources: ResearchSource[];
  price: { marketCap: number | null; date: string; narrative: string };
  years: number; ownership: string;
  capital: ValuationDraft['capital']; notes: ValuationDraft['notes'];
  scenarios: Record<ScenarioKey, { cashFlows: (number | null)[]; discountRate: number | null; terminalEquity: number | null; rationale: string }>;
  evidence: EvidenceSection[];
  recovery: RecoverySchedule | { status: 'unavailable'; reason: string; sourceIds: string[] };
};

// Validate bundled JSON at registration, including nulls and evidence references.
// An unavailable amount must be explicit, never silently defaulted to zero.
export function reviewedStudy(input: unknown): ReviewedStudy {
  const fail = (path: string): never => { throw new Error(`Invalid reviewed study: ${path}.`); };
  const object = (value: unknown, path: string): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : fail(path);
  const text = (value: unknown, path: string) => typeof value === 'string' && value.trim() ? value : fail(path);
  const array = (value: unknown, path: string): unknown[] => Array.isArray(value) ? value : fail(path);
  const amount = (value: unknown, path: string, minimum = -1e12) => value === null || validAmount(value) && value >= minimum ? value : fail(path);
  const day = (value: unknown, path: string) => validDay(value) ? value : fail(path);
  const digest = (value: unknown, path: string) => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value) ? value : fail(path);
  const s = object(input, 'root');
  if (s.version !== 2) fail('version');
  for (const key of ['id', 'company', 'isin', 'name', 'ownership']) text(s[key], key);
  if (typeof s.currency !== 'string' || !/^[A-Z]{3}$/.test(s.currency)) fail('currency');
  const asOf = day(s.asOf, 'asOf');
  const deepDive = object(s.deepDive, 'deepDive');
  day(deepDive.date, 'deepDive.date'); text(deepDive.path, 'deepDive.path'); digest(deepDive.sha256, 'deepDive.sha256');
  const sourceIds = new Set<string>();
  const sources = array(s.sources, 'sources');
  if (!sources.length) fail('sources');
  sources.forEach((value, i) => {
    const source = object(value, `sources[${i}]`), id = text(source.id, `sources[${i}].id`);
    if (sourceIds.has(id)) fail('duplicate source id');
    sourceIds.add(id);
    text(source.title, `sources[${i}].title`); text(source.location, `sources[${i}].location`); day(source.date, `sources[${i}].date`);
    if (source.sha256 !== undefined) digest(source.sha256, `sources[${i}].sha256`);
    if (source.path !== undefined) text(source.path, `sources[${i}].path`);
    if (source.url !== undefined && !/^https?:\/\//.test(text(source.url, `sources[${i}].url`))) fail(`sources[${i}].url`);
  });
  const refs = (value: unknown, path: string) => array(value, path).forEach(id => { if (typeof id !== 'string' || !sourceIds.has(id)) fail(`${path}: unknown source`); });
  const price = object(s.price, 'price');
  amount(price.marketCap, 'price.marketCap', 1e-9); text(price.narrative, 'price.narrative');
  if (day(price.date, 'price.date') > asOf) fail('price.date after asOf');
  if (typeof s.years !== 'number' || !Number.isInteger(s.years) || s.years < 1 || s.years > 50) fail('years');
  const capital = object(s.capital, 'capital'), notes = object(s.notes, 'notes');
  for (const key of Object.keys(capitalLabels)) amount(capital[key], `capital.${key}`);
  for (const key of ['business', 'macro', 'financing', 'recovery', 'decision']) text(notes[key], `notes.${key}`);
  const scenarios = object(s.scenarios, 'scenarios');
  for (const key of scenarioKeys) {
    const scenario = object(scenarios[key], `scenarios.${key}`), flows = array(scenario.cashFlows, `scenarios.${key}.cashFlows`);
    if (flows.length !== s.years) fail(`scenarios.${key}.cashFlows length`);
    flows.forEach(value => amount(value, `scenarios.${key}.cashFlows`));
    const rate = amount(scenario.discountRate, `scenarios.${key}.discountRate`, 0);
    if (rate !== null && rate > 100) fail(`scenarios.${key}.discountRate`);
    amount(scenario.terminalEquity, `scenarios.${key}.terminalEquity`, 0); text(scenario.rationale, `scenarios.${key}.rationale`);
  }
  const evidence = array(s.evidence, 'evidence');
  if (!evidence.length) fail('evidence');
  evidence.forEach((value, i) => {
    const section = object(value, `evidence[${i}]`); text(section.heading, `evidence[${i}].heading`);
    const blocks = array(section.blocks, `evidence[${i}].blocks`);
    if (!blocks.length) fail(`evidence[${i}].blocks`);
    blocks.forEach((value, j) => {
      const path = `evidence[${i}].blocks[${j}]`, block = object(value, path);
      if (!['source', 'assumption', 'calculation'].includes(String(block.classification))) fail(`${path}.classification`);
      refs(block.sourceIds, `${path}.sourceIds`);
      if (block.kind === 'paragraph') text(block.text, `${path}.text`);
      else if (block.kind === 'table') {
        const columns = array(block.columns, `${path}.columns`);
        if (columns.length < 2) fail(`${path}.columns`);
        columns.forEach(value => text(value, `${path}.columns`));
        if (block.precision !== undefined && (typeof block.precision !== 'number' || !Number.isInteger(block.precision) || block.precision < 0 || block.precision > 6)) fail(`${path}.precision`);
        array(block.rows, `${path}.rows`).forEach(value => {
          const row = object(value, `${path}.rows`); text(row.label, `${path}.rows.label`);
          const values = array(row.values, `${path}.rows.values`);
          if (values.length !== columns.length - 1) fail(`${path}.rows.values length`);
          values.forEach(value => { if (typeof value !== 'string') amount(value, `${path}.rows.values`); });
        });
      } else fail(`${path}.kind`);
    });
  });
  const recovery = object(s.recovery, 'recovery'); refs(recovery.sourceIds, 'recovery.sourceIds');
  if (recovery.status === 'unavailable') text(recovery.reason, 'recovery.reason');
  else if (recovery.status === 'available') {
    text(recovery.explanation, 'recovery.explanation'); text(recovery.limitations, 'recovery.limitations');
    const assets = array(recovery.assets, 'recovery.assets');
    if (!assets.length) fail('recovery.assets');
    assets.forEach(value => {
      const asset = object(value, 'recovery.assets'); text(asset.label, 'recovery.assets.label'); amount(asset.book, 'recovery.assets.book', 0);
      const proceeds = object(asset.proceeds, 'recovery.assets.proceeds');
      scenarioKeys.forEach(key => amount(proceeds[key], `recovery.assets.proceeds.${key}`, 0));
    });
    const scenarios = object(recovery.scenarios, 'recovery.scenarios');
    scenarioKeys.forEach(key => {
      const scenario = object(scenarios[key], `recovery.scenarios.${key}`);
      for (const item of ['claims', 'costs', 'cashBurn']) amount(scenario[item], `recovery.scenarios.${key}.${item}`, 0);
      if (scenario.year !== null && (typeof scenario.year !== 'number' || !Number.isInteger(scenario.year) || scenario.year < 0 || scenario.year > 50)) fail(`recovery.scenarios.${key}.year`);
    });
  } else fail('recovery.status');
  return input as ReviewedStudy;
}
