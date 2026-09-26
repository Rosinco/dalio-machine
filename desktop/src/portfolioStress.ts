export type PortfolioStressRow = {
  id: string;
  name: string;
  driver: string;
  weightPercent: string;
  lossPercent: string;
  notes: string;
};

export type PortfolioStressDraft = {
  exposureDate: string;
  exposureBasis: string;
  shock: string;
  evidence: string;
  assumptions: string;
  tolerancePercent: string;
  outsideModelRisks: string;
  rows: PortfolioStressRow[];
};

export type PortfolioStressResult = {
  enteredWeightPercent: number;
  residualWeightPercent: number;
  lossPercent: number;
  remainingCapitalPercent: number;
  tolerancePercent: number;
  toleranceDifferencePercent: number;
  toleranceComparison: 'above' | 'equal' | 'below';
};

export const portfolioStressStorageKey = 'macro-atlas-portfolio-stress-v1';
export const maxPortfolioStressRows = 100;
export type PortfolioStressSaved = { draft: PortfolioStressDraft; savedAt: string; raw: string | null };
export type PortfolioStressSession = PortfolioStressSaved & { dirty: boolean };
type StressStorage = Pick<Storage, 'getItem' | 'setItem'>;
const precision = 1e-9;
const preservedMessage = 'Saved portfolio stress uses an unreadable or unsupported format. Its original bytes are preserved; editing is disabled.';

export function blankPortfolioStressRow(id: string): PortfolioStressRow {
  return { id, name: '', driver: '', weightPercent: '', lossPercent: '', notes: '' };
}

export function blankPortfolioStress(): PortfolioStressDraft {
  return { exposureDate: '', exposureBasis: '', shock: '', evidence: '', assumptions: '', tolerancePercent: '', outsideModelRisks: '', rows: [blankPortfolioStressRow('position-1')] };
}

function percent(value: string): number | null {
  if (!/^(?:\d+(?:\.\d*)?|\.\d+)$/.test(value.trim())) return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 && number <= 100 ? number : null;
}

function validDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T00:00:00.000Z`);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value;
}

/** A single simultaneous loss scenario, measured against starting portfolio capital. */
export function calculatePortfolioStress(draft: PortfolioStressDraft): { errors: string[]; result: PortfolioStressResult | null } {
  const errors: string[] = [];
  if (!validDate(draft.exposureDate)) errors.push('Enter a valid date for the starting portfolio weights.');
  if (!draft.exposureBasis.trim()) errors.push('Record the source and ownership basis for the starting weights.');
  if (!draft.shock.trim()) errors.push('Describe the single shared shock or combined shock scenario.');
  if (!draft.evidence.trim()) errors.push('Record exposure evidence and sources, including what is unknown.');
  if (!draft.assumptions.trim()) errors.push('Record the assumed shock severity and how losses were chosen.');
  const tolerance = percent(draft.tolerancePercent);
  if (tolerance === null) errors.push('Enter your tolerable loss from 0 to 100% of starting portfolio capital.');
  if (!draft.rows.length) errors.push('Add at least one exposure row.');
  let totalWeight = 0;
  let loss = 0;
  for (const [index, row] of draft.rows.entries()) {
    const weight = percent(row.weightPercent), rowLoss = percent(row.lossPercent);
    if (!row.name.trim() || !row.driver.trim()) errors.push(`Row ${index + 1}: enter a position name and exposure driver.`);
    if (weight === null || rowLoss === null) errors.push(`Row ${index + 1}: starting weight and assumed loss must each be from 0 to 100%.`);
    if (weight !== null) totalWeight += weight;
    if (weight !== null && rowLoss !== null) loss += weight * rowLoss / 100;
  }
  if (totalWeight > 100 + precision) errors.push('Entered starting weights exceed 100%. Review overlaps and the exposure basis.');
  if (errors.length || tolerance === null) return { errors, result: null };
  const enteredWeightPercent = Math.min(100, totalWeight);
  const lossPercent = Math.min(100, loss);
  const difference = lossPercent - tolerance;
  const equal = Math.abs(difference) < precision;
  return {
    errors,
    result: {
      enteredWeightPercent,
      residualWeightPercent: Math.max(0, 100 - enteredWeightPercent),
      lossPercent,
      remainingCapitalPercent: Math.max(0, 100 - lossPercent),
      tolerancePercent: tolerance,
      toleranceDifferencePercent: equal ? 0 : difference,
      toleranceComparison: equal ? 'equal' : difference > 0 ? 'above' : 'below',
    },
  };
}

function check(condition: unknown): asserts condition {
  if (!condition) throw new Error(preservedMessage);
}

function object(value: unknown, keys: string[]): Record<string, unknown> {
  check(value !== null && typeof value === 'object' && !Array.isArray(value));
  check(Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key)));
  return value as Record<string, unknown>;
}

function text(value: unknown, maximum: number): void {
  check(typeof value === 'string' && value.length <= maximum);
}

/** Deliberate blanks and invalid draft numbers are retained for later correction. */
function validateDraft(value: unknown): asserts value is PortfolioStressDraft {
  const draft = object(value, ['exposureDate', 'exposureBasis', 'shock', 'evidence', 'assumptions', 'tolerancePercent', 'outsideModelRisks', 'rows']);
  text(draft.exposureDate, 10);
  for (const key of ['exposureBasis', 'shock', 'evidence', 'assumptions', 'outsideModelRisks']) text(draft[key], 10000);
  text(draft.tolerancePercent, 32);
  check(Array.isArray(draft.rows) && draft.rows.length <= maxPortfolioStressRows);
  const ids = new Set<string>();
  for (const value of draft.rows) {
    const row = object(value, ['id', 'name', 'driver', 'weightPercent', 'lossPercent', 'notes']);
    check(typeof row.id === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(row.id) && !ids.has(row.id));
    ids.add(row.id);
    text(row.name, 200); text(row.driver, 1000); text(row.notes, 5000);
    text(row.weightPercent, 32); text(row.lossPercent, 32);
  }
}

export function decodePortfolioStress(raw: string): { draft: PortfolioStressDraft; savedAt: string } {
  check(raw.length <= 2_000_000);
  let value: unknown;
  try { value = JSON.parse(raw); } catch { throw new Error(preservedMessage); }
  const saved = object(value, ['version', 'savedAt', 'draft']);
  check(saved.version === 1 && typeof saved.savedAt === 'string');
  const date = new Date(saved.savedAt);
  check(Number.isFinite(date.getTime()) && date.toISOString() === saved.savedAt);
  validateDraft(saved.draft);
  return { draft: saved.draft, savedAt: saved.savedAt };
}

export function loadPortfolioStress(storage: StressStorage): PortfolioStressSaved {
  const raw = storage.getItem(portfolioStressStorageKey);
  return raw === null ? { draft: blankPortfolioStress(), savedAt: '', raw } : { ...decodePortfolioStress(raw), raw };
}

/** Pass a fresh validated storage read first; transient edits never replace their expected saved bytes. */
export function restorePortfolioStressSession(saved: PortfolioStressSaved, transient: PortfolioStressSession | null): PortfolioStressSession & { storageChanged: boolean } {
  return transient?.dirty
    ? { ...transient, storageChanged: transient.raw !== saved.raw }
    : { ...saved, dirty: false, storageChanged: false };
}

/** Called only by an explicit Save action. Check existing bytes again before replacing them. */
export function savePortfolioStress(storage: StressStorage, draft: PortfolioStressDraft, expectedRaw: string | null, savedAt: string): string {
  const current = storage.getItem(portfolioStressStorageKey);
  if (current !== null) decodePortfolioStress(current);
  if (current !== expectedRaw) throw new Error('Saved portfolio stress changed in another view. Copy your unsaved notes before reloading; newer saved work has been preserved.');
  validateDraft(draft);
  const raw = JSON.stringify({ version: 1, savedAt, draft });
  decodePortfolioStress(raw);
  storage.setItem(portfolioStressStorageKey, raw);
  return raw;
}
