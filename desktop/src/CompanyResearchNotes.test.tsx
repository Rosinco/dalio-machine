import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CompanyResearchNotes from './CompanyResearchNotes';

describe('existing research notes are protected when they cannot be opened', () => {
  afterEach(() => vi.unstubAllGlobals());
  it.each([
    ['malformed JSON', '{unfinished'],
    ['unsupported version', JSON.stringify({ version: 999, notes: { business: 'Keep this original text' } })],
    ['another company', JSON.stringify({ version: 1, companyId: '696', notes: { business: 'Another company’s original note', economics: '', risks: '', next: '' }, updated: '2026-09-13T00:00:00Z', releaseId: 'saved-release', sourceAsOf: '2026-08-10' })],
  ])('does not overwrite %s and disables editing rather than offering a replacement blank notebook', (_label, raw) => {
    const writes = vi.fn(), reads = vi.fn(() => raw);
    vi.stubGlobal('localStorage', { getItem: reads, setItem: writes, removeItem: writes, clear: writes });
    const html = renderToStaticMarkup(createElement(CompanyResearchNotes, { companyId: '102', companyName: 'Fixture Company', releaseId: 'current-release', sourceAsOf: '2026-08-10' }));
    expect(reads).toHaveBeenCalledWith('macro-atlas-company-notes-v1:102');
    expect(writes).not.toHaveBeenCalled();
    expect(html).toContain('have been preserved');
    expect(html.match(/<textarea\b[^>]*\bdisabled=""/g)).toHaveLength(4);
    expect(reads()).toBe(raw);
  });
});
