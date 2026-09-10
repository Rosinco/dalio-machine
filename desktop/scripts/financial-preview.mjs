// Read-only desktop preview. The installed app uses the bundled Rust/SQLite store.
import { DatabaseSync } from 'node:sqlite';
import { closeSync, createReadStream, openSync, readSync, readdirSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('../financial-data/', import.meta.url));
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const idPattern = /^[a-f0-9]{64}$/;
const cache = new Map();
function pack(id) {
  if (!idPattern.test(id ?? '')) throw new Error('Invalid financial pack ID');
  const path = join(root, `${id}.sqlite`), stat = statSync(path);
  if (!stat.isFile() || stat.size > 512 * 1024 * 1024) throw new Error('Unsupported financial pack size');
  const saved = cache.get(id);
  if (saved && stat.size === saved.bytes && stat.mtimeMs === saved.mtimeMs) return saved;
  if (saved) { saved.db.close(); cache.delete(id); }
  const fd = openSync(path, 'r'), hash = createHash('sha256'), chunk = Buffer.alloc(65536);
  try { let n; while ((n = readSync(fd, chunk, 0, chunk.length, null))) hash.update(chunk.subarray(0, n)); } finally { closeSync(fd); }
  if (hash.digest('hex') !== id) throw new Error('Financial pack checksum mismatch');
  const db = new DatabaseSync(path, { readOnly: true, enableDoubleQuotedStringLiterals: false, allowExtension: false });
  try {
    db.exec('PRAGMA query_only=ON; PRAGMA trusted_schema=OFF;');
    const raw = db.prepare('SELECT payload FROM metadata WHERE key=?').get('index');
    const index = JSON.parse(gunzipSync(raw.payload, { maxOutputLength: 16000000 }));
    if (index.format !== 'macro-atlas-financials' || index.version !== 1) throw new Error('Unsupported financial pack');
    const result = { db, index: { ...index, id, bytes: stat.size }, path, bytes: stat.size, mtimeMs: stat.mtimeMs };
    cache.set(id, result); return result;
  } catch (e) { db.close(); throw e; }
}
function middleware(req, res, next) {
  const url = new URL(req.url, 'http://localhost');
  if (!url.pathname.startsWith('/api/financials/')) return next();
  try {
    if (req.method !== 'GET') throw new Error('The browser preview is read-only');
    const operation = url.pathname.slice('/api/financials/'.length);
    let result;
    if (operation === 'index') {
      const taxonomy = url.searchParams.get('taxonomy');
      if (!idPattern.test(taxonomy ?? '')) throw new Error('Invalid directory binding');
      const names = (() => { try { return readdirSync(root); } catch (e) { if (e.code === 'ENOENT') return []; throw e; } })();
      result = names.filter(n => /^[a-f0-9]{64}\.sqlite$/.test(n)).map(n => pack(n.slice(0, -7)).index).filter(i => i.taxonomy_sha256 === taxonomy).sort((a, b) => b.generated_at.localeCompare(a.generated_at))[0] ?? null;
    } else {
      const p = pack(url.searchParams.get('pack'));
      if (operation === 'export') {
        res.setHeader('Content-Type', 'application/octet-stream'); res.setHeader('Content-Length', p.bytes);
        res.setHeader('Content-Disposition', `attachment; filename="Macro-Atlas-Financials-${p.index.as_of}-${p.index.id.slice(0, 12)}.sqlite"`);
        createReadStream(p.path).pipe(res); return;
      }
      if (operation !== 'company') throw new Error('Unknown financial operation');
      const id = url.searchParams.get('id');
      if (!/^[1-9][0-9]{0,9}$/.test(id ?? '') || !p.index.companies[id]) throw new Error('Listing is unavailable in this financial pack');
      const row = p.db.prepare('SELECT payload, sha256 FROM companies WHERE id=?').get(id);
      const bytes = gunzipSync(row.payload, { maxOutputLength: 2000000 });
      if (digest(bytes) !== row.sha256 || row.sha256 !== p.index.companies[id].sha256) throw new Error('Company history checksum mismatch');
      result = JSON.parse(bytes);
    }
    res.setHeader('Content-Type', 'application/json'); res.setHeader('Cache-Control', 'no-store'); res.end(JSON.stringify(result));
  } catch (e) { res.statusCode = 400; res.setHeader('Content-Type', 'text/plain'); res.end(e.message); }
}
export default function financialPreview() {
  return { name: 'atlas-financial-preview', configureServer(server) { server.middlewares.use(middleware); }, configurePreviewServer(server) { server.middlewares.use(middleware); } };
}
