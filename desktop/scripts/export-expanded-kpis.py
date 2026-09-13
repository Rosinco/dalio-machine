#!/usr/bin/env python3
"""Export the immutable provider screener as bounded, source-bound display shards.

No valuation inputs, strategy filters, backtest features, raw files or databases are
changed. One KPI is scanned at a time with Arrow predicate pushdown. Source units
are deliberately variant-specific; metadata alone is not a sufficient unit map.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

DESKTOP = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path('/mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client')
SNAPSHOT = '2026-08-10'
MAX_SHARD_RAW = 16 * 1024 * 1024
MAX_SHARD_ZIP = 4 * 1024 * 1024
MAX_TOTAL_ZIP = 250 * 1024 * 1024
EXCLUDED = {k: 'Known NCAV formula, percentage-scale and cross-currency defects; ADR 0100.' for k in (307, 308, 309, 310)}
LABELS = {23: 'Free cash flow per share', 41: 'Net debt %', 71: 'EBITDA per share',
          110: 'Insider transactions', 144: 'Buybacks', 145: 'Employees and board', 146: 'Short positions',
          148: 'Ordinary dividend yield', 151: 'Stock price and performance', 179: 'Free cash flow stability',
          757: 'Report currency', 758: 'Stock price currency', 759: 'Report-to-quote exchange rate'}
PERCENT_IDS = {1, 16, 17, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 41, 46, 51,
               92, 93, 94, 97, 98, 99, 100, 140, 148, 152, 169, 207, 208, 209, 210, 211, 212,
               216, 217, 218, 219, 220, 221, 223, 224, 225,
               279, 280, 283, 286, 287, 290, 291, 292, 293, 294, 295, 296, 311}
MULTIPLE_IDS = {2, 3, 4, 9, 10, 11, 12, 13, 15, 18, 19, 38, 40, 42, 44, 45, 74, 75, 76, 78, 278, 288, 289, 759}
PER_SHARE_IDS = {5, 6, 7, 8, 23, 66, 68, 69, 70, 71, 73, 277, 282, 285}
MONETARY_IDS = {49, 50, 53, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 137, 138, 281, 284}
COMMON_NOTES = [
    'Provider snapshot captured 2026-08-10; underlying price and report observation dates are not supplied per screener value.',
    'Provider calculations are descriptive snapshot context. They are not Atlas valuation inputs or formal research gates.',
    'Missing values remain missing; zero and negative observations are retained. Unequal observed scope duplicates are withheld.',
    'Provider multi-year aggregates and relative historical slots do not establish point-in-time availability or exact observation counts.',
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()


def category(k: int) -> str:
    if k in {201, 202, 168, 180, 181, 757, 758, 759}: return 'Info & dates'
    if k in {110, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239}: return 'Holdings · insider'
    if k in {144, 213, 214, 215}: return 'Holdings · buybacks'
    if k in {146, 207, 208, 209, 210, 211, 212}: return 'Holdings · shorts'
    if k == 145 or 226 <= k <= 228: return 'Employees & board'
    if 216 <= k <= 273: return 'Holdings · funds'
    if 277 <= k <= 289: return 'Real estate'
    if 290 <= k <= 296: return 'Banks'
    if k in {163, 164, 165, 167, 171, 172, 173, 190, 194, 195}: return 'Provider strategy scores'
    if 174 <= k <= 179: return 'Stability'
    if k in {151, 152, 169}: return 'Stock price & performance'
    if k >= 153: return 'Technical analysis'
    if k in {94, 97, 98, 99, 100}: return 'Growth'
    if k in {1, 7, 20, 21, 26, 66, 148}: return 'Dividends'
    if k in {2, 3, 4, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 49, 50, 74, 75, 76, 78}: return 'Valuation'
    if k in {23, 24, 25, 27, 31, 51, 62, 63, 64, 65, 68, 69, 138, 140}: return 'Cash flow'
    if k in {28, 29, 30, 32, 33, 34, 35, 36, 37, 38}: return 'Profitability'
    if k in {39, 40, 41, 42, 44, 45, 46, 92, 93}: return 'Financial strength'
    if k in {57, 58, 60, 61, 73, 126, 127, 128, 129, 130, 131, 132, 133, 134, 137}: return 'Balance sheet'
    return 'Income statement & per share'


def variant_semantics(k: int, cg: str, calc: str, metadata: dict) -> tuple[str, str, list[str]]:
    notes = []
    if k == 110 and calc in {'ValueBuy', 'ValueSell', 'ValueNet'} or k == 144:
        return 'number', 'unverified', ['Monetary trade or buyback amount withheld: the currency and scale of this aggregate are not established.']
    if k == 146 and calc in {'SumValue', 'AvgValue'}:
        return 'millions', 'unverified', ['Short-position value in provider millions is withheld because its currency is not established.']
    if k == 146 and calc in {'SumCapital', 'AvgCapital'}: return 'percent', 'none', []
    if k in (201, 202): return 'date', 'none', ['String date channel; numeric provider timestamp is not a financial value.']
    if k in (757, 758): return 'text', 'none', []
    if calc in {'cagr', 'growth', 'quarter', 'return'}:
        return 'percent', 'none', ['Provider growth calculation; quarter is a growth comparison, not an absolute quarterly report amount.']
    if k == 151 and cg == 'last': return 'price', 'quote', ['Provider last close; exact trading date is unavailable in this screener field.']
    if k == 153 and calc in {'high', 'low'}: return 'price', 'quote', []
    if k == 153 and calc in {'pricehigh', 'pricelow', 'default'}: return 'percent', 'none', []
    if k == 157 and calc == 'mean': return 'price', 'quote', []
    if k == 161: return 'price', 'quote', ['Bollinger levels and band width are quote-price amounts; width equals upper band minus lower band.']
    if k == 313 and calc == 'mill': return 'millions', 'quote', ['Monetary trading turnover in millions of quote currency, not millions of shares.']
    if k == 313 and calc == 'mean': return 'count', 'none', ['Number of traded shares, not monetary turnover.']
    if k in {321, 322}: return 'percent', 'none', []
    if k == 317: return 'price', 'quote', []
    if k in {49, 50}: return 'millions', calc.upper() if calc in {'sek', 'usd'} else 'quote', ['Provider market capitalization or enterprise value; shares and price bases can differ from saved annual shares.']
    if k == 61: return 'millions', 'none', ['Millions of reported shares, not an independently verified diluted denominator.']
    if calc == 'psh' or k in PER_SHARE_IDS:
        return 'per_share', 'report', ['Provider per-share levels use report currency. Cross-currency levels are withheld except verified latest revenue, earnings, dividend and book-value per-share fields.']
    if k in MONETARY_IDS:
        return 'millions', 'report', ['Provider financial totals in millions of report currency; cross-currency levels are withheld pending field-level reconciliation.']
    if k in PERCENT_IDS or metadata.get('format') == '%': return 'percent', 'none', []
    if k in MULTIPLE_IDS: return 'multiple', 'none', []
    if k in {168, 180, 181}: return 'count', 'none', ['Days or years as identified by the provider field; measured at the saved snapshot.']
    if k == 145 and calc.endswith('Pct') or k == 145 and 'Pct' in calc: return 'percent', 'none', []
    if k == 110 and calc in {'Buyers', 'Sellers', 'BuyersDiffSellers', 'Buys', 'Sells', 'BuyesDiffSells', 'SharesBuy', 'SharesSell', 'SharesNet'}: return 'count', 'none', []
    if k == 146 and calc == 'LastTranDate': return 'date', 'none', ['Only a valid supplied ISO string date is shown.']
    return 'number', 'none', ['Provider-native numeric scale; no undocumented percentage, currency or ratio conversion is inferred.']


def variant_details(k: int, cg: str, calc: str, metadata: dict) -> tuple[str, str, list[str]]:
    unit, basis, notes = variant_semantics(k, cg, calc, metadata)
    if k in {277, 287}: notes.append('Property NAV can use a stale split basis; Atrium Ljungberg is withheld for the documented 5-for-1 mismatch.')
    if k in {23, 24, 25, 26, 27, 31, 51, 62, 63, 64, 65, 68, 69, 138, 140, 178, 179}: notes.append('Provider cash flow is not reconciled owner cash; leases and investment classification require review.')
    if k == 152: notes.append('Provider total return uses simple price change plus non-reinvested dividends, not reinvested total shareholder return.')
    if category(k) == 'Provider strategy scores': notes.append('Provider score or rank only; its construction and universe are not an Atlas research verdict. Net-net measures require independent reconstruction.')
    return unit, basis, notes


def apply_currency_policy(values: list, ids: list[str], report: list, quote: list, k: int, cg: str, calc: str, basis: str) -> tuple[list, int]:
    result, withheld = list(values), 0
    verified_latest = k in {5, 6, 7, 8} and cg == 'last' and calc == 'latest'
    for i, value in enumerate(result):
        if value is None: continue
        if k in {277, 287} and ids[i] == '20': result[i] = None; continue
        if basis == 'unverified' or basis == 'report' and (not report[i] or not quote[i] or report[i] != quote[i] and not verified_latest) or basis == 'quote' and not quote[i]:
            result[i] = None; withheld += 1
    return result, withheld


def resolve_cells(frame: pd.DataFrame, ids: list[str], unit: str) -> tuple[list, dict]:
    """Coalesce observed values only when identical, never prefer a numeric scope."""
    is_text = unit in {'date', 'text'}
    f = frame.copy()
    if is_text:
        f['value'] = f['s'].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else None)
        if unit == 'date':
            def valid_date(v):
                if not isinstance(v, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', v): return None
                try: dt.date.fromisoformat(v)
                except ValueError: return None
                return v
            f['value'] = f['value'].map(valid_date)
        invalid = int((f.s.notna() & f.value.isna()).sum())
    else:
        f['value'] = f['n'].where(np.isfinite(f.n), np.nan)
        # The API often delivers both raw n and a rounded Swedish display string
        # (41.400001525878906 / '41,4%'). Only contradictory companions conflict.
        unexpected = pd.Series([not numeric_companion_agrees(n, s) for n, s in zip(f.n, f.s)], index=f.index)
        invalid = int((f.n.notna() & ~np.isfinite(f.n)).sum())
        f.loc[unexpected, 'value'] = np.nan
    groups = f.groupby('ins_id', sort=False)['value']
    counts = groups.agg(['size', 'count', 'nunique', 'first'])
    conflicts = set(counts.index[counts['nunique'] > 1].astype(int))
    if not is_text:
        conflicts.update(f.loc[unexpected, 'ins_id'].astype(int))
    values = {int(ins): value for ins, value in counts['first'].items() if pd.notna(value) and int(ins) not in conflicts}
    result = [values.get(int(ins)) for ins in ids]
    return result, {'conflictCount': sum(int(ins) in conflicts for ins in ids),
                    'scopeAvailabilityDifferenceCount': int(((counts['size'] > counts['count']) & (counts['count'] > 0)).sum()),
                    'invalidValueCount': invalid}


def numeric_companion_agrees(value: float, text: object) -> bool:
    if not isinstance(text, str) or not text.strip(): return True
    if not isinstance(value, (int, float)) or not math.isfinite(value): return False
    rendered = text.strip().replace('\u00a0', '').replace(' ', '').replace('−', '-')
    if rendered.endswith('%'): rendered = rendered[:-1]
    if not re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?', rendered): return False
    normalized = rendered.replace(',', '.')
    parsed = float(normalized)
    decimals = len(normalized.split('.')[1]) if '.' in normalized else 0
    # Half the displayed last unit plus float32 storage roundoff, not a
    # financial plausibility tolerance or an inferred percent conversion.
    tolerance = 0.5 * 10 ** -decimals + abs(value) * 2e-7
    return math.isfinite(parsed) and abs(parsed - value) <= tolerance


def parse_wiki(root: Path) -> dict:
    out = {}
    for name in ['Kpi-Screener-List.md', 'Kpi-Holdings-List.md', 'Screener-history-kpi-list.md']:
        for line in (root/'reference/api_wiki_2026-08-10'/name).read_text().splitlines():
            bits = [v.strip() for v in line.split('|')]
            if len(bits) >= 6 and bits[1].isdigit():
                k, cg, calc, desc = int(bits[1]), bits[2], bits[3], bits[4]
                if name == 'Kpi-Holdings-List.md': cg, calc = calc, cg
                out[(k, cg, calc)] = desc
                out.setdefault((k, calc, cg), desc)  # Preserved wiki column-order defects.
    return out


def export(raw_root: Path, check: bool = False) -> dict:
    snap = raw_root/'data/raw_api_snapshots'/SNAPSHOT
    output = DESKTOP/'public/data/expanded-kpis'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = DESKTOP/'src/data/expanded-kpi-manifest.json'
    gauge_path = DESKTOP/'src/data/research-gauge-manifest.json'
    taxonomy_path = DESKTOP/'public/data/taxonomy.json'
    gauge = json.loads(gauge_path.read_text())
    taxonomy = json.loads(taxonomy_path.read_text())
    if identity(taxonomy_path)['sha256'] != gauge['taxonomySha256']: raise ValueError('Taxonomy binding mismatch')
    source_paths = ['screener/screener_values.parquet', 'all_kpis/kpi_metadata.parquet', 'all_instruments/all_instruments.parquet']
    source_manifest = (snap/'manifest.md').read_text()
    sources = []
    for relative in source_paths:
        path = snap/relative
        actual = identity(path)
        match = re.search(r'\| `' + re.escape(relative) + r'` \| (\d+) \| (\d+) \| `([a-f0-9]{64})`', source_manifest)
        if not match or int(match[2]) != actual['bytes'] or match[3] != actual['sha256']: raise ValueError(f'Source manifest mismatch: {relative}')
        sources.append({'path': relative, **actual, 'rows': int(match[1])})
    initial_bindings = {str(path): identity(path) for path in [gauge_path, taxonomy_path]}
    ids = sorted(taxonomy['classifications'], key=int)
    if len(ids) != gauge['rows']: raise ValueError('Universe size mismatch')
    master = {str(r['ins_id']): r for r in pq.read_table(snap/source_paths[2]).to_pylist()}
    report = [master.get(ins, {}).get('report_currency') for ins in ids]
    quote = [master.get(ins, {}).get('stock_price_currency') for ins in ids]
    index = {'format': 'macro-atlas-expanded-kpi-index', 'version': 1, 'snapshot': SNAPSHOT, 'ids': ids, 'reportCurrencies': report, 'quoteCurrencies': quote}
    written = []
    def write_artifact(filename: str, value: dict) -> dict:
        raw = json_bytes(value)
        data = gzip.compress(raw, compresslevel=9, mtime=0)
        if len(raw) > MAX_SHARD_RAW or len(data) > MAX_SHARD_ZIP: raise ValueError(f'Artifact size bound exceeded: {filename}')
        path = output/filename
        if check:
            if not path.exists() or path.read_bytes() != data: raise ValueError(f'Artifact differs: {path}')
        else: path.write_bytes(data)
        written.append(len(data))
        return {'path': f'data/expanded-kpis/{filename}', 'sha256': sha(data), 'bytes': len(data), 'uncompressedSha256': sha(raw), 'uncompressedBytes': len(raw)}
    index_descriptor = write_artifact('index.bin', index)
    metadata = {r['kpi_id']: r for r in pq.read_table(snap/source_paths[1]).to_pylist()}
    docs = parse_wiki(raw_root)
    # Metadata is small; kpi-id min/max row-group statistics reduce the data scan.
    provider_ids = sorted({key[0] for key in docs})
    dataset = ds.dataset(snap/source_paths[0])
    metrics, variants, shards, pending, pending_values = [], [], [], [], []
    audit = {'sourceRowsScanned': 0, 'sourceIdsOutsideAtlas': set(), 'excluded': dict(EXCLUDED), 'conflictCount': 0, 'withheldCurrencyCount': 0}
    def flush():
        if not pending: return
        ordinal = len(shards)
        descriptor = write_artifact(f'shard-{ordinal:03d}.bin', {'format': 'macro-atlas-expanded-kpi-shard', 'version': 1, 'snapshot': SNAPSHOT, 'variantIds': [v['id'] for v in pending], 'values': pending_values})
        descriptor['id'] = ordinal
        descriptor['variantIds'] = [v['id'] for v in pending]
        shards.append(descriptor)
        for offset, variant in enumerate(pending): variant.update(shard=ordinal, offset=offset)
        pending.clear(); pending_values.clear()
    for position, k in enumerate(provider_ids):
        if k in EXCLUDED: continue
        frame = dataset.to_table(filter=ds.field('kpi_id') == k).to_pandas()
        if frame.empty: continue
        audit['sourceRowsScanned'] += len(frame)
        audit['sourceIdsOutsideAtlas'].update(set(map(str, frame.ins_id.unique())) - set(ids))
        frame = frame[frame.ins_id.isin(set(map(int, ids)))]
        m = metadata.get(k, {})
        label = LABELS.get(k, (m.get('name_en') or f'Provider KPI {k}').strip())
        metric = {'id': f'provider_{k}', 'providerKpiId': k, 'label': label, 'category': category(k), 'description': f'Börsdata {label}; exact saved provider calculation and selected period.', 'unit': 'number', 'currencyBasis': 'none', 'variants': []}
        for (cg, calc, source), group in frame.groupby(['calc_group', 'calc', 'source'], sort=True):
            unit, basis, notes = variant_details(k, cg, calc, m)
            values, counts = resolve_cells(group, ids, unit)
            values, currency_withheld = apply_currency_policy(values, ids, report, quote, k, cg, calc, basis)
            variant_id = f'provider_{k}_{cg}_{calc}_{source}'
            desc = docs.get((k, cg, calc), f'{label} {cg} {calc}')
            if k == 23: desc = desc.replace('[FCF growth]', '[Free cash flow per share]')
            v = {'id': variant_id, 'metricId': metric['id'], 'calcGroup': cg, 'calculation': calc, 'source': source, 'label': desc.replace('[', '').replace(']', '').strip(), 'unit': unit, 'currencyBasis': basis, 'availableCount': sum(x is not None for x in values), **counts, 'withheldCurrencyCount': currency_withheld, 'notes': notes}
            metric['variants'].append(variant_id)
            variants.append(v); pending.append(v); pending_values.append(values)
            audit['conflictCount'] += counts['conflictCount']; audit['withheldCurrencyCount'] += currency_withheld
            if len(pending) == 16: flush()
        metric['unit'], metric['currencyBasis'] = variant_semantics(k, 'last', 'latest', m)[:2]
        if metric['variants']: metrics.append(metric)
        if position % 15 == 0: print(f'Exported {len(metrics)} metrics / {len(variants)} variants / {len(shards)} shards', flush=True)
    flush()
    if sum(written) > MAX_TOTAL_ZIP: raise ValueError('Total compressed payload exceeds budget')
    for source in sources:
        if identity(snap/source['path']) != {k: source[k] for k in ['sha256', 'bytes']}: raise ValueError('Raw source changed during export')
    for path, actual in initial_bindings.items():
        if identity(Path(path)) != actual: raise ValueError('Atlas source binding changed during export')
    manifest = {'format': 'macro-atlas-expanded-kpi-manifest', 'version': 1, 'snapshot': SNAPSHOT, 'financialPackId': gauge['financialPackId'], 'taxonomySha256': gauge['taxonomySha256'], 'rows': len(ids), 'index': index_descriptor, 'metrics': metrics, 'variants': variants, 'shards': shards, 'sources': sources, 'warnings': COMMON_NOTES, 'excluded': [{'providerKpiId': k, 'reason': v} for k, v in sorted(EXCLUDED.items())], 'stats': {'metrics': len(metrics), 'variants': len(variants), 'availableMetrics': sum(any(v['availableCount'] for v in variants if v['metricId'] == m['id']) for m in metrics), 'availableVariants': sum(v['availableCount'] > 0 for v in variants), 'cells': sum(v['availableCount'] for v in variants), 'compressedBytes': sum(written), 'snapshotInstrumentCount': len(master), 'atlasListingsInSnapshot': sum(ins in master for ins in ids), 'atlasListingsAbsentSnapshot': sum(ins not in master for ins in ids)}, 'fetchMetadata': (snap/'fetch_meta.yaml').read_text(), 'exporterSha256': identity(Path(__file__))['sha256']}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if check:
        if manifest_path.read_text() != encoded: raise ValueError('Manifest differs')
    else: manifest_path.write_text(encoded)
    audit['sourceIdsOutsideAtlas'] = sorted(audit['sourceIdsOutsideAtlas'], key=int)
    audit.update(manifestSha256=sha(encoded.encode()), stats=manifest['stats'], sources=sources)
    receipt = DESKTOP/'test-results/expanded-kpi-export-verification.json'
    if not check: receipt.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(manifest['stats'], indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=DEFAULT_RAW)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    export(args.raw_root, args.check)
