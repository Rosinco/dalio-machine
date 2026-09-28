"""Reproduce the frozen 0.23.2 ranking from hash-bound research assets.

Usage: python3 scripts/verification/build-normal-year-ranking.py test-results/<new-output>
Writes only to a new directory under desktop/test-results; never changes fixtures.
Requires the original local assets and Node for presentation collation only.
"""
import datetime as dt
import gzip
import hashlib
import json
import math
import statistics
import sys
import subprocess
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
if len(sys.argv) != 2:
    raise SystemExit('Provide a new output directory under desktop/test-results.')
OUT = Path(sys.argv[1]).resolve()
if not OUT.is_relative_to(DESKTOP / 'test-results') or OUT == DESKTOP / 'test-results':
    raise SystemExit('Output must be a new directory under desktop/test-results.')
OUT.mkdir(parents=True, exist_ok=False)
WATCH = True
WINDOW = 'normal_2020_2023:5'
read = lambda p: json.loads(p.read_text())
def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
def col(id, kpi=None, window='latest', calculation='latest', minimum=None, maximum=None):
    c = dict(id=id, kpiId=kpi or id, window=window, calculation=calculation)
    if minimum is not None or maximum is not None:
        c['range'] = dict(min='' if minimum is None else str(minimum), max='' if maximum is None else str(maximum))
    return c

# Fixed thresholds chosen before inspecting matching issuer names.
columns = [
    col('sector'), col('branch'),
    col('normal_roce_median', 'normal_roce', WINDOW, 'median', 20),
    col('normal_roce_min', 'normal_roce', WINDOW, 'min', 10),
    col('normal_rota_median', 'normal_rota', WINDOW, 'median', 8),
    col('normal_ebit_margin_median', 'ebit_margin', WINDOW, 'median', 12),
    col('normal_ebit_margin_min', 'ebit_margin', WINDOW, 'min', 8),
    col('normal_revenue_growth', 'revenue', WINDOW, 'growth', 3),
    col('normal_cfo_growth', 'cfo', WINDOW, 'growth', 0),
    col('normal_tangible_revenue', 'tangible_assets_revenue', WINDOW, 'median', 0, 0.5),
    col('normal_positive_fcf', 'positive_fcf', WINDOW, 'latest', 5),
    col('normal_positive_ebit', 'positive_ebit', WINDOW),
    col('net_debt_ebitda', 'provider_42', 'provider:screener:last', 'provider:latest', maximum=1.5),
    col('ebitda_margin', 'provider_32', 'provider:screener:last', 'provider:latest'),
    col('normal_npv_full', 'normal_npv_percent', calculation='terminal_100', minimum=100*0.3/0.7),
    col('normal_npv_half', 'normal_npv_percent', calculation='terminal_50', minimum=0),
    col('normal_npv_cash_only', 'normal_npv_percent', calculation='terminal_0'),
    col('normal_fcf_median', 'fcf', WINDOW, 'median'),
    col('normal_cash_pv'), col('normal_terminal_pv'),
    col('all_year_npv_full', 'valuation_attractiveness', calculation='terminal_100'),
    col('all_year_low_npv', 'low_npv_percent'), col('price_date'), col('annual_date'),
]
view = dict(name='Kvalitetsbolag med rabatt – normalår', columns=columns[:2] + columns[14:17] + columns[2:14] + columns[17:],
    filters=dict(query='', sectorId='all', branchId='all', country='all', route='operating', readiness='all', presence='latest', watchlistOnly=False, preset='normal_quality', numericRules=[dict(column=col('ebitda_positive', 'provider_32', 'provider:screener:last', 'provider:latest'), operator='gt', value=0)]),
    sort=dict(columnId='normal_npv_half', direction='desc'))
if WATCH:
    view['name'] = 'Kvalitetsbolag – normalår, prisbevakning'
    for c in columns:
        if c['kpiId'] == 'normal_npv_percent': c.pop('range', None)
# Reuse the established quality-watch criteria without rewriting its saved view.
gm = read(DESKTOP / 'src/data/research-gauge-manifest.json')
km = read(DESKTOP / 'src/data/expanded-kpi-manifest.json')
assert gm['financialPackId'] == km['financialPackId'] and gm['taxonomySha256'] == km['taxonomySha256']
verified = []
def decode(d):
    packed = (DESKTOP / 'public' / d['path']).read_bytes()
    raw = gzip.decompress(packed)
    for blob, h, size in ((packed, 'sha256', 'bytes'), (raw, 'uncompressedSha256', 'uncompressedBytes')):
        assert len(blob) == d[size] and hashlib.sha256(blob).hexdigest() == d[h]
    verified.append({k:d[k] for k in ('path','sha256','bytes','uncompressedSha256','uncompressedBytes')})
    return json.loads(raw)
gauge = decode(gm['artifact'])
index = decode(km['index'])
positions = {id:i for i,id in enumerate(index['ids'])}
vectors, variants = {}, []
for c in columns:
    if not c['kpiId'].startswith('provider_'): continue
    _, source, group = c['window'].split(':')
    v = next(v for v in km['variants'] if v['metricId'] == c['kpiId'] and v['source'] == source and v['calcGroup'] == group and 'provider:'+v['calculation'] == c['calculation'])
    shard = next(s for s in km['shards'] if s['id'] == v['shard'])
    decoded = decode(shard)
    assert decoded['variantIds'][v['offset']] == v['id']
    vectors[c['id']] = decoded['values'][v['offset']]
    variants.append(v)
finite = lambda x: isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x)
day = dt.date.fromisoformat
def selected(r):
    h = r.get('screeningAnnual')
    if not h: return []
    return [i for i,p in enumerate(h['periods']) if not 2020 <= int(p['end'][:4]) <= 2023][:5]
def eligible(r):
    h, ix = r.get('screeningAnnual'), selected(r)
    if not h or len(ix) != 5 or r['route'] != 'operating' or r['classificationConflict']: return False
    ps = h['periods']
    if int(ps[0]['end'][:4]) <= 2023 or (day(h['asOf']) - day(ps[0]['end'])).days > 550: return False
    if ix[0] != 0 or any(not ps[i]['published'] for i in ix): return False
    for i,p in enumerate(ps):
        if not 330 <= (day(p['end'])-day(p['start'])).days+1 <= 400 or p['currency'] != ps[0]['currency']: return False
        if i and (p['year'] != ps[i-1]['year']-1 or not 1 <= (day(ps[i-1]['start'])-day(p['end'])).days <= 35): return False
    return True
def normal_values(r, key):
    h, ix = r['screeningAnnual'], selected(r)
    def ratio(a,b): return 100*a/b if finite(a) and finite(b) and b>0 else None
    xs=[]
    for i in ix:
        if key == 'ebit_margin': x=ratio(h['ebit'][i],h['revenue'][i])
        elif key == 'normal_roce':
            a,b=h['equity'][i],h['netDebt'][i]
            x=ratio(h['ebit'][i],a+b) if finite(a) and finite(b) else None
        elif key == 'normal_rota':
            a,b=h['assets'][i],h['intangibleAssets'][i]
            x=ratio(h['profit'][i],a-b) if finite(a) and finite(b) else None
        elif key == 'tangible_assets_revenue':
            a,b=h['tangibleAssets'][i],h['revenue'][i]
            x=a/b if finite(a) and finite(b) and b>0 else None
        else: x=h[{'fcf':'cash','cfo':'operatingCash','positive_fcf':'cash','positive_ebit':'ebit'}.get(key,key)][i]
        xs.append(x)
    return xs
def value(r,c):
    k=c['kpiId']
    if c['id'] in vectors: return vectors[c['id']][positions[r['id']]]
    if c['window'] == WINDOW:
        if not eligible(r): return None
        xs=normal_values(r,k)
        if len(xs)!=5 or not all(finite(x) for x in xs): return None
        if k.startswith('positive_'): return sum(x>0 for x in xs)
        calc=c['calculation']
        if calc=='median': return statistics.median(xs)
        if calc=='min': return min(xs)
        if calc=='growth':
            if min(xs)<=0:return None
            ix=selected(r); ps=r['screeningAnnual']['periods']
            years=(day(ps[ix[0]]['end'])-day(ps[ix[-1]]['end'])).days/365.25
            return 100*((xs[0]/xs[-1])**(1/years)-1)
    v=r['valuation']
    if k in ('normal_npv_percent','normal_cash_pv','normal_terminal_pv'):
        if not eligible(r):return None
        xs=normal_values(r,'fcf')
        if not all(finite(x) for x in xs):return None
        median=statistics.median(xs)
        cash=sum(median/1.1**t for t in range(1,11))
        terminal=median/0.1/1.1**10
        if k=='normal_cash_pv':return cash
        if k=='normal_terminal_pv':return terminal
        if not finite(v['candidateEquity']) or v['candidateEquity']<=0 or v['currency']!=r['screeningAnnual']['periods'][0]['currency'] or not v['priceDate']:return None
        credit=int(c['calculation'].split('_')[1])/100
        return 100*((cash+credit*terminal)/v['candidateEquity']-1)
    if k=='valuation_attractiveness':
        if not all(finite(v[x]) for x in ('cashPV','terminalPV','candidateEquity')) or v['candidateEquity']<=0:return None
        return 100*((v['cashPV']+v['terminalPV'])/v['candidateEquity']-1)
    if k=='low_npv_percent':return 100*(v['lowValue']/v['candidateEquity']-1) if finite(v['lowValue']) and finite(v['candidateEquity']) and v['candidateEquity']>0 else None
    return None

# User-confirmed policy: years 1–5 cash only, 60% value and 40% quality.
# The previous 50/50 fixture remains unchanged in its separate evidence folder.
MODEL = 'normal-quality-watch-npv5-60-40-v2'
RANK_KEYS = ('roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda')
saved_watch = read(DESKTOP / 'research/screens/normal-years-quality-watch-2026-09-28.json')['view']
assert {k:v for k,v in saved_watch.items() if k != 'id'} == view, 'Established quality-watch conditions changed'

def priced_discount(r):
    """Reconcile native cash currency, dated price, shares and supported FX."""
    v, h = r['valuation'], r['screeningAnnual']; basis = v['priceBasis']
    if v['currency'] != h['periods'][0]['currency']: return None
    if not finite(v['candidateEquity']) or v['candidateEquity'] <= 0 or not basis or not v['priceDate']: return None
    if v['priceDate'] > basis['sourceAsOf'] or basis['sourceAsOf'] > h['asOf'] or not basis['sourceId']: return None
    if (day(h['asOf']) - day(v['priceDate'])).days > 550: return None
    if not finite(basis['shares']) or basis['shares'] <= 0 or not finite(basis['close']) or basis['close'] <= 0: return None
    price = basis['shares'] * basis['close']
    if basis['method'] == 'local':
        if basis['currency'] != h['periods'][0]['currency']: return None
    elif basis['method'] == 'sek':
        if h['periods'][0]['currency'] != 'SEK' or not finite(basis['fxRate']) or basis['fxRate'] <= 0 or not basis['fxDate'] or basis['fxDate'] > v['priceDate']: return None
        price *= basis['fxRate']
    else: return None
    if not finite(price) or abs(price-v['candidateEquity']) > 1e-9*max(1,abs(price),abs(v['candidateEquity'])): return None
    normal_cash = statistics.median(normal_values(r,'fcf'))
    five_year_cash_pv = sum(normal_cash/1.1**t for t in range(1,6))
    return 100*(five_year_cash_pv/v['candidateEquity']-1)

def quality_eligible(r):
    if r['presence'] != 'latest' or not eligible(r): return False
    for c in columns:
        if 'range' not in c: continue
        x = value(r, c); lo, hi = c['range']['min'], c['range']['max']
        if not finite(x) or lo and x < float(lo) or hi and x > float(hi): return False
    return finite(value(r, columns[13])) and value(r, columns[13]) > 0

quality_rows = [r for r in gauge['rows'] if quality_eligible(r)]
previous_watch = read(DESKTOP / 'tests/fixtures/normal-year-ranking.json')
assert len(quality_rows) == 248 and {r['id'] for r in quality_rows} == set(previous_watch['orderedIds'])
raw = {r['id']: dict(discount=priced_discount(r), roce=value(r,columns[2]), marginFloor=value(r,columns[6]), cfoGrowth=value(r,columns[8]), netDebtEbitda=value(r,columns[12])) for r in quality_rows}
cohort = [r['id'] for r in quality_rows if all(finite(x) for x in raw[r['id']].values())]
assert len(cohort) == 180
assert len(set(cohort)) == len(cohort)

def oriented(id, key):
    x = raw[id][key]
    return -max(0,x) if key == 'netDebtEbitda' else x

def doubled_midrank(id, key):
    x = oriented(id,key)
    worse = sum(oriented(other,key) < x for other in cohort)
    tied = sum(oriented(other,key) == x for other in cohort)
    return 2*worse+tied-1

def percentile(id, key):
    if len(cohort) == 1: return 50.0
    return 100*doubled_midrank(id,key)/(2*(len(cohort)-1))

by_id = {}
for r in quality_rows:
    id = r['id']
    if id not in cohort:
        by_id[id] = dict(score=None,quality=None,discountRank=None,components=None,raw=raw[id],reason='Missing valid price or required quality input; no score or weight substitution.')
        continue
    components = {key:percentile(id,key) for key in RANK_KEYS}
    # Sum integer doubled positions before the final division. This is the same
    # weighted arithmetic, preserving exact mathematical ties across runtimes.
    quality_r2 = sum(doubled_midrank(id,key) for key in RANK_KEYS)
    quality = 50.0 if len(cohort) == 1 else 100*quality_r2/(8*(len(cohort)-1))
    discount = percentile(id,'discount')
    score = 50.0 if len(cohort) == 1 else 100*(6*doubled_midrank(id,'discount')+quality_r2)/(20*(len(cohort)-1))
    by_id[id] = dict(score=score,quality=quality,discountRank=discount,components=components,raw=raw[id],reason=None)
    assert all(0 <= x <= 100 for x in (quality,discount,by_id[id]['score'],*components.values()))

# The financial arithmetic above is Python only. Presentation collation alone
# matches the application's documented en/base/numeric locale policy.
collation_js = """
let input='';process.stdin.setEncoding('utf8');for await(const part of process.stdin)input+=part;
const rows=JSON.parse(input),collator=new Intl.Collator('en',{sensitivity:'base',numeric:true});
rows.sort((a,b)=>{
  if(a.score===null||b.score===null){if(a.score===null&&b.score!==null)return 1;if(b.score===null&&a.score!==null)return -1;}
  else {if(a.score!==b.score)return b.score-a.score;if(a.discount!==b.discount)return b.discount-a.discount;}
  return collator.compare(a.name,b.name)||collator.compare(a.id,b.id);
});process.stdout.write(JSON.stringify(rows.map(r=>r.id)));
"""
ordered_ids = json.loads(subprocess.run(['node','--input-type=module','-e',collation_js],input=json.dumps([dict(id=r['id'],name=r['name'],score=by_id[r['id']]['score'],discount=raw[r['id']]['discount']) for r in quality_rows]),capture_output=True,text=True,check=True).stdout)
strict_ids = [r['id'] for r in quality_rows if finite(raw[r['id']]['discount']) and value(r,columns[14]) >= 100*.3/.7 and value(r,columns[15]) >= 0]
assert strict_ids == []
prior = previous_watch
assert len(cohort) == prior['cohortSize'] and {r['id'] for r in quality_rows} == set(prior['byId'])
assert all(by_id[id]['quality'] == prior['byId'][id]['quality'] for id in cohort)
assert all(by_id[id]['discountRank'] == prior['byId'][id]['discountRank'] for id in cohort), 'Changing the horizon should preserve these flat-cash discount percentiles'
cash_factor_5 = sum(1/1.1**t for t in range(1,6))
old_half_factor = sum(1/1.1**t for t in range(1,11)) + .5/.1/1.1**10
consequences = dict(fiveYearCashFactor=cash_factor_5,previousHalfTerminalFactor=old_half_factor,
    discountPercentilesUnchanged=True,qualityPercentilesUnchanged=True,
    nonnegativeFiveYearNPV=sum(raw[id]['discount'] >= 0 for id in cohort),
    bestFiveYearNPV=max(raw[id]['discount'] for id in cohort),worstFiveYearNPV=min(raw[id]['discount'] for id in cohort),
    changedRankPositions=sum(a!=b for a,b in zip(ordered_ids,prior['orderedIds'])))
identity_data = dict(model=MODEL,gaugeSha256=gm['artifact']['sha256'],kpiIndexSha256=km['index']['sha256'],providerVariants=[v['id'] for v in variants],qualityView=saved_watch,cohortIds=sorted(cohort,key=int))
reference_identity = hashlib.sha256(json.dumps(identity_data,separators=(',',':'),sort_keys=True).encode()).hexdigest()
output = dict(version=2,model=MODEL,policyId=MODEL,assumptions=dict(discountYears=[1,2,3,4,5],discountRatePercent=10,terminalValue=0,normalCash='Median provider FCF from five selected normal reports',npvWeight=.6,qualityWeight=.4,exceptionFiscalEndYears=[2020,2021,2022,2023]),cohortSize=len(cohort),qualityCount=len(quality_rows),unrankedCount=len(quality_rows)-len(cohort),strictCount=len(strict_ids),orderedIds=ordered_ids,rankedIds=[id for id in ordered_ids if id in cohort],unrankedIds=[id for id in ordered_ids if id not in cohort],byId=by_id,referenceIdentity=reference_identity,referenceInputs=identity_data,verifiedArtifacts=verified,policyConsequences=consequences,
    notes=['User-confirmed60/40 weights; score is a relative ordering, not expected return or investment advice.','Discount input is years1–5 discounted median normal-year FCF relative to saved equity price, with no terminal value.','Listing-weighted reference; distinct listings of the same issuer may affect percentiles.','No ranking input changes with text, geography, sector, watchlist or valuation display filters.','All net cash values tie at zero debt for ranking; raw negative net debt remains visible.','Raw accounting-return outliers contribute at most one quarter of quality, one tenth of total points.','Independent financial arithmetic uses Python; standalone JS is used only for presentation collation.'])
approved = (DESKTOP / 'tests/fixtures/normal-year-five-year-ranking.json').read_bytes()
assert (json.dumps(output, ensure_ascii=False, indent=2) + '\n').encode() == approved, 'Regenerated ranking differs from the approved frozen fixture'
write('expected-ranking.json',output)
# Recompute all 20 numeric columns independently for model comparison.
write('legacy-metrics.json', dict(rows=[dict(id=r['id'], metrics={c['id']:value(r,c) for c in view['columns'] if c['id'] not in ('sector','branch','price_date','annual_date')}) for r in quality_rows]))
write('ranking-inputs.json',[dict(id=r['id'],eligible=True,**raw[r['id']]) for r in quality_rows])
write('ranking-receipt.json',dict(status='passed',model=MODEL,referenceIdentity=reference_identity,cohortSize=len(cohort),qualityCount=len(quality_rows),unrankedCount=len(quality_rows)-len(cohort),strictCount=len(strict_ids),scriptSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualityViewRecordSha256=hashlib.sha256((DESKTOP/'research/screens/normal-years-quality-watch-2026-09-28.json').read_bytes()).hexdigest(),approvedFixtureSha256=hashlib.sha256(approved).hexdigest(),expectedSha256=hashlib.sha256((OUT/'expected-ranking.json').read_bytes()).hexdigest()))
print(json.dumps(dict(status='passed',referenceIdentity=reference_identity,cohortSize=len(cohort),qualityCount=len(quality_rows),unrankedCount=len(quality_rows)-len(cohort),strictCount=len(strict_ids),firstRankedIds=ordered_ids[:10],policyConsequences=consequences),indent=2))
