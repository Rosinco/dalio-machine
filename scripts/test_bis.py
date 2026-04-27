"""Smoke-test BIS adapter against the live API."""
from dalio.data_sources.bis import (
    TIER_1_DSR,
    TIER_1_TOTAL_CREDIT,
    BisSource,
)

src = BisSource()
print(f"Total Credit specs: {len(TIER_1_TOTAL_CREDIT)}")
print(f"DSR specs:          {len(TIER_1_DSR)}")
print()

print("--- US Total Credit ---")
for spec in TIER_1_TOTAL_CREDIT:
    if spec.country != "US":
        continue
    try:
        df = src.fetch_total_credit(spec, use_cache=False)
        if df.empty:
            print(f"  ⚠ {spec.indicator:<22} sector={spec.sector}  empty")
            continue
        latest = df.iloc[-1]
        print(f"  ✓ {spec.indicator:<22} sector={spec.sector}  latest={latest['date']}={latest['value']:.1f}%")
    except Exception as e:
        print(f"  ✗ {spec.indicator:<22} sector={spec.sector}  ERROR: {e}")

print()
print("--- US DSR ---")
for spec in TIER_1_DSR:
    if spec.country != "US":
        continue
    try:
        df = src.fetch_dsr(spec, use_cache=False)
        if df.empty:
            print(f"  ⚠ {spec.indicator:<22}  empty")
            continue
        latest = df.iloc[-1]
        print(f"  ✓ {spec.indicator:<22}  latest={latest['date']}={latest['value']:.2f}%")
    except Exception as e:
        print(f"  ✗ {spec.indicator:<22}  ERROR: {e}")
