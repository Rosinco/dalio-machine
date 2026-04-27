"""Allocation-tilt mapper.

Translates regime classifications (short-term stage + long-term phase) into
*tilts* away from a default diversified base portfolio. The output is
deliberately framed as deviations, not absolute weights, because:

  1. Macro-overlay strategies have weak academic evidence vs. simple passive
     diversification (AQR, Vanguard, etc.). Tilting from a base is honest;
     "build me a portfolio from scratch" is not.

  2. Confidence in cycle classification is bounded — saturating confidence
     formulas in both classifiers cap at <100% even on strong signals. Tilts
     should reflect that uncertainty: small magnitudes when confidence is low.

The mapping is built from Dalio's two key frameworks:

  * **Growth × Inflation matrix** (drives short-term tilts):
      - Stage 1 Expansion        → Goldilocks (G↑ I→)         → equities, credit
      - Stage 2 Inflationary peak → Overheating (G↑ I↑)        → commodities, gold, TIPS
      - Stage 3 Recession        → Deflationary (G↓ I↓)       → long bonds, cash
      - Stage 4 Reflation        → Recovery+rising-I (G↓→↑ I↑)→ gold, mixed

  * **Long-term phase risk overlay** (drives long-term tilts):
      - Phase 3 Bubble:          defensive prep (cash, gold; avoid leverage)
      - Phase 4 Top:              very defensive (gold, cash; avoid equity/credit)
      - Phase 5 Ugly deleveraging: deflationary distress (long bonds, gold; avoid credit)
      - Phase 6 Beautiful deleveraging: inflation hedge (gold, TIPS, real assets;
                                       AVOID long nominal bonds — savers lose)

Final tilt = short-term tilt + (long-term tilt × long_confidence). Confidences
weight the long-term overlay since long-term phase classifications have more
inherent uncertainty (slow-moving regimes, fewer indicators).
"""
from __future__ import annotations

from dataclasses import dataclass

from dalio.scoring.long_term import PhaseClassification
from dalio.scoring.short_term import Classification

# Asset classes covered. Order matters for stable display.
ASSET_CLASSES: tuple[str, ...] = (
    "equities",
    "long_bonds",
    "short_bonds",
    "credit",
    "tips",
    "gold",
    "commodities",
    "real_estate",
)

ASSET_LABELS: dict[str, str] = {
    "equities": "Equities (broad market)",
    "long_bonds": "Long-duration govt bonds (10y+)",
    "short_bonds": "Short-duration govt bonds (1–3y) / cash equivalents",
    "credit": "Investment-grade corporate bonds",
    "tips": "Inflation-linked bonds (TIPS)",
    "gold": "Gold",
    "commodities": "Broad commodities",
    "real_estate": "Real estate / REITs",
}


# ─── Tilt tables ─────────────────────────────────────────────────────────

# Short-term stage → asset tilts (units: ~ -2 to +2). Slice 15: derived from
# the 2×2 growth × inflation grid in `scoring/grid.py`. Stages 1–3 map to a
# single quadrant each; Stage 4 (Reflation) is a documented hybrid kept as a
# fifth canonical entry in `GRID_TILTS` because CB-pivot dynamics don't fit a
# single quadrant cleanly. See grid.py module docstring for the full argument.
from dalio.scoring.grid import GRID_TILTS, STAGE_TO_QUADRANT  # noqa: E402

SHORT_TERM_TILTS: dict[int, dict[str, float]] = {
    stage: dict(GRID_TILTS[quadrant])
    for stage, quadrant in STAGE_TO_QUADRANT.items()
} | {0: {}}  # Transition / insufficient data


# Long-term phase → risk overlay. These are added on top of short-term.
LONG_TERM_TILTS: dict[int, dict[str, float]] = {
    1: {},  # Sound money — neutral overlay
    2: {},  # Debt outpaces income — still benign
    3: {  # Bubble — be cautious
        "gold": +0.5,
        "short_bonds": +0.3,
        "equities": -0.3,
        "credit": -0.3,
    },
    4: {  # Top — very defensive
        "gold": +1.0,
        "short_bonds": +0.7,
        "tips": +0.3,
        "equities": -0.7,
        "credit": -0.7,
        "real_estate": -0.3,
        "long_bonds": -0.3,
    },
    5: {  # Ugly deleveraging — deflationary distress
        "long_bonds": +1.5,
        "short_bonds": +1.0,
        "gold": +0.7,
        "credit": -1.5,
        "equities": -1.0,
        "real_estate": -1.2,
        "commodities": -0.5,
    },
    6: {  # Beautiful deleveraging / financial repression
        "gold": +1.2,
        "tips": +1.2,
        "real_estate": +0.7,
        "commodities": +0.5,
        "equities": +0.3,
        "long_bonds": -1.2,
        "short_bonds": -0.5,
    },
    7: {},  # Reset — out of model
    0: {},  # Transition / insufficient data
}


# Direction symbols for tilt magnitudes. Thresholds chosen so that small
# rule-table values still register as a clear arrow.
def _direction(tilt: float) -> str:
    if tilt >= 1.5:
        return "↑↑↑"
    if tilt >= 0.7:
        return "↑↑"
    if tilt >= 0.2:
        return "↑"
    if tilt <= -1.5:
        return "↓↓↓"
    if tilt <= -0.7:
        return "↓↓"
    if tilt <= -0.2:
        return "↓"
    return "→"


# ─── Public types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AssetTilt:
    asset_class: str
    label: str
    tilt: float
    direction: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AllocationView:
    country: str
    short_term_stage: int
    short_term_label: str
    long_term_phase: int
    long_term_label: str
    tilts: tuple[AssetTilt, ...]
    caution_level: str        # "low" | "moderate" | "elevated" | "high"
    real_yield_regime: str    # "repression" | "mild repression" | "neutral" | "savers compensated" | "unknown"
    home_currency: str        # "USD" | "SEK" | "EUR" | etc.
    summary: str


# ─── Computation ─────────────────────────────────────────────────────────


def _caution_from_phase(phase: int) -> str:
    """Translate long-term phase to a single caution level."""
    return {
        1: "low",
        2: "low",
        3: "moderate",
        4: "elevated",
        5: "high",
        6: "elevated",
        7: "high",
        0: "moderate",
    }.get(phase, "moderate")


def _real_yield_multiplier(real_rate: float | None) -> dict[str, float]:
    """Cross-regime tilt overlay driven by the real 10y yield.

    Negative real rates are a financial-repression regime: nominal savers
    are punished, capital flows to scarce real assets (gold, commodities,
    real estate) and away from long-duration nominal bonds — across all
    phases, not just Phase 6. This is additive on top of the phase-specific
    tilts.

    Magnitudes deliberately small (≤0.4 at the deepest negative rate)
    because Phase 6 (beautiful deleveraging) already votes on negative real
    rates. The multiplier amplifies but does not dominate.

    Tapered linearly between rr=0 and rr=-1; full intensity at rr ≤ -1.
    """
    if real_rate is None or real_rate >= 0:
        return {}
    intensity = 1.0 if real_rate <= -1 else -real_rate
    return {
        "gold": +0.4 * intensity,
        "commodities": +0.3 * intensity,
        "real_estate": +0.2 * intensity,
        "long_bonds": -0.4 * intensity,
    }


def _real_yield_regime(real_rate: float | None) -> str:
    """Plain-language label for the real-yield regime."""
    if real_rate is None:
        return "unknown"
    if real_rate < -1:
        return "repression"
    if real_rate < 0:
        return "mild repression"
    if real_rate < 1:
        return "neutral"
    return "savers compensated"


def _summary_for(short_label: str, long_label: str, caution: str) -> str:
    return (
        f"Short-term: {short_label.lower()}; long-term: {long_label.lower()}. "
        f"Caution level: {caution}. Tilts below are deviations from a default "
        f"diversified base, not absolute weights — and not a market-timing signal."
    )


def _resolve_tilts(
    stage: int, votes: tuple, table: dict[int, dict[str, float]],
) -> dict[str, float]:
    """Resolve a tilt dict for a stage. For transition states (stage=0) blend
    by vote weights of the contributing stages — gives partial signal instead
    of "no tilt" when classifier is on a stage boundary.
    """
    if stage != 0:
        return dict(table.get(stage, {}))
    if not votes:
        return {}
    weights_by_stage: dict[int, float] = {}
    for v in votes:
        s = v.stage if hasattr(v, "stage") else v.phase
        weights_by_stage[s] = weights_by_stage.get(s, 0.0) + v.weight
    total = sum(weights_by_stage.values())
    if total == 0:
        return {}
    blended: dict[str, float] = {}
    for s, w in weights_by_stage.items():
        share = w / total
        for asset, tilt in table.get(s, {}).items():
            blended[asset] = blended.get(asset, 0.0) + tilt * share
    return blended


def compute_tilts(
    short_term: Classification,
    long_term: PhaseClassification,
    *,
    home_currency: str = "USD",
    home_real_rate_10y: float | None = None,
) -> AllocationView:
    """Combine short-term + long-term classifications into asset-class tilts.

    Transition states (stage 0) blend their contributing-stage tilts by vote
    weight rather than emitting nothing — gives partial signal at boundaries.

    Both layers are weighted by their own classifier confidence: low-confidence
    classifications produce small tilts. This is honest given typical 25–60%
    confidence figures from the saturating-confidence formulas.

    `home_currency` and `home_real_rate_10y` (Slice 14) drive a small
    interest-rate-parity overlay: when the home currency's real rate is
    higher than the target country's, USD-denominated foreign assets get a
    small tilt-down because the home currency is likely to appreciate.
    `compute_tilts` stays DB-free — the dashboard does the home-country
    real-rate lookup outside and threads the value in.
    """
    from dalio.scoring.currency import home_currency_overlay

    st_tilts_raw = _resolve_tilts(short_term.stage, short_term.votes, SHORT_TERM_TILTS)
    lt_tilts_raw = _resolve_tilts(long_term.phase, long_term.votes, LONG_TERM_TILTS)
    st_weight = max(short_term.confidence, 0.0)
    lt_weight = max(long_term.confidence, 0.0)
    st_tilts = {k: v * st_weight for k, v in st_tilts_raw.items()}
    lt_tilts = {k: v * lt_weight for k, v in lt_tilts_raw.items()}

    # Real-yield overlay: third additive layer. The rate itself IS the signal,
    # so this is NOT confidence-weighted by either classifier.
    real_rate = long_term.features.real_rate_10y
    ry_tilts = _real_yield_multiplier(real_rate)
    ry_regime = _real_yield_regime(real_rate)

    # Home-currency overlay: fourth additive layer. Small magnitude (≤0.3),
    # only fires when home_currency != "USD" and both real rates are present.
    cur_tilts = home_currency_overlay(home_currency, home_real_rate_10y, real_rate)
    fx_diff = (
        home_real_rate_10y - real_rate
        if (home_real_rate_10y is not None and real_rate is not None)
        else None
    )

    asset_tilts: list[AssetTilt] = []
    for asset in ASSET_CLASSES:
        st = st_tilts.get(asset, 0.0)
        lt = lt_tilts.get(asset, 0.0)
        ry = ry_tilts.get(asset, 0.0)
        cur = cur_tilts.get(asset, 0.0)
        total = st + lt + ry + cur
        reasons: list[str] = []
        if abs(st_tilts_raw.get(asset, 0.0)) > 0.01:
            raw = st_tilts_raw[asset]
            reasons.append(
                f"Short-term ({short_term.stage_label}, conf {short_term.confidence:.0%}): "
                f"{raw:+.2f} × {st_weight:.2f} = {st:+.2f}"
            )
        if abs(lt_tilts_raw.get(asset, 0.0)) > 0.01:
            raw = lt_tilts_raw[asset]
            reasons.append(
                f"Long-term ({long_term.phase_label}, conf {long_term.confidence:.0%}): "
                f"{raw:+.2f} × {lt_weight:.2f} = {lt:+.2f}"
            )
        if abs(ry) > 0.01:
            assert real_rate is not None  # guaranteed by ry_tilts non-empty
            reasons.append(
                f"Real-rate regime ({ry_regime}, rr {real_rate:+.1f}%): {ry:+.2f}"
            )
        if abs(cur) > 0.01:
            assert fx_diff is not None  # guaranteed by cur_tilts non-empty
            reasons.append(
                f"{home_currency} overlay (Δreal-rate {fx_diff:+.1f}pp vs target): {cur:+.2f}"
            )
        asset_tilts.append(AssetTilt(
            asset_class=asset,
            label=ASSET_LABELS[asset],
            tilt=total,
            direction=_direction(total),
            reasons=tuple(reasons),
        ))

    caution = _caution_from_phase(long_term.phase)
    summary = _summary_for(short_term.stage_label, long_term.phase_label, caution)

    return AllocationView(
        country=short_term.country,
        short_term_stage=short_term.stage,
        short_term_label=short_term.stage_label,
        long_term_phase=long_term.phase,
        long_term_label=long_term.phase_label,
        tilts=tuple(asset_tilts),
        caution_level=caution,
        real_yield_regime=ry_regime,
        home_currency=home_currency,
        summary=summary,
    )
