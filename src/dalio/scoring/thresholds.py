"""Per-country classifier thresholds.

Default values are US-derived literals from the original Dalio framework
implementations. Per-country overrides come from each country's own historical
quantiles when ≥10 years of data are available — see calibration.py.

Honest tension worth acknowledging: Dalio's original thresholds are absolute
pain points (DSR > 22 historically signals distress somewhere). Per-country
quantiles soften that into a relative-pain framework, which is more
defensible across structurally-different economies (Japan's permanent ~357%
debt is not "extreme" by Japan's history, only by US-anchored intuition) but
loses some of the original signal's universality. The dashboard surfaces both
the default and the per-country values side-by-side so the delta itself is a
credibility signal.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """Distribution-relative thresholds used by the rule classifiers.

    Only the fields that are genuinely distribution-relative are exposed
    here. Universal physics (negative real rate = financial repression,
    yield-curve inversion, Sahm rule 0.5pp) stays as literals in the
    classifiers.
    """
    # Total credit / GDP — country-relative debt ceilings
    debt_late_cycle_low: float = 220.0   # bubble-zone lower bound (q75)
    debt_extreme: float = 280.0          # peak-debt zone (q95)

    # Debt service ratio (private non-financial)
    dsr_stretched: float = 17.0          # q75
    dsr_distress: float = 18.0           # q90
    dsr_extreme: float = 22.0            # q95

    # CPI YoY — what counts as "elevated" / "peak" for THIS country
    cpi_elevated: float = 3.0            # q75 with floor 2.5
    cpi_peak: float = 4.0                # q90 with floor 3.5


DEFAULT_THRESHOLDS = Thresholds()
