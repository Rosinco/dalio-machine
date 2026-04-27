"""Home-currency overlay for allocation tilts.

A Swedish investor sees SEK returns, not USD. A +0.5 gold tilt has a
different magnitude in SEK than USD because SEK/USD itself moves. This
module produces a small additive overlay that biases away from foreign
nominal assets when the home currency is likely to appreciate against
USD (home real rates > USD real rates), and toward them when the home
currency is likely to depreciate.

Honest scope: this is interest-rate-parity intuition, not a full FX
model. Real return differentials drive *some* of currency moves over
multi-year horizons; trade balances, capital flows, and risk-off
dynamics drive the rest. The overlay magnitudes stay small (≤0.3) so
the lens errs toward humility about FX prediction.
"""
from __future__ import annotations


def home_currency_overlay(
    home_currency: str,
    home_real_rate_10y: float | None,
    target_real_rate_10y: float | None,
) -> dict[str, float]:
    """Return a small additive tilt overlay anchored to the home-vs-target
    real-rate differential.

    When `home_real - target_real > 0` (home rates strictly higher), the
    home currency is more attractive — likely to appreciate vs target.
    USD-denominated assets (gold, US equities) get a small tilt-down.
    Conversely when home rates are lower, foreign nominal assets get a
    small tilt-up.

    Returns {} when home_currency is "USD" (no translation), or when
    either real rate is missing (no signal).
    """
    if home_currency == "USD":
        return {}
    if home_real_rate_10y is None or target_real_rate_10y is None:
        return {}

    diff = home_real_rate_10y - target_real_rate_10y
    # Saturate at ±2pp differential — beyond that the IRP intuition is
    # already maxed and we don't want the overlay to dominate.
    intensity = max(-1.0, min(1.0, diff / 2.0))

    # Negative intensity (home rate lower than target) → foreign assets up.
    return {
        "gold": -0.3 * intensity,           # gold is USD-denominated
        "equities": -0.15 * intensity,      # broad equities are USD-heavy
        "long_bonds": -0.2 * intensity,     # long nominal bonds are USD-heavy
        "commodities": -0.15 * intensity,   # commodities are USD-priced
    }
