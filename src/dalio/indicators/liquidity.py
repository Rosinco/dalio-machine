"""Point-in-time, non-additive diagnostics for money and market liquidity.

The module deliberately produces several separate measurements rather than a
single liquidity score.  Levels in different currencies are never added.  A
historical observation loaded from today's release remains labelled
``current_vintage_history`` until enough future releases exist for a genuine
known-at-the-time replay.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    BisGlobalLiquiditySeries,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.money_liquidity import (
    BOE_M4,
    BOE_M4EX,
    BOE_M4EX_QUARTERLY,
    BOJ_BROADLY_DEFINED_LIQUIDITY,
    BOJ_M3,
    ECB_EUROSYSTEM_ASSETS,
    ECB_M3,
    FED_M2,
    FED_TOTAL_ASSETS,
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    SCB_M3,
    MoneyLiquiditySeries,
    money_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_MMF_AGENCY_GSE_INVESTMENTS,
    OFR_MMF_BANK_RELATED_INVESTMENTS,
    OFR_MMF_OTHER_ASSET_INVESTMENTS,
    OFR_MMF_REPO_CLEARED_FICC,
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_REPO_WITH_FED,
    OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS,
    OFR_MMF_REPO_WITH_OTHER_COUNTERPARTIES,
    OFR_MMF_REPO_WITH_US_FINANCIALS,
    OFR_MMF_TOTAL_INVESTMENTS,
    OFR_MMF_TREASURY_INVESTMENTS,
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_REPO_DVP_OUTSTANDING_VOLUME,
    OFR_REPO_DVP_TRANSACTION_VOLUME,
    OFR_REPO_GCF_AVERAGE_RATE,
    OFR_REPO_GCF_OUTSTANDING_VOLUME,
    OFR_REPO_GCF_TRANSACTION_VOLUME,
    OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
    OFR_REPO_TRIPARTY_EX_FED_TRANSACTION_VOLUME,
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    OfrShadowLiquiditySeries,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.storage.db import DataRelease, DataReleaseArtifact
from dalio.storage.releases import load_vintage_panel, make_partition_key

SNAPSHOT_VERSION = 1
METHODOLOGY_VERSION = "liquidity-diagnostics-v1"
HISTORY_MODE = "current_vintage_history"


@dataclass(frozen=True)
class LiquidityParameters:
    """Frozen choices that can change a derived reading."""

    annual_growth_months: int = 12
    annual_growth_quarters: int = 4
    money_required_months: int = 16
    money_acceleration_months: int = 3
    money_summary_requires_common_period: bool = True
    central_bank_anchor_max_lag_days: int = 10
    central_bank_internal_gap_max_days: int = 14
    central_bank_gap_change_months: int = 3
    mmf_required_months: int = 16
    mmf_repo_required_months: int = 13
    repo_smoothing_observations: int = 5
    repo_baseline_observations: int = 252
    repo_required_aligned_observations: int = 257
    repo_latest_max_lag_days: int = 10
    repo_common_date_trail_max_days: int = 3
    repo_activity_window_observations: int = 20
    offshore_required_quarters: int = 6
    offshore_acceleration_quarters: int = 1
    robust_z_scale: float = 0.67448975
    robust_z_mad_floor: float = 1e-12
    robust_z_zero_mad_behavior: str = "unavailable"


PARAMETERS = LiquidityParameters()

PRIMARY_MONEY: tuple[MoneyLiquiditySeries, ...] = (
    FED_M2,
    ECB_M3,
    SCB_M3,
    BOE_M4EX,
    BOJ_BROADLY_DEFINED_LIQUIDITY,
)
CENTRAL_BANK_PAIRS: tuple[tuple[MoneyLiquiditySeries, MoneyLiquiditySeries], ...] = (
    (FED_M2, FED_TOTAL_ASSETS),
    (ECB_M3, ECB_EUROSYSTEM_ASSETS),
)
EXCLUDED_MONEY_DIAGNOSTICS: tuple[MoneyLiquiditySeries, ...] = (
    BOE_M4EX_QUARTERLY,
    BOE_M4,
    BOJ_M3,
)

MMF_ASSET_COMPONENTS: tuple[OfrShadowLiquiditySeries, ...] = (
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_TREASURY_INVESTMENTS,
    OFR_MMF_AGENCY_GSE_INVESTMENTS,
    OFR_MMF_BANK_RELATED_INVESTMENTS,
    OFR_MMF_OTHER_ASSET_INVESTMENTS,
)
MMF_REPO_COUNTERPARTIES: tuple[OfrShadowLiquiditySeries, ...] = (
    OFR_MMF_REPO_WITH_FED,
    OFR_MMF_REPO_CLEARED_FICC,
    OFR_MMF_REPO_WITH_US_FINANCIALS,
    OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS,
    OFR_MMF_REPO_WITH_OTHER_COUNTERPARTIES,
)
REPO_RATE_SERIES: tuple[OfrShadowLiquiditySeries, ...] = (
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_REPO_GCF_AVERAGE_RATE,
    OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
)
REPO_VOLUME_SERIES: tuple[OfrShadowLiquiditySeries, ...] = (
    OFR_REPO_DVP_OUTSTANDING_VOLUME,
    OFR_REPO_DVP_TRANSACTION_VOLUME,
    OFR_REPO_GCF_OUTSTANDING_VOLUME,
    OFR_REPO_GCF_TRANSACTION_VOLUME,
    OFR_REPO_TRIPARTY_EX_FED_TRANSACTION_VOLUME,
)

_ANALYSIS_MONEY = tuple(dict.fromkeys((*PRIMARY_MONEY, FED_TOTAL_ASSETS, ECB_EUROSYSTEM_ASSETS)))
_ANALYSIS_OFR = tuple(
    dict.fromkeys(
        (
            OFR_MMF_TOTAL_INVESTMENTS,
            *MMF_ASSET_COMPONENTS,
            *MMF_REPO_COUNTERPARTIES,
            *REPO_RATE_SERIES,
            *REPO_VOLUME_SERIES,
        )
    )
)

POLICY_RATE_INPUT = {
    "source_family": "FRED",
    "source": "FRED",
    "country": "US",
    "indicator": "policy_rate",
    "series_id": "DFF",
    "title": "Effective Federal Funds Rate",
    "unit": "percent",
}

FORMULAS: dict[str, str] = {
    "annual_log_growth": "100 * ln(x[t] / x[t-12m_or_4q])",
    "money_acceleration": "annual_log_growth[t] - annual_log_growth[t-3m]",
    "offshore_acceleration": "annual_log_growth[t] - annual_log_growth[t-1q]",
    "money_cb_gap": "money_annual_log_growth - central_bank_assets_annual_log_growth",
    "mmf_relative_expansion": "mmf_annual_log_growth - m2_annual_log_growth",
    "component_share": "100 * named_component / declared_parent",
    "repo_policy_premium_bp": "100 * (repo_rate_percent - effective_fed_funds_percent)",
    "repo_fragmentation_bp": "100 * (max(repo_rates) - min(repo_rates))",
    "robust_z": "0.67448975 * (current - median(previous_252)) / MAD(previous_252)",
}

INTERPRETATION_LIMITS: tuple[str, ...] = (
    "There is no composite liquidity, risk, M5, or investment score.",
    "Different currencies and unlike stocks, rates, and transaction volumes are never added.",
    "MMF-versus-M2 relative growth is not proof that bank deposits moved into money funds.",
    "MMF positions are outstanding asset stocks, not transaction-flow observations.",
    "FICC is a clearing counterparty category, not the ultimate repo borrower.",
    "Repo pricing differs by venue, collateral, maturity, and transaction mix; the EFFR comparison is not a pure credit spread.",
    "OFR repo venue aggregates do not identify an end-to-end lender-to-borrower cash path.",
    "Central-bank balance-sheet divergence does not establish causal monetary transmission.",
    "Pre-cutover history is current-vintage history, not a known-at-the-time backtest.",
)


class LiquiditySemanticsError(ValueError):
    """A stored release cannot safely be interpreted with this methodology."""


def _freshness_registry() -> dict[str, int]:
    specs = (*_ANALYSIS_MONEY, *_ANALYSIS_OFR, *BIS_GLOBAL_LIQUIDITY_SERIES)
    return {
        spec.native_series_id: int(spec.max_latest_lag_days)
        for spec in sorted(specs, key=lambda item: item.native_series_id)
    }


def methodology_sha256(parameters: LiquidityParameters = PARAMETERS) -> str:
    """Return the deterministic hash of every interpretation-changing choice."""

    payload = {
        "version": METHODOLOGY_VERSION,
        "primary_money": [spec.native_series_id for spec in PRIMARY_MONEY],
        "central_bank_pairs": [
            [money.native_series_id, assets.native_series_id]
            for money, assets in CENTRAL_BANK_PAIRS
        ],
        "excluded_money_diagnostics": [
            spec.native_series_id for spec in EXCLUDED_MONEY_DIAGNOSTICS
        ],
        "mmf_parent": OFR_MMF_TOTAL_INVESTMENTS.native_series_id,
        "mmf_components": [spec.native_series_id for spec in MMF_ASSET_COMPONENTS],
        "mmf_repo_parent": OFR_MMF_REPO_INVESTMENTS.native_series_id,
        "mmf_repo_counterparties": [spec.native_series_id for spec in MMF_REPO_COUNTERPARTIES],
        "repo_rates": [spec.native_series_id for spec in REPO_RATE_SERIES],
        "repo_volumes": [spec.native_series_id for spec in REPO_VOLUME_SERIES],
        "offshore_credit": [spec.native_series_id for spec in BIS_GLOBAL_LIQUIDITY_SERIES],
        "policy_benchmark": POLICY_RATE_INPUT,
        "formulas": FORMULAS,
        "parameters": asdict(parameters),
        "source_freshness_max_lag_days": _freshness_registry(),
        "alignment_policy": (
            "exact native calendar periods for monthly/quarterly growth; exact common "
            "business dates for repo rates; last weekly observation on or before month-end "
            "within the stated anchor tolerance for central-bank assets; no forward-fill, "
            "interpolation, or zero-fill"
        ),
        "history_mode": HISTORY_MODE,
        "limits": INTERPRETATION_LIMITS,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _partition_key(spec: Any) -> str:
    return make_partition_key(
        spec.source_family,
        spec.native_series_id,
        spec.country,
        spec.indicator,
    )


def analysis_partition_keys() -> tuple[str, ...]:
    """Return the exact, pinned release partitions used by the methodology."""

    specs = (*_ANALYSIS_MONEY, *_ANALYSIS_OFR, *BIS_GLOBAL_LIQUIDITY_SERIES)
    keys = [_partition_key(spec) for spec in specs]
    keys.append(
        make_partition_key(
            POLICY_RATE_INPUT["source_family"],
            POLICY_RATE_INPUT["series_id"],
            POLICY_RATE_INPUT["country"],
            POLICY_RATE_INPUT["indicator"],
        )
    )
    return tuple(dict.fromkeys(keys))


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_known_at must be timezone-aware")
    return value.astimezone(UTC)


def _validate_catalogue_vintages(panel: pd.DataFrame) -> None:
    expected = {
        "FED_FRED": (MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()),
        "ECB_DATA": (MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()),
        "SCB_RIKSBANK_MONEY": (
            MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()
        ),
        "BOE_IADB": (MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()),
        "BOJ_DATA": (MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()),
        "BIS_GLI": (BIS_GLI_CATALOGUE_VINTAGE_PREFIX + bis_global_liquidity_catalogue_sha256()),
        "OFR_STFM": (OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX + ofr_shadow_liquidity_catalogue_sha256()),
    }
    for source_family, exact_label in expected.items():
        family = panel.loc[panel["source_family"] == source_family]
        if family.empty:
            continue
        labels = family["vintage_label"]
        if source_family == "OFR_STFM":
            pattern = re.compile(rf"{re.escape(exact_label)};p:[0-9a-f]{{24}}")
            valid = labels.notna() & labels.astype(str).str.fullmatch(pattern.pattern)
        else:
            valid = labels.notna() & labels.astype(str).eq(exact_label)
        if not valid.all():
            raise LiquiditySemanticsError(
                f"{source_family} release catalogue does not match current semantics"
            )


def load_liquidity_panel(
    session: Session,
    *,
    as_known_at: datetime,
    through_date: date,
) -> pd.DataFrame:
    """Load complete point-in-time partitions, then cap economic observation dates."""

    cutoff = _ensure_aware(as_known_at)
    if through_date > cutoff.date():
        raise ValueError("through_date cannot be later than as_known_at")
    panel = load_vintage_panel(
        session,
        cutoff,
        partition_keys=analysis_partition_keys(),
    )
    if panel.empty:
        return panel
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.date
    panel = panel.loc[panel["date"] <= through_date].reset_index(drop=True)
    _validate_catalogue_vintages(panel)
    return panel


def _series_frame(panel: pd.DataFrame, spec: Any) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    frame = panel.loc[
        (panel["source_family"] == spec.source_family)
        & (panel["series_id"] == spec.native_series_id)
        & (panel["country"] == spec.country)
        & (panel["indicator"] == spec.indicator)
    ].copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def _policy_frame(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    frame = panel.loc[
        (panel["source_family"] == POLICY_RATE_INPUT["source_family"])
        & (panel["series_id"] == POLICY_RATE_INPUT["series_id"])
        & (panel["country"] == POLICY_RATE_INPUT["country"])
        & (panel["indicator"] == POLICY_RATE_INPUT["indicator"])
    ].copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def _unavailable(metric_id: str, family: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "unavailable",
        "missing_reason": reason,
        **extra,
    }


def _valid_positive(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    values = frame["value"].to_numpy(dtype=float)
    return bool(np.isfinite(values).all() and (values > 0).all())


def _valid_finite(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    return bool(np.isfinite(frame["value"].to_numpy(dtype=float)).all())


def _period_points(frame: pd.DataFrame, frequency: str) -> dict[pd.Period, dict[str, Any]]:
    if frame.empty:
        return {}
    points: dict[pd.Period, dict[str, Any]] = {}
    for row in frame.itertuples(index=False):
        period = pd.Period(row.date, freq=frequency)
        if period in points:
            raise LiquiditySemanticsError(f"duplicate {frequency} period {period}")
        points[period] = {
            "date": row.date,
            "value": float(row.value),
            "release_id": int(row.release_id),
            "status": str(row.status),
        }
    return points


def _consecutive(points: dict[pd.Period, Any], end: pd.Period, count: int) -> bool:
    return all(
        period in points for period in pd.period_range(end=end, periods=count, freq=end.freq)
    )


def _log_growth(current: float, prior: float) -> float:
    if not (math.isfinite(current) and math.isfinite(prior) and current > 0 and prior > 0):
        raise ValueError("log growth requires positive finite values")
    return 100.0 * math.log(current / prior)


def _movement(growth: float, acceleration: float) -> str:
    growth_word = "expanding" if growth > 0 else "contracting" if growth < 0 else "flat"
    acceleration_word = (
        "accelerating" if acceleration > 0 else "decelerating" if acceleration < 0 else "unchanged"
    )
    return f"{growth_word}_{acceleration_word}"


def _point_inputs(
    series_ids: Sequence[str],
    points: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    material = list(points)
    return {
        "input_series_ids": list(dict.fromkeys(series_ids)),
        "input_release_ids": sorted({int(point["release_id"]) for point in material}),
        "input_dates": sorted({point["date"].isoformat() for point in material}),
        "input_statuses": sorted({str(point["status"]) for point in material}),
    }


def money_impulse(
    frame: pd.DataFrame,
    spec: MoneyLiquiditySeries,
    *,
    as_of: date,
) -> dict[str, Any]:
    """Calculate exact-calendar annual log growth and three-month acceleration."""

    metric_id = f"money_impulse:{spec.country}"
    family = "broad_money_impulse"
    if frame.empty:
        return _unavailable(metric_id, family, "native series is not available")
    if not _valid_positive(frame):
        return _unavailable(metric_id, family, "series contains nonpositive or nonfinite values")
    points = _period_points(frame, "M")
    current = max(points)
    required = PARAMETERS.money_required_months
    annual_lag = PARAMETERS.annual_growth_months
    acceleration_lag = PARAMETERS.money_acceleration_months
    if not _consecutive(points, current, required):
        return _unavailable(
            metric_id,
            family,
            f"latest {required} calendar months are not complete",
        )
    if (as_of - points[current]["date"]).days > spec.max_latest_lag_days:
        return _unavailable(metric_id, family, "latest observation is stale")

    current_growth = _log_growth(points[current]["value"], points[current - annual_lag]["value"])
    prior_growth = _log_growth(
        points[current - acceleration_lag]["value"],
        points[current - acceleration_lag - annual_lag]["value"],
    )
    acceleration = current_growth - prior_growth
    used = [
        points[current],
        points[current - acceleration_lag],
        points[current - annual_lag],
        points[current - acceleration_lag - annual_lag],
    ]

    history: list[dict[str, Any]] = []
    for period in sorted(points):
        if not _consecutive(points, period, required):
            continue
        growth = _log_growth(points[period]["value"], points[period - annual_lag]["value"])
        old_growth = _log_growth(
            points[period - acceleration_lag]["value"],
            points[period - acceleration_lag - annual_lag]["value"],
        )
        history_inputs = [
            points[period],
            points[period - acceleration_lag],
            points[period - annual_lag],
            points[period - acceleration_lag - annual_lag],
        ]
        history.append(
            {
                "period": str(period),
                "date": points[period]["date"].isoformat(),
                "annual_log_growth_pct": growth,
                "acceleration_3m_pp": growth - old_growth,
                **_point_inputs([spec.native_series_id], history_inputs),
            }
        )

    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "ready",
        "country": spec.country,
        "currency": spec.currency,
        "series_id": spec.native_series_id,
        "title": spec.title,
        "period": str(current),
        "period_date": points[current]["date"].isoformat(),
        "latest_value": points[current]["value"],
        "unit": spec.unit,
        "annual_log_growth_pct": current_growth,
        "growth_3m_ago_pct": prior_growth,
        "acceleration_3m_pp": acceleration,
        "movement": _movement(current_growth, acceleration),
        "formula_ids": ["annual_log_growth", "money_acceleration"],
        "history": history,
        **_point_inputs([spec.native_series_id], used),
    }


def _month_end(period: pd.Period) -> date:
    return date(period.year, period.month, calendar.monthrange(period.year, period.month)[1])


def _weekly_anchor(
    frame: pd.DataFrame,
    target: date,
    *,
    max_lag_days: int = PARAMETERS.central_bank_anchor_max_lag_days,
) -> dict[str, Any] | None:
    eligible = frame.loc[frame["date"] <= target]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    observed = row["date"]
    if (target - observed).days > max_lag_days:
        return None
    return {
        "date": observed,
        "value": float(row["value"]),
        "release_id": int(row["release_id"]),
        "status": str(row["status"]),
    }


def money_central_bank_gap(
    money_frame: pd.DataFrame,
    assets_frame: pd.DataFrame,
    money_spec: MoneyLiquiditySeries,
    assets_spec: MoneyLiquiditySeries,
    *,
    as_of: date,
) -> dict[str, Any]:
    """Compare monthly money growth with weekly CB assets aligned at month end."""

    metric_id = f"money_cb_gap:{money_spec.country}"
    family = "money_central_bank_divergence"
    if money_frame.empty or assets_frame.empty:
        return _unavailable(metric_id, family, "money or central-bank asset series is unavailable")
    if not _valid_positive(money_frame) or not _valid_positive(assets_frame):
        return _unavailable(metric_id, family, "an input contains nonpositive or nonfinite values")
    money = _period_points(money_frame, "M")
    current = max(money)
    required = PARAMETERS.money_required_months
    annual_lag = PARAMETERS.annual_growth_months
    change_lag = PARAMETERS.central_bank_gap_change_months
    if not _consecutive(money, current, required):
        return _unavailable(
            metric_id,
            family,
            f"latest {required} money months are not complete",
        )
    if (as_of - money[current]["date"]).days > money_spec.max_latest_lag_days:
        return _unavailable(metric_id, family, "latest money observation is stale")

    weekly_dates = list(assets_frame["date"])
    if any(
        (right - left).days > PARAMETERS.central_bank_internal_gap_max_days
        for left, right in zip(weekly_dates, weekly_dates[1:], strict=False)
    ):
        return _unavailable(metric_id, family, "central-bank weekly history has an internal gap")

    def gap_at(period: pd.Period) -> tuple[float, float, float, list[dict[str, Any]]] | None:
        current_anchor = _weekly_anchor(assets_frame, _month_end(period))
        prior_anchor = _weekly_anchor(assets_frame, _month_end(period - annual_lag))
        if current_anchor is None or prior_anchor is None:
            return None
        money_growth = _log_growth(money[period]["value"], money[period - annual_lag]["value"])
        asset_growth = _log_growth(current_anchor["value"], prior_anchor["value"])
        return (
            money_growth,
            asset_growth,
            money_growth - asset_growth,
            [money[period], money[period - annual_lag], current_anchor, prior_anchor],
        )

    latest = gap_at(current)
    previous = gap_at(current - change_lag)
    if latest is None or previous is None:
        return _unavailable(metric_id, family, "a weekly month-end anchor is unavailable")
    money_growth, asset_growth, gap, used = latest
    _, _, prior_gap, prior_used = previous

    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "ready",
        "country": money_spec.country,
        "currency": money_spec.currency,
        "period": str(current),
        "period_date": money[current]["date"].isoformat(),
        "money_series_id": money_spec.native_series_id,
        "central_bank_assets_series_id": assets_spec.native_series_id,
        "money_annual_log_growth_pct": money_growth,
        "central_bank_assets_annual_log_growth_pct": asset_growth,
        "money_minus_assets_growth_gap_pp": gap,
        "gap_change_3m_pp": gap - prior_gap,
        "movement": "money_faster" if gap > 0 else "assets_faster" if gap < 0 else "equal",
        "formula_ids": ["annual_log_growth", "money_cb_gap"],
        "interpretation_limit": (
            "Growth divergence is descriptive and does not prove causal monetary transmission."
        ),
        **_point_inputs(
            [money_spec.native_series_id, assets_spec.native_series_id],
            [*used, *prior_used],
        ),
    }


def _latest_common_period(
    point_sets: Sequence[dict[pd.Period, dict[str, Any]]],
    *,
    required_history: int,
) -> pd.Period | None:
    if not point_sets or any(not points for points in point_sets):
        return None
    common = set(point_sets[0])
    for points in point_sets[1:]:
        common.intersection_update(points)
    for candidate in sorted(common, reverse=True):
        required = set(
            pd.period_range(end=candidate, periods=required_history, freq=candidate.freq)
        )
        if all(required.issubset(points) for points in point_sets):
            return candidate
    return None


def _mmf_published_repo_categories(panel: pd.DataFrame, *, as_of: date) -> dict[str, Any]:
    """Return publisher-defined counterparty/clearing ratios without a cash-flow claim."""

    specs = (OFR_MMF_REPO_INVESTMENTS, *MMF_REPO_COUNTERPARTIES)
    series_ids = [spec.native_series_id for spec in specs]

    def unavailable(reason: str) -> dict[str, Any]:
        return {
            "availability_status": "unavailable",
            "missing_reason": reason,
            "period": None,
            "ratios": [],
            "input_series_ids": series_ids,
            "input_release_ids": [],
            "input_dates": [],
            "input_statuses": [],
            "non_additive": True,
            "interpretation_limit": (
                "These are OFR-published counterparty and clearing-category ratios, not "
                "an additive liquidity total or an end-borrower cash-flow map. FICC is a "
                "clearing category."
            ),
        }

    frames = [_series_frame(panel, spec) for spec in specs]
    if any(frame.empty for frame in frames):
        return unavailable("a published repo category input is unavailable")
    if any(not _valid_positive(frame) for frame in frames):
        return unavailable("a published repo category input is nonpositive or nonfinite")
    point_sets = [_period_points(frame, "M") for frame in frames]
    required = PARAMETERS.mmf_repo_required_months
    period = _latest_common_period(point_sets, required_history=required)
    if period is None:
        return unavailable(f"{required} consecutive common calendar months are unavailable")

    repo_total = point_sets[0]
    if (as_of - repo_total[period]["date"]).days > OFR_MMF_REPO_INVESTMENTS.max_latest_lag_days:
        return unavailable("latest published repo category month is stale")

    annual_lag = PARAMETERS.annual_growth_months
    ratios: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = [repo_total[period], repo_total[period - annual_lag]]
    for spec, points in zip(MMF_REPO_COUNTERPARTIES, point_sets[1:], strict=True):
        if spec.parent_native_series_id != OFR_MMF_REPO_INVESTMENTS.native_series_id:
            raise LiquiditySemanticsError(f"{spec.native_series_id} has an unexpected parent")
        share = 100.0 * points[period]["value"] / repo_total[period]["value"]
        old_share = (
            100.0 * points[period - annual_lag]["value"] / repo_total[period - annual_lag]["value"]
        )
        ratios.append(
            {
                "series_id": spec.native_series_id,
                "title": spec.title,
                "share_of_repo_pct": share,
                "share_12m_ago_pct": old_share,
                "share_change_12m_pp": share - old_share,
                **_point_inputs(
                    [spec.native_series_id, OFR_MMF_REPO_INVESTMENTS.native_series_id],
                    [
                        points[period],
                        points[period - annual_lag],
                        repo_total[period],
                        repo_total[period - annual_lag],
                    ],
                ),
            }
        )
        used.extend([points[period], points[period - annual_lag]])

    return {
        "availability_status": "ready",
        "period": str(period),
        "period_date": repo_total[period]["date"].isoformat(),
        "ratios": ratios,
        "non_additive": True,
        "interpretation_limit": (
            "These are OFR-published counterparty and clearing-category ratios, not "
            "an additive liquidity total or an end-borrower cash-flow map. FICC is a "
            "clearing category."
        ),
        **_point_inputs(series_ids, used),
    }


def mmf_overview(panel: pd.DataFrame, *, as_of: date) -> dict[str, Any]:
    """Compare MMF and M2 growth and show explicit parent/component shares."""

    family = "us_mmf_relative_expansion_and_allocation"
    metric_id = "mmf_relative_expansion:US"
    published_categories = _mmf_published_repo_categories(panel, as_of=as_of)
    asset_specs = (OFR_MMF_TOTAL_INVESTMENTS, *MMF_ASSET_COMPONENTS)
    asset_frames = [_series_frame(panel, spec) for spec in asset_specs]
    m2_frame = _series_frame(panel, FED_M2)
    if any(frame.empty for frame in (*asset_frames, m2_frame)):
        return _unavailable(
            metric_id,
            family,
            "an MMF asset or M2 input is unavailable",
            published_repo_counterparty_categories=published_categories,
        )
    if any(not _valid_positive(frame) for frame in (*asset_frames, m2_frame)):
        return _unavailable(
            metric_id,
            family,
            "an input contains nonpositive or nonfinite values",
            published_repo_counterparty_categories=published_categories,
        )
    asset_points = [_period_points(frame, "M") for frame in asset_frames]
    m2_points = _period_points(m2_frame, "M")
    required = PARAMETERS.mmf_required_months
    annual_lag = PARAMETERS.annual_growth_months
    current = _latest_common_period([*asset_points, m2_points], required_history=required)
    if current is None:
        return _unavailable(
            metric_id,
            family,
            f"{required} consecutive common calendar months are unavailable",
            published_repo_counterparty_categories=published_categories,
        )
    total = asset_points[0]
    if (as_of - total[current]["date"]).days > OFR_MMF_TOTAL_INVESTMENTS.max_latest_lag_days:
        return _unavailable(
            metric_id,
            family,
            "latest MMF common month is stale",
            published_repo_counterparty_categories=published_categories,
        )

    mmf_growth = _log_growth(total[current]["value"], total[current - annual_lag]["value"])
    m2_growth = _log_growth(m2_points[current]["value"], m2_points[current - annual_lag]["value"])
    current_ratio = (
        100.0 * total[current]["value"] / (m2_points[current]["value"] * FED_M2.unit_multiplier)
    )
    prior_ratio = (
        100.0
        * total[current - annual_lag]["value"]
        / (m2_points[current - annual_lag]["value"] * FED_M2.unit_multiplier)
    )

    allocations: list[dict[str, Any]] = []
    used: list[dict[str, Any]] = [
        total[current],
        total[current - annual_lag],
        m2_points[current],
        m2_points[current - annual_lag],
    ]
    for spec, points in zip(MMF_ASSET_COMPONENTS, asset_points[1:], strict=True):
        if spec.parent_native_series_id != OFR_MMF_TOTAL_INVESTMENTS.native_series_id:
            raise LiquiditySemanticsError(f"{spec.native_series_id} has an unexpected parent")
        share = 100.0 * points[current]["value"] / total[current]["value"]
        old_share = (
            100.0 * points[current - annual_lag]["value"] / total[current - annual_lag]["value"]
        )
        allocations.append(
            {
                "series_id": spec.native_series_id,
                "title": spec.title,
                "value": points[current]["value"],
                "unit": spec.unit,
                "share_of_total_pct": share,
                "share_12m_ago_pct": old_share,
                "share_change_12m_pp": share - old_share,
                **_point_inputs(
                    [spec.native_series_id, OFR_MMF_TOTAL_INVESTMENTS.native_series_id],
                    [
                        points[current],
                        points[current - annual_lag],
                        total[current],
                        total[current - annual_lag],
                    ],
                ),
            }
        )
        used.extend([points[current], points[current - annual_lag]])

    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "ready",
        "period": str(current),
        "period_date": total[current]["date"].isoformat(),
        "mmf_total_value": total[current]["value"],
        "mmf_unit": OFR_MMF_TOTAL_INVESTMENTS.unit,
        "mmf_annual_log_growth_pct": mmf_growth,
        "m2_annual_log_growth_pct": m2_growth,
        "mmf_minus_m2_growth_gap_pp": mmf_growth - m2_growth,
        "mmf_assets_to_m2_scale_pct": current_ratio,
        "scale_change_12m_pp": current_ratio - prior_ratio,
        "asset_allocation": allocations,
        "published_repo_counterparty_categories": published_categories,
        "formula_ids": ["annual_log_growth", "mmf_relative_expansion", "component_share"],
        "interpretation_limit": (
            "Relative expansion is not a deposit-flow measure; M2 includes retail MMF claims, "
            "OFR covers a broader fund universe, and the adjustment bases differ."
        ),
        **_point_inputs(
            [spec.native_series_id for spec in (*asset_specs, FED_M2)],
            used,
        ),
    }


def _robust_z(values: pd.Series) -> float | None:
    clean = values.dropna()
    required = PARAMETERS.repo_baseline_observations + 1
    if len(clean) < required:
        return None
    current = float(clean.iloc[-1])
    history = clean.iloc[-required:-1]
    centre = float(history.median())
    mad = float((history - centre).abs().median())
    if not math.isfinite(mad) or mad <= PARAMETERS.robust_z_mad_floor:
        return None
    return PARAMETERS.robust_z_scale * (current - centre) / mad


def _volume_context(
    panel: pd.DataFrame,
    spec: OfrShadowLiquiditySeries,
    *,
    as_of: date,
) -> dict[str, Any]:
    frame = _series_frame(panel, spec)
    base = {
        "series_id": spec.native_series_id,
        "title": spec.title,
        "measure_kind": spec.measure_kind,
        "unit": spec.unit,
    }
    window = PARAMETERS.repo_activity_window_observations
    if frame.empty:
        return {
            **base,
            "availability_status": "unavailable",
            "missing_reason": "native volume series is unavailable",
        }
    if len(frame) < window:
        return {
            **base,
            "availability_status": "unavailable",
            "missing_reason": f"fewer than {window} observations are available",
        }
    tail = frame.tail(window)
    if not _valid_positive(tail):
        return {
            **base,
            "availability_status": "unavailable",
            "missing_reason": "context window contains nonpositive or nonfinite values",
        }
    latest_date = tail.iloc[-1]["date"]
    if (as_of - latest_date).days > spec.max_latest_lag_days:
        return {
            **base,
            "availability_status": "unavailable",
            "missing_reason": "latest volume observation is stale",
            "latest_date": latest_date.isoformat(),
        }
    return {
        **base,
        "availability_status": "ready",
        "latest_date": latest_date.isoformat(),
        "latest_value": float(tail.iloc[-1]["value"]),
        "status": str(tail.iloc[-1]["status"]),
        "trailing_observation_median": float(tail["value"].median()),
        "context_start_date": tail.iloc[0]["date"].isoformat(),
        "context_end_date": latest_date.isoformat(),
        "context_observations": window,
        "input_dates": [value.isoformat() for value in tail["date"]],
        "input_release_ids": sorted({int(value) for value in tail["release_id"]}),
        "input_statuses": sorted({str(value) for value in tail["status"]}),
    }


def repo_conditions(panel: pd.DataFrame, *, as_of: date) -> dict[str, Any]:
    """Describe policy-relative repo pricing and venue fragmentation without a verdict."""

    metric_id = "repo_conditions:US"
    family = "repo_pricing_and_activity"
    volumes = [_volume_context(panel, spec, as_of=as_of) for spec in REPO_VOLUME_SERIES]

    def unavailable(reason: str) -> dict[str, Any]:
        return _unavailable(
            metric_id,
            family,
            reason,
            volume_context=volumes,
        )

    rate_frames = [_series_frame(panel, spec) for spec in REPO_RATE_SERIES]
    policy = _policy_frame(panel)
    if any(frame.empty for frame in rate_frames) or policy.empty:
        return unavailable("a repo rate or EFFR input is unavailable")
    if any(not _valid_finite(frame) for frame in (*rate_frames, policy)):
        return unavailable("a rate input contains nonfinite values")

    maps = [dict(zip(frame["date"], frame["value"], strict=True)) for frame in rate_frames]
    policy_map = dict(zip(policy["date"], policy["value"], strict=True))
    common = set(policy_map)
    for values in maps:
        common.intersection_update(values)
    dates = sorted(common)
    required = PARAMETERS.repo_required_aligned_observations
    if len(dates) < required:
        return unavailable(f"fewer than {required} aligned business dates")
    analysis_dates = dates[-required:]
    latest = analysis_dates[-1]
    if (as_of - latest).days > PARAMETERS.repo_latest_max_lag_days:
        return unavailable("latest common repo date is stale")
    if any(
        (max(frame["date"]) - latest).days > PARAMETERS.repo_common_date_trail_max_days
        for frame in rate_frames
    ):
        return unavailable("common repo date trails a venue series")

    table = pd.DataFrame(
        {
            "date": analysis_dates,
            "effr": [policy_map[day] for day in analysis_dates],
            **{
                spec.venue: [values[day] for day in analysis_dates]
                for spec, values in zip(REPO_RATE_SERIES, maps, strict=True)
            },
        }
    ).set_index("date")
    venue_columns = [str(spec.venue) for spec in REPO_RATE_SERIES]
    premiums = table[venue_columns].sub(table["effr"], axis=0) * 100.0
    fragmentation = (table[venue_columns].max(axis=1) - table[venue_columns].min(axis=1)) * 100.0
    max_premium = premiums.max(axis=1)
    smoothing = PARAMETERS.repo_smoothing_observations
    fragmentation_5d = fragmentation.rolling(smoothing, min_periods=smoothing).median()
    max_premium_5d = max_premium.rolling(smoothing, min_periods=smoothing).median()

    venue_rows: list[dict[str, Any]] = []
    rate_release_ids: set[int] = set()
    input_statuses: set[str] = set()
    for spec, frame in zip(REPO_RATE_SERIES, rate_frames, strict=True):
        venue = str(spec.venue)
        exact = frame.loc[frame["date"] == latest].iloc[-1]
        window_rows = frame.loc[frame["date"].isin(analysis_dates)]
        rate_release_ids.update(int(value) for value in window_rows["release_id"])
        input_statuses.update(str(value) for value in window_rows["status"])
        venue_rows.append(
            {
                "venue": venue,
                "series_id": spec.native_series_id,
                "rate_pct": float(table.iloc[-1][venue]),
                "effr_premium_5d_median_bp": float(premiums[venue].tail(smoothing).median()),
                "status": str(exact["status"]),
                "release_id": int(exact["release_id"]),
            }
        )

    policy_exact = policy.loc[policy["date"] == latest].iloc[-1]
    policy_window = policy.loc[policy["date"].isin(analysis_dates)]
    rate_release_ids.update(int(value) for value in policy_window["release_id"])
    input_statuses.update(str(value) for value in policy_window["status"])
    clean_fragmentation = fragmentation_5d.dropna()
    baseline = clean_fragmentation.iloc[: PARAMETERS.repo_baseline_observations]
    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "ready",
        "period_date": latest.isoformat(),
        "effr_pct": float(table.iloc[-1]["effr"]),
        "venues": venue_rows,
        "fragmentation_5d_median_bp": float(fragmentation_5d.iloc[-1]),
        "fragmentation_robust_z": _robust_z(fragmentation_5d),
        "maximum_effr_premium_5d_median_bp": float(max_premium_5d.iloc[-1]),
        "maximum_effr_premium_robust_z": _robust_z(max_premium_5d),
        "volume_context": volumes,
        "raw_input_start_date": analysis_dates[0].isoformat(),
        "raw_input_end_date": latest.isoformat(),
        "smoothing_start_date": analysis_dates[-smoothing].isoformat(),
        "smoothing_end_date": latest.isoformat(),
        "smoothing_observations": smoothing,
        "baseline_start_date": baseline.index[0].isoformat(),
        "baseline_end_date": baseline.index[-1].isoformat(),
        "baseline_observations": PARAMETERS.repo_baseline_observations,
        "robust_z_zero_mad_behavior": PARAMETERS.robust_z_zero_mad_behavior,
        "formula_ids": ["repo_policy_premium_bp", "repo_fragmentation_bp", "robust_z"],
        "interpretation_limit": (
            "This is a pricing-and-activity diagnostic, not a repo stress score. "
            "Volume has no assumed risk direction."
        ),
        "input_series_ids": [
            *[spec.native_series_id for spec in REPO_RATE_SERIES],
            POLICY_RATE_INPUT["series_id"],
        ],
        "input_release_ids": sorted({*rate_release_ids, int(policy_exact["release_id"])}),
        "input_dates": [value.isoformat() for value in analysis_dates],
        "input_statuses": sorted(input_statuses),
    }


def offshore_credit_impulse(
    frame: pd.DataFrame,
    spec: BisGlobalLiquiditySeries,
    *,
    as_of: date,
) -> dict[str, Any]:
    """Calculate separate currency-native BIS offshore-credit growth."""

    metric_id = f"offshore_credit:{spec.currency}"
    family = "offshore_credit_growth_momentum"
    if frame.empty:
        return _unavailable(metric_id, family, "native series is not available")
    if not _valid_positive(frame):
        return _unavailable(metric_id, family, "series contains nonpositive or nonfinite values")
    points = _period_points(frame, "Q")
    current = max(points)
    required = PARAMETERS.offshore_required_quarters
    annual_lag = PARAMETERS.annual_growth_quarters
    acceleration_lag = PARAMETERS.offshore_acceleration_quarters
    if not _consecutive(points, current, required):
        return _unavailable(
            metric_id,
            family,
            f"latest {required} calendar quarters are not complete",
        )
    if (as_of - points[current]["date"]).days > spec.max_latest_lag_days:
        return _unavailable(metric_id, family, "latest observation is stale")
    growth = _log_growth(points[current]["value"], points[current - annual_lag]["value"])
    previous = _log_growth(
        points[current - acceleration_lag]["value"],
        points[current - acceleration_lag - annual_lag]["value"],
    )
    acceleration = growth - previous
    used = [
        points[current],
        points[current - acceleration_lag],
        points[current - annual_lag],
        points[current - acceleration_lag - annual_lag],
    ]

    history: list[dict[str, Any]] = []
    for period in sorted(points):
        if not _consecutive(points, period, required):
            continue
        g = _log_growth(points[period]["value"], points[period - annual_lag]["value"])
        old = _log_growth(
            points[period - acceleration_lag]["value"],
            points[period - acceleration_lag - annual_lag]["value"],
        )
        history_inputs = [
            points[period],
            points[period - acceleration_lag],
            points[period - annual_lag],
            points[period - acceleration_lag - annual_lag],
        ]
        history.append(
            {
                "period": str(period),
                "date": points[period]["date"].isoformat(),
                "annual_log_growth_pct": g,
                "acceleration_1q_pp": g - old,
                **_point_inputs([spec.native_series_id], history_inputs),
            }
        )

    return {
        "metric_id": metric_id,
        "family": family,
        "availability_status": "ready",
        "currency": spec.currency,
        "series_id": spec.native_series_id,
        "title": spec.title,
        "period": str(current),
        "period_date": points[current]["date"].isoformat(),
        "latest_value": points[current]["value"],
        "unit": spec.unit,
        "annual_log_growth_pct": growth,
        "acceleration_1q_pp": acceleration,
        "movement": _movement(growth, acceleration),
        "formula_ids": ["annual_log_growth", "offshore_acceleration"],
        "history": history,
        "interpretation_limit": "Currency-native histories remain separate and are not FX-summed.",
        **_point_inputs([spec.native_series_id], used),
    }


def _release_manifest(session: Session, release_ids: Sequence[int]) -> list[dict[str, Any]]:
    if not release_ids:
        return []
    releases = list(
        session.execute(
            select(DataRelease).where(DataRelease.id.in_(release_ids)).order_by(DataRelease.id)
        ).scalars()
    )
    artifacts = list(
        session.execute(
            select(DataReleaseArtifact)
            .where(DataReleaseArtifact.release_id.in_(release_ids))
            .order_by(DataReleaseArtifact.release_id, DataReleaseArtifact.role)
        ).scalars()
    )
    by_release: dict[int, list[DataReleaseArtifact]] = {}
    for artifact in artifacts:
        by_release.setdefault(int(artifact.release_id), []).append(artifact)

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    money_families = {spec.source_family for spec in _ANALYSIS_MONEY}

    def utc_iso(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()

    def artifact_status(
        source_family: str,
        rows: Sequence[DataReleaseArtifact],
    ) -> tuple[str, list[str]]:
        if source_family == POLICY_RATE_INPUT["source_family"]:
            return "not_required_for_legacy_benchmark", []
        expected_roles = (
            {"source_response"}
            if source_family == "BIS_GLI"
            else {"source_response", "native_series_payload", "missingness_ledger"}
            if source_family in {*money_families, "OFR_STFM"}
            else set()
        )
        failures: list[str] = []
        roles = {str(row.role) for row in rows}
        if roles != expected_roles:
            failures.append(
                "roles:expected="
                + ",".join(sorted(expected_roles))
                + ";actual="
                + ",".join(sorted(roles))
            )
        native_hashes = {str(row.native_payload_sha256) for row in rows}
        missing_hashes = {str(row.missing_provenance_sha256) for row in rows}
        hashes_by_role = {str(row.role): str(row.artifact_sha256) for row in rows}
        for row in rows:
            try:
                path = Path(str(row.artifact_path))
                if not path.is_absolute() or not path.is_file():
                    failures.append(f"{row.role}:artifact_missing")
                elif file_sha256(path) != row.artifact_sha256:
                    failures.append(f"{row.role}:artifact_hash_mismatch")
            except OSError:
                failures.append(f"{row.role}:artifact_unreadable")
            try:
                provenance = json.loads(row.provenance_json)
                canonical = json.dumps(
                    provenance,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if canonical != row.provenance_json:
                    failures.append(f"{row.role}:provenance_not_canonical")
                if (
                    hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    != row.missing_provenance_sha256
                ):
                    failures.append(f"{row.role}:provenance_hash_mismatch")
            except (TypeError, ValueError, json.JSONDecodeError):
                failures.append(f"{row.role}:provenance_invalid")
        if len(native_hashes) != 1:
            failures.append("native_payload_hash_inconsistent")
        if len(missing_hashes) != 1:
            failures.append("missing_provenance_hash_inconsistent")
        if expected_roles == {"source_response"}:
            if hashes_by_role.get("source_response") not in native_hashes:
                failures.append("source_response_native_payload_mismatch")
        elif expected_roles:
            if hashes_by_role.get("native_series_payload") not in native_hashes:
                failures.append("native_payload_artifact_mismatch")
            if hashes_by_role.get("missingness_ledger") not in missing_hashes:
                failures.append("missingness_ledger_artifact_mismatch")
        return ("valid" if not failures else "invalid"), failures

    manifest: list[dict[str, Any]] = []
    for release in releases:
        release_artifacts = by_release.get(int(release.id), [])
        status, failures = artifact_status(release.source_family, release_artifacts)
        manifest.append(
            {
                "release_id": int(release.id),
                "partition_key": release.partition_key,
                "source_family": release.source_family,
                "published_at": utc_iso(release.published_at),
                "available_at": utc_iso(release.available_at),
                "retrieved_at": utc_iso(release.retrieved_at),
                "vintage_label": release.vintage_label,
                "source_url": release.source_url,
                "content_sha256": release.content_sha256,
                "row_count": int(release.row_count),
                "artifact_manifest_status": status,
                "artifact_manifest_failures": failures,
                "artifacts": [
                    {
                        "role": artifact.role,
                        "sha256": artifact.artifact_sha256,
                        "path": artifact.artifact_path,
                    }
                    for artifact in release_artifacts
                ],
            }
        )
    return manifest


def build_liquidity_overview(
    panel: pd.DataFrame,
    *,
    as_of: date,
    as_known_at: datetime,
    input_releases: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build the JSON-serialisable, unscored liquidity overview from a PIT panel."""

    known = _ensure_aware(as_known_at)
    money = [money_impulse(_series_frame(panel, spec), spec, as_of=as_of) for spec in PRIMARY_MONEY]
    ready_money = [row for row in money if row["availability_status"] == "ready"]
    money_periods = sorted({str(row["period"]) for row in ready_money})
    common_money_period = (
        money_periods[0]
        if len(ready_money) == len(PRIMARY_MONEY) and len(money_periods) == 1
        else None
    )
    money_summary: dict[str, Any] = {
        "ready": len(ready_money),
        "expected": len(PRIMARY_MONEY),
        "period_basis": (
            "common"
            if common_money_period is not None
            else "mixed"
            if len(ready_money) == len(PRIMARY_MONEY)
            else "incomplete"
        ),
        "common_period": common_money_period,
        "constituent_periods": {
            str(row.get("country", row["metric_id"])): row.get("period") for row in money
        },
        "median_annual_log_growth_pct": None,
        "positive_growth_breadth": None,
        "accelerating_breadth": None,
        "interpretation_limit": (
            "The equal-country median and breadth are descriptive; they are not a "
            "currency-weighted global money aggregate."
        ),
    }
    if common_money_period is not None:
        money_summary.update(
            {
                "median_annual_log_growth_pct": median(
                    row["annual_log_growth_pct"] for row in ready_money
                ),
                "positive_growth_breadth": sum(
                    row["annual_log_growth_pct"] > 0 for row in ready_money
                ),
                "accelerating_breadth": sum(row["acceleration_3m_pp"] > 0 for row in ready_money),
            }
        )

    central_bank = [
        money_central_bank_gap(
            _series_frame(panel, money_spec),
            _series_frame(panel, assets_spec),
            money_spec,
            assets_spec,
            as_of=as_of,
        )
        for money_spec, assets_spec in CENTRAL_BANK_PAIRS
    ]
    mmf = mmf_overview(panel, as_of=as_of)
    repo = repo_conditions(panel, as_of=as_of)
    offshore = [
        offshore_credit_impulse(_series_frame(panel, spec), spec, as_of=as_of)
        for spec in BIS_GLOBAL_LIQUIDITY_SERIES
    ]

    selected_releases = sorted(input_releases, key=lambda row: int(row["release_id"]))
    liquidity_releases = [
        row
        for row in selected_releases
        if row["source_family"] != POLICY_RATE_INPUT["source_family"]
    ]
    earliest_input = min(
        (row["available_at"] for row in liquidity_releases),
        default=None,
    )
    selected_partitions = {str(row["partition_key"]) for row in selected_releases}
    expected_partitions = set(analysis_partition_keys())
    complete_snapshot = (
        max((row["available_at"] for row in liquidity_releases), default=None)
        if selected_partitions == expected_partitions
        else None
    )
    mmf_categories = mmf.get("published_repo_counterparty_categories", {})
    volume_context = repo.get("volume_context", [])
    coverage = {
        "broad_money": {
            "ready": len(ready_money),
            "expected": len(PRIMARY_MONEY),
        },
        "central_bank_pairs": {
            "ready": sum(row["availability_status"] == "ready" for row in central_bank),
            "expected": len(CENTRAL_BANK_PAIRS),
        },
        "mmf_headline_and_allocation": {
            "ready": int(mmf["availability_status"] == "ready"),
            "expected": 1,
        },
        "mmf_published_repo_categories": {
            "ready": int(mmf_categories.get("availability_status") == "ready"),
            "expected": 1,
        },
        "repo_pricing": {
            "ready": int(repo["availability_status"] == "ready"),
            "expected": 1,
        },
        "repo_activity_context": {
            "ready": sum(row.get("availability_status") == "ready" for row in volume_context),
            "expected": len(REPO_VOLUME_SERIES),
        },
        "offshore_credit": {
            "ready": sum(row["availability_status"] == "ready" for row in offshore),
            "expected": len(BIS_GLOBAL_LIQUIDITY_SERIES),
        },
    }

    overview: dict[str, Any] = {
        "version": SNAPSHOT_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "methodology_sha256": methodology_sha256(),
        "as_of": as_of.isoformat(),
        "as_known_at": known.isoformat(),
        "data_mode": "release_ledger_point_in_time",
        "history_mode": HISTORY_MODE,
        "history_basis": "reported_in_selected_release",
        "earliest_input_available_at": earliest_input,
        "complete_snapshot_available_at": complete_snapshot,
        "formulas": FORMULAS,
        "parameters": {
            **asdict(PARAMETERS),
            "source_freshness_max_lag_days": _freshness_registry(),
        },
        "source_catalogue_sha256": {
            "money_liquidity": money_liquidity_catalogue_sha256(),
            "bis_global_liquidity": bis_global_liquidity_catalogue_sha256(),
            "ofr_shadow_liquidity": ofr_shadow_liquidity_catalogue_sha256(),
        },
        "excluded_from_primary_money": [
            {
                "series_id": spec.native_series_id,
                "title": spec.title,
                "reason": spec.definition_notes,
            }
            for spec in EXCLUDED_MONEY_DIAGNOSTICS
        ],
        "coverage": coverage,
        "money_summary": money_summary,
        "broad_money": money,
        "central_bank_divergence": central_bank,
        "mmf": mmf,
        "repo": repo,
        "offshore_credit": offshore,
        "horizons": [
            {
                "horizon": "0-12 months",
                "supported_context": "repo pricing/fragmentation and MMF allocation conditions",
            },
            {
                "horizon": "1-3 years",
                "supported_context": "broad-money, central-bank, MMF, and offshore-credit momentum",
            },
            {
                "horizon": "3-5 years and longer",
                "supported_context": (
                    "not assessed by this liquidity layer alone; combine with debt, pensions, "
                    "demographics, productivity, fiscal policy, and reviewed institutional reports"
                ),
            },
        ],
        "interpretation_limits": list(INTERPRETATION_LIMITS),
        "input_releases": selected_releases,
        "evidence_integrity": {
            "release_count": len(selected_releases),
            "artifact_manifest_valid": sum(
                row["artifact_manifest_status"] == "valid" for row in selected_releases
            ),
            "legacy_benchmark_without_raw_manifest": sum(
                row["artifact_manifest_status"] == "not_required_for_legacy_benchmark"
                for row in selected_releases
            ),
            "artifact_manifest_invalid": sum(
                row["artifact_manifest_status"] == "invalid" for row in selected_releases
            ),
        },
    }
    fingerprint_payload = json.dumps(
        overview,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    overview["snapshot_sha256"] = hashlib.sha256(fingerprint_payload).hexdigest()
    validate_liquidity_snapshot(overview)
    return overview


def validate_liquidity_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail closed when the generated contract contains unsafe or non-JSON data."""

    required = {
        "version",
        "methodology_version",
        "methodology_sha256",
        "as_of",
        "as_known_at",
        "data_mode",
        "history_mode",
        "history_basis",
        "earliest_input_available_at",
        "complete_snapshot_available_at",
        "formulas",
        "parameters",
        "source_catalogue_sha256",
        "coverage",
        "money_summary",
        "broad_money",
        "central_bank_divergence",
        "mmf",
        "repo",
        "offshore_credit",
        "horizons",
        "interpretation_limits",
        "input_releases",
        "evidence_integrity",
        "snapshot_sha256",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise LiquiditySemanticsError(f"liquidity snapshot missing fields: {missing}")
    if snapshot["version"] != SNAPSHOT_VERSION:
        raise LiquiditySemanticsError(
            f"unsupported liquidity snapshot version: {snapshot['version']!r}"
        )
    if snapshot["methodology_version"] != METHODOLOGY_VERSION:
        raise LiquiditySemanticsError(
            f"unsupported liquidity methodology: {snapshot['methodology_version']!r}"
        )
    if snapshot["methodology_sha256"] != methodology_sha256():
        raise LiquiditySemanticsError(
            "liquidity methodology hash does not match the implementation"
        )
    expected_parameters = {
        **asdict(PARAMETERS),
        "source_freshness_max_lag_days": _freshness_registry(),
    }
    if snapshot["parameters"] != expected_parameters:
        raise LiquiditySemanticsError("liquidity parameters do not match the methodology")
    if snapshot["formulas"] != FORMULAS:
        raise LiquiditySemanticsError("liquidity formulas do not match the methodology")
    if snapshot["history_mode"] != HISTORY_MODE:
        raise LiquiditySemanticsError("liquidity history mode does not match the methodology")
    if snapshot["data_mode"] != "release_ledger_point_in_time":
        raise LiquiditySemanticsError("liquidity snapshot is not release-ledger point-in-time data")

    try:
        as_of = date.fromisoformat(str(snapshot["as_of"]))
        known_at = datetime.fromisoformat(str(snapshot["as_known_at"]))
    except ValueError as exc:
        raise LiquiditySemanticsError("liquidity snapshot has an invalid cutoff clock") from exc
    if known_at.tzinfo is None or known_at.utcoffset() is None or as_of > known_at.date():
        raise LiquiditySemanticsError("liquidity snapshot cutoff clocks are inconsistent")

    for name, coverage in snapshot["coverage"].items():
        if (
            not isinstance(coverage, dict)
            or not isinstance(coverage.get("ready"), int)
            or not isinstance(coverage.get("expected"), int)
            or coverage["ready"] < 0
            or coverage["expected"] < 1
            or coverage["ready"] > coverage["expected"]
        ):
            raise LiquiditySemanticsError(f"invalid liquidity coverage for {name}")

    forbidden = {"score", "overall_score", "composite_score", "global_liquidity_total"}

    def inspect_keys(value: Any, location: str = "snapshot") -> None:
        if isinstance(value, dict):
            unsafe = forbidden & set(value)
            if unsafe:
                raise LiquiditySemanticsError(
                    f"{location} contains forbidden aggregate field(s): {sorted(unsafe)}"
                )
            for key, child in value.items():
                inspect_keys(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect_keys(child, f"{location}[{index}]")

    inspect_keys(snapshot)
    try:
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise LiquiditySemanticsError("liquidity snapshot is not finite canonical JSON") from exc

    supplied_hash = snapshot["snapshot_sha256"]
    if not isinstance(supplied_hash, str) or re.fullmatch(r"[0-9a-f]{64}", supplied_hash) is None:
        raise LiquiditySemanticsError("liquidity snapshot hash is not a lowercase SHA-256 digest")
    unhashed = dict(snapshot)
    del unhashed["snapshot_sha256"]
    canonical = json.dumps(
        unhashed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if supplied_hash != expected_hash:
        raise LiquiditySemanticsError("liquidity snapshot hash does not match its contents")


def build_liquidity_snapshot(
    session: Session,
    *,
    as_of: date,
    as_known_at: datetime,
) -> dict[str, Any]:
    """Load the PIT panel and build a release- and artifact-traceable overview."""

    panel = load_liquidity_panel(session, as_known_at=as_known_at, through_date=as_of)
    release_ids = sorted({int(value) for value in panel.get("release_id", [])})
    releases = _release_manifest(session, release_ids)
    invalid = [
        release
        for release in releases
        if release["source_family"] != POLICY_RATE_INPUT["source_family"]
        and release["artifact_manifest_status"] != "valid"
    ]
    if invalid:
        details = ", ".join(
            f"{release['release_id']}:{release['artifact_manifest_failures']}"
            for release in invalid
        )
        raise LiquiditySemanticsError(f"input artifact manifest validation failed: {details}")
    return build_liquidity_overview(
        panel,
        as_of=as_of,
        as_known_at=as_known_at,
        input_releases=releases,
    )


def methodology_record() -> dict[str, Any]:
    """Expose the frozen formula and input registry for documentation/tests."""

    def json_safe(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(child) for child in value]
        return value

    return {
        "version": METHODOLOGY_VERSION,
        "sha256": methodology_sha256(),
        "formulas": dict(FORMULAS),
        "parameters": {
            **asdict(PARAMETERS),
            "source_freshness_max_lag_days": _freshness_registry(),
        },
        "primary_money": [json_safe(asdict(spec)) for spec in PRIMARY_MONEY],
        "central_bank_pairs": [
            [money.native_series_id, assets.native_series_id]
            for money, assets in CENTRAL_BANK_PAIRS
        ],
        "repo_rates": [spec.native_series_id for spec in REPO_RATE_SERIES],
        "repo_volumes": [spec.native_series_id for spec in REPO_VOLUME_SERIES],
        "offshore_credit": [spec.native_series_id for spec in BIS_GLOBAL_LIQUIDITY_SERIES],
        "source_catalogue_sha256": {
            "money_liquidity": money_liquidity_catalogue_sha256(),
            "bis_global_liquidity": bis_global_liquidity_catalogue_sha256(),
            "ofr_shadow_liquidity": ofr_shadow_liquidity_catalogue_sha256(),
        },
        "limits": list(INTERPRETATION_LIMITS),
    }
