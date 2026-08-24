"""Snapshot loader — the single coupling point between data and presentation.

`dalio-score` writes ``data/snapshots/fundamentals_latest.json``; this module
parses it into typed frames and VALIDATES the contract, raising
``SnapshotError`` naming the missing field. Streamlit-free so it is testable
and reusable from scripts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

SUPPORTED_VERSIONS = (1,)
LATEST_NAME = "fundamentals_latest.json"

_REQUIRED_TOP = ("version", "as_of", "ranking_population", "indicators", "categories",
                 "views", "countries", "coverage")
_REQUIRED_COUNTRY = ("name", "iso3", "tier", "eu_member", "members", "on_map", "fx_regime",
                     "sanctioned", "data_quality", "indicators", "categories", "views",
                     "pressures", "history")
_REQUIRED_CELL = ("value", "date", "source", "pct", "trend", "trend_5y", "uncertainty")
_REQUIRED_CATEGORY = ("score", "n_available", "n_total", "distance_to_best", "best_iso2")


class SnapshotError(ValueError):
    """The snapshot file is missing, unreadable, or violates the contract."""


@dataclass(frozen=True)
class IndicatorMeta:
    name: str
    category: str
    label: str
    unit: str
    uncertainty: str
    higher_is_better: bool
    description: str
    cadence: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Chain:
    """One fired pressure-chain rule (slice 21)."""
    iso2: str
    rule_id: str
    title: str
    triggered: bool
    severity: float
    constraint: str
    forced_options: tuple[str, ...]
    spillovers: tuple[tuple[str, str, str], ...]   # (target, text, channel)
    confidence: float
    inputs: dict[str, float | None]
    uncertainty: str

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t for t, _, _ in self.spillovers))


@dataclass(frozen=True)
class Snapshot:
    as_of: date
    generated_at: str
    ranking_population: tuple[str, ...]
    categories: tuple[str, ...]
    category_labels: dict[str, str]
    catalog: dict[str, IndicatorMeta]
    players: pd.DataFrame          # iso2, iso3, name, tier, eu_member, members, on_map,
                                   # fx_regime, sanctioned, currency, dq_flag, dq_note
    indicators: pd.DataFrame       # iso2, indicator, value, pct, trend, trend_5y, lag_value,
                                   # tier, as_of, source, is_forecast, se
    category_scores: pd.DataFrame  # iso2, category, score, n_available, n_total,
                                   # distance_to_best, best_iso2
    view_scores: pd.DataFrame      # iso2, view, score
    views: dict[str, dict[str, float]]
    chains: tuple[Chain, ...]
    history: pd.DataFrame          # iso2, indicator, year, value, is_forecast
    trade: pd.DataFrame | None
    coverage: dict
    cycles: dict[str, dict] = field(default_factory=dict)   # iso2 → cycle block (cycle basket only)

    @property
    def player_codes(self) -> tuple[str, ...]:
        return tuple(self.players["iso2"])

    def player_name(self, iso2: str) -> str:
        row = self.players.loc[self.players["iso2"] == iso2, "name"]
        return str(row.iloc[0]) if not row.empty else iso2


# ─── Location + freshness ────────────────────────────────────────────────────


def snapshot_dir() -> Path:
    return Path(os.environ.get("FUNDAMENTALS_DIR", "data/snapshots"))


def snapshot_path(d: Path | None = None) -> Path:
    return (d or snapshot_dir()) / LATEST_NAME


def snapshot_fingerprint(d: Path | None = None) -> int:
    """mtime_ns of the latest file (0 if absent) — the cache key for the app."""
    p = snapshot_path(d)
    return p.stat().st_mtime_ns if p.exists() else 0


# ─── Loading ─────────────────────────────────────────────────────────────────


def _require(obj: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise SnapshotError(f"{where}: missing field(s) {missing}")


def parse_snapshot(raw: dict) -> Snapshot:
    """Validate + shape a snapshot dict (as produced by ``build_snapshot``)."""
    _require(raw, _REQUIRED_TOP, "snapshot")
    if raw["version"] not in SUPPORTED_VERSIONS:
        raise SnapshotError(f"snapshot: unsupported version {raw['version']!r}")

    catalog: dict[str, IndicatorMeta] = {}
    for i in raw["indicators"]:
        _require(i, ("name", "category", "label", "unit", "uncertainty", "higher_is_better",
                     "description"), "snapshot.indicators[]")
        catalog[i["name"]] = IndicatorMeta(
            name=i["name"], category=i["category"], label=i["label"], unit=i["unit"],
            uncertainty=i["uncertainty"], higher_is_better=bool(i["higher_is_better"]),
            description=i["description"], cadence=i.get("cadence", "A"),
            sources=tuple(i.get("sources", ())),
        )

    players, cells, cats, vscores, chains, hist = [], [], [], [], [], []
    cycles: dict[str, dict] = {}
    for iso2, c in raw["countries"].items():
        _require(c, _REQUIRED_COUNTRY, f"snapshot.countries[{iso2}]")
        dq = c["data_quality"] or {}
        if isinstance(c.get("cycle"), dict):
            cycles[iso2] = dict(c["cycle"])
        players.append({
            "iso2": iso2, "iso3": c["iso3"], "name": c["name"], "tier": int(c["tier"]),
            "eu_member": bool(c["eu_member"]), "members": tuple(c["members"]),
            "on_map": bool(c["on_map"]), "fx_regime": c["fx_regime"],
            "sanctioned": bool(c["sanctioned"]), "currency": c.get("currency"),
            "dq_flag": dq.get("flag", "high"), "dq_note": dq.get("note"),
        })
        for name, cell in c["indicators"].items():
            _require(cell, _REQUIRED_CELL, f"snapshot.countries[{iso2}].indicators[{name}]")
            if name not in catalog:
                raise SnapshotError(f"snapshot.countries[{iso2}]: indicator {name!r} not in catalog")
            cells.append({
                "iso2": iso2, "indicator": name, "value": cell["value"], "pct": cell["pct"],
                "trend": cell["trend"], "trend_5y": cell["trend_5y"],
                "lag_value": cell.get("lag_value"), "tier": cell["uncertainty"],
                "as_of": cell["date"], "source": cell["source"],
                "is_forecast": bool(cell.get("is_forecast", False)), "se": cell.get("se"),
            })
        for cat, cs in c["categories"].items():
            _require(cs, _REQUIRED_CATEGORY, f"snapshot.countries[{iso2}].categories[{cat}]")
            cats.append({"iso2": iso2, "category": cat, **{k: cs[k] for k in _REQUIRED_CATEGORY}})
        for view, score in c["views"].items():
            vscores.append({"iso2": iso2, "view": view, "score": score})
        for p in c["pressures"]:
            chains.append(Chain(
                iso2=iso2, rule_id=p["rule_id"],
                title=p.get("title") or p["rule_id"].replace("_", " ").capitalize(),
                triggered=bool(p.get("triggered", True)),
                severity=float(p.get("severity", 0.0)), constraint=p.get("constraint", ""),
                forced_options=tuple(p.get("forced_options", ())),
                spillovers=tuple((s["target"], s.get("text", ""), s.get("channel", ""))
                                 for s in p.get("spillovers", ())),
                confidence=float(p.get("confidence", 0.0)), inputs=dict(p.get("inputs", {})),
                uncertainty=p.get("uncertainty", "C"),
            ))
        for name, rows in c["history"].items():
            for r in rows:
                hist.append({"iso2": iso2, "indicator": name, "year": int(r["year"]),
                             "value": float(r["value"]), "is_forecast": bool(r["is_forecast"])})

    cell_cols = ["iso2", "indicator", "value", "pct", "trend", "trend_5y", "lag_value", "tier",
                 "as_of", "source", "is_forecast", "se"]
    trade = raw.get("trade")
    return Snapshot(
        as_of=date.fromisoformat(raw["as_of"]),
        generated_at=raw.get("generated_at", ""),
        ranking_population=tuple(raw["ranking_population"]),
        categories=tuple(raw["categories"]),
        category_labels=dict(raw.get("category_labels", {})),
        catalog=catalog,
        players=pd.DataFrame(players),
        indicators=pd.DataFrame(cells, columns=cell_cols),
        category_scores=pd.DataFrame(cats, columns=["iso2", "category", *_REQUIRED_CATEGORY]),
        view_scores=pd.DataFrame(vscores, columns=["iso2", "view", "score"]),
        views={k: dict(v) for k, v in raw["views"].items()},
        chains=tuple(chains),
        history=pd.DataFrame(hist, columns=["iso2", "indicator", "year", "value", "is_forecast"]),
        trade=pd.DataFrame(trade) if isinstance(trade, list) else None,
        coverage=dict(raw["coverage"]),
        cycles=cycles,
    )


def load_snapshot(d: Path | None = None) -> Snapshot:
    p = snapshot_path(d)
    if not p.exists():
        raise SnapshotError(f"no snapshot at {p} — run `dalio-fetch-fundamentals` then `dalio-score`")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SnapshotError(f"{p}: not valid JSON ({e})") from e
    return parse_snapshot(raw)
