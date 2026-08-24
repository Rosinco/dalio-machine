"""Bilateral goods-trade shares (slice 24) — who trades with whom, and how much
of each player's trade that is.

Reads the partner-suffixed ``exports_to_*`` / ``imports_from_*`` observations
the IMTS adapter stores and turns them into one small frame per snapshot:

    iso2 · partner · year · x_share · m_share · x_usd · m_usd

``x_share`` = reporter's exports to the partner ÷ reporter's world exports × 100
(``m_share`` likewise for imports). Two directions matter and they are not the
same number: *my* share of trade with *you* (``top_partners``) versus how much
of *your* exports come to *me* (``exposed_players`` — the one the pressure
rules use, because a shock in me hits you in proportion to that).

Goods only, tier A (customs data) but with the usual CIF/FOB asymmetry; the
euro-area aggregate's totals include intra-area trade (its member-partner rows
are dropped at fetch time).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import COUNTRIES, Country
from dalio.data_sources.imf_imts import IMTS_FLOWS, SOURCE_IMTS, WORLD_PARTNER
from dalio.storage.db import Observation

TRADE_COLUMNS = ["iso2", "partner", "year", "x_share", "m_share", "x_usd", "m_usd"]
RAW_COLUMNS = ["iso2", "partner", "year", "x_usd", "m_usd"]
TRADE_CAVEATS = ("Goods only (IMF IMTS; exports FOB, imports CIF) — services excluded. "
                 "The euro-area aggregate's world totals include intra-area trade.")
DEFAULT_MIN_SHARE = 2.0
_PREFIX_TO_COL = {"exports_to": "x_usd", "imports_from": "m_usd"}


def partner_indicator_names(iso2s: Sequence[str]) -> list[str]:
    partners = [*iso2s, WORLD_PARTNER]
    return [spec.indicator(p) for spec in IMTS_FLOWS for p in partners]


def load_trade(session: Session, countries: Sequence[Country] = COUNTRIES,
               as_of: date | None = None) -> pd.DataFrame:
    """Latest bilateral year per reporter at or before ``as_of`` — the latest
    year that carries a world total (``WLD``) for that reporter, so shares are
    never computed against a missing denominator. ONE query."""
    as_of = as_of or date.today()
    iso2s = [c.iso2 for c in countries]
    names = partner_indicator_names(iso2s)
    rows = session.execute(
        select(Observation.country, Observation.indicator, Observation.date, Observation.value)
        .where(
            Observation.indicator.in_(names),
            Observation.country.in_(iso2s),
            Observation.source == SOURCE_IMTS,
            Observation.date <= as_of,
        )
    ).all()
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(rows, columns=["iso2", "indicator", "date", "value"])
    split = df["indicator"].str.rsplit("_", n=1, expand=True)
    df["col"] = split[0].map(_PREFIX_TO_COL)
    df["partner"] = split[1]
    df["year"] = [d.year for d in df["date"]]
    df = df.dropna(subset=["col"])
    wide = (df.pivot_table(index=["iso2", "partner", "year"], columns="col", values="value", aggfunc="last")
            .reindex(columns=["x_usd", "m_usd"]).reset_index())
    totals = wide[wide["partner"] == WORLD_PARTNER]
    totals = totals[totals[["x_usd", "m_usd"]].notna().any(axis=1)]
    latest_year = totals.groupby("iso2")["year"].max()
    wide = wide[wide["year"] == wide["iso2"].map(latest_year)]
    return wide[RAW_COLUMNS].sort_values(["iso2", "partner"]).reset_index(drop=True)


def trade_shares(raw: pd.DataFrame) -> pd.DataFrame:
    """Partner rows with shares of the reporter's world totals (NaN when the
    total is missing or zero). The ``WLD`` rows themselves are not emitted."""
    if raw.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    tot = raw[raw["partner"] == WORLD_PARTNER].set_index("iso2")[["x_usd", "m_usd"]]
    part = raw[raw["partner"] != WORLD_PARTNER].copy()
    if part.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    xt = part["iso2"].map(tot["x_usd"]).astype(float)
    mt = part["iso2"].map(tot["m_usd"]).astype(float)
    part["x_share"] = (part["x_usd"] / xt.where(xt > 0) * 100.0).astype(float)
    part["m_share"] = (part["m_usd"] / mt.where(mt > 0) * 100.0).astype(float)
    part["year"] = part["year"].astype(int)
    return (part[TRADE_COLUMNS].sort_values(["iso2", "x_share"], ascending=[True, False], na_position="last")
            .reset_index(drop=True))


def world_totals(raw: pd.DataFrame) -> pd.DataFrame:
    """``iso2 → year, x_usd, m_usd`` world totals (for the UI's denominators)."""
    if raw.empty:
        return pd.DataFrame(columns=["iso2", "year", "x_usd", "m_usd"]).set_index("iso2")
    return raw[raw["partner"] == WORLD_PARTNER].set_index("iso2")[["year", "x_usd", "m_usd"]]


def top_partners(shares: pd.DataFrame, iso2: str, by: str = "x_share", n: int = 3,
                 min_share: float = DEFAULT_MIN_SHARE) -> list[tuple[str, float]]:
    """The reporter's largest partners by ``x_share`` / ``m_share`` / ``total``
    (sum of both), at or above ``min_share`` %."""
    if shares.empty:
        return []
    sub = shares[shares["iso2"] == iso2]
    if by == "total":
        score = sub[["x_share", "m_share"]].sum(axis=1, min_count=1)
    else:
        score = sub[by].astype(float)
    sub = sub.assign(_s=score).dropna(subset=["_s"])
    sub = sub[sub["_s"] >= min_share].sort_values("_s", ascending=False, kind="stable").head(n)
    return [(str(p), float(s)) for p, s in zip(sub["partner"], sub["_s"], strict=True)]


def exposure_to(shares: pd.DataFrame, target: str, iso2: str) -> float | None:
    """Share of ``target``'s exports that go to ``iso2`` — how much *they*
    depend on *us* as a buyer."""
    if shares.empty:
        return None
    row = shares[(shares["iso2"] == target) & (shares["partner"] == iso2)]
    if row.empty or pd.isna(row["x_share"].iloc[0]):
        return None
    return float(row["x_share"].iloc[0])


def import_share(shares: pd.DataFrame, iso2: str, partner: str) -> float | None:
    """Share of ``iso2``'s imports that come from ``partner``."""
    if shares.empty:
        return None
    row = shares[(shares["iso2"] == iso2) & (shares["partner"] == partner)]
    if row.empty or pd.isna(row["m_share"].iloc[0]):
        return None
    return float(row["m_share"].iloc[0])


def exposed_players(shares: pd.DataFrame, iso2: str, n: int = 3,
                    min_share: float = DEFAULT_MIN_SHARE) -> list[tuple[str, float]]:
    """Players whose exports to ``iso2`` are at least ``min_share`` % of their
    own exports, top ``n`` by that share — the mechanical spillover set for a
    demand shock in ``iso2``."""
    if shares.empty:
        return []
    sub = shares[(shares["partner"] == iso2) & (shares["iso2"] != iso2)].dropna(subset=["x_share"])
    sub = sub[sub["x_share"] >= min_share].sort_values("x_share", ascending=False, kind="stable").head(n)
    return [(str(r), float(s)) for r, s in zip(sub["iso2"], sub["x_share"], strict=True)]


def trade_block(shares: pd.DataFrame) -> list[dict] | None:
    """JSON-ready rows for the snapshot (NaN → None); ``None`` when there is no trade."""
    if shares.empty:
        return None
    out = []
    for r in shares[TRADE_COLUMNS].itertuples(index=False):
        d = r._asdict()
        d["year"] = int(d["year"])
        for k in ("x_share", "m_share", "x_usd", "m_usd"):
            v = d[k]
            d[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        out.append(d)
    return out
