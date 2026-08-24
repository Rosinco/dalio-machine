"""Snapshot loader = the executable data↔presentation contract (slice P0)."""
import json
import os
import time

import pandas as pd
import pytest

from dalio.app.fundamentals.snapshot import (
    Chain,
    SnapshotError,
    load_snapshot,
    parse_snapshot,
    snapshot_dir,
    snapshot_fingerprint,
    snapshot_path,
)
from dalio.scoring.fundamentals import CATEGORIES, FUNDAMENTALS


def test_load_snapshot_shapes(synthetic_snapshot_dir):
    snap = load_snapshot(synthetic_snapshot_dir)
    assert snap.as_of.isoformat() == "2026-08-24"
    assert snap.categories == CATEGORIES
    assert set(snap.catalog) == {s.name for s in FUNDAMENTALS}
    assert snap.catalog["old_age_dependency"].higher_is_better is False
    assert snap.ranking_population == ("US", "SE", "CN", "IN", "DE")

    assert list(snap.players["iso2"]) == ["US", "SE", "CN", "IN", "DE", "EU"]
    eu = snap.players.set_index("iso2").loc["EU"]
    assert eu["on_map"] is False or eu["on_map"] == False  # noqa: E712 — numpy bool
    assert len(eu["members"]) == 20
    assert snap.players.set_index("iso2").loc["CN", "dq_flag"] == "low"

    ind = snap.indicators
    assert list(ind.columns) == ["iso2", "indicator", "value", "pct", "trend", "trend_5y",
                                 "lag_value", "tier", "as_of", "source", "is_forecast", "se"]
    assert len(ind) == 6 * 15
    us_gdp = ind[(ind.iso2 == "US") & (ind.indicator == "gdp_pc_ppp")].iloc[0]
    assert us_gdp["pct"] == 100.0 and us_gdp["tier"] == "B" and us_gdp["as_of"] == "2024-12-31"
    eu_mil = ind[(ind.iso2 == "EU") & (ind.indicator == "military_share_world")].iloc[0]
    assert pd.isna(eu_mil["value"]) and pd.isna(eu_mil["pct"])   # numeric columns: None → NaN
    us_rol = ind[(ind.iso2 == "US") & (ind.indicator == "rule_of_law")].iloc[0]
    assert us_rol["se"] == 0.15 and us_rol["tier"] == "C"

    cs = snap.category_scores
    assert set(cs["category"]) == set(CATEGORIES)
    assert len(cs) == 6 * 5
    assert snap.view_scores.query("iso2 == 'US' and view == 'learning'")["score"].iloc[0] > 0

    assert len(snap.chains) == 1
    ch = snap.chains[0]
    assert isinstance(ch, Chain) and ch.iso2 == "US" and ch.rule_id == "fiscal_dominance"
    assert ch.spillovers == (("CN", "real return on USD reserves falls"), ("SA", "peg imports US inflation"))

    h = snap.history
    assert set(h.columns) == {"iso2", "indicator", "year", "value", "is_forecast"}
    assert sorted(h[(h.iso2 == "US") & (h.indicator == "gdp_pc_ppp")]["year"]) == [2019, 2024]
    assert snap.trade is None
    from tests.conftest import SYNTHETIC_FILLED
    assert snap.coverage["filled"] == SYNTHETIC_FILLED
    assert snap.player_name("SE") == "Sweden" and snap.player_name("ZZ") == "ZZ"


def test_missing_file_raises(tmp_path):
    with pytest.raises(SnapshotError, match="no snapshot"):
        load_snapshot(tmp_path)


def test_invalid_json_raises(tmp_path):
    snapshot_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    snapshot_path(tmp_path).write_text("{not json")
    with pytest.raises(SnapshotError, match="not valid JSON"):
        load_snapshot(tmp_path)


def test_missing_field_is_named(synthetic_snapshot_dict):
    raw = json.loads(json.dumps(synthetic_snapshot_dict))
    del raw["countries"]["SE"]["categories"]["promises"]["best_iso2"]
    with pytest.raises(SnapshotError, match=r"countries\[SE\].categories\[promises\].*best_iso2"):
        parse_snapshot(raw)
    raw = json.loads(json.dumps(synthetic_snapshot_dict))
    del raw["coverage"]
    with pytest.raises(SnapshotError, match="coverage"):
        parse_snapshot(raw)


def test_unknown_indicator_in_country_rejected(synthetic_snapshot_dict):
    raw = json.loads(json.dumps(synthetic_snapshot_dict))
    raw["countries"]["US"]["indicators"]["mystery"] = raw["countries"]["US"]["indicators"]["gdp_pc_ppp"]
    with pytest.raises(SnapshotError, match="not in catalog"):
        parse_snapshot(raw)


def test_unsupported_version_rejected(synthetic_snapshot_dict):
    raw = dict(synthetic_snapshot_dict, version=99)
    with pytest.raises(SnapshotError, match="version"):
        parse_snapshot(raw)


def test_fingerprint_tracks_rewrites(synthetic_snapshot_dir):
    fp1 = snapshot_fingerprint(synthetic_snapshot_dir)
    assert fp1 > 0
    p = snapshot_path(synthetic_snapshot_dir)
    later = time.time() + 5
    os.utime(p, (later, later))
    assert snapshot_fingerprint(synthetic_snapshot_dir) > fp1
    assert snapshot_fingerprint(synthetic_snapshot_dir / "nope") == 0


def test_snapshot_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FUNDAMENTALS_DIR", str(tmp_path / "x"))
    assert snapshot_dir() == tmp_path / "x"
