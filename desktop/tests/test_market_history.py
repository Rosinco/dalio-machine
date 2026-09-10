"""Market-cap integration: dates, units and observable gaps are part of the value."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from market_model import market_coverage, market_record, select_fx, select_quote


def annual(**kw):
    d = dict(
        year=2024,
        start="2024-01-01",
        end="2024-12-31",
        published="2025-01-31",
        source_id="2026-08-10-annual",
        as_of="2026-08-10",
        profit=2862,
        equity=57370,
    )
    return {**d, **kw}


def record(**kw):
    return market_record(
        annual(),
        shares=157.668,
        currency="SEK",
        quote=("2025-01-31", 420.4),
        fx=("2025-01-31", 1.0, "identity", []),
        **kw,
    )


def test_holmen_uses_actual_publication_close_and_millions():
    r = record(basis_ok=True)
    assert r["local"] == pytest.approx(66283.6272)
    assert r["sek"] == r["local"]
    assert r["price_date"] == "2025-01-31"
    assert r["year"] == 2024 and r["flags"] == []


def test_entry_close_is_forward_bounded_and_never_uses_future_snapshot_prices():
    prices = [("2025-01-30", 400), ("2025-02-03", 420), ("2025-03-05", 430)]
    assert select_quote(prices, "2025-01-31", "2025-02-03") == prices[1]
    assert select_quote(prices, "2025-01-31", "2025-02-02") is None
    assert select_quote(prices, "2024-11-01", "2025-02-03") is None
    assert select_quote(prices, None, "2025-02-03") is None
    assert select_quote([("2025-01-31", 1e13)], "2025-01-31", "2025-02-03") is None


def test_fx_has_no_static_or_future_fallback_and_keeps_observation_date():
    fx = [("2025-01-30", 11.2, "direct", ["20986"])]
    assert select_fx(fx, "2025-01-31") == fx[0]
    assert select_fx(fx, "2025-01-29") is None
    assert select_fx(fx, "2025-02-07") is None


def test_missing_fx_preserves_local_value_and_does_not_label_it_as_sek():
    r = market_record(
        annual(profit=20, equity=1000),
        shares=10,
        currency="EUR",
        quote=("2025-01-31", 20),
        fx=None,
        basis_ok=True,
    )
    assert r["local"] == 200 and r["sek"] is None
    assert r["flags"] == ["missing_fx"]
    assert market_coverage([r]) == dict(count=1, local=1, sek=0, flagged=0)


@pytest.mark.parametrize(
    "changes,flag",
    [
        ({"shares": None}, "missing_shares"),
        ({"shares": 0}, "missing_shares"),
        ({"quote": None}, "missing_price"),
        ({"currency": None}, "missing_currency"),
        ({"basis_ok": False}, "share_basis"),
        ({"receipt": True}, "receipt_basis"),
    ],
)
def test_unusable_bases_are_withheld_without_zero_filling(changes, flag):
    args = dict(
        shares=10,
        currency="SEK",
        quote=("2025-01-31", 20),
        fx=("2025-01-31", 1, "identity", []),
        basis_ok=True,
    )
    if changes.get("quote", True) is None or changes.get("currency", True) is None:
        args["fx"] = None
    r = market_record(annual(), **{**args, **changes})
    assert r["local"] is None and r["sek"] is None and flag in r["flags"]


def test_scale_check_does_not_call_a_loss_or_zero_equity_an_invalid_cap():
    args = dict(
        shares=10,
        currency="SEK",
        quote=("2025-01-31", 20),
        fx=("2025-01-31", 1, "identity", []),
        basis_ok=True,
    )
    bad = market_record(annual(profit=1000), **args)
    good = market_record(annual(profit=-1000, equity=0), **args)
    assert "scale_suspect" in bad["flags"] and bad["local"] is None
    assert good["local"] == 200


def test_short_year_and_missing_publication_are_explicit():
    args = dict(shares=10, currency="SEK", quote=None, fx=None, basis_ok=True)
    r = market_record(annual(start="2024-07-01", published=None), **args)
    assert "short_period" in r["flags"] and "missing_publication" in r["flags"]


def test_bad_fx_and_price_dates_cannot_be_exported():
    args = dict(
        shares=157.668,
        currency="SEK",
        quote=("2025-01-30", 420.4),
        fx=("2025-01-31", 1, "identity", []),
        basis_ok=True,
    )
    with pytest.raises(ValueError):
        market_record(annual(), **args)
    args["quote"] = ("2025-01-31", 420.4)
    args["fx"] = ("2025-02-01", 1, "identity", [])
    with pytest.raises(ValueError):
        market_record(annual(), **args)
