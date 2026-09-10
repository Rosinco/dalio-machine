"""Downloaded company identity, vintage selection and transparent classification."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "listing_model", Path(__file__).parents[1] / "scripts/listing_model.py"
)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def row(identifier=1, **changes):
    return {
        "ins_id": identifier,
        "name": "Åland A",
        "ticker": "AL A",
        "isin": "FI0000000001",
        "instrument_type": 0,
        "country_id": 3,
        "sector_id": 7,
        "branch_id": 21,
        "stock_price_currency": "EUR",
        "report_currency": "EUR",
        "listing_date": None,
        **changes,
    }


def test_latest_record_wins_without_merging_share_classes_or_resurrecting_old_type():
    source = [
        ("2025-06-21", [row(), row(2, name="Older only"), row(3)]),
        (
            "2026-08-10",
            [
                row(name="Åland renamed"),
                row(3, instrument_type=2, branch_id=None),
                row(4, instrument_type=3),
            ],
        ),
    ]
    rows = model.select_listings(source)
    assert set(rows) == {"1", "2", "4"}
    assert rows["1"]["name"] == "Åland renamed"
    assert rows["1"]["source_as_of"] == "2026-08-10"
    assert rows["2"]["source_as_of"] == "2025-06-21"
    assert rows["4"]["isin"] == rows["1"]["isin"]


def test_duplicate_ids_within_a_source_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        model.select_listings([("2026-08-10", [row(), row()])])


def test_unknown_country_and_missing_fields_are_retained():
    rows = model.select_listings(
        [
            (
                "2026-08-10",
                [row(country_id=999, branch_id=None, sector_id=None, name=None, ticker=None)],
            )
        ]
    )
    result = model.listing_record(rows["1"], {})
    assert result["country_id"] == "999"
    assert result["listing_country"] is None
    assert result["name"] is None and result["ticker"] is None
    assert result["branch_id"] is None


BRANCHES = {"21": {"sector_id": "7"}, "31": {"sector_id": "5"}}


def test_branch_parent_groups_the_row_but_preserves_a_conflicting_source_sector():
    result = model.classify_listing("1", "3", "21", None, BRANCHES)
    assert result["source_sector_id"] == "3"
    assert result["source_branch_id"] == result["branch_id"] == "21"
    assert result["sector_id"] == "7"
    assert result["status"] == "sector_mismatch"
    missing = model.classify_listing("2", "7", "999", None, BRANCHES)
    assert missing["source_branch_id"] == "999" and missing["branch_id"] is None
    assert missing["status"] == "unclassified"


def test_reviewed_correction_for_any_listing_survives_source_drift():
    review = {
        "company_id": "1",
        "expected_sector_id": "3",
        "expected_branch_id": "21",
        "branch_id": "31",
        "reason": "Synthetic review",
        "source": "Synthetic filing",
        "reviewed_at": "2026-09-10",
    }
    changed = model.classify_listing("1", "3", "21", review, BRANCHES)
    assert changed["status"] == "corrected" and changed["branch_id"] == "31"
    drift = model.classify_listing("1", "7", "21", review, BRANCHES)
    assert drift["status"] == "needs_review" and drift["branch_id"] == "21"
    assert drift["correction"] == review
    assert model.classify_listing("1", "5", "31", review, BRANCHES)["status"] == "aligned"
    with pytest.raises(ValueError):
        model.classify_listing("1", "3", "21", {**review, "reason": ""}, BRANCHES)
