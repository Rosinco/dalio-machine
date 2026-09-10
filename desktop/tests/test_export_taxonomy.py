"""Contracts for source classifications, corrections and research inventory."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "taxonomy_model", Path(__file__).parents[1] / "scripts/taxonomy_model.py"
)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def rows():
    return [
        {
            "branch_id": "21",
            "sector_id": "7",
            "name_sv": "Skogsbolag",
            "name_en": "Forest & Wood Products",
            "sector_sv_ui": "Material",
            "sector_en_ui": "Materials",
            "slug": "skogsbolag",
            "sector_slug": "material",
            "study_status": "graduated",
            "corpus_status": "none",
        },
        {
            "branch_id": "31",
            "sector_id": "5",
            "name_sv": "Byggmaterial",
            "name_en": "Construction Materials",
            "sector_sv_ui": "Industri",
            "sector_en_ui": "Industrials",
            "slug": "byggmaterial",
            "sector_slug": "industri",
            "study_status": "scaffold",
            "corpus_status": "none",
        },
        {
            "branch_id": "32",
            "sector_id": "5",
            "name_sv": "Bygginredning",
            "name_en": "Building Fixtures",
            "sector_sv_ui": "Industri",
            "sector_en_ui": "Industrials",
            "slug": "bygginredning",
            "sector_slug": "industri",
            "study_status": "scaffold",
            "corpus_status": "none",
        },
    ]


def correction(**changes):
    return {
        "company_id": "102",
        "expected_sector_id": "7",
        "expected_branch_id": "21",
        "branch_id": "31",
        "reason": "Synthetic reviewed category correction",
        "source": "Synthetic annual report p. 4",
        "reviewed_at": "2026-09-10",
        **changes,
    }


def test_ids_not_translated_names_are_the_keys():
    sectors, branches = model.hierarchy(rows())
    assert set(sectors) == {"5", "7"}
    assert branches["21"]["name_sv"] == "Skogsbolag"
    assert branches["21"]["name_en"] == "Forest & Wood Products"
    changed = rows()
    changed[0]["name_en"] = "Renamed display label"
    assert set(model.hierarchy(changed)[1]) == set(branches)
    with pytest.raises(ValueError, match="Duplicate"):
        model.hierarchy(rows() + [rows()[0]])


def test_conflicting_sector_labels_and_unsafe_folder_slugs_fail():
    invalid = rows()
    invalid[2]["sector_en_ui"] = "Unexpected label"
    with pytest.raises(ValueError, match="sector"):
        model.hierarchy(invalid)
    invalid = rows()
    invalid[0]["slug"] = "../outside"
    with pytest.raises(ValueError, match="slug"):
        model.hierarchy(invalid)


def test_correction_keeps_original_values_and_derives_destination_sector():
    _, branches = model.hierarchy(rows())
    result = model.classify("102", "7", "21", correction(), branches)
    assert result["source_branch_id"] == "21"
    assert result["branch_id"] == "31"
    assert result["sector_id"] == "5"
    assert result["status"] == "corrected"
    assert result["correction"]["reason"] == correction()["reason"]


def test_vendor_refresh_keeps_correction_and_flags_conflicts():
    _, branches = model.hierarchy(rows())
    changed = model.classify("102", "5", "32", correction(), branches)
    assert changed["status"] == "needs_review"
    assert changed["branch_id"] == "32"
    assert changed["correction"] == correction()
    aligned = model.classify("102", "5", "31", correction(), branches)
    assert aligned["status"] == "aligned"
    assert aligned["branch_id"] == "31"


def test_unreviewed_unknown_and_duplicate_corrections_are_rejected():
    _, branches = model.hierarchy(rows())
    with pytest.raises(ValueError, match="reason"):
        model.classify("102", "7", "21", correction(reason=""), branches)
    with pytest.raises(ValueError, match="branch"):
        model.classify("102", "7", "21", correction(branch_id="999"), branches)
    with pytest.raises(ValueError, match="Duplicate"):
        model.corrections_by_company(
            {"version": 1, "corrections": [correction(), correction()]}, branches
        )


def test_inventory_counts_folders_once_and_does_not_infer_completion(tmp_path):
    base = tmp_path / "studies/sectors/material/skogsbolag/deep_dives/holm_b"
    base.mkdir(parents=True)
    (base / "deep_dive.md").write_text("# Old saved analysis")
    (base / "deep_dive_2026-09-10.md").write_text("# New draft, not a completion declaration")
    (base / "sources.md").write_text("Source list")
    (base.parent / "empty_scaffold").mkdir()
    inventory = model.dive_inventory(tmp_path, ["material/skogsbolag"])
    assert len(inventory) == 1
    assert len(inventory[0]["documents"]) == 2
    assert "completed" not in inventory[0]
    assert inventory[0]["documents"][0]["sha256"]


def test_shared_study_folder_is_not_counted_twice(tmp_path):
    base = tmp_path / "studies/sectors/dagligvaror/shared/deep_dives/test"
    base.mkdir(parents=True)
    (base / "deep_dive.md").write_text("# Synthetic shared study")
    assert len(model.dive_inventory(tmp_path, ["dagligvaror/shared", "dagligvaror/shared"])) == 1
