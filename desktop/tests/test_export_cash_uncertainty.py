"""Source identity and immutable-bundle checks for the shipped research factors."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("cash_uncertainty_export", ROOT / "scripts/export-cash-uncertainty.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUNDLE = ROOT / "src/data/cash-uncertainty-2026-09-12.json"
SOURCE = ROOT / "test-results/cash-flow-segmentation-2026-09-12"


def bundle():
    return json.loads(BUNDLE.read_text())


def test_bundle_keeps_original_covid_retained_research_contract():
    data = bundle()
    assert MODULE.file_hash(BUNDLE) == "9d4664c6f433a9fa9dc7bf8256686522db1660d70d51e875e59eb85fb1690ff2"
    MODULE.validate_artifact(data)
    assert data["id"] == "cash-uncertainty-2026-09-12-v1"
    assert data["targetCoverage"] == .8
    assert data["status"] == "research"
    assert data["historyYears"] == 5
    assert data["scope"] == "operating_and_property"
    assert data["weighting"] == "listing_balanced"
    assert data["calibrationLastTargetYear"] == 2020
    assert data["calibrationCutoff"] == "2021-06-30"
    assert data["covidTreatment"]["omittedTargetYears"] == []
    assert data["latestEmpiricalHorizon"] == 4
    assert data["cashDispersion"]["bounds"] == [.25, .75]
    assert data["cashFlowBasis"]["quoteCurrencyFallbackTested"] is False
    assert data["modelDefinitions"]["linear"]["newestFirstWeights"] == [30, 25, 20, 15, 10]
    assert set(data["calibrationByModelHorizon"]) == {"linear", "naive"}
    for horizons in data["calibrationByModelHorizon"].values():
        assert set(horizons) == {"1", "2", "3", "4"}
    assert data["calibrationByModelHorizon"]["naive"]["1"]["global"]["factor"] == 1.406043046357616
    assert data["calibrationByModelHorizon"]["linear"]["1"]["cashDispersion"]["low"]["factor"] == .5428303157971751
    assert data["calibrationByModelHorizon"]["linear"]["4"]["cashDispersion"]["high"]["factor"] == 4.591973746136332


@pytest.mark.parametrize("change", [
    lambda value: value["covidTreatment"]["omittedTargetYears"].append(2020),
    lambda value: value.__setitem__("latestEmpiricalHorizon", 10),
    lambda value: value.__setitem__("targetCoverage", .95),
    lambda value: value["modelDefinitions"]["linear"].__setitem__("newestFirstWeights", [20, 20, 20, 20, 20]),
    lambda value: value["cashFlowBasis"].__setitem__("quoteCurrencyFallbackTested", True),
    lambda value: value["calibrationByModelHorizon"]["naive"]["1"]["global"].__setitem__("factor", -1),
    lambda value: value["calibrationByModelHorizon"]["linear"]["1"]["global"].__setitem__("count", 10),
    lambda value: value["calibrationByModelHorizon"]["linear"].__setitem__("5", value["calibrationByModelHorizon"]["linear"]["4"]),
])
def test_invalid_probability_crisis_filter_or_support_cannot_enter_bundle(change):
    data = bundle()
    change(data)
    with pytest.raises(ValueError):
        MODULE.validate_artifact(data)


def test_immutable_writer_accepts_identical_regeneration_and_refuses_replacement(tmp_path):
    path = tmp_path / "dated-calibration.json"
    MODULE.write_immutable(path, b"original\n")
    MODULE.write_immutable(path, b"original\n")
    with pytest.raises(ValueError, match="immutable"):
        MODULE.write_immutable(path, b"changed\n")
    assert path.read_bytes() == b"original\n"


def test_complete_bundle_matches_exact_source_factors_and_support_without_refitting():
    if not (SOURCE / "group-calibration.json").exists():
        pytest.skip("Source-bound reproduction requires the preserved gitignored research artifacts")
    sources = MODULE.load_verified_sources(SOURCE)
    reproduced = MODULE.build_artifact(sources)
    assert MODULE.serialize(reproduced) == BUNDLE.read_bytes()
    for row in sources["calibration"]["calibrations"]:
        if row["scope"] != "operating_and_property" or row["weighting"] != "listing_balanced":
            continue
        cell = reproduced["calibrationByModelHorizon"][row["model"]][str(row["horizon"])]
        assert cell["global"] == row["groups"]["global"]["all"]
        assert cell["cashDispersion"] == row["groups"]["cash_dispersion"]


def test_newer_evaluation_changes_do_not_reestimate_the_older_factors():
    if not (SOURCE / "group-calibration.json").exists():
        pytest.skip("Source-bound reproduction requires the preserved gitignored research artifacts")
    sources = MODULE.load_verified_sources(SOURCE)
    original = MODULE.build_artifact(sources)
    altered = copy.deepcopy(sources)
    for row in altered["results"]["metrics"]:
        row["meanWidth"] *= 2
        row["meanIntervalScore"] *= 2
    candidate = MODULE.build_artifact(altered)
    assert candidate["calibrationByModelHorizon"] == original["calibrationByModelHorizon"]
    assert candidate["evaluationByModelHorizon"] != original["evaluationByModelHorizon"]


def test_changed_source_file_is_rejected_before_export(tmp_path):
    for name in MODULE.SOURCE_HASHES:
        (tmp_path / name).write_text("{\"changed\":true}\n")
    with pytest.raises(ValueError, match="source identity"):
        MODULE.load_verified_sources(tmp_path)
