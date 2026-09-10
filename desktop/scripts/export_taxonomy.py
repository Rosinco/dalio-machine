"""Export the existing taxonomy and research inventory, without source writes or HTTP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from taxonomy_model import classify, corrections_by_company, dive_inventory, file_record, hierarchy

# These shared themes are explicitly documented in the source crosswalk and INDEX.
SHARED_STUDIES = {
    "branded_beverages": {"path": "dagligvaror/branded_beverages", "branch_ids": ["58", "59"]},
    "banks": {"path": "finans/banks", "branch_ids": ["68", "69"]},
}


def export(root: Path, business: Path | None, corrections: Path, output: Path) -> dict:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Atlas exports must be written outside the Börsdata project")
    crosswalk = root / "data/complementing/taxonomy/branch_crosswalk.csv"
    rows = list(
        csv.DictReader(
            line
            for line in crosswalk.read_text(encoding="utf-8-sig").splitlines()
            if not line.startswith("#")
        )
    )
    sectors, branches = hierarchy(rows)
    correction_raw = json.loads(corrections.read_text(encoding="utf-8"))
    reviewed = corrections_by_company(correction_raw, branches)
    business_bytes = business.read_bytes() if business else None
    business_raw = json.loads(business_bytes) if business_bytes else None
    companies = business_raw["companies"] if business_raw else {}
    missing = set(reviewed) - set(companies)
    if missing:
        raise ValueError(f"Corrections must accompany included company records: {sorted(missing)}")
    classifications = {
        key: classify(key, c["sector_id"], c["branch_id"], reviewed.get(key), branches)
        for key, c in companies.items()
    }
    for branch in branches.values():
        overview = root / branch["study_path"] / "README.md"
        if not overview.is_file():
            raise ValueError(f"The saved branch overview is missing: {branch['study_path']}")
        branch["overview"] = file_record(root, overview)
        branch["deep_dives"] = dive_inventory(
            root, [branch["study_path"].removeprefix("studies/sectors/")]
        )
        branch["shared_study_ids"] = [
            key for key, group in SHARED_STUDIES.items() if branch["id"] in group["branch_ids"]
        ]
    shared = {
        key: {
            "id": key,
            **group,
            "overview": file_record(root, root / "studies/sectors" / group["path"] / "WORKFLOW.md"),
            "deep_dives": dive_inventory(root, [group["path"]]),
        }
        for key, group in SHARED_STUDIES.items()
    }
    mapped = {
        d["folder"] for group in [*branches.values(), *shared.values()] for d in group["deep_dives"]
    }
    all_studies = [
        f"{p.parent.parent.parent.relative_to(root / 'studies/sectors')}"
        for p in (root / "studies/sectors").glob("*/*/deep_dives/*/deep_dive*.md")
    ]
    all_dives = dive_inventory(root, all_studies)
    unmapped = [d for d in all_dives if d["folder"] not in mapped]
    now = datetime.now(UTC).isoformat()
    result = {
        "version": 1,
        "as_of": now[:10],
        "exported_at": now,
        "business_sha256": hashlib.sha256(business_bytes).hexdigest() if business_bytes else None,
        "classification_as_of": business_raw["as_of"] if business_raw else None,
        "sectors": sectors,
        "branches": branches,
        "classifications": classifications,
        "shared_studies": shared,
        "unmapped_deep_dives": unmapped,
        "sources": [
            file_record(root, crosswalk),
            file_record(root, root / "studies/sectors/INDEX.md"),
            file_record(root, root / "decisions/0034-taxonomy-english-alias-crosswalk.md"),
        ],
        "corrections_sha256": hashlib.sha256(corrections.read_bytes()).hexdigest(),
        "notes": [
            "Names and research-status labels come from the saved Börsdata taxonomy crosswalk. Swedish and English labels can differ in scope; IDs are the shared keys.",
            "Branch study and corpus status describe the source project's registry. They are not current investment judgments or proof that every company has been researched.",
            "Deep-dive coverage counts saved research folders containing analysis documents, once per folder. Drafts and older versions may be present; completion and freshness are not inferred.",
            "Shared study groups are displayed separately and counted once in directory totals. Unmapped research is retained for a later classification review.",
            "Only profiles in the active business package can be opened in Atlas. Other project documents are inventoried here; their full contents are not included.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return {
        "sectors": len(sectors),
        "branches": len(branches),
        "branch_studies": sum(b["study_status"] == "graduated" for b in branches.values()),
        "deep_dive_folders": len(all_dives),
        "unmapped": len(unmapped),
        "profiles": len(companies),
        "bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borsdata-root", type=Path, required=True)
    parser.add_argument("--business", type=Path)
    parser.add_argument(
        "--corrections",
        type=Path,
        default=Path(__file__).parents[1] / "research/classification-corrections.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.borsdata_root, args.business, args.corrections, args.output)))
