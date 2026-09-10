"""Börsdata taxonomy identity, reviewed corrections and bounded file inventory."""

from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path


def identifier(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]{0,9}", value) is not None


def hierarchy(rows: list[dict]) -> tuple[dict, dict]:
    sectors, branches = {}, {}
    for row in rows:
        bid, sid = row["branch_id"], row["sector_id"]
        if not identifier(bid) or not identifier(sid):
            raise ValueError("Invalid sector or branch ID")
        if bid in branches:
            raise ValueError("Duplicate branch ID")
        for key in ("slug", "sector_slug"):
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", row[key]):
                raise ValueError("Invalid source folder slug")
        sector = {
            "id": sid,
            "name_sv": row["sector_sv_ui"],
            "name_en": row["sector_en_ui"],
            "slug": row["sector_slug"],
        }
        if sid in sectors and sectors[sid] != sector:
            raise ValueError("Conflicting sector labels or folders")
        if not all(isinstance(v, str) and v.strip() for v in sector.values()):
            raise ValueError("Missing sector metadata")
        if row["study_status"] not in {"scaffold", "graduated"} or row["corpus_status"] not in {
            "none",
            "graduated",
        }:
            raise ValueError("Unknown source research status")
        sectors[sid] = sector
        branches[bid] = {
            "id": bid,
            "sector_id": sid,
            "name_sv": row["name_sv"],
            "name_en": row["name_en"],
            "study_path": f"studies/sectors/{row['sector_slug']}/{row['slug']}",
            "study_status": row["study_status"],
            "corpus_status": row["corpus_status"],
        }
        if not branches[bid]["name_sv"].strip() or not branches[bid]["name_en"].strip():
            raise ValueError("Missing branch name")
    return sectors, branches


def validate_correction(row: dict, branches: dict) -> None:
    for key in ("company_id", "expected_sector_id", "expected_branch_id", "branch_id"):
        if not identifier(row.get(key)):
            raise ValueError(f"Invalid correction {key}")
    if row["branch_id"] not in branches or row["expected_branch_id"] not in branches:
        raise ValueError("Unknown correction branch")
    if branches[row["expected_branch_id"]]["sector_id"] != row["expected_sector_id"]:
        raise ValueError("Correction expected sector does not match its branch")
    for key in ("reason", "source"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"A reviewed correction requires a {key}")
    stamp = row.get("reviewed_at", "")
    if (
        not re.fullmatch(r"20\d\d-\d\d-\d\d", stamp)
        or date.fromisoformat(stamp).isoformat() != stamp
    ):
        raise ValueError("A correction requires a valid review date")


def corrections_by_company(raw: dict, branches: dict) -> dict:
    if raw.get("version") != 1 or not isinstance(raw.get("corrections"), list):
        raise ValueError("Unsupported correction file")
    result = {}
    for row in raw["corrections"]:
        validate_correction(row, branches)
        if row["company_id"] in result:
            raise ValueError("Duplicate company correction")
        result[row["company_id"]] = row
    return result


def classify(
    company_id: str, sector_id: str, branch_id: str, correction: dict | None, branches: dict
) -> dict:
    if branch_id not in branches or branches[branch_id]["sector_id"] != sector_id:
        raise ValueError("Company source branch and sector do not match the saved hierarchy")
    status, effective = "source", branch_id
    if correction:
        validate_correction(correction, branches)
        if correction["company_id"] != company_id:
            raise ValueError("Correction belongs to a different company")
        if branch_id == correction["branch_id"]:
            status = "aligned"
        elif (sector_id, branch_id) == (
            correction["expected_sector_id"],
            correction["expected_branch_id"],
        ):
            status, effective = "corrected", correction["branch_id"]
        else:
            status = "needs_review"
    return {
        "company_id": company_id,
        "source_sector_id": sector_id,
        "source_branch_id": branch_id,
        "sector_id": branches[effective]["sector_id"],
        "branch_id": effective,
        "status": status,
        "correction": correction,
    }


def file_record(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    contents = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
    }


def dive_inventory(root: Path, study_paths: list[str]) -> list[dict]:
    groups = {}
    for relative in set(study_paths):
        base = root / "studies/sectors" / relative
        base.resolve().relative_to((root / "studies/sectors").resolve())
        for path in sorted(base.glob("deep_dives/*/deep_dive*.md")):
            if path.is_file():
                folder = path.parent.relative_to(root).as_posix()
                groups.setdefault(folder, []).append(file_record(root, path))
    return [
        {
            "folder": folder,
            "label": Path(folder).name.replace("_", " "),
            "documents": sorted({d["path"]: d for d in docs}.values(), key=lambda d: d["path"]),
        }
        for folder, docs in sorted(groups.items())
    ]
