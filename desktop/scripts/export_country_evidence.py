"""Project verified macro snapshots into an independent, lazy offline desktop pack.

No network, SQLite access, scoring or acquisition. Original source snapshots are
read-only. All native observations, missing slots, scopes and source clocks in
the selected histories remain unchanged; countries select whole monitoring
profiles, never a mixture of successful signals from different captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

FORMAT = "macro-atlas-country-evidence-v1"


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clock(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None, "Source clock must specify timezone")
    return result.astimezone(UTC)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate source JSON key")
            result[key] = value
        return result

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def validate_snapshot(snapshot):
    require(snapshot.get("schema_version") == 1, "Unsupported country snapshot schema")
    require(
        snapshot.get("snapshot_sha256")
        == sha(canonical({k: v for k, v in snapshot.items() if k != "snapshot_sha256"})),
        "Source snapshot hash mismatch",
    )
    require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot["as_of"])
        and snapshot["as_of"] <= clock(snapshot["as_known_at"]).date().isoformat(),
        "Invalid source assessment date/cutoff",
    )
    countries = snapshot["countries"]
    require(isinstance(countries, list) and 0 < len(countries) <= 250, "Invalid country catalogue")
    codes = [c["country"] for c in countries]
    require(
        all(re.fullmatch(r"[A-Z]{2}", c) for c in codes) and len(codes) == len(set(codes)),
        "Duplicate/invalid country identity",
    )
    for country in countries:
        validate_refs(country, snapshot["citations"], country["country"])


def validate_refs(value, citations, country):
    if isinstance(value, list):
        for child in value:
            validate_refs(child, citations, country)
    elif isinstance(value, dict):
        if "evidence_ref" in value:
            ref = value["evidence_ref"]
            require(
                ref in citations and citations[ref].get("country") == country,
                "Missing/wrong-country citation",
            )
            for key in (
                "value",
                "unit",
                "year",
                "date",
                "period",
                "period_start",
                "period_end",
                "status",
            ):
                if key in value:
                    require(
                        citations[ref].get(key) == value[key], "Source point differs from citation"
                    )
        for key, child in value.items():
            if key.endswith("evidence_refs"):
                require(
                    all(
                        ref in citations and citations[ref].get("country") == country
                        for ref in child
                    ),
                    "Unresolved/wrong-country citation",
                )
            elif key != "evidence_ref":
                validate_refs(child, citations, country)


def histories(value, owner=None):
    """Find source histories through country-specific evidence wrappers."""
    result = []
    if isinstance(value, list):
        for child in value:
            result.extend(histories(child, owner))
    elif isinstance(value, dict):
        country = value.get("country", owner)
        if "observations" in value and "indicator" in value:
            result.append({**deepcopy(value), "country": country})
        else:
            for child in value.values():
                if isinstance(child, (dict, list)):
                    result.extend(histories(child, country))
    return result


def envelope(snapshot, profile):
    country = profile["country"]
    return {
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "as_of": snapshot["as_of"],
        "as_known_at": snapshot["as_known_at"],
        "methodology": deepcopy(snapshot["methodology"]),
        "profile": deepcopy(profile),
        "citations": {
            k: deepcopy(v) for k, v in snapshot["citations"].items() if v.get("country") == country
        },
    }


def project(assessment, monitoring):
    validate_snapshot(assessment)
    annual_selected = {c["country"]: (assessment, c) for c in assessment["countries"]}
    selected = {}
    for snapshot in monitoring:
        validate_snapshot(snapshot)
        embedded = snapshot.get("country_assessment")
        if embedded:
            validate_snapshot(embedded)
            require(
                clock(embedded["as_known_at"]) <= clock(snapshot["as_known_at"]),
                "Embedded assessment available after monitoring cutoff",
            )
            for annual_profile in embedded["countries"]:
                country = annual_profile["country"]
                require(
                    country in annual_selected, "Embedded assessment country outside base catalogue"
                )
                previous = annual_selected[country]
                # Country snapshots may have different panel hashes at one clock;
                # equal-clock profiles must agree on the evidence they display.
                if clock(previous[0]["as_known_at"]) == clock(embedded["as_known_at"]):
                    require(
                        previous[1] == annual_profile,
                        "Ambiguous annual country profiles at same cutoff",
                    )
                elif clock(embedded["as_known_at"]) > clock(previous[0]["as_known_at"]):
                    annual_selected[country] = embedded, annual_profile
        for profile in snapshot["countries"]:
            country = profile["country"]
            require(country in annual_selected, "Monitoring country has no annual assessment")
            previous = selected.get(country)
            if previous and clock(previous[0]["as_known_at"]) == clock(snapshot["as_known_at"]):
                require(
                    previous[0]["snapshot_sha256"] == snapshot["snapshot_sha256"],
                    "Ambiguous monitoring profiles at same cutoff",
                )
            if previous is None or clock(snapshot["as_known_at"]) > clock(
                previous[0]["as_known_at"]
            ):
                selected[country] = snapshot, profile
    result = {}
    for country, (annual_snapshot, profile) in annual_selected.items():
        annual_sources = {c["country"]: c for c in annual_snapshot["source_evidence"]["countries"]}
        annual = envelope(annual_snapshot, profile)
        annual.update(
            baseline_year=annual_snapshot["baseline_year"],
            horizon_end_year=annual_snapshot["horizon_end_year"],
            histories=[],
        )
        points = {**profile["baseline"], **profile.get("structural", {})}
        for indicator, point in points.items():
            if point is None:
                continue
            matches = [
                s
                for s in annual_sources[country]["series"]
                if s["release_id"] == point["release_id"] and s["indicator"] == indicator
            ]
            require(len(matches) == 1, "Annual native history missing/ambiguous")
            meta = annual_snapshot["methodology"]["metrics"][indicator]
            annual["histories"].append(
                {
                    **deepcopy(matches[0]),
                    "country": country,
                    "unit": meta["unit"],
                    "label": meta["label"],
                    "frequency": "annual",
                }
            )
        monitor = None
        if country in selected:
            snapshot, monitoring_profile = selected[country]
            monitor = envelope(snapshot, monitoring_profile)
            monitor.update(
                histories=[],
                input_gaps=[
                    deepcopy(g)
                    for g in snapshot.get("input_gaps", [])
                    if g.get("country") == country
                ],
                remaining_gaps=deepcopy(snapshot.get("remaining_gaps", [])),
            )
            source_histories = histories(snapshot["source_evidence"])
            for signal in monitoring_profile["signals"]:
                candidates = {
                    sha(canonical(s)): s
                    for s in source_histories
                    if s["country"] == country and s["indicator"] == signal["indicator"]
                }
                require(len(candidates) <= 1, "Ambiguous native monitoring history")
                if signal.get("latest") is not None:
                    require(len(candidates) == 1, "Monitoring native source history missing")
                if candidates:
                    series = next(iter(candidates.values()))
                    require(
                        clock(series["available_at"]) <= clock(snapshot["as_known_at"]),
                        "History available after cutoff",
                    )
                    if signal.get("latest"):
                        latest = signal["latest"]
                        matching = [
                            r for r in series["observations"] if r["date"] == latest["date"]
                        ]
                        require(
                            len(matching) == 1
                            and all(
                                matching[0].get(k) == latest.get(k)
                                for k in ("value", "status", "period")
                            ),
                            "Monitoring latest differs from native history",
                        )
                    monitor["histories"].append(series)
        result[country] = {
            "version": 1,
            "country": country,
            "name": profile["name"],
            "listing_iso2": profile.get("listing_iso2", country),
            "assessment": annual,
            "monitoring": monitor,
        }
    return result


def protected_paths(value):
    """Collect retained file references, including nested assessment parents/gaps."""
    result = set()
    if isinstance(value, list):
        for child in value:
            result.update(protected_paths(child))
    elif isinstance(value, dict):
        for key, child in value.items():
            paths = (
                child
                if key == "protected_artifact_paths"
                else [child]
                if key in {"path", "artifact_path"} and isinstance(child, str)
                else []
            )
            for path in paths:
                require(
                    isinstance(path, str) and Path(path).is_absolute(),
                    "Protected artifact path must be absolute",
                )
                result.add(Path(path).resolve())
            if isinstance(child, (dict, list)):
                result.update(protected_paths(child))
    return result


def protect_output(output, sources, documents):
    """Reject overlapping destinations before staging or replacing any files."""
    lexical = output.absolute()
    require(
        not any(p.is_symlink() for p in (lexical, *lexical.parents)),
        "Output has symlinked ancestry",
    )
    destination = output.resolve()
    retained = set(sources) | protected_paths(documents)
    require(
        all(
            destination != p and destination not in p.parents and p not in destination.parents
            for p in retained
        ),
        "Output would overwrite or contain a protected source",
    )
    namespaces = {
        parent for path in retained for parent in path.parents if parent.name == "artifacts"
    }
    require(
        all(destination != root and root not in destination.parents for root in namespaces),
        "Output enters a protected artifact namespace",
    )


def export_pack(assessment_path: Path, monitoring_paths: list[Path], output: Path):
    sources = [Path(assessment_path).resolve(), *(Path(p).resolve() for p in monitoring_paths)]
    output = Path(output)
    require(len(sources) == len(set(sources)), "Duplicate source snapshot paths")
    before = {p: (sha(p.read_bytes()), p.stat().st_mtime_ns) for p in sources}
    documents = [read(p) for p in sources]
    protect_output(output, sources, documents)
    countries = project(documents[0], documents[1:])
    files = {f"countries/{code}.json": canonical(document) for code, document in countries.items()}
    source_records = [
        {
            "kind": "assessment" if i == 0 else "monitoring",
            "file": p.name,
            "snapshot_sha256": doc["snapshot_sha256"],
            "file_sha256": before[p][0],
            "as_of": doc["as_of"],
            "as_known_at": doc["as_known_at"],
        }
        for i, (p, doc) in enumerate(zip(sources, documents, strict=True))
    ]
    base = {
        "format": FORMAT,
        "version": 1,
        "sources": sorted(
            source_records, key=lambda s: (s["kind"], s["as_known_at"], s["snapshot_sha256"])
        ),
        "countries": {
            code: {
                "name": d["name"],
                "listing_iso2": d["listing_iso2"],
                "assessment_as_of": d["assessment"]["as_of"],
                "monitoring_as_of": d["monitoring"]["as_of"] if d["monitoring"] else None,
                "signals": len(d["monitoring"]["profile"]["signals"]) if d["monitoring"] else 0,
                "sha256": sha(files[f"countries/{code}.json"]),
                "bytes": len(files[f"countries/{code}.json"]),
            }
            for code, d in countries.items()
        },
    }
    identity = sha(canonical(base))
    index = {
        **base,
        "id": identity,
        "countries": {
            code: {**entry, "file": f"{identity}/countries/{code}.json"}
            for code, entry in base["countries"].items()
        },
    }
    directory = output / identity
    if directory.exists():
        require(not directory.is_symlink(), "Immutable pack is a symlink")
        actual = {
            p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        require(
            actual == files and not any(p.is_symlink() for p in directory.rglob("*")),
            "Existing immutable pack differs",
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".country-evidence-", dir=output))
        try:
            for filename, body in files.items():
                target = stage / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            stage.rename(directory)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    require(
        all(
            (sha(p.read_bytes()), p.stat().st_mtime_ns) == original
            for p, original in before.items()
        ),
        "Source snapshot changed during export",
    )
    pointer = output / "index.json"
    require(not pointer.is_symlink(), "Index cannot be a symlink")
    body = canonical(index)
    if not pointer.exists() or pointer.read_bytes() != body:
        handle, name = tempfile.mkstemp(prefix=".index-", dir=output)
        os.close(handle)
        temporary = Path(name)
        try:
            temporary.write_bytes(body)
            temporary.replace(pointer)
        finally:
            temporary.unlink(missing_ok=True)
    verify_pack(output)
    return pointer


def verify_pack(output):
    output = Path(output)
    index = read(output / "index.json")
    base = {k: deepcopy(v) for k, v in index.items() if k != "id"}
    for entry in base["countries"].values():
        entry.pop("file")
    require(
        index["format"] == FORMAT and index["id"] == sha(canonical(base)),
        "Country evidence index hash mismatch",
    )
    signals = 0
    for country, entry in index["countries"].items():
        require(
            entry["file"] == f"{index['id']}/countries/{country}.json", "Unsafe country file path"
        )
        path = output / entry["file"]
        require(not path.is_symlink() and path.is_file(), "Country file missing/symlink")
        body = path.read_bytes()
        require(
            len(body) == entry["bytes"] and sha(body) == entry["sha256"],
            "Country file hash mismatch",
        )
        data = json.loads(body)
        require(data["country"] == country, "Country file identity mismatch")
        signals += entry["signals"]
    return {"countries": len(index["countries"]), "monitoring_signals": signals, "id": index["id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--monitoring", type=Path, action="append", default=[])
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parents[1] / "public/data/country-evidence"
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        require(args.assessment is not None, "--assessment is required for export")
        export_pack(args.assessment, args.monitoring, args.output)
    print(json.dumps(verify_pack(args.output), indent=2))


if __name__ == "__main__":
    main()
