"""Build the offline financial companion pack from saved, validated report files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from financial_model import COLUMNS, create_database, encode, pack_report, store_company
from taxonomy_model import file_record


def export(root: Path, taxonomy: Path, output: Path):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Financial output must be outside the Börsdata source project")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "framework"))
    import pandas as pd
    from core.data import read_validated
    from core.schemas import YEARLY_REPORTS_SCHEMA

    raw = json.loads(taxonomy.read_bytes())
    catalogue = raw["catalogue"]
    ids = set(map(int, catalogue["listings"]))
    frames, sources = [], []
    for snapshot in catalogue["snapshots"]:
        folder = Path(snapshot["instruments"]["path"]).parent.parent
        for frequency, stem in [("annual", "yearly"), ("quarterly", "quarterly")]:
            path = root / folder / f"all_reports/all_{stem}_reports.parquet"
            columns = [
                "ins_id",
                "year",
                "period",
                "report_start_date",
                "report_end_date",
                "report_date",
                "currency",
                "currency_ratio",
                *COLUMNS,
            ]
            frame = read_validated(path, YEARLY_REPORTS_SCHEMA, columns=columns)
            if frame.duplicated(["ins_id", "year", "period"]).any():
                raise ValueError(f"Duplicate fiscal periods in {path}")
            if not all(
                pd.api.types.is_numeric_dtype(frame[c]) for c in (*COLUMNS, "currency_ratio")
            ):
                raise ValueError("Financial columns must be numeric")
            identifier = f"{snapshot['as_of']}-{frequency}"
            sources.append(
                {
                    **file_record(root, path),
                    "id": identifier,
                    "as_of": snapshot["as_of"],
                    "frequency": frequency,
                    "rows": len(frame),
                    "outside_directory": int((~frame.ins_id.isin(ids)).sum()),
                }
            )
            frame = frame[frame.ins_id.isin(ids)].copy()
            # Preserve invalid source dates as strings for explicit withholding, including
            # years outside pandas' nanosecond range. Never coerce them into plausible dates.
            for c in ("report_start_date", "report_end_date", "report_date"):
                frame[c] = frame[c].astype(str).str.slice(0, 10)
            frame["source_id"] = identifier
            frame["source_as_of"] = snapshot["as_of"]
            frame["frequency"] = frequency
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.sort_values("source_as_of", kind="stable").drop_duplicates(
        ["ins_id", "year", "period"], keep="last"
    )
    combined = combined.sort_values(["ins_id", "year", "period"])
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="financial-export-", dir=output))
    path = staging / "pack.sqlite"
    conn = create_database(path)
    coverage = {}
    totals = {
        "listings": len(ids),
        "with_reports": 0,
        "annual": 0,
        "quarterly": 0,
        "withheld": 0,
        "source_rows": sum(s["rows"] for s in sources),
        "outside_directory": sum(s["outside_directory"] for s in sources),
        "superseded": before - len(combined),
    }
    source_rows = {s["id"]: {"usable": 0, "withheld": 0} for s in sources}
    try:
        groups = {str(key): frame for key, frame in combined.groupby("ins_id", sort=False)}
        for identifier in sorted(catalogue["listings"], key=int):
            payload = {"id": identifier, "annual": [], "quarterly": [], "withheld": []}
            frame = groups.get(identifier)
            if frame is not None:
                for values in frame.itertuples(index=False, name=None):
                    row = dict(zip(frame.columns, values, strict=True))
                    packed, issue = pack_report(
                        row,
                        row["source_as_of"],
                        row["source_id"],
                        annual=row["frequency"] == "annual",
                    )
                    if issue:
                        payload["withheld"].append(issue)
                        totals["withheld"] += 1
                        source_rows[row["source_id"]]["withheld"] += 1
                    else:
                        payload[row["frequency"]].append(packed)
                        totals[row["frequency"]] += 1
                        source_rows[row["source_id"]]["usable"] += 1
            coverage[identifier] = store_company(conn, payload)
            totals["with_reports"] += bool(payload["annual"] or payload["quarterly"])
        metadata = {
            "format": "macro-atlas-financials",
            "version": 1,
            "taxonomy_sha256": hashlib.sha256(taxonomy.read_bytes()).hexdigest(),
            "as_of": catalogue["as_of"],
            "generated_at": datetime.now(UTC).isoformat(),
            "columns": list(COLUMNS),
            "sources": [{**s, **source_rows[s["id"]]} for s in sources],
            "summary": totals,
            "companies": coverage,
        }
        assert (
            totals["source_rows"]
            == totals["outside_directory"]
            + totals["superseded"]
            + totals["annual"]
            + totals["quarterly"]
            + totals["withheld"]
        )
        conn.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("index", gzip.compress(encode(metadata), mtime=0)),
        )
        conn.commit()
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    with path.open("rb") as f:
        identity = hashlib.file_digest(f, "sha256").hexdigest()
    destination = output / f"{identity}.sqlite"
    if destination.exists():
        raise ValueError("This financial pack already exists")
    os.replace(path, destination)
    staging.rmdir()
    record = {
        "id": identity,
        "bytes": destination.stat().st_size,
        "taxonomy_sha256": metadata["taxonomy_sha256"],
        "as_of": metadata["as_of"],
    }
    (output / "catalog.json").write_bytes(encode({"version": 1, "packs": [record]}))
    (output / "coverage-audit.json").write_bytes(
        encode({"pack": record, "summary": totals, "sources": metadata["sources"]})
    )
    return {**record, **totals}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borsdata-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.borsdata_root, args.taxonomy, args.output)))
