"""Copy one validated Dalio snapshot into a bounded, lazy-loaded desktop data pack.

Reads the source snapshot only; never opens or changes the canonical database.
Run with dalio-machine's installed Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dalio.app.fundamentals.snapshot import parse_snapshot


def export(source: Path, output: Path) -> dict:
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    parse_snapshot(raw)
    output.mkdir(parents=True, exist_ok=True)
    (output / "countries").mkdir(exist_ok=True)
    files = {}
    summaries = {}
    histories = {}
    for code, country in raw["countries"].items():
        if not code.isascii() or not code.isalpha() or len(code) != 2:
            raise ValueError(f"Invalid country code: {code!r}")
        encoded = json.dumps(country, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        (output / "countries" / f"{code}.json").write_bytes(encoded)
        files[code] = hashlib.sha256(encoded).hexdigest()
        summaries[code] = {k: v for k, v in country.items() if k != "history"}
        histories[code] = country["history"]
    index = {k: v for k, v in raw.items() if k != "countries"}
    index["countries"] = summaries
    index["manifest"] = {
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_file": source.name,
        "source_bytes": len(source_bytes),
        "country_files": files,
    }
    # Compact annual history is fetched only when opening History mode.
    for name, obj in (("index", index), ("history", histories)):
        (output / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return index["manifest"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "public" / "data")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
