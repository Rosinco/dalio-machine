"""Verify packaged data against the canonical source and country map before release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(source: Path, public: Path) -> None:
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    index = json.loads((public / "data/index.json").read_text())
    assert index["manifest"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert index["as_of"] == raw["as_of"]
    assert index["ranking_population"] == raw["ranking_population"]
    assert index["trade"] == raw["trade"]
    history = json.loads((public / "data/history.json").read_text())
    world = json.loads((public / "maps/world.geojson").read_text())
    map_codes = {f["properties"]["code"] for f in world["features"]}
    for code, original in raw["countries"].items():
        contents = (public / "data/countries" / f"{code}.json").read_bytes()
        assert hashlib.sha256(contents).hexdigest() == index["manifest"]["country_files"][code]
        assert json.loads(contents) == original, f"Changed evidence for {code}"
        assert history[code] == original["history"]
        for key in ("categories", "indicators", "pressures"):
            assert index["countries"][code][key] == original[key]
        if original["on_map"]:
            assert code in map_codes, f"Missing map geometry for {code}"
    print(f"PASS: {len(raw['countries'])} exact country exports; all mapped countries have geometry; source hashes and histories agree.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=Path(__file__).resolve().parents[1] / "public")
    args = parser.parse_args()
    verify(args.source, args.public)
