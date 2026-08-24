"""Score the World Fundamentals Map and write the snapshot the app renders.

    dalio-score                       # → data/snapshots/fundamentals_latest.json (+ dated copy)
    dalio-score --as-of 2026-06-30    # score as of a past date (history replay)
    dalio-score --out /tmp/snap.json

The app never touches the DB for the fundamentals page; it reads this file
(cached on mtime). Re-run after every fetch.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from dalio.scoring.fundamentals import build_snapshot, jurisdiction_table, write_snapshot
from dalio.storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)


def snapshot_dir() -> Path:
    return Path(os.environ.get("FUNDAMENTALS_DIR", "data/snapshots"))


def run(as_of: date | None = None, out: Path | None = None) -> tuple[Path, dict]:
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        snap = build_snapshot(session, as_of=as_of)

    target = out or (snapshot_dir() / "fundamentals_latest.json")
    write_snapshot(snap, target)
    if out is None:
        write_snapshot(snap, snapshot_dir() / f"fundamentals_{snap['as_of']}.json")
        # Slice 25: the one output that touches real money — §0.2 pre-triage input.
        jurisdiction_table(snap).to_csv(snapshot_dir() / "jurisdiction_tier.csv", index=False)
    return target, snap


def _print_coverage(snap: dict) -> None:
    cov = snap["coverage"]
    print(f"\nSnapshot as of {snap['as_of']}: {cov['filled']}/{cov['cells']} cells filled")
    for name, n in cov["by_indicator"].items():
        print(f"  {name:<28} {n:>2}/{len(snap['countries'])} countries")
    missing = [
        iso2 for iso2, c in snap["countries"].items()
        if all(cell["value"] is None for cell in c["indicators"].values())
    ]
    if missing:
        print(f"  no data at all: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fundamentals snapshot JSON.")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    path, snap = run(as_of=args.as_of, out=args.out)
    print(f"dalio-score — wrote {path}")
    _print_coverage(snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
