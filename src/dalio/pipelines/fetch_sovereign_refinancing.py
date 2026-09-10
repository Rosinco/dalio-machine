"""Collect all 31 harmonized refinancing histories before one atomic DB refresh."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine

from dalio.data_sources.ecb_refinancing import EcbRefinancingSource
from dalio.data_sources.eurostat_refinancing import (
    EurostatRefinancingSeries,
    EurostatRefinancingSource,
)
from dalio.storage.db import make_engine
from dalio.storage.refinancing import (
    PreparedRefinancingPartition,
    ingest_refinancing_batch,
    load_stored_refinancing_batch,
    refinancing_bindings,
    validate_partition,
    write_refinancing_catalogue,
)

logger = logging.getLogger(__name__)


def prepare_batch(
    *,
    eurostat_source: EurostatRefinancingSource | None = None,
    ecb_source: EcbRefinancingSource | None = None,
    artifact_dir: Path = Path("data/artifacts/debt_refinancing/catalogues"),
    use_cache: bool = False,
    retrieved_at: datetime | None = None,
) -> tuple[PreparedRefinancingPartition, ...]:
    """Fetch and validate every source; never open or initialize a database."""
    eurostat = eurostat_source or EurostatRefinancingSource()
    ecb = ecb_source or EcbRefinancingSource()
    bindings = refinancing_bindings()
    frames = []
    for index, binding in enumerate(bindings, start=1):
        logger.info("Fetching %s/31: %s", index, binding.partition.partition_id)
        source = eurostat if isinstance(binding.spec, EurostatRefinancingSeries) else ecb
        frames.append(source.fetch(binding.spec, use_cache=use_cache))
    # Availability is when the entire batch was obtained, never a backdated
    # reference period or an assumed publisher release timestamp.
    run_at = retrieved_at or datetime.now(UTC)
    catalogue_path = write_refinancing_catalogue(artifact_dir)
    return tuple(
        validate_partition(binding, frame, retrieved_at=run_at, catalogue_path=catalogue_path)
        for binding, frame in zip(bindings, frames, strict=True)
    )


def run_pipeline(*, engine: Engine | None = None, **kwargs) -> dict:
    batch = prepare_batch(**kwargs)
    return ingest_refinancing_batch(batch, engine=engine or make_engine())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Explicit target SQLite database")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--use-cache", action="store_true", help="Use validated local response cache"
    )
    source_group.add_argument(
        "--from-db", type=Path, help="Promote verified staging evidence offline without downloading"
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=Path("data/artifacts/debt_refinancing")
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/debt_refinancing"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.from_db:
            if not args.from_db.is_file():
                raise ValueError(f"Staging database does not exist: {args.from_db}")
            source_engine = create_engine(
                "sqlite://",
                creator=lambda: sqlite3.connect(
                    args.from_db.resolve().as_uri() + "?mode=ro", uri=True
                ),
            )
            try:
                batch = load_stored_refinancing_batch(source_engine)
            finally:
                source_engine.dispose()
            result = ingest_refinancing_batch(batch, engine=make_engine(args.db))
        else:
            result = run_pipeline(
                engine=make_engine(args.db),
                use_cache=args.use_cache,
                eurostat_source=EurostatRefinancingSource(
                    artifact_dir=args.evidence_root / "eurostat",
                    cache_dir=args.cache_root / "eurostat",
                ),
                ecb_source=EcbRefinancingSource(
                    artifact_dir=args.evidence_root / "ecb", cache_dir=args.cache_root / "ecb"
                ),
                artifact_dir=args.evidence_root / "catalogues",
            )
    except Exception as exc:  # noqa: BLE001 - report fail-closed CLI failures
        logger.exception("Refinancing batch failed: %s", exc)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
