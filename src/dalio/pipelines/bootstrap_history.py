"""Seed the point-in-time ledger from the legacy latest-value table.

The old database cannot reveal when historical revisions became available. This
command therefore requires an explicit conservative cutover timestamp and never
uses observation dates as publication dates.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime

from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.releases import bootstrap_current_observations


def _parse_timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use an ISO-8601 timestamp, for example 2026-09-08T12:00:00Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--available-at",
        required=True,
        type=_parse_timestamp,
        help=(
            "Conservative time at which the legacy rows are treated as known "
            "(ISO-8601, normally the cutover time)."
        ),
    )
    args = parser.parse_args()

    retrieved_at = datetime.now(UTC)
    if args.available_at > retrieved_at:
        parser.error("--available-at cannot be in the future")

    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        releases, rows = bootstrap_current_observations(
            session,
            available_at=args.available_at,
            retrieved_at=retrieved_at,
        )
    print(f"History bootstrap: {releases} new releases covering {rows} current rows.")
    if releases == 0 and rows:
        print("No changes: matching release snapshots already exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
