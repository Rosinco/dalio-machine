"""Fetch official SCB government-debt holder positions into the typed ledger."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from urllib.parse import quote

from sqlalchemy import Engine

from dalio.data_sources.scb_financial_accounts import (
    SCB_GOVERNMENT_DEBT_HOLDERS,
    SOURCE_SCB_FINANCIAL_ACCOUNTS,
    ScbFinancialAccountsSource,
    ScbFinancialAccountsSpec,
)
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.debt import ingest_debt_holder_snapshot
from dalio.storage.releases import ReleaseMeta

logger = logging.getLogger(__name__)


def partition_key_for(
    spec: ScbFinancialAccountsSpec = SCB_GOVERNMENT_DEBT_HOLDERS,
) -> str:
    """Stable identity for the complete multidimensional SCB selection."""
    parts = (
        SOURCE_SCB_FINANCIAL_ACCOUNTS,
        spec.table_id,
        spec.country,
        spec.issuer_sector_code,
        "+".join(spec.instrument_codes),
        spec.measure_code,
    )
    return "facts:" + ":".join(quote(str(part), safe="") for part in parts)


def run_pipeline(
    *,
    source: ScbFinancialAccountsSource | None = None,
    spec: ScbFinancialAccountsSpec = SCB_GOVERNMENT_DEBT_HOLDERS,
    use_cache: bool = True,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, object]:
    """Fetch and append one complete SCB holder-position release."""
    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    adapter = source or ScbFinancialAccountsSource()

    try:
        frame = adapter.fetch(spec, use_cache=use_cache)
        if frame.empty:
            raise ValueError("SCB debt-holder release snapshot is empty")
        with session_factory() as session:
            result = ingest_debt_holder_snapshot(
                session,
                frame,
                ReleaseMeta(
                    partition_key=partition_key_for(spec),
                    source_family=SOURCE_SCB_FINANCIAL_ACCOUNTS,
                    available_at=run_at,
                    retrieved_at=run_at,
                    vintage_label="SCB TAB1203 complete response",
                    source_url=spec.url,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports a safe per-source failure
        logger.exception("Failed SCB government-debt holder refresh: %s", exc)
        return {"error": str(exc)}

    logger.info(
        "Stored SCB government-debt holders: %d rows (release %d, created=%s)",
        result.row_count,
        result.release_id,
        result.created,
    )
    return {
        "source": SOURCE_SCB_FINANCIAL_ACCOUNTS,
        "rows": result.row_count,
        "release_id": result.release_id,
        "created": result.created,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch official SCB central-government debt holders by sector."
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass the local response cache.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = run_pipeline(use_cache=not args.no_cache)
    if "error" in result:
        print(f"SCB government-debt holders failed: {result['error']}")
        return 1
    action = "created" if result["created"] else "already stored"
    print(
        f"SCB government-debt holders: {result['rows']} rows; "
        f"release {result['release_id']} ({action})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
