"""Build a release-traceable liquidity snapshot and a plain-language brief.

The command is read-only with respect to the database.  It writes generated
JSON and Markdown files beneath ``data/snapshots`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine

from dalio.indicators.liquidity import build_liquidity_snapshot, validate_liquidity_snapshot
from dalio.storage.db import make_session_factory

LATEST_JSON = "liquidity_latest.json"
LATEST_MARKDOWN = "liquidity_latest.md"


def _parse_known_at(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("--known-at must include a timezone offset")
    return parsed.astimezone(UTC)


def _read_only_engine(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database does not exist: {resolved}")
    return create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{resolved}?mode=ro", uri=True),
        future=True,
    )


def _atomic_write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _fmt(value: Any, digits: int = 2, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


def _signed(value: Any, digits: int = 2, *, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):+,.{digits}f}{suffix}"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def safe(value: str) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(safe(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(safe(value) for value in row) + " |" for row in rows),
    ]


def _headline_lines(snapshot: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    summary = snapshot["money_summary"]
    if summary["common_period"] is not None:
        lines.append(
            f"- Broad money is growing in {summary['positive_growth_breadth']}/"
            f"{summary['expected']} covered economies; {summary['accelerating_breadth']}/"
            f"{summary['expected']} accelerated over the latest three months. The equal-country "
            f"median annual log growth rate is {_fmt(summary['median_annual_log_growth_pct'], 2, suffix='%')}."
        )
    mmf = snapshot["mmf"]
    if mmf["availability_status"] == "ready":
        lines.append(
            f"- US MMF assets grew {_fmt(mmf['mmf_annual_log_growth_pct'], 2, suffix='%')} "
            f"and outpaced M2 by {_signed(mmf['mmf_minus_m2_growth_gap_pp'], 2, suffix=' pp')}. "
            "That is relative expansion, not proof that deposits moved into funds."
        )
    repo = snapshot["repo"]
    if repo["availability_status"] == "ready":
        lines.append(
            f"- Repo venue-rate dispersion is {_fmt(repo['fragmentation_5d_median_bp'], 1, suffix=' bp')} "
            f"on a five-observation median (robust z {_signed(repo['fragmentation_robust_z'], 2)}). "
            "Volumes are shown separately because high activity is not automatically stress."
        )
    usd = next(
        (
            row
            for row in snapshot["offshore_credit"]
            if row.get("currency") == "USD" and row["availability_status"] == "ready"
        ),
        None,
    )
    if usd is not None:
        direction = "accelerated" if usd["acceleration_1q_pp"] > 0 else "decelerated"
        lines.append(
            f"- Offshore USD credit grew {_fmt(usd['annual_log_growth_pct'], 2, suffix='%')} year on year "
            f"and {direction} by {_fmt(abs(usd['acceleration_1q_pp']), 2, suffix=' pp')} in the latest quarter."
        )
    return lines or [
        "- The required inputs are incomplete; no headline interpretation is produced."
    ]


def render_markdown(snapshot: dict[str, Any]) -> str:
    """Render a bounded interpretation whose numbers all exist in the JSON."""

    validate_liquidity_snapshot(snapshot)

    lines = [
        f"# Liquidity brief — {snapshot['as_of']}",
        "",
        "> A set of separate monetary and funding diagnostics, not a universal M5, "
        "liquidity score, market-timing signal, or portfolio instruction.",
        "",
        f"Known-at cutoff: `{snapshot['as_known_at']}`  ",
        f"Methodology: `{snapshot['methodology_version']}` / `{snapshot['methodology_sha256']}`  ",
        f"Snapshot: `{snapshot['snapshot_sha256']}`",
        "",
        "## Current reading",
        "",
        *_headline_lines(snapshot),
        "",
        "## Broad-money impulse",
        "",
        "Annual log growth compares the same calendar month one year earlier. Acceleration "
        "is the change in that growth rate over three months. Currency levels are not combined.",
        "",
    ]
    money_rows = []
    for row in snapshot["broad_money"]:
        if row["availability_status"] != "ready":
            money_rows.append([row["metric_id"], "—", "—", "—", row["missing_reason"]])
            continue
        money_rows.append(
            [
                row["country"],
                row["period"],
                _fmt(row["annual_log_growth_pct"], 2, suffix="%"),
                _signed(row["acceleration_3m_pp"], 2, suffix=" pp"),
                row["movement"].replace("_", " "),
            ]
        )
    lines.extend(
        _table(
            ["Economy", "Period", "YoY log growth", "3m acceleration", "Arithmetic direction"],
            money_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Money versus central-bank assets",
            "",
            "Positive gaps mean broad money grew faster than the central-bank balance sheet. "
            "They do not establish what caused money growth.",
            "",
        ]
    )
    cb_rows = []
    for row in snapshot["central_bank_divergence"]:
        if row["availability_status"] != "ready":
            cb_rows.append([row["metric_id"], "—", "—", "—", row["missing_reason"]])
            continue
        cb_rows.append(
            [
                row["country"],
                _fmt(row["money_annual_log_growth_pct"], 2, suffix="%"),
                _fmt(row["central_bank_assets_annual_log_growth_pct"], 2, suffix="%"),
                _signed(row["money_minus_assets_growth_gap_pp"], 2, suffix=" pp"),
                _signed(row["gap_change_3m_pp"], 2, suffix=" pp"),
            ]
        )
    lines.extend(
        _table(
            ["Economy", "Money growth", "CB assets growth", "Money − CB gap", "Gap change / 3m"],
            cb_rows,
        )
    )

    lines.extend(["", "## US money-market funds", ""])
    mmf = snapshot["mmf"]
    if mmf["availability_status"] != "ready":
        lines.append(f"Unavailable: {mmf['missing_reason']}")
    else:
        lines.extend(
            [
                f"At {mmf['period']}, MMF assets were {_fmt(mmf['mmf_assets_to_m2_scale_pct'], 2, suffix='%')} "
                f"of the numerical M2 scale, {_signed(mmf['scale_change_12m_pp'], 2, suffix=' pp')} over twelve months. "
                "This ratio is context only: the universes overlap and are not definitionally equivalent.",
                "",
                *_table(
                    ["MMF asset category", "Share of total", "12m share change"],
                    [
                        [
                            row["title"],
                            _fmt(row["share_of_total_pct"], 2, suffix="%"),
                            _signed(row["share_change_12m_pp"], 2, suffix=" pp"),
                        ]
                        for row in mmf["asset_allocation"]
                    ],
                ),
                "",
            ]
        )
        categories = mmf["published_repo_counterparty_categories"]
        lines.extend(["", "### Published counterparty/clearing-category ratios", ""])
        if categories["availability_status"] != "ready":
            lines.append(f"Unavailable: {categories['missing_reason']}")
        else:
            lines.extend(
                [
                    f"Latest common reported month: `{categories['period']}`. These publisher "
                    "categories are not an additive liquidity total or an end-borrower cash-flow "
                    "map; FICC is a clearing category.",
                    "",
                    *_table(
                        ["Published category", "Share of MMF repo", "12m share change"],
                        [
                            [
                                row["title"],
                                _fmt(row["share_of_repo_pct"], 2, suffix="%"),
                                _signed(row["share_change_12m_pp"], 2, suffix=" pp"),
                            ]
                            for row in categories["ratios"]
                        ],
                    ),
                ]
            )

    lines.extend(["", "## Repo pricing and activity", ""])
    repo = snapshot["repo"]
    if repo["availability_status"] != "ready":
        lines.append(f"Unavailable: {repo['missing_reason']}")
    else:
        lines.extend(
            [
                f"Exact common business date: `{repo['period_date']}`; EFFR {_fmt(repo['effr_pct'], 2, suffix='%')}. "
                f"Five-observation median fragmentation is {_fmt(repo['fragmentation_5d_median_bp'], 1, suffix=' bp')}; "
                f"maximum policy-relative premium is {_fmt(repo['maximum_effr_premium_5d_median_bp'], 1, suffix=' bp')}. "
                "This describes selected venues and does not constitute a stress verdict.",
                "",
                *_table(
                    ["Venue", "Rate", "5-observation EFFR premium", "Status"],
                    [
                        [
                            row["venue"],
                            _fmt(row["rate_pct"], 2, suffix="%"),
                            _signed(row["effr_premium_5d_median_bp"], 1, suffix=" bp"),
                            row["status"],
                        ]
                        for row in repo["venues"]
                    ],
                ),
                "",
                *_table(
                    ["Venue measure", "Latest", "20-observation median", "Status"],
                    [
                        [
                            row.get("title", row["series_id"]),
                            (
                                _fmt(row.get("latest_value") / 1_000_000_000, 1, suffix=" USD bn")
                                if row.get("latest_value") is not None
                                else "—"
                            ),
                            (
                                _fmt(
                                    row.get("trailing_observation_median") / 1_000_000_000,
                                    1,
                                    suffix=" USD bn",
                                )
                                if row.get("trailing_observation_median") is not None
                                else "—"
                            ),
                            row.get("status", "unavailable"),
                        ]
                        for row in repo["volume_context"]
                    ],
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Offshore reserve-currency credit",
            "",
            "These are separate native-currency credit stocks to nonbanks outside each issuing "
            "currency area. They are not converted or summed.",
            "",
        ]
    )
    offshore_rows = []
    for row in snapshot["offshore_credit"]:
        if row["availability_status"] != "ready":
            offshore_rows.append([row["metric_id"], "—", "—", "—", row["missing_reason"]])
            continue
        offshore_rows.append(
            [
                row["currency"],
                row["period"],
                _fmt(row["annual_log_growth_pct"], 2, suffix="%"),
                _signed(row["acceleration_1q_pp"], 2, suffix=" pp"),
                row["movement"].replace("_", " "),
            ]
        )
    lines.extend(
        _table(
            ["Currency", "Period", "YoY log growth", "1q acceleration", "Arithmetic direction"],
            offshore_rows,
        )
    )

    lines.extend(["", "## Horizon use", ""])
    for row in snapshot["horizons"]:
        lines.append(f"- **{row['horizon']}:** {row['supported_context']}.")

    lines.extend(
        [
            "",
            "## Evidence and limits",
            "",
            f"The calculations reference `{len(snapshot['input_releases'])}` immutable input releases. "
            f"The earliest selected liquidity input became available at "
            f"`{snapshot['earliest_input_available_at']}`; the complete selected-input snapshot "
            f"became available at `{snapshot['complete_snapshot_available_at']}`. "
            "Older observation dates currently describe history contained in those releases; they are not a real-time backtest.",
            "",
            *[f"- {item}" for item in snapshot["interpretation_limits"]],
            "",
        ]
    )
    return "\n".join(lines)


def _complete(snapshot: dict[str, Any]) -> bool:
    return all(section["ready"] == section["expected"] for section in snapshot["coverage"].values())


def run(
    *,
    db_path: Path,
    output_dir: Path,
    as_of: date,
    as_known_at: datetime,
    require_complete: bool = True,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    engine = _read_only_engine(db_path)
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        snapshot = build_liquidity_snapshot(
            session,
            as_of=as_of,
            as_known_at=as_known_at,
        )
    if require_complete and not _complete(snapshot):
        raise RuntimeError(f"liquidity overview is incomplete: {snapshot['coverage']}")

    validate_liquidity_snapshot(snapshot)
    json_payload = (
        json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    markdown = render_markdown(snapshot)
    dated_stem = f"liquidity_{as_of.isoformat()}_{snapshot['snapshot_sha256'][:16]}"
    paths = (
        _atomic_write(output_dir / LATEST_JSON, json_payload),
        _atomic_write(output_dir / LATEST_MARKDOWN, markdown),
        _atomic_write(output_dir / f"{dated_stem}.json", json_payload),
        _atomic_write(output_dir / f"{dated_stem}.md", markdown),
    )
    return snapshot, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build separate, point-in-time liquidity diagnostics and a Markdown brief."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database (default: DALIO_DB_PATH or data/dalio.db).",
    )
    parser.add_argument(
        "--as-of", "--through-date", dest="as_of", type=date.fromisoformat, default=None
    )
    parser.add_argument(
        "--known-at", "--as-known-at", dest="known_at", type=_parse_known_at, default=None
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: DALIO_LIQUIDITY_BRIEF_DIR or data/snapshots).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write an explicitly incomplete overview instead of failing closed.",
    )
    args = parser.parse_args(argv)
    load_dotenv()

    db_path = args.db or Path(os.environ.get("DALIO_DB_PATH", "data/dalio.db"))
    output_dir = args.out_dir or Path(os.environ.get("DALIO_LIQUIDITY_BRIEF_DIR", "data/snapshots"))
    known_at = args.known_at or datetime.now(UTC)
    as_of = args.as_of or known_at.date()
    try:
        snapshot, paths = run(
            db_path=db_path,
            output_dir=output_dir,
            as_of=as_of,
            as_known_at=known_at,
            require_complete=not args.allow_partial,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"dalio-liquidity-brief: {exc}", file=sys.stderr)
        return 2

    print(f"Liquidity brief as of {snapshot['as_of']} / known at {snapshot['as_known_at']}")
    for name, coverage in snapshot["coverage"].items():
        print(f"  {name:<24} {coverage['ready']}/{coverage['expected']} ready")
    for path in paths:
        print(f"  wrote {path}")
    print("  composite score: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
