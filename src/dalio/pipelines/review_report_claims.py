"""Prepare, check and explicitly apply human report-claim decisions.

Packet construction remains read-only.  The ``apply`` subcommand is a separate
local-human boundary: it refuses non-TTY input, shows every candidate/outcome,
requires the canonical full-decision hash to be typed, makes a verified SQLite
backup, and only then appends an all-or-nothing decision batch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, select

from dalio.pipelines.build_report_review_packet import _atomic_write
from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import CandidateCatalogue, load_candidate_catalogue
from dalio.reports.review_decisions import (
    ReviewApplicationResult,
    ReviewDecisionFile,
    ValidatedDecisionPlan,
    apply_review_decisions,
    build_decision_template,
    decision_file_sha256,
    load_decision_file,
    preflight_review_decisions,
    validate_review_decisions,
)
from dalio.storage.db import (
    ReportCandidateReview,
    create_verified_sqlite_backup,
    init_db,
    make_engine,
    make_session_factory,
)

_HUMAN_REVIEWER_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")


def _required_source_ids() -> set[str]:
    return {source.source_id for source in REPORT_SOURCES if source.enabled}


def _require_all_sources(source_ids: set[str]) -> None:
    expected = _required_source_ids()
    missing = sorted(expected - source_ids)
    unexpected = sorted(source_ids - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise RuntimeError(
            "human report review requires every enabled official family: " + "; ".join(details)
        )


def _read_only_engine(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database does not exist: {resolved}")
    return create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{resolved}?mode=ro", uri=True),
        future=True,
    )


def _with_read_only_plan(
    *,
    db_path: Path,
    catalogue_path: Path,
    decision_path: Path,
    require_complete: bool,
) -> ValidatedDecisionPlan:
    catalogue = load_candidate_catalogue(catalogue_path)
    decisions = load_decision_file(decision_path)
    return _validate_read_only_plan(
        db_path=db_path,
        catalogue=catalogue,
        decisions=decisions,
        require_complete=require_complete,
    )


def _validate_read_only_plan(
    *,
    db_path: Path,
    catalogue: CandidateCatalogue,
    decisions: ReviewDecisionFile,
    require_complete: bool,
) -> ValidatedDecisionPlan:
    engine = _read_only_engine(db_path)
    try:
        factory = make_session_factory(engine)
        with factory() as session:
            plan = validate_review_decisions(
                session,
                catalogue,
                decisions,
                require_complete=require_complete,
            )
            _require_all_sources(
                {str(document["source_id"]) for document in plan.packet["documents"]}
            )
            if session.new or session.dirty or session.deleted:
                raise RuntimeError("report decision validation attempted to mutate its session")
            return plan
    finally:
        engine.dispose()


def _validated_review_clock(value: datetime, proposal: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("reviewed_at must be a timezone-aware datetime")
    result = value.astimezone(UTC)
    if result < proposal.astimezone(UTC):
        raise ValueError("reviewed_at cannot be earlier than proposal")
    return result


def _preflight_existing_review(
    *,
    db_path: Path,
    plan: ValidatedDecisionPlan,
    reviewer: str,
) -> ReviewApplicationResult | None:
    engine = _read_only_engine(db_path)
    try:
        if not inspect(engine).has_table(ReportCandidateReview.__tablename__):
            return None
        factory = make_session_factory(engine)
        with factory() as session:
            result = preflight_review_decisions(
                session,
                plan,
                reviewer=reviewer,
            )
            if session.new or session.dirty or session.deleted:
                raise RuntimeError("report decision preflight attempted to mutate its session")
            return result
    finally:
        engine.dispose()


def prepare(
    *,
    db_path: Path,
    catalogue_path: Path,
    output_dir: Path,
) -> tuple[dict[str, object], Path]:
    """Write one immutable blank decision template from a rebuilt packet."""
    catalogue = load_candidate_catalogue(catalogue_path)
    _require_all_sources({candidate.source_id for candidate in catalogue.candidates})
    engine = _read_only_engine(db_path)
    try:
        factory = make_session_factory(engine)
        with factory() as session:
            template = build_decision_template(session, catalogue)
            if session.new or session.dirty or session.deleted:
                raise RuntimeError("decision template preparation attempted a database write")
    finally:
        engine.dispose()

    packet_hash = str(template["packet_sha256"])
    known_date = catalogue.as_known_at.astimezone(UTC).date().isoformat()
    path = output_dir / f"report_decisions_{known_date}_{packet_hash[:16]}.json"
    payload = json.dumps(template, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write(path, payload, immutable=True)
    return template, path


def check(
    *,
    db_path: Path,
    catalogue_path: Path,
    decision_path: Path,
    require_complete: bool = False,
) -> ValidatedDecisionPlan:
    """Validate a decision document and all underlying evidence without writes."""
    return _with_read_only_plan(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
        require_complete=require_complete,
    )


def _decision_counts(decisions: ReviewDecisionFile) -> dict[str, int]:
    counts = Counter(decision.outcome or "pending" for decision in decisions.decisions)
    return {key: counts.get(key, 0) for key in ("approve", "revise", "reject", "pending")}


def status(
    *,
    db_path: Path,
    catalogue_path: Path,
    decision_path: Path,
) -> dict[str, Any]:
    """Return editable-file progress and any already recorded decisions."""
    plan = check(
        db_path=db_path,
        catalogue_path=catalogue_path,
        decision_path=decision_path,
    )
    engine = _read_only_engine(db_path)
    try:
        recorded: list[ReportCandidateReview] = []
        if inspect(engine).has_table(ReportCandidateReview.__tablename__):
            factory = make_session_factory(engine)
            with factory() as session:
                ids = tuple(item.decision.candidate_id for item in plan.decisions)
                recorded = list(
                    session.execute(
                        select(ReportCandidateReview)
                        .where(ReportCandidateReview.candidate_id.in_(ids))
                        .order_by(ReportCandidateReview.candidate_id)
                    ).scalars()
                )
        recorded_counts = Counter(row.outcome for row in recorded)
    finally:
        engine.dispose()
    return {
        "packet_sha256": plan.decision_file.packet_sha256,
        "decision_sha256": decision_file_sha256(plan.decision_file),
        "candidate_count": len(plan.decisions),
        "file": _decision_counts(plan.decision_file),
        "recorded": {
            "approve": recorded_counts.get("approve", 0),
            "revise": recorded_counts.get("revise", 0),
            "reject": recorded_counts.get("reject", 0),
            "total": len(recorded),
        },
    }


def _backup_path(backup_dir: Path, packet_sha256: str, reviewed_at: datetime) -> Path:
    stamp = reviewed_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return backup_dir / f"dalio-before-report-review-{stamp}-{packet_sha256[:16]}.db"


def apply(
    *,
    db_path: Path,
    catalogue_path: Path,
    decision_path: Path,
    backup_dir: Path,
    reviewer: str,
    reviewed_at: datetime,
    expected_decision_file: ReviewDecisionFile | None = None,
) -> tuple[ReviewApplicationResult, Path | None]:
    """Back up the database and append one complete decision batch."""
    if _HUMAN_REVIEWER_RE.fullmatch(reviewer) is None:
        raise ValueError("reviewer must be a concrete human:<id> identity")
    catalogue = load_candidate_catalogue(catalogue_path)
    checked_at = _validated_review_clock(reviewed_at, catalogue.created_at)
    decisions = load_decision_file(decision_path)
    if expected_decision_file is not None and decisions != expected_decision_file:
        raise RuntimeError("decision file changed after preview; nothing was written")
    plan = _validate_read_only_plan(
        db_path=db_path,
        catalogue=catalogue,
        decisions=decisions,
        require_complete=True,
    )
    replay = _preflight_existing_review(
        db_path=db_path,
        plan=plan,
        reviewer=reviewer,
    )
    if replay is not None:
        return replay, None
    backup_path = _backup_path(backup_dir, plan.decision_file.packet_sha256, checked_at)
    create_verified_sqlite_backup(db_path, backup_path)

    engine = make_engine(db_path.expanduser().resolve())
    try:
        init_db(engine)
        factory = make_session_factory(engine)
        with factory() as session:
            result = apply_review_decisions(
                session,
                catalogue,
                decisions,
                reviewer=reviewer,
                reviewed_at=checked_at,
            )
        with engine.connect() as connection:
            integrity = connection.exec_driver_sql("PRAGMA integrity_check").all()
            if integrity != [("ok",)]:
                raise RuntimeError(f"database failed integrity_check after review: {integrity[:3]}")
            foreign_keys = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
            if foreign_keys:
                raise RuntimeError(
                    f"database has foreign-key violations after review: {foreign_keys[:3]}"
                )
    finally:
        engine.dispose()
    return result, backup_path


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    db_path = args.db or Path(os.environ.get("DALIO_DB_PATH", "data/dalio.db"))
    catalogue_path = args.catalogue or Path(
        os.environ.get(
            "DALIO_REPORT_CANDIDATES",
            "data/reference/report_claim_candidates.json",
        )
    )
    output_dir = getattr(args, "out_dir", None) or Path(
        os.environ.get("DALIO_REPORT_REVIEW_DIR", "data/review")
    )
    return db_path, catalogue_path, output_dir


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{key} {counts[key]}" for key in ("approve", "revise", "reject", "pending"))


def _windows_unc(path: Path) -> str:
    distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu")
    absolute = str(path.expanduser().resolve()).replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}{absolute}"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database (default: DALIO_DB_PATH or data/dalio.db).",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=None,
        help=(
            "Checked candidate catalogue (default: DALIO_REPORT_CANDIDATES or "
            "data/reference/report_claim_candidates.json)."
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, check or explicitly apply operator-attributed report-claim decisions."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create a blank decision JSON file.")
    _add_common(prepare_parser)
    prepare_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Generated review directory (default: DALIO_REPORT_REVIEW_DIR or data/review).",
    )

    command_help = {
        "check": "Validate decisions and evidence without database writes.",
        "status": "Show file progress and the recorded ledger without writes.",
        "apply": "Interactively back up and record one complete review batch.",
    }
    for command, help_text in command_help.items():
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_common(command_parser)
        command_parser.add_argument(
            "--decisions",
            type=Path,
            required=True,
            help="Packet-hash-bound decision JSON created by prepare.",
        )
        if command == "check":
            command_parser.add_argument(
                "--require-complete",
                action="store_true",
                help="Fail unless every candidate has a structurally complete outcome.",
            )
        elif command == "apply":
            command_parser.add_argument(
                "--backup-dir",
                type=Path,
                default=Path("data/backups"),
                help="Verified pre-write backup directory (default: data/backups).",
            )

    args = parser.parse_args(argv)
    load_dotenv()
    db_path, catalogue_path, output_dir = _default_paths(args)

    try:
        if args.command == "prepare":
            template, path = prepare(
                db_path=db_path,
                catalogue_path=catalogue_path,
                output_dir=output_dir,
            )
            print(f"Prepared {len(template['decisions'])} pending human decisions.")
            print(f"Packet SHA-256: {template['packet_sha256']}")
            print(f"Blank decision SHA-256: {decision_file_sha256(template)}")
            print(f"POSIX path: {path.expanduser().resolve()}")
            print(f"Windows path: {_windows_unc(path)}")
            print("No reviewer identity or decision was supplied; database writes: none")
            return 0

        if args.command == "check":
            plan = check(
                db_path=db_path,
                catalogue_path=catalogue_path,
                decision_path=args.decisions,
                require_complete=args.require_complete,
            )
            counts = _decision_counts(plan.decision_file)
            print(f"Decision file is provenance-valid: {_counts_text(counts)}")
            print(f"Packet SHA-256: {plan.decision_file.packet_sha256}")
            print(f"Decision SHA-256: {decision_file_sha256(plan.decision_file)}")
            print("Semantic review was not performed by this command.")
            print("Database writes: none")
            return 0

        if args.command == "status":
            report = status(
                db_path=db_path,
                catalogue_path=catalogue_path,
                decision_path=args.decisions,
            )
            print(f"Decision file: {_counts_text(report['file'])}")
            print(f"Outstanding decisions: {report['file']['pending']}")
            print(f"Packet SHA-256: {report['packet_sha256']}")
            print(f"Decision SHA-256: {report['decision_sha256']}")
            print(
                "Recorded ledger: "
                f"approve {report['recorded']['approve']}, "
                f"revise {report['recorded']['revise']}, "
                f"reject {report['recorded']['reject']}, "
                f"total {report['recorded']['total']}"
            )
            print("Database writes: none")
            return 0

        plan = check(
            db_path=db_path,
            catalogue_path=catalogue_path,
            decision_path=args.decisions,
            require_complete=True,
        )
        counts = _decision_counts(plan.decision_file)
        decision_hash = decision_file_sha256(plan.decision_file)
        print(f"Write plan: {_counts_text(counts)}")
        print(f"Packet SHA-256: {plan.decision_file.packet_sha256}")
        print(f"Decision file (POSIX): {args.decisions.expanduser().resolve()}")
        print(f"Decision file (Windows): {_windows_unc(args.decisions)}")
        print("Candidate outcomes:")
        for item in plan.decisions:
            decision = item.decision
            print(
                f"  {decision.candidate_id}  {decision.source_id}/{decision.issue_key}  "
                f"{decision.outcome}"
            )
            print(
                "    proposed statement: "
                + json.dumps(decision.proposed_statement, ensure_ascii=False)
            )
        print(f"Decision SHA-256: {decision_hash}")
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise RuntimeError("apply requires a TTY; non-interactive CLI invocation is refused")
        reviewer = input("Human reviewer identity (human:<id>): ").strip()
        if _HUMAN_REVIEWER_RE.fullmatch(reviewer) is None:
            raise ValueError("reviewer must be a concrete human:<id> identity")
        confirmation = input("Type the full decision SHA-256 to apply this review: ").strip()
        if confirmation != decision_hash:
            raise ValueError("decision confirmation did not match; nothing was written")
        reviewed_at = datetime.now(UTC)
        result, backup_path = apply(
            db_path=db_path,
            catalogue_path=catalogue_path,
            decision_path=args.decisions,
            backup_dir=args.backup_dir,
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            expected_decision_file=plan.decision_file,
        )
        applied_counts = Counter(item.outcome for item in result.results)
        print(
            "Recorded operator-attributed decisions: "
            + ", ".join(
                f"{key} {applied_counts.get(key, 0)}" for key in ("approve", "revise", "reject")
            )
        )
        verified_count = sum(item.verified_claim_id is not None for item in result.results)
        print(f"Operator attribution: {reviewer}")
        print(f"Verified successors: {verified_count}")
        print("Review receipt IDs: " + ", ".join(str(item.review_id) for item in result.results))
        print(f"Replay: {'yes' if result.replayed else 'no'}")
        if backup_path is None:
            print("Verified pre-write backup: not created; exact replay made no database write")
        else:
            print(f"Verified pre-write backup (POSIX): {backup_path.resolve()}")
            print(f"Verified pre-write backup (Windows): {_windows_unc(backup_path)}")
        return 0
    except (EOFError, KeyboardInterrupt):
        print(
            "\nReview application cancelled; no review decisions were committed.", file=sys.stderr
        )
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"dalio-report-review: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
