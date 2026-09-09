"""Publish the offline institutional-communications rights-review packet.

Only the checked pilot manifest is read.  This command has no database or
network dependency and never reads or captures communication source content.
It writes reproducible human-review aids, not a rights decision.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from dalio.communications.pilot_manifest import load_pilot_manifest
from dalio.communications.rights_review import (
    build_rights_review_packet,
    render_rights_review_markdown,
    validate_rights_review_packet_sha256,
)

DEFAULT_MANIFEST_PATH = Path("data/reference/communication_pilot_events.json")
DEFAULT_OUTPUT_DIR = Path("data/review")
LATEST_JSON = "communication_rights_latest.json"
LATEST_MARKDOWN = "communication_rights_latest.md"
LOCK_FILENAME = ".communication_rights_review.lock"
_IMMUTABLE_OUTPUT_RE = re.compile(
    r"^communication_rights_\d{4}-\d{2}-\d{2}_[0-9a-f]{16}\.(?:json|md)$"
)


def _require_regular_or_missing(path: Path, *, purpose: str) -> os.stat_result | None:
    """Reject links and special files before they enter the publication boundary."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"refusing non-regular {purpose}: {path}")
    return metadata


def _validate_immutable_output_namespace(output_dir: Path) -> None:
    """Keep historical hash-named outputs independent from mutable aliases."""
    for path in output_dir.iterdir():
        if _IMMUTABLE_OUTPUT_RE.fullmatch(path.name) is not None:
            _require_regular_or_missing(path, purpose="immutable review output")


def _stage_payload(path: Path, payload: bytes) -> Path:
    """Fully write one same-directory temporary without publishing it."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _replace_file(staged: Path, target: Path) -> None:
    """Single indirection for atomic replacement and deterministic fault tests."""
    os.replace(staged, target)


def _read_optional(path: Path) -> bytes | None:
    metadata = _require_regular_or_missing(path, purpose="review output")
    return path.read_bytes() if metadata is not None else None


def _restore_target(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage_payload(path, previous)
    try:
        _replace_file(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _restore_pair(paths: tuple[Path, ...], previous: tuple[bytes | None, ...]) -> None:
    failures: list[OSError] = []
    for path, payload in zip(reversed(paths), reversed(previous), strict=True):
        try:
            _restore_target(path, payload)
        except OSError as exc:  # pragma: no cover - requires two independent I/O failures
            failures.append(exc)
    if failures:
        raise RuntimeError("could not restore the previous review-output pair") from failures[0]


def _preflight_immutable_pair(
    paths: tuple[Path, Path],
    payloads: tuple[bytes, bytes],
    previous: tuple[bytes | None, bytes | None],
) -> None:
    for path, payload, existing in zip(paths, payloads, previous, strict=True):
        if existing is not None and existing != payload:
            raise RuntimeError(f"refusing to overwrite different hash-addressed output: {path}")


def _publish_pair(
    paths: tuple[Path, Path],
    payloads: tuple[bytes, bytes],
    *,
    immutable: bool,
) -> tuple[bytes | None, bytes | None]:
    """Stage both members, then publish or restore the complete prior pair."""
    previous = tuple(_read_optional(path) for path in paths)
    assert len(previous) == 2
    if immutable:
        _preflight_immutable_pair(paths, payloads, previous)
    should_write = tuple(existing is None if immutable else True for existing in previous)
    if not any(should_write):
        return previous

    staged: list[Path | None] = [None, None]
    replaced: list[int] = []
    try:
        # No final pathname moves until every required member is fully staged.
        for index, (path, payload, needed) in enumerate(
            zip(paths, payloads, should_write, strict=True)
        ):
            if needed:
                staged[index] = _stage_payload(path, payload)
        try:
            for index, (path, needed) in enumerate(zip(paths, should_write, strict=True)):
                if not needed:
                    continue
                temporary = staged[index]
                assert temporary is not None
                _replace_file(temporary, path)
                staged[index] = None
                replaced.append(index)
        except BaseException as exc:
            try:
                _restore_pair(
                    tuple(paths[index] for index in replaced),
                    tuple(previous[index] for index in replaced),
                )
            except RuntimeError as rollback_exc:
                raise rollback_exc from exc
            raise
    finally:
        for temporary in staged:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return previous


@contextmanager
def _output_lock(output_dir: Path) -> Iterator[Path]:
    """Serialize cooperating publishers with a persistent advisory lock."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILENAME
    previous = _require_regular_or_missing(lock_path, purpose="review-output lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"refusing non-regular review-output lock: {lock_path}")
        if previous is not None and (previous.st_dev, previous.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise RuntimeError(f"review-output lock changed while opening it: {lock_path}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _packet_date(packet: dict[str, Any]) -> str:
    known_at = packet.get("as_known_at")
    if not isinstance(known_at, str) or len(known_at) < 10:
        raise ValueError("rights-review packet lacks a canonical as_known_at timestamp")
    try:
        return date.fromisoformat(known_at[:10]).isoformat()
    except ValueError as exc:
        raise ValueError("rights-review packet has an invalid as_known_at timestamp") from exc


def _ensure_input_is_not_an_output(manifest_path: Path, targets: tuple[Path, ...]) -> None:
    manifest_resolved = manifest_path.expanduser().resolve()
    for target in targets:
        if target.expanduser().resolve() == manifest_resolved:
            raise RuntimeError("refusing to overwrite the communication pilot manifest")


def run(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Build and publish serialized, failure-recoverable JSON/Markdown pairs."""
    manifest_path = manifest_path.expanduser()
    output_dir = output_dir.expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"pilot manifest does not exist: {manifest_path}")

    input_bytes = manifest_path.read_bytes()
    manifest = load_pilot_manifest(manifest_path)
    if manifest_path.read_bytes() != input_bytes:
        raise RuntimeError("pilot manifest changed while the review packet was being built")

    packet = build_rights_review_packet(manifest)
    packet_hash = validate_rights_review_packet_sha256(packet)

    json_payload = (
        json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown = render_rights_review_markdown(packet)
    if not markdown.endswith("\n"):
        markdown += "\n"
    markdown_payload = markdown.encode("utf-8")

    dated_stem = f"communication_rights_{_packet_date(packet)}_{packet_hash[:16]}"
    dated_json = output_dir / f"{dated_stem}.json"
    dated_markdown = output_dir / f"{dated_stem}.md"
    latest_json = output_dir / LATEST_JSON
    latest_markdown = output_dir / LATEST_MARKDOWN
    targets = (dated_json, dated_markdown, latest_json, latest_markdown)
    _ensure_input_is_not_an_output(manifest_path, targets)
    lock_path = output_dir / LOCK_FILENAME
    if lock_path.expanduser().resolve() == manifest_path.resolve():
        raise RuntimeError("refusing to use the communication pilot manifest as an output lock")
    if manifest_path.read_bytes() != input_bytes:
        raise RuntimeError("pilot manifest changed while the review packet was being built")

    immutable_paths = (dated_json, dated_markdown)
    latest_paths = (latest_json, latest_markdown)
    payloads = (json_payload, markdown_payload)
    with _output_lock(output_dir):
        _validate_immutable_output_namespace(output_dir)
        for path in latest_paths:
            _require_regular_or_missing(path, purpose="latest review output")
        if manifest_path.read_bytes() != input_bytes:
            raise RuntimeError("pilot manifest changed before review outputs were published")
        # Revalidate at the actual publication boundary, after rendering and
        # after waiting for any other publisher holding the output lock.
        if validate_rights_review_packet_sha256(packet) != packet_hash:
            raise ValueError("rights-review packet hash changed before publication")
        _publish_pair(immutable_paths, payloads, immutable=True)
        _validate_immutable_output_namespace(output_dir)
        if manifest_path.read_bytes() != input_bytes:
            raise RuntimeError("pilot manifest changed before latest aliases were published")
        latest_previous = _publish_pair(latest_paths, payloads, immutable=False)
        if manifest_path.read_bytes() != input_bytes:
            _restore_pair(latest_paths, latest_previous)
            raise RuntimeError("pilot manifest changed while latest aliases were published")

    paths = (*immutable_paths, *latest_paths)
    return packet, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline UNVERIFIED RIGHTS REVIEW packet for the closed "
            "institutional-communications pilot."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Checked pilot manifest (default: DALIO_COMMUNICATION_PILOT_MANIFEST or "
            "data/reference/communication_pilot_events.json)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Review output directory (default: DALIO_COMMUNICATION_RIGHTS_REVIEW_DIR "
            "or data/review)."
        ),
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest or Path(
        os.environ.get("DALIO_COMMUNICATION_PILOT_MANIFEST", str(DEFAULT_MANIFEST_PATH))
    )
    output_dir = args.out_dir or Path(
        os.environ.get("DALIO_COMMUNICATION_RIGHTS_REVIEW_DIR", str(DEFAULT_OUTPUT_DIR))
    )
    try:
        packet, paths = run(manifest_path=manifest_path, output_dir=output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"dalio-communication-rights-packet: {exc}", file=sys.stderr)
        return 2

    print(
        "UNVERIFIED RIGHTS REVIEW: "
        f"{packet['event_count']} events and "
        f"{packet['available_chosen_representation_count']} available chosen "
        "representations; content capture authorized=false"
    )
    for path in paths:
        print(f"  wrote {path.name}")
    print("  verified rights decisions: 0")
    print("  database/network/source-content reads: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "LATEST_JSON",
    "LATEST_MARKDOWN",
    "LOCK_FILENAME",
    "main",
    "run",
]
