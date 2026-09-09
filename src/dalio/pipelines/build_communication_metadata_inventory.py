"""Publish the offline Bank of England communication-link metadata inventory."""

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
from pathlib import Path

from dalio.communications.institution_year_manifest import load_checked_boe_2025_manifest
from dalio.communications.metadata_inventory import (
    build_representation_inventory,
    inventory_publication_date,
    render_representation_inventory_markdown,
    validate_communication_metadata_inventory_sha256,
)

DEFAULT_MANIFEST_PATH = Path("data/reference/communication_boe_2025_events.json")
DEFAULT_OUTPUT_DIR = Path("data/review")
LATEST_JSON = "communication_metadata_latest.json"
LATEST_MARKDOWN = "communication_metadata_latest.md"
LOCK_FILENAME = ".communication_metadata_inventory.lock"
_IMMUTABLE_OUTPUT = re.compile(
    r"^communication_metadata_\d{4}-\d{2}-\d{2}_[0-9a-f]{16}\.(?:json|md)$"
)


def _regular_or_missing(path: Path, *, purpose: str) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"refusing non-regular {purpose}: {path}")
    return metadata


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


@contextmanager
def _pinned_output_directory(
    output_dir: Path,
) -> Iterator[tuple[Path, tuple[int, int]]]:
    """Open the requested directory once and keep all writes bound to that inode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        before = output_dir.lstat()
    except FileNotFoundError as exc:  # pragma: no cover - requires a concurrent unlink
        raise RuntimeError(f"inventory output directory disappeared: {output_dir}") from exc
    if not stat.S_ISDIR(before.st_mode) or output_dir.is_symlink():
        raise RuntimeError(f"refusing unsafe inventory output directory: {output_dir}")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(output_dir, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or _identity(opened) != _identity(before):
            raise RuntimeError(f"inventory output directory changed while opening: {output_dir}")
        bound_dir = Path("/proc/self/fd") / str(descriptor)
        if not bound_dir.is_dir():  # pragma: no cover - Linux/WSL contract beside fcntl
            raise RuntimeError("cannot bind inventory publication to its output directory")
        yield bound_dir, _identity(opened)
    finally:
        os.close(descriptor)


def _require_output_directory_unchanged(
    output_dir: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    try:
        current = output_dir.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"inventory output directory changed during publication: {output_dir}"
        ) from exc
    if not stat.S_ISDIR(current.st_mode) or _identity(current) != expected_identity:
        raise RuntimeError(
            f"inventory output directory changed during publication: {output_dir}"
        )


def _manifest_snapshot(path: Path) -> tuple[bytes, tuple[int, int]]:
    try:
        metadata = _regular_or_missing(path, purpose="checked metadata manifest")
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"checked metadata manifest does not exist safely: {path}"
        ) from exc
    if metadata is None:
        raise FileNotFoundError(f"checked metadata manifest does not exist safely: {path}")
    identity = _identity(metadata)
    payload = path.read_bytes()
    after = _regular_or_missing(path, purpose="checked metadata manifest")
    if after is None or _identity(after) != identity:
        raise RuntimeError("metadata manifest changed while it was being read")
    return payload, identity


def _require_manifest_unchanged(
    path: Path,
    *,
    expected_bytes: bytes,
    expected_identity: tuple[int, int],
    phase: str,
) -> None:
    before = _regular_or_missing(path, purpose="checked metadata manifest")
    if before is None or _identity(before) != expected_identity:
        raise RuntimeError(f"metadata manifest changed {phase}")
    payload = path.read_bytes()
    after = _regular_or_missing(path, purpose="checked metadata manifest")
    if (
        after is None
        or _identity(after) != expected_identity
        or payload != expected_bytes
    ):
        raise RuntimeError(f"metadata manifest changed {phase}")


def _require_distinct_existing_files(
    entries: tuple[tuple[str, Path], ...],
) -> None:
    """Reject hard-link aliases across the input and publication namespace."""
    seen: dict[tuple[int, int], tuple[str, Path]] = {}
    for purpose, path in entries:
        metadata = _regular_or_missing(path, purpose=purpose)
        if metadata is None:
            continue
        identity = _identity(metadata)
        prior = seen.get(identity)
        if prior is not None:
            prior_purpose, prior_path = prior
            raise RuntimeError(
                "refusing hard-linked publication paths: "
                f"{prior_purpose} {prior_path} and {purpose} {path}"
            )
        seen[identity] = (purpose, path)


def _read_optional(path: Path) -> bytes | None:
    return path.read_bytes() if _regular_or_missing(path, purpose="inventory output") else None


def _stage(path: Path, payload: bytes) -> Path:
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


def _replace(staged: Path, target: Path) -> None:
    os.replace(staged, target)


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
        return
    staged = _stage(path, previous)
    try:
        _replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _publish_pair(
    paths: tuple[Path, Path],
    payloads: tuple[bytes, bytes],
    *,
    immutable: bool,
) -> tuple[bytes | None, bytes | None]:
    previous = tuple(_read_optional(path) for path in paths)
    if immutable:
        for path, old, new in zip(paths, previous, payloads, strict=True):
            if old is not None and old != new:
                raise RuntimeError(f"refusing to overwrite different hash-addressed output: {path}")
    should_write = tuple(
        old is None if immutable else old != new
        for old, new in zip(previous, payloads, strict=True)
    )
    if not any(should_write):
        return previous
    staged: list[Path | None] = [None, None]
    replaced: list[int] = []
    try:
        for index, (path, payload, write) in enumerate(
            zip(paths, payloads, should_write, strict=True)
        ):
            if write:
                staged[index] = _stage(path, payload)
        for index, (path, write) in enumerate(zip(paths, should_write, strict=True)):
            if write:
                assert staged[index] is not None
                _replace(staged[index], path)
                staged[index] = None
                replaced.append(index)
    except BaseException:
        failures: list[OSError] = []
        for index in reversed(replaced):
            try:
                _restore(paths[index], previous[index])
            except OSError as exc:  # pragma: no cover - independent double I/O failure
                failures.append(exc)
        if failures:
            raise RuntimeError(
                "could not restore communication metadata output pair"
            ) from failures[0]
        raise
    finally:
        for path in staged:
            if path is not None:
                path.unlink(missing_ok=True)
    return previous


@contextmanager
def _output_lock(output_dir: Path) -> Iterator[Path]:
    lock_path = output_dir / LOCK_FILENAME
    previous = _regular_or_missing(lock_path, purpose="inventory-output lock")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"refusing non-regular inventory-output lock: {lock_path}")
        on_disk = _regular_or_missing(lock_path, purpose="inventory-output lock")
        if on_disk is None or _identity(on_disk) != _identity(opened):
            raise RuntimeError(f"inventory-output lock changed while opening it: {lock_path}")
        if previous is not None and _identity(previous) != _identity(opened):
            raise RuntimeError(f"inventory-output lock changed while opening it: {lock_path}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            locked_path = _regular_or_missing(lock_path, purpose="inventory-output lock")
            if locked_path is None or _identity(locked_path) != _identity(opened):
                raise RuntimeError(
                    f"inventory-output lock changed while acquiring it: {lock_path}"
                )
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_namespace(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if _IMMUTABLE_OUTPUT.fullmatch(path.name):
            _regular_or_missing(path, purpose="immutable inventory output")


def run(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    """Build and publish fixed and hash-addressed metadata-only inventory pairs."""
    manifest_path = manifest_path.expanduser()
    output_dir = output_dir.expanduser()
    input_bytes, input_identity = _manifest_snapshot(manifest_path)
    manifest = load_checked_boe_2025_manifest(manifest_path)
    _require_manifest_unchanged(
        manifest_path,
        expected_bytes=input_bytes,
        expected_identity=input_identity,
        phase="while inventory was being built",
    )
    inventory = build_representation_inventory(manifest)
    inventory_hash = validate_communication_metadata_inventory_sha256(inventory)
    json_payload = (
        json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    markdown = render_representation_inventory_markdown(inventory)
    if not markdown.endswith("\n"):
        markdown += "\n"
    payloads = (json_payload, markdown.encode("utf-8"))

    stem = f"communication_metadata_{inventory_publication_date(inventory)}_{inventory_hash[:16]}"
    immutable = (output_dir / f"{stem}.json", output_dir / f"{stem}.md")
    latest = (output_dir / LATEST_JSON, output_dir / LATEST_MARKDOWN)
    targets = (*immutable, *latest)
    resolved_input = manifest_path.resolve()
    if any(path.resolve() == resolved_input for path in targets):
        raise RuntimeError("refusing to overwrite the checked metadata manifest")

    lock_candidate = output_dir / LOCK_FILENAME
    if lock_candidate.resolve() == resolved_input:
        raise RuntimeError("refusing to use the checked metadata manifest as an output lock")

    with _pinned_output_directory(output_dir) as (bound_dir, output_identity):
        bound_immutable = tuple(bound_dir / path.name for path in immutable)
        bound_latest = tuple(bound_dir / path.name for path in latest)
        bound_targets = (*bound_immutable, *bound_latest)
        with _output_lock(bound_dir) as lock_path:
            _require_output_directory_unchanged(
                output_dir, expected_identity=output_identity
            )
            _validate_namespace(bound_dir)
            for path in bound_latest:
                _regular_or_missing(path, purpose="latest inventory output")
            publication_entries = (
                ("checked metadata manifest", manifest_path),
                ("inventory-output lock", lock_path),
                *(("inventory output", path) for path in bound_targets),
            )
            _require_distinct_existing_files(publication_entries)
            _require_manifest_unchanged(
                manifest_path,
                expected_bytes=input_bytes,
                expected_identity=input_identity,
                phase="before inventory publication",
            )
            if validate_communication_metadata_inventory_sha256(inventory) != inventory_hash:
                raise ValueError("communication metadata inventory hash changed before publication")
            _publish_pair(bound_immutable, payloads, immutable=True)
            _require_output_directory_unchanged(
                output_dir, expected_identity=output_identity
            )
            _validate_namespace(bound_dir)
            _require_distinct_existing_files(publication_entries)
            _require_manifest_unchanged(
                manifest_path,
                expected_bytes=input_bytes,
                expected_identity=input_identity,
                phase="before latest aliases were published",
            )
            previous_latest = _publish_pair(bound_latest, payloads, immutable=False)
            try:
                _require_output_directory_unchanged(
                    output_dir, expected_identity=output_identity
                )
                _require_manifest_unchanged(
                    manifest_path,
                    expected_bytes=input_bytes,
                    expected_identity=input_identity,
                    phase="while latest aliases were published",
                )
                _require_distinct_existing_files(publication_entries)
            except (OSError, RuntimeError):
                for path, previous in zip(
                    reversed(bound_latest), reversed(previous_latest), strict=True
                ):
                    _restore(path, previous)
                raise
            _validate_namespace(bound_dir)
    return inventory, targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the offline metadata-only BoE communication representation inventory."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        inventory, paths = run(manifest_path=args.manifest, output_dir=args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"dalio-communication-metadata-inventory: {exc}", file=sys.stderr)
        return 2
    print(
        f"communication metadata inventory {inventory['inventory_sha256']} "
        f"({inventory['event_count']}/{inventory['expected_event_count']} events)"
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
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
