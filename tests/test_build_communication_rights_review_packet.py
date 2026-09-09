"""Atomic offline publication of communication rights-review aids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Event, Thread

import pytest

import dalio.pipelines.build_communication_rights_review_packet as pipeline
from dalio.pipelines.build_communication_rights_review_packet import (
    LATEST_JSON,
    LATEST_MARKDOWN,
    LOCK_FILENAME,
    main,
    run,
)

_ROOT = Path(__file__).resolve().parents[1]
_CHECKED_MANIFEST = _ROOT / "data/reference/communication_pilot_events.json"


def _copy_manifest(tmp_path: Path) -> Path:
    target = tmp_path / "communication_pilot_events.json"
    target.write_bytes(_CHECKED_MANIFEST.read_bytes())
    return target


def test_run_writes_immutable_and_latest_pairs_without_changing_input(tmp_path):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    before = manifest_path.read_bytes()
    before_sha256 = hashlib.sha256(before).hexdigest()

    packet, paths = run(manifest_path=manifest_path, output_dir=output_dir)

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == before_sha256
    hash16 = packet["packet_sha256"][:16]
    stem = f"communication_rights_2026-09-09_{hash16}"
    expected = (
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.md",
        output_dir / LATEST_JSON,
        output_dir / LATEST_MARKDOWN,
    )
    assert paths == expected
    assert all(path.is_file() for path in paths)
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert json.loads(paths[0].read_text(encoding="utf-8")) == packet

    for output in paths:
        rendered = output.read_text(encoding="utf-8")
        assert str(tmp_path) not in rendered
        assert str(manifest_path.resolve()) not in rendered
    assert packet["content_capture_authorized"] is False
    assert packet["verified_rights_decisions"] == 0
    assert packet["event_count"] == 16
    assert packet["available_chosen_representation_count"] == 16


def test_repeat_run_is_byte_and_path_idempotent(tmp_path):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    first_packet, first_paths = run(manifest_path=manifest_path, output_dir=output_dir)
    first_bytes = {path: path.read_bytes() for path in first_paths}

    second_packet, second_paths = run(manifest_path=manifest_path, output_dir=output_dir)

    assert second_packet == first_packet
    assert second_paths == first_paths
    assert {path: path.read_bytes() for path in second_paths} == first_bytes
    assert set(output_dir.iterdir()) == {*first_paths, output_dir / LOCK_FILENAME}


def test_persistent_advisory_lock_serializes_publishers(tmp_path):
    output_dir = tmp_path / "review"
    first_holds_lock = Event()
    release_first = Event()
    second_attempted = Event()
    second_entered = Event()

    def first_publisher():
        with pipeline._output_lock(output_dir):
            first_holds_lock.set()
            assert release_first.wait(timeout=2)

    def second_publisher():
        assert first_holds_lock.wait(timeout=2)
        second_attempted.set()
        with pipeline._output_lock(output_dir):
            second_entered.set()

    first = Thread(target=first_publisher)
    second = Thread(target=second_publisher)
    first.start()
    second.start()
    assert second_attempted.wait(timeout=2)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert set(output_dir.iterdir()) == {output_dir / LOCK_FILENAME}


def test_immutable_collision_fails_before_latest_aliases_move(tmp_path):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    _packet, paths = run(manifest_path=manifest_path, output_dir=output_dir)
    latest_before = {path: path.read_bytes() for path in paths[2:]}
    paths[0].write_text("different bytes\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite different hash-addressed"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert {path: path.read_bytes() for path in paths[2:]} == latest_before
    assert manifest_path.read_bytes() == _CHECKED_MANIFEST.read_bytes()


def test_symlinked_historical_immutable_cannot_follow_a_future_latest_alias(tmp_path):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    old_packet, old_paths = run(manifest_path=manifest_path, output_dir=output_dir)
    old_latest = old_paths[2].read_bytes()
    old_paths[0].unlink()
    old_paths[0].symlink_to(old_paths[2].name)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["events"][0]["title"] = "Updated checked metadata title"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-regular immutable review output"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert old_paths[0].is_symlink()
    assert old_paths[0].read_bytes() == old_latest
    assert old_paths[2].read_bytes() == old_latest
    assert (
        json.loads(old_paths[0].read_text(encoding="utf-8"))["packet_sha256"]
        == (old_packet["packet_sha256"])
    )
    assert set(output_dir.iterdir()) == {*old_paths, output_dir / LOCK_FILENAME}


def test_symlinked_lock_is_rejected_without_touching_its_target(tmp_path):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"must remain unchanged")
    (output_dir / LOCK_FILENAME).symlink_to(lock_target)

    with pytest.raises(RuntimeError, match="non-regular review-output lock"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert lock_target.read_bytes() == b"must remain unchanged"
    assert set(output_dir.iterdir()) == {output_dir / LOCK_FILENAME}


def test_failure_replacing_second_immutable_member_removes_the_first(tmp_path, monkeypatch):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    original_replace = pipeline._replace_file
    failed = False

    def fail_dated_markdown_once(staged, target):
        nonlocal failed
        if not failed and target.suffix == ".md" and target.name != LATEST_MARKDOWN:
            failed = True
            raise OSError("injected immutable Markdown failure")
        original_replace(staged, target)

    monkeypatch.setattr(pipeline, "_replace_file", fail_dated_markdown_once)

    with pytest.raises(OSError, match="injected immutable Markdown failure"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert failed is True
    assert set(output_dir.iterdir()) == {output_dir / LOCK_FILENAME}


def test_failure_replacing_second_latest_member_restores_the_previous_pair(tmp_path, monkeypatch):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    _old_packet, old_paths = run(manifest_path=manifest_path, output_dir=output_dir)
    latest_paths = old_paths[2:]
    latest_before = {path: path.read_bytes() for path in latest_paths}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["events"][0]["title"] = "Updated checked metadata title"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    original_replace = pipeline._replace_file
    failed = False

    def fail_latest_markdown_once(staged, target):
        nonlocal failed
        if not failed and target.name == LATEST_MARKDOWN:
            failed = True
            raise OSError("injected latest Markdown failure")
        original_replace(staged, target)

    monkeypatch.setattr(pipeline, "_replace_file", fail_latest_markdown_once)

    with pytest.raises(OSError, match="injected latest Markdown failure"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert failed is True
    assert {path: path.read_bytes() for path in latest_paths} == latest_before
    immutable = [
        path for path in output_dir.iterdir() if path.name.startswith("communication_rights_2026-")
    ]
    stems = {path.stem for path in immutable}
    assert len(immutable) == 4
    assert len(stems) == 2


def test_packet_is_revalidated_after_rendering_and_before_publication(tmp_path, monkeypatch):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"
    original_render = pipeline.render_rights_review_markdown

    def render_then_tamper(packet):
        rendered = original_render(packet)
        packet["event_count"] = 15
        return rendered

    monkeypatch.setattr(pipeline, "render_rights_review_markdown", render_then_tamper)

    with pytest.raises(ValueError, match="does not match its canonical payload"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert set(output_dir.iterdir()) == {output_dir / LOCK_FILENAME}


def test_run_refuses_to_use_manifest_as_an_output_alias(tmp_path):
    output_dir = tmp_path / "review"
    output_dir.mkdir()
    manifest_path = output_dir / LATEST_JSON
    manifest_path.write_bytes(_CHECKED_MANIFEST.read_bytes())
    before = manifest_path.read_bytes()

    with pytest.raises(RuntimeError, match="refusing to overwrite the communication pilot"):
        run(manifest_path=manifest_path, output_dir=output_dir)

    assert manifest_path.read_bytes() == before
    assert set(output_dir.iterdir()) == {manifest_path}


def test_cli_success_prints_only_output_names_and_closed_gate(tmp_path, capsys):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"

    result = main(
        [
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert "UNVERIFIED RIGHTS REVIEW" in captured.out
    assert "content capture authorized=false" in captured.out
    assert "verified rights decisions: 0" in captured.out
    assert str(tmp_path) not in captured.out
    assert len(list(output_dir.iterdir())) == 5
    assert (output_dir / LOCK_FILENAME).is_file()


def test_cli_catches_publication_oserror_with_its_registered_command_name(
    tmp_path, monkeypatch, capsys
):
    manifest_path = _copy_manifest(tmp_path)
    output_dir = tmp_path / "review"

    def fail_replace(_staged, _target):
        raise OSError("injected disk error")

    monkeypatch.setattr(pipeline, "_replace_file", fail_replace)

    result = main(["--manifest", str(manifest_path), "--out-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err.startswith("dalio-communication-rights-packet:")
    assert "injected disk error" in captured.err
    assert set(output_dir.iterdir()) == {output_dir / LOCK_FILENAME}


def test_cli_missing_or_invalid_manifest_writes_nothing(tmp_path, capsys):
    output_dir = tmp_path / "review"
    missing = tmp_path / "missing.json"

    result = main(["--manifest", str(missing), "--out-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert result == 2
    assert "pilot manifest does not exist" in captured.err
    assert not output_dir.exists()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}\n", encoding="utf-8")
    result = main(["--manifest", str(invalid), "--out-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert result == 2
    assert "missing fields" in captured.err
    assert not output_dir.exists()
