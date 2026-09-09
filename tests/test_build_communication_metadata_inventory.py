"""Offline publication of the communication metadata coverage inventory."""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from dalio.pipelines import build_communication_metadata_inventory as pipeline

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "reference" / "communication_boe_2025_events.json"
RBA_MANIFEST = ROOT / "data" / "reference" / "communication_rba_2025_events.json"
RIKSBANK_MANIFEST = ROOT / "data" / "reference" / "communication_riksbank_2025_events.json"
CHECKED_JSON_SHA256 = "e17cd292bfba8c509716e158964d5e021bb12d21f654bf84e0788dff42dc6073"
CHECKED_MARKDOWN_SHA256 = "644b2fdac5932293fe5ac1fdf75281002d2289360de7b3aa1eb5176733430214"
CHECKED_RIKSBANK_JSON_SHA256 = "8485f8cd23d740539837921e3699704c8675d1ab908a27a495b7ee966d07f491"
CHECKED_RIKSBANK_MARKDOWN_SHA256 = (
    "35d38c1d7982f603c24f3c3c11fc57c0e2024d4b3212451d37abb612137f9556"
)
CHECKED_RBA_JSON_SHA256 = "9028b5c26f720139869360aa1ab158035ab64786f6b4c2fce9f621d841bab4c9"
CHECKED_RBA_MARKDOWN_SHA256 = "913233696151c9b26e6dfd173aa6a74ed52d6d850111152c220943dab18b0490"
TEST_COHORT_ID = "test_boe_2025_mpr_press_conferences"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_test_cohort(monkeypatch, *, loader, cohort_id=TEST_COHORT_ID):
    publication = pipeline.CheckedCohortPublication(
        cohort_id=cohort_id,
        default_manifest_path=MANIFEST,
        checked_loader=loader,
    )
    monkeypatch.setattr(
        pipeline,
        "CHECKED_COHORT_PUBLICATIONS",
        MappingProxyType({**pipeline.CHECKED_COHORT_PUBLICATIONS, cohort_id: publication}),
    )
    return publication


def test_checked_cohort_registry_is_immutable_and_defaults_to_boe():
    publication = pipeline.CHECKED_COHORT_PUBLICATIONS[pipeline.DEFAULT_COHORT_ID]
    rba = pipeline.CHECKED_COHORT_PUBLICATIONS[pipeline.RBA_2025_MANIFEST_ID]
    riksbank = pipeline.CHECKED_COHORT_PUBLICATIONS[pipeline.RIKSBANK_2025_MANIFEST_ID]

    assert pipeline.DEFAULT_COHORT_ID == "boe_2025_mpr_press_conferences"
    assert publication.cohort_id == pipeline.DEFAULT_COHORT_ID
    assert publication.default_manifest_path == pipeline.DEFAULT_MANIFEST_PATH
    assert publication.checked_loader is pipeline.load_checked_boe_2025_manifest
    assert rba.default_manifest_path == pipeline.RBA_2025_MANIFEST_PATH
    assert rba.checked_loader is pipeline.load_checked_rba_2025_manifest
    assert riksbank.default_manifest_path == pipeline.RIKSBANK_2025_MANIFEST_PATH
    assert riksbank.checked_loader is pipeline.load_checked_riksbank_2025_manifest
    with pytest.raises(TypeError):
        pipeline.CHECKED_COHORT_PUBLICATIONS["another_cohort"] = publication  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        publication.cohort_id = "another_cohort"  # type: ignore[misc]


@pytest.mark.parametrize("cohort_id", ["", "Uppercase", "../escape", "two-cohorts"])
def test_checked_cohort_publication_rejects_unsafe_ids(cohort_id):
    with pytest.raises(ValueError, match="lowercase snake_case"):
        pipeline.CheckedCohortPublication(
            cohort_id=cohort_id,
            default_manifest_path=MANIFEST,
            checked_loader=pipeline.load_checked_boe_2025_manifest,
        )


def test_run_publishes_exact_immutable_and_latest_pairs(tmp_path):
    inventory, paths = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)

    stem = f"communication_metadata_2026-09-09_{inventory['inventory_sha256'][:16]}"
    assert paths == (
        tmp_path / f"{stem}.json",
        tmp_path / f"{stem}.md",
        tmp_path / pipeline.LATEST_JSON,
        tmp_path / pipeline.LATEST_MARKDOWN,
    )
    assert all(path.is_file() and not path.is_symlink() for path in paths)
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert _sha256(paths[0]) == CHECKED_JSON_SHA256
    assert _sha256(paths[1]) == CHECKED_MARKDOWN_SHA256
    assert json.loads(paths[0].read_text(encoding="utf-8")) == inventory
    assert "NO CONTENT CAPTURE IS AUTHORIZED" in paths[1].read_text(encoding="utf-8")

    before = {path: path.read_bytes() for path in paths}
    replay, replay_paths = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    assert replay == inventory
    assert replay_paths == paths
    assert {path: path.read_bytes() for path in paths} == before


def test_explicit_default_cohort_preserves_boe_paths_bytes_and_immutable_files(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    implicit_inventory, implicit_paths = pipeline.run(output_dir=tmp_path)
    immutable_identity = tuple(
        (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
        for path in implicit_paths[:2]
    )

    explicit_inventory, explicit_paths = pipeline.run(
        manifest_path=MANIFEST,
        output_dir=tmp_path,
        cohort=pipeline.DEFAULT_COHORT_ID,
    )

    assert explicit_inventory == implicit_inventory
    assert explicit_paths == implicit_paths
    assert explicit_paths[2:] == (
        tmp_path / pipeline.LATEST_JSON,
        tmp_path / pipeline.LATEST_MARKDOWN,
    )
    assert (
        tuple(
            (path.stat().st_dev, path.stat().st_ino, path.stat().st_mtime_ns)
            for path in explicit_paths[:2]
        )
        == immutable_identity
    )
    assert _sha256(explicit_paths[0]) == CHECKED_JSON_SHA256
    assert _sha256(explicit_paths[1]) == CHECKED_MARKDOWN_SHA256


def test_nondefault_cohort_uses_qualified_latest_aliases_and_leaves_boe_aliases(
    tmp_path, monkeypatch
):
    _, boe_paths = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    boe_aliases = {path: path.read_bytes() for path in boe_paths[2:]}
    boe_immutable_identity = tuple(
        (path.stat().st_dev, path.stat().st_ino) for path in boe_paths[:2]
    )
    loaded_paths = []

    def load_test_checked_manifest(path: Path):
        loaded_paths.append(path)
        return replace(pipeline.load_checked_boe_2025_manifest(path), manifest_id=TEST_COHORT_ID)

    _register_test_cohort(monkeypatch, loader=load_test_checked_manifest)

    inventory, paths = pipeline.run(
        manifest_path=MANIFEST,
        output_dir=tmp_path,
        cohort=TEST_COHORT_ID,
    )

    stem = f"communication_metadata_2026-09-09_{inventory['inventory_sha256'][:16]}"
    assert loaded_paths == [MANIFEST]
    assert inventory["manifest_id"] == TEST_COHORT_ID
    assert paths == (
        tmp_path / f"{stem}.json",
        tmp_path / f"{stem}.md",
        tmp_path / f"communication_metadata_{TEST_COHORT_ID}_latest.json",
        tmp_path / f"communication_metadata_{TEST_COHORT_ID}_latest.md",
    )
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert {path: path.read_bytes() for path in boe_paths[2:]} == boe_aliases
    assert (
        tuple((path.stat().st_dev, path.stat().st_ino) for path in boe_paths[:2])
        == boe_immutable_identity
    )


def test_riksbank_cohort_uses_registered_default_and_does_not_move_boe_latest(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    boe_inventory, boe_paths = pipeline.run(output_dir=tmp_path)
    boe_aliases = {path: path.read_bytes() for path in boe_paths[2:]}

    inventory, paths = pipeline.run(
        output_dir=tmp_path,
        cohort=pipeline.RIKSBANK_2025_MANIFEST_ID,
    )

    stem = f"communication_metadata_2026-09-09_{inventory['inventory_sha256'][:16]}"
    assert inventory["manifest_id"] == pipeline.RIKSBANK_2025_MANIFEST_ID
    assert inventory["event_count"] == inventory["expected_event_count"] == 8
    assert paths == (
        tmp_path / f"{stem}.json",
        tmp_path / f"{stem}.md",
        tmp_path
        / "communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.json",
        tmp_path
        / "communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.md",
    )
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert {path: path.read_bytes() for path in boe_paths[2:]} == boe_aliases
    assert json.loads(boe_paths[0].read_text(encoding="utf-8")) == boe_inventory
    assert _sha256(paths[0]) == CHECKED_RIKSBANK_JSON_SHA256
    assert _sha256(paths[1]) == CHECKED_RIKSBANK_MARKDOWN_SHA256
    markdown = paths[1].read_text(encoding="utf-8")
    assert "first-party replay-page locator" in markdown
    assert "Bank of England publication" not in markdown


def test_rba_cohort_uses_registered_default_and_does_not_move_existing_latest_aliases(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    _, boe_paths = pipeline.run(output_dir=tmp_path)
    _, riksbank_paths = pipeline.run(
        output_dir=tmp_path,
        cohort=pipeline.RIKSBANK_2025_MANIFEST_ID,
    )
    existing_aliases = {path: path.read_bytes() for path in (*boe_paths[2:], *riksbank_paths[2:])}

    inventory, paths = pipeline.run(
        output_dir=tmp_path,
        cohort=pipeline.RBA_2025_MANIFEST_ID,
    )

    stem = f"communication_metadata_2026-09-09_{inventory['inventory_sha256'][:16]}"
    assert inventory["manifest_id"] == pipeline.RBA_2025_MANIFEST_ID
    assert inventory["event_count"] == inventory["expected_event_count"] == 8
    assert paths == (
        tmp_path / f"{stem}.json",
        tmp_path / f"{stem}.md",
        tmp_path / "communication_metadata_rba_2025_monetary_policy_media_conferences_latest.json",
        tmp_path / "communication_metadata_rba_2025_monetary_policy_media_conferences_latest.md",
    )
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert {path: path.read_bytes() for path in existing_aliases} == existing_aliases
    assert _sha256(boe_paths[0]) == CHECKED_JSON_SHA256
    assert _sha256(boe_paths[1]) == CHECKED_MARKDOWN_SHA256
    assert _sha256(riksbank_paths[0]) == CHECKED_RIKSBANK_JSON_SHA256
    assert _sha256(riksbank_paths[1]) == CHECKED_RIKSBANK_MARKDOWN_SHA256
    assert _sha256(paths[0]) == CHECKED_RBA_JSON_SHA256
    assert _sha256(paths[1]) == CHECKED_RBA_MARKDOWN_SHA256
    markdown = paths[1].read_text(encoding="utf-8")
    assert "first-party institutional publication" in markdown
    assert "first-party replay-page locator" not in markdown
    assert "Bank of England publication" not in markdown


def test_unknown_or_mismatched_cohort_fails_before_output(tmp_path, monkeypatch):
    unknown_output = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unknown checked communication cohort"):
        pipeline.run(
            manifest_path=MANIFEST,
            output_dir=unknown_output,
            cohort="unknown_cohort",
        )
    assert not unknown_output.exists()

    _register_test_cohort(monkeypatch, loader=pipeline.load_checked_boe_2025_manifest)
    mismatch_output = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="manifest cohort does not match selected cohort"):
        pipeline.run(
            manifest_path=MANIFEST,
            output_dir=mismatch_output,
            cohort=TEST_COHORT_ID,
        )
    assert not mismatch_output.exists()


def test_run_has_no_network_or_database_dependency(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/database access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    inventory, _ = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    assert inventory["content_capture_authorized"] is False


def test_run_rejects_a_changed_checked_manifest_before_output(tmp_path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["events"][0]["title"] = "Changed but structurally valid"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "review"

    with pytest.raises(ValueError, match="pinned semantic SHA-256"):
        pipeline.run(manifest_path=changed, output_dir=output)
    assert not output.exists()


def test_run_refuses_to_overwrite_a_different_hash_addressed_file(tmp_path):
    inventory, paths = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    paths[0].write_bytes(b"different")

    with pytest.raises(RuntimeError, match="hash-addressed output"):
        pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    assert paths[0].read_bytes() == b"different"
    assert inventory["content_capture_authorized"] is False


def test_latest_pair_is_restored_if_second_replace_fails(tmp_path, monkeypatch):
    _, paths = pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)
    latest_json, latest_markdown = paths[2:]
    latest_json.write_bytes(b"previous-json")
    latest_markdown.write_bytes(b"previous-markdown")
    original_replace = pipeline._replace
    failed = False

    def fail_once(staged: Path, target: Path) -> None:
        nonlocal failed
        if target.name == latest_markdown.name and not failed:
            failed = True
            raise OSError("injected second replace failure")
        original_replace(staged, target)

    monkeypatch.setattr(pipeline, "_replace", fail_once)
    with pytest.raises(OSError, match="injected"):
        pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)

    assert latest_json.read_bytes() == b"previous-json"
    assert latest_markdown.read_bytes() == b"previous-markdown"


def test_new_immutable_pair_is_removed_if_second_replace_fails(tmp_path, monkeypatch):
    expected_inventory = pipeline.build_representation_inventory(
        pipeline.load_checked_boe_2025_manifest(MANIFEST)
    )
    stem = f"communication_metadata_2026-09-09_{expected_inventory['inventory_sha256'][:16]}"
    immutable_json = tmp_path / f"{stem}.json"
    immutable_markdown = tmp_path / f"{stem}.md"
    original_replace = pipeline._replace

    def fail_on_immutable_markdown(staged: Path, target: Path) -> None:
        if target.name == immutable_markdown.name:
            raise OSError("injected immutable-pair failure")
        original_replace(staged, target)

    monkeypatch.setattr(pipeline, "_replace", fail_on_immutable_markdown)
    with pytest.raises(OSError, match="immutable-pair"):
        pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)

    assert not immutable_json.exists()
    assert not immutable_markdown.exists()
    assert not (tmp_path / pipeline.LATEST_JSON).exists()
    assert not (tmp_path / pipeline.LATEST_MARKDOWN).exists()


def test_run_rejects_symlink_manifest_and_output_alias(tmp_path):
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(MANIFEST)
    with pytest.raises(FileNotFoundError, match="does not exist safely"):
        pipeline.run(manifest_path=manifest_link, output_dir=tmp_path / "review")

    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias_manifest = alias_dir / pipeline.LATEST_JSON
    alias_manifest.write_bytes(MANIFEST.read_bytes())
    with pytest.raises(RuntimeError, match="overwrite the checked metadata manifest"):
        pipeline.run(manifest_path=alias_manifest, output_dir=alias_dir)


def test_run_rejects_symlink_lock_and_hard_link_aliases(tmp_path):
    symlink_output = tmp_path / "symlink-output"
    symlink_output.mkdir()
    lock_target = tmp_path / "lock-target"
    lock_target.write_bytes(b"do not follow")
    (symlink_output / pipeline.LOCK_FILENAME).symlink_to(lock_target)
    with pytest.raises(RuntimeError, match="non-regular inventory-output lock"):
        pipeline.run(manifest_path=MANIFEST, output_dir=symlink_output)
    assert lock_target.read_bytes() == b"do not follow"

    hard_link_output = tmp_path / "hard-link-output"
    hard_link_output.mkdir()
    local_manifest = tmp_path / "checked-manifest.json"
    local_manifest.write_bytes(MANIFEST.read_bytes())
    (hard_link_output / pipeline.LATEST_JSON).hardlink_to(local_manifest)
    with pytest.raises(RuntimeError, match="hard-linked publication paths"):
        pipeline.run(manifest_path=local_manifest, output_dir=hard_link_output)
    assert local_manifest.read_bytes() == (hard_link_output / pipeline.LATEST_JSON).read_bytes()

    pair_output = tmp_path / "hard-linked-pair"
    pair_output.mkdir()
    latest_json = pair_output / pipeline.LATEST_JSON
    latest_json.write_bytes(b"old pair")
    (pair_output / pipeline.LATEST_MARKDOWN).hardlink_to(latest_json)
    with pytest.raises(RuntimeError, match="hard-linked publication paths"):
        pipeline.run(manifest_path=MANIFEST, output_dir=pair_output)
    assert latest_json.read_bytes() == b"old pair"


def test_run_rejects_lock_path_replacement_during_acquisition(tmp_path, monkeypatch):
    original_flock = pipeline.fcntl.flock
    lock_path = tmp_path / pipeline.LOCK_FILENAME
    replaced = False

    def replace_lock_path(descriptor: int, operation: int) -> None:
        nonlocal replaced
        original_flock(descriptor, operation)
        if operation == pipeline.fcntl.LOCK_EX and not replaced:
            replaced = True
            lock_path.unlink()
            lock_path.write_bytes(b"replacement lock")

    monkeypatch.setattr(pipeline.fcntl, "flock", replace_lock_path)
    with pytest.raises(RuntimeError, match="lock changed while acquiring"):
        pipeline.run(manifest_path=MANIFEST, output_dir=tmp_path)

    assert not (tmp_path / pipeline.LATEST_JSON).exists()
    assert not (tmp_path / pipeline.LATEST_MARKDOWN).exists()


def test_run_pins_output_directory_and_rejects_path_substitution(tmp_path, monkeypatch):
    output_dir = tmp_path / "review"
    displaced_dir = tmp_path / "displaced-review"
    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    original_lock = pipeline._output_lock

    @contextmanager
    def substitute_before_lock(bound_dir: Path):
        output_dir.rename(displaced_dir)
        output_dir.symlink_to(victim_dir, target_is_directory=True)
        with original_lock(bound_dir) as lock_path:
            yield lock_path

    monkeypatch.setattr(pipeline, "_output_lock", substitute_before_lock)
    with pytest.raises(RuntimeError, match="output directory changed during publication"):
        pipeline.run(manifest_path=MANIFEST, output_dir=output_dir)

    assert list(victim_dir.iterdir()) == []
    assert not (victim_dir / pipeline.LATEST_JSON).exists()
    assert not (victim_dir / pipeline.LATEST_MARKDOWN).exists()


def test_main_prints_hash_counts_and_paths(tmp_path, capsys):
    assert pipeline.main(["--manifest", str(MANIFEST), "--output-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "communication metadata inventory 97c6116f" in output
    assert "(4/4 events)" in output
    assert str(tmp_path / pipeline.LATEST_MARKDOWN) in output


def test_main_accepts_explicit_default_cohort_without_changing_output(tmp_path, capsys):
    assert (
        pipeline.main(
            [
                "--cohort",
                pipeline.DEFAULT_COHORT_ID,
                "--manifest",
                str(MANIFEST),
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "communication metadata inventory 97c6116f" in output
    assert str(tmp_path / pipeline.LATEST_JSON) in output
    assert str(tmp_path / pipeline.LATEST_MARKDOWN) in output


def test_main_dispatches_riksbank_registered_default_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert (
        pipeline.main(
            [
                "--cohort",
                pipeline.RIKSBANK_2025_MANIFEST_ID,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "(8/8 events)" in output
    assert (
        str(
            tmp_path
            / "communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.md"
        )
        in output
    )


def test_main_dispatches_rba_registered_default_path(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert (
        pipeline.main(
            [
                "--cohort",
                pipeline.RBA_2025_MANIFEST_ID,
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "(8/8 events)" in output
    assert (
        str(
            tmp_path / "communication_metadata_rba_2025_monetary_policy_media_conferences_latest.md"
        )
        in output
    )


def test_main_reports_expected_errors_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / "missing.json"

    assert pipeline.main(["--manifest", str(missing), "--output-dir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("dalio-communication-metadata-inventory: ")
    assert "does not exist safely" in captured.err
    assert "Traceback" not in captured.err
