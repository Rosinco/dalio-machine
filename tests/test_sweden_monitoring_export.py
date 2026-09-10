"""The pilot exporter must preserve source files and publish complete snapshots."""

import json
import sqlite3
from copy import deepcopy
from datetime import UTC, date, datetime

import pytest
from sqlalchemy.exc import OperationalError

from dalio.assessments.core import content_hash
from dalio.pipelines import build_sweden_monitoring as export

KNOWN = datetime(2026, 9, 10, 23, tzinfo=UTC)
AS_OF = date(2026, 9, 10)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    db = tmp_path / "source.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE sentinel (n INTEGER)")
    artifact = tmp_path / "artifacts" / "source.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"original source")
    bundle = tmp_path / "artifacts" / "bundles" / "input.json"
    bundle.parent.mkdir()
    bundle.write_text("{}")
    snapshot = {"as_of": AS_OF.isoformat(), "as_known_at": KNOWN.isoformat(),
                "signals": [], "scenarios": [], "citations": {},
                "protected_artifact_paths": [str(artifact)]}
    snapshot["snapshot_sha256"] = content_hash(snapshot)

    def evidence(engine, *, as_known_at, bundle_paths):
        assert as_known_at == KNOWN and bundle_paths == [bundle]
        with engine.connect() as connection, pytest.raises(OperationalError, match="readonly"):
            connection.exec_driver_sql("INSERT INTO sentinel VALUES (1)")
        return {"protected_artifact_paths": [str(artifact)]}

    monkeypatch.setattr(export, "load_evidence", evidence)
    monkeypatch.setattr(export, "load_assessment", lambda engine, **kwargs: {})
    monkeypatch.setattr(export, "build_monitoring", lambda *args, **kwargs: deepcopy(snapshot))
    monkeypatch.setattr(export, "render_monitoring", lambda value: "# Sweden\n\nVerified report.\n")
    return db, artifact, bundle, snapshot


def run(setup, output_dir, **kwargs):
    db, _, bundle, _ = setup
    return export.export_monitoring(db_path=db, as_of=AS_OF, as_known_at=KNOWN,
                                    bundle_paths=[bundle], output_dir=output_dir, **kwargs)


def test_readonly_complete_publication_and_repeat_preserve_all_bytes_and_mtimes(setup, tmp_path):
    db, artifact, bundle, snapshot = setup
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (db, artifact, bundle)}
    directory = run(setup, tmp_path / "output")
    assert directory.name == snapshot["snapshot_sha256"]
    assert {p.name for p in directory.iterdir()} == {"snapshot.json", "SE.md"}
    files = [*directory.iterdir(), directory.parent / "LATEST.json"]
    rendered = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert run(setup, tmp_path / "output") == directory
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == values for p, values in original.items())
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == values for p, values in rendered.items())


def test_no_latest_is_available_for_review_before_publication(setup, tmp_path):
    directory = run(setup, tmp_path / "output", update_latest=False)
    assert directory.is_dir()
    assert not (directory.parent / "LATEST.json").exists()


def test_invalid_hash_or_render_failure_cannot_replace_latest(setup, tmp_path, monkeypatch):
    directory = run(setup, tmp_path / "output")
    pointer = directory.parent / "LATEST.json"
    old = pointer.read_bytes()
    setup[3]["as_of"] = "2026-09-09"
    with pytest.raises(ValueError, match="hash"):
        run(setup, directory.parent)
    setup[3]["as_of"] = AS_OF.isoformat()
    monkeypatch.setattr(export, "render_monitoring", lambda value: (_ for _ in ()).throw(ValueError("bad report")))
    with pytest.raises(ValueError, match="bad report"):
        run(setup, directory.parent)
    assert pointer.read_bytes() == old


def test_protected_sources_and_artifact_namespace_cannot_be_outputs(setup, tmp_path):
    db, artifact, bundle, _ = setup
    for target in (db, artifact, bundle, artifact.parent / "reports"):
        with pytest.raises(ValueError, match="protected|artifacts"):
            run(setup, target)


def test_mutated_immutable_report_is_not_overwritten(setup, tmp_path):
    directory = run(setup, tmp_path / "output")
    report = directory / "SE.md"
    report.write_text("tampered")
    with pytest.raises(ValueError, match="immutable"):
        run(setup, directory.parent)
    assert report.read_text() == "tampered"


def test_missing_database_naive_clock_and_future_asof_fail_without_outputs(tmp_path):
    absent = tmp_path / "absent.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        export.export_monitoring(db_path=absent)
    assert not absent.exists()
    with sqlite3.connect(absent):
        pass
    with pytest.raises(ValueError, match="timezone"):
        export.export_monitoring(db_path=absent, as_known_at=datetime(2026, 9, 10))
    with pytest.raises(ValueError, match="cutoff"):
        export.export_monitoring(db_path=absent, as_of=date(2026, 9, 11), as_known_at=KNOWN)


def test_cli_resolves_explicit_bundle_paths_and_reports_directory(setup, tmp_path, capsys):
    db, _, bundle, _ = setup
    assert export.main(["--db", str(db), "--as-of", str(AS_OF), "--known-at", KNOWN.isoformat(),
                        "--bundle", str(bundle), "--output-dir", str(tmp_path / "output")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["snapshot_sha256"] in result["directory"]
