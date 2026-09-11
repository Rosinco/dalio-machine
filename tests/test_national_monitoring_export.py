"""The National panel is published offline as one complete immutable report set."""

import sqlite3
from datetime import date, datetime

import pytest
from sqlalchemy.exc import OperationalError

from dalio.pipelines import build_national_monitoring as export
from tests.test_national_monitoring_core import inputs  # noqa: F401 - shared fixture


@pytest.fixture
def setup(inputs, monkeypatch, tmp_path):  # noqa: F811 - pytest injects imported fixture
    db = tmp_path / "source.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE sentinel (n INTEGER)")
    artifact = tmp_path / "artifacts" / "source.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"original")
    n, p, _ = inputs
    n["protected_artifact_paths"] = [str(artifact)]
    from tests.test_sweden_monitoring import seal
    seal(n)
    def load(engine, **kwargs):
        with engine.connect() as conn, pytest.raises(OperationalError, match="readonly"):
            conn.exec_driver_sql("INSERT INTO sentinel VALUES (1)")
        return n, p
    monkeypatch.setattr(export, "load_inputs", load)
    return db, artifact, datetime.fromisoformat(n["as_known_at"])


def run(setup, target, **kwargs):
    db, _, cutoff = setup
    return export.export_monitoring(db_path=db, as_of=date(2026, 9, 10), as_known_at=cutoff,
                                    bundle_paths=[], output_dir=target, **kwargs)


def test_full_offline_publication_readonly_and_repeat_preserves_bytes_mtimes(setup, tmp_path):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in setup[:2]}
    directory = run(setup, tmp_path / "output")
    names = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    assert names == {"snapshot.json", "index.md", "US.md", "DE.md", "CA.md",
                     "context/US.md", "context/DE.md", "context/CA.md"}
    before.update({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()})
    before[directory.parent / "LATEST.json"] = ((directory.parent / "LATEST.json").read_bytes(),
                                                (directory.parent / "LATEST.json").stat().st_mtime_ns)
    assert run(setup, directory.parent) == directory
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == original for p, original in before.items())


def test_source_protection_and_mutated_immutable_output(setup, tmp_path):
    for output in (setup[0], setup[1], setup[1].parent / "reports"):
        with pytest.raises(ValueError, match="protected|artifacts|directory"):
            run(setup, output)
    directory = run(setup, tmp_path / "output")
    (directory / "DE.md").write_text("tamper")
    with pytest.raises(ValueError, match="immutable"):
        run(setup, directory.parent)


def test_report_failure_does_not_replace_latest(setup, tmp_path, monkeypatch):
    directory = run(setup, tmp_path / "output")
    pointer = directory.parent / "LATEST.json"
    before = pointer.read_bytes()
    monkeypatch.setattr(export, "render_country", lambda *args: (_ for _ in ()).throw(ValueError("render fail")))
    with pytest.raises(ValueError, match="render fail"):
        run(setup, directory.parent)
    assert pointer.read_bytes() == before


def test_missing_database_and_naive_cutoff_do_not_create_source(tmp_path):
    db = tmp_path / "absent.sqlite"
    with pytest.raises(ValueError, match="does not exist"):
        export.export_monitoring(db_path=db)
    assert not db.exists()
    with sqlite3.connect(db):
        pass
    with pytest.raises(ValueError, match="timezone"):
        export.export_monitoring(db_path=db, as_known_at=datetime(2026, 9, 10))
