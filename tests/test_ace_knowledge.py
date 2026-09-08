from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from ace_knowledge import publish_project, fetch_reader_copy, checksum

PID = "123e4567-e89b-12d3-a456-426614174000"

class Store:
    def __init__(self): self.record = None
    def read_compiled_snapshot(self, project_id, version=None): return self.record
    def publish_compiled_snapshot(self, project_id, version, snapshot):
        self.record = {"version": version, "snapshot": snapshot, "checksum": checksum(snapshot)}
        return {"version": version}

def test_versioned_reader_round_trip_never_writes_master(tmp_path):
    master = tmp_path / "master" / "knowledge"
    master.mkdir(parents=True)
    (master / "index.md").write_text("# Knowledge\nDecision source: daily/2026-09-07.md")
    info = {"id": PID, "vault_dir": str(master.parent)}
    store = Store()
    assert publish_project(info, store)["version"] == 1
    assert publish_project(info, store)["unchanged"]
    result = fetch_reader_copy(PID, store=store, cache_root=tmp_path / "reader")
    reader = Path(result["path"])
    assert (reader / "index.md").read_bytes() == (master / "index.md").read_bytes()
    assert (reader / "index.md").stat().st_mode & 0o222 == 0
    assert fetch_reader_copy(PID, store=store, cache_root=tmp_path / "reader")["unchanged"]
    (master / "index.md").write_text("# Knowledge changed")
    assert publish_project(info, store)["version"] == 2
    assert (reader / "index.md").read_text().startswith("# Knowledge\n")

def test_reader_rejects_tampering_and_escaping_paths(tmp_path):
    store = Store()
    snapshot = {"schema_version":1,"project_id":PID,"files":{"../escape.md":"bad"}}
    store.record = {"version":1,"snapshot":snapshot,"checksum":checksum(snapshot)}
    with pytest.raises(ValueError,match="path"):
        fetch_reader_copy(PID,store=store,cache_root=tmp_path)
    store.record["checksum"] = "0" * 64
    with pytest.raises(ValueError,match="checksum"):
        fetch_reader_copy(PID,store=store,cache_root=tmp_path)


def test_publish_rejects_incomplete_bundle_before_store_write(tmp_path):
    master = tmp_path / "master" / "knowledge"
    (master / "concepts").mkdir(parents=True)
    (master / "index.md").write_text(
        "# Knowledge\n\n* [Broken](/concepts/broken.md)\n",
        encoding="utf-8",
    )
    (master / "concepts" / "broken.md").write_text(
        "# Broken\n\n[Missing](/concepts/missing.md)\n",
        encoding="utf-8",
    )
    store = Store()
    with pytest.raises(ValueError, match="incomplete"):
        publish_project({"id": PID, "vault_dir": str(master.parent)}, store)
    assert store.record is None
