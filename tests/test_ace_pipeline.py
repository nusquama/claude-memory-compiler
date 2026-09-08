from __future__ import annotations

import os
import json
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ace_pipeline as pipeline


@dataclass
class Project:
    root: Path
    vault_dir: Path
    project_id: str = "project-1"
    name: str = "demo"


class Projects:
    def __init__(self, project: Project | None):
        self.project = project
        self.resolve_calls = 0

    def resolve_project(self, root, *, vault_root=None, strict=False):
        self.resolve_calls += 1
        if self.project is None:
            return None
        return self.project if Path(root).resolve() == self.project.root.resolve() else None

    def list_initialized_projects(self):
        return [self.project] if self.project else []


class Transcripts:
    def __init__(self):
        self.parse_calls = 0

    def iter_snapshots(self, sources, project):
        return [{"path": Path(source), "source": "test", "mtime": Path(source).stat().st_mtime} for source in sources]

    def parse_transcript(self, path, source, project, host_id=None):
        self.parse_calls += 1
        return {"snapshot_id": Path(path).stem, "source": source, "context": Path(path).read_text(), "host_id": host_id}


class Outbox:
    def __init__(self):
        self.items: list[dict] = []

    def enqueue(self, envelope):
        if not any(item["key"] == envelope["snapshot_id"] for item in self.items):
            self.items.append({"key": envelope["snapshot_id"], "envelope": envelope, "status": "pending"})
        return envelope["snapshot_id"]

    def pending(self, limit=None):
        values = [item for item in self.items if item["status"] == "pending"]
        return values[:limit] if limit else values

    def ack(self, key, receipt=None):
        for item in self.items:
            if item["key"] == key:
                item["status"] = "acked"
                item["receipt"] = receipt

    def fail(self, key, error):
        for item in self.items:
            if item["key"] == key:
                item["error"] = error


class Store:
    def __init__(self):
        self.snapshots: list[dict] = []

    def upsert_snapshot(self, envelope):
        if not any(item["snapshot_id"] == envelope["snapshot_id"] for item in self.snapshots):
            self.snapshots.append(envelope)
        return {"receipt": "db-receipt"}

    def list_acquitted_snapshots(self, *, project_id, limit):
        return [item for item in self.snapshots if item["project_id"] == project_id][:limit]


def make_pipeline(tmp_path: Path, *, project=True, store=None, extractor=None, learning=None):
    repo = tmp_path / "repo"
    repo.mkdir()
    vault = tmp_path / "vault" / "demo"
    (vault / "daily").mkdir(parents=True)
    info = Project(repo, vault) if project else None
    projects = Projects(info)
    transcripts = Transcripts()
    outbox = Outbox()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=projects,
        transcripts=transcripts,
        outbox=outbox,
        store=store,
        extractor=extractor,
        learning=learning,
    )
    return runner, info, projects, transcripts, outbox


def test_end_to_end_collect_process_writes_daily_before_db_ack(tmp_path):
    """Memory first: the daily log comes from the local envelope, the DB later."""
    source = tmp_path / "source.jsonl"
    source.write_text("**User:** keep this decision", encoding="utf-8")
    store = Store()
    runner, project, _projects, transcripts, outbox = make_pipeline(
        tmp_path, store=store, extractor=lambda context: "**Problème original**\n- decision"
    )

    collected = runner.collect([source], cwd=project.root, source="test", host_id="host-a")
    assert collected["queued"] == 1
    assert transcripts.parse_calls == 1
    assert outbox.items[0]["status"] == "pending"
    processed = runner.process(project=project)
    assert processed["processed"] == 1
    assert processed["local"]["processed"] == 1
    daily = next((project.vault_dir / "daily").glob("*.md"))
    assert "decision" in daily.read_text(encoding="utf-8")
    assert "ace-snapshot" in daily.read_text(encoding="utf-8")

    synced = runner.sync(project=project)
    assert synced["synced"] == 1
    assert synced["pending"] == 0
    # The remote path never replays a locally extracted revision.
    assert runner.process(project=project)["processed"] == 0
    assert len(list((project.vault_dir / "daily").glob("*.md"))) == 1


def test_database_mode_keeps_extraction_after_db_ack(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_EXTRACTION_MODE", "database")
    source = tmp_path / "source.jsonl"
    source.write_text("**User:** keep this decision", encoding="utf-8")
    store = Store()
    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path, store=store, extractor=lambda context: "- decision"
    )
    runner.collect([source], cwd=project.root, source="test", host_id="host-a")
    assert runner.process(project=project)["processed"] == 0
    runner.sync(project=project)
    assert runner.process(project=project)["processed"] == 1


def test_process_local_works_without_any_store(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("**User:** offline decision", encoding="utf-8")
    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path, store=None, extractor=lambda context: "- offline decision"
    )
    runner.collect([source], cwd=project.root, source="test", host_id="host-a")
    processed = runner.process(project=project)
    assert processed["offline"] is True
    assert processed["processed"] == 1
    daily = next((project.vault_dir / "daily").glob("*.md"))
    assert "offline decision" in daily.read_text(encoding="utf-8")


def test_process_sends_only_new_messages_after_session_cursor(tmp_path):
    store_rows = [
        {
            "project_id": "project-1",
            "source": "test",
            "session_id": "session-1",
            "revision": "a" * 64,
            "snapshot": {
                "project": {"id": "project-1"},
                "source": "test",
                "session_id": "session-1",
                "revision": "a" * 64,
                "messages": [
                    {"id": "m-1", "ordinal": 1, "role": "user", "type": "message", "content": "old one"},
                    {"id": "m-2", "ordinal": 2, "role": "assistant", "type": "message", "content": "old two"},
                ],
            },
        },
        {
            "project_id": "project-1",
            "source": "test",
            "session_id": "session-1",
            "revision": "b" * 64,
            "snapshot": {
                "project": {"id": "project-1"},
                "source": "test",
                "session_id": "session-1",
                "revision": "b" * 64,
                "messages": [
                    {"id": "m-1", "ordinal": 1, "role": "user", "type": "message", "content": "old one"},
                    {"id": "m-2", "ordinal": 2, "role": "assistant", "type": "message", "content": "old two"},
                    {"id": "m-3", "ordinal": 3, "role": "user", "type": "message", "content": "new three"},
                ],
            },
        },
    ]

    class Store:
        def __init__(self):
            self.marked: set[str] = set()

        def list_acquitted_snapshots(self, *, project_id, limit, stage="extraction"):
            return [
                row
                for row in store_rows
                if row["project_id"] == project_id and row["revision"] not in self.marked
            ][:limit]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error=None):
            assert stage == "extraction"
            assert status == "succeeded"
            self.marked.add(revision)

    contexts = []
    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path,
        store=Store(),
        extractor=lambda context: contexts.append(context) or "delta extracted",
    )

    assert runner.process(project=project, limit=1)["processed"] == 1
    assert runner.process(project=project, limit=1)["processed"] == 1
    assert len(contexts) == 2
    assert "old one" in contexts[0] and "old two" in contexts[0]
    assert "new three" in contexts[1]
    assert "old one" not in contexts[1]
    assert "old two" not in contexts[1]


def test_incremental_first_pass_baselines_existing_session_then_processes_delta(tmp_path):
    cutoff = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()
    store_rows = [
        {
            "project_id": "project-1",
            "source": "test",
            "session_id": "long-session",
            "revision": "c" * 64,
            "updated_at": "2026-09-07T10:00:01Z",
            "snapshot": {
                "project": {"id": "project-1"},
                "source": "test",
                "session_id": "long-session",
                "revision": "c" * 64,
                "started_at": "2026-09-07T09:00:00Z",
                "updated_at": "2026-09-07T10:00:01Z",
                "messages": [
                    {"id": "old-1", "ordinal": 1, "timestamp": "2026-09-07T09:01:00Z", "role": "user", "type": "message", "content": "historical"},
                    {"id": "old-2", "ordinal": 2, "timestamp": "2026-09-07T09:02:00Z", "role": "assistant", "type": "message", "content": "historical answer"},
                ],
            },
        },
        {
            "project_id": "project-1",
            "source": "test",
            "session_id": "long-session",
            "revision": "d" * 64,
            "updated_at": "2026-09-07T10:01:00Z",
            "snapshot": {
                "project": {"id": "project-1"},
                "source": "test",
                "session_id": "long-session",
                "revision": "d" * 64,
                "started_at": "2026-09-07T09:00:00Z",
                "updated_at": "2026-09-07T10:01:00Z",
                "messages": [
                    {"id": "old-1", "ordinal": 1, "timestamp": "2026-09-07T09:01:00Z", "role": "user", "type": "message", "content": "historical"},
                    {"id": "old-2", "ordinal": 2, "timestamp": "2026-09-07T09:02:00Z", "role": "assistant", "type": "message", "content": "historical answer"},
                    {"id": "new-3", "ordinal": 3, "timestamp": "2026-09-07T10:01:00Z", "role": "user", "type": "message", "content": "current request"},
                ],
            },
        },
    ]

    class Store:
        def __init__(self):
            self.marked: set[str] = set()

        def list_acquitted_snapshots(self, *, project_id, limit, stage="extraction"):
            return [
                row
                for row in store_rows
                if row["project_id"] == project_id and row["revision"] not in self.marked
            ][:limit]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error=None):
            assert stage == "extraction"
            assert status == "succeeded"
            self.marked.add(revision)

    contexts = []
    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path,
        store=Store(),
        extractor=lambda context: contexts.append(context) or "delta extracted",
    )

    first = runner.process(project=project, limit=1, minimum_started_at=cutoff)
    assert first["baseline"] == 1
    assert first["processed"] == 0
    assert contexts == []

    second = runner.process(project=project, limit=1, minimum_started_at=cutoff)
    assert second["processed"] == 1
    assert len(contexts) == 1
    assert "current request" in contexts[0]
    assert "historical" not in contexts[0]


def test_interrupted_batch_keeps_failed_snapshot_pending_for_retry(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    store = Store()
    calls = {"count": 0}

    def extractor(_context):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("interrupted")
        return "first result" if calls["count"] == 1 else "second result"

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=store, extractor=extractor)
    runner.collect([first, second], cwd=project.root, source="test", limit=4)
    runner.sync(project=project)
    result = runner.process(project=project)
    assert result["processed"] == 1
    assert result["pending"] == 1
    assert runner.process(project=project)["processed"] == 1


def test_collect_refuses_uninitialized_cwd_before_parsing(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("must not be read", encoding="utf-8")
    runner, _project, projects, transcripts, _outbox = make_pipeline(tmp_path, project=False)
    with pytest.raises(pipeline.NotInitializedError):
        runner.collect([source], cwd=tmp_path / "unknown", source="test")
    assert projects.resolve_calls == 1
    assert transcripts.parse_calls == 0


def test_daily_retry_does_not_mark_success_after_compile_failure(tmp_path):
    class Learning:
        def __init__(self):
            self.calls = 0

        def compile_daily(self, project, day):
            self.calls += 1
            if self.calls == 1:
                return False
            return True

    learning = Learning()
    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=Store(), learning=learning)
    first = runner.daily(project=project, day="2026-09-06")
    assert first["failed"] == 1
    assert runner.state.read("compile").get("last_successful_day") is None
    second = runner.daily(project=project, day="2026-09-06")
    assert second["compiled"] == 1
    assert runner.state.read("compile")["last_successful_day"] == "2026-09-06"


def test_default_extractor_sets_recursion_guard_and_restores_environment(monkeypatch, tmp_path):
    observed = {}

    def run_flush(context):
        observed["context"] = context
        observed["guard"] = os.environ.get("CLAUDE_INVOKED_BY")
        observed["backfill"] = os.environ.get("CODEX_ACE_BACKFILL_ENABLED")
        return "extracted", []

    fake_flush = types.SimpleNamespace(run_flush=run_flush)
    monkeypatch.setitem(sys.modules, "flush", fake_flush)
    monkeypatch.delenv("CLAUDE_INVOKED_BY", raising=False)
    monkeypatch.delenv("CODEX_ACE_BACKFILL_ENABLED", raising=False)
    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=Store())
    assert runner._default_extractor("safe context") == ("extracted", [])
    assert observed == {"context": "safe context", "guard": "memory_flush", "backfill": "0"}
    assert "CLAUDE_INVOKED_BY" not in os.environ
    assert "CODEX_ACE_BACKFILL_ENABLED" not in os.environ


def test_offline_sync_is_not_reported_as_success(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("queued", encoding="utf-8")
    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=None)
    assert runner.collect([source], cwd=project.root, source="test")["queued"] == 1
    result = runner.sync(project=project)
    assert result["synced"] == 0
    assert result["offline"] is True
    assert result["pending"] == 1


def test_incremental_automation_freezes_first_run_cutoff(tmp_path, monkeypatch):
    old = tmp_path / "old.jsonl"
    old.write_text("historical", encoding="utf-8")
    old_timestamp = time.time() - 120
    os.utime(old, (old_timestamp, old_timestamp))
    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    monkeypatch.setenv("ACE_AUTOMATION_MODE", "incremental")

    assert runner._source_paths([old], all_history=False, days=7) == []
    since = runner.state.read("collection")["automation_since"]
    assert float(since) > old_timestamp

    fresh = tmp_path / "fresh.jsonl"
    fresh.write_text("new", encoding="utf-8")
    selected = runner._source_paths([fresh], all_history=False, days=7)
    assert [item["path"] for item in selected] == [fresh.resolve()]


def test_incremental_processing_uses_snapshot_update_time_not_session_start(tmp_path, monkeypatch):
    class Store:
        def pending_snapshots(self, *, project_id=None, limit=100, stage="extraction"):
            rows = []
            for index in range(5):
                rows.append({
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": f"old-{index}",
                    "revision": (str(index) + "b" * 63),
                    "snapshot": {
                        "project": {"id": project.project_id},
                        "source": "codex",
                        "session_id": f"old-{index}",
                        "revision": (str(index) + "b" * 63),
                        "started_at": "2026-09-01T10:00:00Z",
                        "updated_at": "2026-09-01T10:00:00Z",
                        "messages": [{"id": "m", "ordinal": 0, "role": "user", "type": "message", "content": "old"}],
                        "attachments": [],
                    },
                })
            rows.append({
                "project_id": project.project_id,
                "source": "codex",
                "session_id": "old-session-now-updated",
                "revision": "a" * 64,
                "snapshot": {
                    "project": {"id": project.project_id},
                    "source": "codex",
                    "session_id": "old-session-now-updated",
                    "revision": "a" * 64,
                    "started_at": "2026-09-01T10:00:00Z",
                    "updated_at": "2026-09-07T19:00:00Z",
                    "messages": [{"id": "m", "ordinal": 0, "role": "user", "type": "message", "content": "now"}],
                    "attachments": [],
                },
            })
            return rows[:limit]

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=Store())
    monkeypatch.setenv("ACE_AUTOMATION_MODE", "incremental")
    monkeypatch.setattr(runner, "_automation_since", lambda *, create=False: 1788799000.0)

    rows = runner._db_snapshots(
        runner._get_store(), project, 1, minimum_started_at=1788799000.0
    )
    assert len(rows) == 1


def test_incremental_supabase_refs_skip_historical_pending_queue(tmp_path):
    cutoff = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()

    class RefStore:
        def pending_snapshot_refs(self, *, project_id, limit, stage="extraction"):
            assert project_id == "project-1"
            assert stage == "extraction"
            return [
                {
                    "project_id": project_id,
                    "source": "codex",
                    "session_id": "old-session",
                    "revision": "a" * 64,
                    "updated_at": "2026-09-01T10:00:00Z",
                    "message_count": 200,
                },
                {
                    "project_id": project_id,
                    "source": "codex",
                    "session_id": "new-session",
                    "revision": "b" * 64,
                    "updated_at": "2026-09-07T10:01:00Z",
                    "message_count": 3,
                },
            ][:limit]

        def snapshot_delta(self, *args, **kwargs):
            raise AssertionError("a first observation must baseline, not fetch a delta")

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=RefStore())
    rows = runner._db_snapshots(
        runner._get_store(), project, 1, minimum_started_at=cutoff, extraction_cursors={}
    )
    assert [row["session_id"] for row in rows] == ["new-session"]


def test_incremental_supabase_refs_fetch_first_delta_for_new_session(tmp_path):
    cutoff = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()

    class RefStore:
        def pending_snapshot_refs(self, *, project_id, limit, stage="extraction"):
            return [
                {
                    "project_id": project_id,
                    "source": "codex",
                    "session_id": "new-session",
                    "revision": "c" * 64,
                    "started_at": "2026-09-07T10:01:00Z",
                    "updated_at": "2026-09-07T10:01:00Z",
                    "message_count": 2,
                }
            ]

        def snapshot_delta(self, *args, **kwargs):
            assert kwargs.get("last_ordinal", args[-1] if args else None) == -1
            return {
                "project_id": "project-1",
                "source": "codex",
                "session_id": "new-session",
                "revision": "c" * 64,
                "messages": [
                    {"id": "m-1", "ordinal": 0, "role": "user", "type": "message", "content": "current"},
                    {"id": "m-2", "ordinal": 1, "role": "assistant", "type": "message", "content": "answer"},
                ],
            }

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=RefStore())
    rows = runner._db_snapshots(
        runner._get_store(), project, 1, minimum_started_at=cutoff, extraction_cursors={}
    )
    assert len(rows) == 1
    assert [message["content"] for message in rows[0]["messages"]] == ["current", "answer"]


def test_snapshot_ref_window_pushes_effective_lower_bound_before_limit(tmp_path):
    cutoff = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()
    captured = {}

    class RefStore:
        def pending_snapshot_refs(self, **kwargs):
            captured.update(kwargs)
            return [
                {
                    "project_id": "project-1",
                    "source": "codex",
                    "session_id": "current",
                    "revision": "e" * 64,
                    "started_at": "2026-09-07T10:01:00Z",
                    "updated_at": "2026-09-07T10:01:00Z",
                }
            ]

        def snapshot_delta(self, *args, **kwargs):
            return None

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=RefStore())
    rows = runner._db_snapshots(
        runner._get_store(),
        project,
        1,
        minimum_started_at=cutoff,
        source_after=cutoff - 3600,
        source_before=cutoff + 3600,
        extraction_cursors={},
    )
    assert captured["source_after"] == cutoff
    assert captured["source_before"] == cutoff + 3600
    assert captured["limit"] == 64
    assert len(rows) == 1


def test_snapshot_ref_window_without_incremental_cutoff_does_not_compare_none(tmp_path):
    window_start = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()

    class RefStore:
        def pending_snapshot_refs(self, **kwargs):
            return [
                {
                    "project_id": "project-1",
                    "source": "codex",
                    "session_id": "manual-window",
                    "revision": "f" * 64,
                    "started_at": "2026-09-07T10:01:00Z",
                    "updated_at": "2026-09-07T10:01:00Z",
                }
            ]

        def snapshot_delta(self, *args, **kwargs):
            assert args[-1] == -1
            return {"project_id": "project-1", "source": "codex", "session_id": "manual-window",
                    "revision": "f" * 64, "messages": [{"id": "current", "ordinal": 0,
                    "role": "user", "type": "message", "timestamp": "2026-09-07T10:01:00Z",
                    "content": "current evidence"}]}

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=RefStore())
    rows = runner._db_snapshots(
        runner._get_store(),
        project,
        1,
        stage="analysis",
        source_after=window_start,
        source_before=window_start + 3600,
        extraction_cursors={},
    )
    assert len(rows) == 1
    assert rows[0]["session_id"] == "manual-window"
    assert rows[0]["messages"][0]["id"] == "current"


def test_manual_snapshot_ref_window_batches_delta_reads(tmp_path):
    window_start = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()

    class RefStore:
        def pending_snapshot_refs(self, **kwargs):
            return [
                {
                    "project_id": "project-1",
                    "source": "codex",
                    "session_id": "manual-one",
                    "revision": "1" + "a" * 63,
                    "started_at": "2026-09-07T10:01:00Z",
                    "updated_at": "2026-09-07T10:01:00Z",
                },
                {
                    "project_id": "project-1",
                    "source": "codex",
                    "session_id": "manual-two",
                    "revision": "2" + "b" * 63,
                    "started_at": "2026-09-07T10:02:00Z",
                    "updated_at": "2026-09-07T10:02:00Z",
                },
            ]

        def snapshot_deltas(self, requests):
            assert len(requests) == 2
            return [
                {
                    "project_id": request["project_id"],
                    "source": request["source"],
                    "session_id": request["session_id"],
                    "revision": request["revision"],
                    "started_at": "2026-09-07T10:01:00Z",
                    "updated_at": "2026-09-07T10:01:00Z",
                    "messages": [{
                        "id": f"{request['session_id']}-message",
                        "ordinal": 0,
                        "role": "user",
                        "type": "message",
                        "timestamp": "2026-09-07T10:01:00Z",
                        "content": "current evidence",
                    }],
                }
                for request in requests
            ]

        def snapshot_delta(self, *args, **kwargs):
            raise AssertionError("manual source windows must use the batch reader")

    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path, store=RefStore())
    rows = runner._db_snapshots(
        runner._get_store(),
        project,
        10,
        stage="analysis",
        source_after=window_start,
        source_before=window_start + 3600,
    )

    assert [row["session_id"] for row in rows] == ["manual-one", "manual-two"]


def test_automation_daily_window_is_once_per_morning(tmp_path, monkeypatch):
    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    monkeypatch.setenv("ACE_AUTOMATION_MODE", "incremental")
    monkeypatch.setenv("ACE_DAILY_REPORT_TARGET", "08:00")
    morning = datetime(2026, 9, 7, 8, 15, tzinfo=timezone.utc)
    evening = datetime(2026, 9, 7, 20, 15, tzinfo=timezone.utc)

    assert runner._automation_daily_due(morning) is True
    runner._claim_automation_daily(morning)
    assert runner._automation_daily_due(morning) is False
    assert runner._automation_daily_due(evening) is False


def test_incremental_new_codex_session_starts_at_zero_and_old_session_baselines(tmp_path):
    source = tmp_path / "new-session.jsonl"
    source.write_text("first\nsecond\n", encoding="utf-8")
    cutoff = datetime(2026, 9, 7, 10, tzinfo=timezone.utc).timestamp()

    class IncrementalTranscripts:
        def __init__(self):
            self.offsets = []

        def parse_codex_incremental(self, path, project, offset, *, ordinal_start, host_id, session_id):
            self.offsets.append((offset, ordinal_start, session_id))
            return {
                "schema_version": 1,
                "session_id": session_id,
                "source": "codex",
                "revision": "a" * 64,
                "messages": [
                    {"id": "m1", "ordinal": 0, "role": "user", "type": "message", "content": "first"}
                ],
            }, path.stat().st_size

    transcripts = IncrementalTranscripts()
    runner, project, _projects, _old_transcripts, _outbox = make_pipeline(tmp_path, store=None)
    runner.transcripts = transcripts
    new = runner._incremental_codex_capture(
        {
            "path": source,
            "source": "codex",
            "session_id": "new-session",
            "started_at": "2026-09-07T10:01:00Z",
        },
        project,
        "host-a",
        {},
        automation_since=cutoff,
    )
    assert new is not None
    assert transcripts.offsets == [(0, 0, "new-session")]
    assert new["envelope"] is not None

    old = runner._incremental_codex_capture(
        {
            "path": source,
            "source": "codex",
            "session_id": "old-session",
            "started_at": "2026-09-01T10:01:00Z",
        },
        project,
        "host-a",
        {},
        automation_since=cutoff,
    )
    assert old is not None
    assert old["state"]["status"] == "baseline"
    assert old["envelope"] is None
    assert len(transcripts.offsets) == 1


def test_automation_daily_failure_is_retryable_then_bounded(tmp_path, monkeypatch):
    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    monkeypatch.setenv("ACE_AUTOMATION_MODE", "incremental")
    current = datetime(2026, 9, 7, 8, 15, tzinfo=timezone.utc)

    assert runner._automation_daily_due(current) is True
    runner._claim_automation_daily(current, {"failed": 1, "pending": 1})
    assert runner._automation_daily_due(current) is True
    runner._claim_automation_daily(current, {"failed": 1, "pending": 1})
    assert runner._automation_daily_due(current) is True
    runner._claim_automation_daily(current, {"failed": 1, "pending": 1})
    assert runner._automation_daily_due(current) is False
    record = runner.state.read("collection")["automation_daily"]["2026-09-07"]
    assert record["attempts"] == 3
    assert record["retry_exhausted"] is True


def test_automation_daily_pending_continuations_do_not_consume_failure_budget(tmp_path, monkeypatch):
    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    monkeypatch.setenv("ACE_AUTOMATION_MODE", "incremental")
    current = datetime(2026, 9, 7, 8, 15, tzinfo=timezone.utc)

    for _ in range(4):
        assert runner._automation_daily_due(current) is True
        runner._claim_automation_daily(current, {"failed": 0, "pending": 1})

    record = runner.state.read("collection")["automation_daily"]["2026-09-07"]
    assert record["status"] == "pending"
    assert record["attempts"] == 4
    assert record["retry_failures"] == 0
    assert record.get("retry_exhausted") is False
    assert runner._automation_daily_due(current) is True

    runner._claim_automation_daily(current, {"failed": 0, "pending": 0})
    assert runner._automation_daily_due(current) is False


def test_outbox_uses_canonical_limit_keywords(tmp_path):
    captured = {}

    class CaptureOutbox:
        def __init__(self, db_path, **kwargs):
            captured["db_path"] = db_path
            captured["kwargs"] = kwargs

    runner, _project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    runner.outbox_integration = types.SimpleNamespace(Outbox=CaptureOutbox)
    runner._outbox_instance = None
    runner._get_outbox()
    assert captured["kwargs"] == {
        "max_payload_bytes": 50_000_000,
        "max_lot_items": 100,
    }


def test_safe_analysis_value_keeps_deep_cause_as_redacted_structure():
    value = {
        "a": {
            "b": {
                "c": {
                    "d": {
                        "cause": {
                            "status": "verified",
                            "evidence_refs": ["token=should-not-leak"],
                            "nested": {"secret": "password=should-not-leak"},
                        }
                    }
                }
            }
        }
    }
    safe = pipeline.ACEPipeline._safe_analysis_value(value)
    cause = safe["a"]["b"]["c"]["d"]["cause"]
    assert isinstance(cause, dict)
    assert cause["status"] == "verified"
    assert cause["evidence_refs"] == ["<truncated>"]
    assert cause["nested"] == {"_truncated": True}
    assert "should-not-leak" not in str(safe)


def test_daily_filters_message_day_before_analysis_limit(tmp_path):
    target = datetime(2026, 9, 7, tzinfo=timezone.utc).date()
    project_id = "project-1"
    stale = {
        "project_id": project_id,
        "source": "codex",
        "session_id": "stale",
        "revision": "a" * 64,
        # The revision was updated on the target day, but all its evidence is
        # from an older source day and must not consume the bounded batch.
        "started_at": "2026-09-06T10:00:00Z",
        "updated_at": "2026-09-07T10:00:00Z",
        "messages": [
            {
                "id": "old",
                "ordinal": 0,
                "role": "user",
                "type": "message",
                "timestamp": "2026-09-06T10:00:00Z",
                "content": "old evidence",
            }
        ],
    }
    current = {
        "project_id": project_id,
        "source": "codex",
        "session_id": "current",
        "revision": "b" * 64,
        "started_at": "2026-09-07T10:00:00Z",
        "updated_at": "2026-09-07T10:00:00Z",
        "messages": [
            {
                "id": "new",
                "ordinal": 0,
                "role": "user",
                "type": "message",
                "timestamp": "2026-09-07T10:00:00Z",
                "content": "current evidence",
            }
        ],
    }

    class Store:
        def pending_snapshots(self, **kwargs):
            assert kwargs["source_after"] < kwargs["source_before"]
            return [stale, current][: kwargs["limit"]]

        def mark_processed(self, *args, **kwargs):
            return {"accepted": True}

    class Learning:
        def __init__(self):
            self.sessions = []

        def audit_snapshots_sync(self, snapshots, **kwargs):
            self.sessions.append([row["session_id"] for row in snapshots])
            return {
                "reports": [
                    {"session_id": row["session_id"], "analysis_status": "ok"}
                    for row in snapshots
                ],
                "coverage": {"sessions": len(snapshots)},
                "errors": [],
            }

    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path,
        store=Store(),
        learning=Learning(),
    )
    assert runner._default_analyze_daily(project, target.isoformat()) is True
    assert runner.learning.sessions == [["current"]]


def test_extraction_claim_is_renewed_before_each_chunk_and_ack(tmp_path):
    project_id = "project-1"
    row = {
        "project_id": project_id,
        "source": "codex",
        "session_id": "claimed-session",
        "revision": "c" * 64,
        "messages": [
            {"id": "m1", "ordinal": 0, "role": "user", "type": "message", "content": "one"},
            {"id": "m2", "ordinal": 1, "role": "assistant", "type": "message", "content": "two"},
        ],
    }

    class ClaimedStore:
        def __init__(self):
            self.claims = []
            self.marks = []

        def list_acquitted_snapshots(self, **kwargs):
            return [row]

        def claim_stage(self, project_id, host_id, session_id, revision, *, source, stage, lease_owner, lease_seconds=1800):
            self.claims.append((project_id, host_id, session_id, revision, source, stage, lease_owner))
            return {"claimed": True, "lease_id": f"lease-{len(self.claims)}"}

        def mark_stage(self, source, session_id, revision, project_id, stage, status, error=None, *, lease_owner, host_id):
            self.marks.append((source, session_id, revision, project_id, stage, status, lease_owner, host_id))
            return {"accepted": True, "status": status}

    store = ClaimedStore()
    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path,
        store=store,
        extractor=lambda chunk: f"extracted:{chunk}",
    )
    result = runner.process(project=project, max_context_chars=8)
    assert result["processed"] == 1
    # Initial claim plus at least one renewal per chunk and before the final
    # write/ack. Every call keeps one owner and the execution host identity.
    assert len(store.claims) >= 5
    assert {claim[-1] for claim in store.claims} == {runner._lease_owner}
    assert len(store.marks) == 1
    assert store.marks[0][6:] == (
        runner._lease_owner,
        runner.host_id,
    )


def test_extraction_diagnostics_are_persisted_per_snapshot_and_include_chunks(tmp_path):
    row = {
        "project_id": "project-1",
        "source": "codex",
        "session_id": "measured-session",
        "revision": "d" * 64,
        "started_at": "2026-09-07T10:00:00Z",
        "snapshot": {
            "project": {"id": "project-1"},
            "source": "codex",
            "session_id": "measured-session",
            "revision": "d" * 64,
            "started_at": "2026-09-07T10:00:00Z",
            "messages": [
                {"id": "m1", "ordinal": 0, "role": "user", "type": "message", "content": "one"},
                {"id": "m2", "ordinal": 1, "role": "assistant", "type": "message", "content": "two"},
            ],
        },
    }

    class Diagnostics:
        def __init__(self):
            self.calls = 1
            self.duration = 0.25

        def merge(self, other):
            self.calls += other.calls
            self.duration += other.duration

        def as_metrics(self):
            return {
                "call_count": self.calls,
                "duration_seconds": self.duration,
                "token_usage": {
                    "input_tokens": self.calls * 10,
                    "cached_input_tokens": 0,
                    "output_tokens": self.calls * 2,
                },
                "usage_status": "available",
            }

    class Store:
        def list_acquitted_snapshots(self, **kwargs):
            return [row]

        def mark_processed(self, *args, **kwargs):
            return {"accepted": True}

    runner, project, _projects, _transcripts, _outbox = make_pipeline(
        tmp_path,
        store=Store(),
        extractor=lambda chunk: ("measured extraction", Diagnostics()),
    )
    result = runner.process(project=project, max_context_chars=8)
    assert result["processed"] == 1

    state = runner.state.read("extraction")
    record = next(iter(state["snapshots"].values()))
    metrics = record["stage_metrics"]["extraction"]
    assert record["source_day"] == "2026-09-07"
    assert record["session_id"] == "measured-session"
    assert metrics["call_count"] >= 2
    assert metrics["duration_seconds"] == pytest.approx(metrics["call_count"] * 0.25)
    assert metrics["token_usage"]["output_tokens"] == metrics["call_count"] * 2
    assert metrics["usage_status"] == "available"


def test_daily_audit_exposes_measured_extraction_and_compile_usage(tmp_path):
    runner, project, _projects, _transcripts, _outbox = make_pipeline(tmp_path)
    runner.state.write(
        "extraction",
        {
            "snapshots": {
                "project-1:codex:s1:r1": {
                    "project_id": "project-1",
                    "source": "codex",
                    "session_id": "s1",
                    "revision": "r1",
                    "source_day": "2026-09-07",
                    "stage_metrics": {
                        "extraction": {
                            "call_count": 2,
                            "duration_seconds": 1.5,
                            "token_usage": {"input_tokens": 20, "output_tokens": 4},
                            "usage_status": "available",
                        }
                    },
                }
            }
        },
    )
    compile_state = project.vault_dir / ".state" / "state.json"
    compile_state.parent.mkdir(parents=True)
    compile_state.write_text(
        '{"ingested": {"2026-09-07.md": {"stage_metrics": {"compile": '
        '{"call_count": 1, "duration_seconds": 0.5, '
        '"token_usage": {"input_tokens": 8, "output_tokens": 2}, '
        '"usage_status": "available"}}}}}',
        encoding="utf-8",
    )

    assert runner._default_analyze_daily(project, "2026-09-07") is True
    audit = tmp_path / "private" / "audits" / project.project_id / "2026-09-07.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["stage_usage"]["extraction"]["call_count"] == 2
    assert payload["stage_usage"]["compile"]["token_usage"]["output_tokens"] == 2
    assert payload["stage_metrics"] == payload["stage_usage"]
