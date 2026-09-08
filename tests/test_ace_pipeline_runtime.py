from __future__ import annotations

import sys
import types
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ace_pipeline as pipeline
import ace_daily_report
from ace_outbox import Outbox as SQLiteOutbox


@dataclass
class Project:
    root: Path
    vault_dir: Path
    project_id: str
    name: str


class Registry:
    def __init__(self, project: Project):
        self.project = project
        self.vault_root_seen = None

    def resolve_project(self, root, *, vault_root=None, strict=False):
        return self.project if Path(root).resolve() == self.project.root.resolve() else None

    def list_initialized_projects(self, *, vault_root=None):
        self.vault_root_seen = vault_root
        return [self.project]


class MetadataTranscripts:
    def __init__(self):
        self.parsed: list[tuple[str, str]] = []

    def iter_snapshots(self, sources, project, *, parse=False, limit=None, host_id=None):
        assert parse is False
        result = []
        for item in sources:
            result.append(
                {
                    "metadata_only": True,
                    "path": Path(item["path"]),
                    "source": item["source"],
                    "session_id": Path(item["path"]).stem,
                    "snapshot_id": f"meta-{Path(item['path']).stem}",
                    "mtime": Path(item["path"]).stat().st_mtime,
                }
            )
        return result[:limit] if limit is not None and limit >= 0 else result

    def parse_transcript(self, path, source, project, *, host_id=None, session_id=None):
        self.parsed.append((source, str(session_id)))
        return {
            "source": source,
            "session_id": session_id,
            "context": Path(path).read_text(encoding="utf-8"),
        }


class Outbox:
    def __init__(self):
        self.items = []

    def enqueue(self, envelope):
        self.items.append(envelope)
        return envelope["snapshot_id"]

    def pending(self, limit=None):
        return []


def _runner(tmp_path: Path, *, learning=None, store=None):
    root = tmp_path / "repo"
    root.mkdir()
    vault = tmp_path / "vault" / "project"
    (vault / "daily").mkdir(parents=True)
    project = Project(root, vault, "project-id", "project")
    transcripts = MetadataTranscripts()
    outbox = Outbox()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=Registry(project),
        transcripts=transcripts,
        outbox=outbox,
        store=store,
        learning=learning,
    )
    return runner, project, transcripts, outbox


def test_collection_limit_is_applied_before_body_parse_and_provider_is_kept(tmp_path):
    runner, project, transcripts, outbox = _runner(tmp_path)
    files = []
    for index in range(3):
        path = tmp_path / f"session-{index}.jsonl"
        path.write_text(f"session {index}", encoding="utf-8")
        files.append({"path": path, "source": "claude" if index == 0 else "codex"})

    result = runner.collect(files, cwd=project.root, all_history=True, limit=1)

    assert result["candidates"] == 3
    assert result["queued"] == 1
    assert result["unexamined"] == 2
    assert len(transcripts.parsed) == 1
    assert outbox.items[0]["source"] == "codex" or outbox.items[0]["source"] == "claude"
    assert outbox.items[0]["source"] == transcripts.parsed[0][0]
    collection_state = runner.state.read("collection")
    coverage = collection_state["projects"][project.project_id]
    assert coverage["last_run_at"]
    assert coverage["coverage"]["candidates"] == 3
    assert coverage["coverage"]["ingested"] == 1
    assert coverage["coverage"]["unexamined"] == 2


def test_project_ids_pass_vault_root_and_skip_legacy_records(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)

    class MixedRegistry(Registry):
        def list_initialized_projects(self, *, vault_root=None):
            self.vault_root_seen = vault_root
            return [project, {"id": None, "name": "legacy", "root": None}]

    registry = MixedRegistry(project)
    runner.projects = registry

    assert runner._project_ids() == {project.project_id}
    assert registry.vault_root_seen == (tmp_path / "vault").resolve()


def test_daily_cursor_is_project_scoped_and_analysis_failure_is_retryable(tmp_path):
    class Learning:
        def __init__(self):
            self.analysis_calls = 0

        def compile_daily(self, project, day):
            return True

        def analyze_daily(self, project, day):
            self.analysis_calls += 1
            return self.analysis_calls > 1

    learning = Learning()
    runner, project, _transcripts, _outbox = _runner(tmp_path, learning=learning)
    first = runner.daily(project=project, day="2026-09-06")
    assert first["failed"] == 1
    assert runner.state.read("compile")["projects"][project.project_id]["last_successful_day"] == "2026-09-06"

    second = runner.daily(project=project, day="2026-09-06")
    assert second["analyzed"] == 1
    assert runner.state.read("compile")["projects"][project.project_id]["last_successful_day"] == "2026-09-06"

    other = Project(tmp_path / "other-repo", tmp_path / "vault" / "other", "other-id", "other")
    other.root.mkdir()
    (other.vault_dir / "daily").mkdir(parents=True)
    third = runner.daily(project=other, day="2026-09-06")
    assert third["days"] == 1
    assert runner.state.read("compile")["projects"][other.project_id]["last_successful_day"] == "2026-09-06"


def _write_legacy_compile_proof(project, day: str, content: str, *, reference: bool = True):
    daily = project.vault_dir / "daily" / f"{day}.md"
    daily.write_text(content, encoding="utf-8")
    state_path = project.vault_dir / ".state" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({
        "ingested": {
            daily.name: {
                "hash": pipeline.hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
                "compiled_at": "2026-09-07T10:00:00+02:00",
            }
        }
    }), encoding="utf-8")
    if reference:
        article = project.vault_dir / "knowledge" / "concepts" / "legacy.md"
        article.parent.mkdir(parents=True)
        article.write_text(f"---\nsources: [daily/{daily.name}]\n---\n# Legacy\n", encoding="utf-8")
    return daily


def test_legacy_compile_proof_skips_unchanged_daily_without_llm(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    daily = _write_legacy_compile_proof(project, "2026-09-04", "unchanged legacy daily")

    assert runner._due_days({}, date(2026, 9, 7), project) == []
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("legacy proof must not invoke compile.py"),
    )
    assert runner._default_compile_daily(project, daily.stem) is True


def test_changed_daily_with_legacy_ledger_stays_due_and_recompiles(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    daily = _write_legacy_compile_proof(project, "2026-09-04", "original legacy daily")
    daily.write_text("changed after legacy compile", encoding="utf-8")
    calls = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: calls.append(args) or Completed())
    monkeypatch.setattr(runner, "_publish_knowledge", lambda project: None)

    assert runner._due_days({}, date(2026, 9, 7), project) == ["2026-09-04"]
    assert runner._default_compile_daily(project, daily.stem) is True
    assert calls


def test_legacy_hash_without_article_reference_remains_retryable(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    daily = _write_legacy_compile_proof(project, "2026-09-04", "legacy without article", reference=False)

    assert runner._due_days({}, date(2026, 9, 7), project) == ["2026-09-04"]


def test_historical_compile_due_days_are_bounded_and_keep_target(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    for index in range(6):
        day = date(2026, 9, 1 + index).isoformat()
        (project.vault_dir / "daily" / f"{day}.md").write_text(f"daily {day}", encoding="utf-8")

    due = runner._due_days({}, date(2026, 9, 6), project)

    assert len(due) == pipeline.DEFAULT_LIMIT
    assert "2026-09-06" in due


def test_daily_audits_target_without_new_daily_file_after_legacy_skip(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path, store=None, learning=None)
    _write_legacy_compile_proof(project, "2026-09-04", "already compiled legacy daily")

    result = runner.daily(project=project, day="2026-09-06")

    assert result["days"] == 1
    assert result["compiled"] == 0
    assert result["analyzed"] == 1
    assert runner.state.read("compile")["projects"][project.project_id]["days"]["2026-09-06"]["status"] == "analysis_only"
    audit = tmp_path / "private" / "audits" / project.project_id / "2026-09-06.json"
    assert json.loads(audit.read_text(encoding="utf-8"))["coverage"]["status"] == "no_evidence"


def test_historical_compile_does_not_replace_valid_target_no_evidence_report(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path, store=None, learning=None)
    first = runner.daily(project=project, day="2026-09-06")
    assert first["analyzed"] == 1
    target_audit = tmp_path / "private" / "audits" / project.project_id / "2026-09-06.json"
    before = target_audit.read_text(encoding="utf-8")

    historical = project.vault_dir / "daily" / "2026-09-04.md"
    historical.write_text("late historical daily", encoding="utf-8")
    calls = []

    class Learning:
        def compile_daily(self, project, day):
            calls.append(("compile", day))
            return True

        def analyze_daily(self, project, day):
            calls.append(("analyze", day))
            return True

    runner.learning = Learning()
    result = runner.daily(project=project, day="2026-09-06")

    # Explicit --day is exact: a late historical file is left for an
    # intentional historical compilation request.
    assert result["compiled"] == 0
    assert result["analyzed"] == 0
    assert calls == []
    assert target_audit.read_text(encoding="utf-8") == before


def test_pending_target_analysis_does_not_block_due_compile_backlog(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    historical = project.vault_dir / "daily" / "2026-09-04.md"
    historical.write_text("late historical daily", encoding="utf-8")
    calls = []

    class Store:
        def __init__(self):
            self.rows = [
                {
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": "pending-a",
                    "revision": "a" * 64,
                    "started_at": "2026-09-06T10:00:00Z",
                    "updated_at": "2026-09-06T10:00:00Z",
                    "messages": [{"role": "user", "content": "a"}],
                },
                {
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": "pending-b",
                    "revision": "b" * 64,
                    "started_at": "2026-09-06T11:00:00Z",
                    "updated_at": "2026-09-06T11:00:00Z",
                    "messages": [{"role": "user", "content": "b"}],
                },
            ]
            self.marked = set()

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            return [
                row
                for row in self.rows
                if (row["project_id"], row["source"], row["session_id"], row["revision"])
                not in self.marked
            ][:limit]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error):
            self.marked.add((project_id, source, session_id, revision))

    class Learning:
        def compile_daily(self, _project, day):
            calls.append(("compile", day))
            return True

        def audit_snapshots_sync(self, snapshots, **kwargs):
            calls.append(("audit", [row["session_id"] for row in snapshots]))
            return {
                "reports": [
                    {"session_id": row["session_id"], "analysis_status": "ok"}
                    for row in snapshots
                ],
                "coverage": {"sessions": len(snapshots)},
                "errors": [],
            }

    store = Store()
    runner.store = store
    runner.learning = Learning()
    result = runner.daily(project=project, day="2026-09-06")

    assert result["compiled"] == 0
    assert result["analyzed"] == 0
    assert result["failed"] == 0
    assert result["pending"] == 1
    assert calls == [("audit", ["pending-a"])]
    assert runner.state.read("analysis")["projects"][project.project_id]["days"]["2026-09-06"]["status"] == "pending"


def test_compile_failure_does_not_block_target_analysis(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    target_daily = project.vault_dir / "daily" / "2026-09-06.md"
    target_daily.write_text("target daily", encoding="utf-8")
    calls = []

    class Store:
        def __init__(self):
            self.marked = False

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            if self.marked:
                return []
            return [{
                "project_id": project.project_id,
                "source": "codex",
                "session_id": "target-session",
                "revision": "c" * 64,
                "started_at": "2026-09-06T12:00:00Z",
                "updated_at": "2026-09-06T12:00:00Z",
                "messages": [{"role": "user", "content": "target evidence"}],
            }][:limit]

        def mark_processed(self, *args, **kwargs):
            self.marked = True

    class Learning:
        def compile_daily(self, _project, day):
            calls.append(("compile", day))
            return False

        def audit_snapshots_sync(self, snapshots, **kwargs):
            calls.append(("audit", [row["session_id"] for row in snapshots]))
            return {
                "reports": [{"session_id": "target-session", "analysis_status": "ok"}],
                "coverage": {"sessions": 1},
                "errors": [],
            }

    store = Store()
    runner.store = store
    runner.learning = Learning()
    result = runner.daily(project=project, day="2026-09-06")

    assert result["compiled"] == 0
    assert result["analyzed"] == 1
    assert result["failed"] == 1
    assert result["pending"] == 1
    assert calls == [("compile", "2026-09-06"), ("audit", ["target-session"])]
    assert (tmp_path / "private" / "audits" / project.project_id / "2026-09-06.json").exists()


def test_analysis_ack_failure_leaves_target_day_retryable(tmp_path):
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def __init__(self):
            self.attempts = 0

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            return [{
                "project_id": project.project_id,
                "source": "codex",
                "session_id": "retry-session",
                "revision": "e" * 64,
                "started_at": "2026-09-06T12:00:00Z",
                "updated_at": "2026-09-06T12:00:00Z",
                "messages": [{"role": "user", "content": "evidence"}],
            }]

        def mark_processed(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("ack unavailable")

    class Learning:
        def audit_snapshots_sync(self, snapshots, **kwargs):
            return {
                "reports": [{"session_id": "retry-session", "analysis_status": "ok"}],
                "coverage": {"sessions": 1},
                "errors": [],
            }

    store = Store()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=store,
        learning=Learning(),
    )

    first = runner.daily(project=project, day="2026-09-06")
    assert first["failed"] == 1
    assert runner.state.read("analysis")["projects"][project.project_id]["days"]["2026-09-06"]["status"] == "pending"
    second = runner.daily(project=project, day="2026-09-06")
    assert second["analyzed"] == 1
    assert store.attempts == 2


def test_analysis_error_acknowledges_only_individually_valid_snapshots(tmp_path, monkeypatch):
    # This test exercises per-session partial ACK semantics independently of
    # the native one-conversation analysis budget.
    monkeypatch.setattr(pipeline, "ANALYSIS_BATCH_LIMIT", 2)
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def __init__(self):
            self.marked = []

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            rows = [
                {
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": "good-session",
                    "revision": "f" * 64,
                    "started_at": "2026-09-06T12:00:00Z",
                    "updated_at": "2026-09-06T12:00:00Z",
                    "messages": [{"role": "user", "content": "good evidence"}],
                },
                {
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": "bad-session",
                    "revision": "1" * 64,
                    "started_at": "2026-09-06T13:00:00Z",
                    "updated_at": "2026-09-06T13:00:00Z",
                    "messages": [{"role": "user", "content": "bad evidence"}],
                },
            ]
            marked_sessions = {item[0] for item in self.marked}
            return [row for row in rows if row["session_id"] not in marked_sessions]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error):
            self.marked.append((session_id, status, error))

    class Learning:
        def audit_snapshots_sync(self, snapshots, **kwargs):
            return {
                "reports": [
                    {"session_id": "good-session", "analysis_status": "ok"},
                    {"session_id": "bad-session", "analysis_status": "model-error"},
                ],
                "coverage": {"sessions": 2},
                "errors": [{"session_id": "bad-session", "kind": "audit_runner_error"}],
            }

    store = Store()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=store,
        learning=Learning(),
    )

    with pytest.raises(pipeline.PipelineError):
        runner._default_analyze_daily(project, "2026-09-06")
    assert store.marked == [
        ("good-session", "succeeded", None),
        ("bad-session", "failed", "analysis_model_error"),
    ]
    attempt = tmp_path / "private" / "audits" / project.project_id / "2026-09-06.attempt.json"
    assert attempt.exists()


def test_default_analysis_leaves_day_pending_when_more_than_one_batch_remains(tmp_path):
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def __init__(self):
            self.rows = [
                {
                    "project_id": project.project_id,
                    "source": "codex",
                    "session_id": f"session-{index}",
                    "revision": f"{index:064x}",
                    "started_at": f"2026-09-06T{10 + index:02d}:00:00Z",
                    "updated_at": f"2026-09-06T{10 + index:02d}:00:00Z",
                    "messages": [{"role": "user", "content": f"evidence {index}"}],
                }
                for index in range(pipeline.ANALYSIS_BATCH_LIMIT + 2)
            ]
            self.marked: set[tuple[str, str, str, str]] = set()

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            assert project_id == project.project_id
            assert stage == "analysis"
            return [
                row
                for row in self.rows
                if (row["project_id"], row["source"], row["session_id"], row["revision"])
                not in self.marked
            ][:limit]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error):
            assert stage == "analysis"
            assert status == "succeeded"
            self.marked.add((project_id, source, session_id, revision))

    class Learning:
        def __init__(self):
            self.batch_sizes = []

        def audit_snapshots_sync(self, snapshots, **kwargs):
            self.batch_sizes.append(len(snapshots))
            return {
                "reports": [
                    {
                        "session_id": row["session_id"],
                        "analysis_status": "ok",
                        "conversations": [
                            {
                                "conversation_id": f"codex:{row['session_id']}",
                                "status": "ok",
                            }
                        ],
                    }
                    for row in snapshots
                ],
                "coverage": {"sessions": len(snapshots)},
                "errors": [],
            }

    store = Store()
    learning = Learning()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=store,
        learning=learning,
    )

    result = runner.daily(project=project, day="2026-09-06")

    assert result["failed"] == 0
    assert result["pending"] == 1
    assert learning.batch_sizes == [pipeline.ANALYSIS_BATCH_LIMIT]
    assert len(store.marked) == pipeline.ANALYSIS_BATCH_LIMIT
    assert runner.state.read("analysis")["projects"][project.project_id]["days"]["2026-09-06"]["status"] == "pending"


def test_daily_reopens_completed_analysis_for_new_revision_same_day(tmp_path):
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def __init__(self):
            self.rows = [self._row("revision-a")]
            self.marked = set()

        @staticmethod
        def _row(revision):
            return {
                "project_id": project.project_id,
                "source": "codex",
                "session_id": "same-session",
                "revision": revision,
                "started_at": "2026-09-06T10:00:00Z",
                "updated_at": "2026-09-06T10:00:00Z",
                "messages": [{"role": "user", "timestamp": "2026-09-06T10:00:00Z", "content": revision}],
            }

        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            assert project_id == project.project_id
            assert stage == "analysis"
            return [
                row
                for row in self.rows
                if (row["project_id"], row["source"], row["session_id"], row["revision"])
                not in self.marked
            ][:limit]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error):
            self.marked.add((project_id, source, session_id, revision))

    class Learning:
        def __init__(self):
            self.calls = []

        def audit_snapshots_sync(self, snapshots, **kwargs):
            self.calls.append([row["revision"] for row in snapshots])
            return {
                "reports": [
                    {
                        "session_id": row["session_id"],
                        "analysis_status": "ok",
                        "conversations": [
                            {
                                "conversation_id": f"codex:{row['session_id']}",
                                "status": "ok",
                            }
                        ],
                    }
                    for row in snapshots
                ],
                "coverage": {"sessions": len(snapshots)},
                "errors": [],
            }

    store = Store()
    learning = Learning()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=store,
        learning=learning,
    )

    first = runner.daily(project=project, day="2026-09-06")
    assert first["analyzed"] == 1
    assert first["failed"] == 0
    assert first["compiled"] == 0

    store.rows.append(store._row("revision-b"))
    second = runner.daily(project=project, day="2026-09-06")

    assert second["analyzed"] == 1
    assert second["failed"] == 0
    assert second["compiled"] == 0
    assert learning.calls == [["revision-a"], ["revision-b"]]
    assert {key[-1] for key in store.marked} == {"revision-a", "revision-b"}
    audit = json.loads(
        (tmp_path / "private" / "audits" / project.project_id / "2026-09-06.json").read_text(encoding="utf-8")
    )
    assert len(audit["reports"]) == 2
    assert {item["revision"] for item in audit["snapshot_identities"]} == {
        "revision-a",
        "revision-b",
    }
    assert len(audit["conversations"]) == 2


def test_daily_write_uses_stable_session_id_and_started_at(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    item = {
        "source": "test",
        "snapshot": {
            "session_id": "stable-session",
            "revision": "a" * 64,
            "started_at": "2026-09-06T23:30:00Z",
            "messages": [],
        },
    }
    first = runner._daily_write(project, item, "first extraction")
    item["snapshot"]["revision"] = "b" * 64
    second = runner._daily_write(project, item, "updated extraction")

    assert first == second
    content = first.read_text(encoding="utf-8")
    assert content.count("ace-snapshot: stable-session") == 1
    assert "updated extraction" in content
    assert "first extraction" not in content
    assert first.name == "2026-09-07.md"


def test_default_analysis_marks_invalid_snapshots_failed(tmp_path):
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def __init__(self):
            self.marked = []

        def pending_snapshots(self, *, project_id=None, limit=100, stage="extraction"):
            return [{
                "project_id": project.project_id,
                "source": "codex",
                "session_id": "session",
                "revision": "c" * 64,
                "started_at": "2026-09-06T10:00:00Z",
                "updated_at": "2026-09-06T10:00:00Z",
                "messages": [{"role": "user", "content": "evidence"}],
            }]

        def mark_processed(self, source, session_id, revision, project_id, stage, status, error):
            self.marked.append((session_id, status, error))

    class Learning:
        def audit_snapshots_sync(self, *args, **kwargs):
            return {"errors": [{"kind": "audit_runner_error"}], "coverage": {"sessions": 1}, "reports": [{}]}

    store = Store()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=store,
        learning=Learning(),
    )
    with pytest.raises(pipeline.PipelineError):
        runner._default_analyze_daily(project, "2026-09-06")
    assert store.marked == [("session", "failed", "analysis_model_error")]
    attempt = tmp_path / "private" / "audits" / project.project_id / "2026-09-06.attempt.json"
    assert attempt.exists()
    assert json.loads(attempt.read_text(encoding="utf-8"))["analysis_status"] == "degraded"


def test_schedule_json_is_forwarded_to_native_delegate(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setitem(sys.modules, "ace_schedule", types.SimpleNamespace(main=lambda argv: calls.append(argv) or 0))
    runner = pipeline.ACEPipeline(private_root=tmp_path / "private", projects=None)

    assert pipeline.main(["schedule", "--json"], pipeline=runner) == 0
    assert calls == [["--json"]]
    assert '"returncode": 0' in capsys.readouterr().out


def test_query_delegate_returns_captured_stdout(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    observed = {}

    class Completed:
        returncode = 0
        stdout = "Question: decision\n\nanswer with [daily/2026-09-06.md]"
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["cwd"] = kwargs["cwd"]
        return Completed()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = runner._delegate("query", project.root, ["decision"])

    assert result["returncode"] == 0
    assert result["stdout"] == Completed.stdout
    assert result["stderr"] == ""
    assert observed["command"][1].endswith("/scripts/query.py")


def test_scan_delegate_passes_verified_source_cwd_and_shared_tick_lock(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    observed = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    result = runner._delegate("scan_md", project.root, ["--dry-run"])

    assert result["ok"] is True
    command = observed["command"]
    assert command[1].endswith("/scripts/scan_md.py")
    assert command[2:4] == ["--cwd", str(project.root.resolve())]
    assert command[4:] == ["--dry-run"]
    assert observed["kwargs"]["cwd"] == str(pipeline.CONFIG_ROOT)
    assert observed["kwargs"]["env"]["ACE_PROJECT_DIR"] == str(project.vault_dir)


def test_delegate_cannot_bypass_tick_lock_when_already_held(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path)

    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("delegate must be rejected before subprocess"),
    )
    with pipeline.advisory_lock(runner.private_root / "tick.lock"):
        with pytest.raises(pipeline.PipelineBusyError):
            runner._delegate("compile", project.root, ["--dry-run"])


def test_hermes_without_session_metadata_is_left_unrouted(tmp_path):
    runner, project, _transcripts, _outbox = _runner(tmp_path)
    database = tmp_path / "hermes.sqlite"
    database.write_bytes(b"not parsed by the pipeline fallback")

    class NoMetadata:
        def iter_snapshots(self, *args, **kwargs):
            return []

    runner.transcripts = NoMetadata()
    groups, unrouted, candidates = runner._auto_source_groups(
        [database],
        [project],
        source="hermes",
        all_history=True,
        days=7,
    )

    assert groups == {}
    assert unrouted == 1
    assert candidates == 1


def test_compile_publication_failure_keeps_local_daily_retryable(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path, learning=None, store=None)
    daily = project.vault_dir / "daily" / "2026-09-06.md"
    daily.write_text("# Daily Log\n\nlocal source of truth\n", encoding="utf-8")
    calls = {"publish": 0, "compile": 0}

    class Completed:
        returncode = 0

    def fake_compile(*args, **kwargs):
        calls["compile"] += 1
        return Completed()

    def publish(_project, _store):
        calls["publish"] += 1
        if calls["publish"] == 1:
            raise RuntimeError("publication unavailable")
        return {"published": True}

    monkeypatch.setattr(pipeline.subprocess, "run", fake_compile)
    monkeypatch.setitem(sys.modules, "ace_knowledge", types.SimpleNamespace(publish_project=publish))

    with pytest.raises(pipeline.PipelineError):
        runner._default_compile_daily(project, "2026-09-06")
    assert daily.read_text(encoding="utf-8") == "# Daily Log\n\nlocal source of truth\n"

    assert runner._default_compile_daily(project, "2026-09-06") is True
    assert calls == {"publish": 2, "compile": 2}


def test_compile_failure_persists_safe_project_day_diagnostic(tmp_path, monkeypatch):
    runner, project, _transcripts, _outbox = _runner(tmp_path, learning=None, store=None)
    daily = project.vault_dir / "daily" / "2026-09-06.md"
    daily.write_text("# Daily Log\n\nprivate user prompt\n", encoding="utf-8")

    class Completed:
        returncode = 1
        stdout = "compiler saw private user prompt\n"
        stderr = "NameError: name 'save_state' is not defined\nraw transcript body"

    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: Completed())

    result = runner.daily(project=project, day="2026-09-06")
    assert result["failed"] == 1

    state = runner.state.read("compile")
    diagnostic = state["projects"][project.project_id]["diagnostics"]["2026-09-06"]
    assert diagnostic["returncode"] == 1
    assert "NameError" in diagnostic["diagnostic"]
    assert "save_state" in diagnostic["diagnostic"]
    assert "private user prompt" not in json.dumps(diagnostic)
    assert "raw transcript body" not in json.dumps(diagnostic)


def test_default_analysis_empty_snapshots_is_deterministic_and_does_not_call_learning(tmp_path):
    project = Project(tmp_path / "repo", tmp_path / "vault" / "project", "project-id", "project")
    project.root.mkdir()
    (project.vault_dir / "daily").mkdir(parents=True)

    class Store:
        def pending_snapshots(self, *, project_id=None, limit=100, stage="analysis"):
            assert project_id == project.project_id
            assert stage == "analysis"
            return []

    class Learning:
        def __init__(self):
            self.audit_calls = 0
            self.render_calls = 0

        def audit_snapshots_sync(self, *args, **kwargs):
            self.audit_calls += 1
            raise AssertionError("empty analysis must not launch a model audit")

        def render_reports(self, *args, **kwargs):
            self.render_calls += 1

    learning = Learning()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=None,
        transcripts=None,
        outbox=None,
        store=Store(),
        learning=learning,
    )

    assert runner._default_analyze_daily(project, "2026-09-06") is True
    audit = json.loads(
        (tmp_path / "private" / "audits" / project.project_id / "2026-09-06.json").read_text(encoding="utf-8")
    )
    assert audit["coverage"] == {"sessions": 0, "status": "no_evidence", "evidence": "none"}
    assert audit["reports"] == []
    assert audit["limitations"] == ["no acknowledged snapshots available"]
    assert learning.audit_calls == 0
    assert learning.render_calls == 0


def test_main_propagates_delegate_and_nested_stage_failures(tmp_path, monkeypatch):
    class FailedStages:
        def process(self, **kwargs):
            return {"failed": 1, "pending": 1}

        def tick(self, **kwargs):
            return {"process": {"failed": 1}, "daily": {"failed": 0}}

    runner = FailedStages()
    assert pipeline.main(["process"], pipeline=runner) == 1
    assert pipeline.main(["tick"], pipeline=runner) == 1

    monkeypatch.setitem(sys.modules, "ace_schedule", types.SimpleNamespace(main=lambda argv: 7))
    delegated = pipeline.ACEPipeline(private_root=tmp_path / "private", projects=None)
    assert pipeline.main(["schedule"], pipeline=delegated) == 7


def test_process_daily_share_tick_lock_without_nested_tick_lock(tmp_path, monkeypatch):
    class Learning:
        def compile_daily(self, project, day):
            return True

        def analyze_daily(self, project, day):
            return True

    runner, project, _transcripts, _outbox = _runner(tmp_path, learning=Learning(), store=None)
    runner.private_root.mkdir(mode=0o700)
    acquired = []

    @contextmanager
    def fake_lock(path):
        acquired.append(Path(path))
        yield

    monkeypatch.setattr(pipeline, "advisory_lock", fake_lock)

    runner.process(project=project)
    runner.daily(project=project, day="2026-09-06")
    assert acquired == [runner.private_root / "tick.lock", runner.private_root / "tick.lock"]

    acquired.clear()
    result = runner.tick(
        cwd=project.root,
        now=pipeline.datetime(2026, 9, 7, 8, tzinfo=pipeline.timezone.utc),
        limit=1,
    )
    assert result["daily"]["failed"] == 0
    assert acquired == [runner.private_root / "tick.lock"]


def test_extraction_ack_failure_keeps_local_result_pending_for_retry(tmp_path):
    class Store:
        def __init__(self):
            self.attempts = 0
            self.acknowledged = False

        def pending_snapshots(self, *, project_id=None, limit=100, stage="extraction"):
            if self.acknowledged or project_id != "project-id" or stage != "extraction":
                return []
            return [{
                "project_id": project_id,
                "source": "test",
                "session_id": "ack-session",
                "revision": "d" * 64,
                "started_at": "2026-09-06T23:30:00Z",
                "messages": [{"id": "message-1", "role": "user", "type": "message", "content": "evidence"}],
            }][:limit]

        def mark_processed(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("database unavailable")
            self.acknowledged = True

    store = Store()
    runner, project, _transcripts, _outbox = _runner(
        tmp_path,
        store=store,
    )
    runner.extractor = lambda _context: "idempotent extraction"

    first = runner.process(project=project, limit=1)
    assert first["processed"] == 0
    assert first["failed"] == 1
    assert first["pending"] == 1
    daily_files = list((project.vault_dir / "daily").glob("*.md"))
    assert len(daily_files) == 1
    state = runner.state.read("extraction")
    assert next(iter(state["snapshots"].values()))["status"] == "pending"

    second = runner.process(project=project, limit=1)
    assert second["processed"] == 1
    assert store.acknowledged is True


def test_snapshot_context_preserves_source_metadata(tmp_path):
    runner, _project, _transcripts, _outbox = _runner(tmp_path)
    context = runner._snapshot_context({
        "messages": [
            {"role": "system", "type": "message", "content": "injected system context"},
            {"role": "developer", "type": "message", "content": "developer policy"},
            {
                "id": "user-message",
                "role": "user",
                "type": "message",
                "content": "on va reprendre le projet",
            },
            {
                "id": "assistant-message",
                "role": "assistant",
                "type": "message",
                "content": "je vérifie le dossier cité",
            },
            {
                "id": "task-event",
                "role": "assistant",
                "type": "task_started",
                "content": "session devait produire daily log",
            },
            {
                "id": "usage-event",
                "role": "assistant",
                "type": "tokenusage",
                "content": "usage telemetry",
            },
            {
                "id": "tool-message",
                "role": "tool",
                "type": "tool_result",
                "timestamp": "2026-09-07T10:11:12Z",
                "call_id": "call-42",
                "refs": {"source_ref": "session#L18", "line": "18"},
                "content": "read_thread error",
            },
        ],
    })

    assert "injected system context" not in context
    assert "developer policy" not in context
    assert "on va reprendre le projet" in context
    assert "je vérifie le dossier cité" in context
    assert "session devait produire daily log" not in context
    assert "usage telemetry" not in context
    assert "type=tool_result" in context
    assert "timestamp=2026-09-07T10:11:12Z" in context
    assert "call_id=call-42" in context
    assert '"source_ref": "session#L18"' in context
    assert "read_thread error" in context


def test_snapshot_context_strips_only_complete_runtime_preamble(tmp_path):
    runner, _project, _transcripts, _outbox = _runner(tmp_path)
    injected = (
        "# AGENTS.md instructions for /Users/franck/.agents\n"
        "<INSTRUCTIONS>runtime instructions, not a user request</INSTRUCTIONS>\n"
        "<environment_context>cwd=/tmp\ncurrent_date=2026-09-07</environment_context>\n"
        "On va reprendre le projet et auditer le document cité."
    )
    context = runner._snapshot_context({"messages": [{"role": "user", "content": injected}]})

    assert "runtime instructions" not in context
    assert "cwd=/tmp" not in context
    assert "On va reprendre le projet et auditer le document cité." in context


def test_snapshot_context_keeps_user_citation_without_exact_runtime_wrapper(tmp_path):
    runner, _project, _transcripts, _outbox = _runner(tmp_path)
    cited = "# AGENTS.md instructions\nPlease audit this document and tell me what to change."

    context = runner._snapshot_context({"messages": [{"role": "user", "content": cited}]})

    assert cited in context


def test_snapshot_context_strips_runtime_prefix_inside_visible_content_blocks(tmp_path):
    runner, _project, _transcripts, _outbox = _runner(tmp_path)
    context = runner._snapshot_context({
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": (
                    "# AGENTS.md instructions\n"
                    "<INSTRUCTIONS>runtime only</INSTRUCTIONS>\n"
                    "<environment_context>cwd=/tmp</environment_context>\n"
                    "demande humaine conservée"
                ),
            }],
        }],
    })

    assert "runtime only" not in context
    assert "cwd=/tmp" not in context
    assert "demande humaine conservée" in context


def test_daily_report_selects_explicit_project_collection_coverage(tmp_path):
    collection = tmp_path / "collection.json"
    collection.write_text(json.dumps({
        "projects": {
            "project-one": {
                "last_run_at": "2026-09-07T10:00:00+02:00",
                "coverage": {"candidates": 3, "ingested": 3, "unexamined": 0, "failed": 0},
            },
            "project-two": {
                "last_run_at": "2026-09-07T10:00:00+02:00",
                "coverage": {"candidates": 99, "ingested": 99, "unexamined": 0, "failed": 0},
            },
        }
    }), encoding="utf-8")

    report = ace_daily_report.build_report(
        report_date=date(2026, 9, 7),
        collection_state_path=collection,
        incident_state_path=tmp_path / "incidents.json",
        audit_dir=tmp_path / "audit",
        project_id="project-one",
    )

    assert "| candidates | 3 |" in report
    assert "| candidates | 99 |" not in report
    assert "/projects/project-one/coverage" in report


def test_status_reads_existing_outbox_sqlite_without_constructing_outbox(tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "outbox.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE ace_outbox (status TEXT, payload_bytes INTEGER)")
        connection.executemany(
            "INSERT INTO ace_outbox(status, payload_bytes) VALUES (?, ?)",
            [("pending", 12), ("acknowledged", 20)],
        )
        connection.commit()
    finally:
        connection.close()

    runner = pipeline.ACEPipeline(private_root=private, projects=None, outbox=None)
    result = runner.status()

    assert result["outbox"]["total"] == 2
    assert result["outbox"]["pending"] == 1
    assert result["outbox"]["statuses"]["pending"]["count"] == 1


def _scoped_envelope(project: Project, session_id: str, revision: str) -> dict:
    return {
        "schema_version": 1,
        "project": {
            "id": project.project_id,
            "name": project.name,
            "root": str(project.root),
            "vault_dir": str(project.vault_dir),
        },
        "source": "codex",
        "session_id": session_id,
        "revision": revision,
        "source_path": str(project.root / f"{session_id}.jsonl"),
        "messages": [{"role": "user", "type": "message", "content": "evidence"}],
        "attachments": [],
    }


def test_outbox_project_scope_is_applied_before_batch_claim(tmp_path):
    queue = SQLiteOutbox(tmp_path / "outbox.sqlite3")
    project_a = Project(tmp_path / "repo-a", tmp_path / "vault" / "a", "project-a", "a")
    project_b = Project(tmp_path / "repo-b", tmp_path / "vault" / "b", "project-b", "b")
    claimed_a = queue.enqueue(_scoped_envelope(project_a, "session-a", "a" * 64))
    claimed_b = queue.enqueue(_scoped_envelope(project_b, "session-b", "b" * 64))

    first = queue.pending(limit=1, project_id=project_a.project_id)
    assert [item.key for item in first] == [str(claimed_a)]
    assert queue.connection.execute(
        "SELECT status FROM ace_outbox WHERE key=?", (str(claimed_b),)
    ).fetchone()[0] == "pending"

    second = queue.pending(limit=1, project_id=project_b.project_id)
    assert [item.key for item in second] == [str(claimed_b)]
    queue.close()


def test_outbox_incremental_cutoff_filters_before_claim(tmp_path):
    queue = SQLiteOutbox(tmp_path / "outbox.sqlite3")
    project = Project(tmp_path / "repo", tmp_path / "vault", "project", "project")
    old_key = queue.enqueue(_scoped_envelope(project, "old", "c" * 64))
    queue.connection.execute(
        "UPDATE ace_outbox SET created_at=100 WHERE key=?", (str(old_key),)
    )
    new_key = queue.enqueue(_scoped_envelope(project, "new", "d" * 64))

    claimed = queue.pending(limit=10, created_after=200)

    assert [item.key for item in claimed] == [str(new_key)]
    assert queue.connection.execute(
        "SELECT status FROM ace_outbox WHERE key=?", (str(old_key),)
    ).fetchone()[0] == "pending"
    queue.close()


def test_tick_two_projects_never_claims_foreign_outbox_rows(tmp_path):
    project_a = Project(tmp_path / "repo-a", tmp_path / "vault" / "a", "project-a", "a")
    project_b = Project(tmp_path / "repo-b", tmp_path / "vault" / "b", "project-b", "b")
    for project in (project_a, project_b):
        project.root.mkdir()
        (project.vault_dir / "daily").mkdir(parents=True)

    class Projects:
        def list_initialized_projects(self, *, vault_root=None):
            return [project_a, project_b]

        def resolve_project(self, root, *, vault_root=None, strict=False):
            return next(
                project for project in (project_a, project_b)
                if Path(root).resolve() == project.root.resolve()
            )

    class Store:
        def __init__(self):
            self.received = []

        def upsert_snapshot(self, envelope):
            self.received.append(envelope["project"]["id"])
            return {"receipt": f"receipt-{len(self.received)}"}

        def list_acquitted_snapshots(self, *, project_id, limit, stage="extraction"):
            return []

    queue = SQLiteOutbox(tmp_path / "outbox.sqlite3")
    queue.enqueue(_scoped_envelope(project_a, "session-a", "a" * 64))
    queue.enqueue(_scoped_envelope(project_b, "session-b", "b" * 64))
    store = Store()
    runner = pipeline.ACEPipeline(
        vault_root=tmp_path / "vault",
        private_root=tmp_path / "private",
        projects=Projects(),
        transcripts=None,
        outbox=queue,
        store=store,
    )
    runner.collect = lambda *args, **kwargs: {
        "projects": 0,
        "candidates": 0,
        "queued": 0,
        "unchanged": 0,
        "failed": 0,
        "deferred": 0,
        "unexamined": 0,
        "unrouted": 0,
        "offline": False,
    }

    result = runner.tick(
        now=pipeline.datetime(2026, 9, 7, 4, tzinfo=pipeline.timezone.utc),
        limit=1,
    )

    assert result["sync"]["synced"] == 2
    assert result["sync"]["failed"] == 0
    assert sorted(store.received) == ["project-a", "project-b"]
    statuses = [row[0] for row in queue.connection.execute("SELECT status FROM ace_outbox ORDER BY project_id")]
    assert statuses == ["acknowledged", "acknowledged"]
    queue.close()
