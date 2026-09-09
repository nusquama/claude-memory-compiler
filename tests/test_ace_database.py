from __future__ import annotations

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ace_database import (  # noqa: E402
    EnvelopeValidationError,
    SupabaseStore,
    SupabaseStoreError,
    normalize_envelope,
    normalise_stored_envelope,
)
from ace_transcripts import _AttachmentCollector, _clean_content  # noqa: E402


PROJECT_ID = "123e4567-e89b-12d3-a456-426614174000"
REVISION = "a" * 64


def envelope() -> dict:
    return {
        "schema_version": 1,
        "project": {
            "id": PROJECT_ID,
            "name": "Conversations",
            "root": "/vault/Conversations",
            "vault_dir": "/vault",
        },
        "source": "Codex",
        "session_id": "native-session-1",
        "revision": REVISION,
        "source_path": "/tmp/transcript.jsonl",
        "host_id": "local",
        "started_at": "2026-09-07T10:00:00+00:00",
        "updated_at": "2026-09-07T10:01:00+00:00",
        "messages": [
            {
                "id": "same-id",
                "ordinal": 0,
                "role": "user",
                "type": "message",
                "timestamp": "2026-09-07T10:00:00+00:00",
                "content": {"text": "Keep this context"},
                "raw_state": {"secret": "must not be persisted"},
                "model": "gpt-5.6-luna",
            },
            {
                "id": "same-id",
                "ordinal": 1,
                "role": "assistant",
                "type": "message",
                "content": {"text": "Repeated ids are valid"},
            },
        ],
        "attachments": [
            {
                "id": "attachment-1",
                "name": "note.txt",
                "mime_type": "text/plain",
                "size": 3,
                "kind": "text",
                "content": "raw attachment content is intentionally dropped",
                "metadata": {"source": "synthetic"},
            }
        ],
        "raw_state": {"transcript": "never store this"},
    }


def receipt(*, status: str = "accepted", inserted: bool = True) -> dict:
    return {
        "project_id": PROJECT_ID,
        "source": "codex",
        "session_id": "native-session-1",
        "revision": REVISION,
        "inserted": inserted,
        "message_count": 2,
        "attachment_count": 1,
        "status": status,
    }


class FakeStore(SupabaseStore):
    def __init__(self, responses: list[list[dict]]) -> None:
        super().__init__(wrapper="/unused", timeout=4)
        self.responses = list(responses)
        self.sql: list[str] = []

    def _rows(self, sql: str) -> list[dict]:  # type: ignore[override]
        self.sql.append(sql)
        if not self.responses:
            raise AssertionError("unexpected SQL call")
        return self.responses.pop(0)


def test_normalize_envelope_filters_untrusted_fields_and_preserves_repeats() -> None:
    normalized = normalize_envelope(envelope())

    assert normalized["source"] == "codex"
    assert "raw_state" not in normalized
    assert "raw_state" not in normalized["messages"][0]
    assert "content" not in normalized["attachments"][0]
    assert normalized["attachments"][0]["kind"] == "text"
    assert [item["id"] for item in normalized["messages"]] == ["same-id", "same-id"]


def test_normalize_envelope_rejects_unbounded_identity() -> None:
    value = envelope()
    value["revision"] = "not-a-sha"
    with pytest.raises(EnvelopeValidationError):
        normalize_envelope(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", {"text": "API_KEY=ACE_TEST_MARKER"}),
        ("content", {"type": "reasoning", "text": "hidden"}),
        ("content", {"data": "QUJDREVGR0hJSktMTU5PUA=="}),
        ("refs", {"token": "ACE_TEST_MARKER"}),
        ("metadata", {"password": "ACE_TEST_MARKER"}),
    ],
)
def test_normalize_envelope_rejects_payloads_not_cleaned_by_source_adapter(
    field: str, value: dict
) -> None:
    candidate = envelope()
    if field == "metadata":
        candidate["attachments"][0][field] = value
    else:
        candidate["messages"][0][field] = value

    with pytest.raises(EnvelopeValidationError):
        normalize_envelope(candidate)


def test_normalize_envelope_keeps_visible_code_concepts() -> None:
    candidate = envelope()
    candidate["messages"][0]["content"] = {
        "text": "This code explains tokenization and reasoning concepts."
    }

    normalized = normalize_envelope(candidate)

    assert normalized["messages"][0]["content"] == candidate["messages"][0]["content"]


def test_normalize_envelope_accepts_idempotent_provider_telemetry() -> None:
    telemetry = {
        "info": {
            "last_token_usage": {
                "cache_write_input_tokens": 1,
                "cached_input_tokens": 2,
                "input_tokens": 3,
                "output_tokens": 4,
                "reasoning_output_tokens": 5,
                "total_tokens": 6,
            },
            "total_token_usage": {"total_tokens": 6},
            "time_to_first_token_ms": 7,
        },
        "thread_token_usage": {"total_tokens": 8},
        "turn_token_usage": {"total_tokens": 9},
        "latest_token_usage_record": {"total_tokens": 10},
    }
    first = _clean_content(
        telemetry,
        collector=_AttachmentCollector("<test>"),
        source_line=0,
    )
    second = _clean_content(
        first,
        collector=_AttachmentCollector("<test>"),
        source_line=0,
    )
    candidate = envelope()
    candidate["messages"][0]["content"] = first

    assert first == telemetry
    assert second == first
    assert normalize_envelope(candidate)["messages"][0]["content"] == first


def test_normalize_envelope_accepts_only_redacted_secret_markers() -> None:
    candidate = envelope()
    candidate["messages"][0]["content"] = {
        "token": "<REDACTED>",
        "text": "API_KEY=<REDACTED>",
    }

    normalized = normalize_envelope(candidate)

    assert normalized["messages"][0]["content"] == candidate["messages"][0]["content"]


def test_normalize_envelope_accepts_marker_context_without_raw_secret() -> None:
    candidate = envelope()
    candidate["messages"][0]["content"] = {
        "text": "TOKEN=<REDACTED>\nnextword",
        "json_example": '{"token": "<REDACTED>"}',
    }

    normalized = normalize_envelope(candidate)

    assert normalized["messages"][0]["content"] == candidate["messages"][0]["content"]


def test_transport_uses_wrapper_stdin_and_never_puts_payload_in_args() -> None:
    response = {"ok": True, "data": [receipt()]}
    completed = CompletedProcess([], 0, stdout=json.dumps(response), stderr="")
    store = SupabaseStore(wrapper="/private/wrapper", timeout=17)

    with patch("ace_database.subprocess.run", return_value=completed) as run:
        ack_receipt = store.ingest_snapshot(envelope())

    command = run.call_args.args[0]
    assert command == ["/private/wrapper", "--profile", "amastuces", "sql", "exec"]
    assert run.call_args.kwargs["timeout"] == 17
    assert REVISION in run.call_args.kwargs["input"]
    assert "SET LOCAL ROLE ace_ingest" in run.call_args.kwargs["input"]
    assert "never store this" not in run.call_args.kwargs["input"]
    assert "revision" in ack_receipt


@pytest.mark.parametrize(
    "bad_receipt",
    [
        {**receipt(), "status": "failed"},
        {**receipt(), "inserted": "true"},
        {**receipt(), "project_id": "00000000-0000-0000-0000-000000000000"},
        {**receipt(), "source": "claude"},
        {**receipt(), "session_id": "other-session"},
        {**receipt(), "revision": "b" * 64},
    ],
)
def test_ingest_rejects_non_acknowledged_or_mismatched_receipts(bad_receipt: dict) -> None:
    store = FakeStore([[bad_receipt]])

    with pytest.raises(SupabaseStoreError):
        store.ingest_snapshot(envelope())


def test_role_transaction_unwraps_statement_results():
    store = SupabaseStore(wrapper="/private/wrapper")
    completed = CompletedProcess([], 0, stdout=json.dumps({"ok": True, "data": [[], [], [{"enabled": True}], []]}), stderr="")
    with patch("ace_database.subprocess.run", return_value=completed) as run:
        assert store._rows("SELECT * FROM ace.list_projects()") == [{"enabled": True}]
    assert "SET LOCAL ROLE ace_reader" in run.call_args.kwargs["input"]


def test_pending_filters_project_before_applying_limit():
    store = FakeStore([[]])
    store.pending_snapshots(limit=2, project_id=PROJECT_ID)
    assert f"ace.pending_snapshots(2, 'extraction', '{PROJECT_ID}'::uuid)" in store.sql[0]


def test_snapshot_deltas_batches_requests_in_one_wrapper_transaction():
    first = envelope()
    second = envelope()
    second["session_id"] = "native-session-2"
    second["revision"] = "b" * 64
    response = {
        "ok": True,
        "data": [
            {"envelope": first},
            {"envelope": second},
        ],
    }
    store = SupabaseStore(wrapper="/private/wrapper")
    requests = [
        {
            "project_id": PROJECT_ID,
            "source": "codex",
            "session_id": first["session_id"],
            "revision": first["revision"],
            "last_ordinal": -1,
        },
        {
            "project_id": PROJECT_ID,
            "source": "codex",
            "session_id": second["session_id"],
            "revision": second["revision"],
            "last_ordinal": -1,
        },
    ]
    completed = CompletedProcess([], 0, stdout=json.dumps(response), stderr="")

    with patch("ace_database.subprocess.run", return_value=completed) as run:
        rows = store.snapshot_deltas(requests)

    assert [row["session_id"] for row in rows] == [first["session_id"], second["session_id"]]
    command = run.call_args.args[0]
    assert command == ["/private/wrapper", "--profile", "amastuces", "sql", "exec"]
    sql = run.call_args.kwargs["input"]
    assert sql.count("SELECT * FROM ace.snapshot_delta(") == 2
    assert "SET LOCAL ROLE ace_processor" in sql


def test_transport_errors_are_generic_and_do_not_echo_payload_or_stderr() -> None:
    secret_marker = "payload-secret-marker"
    completed = CompletedProcess([], 1, stdout="", stderr=secret_marker)
    store = SupabaseStore(wrapper="/private/wrapper")

    with patch("ace_database.subprocess.run", return_value=completed):
        with pytest.raises(SupabaseStoreError) as caught:
            store._rows(f"SELECT '{secret_marker}'")
    assert secret_marker not in str(caught.value)


def test_register_pending_and_mark_processed_use_canonical_schema_and_stage() -> None:
    pending = normalize_envelope(envelope())
    store = FakeStore(
        [
            [{"id": PROJECT_ID, "name": "Conversations", "root": "/vault/Conversations", "vault_dir": "/vault", "enabled": True}],
            [{"envelope": pending}],
            [{"status": "succeeded", "stage": "analysis"}],
        ]
    )

    project = store.register_project(envelope()["project"])
    snapshots = store.pending_snapshots(limit=3, stage="analysis")
    run = store.mark_processed(
        "codex",
        "native-session-1",
        REVISION,
        PROJECT_ID,
        "analysis",
        "succeeded",
        lease_owner="worker-a",
        host_id="host-a",
    )

    assert project["enabled"] is True
    assert snapshots[0]["revision"] == REVISION
    assert run["stage"] == "analysis"
    assert "ace.register_project" in store.sql[0]
    assert "ace.pending_snapshots(3, 'analysis')" in store.sql[1]
    assert "ace.mark_stage" in store.sql[2]


def test_mark_processed_requires_a_lease_owner() -> None:
    store = FakeStore([])

    with pytest.raises(ValueError, match="lease_owner and host_id"):
        store.mark_processed(
            "codex", "native-session-1", REVISION, PROJECT_ID, "analysis", "succeeded"
        )


def test_claim_release_and_expiry_use_bounded_scoped_rpcs() -> None:
    claim = {
        "claimed": True,
        "lease_id": "lease-1",
        "project_id": PROJECT_ID,
        "source": "codex",
        "session_id": "native-session-1",
        "revision": REVISION,
        "stage": "analysis",
        "lease_owner": "worker-a",
        "host_id": "host-a",
        "lease_until": "2026-09-07T10:30:00+00:00",
    }
    store = FakeStore(
        [
            [claim],
            [{"released": True, "lease_id": "lease-1", "outcome": "failed"}],
            [{"expired_count": 2}],
        ]
    )

    assert store.claim_stage(
        PROJECT_ID,
        "host-a",
        "native-session-1",
        REVISION,
        source="codex",
        stage="analysis",
        lease_owner="worker-a",
    )["claimed"] is True
    assert store.release_stage(
        PROJECT_ID,
        "host-a",
        "native-session-1",
        REVISION,
        source="codex",
        stage="analysis",
        lease_owner="worker-a",
    )["released"] is True
    assert store.expire_stage_leases(limit=3)["expired_count"] == 2
    assert "ace.claim_stage" in store.sql[0]
    assert "ace.release_stage" in store.sql[1]
    assert "ace.expire_stage_leases(3)" in store.sql[2]


def test_pending_source_window_is_utc_and_applied_before_database_limit() -> None:
    store = FakeStore([[]])

    store.pending_snapshot_refs(
        limit=4,
        stage="analysis",
        project_id=PROJECT_ID,
        source_after="2026-09-07T00:00:00+02:00",
        source_before="2026-09-08T00:00:00+02:00",
    )

    sql = store.sql[0]
    assert "ace.pending_snapshot_refs_window" in sql
    assert "2026-09-06T22:00:00+00:00" in sql
    assert "2026-09-07T22:00:00+00:00" in sql


def test_pending_source_window_rejects_reversed_bounds() -> None:
    store = FakeStore([])

    with pytest.raises(ValueError, match="source_after"):
        store.pending_snapshots(
            limit=2,
            stage="analysis",
            project_id=PROJECT_ID,
            source_after="2026-09-08T00:00:00Z",
            source_before="2026-09-07T00:00:00Z",
        )


def test_persistence_accepts_envelope_context_and_compiled_snapshot_round_trip_shape() -> None:
    store = FakeStore(
        [
            [{"observations_saved": 1, "recommendations_saved": 1}],
            [{"result_id": "result-id"}],
            [{"project_id": PROJECT_ID, "version": 2, "checksum": "b" * 64}],
            [{"version": 2, "snapshot": {"index": []}, "checksum": "b" * 64}],
        ]
    )

    analysis = store.save_analysis(
        envelope(),
        {"observations": [{"problem_signature": "bounded"}], "recommendations": [{"text": "keep"}]},
    )
    result = store.save_result(envelope(), {"status": "compiled"})
    published = store.publish_compiled_snapshot(PROJECT_ID, 2, {"index": []})
    read = store.read_compiled_snapshot(PROJECT_ID, 2)

    assert analysis["observations_saved"] == 1
    assert result["result_id"] == "result-id"
    assert published["version"] == 2
    assert read is not None and read["version"] == 2
    assert all("ace." in statement for statement in store.sql)


def test_migration_is_additive_and_has_default_deny_controls() -> None:
    migration = (Path(__file__).parents[1] / "migrations" / "001_ace.sql").read_text()
    lowered = migration.lower()
    assert "create schema if not exists ace" in lowered
    assert "create table if not exists ace.projects" in lowered
    assert "create table if not exists ace.revisions" in lowered
    assert "create table if not exists ace.processing_runs" in lowered
    assert "create table if not exists ace.knowledge_versions" in lowered
    assert "create role ace_ingest nologin" in lowered
    assert "create role ace_processor nologin" in lowered
    assert "create role ace_reader nologin" in lowered
    assert "enable row level security" in lowered
    assert "revoke all on all tables in schema ace from public" in lowered
    assert "revoke all on all functions in schema ace from public" in lowered
    assert "create or replace function ace.jsonb_is_clean" in lowered
    assert "ace snapshot is not sanitized" in lowered or "ace message is not sanitized" in lowered
    assert "when excluded.updated_at is null then sessions.latest_revision" in lowered
    assert "coalesce(pr.finished_at, pr.started_at, pr.created_at)" in lowered
    assert "case when candidates.attempt_id is null then 0 else 1 end" in lowered
    assert "started_at = excluded.started_at" in lowered
    assert "clock_timestamp()" in lowered
    for key in (
        "last_token_usage",
        "total_token_usage",
        "thread_token_usage",
        "turn_token_usage",
        "latest_token_usage_record",
        "time_to_first_token_ms",
    ):
        assert key in lowered
    assert "<redacted>" in lowered
    assert "disabled compiled publication was accepted" in Path(
        Path(__file__).parents[1] / "scripts" / "ace_schema.py"
    ).read_text().lower()
    assert "drop table" not in lowered
    assert "drop schema" not in lowered


def test_stored_envelope_is_repaired_instead_of_rejected() -> None:
    """A rule change must not turn a stored conversation into a dead row."""
    candidate = envelope()
    candidate["messages"][0]["content"] = "token: hunter2"
    with pytest.raises(EnvelopeValidationError):
        normalize_envelope(candidate)
    repaired = normalise_stored_envelope(candidate)
    assert "hunter2" not in json.dumps(repaired, ensure_ascii=False)
    assert "<REDACTED>" in json.dumps(repaired, ensure_ascii=False)


def test_unrepairable_stored_envelope_still_raises() -> None:
    candidate = envelope()
    candidate["messages"] = "not a list"
    with pytest.raises(EnvelopeValidationError):
        normalise_stored_envelope(candidate)
