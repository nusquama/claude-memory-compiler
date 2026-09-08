from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_learning as learning  # noqa: E402


def _snapshot(session_id: str = "safety") -> dict[str, object]:
    return {
        "project": {"id": "project-a"},
        "source": "codex",
        "session_id": session_id,
        "revision": "rev-1",
        "messages": [
            {"id": "u", "ordinal": 1, "role": "user", "type": "message", "content": "request"},
            {"id": "a", "ordinal": 2, "role": "assistant", "type": "message", "content": "result"},
            {
                "id": "t",
                "ordinal": 3,
                "role": "assistant",
                "type": "task_complete",
                "status": "completed",
                "content": "complete",
            },
        ],
    }


def test_terminal_task_complete_without_role_is_not_success_evidence() -> None:
    assert not learning._is_terminal({"type": "task_complete", "status": "completed"})
    source = _snapshot("missing-role")
    source["messages"][-1] = {"id": "t", "ordinal": 3, "type": "task_complete", "status": "completed"}
    record = learning.snapshot_to_record(source)
    assert record["_completeness"]["terminal_evidence"] is False


def test_terminal_requires_explicit_marker_real_user_and_final_nonruntime_event() -> None:
    assert not learning._is_terminal({"role": "assistant", "type": "message", "status": "completed"})
    assert not learning._is_terminal({"role": "user", "type": "assistant_message", "status": "completed"})
    source_with_stop_reason = _snapshot("stop-reason")
    source_with_stop_reason["messages"][-1] = {
        "id": "a-stop",
        "ordinal": 3,
        "role": "assistant",
        "type": "message",
        "stop_reason": "end_turn",
        "content": "complete",
    }
    assert learning.snapshot_to_record(source_with_stop_reason)["_completeness"]["terminal_evidence"] is True
    source = _snapshot("late-output")
    source["messages"].append({"id": "late", "ordinal": 4, "role": "assistant", "type": "message", "content": "after marker"})
    record = learning.snapshot_to_record(source)
    assert record["_completeness"]["real_user_turn"] is True
    assert record["_completeness"]["terminal_evidence"] is False


def test_nested_mapping_in_list_is_redacted_and_bounded() -> None:
    value = {
        "events": [
            {"access_token": "SECRET_SENTINEL", "safe": "kept"},
            {"nested": [{"password": "PASSWORD_SENTINEL"}]},
        ]
    }
    row = learning._safe_row({"payload": value})
    encoded = json.dumps(row, ensure_ascii=False)
    assert "SECRET_SENTINEL" not in encoded
    assert "PASSWORD_SENTINEL" not in encoded
    assert row["payload"]["events"][0]["access_token"] == "<REDACTED>"


def test_unknown_model_claim_fields_do_not_reach_normalized_or_persisted_row() -> None:
    source = _snapshot("unknown-claim")
    record = learning.snapshot_to_record(source)
    evidence = record["_evidence"][0]
    report = {
        "conversations": [{"conversation_id": "codex:unknown-claim", "status": "insufficient_evidence"}],
        "observations": [
            {
                "type": "preference",
                "message": "bounded",
                "evidence_refs": [evidence["ref"]],
                "message_ids": [evidence["message_id"]],
                "synthetic_claim_field": "SYNTHETIC_SENTINEL",
            }
        ],
    }
    normalized = learning._normalise_model_report(report, record)
    assert normalized is not None
    assert "SYNTHETIC_SENTINEL" not in json.dumps(normalized, ensure_ascii=False)
    row = learning._analysis_row(normalized, record, "2026-09-07T12:00:00Z")
    assert "SYNTHETIC_SENTINEL" not in json.dumps(row, ensure_ascii=False)


def test_save_analysis_requires_explicit_positive_receipt() -> None:
    for receipt in (False, None, {"ok": False}, {"status": "failed"}, {"unexpected": "value"}):
        try:
            learning._require_positive_save_receipt(receipt)
        except RuntimeError:
            pass
        else:  # pragma: no cover - assertion gives a clearer failure below.
            raise AssertionError(f"receipt unexpectedly accepted: {receipt!r}")
    assert learning._require_positive_save_receipt({"observations_saved": 0, "recommendations_saved": 0})[
        "observations_saved"
    ] == 0


def test_learning_events_require_positive_ack_and_saved_identifier() -> None:
    class Store:
        def __init__(self, receipt: object) -> None:
            self.receipt = receipt

        def save_decision(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return self.receipt

        def save_correction(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return self.receipt

        def save_evaluation(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return self.receipt

    common = {
        "project_id": "project-a",
        "source": "codex",
        "session_id": "session-a",
        "revision": "revision-a",
        "payload": {"scope": "project-a", "type": "tool_failure", "signature": "read"},
    }
    for event, identifier in (
        ("decision", "decision_id"),
        ("correction", "correction_id"),
        ("evaluation", "evaluation_id"),
    ):
        for receipt in (None, False, {}, {identifier: ""}, {identifier: "saved", "ok": False}, {identifier: "saved", "status": "failed"}):
            try:
                learning.save_learning_event(Store(receipt), event, **common)
            except RuntimeError:
                pass
            else:  # pragma: no cover - assertion gives a clearer failure below.
                raise AssertionError(f"{event} receipt unexpectedly accepted: {receipt!r}")
        assert learning.save_learning_event(Store({identifier: f"{event}-1"}), event, **common)[identifier]


def test_refusal_learning_event_is_explicit_at_existing_decision_store_boundary() -> None:
    class Store:
        def __init__(self) -> None:
            self.payload = None

        def save_decision(self, project_id: object, payload: object, **kwargs: object) -> dict[str, str]:
            del project_id, kwargs
            self.payload = payload
            return {"decision_id": "decision-refusal-1"}

    store = Store()
    receipt = learning.save_learning_event(
        store,
        "refusal",
        project_id="project-a",
        source="codex",
        session_id="session-a",
        revision="revision-a",
        payload={
            "scope": "project-a",
            "type": "tool_failure",
            "signature": "read",
            "reason": "user declined",
            "evidence_refs": ["snapshot:session-a:message:u"],
        },
    )
    assert receipt["decision_id"] == "decision-refusal-1"
    assert store.payload["event"] == "refusal"
    assert store.payload["refused"] is True
    assert store.payload["reason"] == "user declined"
    assert store.payload["refusal_reason"] == "user declined"
    assert store.payload["evidence_refs"] == ["snapshot:session-a:message:u"]
    assert store.payload["refusal_evidence_refs"] == ["snapshot:session-a:message:u"]


def test_record_cli_does_not_report_success_for_missing_event_ack(monkeypatch, capsys) -> None:
    class Store:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def save_decision(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            return None

    module = types.ModuleType("ace_database")
    module.SupabaseStore = Store
    monkeypatch.setitem(sys.modules, "ace_database", module)
    exit_code = learning.main(
        [
            "record",
            "refusal",
            "--project-id",
            "project-a",
            "--source",
            "codex",
            "--session-id",
            "session-a",
            "--revision",
            "revision-a",
            "--payload-json",
            '{"scope":"project-a","type":"tool_failure","signature":"read"}',
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["status"] == "failed"


def test_false_save_ack_is_degraded_and_never_committed(tmp_path: Path) -> None:
    class Store:
        async def save_analysis(self, row):
            del row
            return False

    result = asyncio.run(learning.audit_snapshots([_snapshot("false-ack")], store=Store(), state_dir=tmp_path))
    assert result["status"] == "degraded"
    assert result["errors"][0]["kind"] == "save_analysis rejected"
    history = learning._load_jsonl(tmp_path / "analysis-history.jsonl")
    assert history == []


def test_refusal_history_blocks_followup_across_sessions(tmp_path: Path) -> None:
    proposal = {"scope": "project-a", "type": "tool_failure", "signature": "read", "text": "inspect"}
    learning.record_refusal(tmp_path, proposal)
    evaluations = [
        {**proposal, "session_id": f"session-{index}", "outcome": "failed"}
        for index in range(3)
    ]
    assert learning.followup_recommendations(evaluations, state_dir=tmp_path) == []


def test_recurrence_requires_known_session_and_resolved_proof_per_occurrence() -> None:
    invalid = [
        {"session_id": "s-missing-proof", "scope": "p", "type": "gap", "signature": "same"},
        {"session_id": "unknown", "scope": "p", "type": "gap", "signature": "same", "evidence_refs": ["ev-unknown"]},
        {"session_id": "s-sentinel-proof", "scope": "p", "type": "gap", "signature": "same", "evidence_refs": ["unknown"]},
        {
            "session_id": "s-child-proof",
            "scope": "p",
            "type": "gap",
            "signature": "same",
            "evidence_refs": ["row-level-only"],
            "incidents": [{"scope": "p", "type": "gap", "signature": "same"}],
        },
    ]
    assert learning.detect_recurrence(invalid) == []

    valid = [
        {"session_id": f"session-{index}", "scope": "p", "type": "gap", "signature": "same", "evidence_refs": [f"ev-{index}"]}
        for index in range(3)
    ]
    patterns = learning.detect_recurrence(valid)
    assert len(patterns) == 1
    assert patterns[0]["sessions"] == ["session-0", "session-1", "session-2"]
    assert patterns[0]["occurrences"] == 3
    assert patterns[0]["evidence_refs"] == ["ev-0", "ev-1", "ev-2"]


def test_business_metrics_keep_states_proofs_elapsed_recurrence_and_tokens_distinct() -> None:
    records = [
        {
            "session_id": "after-fix",
            "scope": "project-a",
            "type": "tool_failure",
            "signature": "read",
            "created_at": "2026-09-07T12:00:00Z",
            "incidents": [
                {
                    "scope": "project-a",
                    "type": "tool_failure",
                    "signature": "read",
                    "created_at": "2026-09-07T12:00:00Z",
                }
            ],
        },
        {
            "accepted": True,
            "accepted_evidence_refs": ["ev-accepted"],
            "applied": True,
            "applied_evidence_refs": ["ev-applied"],
            "effective": False,
            "requested_at": "2026-09-07T10:00:00Z",
            "accepted_at": "2026-09-07T10:02:00Z",
        },
    ]
    correction = {
        "event": "correction",
        "scope": "project-a",
        "type": "tool_failure",
        "signature": "read",
        "applied": True,
        "applied_evidence_refs": ["ev-applied"],
        "applied_at": "2026-09-07T11:00:00Z",
    }
    metrics = learning._business_metrics(
        records,
        corrections=[correction],
        usage_values=[{"stage": "analysis", "token_usage": {"input_tokens": 4, "output_tokens": 5}}],
    )
    assert metrics["accepted"]["yes"] == 1
    assert metrics["accepted"]["with_proof"] == 1
    assert metrics["applied"]["with_proof"] == 2  # state row + explicit correction event
    assert metrics["effective"]["no"] == 1
    assert metrics["elapsed_request_to_accepted"]["seconds"] == 120.0
    assert metrics["recurrence_after_correction"]["recidive_count"] == 1
    assert metrics["tokens_by_stage"]["analysis"] == {"input_tokens": 4, "output_tokens": 5}
    assert "total_tokens" not in metrics["tokens_by_stage"]["analysis"]


def test_audit_accepts_runner_diagnostics_without_treating_it_as_model_json() -> None:
    class Diagnostics:
        def as_metrics(self) -> dict[str, object]:
            return {
                "call_count": 1,
                "duration_seconds": 0.25,
                "token_usage": {"input_tokens": 2, "output_tokens": 3},
                "usage_status": "available",
            }

    async def runner(records, prompt):
        del records, prompt
        return ({"status": "insufficient_evidence", "conversations": []}, Diagnostics())

    result = asyncio.run(learning.audit_snapshots([_snapshot("diagnostics")], audit_runner=runner))
    assert result["status"] == "ok"
    assert result["business_metrics"]["tokens_by_stage"]["analysis"] == {
        "input_tokens": 2,
        "output_tokens": 3,
    }
    assert result["usage"]["usage_status"] == "available"
    assert result["runner_metrics"]["call_count"] == 1
