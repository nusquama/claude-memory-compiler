from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_learning as learning  # noqa: E402


def test_analysis_output_schema_requires_claim_proof_fields() -> None:
    schema = learning.build_analysis_output_schema()
    assert schema["required"] == [
        "schema_version",
        "verdict",
        "status",
        "conversations",
        "incidents",
        "successes",
        "observations",
        "recommendations",
        "limitations",
    ]
    incident = schema["properties"]["incidents"]["items"]
    assert {"evidence_refs", "message_ids", "expected", "observed", "recommendation", "test"} <= set(
        incident["required"]
    )
    success = schema["properties"]["successes"]["items"]
    assert {"evidence_refs", "message_ids", "summary"} <= set(success["required"])
    conversation = schema["properties"]["conversations"]["items"]
    assert conversation["properties"]["status"]["enum"] == [
        "success",
        "incident",
        "insufficient_evidence",
    ]
    assert schema["additionalProperties"] is False
    for key in ("cause", "incidents", "successes", "observations", "recommendations"):
        value = schema["properties"][key]["items"] if key != "cause" else schema["properties"]["incidents"]["items"]["properties"]["cause"]
        assert value["additionalProperties"] is False
        assert set(value["required"]) == set(value["properties"])


def _snapshot(session_id: str, *, coverage: str = "complete", extra: str = "") -> dict[str, object]:
    messages = [
        {"id": "u", "ordinal": 1, "role": "user", "type": "message", "content": "only one file"},
        {"id": "a", "ordinal": 2, "role": "assistant", "type": "message", "content": extra or "checked"},
        {"id": "t", "ordinal": 3, "role": "assistant", "type": "task_complete", "status": "completed", "content": "finished"},
    ]
    return {
        "project": {"id": "project-a", "name": "Fixture"},
        "source": "codex",
        "session_id": session_id,
        "revision": "rev-1",
        "coverage": {"completeness": {"observation": coverage}},
        "messages": messages,
    }


def test_luna_json_requires_resolved_evidence_and_marks_model_error() -> None:
    source = _snapshot("luna")

    async def runner(records, prompt):
        del records, prompt
        return "```json\n" + json.dumps(
            {
                "verdict": "success",
                "conversations": [{"conversation_id": "codex:luna", "status": "success"}],
                "successes": [{"evidence_refs": ["missing-ref"]}],
            }
        ) + "\n```"

    result = asyncio.run(learning.audit_snapshots([source], audit_runner=runner))
    assert result["status"] == "model-error"
    assert result["errors"][0]["status"] == "model-error"
    assert result["conversations"][0]["status"] == "model-error"
    assert result["successes"] == []


def test_invalid_model_report_gets_one_bounded_repair_attempt() -> None:
    source = _snapshot("repair")
    valid = {
        "schema_version": "1",
        "verdict": "insufficient_evidence",
        "status": "insufficient_evidence",
        "conversations": [
            {
                "conversation_id": "codex:repair",
                "subject": "Fixture",
                "level": "none",
                "status": "insufficient_evidence",
                "summary": "No supported incident.",
                "incidents": [],
                "skills": [],
            }
        ],
        "incidents": [],
        "successes": [],
        "observations": [],
        "recommendations": [],
        "limitations": [],
    }
    calls = 0

    async def runner(records, prompt):
        nonlocal calls
        del records, prompt
        calls += 1
        if calls == 1:
            return {**valid, "incidents": [{"evidence_refs": ["missing"], "message_ids": ["missing"]}]}
        return valid

    result = asyncio.run(learning.audit_snapshots([source], audit_runner=runner))
    assert calls == 2
    assert result["status"] == "ok"
    assert result["errors"] == []


def test_model_error_stays_private_while_valid_report_is_saved(tmp_path: Path) -> None:
    bad = _snapshot("bad-save")
    good = _snapshot("good-save")

    class Store:
        def __init__(self) -> None:
            self.rows: list[dict[str, object]] = []

        async def save_analysis(self, row: dict[str, object]) -> dict[str, object]:
            self.rows.append(row)
            return {"ok": True}

    async def runner(records, prompt):
        del records, prompt
        return [
            {
                "verdict": "insufficient_evidence",
                "conversations": [{"conversation_id": "codex:bad-save", "status": "insufficient_evidence"}],
                "incidents": [{"evidence_refs": ["missing"], "message_ids": ["missing"]}],
                "successes": [],
                "observations": [],
                "recommendations": [],
                "limitations": [],
            },
            {
                "verdict": "insufficient_evidence",
                "conversations": [{"conversation_id": "codex:good-save", "status": "insufficient_evidence"}],
                "incidents": [],
                "successes": [],
                "observations": [],
                "recommendations": [],
                "limitations": [],
            },
        ]

    store = Store()
    result = asyncio.run(
        learning.audit_snapshots(
            [bad, good],
            store=store,
            state_dir=tmp_path,
            audit_runner=runner,
        )
    )
    assert result["status"] == "model-error"
    assert [row["session_id"] for row in store.rows] == ["good-save"]
    history = learning._load_jsonl(tmp_path / "analysis-history.jsonl")
    assert {row["session_id"] for row in history} == {"bad-save", "good-save"}
    assert any(row.get("analysis_status") == "model-error" for row in history)


def test_nested_luna_claims_are_flattened_with_real_message_proof() -> None:
    source = _snapshot("nested")
    record = learning.snapshot_to_record(source)
    window = record["_evidence"][0]
    model = {
        "schema_version": "1",
        "verdict": "insufficient_evidence",
        "conversations": [
            {
                "conversation_id": "codex:nested",
                "status": "insufficient_evidence",
                "summary": "runner",
                "incidents": [
                    {
                        "type": "calm_frustration_signal",
                        "evidence": {"ref": window["ref"], "message_id": window["message_id"]},
                        "cause": "unknown",
                    }
                ],
                "observations": [],
                "successes": [],
            }
        ],
        "incidents": [],
        "observations": [],
        "recommendations": [],
        "successes": [],
        "limitations": [],
    }

    async def runner(records, prompt):
        del records, prompt
        return model

    result = asyncio.run(learning.audit_snapshots([source], audit_runner=runner))
    assert result["status"] == "ok"
    assert result["incidents"][0]["evidence_refs"] == [window["ref"]]
    assert result["incidents"][0]["message_ids"] == [window["message_id"]]
    assert result["incidents"][0]["expected"]
    assert result["incidents"][0]["observed"]
    assert result["incidents"][0]["recommendation"]
    assert result["incidents"][0]["test"]
    assert result["incidents"][0]["detail_source"] == "generic_suggestion_missing_model_detail"
    assert result["reports"][0]["analysis_contract_version"] == learning.ANALYSIS_CONTRACT_VERSION


def test_nested_luna_message_outside_window_is_model_error() -> None:
    source = _snapshot("bad-proof")
    record = learning.snapshot_to_record(source)
    window = record["_evidence"][0]
    model = {
        "conversations": [
            {
                "conversation_id": "codex:bad-proof",
                "status": "insufficient_evidence",
                "incidents": [
                    {
                        "type": "tool_errors",
                        "evidence": {"ref": window["ref"], "message_id": "not-in-window"},
                    }
                ],
            }
        ]
    }

    async def runner(records, prompt):
        del records, prompt
        return model

    result = asyncio.run(learning.audit_snapshots([source], audit_runner=runner))
    assert result["status"] == "model-error"


def test_cause_can_use_a_second_valid_evidence_window() -> None:
    source = _snapshot("cause-window")
    record = learning.snapshot_to_record(learning.normalize_snapshot(source))
    record["_evidence"] = [
        {"ref": "ev-root", "message_ids": ["u"]},
        {"ref": "ev-cause", "message_ids": ["a"]},
    ]
    report = {
        "verdict": "insufficient_evidence",
        "conversations": [{"conversation_id": "codex:cause-window", "status": "insufficient_evidence"}],
        "incidents": [
            {
                "type": "tool_failure",
                "evidence_refs": ["ev-root"],
                "message_ids": ["u"],
                "cause": {
                    "status": "verified",
                    "summary": "The tool failed.",
                    "evidence_refs": ["ev-cause"],
                },
            }
        ],
    }
    diagnostics: list[dict[str, object]] = []
    normalized = learning._normalise_model_report(report, record, diagnostics)
    assert normalized is not None
    incident = normalized["incidents"][0]
    assert incident["evidence_refs"] == ["ev-root", "ev-cause"]
    assert incident["message_ids"] == ["u", "a"]
    assert incident["cause"]["evidence_refs"] == ["ev-cause"]
    assert diagnostics == []


def test_disjoint_evidence_pairs_are_validated_per_window() -> None:
    record = {
        "source": "codex",
        "session_id": "pairs",
        "revision": "r",
        "source_hash": "h",
        "project_id": "p",
        "_evidence": [
            {"ref": "ev-a", "message_ids": ["m-a"]},
            {"ref": "ev-b", "message_ids": ["m-b"]},
        ],
        "_completeness": {"source_available": True, "terminal_evidence": False, "observation": "partial"},
    }
    report = {
        "conversations": [{"conversation_id": "codex:pairs", "status": "insufficient_evidence"}],
        "incidents": [
            {
                "type": "gap",
                "evidence_refs": ["ev-a", "ev-b"],
                "message_ids": ["m-a", "m-b"],
                "expected": "bounded",
                "observed": "gap",
                "recommendation": "inspect",
                "test": "rerun",
            }
        ],
        "successes": [],
        "observations": [],
        "recommendations": [],
        "limitations": [],
    }
    normalized = learning._normalise_model_report(report, record)
    assert normalized is not None
    assert normalized["incidents"][0]["message_ids"] == ["m-a", "m-b"]

    invalid = {**report, "incidents": [{**report["incidents"][0], "message_ids": ["m-a", "not-in-window"]}]}
    assert learning._normalise_model_report(invalid, record) is None


def test_luna_union_evidence_windows_preserves_messages_and_rejects_unsupported_claims() -> None:
    record = {
        "source": "codex",
        "session_id": "union",
        "revision": "r",
        "source_hash": "h",
        "project_id": "p",
        "_evidence": [
            {"ref": "ev-a", "message_ids": ["m-a1", "m-a2"]},
            {"ref": "ev-b", "message_ids": ["m-b1", "m-b2"]},
        ],
        "_completeness": {"source_available": True, "terminal_evidence": False, "observation": "partial"},
    }
    report = {
        "conversations": [{"conversation_id": "codex:union", "status": "partial"}],
        "incidents": [
            {
                "id": "incident-union",
                "evidence_refs": ["ev-a", "ev-b"],
                "message_ids": ["m-a1", "m-a2", "m-b1", "m-b2"],
                "expected": "bounded result",
                "observed": "partial result",
                "recommendation": "inspect the final marker",
                "test": "replay the bounded case",
            }
        ],
        "observations": [
            {
                "message": "Observed model message",
                "evidence_refs": ["ev-a", "ev-b"],
                "message_ids": ["m-a1", "m-b2"],
            }
        ],
        "recommendations": [
            {
                "message": "Model recommendation message",
                "evidence_refs": ["ev-b"],
                "message_ids": ["m-b1", "m-b2"],
            }
        ],
        "successes": [],
        "limitations": [],
    }

    normalized = learning._normalise_model_report(report, record)
    assert normalized is not None
    assert normalized["incidents"][0]["message_ids"] == ["m-a1", "m-a2", "m-b1", "m-b2"]
    assert normalized["observations"][0]["message"] == "Observed model message"
    assert normalized["recommendations"][0]["message"] == "Model recommendation message"
    assert normalized["recommendations"][0]["text"] == "Model recommendation message"

    outside = {
        **report,
        "incidents": [{**report["incidents"][0], "message_ids": ["m-a1", "outside"]}],
    }
    assert learning._normalise_model_report(outside, record) is None

    unsupported_window = {
        **report,
        "incidents": [{**report["incidents"][0], "message_ids": ["m-a1"]}],
    }
    assert learning._normalise_model_report(unsupported_window, record) is None


def test_renderer_derives_status_and_uses_normalized_recommendation_text() -> None:
    report = learning._report_records(
        [
            {
                "source": "codex",
                "session_id": "renderer-ok",
                "revision": "r1",
                "audit_date": "2026-09-09",
                "analysis_status": "ok",
                "recommendations": [{"text": "Use the observed minimal check."}],
            },
            {
                "source": "codex",
                "session_id": "renderer-degraded",
                "revision": "r1",
                "audit_date": "2026-09-09",
                "analysis_status": "degraded",
            },
        ],
        period="daily",
        end_day=date(2026, 9, 9),
    )
    assert report["status"] == "degraded"
    assert report["recommendations"][0]["text"] == "Use the observed minimal check."


def test_analysis_row_projects_incidents_without_inventing_missing_recommendations() -> None:
    record = {"source": "codex", "session_id": "mapping", "revision": "r", "source_hash": "h"}
    report = {
        "incidents": [
            {
                "id": "incident-a",
                "type": "tool_failure",
                "expected": "success",
                "observed": "error",
                "recommendation": "retry safely",
                "test": "rerun",
                "evidence_refs": ["ev-a"],
                "message_ids": ["m-a"],
            },
            {
                "id": "incident-b",
                "type": "gap",
                "expected": "proof",
                "observed": "missing",
                "evidence_refs": ["ev-b"],
                "message_ids": ["m-b"],
            },
        ],
        "observations": [],
        "recommendations": [],
        "successes": [],
    }
    row = learning._analysis_row(report, record, "2026-09-09T00:00:00Z")
    assert {item["incident_id"] for item in row["observations"]} == {"incident-a", "incident-b"}
    assert [item["incident_id"] for item in row["recommendations"]] == ["incident-a"]
    assert row["recommendations"][0]["text"] == "retry safely"


def test_partial_coverage_cannot_become_success() -> None:
    record = learning.snapshot_to_record(_snapshot("partial", coverage="partial"))
    report = learning._heuristic_report(record)
    assert report["conversations"][0]["status"] == "insufficient_evidence"
    assert report["successes"] == []


def test_overengineering_requires_scope_mismatch() -> None:
    neutral = learning._heuristic_report(learning.snapshot_to_record(_snapshot("neutral", extra="architecture discussion")))
    mismatch = learning._heuristic_report(
        learning.snapshot_to_record(_snapshot("mismatch", extra="new architecture refactor three files"))
    )
    assert not any(item["type"] == "overengineering" for item in neutral["incidents"])
    assert any(item["type"] == "overengineering" for item in mismatch["incidents"])
    incident = next(item for item in mismatch["incidents"] if item["type"] == "overengineering")
    assert incident["evidence_refs"]
    assert "minimal" in incident["recommendation"]
    assert incident["test"]


def test_history_recurrence_is_distinct_session_and_retry_safe(tmp_path: Path) -> None:
    rows = [
        {"session_id": "a", "scope": "p", "type": "gap", "signature": "same", "revision": "r", "source_hash": "h", "evidence_refs": ["a"]},
        {"session_id": "a", "scope": "p", "type": "gap", "signature": "same", "revision": "r", "source_hash": "h", "evidence_refs": ["a"]},
        {"session_id": "b", "scope": "p", "type": "gap", "signature": "same", "revision": "r", "source_hash": "h2", "evidence_refs": ["b"]},
    ]
    patterns = learning.detect_recurrence(rows, history=[{"session_id": "c", "scope": "p", "type": "gap", "signature": "same", "revision": "r", "source_hash": "h3", "evidence_refs": ["c"]}])
    assert patterns[0]["sessions"] == ["a", "b", "c"]
    assert patterns[0]["occurrences"] == 3

    record = {"source": "codex", "session_id": "s", "revision": "r", "source_hash": "h"}
    first = learning._analysis_history_key(record)
    second = learning._analysis_history_key(record)
    assert first == second
    assert first[-1] == "legacy"
    current = learning._analysis_row({"incidents": [], "observations": [], "recommendations": [], "successes": []}, record, "2026-09-09T00:00:00Z")
    assert current["analysis_identity"]["analysis_contract_version"] == learning.ANALYSIS_CONTRACT_VERSION
    assert current["analysis_key"] != learning.hashlib.sha256(learning._canonical_json(first).encode()).hexdigest()


def test_history_keeps_changed_retry_outcome_and_report_uses_latest_attempt(tmp_path: Path) -> None:
    base = {
        "source": "codex",
        "session_id": "retry-session",
        "revision": "revision-1",
        "source_hash": "snapshot-hash",
        "analysis_contract_version": learning.ANALYSIS_CONTRACT_VERSION,
        "audit_date": "2026-09-09",
    }
    failed = {
        **base,
        "created_at": "2026-09-09T10:00:00+00:00",
        "status": "model-error",
        "analysis_status": "model-error",
        "incidents": [{"id": "failed-incident", "type": "tool_failure"}],
        "recommendations": [{"text": "fallback"}],
    }
    succeeded = {
        **base,
        "created_at": "2026-09-09T10:01:00+00:00",
        "status": "ok",
        "analysis_status": "ok",
        "incidents": [{"id": "real-incident", "type": "tool_failure"}],
        "recommendations": [{"text": "real recommendation"}],
    }
    history_path = tmp_path / "analysis-history.jsonl"
    first = learning._append_unique_analysis_history(history_path, [failed])
    second = learning._append_unique_analysis_history(history_path, [succeeded, succeeded])
    assert len(first) == 1
    assert len(second) == 2
    assert len(learning._load_jsonl(history_path)) == 2

    latest = learning._latest_analysis_attempts(learning._load_jsonl(history_path))
    assert len(latest) == 1
    assert latest[0]["analysis_status"] == "ok"
    report = learning._report_records(latest + [failed], period="daily", end_day=date(2026, 9, 9))
    assert report["records"] == 1
    assert report["status"] == "ok"
    assert report["recommendations"][0]["text"] == "real recommendation"


def test_usage_unknown_mapping_is_not_known_and_report_contains_proof_fields(tmp_path: Path) -> None:
    payload = {
        "records": [
            {
                "source_date": "2026-09-09",
                "usage": {"status": "unknown"},
                "incidents": [
                    {
                        "id": "i",
                        "type": "gap",
                        "evidence_refs": ["ev-1"],
                        "cause": {"status": "unknown"},
                        "observed": "mismatch",
                        "recommendation": "minimal fix",
                        "test": "rerun",
                    }
                ],
            }
        ]
    }
    learning.render_reports(payload, tmp_path, now="2026-09-09T12:00:00Z")
    report = json.loads((tmp_path / "daily-2026-09-09.json").read_text())
    assert report["usage"]["status"] == "unknown"
    assert report["proofs"][0]["evidence_refs"] == ["ev-1"]
    assert report["minimal_solutions"][0]["test"] == "rerun"
    assert "Preuves et solution minimale" in (tmp_path / "daily-2026-09-09.md").read_text()


def test_rejection_diagnostics_identify_claim_without_raw_text() -> None:
    record = learning.snapshot_to_record(learning.normalize_snapshot(_snapshot("diagnostics")))
    diagnostics = []
    report = {"incidents": [{"evidence_refs": ["unknown-ref"], "message_ids": ["u"],
                              "observed": "PRIVATE BODY MUST NOT APPEAR"}]}
    assert learning._normalise_model_report(report, record, diagnostics) is None
    assert diagnostics == [{"field": "incidents", "claim_index": 0,
                            "reason": "unknown_evidence_ref", "ref": "unknown-ref", "message_id": None}]
    assert "PRIVATE BODY" not in json.dumps(diagnostics)


def test_repair_keeps_both_structured_rejections_and_stays_fail_closed() -> None:
    calls = 0
    async def runner(records, prompt):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert "unknown_evidence_ref" in prompt
        return {"incidents": [{"evidence_refs": ["bad-ref"], "message_ids": ["u"]}]}
    result = asyncio.run(learning.audit_snapshots([_snapshot("rejected")], audit_runner=runner))
    assert calls == 2
    assert result["status"] == "model-error"
    assert result["incidents"] == []
    error = result["errors"][0]
    assert error["validation_errors"][0]["reason"] == "unknown_evidence_ref"
    assert error["repair_validation_errors"][0]["reason"] == "unknown_evidence_ref"


def test_diagnostics_hash_non_identifier_values() -> None:
    diagnostics = []
    learning._validation_diagnostic(diagnostics, "incidents", 2, "unknown_evidence_ref", "arbitrary private prose")
    assert diagnostics[0]["ref"].startswith("sha256:")
    assert "arbitrary private prose" not in json.dumps(diagnostics)


def test_report_keeps_verifiable_claims_and_drops_only_bad_citations() -> None:
    """One bad citation must not discard three good findings."""
    record = {
        "source": "codex",
        "session_id": "salvage",
        "revision": "r",
        "source_hash": "h",
        "project_id": "p",
        "_evidence": [
            {"ref": "ev-a", "message_ids": ["m-a1", "m-a2"]},
            {"ref": "ev-b", "message_ids": ["m-b1", "m-b2"]},
        ],
        "_completeness": {"source_available": True, "terminal_evidence": False, "observation": "partial"},
    }
    good_incident = {
        "id": "inc-good",
        "evidence_refs": ["ev-a"],
        "message_ids": ["m-a1", "m-a2"],
        "expected": "bounded result",
        "observed": "partial result",
        "recommendation": "inspect the final marker",
        "test": "replay the bounded case",
    }
    bad_incident = {**good_incident, "id": "inc-bad", "evidence_refs": ["ev-zzz"], "message_ids": ["m-a1"]}
    report = {
        "conversations": [{"conversation_id": "codex:salvage", "status": "partial"}],
        "incidents": [good_incident, bad_incident],
        "observations": [{"message": "seen", "evidence_refs": ["ev-b"], "message_ids": ["m-b1", "m-b2"]}],
        "recommendations": [],
        "successes": [],
        "limitations": [],
    }
    diagnostics: list[dict] = []
    normalized = learning._normalise_model_report(report, record, diagnostics)
    assert normalized is not None
    assert [item["id"] for item in normalized["incidents"]] == ["inc-good"]
    assert normalized["dropped_claims"] >= 1
    assert any("écarté" in item for item in normalized["limitations"])

    only_bad = {**report, "incidents": [bad_incident], "observations": []}
    assert learning._normalise_model_report(only_bad, record) is None
