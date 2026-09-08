from __future__ import annotations

import sys
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ace_overengineering_audit import (  # noqa: E402
    _manifest_records,
    build_report_schema,
    extract_evidence_windows,
    missing_report_requirements,
    observe_source_completeness,
    record_date_metadata,
    select_pending_batch,
    update_incident_tracking,
    validate_structured_report,
    should_trigger,
    structural_metrics,
)


class OverengineeringTriggerTests(unittest.TestCase):
    def test_triggers_at_ten_pending_conversations(self) -> None:
        trigger, reason = should_trigger(
            10,
            batch_size=10,
            max_age_days=7,
            last_report_at=datetime.now(timezone.utc).isoformat(),
        )
        self.assertTrue(trigger)
        self.assertIn("batch threshold", reason)

    def test_triggers_weekly_when_new_conversations_exist(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        trigger, reason = should_trigger(
            2,
            batch_size=10,
            max_age_days=7,
            last_report_at=old,
        )
        self.assertTrue(trigger)
        self.assertIn("weekly threshold", reason)

    def test_does_not_run_without_new_conversations(self) -> None:
        trigger, reason = should_trigger(
            0,
            batch_size=10,
            max_age_days=7,
            last_report_at=None,
        )
        self.assertFalse(trigger)
        self.assertEqual("no new conversations", reason)

    def test_first_run_waits_for_full_batch(self) -> None:
        trigger, reason = should_trigger(
            3,
            batch_size=10,
            max_age_days=7,
            last_report_at=None,
        )
        self.assertFalse(trigger)
        self.assertIn("first batch", reason)

    def test_waits_below_threshold_during_same_week(self) -> None:
        trigger, reason = should_trigger(
            3,
            batch_size=10,
            max_age_days=7,
            last_report_at=datetime.now(timezone.utc).isoformat(),
        )
        self.assertFalse(trigger)
        self.assertIn("waiting", reason)

    def test_frustration_signal_triggers_immediately(self) -> None:
        trigger, reason = should_trigger(
            1,
            batch_size=10,
            max_age_days=7,
            last_report_at=datetime.now(timezone.utc).isoformat(),
            frustration_count=1,
        )
        self.assertTrue(trigger)
        self.assertIn("frustration signal", reason)

    def test_structural_metrics_keep_only_frustration_metadata(self) -> None:
        def fake_detector(event):
            signaled = "angry-example" in event["prompt"]
            return {
                "classification": "frustration_requires_analysis" if signaled else "no_signal",
                "signal": {
                    "kind": "insult_or_vulgarity" if signaled else "none",
                    "categories": ["profanity"] if signaled else [],
                    "count": 1 if signaled else 0,
                    "frustration_context": signaled,
                },
                "confidence": {"cause": {"status": "unknown"}},
            }

        with tempfile.TemporaryDirectory(prefix="cmc-frustration-test-") as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            rows = [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "angry-example private text"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "response"}],
                    },
                },
            ]
            rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
            metrics = structural_metrics(rollout, "fixture", detector=fake_detector)
            self.assertEqual(len(metrics["frustration_signals"]), 1)
            self.assertFalse(metrics["frustration_signals"][0]["fault_established"])
            self.assertNotIn("private text", repr(metrics))

    def test_report_requires_every_conversation_and_frustration_section(self) -> None:
        records = [{"session_id": "abcdefgh-1"}, {"session_id": "ijklmnop-2"}]
        incomplete = "abcdefgh\n## Incidents de frustration"
        self.assertEqual(
            missing_report_requirements(incomplete, records),
            ["ijklmnop"],
        )
        complete = "abcdefgh\nijklmnop\n## Incidents de frustration"
        self.assertEqual(missing_report_requirements(complete, records), [])

    def test_manifest_accepts_ingested_claude_and_ignores_non_ingested(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-manifest-test-") as tmp:
            root = Path(tmp)
            manifest = root / "collection-state.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sessions": {
                            "claude:one": {
                                "source": "claude",
                                "session_id": "one",
                                "path": str(root / "one.jsonl"),
                                "source_hash": "hash-one",
                                "project": "demo",
                                "status": "ingested",
                                "source_mtime": 1,
                            },
                            "codex:two": {
                                "source": "codex",
                                "session_id": "two",
                                "path": str(root / "two.jsonl"),
                                "source_hash": "hash-two",
                                "project": "demo",
                                "status": "failed",
                                "source_mtime": 2,
                            },
                        }
                    }
                )
            )
            records = _manifest_records(manifest)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source"], "claude")
            self.assertEqual(records[0]["source_hash"], "hash-one")

    def test_evidence_excludes_analysis_retains_error_tail_and_redacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-evidence-test-") as tmp:
            source = Path(tmp) / "session.jsonl"
            rows = [
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "request"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "analysis",
                        "content": [{"type": "output_text", "text": "private analysis"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "exec",
                        "call_id": "call-1",
                        "arguments": {"command": "echo token=hidden"},
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "prefix " + ("x" * 1000) + "\nERROR at tail",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            windows = extract_evidence_windows(
                source, {"source": "codex", "session_id": "evidence1"}, max_window_chars=300
            )
            text = "\n".join(window["text"] for window in windows)
            self.assertNotIn("private analysis", text)
            self.assertIn("ERROR at tail", text)
            self.assertNotIn("token=hidden", text)
            self.assertIn("call-1", text)

    def test_structured_validator_rejects_incident_success_contradiction(self) -> None:
        records = [{"source": "codex", "session_id": "s1"}]
        evidence = {"codex:s1": [{"ref": "ev-codex-s1-01", "text": "request"}]}
        report = {
            "schema_version": "1",
            "conversations": [
                {
                    "conversation_id": "codex:s1",
                    "subject": "x",
                    "level": "medium",
                    "status": "success",
                    "summary": "bad",
                    "incidents": ["incident-1"],
                    "skills": [],
                }
            ],
            "incidents": [
                {
                    "id": "incident-1",
                    "conversation_id": "codex:s1",
                    "type": "scope",
                    "expected": "answer",
                    "observed": "detour",
                    "cause": {"status": "verified", "summary": "evidence"},
                    "evidence_refs": ["ev-codex-s1-01"],
                    "recommendation": "simplify",
                    "test": "replay",
                }
            ],
            "successes": [],
            "limitations": [],
        }
        errors = validate_structured_report(report, records, evidence)
        self.assertTrue(any("status incident" in error for error in errors))

    def test_report_schema_requires_structured_audit_objects(self) -> None:
        schema = build_report_schema()
        schema_text = json.dumps(schema)
        self.assertNotIn("uniqueItems", schema_text)
        self.assertEqual(
            schema["required"],
            ["schema_version", "verdict", "conversations", "incidents", "successes", "limitations"],
        )
        self.assertEqual(
            schema["properties"]["incidents"]["items"]["required"],
            [
                "id",
                "conversation_id",
                "type",
                "expected",
                "observed",
                "cause",
                "evidence_refs",
                "recommendation",
                "test",
            ],
        )

    def test_incident_tracking_preserves_human_status_and_deduplicates(self) -> None:
        state = {
            "incidents": {
                "incident-1": {
                    "status": "applied",
                    "test_status": "passed",
                    "occurrences": [],
                }
            }
        }
        report = {
            "metadata": {"record_hashes": {"codex:s1": "hash-1"}},
            "incidents": [
                {
                    "id": "incident-1",
                    "conversation_id": "codex:s1",
                    "type": "scope",
                    "expected": "answer",
                    "observed": "detour",
                    "cause": {"status": "verified", "summary": "evidence"},
                    "evidence_refs": ["ev-codex-s1-01"],
                    "recommendation": "simplify",
                    "test": "replay",
                }
            ],
        }
        updated = update_incident_tracking(state, report)
        updated = update_incident_tracking(updated, report)
        tracked = updated["incidents"]["incident-1"]
        self.assertEqual(tracked["status"], "applied")
        self.assertEqual(tracked["test_status"], "passed")
        self.assertEqual(len(tracked["occurrences"]), 1)
        self.assertEqual(tracked["occurrence_count"], 1)

    def test_pending_batch_alternates_oldest_and_newest(self) -> None:
        pending = [{"session_id": f"s{i}", "sort_at": i} for i in range(6)]
        selected = select_pending_batch(pending, 4)
        self.assertEqual([item["session_id"] for item in selected], ["s0", "s5", "s1", "s4"])

    def test_completeness_is_independent_from_bounded_windows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-completeness-test-") as tmp:
            source = Path(tmp) / "session.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "session_meta", "payload": {"id": "fixture"}}),
                        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user"}}),
                        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "assistant"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            partial = observe_source_completeness(source)
            source.write_text(
                source.read_text(encoding="utf-8")
                + json.dumps({"type": "task_complete"})
                + "\n",
                encoding="utf-8",
            )
            complete = observe_source_completeness(source)
        self.assertEqual(partial["observation"], "partial")
        self.assertFalse(partial["terminal_evidence"])
        self.assertEqual(complete["observation"], "complete")
        self.assertTrue(complete["terminal_evidence"])
        self.assertGreaterEqual(complete["event_count"], partial["event_count"])

    def test_old_terminal_does_not_complete_a_new_open_turn(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-multiturn-completeness-") as tmp:
            source = Path(tmp) / "session.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "task_started"}),
                        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user"}}),
                        json.dumps({"type": "task_complete"}),
                        json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user"}}),
                        json.dumps({"type": "response_item", "payload": {"type": "function_call", "name": "exec"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            observed = observe_source_completeness(source)
        self.assertEqual(observed["observation"], "partial")
        self.assertFalse(observed["terminal_evidence"])
        self.assertGreater(observed["last_turn_boundary_line"], observed["last_terminal_line"])

    def test_incomplete_source_rejects_success_without_masking_observable_error(self) -> None:
        records = [{"source": "codex", "session_id": "s1", "_metrics": {"frustration_signals": []}}]
        evidence = {"codex:s1": [{"ref": "ev-codex-s1-01", "kind": "user_request", "text": "request"}]}
        incomplete = {"codex:s1": {"observation": "partial", "terminal_evidence": False}}
        success_report = {
            "schema_version": "1",
            "conversations": [{
                "conversation_id": "codex:s1", "subject": "x", "level": "none",
                "status": "success", "summary": "delivered", "incidents": [], "skills": [],
            }],
            "incidents": [], "successes": [], "limitations": [],
        }
        errors = validate_structured_report(success_report, records, evidence, incomplete)
        self.assertTrue(any("insufficient_evidence" in error for error in errors))

        error_evidence = {"codex:s1": [{"ref": "ev-codex-s1-01", "kind": "tool_error", "text": "ERROR observed"}]}
        incident_report = {
            "schema_version": "1",
            "conversations": [{
                "conversation_id": "codex:s1", "subject": "x", "level": "medium",
                "status": "incident", "summary": "tool failed", "incidents": ["i1"], "skills": [],
            }],
            "incidents": [{
                "id": "i1", "conversation_id": "codex:s1", "type": "tool",
                "expected": "answer", "observed": "ERROR", "cause": {"status": "verified", "summary": "tool"},
                "evidence_refs": ["ev-codex-s1-01"], "recommendation": "inspect", "test": "replay",
            }],
            "successes": [], "limitations": [],
        }
        self.assertFalse(
            any("incomplete source" in error for error in validate_structured_report(
                incident_report, records, error_evidence, incomplete
            ))
        )

        absence_report = {
            **incident_report,
            "conversations": [{
                "conversation_id": "codex:s1", "subject": "x", "level": "medium",
                "status": "incident", "summary": "aucun final", "incidents": ["i1"], "skills": [],
            }],
            "incidents": [{
                **incident_report["incidents"][0],
                "type": "livraison absente",
                "observed": "aucun final",
                "evidence_refs": ["ev-codex-s1-01"],
            }],
        }
        absence_errors = validate_structured_report(
            absence_report,
            records,
            {"codex:s1": [
                {"ref": "ev-codex-s1-01", "kind": "user_request", "text": "aucun final"},
                {"ref": "ev-codex-s1-02", "kind": "tool_error", "text": "ERROR unrelated"},
            ]},
            incomplete,
        )
        self.assertTrue(any("absence-of-delivery" in error for error in absence_errors))

    def test_record_dates_keep_unknown_ingestion_explicit(self) -> None:
        dates = record_date_metadata(
            {"sort_at": 1788602400, "ingested_at": "", "session_id": "s1"},
            "2026-09-05T12:00:00+02:00",
        )
        self.assertNotEqual(dates["source_date"], "unknown")
        self.assertEqual(dates["ingestion_date"], "unknown")
        self.assertEqual(dates["audit_date"], "2026-09-05")


if __name__ == "__main__":
    unittest.main()
