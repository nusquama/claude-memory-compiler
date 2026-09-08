from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_learning as learning  # noqa: E402


def snapshot(
    session_id: str = "session-a",
    *,
    terminal: bool = True,
    messages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = messages or [
        {
            "id": "m-1",
            "ordinal": 1,
            "role": "user",
            "type": "message",
            "timestamp": "2026-09-07T10:00:00Z",
            "content": "Je veux seulement le fichier demandé, sans refonte.",
        },
        {
            "id": "m-2",
            "ordinal": 2,
            "role": "assistant",
            "type": "message",
            "content": "Done, je vais vérifier le résultat.",
        },
    ]
    if terminal:
        rows.append(
            {
                "id": "m-3",
                "ordinal": 3,
                "role": "assistant",
                "type": "task_complete",
                "status": "completed",
                "content": "task complete",
            }
        )
    return {
        "project": {"id": "project-a", "name": "Fixture", "root": "/private/root", "vault_dir": "/private/vault"},
        "source": "codex",
        "session_id": session_id,
        "revision": "rev-1",
        "started_at": "2026-09-07T10:00:00Z",
        "updated_at": "2026-09-07T10:01:00Z",
        "messages": rows,
        "attachments": [],
    }


class AceSnapshotEvidenceTests(unittest.TestCase):
    def test_evidence_resolves_to_snapshot_messages_without_paths(self) -> None:
        source = snapshot()
        windows = learning.build_evidence_windows(source)
        ids = {message["id"] for message in source["messages"]}
        self.assertTrue(windows)
        for window in windows:
            self.assertTrue(set(window["message_ids"]).issubset(ids))
            self.assertTrue(all(ref.startswith("snapshot:session-a:message:") for ref in window["message_refs"]))
            self.assertNotIn("/private/root", window["text"])

    def test_supabase_values_are_redacted_before_model_context(self) -> None:
        source = snapshot(
            messages=[
                {
                    "id": "m-secret",
                    "ordinal": 1,
                    "role": "user",
                    "type": "message",
                    "content": {"SUPABASE_SERVICE_ROLE_KEY": "super-secret-value", "text": "request"},
                }
            ],
            terminal=False,
        )
        normalized = learning.normalize_snapshot(source)
        self.assertNotIn("super-secret-value", json.dumps(normalized, ensure_ascii=False))
        self.assertNotIn("super-secret-value", learning.build_snapshot_prompt([source]))

    def test_old_terminal_marker_does_not_complete_new_open_turn(self) -> None:
        source = snapshot(
            terminal=False,
            messages=[
                {"id": "u1", "ordinal": 1, "role": "user", "type": "message", "content": "first"},
                {"id": "t1", "ordinal": 2, "role": "assistant", "type": "task_complete", "content": "complete"},
                {"id": "u2", "ordinal": 3, "role": "user", "type": "message", "content": "new request"},
            ],
        )
        record = learning.snapshot_to_record(source)
        self.assertFalse(record["_completeness"]["terminal_evidence"])

    def test_runtime_preamble_does_not_displace_real_frustration_window(self) -> None:
        source = snapshot(
            terminal=False,
            messages=[
                {
                    "id": "runtime-developer",
                    "ordinal": 1,
                    "role": "developer",
                    "type": "message",
                    "content": "# AGENTS.md instructions\n" + ("configuration " * 500),
                },
                {
                    "id": "runtime-system",
                    "ordinal": 2,
                    "role": "system",
                    "type": "unknown_event",
                    "content": {"full": "<environment_context>" + ("config " * 500)},
                },
                {
                    "id": "real-user",
                    "ordinal": 3,
                    "role": "user",
                    "type": "message",
                    "content": "Non, ce n'est pas ce que j'ai demandé.",
                },
                {
                    "id": "real-assistant",
                    "ordinal": 4,
                    "role": "assistant",
                    "type": "message",
                    "content": "Je vérifie le résultat.",
                },
            ],
        )
        windows = learning.build_evidence_windows(source)
        frustration = next(item for item in windows if item["kind"] == "calm_rejection")
        self.assertIn("real-user", frustration["message_ids"])
        self.assertNotIn("runtime-developer", frustration["message_ids"])
        self.assertNotIn("runtime-system", frustration["message_ids"])
        self.assertIn("real-user", frustration["text"])


class AceAuditIntegrationTests(unittest.TestCase):
    def test_incomplete_snapshot_cannot_claim_success(self) -> None:
        async def fake_runner(records, prompt):
            return {
                "schema_version": "1",
                "verdict": "success",
                "conversations": [
                    {
                        "conversation_id": "codex:session-a",
                        "subject": "x",
                        "level": "none",
                        "status": "success",
                        "summary": "claimed",
                        "incidents": [],
                        "skills": [],
                    }
                ],
                "incidents": [],
                "successes": [],
                "limitations": [],
            }

        result = asyncio.run(learning.audit_snapshots([snapshot(terminal=False)], audit_runner=fake_runner))
        self.assertNotEqual(result["conversations"][0]["status"], "success")
        self.assertEqual(result["conversations"][0]["status"], "insufficient_evidence")

    def test_store_is_called_only_after_batch_audit(self) -> None:
        class Store:
            def __init__(self):
                self.rows = []

            async def save_analysis(self, analysis):
                self.rows.append(analysis)

        store = Store()
        result = asyncio.run(learning.audit_snapshots([snapshot()], store=store))
        self.assertEqual(len(store.rows), 1)
        self.assertEqual(store.rows[0]["session_id"], "session-a")
        self.assertEqual(len(result["reports"]), 1)

    def test_calm_rejection_is_observation_with_unknown_cause(self) -> None:
        source = snapshot(
            terminal=False,
            messages=[
                {"id": "u1", "ordinal": 1, "role": "user", "type": "message", "content": "Non, ce n'est pas ce que j'ai demandé."},
                {"id": "a1", "ordinal": 2, "role": "assistant", "type": "message", "content": "Je comprends."},
            ],
        )
        report = learning._heuristic_report(learning.snapshot_to_record(source))
        self.assertTrue(any(item["type"] == "frustration_mismatch" for item in report["incidents"]))
        self.assertEqual(report["incidents"][0]["cause"]["status"], "not_established")


class AceRecurrenceAndProposalTests(unittest.TestCase):
    def test_recurrence_requires_exact_scope_signature_and_three_sessions(self) -> None:
        rows = [
            {"session_id": "a", "scope": "p", "type": "tool_failure", "signature": "read", "evidence_refs": ["a1"]},
            {"session_id": "b", "scope": "p", "type": "tool_failure", "signature": "read", "evidence_refs": ["b1"]},
            {"session_id": "c", "scope": "p", "type": "tool_failure", "signature": "write", "evidence_refs": ["c1"]},
            {"session_id": "d", "scope": "p", "type": "tool_failure", "signature": "read", "evidence_refs": ["d1"]},
        ]
        patterns = learning.detect_recurrence(rows)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["sessions"], ["a", "b", "d"])
        self.assertFalse(patterns[0]["semantic_merge_proved"])

    def test_proposal_and_evaluations_are_append_only_and_three_failures_change_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = {"scope": "p", "type": "tool_failure", "signature": "read", "text": "inspect"}
            learning.record_proposal(root, proposal)
            for _ in range(3):
                learning.record_evaluation(root, {**proposal, "outcome": "failed"})
            rows = learning._load_jsonl(root / "evaluation-history.jsonl")
            followups = learning.followup_recommendations(rows)
            self.assertEqual(len(rows), 3)
            self.assertTrue(any(item["type"] == "changed_diagnosis" for item in followups))

    def test_suggestions_only_target_existing_rules_or_skills(self) -> None:
        suggestions = [
            {"skill": "known-skill", "text": "review"},
            {"skill": "not-installed", "text": "unknown"},
        ]
        filtered = learning.filter_suggestions(suggestions, existing_skills=["known-skill"])
        self.assertEqual(len(filtered), 1)
        self.assertFalse(filtered[0]["auto_apply"])
        self.assertTrue(filtered[0]["requires_authorization"])


class AceReportTests(unittest.TestCase):
    def test_render_reports_keeps_date_dimensions_usage_and_statuses_distinct(self) -> None:
        payload = {
            "records": [
                {
                    "session_id": "s1",
                    "source_date": "2026-09-07",
                    "ingestion_date": "2026-09-08",
                    "audit_date": "2026-09-09",
                    "coverage": {"completeness": {"observation": "partial"}},
                    "usage": None,
                    "accepted": True,
                    "applied": None,
                    "effective": None,
                    "decisions": ["keep bounded"],
                    "refusals": ["no auto-apply"],
                    "incidents": [{"type": "gap", "priority": "high", "risk": "unknown"}],
                    "recommendations": [],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            outputs = learning.render_reports(payload, tmp, now="2026-09-09T12:00:00Z")
            report = json.loads((Path(tmp) / "daily-2026-09-09.json").read_text())
            self.assertTrue(outputs["daily"].is_file())
            self.assertIn("2026-09-07", report["source_dates"])
            self.assertIn("2026-09-08", report["ingestion_dates"])
            self.assertNotEqual(report["source_dates"], report["ingestion_dates"])
            self.assertEqual(report["usage"]["status"], "unknown")
            self.assertIn("decisions", report)
            self.assertIn("refusals", report)
            self.assertIn("accepted", report["status_counts"])
            self.assertIn("applied", report["status_counts"])
            self.assertIn("effective", report["status_counts"])


if __name__ == "__main__":
    unittest.main()
