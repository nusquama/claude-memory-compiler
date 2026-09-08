from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

_previous_claude_invoked_by = os.environ.get("CLAUDE_INVOKED_BY")

from backfill_codex import (  # noqa: E402
    CodexSession,
    extract_codex_session,
    upsert_daily_entry,
)
from checkpoint_cursor import (  # noqa: E402
    CURSOR_SCHEMA_VERSION,
    bounded_redacted_text,
    extract_incremental_slice,
    extract_turns_from_jsonl,
)
from flush import _build_single_pass_prompt  # noqa: E402

if _previous_claude_invoked_by is None:
    os.environ.pop("CLAUDE_INVOKED_BY", None)
else:
    os.environ["CLAUDE_INVOKED_BY"] = _previous_claude_invoked_by


class CmcEvidenceTests(unittest.TestCase):
    def test_bounded_evidence_redacts_and_keeps_tail_error(self) -> None:
        source = (
            "HEAD command: curl /health "
            + ("x" * 260)
            + '\nERROR: tail failure password="super-secret-value"\nTAIL marker'
        )

        evidence = bounded_redacted_text(source, 180)

        self.assertLessEqual(len(evidence), 180)
        self.assertIn("HEAD command", evidence)
        self.assertIn("ERROR: tail failure", evidence)
        self.assertIn("TAIL marker", evidence)
        self.assertNotIn("super-secret-value", evidence)
        self.assertIn("<REDACTED>", evidence)

    def test_codex_preserves_call_id_source_ref_and_tail_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-codex-evidence-") as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            rows = [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "session-evidence",
                        "timestamp": "2026-09-05T10:00:00Z",
                        "source": "fixture",
                    },
                },
                {
                    "type": "response_item",
                    "id": "response-call",
                    "payload": {
                        "type": "function_call",
                        "name": "exec",
                        "call_id": "call-evidence-1",
                        "arguments": {
                            "cmd": "run " + ("x" * 260),
                            "api_key": "secret-value-must-not-survive",
                        },
                    },
                },
                {
                    "type": "response_item",
                    "id": "response-result",
                    "payload": {
                        "type": "function_call_output",
                        "role": "tool",
                        "call_id": "call-evidence-1",
                        "output": (
                            ("output " + ("y" * 260))
                            + "\nERROR: command failed at tail"
                        ),
                    },
                },
            ]
            rollout.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            session = extract_codex_session(
                rollout,
                tool_arg_chars=180,
                tool_output_chars=180,
            )

        self.assertIsNotNone(session)
        assert session is not None
        self.assertIn("call_id=call-evidence-1", session.context)
        self.assertIn("source_ref=response-call", session.context)
        self.assertIn("source_ref=response-result", session.context)
        self.assertIn("ERROR: command failed at tail", session.context)
        self.assertNotIn("secret-value-must-not-survive", session.context)

    def test_codex_skips_internal_assistant_channel_and_keeps_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-codex-channel-") as tmp:
            rollout = Path(tmp) / "rollout.jsonl"
            rows = [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "session-channel",
                        "timestamp": "2026-09-05T10:00:00Z",
                        "source": "fixture",
                    },
                },
                {
                    "type": "turn_context",
                    "payload": {
                        "model": "gpt-source-fixture",
                        "reasoning_effort": "high",
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "channel": "analysis",
                        "content": [{"type": "output_text", "text": "private reasoning"}],
                    },
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "channel": "commentary",
                        "content": [{"type": "output_text", "text": "Visible result"}],
                    },
                },
            ]
            rollout.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            session = extract_codex_session(
                rollout,
                tool_arg_chars=200,
                tool_output_chars=200,
            )

        self.assertIsNotNone(session)
        assert session is not None
        self.assertNotIn("private reasoning", session.context)
        self.assertIn("Visible result", session.context)
        self.assertIn("Source Model:** gpt-source-fixture", session.context)
        self.assertIn("Source Reasoning Effort:** high", session.context)

    def test_claude_preserves_tool_result_and_refs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-claude-evidence-") as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            rows = [
                {
                    "uuid": "claude-assistant-1",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-use-1",
                                "name": "Bash",
                                "input": {
                                    "command": "echo " + ("x" * 220),
                                    "token": "claude-secret-value",
                                },
                            }
                        ],
                    },
                },
                {
                    "uuid": "claude-tool-result-1",
                    "parentUuid": "claude-assistant-1",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-use-1",
                                "is_error": True,
                                "content": (
                                    ("stdout " + ("z" * 220))
                                    + "\nERROR: observed command failure"
                                ),
                            }
                        ],
                    },
                },
            ]
            transcript.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            turns = extract_turns_from_jsonl(transcript)

        self.assertEqual(len(turns), 2)
        rendered = "\n".join(turns)
        self.assertIn("tool-use-1", rendered)
        self.assertIn("status=error", rendered)
        self.assertIn("ERROR: observed command failure", rendered)
        self.assertIn("source_ref=claude-tool-result-1", rendered)
        self.assertIn("parent_ref=claude-assistant-1", rendered)
        self.assertNotIn("claude-secret-value", rendered)

    def test_legacy_cursor_resets_before_new_tool_result_counting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-cursor-evidence-") as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            rows = [
                {
                    "uuid": "user-1",
                    "message": {"role": "user", "content": "Run it"},
                },
                {
                    "uuid": "assistant-1",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-1",
                                "name": "Bash",
                                "input": {"command": "true"},
                            }
                        ],
                    },
                },
                {
                    "uuid": "result-1",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-1",
                                "content": "ok",
                            }
                        ],
                    },
                },
            ]
            transcript.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            context, new_total, next_state = extract_incremental_slice(
                transcript,
                {
                    # Pre-v2 cursors counted only the first two messages.
                    "last_flush_ts": 10.0,
                    "last_main_turn_count": 2,
                    "last_subagent_counts": {},
                },
                10000,
            )

        self.assertIn("Tool result", context)
        self.assertEqual(new_total, 3)
        self.assertEqual(next_state["schema_version"], CURSOR_SCHEMA_VERSION)
        self.assertEqual(next_state["last_main_turn_count"], 3)

    def test_flush_prompt_separates_claims_observations_and_statuses(self) -> None:
        prompt = _build_single_pass_prompt(
            "**Tool Output role=tool status=error call_id=call-1 source_ref=r1:**\nERROR"
        )

        lowered = prompt.lower()
        self.assertIn("agent claim", lowered)
        self.assertIn("observed result", lowered)
        self.assertIn("source_ref", lowered)
        self.assertIn("call_id", lowered)
        self.assertIn("erreur", lowered)
        self.assertIn("source model", lowered)

    def test_codex_daily_upsert_handles_backslashes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-daily-evidence-") as tmp:
            project = Path(tmp) / "project"
            source = project / "rollout.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text("{}\n", encoding="utf-8")
            session = CodexSession(
                path=source,
                session_id="session-backslash",
                timestamp=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
                cwd=str(project),
                cli_version="fixture",
                originator="Codex",
                is_subagent=False,
                context="fixture",
                turn_count=1,
            )

            log_path = upsert_daily_entry(project, session, r"result with \1 intact")
            upsert_daily_entry(project, session, r"updated result with \2 intact")
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(content.count("ace-codex-session: session-backslash"), 1)
        self.assertIn(r"updated result with \2 intact", content)
        self.assertNotIn(r"result with \1 intact", content)

    def test_codex_daily_upsert_redacts_model_output_before_write(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-daily-redaction-") as tmp:
            project = Path(tmp) / "project"
            source = project / "rollout.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text("{}\n", encoding="utf-8")
            session = CodexSession(
                path=source,
                session_id="session-redaction",
                timestamp=datetime(2026, 9, 5, 10, 0, tzinfo=timezone.utc),
                cwd=str(project),
                cli_version="fixture",
                originator="Codex",
                is_subagent=False,
                context="fixture",
                turn_count=1,
            )

            log_path = upsert_daily_entry(
                project,
                session,
                'result Authorization: Bearer sk-live-abcdefghijklmnopqrstuvwxyz',
            )
            content = log_path.read_text(encoding="utf-8")

        self.assertNotIn("sk-live-abcdefghijklmnopqrstuvwxyz", content)
        self.assertIn("<REDACTED>", content)


if __name__ == "__main__":
    unittest.main()
