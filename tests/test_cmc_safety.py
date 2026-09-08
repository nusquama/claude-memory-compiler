from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from backfill_codex import is_subagent_source, rollout_is_subagent  # noqa: E402
from utils import (  # noqa: E402
    redact_sensitive_text,
    safe_codex_diagnostic_lines,
    sensitive_text_findings,
)
import ace_pipeline  # noqa: E402


SLACK_SHAPED = "xoxb-" + "1234567890-abcdefghijklmnop"  # assembled at runtime; not a stored token


class CmcSafetyTests(unittest.TestCase):
    def test_redacts_common_secret_forms(self) -> None:
        source = "\n".join(
            [
                "Authorization: Bearer sk-live-abcdefghijklmnopqrstuvwxyz",
                '"api_key": "top-secret-value"',
                "password=hunter-example",
                "token=" + SLACK_SHAPED,
                "url=https://example.test/hook?token=url-secret&mode=fast",
                "-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----",
            ]
        )
        redacted = redact_sensitive_text(source)
        for secret in (
            "sk-live-abcdefghijklmnopqrstuvwxyz",
            "top-secret-value",
            "hunter-example",
            SLACK_SHAPED,
            "url-secret",
            "private-material",
        ):
            self.assertNotIn(secret, redacted)
        self.assertGreaterEqual(redacted.count("<REDACTED>"), 6)

    def test_keeps_normal_identifiers(self) -> None:
        source = "task_id=1218090961433023 model=gpt-5.6-luna status=active"
        self.assertEqual(redact_sensitive_text(source), source)

    def test_secret_audit_reports_categories_without_values(self) -> None:
        findings = sensitive_text_findings(
            '"api_key": "top-secret-value"\nAuthorization: Bearer sample-secret'
        )
        self.assertGreaterEqual(findings.get("quoted_credential", 0), 1)
        self.assertGreaterEqual(findings.get("plain_credential", 0), 1)
        self.assertNotIn("top-secret-value", repr(findings))

    def test_codex_diagnostics_drop_echoed_content(self) -> None:
        stderr = "\n".join(
            [
                "2026-09-04T10:00:00Z WARN runtime state discrepancy",
                "user",
                "Authorization: Bearer should-not-reach-log",
                "assistant",
                "A long extracted response",
            ]
        )
        diagnostics = safe_codex_diagnostic_lines(stderr)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("runtime state discrepancy", diagnostics[0])
        self.assertNotIn("should-not-reach-log", repr(diagnostics))

    def test_codex_diagnostics_deduplicate_timestamped_warnings(self) -> None:
        stderr = "\n".join(
            [
                "2026-09-04T10:00:00Z WARN runtime state discrepancy",
                "2026-09-04T10:00:01Z WARN runtime state discrepancy",
            ]
        )
        self.assertEqual(len(safe_codex_diagnostic_lines(stderr)), 1)

    def test_detects_codex_subagent_sources(self) -> None:
        self.assertTrue(is_subagent_source({"subagent": {"depth": 1}}))
        self.assertTrue(is_subagent_source({"thread_spawn": {"depth": 1}}))
        self.assertFalse(is_subagent_source("vscode"))
        self.assertFalse(is_subagent_source(None))

    def test_detects_subagent_without_parsing_full_rollout(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory(prefix="cmc-subagent-test-") as tmp:
            path = Path(tmp) / "rollout.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"source": {"thread_spawn": {"depth": 1}}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(rollout_is_subagent(path))

    def test_ace_collect_has_explicit_scope_and_no_legacy_fallback_flag(self) -> None:
        parser = ace_pipeline.build_parser()
        parsed = parser.parse_args(
            [
                "collect",
                "--source",
                "claude",
                "--path",
                "/synthetic/session.jsonl",
                "--cwd",
                "/synthetic/initialized-project",
                "--limit",
                "1",
            ]
        )
        self.assertEqual(parsed.command, "collect")
        self.assertEqual(parsed.source, "claude")
        self.assertEqual(parsed.paths, ["/synthetic/session.jsonl"])
        self.assertEqual(parsed.cwd, "/synthetic/initialized-project")
        self.assertEqual(parsed.limit, 1)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "collect",
                    "--source",
                    "claude",
                    "--fallback-project",
                    "Conversations",
                ]
            )


if __name__ == "__main__":
    unittest.main()
