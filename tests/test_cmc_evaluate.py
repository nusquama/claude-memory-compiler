from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ace_evaluate import (  # noqa: E402
    CODEX_EXEC_PATH,
    MODEL_SPECS,
    VAULT_ROOT,
    InvocationResult,
    build_cases,
    build_codex_command,
    build_output_schema,
    main,
    parse_response,
    run_evaluation,
    safe_child_environment,
    score_response,
    select_specs,
    _safe_error_fields,
    write_report,
)


class CmcEvaluationHarnessTests(unittest.TestCase):
    def test_cases_are_small_labelled_and_have_stable_evidence_refs(self) -> None:
        cases = build_cases()
        self.assertEqual(len(cases), 7)
        self.assertEqual(
            {case.case_id for case in cases},
            {
                "surengineering_justified",
                "surengineering_unjustified",
                "frustration_actual_mismatch",
                "frustration_quoted_language",
                "failed_tool_recovered",
                "unverified_success",
                "clean_control",
            },
        )
        self.assertTrue(all(ref.startswith("ev-") for case in cases for ref in case.evidence_refs))
        self.assertEqual(sum(case.supported_finding for case in cases), 5)

    def test_only_explicit_model_effort_pairs_are_selectable(self) -> None:
        self.assertEqual(select_specs(), MODEL_SPECS)
        self.assertEqual(select_specs("gpt-5.6-luna"), MODEL_SPECS[:2])
        self.assertEqual(select_specs(effort="medium"), MODEL_SPECS[1:])
        with self.assertRaises(ValueError):
            select_specs("gpt-6-astra", "low")

    def test_child_command_is_ephemeral_read_only_and_non_recursive(self) -> None:
        command = build_codex_command(
            MODEL_SPECS[0],
            cwd=Path("/tmp/cmc-evaluation"),
            schema_path=Path("/tmp/schema.json"),
            output_path=Path("/tmp/output.json"),
            codex_exec="codex-test",
        )
        self.assertEqual(command[0:2], ["codex-test", "exec"])
        for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertIn("--model", command)
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn("model_reasoning_effort=low", command)

        env = safe_child_environment(
            {
                "PATH": "/bin",
                "HOME": "/tmp/home",
                "OPENAI_API_KEY": "must-not-pass",
                "SERVICE_TOKEN": "must-not-pass",
                "NORMAL_FLAG": "kept",
            }
        )
        # Auth is passed opaquely to the CLI; tests inspect only the key names
        # and synthetic values, never real credentials.
        self.assertEqual(env["OPENAI_API_KEY"], "must-not-pass")
        self.assertEqual(env["SERVICE_TOKEN"], "must-not-pass")
        self.assertEqual(env["NORMAL_FLAG"], "kept")
        self.assertEqual(env["CODEX_ACE_BACKFILL_ENABLED"], "0")
        self.assertEqual(env["ACE_EVALUATION"], "1")

    def test_api_schema_omits_unique_items_but_local_validation_remains_strict(self) -> None:
        schema_text = json.dumps(build_output_schema())
        self.assertNotIn("uniqueItems", schema_text)
        duplicate_refs = {
            "findings": [
                {
                    "case_id": "surengineering_justified",
                    "finding_type": "surengineering",
                    "classification": "justified",
                    "evidence_refs": ["ev-sj-1", "ev-sj-1"],
                }
            ]
        }
        self.assertFalse(parse_response(json.dumps(duplicate_refs), build_cases()).structured_valid)

    def test_error_fields_are_bounded_and_redacted(self) -> None:
        fields = _safe_error_fields(
            'ERROR: {"type":"invalid_request_error","code":"bad_schema",'
            '"message":"Authorization: Bearer should-not-leak"}'
        )
        self.assertEqual(fields["type"], "invalid_request_error")
        self.assertEqual(fields["code"], "bad_schema")
        self.assertNotIn("should-not-leak", repr(fields))
        nested = _safe_error_fields(
            'ERROR: {"type":"error","error":{"type":"invalid_request_error",'
            '"message":"nested failure"}}'
        )
        self.assertEqual(nested["type"], "invalid_request_error")
        self.assertEqual(nested["outer_type"], "error")
        trailing_error = ("noise\n" * 4000) + (
            'ERROR: {"type":"error","code":"late_code",'
            '"message":"late bounded message"}'
        )
        self.assertEqual(_safe_error_fields(trailing_error)["code"], "late_code")

    def test_exact_grounded_output_scores_perfectly(self) -> None:
        cases = build_cases()
        findings = [
            {
                "case_id": case.case_id,
                "finding_type": case.finding_type,
                "classification": case.expected_classification,
                "evidence_refs": list(case.evidence_refs),
            }
            for case in cases
            if case.finding_type != "none"
        ]
        parsed = parse_response(json.dumps({"findings": findings}), cases)
        metrics = score_response(parsed, cases, latency_ms=12.345)
        self.assertTrue(parsed.structured_valid)
        self.assertEqual(metrics["supported_findings"]["true_positive"], 5)
        self.assertEqual(metrics["supported_findings"]["false_positive"], 0)
        self.assertEqual(metrics["supported_findings"]["false_negative"], 0)
        self.assertEqual(metrics["supported_findings"]["precision"], 1.0)
        self.assertEqual(metrics["supported_findings"]["recall"], 1.0)
        self.assertEqual(metrics["classification_accuracy"], 1.0)
        self.assertEqual(metrics["latency_ms"], 12.35)
        self.assertIsNone(metrics["token_usage"])
        self.assertFalse(metrics["token_usage_measurable"])

    def test_false_positive_and_false_negative_are_counted(self) -> None:
        cases = build_cases()
        findings = [
            {
                "case_id": "frustration_quoted_language",
                "finding_type": "frustration",
                "classification": "actual_mismatch",
                "evidence_refs": ["ev-fq-1"],
            },
            {
                "case_id": "unverified_success",
                "finding_type": "verification",
                "classification": "verified",
                "evidence_refs": ["ev-uv-1"],
            },
        ]
        parsed = parse_response(json.dumps({"findings": findings}), cases)
        self.assertTrue(parsed.structured_valid)
        metrics = score_response(parsed, cases)
        supported = metrics["supported_findings"]
        self.assertEqual(supported["true_positive"], 0)
        # One negative-case false alarm plus one wrong positive classification
        # (which is both a missed gold finding and a false positive).
        self.assertEqual(supported["false_positive"], 2)
        self.assertEqual(supported["false_negative"], 5)
        self.assertEqual(supported["precision"], 0.0)
        self.assertEqual(supported["recall"], 0.0)

    def test_strict_json_and_case_aware_validation(self) -> None:
        cases = build_cases()
        malformed = parse_response("```json\n{\"findings\": []}\n```", cases)
        self.assertFalse(malformed.structured_valid)
        wrong_ref = {
            "findings": [
                {
                    "case_id": "surengineering_justified",
                    "finding_type": "surengineering",
                    "classification": "justified",
                    "evidence_refs": ["ev-not-listed-1"],
                }
            ]
        }
        invalid_ref = parse_response(json.dumps(wrong_ref), cases)
        self.assertFalse(invalid_ref.structured_valid)
        self.assertEqual(build_output_schema()["required"], ["findings"])

    def test_process_failure_does_not_claim_model_quality(self) -> None:
        parsed = parse_response(None, build_cases())
        metrics = score_response(parsed, build_cases(), latency_ms=25.0)
        self.assertIsNone(metrics["classification_accuracy"])
        self.assertIsNone(metrics["supported_findings"]["precision"])
        self.assertIsNone(metrics["supported_findings"]["recall"])
        self.assertEqual(metrics["structured_validity"], 0.0)

    def test_default_cli_is_dry_run_and_makes_no_subprocess_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cmc-evaluate-test-") as tmp, patch(
            "ace_evaluate.subprocess.run", side_effect=AssertionError("dry-run invoked subprocess")
        ), contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(["--output-dir", tmp])
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(result, 0)
        self.assertIn("dry-run", output.getvalue())

    def test_report_writer_rejects_vault_paths_and_keeps_private_mode(self) -> None:
        with self.assertRaises(ValueError):
            write_report({"test": True}, VAULT_ROOT / "reports")
        with tempfile.TemporaryDirectory(prefix="cmc-report-test-") as tmp:
            path = write_report({"test": True}, Path(tmp))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"test": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_real_or_injected_run_is_bounded_and_report_is_transcript_free(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_invoker(spec, prompt, **_kwargs):
            calls.append((spec.name, prompt))
            findings = [
                {
                    "case_id": case.case_id,
                    "finding_type": case.finding_type,
                    "classification": case.expected_classification,
                    "evidence_refs": list(case.evidence_refs),
                }
                for case in build_cases()
                if case.finding_type != "none"
            ]
            return InvocationResult(json.dumps({"findings": findings}), 4.0)

        with tempfile.TemporaryDirectory(prefix="cmc-evaluate-report-") as tmp:
            report_path = run_evaluation(MODEL_SPECS, output_dir=Path(tmp), invoker=fake_invoker)
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 3)
        self.assertEqual(len({prompt for _, prompt in calls}), 1)
        self.assertEqual(report["experiment"]["calls"], 3)
        self.assertEqual(report["experiment"]["codex_exec"], CODEX_EXEC_PATH)
        self.assertEqual([run["status"] for run in report["runs"]], ["ok", "ok", "ok"])
        self.assertEqual(report["runs"][0]["metrics"]["supported_findings"]["recall"], 1.0)
        self.assertNotIn("Observation:", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
