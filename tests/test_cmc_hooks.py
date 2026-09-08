from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"
sys.path.insert(0, str(HOOKS))

import ace_capture  # noqa: E402


class AceHookTests(unittest.TestCase):
    """Synthetic checks for capture-only hook safety and routing."""

    def setUp(self) -> None:
        self._claude_invoked_by = os.environ.pop("CLAUDE_INVOKED_BY", None)

    def tearDown(self) -> None:
        if self._claude_invoked_by is not None:
            os.environ["CLAUDE_INVOKED_BY"] = self._claude_invoked_by

    def _load_hook(self, filename: str):
        name = f"ace_test_{filename.replace('-', '_').replace('.', '_')}"
        spec = importlib.util.spec_from_file_location(name, HOOKS / filename)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_collector_command_uses_only_supported_ace_options_and_source_cwd(self) -> None:
        transcript = Path("/tmp/ACE transcript/session.jsonl")
        source_cwd = "/tmp/initialized-project"
        command = ace_capture.collector_command(transcript, cwd=source_cwd)

        self.assertEqual(
            command[command.index("--source") :],
            [
                "--source",
                "claude",
                "--path",
                str(transcript),
                "--cwd",
                source_cwd,
                "--limit",
                "1",
                "--extract",
            ],
        )
        self.assertNotIn("--stable-seconds", command)
        self.assertNotIn("--state-file", command)
        self.assertNotIn("--fallback-project", command)
        self.assertNotIn("--cwd", ace_capture.collector_command(transcript))

    def test_spawn_collector_is_detached_and_does_not_read_transcript(self) -> None:
        process = Mock()
        with patch.object(ace_capture.subprocess, "Popen", process):
            with tempfile.TemporaryDirectory() as tmp:
                transcript = Path(tmp) / "session.jsonl"
                transcript.write_text("user secret text\n", encoding="utf-8")
                self.assertTrue(
                    ace_capture.spawn_collector(transcript, cwd="/tmp/project")
                )
        process.assert_called_once()
        args, kwargs = process.call_args
        self.assertIn("--path", args[0])
        self.assertEqual(args[0][args[0].index("--path") + 1], str(transcript))
        self.assertEqual(args[0][args[0].index("--cwd") + 1], "/tmp/project")
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(kwargs["start_new_session"])

    def test_stop_throttle_uses_ace_success_record_then_launch_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session.jsonl"
            transcript.write_text("not parsed by the hook\n", encoding="utf-8")
            state_file = root / "collection-state.json"
            throttle_file = root / "stop-throttle.json"
            now = transcript.stat().st_mtime + 120
            state_file.write_text(
                json.dumps(
                    {
                        "sessions": {
                            "claude:session": {
                                "source": "claude",
                                "path": str(transcript),
                                "source_mtime": transcript.stat().st_mtime,
                                "status": "ingested",
                                "collected_at": datetime.fromtimestamp(
                                    now - 10, timezone.utc
                                ).isoformat(),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "ACE_COLLECTION_STATE_FILE": str(state_file),
                    "ACE_STOP_THROTTLE_FILE": str(throttle_file),
                },
                clear=False,
            ):
                self.assertTrue(ace_capture.stop_is_throttled(transcript, now=now))
                # After the 30-minute gate opens, the collector owns its
                # unchanged-source decision.  A launch timestamp closes the
                # hot-path gate again even if collection later fails.
                self.assertFalse(
                    ace_capture.stop_is_throttled(transcript, now=now + 1801)
                )
                ace_capture.mark_stop_launch(transcript, now=now + 1801)
                self.assertTrue(
                    ace_capture.stop_is_throttled(transcript, now=now + 1802)
                )

    def test_failed_or_missing_collection_does_not_repeat_stop_launches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "session.jsonl"
            transcript.write_text("content\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "ACE_COLLECTION_STATE_FILE": str(root / "missing-state.json"),
                    "ACE_STOP_THROTTLE_FILE": str(root / "stop-throttle.json"),
                },
                clear=False,
            ):
                now = transcript.stat().st_mtime + 10
                self.assertFalse(ace_capture.stop_is_throttled(transcript, now=now))
                ace_capture.mark_stop_launch(transcript, now=now)
                self.assertTrue(ace_capture.stop_is_throttled(transcript, now=now + 1))
                self.assertFalse(
                    ace_capture.stop_is_throttled(transcript, now=now + 1801)
                )

    def test_hooks_have_no_cursor_flush_or_second_summary_pipeline(self) -> None:
        for filename in (
            "session-end.py",
            "pre-compact.py",
            "stop-flush-checkpoint.py",
        ):
            source = (HOOKS / filename).read_text(encoding="utf-8")
            self.assertNotIn("checkpoint_cursor", source)
            self.assertNotIn("checkpoint-cursor", source)
            self.assertNotIn("flush.py", source)
            self.assertIn("ace_capture", source)
            self.assertNotIn("extract_native_summaries", source)
            self.assertNotIn("spawn_native_summaries", source)
            self.assertNotIn("fallback-project", source)
            self.assertNotIn("CMC_", source)

    def test_session_and_precompact_launch_only_the_canonical_collector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            payload = {
                "session_id": "session-123",
                "transcript_path": str(transcript),
                "cwd": "/tmp/initialized-project",
            }
            for filename in ("session-end.py", "pre-compact.py"):
                module = self._load_hook(filename)
                with (
                    patch.object(module, "spawn_collector", return_value=True) as collector,
                    patch.object(module.sys, "stdin", io.StringIO(json.dumps(payload))),
                ):
                    module.main()
                self.assertEqual(collector.call_args.args[0], transcript)
                self.assertEqual(
                    collector.call_args.kwargs.get("cwd"), "/tmp/initialized-project"
                )

    def test_stop_routes_exact_path_and_source_cwd_without_cursor_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            module = self._load_hook("stop-flush-checkpoint.py")
            with (
                patch.object(module, "stop_is_throttled", return_value=False),
                patch.object(module, "spawn_collector", return_value=True) as collector,
                patch.object(module, "mark_stop_launch") as mark,
                patch.object(
                    module.sys,
                    "stdin",
                    io.StringIO(
                        json.dumps(
                            {
                                "transcript_path": str(transcript),
                                "cwd": "/tmp/initialized-project",
                            }
                        )
                    ),
                ),
            ):
                module.main()
            self.assertEqual(collector.call_args.args[0], transcript)
            self.assertEqual(
                collector.call_args.kwargs.get("cwd"), "/tmp/initialized-project"
            )
            mark.assert_called_once_with(transcript)

    def test_stop_spawn_failure_remains_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            module = self._load_hook("stop-flush-checkpoint.py")
            with (
                patch.object(module, "stop_is_throttled", return_value=False),
                patch.object(module, "spawn_collector", return_value=False),
                patch.object(module, "mark_stop_launch") as mark,
                patch.object(
                    module.sys,
                    "stdin",
                    io.StringIO(
                        json.dumps(
                            {
                                "transcript_path": str(transcript),
                                "cwd": "/tmp/initialized-project",
                            }
                        )
                    ),
                ),
            ):
                module.main()
            mark.assert_not_called()

    def test_missing_source_cwd_skips_without_process_cwd_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            payload = json.dumps({"transcript_path": str(transcript)})
            for filename in ("session-end.py", "pre-compact.py", "stop-flush-checkpoint.py"):
                module = self._load_hook(filename)
                with (
                    patch.object(module, "spawn_collector") as collector,
                    patch.object(module.sys, "stdin", io.StringIO(payload)),
                ):
                    module.main()
                collector.assert_not_called()

    def test_hooks_skip_children_and_missing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            transcript.write_text("{}\n", encoding="utf-8")
            payload = json.dumps({"transcript_path": str(transcript)})
            for filename in (
                "session-end.py",
                "pre-compact.py",
                "stop-flush-checkpoint.py",
            ):
                path = HOOKS / filename
                result = subprocess.run(
                    [sys.executable, str(path)],
                    input=payload,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "CLAUDE_INVOKED_BY": "ace_test_child"},
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, filename)
                result = subprocess.run(
                    [sys.executable, str(path)],
                    input="{}",
                    text=True,
                    capture_output=True,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key != "CLAUDE_INVOKED_BY"
                    },
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, filename)


if __name__ == "__main__":
    unittest.main()
