from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


WORKER = Path("/Users/franck/.agents/bin/codex-ace-backfill")
NOTIFY = Path("/Users/franck/.agents/bin/codex-turn-ended")


class AceWorkerSelectionTests(unittest.TestCase):
    """The historical worker name now verifies the canonical ACE collector."""

    def _runtime_fixture(self, root: Path) -> tuple[Path, Path]:
        runtime = root / "runtime"
        (runtime / "scripts").mkdir(parents=True)
        (runtime / "scripts" / "ace_pipeline.py").write_text(
            "# synthetic runtime marker\n", encoding="utf-8"
        )
        fake_bin = root / "bin"
        fake_bin.mkdir()
        args_log = root / "ace-args.log"
        fake_uv = fake_bin / "uv"
        fake_uv.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$ACE_TEST_ARGS\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        fake_uv.chmod(0o700)
        return runtime, args_log

    def test_worker_is_a_simple_collect_wrapper_with_exact_scope(self) -> None:
        self.assertTrue(WORKER.is_file())
        with tempfile.TemporaryDirectory(prefix="ace-worker-collect-") as tmp:
            root = Path(tmp)
            runtime, args_log = self._runtime_fixture(root)
            transcript = root / "session.jsonl"
            transcript.write_text("synthetic transcript\n", encoding="utf-8")
            source_cwd = root / "initialized-project"
            source_cwd.mkdir()
            env = {
                **os.environ,
                "ACE_RUNTIME_DIR": str(runtime),
                "ACE_TEST_ARGS": str(args_log),
                "PATH": f"{root / 'bin'}:{os.environ.get('PATH', '')}",
            }
            result = subprocess.run(
                [
                    str(WORKER),
                    "--source",
                    "claude",
                    "--path",
                    str(transcript),
                    "--cwd",
                    str(source_cwd),
                    "--limit",
                    "1",
                ],
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            args = args_log.read_text(encoding="utf-8").splitlines()

        self.assertEqual(args[:4], ["run", "--frozen", "--directory", str(runtime)])
        self.assertIn("collect", args)
        self.assertEqual(args[args.index("--source") + 1], "claude")
        self.assertEqual(args[args.index("--path") + 1], str(transcript))
        self.assertEqual(args[args.index("--cwd") + 1], str(source_cwd))
        self.assertEqual(args[args.index("--limit") + 1], "1")
        self.assertNotIn("tick", args)
        self.assertNotIn("schedule", args)
        self.assertNotIn("--fallback-project", args)

    def test_turn_ended_notify_does_not_start_ace_collection(self) -> None:
        """A notification/disconnect event must not become a scheduler."""

        with tempfile.TemporaryDirectory(prefix="ace-notify-") as tmp:
            root = Path(tmp)
            marker = root / "unexpected-ace-call"
            result = subprocess.run(
                [str(NOTIFY), "turn-ended"],
                env={
                    **os.environ,
                    "ACE_RUNTIME_DIR": str(root / "missing-runtime"),
                    "ACE_TEST_ARGS": str(marker),
                    "CODEX_TURN_ENDED_FORWARD_SKY": "0",
                },
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
