from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "hooks"

sys.path.insert(0, str(HOOKS))

import ace_capture  # noqa: E402


def _hook_state_environment(root: Path) -> dict[str, str]:
    return {
        "ACE_STOP_THROTTLE_FILE": str(root / "stop-throttle.json"),
        "ACE_STOP_THROTTLE_LOCK_FILE": str(root / "stop-throttle.lock"),
        "ACE_COLLECTION_STATE_FILE": str(root / "collection-state.json"),
    }


def test_concurrent_callbacks_start_one_child_for_one_snapshot(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("snapshot\n", encoding="utf-8")
    calls: list[list[str]] = []
    barrier = threading.Barrier(2)

    def fake_popen(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        time.sleep(0.05)
        return object()

    def launch() -> bool:
        barrier.wait()
        return ace_capture.spawn_collector(
            transcript,
            cwd="/tmp/initialized-project",
            popen=fake_popen,
            now=100.0,
        )

    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _item: launch(), (1, 2)))

    assert sorted(results) == [False, True]
    assert len(calls) == 1
    key = str(transcript.resolve())
    state = json.loads((tmp_path / "stop-throttle.json").read_text(encoding="utf-8"))
    assert state["launches"][key] == {
        "fingerprint": ace_capture._transcript_fingerprint(transcript),
        "launched_at": 100.0,
    }
    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        assert ace_capture.stop_is_throttled(transcript, now=101.0)


def test_changed_snapshot_gets_a_second_claim_without_reading_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("before\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_popen(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return object()

    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        assert ace_capture.spawn_collector(transcript, popen=fake_popen, now=100.0)
        transcript.write_text("before\nafter\n", encoding="utf-8")
        stat = transcript.stat()
        os.utime(transcript, ns=(stat.st_atime_ns, max(stat.st_mtime_ns, 101_000_000_000)))
        assert ace_capture.spawn_collector(transcript, popen=fake_popen, now=101.0)

    assert len(calls) == 2


def test_failed_popen_does_not_leave_a_dedupe_claim(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("retry\n", encoding="utf-8")
    attempts = 0

    def fake_popen(_command: list[str], **_kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("synthetic launch failure")
        return object()

    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        assert not ace_capture.spawn_collector(transcript, popen=fake_popen, now=100.0)
        assert ace_capture.spawn_collector(transcript, popen=fake_popen, now=101.0)

    assert attempts == 2


def test_concurrent_stop_marks_preserve_each_path(tmp_path: Path) -> None:
    transcripts = []
    for index in range(2):
        path = tmp_path / f"session-{index}.jsonl"
        path.write_text(f"session {index}\n", encoding="utf-8")
        transcripts.append(path)
    barrier = threading.Barrier(2)

    def mark(path: Path) -> None:
        barrier.wait()
        ace_capture.mark_stop_launch(path, now=100.0)

    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(mark, transcripts))

    state = json.loads((tmp_path / "stop-throttle.json").read_text(encoding="utf-8"))
    assert set(state["paths"]) == {str(path.resolve()) for path in transcripts}


def test_concurrent_independent_processes_start_one_child(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("snapshot\n", encoding="utf-8")
    child_log = tmp_path / "child.log"
    helper = tmp_path / "fake-ace"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['ACE_CHILD_LOG']).open('a', encoding='utf-8').write('child\\n')\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "import ace_capture\n"
        "print('true' if ace_capture.spawn_collector(\n"
        "    Path(sys.argv[1]), cwd='/tmp/project'\n"
        ") else 'false', flush=True)\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(_hook_state_environment(tmp_path))
    environment.update(
        {
            "ACE_BIN": str(helper),
            "ACE_CHILD_LOG": str(child_log),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    processes = [
        subprocess.Popen(
            [sys.executable, str(worker), str(transcript), str(HOOKS)],
            env=environment,
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        outputs.append(stdout.strip())

    assert sorted(outputs) == ["false", "true"]
    deadline = time.monotonic() + 2.0
    while not child_log.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_log.read_text(encoding="utf-8").splitlines() == ["child"]


def test_launch_ledger_expires_and_stays_bounded(tmp_path: Path) -> None:
    transcript = tmp_path / "new-session.jsonl"
    transcript.write_text("new\n", encoding="utf-8")
    now = 10_000.0
    launches = {
        str(tmp_path / f"old-{index}.jsonl"): {
            "fingerprint": {"size": index},
            "launched_at": (
                now - ace_capture.LAUNCH_LEDGER_RETENTION_SECONDS - 1
                if index == 0
                else now - index - 1
            ),
        }
        for index in range(ace_capture.LAUNCH_LEDGER_MAX_ENTRIES + 8)
    }
    state_path = tmp_path / "stop-throttle.json"
    state_path.write_text(json.dumps({"version": 1, "paths": {}, "launches": launches}))

    def fake_popen(_command: list[str], **_kwargs: object) -> object:
        return object()

    with patch.dict(os.environ, _hook_state_environment(tmp_path), clear=False):
        assert ace_capture.spawn_collector(transcript, popen=fake_popen, now=now)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(state["launches"]) <= ace_capture.LAUNCH_LEDGER_MAX_ENTRIES
    assert str(transcript.resolve()) in state["launches"]
    assert str(tmp_path / "old-0.jsonl") not in state["launches"]
