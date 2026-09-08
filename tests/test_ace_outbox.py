from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ace_outbox import Outbox  # noqa: E402


PROJECT_ID = "123e4567-e89b-12d3-a456-426614174000"


def _envelope(session_id: str, revision: str, *, project_id: str = PROJECT_ID) -> dict:
    return {
        "schema_version": 1,
        "project": {
            "id": project_id,
            "name": "ACE test",
            "root": "/tmp/ace-test",
            "vault_dir": "/tmp/ace-test-vault",
        },
        "source": "codex",
        "session_id": session_id,
        "revision": revision,
        "source_path": f"/tmp/{session_id}.jsonl",
        "messages": [],
        "attachments": [],
    }


def test_rotating_window_does_not_starve_middle_rows_under_continuous_arrivals(tmp_path: Path) -> None:
    queue = Outbox(tmp_path / "outbox.sqlite3", lease_seconds=0, max_lot_items=2)
    initial = [
        queue.enqueue(_envelope(f"initial-{index}", f"{index + 1:064x}"))
        for index in range(6)
    ]

    seen: list[str] = []
    for index in range(8):
        # A new row arrives before every claim.  A newest-plus-oldest
        # selector would repeatedly leave the middle of the initial batch.
        queue.enqueue(_envelope(f"arrival-{index}", f"{100 + index:064x}"))
        claimed = queue.pending(limit=2)
        seen.extend(item.key for item in claimed)
        for item in claimed:
            queue.fail(item.key, "synthetic retry", retry_at=0)

    assert {str(key) for key in initial}.issubset(seen)
    queue.close()


def test_fairness_cursor_is_stable_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "outbox.sqlite3"
    queue = Outbox(path, lease_seconds=0, max_lot_items=1)
    keys = [
        queue.enqueue(_envelope(f"session-{index}", f"{index + 1:064x}"))
        for index in range(4)
    ]
    first = queue.pending(limit=1)[0]
    queue.fail(first.key, "retry", retry_at=0)
    queue.close()

    restarted = Outbox(path, lease_seconds=0, max_lot_items=1)
    second = restarted.pending(limit=1)[0]
    assert second.key != first.key
    restarted.close()
