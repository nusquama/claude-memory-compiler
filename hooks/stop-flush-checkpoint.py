"""
Stop hook — periodic flush checkpoint (incremental).

Fires after every assistant response. If at least CHECKPOINT_INTERVAL_SECONDS
have elapsed since the last flush of this session AND new turns have been
written to the transcript since then, spawns flush.py with ONLY the new
slice (turns since last cursor position). Never re-extracts already-captured
content.

Goal: capture the conversation continuously while Claude Code is open, so a
crash, force-quit, or any failure to fire SessionEnd does not lose more than
~30 minutes of conversation.

Cost on the hot path: ~5–10 ms when the elapsed gate trips early. The full
extraction + slice happens only when an actual checkpoint fires.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Recursion guard: spawned flush.py runs Claude Agent SDK which fires Stop.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (  # noqa: E402
    FLUSH_LOG,
    FLUSH_MAX_CHARS,
    PROJECT_DIR,
    SCRIPTS_DIR,
    STATE_DIR,
    TOOL_DIR as ROOT,
)
from checkpoint_cursor import (  # noqa: E402
    extract_incremental_slice,
    gc_old_entries,
    get_session_state,
    load_cursor,
    save_cursor,
)

if PROJECT_DIR is None:
    sys.exit(0)

STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(FLUSH_LOG),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [checkpoint] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CHECKPOINT_INTERVAL_SECONDS = int(
    os.environ.get("CMC_CHECKPOINT_INTERVAL_SECONDS", "1800")
)
MIN_NEW_TURNS = int(os.environ.get("CMC_CHECKPOINT_MIN_NEW_TURNS", "1"))
CURSOR_FILE = STATE_DIR / "checkpoint-cursor.json"


def main() -> None:
    try:
        raw = sys.stdin.read()
        try:
            hook_input = json.loads(raw)
        except json.JSONDecodeError:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw)
            hook_input = json.loads(fixed)
    except (json.JSONDecodeError, ValueError, EOFError):
        return

    session_id = hook_input.get("session_id", "")
    transcript_path_str = hook_input.get("transcript_path", "")
    if not session_id or not transcript_path_str:
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        return

    cursor = load_cursor(CURSOR_FILE)
    state = get_session_state(cursor, session_id)

    now = time.time()
    elapsed = now - state["last_flush_ts"]

    # Hot-path bail: most Stop fires don't trigger a checkpoint.
    if elapsed < CHECKPOINT_INTERVAL_SECONDS:
        return

    try:
        context, new_turns, next_state = extract_incremental_slice(
            transcript_path, state, FLUSH_MAX_CHARS
        )
    except Exception as e:
        logging.error("Slice extraction failed for session %s: %s", session_id, e)
        return

    if new_turns < MIN_NEW_TURNS or not context.strip():
        return

    # Stamp + persist new cursor state BEFORE spawning flush.py. This way,
    # a fast-firing duplicate Stop hook (e.g., global+project-local both
    # configured) reads the advanced cursor and skips, instead of double-
    # extracting the same slice.
    next_state["last_flush_ts"] = now
    cursor[session_id] = next_state
    cursor = gc_old_entries(cursor, now)
    save_cursor(CURSOR_FILE, cursor)

    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    context_file = STATE_DIR / f"checkpoint-{session_id}-{timestamp}.md"
    try:
        context_file.write_text(context, encoding="utf-8")
    except OSError as e:
        logging.error("Failed to write context file: %s", e)
        return

    flush_script = SCRIPTS_DIR / "flush.py"
    cmd = [
        "uv", "run", "--directory", str(ROOT),
        "python", str(flush_script),
        str(context_file), session_id,
        "--label", "checkpoint",
    ]
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        logging.info(
            "Checkpoint fired for session %s (elapsed %.0fs, %d new turns, %d chars)",
            session_id, elapsed, new_turns, len(context),
        )
    except Exception as e:
        logging.error("Failed to spawn flush.py from checkpoint: %s", e)


if __name__ == "__main__":
    main()
