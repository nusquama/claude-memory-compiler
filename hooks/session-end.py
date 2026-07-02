"""
SessionEnd hook - captures conversation transcript for memory extraction.

When a Claude Code session ends, this hook reads the transcript path from
stdin, extracts the INCREMENTAL slice since the last cursor position, and
spawns flush.py as a background process to extract knowledge into the
daily log.

The hook itself does NO API calls - only local file I/O for speed (<10s).

Cursor coordination: shares state with hooks/stop-flush-checkpoint.py and
hooks/pre-compact.py via .state/checkpoint-cursor.json. Each hook slices
forward from the same cursor and advances it on spawn — never re-extracts
content already captured. If no checkpoints fired during the session
(short conversation, or feature was just enabled), the cursor is fresh
(turn count = 0) and SessionEnd extracts the full transcript.
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

# Recursion guard: if we were spawned by flush.py (which calls Agent SDK,
# which runs Claude Code, which would fire this hook again), exit immediately.
if os.environ.get("CLAUDE_INVOKED_BY"):
    sys.exit(0)

# Make scripts/ importable when this hook is invoked from anywhere.
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

# No project (e.g. session opened outside any git repo) → exit silently.
if PROJECT_DIR is None:
    sys.exit(0)

STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(FLUSH_LOG),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [hook] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MIN_TURNS_TO_FLUSH = 1
CURSOR_FILE = STATE_DIR / "checkpoint-cursor.json"


def main() -> None:
    # Read hook input from stdin.
    # Claude Code on Windows may pass paths with unescaped backslashes.
    try:
        raw_input = sys.stdin.read()
        try:
            hook_input: dict = json.loads(raw_input)
        except json.JSONDecodeError:
            fixed_input = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw_input)
            hook_input = json.loads(fixed_input)
    except (json.JSONDecodeError, ValueError, EOFError) as e:
        logging.error("Failed to parse stdin: %s", e)
        return

    session_id = hook_input.get("session_id", "unknown")
    source = hook_input.get("source", "unknown")
    transcript_path_str = hook_input.get("transcript_path", "")

    logging.info("SessionEnd fired: session=%s source=%s", session_id, source)

    if not transcript_path_str or not isinstance(transcript_path_str, str):
        logging.info("SKIP: no transcript path")
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        logging.info("SKIP: transcript missing: %s", transcript_path_str)
        return

    cursor = load_cursor(CURSOR_FILE)
    state = get_session_state(cursor, session_id)

    # Extract only the incremental slice since the last cursor position.
    try:
        context, new_turns, next_state = extract_incremental_slice(
            transcript_path, state, FLUSH_MAX_CHARS
        )
    except Exception as e:
        logging.error("Slice extraction failed: %s", e)
        return

    if new_turns < MIN_TURNS_TO_FLUSH or not context.strip():
        logging.info(
            "SKIP: no new turns since last cursor (session %s, last_main=%d)",
            session_id, state["last_main_turn_count"],
        )
        # Still trigger native-summary capture below — that path is
        # independent and handles its own dedup.
    else:
        # Advance cursor BEFORE spawning flush.py so a duplicate SessionEnd
        # firing (global + project-local hook configs both registered) reads
        # the advanced cursor and skips with an empty slice.
        next_state["last_flush_ts"] = time.time()
        cursor[session_id] = next_state
        cursor = gc_old_entries(cursor, time.time())
        save_cursor(CURSOR_FILE, cursor)

        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        context_file = STATE_DIR / f"session-flush-{session_id}-{timestamp}.md"
        context_file.write_text(context, encoding="utf-8")

        flush_script = SCRIPTS_DIR / "flush.py"
        cmd = [
            "uv", "run", "--directory", str(ROOT),
            "python", str(flush_script),
            str(context_file), session_id,
            "--label", "final",
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
                "Spawned flush.py for session %s (%d new turns, %d chars)",
                session_id, new_turns, len(context),
            )
        except Exception as e:
            logging.error("Failed to spawn flush.py: %s", e)

    # Independent of the slice flush: capture Claude Code's native /compact
    # summaries from the transcript (idempotent — tracks already-emitted
    # summary UUIDs in its own state).
    extract_native_script = SCRIPTS_DIR / "extract_native_summaries.py"
    if extract_native_script.exists():
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        native_cmd = [
            "uv", "run", "--directory", str(ROOT),
            "python", str(extract_native_script),
            str(transcript_path), session_id,
        ]
        try:
            subprocess.Popen(
                native_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
            logging.info(
                "Spawned extract_native_summaries.py for session %s",
                session_id,
            )
        except Exception as e:
            logging.error("Failed to spawn extract_native_summaries.py: %s", e)


if __name__ == "__main__":
    main()
