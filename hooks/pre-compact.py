"""
PreCompact hook - captures conversation transcript before auto-compaction.

When Claude Code's context window fills up, it auto-compacts (summarizes
and discards detail). This hook fires BEFORE that happens, slicing the
INCREMENTAL conversation context since the last cursor position and
spawning flush.py to extract knowledge that would otherwise be lost to
summarization.

The hook itself does NO API calls - only local file I/O for speed (<10s).

Cursor coordination: shares state with hooks/stop-flush-checkpoint.py and
hooks/session-end.py via .state/checkpoint-cursor.json. Slice forward,
advance on spawn — never re-extract content already captured.
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

# Recursion guard
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
    format="%(asctime)s %(levelname)s [pre-compact] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Pre-compact fires when Claude Code is about to summarize away detail.
# Lower threshold than periodic checkpoint: even a small slice is worth
# saving since the alternative is loss to summarization.
MIN_TURNS_TO_FLUSH = 5
CURSOR_FILE = STATE_DIR / "checkpoint-cursor.json"


def main() -> None:
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
    transcript_path_str = hook_input.get("transcript_path", "")

    logging.info("PreCompact fired: session=%s", session_id)

    # transcript_path can be empty (known Claude Code bug #13668)
    if not transcript_path_str or not isinstance(transcript_path_str, str):
        logging.info("SKIP: no transcript path")
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        logging.info("SKIP: transcript missing: %s", transcript_path_str)
        return

    cursor = load_cursor(CURSOR_FILE)
    state = get_session_state(cursor, session_id)

    try:
        context, new_turns, next_state = extract_incremental_slice(
            transcript_path, state, FLUSH_MAX_CHARS
        )
    except Exception as e:
        logging.error("Slice extraction failed: %s", e)
        return

    if new_turns >= MIN_TURNS_TO_FLUSH and context.strip():
        next_state["last_flush_ts"] = time.time()
        cursor[session_id] = next_state
        cursor = gc_old_entries(cursor, time.time())
        save_cursor(CURSOR_FILE, cursor)

        timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        context_file = STATE_DIR / f"flush-context-{session_id}-{timestamp}.md"
        context_file.write_text(context, encoding="utf-8")

        flush_script = SCRIPTS_DIR / "flush.py"
        cmd = [
            "uv", "run", "--directory", str(ROOT),
            "python", str(flush_script),
            str(context_file), session_id,
            "--label", "pre-compact",
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
    else:
        logging.info(
            "SKIP: only %d new turns (min %d) since last cursor",
            new_turns, MIN_TURNS_TO_FLUSH,
        )

    # Capture Claude Code's native /compact summaries — independent path
    # with its own dedup.
    extract_native_script = SCRIPTS_DIR / "extract_native_summaries.py"
    if extract_native_script.exists():
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # --watch 600: large compacts can take 1–3 min before the summary
        # lands in the JSONL. The script polls patiently in background.
        native_cmd = [
            "uv", "run", "--directory", str(ROOT),
            "python", str(extract_native_script),
            str(transcript_path), session_id,
            "--watch", "600",
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
