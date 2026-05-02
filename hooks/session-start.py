"""
SessionStart hook - injects knowledge base context into every conversation.

This is the "context injection" layer. When Claude Code starts a session,
this hook reads the knowledge base index and recent daily log, then injects
them as additional context so Claude always "remembers" what it has learned.

Configure in .claude/settings.json:
{
    "hooks": {
        "SessionStart": [{
            "matcher": "",
            "command": "uv run python hooks/session-start.py"
        }]
    }
}
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from config import (  # noqa: E402
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    PROJECT_DIR,
    SCRIPTS_DIR,
    TOOL_DIR as ROOT,
)


def maybe_extract_native_summaries() -> None:
    """Safety net — re-run native summary extraction at session start.

    Catches any compact summaries that PreCompact missed because they
    were written to the JSONL after the hook fired. The extract script
    is idempotent (UUID-tracked), so this is a no-op when nothing's new.
    """
    if os.environ.get("CLAUDE_INVOKED_BY"):
        return  # recursion guard

    raw = ""
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
    except (ValueError, OSError):
        return
    if not raw:
        return

    try:
        hook_input = json.loads(raw)
    except json.JSONDecodeError:
        try:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', raw)
            hook_input = json.loads(fixed)
        except json.JSONDecodeError:
            return

    transcript_path_str = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "unknown")
    if not transcript_path_str or not Path(transcript_path_str).exists():
        return

    extract_script = SCRIPTS_DIR / "extract_native_summaries.py"
    if not extract_script.exists():
        return

    cmd = [
        "uv",
        "run",
        "--directory",
        str(ROOT),
        "python",
        str(extract_script),
        transcript_path_str,
        session_id,
    ]
    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
    except Exception:
        pass  # never block SessionStart on this side-task

# No project context → emit empty additionalContext and exit (neutral behavior).
if PROJECT_DIR is None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "",
        }
    }))
    sys.exit(0)

MAX_CONTEXT_CHARS = 20_000
MAX_LOG_LINES = 30


def get_recent_log() -> str:
    """Read the most recent daily log (today or yesterday)."""
    today = datetime.now(timezone.utc).astimezone()

    for offset in range(2):
        date = today - timedelta(days=offset)
        log_path = DAILY_DIR / f"{date.strftime('%Y-%m-%d')}.md"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            # Return last N lines to keep context small
            recent = lines[-MAX_LOG_LINES:] if len(lines) > MAX_LOG_LINES else lines
            return "\n".join(recent)

    return "(no recent daily log)"


def build_context() -> str:
    """Assemble the context to inject into the conversation."""
    parts = []

    # Today's date
    today = datetime.now(timezone.utc).astimezone()
    parts.append(f"## Today\n{today.strftime('%A, %B %d, %Y')}")

    # Knowledge base index (the core retrieval mechanism)
    if INDEX_FILE.exists():
        index_content = INDEX_FILE.read_text(encoding="utf-8")
        parts.append(f"## Knowledge Base Index\n\n{index_content}")
    else:
        parts.append("## Knowledge Base Index\n\n(empty - no articles compiled yet)")

    # Recent daily log
    recent_log = get_recent_log()
    parts.append(f"## Recent Daily Log\n\n{recent_log}")

    context = "\n\n---\n\n".join(parts)

    # Truncate if too long
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n...(truncated)"

    return context


def main():
    # Safety net: catch any native compact summaries the PreCompact hook
    # missed (the summary often lands in the JSONL AFTER PreCompact runs).
    # This consumes stdin, so it must run before anything else that needs it.
    try:
        maybe_extract_native_summaries()
    except Exception:
        pass

    context = build_context()

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()
