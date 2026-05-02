"""Extract Claude Code's native compaction summaries from a session JSONL.

Claude Code, when running `/compact` or auto-compaction, produces a summary
turn that is written into the session transcript with `isCompactSummary: true`
(and the summary text in the message content). This script reads the
transcript, finds every such summary, and appends them to the project's
daily log so they survive next to the LLM-extracted memory entries.

Idempotent: tracks which (session_id, summary_uuid) pairs have already been
emitted in `STATE_DIR / native-summaries.json`, so re-running on the same
transcript is a no-op.

Usage:
    uv run python extract_native_summaries.py <transcript_jsonl> <session_id>

Exits silently (status 0) if PROJECT_DIR is not resolved (no vault folder
for this project).
"""

from __future__ import annotations

# Recursion guard — match flush.py: never trigger SessionEnd recursively.
import os
os.environ.setdefault("CLAUDE_INVOKED_BY", "memory_native_summary")

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DAILY_DIR,
    FLUSH_LOG as LOG_FILE,
    PROJECT_DIR,
    STATE_DIR,
)


if PROJECT_DIR is None:
    sys.exit(0)

DAILY_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [native-summary] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


SUMMARY_STATE_FILE = STATE_DIR / "native-summaries.json"


def load_emitted_uuids() -> set[str]:
    if not SUMMARY_STATE_FILE.exists():
        return set()
    try:
        return set(json.loads(SUMMARY_STATE_FILE.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def save_emitted_uuids(uuids: set[str]) -> None:
    SUMMARY_STATE_FILE.write_text(json.dumps(sorted(uuids)), encoding="utf-8")


def extract_text_from_content(content) -> str:
    """Pull the user-facing text out of a JSONL message.content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    parts.append(txt)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def find_compact_summaries(transcript_path: Path) -> list[dict]:
    """Return every compactSummary turn in the transcript.

    Each entry is `{uuid, timestamp, text, session_id}`. Order preserved.
    """
    summaries: list[dict] = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not entry.get("isCompactSummary"):
                continue

            uuid = entry.get("uuid", "")
            timestamp = entry.get("timestamp", "")
            session_id = entry.get("sessionId", "")
            msg = entry.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            text = extract_text_from_content(content).strip()

            if not text or not uuid:
                continue

            summaries.append(
                {
                    "uuid": uuid,
                    "timestamp": timestamp,
                    "text": text,
                    "session_id": session_id,
                }
            )
    return summaries


def append_to_daily_log(text: str, when: datetime) -> None:
    """Append a single native-summary entry to today's daily log."""
    log_path = DAILY_DIR / f"{when.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        log_path.write_text(
            f"# Daily Log: {when.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = when.strftime("%H:%M")
    entry = (
        f"### Claude Code Compact Summary ({time_str})\n\n"
        f"_Native summary produced by Claude Code during `/compact`._\n\n"
        f"{text}\n\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def parse_iso_timestamp(ts: str) -> datetime:
    """Parse an ISO timestamp from the JSONL into a tz-aware local datetime."""
    if not ts:
        return datetime.now(timezone.utc).astimezone()
    # Python's fromisoformat handles the trailing 'Z' from 3.11+.
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone()
    except ValueError:
        return datetime.now(timezone.utc).astimezone()


def process_once(transcript_path: Path, session_id: str) -> int:
    """Run extraction once. Returns number of NEW summaries captured."""
    if not transcript_path.exists():
        return 0
    summaries = find_compact_summaries(transcript_path)
    if not summaries:
        return 0

    emitted = load_emitted_uuids()
    new_count = 0
    for summary in summaries:
        if summary["uuid"] in emitted:
            continue
        when = parse_iso_timestamp(summary["timestamp"])
        try:
            append_to_daily_log(summary["text"], when)
        except OSError as e:
            logging.error(
                "Failed to append native summary %s: %s",
                summary["uuid"],
                e,
            )
            continue
        emitted.add(summary["uuid"])
        new_count += 1

    if new_count:
        save_emitted_uuids(emitted)
    return new_count


def main() -> None:
    # Args: <transcript_jsonl> <session_id> [--watch <seconds>]
    if len(sys.argv) < 3:
        logging.error(
            "Usage: %s <transcript_jsonl> <session_id> [--watch <seconds>]",
            sys.argv[0],
        )
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    session_id = sys.argv[2]

    # Optional --watch flag to poll the JSONL with retries.
    # Needed for PreCompact, which fires BEFORE Claude Code writes the
    # summary line. We poll until a new summary appears or budget expires.
    watch_seconds = 0
    if "--watch" in sys.argv:
        i = sys.argv.index("--watch")
        if i + 1 < len(sys.argv):
            try:
                watch_seconds = int(sys.argv[i + 1])
            except ValueError:
                watch_seconds = 0

    logging.info(
        "extract_native_summaries started for session %s, transcript: %s, watch=%ds",
        session_id,
        transcript_path,
        watch_seconds,
    )

    if not transcript_path.exists():
        logging.info("SKIP: transcript missing: %s", transcript_path)
        return

    # Baseline pass — captures whatever is already in the JSONL.
    new_count = process_once(transcript_path, session_id)

    if watch_seconds > 0 and new_count == 0:
        # Poll loop: 2s interval up to watch_seconds. With 2s polling we
        # catch the summary within seconds of it landing — feels close to
        # event-driven. Cost is negligible (small file, just sleep + read
        # between checks).
        deadline = time.time() + watch_seconds
        attempts = 0
        while time.time() < deadline:
            time.sleep(2)
            attempts += 1
            captured = process_once(transcript_path, session_id)
            if captured:
                logging.info(
                    "Watch caught %d new summary on attempt %d (session %s)",
                    captured,
                    attempts,
                    session_id,
                )
                new_count = captured
                break
        else:
            logging.info(
                "Watch budget (%ds) elapsed without summary appearing (session %s)",
                watch_seconds,
                session_id,
            )

    if new_count:
        logging.info(
            "Captured %d new native summary entries (session %s)",
            new_count,
            session_id,
        )
    else:
        logging.info("All native summaries already captured (session %s)", session_id)


if __name__ == "__main__":
    main()
