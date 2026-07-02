"""
Shared cursor + incremental-slice helper for CMC capture hooks.

Used by:
- hooks/stop-flush-checkpoint.py  (periodic 30-min flush)
- hooks/session-end.py            (final flush at session end)
- hooks/pre-compact.py            (flush before Claude Code auto-compacts)

All three hooks slice the conversation transcript starting from the cursor
position and only ship NEW content to flush.py — never re-extracts what
was already captured. The caller advances the cursor atomically right
before spawning flush.py.

Cursor schema (JSON, per-project file at <STATE_DIR>/checkpoint-cursor.json):
    {
      "<session_id>": {
        "last_flush_ts": 1778293267.49,
        "last_main_turn_count": 148,
        "last_subagent_counts": {"agent-stem-1": 12, "agent-stem-2": 5}
      }
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

CURSOR_GC_DAYS = 7


def empty_state() -> dict[str, Any]:
    return {
        "last_flush_ts": 0.0,
        "last_main_turn_count": 0,
        "last_subagent_counts": {},
    }


def load_cursor(cursor_file: Path) -> dict[str, dict[str, Any]]:
    if not cursor_file.exists():
        return {}
    try:
        data = json.loads(cursor_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cursor(cursor_file: Path, cursor: dict[str, dict[str, Any]]) -> None:
    cursor_file.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file + rename. Avoids partial writes if the process
    # is killed mid-write (which would corrupt cursor state and cause either
    # a re-extraction of already-captured content, or worse, data loss).
    tmp = cursor_file.with_suffix(f"{cursor_file.suffix}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cursor), encoding="utf-8")
    tmp.replace(cursor_file)


def get_session_state(
    cursor: dict[str, dict[str, Any]], session_id: str
) -> dict[str, Any]:
    """Return the per-session state, defaulting to zeros if absent.

    Defensive: if a stored entry is missing fields (e.g. from an older
    schema version), fill in the missing pieces.
    """
    raw = cursor.get(session_id, {})
    state = empty_state()
    if isinstance(raw, dict):
        state["last_flush_ts"] = float(raw.get("last_flush_ts", 0.0) or 0.0)
        state["last_main_turn_count"] = int(raw.get("last_main_turn_count", 0) or 0)
        sub = raw.get("last_subagent_counts", {})
        state["last_subagent_counts"] = sub if isinstance(sub, dict) else {}
    return state


def gc_old_entries(
    cursor: dict[str, dict[str, Any]], now: float | None = None
) -> dict[str, dict[str, Any]]:
    """Remove cursor entries older than CURSOR_GC_DAYS."""
    if now is None:
        now = time.time()
    cutoff = now - CURSOR_GC_DAYS * 24 * 3600
    return {
        sid: s for sid, s in cursor.items()
        if isinstance(s, dict) and float(s.get("last_flush_ts", 0) or 0) > cutoff
    }


def extract_turns_from_jsonl(jsonl_path: Path) -> list[str]:
    """Extract user/assistant turns as markdown.

    Mirrors hooks/session-end.py for backward compatibility — kept here so
    the cursor module is the single source of truth for what counts as a
    "turn" (the unit the cursor indexes).
    """
    turns: list[str] = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = entry.get("message", {})
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                else:
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                if role not in ("user", "assistant"):
                    continue
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if not isinstance(block, dict):
                            if isinstance(block, str):
                                parts.append(block)
                            continue
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            name = block.get("name", "?")
                            inp = block.get("input", {})
                            inp_str = json.dumps(inp, ensure_ascii=False)[:200]
                            parts.append(f"[Tool: {name}] {inp_str}")
                    content = "\n".join(p for p in parts if p.strip())
                if isinstance(content, str) and content.strip():
                    label = "User" if role == "user" else "Assistant"
                    turns.append(f"**{label}:** {content.strip()}\n")
    except OSError:
        return []
    return turns


def extract_incremental_slice(
    transcript_path: Path,
    state: dict[str, Any],
    max_chars: int,
) -> tuple[str, int, dict[str, Any]]:
    """Extract ONLY new turns since the cursor position.

    Returns:
        context: markdown string with new main turns + new subagent turns
        new_turn_total: number of new turns across main + subagents
        next_state: cursor state to write back AFTER successful spawn
                    (caller decides when to commit)

    First-run semantics: if state is fresh (last_main_turn_count=0,
    no subagent counts), this returns the full transcript content.
    """
    main_turns = extract_turns_from_jsonl(transcript_path)
    last_main = state["last_main_turn_count"]
    new_main = main_turns[last_main:]

    subagents_dir = transcript_path.parent / transcript_path.stem / "subagents"
    last_sub_counts: dict[str, int] = dict(state["last_subagent_counts"])
    next_sub_counts: dict[str, int] = dict(last_sub_counts)
    sub_pieces: list[str] = []

    if subagents_dir.exists():
        for sub_file in sorted(subagents_dir.glob("*.jsonl")):
            stem = sub_file.stem
            sub_turns = extract_turns_from_jsonl(sub_file)
            prev = last_sub_counts.get(stem, 0)
            new_sub = sub_turns[prev:]
            if new_sub:
                sub_pieces.append(f"**[Subagent: {stem}]**\n")
                sub_pieces.extend(new_sub)
            next_sub_counts[stem] = len(sub_turns)

    new_total = len(new_main) + sum(
        next_sub_counts.get(k, 0) - last_sub_counts.get(k, 0)
        for k in next_sub_counts
    )

    pieces = list(new_main) + sub_pieces
    context = "\n".join(pieces)

    # Hard safety cap on a single slice — protects against pathological
    # inputs (corrupted transcripts, etc.). flush.py also has its own
    # map-reduce chunking for large but legitimate slices.
    if len(context) > max_chars:
        elided = len(context) - max_chars
        context = (
            f"...[{elided} chars elided — exceeded hard cap of "
            f"{max_chars} chars; kept tail only]...\n\n"
            + context[-max_chars:]
        )

    next_state = {
        "last_flush_ts": state["last_flush_ts"],  # caller updates if it spawns
        "last_main_turn_count": len(main_turns),
        "last_subagent_counts": next_sub_counts,
    }
    return context, new_total, next_state
