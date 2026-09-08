"""
Shared cursor + incremental-slice helper for ACE capture hooks.

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
        "schema_version": 2,
        "last_flush_ts": 1778293267.49,
        "last_main_turn_count": 148,
        "last_subagent_counts": {"agent-stem-1": 12, "agent-stem-2": 5}
      }
    }
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from utils import redact_sensitive_text

CURSOR_GC_DAYS = 7
# Increment when the extracted turn representation changes in a way that
# makes an existing count unsafe.  Version 2 adds tool results to Claude
# turns; an old cursor is therefore replayed from the beginning once instead
# of silently skipping the newly preserved evidence.
CURSOR_SCHEMA_VERSION = 2
TOOL_EVIDENCE_CHARS = 1200

_ERROR_MARKER_RE = re.compile(
    r"(?i)(?:\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\bfailure\b|"
    r"\bfatal\b|\btimeout\b|\btimed out\b|\bHTTP/\d(?:\.\d)?\s+[45]\d\d\b|"
    r"\b[45]\d\d\b)"
)


def empty_state() -> dict[str, Any]:
    return {
        "schema_version": CURSOR_SCHEMA_VERSION,
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
    # Before schema v2 the extractor dropped tool_result blocks.  Reusing its
    # count would make the first post-upgrade slice start after evidence that
    # was never represented.  Reset just this session; other sessions migrate
    # lazily when they are next observed.
    if isinstance(raw, dict) and raw.get("schema_version", 1) != CURSOR_SCHEMA_VERSION:
        return empty_state()
    state = empty_state()
    if isinstance(raw, dict):
        state["last_flush_ts"] = float(raw.get("last_flush_ts", 0.0) or 0.0)
        state["last_main_turn_count"] = int(raw.get("last_main_turn_count", 0) or 0)
        sub = raw.get("last_subagent_counts", {})
        state["last_subagent_counts"] = sub if isinstance(sub, dict) else {}
    return state


def _stringify_evidence(value: Any) -> str:
    """Convert a transcript value to redacted text without exposing secrets."""
    if value is None:
        return ""
    if isinstance(value, str):
        raw = value
    else:
        try:
            raw = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            raw = str(value)
    return redact_sensitive_text(raw.strip())


def _error_line(text: str) -> str:
    """Return the last likely error line, without returning any raw secret."""
    candidates = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and _ERROR_MARKER_RE.search(line)
    ]
    return candidates[-1] if candidates else ""


def bounded_redacted_text(value: Any, limit: int) -> str:
    """Keep bounded, redacted head/tail evidence and retain a likely error.

    Tool arguments and results are often large JSON blobs.  Taking only the
    head hides the useful failure at the end, while taking only the tail hides
    the command or identifier at the beginning.  Redaction happens *before*
    slicing so a secret cannot survive solely because it was near a boundary.
    The returned string is always at most ``limit`` characters.
    """
    if limit <= 0:
        return ""
    text = _stringify_evidence(value)
    if len(text) <= limit:
        return text

    marker = f"...[truncated {len(text) - limit} chars]..."
    # A very small limit cannot fit useful framing and content simultaneously.
    if len(marker) >= limit:
        return marker[:limit]

    available = limit - len(marker)
    head_size = available // 2
    tail_size = available - head_size
    head = text[:head_size]
    tail = text[-tail_size:] if tail_size else ""
    base = f"{head}{marker}{tail}"

    error = _error_line(text)
    if not error or error in base:
        return base

    # Reserve a compact, explicitly labelled error excerpt while retaining
    # both sides.  This is especially important for outputs whose final
    # diagnostic is not literally the final line.
    error_prefix = "[error evidence: "
    error_suffix = "]"
    error_budget = min(max(48, available // 3), available)
    excerpt_size = error_budget - len(error_prefix) - len(error_suffix)
    if excerpt_size <= 0:
        return base
    error_excerpt = error[:excerpt_size]
    error_clause = f"{error_prefix}{error_excerpt}{error_suffix}"

    content_budget = available - len(error_clause) - 1
    if content_budget <= 0:
        return (marker + error_clause)[:limit]
    head_size = content_budget // 2
    tail_size = content_budget - head_size
    head = text[:head_size]
    tail = text[-tail_size:] if tail_size else ""
    return f"{head}{marker}{error_clause}{tail}"[:limit]


def _scalar_ref(value: Any) -> str:
    """Render a source reference without dumping arbitrary nested metadata."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return redact_sensitive_text(str(value).strip())


def source_reference(entry: dict[str, Any], message: dict[str, Any] | None = None) -> str:
    """Return stable transcript refs (uuid/id/parent) for evidence markers."""
    message = message or {}
    fields = (
        ("uuid", "source_ref"),
        ("id", "source_ref"),
        ("source_ref", "source_ref"),
        ("source", "source"),
        ("parentUuid", "parent_ref"),
        ("parent_id", "parent_ref"),
    )
    refs: list[str] = []
    seen: set[str] = set()
    for key, label in fields:
        value = _scalar_ref(message.get(key)) or _scalar_ref(entry.get(key))
        if value and f"{label}={value}" not in seen:
            refs.append(f"{label}={value}")
            seen.add(f"{label}={value}")
    return " ".join(refs)


def _call_id(block: dict[str, Any], entry: dict[str, Any] | None = None) -> str:
    """Extract Claude's tool-call id from the common schema variants."""
    entry = entry or {}
    for key in ("call_id", "tool_use_id", "tool_call_id", "id"):
        value = _scalar_ref(block.get(key)) or _scalar_ref(entry.get(key))
        if value:
            return value
    return "?"


def _content_text(value: Any) -> str:
    """Serialize tool output content while retaining structured details."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(parts)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


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

    The cursor module is the single source of truth for what counts as a
    "turn" (the unit the cursor indexes).  Tool results stay attached to the
    message that carried them, so a call and its result remain auditable.
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
                    role = str(msg.get("role", "") or "")
                    content = msg.get("content", "")
                else:
                    msg = {}
                    role = str(entry.get("role", "") or "")
                    content = entry.get("content", "")

                # Claude normally nests tool_result blocks in a user message,
                # but accepting a top-level tool role makes this extractor
                # tolerant of transcript format revisions.
                is_tool_record = role == "tool" or entry.get("type") in {
                    "tool_result",
                    "tool_response",
                }
                if role not in ("user", "assistant", "tool") and not is_tool_record:
                    continue

                parts: list[str] = []
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            if isinstance(block, str) and block.strip():
                                parts.append(redact_sensitive_text(block.strip()))
                            continue
                        btype = str(block.get("type", "") or "")
                        if btype in {"text", "input_text", "output_text"}:
                            text = block.get("text", "")
                            if text:
                                parts.append(redact_sensitive_text(str(text).strip()))
                        elif btype == "tool_use":
                            name = _scalar_ref(block.get("name")) or "?"
                            call_id = _call_id(block, entry)
                            inp = bounded_redacted_text(
                                block.get("input", {}), TOOL_EVIDENCE_CHARS
                            )
                            refs = source_reference(entry, msg)
                            ref_text = f" {refs}" if refs else ""
                            parts.append(
                                f"[Tool call role=assistant name={name} call_id={call_id}{ref_text}]"
                                f" {inp}"
                            )
                        elif btype == "tool_result":
                            call_id = _call_id(block, entry)
                            result = bounded_redacted_text(
                                block.get(
                                    "content",
                                    block.get("output", block.get("result", "")),
                                ),
                                TOOL_EVIDENCE_CHARS,
                            )
                            is_error = bool(
                                block.get("is_error")
                                or block.get("isError")
                                or str(block.get("status", "")).lower()
                                in {"error", "failed", "failure"}
                            )
                            status = "error" if is_error else str(
                                block.get("status") or "observed"
                            )
                            refs = source_reference(entry, msg)
                            ref_text = f" {refs}" if refs else ""
                            parts.append(
                                f"[Tool result role=tool status={status} call_id={call_id}"
                                f"{ref_text}] {result}"
                            )
                elif isinstance(content, str) and content.strip() and not is_tool_record:
                    parts.append(redact_sensitive_text(content.strip()))
                elif is_tool_record:
                    # Some exporters put the result directly on the record.
                    call_id = _call_id(entry, entry)
                    result = bounded_redacted_text(
                        content
                        if content not in (None, "")
                        else entry.get("output", entry.get("result", "")),
                        TOOL_EVIDENCE_CHARS,
                    )
                    status = str(entry.get("status") or "observed")
                    if (
                        entry.get("is_error")
                        or entry.get("isError")
                        or _ERROR_MARKER_RE.search(result)
                    ):
                        status = "error"
                    refs = source_reference(entry, msg)
                    ref_text = f" {refs}" if refs else ""
                    parts.append(
                        f"[Tool result role=tool status={status} call_id={call_id}"
                        f"{ref_text}] {result}"
                    )

                if parts:
                    label = (
                        "Tool"
                        if role == "tool"
                        else ("User" if role == "user" else "Assistant")
                    )
                    refs = source_reference(entry, msg)
                    ref_text = f" [{refs}]" if refs else ""
                    rendered_parts = "\n".join(parts).strip()
                    turns.append(f"**{label}{ref_text}:** {rendered_parts}\n")
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
    if state.get("schema_version", 1) != CURSOR_SCHEMA_VERSION:
        state = empty_state()
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
        "schema_version": CURSOR_SCHEMA_VERSION,
        "last_flush_ts": state["last_flush_ts"],  # caller updates if it spawns
        "last_main_turn_count": len(main_turns),
        "last_subagent_counts": next_sub_counts,
    }
    return context, new_total, next_state
