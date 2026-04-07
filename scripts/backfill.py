"""
Backfill existing transcripts into the knowledge base.

Reads all JSONL transcripts from the Claude Code projects directory,
extracts conversation context (including tool calls and subagents),
and runs flush.py on each one.

Usage:
    uv run python scripts/backfill.py
    uv run python scripts/backfill.py --dry-run
    uv run python scripts/backfill.py --transcripts-dir /path/to/projects/<project-slug>
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

MAX_TURNS = 30
MAX_CONTEXT_CHARS = 15_000
MIN_TURNS_TO_FLUSH = 2


def find_transcripts_dir() -> Path:
    """Auto-detect the Claude Code transcripts directory for the current project."""
    # Try CCS path first (newer Claude Code versions)
    ccs_base = Path.home() / ".ccs" / "shared" / "context-groups" / "default" / "projects"
    # Try standard path
    claude_base = Path.home() / ".claude" / "projects"

    cwd_slug = str(Path.cwd()).replace("/", "-").replace("\\", "-").lstrip("-")

    for base in [ccs_base, claude_base]:
        candidate = base / cwd_slug
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find transcripts directory. Pass --transcripts-dir explicitly.\n"
        f"Tried:\n  {ccs_base / cwd_slug}\n  {claude_base / cwd_slug}"
    )


def extract_turns_from_jsonl(jsonl_path: Path) -> list[str]:
    """Extract conversation turns from a single JSONL file."""
    turns: list[str] = []
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
                    # skip tool_result and thinking blocks
                content = "\n".join(p for p in parts if p.strip())

            if isinstance(content, str) and content.strip():
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {content.strip()}\n")

    return turns


def extract_conversation_context(transcript_path: Path) -> tuple[str, int]:
    turns = extract_turns_from_jsonl(transcript_path)

    # Also include subagent transcripts if present
    subagents_dir = transcript_path.parent / transcript_path.stem / "subagents"
    if subagents_dir.exists():
        for subagent_file in sorted(subagents_dir.glob("*.jsonl")):
            sub_turns = extract_turns_from_jsonl(subagent_file)
            if sub_turns:
                turns.append(f"**[Subagent: {subagent_file.stem}]**\n")
                turns.extend(sub_turns)

    recent = turns[-MAX_TURNS:]
    context = "\n".join(recent)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[-MAX_CONTEXT_CHARS:]
        boundary = context.find("\n**")
        if boundary > 0:
            context = context[boundary + 1:]

    return context, len(recent)


def main():
    dry_run = "--dry-run" in sys.argv

    # Allow explicit --transcripts-dir override
    transcripts_dir: Path | None = None
    for i, arg in enumerate(sys.argv):
        if arg == "--transcripts-dir" and i + 1 < len(sys.argv):
            transcripts_dir = Path(sys.argv[i + 1])

    if transcripts_dir is None:
        try:
            transcripts_dir = find_transcripts_dir()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    transcripts = sorted(transcripts_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)

    print(f"Transcripts dir: {transcripts_dir}")
    print(f"Found {len(transcripts)} transcripts")
    if dry_run:
        print("[dry-run] No API calls will be made\n")

    flush_script = SCRIPTS_DIR / "flush.py"

    for i, transcript in enumerate(transcripts, 1):
        session_id = transcript.stem
        mtime = datetime.fromtimestamp(transcript.stat().st_mtime, tz=timezone.utc).astimezone()
        size_kb = transcript.stat().st_size // 1024

        print(f"[{i}/{len(transcripts)}] {session_id[:8]}... ({size_kb}KB, {mtime.strftime('%Y-%m-%d')})", end="")

        try:
            context, turn_count = extract_conversation_context(transcript)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if turn_count < MIN_TURNS_TO_FLUSH:
            print(f" SKIP ({turn_count} turns)")
            continue

        if not context.strip():
            print(f" SKIP (empty)")
            continue

        print(f" {turn_count} turns, {len(context)} chars", end="")

        if dry_run:
            print(" [dry-run]")
            continue

        # Write context to temp file
        timestamp = mtime.strftime("%Y%m%d-%H%M%S")
        context_file = SCRIPTS_DIR / f"backfill-{session_id}-{timestamp}.md"
        context_file.write_text(context, encoding="utf-8")

        # Run flush.py synchronously
        cmd = [
            "uv", "run", "--directory", str(ROOT),
            "python", str(flush_script),
            str(context_file), session_id,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f" FLUSH ERROR: {result.stderr[:100]}")
        else:
            print(f" done")

        time.sleep(2)

    print("\nBackfill complete.")
    if not dry_run:
        print("Run: uv run python scripts/compile.py  to compile daily logs -> knowledge articles")


if __name__ == "__main__":
    main()
