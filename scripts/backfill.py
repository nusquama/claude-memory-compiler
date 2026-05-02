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
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import PROJECT_DIR, SCRIPTS_DIR, TOOL_DIR as ROOT

MAX_TURNS = int(os.environ.get("CMC_MAX_TURNS", 300))
MAX_CONTEXT_CHARS = int(os.environ.get("CMC_MAX_CONTEXT_CHARS", 120_000))
MIN_TURNS_TO_FLUSH = 2


def find_transcripts_dir() -> Path:
    """Auto-detect the Claude Code transcripts directory for the current project.

    If both CCS and standard paths exist, picks the one with the most recent transcript.
    """
    import os
    project_root = os.environ.get("CLAUDE_PROJECT_DIR") or str(Path.cwd())
    cwd_slug = project_root.replace("/", "-").replace("\\", "-")

    candidates = [
        Path.home() / ".ccs" / "shared" / "context-groups" / "default" / "projects" / cwd_slug,
        Path.home() / ".claude" / "projects" / cwd_slug,
    ]

    existing = [p for p in candidates if p.exists()]

    if not existing:
        raise FileNotFoundError(
            f"Could not find transcripts directory. Pass --transcripts-dir explicitly.\n"
            + "\n".join(f"  {p}" for p in candidates)
        )

    if len(existing) == 1:
        return existing[0]

    # Both exist — pick the one with the most recently modified transcript
    def latest_mtime(d: Path) -> float:
        files = list(d.glob("*.jsonl"))
        return max((f.stat().st_mtime for f in files), default=0.0)

    return max(existing, key=latest_mtime)


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


def _prepare_one(transcript: Path):
    """Build the (transcript, session_id, context_file, label) tuple for a
    single transcript, or None if it should be skipped."""
    session_id = transcript.stem
    mtime = datetime.fromtimestamp(transcript.stat().st_mtime, tz=timezone.utc).astimezone()
    size_kb = transcript.stat().st_size // 1024
    label = f"{session_id[:8]}... ({size_kb}KB, {mtime.strftime('%Y-%m-%d')})"

    try:
        context, turn_count = extract_conversation_context(transcript)
    except Exception as e:
        return ("error", label, f"ERROR: {e}")

    if turn_count < MIN_TURNS_TO_FLUSH:
        return ("skip", label, f"SKIP ({turn_count} turns)")
    if not context.strip():
        return ("skip", label, "SKIP (empty)")

    timestamp = mtime.strftime("%Y%m%d-%H%M%S")
    context_file = SCRIPTS_DIR / f"backfill-{session_id}-{timestamp}.md"
    context_file.write_text(context, encoding="utf-8")
    return ("ok", label, session_id, context_file, turn_count, len(context))


def _run_flush(session_id: str, context_file: Path, flush_script: Path, parallel_mode: bool) -> tuple[int, str]:
    """Spawn flush.py for one session. Returns (returncode, stderr_tail)."""
    cmd = [
        "uv", "run", "--directory", str(ROOT),
        "python", str(flush_script),
        str(context_file), session_id,
    ]
    flush_env = os.environ.copy()
    flush_env.setdefault("CMC_FLUSH_MAX_RETRIES", "1")
    flush_env.setdefault("CMC_FLUSH_ATTEMPT_TIMEOUT", "90")
    if parallel_mode:
        # Caller (backfill) is managing concurrency itself — bypass the
        # vault-wide lock. Each pool worker still spawns its own flush.py
        # subprocess; serialising via the lock would defeat the parallelism.
        flush_env["CMC_FLUSH_SKIP_LOCK"] = "1"
    result = subprocess.run(cmd, capture_output=True, text=True, env=flush_env)
    return result.returncode, result.stderr


def main():
    if PROJECT_DIR is None:
        print("error: no project detected. Run from inside a git repo.", file=sys.stderr)
        sys.exit(1)
    dry_run = "--dry-run" in sys.argv

    # Parallelism (default 1 = current sequential behaviour). Pre-warm runs
    # the first eligible transcript sequentially to refresh OAuth tokens
    # before the pool starts — avoids the auth race that crashes the
    # bundled CLI on cold-start.
    parallel = 1
    for i, arg in enumerate(sys.argv):
        if arg == "--parallel" and i + 1 < len(sys.argv):
            try:
                parallel = max(1, int(sys.argv[i + 1]))
            except ValueError:
                print(f"error: --parallel needs an integer, got {sys.argv[i + 1]!r}", file=sys.stderr)
                sys.exit(2)

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
    if parallel > 1:
        print(f"Parallel mode: pre-warm + pool of {parallel} workers")
    if dry_run:
        print("[dry-run] No API calls will be made\n")

    flush_script = SCRIPTS_DIR / "flush.py"
    total = len(transcripts)
    parallel_mode = parallel > 1

    # Phase 1: prepare all transcripts (extract context, write temp files,
    # filter skips). Cheap, no API calls.
    prepared: list[tuple] = []
    for i, transcript in enumerate(transcripts, 1):
        prep = _prepare_one(transcript)
        prefix = f"[{i}/{total}] {prep[1]}"
        if prep[0] == "ok":
            _, label, session_id, ctx_file, turns, chars = prep
            print(f"{prefix} {turns} turns, {chars} chars", end="")
            if dry_run:
                print(" [dry-run]")
                continue
            print(" queued")
            prepared.append(prep)
        else:
            print(f"{prefix} {prep[2]}")

    if dry_run or not prepared:
        print("\nBackfill complete.")
        return

    # Phase 2: run flushes. Pre-warm with the first one sequentially.
    print(f"\nProcessing {len(prepared)} sessions...")
    print("Pre-warm: first session sequential to establish auth state")
    first = prepared[0]
    _, label, session_id, ctx_file, _, _ = first
    print(f"[1/{len(prepared)}] {label}", end="", flush=True)
    rc, err = _run_flush(session_id, ctx_file, flush_script, parallel_mode=False)
    print(" done" if rc == 0 else f" FLUSH ERROR: {err[:100]}")

    rest = prepared[1:]
    if not rest:
        print("\nBackfill complete.")
        return

    if not parallel_mode:
        for idx, item in enumerate(rest, 2):
            _, label, sid, cf, _, _ = item
            print(f"[{idx}/{len(prepared)}] {label}", end="", flush=True)
            rc, err = _run_flush(sid, cf, flush_script, parallel_mode=False)
            print(" done" if rc == 0 else f" FLUSH ERROR: {err[:100]}")
            time.sleep(2)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Pool mode: {parallel} concurrent flushes (lock bypassed)")
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_run_flush, item[2], item[3], flush_script, True): (idx, item)
                for idx, item in enumerate(rest, 2)
            }
            for fut in as_completed(futures):
                idx, item = futures[fut]
                _, label, _, _, _, _ = item
                try:
                    rc, err = fut.result()
                except Exception as e:
                    rc, err = -1, str(e)
                tag = "done" if rc == 0 else f"FLUSH ERROR: {err[:100]}"
                print(f"[{idx}/{len(prepared)}] {label} {tag}")

    print("\nBackfill complete.")
    if not dry_run:
        print("Run: uv run python scripts/compile.py  to compile daily logs -> knowledge articles")


if __name__ == "__main__":
    main()
