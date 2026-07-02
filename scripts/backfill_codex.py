"""
Backfill Codex Desktop/CLI sessions into CMC daily logs.

Codex stores rollouts under ~/.codex/sessions/YYYY/MM/DD/*.jsonl with a
different schema from Claude Code transcripts. This script converts those
JSONL records into the same conversation-context shape consumed by flush.py,
then writes the extracted daily-log entry into the matching CMC vault project.

Usage:
    uv run python scripts/backfill_codex.py --dry-run
    uv run python scripts/backfill_codex.py --since 2026-05-30 --dry-run
    uv run python scripts/backfill_codex.py --fallback-project Conversations
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config import FLUSH_MAX_CHARS, FLUSH_MAX_TURNS, TOOL_DIR, VAULT_ROOT

# Historical backfills should fail fast. Interactive hooks keep the defaults
# from flush.py; this script may process many sessions and must not spend
# several minutes retrying one stale rollout.
os.environ.setdefault("CMC_FLUSH_MAX_RETRIES", "1")
os.environ.setdefault("CMC_FLUSH_ATTEMPT_TIMEOUT", "120")

from flush import run_flush


MIN_TURNS_TO_FLUSH = 2
DEFAULT_TOOL_ARG_CHARS = 600
DEFAULT_TOOL_OUTPUT_CHARS = 1000
DEFAULT_SKIP_ACTIVE_SECONDS = 300
DEFAULT_MAX_CONTEXT_CHARS = 350000


@dataclass
class CodexSession:
    path: Path
    session_id: str
    timestamp: datetime
    cwd: str
    cli_version: str
    originator: str
    context: str
    turn_count: int


def parse_iso_datetime(value: str | None, fallback: float) -> datetime:
    if value:
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback, tz=timezone.utc).astimezone()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def truncate_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    value = value.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...[truncated {len(value) - limit} chars]"


def text_from_content_blocks(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in {"input_text", "output_text", "text"}:
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(p.strip() for p in parts if p and p.strip())


def extract_codex_session(
    path: Path,
    *,
    tool_arg_chars: int,
    tool_output_chars: int,
) -> CodexSession | None:
    meta: dict[str, Any] = {}
    turns: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = entry.get("type")
            payload = entry.get("payload", {})
            if etype == "session_meta" and isinstance(payload, dict) and not meta:
                meta = payload
                continue
            if etype != "response_item" or not isinstance(payload, dict):
                continue

            ptype = payload.get("type")
            role = payload.get("role")

            if ptype == "message":
                if role not in {"user", "assistant"}:
                    continue
                text = text_from_content_blocks(payload.get("content"))
                if not text:
                    continue
                label = "User" if role == "user" else "Assistant"
                turns.append(f"**{label}:** {text}\n")
                continue

            if ptype in {"function_call", "custom_tool_call"}:
                name = payload.get("name", "?")
                args = payload.get("arguments", payload.get("input", ""))
                rendered = truncate_text(args, tool_arg_chars)
                turns.append(f"**Assistant Tool Call:** {name}\n{rendered}\n")
                continue

            if ptype in {"function_call_output", "custom_tool_call_output"}:
                call_id = payload.get("call_id", "?")
                output = truncate_text(payload.get("output", ""), tool_output_chars)
                if output:
                    turns.append(f"**Tool Output:** {call_id}\n{output}\n")

    if not turns:
        return None

    session_id = str(meta.get("id") or path.stem)
    timestamp = parse_iso_datetime(meta.get("timestamp"), path.stat().st_mtime)
    cwd = str(meta.get("cwd") or "")
    cli_version = str(meta.get("cli_version") or "")
    originator = str(meta.get("originator") or "Codex")

    recent = turns[-FLUSH_MAX_TURNS:]
    header = [
        f"**Codex Session:** {session_id}",
        f"**Started:** {timestamp.isoformat()}",
        f"**CWD:** {cwd or '(unknown)'}",
        f"**Originator:** {originator}",
        f"**CLI Version:** {cli_version or '(unknown)'}",
        f"**Source JSONL:** {path}",
        "",
    ]
    context = "\n".join(header + recent)

    if len(context) > FLUSH_MAX_CHARS:
        elided = len(context) - FLUSH_MAX_CHARS
        context = (
            f"...[{elided} chars elided - exceeded hard cap of "
            f"{FLUSH_MAX_CHARS} chars; kept tail only]...\n\n"
            + context[-FLUSH_MAX_CHARS:]
        )

    return CodexSession(
        path=path,
        session_id=session_id,
        timestamp=timestamp,
        cwd=cwd,
        cli_version=cli_version,
        originator=originator,
        context=context,
        turn_count=len(recent),
    )


def canonical_project_name(cwd: str) -> str | None:
    if not cwd:
        return None
    try:
        top = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        common = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if top.returncode != 0:
        return None

    top_path = Path(top.stdout.strip())
    if common.returncode == 0:
        common_path = Path(common.stdout.strip())
        if not common_path.is_absolute():
            common_path = (Path(cwd) / common_path).resolve()
        if common_path.name == ".git" and common_path.parent != top_path:
            top_path = common_path.parent
    return top_path.name


def resolve_target_project(
    session: CodexSession,
    *,
    target_project: str | None,
    fallback_project: str | None,
) -> Path | None:
    if target_project:
        candidate = VAULT_ROOT / target_project
        return candidate if candidate.is_dir() else None

    project_name = canonical_project_name(session.cwd)
    if project_name:
        candidate = VAULT_ROOT / project_name
        if candidate.is_dir():
            return candidate

    if fallback_project:
        candidate = VAULT_ROOT / fallback_project
        if candidate.is_dir():
            return candidate

    return None


def discover_sessions(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def passes_date_filter(path: Path, since: datetime | None, days: int | None) -> bool:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()
    if since and mtime < since:
        return False
    if days is not None:
        cutoff = datetime.now(timezone.utc).astimezone() - timedelta(days=days)
        if mtime < cutoff:
            return False
    return True


def load_codex_state(project_dir: Path) -> dict[str, Any]:
    state_file = project_dir / ".state" / "codex-backfill.json"
    if not state_file.exists():
        return {"sessions": {}}
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"sessions": {}}
    except (json.JSONDecodeError, OSError):
        return {"sessions": {}}


def save_codex_state(project_dir: Path, state: dict[str, Any]) -> None:
    state_dir = project_dir / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "codex-backfill.json"
    tmp = state_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_file)


def already_ingested(project_dir: Path, session: CodexSession, source_hash: str) -> bool:
    state = load_codex_state(project_dir)
    prior = state.get("sessions", {}).get(session.session_id)
    if not prior or prior.get("source_hash") != source_hash:
        return False
    # Backward compatible with entries written before the status field existed.
    return prior.get("status", "ingested") == "ingested"


def previous_failure(project_dir: Path, session: CodexSession, source_hash: str) -> dict[str, Any] | None:
    state = load_codex_state(project_dir)
    prior = state.get("sessions", {}).get(session.session_id)
    if (
        isinstance(prior, dict)
        and prior.get("status") == "failed"
        and prior.get("source_hash") == source_hash
    ):
        return prior
    return None


def mark_ingested(project_dir: Path, session: CodexSession, source_hash: str) -> None:
    state = load_codex_state(project_dir)
    sessions = state.setdefault("sessions", {})
    sessions[session.session_id] = {
        "status": "ingested",
        "source": str(session.path),
        "source_hash": source_hash,
        "source_mtime": session.path.stat().st_mtime,
        "cwd": session.cwd,
        "ingested_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    save_codex_state(project_dir, state)


def mark_failed(project_dir: Path, session: CodexSession, source_hash: str, reason: str) -> None:
    state = load_codex_state(project_dir)
    sessions = state.setdefault("sessions", {})
    prior = sessions.get(session.session_id, {})
    attempts = int(prior.get("attempts", 0)) + 1 if isinstance(prior, dict) else 1
    sessions[session.session_id] = {
        "status": "failed",
        "source": str(session.path),
        "source_hash": source_hash,
        "source_mtime": session.path.stat().st_mtime,
        "cwd": session.cwd,
        "failed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "attempts": attempts,
        "reason": reason[:500],
    }
    save_codex_state(project_dir, state)


def upsert_daily_entry(project_dir: Path, session: CodexSession, extracted: str) -> Path:
    local_ts = session.timestamp.astimezone()
    daily_dir = project_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    log_path = daily_dir / f"{local_ts.strftime('%Y-%m-%d')}.md"
    if not log_path.exists():
        log_path.write_text(
            f"# Daily Log: {local_ts.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    short_id = session.session_id[:8]
    open_tag = f"<!-- cmc-codex-session: {session.session_id} -->"
    close_tag = "<!-- /cmc-codex-session -->"
    section = f"### Codex Session {short_id} ({local_ts.strftime('%H:%M')})"
    body = f"{open_tag}\n{section}\n\n{extracted.strip()}\n{close_tag}\n\n"

    existing = log_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(open_tag) + r".*?" + re.escape(close_tag) + r"\n*",
        re.DOTALL,
    )
    if pattern.search(existing):
        updated = pattern.sub(body, existing)
    else:
        updated = existing.rstrip() + "\n\n" + body
    log_path.write_text(updated, encoding="utf-8")
    return log_path


def parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise SystemExit(f"error: --since expects YYYY-MM-DD, got {value!r} ({e})")
    return parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill Codex JSONL sessions into CMC")
    parser.add_argument("--sessions-root", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--since", help="Only sessions modified since YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Only sessions modified in the last N days")
    parser.add_argument("--limit", type=int, help="Limit number of eligible sessions")
    parser.add_argument("--target-project", help="Force all sessions into this CMC vault project")
    parser.add_argument(
        "--fallback-project",
        help="CMC vault project for sessions whose cwd has no initialized project",
    )
    parser.add_argument("--force", action="store_true", help="Re-ingest even if source hash is unchanged")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry sessions marked failed for the same source hash",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="Include sessions modified recently; default skips likely active Codex rollouts",
    )
    parser.add_argument(
        "--skip-active-seconds",
        type=int,
        default=DEFAULT_SKIP_ACTIVE_SECONDS,
        help="Skip sessions modified in the last N seconds unless --include-active is set",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help="Skip extracted session contexts larger than this unless --force is set",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview routing without LLM calls or writes")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile affected CMC projects after writing Codex daily entries",
    )
    parser.add_argument(
        "--compile-timeout",
        type=int,
        default=900,
        help="Seconds to allow each post-backfill compile before giving up",
    )
    parser.add_argument("--tool-arg-chars", type=int, default=DEFAULT_TOOL_ARG_CHARS)
    parser.add_argument("--tool-output-chars", type=int, default=DEFAULT_TOOL_OUTPUT_CHARS)
    return parser


def compile_project(project_dir: Path, timeout: int) -> int:
    env = os.environ.copy()
    env["CMC_PROJECT_DIR"] = str(project_dir)
    cmd = [sys.executable, str(TOOL_DIR / "scripts" / "compile.py")]
    print(f"COMPILE {project_dir.name}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(TOOL_DIR),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"ERROR compile-timeout: {project_dir.name} >{timeout}s", file=sys.stderr)
        return 124

    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"ERROR compile: {project_dir.name} exit={result.returncode}", file=sys.stderr)
    return result.returncode


def main() -> None:
    args = build_parser().parse_args()
    sessions_root = Path(args.sessions_root).expanduser()
    since = parse_since(args.since)

    paths = [
        path for path in discover_sessions(sessions_root)
        if passes_date_filter(path, since, args.days)
    ]
    if args.limit is not None:
        paths = paths[-args.limit:]

    print(f"Codex sessions root: {sessions_root}")
    print(f"Found {len(paths)} candidate session(s)")
    if args.dry_run:
        print("[dry-run] No LLM calls and no files will be written")

    processed = 0
    skipped = 0
    written = 0
    failed = 0
    affected_projects: set[Path] = set()

    for path in paths:
        if not args.include_active and args.skip_active_seconds > 0:
            age = datetime.now(timezone.utc).astimezone().timestamp() - path.stat().st_mtime
            if age < args.skip_active_seconds:
                skipped += 1
                print(f"SKIP active: {path.name} modified {int(age)}s ago")
                continue

        session = extract_codex_session(
            path,
            tool_arg_chars=args.tool_arg_chars,
            tool_output_chars=args.tool_output_chars,
        )
        if session is None or session.turn_count < MIN_TURNS_TO_FLUSH:
            skipped += 1
            print(f"SKIP empty: {path.name}")
            continue

        target_dir = resolve_target_project(
            session,
            target_project=args.target_project,
            fallback_project=args.fallback_project,
        )
        if target_dir is None:
            skipped += 1
            print(f"SKIP no-target: {session.session_id[:8]} cwd={session.cwd or '(unknown)'}")
            continue

        if len(session.context) > args.max_context_chars and not args.force:
            skipped += 1
            print(
                f"SKIP too-large: {session.session_id[:8]} "
                f"({len(session.context)} chars > {args.max_context_chars}; use --force)"
            )
            continue

        source_hash = file_hash(path)
        if not args.force and already_ingested(target_dir, session, source_hash):
            skipped += 1
            print(f"SKIP ingested: {session.session_id[:8]} -> {target_dir.name}")
            continue
        prior_failure = previous_failure(target_dir, session, source_hash)
        if prior_failure and not (args.force or args.retry_failed):
            skipped += 1
            print(
                f"SKIP failed: {session.session_id[:8]} -> {target_dir.name} "
                f"({prior_failure.get('reason', 'previous failure')[:120]}; use --retry-failed)"
            )
            continue

        processed += 1
        print(
            f"QUEUE {session.session_id[:8]} -> {target_dir.name} "
            f"({session.turn_count} turns, {len(session.context)} chars)"
        )
        if args.dry_run:
            continue

        try:
            response, _stderr = asyncio.run(run_flush(session.context))
        except Exception as e:
            failed += 1
            reason = f"{type(e).__name__}: {e}"
            mark_failed(target_dir, session, source_hash, reason)
            print(f"ERROR flush: {session.session_id[:8]} {reason[:160]}", file=sys.stderr)
            continue
        stripped = response.strip()
        if stripped == "FLUSH_OK":
            mark_ingested(target_dir, session, source_hash)
            print(f"SKIP model-empty: {session.session_id[:8]}")
            continue
        if stripped.startswith("FLUSH_ERROR"):
            failed += 1
            mark_failed(target_dir, session, source_hash, stripped)
            print(f"ERROR flush: {session.session_id[:8]} {stripped[:160]}", file=sys.stderr)
            continue

        log_path = upsert_daily_entry(target_dir, session, stripped)
        mark_ingested(target_dir, session, source_hash)
        affected_projects.add(target_dir)
        written += 1
        print(f"WROTE {log_path}")

    print(f"\nDone. queued={processed} written={written} skipped={skipped} failed={failed}")
    if not args.dry_run and args.compile and affected_projects:
        print("\nCompiling affected project(s)...")
        for project_dir in sorted(affected_projects, key=lambda p: p.name):
            compile_project(project_dir, args.compile_timeout)
    elif not args.dry_run and written:
        print(f"Run compile for affected projects with: uv run --directory {TOOL_DIR} python scripts/compile.py")


if __name__ == "__main__":
    main()
