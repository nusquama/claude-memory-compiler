"""
Backfill Codex Desktop/CLI sessions into ACE daily logs.

Codex stores rollouts under ~/.codex/sessions/YYYY/MM/DD/*.jsonl with a
different schema from Claude Code transcripts. This script converts those
JSONL records into the same conversation-context shape consumed by flush.py,
then writes the extracted daily-log entry into the matching ACE vault project.

Usage:
    uv run python scripts/backfill_codex.py --dry-run
    uv run python scripts/backfill_codex.py --since 2026-05-30 --dry-run
    uv run python scripts/backfill_codex.py --target-project Conversations
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Codex rollouts are extracted by the Codex CLI itself. Set these before
# importing config/flush.py because those modules read environment defaults
# at import time. The shared flush default is Codex for Claude sessions too.
os.environ.setdefault("ACE_FLUSH_ENGINE", "codex")
os.environ.setdefault("ACE_CODEX_MODEL", "gpt-5.6-luna")
os.environ.setdefault("ACE_CODEX_REASONING_EFFORT", "medium")

from config import (
    FLUSH_MAX_CHARS,
    FLUSH_MAX_TURNS,
    TOOL_DIR,
    VAULT_ROOT,
    ProjectRoute,
    canonical_project_name,
    resolve_project_route,
)
from checkpoint_cursor import bounded_redacted_text
from utils import redact_sensitive_text

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

# Historical backfills should fail fast. Interactive hooks keep the defaults
# from flush.py; this script may process many sessions and must not spend
# several minutes retrying one stale rollout.
os.environ.setdefault("ACE_FLUSH_MAX_RETRIES", "1")
os.environ.setdefault("ACE_FLUSH_ATTEMPT_TIMEOUT", "120")

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
    is_subagent: bool
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
    """Return bounded, secret-redacted head/tail evidence for tool data."""
    return bounded_redacted_text(value, limit)


def _scalar_ref(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return redact_sensitive_text(str(value).strip())


def source_reference(entry: dict[str, Any], payload: dict[str, Any] | None = None) -> str:
    """Preserve stable rollout refs without dumping arbitrary metadata."""
    payload = payload or {}
    fields = (
        ("source_ref", "source_ref"),
        ("uuid", "source_ref"),
        ("id", "source_ref"),
        ("source", "source"),
        ("parentUuid", "parent_ref"),
        ("parent_id", "parent_ref"),
    )
    refs: list[str] = []
    seen: set[str] = set()
    for key, label in fields:
        value = _scalar_ref(payload.get(key)) or _scalar_ref(entry.get(key))
        if value and f"{label}={value}" not in seen:
            refs.append(f"{label}={value}")
            seen.add(f"{label}={value}")
    return " ".join(refs)


def _call_id(payload: dict[str, Any], entry: dict[str, Any] | None = None) -> str:
    entry = entry or {}
    for key in ("call_id", "tool_use_id", "tool_call_id", "id"):
        value = _scalar_ref(payload.get(key)) or _scalar_ref(entry.get(key))
        if value:
            return value
    return "?"


def _observed_status(payload: dict[str, Any], output: str = "") -> str:
    explicit = payload.get("status")
    if explicit:
        return _scalar_ref(explicit) or "observed"
    if payload.get("is_error") or payload.get("isError"):
        return "error"
    exit_code = payload.get("exit_code", payload.get("returncode"))
    if exit_code is not None:
        try:
            return "success" if int(exit_code) == 0 else "error"
        except (TypeError, ValueError):
            pass
    lowered = output.lower()
    if any(marker in lowered for marker in ("error", "exception", "traceback", "failed", "failure")):
        return "error"
    # A normal-looking output is still not independent proof of success.
    return "observed"


def _visible_assistant_message(payload: dict[str, Any]) -> bool:
    """Keep only user-visible final/commentary assistant messages.

    Rollout schemas differ: some use ``channel`` and others use ``phase``;
    older records omit both.  Missing metadata remains compatible, while an
    explicit channel must be a user-visible final/commentary alias. Internal
    and unknown channels are never copied into the knowledge base.
    """
    markers = [payload.get("channel"), payload.get("phase")]
    allowed = {"final", "commentary", "output", "result"}
    internal = {"analysis", "reasoning", "thinking", "internal", "hidden"}
    for marker in markers:
        if marker is None:
            continue
        normalized = str(marker).strip().lower()
        if normalized in internal:
            return False
        if normalized in allowed:
            continue
        return False
    return True


def is_subagent_source(source: Any) -> bool:
    """Return true for Codex rollouts created by delegated subagents."""
    if not isinstance(source, dict):
        return False
    if "subagent" in source or "thread_spawn" in source:
        return True
    return any(is_subagent_source(value) for value in source.values())


def rollout_is_subagent(path: Path) -> bool:
    """Inspect only session metadata so subagents can be skipped cheaply."""
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "session_meta":
                    payload = entry.get("payload", {})
                    return is_subagent_source(payload.get("source"))
    except OSError:
        return False
    return False


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
                parts.append(str(text))
    return "\n".join(p.strip() for p in parts if p and p.strip())


def extract_codex_session(
    path: Path,
    *,
    tool_arg_chars: int,
    tool_output_chars: int,
) -> CodexSession | None:
    meta: dict[str, Any] = {}
    source_model: str | None = None
    source_effort: str | None = None
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
            if not isinstance(entry, dict):
                continue

            etype = entry.get("type")
            payload = entry.get("payload", {})
            if etype == "session_meta" and isinstance(payload, dict) and not meta:
                meta = payload
                source_model = _scalar_ref(payload.get("model")) or source_model
                source_effort = (
                    _scalar_ref(
                        payload.get("reasoning_effort", payload.get("effort"))
                    )
                    or source_effort
                )
                continue
            if etype == "turn_context" and isinstance(payload, dict):
                # These fields describe the source Codex rollout.  They are
                # intentionally distinct from the model used later by
                # flush.py and are omitted when the source did not provide
                # them.
                source_model = _scalar_ref(payload.get("model")) or source_model
                source_effort = (
                    _scalar_ref(
                        payload.get(
                            "reasoning_effort",
                            payload.get("model_reasoning_effort", payload.get("effort")),
                        )
                    )
                    or source_effort
                )
                continue
            if etype != "response_item" or not isinstance(payload, dict):
                continue

            ptype = payload.get("type")
            role = payload.get("role")

            if ptype == "message":
                if role not in {"user", "assistant"}:
                    continue
                if role == "assistant" and not _visible_assistant_message(payload):
                    continue
                text = text_from_content_blocks(payload.get("content"))
                if not text:
                    continue
                label = "User" if role == "user" else "Assistant"
                refs = source_reference(entry, payload)
                ref_text = f" [{refs}]" if refs else ""
                turns.append(f"**{label}{ref_text}:** {redact_sensitive_text(text)}\n")
                continue

            if ptype in {"function_call", "custom_tool_call"}:
                name = _scalar_ref(payload.get("name")) or "?"
                args = payload.get("arguments", payload.get("input", ""))
                rendered = truncate_text(args, tool_arg_chars)
                call_id = _call_id(payload, entry)
                refs = source_reference(entry, payload)
                ref_text = f" {refs}" if refs else ""
                tool_role = _scalar_ref(role) or "assistant"
                turns.append(
                    f"**Assistant Tool Call role={tool_role} call_id={call_id}{ref_text}:** "
                    f"{name}\n{rendered}\n"
                )
                continue

            if ptype in {"function_call_output", "custom_tool_call_output"}:
                output = truncate_text(payload.get("output", ""), tool_output_chars)
                call_id = _call_id(payload, entry)
                refs = source_reference(entry, payload)
                ref_text = f" {refs}" if refs else ""
                tool_role = _scalar_ref(role) or "tool"
                status = _observed_status(payload, output)
                if output or status == "error" or payload.get("status"):
                    turns.append(
                        f"**Tool Output role={tool_role} status={status} call_id={call_id}"
                        f"{ref_text}:**\n{output}\n"
                    )

    if not turns:
        return None

    session_id = str(meta.get("id") or path.stem)
    timestamp = parse_iso_datetime(meta.get("timestamp"), path.stat().st_mtime)
    cwd = str(meta.get("cwd") or "")
    cli_version = str(meta.get("cli_version") or "")
    originator = str(meta.get("originator") or "Codex")
    subagent = is_subagent_source(meta.get("source"))

    recent = turns[-FLUSH_MAX_TURNS:]
    header = [
        f"**Codex Session:** {session_id}",
        f"**Started:** {timestamp.isoformat()}",
        f"**CWD:** {cwd or '(unknown)'}",
        f"**Originator:** {originator}",
        f"**CLI Version:** {cli_version or '(unknown)'}",
        f"**Source JSONL:** {path}",
    ]
    if source_model:
        header.append(f"**Source Model:** {source_model}")
    if source_effort:
        header.append(f"**Source Reasoning Effort:** {source_effort}")
    header.append("")
    context = redact_sensitive_text("\n".join(header + recent))

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
        is_subagent=subagent,
        context=context,
        turn_count=len(recent),
    )


def resolve_target_route(
    session: CodexSession,
    *,
    target_project: str | None,
    fallback_project: str | None,
) -> ProjectRoute:
    """Resolve only an initialized source or an explicitly initialized target."""
    if fallback_project:
        raise ValueError("fallback projects are disabled; initialize the source project")
    return resolve_project_route(
        session.cwd,
        target_project=target_project,
        fallback_project=None,
        vault_root=VAULT_ROOT,
    )


def resolve_target_project(
    session: CodexSession,
    *,
    target_project: str | None,
    fallback_project: str | None,
) -> Path | None:
    """Backward-compatible destination-only wrapper."""
    return resolve_target_route(
        session,
        target_project=target_project,
        fallback_project=fallback_project,
    ).destination_dir


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
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_file = state_dir / "codex-backfill.json"
    tmp = state_file.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(state_file)
    finally:
        tmp.unlink(missing_ok=True)


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


def _route_state_fields(route: ProjectRoute | None, session: CodexSession) -> dict[str, Any]:
    """Return non-content routing metadata for durable state records."""
    if route is None:
        return {
            "source_project": canonical_project_name(session.cwd),
            "source_cwd": session.cwd,
            "destination_project": None,
        }
    return {
        "source_project": route.source_project,
        "source_cwd": route.source_cwd,
        "destination_project": route.destination_project,
        "used_fallback": route.used_fallback,
        "route_reason": route.reason,
    }


def mark_ingested(
    project_dir: Path,
    session: CodexSession,
    source_hash: str,
    route: ProjectRoute | None = None,
) -> None:
    state = load_codex_state(project_dir)
    sessions = state.setdefault("sessions", {})
    record = {
        "status": "ingested",
        "source": str(session.path),
        "source_hash": source_hash,
        "source_mtime": session.path.stat().st_mtime,
        "cwd": session.cwd,
        "ingested_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    record.update(_route_state_fields(route, session))
    sessions[session.session_id] = record
    save_codex_state(project_dir, state)


def mark_failed(
    project_dir: Path,
    session: CodexSession,
    source_hash: str,
    reason: str,
    route: ProjectRoute | None = None,
) -> None:
    state = load_codex_state(project_dir)
    sessions = state.setdefault("sessions", {})
    prior = sessions.get(session.session_id, {})
    attempts = int(prior.get("attempts", 0)) + 1 if isinstance(prior, dict) else 1
    record = {
        "status": "failed",
        "source": str(session.path),
        "source_hash": source_hash,
        "source_mtime": session.path.stat().st_mtime,
        "cwd": session.cwd,
        "failed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "attempts": attempts,
        "reason": redact_sensitive_text(reason)[:500],
    }
    record.update(_route_state_fields(route, session))
    sessions[session.session_id] = record
    save_codex_state(project_dir, state)


@contextmanager
def _daily_log_lock(log_path: Path):
    """Serialize Codex workers updating one daily log on this host."""
    lock_path = log_path.with_name(f".{log_path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a daily log atomically using a unique same-directory temp."""
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=f".{os.getpid()}.tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def upsert_daily_entry(project_dir: Path, session: CodexSession, extracted: str) -> Path:
    local_ts = session.timestamp.astimezone()
    daily_dir = project_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    log_path = daily_dir / f"{local_ts.strftime('%Y-%m-%d')}.md"
    with _daily_log_lock(log_path):
        if not log_path.exists():
            initial = (
                f"# Daily Log: {local_ts.strftime('%Y-%m-%d')}\n\n"
                "## Sessions\n\n## Memory Maintenance\n\n"
            )
            _atomic_write_text(log_path, initial)

        short_id = session.session_id[:8]
        open_tag = f"<!-- ace-codex-session: {session.session_id} -->"
        close_tag = "<!-- /ace-codex-session -->"
        legacy_open_tag = f"<!-- cmc-codex-session: {session.session_id} -->"
        legacy_close_tag = "<!-- /cmc-codex-session -->"
        section = f"### Codex Session {short_id} ({local_ts.strftime('%H:%M')})"
        safe_extracted = redact_sensitive_text(extracted.strip())
        body = f"{open_tag}\n{section}\n\n{safe_extracted}\n{close_tag}\n\n"

        existing = log_path.read_text(encoding="utf-8")
        updated = existing
        for marker_start, marker_end in (
            (open_tag, close_tag),
            (legacy_open_tag, legacy_close_tag),
        ):
            pattern = re.compile(
                re.escape(marker_start) + r".*?" + re.escape(marker_end) + r"\n*",
                re.DOTALL,
            )
            if pattern.search(updated):
                # Use a lambda: extracted text may contain backslashes that
                # are meaningful to re.sub replacement templates.
                updated = pattern.sub(lambda _match: body, updated, count=1)
                break
        else:
            updated = existing.rstrip() + "\n\n" + body
        _atomic_write_text(log_path, updated)
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
    parser = argparse.ArgumentParser(description="Backfill Codex JSONL sessions into ACE")
    parser.add_argument("--sessions-root", default=str(Path.home() / ".codex" / "sessions"))
    parser.add_argument("--since", help="Only sessions modified since YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="Only sessions modified in the last N days")
    parser.add_argument("--limit", type=int, help="Limit stable, non-subagent sessions")
    parser.add_argument("--target-project", help="Force all sessions into this ACE vault project")
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
        "--include-subagents",
        action="store_true",
        help="Include delegated Codex subagent rollouts; excluded by default",
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
        help="Compile affected ACE projects after writing Codex daily entries",
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
    env["ACE_PROJECT_DIR"] = str(project_dir)
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
        # Start with the newest rollouts, but count only sessions that reach an
        # LLM call. Active, subagent, oversized, ingested, and failed entries do
        # not consume the user's requested processing limit.
        paths = list(reversed(paths))

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
        if args.limit is not None and processed >= args.limit:
            break
        if not args.include_active and args.skip_active_seconds > 0:
            age = datetime.now(timezone.utc).astimezone().timestamp() - path.stat().st_mtime
            if age < args.skip_active_seconds:
                skipped += 1
                print(f"SKIP active: {path.name} modified {int(age)}s ago")
                continue

        if not args.include_subagents and rollout_is_subagent(path):
            skipped += 1
            print(f"SKIP subagent: {path.name}")
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

        if session.is_subagent and not args.include_subagents:
            skipped += 1
            print(f"SKIP subagent: {session.session_id[:8]}")
            continue

        route = resolve_target_route(
            session,
            target_project=args.target_project,
            fallback_project=None,
        )
        target_dir = route.destination_dir
        if target_dir is None:
            skipped += 1
            print(
                f"SKIP no-target: {session.session_id[:8]} "
                f"source={route.source_project or '(unknown)'} "
                f"cwd={session.cwd or '(unknown)'}"
            )
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
        fallback_note = " fallback" if route.used_fallback else ""
        print(
            f"QUEUE {session.session_id[:8]} -> {target_dir.name}{fallback_note} "
            f"({session.turn_count} turns, {len(session.context)} chars)"
        )
        if args.dry_run:
            continue

        try:
            response, _stderr = asyncio.run(run_flush(session.context))
        except Exception as e:
            failed += 1
            reason = redact_sensitive_text(f"{type(e).__name__}: {e}")
            mark_failed(target_dir, session, source_hash, reason, route)
            print(f"ERROR flush: {session.session_id[:8]} {reason[:160]}", file=sys.stderr)
            continue
        # The model output crosses the vault boundary here. Redact it again
        # after the runner returns, before any daily log write.
        stripped = redact_sensitive_text(response.strip())
        if stripped == "FLUSH_OK":
            mark_ingested(target_dir, session, source_hash, route)
            print(f"SKIP model-empty: {session.session_id[:8]}")
            continue
        if stripped.startswith("FLUSH_ERROR"):
            failed += 1
            safe_error = redact_sensitive_text(stripped)
            mark_failed(target_dir, session, source_hash, safe_error, route)
            print(f"ERROR flush: {session.session_id[:8]} {safe_error[:160]}", file=sys.stderr)
            continue

        log_path = upsert_daily_entry(target_dir, session, stripped)
        mark_ingested(target_dir, session, source_hash, route)
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
