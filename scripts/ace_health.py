"""Read-only ACE coverage reports for Claude and Codex capture.

The historical :func:`build_report` API remains Codex-only. The CLI keeps
that mode as its default and exposes ``--source all`` for the bounded
cross-source report. Reports contain counts, project identities and safe
structural errors only; transcript content is never printed or persisted.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from backfill_codex import (
    DEFAULT_SKIP_ACTIVE_SECONDS,
    MIN_TURNS_TO_FLUSH,
    CodexSession,
    already_ingested,
    discover_sessions,
    extract_codex_session,
    file_hash,
    previous_failure,
    resolve_target_project,
    resolve_target_route,
    rollout_is_subagent,
)
from checkpoint_cursor import extract_turns_from_jsonl
from config import VAULT_ROOT, ProjectRoute, resolve_project_route


def _parse_error_lines(path: Path) -> list[int]:
    """Return malformed JSONL line numbers without retaining line contents."""
    errors: list[int] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        errors.append(number)
                except json.JSONDecodeError:
                    errors.append(number)
    except OSError:
        return []
    return errors


def _read_json_status(path: Path) -> tuple[str, dict[str, Any] | None]:
    """Read a state file as known, missing or corrupt, without repairing it."""
    if not path.exists():
        return "missing", None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "corrupt", None
    return ("known", value) if isinstance(value, dict) else ("corrupt", None)


def _codex_state_status(project: Path) -> str:
    status, _ = _read_json_status(project / ".state" / "codex-backfill.json")
    return status


def _collection_state_status(path: Path) -> tuple[str, dict[str, Any] | None]:
    return _read_json_status(path)


def _state_record_matches(
    state: dict[str, Any] | None,
    source: str,
    session_id: str,
    source_hash: str,
) -> bool:
    if not isinstance(state, dict):
        return False
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        return False
    record = sessions.get(f"{source}:{session_id}")
    if not isinstance(record, dict):
        return False
    return (
        record.get("source_hash") == source_hash
        and record.get("status") in {"ingested", "empty"}
    )


def _state_record_failed(
    state: dict[str, Any] | None,
    source: str,
    session_id: str,
    source_hash: str,
) -> bool:
    if not isinstance(state, dict) or not isinstance(state.get("sessions"), dict):
        return False
    record = state["sessions"].get(f"{source}:{session_id}")
    return (
        isinstance(record, dict)
        and record.get("source_hash") == source_hash
        and record.get("status") == "failed"
    )


def _empty_counts() -> dict[str, Any]:
    return {
        "rollouts": 0,
        "subagents_excluded": 0,
        "active": 0,
        "empty": 0,
        "no_target": 0,
        "too_large": 0,
        "ingested": 0,
        "failed": 0,
        "parse_errors": 0,
        "duplicates": 0,
        "pending": 0,
        "state_unknown": 0,
    }


def _bump(item: dict[str, Any], key: str, amount: int = 1) -> None:
    item[key] = int(item.get(key, 0)) + amount


def _finish_counts(report: dict[str, Any]) -> dict[str, Any]:
    denominator = sum(
        int(report[key])
        for key in (
            "ingested",
            "failed",
            "parse_errors",
            "pending",
            "too_large",
            "no_target",
            "state_unknown",
        )
    )
    # Empty windows and unavailable state are not proof of 100% capture.
    if denominator == 0 or int(report.get("state_unknown", 0)):
        report["coverage_percent"] = None
        report["status"] = "unknown"
    else:
        report["coverage_percent"] = round(
            100.0 * int(report["ingested"]) / denominator, 1
        )
        report["status"] = "ok" if not any(
            int(report[key])
            for key in (
                "failed",
                "parse_errors",
                "pending",
                "too_large",
                "no_target",
            )
        ) else "attention"
    return report


def _source_project_counts(report: dict[str, Any], route: ProjectRoute | None) -> dict[str, Any]:
    """Return the source-project bucket with explicit destinations."""
    source_name = route.source_project if route and route.source_project else "(unknown)"
    destination_name = route.destination_project if route else None
    projects = report.setdefault("projects", {})
    source_bucket = projects.setdefault(
        source_name,
        {
            "source_project": route.source_project if route else None,
            "source_cwds": [],
            "destinations": {},
        },
    )
    source_cwd = route.source_cwd if route else ""
    if source_cwd and source_cwd not in source_bucket["source_cwds"]:
        source_bucket["source_cwds"].append(source_cwd)
    destination = destination_name or "(unknown)"
    return source_bucket["destinations"].setdefault(destination, _empty_counts())


def build_report(
    sessions_root: Path,
    *,
    days: int,
    fallback_project: str | None,
    skip_active_seconds: int,
    max_context_chars: int,
) -> dict[str, Any]:
    """Build the historical Codex-only report with its legacy signature."""
    cutoff = datetime.now(timezone.utc).astimezone() - timedelta(days=days)
    report: dict[str, Any] = {"source": "codex", "window_days": days}
    report.update(_empty_counts())
    now = datetime.now(timezone.utc).astimezone().timestamp()
    seen_sessions: dict[str, set[str]] = {}

    for path in discover_sessions(sessions_root):
        try:
            stat = path.stat()
        except OSError:
            continue
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).astimezone()
        if modified < cutoff:
            continue
        _bump(report, "rollouts")
        if rollout_is_subagent(path):
            _bump(report, "subagents_excluded")
            continue
        if now - stat.st_mtime < skip_active_seconds:
            _bump(report, "active")
            continue
        if _parse_error_lines(path):
            _bump(report, "parse_errors")
            continue

        session = extract_codex_session(path, tool_arg_chars=200, tool_output_chars=200)
        if session is None or session.turn_count < MIN_TURNS_TO_FLUSH:
            _bump(report, "empty")
            continue
        source_hash = file_hash(path)
        hashes = seen_sessions.setdefault(session.session_id, set())
        if source_hash in hashes:
            _bump(report, "duplicates")
            continue
        hashes.add(source_hash)

        route = resolve_target_route(
            session, target_project=None, fallback_project=fallback_project
        )
        target = route.destination_dir
        if target is None:
            _bump(report, "no_target")
            continue

        state_status = _codex_state_status(target)
        # An exact successful hash wins before the health-only context bound:
        # collector and worker limits intentionally differ.
        if state_status == "known" and already_ingested(target, session, source_hash):
            _bump(report, "ingested")
            continue
        if state_status != "known":
            _bump(report, "state_unknown")
            continue
        if len(session.context) > max_context_chars:
            _bump(report, "too_large")
            continue
        if previous_failure(target, session, source_hash):
            _bump(report, "failed")
        else:
            _bump(report, "pending")

    return _finish_counts(report)


def _extract_claude_session(path: Path) -> CodexSession | None:
    """Extract only structural Claude metadata and bounded turn count."""
    turns = extract_turns_from_jsonl(path)
    if not turns:
        return None
    cwd = ""
    session_id = path.stem
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cwd = cwd or str(entry.get("cwd") or "")
                session_id = str(entry.get("sessionId") or session_id)
                if cwd and session_id:
                    break
    except OSError:
        return None
    return CodexSession(
        session_id=session_id,
        path=path,
        cwd=cwd,
        timestamp=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
        cli_version="",
        originator="Claude",
        is_subagent=False,
        context="\n".join(turns),
        turn_count=len(turns),
    )


def _source_paths(
    source: str,
    *,
    codex_root: Path,
    claude_roots: Iterable[Path],
    cutoff: datetime,
) -> list[Path]:
    roots = [codex_root] if source == "codex" else list(claude_roots)
    paths: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            try:
                modified = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).astimezone()
            except OSError:
                continue
            if modified >= cutoff:
                paths[str(path.resolve())] = path
    return sorted(paths.values(), key=lambda item: (item.stat().st_mtime, str(item)))


def build_source_report(
    source: str,
    *,
    codex_root: Path,
    claude_roots: Iterable[Path],
    days: int,
    fallback_project: str | None,
    skip_active_seconds: int,
    max_context_chars: int,
    collection_state_file: Path | None = None,
) -> dict[str, Any]:
    """Build a bounded report for one source, grouped by destination project."""
    if source not in {"codex", "claude"}:
        raise ValueError("source must be codex or claude")
    cutoff = datetime.now(timezone.utc).astimezone() - timedelta(days=days)
    report: dict[str, Any] = {"source": source, "window_days": days}
    report.update(_empty_counts())
    report["projects"] = {}
    collection_status, collection_state = (
        _collection_state_status(collection_state_file)
        if collection_state_file is not None
        else ("missing", None)
    )
    now = datetime.now(timezone.utc).astimezone().timestamp()
    seen_sessions: dict[str, set[str]] = {}
    for path in _source_paths(
        source,
        codex_root=codex_root,
        claude_roots=claude_roots,
        cutoff=cutoff,
    ):
        _bump(report, "rollouts")
        if source == "codex" and rollout_is_subagent(path):
            _bump(report, "subagents_excluded")
            continue
        if now - path.stat().st_mtime < skip_active_seconds:
            _bump(report, "active")
            continue
        if _parse_error_lines(path):
            _bump(report, "parse_errors")
            continue
        session = (
            extract_codex_session(path, tool_arg_chars=200, tool_output_chars=200)
            if source == "codex"
            else _extract_claude_session(path)
        )
        if session is None or (source == "codex" and session.turn_count < MIN_TURNS_TO_FLUSH):
            _bump(report, "empty")
            continue
        source_hash = file_hash(path)
        hashes = seen_sessions.setdefault(session.session_id, set())
        if source_hash in hashes:
            _bump(report, "duplicates")
            continue
        hashes.add(source_hash)

        route = resolve_project_route(
            session.cwd,
            fallback_project=fallback_project,
            vault_root=VAULT_ROOT,
        )
        project = _source_project_counts(report, route)
        _bump(project, "rollouts")
        if route.destination_dir is None:
            _bump(report, "no_target")
            _bump(project, "no_target")
            continue

        ingested = False
        failed = False
        if collection_status == "known":
            ingested = _state_record_matches(
                collection_state, source, session.session_id, source_hash
            )
        elif collection_status in {"missing", "corrupt"} and source == "claude":
            _bump(report, "state_unknown")
            _bump(project, "state_unknown")
            continue
        if source == "codex" and not ingested:
            backfill_status = _codex_state_status(route.destination_dir)
            if backfill_status == "known":
                ingested = already_ingested(route.destination_dir, session, source_hash)
                failed = previous_failure(route.destination_dir, session, source_hash) is not None
            elif collection_status != "known":
                _bump(report, "state_unknown")
                _bump(project, "state_unknown")
                continue
        if source == "codex" and collection_status == "known" and not ingested:
            failed = _state_record_failed(
                collection_state, source, session.session_id, source_hash
            )
        if ingested:
            _bump(report, "ingested")
            _bump(project, "ingested")
            continue
        if len(session.context) > max_context_chars:
            _bump(report, "too_large")
            _bump(project, "too_large")
            continue
        if failed or (collection_status == "known" and _state_record_failed(
            collection_state, source, session.session_id, source_hash
        )):
            _bump(report, "failed")
            _bump(project, "failed")
        else:
            _bump(report, "pending")
            _bump(project, "pending")

    _finish_counts(report)
    for source_bucket in report["projects"].values():
        for destination in source_bucket.get("destinations", {}).values():
            _finish_counts(destination)
    return report


def build_all_sources_report(
    *,
    codex_root: Path,
    claude_roots: Iterable[Path],
    days: int,
    fallback_project: str | None,
    skip_active_seconds: int,
    max_context_chars: int,
    collection_state_file: Path | None = None,
) -> dict[str, Any]:
    """Build the opt-in cross-source report without writing state."""
    sources = {
        source: build_source_report(
            source,
            codex_root=codex_root,
            claude_roots=claude_roots,
            days=days,
            fallback_project=fallback_project,
            skip_active_seconds=skip_active_seconds,
            max_context_chars=max_context_chars,
            collection_state_file=collection_state_file,
        )
        for source in ("codex", "claude")
    }
    report: dict[str, Any] = {
        "source": "all",
        "window_days": days,
        "sources": sources,
        "projects": {},
    }
    for source_report in sources.values():
        for name, values in source_report.get("projects", {}).items():
            bucket = report["projects"].setdefault(
                name,
                {
                    "source_project": values.get("source_project"),
                    "source_cwds": [],
                    "destinations": {},
                    "sources": {},
                },
            )
            for cwd in values.get("source_cwds", []):
                if cwd not in bucket["source_cwds"]:
                    bucket["source_cwds"].append(cwd)
            for destination, counts in values.get("destinations", {}).items():
                existing = bucket["destinations"].setdefault(
                    destination, _empty_counts()
                )
                for key in _empty_counts():
                    existing[key] += int(counts.get(key, 0))
            bucket["sources"][source_report["source"]] = values
    totals = _empty_counts()
    for source_report in sources.values():
        for key in totals:
            totals[key] += int(source_report.get(key, 0))
    report.update(totals)
    _finish_counts(report)
    # A successful source cannot turn an empty or unavailable sibling source
    # into a false global 100%.
    if any(values.get("status") == "unknown" for values in sources.values()):
        report["coverage_percent"] = None
        report["status"] = "unknown"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Report ACE capture coverage without writes")
    parser.add_argument(
        "--source",
        choices=("codex", "claude", "all"),
        default="codex",
        help="Coverage source (default: codex; use all for Claude + Codex)",
    )
    parser.add_argument(
        "--sessions-root",
        "--codex-root",
        dest="codex_root",
        default=str(Path.home() / ".codex" / "sessions"),
    )
    parser.add_argument("--claude-root", default=str(Path.home() / ".claude" / "projects"))
    parser.add_argument(
        "--ccs-root",
        default=str(Path.home() / ".ccs" / "shared/context-groups/default/projects"),
    )
    parser.add_argument(
        "--state-file", default=str(Path.home() / ".codex/ace/collection-state.json")
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--fallback-project",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-active-seconds", type=int, default=DEFAULT_SKIP_ACTIVE_SECONDS)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=int(os.environ.get("ACE_CODEX_MAX_CONTEXT_CHARS", "120000")),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    days = max(0, args.days)
    skip_active_seconds = max(0, args.skip_active_seconds)
    max_context_chars = max(1, args.max_context_chars)
    codex_root = Path(args.codex_root).expanduser()
    claude_roots = [Path(args.claude_root).expanduser(), Path(args.ccs_root).expanduser()]
    state_file = Path(args.state_file).expanduser()

    if args.source == "codex":
        report = build_report(
            codex_root,
            days=days,
            fallback_project=args.fallback_project or None,
            skip_active_seconds=skip_active_seconds,
            max_context_chars=max_context_chars,
        )
    elif args.source == "claude":
        report = build_source_report(
            "claude",
            codex_root=codex_root,
            claude_roots=claude_roots,
            days=days,
            fallback_project=args.fallback_project or None,
            skip_active_seconds=skip_active_seconds,
            max_context_chars=max_context_chars,
            collection_state_file=state_file,
        )
    else:
        report = build_all_sources_report(
            codex_root=codex_root,
            claude_roots=claude_roots,
            days=days,
            fallback_project=args.fallback_project or None,
            skip_active_seconds=skip_active_seconds,
            max_context_chars=max_context_chars,
            collection_state_file=state_file,
        )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"ACE {args.source} coverage, last {report['window_days']} day(s): "
            f"{report['status']}"
        )
        for key in (
            "rollouts",
            "subagents_excluded",
            "active",
            "empty",
            "no_target",
            "too_large",
            "ingested",
            "failed",
            "parse_errors",
            "duplicates",
            "pending",
            "state_unknown",
            "coverage_percent",
        ):
            if key in report:
                print(f"  {key}: {report[key]}")
        if args.source == "all":
            for source, values in report["sources"].items():
                print(f"  {source}: {values['status']} ({values['rollouts']} rollout(s))")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
