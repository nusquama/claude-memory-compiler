"""Bounded periodic Claude/Codex collection. Checkpoint only durable outcomes."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from backfill_codex import (
    CodexSession,
    extract_codex_session,
    canonical_project_name,
    file_hash,
    already_ingested,
    mark_ingested,
    upsert_daily_entry,
)
from checkpoint_cursor import extract_turns_from_jsonl
from config import ProjectRoute, VAULT_ROOT, resolve_project_route
from flush import run_flush
from utils import redact_sensitive_text

DEFAULT_STATE = Path.home() / '.codex/ace/collection-state.json'
cached_project_name = lru_cache(maxsize=256)(canonical_project_name)


def _jsonl_parse_errors(path: Path) -> list[int]:
    """Return malformed JSONL line numbers without exposing line contents."""
    errors: list[int] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        errors.append(line_number)
                except json.JSONDecodeError:
                    errors.append(line_number)
    except OSError:
        return []
    return errors


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.collect-')
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_state(path: Path) -> dict:
    if not path.exists():
        return {'version': 1, 'sessions': {}}
    state = json.loads(path.read_text())
    if not isinstance(state.get('sessions'), dict):
        raise ValueError('invalid collection state; preserved without reset')
    return state


def _candidate_key(source: str, path: Path) -> tuple[str, str]:
    return source, str(path.resolve())


def _pending_candidate_keys(state: dict) -> set[tuple[str, str]]:
    """Return paths that need a retry or remain in the durable backlog."""
    pending: set[tuple[str, str]] = set()
    sessions = state.get('sessions', {})
    if isinstance(sessions, dict):
        for record in sessions.values():
            if not isinstance(record, dict) or record.get('status') not in {'failed', 'deferred'}:
                continue
            source = record.get('source')
            raw_path = record.get('path')
            if source and raw_path:
                pending.add(_candidate_key(str(source), Path(str(raw_path))))
    for record in state.get('backlog', []):
        if not isinstance(record, dict):
            continue
        source = record.get('source')
        raw_path = record.get('path')
        if source and raw_path:
            pending.add(_candidate_key(str(source), Path(str(raw_path))))
    return pending


def _fair_candidate_order(
    candidates: list[tuple[str, Path]], state: dict, cutoff: float
) -> list[tuple[str, Path]]:
    """Interleave fresh work, retries, and older backlog deterministically.

    New sessions get a turn between retry/backlog items. A persisted cursor
    rotates the first bucket across collection runs, so a steady stream of new
    sessions cannot starve older pending work when the batch limit is one.
    """
    retry_keys = _pending_candidate_keys(state)
    backlog_keys = {
        _candidate_key(str(record.get('source')), Path(str(record.get('path'))))
        for record in state.get('backlog', [])
        if isinstance(record, dict) and record.get('source') and record.get('path')
    }
    queues: dict[str, list[tuple[str, Path]]] = {'new': [], 'retry': [], 'old': []}
    for source, path in candidates:
        key = _candidate_key(source, path)
        if key in retry_keys and key not in backlog_keys:
            bucket = 'retry'
        elif key in backlog_keys or path.stat().st_mtime < cutoff:
            bucket = 'old'
        else:
            bucket = 'new'
        queues[bucket].append((source, path))

    queues['new'].sort(key=lambda item: (-item[1].stat().st_mtime, str(item[1])))
    queues['retry'].sort(key=lambda item: (item[1].stat().st_mtime, str(item[1])))
    queues['old'].sort(key=lambda item: (item[1].stat().st_mtime, str(item[1])))

    present = [name for name in ('new', 'retry', 'old') if queues[name]]
    if not present:
        return []
    try:
        cursor = int(state.get('selection_cursor', 0) or 0) % len(present)
    except (TypeError, ValueError):
        cursor = 0
    state['selection_cursor'] = (cursor + 1) % len(present)

    ordered: list[tuple[str, Path]] = []
    while any(queues.values()):
        for offset in range(len(present)):
            bucket = present[(cursor + offset) % len(present)]
            if queues[bucket]:
                ordered.append(queues[bucket].pop(0))
    return ordered


def _pending_coverage(state: dict, candidates: list[tuple[str, Path]]) -> dict:
    """Expose current pending age without changing historical daily logs."""
    pending: dict[tuple[str, str], float] = {}
    sessions = state.get('sessions', {})
    if isinstance(sessions, dict):
        records = [record for record in sessions.values() if isinstance(record, dict)]
    else:
        records = []
    records.extend(record for record in state.get('backlog', []) if isinstance(record, dict))
    for record in records:
        if record.get('status') not in {'failed', 'deferred', 'unexamined'}:
            continue
        source = record.get('source')
        raw_path = record.get('path')
        if not source or not raw_path:
            continue
        path = Path(str(raw_path))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        pending[_candidate_key(str(source), path)] = mtime

    now = time.time()
    coverage: dict[str, object] = {
        'pending_count': len(pending),
        'pending_oldest_mtime': None,
        'pending_oldest_age_seconds': None,
        'freshest_candidate_mtime': None,
        'freshest_candidate_age_seconds': None,
    }
    if pending:
        oldest = min(pending.values())
        coverage['pending_oldest_mtime'] = datetime.fromtimestamp(
            oldest, timezone.utc
        ).astimezone().isoformat(timespec='seconds')
        coverage['pending_oldest_age_seconds'] = max(0, int(now - oldest))
    candidate_mtimes = [path.stat().st_mtime for _, path in candidates if path.is_file()]
    if candidate_mtimes:
        newest = max(candidate_mtimes)
        coverage['freshest_candidate_mtime'] = datetime.fromtimestamp(
            newest, timezone.utc
        ).astimezone().isoformat(timespec='seconds')
        coverage['freshest_candidate_age_seconds'] = max(0, int(now - newest))
    return coverage


def discover(roots: list[tuple[str, Path]], days: int, state: dict) -> list[tuple[str, Path]]:
    """Keep failed/deferred known sources eligible after the lookback window."""
    cutoff = time.time() - days * 86400
    candidates = {}
    for source, root in roots:
        for path in root.rglob('*.jsonl') if root.exists() else []:
            if 'subagents' in path.parts or path.name.startswith('agent-'):
                continue
            if path.stat().st_mtime >= cutoff:
                candidates[(source, str(path.resolve()))] = (source, path)
    sessions = state.get('sessions', {})
    for record in sessions.values() if isinstance(sessions, dict) else []:
        if not isinstance(record, dict) or not record.get('path'):
            continue
        path = Path(record['path'])
        if record.get('status') in {'failed', 'deferred'} and path.is_file():
            source = str(record.get('source') or 'codex')
            candidates[(source, str(path.resolve()))] = (source, path)
    for record in state.get('backlog', []):
        if not isinstance(record, dict) or not record.get('path'):
            continue
        path = Path(record['path'])
        if path.is_file():
            source = str(record.get('source') or 'codex')
            candidates[(source, str(path.resolve()))] = (source, path)
    return _fair_candidate_order(list(candidates.values()), state, cutoff)


def extract(source: str, path: Path) -> CodexSession | None:
    if source == 'codex':
        return extract_codex_session(path, tool_arg_chars=1600, tool_output_chars=4000)
    turns = extract_turns_from_jsonl(path)
    if not turns:
        return None
    cwd, session_id = '', path.stem
    with path.open() as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cwd = cwd or entry.get('cwd', '')
            session_id = entry.get('sessionId') or session_id
            if cwd:
                break
    context = redact_sensitive_text('\n'.join(turns))
    header = f'Claude session: {session_id}\nSource: {path}\n'
    return CodexSession(session_id=str(session_id), path=path, cwd=cwd,
                        timestamp=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
                        cli_version='', originator='Claude', is_subagent=False,
                        context=header + context, turn_count=len(turns))


def project_route(session: CodexSession, fallback: str | None = None) -> ProjectRoute:
    """Resolve source identity and initialized destination consistently."""
    return resolve_project_route(
        session.cwd,
        fallback_project=None,
        vault_root=VAULT_ROOT,
    )


def target_project(session: CodexSession, fallback: str | None = None) -> Path:
    """Backward-compatible destination-only wrapper."""
    route = project_route(session, fallback)
    if route.destination_dir is None:
        raise ValueError("no initialized source project")
    return route.destination_dir


def store_claude(project: Path, session: CodexSession, body: str, previous: dict) -> Path:
    # Keep the original daily location when a conversation grows on a later day.
    body = redact_sensitive_text(body)
    filename = previous.get('daily_file') or session.timestamp.strftime('%Y-%m-%d') + '.md'
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}\.md', filename):
        raise ValueError('invalid daily filename')
    daily = project / 'daily' / filename
    daily.parent.mkdir(parents=True, exist_ok=True)
    sid = hashlib.sha256(session.session_id.encode()).hexdigest()[:24]
    start, end = f'<!-- ace-claude-session: {sid} -->', '<!-- /ace-claude-session -->'
    legacy_start, legacy_end = f'<!-- cmc-claude-session: {sid} -->', '<!-- /cmc-claude-session -->'
    section = f'{start}\n### Claude Session {session.session_id[:8]}\n\n{body}\n{end}\n'
    with daily.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = daily.read_text() if daily.exists() else f'# Daily Log: {filename[:-3]}\n'
        patterns = (
            (start, end),
            (legacy_start, legacy_end),
        )
        for marker_start, marker_end in patterns:
            pattern = re.compile(re.escape(marker_start) + '.*?' + re.escape(marker_end), re.S)
            if marker_start in text:
                text = pattern.sub(lambda _: section, text, count=1)
                break
        else:
            text += '\n' + section
        fd, name = tempfile.mkstemp(dir=daily.parent, prefix='.daily-')
        try:
            with os.fdopen(fd, 'w') as handle:
                handle.write(text)
            os.replace(name, daily)
        finally:
            Path(name).unlink(missing_ok=True)
    return daily


def collect(args) -> dict:
    state_path = Path(args.state_file).expanduser()
    state = load_state(state_path)
    roots = [('codex', Path(args.codex_root).expanduser()),
             ('claude', Path(args.claude_root).expanduser()),
             ('claude', Path(args.ccs_root).expanduser())]
    counts = dict(
        candidates=0,
        unchanged=0,
        duplicates=0,
        active=0,
        ingested=0,
        failed=0,
        parse_errors=0,
        deferred=0,
        unexamined=0,
        calls=0,
    )
    candidates = ([(args.source, Path(args.path).expanduser())] if args.path
                  else discover(roots, args.days, state))
    counts['candidates'] = len(candidates)
    seen_sessions: dict[tuple[str, str], set[str]] = {}
    if not args.path:
        state['backlog'] = []
    for position, (source, path) in enumerate(candidates):
        if counts['calls'] >= args.limit:
            counts['unexamined'] += len(candidates) - position
            if not args.path:
                state['backlog'] = []
                for source, pending_path in candidates[position:]:
                    entry = dict(source=source, path=str(pending_path), status='unexamined')
                    try:
                        entry['source_mtime'] = pending_path.stat().st_mtime
                    except OSError:
                        pass
                    state['backlog'].append(entry)
            break
        stat = path.stat()
        if time.time() - stat.st_mtime < args.stable_seconds:
            counts['active'] += 1
            continue
        parse_errors = _jsonl_parse_errors(path)
        if parse_errors:
            counts['failed'] += 1
            counts['parse_errors'] += 1
            if not args.dry_run:
                # Keep only structural diagnostics; never persist malformed
                # line contents or transcript excerpts.
                state['sessions'][f'{source}:path:{path}'] = {
                    'source': source,
                    'path': str(path),
                    'status': 'failed',
                    'error_type': 'JSONDecodeError',
                    'parse_error_lines': parse_errors[:20],
                    'parse_error_count': len(parse_errors),
                }
                save_state(state_path, state)
            continue
        try:
            digest = file_hash(path)
            session = extract(source, path)
        except (OSError, ValueError, TypeError):
            counts['failed'] += 1
            continue
        if session is None or session.is_subagent:
            continue
        session_key = (source, str(session.session_id))
        hashes = seen_sessions.setdefault(session_key, set())
        if digest in hashes:
            counts['duplicates'] += 1
            continue
        hashes.add(digest)
        # A previously malformed version of this path is no longer pending
        # once the same file becomes valid.
        state['sessions'].pop(f'{source}:path:{path}', None)
        key = f'{source}:{session.session_id}'
        prior = state['sessions'].get(key, {})
        if prior.get('source_hash') == digest and prior.get('status') in {'ingested', 'empty'}:
            counts['unchanged'] += 1
            continue
        route = project_route(session)
        record = dict(
            source=source,
            session_id=session.session_id,
            path=str(path),
            source_hash=digest,
            source_mtime=stat.st_mtime,
            source_project=route.source_project,
            source_cwd=route.source_cwd,
            destination_project=route.destination_project,
            used_fallback=route.used_fallback,
            route_reason=route.reason,
            status='failed',
        )
        try:
            project = route.destination_dir
            if project is None:
                raise ValueError('no initialized source project')
            # ``project`` remains a compatibility alias for older state
            # readers; destination_project is the explicit new field.
            record['project'] = project.name
            if source == 'codex' and already_ingested(project, session, digest):
                record['status'] = 'ingested'
                counts['unchanged'] += 1
            elif len(session.context) > args.max_chars:
                record['status'] = 'deferred'
                counts['deferred'] += 1
            elif counts['calls'] >= args.limit:
                counts['deferred'] += 1
                record['status'] = 'deferred'
            elif args.dry_run:
                counts['calls'] += 1
                continue
            else:
                counts['calls'] += 1
                response, _ = asyncio.run(run_flush(session.context))
                response = redact_sensitive_text(response.strip())
                if not response or response.startswith('FLUSH_ERROR'):
                    raise RuntimeError('extraction failed')
                if response == 'FLUSH_OK':
                    record['status'] = 'empty'
                else:
                    daily = (upsert_daily_entry(project, session, response) if source == 'codex'
                             else store_claude(project, session, response, prior))
                    record.update(status='ingested', daily_file=daily.name,
                                  ingested_at=datetime.now(timezone.utc).isoformat())
                    pending = state.setdefault('compile_pending', [])
                    if project.name not in pending:
                        pending.append(project.name)
                    counts['ingested'] += 1
                if source == 'codex':
                    mark_ingested(project, session, digest, route)
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            counts['failed'] += 1
        if not args.dry_run:
            state['sessions'][key] = record
            save_state(state_path, state)
    if not args.dry_run:
        if args.compile:
            pending = state.setdefault('compile_pending', [])
            for project_name in pending[:1]:
                project = VAULT_ROOT / project_name
                env = dict(os.environ, ACE_PROJECT_DIR=str(project), CODEX_ACE_BACKFILL_ENABLED='0')
                try:
                    result = subprocess.run([sys.executable, str(Path(__file__).with_name('compile.py'))],
                                            env=env, capture_output=True, timeout=1200)
                    if result.returncode:
                        counts['failed'] += 1
                    else:
                        pending.remove(project_name)
                except subprocess.TimeoutExpired:
                    counts['failed'] += 1
        state['last_run_at'] = datetime.now(timezone.utc).isoformat()
        coverage = dict(counts)
        coverage.update(_pending_coverage(state, candidates))
        state['coverage'] = coverage
        save_state(state_path, state)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-file', default=str(DEFAULT_STATE))
    parser.add_argument('--codex-root', default=str(Path.home() / '.codex/sessions'))
    parser.add_argument('--claude-root', default=str(Path.home() / '.claude/projects'))
    parser.add_argument('--ccs-root', default=str(Path.home() / '.ccs/shared/context-groups/default/projects'))
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--stable-seconds', type=int, default=120)
    parser.add_argument('--max-chars', type=int, default=500000)
    parser.add_argument('--source', choices=['claude', 'codex'])
    parser.add_argument('--path', help='Capture one exact transcript (hook mode)')
    parser.add_argument('--compile', action='store_true', help='Compile one pending project after collection')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if bool(args.path) != bool(args.source):
        parser.error('--path and --source must be provided together')
    if min(args.days, args.limit, args.max_chars) < 1 or args.stable_seconds < 0:
        parser.error('limits must be positive')
    if args.dry_run:
        result = collect(args)
    else:
        lock_path = Path(args.state_file).expanduser().with_suffix('.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('SKIP collection already running')
                return 0
            result = collect(args)
    print(json.dumps(result, sort_keys=True))
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
