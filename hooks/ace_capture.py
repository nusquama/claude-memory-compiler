"""Small, detached launchers shared by the Claude Code ACE hooks.

The hooks only validate the JSON payload and the transcript path.  The
collector owns transcript parsing, redaction, normalized queueing, and its
process lock.  Database acknowledgement and the single processor own later
extraction and compilation.  Keeping those operations out of the hook
prevents a hook from advancing a cursor before collection is durable.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # macOS and Linux.  The runtime is POSIX, but keep imports testable elsewhere.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]


ACE_ENTRYPOINT = Path(
    os.environ.get("ACE_BIN", str(Path.home() / ".agents" / "bin" / "ace"))
).expanduser()

COLLECTOR_LIMIT = 1
STOP_INTERVAL_SECONDS = int(
    os.environ.get("ACE_CHECKPOINT_INTERVAL_SECONDS", "1800")
)
# Keep callback claims long enough to deduplicate late duplicate hook events,
# then discard them so a long-lived runtime cannot grow this local ledger
# without bound.
LAUNCH_LEDGER_RETENTION_SECONDS = max(STOP_INTERVAL_SECONDS * 2, 3600)
LAUNCH_LEDGER_MAX_ENTRIES = 256
DEFAULT_COLLECTION_STATE = Path.home() / ".codex" / "ace" / "collection-state.json"
DEFAULT_STOP_THROTTLE = Path.home() / ".codex" / "ace" / "claude-stop-throttle.json"


def parse_hook_payload(raw: str) -> dict[str, Any] | None:
    """Parse a Claude hook payload without reading transcript content."""

    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Claude Code can emit unescaped Windows path separators in hook JSON.
        try:
            fixed = re.sub(r'(?<!\\)\\(?!["\\])', r"\\\\", raw)
            value = json.loads(fixed)
        except (json.JSONDecodeError, TypeError):
            return None
    return value if isinstance(value, dict) else None


def transcript_from_payload(payload: dict[str, Any]) -> Path | None:
    """Return a usable transcript path from a hook payload."""

    raw_path = payload.get("transcript_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    return path


def cwd_from_payload(payload: dict[str, Any]) -> str | None:
    """Return the source cwd from hook metadata, without a process fallback.

    Claude includes the project root in the hook JSON.  Passing it through is
    what lets the ACE collector resolve an initialized destination.  Missing
    or non-string metadata is intentionally left unset: the collector must
    not infer a project from the hook process cwd or a vault-folder fallback.
    """

    value = payload.get("cwd")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def collection_state_path() -> Path:
    """Resolve the collector state path used by the stop throttle."""

    configured = os.environ.get("ACE_COLLECTION_STATE_FILE") or os.environ.get(
        "ACE_STATE_FILE"
    )
    return Path(configured).expanduser() if configured else DEFAULT_COLLECTION_STATE


def stop_throttle_path() -> Path:
    configured = os.environ.get("ACE_STOP_THROTTLE_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_STOP_THROTTLE


def stop_throttle_lock_path() -> Path:
    """Return the lock used for the throttle and launch ledger.

    The lock is deliberately separate from the ACE collector lock.  Hooks
    hold it only while reading or updating their small local state and while
    making the detached ``Popen`` call; the collector owns serialization of
    collection work itself.
    """

    configured = os.environ.get("ACE_STOP_THROTTLE_LOCK_FILE")
    if configured:
        return Path(configured).expanduser()
    return stop_throttle_path().with_suffix(".lock")


@contextlib.contextmanager
def _throttle_lock() -> Iterator[None]:
    """Serialize hook state changes across processes without a stale pid lock."""

    path = stop_throttle_lock_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = path.open("a+")
    except OSError as exc:
        # A hook must remain a best-effort, non-blocking launcher when its
        # optional local ledger cannot be opened (for example, a read-only
        # home during an isolated test).  Normal runtime paths are writable,
        # so this fallback is only for the existing launch semantics.
        logging.getLogger("ace.capture").warning(
            "could not open throttle lock: %s", exc
        )
        yield
        return
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def collector_command(
    transcript_path: Path,
    *,
    cwd: str | Path | None = None,
) -> list[str]:
    """Build the bounded, explicit Claude collector command.

    ``bin/ace collect`` owns its parser and routing.  Keep this launcher to
    options supported by that public command; local stop-throttle state is
    deliberately not passed as an unknown CLI option.
    """

    command = [
        str(ACE_ENTRYPOINT),
        "collect",
        "--source",
        "claude",
        "--path",
        str(transcript_path),
    ]
    if cwd is not None:
        value = str(cwd).strip()
        if value:
            command.extend(["--cwd", value])
    command.extend([
        "--limit",
        str(COLLECTOR_LIMIT),
        # Memory first: the daily log is written from the local envelope as
        # soon as the capture is durable, like the pre-ACE flush.
        "--extract",
    ])
    return command


def _detached_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _spawn_detached_unlocked(
    command: list[str],
    *,
    popen: Callable[..., Any] | None = None,
) -> bool:
    """Spawn a bounded child; the caller owns the launch lock."""

    launcher = popen or subprocess.Popen
    try:
        launcher(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=(sys.platform != "win32"),
            start_new_session=(sys.platform != "win32"),
            creationflags=_detached_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logging.getLogger("ace.capture").warning("detached child failed: %s", exc)
        return False
    return True


def spawn_detached(
    command: list[str],
    *,
    popen: Callable[..., Any] | None = None,
) -> bool:
    """Spawn a bounded child without keeping the Claude hook alive."""

    # Keep direct callers safe too.  ``spawn_collector`` already owns this
    # same lock while it atomically claims a transcript snapshot and therefore
    # calls the unlocked helper to avoid nested flock acquisition.
    with _throttle_lock():
        return _spawn_detached_unlocked(command, popen=popen)


def spawn_collector(
    transcript_path: Path,
    *,
    cwd: str | Path | None = None,
    popen: Callable[..., Any] | None = None,
    now: float | None = None,
) -> bool:
    """Launch one bounded Claude collection for an exact transcript path.

    PreCompact and SessionEnd can arrive at the same time for one transcript.
    The short launch ledger claims the current filesystem snapshot under the
    throttle lock, so only one callback starts a child.  A changed snapshot
    gets its own claim, which lets a later callback collect messages appended
    while the first child was starting.  A failed ``Popen`` never leaves a
    claim behind, so the next callback remains retryable.
    """

    command = collector_command(transcript_path, cwd=cwd)
    key = str(transcript_path.expanduser().resolve())
    fingerprint = _transcript_fingerprint(transcript_path)
    current_time = time.time() if now is None else now

    with _throttle_lock():
        state = _read_stop_throttle()
        launches = state.get("launches")
        if not isinstance(launches, dict):
            launches = {}
            state["launches"] = launches
        _prune_launches(launches, current_time)
        previous = launches.get(key)
        if (
            fingerprint is not None
            and isinstance(previous, dict)
            and previous.get("fingerprint") == fingerprint
            and current_time - _parse_time(previous.get("launched_at")) < STOP_INTERVAL_SECONDS
        ):
            return False

        launched = _spawn_detached_unlocked(command, popen=popen)
        if not launched:
            return False

        if fingerprint is not None:
            launches[key] = {
                "fingerprint": fingerprint,
                "launched_at": current_time,
            }
            _prune_launches(launches, current_time, preserve_key=key)
            try:
                _write_stop_throttle(state)
            except OSError as exc:
                # The child is already detached.  Keep the historical Popen
                # result semantics even if a local best-effort ledger write
                # is unavailable; the next callback can retry the ledger.
                logging.getLogger("ace.capture").warning(
                    "could not persist launch ledger: %s", exc
                )
        return True


def _transcript_fingerprint(path: Path) -> dict[str, int] | None:
    """Read only stable file metadata for callback deduplication."""

    try:
        stat = path.stat()
    except OSError:
        return None
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _prune_launches(
    launches: dict[str, Any],
    current_time: float,
    *,
    preserve_key: str | None = None,
) -> None:
    """Expire old callback claims and keep the newest bounded set."""

    recent: list[tuple[float, str, dict[str, Any]]] = []
    for key, value in launches.items():
        if not isinstance(value, dict) or "launched_at" not in value:
            continue
        launched_at = _parse_time(value.get("launched_at"))
        if current_time - launched_at >= LAUNCH_LEDGER_RETENTION_SECONDS:
            continue
        recent.append((launched_at, key, value))

    recent.sort(key=lambda item: (item[0], item[1]), reverse=True)
    kept = recent[:LAUNCH_LEDGER_MAX_ENTRIES]
    if preserve_key and not any(key == preserve_key for _, key, _ in kept):
        preserved = next(
            (entry for entry in recent if entry[1] == preserve_key),
            None,
        )
        if preserved is not None:
            if len(kept) == LAUNCH_LEDGER_MAX_ENTRIES:
                kept[-1] = preserved
            else:
                kept.append(preserved)
    launches.clear()
    launches.update({key: value for _, key, value in kept})


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_time(value: Any) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _successful_collection(path: Path) -> tuple[float, float] | None:
    """Return (source_mtime, success_time) for the latest durable record."""

    state = _read_json(collection_state_path())
    if not state:
        return None
    records = state.get("sessions")
    if not isinstance(records, dict):
        return None

    resolved = str(path.expanduser().resolve())
    matches: list[tuple[float, float]] = []
    for record in records.values():
        if not isinstance(record, dict) or record.get("source") != "claude":
            continue
        try:
            record_path = str(Path(str(record.get("path", ""))).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
        if record_path != resolved or record.get("status") not in {"ingested", "empty"}:
            continue
        source_mtime = float(record.get("source_mtime", 0.0) or 0.0)
        success_time = 0.0
        for key in ("collected_at", "success_at", "updated_at", "ingested_at"):
            success_time = max(success_time, _parse_time(record.get(key)))
        # Empty durable extractions may not have an ingested_at field.  The
        # source mtime is the safe lower-bound fallback for the throttle.
        success_time = max(success_time, source_mtime)
        matches.append((source_mtime, success_time))
    return max(matches, key=lambda item: item[1]) if matches else None


def _read_stop_throttle() -> dict[str, Any]:
    value = _read_json(stop_throttle_path())
    return value if value is not None else {"version": 1, "paths": {}}


def _write_stop_throttle(state: dict[str, Any]) -> None:
    path = stop_throttle_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=".claude-stop-")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def stop_is_throttled(transcript_path: Path, *, now: float | None = None) -> bool:
    """Apply the 30-minute gate without reading transcript tokens.

    A successful collector record is authoritative.  Before that record
    appears, the local attempted-launch timestamp prevents Stop from spawning
    one child per assistant response.  Once the gate opens, the collector's
    source hash handles an unchanged transcript without duplicate ingestion.
    """

    current_time = time.time() if now is None else now
    try:
        transcript_path.stat()
    except OSError:
        return True

    with _throttle_lock():
        successful = _successful_collection(transcript_path)
        if successful is not None:
            _, success_time = successful
            if current_time - success_time < STOP_INTERVAL_SECONDS:
                return True

        state = _read_stop_throttle()
        key = str(transcript_path.expanduser().resolve())
        paths = state.setdefault("paths", {})
        previous = paths.get(key, {})
        if isinstance(previous, dict):
            attempted_at = _parse_time(previous.get("attempted_at"))
            if current_time - attempted_at < STOP_INTERVAL_SECONDS:
                return True

        # A PreCompact or SessionEnd launch is also a recent attempt for the
        # same snapshot.  Checking its fingerprint keeps Stop from racing a
        # callback that already claimed this transcript.
        launches = state.get("launches", {})
        launch = launches.get(key) if isinstance(launches, dict) else None
        fingerprint = _transcript_fingerprint(transcript_path)
        if (
            fingerprint is not None
            and isinstance(launch, dict)
            and launch.get("fingerprint") == fingerprint
            and current_time - _parse_time(launch.get("launched_at")) < STOP_INTERVAL_SECONDS
        ):
            return True
        return False


def mark_stop_launch(transcript_path: Path, *, now: float | None = None) -> None:
    current_time = time.time() if now is None else now
    with _throttle_lock():
        state = _read_stop_throttle()
        paths = state.setdefault("paths", {})
        key = str(transcript_path.expanduser().resolve())
        paths[key] = {
            "attempted_at": current_time,
            "source_mtime": transcript_path.stat().st_mtime,
        }
        _write_stop_throttle(state)


__all__ = [
    "ACE_ENTRYPOINT",
    "STOP_INTERVAL_SECONDS",
    "collector_command",
    "cwd_from_payload",
    "mark_stop_launch",
    "parse_hook_payload",
    "spawn_collector",
    "stop_throttle_lock_path",
    "stop_is_throttled",
    "transcript_from_payload",
]
