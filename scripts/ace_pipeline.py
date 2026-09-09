#!/usr/bin/env python3
"""Orchestrate the bounded ACE collection and processing pipeline.

The module is deliberately small at its integration boundaries.  Project
registration, transcript parsing, the outbox, the database store, learning,
and scheduling are supplied by the sibling ``ace_*`` modules.  The pipeline
owns ordering, durable stage state, retry behaviour, and the user-facing CLI.

No command in this module uses a fallback project.  Collection reads a body
only after strict project resolution and after metadata discovery selected the
source.  Processing reads only database-acquitted snapshots; a local outbox
is never treated as an extraction source.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

try:  # macOS and Linux.  The fallback keeps imports testable on Windows.
    import fcntl
except ImportError:  # pragma: no cover - the supported runtime is POSIX
    fcntl = None


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = SCRIPT_DIR.parent
PIPELINE_VERSION = "ace-pipeline-v1"
DEFAULT_LIMIT = 4
# The native 30-minute pass is deliberately bounded to one revision per
# project. Explicit/manual processing keeps the normal DEFAULT_LIMIT.
AUTOMATION_PROCESS_LIMIT = 4
# Metadata discovery and parsing never call a model: examine a wide lot per
# tick so a deferred backlog drains within a few cycles instead of days.
AUTOMATION_COLLECT_LIMIT = 40
# Identity-only Supabase reads must stay below the wrapper's pipe limit.
# Later ticks continue from the same bounded queue.
REF_QUERY_PAGE_LIMIT = 64
AUTOMATION_REPORT_WINDOW_MINUTES = 120
AUTOMATION_MAX_PAYLOAD_BYTES = 4_000_000
AUTOMATION_DAILY_MAX_ATTEMPTS = 3
AUTOMATION_DAILY_START = "07:00"
ANALYSIS_MAX_DEPTH = 5
ANALYSIS_MAX_NODES = 1000
# Native analysis is deliberately one conversation per pass.  This keeps the
# evidence window intact; unresolved snapshots remain pending for the next
# scheduled pass instead of being compressed into one shared prompt.
# Conversations analysed per daily call. One call covers the batch; the
# evidence budget per conversation is MAX_MODEL_CONTEXT_CHARS / batch.
ANALYSIS_BATCH_LIMIT = 6
# Continuations of the daily analysis attempted inside one native tick.
DAILY_CONTINUATIONS_PER_TICK = 4
DEFAULT_MAX_CONTEXT_CHARS = 120_000
DEFAULT_MAX_HISTORY_CHARS = 8_000
EXTRACTION_CURSOR_VERSION = 1
# "local": the daily log is written from the local sanitized envelope first;
# the central database is synchronized and acknowledged asynchronously.
# "database": the historical contract, extraction only after the DB ack.
DEFAULT_EXTRACTION_MODE = "local"
# Marker the extractor appends after the daily body, followed by one JSON
# object holding the improvement signals it observed in the raw transcript.
SIGNAL_MARKER = "<<<ACE_SIGNAUX>>>"
SIGNAL_TYPES = frozenset(
    {
        "frustration",
        "correction_utilisateur",
        "demande_repetee",
        "tool_error",
        "fausse_completion",
        "perte_de_contexte",
        "preference_recurrente",
    }
)
MAX_SIGNALS_PER_SNAPSHOT = 40
MAX_SIGNAL_QUOTE_CHARS = 200
MAX_SIGNAL_TEXT_CHARS = 400
PARIS = ZoneInfo("Europe/Paris")
STAGES = ("collection", "sync", "extraction", "compile", "analysis")
# Structural validation messages from ``compile.validate_knowledge_bundle``.
# They carry knowledge-base slugs and file names only, never transcript or
# prompt content, so they are preserved verbatim: without them a failed
# compilation is reported as a generic "compiler error" and its real cause
# stays invisible for days.
_COMPILE_STRUCTURAL_DIAGNOSTIC = re.compile(
    r"(?i)\b("
    r"broken internal link|index link has no file|build log link has no file|"
    r"article missing from index|link escapes knowledge root|"
    r"knowledge index is (?:missing|unreadable)|knowledge build log is (?:missing|unreadable)|"
    r"knowledge file escapes its root|symlinked knowledge file|knowledge file is unreadable|"
    r"build log has no entry for|no article references|"
    r"incomplete knowledge build"
    r")\b[^;]*"
)
_COMPILE_DIAGNOSTIC_HINT = re.compile(
    r"(?i)(error|exception|traceback|failed|failure|missing|not found|no such file|"
    r"timeout|timed out|permission denied|unavailable|undefined|cannot)"
)
_RUNTIME_PREAMBLE_HEADER = re.compile(
    r"\A\s*#\s*AGENTS\.md instructions[^\n]*\n", re.IGNORECASE
)
_RUNTIME_INSTRUCTIONS_BLOCK = re.compile(
    r"\s*<INSTRUCTIONS>.*?</INSTRUCTIONS>", re.IGNORECASE | re.DOTALL
)
_RUNTIME_ENVIRONMENT_BLOCK = re.compile(
    r"\s*<environment_context>.*?</environment_context>",
    re.IGNORECASE | re.DOTALL,
)
_NON_ACTIVITY_MESSAGE_TYPES = {
    "task_started",
    "task_complete",
    "turn_started",
    "turn_complete",
    "token_usage",
    "tokenusage",
    "token_count",
    "token_counts",
    "telemetry",
}

_MISSING = object()


class PipelineError(RuntimeError):
    """A user-actionable pipeline error."""


class PipelinePendingError(PipelineError):
    """Work remains pending after a bounded stage pass."""


class NotInitializedError(PipelineError):
    """The requested source does not belong to an initialized project."""


class PipelineBusyError(PipelineError):
    """Another processor tick owns the advisory lock."""


class StateSecurityError(PipelineError):
    """A state path is not private enough for durable pipeline state."""


class OfflineError(PipelineError):
    """The database is not available; pending work remains in the outbox."""


def _import_optional(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        return None


def _redact(value: Any) -> str:
    """Redact text before it can reach diagnostics or bounded history output."""
    text = value if isinstance(value, str) else str(value)
    try:
        utils = _import_optional("utils")
        if utils is not None and hasattr(utils, "redact_sensitive_text"):
            return str(utils.redact_sensitive_text(text))
    except Exception:
        pass
    text = re.sub(r"(?i)(authorization|cookie|token|secret|password|api[_-]?key)\s*[:=]\s*[^\s,;]+", r"\1=<REDACTED>", text)
    return re.sub(r"\b(?:sk|ghp|xox[baprs])-[A-Za-z0-9_-]{12,}\b", "<REDACTED>", text)


def _json_safe(value: Any) -> Any:
    """Convert integration objects to JSON-safe values without dumping secrets."""
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return _redact(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    safe = _json_safe(value)
    if isinstance(safe, Mapping):
        return dict(safe)
    return {"value": safe}


def _first_attr(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
    for name in names:
        try:
            candidate = getattr(value, name)
        except AttributeError:
            continue
        if candidate is not None:
            return candidate
    return default


def _project_id(project: Any) -> str:
    value = _first_attr(project, ("project_id", "id", "uuid", "name", "slug"))
    if value:
        return str(value)
    root = _first_attr(project, ("root", "repo_root", "path"), "unknown")
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:24]


def _project_name(project: Any) -> str:
    value = _first_attr(project, ("name", "slug", "project_name"))
    if value:
        return str(value)
    root = _first_attr(project, ("root", "repo_root", "path"))
    return Path(str(root)).name if root else _project_id(project)


def _normalise_datetime(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif value:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = fallback or datetime.now(timezone.utc)
    else:
        parsed = fallback or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _call_variants(function: Callable[..., Any], variants: Sequence[tuple[tuple[Any, ...], dict[str, Any]]]) -> Any:
    """Call an integration across the two compatible signatures in transition."""
    last_error: TypeError | None = None
    for args, kwargs in variants:
        try:
            return function(*args, **kwargs)
        except TypeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise PipelineError("empty integration call variants")


def _await_if_needed(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    # The CLI is synchronous, but injected tests can call an async fake from
    # an already-running loop.  Keep the event loop untouched.
    import threading

    result: list[Any] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as error:  # pragma: no cover - defensive bridge
            errors.append(error)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0] if result else None


class AtomicState:
    """Private, atomic JSON state split by pipeline stage."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def _ensure_private_dir(self, create: bool = False) -> None:
        if not self.root.exists():
            if not create:
                return
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.root.is_dir():
            raise StateSecurityError(f"state root is not a directory: {self.root}")
        try:
            mode = self.root.stat().st_mode & 0o777
        except OSError as error:
            raise StateSecurityError("cannot inspect state permissions") from error
        if mode & 0o077:
            raise StateSecurityError("state root permissions are not private")

    def path(self, stage: str) -> Path:
        if stage not in STAGES and stage != "projects":
            raise ValueError(f"unknown state stage: {stage}")
        return self.root / f"{stage}.json"

    def read(self, stage: str) -> dict[str, Any]:
        path = self.path(stage)
        self._ensure_private_dir(create=False)
        if not path.exists():
            return {}
        try:
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                raise StateSecurityError(f"state file permissions are not private: {path.name}")
            value = json.loads(path.read_text(encoding="utf-8"))
        except StateSecurityError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise PipelineError(f"invalid {stage} state") from error
        if not isinstance(value, dict):
            raise PipelineError(f"invalid {stage} state shape")
        return value

    def write(self, stage: str, value: Mapping[str, Any]) -> None:
        self._ensure_private_dir(create=True)
        path = self.path(stage)
        fd, temp_name = tempfile.mkstemp(prefix=f".{stage}.", suffix=".tmp", dir=str(self.root))
        temp_path = Path(temp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_json_safe(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            os.chmod(path, 0o600)
        finally:
            temp_path.unlink(missing_ok=True)


@contextlib.contextmanager
def advisory_lock(path: str | Path) -> Iterable[None]:
    """Hold a process lock that the kernel releases after a crash."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        mode = target.parent.stat().st_mode & 0o777
        if mode & 0o077:
            raise StateSecurityError("lock directory permissions are not private")
    except OSError as error:
        raise StateSecurityError("cannot inspect lock directory") from error
    handle = target.open("a+", encoding="utf-8")
    try:
        os.chmod(target, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PipelineBusyError("another ACE processor is running") from error
        yield
    finally:
        if fcntl is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class _LocalOutbox:
    """Small in-memory adapter used only when tests inject no outbox module."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def enqueue(self, envelope: Mapping[str, Any]) -> str:
        key = str(envelope.get("snapshot_id"))
        if not any(item["key"] == key for item in self.items):
            self.items.append({"key": key, "envelope": dict(envelope), "status": "pending"})
        return key

    def pending(self, limit: int | None = None) -> list[dict[str, Any]]:
        values = [item for item in self.items if item["status"] == "pending"]
        return values[:limit] if limit else values

    def ack(self, key: str, receipt: Any = None) -> None:
        for item in self.items:
            if item["key"] == key:
                item["status"] = "acked"
                item["receipt"] = _json_safe(receipt)

    def fail(self, key: str, error: str) -> None:
        for item in self.items:
            if item["key"] == key:
                item["error"] = _redact(error)

    def summary(self) -> dict[str, int]:
        return {
            "pending": sum(item["status"] == "pending" for item in self.items),
            "acked": sum(item["status"] == "acked" for item in self.items),
            "total": len(self.items),
        }


def _first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in ``text``, string-aware."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _normalise_signal(value: Any) -> dict[str, Any] | None:
    """Keep one bounded improvement signal, or nothing.

    The extractor sees the raw transcript, so it is the only stage able to
    quote the user verbatim.  Everything here is bounded and typed; an unknown
    type or a missing quote/message pair is dropped rather than guessed.
    """
    if not isinstance(value, Mapping):
        return None
    kind = str(value.get("type") or "").strip().lower()
    if kind not in SIGNAL_TYPES:
        return None
    message_ids = [
        str(item).strip()
        for item in (value.get("message_ids") or [])
        if isinstance(item, (str, int)) and str(item).strip()
    ][:20]
    quote = _redact(" ".join(str(value.get("quote") or "").split()))[:MAX_SIGNAL_QUOTE_CHARS]
    observed = _redact(" ".join(str(value.get("observed") or "").split()))[:MAX_SIGNAL_TEXT_CHARS]
    signature = _redact(" ".join(str(value.get("signature") or kind).split()))[:120]
    if not message_ids and not quote:
        # A signal with neither a message nor a quote cannot be verified later.
        return None
    return {
        "type": kind,
        "signature": signature or kind,
        "message_ids": message_ids,
        "quote": quote,
        "observed": observed,
    }


def split_extraction_signals(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split the daily body from the appended SIGNAUX blocks.

    Several extractor outputs may be concatenated, so every marker occurrence
    is parsed and its signals merged.  A malformed block is ignored: the daily
    log must never be lost because the signal JSON was wrong.
    """
    if not text or SIGNAL_MARKER not in text:
        return text, []
    segments = text.split(SIGNAL_MARKER)
    body_parts = [segments[0]]
    signals: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for segment in segments[1:]:
        payload = _first_json_object(segment)
        if payload is not None:
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            items = parsed.get("signals") if isinstance(parsed, Mapping) else None
            for item in items or []:
                signal = _normalise_signal(item)
                if signal is None:
                    continue
                identity = (signal["type"], signal["signature"], signal["quote"])
                if identity in seen:
                    continue
                seen.add(identity)
                signals.append(signal)
                if len(signals) >= MAX_SIGNALS_PER_SNAPSHOT:
                    break
            # Anything after the JSON object belongs to the next daily chunk.
            remainder = segment[segment.find(payload) + len(payload) :]
        else:
            remainder = segment
        if remainder.strip():
            body_parts.append(remainder)
    return "\n\n".join(part.strip() for part in body_parts if part.strip()), signals


class ACEPipeline:
    """Coordinate collection, synchronization, extraction, and daily work."""

    def __init__(
        self,
        *,
        vault_root: str | Path | None = None,
        private_root: str | Path | None = None,
        projects: Any = _MISSING,
        transcripts: Any = _MISSING,
        outbox: Any = _MISSING,
        store: Any = _MISSING,
        extractor: Callable[[str], Any] | None = None,
        learning: Any = _MISSING,
        schedule: Any = _MISSING,
        now: Callable[[], datetime] | None = None,
    ):
        self.vault_root = Path(vault_root or os.environ.get("ACE_VAULT_ROOT", CONFIG_ROOT.parent)).expanduser().resolve()
        configured_private = private_root or os.environ.get("ACE_PRIVATE_ROOT") or os.environ.get("ACE_STATE_ROOT")
        self.private_root = Path(configured_private or (Path.home() / ".agents" / "private" / "ace")).expanduser()
        self.state = AtomicState(self.private_root)
        self.projects = _import_optional("ace_projects") if projects is _MISSING else projects
        self.transcripts = _import_optional("ace_transcripts") if transcripts is _MISSING else transcripts
        self.outbox_integration = _import_optional("ace_outbox") if outbox is _MISSING else outbox
        self.database = _import_optional("ace_database")
        self.store = _MISSING if store is _MISSING else store
        self.learning = _import_optional("ace_learning") if learning is _MISSING else learning
        self.schedule = _import_optional("ace_schedule") if schedule is _MISSING else schedule
        self.extractor = extractor
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._outbox_instance: Any = outbox if outbox is not _MISSING and hasattr(outbox, "enqueue") else None
        self._store_instance: Any = None
        self._compile_diagnostic_cache: dict[tuple[str, str], dict[str, Any]] = {}
        # One owner is stable for the whole bounded process pass.  Claims are
        # renewed with this same value before long work and before the final
        # acknowledgement, so a lease cannot be completed by a stale worker.
        self._lease_owner = f"ace-{uuid.uuid4().hex}"
        self._last_stage_claim_error: str | None = None

    @property
    def host_id(self) -> str:
        return os.environ.get("ACE_HOST_ID") or socket.gethostname()

    def _automation_enabled(self) -> bool:
        """Return whether the native scheduler is in incremental mode."""
        return os.environ.get("ACE_AUTOMATION_MODE", "").strip().lower() == "incremental"

    def _automation_since(self, *, create: bool = False) -> float | None:
        """Return the first-run cutoff for the native incremental service.

        The cutoff is stored in the existing collection state so reinstalling
        the scheduler cannot silently turn an incremental service into a
        historical replay. Explicit ``--all-history`` callers bypass it.
        """
        if not self._automation_enabled():
            return None
        state = self.state.read("collection")
        raw = state.get("automation_since")
        if raw not in (None, ""):
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise PipelineError("invalid incremental collection cutoff") from error
            if value <= 0:
                raise PipelineError("invalid incremental collection cutoff")
            return value
        if not create:
            return None
        value = _normalise_datetime(self.now()).timestamp()
        state["automation_since"] = value
        self.state.write("collection", state)
        return value

    def _automation_daily_due(self, current: datetime) -> bool:
        """Allow bounded same-day catch-up/retries after the 07:00 start."""
        raw_start = os.environ.get(
            "ACE_DAILY_REPORT_START",
            os.environ.get("ACE_DAILY_START", AUTOMATION_DAILY_START),
        ).strip()
        try:
            hour_text, minute_text = raw_start.split(":", 1)
            start_minutes = int(hour_text) * 60 + int(minute_text)
        except (TypeError, ValueError):
            start_minutes = 7 * 60
        if not (0 <= start_minutes < 24 * 60):
            start_minutes = 7 * 60
        current_minutes = current.hour * 60 + current.minute
        if current_minutes < start_minutes:
            return False
        state = self.state.read("collection")
        day = current.date().isoformat()
        if state.get("last_automation_report_day") == day:
            return False
        runs = state.get("automation_daily", {})
        record = runs.get(day) if isinstance(runs, Mapping) else None
        if not isinstance(record, Mapping):
            return True
        if str(record.get("status") or "").lower() == "complete":
            return False
        retry_value: Any = record.get("retry_failures", _MISSING)
        if retry_value is _MISSING:
            # Upgrade an old record that used ``status=failed`` for a
            # pending-only continuation.  It must not inherit the old total
            # attempt count as a failure budget.
            stages = record.get("stages")
            if isinstance(stages, Mapping):
                try:
                    stage_failed = int(stages.get("failed", 0) or 0)
                    stage_pending = int(stages.get("pending", 0) or 0)
                except (TypeError, ValueError):
                    stage_failed, stage_pending = 1, 0
                if stage_failed == 0 and stage_pending > 0:
                    retry_value = 0
        if retry_value is _MISSING:
            retry_value = (
                record.get("attempts", 0)
                if str(record.get("status") or "").lower() == "failed"
                else 0
            )
        try:
            # ``attempts`` counts bounded scheduler passes for visibility;
            # only actual failed stages consume the retry budget.  Pending
            # continuation batches must therefore remain due until their
            # successful stages drain the queue.
            attempts = int(retry_value or 0)
        except (TypeError, ValueError):
            attempts = AUTOMATION_DAILY_MAX_ATTEMPTS
        return attempts < AUTOMATION_DAILY_MAX_ATTEMPTS

    def _claim_automation_daily(
        self, current: datetime, result: Mapping[str, Any] | None = None
    ) -> None:
        """Record the result after a bounded daily attempt.

        The compatibility path without a result is kept for old direct
        callers/tests. The native tick always supplies the completed result,
        so a failed attempt remains retryable and is never claimed before its
        work starts.
        """
        state = self.state.read("collection")
        day = current.date().isoformat()
        runs = state.setdefault("automation_daily", {})
        if not isinstance(runs, dict):
            runs = {}
            state["automation_daily"] = runs
        prior = runs.get(day)
        prior = dict(prior) if isinstance(prior, Mapping) else {}
        if result is None:
            # Compatibility path only; the scheduler calls this method after
            # the daily stage and passes its result explicitly.
            status = "complete"
            failed = 0
            pending = 0
        else:
            try:
                failed = max(0, int(result.get("failed", 0) or 0))
            except (TypeError, ValueError):
                failed = 1
            try:
                pending = max(0, int(result.get("pending", 0) or 0))
            except (TypeError, ValueError):
                pending = 1
            status = (
                "complete"
                if failed == 0 and pending == 0
                else "failed"
                if failed > 0
                else "pending"
            )
        try:
            attempts = max(0, int(prior.get("attempts", 0) or 0))
        except (TypeError, ValueError):
            attempts = 0
        attempts += 1
        try:
            retry_failures = max(0, int(prior.get("retry_failures", 0) or 0))
        except (TypeError, ValueError):
            retry_failures = AUTOMATION_DAILY_MAX_ATTEMPTS
        if result is not None and failed > 0:
            retry_failures += 1
        record: dict[str, Any] = {
            **prior,
            "status": status,
            "attempts": attempts,
            "retry_failures": retry_failures,
            "last_attempt_at": _normalise_datetime(self.now()).astimezone(PARIS).isoformat(),
            "stages": {
                "failed": failed,
                "pending": pending,
            },
        }
        if status == "complete":
            state["last_automation_report_day"] = day
            record.pop("retry_exhausted", None)
            record.pop("last_error", None)
        else:
            record["retry_exhausted"] = retry_failures >= AUTOMATION_DAILY_MAX_ATTEMPTS
            record["last_error"] = (
                "daily stages failed" if failed > 0 else "daily stages remain pending"
            )
        runs[day] = record
        self.state.write("collection", state)

    def _cwd(self, value: str | Path | None = None) -> Path:
        raw = value or os.environ.get("ACE_PROJECT_CWD") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        return Path(raw).expanduser().resolve()

    def _resolve_project(self, cwd: str | Path | None = None) -> Any:
        if self.projects is None:
            raise NotInitializedError("ACE project registry is unavailable")
        root = self._cwd(cwd)
        resolver = getattr(self.projects, "resolve_project", None)
        if resolver is None:
            registry = getattr(self.projects, "ProjectRegistry", None)
            if registry is not None:
                resolver = getattr(registry(self.vault_root), "resolve_project", None)
        if resolver is None:
            raise NotInitializedError("ACE project resolver is unavailable")
        result = _call_variants(
            resolver,
            [
                ((root,), {"vault_root": self.vault_root, "strict": True}),
                ((root,), {"strict": True}),
                ((root,), {}),
            ],
        )
        result = _await_if_needed(result)
        if result is None:
            raise NotInitializedError(f"cwd is not an initialized ACE project: {root.name}")
        return result

    def _project_vault_dir(self, project: Any) -> Path:
        candidate = _first_attr(project, ("vault_dir", "vault_path", "destination_dir", "project_dir"))
        if candidate:
            path = Path(str(candidate)).expanduser()
            if path.is_dir() or not path.exists():
                return path
        name = _project_name(project)
        candidate = self.vault_root / name
        if candidate.is_dir() or not candidate.exists():
            return candidate
        candidate = _first_attr(project, ("directory", "path"))
        if candidate:
            return Path(str(candidate)).expanduser()
        raise PipelineError("initialized project has no vault directory")

    def _processable_projects(self) -> list[Any]:
        """Return only explicitly registered projects with a source root.

        ``list_initialized_projects`` also exposes marker-less legacy vault
        folders.  Those records intentionally have no root/id and must not be
        treated as eligible for native collection, synchronisation, or status.
        Keep the vault root explicit because the real registry exposes it as a
        keyword-only argument.
        """
        if self.projects is None:
            return []
        listing = getattr(self.projects, "list_initialized_projects", None)
        if listing is None:
            return []
        try:
            values = _await_if_needed(
                _call_variants(
                    listing,
                    [
                        ((), {"vault_root": self.vault_root}),
                        ((), {}),
                        ((self.vault_root,), {}),
                    ],
                )
            )
        except Exception:
            return []
        if isinstance(values, Mapping):
            values = values.values()
        result: list[Any] = []
        for item in values or []:
            root = _first_attr(item, ("root", "repo_root", "path"))
            project_id = _first_attr(item, ("project_id", "id", "uuid"))
            if root is None or project_id is None:
                continue
            if not bool(_first_attr(item, ("processable",), True)):
                continue
            result.append(item)
        return result

    def _project_ids(self) -> set[str]:
        return {_project_id(item) for item in self._processable_projects()}

    def _init_project(self, cwd: str | Path) -> dict[str, Any]:
        if self.projects is None:
            raise NotInitializedError("ACE project registry is unavailable")
        root = self._cwd(cwd)
        canonical = getattr(self.projects, "canonical_git_root", None)
        if canonical is not None:
            root = _await_if_needed(_call_variants(canonical, [((root,), {})])) or root
            root = Path(root).expanduser().resolve()
        init = getattr(self.projects, "init_project", None)
        if init is None:
            raise PipelineError("ACE project initializer is unavailable")
        project = _await_if_needed(
            _call_variants(init, [((root,), {"vault_root": self.vault_root}), ((root,), {})])
        )
        register = getattr(self.projects, "register_project", None)
        if register is not None:
            # Local registration is idempotent.  Keep it separate from the
            # optional database registration below.
            with contextlib.suppress(TypeError):
                _await_if_needed(
                    _call_variants(
                        register,
                        [
                            ((root,), {"vault_root": self.vault_root}),
                            ((project,), {}),
                            ((root,), {}),
                        ],
                    )
                )
        project_id = _project_id(project)
        projects_state = self.state.read("projects")
        projects_state.setdefault("projects", {})[project_id] = {
            "project_id": project_id,
            "name": _project_name(project),
            "status": "local",
            "registered_at": self.now().isoformat(),
        }
        self.state.write("projects", projects_state)
        database_status = "pending"
        try:
            store = self._get_store()
            registrar = self._store_method(store, ("register_project", "upsert_project")) if store else None
            if registrar is not None:
                _await_if_needed(_call_variants(registrar, [((project,), {}), ((_as_mapping(project)), {})]))
                database_status = "registered"
        except Exception:
            # Registration stays local and retryable.  Do not fail init because
            # the database is offline.
            database_status = "pending"
        return {"initialized": 1, "project_id": project_id, "database": database_status}

    def _get_outbox(self) -> Any:
        if self._outbox_instance is not None:
            return self._outbox_instance
        integration = self.outbox_integration
        if integration is None:
            self._outbox_instance = _LocalOutbox()
            return self._outbox_instance
        constructor = getattr(integration, "Outbox", integration if callable(integration) else None)
        if constructor is None:
            self._outbox_instance = _LocalOutbox()
            return self._outbox_instance
        db_path = self.private_root / "outbox.sqlite3"
        self._outbox_instance = _call_variants(
            constructor,
            [
                ((db_path,), {"max_payload_bytes": 50_000_000, "max_lot_items": 100}),
                ((str(db_path),), {"max_payload_bytes": 50_000_000, "max_lot_items": 100}),
                # Compatibility for the pre-ACE queue adapter. The native
                # adapter receives the canonical names above.
                ((db_path,), {"max_bytes": 50_000_000, "max_lot": 100}),
                ((str(db_path),), {"max_bytes": 50_000_000, "max_lot": 100}),
                ((db_path,), {}),
                ((), {"db_path": db_path}),
            ],
        )
        return self._outbox_instance

    def _get_store(self) -> Any | None:
        if self.store is not _MISSING:
            if self._store_instance is None and self.store is not None:
                self._store_instance = self.store() if inspect.isclass(self.store) else self.store
            return self._store_instance
        if self._store_instance is not None:
            return self._store_instance
        integration = self.database
        if integration is None:
            return None
        constructor = getattr(integration, "SupabaseStore", integration if callable(integration) else None)
        if constructor is None:
            return None
        try:
            self._store_instance = constructor()
        except Exception:
            self._store_instance = None
        return self._store_instance

    @staticmethod
    def _store_method(store: Any, names: Sequence[str]) -> Callable[..., Any] | None:
        if store is None:
            return None
        for name in names:
            method = getattr(store, name, None)
            if callable(method):
                return method
        return None

    def _source_paths(self, source_paths: Sequence[str | Path | Mapping[str, Any]], all_history: bool, days: int, since: float | None = None) -> list[dict[str, Any]]:
        automation_since = None if all_history else self._automation_since(create=True)
        cutoff = automation_since if automation_since is not None else time.time() - max(0, days) * 86400
        if since is not None:
            # An explicit start date (``--since YYYY-MM-DD``) is the operator's
            # bound: it replaces both the activation cutoff and ``--days``.
            cutoff = float(since)
        candidates: list[dict[str, Any]] = []
        for raw in source_paths:
            configured_source: str | None = None
            configured_path: Any = raw
            if isinstance(raw, Mapping):
                configured_source = _first_attr(raw, ("source", "provider", "kind"))
                configured_path = _first_attr(raw, ("path", "source_path", "file"))
            elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
                configured_source = str(raw[0])
                configured_path = raw[1]
            if configured_path is None:
                continue
            path = Path(str(configured_path)).expanduser()
            if path.is_file():
                values = [path]
            elif path.is_dir():
                values = sorted(item for item in path.rglob("*") if item.is_file())
            else:
                continue
            for item in values:
                try:
                    info = item.stat()
                except OSError:
                    continue
                if not all_history and info.st_mtime < cutoff:
                    continue
                candidate: dict[str, Any] = {"path": item.resolve(), "mtime": info.st_mtime}
                if configured_source:
                    candidate["source"] = str(configured_source)
                if isinstance(raw, Mapping):
                    for key in ("session_id", "target_session_id", "project_id", "project_root"):
                        if raw.get(key) not in (None, ""):
                            candidate[key] = raw[key]
                candidates.append(candidate)
        candidates.sort(key=lambda item: (-float(item["mtime"]), str(item["path"])))
        return candidates

    def _default_source_paths(self, source: str | None = None) -> list[Path]:
        """Discover configured local source files without discovering projects."""
        roots: list[tuple[str, Path, tuple[str, ...]]] = [
            ("codex", Path.home() / ".codex" / "sessions", ("*.jsonl",)),
            ("claude", Path.home() / ".claude" / "projects", ("*.jsonl",)),
            (
                "claude",
                Path.home() / ".ccs" / "shared" / "context-groups" / "default" / "projects",
                ("*.jsonl",),
            ),
            (
                "hermes",
                Path(os.environ.get("ACE_HERMES_ROOT", str(Path.home() / ".hermes"))).expanduser(),
                ("*.db", "*.sqlite", "*.sqlite3"),
            ),
        ]
        found: list[Path] = []
        for kind, root, patterns in roots:
            if source and kind != source:
                continue
            if not root.is_dir():
                continue
            for pattern in patterns:
                found.extend(path for path in root.rglob(pattern) if path.is_file())
        return sorted(set(found), key=lambda path: (path.stat().st_mtime, str(path)))

    def _source_project_hint(self, path: Path, source: str) -> Path | None:
        """Read provider metadata needed for automatic project routing."""
        if source == "hermes":
            # A Hermes database can contain sessions from several projects.
            # Without the source adapter's per-session metadata, selecting a
            # first row would silently assign the whole database to one
            # project.  Leave it unrouted until ``iter_snapshots`` provides a
            # project root or project id for each candidate.
            return None
        if source not in {"codex", "claude"}:
            return None
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                max_lines = 8 if source == "codex" else 32
                for line_number, raw in enumerate(handle, 1):
                    # Codex writes cwd in session_meta (normally line 1).
                    # Claude's startup metadata is also at the head of the
                    # JSONL, but allow a few more records for older exports.
                    # Do not parse hundreds of large prompt records merely to
                    # route a file; missing metadata is intentionally unrouted.
                    if line_number > max_lines:
                        break
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(entry, Mapping):
                        continue
                    candidates: list[Any] = [entry]
                    for key in ("payload", "message"):
                        value = entry.get(key)
                        if isinstance(value, Mapping):
                            candidates.append(value)
                    for value in candidates:
                        hint = _first_attr(
                            value,
                            ("cwd", "working_directory", "project_root", "repo_root"),
                        )
                        if hint:
                            return Path(str(hint)).expanduser()
        except OSError:
            return None
        return None

    def _auto_source_groups(
        self,
        source_paths: Sequence[str | Path],
        projects: Sequence[Any],
        *,
        source: str | None,
        all_history: bool,
        days: int,
        since: float | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
        """Group files only by an explicitly registered source root."""
        registry = self.projects
        canonical = getattr(registry, "canonical_git_root", None) if registry else None
        if source:
            kinds = (source,)
        elif not source_paths:
            # A native ACE tick has no positional source paths. It must cover
            # every local provider while still routing each file through its
            # provider-specific parser. Explicit paths without --source retain
            # the historical Codex default; Claude/Hermes callers must identify
            # their adapter when they pass a path.
            kinds = ("codex", "claude", "hermes")
        elif all(Path(item).suffix.lower() in {".db", ".sqlite", ".sqlite3"} for item in source_paths):
            kinds = ("hermes",)
        else:
            kinds = ("codex",)
        # Keep the provider next to the path.  A plain ``project -> [Path]``
        # grouping loses the parser identity as soon as Codex and Claude
        # files belong to the same project.
        groups: dict[str, list[dict[str, Any]]] = {}
        unrouted = 0
        candidates_count = 0
        for kind in kinds:
            roots = source_paths if source_paths else self._default_source_paths(kind)
            paths = self._source_paths(roots, all_history, days, since)
            for item in paths:
                path = Path(item["path"])
                route_candidates: list[dict[str, Any]] = []
                # Hermes databases can contain sessions from several
                # projects.  Its metadata iterator reads only routing fields,
                # so route each session before any body parse.  JSONL
                # providers remain file-routed through their bounded prefix.
                if kind == "hermes" and self.transcripts is not None:
                    iterator = getattr(self.transcripts, "iter_snapshots", None)
                    if iterator is not None:
                        try:
                            metadata = _await_if_needed(
                                _call_variants(
                                    iterator,
                                    [
                                        (([{"source": kind, "path": path}], None), {"parse": False}),
                                        (([{"source": kind, "path": path}],), {"project": None, "parse": False}),
                                        (([{"source": kind, "path": path}], None), {}),
                                    ],
                                )
                            )
                            for candidate in list(metadata or []):
                                route_candidates.append(
                                    {
                                        "hint": _first_attr(
                                            candidate,
                                            ("project_root", "cwd", "root", "repo_root"),
                                        ),
                                        "project_id": _first_attr(candidate, ("project_id", "project")),
                                        "session_id": _first_attr(candidate, ("session_id", "source_id")),
                                    }
                                )
                        except Exception:
                            route_candidates = []
                if not route_candidates:
                    route_candidates = [{"hint": self._source_project_hint(path, kind)}]
                candidates_count += len(route_candidates)
                for route in route_candidates:
                    hint = route.get("hint")
                    project_id_hint = route.get("project_id")
                    hint_root: Path | None = None
                    if hint is not None:
                        try:
                            if canonical is None:
                                hint_root = Path(hint).expanduser().resolve()
                            else:
                                hint_root = Path(
                                    _await_if_needed(_call_variants(canonical, [((hint,), {})]))
                                ).resolve()
                        except Exception:
                            hint_root = None
                    matches = []
                    for project in projects:
                        project_root = _first_attr(project, ("root", "repo_root"))
                        project_matches_root = False
                        if hint_root is not None and project_root:
                            with contextlib.suppress(OSError, RuntimeError):
                                project_matches_root = Path(str(project_root)).expanduser().resolve() == hint_root
                        project_matches_id = project_id_hint not in (None, "") and str(
                            _first_attr(project, ("project_id", "id", "uuid"), "")
                        ) == str(project_id_hint)
                        if project_matches_root or project_matches_id:
                            matches.append(project)
                    if len(matches) != 1:
                        unrouted += 1
                        continue
                    descriptor: dict[str, Any] = {"path": path, "source": kind}
                    if route.get("session_id") not in (None, ""):
                        descriptor["session_id"] = str(route["session_id"])
                    groups.setdefault(_project_id(matches[0]), []).append(descriptor)
        return groups, unrouted, candidates_count

    def _iter_metadata_or_snapshots(
        self,
        paths: list[dict[str, Any]],
        project: Any,
        *,
        source: str | None = None,
        host_id: str | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        function = getattr(self.transcripts, "iter_snapshots", None) if self.transcripts else None
        selected_paths = paths if limit is None or limit < 0 else paths[:limit]
        source_values = [
            {
                "source": source or item.get("source") or "codex",
                "path": item["path"],
                "host_id": host_id,
                **{
                    key: item[key]
                    for key in ("session_id", "target_session_id")
                    if item.get(key) not in (None, "")
                },
            }
            for item in selected_paths
        ]
        if function is None:
            return selected_paths
        try:
            result = _await_if_needed(
                _call_variants(
                    function,
                    [
                        (
                            (source_values, project),
                            {"host_id": host_id, "parse": False, "limit": limit},
                        ),
                        ((source_values, project), {}),
                        ((source_values,), {"project": project}),
                    ],
                )
            )
            values = list(result or [])
            # New ingestion adapters may already return normalized snapshots.
            # Preserve them and keep metadata discovery separate from body
            # parsing.  Older adapters return path metadata and are parsed
            # below with the exact ``parse_transcript`` contract.
            normalized: list[Any] = []
            for item in values:
                has_body = _first_attr(item, ("context", "text", "transcript", "content", "body")) is not None
                has_identity = _first_attr(item, ("snapshot_id", "session_id", "source_id", "uuid")) is not None
                is_canonical = _first_attr(item, ("schema_version",)) == 1 and isinstance(_first_attr(item, ("messages",)), list)
                if (has_body and has_identity) or (is_canonical and has_identity):
                    path = self._candidate_path(item)
                    mtime = path.stat().st_mtime if path and path.exists() else time.time()
                    normalized.append({"_parsed": item, "path": path, "mtime": mtime, "source": _first_attr(item, ("source", "provider"))})
                else:
                    normalized.append(item)
            return normalized
        except (OSError, ValueError, TypeError):
            # Metadata discovery remains available when an older ingestion
            # adapter does not implement the iterator contract.
            # The compatibility fallback remains metadata-only.  It does not
            # open a transcript body until ``_parse_candidate`` below.
            return paths

    def _candidate_path(self, candidate: Any) -> Path | None:
        value = _first_attr(candidate, ("path", "source_path", "file"))
        if value is None:
            return None
        return Path(str(value)).expanduser()

    def _candidate_source(self, candidate: Any, default: str | None) -> str:
        value = _first_attr(candidate, ("source", "provider", "kind"), default or "codex")
        return str(value)

    def _candidate_started_at(self, candidate: Any) -> float | None:
        value = _first_attr(
            candidate,
            ("started_at", "source_timestamp", "captured_at", "timestamp", "created_at"),
        )
        return self._message_timestamp(value)

    def _incremental_codex_capture(
        self,
        candidate: Any,
        project: Any,
        host: str,
        prior: Mapping[str, Any],
        *,
        automation_since: float | None = None,
    ) -> dict[str, Any] | None:
        """Capture only appended Codex JSONL records for native automation.

        ``None`` means the provider is not handled by this fast path.  A
        handled result always includes a ``state`` update and may omit the
        envelope when the first observation is only a baseline or no new
        visible turn was appended.
        """
        if self._candidate_source(candidate, "") != "codex":
            return None
        path = self._candidate_path(candidate)
        parser = getattr(self.transcripts, "parse_codex_incremental", None) if self.transcripts else None
        if path is None or parser is None or not path.is_file():
            return None
        try:
            size = path.stat().st_size
            mtime = path.stat().st_mtime
        except OSError as error:
            raise PipelineError("Codex source metadata is unavailable") from error
        if automation_since is None:
            automation_since = self._automation_since(create=False)
        session_id = str(_first_attr(candidate, ("session_id", "source_id"), path.stem))
        raw_offset = prior.get("source_offset")
        raw_count = prior.get("source_message_count", 0)
        try:
            offset = int(raw_offset)
            message_count = max(0, int(raw_count))
        except (TypeError, ValueError):
            offset = -1
            message_count = 0
        # A cursor belongs to one concrete file/session.  Never apply a
        # previous file's offset to a newly discovered session.
        if (
            str(prior.get("source_path") or "") != str(path)
            or str(prior.get("session_id") or "") != session_id
        ):
            offset = -1
        state_update: dict[str, Any] = {
            "source": "codex",
            "source_path": str(path),
            "source_mtime": mtime,
            "source_size": size,
            "source_offset": max(0, min(offset, size)) if offset >= 0 else size,
            "source_message_count": message_count,
            "session_id": session_id,
            "host_id": host,
        }
        if offset < 0 or offset > size:
            # A session that started after the frozen cutoff is new work.  Its
            # complete bounded file must be parsed from byte zero on the first
            # pass; baselining it at EOF would lose the opening turns forever.
            started_at = self._candidate_started_at(candidate)
            if (
                automation_since is None
                or started_at is None
                or started_at < automation_since
            ):
                state_update["source_offset"] = size
                state_update["status"] = "baseline"
                return {"state": state_update, "envelope": None}
            offset = 0
            message_count = 0
            state_update["source_offset"] = 0
            state_update["source_message_count"] = 0
        if offset == size:
            state_update["status"] = "unchanged"
            return {"state": state_update, "envelope": None}
        result = _await_if_needed(
            _call_variants(
                parser,
                [
                    (
                        (path, project),
                        {
                            "offset": offset,
                            "ordinal_start": message_count,
                            "host_id": host,
                            "session_id": session_id,
                        },
                    ),
                    ((path, project, offset), {"ordinal_start": message_count, "host_id": host, "session_id": session_id}),
                ],
            )
        )
        parsed, next_offset = result
        try:
            next_offset = int(next_offset)
        except (TypeError, ValueError) as error:
            raise PipelineError("Codex incremental parser returned an invalid cursor") from error
        if next_offset < offset or next_offset > size:
            raise PipelineError("Codex incremental parser returned an invalid cursor")
        state_update["source_offset"] = next_offset
        messages = _as_mapping(parsed).get("messages", []) if parsed is not None else []
        if not isinstance(messages, list) or not messages:
            state_update["status"] = "unchanged"
            return {"state": state_update, "envelope": None}
        state_update["source_message_count"] = message_count + len(messages)
        state_update["status"] = "queued"
        return {"state": state_update, "envelope": parsed}

    def _fair_order(self, candidates: list[Any], state: Mapping[str, Any]) -> list[Any]:
        sessions = state.get("sessions", {})
        backlog: list[Any] = []
        current: list[Any] = []
        now = time.time()
        for item in candidates:
            path = self._candidate_path(item)
            key = str(path) if path else str(_first_attr(item, ("snapshot_id", "id"), ""))
            prior = sessions.get(key, {}) if isinstance(sessions, Mapping) else {}
            mtime = float(_first_attr(item, ("mtime",), path.stat().st_mtime if path and path.exists() else now))
            if prior.get("status") in {"failed", "deferred", "pending"} or mtime < now - 86400:
                backlog.append(item)
            else:
                current.append(item)
        current.sort(key=lambda item: -float(_first_attr(item, ("mtime",), 0)))
        backlog.sort(key=lambda item: float(_first_attr(item, ("mtime",), 0)))
        output: list[Any] = []
        fresh_first = int(state.get("fair_cursor", 0)) % 2 == 0
        while current or backlog:
            if fresh_first and current:
                output.append(current.pop(0))
            elif not fresh_first and backlog:
                output.append(backlog.pop(0))
            elif current:
                output.append(current.pop(0))
            elif backlog:
                output.append(backlog.pop(0))
            fresh_first = not fresh_first
        return output

    def _parse_candidate(self, candidate: Any, source: str, project: Any, host_id: str) -> Any:
        # ``iter_snapshots`` can return an already-normalized item.  A path
        # item is metadata only and enters the parser here, after auth.
        parsed = _first_attr(candidate, ("_parsed",))
        if parsed is not None:
            return parsed
        if _first_attr(candidate, ("schema_version",)) == 1 and isinstance(_first_attr(candidate, ("messages",)), list):
            return candidate
        if self._candidate_path(candidate) is None and _first_attr(candidate, ("snapshot_id", "id")) is not None:
            return candidate
        path = self._candidate_path(candidate)
        if path is None:
            raise PipelineError("metadata candidate has no source path")
        parser = getattr(self.transcripts, "parse_transcript", None) if self.transcripts else None
        if parser is None:
            raise PipelineError("ACE transcript parser is unavailable")
        return _await_if_needed(
            _call_variants(
                parser,
                [
                    (
                        (path, source, project),
                        {
                            "host_id": host_id,
                            "session_id": _first_attr(candidate, ("session_id", "source_id")),
                        },
                    ),
                    ((path, source, project), {"host_id": host_id}),
                    ((path, source, project, host_id), {}),
                    ((path, source, project), {}),
                ],
            )
        )

    def _snapshot_id(self, snapshot: Any, source: str, path: Path | None, mtime: float) -> str:
        value = _first_attr(snapshot, ("snapshot_id", "session_id", "source_id", "id", "uuid"))
        if value:
            return str(value)
        seed = f"{source}|{path or ''}|{mtime}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]

    def _envelope(self, snapshot: Any, project: Any, source: str, host_id: str, path: Path | None, mtime: float) -> dict[str, Any]:
        payload = _as_mapping(snapshot)
        raw = json.dumps(payload, ensure_ascii=False)
        try:
            clean = json.loads(_redact(raw))
        except json.JSONDecodeError:
            clean = payload
        snapshot_id = self._snapshot_id(snapshot, source, path, mtime)
        project_descriptor = _as_mapping(project)
        project_descriptor = {
            "id": str(_first_attr(project_descriptor, ("id", "project_id"), _project_id(project))),
            "name": str(_first_attr(project_descriptor, ("name", "slug"), _project_name(project))),
            "root": str(_first_attr(project_descriptor, ("root", "repo_root"), _first_attr(project, ("root",), ""))),
            "vault_dir": str(_first_attr(project_descriptor, ("vault_dir", "vault_path"), self._project_vault_dir(project))),
        }
        # Ingestion adapters already produce the canonical ACE envelope.  Do
        # not wrap it in a second ``snapshot`` object because the outbox and
        # database share this exact normalized shape.
        is_canonical = clean.get("schema_version") == 1 and isinstance(clean.get("messages"), list)
        if is_canonical:
            canonical = dict(clean)
            canonical.setdefault("project", project_descriptor)
            # Keep a flat identity for local integrations and legacy test
            # stores.  Supabase normalisation uses the canonical nested
            # ``project.id`` field and ignores this compatibility key.
            canonical.setdefault("project_id", project_descriptor["id"])
            canonical.setdefault("source", source)
            canonical.setdefault("host_id", host_id)
            canonical.setdefault("source_path", str(path) if path else None)
            canonical["snapshot_id"] = snapshot_id
            return canonical

        context = _first_attr(clean, ("context", "text", "transcript", "content", "body"), "")
        messages = clean.get("messages") if isinstance(clean.get("messages"), list) else []
        if not messages and context:
            messages = [{"id": f"msg-{snapshot_id[:16]}", "ordinal": 0, "role": "user", "type": "message", "content": context}]
        session_id = str(_first_attr(clean, ("session_id", "source_id", "id"), snapshot_id))
        revision = str(_first_attr(clean, ("revision",), ""))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", revision):
            revision = hashlib.sha256(json.dumps({"session_id": session_id, "messages": messages}, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        timestamp = _normalise_datetime(_first_attr(clean, ("started_at", "captured_at", "timestamp", "created_at")), datetime.fromtimestamp(mtime, tz=timezone.utc)).isoformat()
        return {
            "schema_version": 1,
            "project": project_descriptor,
            "project_id": project_descriptor["id"],
            "source": source,
            "session_id": session_id,
            "revision": revision.lower(),
            "source_path": str(_first_attr(clean, ("source_path", "path"), path or "snapshot.json")),
            "host_id": host_id,
            "started_at": timestamp,
            "updated_at": timestamp,
            "messages": messages,
            "attachments": clean.get("attachments", []) if isinstance(clean.get("attachments"), list) else [],
            "snapshot_id": snapshot_id,
        }

    def _outbox_key(self, item: Any, envelope: Mapping[str, Any] | None = None) -> str:
        value = _first_attr(item, ("key", "id", "snapshot_id"))
        if value is None and isinstance(item, Mapping):
            nested = item.get("envelope")
            value = _first_attr(nested, ("snapshot_id", "id"))
        if value is None and envelope is not None:
            value = envelope.get("snapshot_id")
        return str(value or "")

    def collect(
        self,
        source_paths: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        source: str | None = None,
        host_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
        days: int = 7,
        all_history: bool = False,
        sync: bool = False,
        extract: bool = False,
        since: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        # Native scheduling invokes ``collect --sync`` without a project path.
        # Enumerate only explicitly registered projects and run the same
        # bounded collector once per project.  No vault-folder basename is a
        # permission grant.
        if not source_paths and cwd is None:
            listing = getattr(self.projects, "list_initialized_projects", None) if self.projects else None
            projects = self._processable_projects() if listing is not None else []
            aggregate: dict[str, Any] = {
                "projects": 0,
                "candidates": 0,
                "queued": 0,
                "unchanged": 0,
                "failed": 0,
                "deferred": 0,
                "unexamined": 0,
                "unrouted": 0,
                "offline": False,
            }
            groups, unrouted, discovered_count = self._auto_source_groups(
                source_paths,
                projects,
                source=source,
                all_history=all_history,
                days=days,
                since=since,
            )
            aggregate["candidates"] = discovered_count
            aggregate["unrouted"] = unrouted
            by_id = {_project_id(item): item for item in projects}
            for project_id, discovered in groups.items():
                item = by_id[project_id]
                result = self.collect(
                    discovered,
                    cwd=_first_attr(item, ("root",), None),
                    source=source,
                    host_id=host_id,
                    limit=limit,
                    days=days,
                    all_history=all_history,
                    sync=sync,
                    extract=extract,
                    since=since,
                    dry_run=dry_run,
                )
                aggregate["projects"] += 1
                if dry_run:
                    aggregate.setdefault("would_queue", []).extend(result.get("would_queue", []))
                for key in ("queued", "unchanged", "failed", "deferred", "unexamined"):
                    aggregate[key] += int(result.get(key, 0) or 0)
                aggregate["offline"] = aggregate["offline"] or bool(result.get("offline"))
            return aggregate

        project = self._resolve_project(cwd)
        state = self.state.read("collection")
        # Route/order metadata first, then parse only the bounded prefix.  The
        # native transcript iterator may parse bodies while yielding; calling
        # it for the whole discovered set would defeat ``limit`` and make an
        # interrupted tick expensive and non-resumable.
        metadata_candidates = self._source_paths(source_paths, all_history, days, since)
        metadata_candidates = self._fair_order(metadata_candidates, state)
        if dry_run:
            # Preview only: list what a real run would examine for this
            # project. No parse, no queue, no cursor, no state write.
            selected = metadata_candidates if limit < 0 else metadata_candidates[:limit]
            return {
                "dry_run": True,
                "project": _project_name(project),
                "candidates": len(metadata_candidates),
                "would_examine": len(selected),
                "would_queue": [
                    str(self._candidate_path(item) or _first_attr(item, ("snapshot_id", "id"), ""))
                    for item in selected
                ][:200],
                "queued": 0,
                "unchanged": 0,
                "failed": 0,
                "deferred": 0,
                "unexamined": len(metadata_candidates) - len(selected),
                "offline": False,
            }
        if limit >= 0:
            selected_metadata = metadata_candidates[:limit]
            deferred_metadata = metadata_candidates[limit:]
        else:
            selected_metadata = metadata_candidates
            deferred_metadata = []
        candidates = self._iter_metadata_or_snapshots(
            selected_metadata,
            project,
            source=source,
            host_id=host_id or self.host_id,
        )
        # A provider (notably Hermes) can yield more than one snapshot for a
        # single source file.  The public limit is a snapshot bound too.
        overflow = []
        if limit >= 0 and len(candidates) > limit:
            overflow = candidates[limit:]
            candidates = candidates[:limit]
        outbox = self._get_outbox()
        host = host_id or self.host_id
        counts: dict[str, Any] = {"candidates": len(metadata_candidates), "queued": 0, "unchanged": 0, "failed": 0, "deferred": 0, "unexamined": len(deferred_metadata) + len(overflow), "offline": False}
        sessions = state.setdefault("sessions", {})
        for candidate in deferred_metadata + overflow:
            key = str(self._candidate_path(candidate) or _first_attr(candidate, ("snapshot_id", "id"), ""))
            if key:
                prior = sessions.get(key)
                deferred = dict(prior) if isinstance(prior, Mapping) else {}
                deferred["status"] = "deferred"
                path = self._candidate_path(candidate)
                if path is not None:
                    deferred.setdefault("source_path", str(path))
                sessions[key] = deferred
        processed = 0
        for position, candidate in enumerate(candidates):
            if limit >= 0 and processed >= limit:
                key = str(self._candidate_path(candidate) or _first_attr(candidate, ("snapshot_id", "id"), position))
                sessions[key] = {"status": "deferred"}
                counts["unexamined"] += len(candidates) - position
                continue
            path = self._candidate_path(candidate)
            mtime = float(_first_attr(candidate, ("mtime",), path.stat().st_mtime if path and path.exists() else time.time()))
            candidate_source = self._candidate_source(candidate, source)
            key = str(path or _first_attr(candidate, ("snapshot_id", "id"), position))
            prior = sessions.get(key, {})
            capture_state: dict[str, Any] = {}
            try:
                automatic_capture = (
                    self._automation_since(create=False) is not None
                    and not all_history
                    and candidate_source == "codex"
                )
                incremental = (
                    self._incremental_codex_capture(
                        candidate,
                        project,
                        host,
                        prior,
                        automation_since=self._automation_since(create=False),
                    )
                    if automatic_capture
                    else None
                )
                if automatic_capture and incremental is not None:
                    capture_state = dict(incremental["state"])
                    envelope = None
                    if incremental.get("envelope") is not None:
                        envelope = self._envelope(
                            incremental["envelope"], project, candidate_source, host, path, mtime
                        )
                    if envelope is None:
                        sessions[key] = capture_state
                        counts["unchanged"] += 1
                        state["fair_cursor"] = int(state.get("fair_cursor", 0)) + 1
                        self.state.write("collection", state)
                        continue
                else:
                    parsed = self._parse_candidate(candidate, candidate_source, project, host)
                    envelope = self._envelope(parsed, project, candidate_source, host, path, mtime)
                    capture_state = {}
                if (
                    prior.get("snapshot_id") == envelope["snapshot_id"]
                    and prior.get("status") in {"queued", "synced", "processed"}
                    and prior.get("source_mtime") == mtime
                    and prior.get("source_size") == (path.stat().st_size if path and path.exists() else None)
                ):
                    counts["unchanged"] += 1
                    continue
                result = _await_if_needed(outbox.enqueue(envelope))
                sessions[key] = {
                    **capture_state,
                    "status": "queued",
                    "snapshot_id": envelope["snapshot_id"],
                    "source": candidate_source,
                    "host_id": host,
                    "source_mtime": mtime,
                    "source_size": path.stat().st_size if path and path.exists() else None,
                }
                counts["queued"] += 1
                processed += 1
                _ = result
            except Exception as error:
                # Keep the last acknowledged byte/message cursor on parser,
                # queue, or transport failure.  The next retry then resumes
                # from the last safe point instead of silently resetting to a
                # baseline or losing an appended turn.
                failed_state = dict(prior) if isinstance(prior, Mapping) else {}
                failed_state.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "source": candidate_source,
                        "host_id": host,
                        "source_mtime": mtime,
                        "source_size": path.stat().st_size if path and path.exists() else None,
                    }
                )
                if path is not None:
                    failed_state.setdefault("source_path", str(path))
                sessions[key] = failed_state
                counts["failed"] += 1
                processed += 1
            state["fair_cursor"] = int(state.get("fair_cursor", 0)) + 1
            self.state.write("collection", state)
        if sync:
            counts["sync"] = self.sync(
                project=project,
                limit=max(1, limit),
                created_after=None if all_history else self._automation_since(create=True),
            )
        self._record_collection_coverage(state, project, counts)
        self.state.write("collection", state)
        if extract and int(counts.get("queued", 0) or 0) > 0:
            # Session-end immediacy of the pre-ACE flush: write the daily log
            # from the local envelope right after capture, without the
            # database and without the native 30-minute cycle.
            try:
                counts["extract"] = self.process_local(project=project, limit=max(1, limit))
            except PipelineBusyError:
                counts["extract"] = {"status": "busy"}
        return counts

    def _pending_items(
        self,
        limit: int | None = None,
        project_id: str | None = None,
        created_after: float | None = None,
    ) -> list[Any]:
        outbox = self._get_outbox()
        method = getattr(outbox, "pending", None)
        if method is None:
            return []
        if created_after is not None:
            # The native SQLite adapter applies this predicate before
            # leasing. If an older adapter cannot do that, fail closed rather
            # than claiming an historical row and filtering it too late.
            if project_id is None:
                variants = [
                    ((limit,), {"created_after": created_after}),
                    ((), {"limit": limit, "created_after": created_after}),
                ]
            else:
                variants = [
                    ((limit,), {"project_id": project_id, "created_after": created_after}),
                    ((), {"limit": limit, "project_id": project_id, "created_after": created_after}),
                ]
            try:
                values = _await_if_needed(_call_variants(method, variants))
            except TypeError:
                return []
            return list(values or [])
        if project_id is None:
            variants = [((limit,), {}), ((), {"limit": limit}), ((), {})]
        else:
            # The native SQLite adapter accepts the scoped keyword and applies
            # it inside its claim transaction.  Keep older in-memory fakes
            # usable by falling back to their historical limit-only method;
            # those adapters do not lease rows in the real queue.
            variants = [
                ((limit,), {"project_id": project_id}),
                ((), {"limit": limit, "project_id": project_id}),
                ((limit,), {}),
                ((), {"limit": limit}),
                ((), {}),
            ]
        values = _await_if_needed(_call_variants(method, variants))
        return list(values or [])

    def _outbox_pending_count(self, outbox: Any, project_id: str | None = None) -> int | None:
        """Read pending outbox rows without claiming a new lot.

        ``Outbox.pending`` is a claim operation, so status and coverage paths
        must use a read-only SQL query when the SQLite connection is present.
        Small injected fakes retain a non-mutating ``pending`` fallback.
        """
        connection = getattr(outbox, "connection", None)
        if connection is not None:
            try:
                if project_id:
                    row = connection.execute(
                        "SELECT COUNT(*) FROM ace_outbox "
                        "WHERE project_id=? AND status IN ('pending','retry','inflight')",
                        (project_id,),
                    ).fetchone()
                else:
                    row = connection.execute(
                        "SELECT COUNT(*) FROM ace_outbox "
                        "WHERE status IN ('pending','retry','inflight')"
                    ).fetchone()
                return int(row[0]) if row is not None else 0
            except (sqlite3.Error, TypeError, ValueError):
                return None
        summary = getattr(outbox, "summary", None)
        if callable(summary):
            try:
                value = summary()
                if (
                    not project_id
                    and isinstance(value, Mapping)
                    and isinstance(value.get("pending"), (int, float))
                ):
                    return int(value["pending"])
                statuses = value.get("statuses") if isinstance(value, Mapping) else None
                if isinstance(statuses, Mapping) and not project_id:
                    return sum(
                        int(statuses.get(name, {}).get("count", 0) or 0)
                        for name in ("pending", "retry", "inflight")
                        if isinstance(statuses.get(name), Mapping)
                    )
            except Exception:
                pass
        # Only fakes without a SQLite connection reach this branch.  Calling
        # their pending method is safe for the test/in-memory adapters.
        method = getattr(outbox, "pending", None)
        if not callable(method):
            return None
        try:
            values = _await_if_needed(_call_variants(method, [((), {}), ((None,), {})]))
            if not project_id:
                return len(list(values or []))
            count = 0
            for item in values or []:
                envelope = _first_attr(item, ("envelope", "payload"), item)
                item_project = str(
                    _first_attr(envelope, ("project_id",), "")
                    or _first_attr(_first_attr(envelope, ("project",), {}), ("id", "project_id"), "")
                )
                if item_project == project_id:
                    count += 1
            return count
        except Exception:
            return None

    def _record_collection_coverage(
        self, state: dict[str, Any], project: Any, counts: Mapping[str, Any]
    ) -> None:
        """Persist one timestamped, project-scoped collection observation."""
        projects = state.setdefault("projects", {})
        if not isinstance(projects, dict):
            projects = {}
            state["projects"] = projects
        project_id = _project_id(project)
        record = projects.setdefault(project_id, {})
        if not isinstance(record, dict):
            record = {}
            projects[project_id] = record
        candidates = int(counts.get("candidates", 0) or 0)
        unexamined = int(counts.get("unexamined", 0) or 0)
        queued = int(counts.get("queued", 0) or 0)
        unchanged = int(counts.get("unchanged", 0) or 0)
        coverage: dict[str, Any] = {
            "candidates": max(0, candidates),
            "ingested": max(0, queued + unchanged),
            "unexamined": max(0, unexamined),
            "failed": max(0, int(counts.get("failed", 0) or 0)),
            "calls": max(0, candidates - unexamined),
            "unchanged": max(0, unchanged),
        }
        # ``deferred``/``active`` are optional until a source adapter measures
        # them explicitly; omitting them lets the renderer say "inconnu".
        if counts.get("deferred") not in (None, 0, "0"):
            coverage["deferred"] = max(0, int(counts["deferred"]))
        pending = (
            self._outbox_pending_count(self._outbox_instance, project_id)
            if self._outbox_instance is not None
            else None
        )
        if pending is not None:
            coverage["pending_count"] = max(0, pending)
        record["last_run_at"] = _normalise_datetime(self.now()).astimezone(PARIS).isoformat()
        record["coverage"] = coverage
        record["selection_cursor"] = state.get("fair_cursor")

    def sync(
        self,
        *,
        project: Any | None = None,
        cwd: str | Path | None = None,
        limit: int = 100,
        created_after: float | None = None,
    ) -> dict[str, Any]:
        if project is None and cwd is not None:
            project = self._resolve_project(cwd)
        allowed = {_project_id(project)} if project is not None else self._project_ids()
        items = self._pending_items(
            limit,
            project_id=_project_id(project) if project is not None else None,
            created_after=created_after,
        )
        store = self._get_store()
        outbox = self._outbox_instance
        counts: dict[str, Any] = {"pending": len(items), "synced": 0, "failed": 0, "offline": False, "skipped": 0}
        sync_state = self.state.read("sync")
        records = sync_state.setdefault("snapshots", {})
        if store is None:
            counts["offline"] = bool(items)
            for item in items:
                envelope = _first_attr(item, ("envelope", "payload"), item)
                if allowed and str(_first_attr(envelope, ("project_id",), "")) not in allowed:
                    counts["skipped"] += 1
            remaining = self._outbox_pending_count(outbox, _project_id(project) if project is not None else None)
            if remaining is not None:
                counts["pending"] = remaining
            sync_state["last_status"] = "offline"
            self.state.write("sync", sync_state)
            return counts
        writer = self._store_method(
            store,
            (
                "ingest_snapshot",
                "upsert_normalized_snapshot",
                "upsert_snapshot",
                "sync_snapshot",
                "write_snapshot",
                "upsert",
            ),
        )
        if writer is None:
            counts["offline"] = bool(items)
            remaining = self._outbox_pending_count(outbox, _project_id(project) if project is not None else None)
            if remaining is not None:
                counts["pending"] = remaining
            self.state.write("sync", sync_state)
            return counts
        for item in items:
            envelope = _first_attr(item, ("envelope", "payload"), item)
            envelope = dict(_as_mapping(envelope))
            item_project = str(
                envelope.get("project_id")
                or _first_attr(envelope.get("project"), ("id", "project_id"), "")
            )
            if allowed and item_project not in allowed:
                counts["skipped"] += 1
                continue
            key = self._outbox_key(item, envelope)
            payload_bytes = _first_attr(item, ("payload_bytes",), None)
            if (
                self._automation_enabled()
                and isinstance(payload_bytes, (int, float))
                and payload_bytes > AUTOMATION_MAX_PAYLOAD_BYTES
            ):
                fail = getattr(outbox, "fail", None)
                if fail is not None:
                    with contextlib.suppress(Exception):
                        _await_if_needed(
                            _call_variants(
                                fail,
                                [
                                    ((key, "automatic payload deferred"), {"retry_at": time.time() + 86400}),
                                    ((key, "automatic payload deferred"), {}),
                                ],
                            )
                        )
                records[key] = {
                    "status": "deferred",
                    "project_id": item_project,
                    "error_type": "PayloadTooLargeError",
                }
                counts["skipped"] += 1
                continue
            try:
                receipt = _await_if_needed(_call_variants(writer, [((envelope,), {}), ((envelope.get("snapshot"),), {"project_id": item_project})]))
                ack = getattr(outbox, "ack", None)
                if ack is not None:
                    _await_if_needed(_call_variants(ack, [((key, receipt), {}), ((key,), {})]))
                records[key] = {"status": "acquitted", "project_id": item_project, "receipt": _redact(receipt or "")}
                counts["synced"] += 1
            except Exception as error:
                fail = getattr(outbox, "fail", None)
                if fail is not None:
                    with contextlib.suppress(Exception):
                        _await_if_needed(_call_variants(fail, [((key, type(error).__name__), {}), ((key,), {})]))
                records[key] = {"status": "failed", "project_id": item_project, "error_type": type(error).__name__}
                counts["failed"] += 1
        remaining = self._outbox_pending_count(outbox, _project_id(project) if project is not None else None)
        if remaining is not None:
            counts["pending"] = remaining
        sync_state["last_status"] = "online" if counts["failed"] == 0 else "partial"
        self.state.write("sync", sync_state)
        return counts

    def _db_snapshots(
        self,
        store: Any,
        project: Any | None,
        limit: int,
        *,
        stage: str = "extraction",
        minimum_started_at: float | None = None,
        extraction_cursors: Mapping[str, Any] | None = None,
        source_after: float | None = None,
        source_before: float | None = None,
    ) -> list[Any]:
        if limit <= 0:
            return []
        project_id = _project_id(project) if project is not None else None
        # Push every lower bound into the storage query before its LIMIT.  A
        # historical pending prefix must not hide a current row when the
        # native cutoff and an exact source-day window are combined.
        effective_after = source_after
        if minimum_started_at is not None:
            if effective_after is None or minimum_started_at > effective_after:
                effective_after = minimum_started_at
        # The old pending_snapshots RPC returns the complete JSON envelope.
        # That is safe for short fixtures but can exceed the SQL transport for
        # a long live conversation.  The incremental service first requests
        # small identities, then asks Supabase for one message delta per
        # session.  Historical/manual calls retain the legacy full-envelope
        # path until an explicit cursor exists.
        ref_reader = self._store_method(store, ("pending_snapshot_refs",))
        delta_reader = self._store_method(store, ("snapshot_delta",))
        batch_delta_reader = self._store_method(store, ("snapshot_deltas", "snapshot_delta_many"))
        if (
            minimum_started_at is None
            and (source_after is not None or source_before is not None)
            and ref_reader is not None
            and batch_delta_reader is not None
        ):
            # An explicit daily audit has an exact source-day window and no
            # live cursor. Fetch only the matching identities, then resolve
            # their bodies in one bounded database-wrapper call. This keeps
            # the manual path from paying the Bitwarden throttle once per
            # conversation while preserving the same evidence envelopes.
            query_limit = max(REF_QUERY_PAGE_LIMIT, min(500, limit))
            ref_window = {
                "project_id": project_id,
                "limit": query_limit,
                "stage": stage,
                "source_after": source_after,
                "source_before": source_before,
            }
            refs = _await_if_needed(
                _call_variants(
                    ref_reader,
                    [
                        ((), ref_window),
                        ((), {"project_id": project_id, "limit": query_limit}),
                        ((query_limit, stage, project_id), {}),
                    ],
                )
            )
            matching_refs: list[Mapping[str, Any]] = []
            requests: list[dict[str, Any]] = []
            for ref in list(refs or []):
                if not isinstance(ref, Mapping):
                    continue
                source = str(ref.get("source") or "")
                session_id = str(ref.get("session_id") or "")
                revision = str(ref.get("revision") or "")
                if not source or not session_id or not revision:
                    continue
                if not self._source_window_matches(ref, source_after, source_before):
                    continue
                matching_refs.append(ref)
                requests.append(
                    {
                        "project_id": str(ref.get("project_id") or project_id or ""),
                        "source": source,
                        "session_id": session_id,
                        "revision": revision,
                        "last_ordinal": -1,
                    }
                )
            if not requests:
                return []
            values = _await_if_needed(
                _call_variants(
                    batch_delta_reader,
                    [((requests,), {}), ((), {"requests": requests})],
                )
            )

            def identity(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
                payload = self._snapshot_payload(value)
                project_value = value.get("project_id") or payload.get("project_id")
                if not project_value and isinstance(payload.get("project"), Mapping):
                    project_value = payload["project"].get("id")
                return (
                    str(project_value or ""),
                    str(value.get("source") or payload.get("source") or ""),
                    str(value.get("session_id") or payload.get("session_id") or ""),
                    str(value.get("revision") or payload.get("revision") or ""),
                )

            resolved = {
                identity(dict(_as_mapping(value))): dict(_as_mapping(value))
                for value in list(values or [])
                if isinstance(value, Mapping)
            }
            rows: list[Any] = []
            for ref in matching_refs:
                key = (
                    str(ref.get("project_id") or project_id or ""),
                    str(ref.get("source") or ""),
                    str(ref.get("session_id") or ""),
                    str(ref.get("revision") or ""),
                )
                delta = resolved.get(key)
                if delta is None:
                    continue
                for field in (
                    "project_id",
                    "source",
                    "session_id",
                    "revision",
                    "source_path",
                    "host_id",
                    "started_at",
                    "updated_at",
                ):
                    if delta.get(field) in (None, "") and ref.get(field) not in (None, ""):
                        delta[field] = ref[field]
                if self._source_window_matches(delta, source_after, source_before):
                    rows.append(delta)
            return rows[:limit]
        if (
            (minimum_started_at is not None or source_after is not None or source_before is not None)
            and ref_reader is not None
            and delta_reader is not None
        ):
            # A singleton automation pass must look past the historical
            # pending prefix before applying the time cutoff.  Asking
            # Supabase for only ``limit`` refs (normally one) can return an
            # old row, filter it out, and incorrectly report zero current
            # candidates forever.
            # Filtered windows must page past old rows before applying the
            # caller's small processing limit.
            query_limit = REF_QUERY_PAGE_LIMIT
            ref_window = {
                "project_id": project_id,
                "limit": query_limit,
                "stage": stage,
            }
            if effective_after is not None:
                ref_window["source_after"] = effective_after
            if source_before is not None:
                ref_window["source_before"] = source_before
            refs = _await_if_needed(
                _call_variants(
                    ref_reader,
                    [
                        ((), ref_window),
                        ((), {"project_id": project_id, "limit": query_limit}),
                        ((query_limit, stage, project_id), {}),
                    ],
                )
            )
            rows: list[Any] = []
            known_cursors = extraction_cursors or {}
            for ref in list(refs or []):
                if not isinstance(ref, Mapping):
                    continue
                source = str(ref.get("source") or "")
                session_id = str(ref.get("session_id") or "")
                revision = str(ref.get("revision") or "")
                if not source or not session_id or not revision:
                    continue
                # Do not inherit the historical pending queue into the live
                # 30-minute pass. Only a revision changed after the service's
                # cutoff is eligible for automatic processing.
                changed_at = self._message_timestamp(
                    _first_attr(
                        ref,
                        ("updated_at", "received_at", "ingested_at", "captured_at", "timestamp", "started_at"),
                    )
                )
                if (
                    minimum_started_at is not None
                    and (changed_at is None or changed_at < minimum_started_at)
                ):
                    continue
                if not self._source_window_matches(ref, source_after, source_before):
                    continue
                baseline_snapshot = {
                    "schema_version": 1,
                    "project": {"id": str(ref.get("project_id") or project_id or "")},
                    "source": source,
                    "session_id": session_id,
                    "revision": revision,
                    "source_path": ref.get("source_path"),
                    "host_id": ref.get("host_id"),
                    "started_at": ref.get("started_at"),
                    "updated_at": ref.get("updated_at"),
                    "messages": [],
                    "attachments": [],
                }
                baseline_item = dict(ref)
                baseline_item["snapshot"] = baseline_snapshot
                cursor_key = self._extraction_cursor_key(project, baseline_item)
                cursor = known_cursors.get(cursor_key, {})
                if not isinstance(cursor, Mapping) or "last_ordinal" not in cursor:
                    # A session created after the automation cutoff is new
                    # live work, not historical backlog. Fetch its complete
                    # bounded first delta so the current conversation reaches
                    # extraction and the daily log. Older sessions remain
                    # baselined without replaying their body.
                    started_at = self._message_timestamp(ref.get("started_at"))
                    if (
                        (minimum_started_at is None and stage == "analysis")
                        or (
                            minimum_started_at is not None
                            and started_at is not None
                            and started_at >= minimum_started_at
                        )
                    ):
                        # A manual daily audit has a source-day window but
                        # no automation cutoff. It still needs the actual
                        # snapshot; a metadata baseline is not evidence.
                        delta = _await_if_needed(
                            _call_variants(
                                delta_reader,
                                [
                                    (
                                        (
                                            str(ref.get("project_id") or project_id),
                                            source,
                                            session_id,
                                            revision,
                                            -1,
                                        ),
                                        {},
                                    ),
                                    (
                                        (),
                                        {
                                            "project_id": str(ref.get("project_id") or project_id),
                                            "source": source,
                                            "session_id": session_id,
                                            "revision": revision,
                                            "last_ordinal": -1,
                                        },
                                    ),
                                ],
                            )
                        )
                        if delta is not None:
                            delta = dict(_as_mapping(delta))
                            for key in (
                                "project_id",
                                "source",
                                "session_id",
                                "revision",
                                "source_path",
                                "host_id",
                                "started_at",
                                "updated_at",
                            ):
                                if delta.get(key) in (None, "") and ref.get(key) not in (None, ""):
                                    delta[key] = ref[key]
                            if self._source_window_matches(delta, source_after, source_before):
                                rows.append(delta)
                            continue
                    if self._source_window_matches(baseline_item, source_after, source_before):
                        rows.append(baseline_item)
                    continue
                try:
                    last_ordinal = int(cursor.get("last_ordinal", -1))
                except (TypeError, ValueError):
                    last_ordinal = -1
                delta = _await_if_needed(
                    _call_variants(
                        delta_reader,
                        [
                            ((str(ref.get("project_id") or project_id), source, session_id, revision, last_ordinal), {}),
                            ((), {
                                "project_id": str(ref.get("project_id") or project_id),
                                "source": source,
                                "session_id": session_id,
                                "revision": revision,
                                "last_ordinal": last_ordinal,
                            }),
                        ],
                    )
                )
                if delta is not None:
                    delta = dict(_as_mapping(delta))
                    for key in (
                        "project_id",
                        "source",
                        "session_id",
                        "revision",
                        "source_path",
                        "host_id",
                        "started_at",
                        "updated_at",
                    ):
                        if delta.get(key) in (None, "") and ref.get(key) not in (None, ""):
                            delta[key] = ref[key]
                    if self._source_window_matches(delta, source_after, source_before):
                        rows.append(delta)
            return rows[:limit]
        reader = self._store_method(
            store,
            (
                "list_acquitted_snapshots",
                "acquitted_snapshots",
                "fetch_acquitted_snapshots",
                "pending_extraction",
                "pending_snapshots",
            ),
        )
        if reader is not None:
            # Incremental processing must see past the historical pending
            # prefix.  The database function orders unattempted rows first,
            # so using the caller's small extraction limit here could hide a
            # newly received conversation behind old backlog forever.
            query_limit = (
                500
                if minimum_started_at is not None
                or source_after is not None
                or source_before is not None
                else limit
            )
            reader_window = {
                "project_id": project_id,
                "limit": query_limit,
                "stage": stage,
            }
            if effective_after is not None:
                reader_window["source_after"] = effective_after
            if source_before is not None:
                reader_window["source_before"] = source_before
            values = _await_if_needed(
                _call_variants(
                    reader,
                    [
                        ((), reader_window),
                        ((), {"project_id": project_id, "limit": query_limit}),
                        ((), {"limit": query_limit, "stage": stage}),
                        ((project_id, query_limit), {}),
                        ((project_id,), {"limit": query_limit}),
                        ((), {"limit": query_limit}),
                    ],
                )
            )
            rows = list(values or [])
            if project_id is not None:
                rows = [
                    row
                    for row in rows
                    if str(
                        _first_attr(
                            _first_attr(row, ("project",), row),
                            ("id", "project_id"),
                            _first_attr(row, ("project_id",), ""),
                        )
                    )
                    == project_id
                ]
            if minimum_started_at is not None:
                filtered: list[Any] = []
                for row in rows:
                    payload = self._snapshot_payload(row)
                    value = _first_attr(
                        row,
                        ("received_at", "ingested_at", "updated_at", "captured_at", "timestamp", "started_at"),
                        _first_attr(payload, ("received_at", "ingested_at", "updated_at", "captured_at", "timestamp", "started_at")),
                    )
                    if value in (None, ""):
                        continue
                    with contextlib.suppress(TypeError, ValueError, OverflowError):
                        if _normalise_datetime(value).timestamp() >= minimum_started_at:
                            filtered.append(row)
                rows = filtered
            if source_after is not None or source_before is not None:
                rows = [
                    row
                    for row in rows
                    if self._source_window_matches(row, source_after, source_before)
                ]
            return rows[:limit]
        generic = self._store_method(store, ("list_snapshots", "snapshots"))
        if generic is None:
            return []
        query_limit = (
            500
            if minimum_started_at is not None
            or source_after is not None
            or source_before is not None
            else limit
        )
        generic_window = {
            "project_id": project_id,
            "status": "acquitted",
            "limit": query_limit,
        }
        if effective_after is not None:
            generic_window["source_after"] = effective_after
        if source_before is not None:
            generic_window["source_before"] = source_before
        values = _await_if_needed(_call_variants(generic, [((), generic_window), ((), {})]))
        result: list[Any] = []
        for value in values or []:
            status = str(_first_attr(value, ("status", "sync_status"), "acquitted")).lower()
            if status in {"acquitted", "accepted", "synced", "db_ack"}:
                result.append(value)
        if minimum_started_at is not None:
            filtered_result: list[Any] = []
            for item in result:
                payload = self._snapshot_payload(item)
                value = _first_attr(payload, ("received_at", "ingested_at", "updated_at", "captured_at", "timestamp", "started_at"))
                if value in (None, ""):
                    continue
                with contextlib.suppress(TypeError, ValueError, OverflowError):
                    if _normalise_datetime(value).timestamp() >= minimum_started_at:
                        filtered_result.append(item)
            result = filtered_result
        if source_after is not None or source_before is not None:
            result = [
                item
                for item in result
                if self._source_window_matches(item, source_after, source_before)
            ]
        return result[:limit]

    def _snapshot_payload(self, item: Any) -> dict[str, Any]:
        payload = _first_attr(item, ("snapshot", "normalized_snapshot", "payload", "envelope"), item)
        return _as_mapping(payload)

    @staticmethod
    def _project_timezone(project: Any | None) -> ZoneInfo:
        raw = _first_attr(project, ("timezone", "time_zone", "tz"), "Europe/Paris")
        if isinstance(raw, ZoneInfo):
            return raw
        try:
            return ZoneInfo(str(raw))
        except (TypeError, ValueError):
            return PARIS

    def _source_day_bounds(self, project: Any, target: date) -> tuple[float, float]:
        """Return the exact local project day as a half-open UTC window."""
        tz = self._project_timezone(project)
        start = datetime.combine(target, datetime.min.time(), tzinfo=tz)
        end = datetime.combine(target + timedelta(days=1), datetime.min.time(), tzinfo=tz)
        return (
            start.astimezone(timezone.utc).timestamp(),
            end.astimezone(timezone.utc).timestamp(),
        )

    def _snapshot_window_timestamp(self, item: Any) -> float | None:
        """Return the DB activity timestamp used by source window queries."""
        payload = self._snapshot_payload(item)
        for value in (item, payload):
            parsed = self._message_timestamp(
                _first_attr(
                    value,
                    (
                        "updated_at",
                        "received_at",
                        "ingested_at",
                        "captured_at",
                        "timestamp",
                        "started_at",
                    ),
                )
            )
            if parsed is not None:
                return parsed
        return None

    def _source_window_matches(
        self,
        item: Any,
        source_after: float | None,
        source_before: float | None,
    ) -> bool:
        if source_after is None and source_before is None:
            return True
        timestamp = self._snapshot_window_timestamp(item)
        if timestamp is None:
            return False
        if source_after is not None and timestamp < source_after:
            return False
        if source_before is not None and timestamp >= source_before:
            return False
        return True

    def _snapshot_matches_day(self, item: Any, target: date, project: Any) -> bool:
        """Check source/message dates in the project's timezone."""
        tz = self._project_timezone(project)
        payload = self._snapshot_payload(item)
        source_dates: set[date] = set()
        for value in (item, payload):
            raw = _first_attr(
                value,
                ("started_at", "source_timestamp", "captured_at", "timestamp", "created_at"),
            )
            parsed = self._message_timestamp(raw)
            if parsed is not None:
                source_dates.add(datetime.fromtimestamp(parsed, tz=timezone.utc).astimezone(tz).date())
        if target in source_dates:
            return True
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            parsed = self._message_timestamp(message.get("timestamp"))
            if parsed is not None and datetime.fromtimestamp(parsed, tz=timezone.utc).astimezone(tz).date() == target:
                return True
        return False

    def _snapshot_for_day(self, item: Any, target: date, project: Any) -> dict[str, Any] | None:
        """Bound evidence to one project's local source day."""
        if not self._snapshot_matches_day(item, target, project):
            return None
        payload = self._snapshot_payload(item)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return dict(_as_mapping(item))
        tz = self._project_timezone(project)
        timestamped = [
            (
                message,
                self._message_timestamp(message.get("timestamp")),
            )
            for message in messages
            if isinstance(message, Mapping)
        ]
        known = [(message, stamp) for message, stamp in timestamped if stamp is not None]
        if known:
            filtered = [
                message
                for message, stamp in known
                if datetime.fromtimestamp(stamp, tz=timezone.utc).astimezone(tz).date() == target
            ]
            if not filtered:
                return None
            payload = dict(payload)
            payload["messages"] = filtered
        if isinstance(item, Mapping):
            wrapper_key = next(
                (
                    key
                    for key in ("snapshot", "normalized_snapshot", "payload", "envelope")
                    if isinstance(item.get(key), Mapping)
                ),
                None,
            )
            if wrapper_key is not None:
                result = dict(item)
                result[wrapper_key] = payload
                return result
        return payload

    def _snapshot_identity(self, item: Any) -> dict[str, str]:
        """Return the stable DB identity without reading a source transcript."""
        payload = self._snapshot_payload(item)
        project = _first_attr(item, ("project",), payload.get("project", {}))
        return {
            "project_id": str(
                _first_attr(item, ("project_id",), _first_attr(project, ("id", "project_id"), ""))
            ),
            "source": str(_first_attr(item, ("source",), payload.get("source", "")) or ""),
            "session_id": str(
                _first_attr(item, ("session_id", "snapshot_id", "source_id"), payload.get("session_id", ""))
                or ""
            ),
            "revision": str(_first_attr(item, ("revision",), payload.get("revision", "")) or ""),
        }

    @staticmethod
    def _store_uses_stage_claim(store: Any) -> bool:
        """Require claims for the real Supabase adapter, while keeping small
        injected stores compatible until they explicitly implement the API."""
        if store is None:
            return False
        if callable(getattr(store, "claim_stage", None)):
            return True
        store_type = type(store)
        return (
            store_type.__name__ == "SupabaseStore"
            and str(getattr(store_type, "__module__", "")).endswith("ace_database")
        )

    def _claim_stage(
        self,
        store: Any,
        item: Any,
        *,
        stage: str,
        lease_seconds: int = 1800,
    ) -> dict[str, Any] | None:
        """Claim one stage identity, or return None when it is not owned."""
        self._last_stage_claim_error = None
        claim = getattr(store, "claim_stage", None)
        if not callable(claim):
            self._last_stage_claim_error = "claim_api_missing"
            return None
        identity = self._snapshot_identity(item)
        # ``host_id`` identifies the worker holding the processing lease.
        # Source host metadata remains in the snapshot and is not reused as
        # the lease owner, otherwise two workers on one execution host could
        # be attributed to the collector that produced the source.
        host_id = str(self.host_id or "")
        if not all(identity.values()) or not host_id:
            self._last_stage_claim_error = "claim_identity_incomplete"
            return None
        kwargs = {
            "source": identity["source"],
            "stage": stage,
            "lease_owner": self._lease_owner,
            "lease_seconds": lease_seconds,
        }
        try:
            result = _await_if_needed(
                _call_variants(
                    claim,
                    [
                        (
                            (
                                identity["project_id"],
                                host_id,
                                identity["session_id"],
                                identity["revision"],
                            ),
                            kwargs,
                        ),
                        (
                            (),
                            {
                                "project_id": identity["project_id"],
                                "host_id": host_id,
                                "session_id": identity["session_id"],
                                "revision": identity["revision"],
                                **kwargs,
                            },
                        ),
                    ],
                )
            )
        except Exception as error:
            # Keep only a type-level diagnostic; SQL/connector details can
            # contain credentials or transcript fragments.
            self._last_stage_claim_error = f"claim_rpc_{type(error).__name__}"[:96]
            return None
        if not isinstance(result, Mapping):
            self._last_stage_claim_error = "claim_invalid_response"
            return None
        if result.get("claimed") is not True:
            self._last_stage_claim_error = "claim_busy"
            return None
        return {
            "lease_owner": self._lease_owner,
            "lease_id": result.get("lease_id"),
            "host_id": host_id,
        }

    def _renew_stage_claim(
        self,
        store: Any,
        item: Any,
        *,
        stage: str,
        lease_seconds: int = 1800,
    ) -> dict[str, Any] | None:
        """Renew a claim with the same owner before another long operation."""
        if not self._store_uses_stage_claim(store):
            return {}
        return self._claim_stage(
            store,
            item,
            stage=stage,
            lease_seconds=lease_seconds,
        )

    def _release_stage(
        self,
        store: Any,
        item: Any,
        *,
        stage: str,
        lease_owner: str,
        host_id: str,
        outcome: str = "failed",
    ) -> bool:
        release = getattr(store, "release_stage", None)
        if not callable(release):
            return False
        identity = self._snapshot_identity(item)
        if not all(identity.values()) or not host_id:
            return False
        kwargs = {
            "source": identity["source"],
            "stage": stage,
            "lease_owner": lease_owner,
            "outcome": outcome,
        }
        try:
            result = _await_if_needed(
                _call_variants(
                    release,
                    [
                        (
                            (
                                identity["project_id"],
                                host_id,
                                identity["session_id"],
                                identity["revision"],
                            ),
                            kwargs,
                        ),
                        (
                            (),
                            {
                                "project_id": identity["project_id"],
                                "host_id": host_id,
                                "session_id": identity["session_id"],
                                "revision": identity["revision"],
                                **kwargs,
                            },
                        ),
                    ],
                )
            )
        except Exception:
            return False
        return isinstance(result, Mapping) and result.get("released") is True

    def _mark_stage(
        self,
        store: Any,
        item: Any,
        *,
        stage: str,
        status: str,
        error: str | None = None,
        lease_owner: str | None = None,
        host_id: str | None = None,
    ) -> bool:
        claims_required = self._store_uses_stage_claim(store)
        marker = self._store_method(
            store,
            ("mark_stage", "mark_processed", "mark_extracted", "complete_extraction"),
        )
        if marker is None:
            # Legacy in-memory stores remain compatible; a real store with
            # claims must expose a marker too.
            return not claims_required
        identity = self._snapshot_identity(item)
        if not all(identity.values()):
            return False
        payload = self._snapshot_payload(item)
        marker_host = str(
            host_id
            or _first_attr(
                item,
                ("host_id",),
                _first_attr(payload, ("host_id",), self.host_id),
            )
            or ""
        )
        if claims_required and (not lease_owner or not marker_host):
            return False
        marker_kwargs = (
            {"lease_owner": lease_owner, "host_id": marker_host}
            if claims_required
            else {}
        )
        variants = [
            (
                (
                    identity["source"],
                    identity["session_id"],
                    identity["revision"],
                    identity["project_id"],
                    stage,
                    status,
                    error,
                ),
                marker_kwargs,
            ),
            (
                (),
                {
                    "source": identity["source"],
                    "session_id": identity["session_id"],
                    "revision": identity["revision"],
                    "project_id": identity["project_id"],
                    "stage": stage,
                    "status": status,
                    "error": error,
                    **marker_kwargs,
                },
            ),
        ]
        try:
            result = _await_if_needed(_call_variants(marker, variants))
        except Exception:
            return False
        if result is False:
            return False
        if isinstance(result, Mapping):
            if result.get("ok") is False or result.get("accepted") is False:
                return False
            if str(result.get("status") or "").lower() in {"failed", "error", "rejected"}:
                return False
        return True

    def _extraction_cursor_key(self, project: Any, item: Any) -> str:
        """Return one durable cursor key per authorized project/session."""
        identity = self._snapshot_identity(item)
        project_id = _project_id(project) if project is not None else identity["project_id"]
        parts = (
            project_id,
            identity.get("source", ""),
            identity.get("session_id", ""),
        )
        key = ":".join(part for part in parts if part)
        return key or identity.get("revision", "snapshot")

    @staticmethod
    def _message_ordinal(message: Mapping[str, Any], index: int) -> int:
        """Use the provider ordinal, with a stable list-position fallback."""
        value = message.get("ordinal")
        if isinstance(value, bool):
            return index
        try:
            ordinal = int(value)
        except (TypeError, ValueError):
            return index
        return ordinal if ordinal >= 0 else index

    @staticmethod
    def _message_timestamp(value: Any) -> float | None:
        """Parse a message timestamp without turning malformed input into now."""
        if value in (None, ""):
            return None
        try:
            if isinstance(value, datetime):
                parsed = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
            else:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).timestamp()
        except (TypeError, ValueError, OverflowError, OSError):
            return None

    @classmethod
    def _snapshot_watermark(
        cls, messages: Sequence[Mapping[str, Any]], revision: str = ""
    ) -> dict[str, Any] | None:
        """Build the cursor that represents the end of a normalized snapshot."""
        if not messages:
            return None
        best_index = max(
            range(len(messages)),
            key=lambda index: (cls._message_ordinal(messages[index], index), index),
        )
        last = messages[best_index]
        return {
            "schema_version": EXTRACTION_CURSOR_VERSION,
            "last_ordinal": cls._message_ordinal(last, best_index),
            "last_message_id": str(last.get("id") or ""),
            "last_revision": revision,
        }

    def _snapshot_delta(
        self,
        project: Any,
        item: Any,
        cursor: Mapping[str, Any] | None,
        *,
        minimum_started_at: float | None = None,
    ) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any] | None, bool]:
        """Return only messages after the session cursor.

        ``None`` means the source is a legacy text-only snapshot and the
        existing context path should be used.  ``baseline`` is true when an
        automatic first pass deliberately records an already-running session
        without sending its historical body to the model.
        """
        payload = self._snapshot_payload(item)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            if minimum_started_at is None or cursor:
                return None, None, False
            started = self._message_timestamp(
                _first_attr(
                    item,
                    ("started_at", "captured_at", "timestamp", "created_at"),
                    _first_attr(payload, ("started_at", "captured_at", "timestamp", "created_at")),
                )
            )
            return ([], None, True) if started is None or started < minimum_started_at else (None, None, False)

        messages = [message for message in raw_messages if isinstance(message, Mapping)]
        identity = self._snapshot_identity(item)
        watermark = self._snapshot_watermark(messages, identity.get("revision", ""))
        if not messages:
            if minimum_started_at is not None and not cursor:
                raw_count = _first_attr(item, ("message_count",), _first_attr(payload, ("message_count",), 0))
                try:
                    message_count = int(raw_count or 0)
                except (TypeError, ValueError):
                    message_count = 0
                if message_count > 0:
                    watermark = {
                        "schema_version": EXTRACTION_CURSOR_VERSION,
                        "last_ordinal": message_count,
                        "last_message_id": "",
                        "last_revision": identity.get("revision", ""),
                    }
            return [], watermark, minimum_started_at is not None and not cursor

        has_cursor = isinstance(cursor, Mapping) and "last_ordinal" in cursor
        if has_cursor:
            try:
                last_ordinal = int(cursor.get("last_ordinal", -1))
            except (TypeError, ValueError):
                last_ordinal = -1
            return (
                [
                    message
                    for index, message in enumerate(messages)
                    if self._message_ordinal(message, index) > last_ordinal
                ],
                watermark,
                False,
            )

        # Explicit processing and historical backfill preserve the CMC
        # first-run semantics.  The native 30-minute automation is different:
        # an existing long-running session is baselined instead of replayed.
        if minimum_started_at is None:
            return messages, watermark, False

        started = self._message_timestamp(
            _first_attr(
                item,
                ("started_at", "captured_at", "timestamp", "created_at"),
                _first_attr(payload, ("started_at", "captured_at", "timestamp", "created_at")),
            )
        )
        if started is not None and started >= minimum_started_at:
            return messages, watermark, False

        recent = [
            message
            for message in messages
            if (timestamp := self._message_timestamp(message.get("timestamp"))) is not None
            and timestamp >= minimum_started_at
        ]
        if recent:
            return recent, watermark, False
        return [], watermark, True

    def _snapshot_context(
        self, item: Any, *, messages: Sequence[Mapping[str, Any]] | None = None
    ) -> str:
        payload = self._snapshot_payload(item)
        value = _first_attr(payload, ("context", "text", "transcript", "content", "body"), "")
        source_messages = payload.get("messages") if messages is None else messages
        # An explicit empty delta means "nothing new".  Do not fall back to
        # a legacy full-text field and accidentally replay the snapshot.
        if messages is not None:
            value = ""
        if isinstance(source_messages, list) and source_messages:
            lines: list[str] = []
            for message in source_messages:
                if not isinstance(message, Mapping):
                    continue
                if not self._snapshot_message_is_extractable(message):
                    continue
                role = str(message.get("role") or "system").title()
                content = self._strip_runtime_preamble(message.get("content", ""))
                if isinstance(content, (dict, list)):
                    content = json.dumps(_json_safe(content), ensure_ascii=False)
                metadata: list[str] = []
                for key in ("id", "type", "timestamp", "call_id", "status", "model"):
                    metadata_value = message.get(key)
                    if metadata_value not in (None, "", [], {}):
                        metadata.append(f"{key}={_json_safe(metadata_value)}")
                refs = message.get("refs")
                if refs not in (None, "", [], {}):
                    metadata.append(
                        "refs="
                        + json.dumps(_json_safe(refs), ensure_ascii=False, sort_keys=True)
                    )
                if str(content).strip() or metadata:
                    label = f"**{role}:**"
                    if metadata:
                        label = f"**{role} [{'; '.join(metadata)}]:**"
                    lines.append(f"{label} {content}".rstrip())
            if lines:
                value = "\n\n".join(lines)
        if isinstance(value, (dict, list)):
            return json.dumps(_json_safe(value), ensure_ascii=False)
        return str(value or "")

    @staticmethod
    def _normalise_message_type(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")

    @classmethod
    def _snapshot_message_is_extractable(cls, message: Mapping[str, Any]) -> bool:
        """Keep only visible transcript turns for extraction.

        The normalized DB transcript remains untouched.  This filter is only
        for the bounded extraction context: provider lifecycle/usage events,
        developer/system instructions, and telemetry are not user activity.
        """
        candidates: list[Any] = [
            message.get("type"),
            message.get("event_type"),
            message.get("subtype"),
            message.get("kind"),
        ]
        refs = message.get("refs")
        if isinstance(refs, Mapping):
            candidates.extend(refs.get(key) for key in ("event_type", "subtype", "kind", "type"))
        content = message.get("content")
        if isinstance(content, Mapping):
            candidates.extend(
                content.get(key) for key in ("type", "event_type", "subtype", "kind")
            )
        if any(
            cls._normalise_message_type(candidate) in _NON_ACTIVITY_MESSAGE_TYPES
            for candidate in candidates
            if candidate not in (None, "")
        ):
            return False
        role = cls._normalise_message_type(message.get("role"))
        return role in {"user", "assistant", "tool"}

    @staticmethod
    def _strip_runtime_preamble(value: Any) -> Any:
        """Remove only the exact injected Agent Central prefix.

        A user may legitimately cite AGENTS instructions or an audit in their
        request.  Stripping therefore requires the complete runtime wrapper
        at the beginning: the ``# AGENTS.md instructions`` header, a closed
        ``<INSTRUCTIONS>`` block, and a closed ``<environment_context>``
        block.  Human text after that wrapper is retained verbatim.
        """
        if isinstance(value, list):
            return [ACEPipeline._strip_runtime_preamble(item) for item in value]
        if isinstance(value, Mapping):
            return {
                key: (
                    ACEPipeline._strip_runtime_preamble(item)
                    if str(key).lower() in {"text", "content"}
                    else item
                )
                for key, item in value.items()
            }
        if not isinstance(value, str):
            return value
        header = _RUNTIME_PREAMBLE_HEADER.match(value)
        if header is None:
            return value
        cursor = header.end()
        instructions = _RUNTIME_INSTRUCTIONS_BLOCK.match(value, cursor)
        if instructions is None:
            return value
        cursor = instructions.end()
        environment = _RUNTIME_ENVIRONMENT_BLOCK.match(value, cursor)
        if environment is None:
            return value
        return value[environment.end() :].lstrip()

    def _bounded_chunks(self, context: str, maximum: int) -> list[str]:
        if len(context) <= maximum:
            return [context]
        boundaries = re.split(r"(?=\n\*\*(?:User|Assistant|\[Subagent)|\n#{1,3} )", context)
        chunks: list[str] = []
        current = ""
        for part in boundaries:
            if len(part) > maximum:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(part[index : index + maximum] for index in range(0, len(part), maximum))
                continue
            if current and len(current) + len(part) > maximum:
                chunks.append(current)
                current = part
            else:
                current += part
        if current:
            chunks.append(current)
        if "".join(chunks) != context:
            raise PipelineError("context chunking lost source bytes")
        return chunks

    def _default_extractor(self, context: str) -> Any:
        previous = os.environ.get("CLAUDE_INVOKED_BY")
        previous_backfill = os.environ.get("CODEX_ACE_BACKFILL_ENABLED")
        os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"
        os.environ["CODEX_ACE_BACKFILL_ENABLED"] = "0"
        try:
            flush = _import_optional("flush")
            if flush is None or not hasattr(flush, "run_flush"):
                raise PipelineError("ACE extractor is unavailable")
            return _await_if_needed(flush.run_flush(context))
        finally:
            if previous is None:
                os.environ.pop("CLAUDE_INVOKED_BY", None)
            else:
                os.environ["CLAUDE_INVOKED_BY"] = previous
            if previous_backfill is None:
                os.environ.pop("CODEX_ACE_BACKFILL_ENABLED", None)
            else:
                os.environ["CODEX_ACE_BACKFILL_ENABLED"] = previous_backfill

    def _extraction_result(self, value: Any) -> str:
        if isinstance(value, tuple):
            return str(value[0] or "")
        if isinstance(value, Mapping):
            return str(value.get("text", value.get("result", value.get("extraction", ""))) or "")
        return str(value or "")

    @staticmethod
    def _run_diagnostics_from_result(value: Any) -> Any | None:
        """Return the measured diagnostics carried by a runner result.

        ``flush.run_flush`` and the Codex runner use the tuple contract
        ``(response, RunDiagnostics)``.  Keep this adapter deliberately
        narrow so a response mapping cannot accidentally be interpreted as
        usage metadata.
        """
        if isinstance(value, tuple) and len(value) == 2:
            diagnostics = value[1]
            if callable(getattr(diagnostics, "as_metrics", None)):
                return diagnostics
        return None

    @staticmethod
    def _diagnostics_metrics(diagnostics: Any | None) -> dict[str, Any] | None:
        """Return an allowlisted, JSON-safe diagnostics snapshot.

        Missing diagnostics stay missing.  In particular, this method never
        turns an unavailable usage report into fabricated numeric zeroes.
        """
        if diagnostics is None:
            return None
        method = getattr(diagnostics, "as_metrics", None)
        if not callable(method):
            return None
        try:
            raw = method()
        except Exception:
            return {"usage_status": "unavailable"}
        if not isinstance(raw, Mapping):
            return {"usage_status": "unavailable"}
        result: dict[str, Any] = {}
        for key in ("call_count", "duration_seconds"):
            value = raw.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = value
        status = str(raw.get("usage_status") or "").strip().lower()
        if status in {"available", "partial", "unavailable"}:
            result["usage_status"] = status
        token_usage = raw.get("token_usage")
        if token_usage is None:
            if "token_usage" in raw:
                result["token_usage"] = None
        elif isinstance(token_usage, Mapping):
            bounded: dict[str, int] = {}
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                value = token_usage.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    bounded[key] = value
            if bounded:
                result["token_usage"] = bounded
        if not result:
            return {"usage_status": "unavailable"}
        return result

    @staticmethod
    def _merge_run_diagnostics(current: Any | None, incoming: Any | None) -> Any | None:
        """Merge measured runner calls while retaining the runner object."""
        if incoming is None:
            return current
        if current is None:
            return incoming
        merge = getattr(current, "merge", None)
        if callable(merge):
            try:
                merge(incoming)
            except Exception:
                # A malformed injected diagnostics object must not make a
                # successful extraction look measured.  Keep the first
                # object, whose own metrics remain valid.
                return current
        return current

    def _extraction_state_record(
        self,
        previous: Any,
        item: Any,
        project: Any,
        *,
        status: str,
        diagnostics: Any | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Build one exact snapshot/session record without dropping proof."""
        record = dict(previous) if isinstance(previous, Mapping) else {}
        identity = self._snapshot_identity(item)
        project_id = identity.get("project_id") or _project_id(project)
        if project_id:
            record["project_id"] = project_id
        for field in ("source", "session_id", "revision"):
            value = identity.get(field)
            if value:
                record[field] = value
        source_day = self._snapshot_source_day(item, project)
        if source_day is not None:
            record["source_day"] = source_day
        record.update(fields)
        record["status"] = status
        if diagnostics is not None:
            record.setdefault("stage_metrics", {})["extraction"] = (
                self._diagnostics_metrics(diagnostics) or {"usage_status": "unavailable"}
            )
        return record

    def _snapshot_source_day(self, item: Any, project: Any) -> str | None:
        """Resolve a snapshot's source workday in the project timezone."""
        payload = self._snapshot_payload(item)
        metadata_timestamps: list[float] = []
        for value in (item, payload):
            for field in (
                "started_at",
                "source_timestamp",
                "captured_at",
                "timestamp",
                "created_at",
                "received_at",
                "updated_at",
            ):
                parsed = self._message_timestamp(_first_attr(value, (field,)))
                if parsed is not None:
                    metadata_timestamps.append(parsed)
                    break
        timestamps = metadata_timestamps
        messages = payload.get("messages")
        if not timestamps and isinstance(messages, list):
            for message in messages:
                if isinstance(message, Mapping):
                    parsed = self._message_timestamp(message.get("timestamp"))
                    if parsed is not None:
                        timestamps.append(parsed)
        if not timestamps:
            return None
        tz = self._project_timezone(project)
        return datetime.fromtimestamp(min(timestamps), tz=timezone.utc).astimezone(tz).date().isoformat()

    @staticmethod
    def _aggregate_stage_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Aggregate measured stage metrics without inventing absent values."""
        if not values:
            return None
        result: dict[str, Any] = {}
        for key in ("call_count", "duration_seconds"):
            numeric = [value[key] for value in values if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)]
            if numeric:
                result[key] = sum(numeric)
        token_maps = [value.get("token_usage") for value in values if isinstance(value.get("token_usage"), Mapping)]
        if token_maps:
            token_usage: dict[str, int] = {}
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                numbers = [item.get(key) for item in token_maps if isinstance(item.get(key), int) and not isinstance(item.get(key), bool)]
                if numbers:
                    token_usage[key] = sum(numbers)
            if token_usage:
                result["token_usage"] = token_usage
        statuses = [str(value.get("usage_status") or "").lower() for value in values]
        statuses = [status for status in statuses if status in {"available", "partial", "unavailable"}]
        if statuses:
            if all(status == "available" for status in statuses):
                result["usage_status"] = "available"
            elif any(status in {"available", "partial"} for status in statuses):
                result["usage_status"] = "partial"
            else:
                result["usage_status"] = "unavailable"
        return result or {"usage_status": "unavailable"}

    def _compile_stage_metrics(self, project: Any, day: str) -> dict[str, Any] | None:
        """Read the compiler's measured proof for one exact daily source."""
        path = self._project_vault_dir(project) / ".state" / "state.json"
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(state, Mapping):
            return None
        ingested = state.get("ingested")
        if not isinstance(ingested, Mapping):
            return None
        # The compiler ledger historically keyed this map by the basename;
        # newer writers may retain the source-relative ``daily/`` key.  Both
        # identify the exact requested daily source.
        record = ingested.get(f"daily/{day}.md")
        if not isinstance(record, Mapping):
            record = ingested.get(f"{day}.md")
        if not isinstance(record, Mapping):
            return None
        metrics = record.get("stage_metrics")
        if not isinstance(metrics, Mapping):
            return None
        compile_metrics = metrics.get("compile")
        return dict(compile_metrics) if isinstance(compile_metrics, Mapping) else None

    def _stage_usage_for_day(self, project: Any, day: str) -> dict[str, Any]:
        """Expose measured extraction and compile usage for daily reports."""
        result: dict[str, Any] = {}
        extraction_state = self.state.read("extraction")
        records = extraction_state.get("snapshots")
        metrics: list[Mapping[str, Any]] = []
        if isinstance(records, Mapping):
            project_id = _project_id(project)
            for record in records.values():
                if not isinstance(record, Mapping):
                    continue
                if project_id and str(record.get("project_id") or "") not in {"", project_id}:
                    continue
                if str(record.get("source_day") or "") != day:
                    continue
                stages = record.get("stage_metrics")
                extraction = stages.get("extraction") if isinstance(stages, Mapping) else None
                if isinstance(extraction, Mapping):
                    metrics.append(extraction)
        extraction_metrics = self._aggregate_stage_metrics(metrics)
        if extraction_metrics is not None:
            result["extraction"] = extraction_metrics
        compile_metrics = self._compile_stage_metrics(project, day)
        if compile_metrics is not None:
            result["compile"] = compile_metrics
        return result

    def _attach_stage_usage(self, payload: Mapping[str, Any], project: Any, day: str) -> dict[str, Any]:
        """Attach the stable stage metrics interface consumed by reports."""
        current = dict(payload)
        usage = self._stage_usage_for_day(project, day)
        if usage:
            for field in ("stage_usage", "stage_metrics", "stages"):
                existing = current.get(field)
                merged = dict(existing) if isinstance(existing, Mapping) else {}
                merged.update(usage)
                current[field] = merged
        return current

    def _daily_write(
        self,
        project: Any,
        item: Any,
        extracted: str,
        *,
        context: str | None = None,
    ) -> Path:
        project_dir = self._project_vault_dir(project)
        snapshot = self._snapshot_payload(item)
        source = str(_first_attr(item, ("source", "provider"), _first_attr(snapshot, ("source", "provider"), ""))).lower()
        # ``snapshot_id`` is a revision/transport identity in the DB and may
        # change when one session receives new turns.  Daily markers must use
        # the stable provider session id so a revision update replaces the
        # existing entry instead of appending a duplicate.
        session_id = str(
            _first_attr(
                item,
                ("session_id", "source_id"),
                _first_attr(snapshot, ("session_id", "source_id"), ""),
            )
        )
        if not session_id:
            session_id = str(_first_attr(item, ("snapshot_id",), _first_attr(snapshot, ("snapshot_id", "id"), "")))
        if not session_id:
            session_id = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()[:32]
        timestamp = _normalise_datetime(
            _first_attr(
                item,
                ("started_at", "captured_at", "timestamp", "created_at"),
                _first_attr(snapshot, ("started_at", "captured_at", "timestamp", "created_at")),
            )
        )
        session = SimpleNamespace(
            path=Path(str(_first_attr(snapshot, ("path", "source_path"), "snapshot.json"))),
            session_id=session_id,
            timestamp=timestamp,
            cwd=str(_first_attr(snapshot, ("cwd", "project_root"), project_dir)),
            context=context if context is not None else self._snapshot_context(item),
            turn_count=0,
        )
        if source == "codex":
            module = _import_optional("backfill_codex")
            writer = getattr(module, "upsert_daily_entry", None) if module else None
            if writer is not None:
                return Path(_await_if_needed(writer(project_dir, session, _redact(extracted))))
        if source == "claude":
            module = _import_optional("ace_collect")
            writer = getattr(module, "store_claude", None) if module else None
            if writer is not None:
                return Path(_await_if_needed(writer(project_dir, session, _redact(extracted), {})))
        return self._upsert_ace_daily(project_dir, session_id, timestamp, extracted)

    def _upsert_ace_daily(self, project_dir: Path, session_id: str, timestamp: datetime, extracted: str) -> Path:
        daily_dir = project_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily = daily_dir / f"{timestamp.astimezone(PARIS).strftime('%Y-%m-%d')}.md"
        opening = f"<!-- ace-snapshot: {session_id} -->"
        closing = "<!-- /ace-snapshot -->"
        body = f"{opening}\n### ACE Snapshot {session_id[:8]} ({timestamp.astimezone(PARIS).strftime('%H:%M')})\n\n{_redact(extracted).strip()}\n{closing}\n"
        lock_path = daily.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            existing = daily.read_text(encoding="utf-8") if daily.exists() else f"# Daily Log: {daily.stem}\n\n"
            pattern = re.compile(re.escape(opening) + r".*?" + re.escape(closing) + r"\n*", re.S)
            updated = pattern.sub(lambda _: body, existing, count=1) if pattern.search(existing) else existing.rstrip() + "\n\n" + body
            fd, temporary = tempfile.mkstemp(prefix=f".{daily.name}.", dir=str(daily.parent))
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(updated)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, daily)
            finally:
                Path(temporary).unlink(missing_ok=True)
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return daily

    def process(
        self,
        *,
        project: Any | None = None,
        cwd: str | Path | None = None,
        limit: int = DEFAULT_LIMIT,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        minimum_started_at: float | None = None,
        _lock_held: bool = False,
    ) -> dict[str, Any]:
        # Manual ``process`` invocations share the processor lock with the
        # native tick.  ``tick`` already owns it and opts into the private
        # flag below so the same process does not attempt a nested flock.
        if not _lock_held:
            with advisory_lock(self.private_root / "tick.lock"):
                return self.process(
                    project=project,
                    cwd=cwd,
                    limit=limit,
                    max_context_chars=max_context_chars,
                    minimum_started_at=minimum_started_at,
                    _lock_held=True,
                )
        if project is None and cwd is not None:
            project = self._resolve_project(cwd)
        elif project is None:
            project = self._resolve_project(None)
        counts: dict[str, Any] = {"candidates": 0, "processed": 0, "empty": 0, "baseline": 0, "failed": 0, "pending": 0, "offline": False}
        local_counts: dict[str, Any] | None = None
        if self._extraction_mode() == "local":
            # Memory first: the daily log never waits for the remote store.
            local_counts = self.process_local(
                project=project,
                limit=limit,
                max_context_chars=max_context_chars,
                minimum_started_at=minimum_started_at,
                _lock_held=True,
            )
            for key in ("candidates", "processed", "empty", "baseline", "failed", "pending"):
                counts[key] += int(local_counts.get(key, 0) or 0)
            counts["signals"] = int(local_counts.get("signals", 0) or 0)
            counts["local"] = local_counts
        store = self._get_store()
        if store is None:
            counts["offline"] = True
            return counts
        extraction_state = self.state.read("extraction")
        records = extraction_state.setdefault("snapshots", {})
        cursors = extraction_state.setdefault("cursors", {})
        snapshots = self._db_snapshots(
            store,
            project,
            max(0, limit),
            minimum_started_at=minimum_started_at,
            extraction_cursors=cursors,
        )
        counts["candidates"] += len(snapshots)
        extractor = self.extractor or self._default_extractor
        for item in snapshots:
            identity = self._snapshot_identity(item)
            snapshot_id = str(
                _first_attr(
                    item,
                    ("snapshot_id", "id", "source_id"),
                    identity["session_id"] or identity["revision"],
                )
            )
            state_key = ":".join(
                part
                for part in (
                    _project_id(project) if project is not None else identity["project_id"],
                    identity["source"],
                    identity["session_id"],
                    identity["revision"],
                )
                if part
            ) or snapshot_id
            prior = records.get(state_key, {})
            if snapshot_id and prior.get("status") in {"processed", "empty", "baseline"}:
                # Already extracted from the local envelope.  Close the remote
                # stage so the database stops listing it; a failure here is
                # harmless and retried on a later pass.
                if local_counts is not None:
                    self._reconcile_local_extraction(store, item)
                continue
            if local_counts is not None and prior.get("path") == "local":
                # The local path owns this snapshot and will retry it on the
                # next pass; the remote path only extracts snapshots that have
                # no local envelope, for example those captured by another host.
                continue
            claims_required = self._store_uses_stage_claim(store)
            claim_info = (
                self._claim_stage(store, item, stage="extraction")
                if claims_required
                else None
            )
            if claims_required and claim_info is None:
                records[state_key] = self._extraction_state_record(
                    prior,
                    item,
                    project,
                    status="pending",
                    error_type="StageClaimUnavailable",
                    claim_reason=self._last_stage_claim_error or "claim_unavailable",
                )
                counts["failed"] += 1
                counts["pending"] += 1
                self.state.write("extraction", extraction_state)
                continue
            lease_owner = claim_info.get("lease_owner") if claim_info else None
            lease_host = claim_info.get("host_id") if claim_info else None

            def renew_extraction_claim() -> None:
                if claims_required and self._renew_stage_claim(
                    store, item, stage="extraction"
                ) is None:
                    raise PipelineError("stage lease lost")

            def release_extraction_claim() -> None:
                if claims_required and lease_owner and lease_host:
                    self._release_stage(
                        store,
                        item,
                        stage="extraction",
                        lease_owner=lease_owner,
                        host_id=lease_host,
                    )

            cursor_key = self._extraction_cursor_key(project, item)
            cursor = cursors.get(cursor_key, {})
            delta_messages, next_cursor, baselined = self._snapshot_delta(
                project,
                item,
                cursor,
                minimum_started_at=minimum_started_at,
            )
            context = (
                self._snapshot_context(item, messages=delta_messages)
                if delta_messages is not None
                else self._snapshot_context(item)
            )
            if baselined:
                if claims_required and self._renew_stage_claim(
                    store, item, stage="extraction"
                ) is None:
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="pending", error_type="StageLeaseLost"
                    )
                    self._release_stage(
                        store,
                        item,
                        stage="extraction",
                        lease_owner=lease_owner,
                        host_id=lease_host,
                    )
                    counts["failed"] += 1
                    counts["pending"] += 1
                elif self._mark_stage(
                    store,
                    item,
                    stage="extraction",
                    status="succeeded",
                    lease_owner=lease_owner,
                    host_id=lease_host,
                ):
                    if next_cursor is not None:
                        cursors[cursor_key] = next_cursor
                    records[state_key] = self._extraction_state_record(
                        prior,
                        item,
                        project,
                        status="baseline",
                        cursor_key=cursor_key,
                        processed_at=self.now().isoformat(),
                    )
                    counts["baseline"] += 1
                else:
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="pending", error_type="PipelineError"
                    )
                    if claims_required:
                        self._release_stage(
                            store,
                            item,
                            stage="extraction",
                            lease_owner=lease_owner,
                            host_id=lease_host,
                        )
                    counts["failed"] += 1
                    counts["pending"] += 1
                self.state.write("extraction", extraction_state)
                continue
            if not context.strip():
                if claims_required and self._renew_stage_claim(
                    store, item, stage="extraction"
                ) is None:
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="pending", error_type="StageLeaseLost"
                    )
                    self._release_stage(
                        store,
                        item,
                        stage="extraction",
                        lease_owner=lease_owner,
                        host_id=lease_host,
                    )
                    counts["failed"] += 1
                    counts["pending"] += 1
                elif self._mark_stage(
                    store,
                    item,
                    stage="extraction",
                    status="succeeded",
                    lease_owner=lease_owner,
                    host_id=lease_host,
                ):
                    if next_cursor is not None:
                        cursors[cursor_key] = next_cursor
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="empty"
                    )
                    counts["empty"] += 1
                else:
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="pending", error_type="PipelineError"
                    )
                    if claims_required:
                        self._release_stage(
                            store,
                            item,
                            stage="extraction",
                            lease_owner=lease_owner,
                            host_id=lease_host,
                        )
                    counts["failed"] += 1
                    counts["pending"] += 1
                continue
            acknowledgement_failed = False
            extraction_diagnostics: Any | None = None
            try:
                outputs: list[str] = []
                chunks = self._bounded_chunks(context, max(1, max_context_chars))
                for chunk in chunks:
                    # Flush may retry a large chunk. Renew immediately before
                    # every call so a later chunk cannot complete an expired
                    # lease owned by another worker.
                    renew_extraction_claim()
                    result = _await_if_needed(extractor(chunk))
                    extraction_diagnostics = self._merge_run_diagnostics(
                        extraction_diagnostics,
                        self._run_diagnostics_from_result(result),
                    )
                    text = self._extraction_result(result).strip()
                    if text.startswith("FLUSH_ERROR"):
                        raise PipelineError("extraction failed")
                    if text and text != "FLUSH_OK":
                        outputs.append(text)
                if not outputs:
                    renew_extraction_claim()
                    if not self._mark_stage(
                        store,
                        item,
                        stage="extraction",
                        status="succeeded",
                        lease_owner=lease_owner,
                        host_id=lease_host,
                    ):
                        acknowledgement_failed = True
                        raise PipelineError("database acknowledgement failed")
                    records[state_key] = self._extraction_state_record(
                        prior,
                        item,
                        project,
                        status="empty",
                        diagnostics=extraction_diagnostics,
                    )
                    counts["empty"] += 1
                    continue
                extracted, signals = split_extraction_signals("\n\n".join(outputs))
                renew_extraction_claim()
                daily = self._daily_write(project, item, extracted, context=context)
                counts["signals"] = counts.get("signals", 0) + self._record_signals(
                    project, item, signals, daily.stem
                )
                renew_extraction_claim()
                if not self._mark_stage(
                    store,
                    item,
                    stage="extraction",
                    status="succeeded",
                    lease_owner=lease_owner,
                    host_id=lease_host,
                ):
                    acknowledgement_failed = True
                    raise PipelineError("database acknowledgement failed")
                if next_cursor is not None:
                    cursors[cursor_key] = next_cursor
                records[state_key] = self._extraction_state_record(
                    prior,
                    item,
                    project,
                    status="processed",
                    diagnostics=extraction_diagnostics,
                    daily_file=daily.name,
                    processed_at=self.now().isoformat(),
                )
                counts["processed"] += 1
            except Exception as error:
                extraction_diagnostics = self._merge_run_diagnostics(
                    extraction_diagnostics,
                    getattr(error, "diagnostics", None),
                )
                records[state_key] = self._extraction_state_record(
                    prior,
                    item,
                    project,
                    status="pending",
                    diagnostics=extraction_diagnostics,
                    error_type=type(error).__name__,
                )
                if not acknowledgement_failed:
                    self._mark_stage(
                        store,
                        item,
                        stage="extraction",
                        status="failed",
                        error=type(error).__name__,
                        lease_owner=lease_owner,
                        host_id=lease_host,
                    )
                release_extraction_claim()
                counts["failed"] += 1
                counts["pending"] += 1
            self.state.write("extraction", extraction_state)
        self.state.write("extraction", extraction_state)
        return counts

    def _record_signals(
        self, project: Any, item: Any, signals: Sequence[Mapping[str, Any]], day: str
    ) -> int:
        """Append improvement signals observed during extraction, append-only.

        The extractor is the only stage holding the raw transcript, so it is
        the only one that can quote the user verbatim.  Storing the signals
        here lets the morning analysis skip conversations with no signal at
        all, and lets the report show the exact wording the daily log
        deliberately neutralises.
        """
        if not signals:
            return 0
        identity = self._snapshot_identity(item)
        project_id = _project_id(project) if project is not None else identity["project_id"]
        directory = self.private_root / "signals" / (project_id or "unknown")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / f"{day}.jsonl"
        recorded = 0
        try:
            with path.open("a", encoding="utf-8") as handle:
                for signal in signals:
                    row = {
                        "recorded_at": self.now().isoformat(),
                        "project_id": project_id,
                        "source": identity.get("source", ""),
                        "session_id": identity.get("session_id", ""),
                        "revision": identity.get("revision", ""),
                        **{key: signal.get(key) for key in ("type", "signature", "message_ids", "quote", "observed")},
                    }
                    handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
                    recorded += 1
            with contextlib.suppress(OSError):
                path.chmod(0o600)
        except OSError:
            return 0
        return recorded

    @staticmethod
    def _accepts_keyword(function: Any, name: str) -> bool:
        """Whether ``function`` accepts ``name``, directly or through kwargs."""
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            return False
        if name in parameters:
            return True
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _capture_signals_for(self, project: Any, day: str) -> dict[str, list[dict[str, Any]]]:
        """Load the signals the extractor observed for one project and day.

        Keyed by session id so the morning analysis starts from what was seen
        on the raw transcript instead of re-deriving it from the evidence
        windows alone.
        """
        project_id = _project_id(project) if project is not None else ""
        directory = self.private_root / "signals" / (project_id or "unknown")
        if not directory.is_dir():
            return {}
        # A signal file is named after the conversation's own day, which is
        # often earlier than the extraction day. Read every file and key on the
        # session: the analysis must receive what was seen for the exact
        # conversation it examines, whatever date the file carries.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(directory.glob("*.jsonl")):
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, Mapping) or not row.get("type"):
                    continue
                session_id = str(row.get("session_id") or "")
                if not session_id:
                    continue
                entry = {
                    key: row.get(key)
                    for key in ("type", "signature", "message_ids", "quote", "observed")
                    if row.get(key)
                }
                bucket = grouped.setdefault(session_id, [])
                if entry not in bucket:
                    bucket.append(entry)
        return grouped

    @staticmethod
    def _extraction_mode() -> str:
        value = os.environ.get("ACE_EXTRACTION_MODE", DEFAULT_EXTRACTION_MODE).strip().lower()
        return value if value in {"local", "database"} else DEFAULT_EXTRACTION_MODE

    def _reconcile_local_extraction(self, store: Any, item: Any) -> None:
        """Best-effort remote stage close for a locally extracted snapshot."""
        try:
            lease_owner = None
            lease_host = None
            if self._store_uses_stage_claim(store):
                claim_info = self._claim_stage(store, item, stage="extraction")
                if claim_info is None:
                    return
                lease_owner = claim_info.get("lease_owner")
                lease_host = claim_info.get("host_id")
            self._mark_stage(
                store,
                item,
                stage="extraction",
                status="succeeded",
                lease_owner=lease_owner,
                host_id=lease_host,
            )
        except Exception:
            return

    def _local_snapshots(self, outbox: Any, project_id: str | None, limit: int) -> list[Any]:
        """Read local envelopes for one project without claiming them."""
        if outbox is None or limit <= 0:
            return []
        reader = getattr(outbox, "snapshots", None)
        if callable(reader):
            try:
                return list(reader(project_id=project_id, limit=limit))
            except TypeError:
                return list(reader(project_id, limit))
        items = getattr(outbox, "items", None)
        if not isinstance(items, list):
            return []
        result: list[Any] = []
        for entry in items:
            envelope = _as_mapping(_first_attr(entry, ("envelope", "payload"), entry))
            item_project = str(
                envelope.get("project_id")
                or _first_attr(envelope.get("project"), ("id", "project_id"), "")
            )
            if project_id is not None and item_project != project_id:
                continue
            result.append(entry)
            if len(result) >= limit:
                break
        return result

    def process_local(
        self,
        *,
        project: Any | None = None,
        cwd: str | Path | None = None,
        limit: int = DEFAULT_LIMIT,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        minimum_started_at: float | None = None,
        _lock_held: bool = False,
    ) -> dict[str, Any]:
        """Write daily logs from the local sanitized envelopes.

        This is the pre-ACE capture path: the filtered conversation slice is
        extracted and appended to ``daily/`` without waiting for the central
        database.  The same session cursor, extractor, chunking and daily
        writer as the database path are used, so a revision extracted here is
        never replayed by the remote path.
        """
        if not _lock_held:
            with advisory_lock(self.private_root / "tick.lock"):
                return self.process_local(
                    project=project,
                    cwd=cwd,
                    limit=limit,
                    max_context_chars=max_context_chars,
                    minimum_started_at=minimum_started_at,
                    _lock_held=True,
                )
        if project is None and cwd is not None:
            project = self._resolve_project(cwd)
        elif project is None:
            project = self._resolve_project(None)
        counts: dict[str, Any] = {"candidates": 0, "processed": 0, "empty": 0, "baseline": 0, "failed": 0, "pending": 0, "offline": False}
        project_id = _project_id(project) if project is not None else None
        extraction_state = self.state.read("extraction")
        records = extraction_state.setdefault("snapshots", {})
        cursors = extraction_state.setdefault("cursors", {})
        extractor = self.extractor or self._default_extractor
        budget = max(0, limit)
        for entry in self._local_snapshots(self._get_outbox(), project_id, 10_000):
            if budget <= 0:
                break
            item = _as_mapping(_first_attr(entry, ("envelope", "payload"), entry))
            identity = self._snapshot_identity(item)
            if not identity["revision"]:
                continue
            state_key = ":".join(
                part
                for part in (project_id or identity["project_id"], identity["source"], identity["session_id"], identity["revision"])
                if part
            )
            prior = records.get(state_key, {})
            if prior.get("status") in {"processed", "empty", "baseline"}:
                continue
            cursor_key = self._extraction_cursor_key(project, item)
            cursor = cursors.get(cursor_key, {})
            if isinstance(cursor, Mapping) and cursor.get("last_revision") == identity["revision"]:
                continue
            counts["candidates"] += 1
            budget -= 1
            delta_messages, next_cursor, baselined = self._snapshot_delta(
                project, item, cursor, minimum_started_at=minimum_started_at
            )
            context = (
                self._snapshot_context(item, messages=delta_messages)
                if delta_messages is not None
                else self._snapshot_context(item)
            )
            if baselined:
                if next_cursor is not None:
                    cursors[cursor_key] = next_cursor
                records[state_key] = self._extraction_state_record(
                    prior, item, project, status="baseline", cursor_key=cursor_key, processed_at=self.now().isoformat(), path="local"
                )
                counts["baseline"] += 1
                self.state.write("extraction", extraction_state)
                continue
            if not context.strip():
                if next_cursor is not None:
                    cursors[cursor_key] = next_cursor
                records[state_key] = self._extraction_state_record(prior, item, project, status="empty", path="local")
                counts["empty"] += 1
                self.state.write("extraction", extraction_state)
                continue
            diagnostics: Any | None = None
            try:
                outputs: list[str] = []
                for chunk in self._bounded_chunks(context, max(1, max_context_chars)):
                    result = _await_if_needed(extractor(chunk))
                    diagnostics = self._merge_run_diagnostics(diagnostics, self._run_diagnostics_from_result(result))
                    text = self._extraction_result(result).strip()
                    if text.startswith("FLUSH_ERROR"):
                        raise PipelineError("extraction failed")
                    if text and text != "FLUSH_OK":
                        outputs.append(text)
                if not outputs:
                    if next_cursor is not None:
                        cursors[cursor_key] = next_cursor
                    records[state_key] = self._extraction_state_record(
                        prior, item, project, status="empty", diagnostics=diagnostics, path="local"
                    )
                    counts["empty"] += 1
                else:
                    body, signals = split_extraction_signals("\n\n".join(outputs))
                    daily = self._daily_write(project, item, body, context=context)
                    recorded = self._record_signals(project, item, signals, daily.stem)
                    counts["signals"] = counts.get("signals", 0) + recorded
                    if next_cursor is not None:
                        cursors[cursor_key] = next_cursor
                    records[state_key] = self._extraction_state_record(
                        prior,
                        item,
                        project,
                        status="processed",
                        diagnostics=diagnostics,
                        daily_file=daily.name,
                        processed_at=self.now().isoformat(),
                        path="local",
                        signals=recorded,
                    )
                    counts["processed"] += 1
            except Exception as error:
                diagnostics = self._merge_run_diagnostics(diagnostics, getattr(error, "diagnostics", None))
                records[state_key] = self._extraction_state_record(
                    prior, item, project, status="pending", diagnostics=diagnostics, error_type=type(error).__name__, path="local"
                )
                counts["failed"] += 1
                counts["pending"] += 1
            self.state.write("extraction", extraction_state)
        self.state.write("extraction", extraction_state)
        return counts

    def _learning_call(self, names: Sequence[str], project: Any, day: str) -> Any:
        method = self._store_method(self.learning, names) if self.learning is not None else None
        if method is None:
            if names and names[0] == "compile_daily":
                return self._default_compile_daily(project, day)
            if names and names[0] == "analyze_daily":
                return self._default_analyze_daily(project, day)
            return _MISSING
        project_dir = self._project_vault_dir(project)
        return _await_if_needed(
            _call_variants(
                method,
                [
                    ((project, day), {}),
                    ((project_dir, day), {}),
                    ((), {"project": project, "day": day}),
                    ((), {"project_dir": project_dir, "day": day}),
                    ((project_dir,), {}),
                ],
            )
        )

    def _default_compile_daily(self, project: Any, day: str) -> bool:
        """Run the existing Luna compiler for one changed ACE daily log.

        The pipeline owns a project-scoped content-hash ledger because a
        historical or late daily file can arrive after the date cursor has
        moved on.  A matching completed hash is a safe no-op; a new/changed
        file is delegated to the existing compiler with the unchanged Luna
        contract.
        """
        project_dir = self._project_vault_dir(project)
        daily = project_dir / "daily" / f"{day}.md"
        if not daily.exists():
            # The target day may legitimately have no local daily source yet;
            # daily() still runs its deterministic no-evidence audit.  There
            # is nothing to compile or publish in this branch.
            return True
        if self._legacy_compile_proof_valid(project, daily):
            # CMC already compiled this unchanged source and left a matching
            # article reference.  Treat that proof as a safe no-op; do not
            # invoke the LLM compiler merely because ACE has no local ledger.
            return True
        digest = self._daily_file_hash(daily)
        if digest is not None:
            compile_state = self.state.read("compile")
            project_state = self._project_stage_state(compile_state, project)
            record = project_state.get("files", {}).get(f"daily/{daily.name}", {})
            if isinstance(record, Mapping) and record.get("hash") == digest and record.get("status") == "complete":
                self._publish_knowledge(project)
                return True
        script = SCRIPT_DIR / "compile.py"
        if not script.exists():
            self._record_compile_diagnostic(
                project,
                day,
                returncode=None,
                error_type="FileNotFoundError",
                diagnostic="compiler script unavailable",
            )
            return False
        environment = dict(os.environ)
        environment.update(
            {
                "ACE_PROJECT_DIR": str(project_dir),
                "CLAUDE_PROJECT_DIR": str(_first_attr(project, ("root",), project_dir)),
                "CLAUDE_INVOKED_BY": "ace_daily_compile",
                "CODEX_ACE_BACKFILL_ENABLED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        try:
            completed = subprocess.run(
                [sys.executable, str(script), "--file", str(daily)],
                cwd=str(CONFIG_ROOT),
                env=environment,
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._record_compile_diagnostic(
                project,
                day,
                returncode=None,
                stdout=getattr(error, "stdout", ""),
                stderr=getattr(error, "stderr", ""),
                error_type="TimeoutExpired",
            )
            return False
        except OSError as error:
            self._record_compile_diagnostic(
                project,
                day,
                returncode=None,
                stderr=str(error),
                error_type=type(error).__name__,
            )
            return False
        if completed.returncode != 0:
            self._record_compile_diagnostic(
                project,
                day,
                returncode=completed.returncode,
                stdout=getattr(completed, "stdout", ""),
                stderr=getattr(completed, "stderr", ""),
            )
            return False
        self._publish_knowledge(project)
        return True

    def _record_compile_diagnostic(
        self,
        project: Any,
        day: str,
        *,
        returncode: Any = None,
        stdout: Any = "",
        stderr: Any = "",
        error_type: str | None = None,
        diagnostic: str | None = None,
    ) -> dict[str, Any]:
        """Persist a short, redacted compiler failure for the next retry.

        Compiler output can contain the prompt or transcript body.  Keep only
        whitelisted diagnostic categories and identifiers, never the raw
        stdout/stderr, so a retry has an actionable reason without turning
        stage state into a content dump.
        """
        if diagnostic is None:
            diagnostic = self._safe_compile_diagnostic(stdout, stderr, returncode)
        record: dict[str, Any] = {
            "status": "failed",
            "diagnostic": _redact(diagnostic)[:240],
            "recorded_at": self.now().isoformat(),
        }
        if returncode is not None:
            with contextlib.suppress(TypeError, ValueError):
                record["returncode"] = int(returncode)
        if error_type:
            record["error_type"] = str(error_type)[:80]
        self._compile_diagnostic_cache[(_project_id(project), str(day))] = record

        compile_state = self.state.read("compile")
        project_state = self._project_stage_state(compile_state, project)
        project_state.setdefault("diagnostics", {})[str(day)] = record
        self._record_stage_alias(compile_state, project, project_state)
        self.state.write("compile", compile_state)
        return record

    @staticmethod
    def _safe_compile_diagnostic(stdout: Any, stderr: Any, returncode: Any) -> str:
        """Reduce compiler streams to safe, useful diagnostic labels."""
        candidates: list[str] = []
        for stream in (stderr, stdout):
            if stream is None:
                continue
            text = stream.decode("utf-8", errors="replace") if isinstance(stream, bytes) else str(stream)
            candidates.extend(text.splitlines())

        details: list[str] = []
        for raw_line in candidates:
            line = _redact(" ".join(raw_line.split()))
            if not line:
                continue
            structural = _COMPILE_STRUCTURAL_DIAGNOSTIC.findall(line)
            if structural:
                # Keep the exact structural reasons, bounded and de-duplicated.
                for match in _COMPILE_STRUCTURAL_DIAGNOSTIC.finditer(line):
                    detail = match.group(0).strip()[:200]
                    if detail and detail not in details:
                        details.append(detail)
                continue
            if not _COMPILE_DIAGNOSTIC_HINT.search(line):
                continue
            # Preserve only exception classes and a few safe, actionable
            # identifiers.  This deliberately drops arbitrary line context,
            # which may be user prompt or transcript content.
            kind_match = re.search(
                r"\b([A-Za-z_][\w.]*(?:Error|Exception|Failure|Timeout))\b",
                line,
            )
            kind = kind_match.group(1) if kind_match else None
            if kind == "NameError":
                name_match = re.search(r"name\s+['\"]([A-Za-z_][\w.]*)['\"]\s+is\s+not\s+defined", line, re.I)
                detail = f"NameError: undefined name {name_match.group(1)}" if name_match else "NameError"
            elif kind in {"ModuleNotFoundError", "ImportError"}:
                module_match = re.search(r"module named ['\"]([A-Za-z_][\w.]*)['\"]", line, re.I)
                detail = f"{kind}: missing module {module_match.group(1)}" if module_match else kind
            elif re.search(r"no such file|not found|missing", line, re.I):
                detail = f"{kind}: missing file" if kind else "missing file"
            elif re.search(r"permission denied", line, re.I):
                detail = f"{kind}: permission denied" if kind else "permission denied"
            elif re.search(r"timed? out|timeout", line, re.I):
                detail = f"{kind}: timeout" if kind else "compiler timeout"
            elif kind:
                detail = kind
            elif re.search(r"unavailable", line, re.I):
                detail = "compiler unavailable"
            else:
                detail = "compiler error"
            if detail not in details:
                details.append(detail)
            if len(details) >= 4:
                break
        if details:
            return "; ".join(details)[:240]
        if returncode is None:
            return "compiler failed"
        with contextlib.suppress(TypeError, ValueError):
            return f"compiler exited with returncode {int(returncode)}"
        return "compiler failed"

    def _publish_knowledge(self, project: Any) -> Any:
        """Publish the compiled project snapshot after local compilation.

        The local vault remains the source of truth.  A publisher failure is
        deliberately raised so the daily stage stays pending and retries on a
        later tick; no local knowledge file is replaced or rolled back.
        """
        module = _import_optional("ace_knowledge")
        publisher = getattr(module, "publish_project", None) if module else None
        if publisher is None:
            return None
        try:
            result = _await_if_needed(
                _call_variants(
                    publisher,
                    [
                        ((project, self._get_store()), {}),
                        ((self._project_vault_dir(project), self._get_store()), {}),
                        ((), {"project": project, "store": self._get_store()}),
                    ],
                )
            )
        except Exception as error:
            raise PipelineError("knowledge publication failed") from error
        if result is False:
            raise PipelineError("knowledge publication failed")
        return result

    def _default_analysis_runner(self, project: Any) -> Callable[..., Any] | None:
        """Build the bounded Luna runner used by the daily audit."""
        runner_module = _import_optional("codex_runner")
        learning_module = self.learning
        if runner_module is None or learning_module is None or not hasattr(learning_module, "build_snapshot_prompt"):
            return None
        root = Path(str(_first_attr(project, ("root",), self._project_vault_dir(project))))
        schema_builder = getattr(learning_module, "build_analysis_output_schema", None)
        output_schema = schema_builder() if callable(schema_builder) else None

        aggregate_diagnostics: Any = None

        async def runner(records: Sequence[Mapping[str, Any]], prompt: str) -> Any:
            nonlocal aggregate_diagnostics
            del records
            try:
                result = await runner_module.run_codex(
                    prompt,
                    cwd=root if root.is_dir() else self._project_vault_dir(project),
                    sandbox="read-only",
                    timeout=600,
                    output_schema=output_schema,
                )
            except Exception as error:
                diagnostics = getattr(error, "diagnostics", None)
                if diagnostics is not None:
                    if aggregate_diagnostics is None:
                        aggregate_diagnostics = diagnostics
                    elif callable(getattr(aggregate_diagnostics, "merge", None)):
                        aggregate_diagnostics.merge(diagnostics)
                raise
            if isinstance(result, tuple) and len(result) == 2:
                response, diagnostics = result
            else:
                response, diagnostics = result, None
            if diagnostics is not None:
                if aggregate_diagnostics is None:
                    aggregate_diagnostics = diagnostics
                elif callable(getattr(aggregate_diagnostics, "merge", None)):
                    aggregate_diagnostics.merge(diagnostics)
            return response, aggregate_diagnostics

        return runner

    @staticmethod
    def _safe_analysis_scalar(value: Any, limit: int = 1000) -> str:
        try:
            return _redact(value)[:limit]
        except Exception:
            return "<unserializable>"

    @classmethod
    def _safe_analysis_value(
        cls,
        value: Any,
        depth: int = 0,
        _seen: set[int] | None = None,
        _budget: list[int] | None = None,
    ) -> Any:
        """Bound, redact, and structurally preserve one analysis payload.

        Deep mappings/lists are retained as bounded containers instead of
        being stringified. Cycles, hostile iterables, and exhausted budgets
        fail closed with an explicit truncation marker.
        """
        seen = _seen if _seen is not None else set()
        budget = _budget if _budget is not None else [ANALYSIS_MAX_NODES]
        if budget[0] <= 0:
            if isinstance(value, Mapping):
                return {"_truncated": True}
            if isinstance(value, (list, tuple, set)):
                return ["<truncated>"]
            return cls._safe_analysis_scalar(value)
        if dataclasses.is_dataclass(value):
            try:
                value = dataclasses.asdict(value)
            except Exception:
                return {"_truncated": True}
        if isinstance(value, Mapping):
            if depth > ANALYSIS_MAX_DEPTH:
                return {"_truncated": True}
            identity = id(value)
            if identity in seen:
                return {"_truncated": True}
            seen.add(identity)
            result: dict[str, Any] = {}
            truncated = False
            try:
                for index, (key, item) in enumerate(value.items()):
                    if index >= 100 or budget[0] <= 0:
                        truncated = True
                        break
                    budget[0] -= 1
                    safe_key = cls._safe_analysis_scalar(key, 200)
                    if depth >= ANALYSIS_MAX_DEPTH and isinstance(item, (Mapping, list, tuple, set)):
                        safe_value: Any = (
                            {"_truncated": True}
                            if isinstance(item, Mapping)
                            else ["<truncated>"]
                        )
                    else:
                        safe_value = cls._safe_analysis_value(item, depth + 1, seen, budget)
                    result[safe_key] = safe_value
            except Exception:
                truncated = True
            finally:
                seen.discard(identity)
            if truncated:
                result["_truncated"] = True
            return result
        if isinstance(value, (list, tuple, set)):
            if depth > ANALYSIS_MAX_DEPTH:
                return ["<truncated>"]
            identity = id(value)
            if identity in seen:
                return ["<truncated>"]
            seen.add(identity)
            result_list: list[Any] = []
            truncated = False
            try:
                for index, item in enumerate(value):
                    if index >= 100 or budget[0] <= 0:
                        truncated = True
                        break
                    budget[0] -= 1
                    if depth >= ANALYSIS_MAX_DEPTH and isinstance(item, (Mapping, list, tuple, set)):
                        result_list.append(
                            {"_truncated": True}
                            if isinstance(item, Mapping)
                            else ["<truncated>"]
                        )
                    else:
                        result_list.append(cls._safe_analysis_value(item, depth + 1, seen, budget))
            except Exception:
                truncated = True
            finally:
                seen.discard(identity)
            if truncated:
                result_list.append("<truncated>")
            return result_list
        if isinstance(value, str):
            return cls._safe_analysis_scalar(value, 4000)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return cls._safe_analysis_scalar(value)

    @staticmethod
    def _write_analysis_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _analysis_success_indices(
        self, payload: Mapping[str, Any], snapshots: Sequence[Any]
    ) -> list[int]:
        """Return only rows whose individual analysis proof is valid."""
        raw_errors = payload.get("errors")
        if raw_errors in (None, [], {}):
            errors: list[Any] = []
        elif isinstance(raw_errors, list):
            errors = raw_errors
        else:
            errors = [raw_errors]
        error_ids: set[str] = set()
        payload_status = str(
            payload.get("analysis_status") or payload.get("status") or ""
        ).strip().lower()
        status_failure = payload_status in {"model-error", "error", "failed", "failure", "degraded"}
        global_error = False
        coverage = payload.get("coverage")
        try:
            global_error = global_error or not isinstance(coverage, Mapping) or int(
                coverage.get("sessions", -1)
            ) != len(snapshots)
        except (TypeError, ValueError):
            global_error = True
        for error in errors:
            if not isinstance(error, Mapping):
                global_error = True
                continue
            identity = (
                error.get("session_id")
                or error.get("conversation_id")
                or error.get("snapshot_id")
            )
            if identity in (None, ""):
                global_error = True
            else:
                error_ids.add(str(identity))
        # A degraded batch can still contain individually valid reports.  A
        # status without per-session attribution remains a global failure.
        if status_failure and (global_error or not error_ids):
            global_error = True
        reports = payload.get("reports")
        if not isinstance(reports, list):
            return []
        if len(reports) != len(snapshots):
            global_error = True
        accepted: list[int] = []
        for index, snapshot in enumerate(snapshots):
            report = reports[index] if index < len(reports) else None
            if not isinstance(report, Mapping):
                continue
            status = str(
                report.get("analysis_status")
                or report.get("status")
                or ""
            ).strip().lower()
            if status not in {"ok", "complete", "completed", "success", "succeeded"}:
                continue
            identity = self._snapshot_identity(snapshot)
            session_id = identity.get("session_id", "")
            report_id = (
                report.get("session_id")
                or report.get("conversation_id")
                or report.get("snapshot_id")
            )
            if report_id not in (None, "") and str(report_id) != session_id:
                continue
            if session_id and session_id in error_ids:
                continue
            if global_error:
                continue
            accepted.append(index)
        return accepted

    def _snapshot_stage_key(self, item: Any) -> tuple[str, str, str, str]:
        identity = self._snapshot_identity(item)
        return (
            identity.get("project_id", ""),
            identity.get("source", ""),
            identity.get("session_id", ""),
            identity.get("revision", ""),
        )

    @staticmethod
    def _analysis_row_key(row: Any, identity: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """Build a stable private key for one persisted analysis row."""
        if isinstance(identity, Mapping):
            values = tuple(
                str(identity.get(field) or "")
                for field in ("project_id", "source", "session_id", "revision")
            )
            if all(values):
                return ("snapshot", *values)
        if isinstance(row, Mapping):
            analysis_key = str(row.get("analysis_key") or "").strip()
            if analysis_key:
                return ("analysis", analysis_key)
            source = str(row.get("source") or "").strip()
            session = str(row.get("session_id") or "").strip()
            conversation = str(row.get("conversation_id") or "").strip()
            revision = str(row.get("revision") or "").strip()
            if not session and conversation:
                source, _, session = conversation.partition(":")
            if source or session or revision:
                return ("report", source, session, revision or conversation)
        return ("json", json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True))

    def _merge_analysis_rows(
        self,
        existing: Any,
        current: Any,
        existing_identities: Sequence[Any] = (),
        current_identities: Sequence[Any] = (),
    ) -> tuple[list[Any], list[dict[str, str]]]:
        """Merge successful batches while retaining snapshot provenance."""
        old_rows = existing if isinstance(existing, list) else []
        new_rows = current if isinstance(current, list) else []
        values: dict[tuple[str, ...], tuple[Any, dict[str, str]]] = {}
        order: list[tuple[str, ...]] = []

        def add(row: Any, identity: Any = None) -> None:
            key = self._analysis_row_key(row, identity if isinstance(identity, Mapping) else None)
            normalized_identity = {
                field: str(identity.get(field) or "")
                for field in ("project_id", "source", "session_id", "revision")
            } if isinstance(identity, Mapping) else {}
            if key not in values:
                order.append(key)
            values[key] = (row, normalized_identity)

        for index, row in enumerate(old_rows):
            identity = existing_identities[index] if index < len(existing_identities) else None
            add(row, identity)
        for index, row in enumerate(new_rows):
            identity = current_identities[index] if index < len(current_identities) else None
            add(row, identity)
        rows: list[Any] = []
        identities: list[dict[str, str]] = []
        for key in order:
            row, identity = values[key]
            rows.append(row)
            identities.append(identity)
        return rows, identities

    def _merge_daily_analysis_payload(
        self,
        path: Path,
        payload: Mapping[str, Any],
        snapshots: Sequence[Any],
    ) -> dict[str, Any]:
        """Keep prior successful same-day batches in the daily audit file."""
        current = dict(payload)
        current_identities = [self._snapshot_identity(item) for item in snapshots]
        current["snapshot_identities"] = current_identities
        try:
            previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous = {}
        if not isinstance(previous, Mapping):
            return current
        previous_reports = previous.get("reports")
        previous_conversations = previous.get("conversations")
        if not current_identities and (
            isinstance(previous_reports, list) and previous_reports
            or isinstance(previous_conversations, list) and previous_conversations
        ):
            # A deterministic no-evidence retry must not erase a valid report
            # already assembled from an earlier batch.
            return dict(previous)
        previous_status = str(
            previous.get("analysis_status") or previous.get("status") or ""
        ).strip().lower()
        if previous_status in {"model-error", "error", "failed", "failure", "degraded"}:
            previous = {}
        previous_identities = previous.get("snapshot_identities")
        previous_identities = previous_identities if isinstance(previous_identities, list) else []
        merged_reports, merged_identities = self._merge_analysis_rows(
            previous.get("reports"),
            current.get("reports"),
            previous_identities,
            current_identities,
        )
        if merged_reports:
            current["reports"] = merged_reports
            current["snapshot_identities"] = merged_identities

        # The deterministic renderers consume the flattened fields.  Rebuild
        # them from merged report rows when the native adapter supplied nested
        # conversation data; simple custom adapters keep their existing shape.
        for field in ("conversations", "incidents", "observations", "recommendations", "successes"):
            nested: list[Any] = []
            for report in merged_reports:
                if isinstance(report, Mapping) and isinstance(report.get(field), list):
                    nested.extend(report[field])
            if nested:
                current[field] = nested
            else:
                merged_field, _ = self._merge_analysis_rows(
                    previous.get(field),
                    current.get(field),
                    (),
                    current_identities
                    if isinstance(current.get(field), list)
                    and len(current.get(field, [])) == len(current_identities)
                    else (),
                )
                if merged_field:
                    current[field] = merged_field

        if merged_identities:
            coverage = current.get("coverage")
            coverage = dict(coverage) if isinstance(coverage, Mapping) else {}
            coverage["sessions"] = len(
                {
                    tuple(identity.get(field, "") for field in ("project_id", "source", "session_id", "revision"))
                    for identity in merged_identities
                    if all(identity.get(field) for field in ("project_id", "source", "session_id", "revision"))
                }
            )
            current["coverage"] = coverage
        return current

    def _default_analyze_daily(self, project: Any, day: str) -> bool:
        """Audit DB-acknowledged snapshots and write private project reports."""
        store = self._get_store()
        automation_since = self._automation_since(create=False)
        target = date.fromisoformat(day)
        source_after, source_before = self._source_day_bounds(project, target)
        snapshots = (
            self._db_snapshots(
                store,
                project,
                # Apply the per-message local-day filter before the bounded
                # analysis batch limit. A snapshot updated today may contain
                # only yesterday's messages and must not hide a later valid
                # candidate in the same source window.
                500,
                stage="analysis",
                minimum_started_at=automation_since,
                source_after=source_after,
                source_before=source_before,
            )
            if store is not None
            else []
        )
        snapshots = [
            bounded
            for item in snapshots
            if (bounded := self._snapshot_for_day(item, target, project)) is not None
        ][:ANALYSIS_BATCH_LIMIT]
        analysis_claims: dict[int, dict[str, Any]] = {}
        claims_required = self._store_uses_stage_claim(store)
        if store is not None and snapshots and claims_required:
            for index, snapshot in enumerate(snapshots):
                claim = self._claim_stage(store, snapshot, stage="analysis")
                if claim is None:
                    for claimed_index, claimed in analysis_claims.items():
                        self._release_stage(
                            store,
                            snapshots[claimed_index],
                            stage="analysis",
                            lease_owner=str(claimed.get("lease_owner") or ""),
                            host_id=str(claimed.get("host_id") or ""),
                        )
                    raise PipelineError("analysis stage claim unavailable")
                analysis_claims[index] = claim

        def release_analysis_claims(indices: Iterable[int] | None = None) -> None:
            if not claims_required:
                return
            selected = set(analysis_claims) if indices is None else set(indices)
            for index in selected:
                claim = analysis_claims.get(index)
                if claim is None:
                    continue
                self._release_stage(
                    store,
                    snapshots[index],
                    stage="analysis",
                    lease_owner=str(claim.get("lease_owner") or ""),
                    host_id=str(claim.get("host_id") or ""),
                )

        project_id = _project_id(project)
        analysis_state_dir = self.private_root / "analysis" / project_id
        audit_dir = self.private_root / "audits" / project_id
        report_root = self.private_root / "reports" / project_id
        learning = self.learning
        payload: Any
        no_evidence = not snapshots
        if no_evidence:
            # An empty acknowledged set is a valid deterministic outcome, not
            # an invitation to call Luna with an empty prompt.  Keep the
            # report explicit about the absence of evidence so coverage is
            # not mistaken for a measured zero.
            payload = {
                "schema_version": 1,
                "status": "no_evidence",
                "analysis_status": "no_evidence",
                "generated_at": self.now().isoformat(),
                "reports": [],
                "records": [],
                "coverage": {
                    "sessions": 0,
                    "status": "no_evidence",
                    "evidence": "none",
                },
                "errors": [],
                "limitations": ["no acknowledged snapshots available"],
            }
        elif learning is not None and hasattr(learning, "audit_snapshots_sync"):
            runner = self._default_analysis_runner(project)
            try:
                analysis_kwargs: dict[str, Any] = {
                    "store": store,
                    "state_dir": analysis_state_dir,
                    "audit_runner": runner,
                    "now": self.now(),
                }
                signals = self._capture_signals_for(project, day)
                if signals and self._accepts_keyword(learning.audit_snapshots_sync, "capture_signals"):
                    analysis_kwargs["capture_signals"] = signals
                payload = learning.audit_snapshots_sync(
                    [self._snapshot_payload(item) for item in snapshots],
                    **analysis_kwargs,
                )
            except Exception:
                release_analysis_claims()
                raise
        else:
            payload = {
                "schema_version": 1,
                "reports": [],
                "records": snapshots,
                "coverage": {"sessions": len(snapshots)},
                "errors": [],
            }
        # Extraction and compilation are measured by their own runners.  Add
        # those proofs to the exact source day before validation and audit
        # persistence, including a no-evidence report.  Existing analysis
        # metrics remain intact and are merged by stage name.
        if isinstance(payload, Mapping):
            payload = self._attach_stage_usage(payload, project, day)

        validation_errors: list[str] = []
        if not isinstance(payload, Mapping):
            payload = {
                "schema_version": 1,
                "status": "error",
                "analysis_status": "error",
                "reports": [],
                "coverage": {"sessions": 0},
                "errors": [{"kind": "invalid_payload", "status": "error"}],
            }
            validation_errors.append("daily analysis returned an invalid payload")
        else:
            payload = dict(payload)
        errors = payload.get("errors")
        errors_present = errors not in (None, [], {})
        coverage = payload.get("coverage")
        if not isinstance(coverage, Mapping):
            validation_errors.append("daily analysis coverage is unavailable")
            coverage = {"sessions": -1}
        try:
            covered = int(coverage.get("sessions", -1))
        except (TypeError, ValueError):
            covered = -1
        if covered != len(snapshots):
            validation_errors.append("daily analysis coverage is incomplete")
        reports = payload.get("reports")
        if len(snapshots) and (not isinstance(reports, list) or len(reports) != len(snapshots)):
            validation_errors.append("daily analysis reports are incomplete")
        payload_status = str(
            payload.get("analysis_status") or payload.get("status") or ""
        ).strip().lower()
        failed_status = payload_status in {"model-error", "error", "failed", "failure"}
        attempt_failed = bool(errors_present or validation_errors or failed_status)
        if attempt_failed:
            payload["status"] = "degraded" if not failed_status else payload_status
            payload["analysis_status"] = "model-error" if failed_status else "degraded"
            limitations = payload.get("limitations")
            if not isinstance(limitations, list):
                limitations = []
            payload["limitations"] = [
                *limitations[:20],
                *validation_errors,
                "last analysis attempt was not complete; unresolved snapshots remain pending",
            ]
        safe_payload = self._safe_analysis_value(payload)
        if not isinstance(safe_payload, Mapping):
            safe_payload = {"status": "error", "errors": [{"kind": "unsafe_payload"}]}

        # Keep a failed attempt separately so it never silently overwrites a
        # previously validated daily/no-evidence report.  If no validated
        # report exists yet, expose the degraded attempt at the normal path
        # as well so the daily renderer has something explicit to aggregate.
        audit_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        audit_path = audit_dir / f"{day}.json"
        if attempt_failed:
            self._write_analysis_json(audit_dir / f"{day}.attempt.json", safe_payload)
            if not audit_path.exists():
                self._write_analysis_json(audit_path, safe_payload)
        else:
            safe_payload = self._merge_daily_analysis_payload(
                audit_path,
                safe_payload,
                snapshots,
            )
            self._write_analysis_json(audit_path, safe_payload)

        # Nothing below this point mutates stage status until all validation
        # and report writes have succeeded.
        if not attempt_failed and not no_evidence and learning is not None and hasattr(learning, "render_reports"):
            learning.render_reports(
                payload,
                report_root / "analysis",
                now=datetime.fromisoformat(day).replace(tzinfo=PARIS),
                state_dir=analysis_state_dir,
            )

        report_module = _import_optional("ace_daily_report")
        if report_module is not None and hasattr(report_module, "build_report") and hasattr(report_module, "write_report"):
            content = report_module.build_report(
                report_date=date.fromisoformat(day),
                collection_state_path=self.state.path("collection"),
                incident_state_path=self.private_root / "incident-tracking.json",
                audit_dir=audit_dir,
                project_id=project_id,
            )
            report_module.write_report(report_root / "daily", content, date.fromisoformat(day))

        weekly_module = _import_optional("ace_weekly_report")
        if weekly_module is not None and hasattr(weekly_module, "build_report") and hasattr(weekly_module, "write_report"):
            weekly_content = weekly_module.build_report(
                report_date=date.fromisoformat(day),
                incident_state_path=self.private_root / "incident-tracking.json",
                audit_dir=audit_dir,
            )
            weekly_module.write_report(report_root / "weekly", weekly_content, date.fromisoformat(day))

        if attempt_failed:
            indices = self._analysis_success_indices(payload, snapshots)
        else:
            indices = list(range(len(snapshots)))
        acknowledgement_failed = False
        acknowledged_keys: set[tuple[str, str, str, str]] = set()
        acknowledged_indices: set[int] = set()
        if store is not None:
            for index in indices:
                claim = analysis_claims.get(index)
                if claim is not None and self._renew_stage_claim(
                    store, snapshots[index], stage="analysis"
                ) is None:
                    acknowledgement_failed = True
                    break
                if not self._mark_stage(
                    store,
                    snapshots[index],
                    stage="analysis",
                    status="succeeded",
                    lease_owner=claim.get("lease_owner") if claim else None,
                    host_id=claim.get("host_id") if claim else None,
                ):
                    acknowledgement_failed = True
                    break
                acknowledged_indices.add(index)
                acknowledged_keys.add(self._snapshot_stage_key(snapshots[index]))
            if attempt_failed:
                failure_code = "analysis_model_error" if errors_present or failed_status else "analysis_invalid_payload"
                for index, snapshot in enumerate(snapshots):
                    if index in acknowledged_indices:
                        continue
                    claim = analysis_claims.get(index)
                    if claim is not None and self._renew_stage_claim(
                        store, snapshot, stage="analysis"
                    ) is None:
                        acknowledgement_failed = True
                        continue
                    if not self._mark_stage(
                        store,
                        snapshot,
                        stage="analysis",
                        status="failed",
                        error=failure_code,
                        lease_owner=claim.get("lease_owner") if claim else None,
                        host_id=claim.get("host_id") if claim else None,
                    ):
                        acknowledgement_failed = True
        if acknowledgement_failed:
            release_analysis_claims(set(analysis_claims) - acknowledged_indices)
            raise PipelineError("database acknowledgement failed")
        if store is not None and snapshots:
            # The first query is intentionally bounded.  After ACK, probe a
            # bounded page again so a successful report cannot close the day
            # while more snapshots remain (or arrived during the analysis).
            remaining = self._db_snapshots(
                store,
                project,
                max(500, len(snapshots) + 1),
                stage="analysis",
                minimum_started_at=automation_since,
                source_after=source_after,
                source_before=source_before,
            )
            remaining = [
                bounded
                for item in remaining
                if (bounded := self._snapshot_for_day(item, target, project)) is not None
                if self._snapshot_stage_key(item) not in acknowledged_keys
            ]
            if remaining and not attempt_failed:
                release_analysis_claims(set(analysis_claims) - acknowledged_indices)
                raise PipelinePendingError("daily analysis has pending snapshots")
        if attempt_failed:
            release_analysis_claims(set(analysis_claims) - acknowledged_indices)
            raise PipelineError("daily analysis reported errors")
        return True

    def _write_weekly_report(self, project: Any, day: str) -> None:
        """Materialize the weekly view for custom analysis integrations too."""
        module = _import_optional("ace_weekly_report")
        if module is None or not hasattr(module, "build_report") or not hasattr(module, "write_report"):
            return
        project_id = _project_id(project)
        content = module.build_report(
            report_date=date.fromisoformat(day),
            incident_state_path=self.private_root / "incident-tracking.json",
            audit_dir=self.private_root / "audits" / project_id,
        )
        module.write_report(
            self.private_root / "reports" / project_id / "weekly",
            content,
            date.fromisoformat(day),
        )

    def _project_stage_state(
        self, state: dict[str, Any], project: Any
    ) -> dict[str, Any]:
        """Return durable state scoped to one registered project.

        Older ACE state files stored ``days`` and ``last_successful_day`` at
        the stage root.  Migrate that small shape lazily to the first project
        that resumes it, while keeping a compatibility mirror for callers
        that inspect the historical top-level fields.  Pipeline decisions
        always use the nested project record, so one project's date cannot
        suppress another project's work.
        """
        project_id = _project_id(project)
        projects = state.setdefault("projects", {})
        if not isinstance(projects, dict):
            projects = {}
            state["projects"] = projects
        record = projects.get(project_id)
        if not isinstance(record, dict):
            record = {}
            # A legacy state has no project identity.  Associate it only when
            # this project first resumes; future projects use independent
            # state instead of inheriting the global cursor.
            if not projects and not state.get("legacy_alias_for"):
                for key in ("days", "last_successful_day", "files"):
                    value = state.get(key)
                    if value is not None:
                        record[key] = value.copy() if isinstance(value, dict) else value
            projects[project_id] = record
        record.setdefault("days", {})
        record.setdefault("files", {})
        return record

    @staticmethod
    def _daily_file_hash(path: Path) -> str | None:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except OSError:
            return None

    def _legacy_compile_records(self, project: Any) -> Mapping[str, Any]:
        """Read the pre-ACE compiler ledger without modifying it."""
        state_path = self._project_vault_dir(project) / ".state" / "state.json"
        try:
            value = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        records = value.get("ingested") if isinstance(value, Mapping) else None
        return records if isinstance(records, Mapping) else {}

    @staticmethod
    def _legacy_daily_keys(path: Path) -> tuple[str, ...]:
        return (path.name, f"daily/{path.name}", path.stem)

    def _legacy_article_references(self, project: Any) -> set[str]:
        """Return daily filenames explicitly cited by existing knowledge."""
        knowledge_dir = self._project_vault_dir(project) / "knowledge"
        if not knowledge_dir.is_dir():
            return set()
        references: set[str] = set()
        pattern = re.compile(r"(?<![A-Za-z0-9_.-])daily/(\d{4}-\d{2}-\d{2}\.md)")
        try:
            articles = knowledge_dir.rglob("*.md")
        except OSError:
            return set()
        for article in articles:
            if not article.is_file():
                continue
            try:
                content = article.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            references.update(match.group(1) for match in pattern.finditer(content))
        return references

    def _legacy_compile_proof_valid(
        self,
        project: Any,
        daily: Path,
        *,
        records: Mapping[str, Any] | None = None,
        references: set[str] | None = None,
        current_hash: str | None = None,
    ) -> bool:
        """Validate a legacy compile mark against today’s file and article."""
        records = self._legacy_compile_records(project) if records is None else records
        record: Any = None
        for key in self._legacy_daily_keys(daily):
            candidate = records.get(key)
            if isinstance(candidate, Mapping):
                record = candidate
                break
        if record is None or not record.get("compiled_at"):
            return False
        current_hash = self._daily_file_hash(daily) if current_hash is None else current_hash
        if current_hash is None or str(record.get("hash", "")).lower() != current_hash[:16].lower():
            return False
        references = self._legacy_article_references(project) if references is None else references
        return daily.name in references

    def _record_stage_alias(
        self, state: dict[str, Any], project: Any, project_state: Mapping[str, Any]
    ) -> None:
        """Keep the old inspection shape without using it for scheduling."""
        state["last_successful_day"] = project_state.get("last_successful_day")
        state["days"] = dict(project_state.get("days", {}))
        state["files"] = dict(project_state.get("files", {}))
        state["legacy_alias_for"] = _project_id(project)

    def _due_days(
        self,
        compile_state: Mapping[str, Any],
        target: date,
        project: Any | None = None,
        minimum_mtime: float | None = None,
    ) -> list[str]:
        """Find normal and late daily files not yet compiled by content hash."""
        previous = compile_state.get("last_successful_day")
        days: set[date] = set()
        daily_files: list[tuple[date, Path]] = []
        if project is not None:
            daily_dir = self._project_vault_dir(project) / "daily"
            if daily_dir.is_dir():
                for path in sorted(daily_dir.glob("*.md")):
                    if minimum_mtime is not None:
                        with contextlib.suppress(OSError):
                            if path.stat().st_mtime < minimum_mtime:
                                continue
                    try:
                        file_day = date.fromisoformat(path.stem)
                    except ValueError:
                        continue
                    if file_day <= target:
                        daily_files.append((file_day, path))
        if previous:
            try:
                start = date.fromisoformat(str(previous)) + timedelta(days=1)
            except ValueError:
                start = target
            if start <= target and daily_files:
                days.update(
                    file_day
                    for file_day, _path in daily_files
                    if start <= file_day <= target
                )
        else:
            start = None
        # Initial historical compilation must start from actual source files,
        # not from today's date, and late files must be reconsidered even after
        # the cursor has advanced.
        if start is None and daily_files:
            days.update(file_day for file_day, _path in daily_files)
        elif start is None and not daily_files:
            days.add(target)
        known_files = compile_state.get("files", {})
        if not isinstance(known_files, Mapping):
            known_files = {}
        legacy_records = self._legacy_compile_records(project) if project is not None else {}
        legacy_references = self._legacy_article_references(project) if project is not None else set()
        for file_day, path in daily_files:
            digest = self._daily_file_hash(path)
            key = f"daily/{path.name}"
            record = known_files.get(key, {})
            if project is not None and self._legacy_compile_proof_valid(
                project,
                path,
                records=legacy_records,
                references=legacy_references,
                current_hash=digest,
            ):
                days.discard(file_day)
                continue
            if not isinstance(record, Mapping) or digest is None or record.get("hash") != digest or record.get("status") != "complete":
                days.add(file_day)
        due = sorted(item for item in days if item <= target)
        if len(due) <= DEFAULT_LIMIT:
            return [item.isoformat() for item in due]
        # A first ACE run can inherit hundreds of old CMC daily files.  Give
        # the current target priority and spend the remaining bounded budget
        # on the oldest late days so the historical backlog resumes fairly.
        if target in due:
            due = [target] + [item for item in due if item != target]
        return [item.isoformat() for item in due[:DEFAULT_LIMIT]]

    def daily(
        self,
        *,
        project: Any | None = None,
        cwd: str | Path | None = None,
        day: str | None = None,
        now: datetime | None = None,
        _lock_held: bool = False,
    ) -> dict[str, Any]:
        # Keep direct daily runs serialized with tick/process.  Calls from
        # tick use the already-held lock instead of re-entering it.
        if not _lock_held:
            with advisory_lock(self.private_root / "tick.lock"):
                return self.daily(
                    project=project,
                    cwd=cwd,
                    day=day,
                    now=now,
                    _lock_held=True,
                )
        if project is None:
            project = self._resolve_project(cwd)
        project_tz = self._project_timezone(project)
        current = (now or self.now()).astimezone(project_tz)
        target = date.fromisoformat(day) if day else current.date() - timedelta(days=1)
        target_day = target.isoformat()
        automation_since = self._automation_since(create=False)
        native_target_allowed = True
        if day is None and automation_since is not None:
            cutoff_day = datetime.fromtimestamp(
                automation_since,
                tz=timezone.utc,
            ).astimezone(project_tz).date()
            # Activation on a new day must not compile or analyze the prior
            # calendar day as an implicit historical backfill.  An explicit
            # --day remains an exact, intentional request.
            native_target_allowed = target >= cutoff_day
        compile_state = self.state.read("compile")
        analysis_state = self.state.read("analysis")
        project_compile = self._project_stage_state(compile_state, project)
        project_analysis = self._project_stage_state(analysis_state, project)
        counts: dict[str, Any] = {"days": 0, "compiled": 0, "analyzed": 0, "failed": 0, "pending": 0}
        compile_days = self._due_days(
            project_compile,
            target,
            project,
            minimum_mtime=None if day is not None else automation_since,
        )
        if day is not None:
            compile_days = [item for item in compile_days if item == target_day]
        elif automation_since is not None:
            compile_days = [item for item in compile_days if item == target_day] if native_target_allowed else []
        target_daily_exists = (
            self._project_vault_dir(project) / "daily" / f"{target.isoformat()}.md"
        ).is_file()
        custom_compile = self._store_method(
            self.learning,
            ("compile_daily", "compile_day", "compile"),
        ) if self.learning is not None else None
        if not target_daily_exists and custom_compile is None:
            # _due_days retains its historical no-file target sentinel for
            # callers that only want a stage cursor.  At this boundary it is
            # analysis-only work, never a custom compiler/LLM invocation.
            compile_days = [item for item in compile_days if item != target.isoformat()]
        analysis_days = project_analysis.get("days", {})
        target_analysis_pending = not (
            isinstance(analysis_days, Mapping)
            and isinstance(analysis_days.get(target_day), Mapping)
            and analysis_days[target_day].get("status") == "complete"
        )
        if not native_target_allowed:
            target_analysis_pending = False
        if not target_analysis_pending:
            # A complete report is not a permanent cursor: a newer DB
            # revision for the same day must reopen the analysis stage.  Use
            # a bounded, metadata-only probe so a large backlog cannot make
            # the daily scheduler unbounded.
            store = self._get_store()
            if store is not None:
                source_after, source_before = self._source_day_bounds(project, target)
                target_analysis_pending = any(
                    self._snapshot_for_day(item, target, project) is not None
                    for item in self._db_snapshots(
                        store,
                        project,
                        500,
                        stage="analysis",
                        minimum_started_at=automation_since,
                        source_after=source_after,
                        source_before=source_before,
                    )
                )
        compile_day_set = set(compile_days)
        # Compilation and analysis are independent stage cursors.  Compile
        # every due daily file first; a pending target audit must never break
        # the historical/late compilation backlog.
        for target_day in compile_days:
            counts["days"] += 1
            compile_succeeded = False
            try:
                result = self._learning_call(("compile_daily", "compile_day", "compile"), project, target_day)
                if result is _MISSING:
                    raise PipelineError("daily compiler is unavailable")
                if result is False:
                    raise PipelineError("daily compilation failed")
                compile_succeeded = True
                # A successful retry clears the previous failure detail.  The
                # local cache also prevents a stale diagnostic written by the
                # compiler from being reintroduced when this method persists
                # its already-loaded stage state below.
                project_compile.setdefault("diagnostics", {}).pop(target_day, None)
                self._compile_diagnostic_cache.pop((_project_id(project), target_day), None)
                project_compile.setdefault("days", {})[target_day] = {"status": "complete", "completed_at": self.now().isoformat()}
                daily_path = self._project_vault_dir(project) / "daily" / f"{target_day}.md"
                digest = self._daily_file_hash(daily_path)
                if digest is not None:
                    project_compile.setdefault("files", {})[f"daily/{daily_path.name}"] = {
                        "hash": digest,
                        "status": "complete",
                        "compiled_at": self.now().isoformat(),
                    }
                counts["compiled"] += 1
                # Compilation has its own cursor.  Analysis can be retried
                # independently without recompiling the unchanged source.
                previous = project_compile.get("last_successful_day")
                if not previous or str(target_day) > str(previous):
                    project_compile["last_successful_day"] = target_day
            except Exception as error:
                diagnostic_key = (_project_id(project), target_day)
                diagnostic = self._compile_diagnostic_cache.pop(diagnostic_key, None)
                pending_continuation = isinstance(error, PipelinePendingError)
                if compile_succeeded:
                    project_compile.setdefault("diagnostics", {}).pop(target_day, None)
                elif diagnostic is not None:
                    project_compile.setdefault("diagnostics", {})[target_day] = diagnostic
                if compile_succeeded:
                    project_compile.setdefault("days", {})[target_day] = {
                        "status": "complete",
                        "error_type": type(error).__name__,
                    }
                else:
                    project_compile.setdefault("days", {})[target_day] = {
                        "status": "failed",
                        "error_type": type(error).__name__,
                    }
                counts["failed"] += 1
                counts["pending"] += 1
                self._record_stage_alias(compile_state, project, project_compile)
                self._record_stage_alias(analysis_state, project, project_analysis)
                self.state.write("compile", compile_state)
                self.state.write("analysis", analysis_state)
                continue
            self._record_stage_alias(compile_state, project, project_compile)
            self._record_stage_alias(analysis_state, project, project_analysis)
            self.state.write("compile", compile_state)
            self.state.write("analysis", analysis_state)

        # Audit only the requested target date, never each late historical
        # compile date.  It runs after the compile pass so neither stage can
        # starve the other when the target has a pending continuation.
        if target_analysis_pending:
            target_in_compile = target_day in compile_day_set
            target_compile_record = project_compile.setdefault("days", {}).get(target_day)
            if not target_in_compile:
                counts["days"] += 1
                if not isinstance(target_compile_record, Mapping):
                    project_compile.setdefault("days", {})[target_day] = {
                        "status": "analysis_only",
                        "completed_at": self.now().isoformat(),
                    }
            try:
                analysis = self._learning_call(
                    ("analyze_daily", "analysis", "analyze", "build_report"),
                    project,
                    target_day,
                )
                if analysis is False:
                    raise PipelineError("daily analysis failed")
                if analysis is _MISSING:
                    project_analysis.setdefault("days", {})[target_day] = {
                        "status": "skipped",
                        "reason": "not_configured",
                    }
                else:
                    # Custom learning adapters may only implement the
                    # analysis call; keep the weekly artifact contract at
                    # the pipeline boundary as well.
                    if self.learning is not None and any(
                        callable(getattr(self.learning, name, None))
                        for name in ("analyze_daily", "analysis", "analyze", "build_report")
                    ):
                        self._write_weekly_report(project, target_day)
                    project_analysis.setdefault("days", {})[target_day] = {
                        "status": "complete",
                        "completed_at": self.now().isoformat(),
                    }
                    counts["analyzed"] += 1
            except Exception as error:
                pending_continuation = isinstance(error, PipelinePendingError)
                target_compile_record = project_compile.setdefault("days", {}).get(target_day)
                if not isinstance(target_compile_record, Mapping):
                    target_compile_record = {"status": "analysis_only"}
                else:
                    target_compile_record = dict(target_compile_record)
                if pending_continuation:
                    target_compile_record.update(
                        {
                            "analysis_status": "pending",
                            "pending_reason": "snapshots_remaining",
                        }
                    )
                else:
                    target_compile_record.update(
                        {
                            "analysis_status": "failed",
                            "error_type": type(error).__name__,
                        }
                    )
                project_compile.setdefault("days", {})[target_day] = target_compile_record
                project_analysis.setdefault("days", {})[target_day] = {
                    "status": "pending",
                    "reason": "snapshots_remaining" if pending_continuation else "stage_failed",
                }
                if not pending_continuation:
                    counts["failed"] += 1
                counts["pending"] += 1
            self._record_stage_alias(compile_state, project, project_compile)
            self._record_stage_alias(analysis_state, project, project_analysis)
            self.state.write("compile", compile_state)
            self.state.write("analysis", analysis_state)
        return counts

    def _read_existing_outbox_summary(self) -> dict[str, Any] | None:
        """Read an existing SQLite outbox without opening/creating it."""
        path = self.private_root / "outbox.sqlite3"
        if not path.is_file():
            return None
        try:
            connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    "SELECT status, COUNT(*) AS count, COALESCE(SUM(payload_bytes), 0) AS bytes "
                    "FROM ace_outbox GROUP BY status"
                ).fetchall()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            return None
        statuses = {
            str(row[0]): {"count": int(row[1]), "bytes": int(row[2])}
            for row in rows
        }
        pending = sum(
            int(statuses.get(name, {}).get("count", 0) or 0)
            for name in ("pending", "retry", "inflight")
        )
        return {
            "total": sum(item["count"] for item in statuses.values()),
            "pending": pending,
            "statuses": statuses,
        }

    def morning_report(self, *, day: str | None = None) -> dict[str, Any]:
        """Write the consolidated, model-free morning report for ``day``.

        The renderer only reads analyses and audits already written under the
        private root. A missing renderer or a rendering failure is reported,
        never raised, so the native tick keeps its stage counters intact.
        """
        target = day or self.now().astimezone(PARIS).date().isoformat()
        module = _import_optional("ace_morning_report")
        if module is None or not hasattr(module, "build_report") or not hasattr(module, "write_report"):
            return {"status": "unavailable", "day": target}
        try:
            content = module.build_report(self.private_root, target)
            path = module.write_report(self.private_root / "reports" / "morning", content, target)
        except Exception as error:  # noqa: BLE001 - reported, never raised
            return {"status": "failed", "day": target, "error_type": type(error).__name__}
        return {"status": "written", "day": target, "path": str(path)}

    def status(self) -> dict[str, Any]:
        result: dict[str, Any] = {"version": PIPELINE_VERSION, "projects": len(self._project_ids())}
        if self._outbox_instance is not None:
            summary = getattr(self._outbox_instance, "summary", None)
            if callable(summary):
                with contextlib.suppress(Exception):
                    value = summary()
                    if isinstance(value, Mapping):
                        result["outbox"] = dict(value)
                        if "pending" not in result["outbox"]:
                            result["outbox"]["pending"] = self._outbox_pending_count(self._outbox_instance)
        else:
            result["outbox"] = self._read_existing_outbox_summary() or {"available": False}
        for stage in STAGES:
            with contextlib.suppress(Exception):
                value = self.state.read(stage)
                result[stage] = {"records": len(value.get("snapshots", value.get("days", value.get("sessions", {}))))}
        return result

    def history(self, *, project_id: str, query: str, limit: int = 10, max_chars: int = DEFAULT_MAX_HISTORY_CHARS) -> dict[str, Any]:
        store = self._get_store()
        if store is None:
            return {"results": [], "offline": True}
        if project_id not in self._project_ids() and project_id != "*":
            raise NotInitializedError("history project is not initialized")
        reader = self._store_method(store, ("history", "search_history", "query_history", "search"))
        if reader is None:
            return {"results": [], "offline": True}
        values = _await_if_needed(_call_variants(reader, [((project_id, query), {"limit": limit}), ((), {"project_id": project_id, "query": query, "limit": limit}), ((query,), {"limit": limit})]))
        bounded: list[str] = []
        remaining = max(0, max_chars)
        for value in list(values or [])[: max(0, limit)]:
            text = _redact(json.dumps(_json_safe(value), ensure_ascii=False))
            text = text[:remaining]
            if not text:
                break
            bounded.append(text)
            remaining -= len(text)
        return {"results": bounded, "offline": False}

    def _delegate(self, command: str, cwd: str | Path, flags: Sequence[str]) -> dict[str, Any]:
        project = self._resolve_project(cwd)
        script = SCRIPT_DIR / f"{command}.py"
        if not script.exists():
            raise PipelineError(f"delegate unavailable: {command}")
        env = dict(os.environ)
        env["ACE_PROJECT_DIR"] = str(self._project_vault_dir(project))
        delegate_flags = list(flags)
        if command == "scan_md" and "--cwd" not in delegate_flags:
            # scan-md must discover files from the verified source project
            # root.  ACE_PROJECT_DIR remains the vault target for its output.
            source_root = _first_attr(project, ("root", "repo_root", "source_root"), cwd)
            delegate_flags = ["--cwd", str(Path(str(source_root)).expanduser().resolve()), *delegate_flags]
        # Delegated compile/scan work shares the native pipeline lock.  This
        # prevents a direct command from racing a tick that owns collection,
        # processing, or daily stages.
        lock_path = self.private_root / "tick.lock"
        with advisory_lock(lock_path):
            result = subprocess.run([sys.executable, str(script), *delegate_flags], cwd=str(CONFIG_ROOT), env=env, capture_output=True, text=True, check=False)
        # Keep the delegated command's answer visible to the native CLI.  In
        # particular ``ace query`` writes its synthesized answer to stdout;
        # returning only the exit code made a successful query appear empty to
        # callers.  Stderr is retained for actionable failures, with the same
        # redaction used by other pipeline diagnostics.
        return {
            "command": command,
            "returncode": int(result.returncode),
            "ok": result.returncode == 0,
            "stdout": _redact(result.stdout or ""),
            "stderr": _redact(result.stderr or ""),
        }

    def tick(self, *, cwd: str | Path | None = None, role: str = "processor", source_paths: Sequence[str | Path] = (), source: str | None = None, limit: int = DEFAULT_LIMIT, all_history: bool = False, now: datetime | None = None) -> dict[str, Any]:
        with advisory_lock(self.private_root / "tick.lock"):
            result: dict[str, Any] = {"role": role}
            automation_since = None if all_history else self._automation_since(create=True)
            effective_limit = (
                min(limit, AUTOMATION_PROCESS_LIMIT)
                if automation_since is not None
                else limit
            )
            if role == "collector":
                result["collect"] = self.collect(source_paths, cwd=cwd, source=source, limit=limit, all_history=all_history, sync=True)
                return result
            if cwd is None and not source_paths:
                # The installed native service invokes ``ace tick`` without a
                # project argument. Resolve every explicitly registered
                # project, collect all local providers, then process and run
                # the due daily work for each project. Never resolve the
                # service's working directory as an implicit project.
                projects: list[Any] = self._processable_projects()
                # Every stage is isolated below. One failing project, or one
                # failing stage, must never cancel the whole cycle: that is how
                # a single unusable conversation silently stopped collection,
                # analysis and the morning report for hours.
                stage_errors: list[dict[str, str]] = []

                def run_stage(name: str, project_name: str, call: Callable[[], Any]) -> Any:
                    try:
                        return call()
                    except PipelineBusyError:
                        raise
                    except Exception as error:  # noqa: BLE001 - reported, never raised
                        stage_errors.append(
                            {
                                "stage": name,
                                "project": project_name,
                                "error_type": type(error).__name__,
                            }
                        )
                        logging.warning("ace tick stage failed: %s/%s %s", project_name, name, type(error).__name__)
                        return None

                result["collect"] = run_stage(
                    "collect",
                    "all",
                    lambda: self.collect(
                        (),
                        source=source,
                        limit=max(limit, AUTOMATION_COLLECT_LIMIT),
                        all_history=all_history,
                        sync=False,
                    ),
                ) or {"candidates": 0, "queued": 0, "failed": 0, "offline": False}
                result["projects"] = len(projects)
                result["sync"] = {"synced": 0, "failed": 0, "offline": False}
                result["process"] = {"candidates": 0, "processed": 0, "empty": 0, "baseline": 0, "failed": 0, "pending": 0, "offline": False}
                result["daily"] = {"projects": 0, "days": 0, "compiled": 0, "analyzed": 0, "failed": 0, "pending": 0}
                local_now = (now or self.now()).astimezone(PARIS)
                daily_due = automation_since is None or self._automation_daily_due(local_now)
                for project in projects:
                    project_name = _project_name(project) or "unknown"
                    sync_result = run_stage(
                        "sync",
                        project_name,
                        lambda project=project: self.sync(
                            project=project,
                            limit=max(1, effective_limit * 4),
                            created_after=automation_since,
                        ),
                    ) or {"synced": 0, "failed": 1, "offline": False}
                    for key in ("synced", "failed"):
                        result["sync"][key] += int(sync_result.get(key, 0) or 0)
                    result["sync"]["offline"] = result["sync"]["offline"] or bool(sync_result.get("offline"))
                    process_result = run_stage(
                        "process",
                        project_name,
                        lambda project=project: self.process(
                            project=project,
                            limit=effective_limit,
                            minimum_started_at=automation_since,
                            _lock_held=True,
                        ),
                    ) or {"candidates": 0, "processed": 0, "empty": 0, "baseline": 0, "failed": 1, "pending": 0, "offline": False}
                    for key in ("candidates", "processed", "empty", "baseline", "failed", "pending"):
                        result["process"][key] += int(process_result.get(key, 0) or 0)
                    result["process"]["offline"] = result["process"]["offline"] or bool(process_result.get("offline"))
                    if daily_due:
                        daily_result = run_stage(
                            "daily",
                            project_name,
                            lambda project=project: self.daily(project=project, now=local_now, _lock_held=True),
                        ) or {"days": 0, "compiled": 0, "analyzed": 0, "failed": 1, "pending": 0}
                        # A day with more conversations than one analysis batch
                        # reports ``pending``; continue a few times in the same
                        # tick instead of waiting 30 minutes per batch.
                        attempts = 1
                        while (
                            int(daily_result.get("pending", 0) or 0) > 0
                            and int(daily_result.get("failed", 0) or 0) == 0
                            and int(daily_result.get("analyzed", 0) or 0) == 0
                            and attempts < DAILY_CONTINUATIONS_PER_TICK
                        ):
                            attempts += 1
                            daily_result = run_stage(
                                "daily",
                                project_name,
                                lambda project=project: self.daily(project=project, now=local_now, _lock_held=True),
                            ) or {"days": 0, "compiled": 0, "analyzed": 0, "failed": 1, "pending": 0}
                        result["daily"]["projects"] += 1
                        for key in ("days", "compiled", "analyzed", "failed", "pending"):
                            result["daily"][key] += int(daily_result.get(key, 0) or 0)
                if automation_since is not None and daily_due:
                    # Persist the attempt only after every due project has
                    # returned. Failed/pending stages therefore remain
                    # retryable on the same day.
                    self._claim_automation_daily(local_now, result["daily"])
                if daily_due:
                    # One readable report across every project, written after
                    # the per-project analyses. It never calls a model.
                    result["report"] = self.morning_report(day=local_now.date().isoformat())
                if stage_errors:
                    result["stage_errors"] = stage_errors[:20]
                return result
            project = self._resolve_project(cwd)
            result["sync"] = self.sync(
                project=project,
                limit=max(1, effective_limit * 4),
                created_after=automation_since,
            )
            result["process"] = self.process(
                project=project,
                limit=effective_limit,
                minimum_started_at=automation_since,
                _lock_held=True,
            )
            local_now = (now or self.now()).astimezone(PARIS)
            daily_due = automation_since is None or self._automation_daily_due(local_now)
            if daily_due:
                result["daily"] = self.daily(project=project, now=local_now, _lock_held=True)
                if automation_since is not None:
                    self._claim_automation_daily(local_now, result["daily"])
            else:
                result["daily"] = {"due": False}
            return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACE bounded collection and processing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize the explicit project at --cwd")
    init.add_argument("--cwd", required=True)

    collect = sub.add_parser("collect", help="discover metadata and enqueue authorized transcripts")
    collect.add_argument("sourcepaths", nargs="*")
    collect.add_argument("--path", action="append", dest="paths")
    collect.add_argument("--cwd")
    collect.add_argument("--source", choices=("claude", "codex", "hermes"))
    collect.add_argument("--host-id")
    collect.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    collect.add_argument("--days", type=int, default=7)
    collect.add_argument("--since", help="only transcripts modified since YYYY-MM-DD (explicit history bound)")
    collect.add_argument("--dry-run", action="store_true", help="preview routing; no queue, no cursor, no state write")
    collect.add_argument("--all-history", action="store_true")
    collect.add_argument("--sync", action="store_true")
    collect.add_argument("--extract", action="store_true", help="write daily logs locally right after capture")

    sync = sub.add_parser("sync", help="send pending normalized snapshots to the database")
    sync.add_argument("--cwd")
    sync.add_argument("--limit", type=int, default=100)

    process = sub.add_parser("process", help="extract only database-acquitted snapshots")
    process.add_argument("--cwd")
    process.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    process.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)

    tick = sub.add_parser("tick", help="run one locked collector or processor tick")
    tick.add_argument("sourcepaths", nargs="*")
    tick.add_argument("--cwd")
    tick.add_argument("--source", choices=("claude", "codex", "hermes"))
    tick.add_argument("--role", choices=("collector", "processor"), default="processor")
    tick.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    tick.add_argument("--all-history", action="store_true")

    daily = sub.add_parser("daily", help="compile and analyze pending daily logs")
    daily.add_argument("--cwd")
    daily.add_argument("--day")

    status = sub.add_parser("status", help="show sanitized counts without an LLM")
    status.add_argument("--cwd")

    report = sub.add_parser("report", help="write the consolidated morning report without an LLM")
    report.add_argument("--day", help="local day YYYY-MM-DD; default today")

    history = sub.add_parser("history", help="query bounded, explicitly requested history")
    history.add_argument("--project", required=True)
    history.add_argument("--query", required=True)
    history.add_argument("--limit", type=int, default=10)
    history.add_argument("--max-chars", type=int, default=DEFAULT_MAX_HISTORY_CHARS)

    lint_parser = sub.add_parser("lint", help="explicitly delegate the knowledge base health checks")
    lint_parser.add_argument("--cwd")
    lint_parser.add_argument("--structural-only", action="store_true", help="skip the LLM contradiction check")

    compile_parser = sub.add_parser("compile", help="explicitly delegate the existing compiler")
    compile_parser.add_argument("--cwd", required=True)
    compile_parser.add_argument("--all", action="store_true")
    compile_parser.add_argument("--file")
    compile_parser.add_argument("--dry-run", action="store_true")

    query = sub.add_parser("query", help="explicitly delegate the existing query command")
    query.add_argument("--cwd", required=True)
    query.add_argument("question")

    scan = sub.add_parser("scan-md", help="explicitly delegate Markdown discovery")
    scan.add_argument("--cwd", required=True)
    scan.add_argument("--all", action="store_true")
    scan.add_argument("--days", type=int)
    scan.add_argument("--since")
    scan.add_argument("--dry-run", action="store_true")

    for name in ("migration", "schema", "schedule", "knowledge"):
        delegated = sub.add_parser(name, help=f"explicitly delegate {name} operations")
        # ``argparse.REMAINDER`` does not consume an option when it is the
        # first token after a subcommand (notably ``ace schedule --json``).
        # Capture the common JSON flag explicitly, then forward it below.
        delegated.add_argument("--json", action="store_true", dest="json_flag")
        delegated.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _print_result(value: Mapping[str, Any]) -> None:
    # Never print integration output or transcript bodies on normal status
    # paths.  History is already explicitly bounded and redacted.
    print(json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True))


def _result_exit_code(value: Any) -> int:
    """Translate a command result into the native CLI exit status.

    Pipeline commands return structured counters, while delegated commands
    expose their subprocess status as ``returncode``.  Keep the JSON result
    intact for callers, but make failures observable to launchd/systemd and
    shell callers.  Tick nests stage counters, so mappings are traversed
    recursively.
    """
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if not isinstance(item, Mapping):
            return 0
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        returncode = item.get("returncode")
        if isinstance(returncode, (int, float)) and not isinstance(returncode, bool) and returncode != 0:
            return int(returncode)
        failed = item.get("failed")
        if isinstance(failed, (int, float)) and not isinstance(failed, bool) and failed > 0:
            return 1
        for child in item.values():
            code = visit(child)
            if code:
                return code
        return 0

    return visit(value)


def dispatch(args: argparse.Namespace, pipeline: ACEPipeline) -> dict[str, Any]:
    command = args.command
    if command == "init":
        return pipeline._init_project(args.cwd)
    if command == "collect":
        paths = list(args.sourcepaths or []) + list(args.paths or [])
        since = None
        if args.since:
            since = datetime.combine(date.fromisoformat(args.since), datetime.min.time(), tzinfo=PARIS).timestamp()
        return pipeline.collect(paths, cwd=args.cwd, source=args.source, host_id=args.host_id, limit=args.limit, days=args.days, all_history=args.all_history or since is not None, sync=args.sync, extract=args.extract, since=since, dry_run=args.dry_run)
    if command == "sync":
        return pipeline.sync(cwd=args.cwd, limit=args.limit)
    if command == "process":
        return pipeline.process(cwd=args.cwd, limit=args.limit, max_context_chars=args.max_context_chars)
    if command == "tick":
        return pipeline.tick(cwd=args.cwd, role=args.role, source_paths=args.sourcepaths, source=args.source, limit=args.limit, all_history=args.all_history)
    if command == "daily":
        return pipeline.daily(cwd=args.cwd, day=args.day)
    if command == "status":
        return pipeline.status()
    if command == "report":
        return pipeline.morning_report(day=args.day)
    if command == "history":
        return pipeline.history(project_id=args.project, query=args.query, limit=args.limit, max_chars=args.max_chars)
    if command == "compile":
        flags: list[str] = []
        if args.all:
            flags.append("--all")
        if args.file:
            flags.extend(("--file", args.file))
        if args.dry_run:
            flags.append("--dry-run")
        return pipeline._delegate("compile", args.cwd, flags)
    if command == "query":
        return pipeline._delegate("query", args.cwd, [args.question])
    if command == "lint":
        return pipeline._delegate("lint", args.cwd, ["--structural-only"] if args.structural_only else [])
    if command == "scan-md":
        flags = []
        if args.all:
            flags.append("--all")
        if args.days is not None:
            flags.extend(("--days", str(args.days)))
        if args.since:
            flags.extend(("--since", args.since))
        if args.dry_run:
            flags.append("--dry-run")
        return pipeline._delegate("scan_md", args.cwd, flags)
    if command in {"migration", "schema", "schedule", "knowledge"}:
        module_name = {
            "migration": "ace_migrate",
            "schema": "ace_schema",
            "schedule": "ace_schedule",
            "knowledge": "ace_knowledge",
        }[command]
        module = _import_optional(module_name)
        if module is None:
            raise PipelineError(f"delegate unavailable: {command}")
        function = getattr(module, "main", None) or getattr(module, command, None)
        if function is None:
            raise PipelineError(f"delegate unavailable: {command}")
        forwarded = list(getattr(args, "args", []) or [])
        if bool(getattr(args, "json_flag", False)) and "--json" not in forwarded:
            forwarded.insert(0, "--json")
        return {"command": command, "returncode": int(_await_if_needed(function(forwarded)) or 0)}
    raise PipelineError(f"unknown command: {command}")


def main(argv: Sequence[str] | None = None, *, pipeline: ACEPipeline | None = None) -> int:
    if os.environ.get("ACE_LLM_CHILD") == "1":
        print(json.dumps({"error": "recursive_ace_invocation", "status": "failed"}))
        return 2
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    delegated_commands = {"migration", "schema", "schedule", "knowledge"}
    # Delegated CLIs own their option grammar.  Parsing them as an argparse
    # REMAINDER subcommand rejects options placed immediately after the
    # command (``ace schedule --platform ...``).  Hand the complete suffix to
    # the delegate and keep the normal parser for pipeline-owned commands.
    if raw_argv and raw_argv[0] in delegated_commands:
        args = SimpleNamespace(command=raw_argv[0], args=raw_argv[1:], json_flag=False)
    else:
        args = build_parser().parse_args(raw_argv)
    runner = pipeline or ACEPipeline()
    try:
        result = dispatch(args, runner)
        _print_result(result)
        return _result_exit_code(result)
    except PipelineBusyError as error:
        print(json.dumps({"error": type(error).__name__, "status": "busy"}, sort_keys=True))
        return 2
    except (PipelineError, OSError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "status": "failed"}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through the wrapper
    raise SystemExit(main())
