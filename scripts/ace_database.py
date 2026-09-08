"""Bounded transport for the ACE Supabase schema.

The store deliberately has a small fixed SQL surface.  It sends SQL through
the Agent Supabase wrapper on stdin, so neither envelopes nor SQL appear in a
process argument list.  The wrapper resolves credentials; this module never
reads credentials, calls Postgres directly, or talks HTTP to Supabase.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "ace"
DEFAULT_PROFILE = "amastuces"
DEFAULT_WRAPPER = "/Users/franck/.agents/bin/supabase"
DEFAULT_TIMEOUT_SECONDS = 90.0
MAX_TIMEOUT_SECONDS = 120.0
MAX_MESSAGES = 10_000
MAX_ATTACHMENTS = 1_000
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_CONTENT_BYTES = 1024 * 1024
MAX_REFERENCE_BYTES = 256 * 1024
MAX_ANALYSIS_BYTES = 2 * 1024 * 1024
MAX_COMPILED_BYTES = 8 * 1024 * 1024
MAX_SOURCE = 64
MAX_SESSION_ID = 512
MAX_REVISION = 64
MAX_HOST_ID = 256
MAX_LEASE_OWNER = 256
DEFAULT_STAGE_LEASE_SECONDS = 1_800
MAX_STAGE_LEASE_SECONDS = 86_400
_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STAGES = {"extraction", "analysis", "result", "decision", "evaluation", "compile"}
_STATUSES = {"pending", "running", "succeeded", "failed", "skipped"}
_FUNCTION_ROLES = {
    "register_project": "ace_ingest", "ingest_snapshot": "ace_ingest",
    "list_projects": "ace_reader", "search_history": "ace_reader",
    "read_compiled_snapshot": "ace_reader",
    **{name: "ace_processor" for name in (
        "pending_snapshots", "pending_snapshots_since", "pending_snapshots_window",
        "pending_snapshot_refs", "pending_snapshot_refs_since", "pending_snapshot_refs_window",
        "snapshot_delta", "claim_stage", "release_stage", "expire_stage_leases", "mark_processed", "mark_stage", "save_analysis", "save_result",
        "save_decision", "save_correction", "save_evaluation", "publish_compiled_snapshot",
    )},
}


class SupabaseStoreError(RuntimeError):
    """A safe, non-sensitive wrapper/transport failure."""


class EnvelopeValidationError(ValueError):
    """The caller supplied an envelope outside the transport contract."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EnvelopeValidationError("value is not JSON serializable") from exc


def _json_size(value: Any, *, maximum: int, label: str) -> None:
    if len(_json_bytes(value)) > maximum:
        raise EnvelopeValidationError(f"{label} exceeds its bounded size")


def _require_sanitized(value: Any, label: str) -> None:
    """Reject payloads that the canonical transcript filter would change.

    The source adapters own sanitisation and attachment extraction.  The
    persistence boundary must not silently rewrite an already hashed
    envelope, so it compares the value with the canonical filtered result and
    fails closed when the caller skipped that adapter.
    """

    try:
        from ace_transcripts import _AttachmentCollector, _clean_content

        cleaned = _clean_content(
            value,
            collector=_AttachmentCollector("<ace-transport>"),
            source_line=0,
        )
    except Exception as exc:
        raise EnvelopeValidationError(f"{label} could not be checked for sanitisation") from exc
    if cleaned != value:
        raise EnvelopeValidationError(f"{label} must be sanitised before transport")


def _string(value: Any, field: str, maximum: int, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise EnvelopeValidationError(f"{field} is required")
        return None
    if not isinstance(value, str):
        raise EnvelopeValidationError(f"{field} must be a string")
    if required and not value.strip():
        raise EnvelopeValidationError(f"{field} is required")
    if len(value) > maximum:
        raise EnvelopeValidationError(f"{field} exceeds its bound")
    return value


def _uuid_text(value: Any, field: str) -> str:
    if isinstance(value, uuid.UUID):
        return str(value)
    text = _string(value, field, 36)
    assert text is not None
    try:
        return str(uuid.UUID(text))
    except (ValueError, AttributeError) as exc:
        raise EnvelopeValidationError(f"{field} must be a UUID") from exc


def _timestamp(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = _string(value, field, 80)
    assert text is not None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EnvelopeValidationError(f"{field} must be an ISO timestamp") from exc
    return text


def _instant(value: Any, field: str) -> str | None:
    """Return a UTC ISO timestamp for server-side comparisons.

    The pipeline's automation cutoff is a Unix timestamp, while callers may
    provide an ISO value or ``datetime``.  Normalising all three forms to an
    explicit UTC offset prevents the database session timezone from changing
    which source revisions are eligible.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be a finite timestamp")
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise ValueError(f"{field} is outside the timestamp bound") from exc
        return parsed.isoformat()
    if isinstance(value, datetime):
        parsed = value
    else:
        text = _timestamp(value, field)
        if text is None:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:  # pragma: no cover - guarded by _timestamp
            raise ValueError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _lease_seconds(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("lease_seconds must be an integer")
    if not 1 <= value <= MAX_STAGE_LEASE_SECONDS:
        raise ValueError(
            f"lease_seconds must be between 1 and {MAX_STAGE_LEASE_SECONDS}"
        )
    return value


def _lease_owner(value: Any) -> str:
    text = _string(value, "lease_owner", MAX_LEASE_OWNER)
    assert text is not None
    return text


def _host_id(value: Any) -> str:
    text = _string(value, "host_id", MAX_HOST_ID)
    assert text is not None
    return text


def _source(value: Any) -> str:
    text = _string(value, "source", MAX_SOURCE)
    assert text is not None
    text = text.strip().lower()
    if not _SOURCE_RE.fullmatch(text):
        raise EnvelopeValidationError("source has an invalid format")
    return text


def _revision(value: Any) -> str:
    text = _string(value, "revision", MAX_REVISION)
    assert text is not None
    text = text.strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise EnvelopeValidationError("revision must be a SHA-256 hex digest")
    return text


def _project_descriptor(project: Any, *, defaults_enabled: bool = True) -> dict[str, Any]:
    if not isinstance(project, Mapping):
        if hasattr(project, "as_dict") and callable(project.as_dict):
            project = project.as_dict()
        elif hasattr(project, "__dict__"):
            project = vars(project)
        else:
            raise EnvelopeValidationError("project must be an object")
    descriptor = {
        "id": _uuid_text(project.get("id"), "project.id"),
        "name": _string(project.get("name"), "project.name", 200),
        "root": _string(project.get("root"), "project.root", 2048),
        "vault_dir": _string(project.get("vault_dir"), "project.vault_dir", 2048),
    }
    # The booleans are registration controls, not part of a shared snapshot.
    enabled = project.get("enabled", defaults_enabled)
    initialized = project.get("initialized", project.get("init_opt_in", defaults_enabled))
    if not isinstance(enabled, bool) or not isinstance(initialized, bool):
        raise EnvelopeValidationError("project registration flags must be booleans")
    descriptor["enabled"] = enabled
    descriptor["initialized"] = initialized
    return descriptor


def _normalise_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise EnvelopeValidationError("messages must contain objects")
    message_id = _string(message.get("id"), "message.id", 512)
    role = _string(message.get("role"), "message.role", 64)
    message_type = _string(message.get("type"), "message.type", 64)
    ordinal = message.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal <= 1_000_000:
        raise EnvelopeValidationError("message.ordinal is outside its bound")
    if "content" not in message:
        raise EnvelopeValidationError("message.content is required")
    content = message.get("content")
    _json_size(content, maximum=MAX_MESSAGE_CONTENT_BYTES, label="message.content")
    _require_sanitized(content, "message.content")
    normalized: dict[str, Any] = {
        "id": message_id,
        "ordinal": ordinal,
        "role": role,
        "type": message_type,
        "content": content,
    }
    timestamp = _timestamp(message.get("timestamp"), "message.timestamp")
    if timestamp is not None:
        normalized["timestamp"] = timestamp
    for key, maximum in (("call_id", 512), ("status", 128), ("model", 256)):
        value = message.get(key)
        if value is not None:
            normalized[key] = _string(value, f"message.{key}", maximum)
    if "refs" in message and message.get("refs") is not None:
        _json_size(message["refs"], maximum=MAX_REFERENCE_BYTES, label="message.refs")
        _require_sanitized(message["refs"], "message.refs")
        normalized["refs"] = message["refs"]
    return normalized


def _normalise_attachment(attachment: Any) -> dict[str, Any]:
    if not isinstance(attachment, Mapping):
        raise EnvelopeValidationError("attachments must contain objects")
    normalized: dict[str, Any] = {}
    aliases = {
        "mime_type": ("mime_type", "media_type"),
        "size": ("size", "bytes"),
        "uri": ("uri", "source_path"),
    }
    for key, maximum in (("id", 512), ("name", 512), ("mime_type", 256), ("uri", 4096), ("kind", 128)):
        value = next((attachment.get(alias) for alias in aliases.get(key, (key,)) if attachment.get(alias) is not None), None)
        if value is not None:
            normalized[key] = _string(value, f"attachment.{key}", maximum)
    size = next((attachment.get(alias) for alias in aliases["size"] if attachment.get(alias) is not None), None)
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 1_073_741_824:
            raise EnvelopeValidationError("attachment.size is outside its bound")
        normalized["size"] = size
    sha256 = attachment.get("sha256")
    if sha256 is not None:
        sha256_text = _string(sha256, "attachment.sha256", 64)
        assert sha256_text is not None
        sha256_text = sha256_text.lower()
        if not _SHA256_RE.fullmatch(sha256_text):
            raise EnvelopeValidationError("attachment.sha256 must be a SHA-256 hex digest")
        normalized["sha256"] = sha256_text
    metadata = attachment.get("metadata")
    if metadata is None:
        metadata = {}
        if attachment.get("source_line") is not None:
            source_line = attachment.get("source_line")
            if isinstance(source_line, bool) or not isinstance(source_line, int) or source_line < 0:
                raise EnvelopeValidationError("attachment.source_line is outside its bound")
            metadata["source_line"] = source_line
    if not isinstance(metadata, Mapping):
        raise EnvelopeValidationError("attachment.metadata must be an object")
    _json_size(metadata, maximum=MAX_REFERENCE_BYTES, label="attachment.metadata")
    _require_sanitized(metadata, "attachment.metadata")
    normalized["metadata"] = metadata
    return normalized


def normalize_envelope(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and keep only the stable, normalized snapshot contract."""

    if not isinstance(envelope, Mapping):
        raise EnvelopeValidationError("snapshot envelope must be an object")
    schema_version = envelope.get("schema_version")
    if schema_version != 1:
        raise EnvelopeValidationError("schema_version must be 1")
    project = _project_descriptor(envelope.get("project"), defaults_enabled=True)
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "project": {
            "id": project["id"],
            "name": project["name"],
            "root": project["root"],
            "vault_dir": project["vault_dir"],
        },
        "source": _source(envelope.get("source")),
        "session_id": _string(envelope.get("session_id"), "session_id", MAX_SESSION_ID),
        "revision": _revision(envelope.get("revision")),
        "messages": [],
        "attachments": [],
    }
    for key, maximum in (("source_path", 2048), ("host_id", 256)):
        value = envelope.get(key)
        if value is not None:
            normalized[key] = _string(value, key, maximum)
    for key in ("started_at", "updated_at"):
        value = _timestamp(envelope.get(key), key)
        if value is not None:
            normalized[key] = value
    messages = envelope.get("messages", [])
    attachments = envelope.get("attachments", [])
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise EnvelopeValidationError("messages must be an array")
    if not isinstance(attachments, Sequence) or isinstance(attachments, (str, bytes, bytearray)):
        raise EnvelopeValidationError("attachments must be an array")
    if len(messages) > MAX_MESSAGES or len(attachments) > MAX_ATTACHMENTS:
        raise EnvelopeValidationError("snapshot collection bounds exceeded")
    normalized["messages"] = [_normalise_message(item) for item in messages]
    normalized["attachments"] = [_normalise_attachment(item) for item in attachments]
    _json_size(normalized, maximum=MAX_ENVELOPE_BYTES, label="snapshot envelope")
    return normalized


def _sql_text(value: str) -> str:
    """Return a SQL string literal; the SQL itself is sent only on stdin."""

    return "'" + value.replace("'", "''") + "'"


def _sql_json(value: Any) -> str:
    return _sql_text(_json_bytes(value).decode("utf-8")) + "::jsonb"


def _first_row(rows: list[Any], *, operation: str) -> dict[str, Any]:
    if not rows:
        raise SupabaseStoreError(f"Supabase operation returned no row: {operation}")
    row = rows[0]
    if not isinstance(row, Mapping):
        raise SupabaseStoreError(f"Supabase operation returned an invalid row: {operation}")
    return dict(row)


class SupabaseStore:
    """Small, bounded ACE persistence facade over the Agent Supabase wrapper."""

    def __init__(
        self,
        profile: str = DEFAULT_PROFILE,
        wrapper: str | Path = DEFAULT_WRAPPER,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(profile, str) or not profile.strip() or len(profile) > 64:
            raise ValueError("profile must be a bounded name")
        if not isinstance(wrapper, (str, Path)) or not str(wrapper):
            raise ValueError("wrapper must be a path")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be numeric") from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise ValueError("timeout must be positive")
        self.profile = profile.strip()
        self.wrapper = str(wrapper)
        self.timeout = min(timeout_value, MAX_TIMEOUT_SECONDS)

    def _rows(self, sql: str) -> list[Any]:
        """Run one fixed query through the wrapper, without exposing details."""
        function = re.match(r"^SELECT \* FROM ace\.([a-z_]+)\(", sql)
        role = _FUNCTION_ROLES.get(function.group(1)) if function else None
        if role:
            # The existing wrapper owns the connection credential. Restrict
            # each ACE operation to its granted role for this transaction.
            sql = f"BEGIN; SET LOCAL ROLE {role}; {sql.rstrip(';')}; COMMIT;"
        try:
            completed = subprocess.run(
                [self.wrapper, "--profile", self.profile, "sql", "exec"],
                input=sql,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SupabaseStoreError("Supabase operation timed out") from exc
        except OSError as exc:
            raise SupabaseStoreError("Supabase wrapper could not be executed") from exc
        if completed.returncode != 0:
            raise SupabaseStoreError("Supabase operation failed")
        try:
            response = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SupabaseStoreError("Supabase wrapper returned invalid JSON") from exc
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise SupabaseStoreError("Supabase wrapper returned an error")
        data = response.get("data", [])
        if data is None:
            return []
        if isinstance(data, list):
            if data and all(isinstance(result, list) for result in data):
                return [row for result in data for row in result]
            return data
        if isinstance(data, Mapping) and isinstance(data.get("rows"), list):
            return list(data["rows"])
        if isinstance(data, Mapping):
            return [dict(data)]
        raise SupabaseStoreError("Supabase wrapper returned an invalid data shape")

    def register_project(self, project: Mapping[str, Any]) -> dict[str, Any]:
        """Explicitly register and opt a project into ACE ingestion."""

        descriptor = _project_descriptor(project, defaults_enabled=True)
        sql = (
            "SELECT * FROM ace.register_project("
            f"{_sql_text(descriptor['id'])}::uuid, "
            f"{_sql_text(descriptor['name'])}, "
            f"{_sql_text(descriptor['root'])}, "
            f"{_sql_text(descriptor['vault_dir'])}, "
            f"{str(descriptor['enabled']).lower()}, "
            f"{str(descriptor['initialized']).lower()})"
        )
        row = _first_row(self._rows(sql), operation="register_project")
        return {
            "id": str(row.get("id", descriptor["id"])),
            "name": row.get("name", descriptor["name"]),
            "root": row.get("root", descriptor["root"]),
            "vault_dir": row.get("vault_dir", descriptor["vault_dir"]),
            "enabled": bool(row.get("enabled", descriptor["enabled"])),
        }

    def ingest_snapshot(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically ingest a normalized snapshot and return its receipt."""

        normalized = normalize_envelope(envelope)
        sql = f"SELECT * FROM ace.ingest_snapshot({_sql_json(normalized)})"
        receipt = _first_row(self._rows(sql), operation="ingest_snapshot")
        if receipt.get("status") != "accepted":
            raise SupabaseStoreError("Supabase did not accept the snapshot")
        if not isinstance(receipt.get("inserted"), bool):
            raise SupabaseStoreError("Supabase returned an invalid insertion receipt")
        expected_identity = {
            "project_id": normalized["project"]["id"],
            "source": normalized["source"],
            "session_id": normalized["session_id"],
            "revision": normalized["revision"],
        }
        if any(receipt.get(key) != expected for key, expected in expected_identity.items()):
            raise SupabaseStoreError("Supabase returned a mismatched snapshot receipt")
        return receipt

    def pending_snapshots(
        self,
        limit: int = 100,
        stage: str = "extraction",
        project_id: str | uuid.UUID | None = None,
        minimum_started_at: Any = None,
        source_after: Any = None,
        source_before: Any = None,
    ) -> list[dict[str, Any]]:
        """Return latest, not-yet-succeeded revisions for one processing stage."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if stage not in _STAGES:
            raise ValueError("stage is not supported")
        if source_after is None:
            source_after = minimum_started_at
        after = _instant(source_after, "source_after")
        before = _instant(source_before, "source_before")
        if after is not None and before is not None and after > before:
            raise ValueError("source_after must not be later than source_before")
        if after is None and before is None:
            if project_id is None:
                sql = f"SELECT * FROM ace.pending_snapshots({limit}, {_sql_text(stage)})"
            else:
                project_text = _uuid_text(project_id, "project_id")
                sql = f"SELECT * FROM ace.pending_snapshots({limit}, {_sql_text(stage)}, {_sql_text(project_text)}::uuid)"
        else:
            project_sql = "NULL::uuid" if project_id is None else f"{_sql_text(_uuid_text(project_id, 'project_id'))}::uuid"
            sql = (
                "SELECT * FROM ace.pending_snapshots_window("
                f"{limit}, {_sql_text(stage)}, {project_sql}, "
                f"{_sql_text(after) + '::timestamptz' if after is not None else 'NULL::timestamptz'}, "
                f"{_sql_text(before) + '::timestamptz' if before is not None else 'NULL::timestamptz'})"
            )
        pending: list[dict[str, Any]] = []
        for row in self._rows(sql):
            if not isinstance(row, Mapping):
                raise SupabaseStoreError("Supabase returned an invalid snapshot row")
            envelope = row.get("envelope", row.get("snapshot"))
            if not isinstance(envelope, Mapping):
                # A mocked or older function may return the envelope directly.
                envelope = row
            pending.append(normalize_envelope(envelope))
        return pending

    def pending_snapshot_refs(
        self,
        limit: int = 100,
        stage: str = "extraction",
        project_id: str | uuid.UUID | None = None,
        minimum_started_at: Any = None,
        source_after: Any = None,
        source_before: Any = None,
    ) -> list[dict[str, Any]]:
        """Return small pending identities without transferring transcripts."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        if stage not in _STAGES:
            raise ValueError("stage is not supported")
        project_sql = "NULL::uuid" if project_id is None else f"{_sql_text(_uuid_text(project_id, 'project_id'))}::uuid"
        if source_after is None:
            source_after = minimum_started_at
        after = _instant(source_after, "source_after")
        before = _instant(source_before, "source_before")
        if after is not None and before is not None and after > before:
            raise ValueError("source_after must not be later than source_before")
        if after is None and before is None:
            sql = (
                "SELECT * FROM ace.pending_snapshot_refs("
                f"{limit}, {_sql_text(stage)}, {project_sql})"
            )
        else:
            sql = (
                "SELECT * FROM ace.pending_snapshot_refs_window("
                f"{limit}, {_sql_text(stage)}, {project_sql}, "
                f"{_sql_text(after) + '::timestamptz' if after is not None else 'NULL::timestamptz'}, "
                f"{_sql_text(before) + '::timestamptz' if before is not None else 'NULL::timestamptz'})"
            )
        return [dict(row) for row in self._rows(sql) if isinstance(row, Mapping)]

    def snapshot_delta(
        self,
        project_id: str | uuid.UUID,
        source: str,
        session_id: str,
        revision: str,
        last_ordinal: int = -1,
    ) -> dict[str, Any] | None:
        """Read one bounded message delta from an already selected revision."""

        project_text = _uuid_text(project_id, "project_id")
        source_text = _source(source)
        session_text = _string(session_id, "session_id", MAX_SESSION_ID)
        assert session_text is not None
        revision_text = _revision(revision)
        if isinstance(last_ordinal, bool) or not isinstance(last_ordinal, int) or last_ordinal < -1:
            raise ValueError("last_ordinal must be an integer >= -1")
        sql = (
            "SELECT * FROM ace.snapshot_delta("
            f"{_sql_text(project_text)}::uuid, {_sql_text(source_text)}, "
            f"{_sql_text(session_text)}, {_sql_text(revision_text)}, {last_ordinal})"
        )
        rows = self._rows(sql)
        if not rows:
            return None
        row = _first_row(rows, operation="snapshot_delta")
        envelope = row.get("envelope", row.get("snapshot"))
        if not isinstance(envelope, Mapping):
            envelope = row
        return normalize_envelope(envelope)

    def snapshot_deltas(self, requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Read several bounded message deltas through one wrapper process.

        The native service used to call ``snapshot_delta`` once per selected
        conversation.  Each call starts the Supabase wrapper, which resolves
        the same Bitwarden-backed connection again and is deliberately rate
        limited.  A manual daily audit could therefore spend minutes waiting
        before it reached Luna.  Keep the SQL function and proof boundary the
        same, but send one bounded transaction containing the selected deltas.
        """

        if isinstance(requests, (str, bytes, bytearray)) or not isinstance(requests, Sequence):
            raise ValueError("requests must be an array")
        bounded = list(requests)
        if len(bounded) > 64:
            raise ValueError("requests must contain at most 64 deltas")
        statements: list[str] = []
        for request in bounded:
            if not isinstance(request, Mapping):
                raise ValueError("each delta request must be an object")
            project_text = _uuid_text(request.get("project_id"), "project_id")
            source_text = _source(request.get("source"))
            session_text = _string(request.get("session_id"), "session_id", MAX_SESSION_ID)
            assert session_text is not None
            revision_text = _revision(request.get("revision"))
            last_ordinal = request.get("last_ordinal", -1)
            if isinstance(last_ordinal, bool) or not isinstance(last_ordinal, int) or last_ordinal < -1:
                raise ValueError("last_ordinal must be an integer >= -1")
            statements.append(
                "SELECT * FROM ace.snapshot_delta("
                f"{_sql_text(project_text)}::uuid, {_sql_text(source_text)}, "
                f"{_sql_text(session_text)}, {_sql_text(revision_text)}, {last_ordinal})"
            )
        if not statements:
            return []
        rows = self._rows("; ".join(statements))
        output: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise SupabaseStoreError("Supabase returned an invalid snapshot delta row")
            envelope = row.get("envelope", row.get("snapshot"))
            if not isinstance(envelope, Mapping):
                envelope = row
            output.append(normalize_envelope(envelope))
        return output

    def claim_stage(
        self,
        project_id: str | uuid.UUID,
        host_id: str,
        session_id: str,
        revision: str,
        *,
        source: str,
        stage: str,
        lease_owner: str,
        lease_seconds: int = DEFAULT_STAGE_LEASE_SECONDS,
    ) -> dict[str, Any]:
        """Atomically claim one processing stage for a bounded lease.

        The project and source are part of the SQL predicate, and the
        database's unique stage key serialises competing hosts.  Repeating a
        claim with the same owner and host renews the same lease; another
        owner receives ``claimed=false`` without learning the current owner.
        """

        project_text = _uuid_text(project_id, "project_id")
        host_text = _host_id(host_id)
        source_text = _source(source)
        session_text = _string(session_id, "session_id", MAX_SESSION_ID)
        assert session_text is not None
        revision_text = _revision(revision)
        if stage not in _STAGES:
            raise ValueError("stage is not supported")
        owner_text = _lease_owner(lease_owner)
        seconds = _lease_seconds(lease_seconds)
        sql = (
            "SELECT * FROM ace.claim_stage("
            f"{_sql_text(project_text)}::uuid, {_sql_text(source_text)}, "
            f"{_sql_text(session_text)}, {_sql_text(revision_text)}, {_sql_text(stage)}, "
            f"{_sql_text(owner_text)}, {_sql_text(host_text)}, {seconds})"
        )
        row = _first_row(self._rows(sql), operation="claim_stage")
        if not isinstance(row.get("claimed"), bool):
            raise SupabaseStoreError("Supabase returned an invalid stage claim")
        if row["claimed"]:
            expected = {
                "project_id": project_text,
                "source": source_text,
                "session_id": session_text,
                "revision": revision_text,
                "stage": stage,
                "lease_owner": owner_text,
                "host_id": host_text,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise SupabaseStoreError("Supabase returned a mismatched stage claim")
        return row

    def release_stage(
        self,
        project_id: str | uuid.UUID,
        host_id: str,
        session_id: str,
        revision: str,
        *,
        source: str,
        stage: str,
        lease_owner: str,
        outcome: str = "failed",
    ) -> dict[str, Any]:
        """Release a lease only when its owner and host still match."""

        project_text = _uuid_text(project_id, "project_id")
        host_text = _host_id(host_id)
        source_text = _source(source)
        session_text = _string(session_id, "session_id", MAX_SESSION_ID)
        assert session_text is not None
        revision_text = _revision(revision)
        if stage not in _STAGES:
            raise ValueError("stage is not supported")
        owner_text = _lease_owner(lease_owner)
        outcome_text = _string(outcome, "outcome", 64)
        assert outcome_text is not None
        sql = (
            "SELECT * FROM ace.release_stage("
            f"{_sql_text(project_text)}::uuid, {_sql_text(source_text)}, "
            f"{_sql_text(session_text)}, {_sql_text(revision_text)}, {_sql_text(stage)}, "
            f"{_sql_text(owner_text)}, {_sql_text(host_text)}, {_sql_text(outcome_text)})"
        )
        row = _first_row(self._rows(sql), operation="release_stage")
        if not isinstance(row.get("released"), bool):
            raise SupabaseStoreError("Supabase returned an invalid stage release")
        return row

    def expire_stage_leases(self, limit: int = 500) -> dict[str, Any]:
        """Reclaim expired stage leases in one bounded database call."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        sql = f"SELECT * FROM ace.expire_stage_leases({limit})"
        row = _first_row(self._rows(sql), operation="expire_stage_leases")
        expired = row.get("expired_count", row.get("expired"))
        if isinstance(expired, bool) or not isinstance(expired, int) or expired < 0:
            raise SupabaseStoreError("Supabase returned an invalid expired-lease count")
        return {**row, "expired_count": expired}

    def mark_stage(
        self,
        source: str,
        session_id: str,
        revision: str,
        project_id: str | uuid.UUID,
        stage: str,
        status: str,
        error: str | None = None,
        *,
        lease_owner: str,
        host_id: str,
    ) -> dict[str, Any]:
        """Record an outcome only while the caller owns an unexpired lease."""

        source_text = _source(source)
        session_text = _string(session_id, "session_id", MAX_SESSION_ID)
        assert session_text is not None
        revision_text = _revision(revision)
        project_text = _uuid_text(project_id, "project_id")
        host_text = _host_id(host_id)
        owner_text = _lease_owner(lease_owner)
        if stage not in _STAGES:
            raise ValueError("stage is not supported")
        if status not in _STATUSES:
            raise ValueError("status is not supported")
        if error is not None:
            error = _string(error, "error", 256, required=False)
        sql = (
            "SELECT * FROM ace.mark_stage("
            f"{_sql_text(source_text)}, {_sql_text(session_text)}, {_sql_text(revision_text)}, "
            f"{_sql_text(project_text)}::uuid, {_sql_text(stage)}, {_sql_text(status)}, "
            f"{_sql_text(owner_text)}, {_sql_text(host_text)}, "
            f"{_sql_text(error) if error is not None else 'NULL'})"
        )
        return _first_row(self._rows(sql), operation="mark_stage")

    def mark_processed(
        self,
        source: str,
        session_id: str,
        revision: str,
        project_id: str | uuid.UUID,
        stage: str,
        status: str,
        error: str | None = None,
        *,
        lease_owner: str | None = None,
        host_id: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility name for the lease-bound stage acknowledgement."""

        if lease_owner is None or host_id is None:
            raise ValueError("lease_owner and host_id are required for stage acknowledgement")
        return self.mark_stage(
            source,
            session_id,
            revision,
            project_id,
            stage,
            status,
            error,
            lease_owner=lease_owner,
            host_id=host_id,
        )

    @staticmethod
    def _context_payload(
        args: tuple[Any, ...], kwargs: dict[str, Any], payload_name: str
    ) -> tuple[str, str, str, str, Any]:
        """Accept envelope-first and explicit context forms for learning code."""

        values = dict(kwargs)
        payload = values.pop(payload_name, values.pop("payload", None))
        envelope = values.pop("envelope", None)
        positional = list(args)
        target = positional.pop(0) if positional else envelope

        if (
            isinstance(target, Mapping)
            and "project" in target
            and isinstance(target.get("project"), Mapping)
            and "schema_version" in target
        ):
            normalized = normalize_envelope(target)
            project_id = normalized["project"]["id"]
            source = normalized["source"]
            session_id = normalized["session_id"]
            revision = normalized["revision"]
            if positional and payload is None:
                payload = positional.pop(0)
        elif isinstance(target, Mapping) and target.get("project_id") is not None:
            # ``ace_learning._analysis_row`` is a deliberately small model
            # result rather than a full envelope.  Keep its explicit context
            # fields while storing the row itself (unless a separate payload
            # was supplied).
            project_id = _uuid_text(target.get("project_id"), "project_id")
            source = _source(target.get("source"))
            session_text = _string(target.get("session_id"), "session_id", MAX_SESSION_ID)
            assert session_text is not None
            session_id = session_text
            revision = _revision(target.get("revision"))
            if payload is None:
                payload = target
            if positional and payload is None:
                payload = positional.pop(0)
        else:
            project_id = target if target is not None else values.pop("project_id", None)
            # ``save_analysis(project_id, payload, source=..., ...)`` is a
            # useful short form used by learning code.  Explicit context
            # keywords disambiguate the second positional value as payload.
            if (
                positional
                and payload is None
                and {"source", "session_id", "revision"}.issubset(values)
            ):
                payload = positional.pop(0)
            if positional and "source" not in values:
                values["source"] = positional.pop(0)
            if positional and "session_id" not in values:
                values["session_id"] = positional.pop(0)
            if positional and "revision" not in values:
                values["revision"] = positional.pop(0)
            if positional and payload is None:
                payload = positional.pop(0)
            project_id = _uuid_text(project_id, "project_id")
            source = _source(values.pop("source", None))
            session_id_value = _string(values.pop("session_id", None), "session_id", MAX_SESSION_ID)
            assert session_id_value is not None
            session_id = session_id_value
            revision = _revision(values.pop("revision", None))

        if positional or values:
            raise TypeError("unexpected persistence arguments")
        if payload is None:
            raise ValueError(f"{payload_name} is required")
        _json_size(payload, maximum=MAX_ANALYSIS_BYTES, label=payload_name)
        return project_id, source, session_id, revision, payload

    def save_analysis(self, target: Any = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
        project_id, source, session_id, revision, analysis = self._context_payload(
            (target, *args), kwargs, "analysis"
        )
        sql = (
            "SELECT * FROM ace.save_analysis("
            f"{_sql_text(project_id)}::uuid, {_sql_text(source)}, {_sql_text(session_id)}, "
            f"{_sql_text(revision)}, {_sql_json(analysis)})"
        )
        return _first_row(self._rows(sql), operation="save_analysis")

    def save_result(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        project_id, source, session_id, revision, result = self._context_payload(
            args, kwargs, "result"
        )
        sql = (
            "SELECT * FROM ace.save_result("
            f"{_sql_text(project_id)}::uuid, {_sql_text(source)}, {_sql_text(session_id)}, "
            f"{_sql_text(revision)}, {_sql_json(result)})"
        )
        return _first_row(self._rows(sql), operation="save_result")

    def save_decision(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        actor = kwargs.pop("actor", None)
        project_id, source, session_id, revision, decision = self._context_payload(
            args, kwargs, "decision"
        )
        actor_text = _string(actor, "actor", 256, required=False) if actor is not None else None
        sql = (
            "SELECT * FROM ace.save_decision("
            f"{_sql_text(project_id)}::uuid, {_sql_text(source)}, {_sql_text(session_id)}, "
            f"{_sql_text(revision)}, {_sql_json(decision)}, "
            f"{_sql_text(actor_text) if actor_text is not None else 'NULL'})"
        )
        return _first_row(self._rows(sql), operation="save_decision")

    def save_correction(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        actor = kwargs.pop("actor", None)
        project_id, source, session_id, revision, correction = self._context_payload(
            args, kwargs, "correction"
        )
        actor_text = _string(actor, "actor", 256, required=False) if actor is not None else None
        sql = (
            "SELECT * FROM ace.save_correction("
            f"{_sql_text(project_id)}::uuid, {_sql_text(source)}, {_sql_text(session_id)}, "
            f"{_sql_text(revision)}, {_sql_json(correction)}, "
            f"{_sql_text(actor_text) if actor_text is not None else 'NULL'})"
        )
        return _first_row(self._rows(sql), operation="save_correction")

    def save_evaluation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        score = kwargs.pop("score", None)
        project_id, source, session_id, revision, evaluation = self._context_payload(
            args, kwargs, "evaluation"
        )
        score_sql = "NULL"
        if score is not None:
            try:
                score_value = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError("score must be numeric") from exc
            if not math.isfinite(score_value):
                raise ValueError("score must be finite")
            score_sql = format(score_value, ".10g")
        sql = (
            "SELECT * FROM ace.save_evaluation("
            f"{_sql_text(project_id)}::uuid, {_sql_text(source)}, {_sql_text(session_id)}, "
            f"{_sql_text(revision)}, {_sql_json(evaluation)}, {score_sql})"
        )
        return _first_row(self._rows(sql), operation="save_evaluation")

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM ace.list_projects()")
        projects: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise SupabaseStoreError("Supabase returned an invalid project row")
            projects.append(
                {
                    "id": str(row.get("id")),
                    "name": row.get("name"),
                    "root": row.get("root"),
                    "vault_dir": row.get("vault_dir"),
                    "enabled": bool(row.get("enabled", False)),
                }
            )
        return projects

    def search_history(
        self, project_id: str | uuid.UUID, query: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        project_text = _uuid_text(project_id, "project_id")
        query_text = _string(query, "query", 200)
        assert query_text is not None
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        sql = (
            "SELECT * FROM ace.search_history("
            f"{_sql_text(project_text)}::uuid, {_sql_text(query_text)}, {limit})"
        )
        rows = self._rows(sql)
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def publish_compiled_snapshot(
        self,
        project_id: str | uuid.UUID | Mapping[str, Any],
        version: int,
        snapshot: Any,
    ) -> dict[str, Any]:
        if isinstance(project_id, Mapping):
            project_value = project_id.get("project")
            if isinstance(project_value, Mapping):
                project_id = project_value.get("id")
            else:
                project_id = project_id.get("id")
        project_text = _uuid_text(project_id, "project_id")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("version must be a positive integer")
        _json_size(snapshot, maximum=MAX_COMPILED_BYTES, label="compiled snapshot")
        checksum = hashlib.sha256(_json_bytes(snapshot)).hexdigest()
        sql = (
            "SELECT * FROM ace.publish_compiled_snapshot("
            f"{_sql_text(project_text)}::uuid, {version}, {_sql_json(snapshot)}, "
            f"{_sql_text(checksum)})"
        )
        return _first_row(self._rows(sql), operation="publish_compiled_snapshot")

    def read_compiled_snapshot(
        self, project_id: str | uuid.UUID, version: int | None = None
    ) -> dict[str, Any] | None:
        project_text = _uuid_text(project_id, "project_id")
        if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
            raise ValueError("version must be a positive integer")
        version_sql = str(version) if version is not None else "NULL"
        sql = (
            "SELECT * FROM ace.read_compiled_snapshot("
            f"{_sql_text(project_text)}::uuid, {version_sql})"
        )
        rows = self._rows(sql)
        if not rows:
            return None
        row = _first_row(rows, operation="read_compiled_snapshot")
        return row


__all__ = [
    "EnvelopeValidationError",
    "SupabaseStore",
    "SupabaseStoreError",
    "normalize_envelope",
]
