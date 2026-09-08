"""Read-only transcript adapters for the ACE ingestion boundary.

The adapters produce the shared JSON envelope directly.  They do not call a
model, mutate source files, or truncate tool arguments/results.  Content is
sanitised recursively before the revision is calculated; images and encoded
binary values become attachment references rather than entering the payload.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from checkpoint_cursor import bounded_redacted_text

try:  # The project module is intentionally optional for standalone adapters.
    from ace_projects import ProjectInfo, deterministic_project_id
except ImportError:  # pragma: no cover - direct package execution fallback
    ProjectInfo = Any  # type: ignore[misc,assignment]

    def deterministic_project_id(root: str | Path) -> str:
        return str(hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:32])


try:
    from utils import redact_sensitive_text, sensitive_text_findings
except ImportError:  # pragma: no cover - a small safe fallback for direct use
    _FALLBACK_SECRET = re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|cookie)"
        r"\s*[:=]\s*([\"']?)([^\s,;&\"']+)\1"
    )

    def redact_sensitive_text(value: str) -> str:
        return _FALLBACK_SECRET.sub(lambda match: f"<REDACTED>", value)

    def sensitive_text_findings(value: str) -> dict[str, int]:
        return {"candidate": 1} if _FALLBACK_SECRET.search(value) else {}


SCHEMA_VERSION = 1
_KNOWN_SOURCES = {"codex", "claude", "hermes"}
_HIDDEN_BLOCK_TYPES = {"analysis", "thinking", "reasoning", "redacted_thinking"}
_SECRET_KEY = re.compile(
    r"(?i)(?:^|[_-])(?:authorization|proxy-authorization|api[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|client[_-]?secret|password|passwd|"
    r"secret|token|cookie|set-cookie)(?:$|[_-])"
)
_BASE64_CANDIDATE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_TOKEN_CANDIDATE = re.compile(
    r"(?i)(?:bearer\s+[A-Za-z0-9._~+/=-]{8,}|sk-(?:live-|test-|proj-)?[A-Za-z0-9_-]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[A-Z0-9]{16})"
)
_SAFE_TELEMETRY_KEYS = {
    "last_token_usage",
    "total_token_usage",
    "thread_token_usage",
    "turn_token_usage",
    "latest_token_usage_record",
    "time_to_first_token_ms",
}

_HERMES_SESSION_KEYS = ("session_id", "conversation_id", "thread_id", "session", "conversation")
_HERMES_ROOT_KEYS = ("project_root", "repo_root", "cwd", "working_directory")
_HERMES_PROJECT_ID_KEYS = ("project_id", "workspace_id")
_HERMES_PROJECT_NAME_KEYS = ("project_name", "workspace_name")
_TOOL_CALL_CONTENT_CHARS = 1_200
_TOOL_RESULT_CONTENT_CHARS = 2_000


class TranscriptError(ValueError):
    """Base class for source parsing failures."""


class IncompleteTranscriptError(TranscriptError):
    """The final JSONL record is truncated and must be deferred."""

    defer = True
    incomplete = True


class MalformedTranscriptError(TranscriptError):
    """A non-terminal source record is malformed."""


class EmptyTranscriptError(TranscriptError):
    """The source contains no visible records to ingest."""


class HermesAdapterUnavailable(TranscriptError):
    """Hermes is absent or its SQLite schema is not supported by this adapter."""


class UnsupportedSourceError(TranscriptError):
    """No adapter exists for the requested source kind."""


@dataclass
class _AttachmentCollector:
    source_path: str
    values: dict[str, dict[str, Any]]

    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.values = {}

    def add(
        self,
        payload: bytes,
        *,
        media_type: str | None,
        source_line: int,
    ) -> dict[str, str | int]:
        digest = hashlib.sha256(payload).hexdigest()
        attachment_id = f"att-{digest[:24]}"
        self.values.setdefault(
            attachment_id,
            {
                "id": attachment_id,
                "sha256": digest,
                "source_path": self.source_path,
                "source_line": int(source_line),
                "media_type": media_type or "application/octet-stream",
                "bytes": len(payload),
            },
        )
        return {
            "attachment_id": attachment_id,
            "source_ref": f"{self.source_path}#L{int(source_line)}",
        }

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.values.values())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except OSError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _project_dict(project: Any) -> dict[str, str]:
    if isinstance(project, ProjectInfo):
        raw = project.as_dict()
    elif hasattr(project, "as_dict"):
        raw = project.as_dict()
    elif isinstance(project, Mapping):
        raw = dict(project)
    else:
        raise TranscriptError("a registered project identity is required")
    root = str(raw.get("root") or "")
    name = str(raw.get("name") or (Path(root).name if root else ""))
    vault_dir = str(raw.get("vault_dir") or "")
    project_id = str(raw.get("id") or raw.get("project_id") or "")
    if not root or not name or not vault_dir:
        raise TranscriptError("project identity must contain id, name, root and vault_dir")
    if not project_id:
        project_id = deterministic_project_id(root)
    return {"id": project_id, "name": name, "root": root, "vault_dir": vault_dir}


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return redact_sensitive_text(str(value).strip())


def _looks_like_base64(value: str, *, force: bool = False) -> bool:
    text = value.strip()
    if text.startswith("data:") and ";base64," in text:
        return True
    if len(text) < (16 if force else 64) or len(text) % 4:
        return False
    if not _BASE64_CANDIDATE.fullmatch(text):
        return False
    try:
        decoded = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(decoded) >= (8 if force else 32)


def _decode_base64(value: str) -> tuple[bytes, str | None] | None:
    text = value.strip()
    media_type = None
    if text.startswith("data:") and ";base64," in text:
        prefix, text = text.split(",", 1)
        media_type = prefix[5:].split(";", 1)[0] or None
    try:
        return base64.b64decode(text, validate=True), media_type
    except (binascii.Error, ValueError):
        return None


def _candidate_gate(value: str) -> str:
    """Apply the shared redactor and a final provider-token candidate gate."""

    redacted = redact_sensitive_text(value)
    if _TOKEN_CANDIDATE.search(redacted):
        return _TOKEN_CANDIDATE.sub("<REDACTED>", redacted)
    # Calling the utility's findings function is intentional: installations
    # may add a provider-specific detector without this adapter having to
    # know its token grammar.
    try:
        findings = sensitive_text_findings(redacted)
    except Exception:  # the source must remain ingestible if an old utility is loaded
        findings = {}
    return redacted if not findings else redact_sensitive_text(redacted)


def _sanitize_value(
    value: Any,
    *,
    collector: _AttachmentCollector,
    source_line: int,
    key_hint: str = "",
    media_type_hint: str | None = None,
) -> Any:
    if isinstance(value, bytes):
        return collector.add(value, media_type=media_type_hint, source_line=source_line)
    if isinstance(value, str):
        force_binary = key_hint.lower() in {
            "data",
            "base64",
            "bytes",
            "binary",
            "blob",
            "image",
            "image_data",
            "raw_image",
        }
        if _looks_like_base64(value, force=force_binary):
            decoded = _decode_base64(value)
            if decoded is not None:
                payload, detected_type = decoded
                return collector.add(
                    payload,
                    media_type=detected_type or media_type_hint,
                    source_line=source_line,
                )
        return _candidate_gate(value)
    if isinstance(value, Mapping):
        raw_media = value.get("media_type") or value.get("mediaType") or value.get("mime_type")
        media_type = str(raw_media) if raw_media else media_type_hint
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _SAFE_TELEMETRY_KEYS:
                # These are bounded provider telemetry objects, not
                # credentials.  Recurse into them so a secret-looking child
                # key is still redacted while preserving parser output.
                result[key] = _sanitize_value(
                    raw_value,
                    collector=collector,
                    source_line=source_line,
                    key_hint=key,
                    media_type_hint=media_type,
                )
                continue
            if _SECRET_KEY.search(lowered) or re.search(
                r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
                r"client[_-]?secret|password|passwd|secret|token|cookie|set-cookie)$",
                lowered,
            ):
                result[key] = "<REDACTED>"
                continue
            # Keep the media type but never copy image/binary bytes into a
            # message.  _sanitize_value turns the value into an attachment
            # reference and records its source line.
            result[key] = _sanitize_value(
                raw_value,
                collector=collector,
                source_line=source_line,
                key_hint=key,
                media_type_hint=media_type,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_value(
                item,
                collector=collector,
                source_line=source_line,
                key_hint=key_hint,
                media_type_hint=media_type_hint,
            )
            for item in value
        ]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _candidate_gate(str(value))


def _clean_content(
    value: Any,
    *,
    collector: _AttachmentCollector,
    source_line: int,
) -> Any:
    """Drop hidden thinking blocks while retaining all visible structure."""

    if isinstance(value, Mapping):
        block_type = str(value.get("type") or "").lower()
        channel = str(value.get("channel") or value.get("phase") or "").lower()
        if block_type in _HIDDEN_BLOCK_TYPES or channel in _HIDDEN_BLOCK_TYPES:
            return None
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"[_-]", "", str(key).lower())
            # Providers do not use one stable shape for hidden reasoning.  A
            # block may be tagged by ``type``/``phase`` (handled above), or
            # expose the text directly under a reasoning-named field.
            if normalized_key in {
                "analysis",
                "thinking",
                "reasoning",
                "redactedthinking",
                "reasoningcontent",
            }:
                continue
            child = _clean_content(item, collector=collector, source_line=source_line)
            if child is not None:
                cleaned[str(key)] = child
        return _sanitize_value(cleaned, collector=collector, source_line=source_line)
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            child = _clean_content(item, collector=collector, source_line=source_line)
            if child is not None:
                result.append(child)
        return _sanitize_value(result, collector=collector, source_line=source_line)
    return _sanitize_value(value, collector=collector, source_line=source_line)


def _has_visible_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_visible_content(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_visible_content(item) for item in value)
    return True


def _bound_message_content(content: Any, message_type: str) -> Any:
    """Keep tool evidence useful without letting one rollout fill the queue."""

    if message_type == "tool_call":
        if isinstance(content, Mapping):
            bounded = dict(content)
            for key in ("arguments", "input"):
                if key in bounded:
                    bounded[key] = bounded_redacted_text(
                        bounded[key], _TOOL_CALL_CONTENT_CHARS
                    )
            return bounded
        return bounded_redacted_text(content, _TOOL_CALL_CONTENT_CHARS)
    if message_type == "tool_result":
        return bounded_redacted_text(content, _TOOL_RESULT_CONTENT_CHARS)
    return content


def _content_fingerprint(value: Any) -> str:
    # Codex's event_msg mirror commonly stores the text as a string while
    # response_item stores the same text in an input_text/output_text block.
    # Normalize those two shapes for deduplication, without changing the
    # actual envelope content that is retained.
    text_parts: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, Mapping):
            if "text" in item:
                collect(item.get("text"))
            elif "content" in item and len(item) <= 2:
                collect(item.get("content"))
            else:
                for child in item.values():
                    if not isinstance(child, (bool, int, float)):
                        collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    normalized = "\n".join(part.strip() for part in text_parts if part.strip())
    material = {"text": normalized} if normalized else value
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def _refs(entry: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> dict[str, str]:
    payload = payload or {}
    refs: dict[str, str] = {}
    for key, label in (
        ("source_ref", "source_ref"),
        ("uuid", "uuid"),
        ("parentUuid", "parent_ref"),
        ("parent_id", "parent_ref"),
        ("turn_id", "turn_id"),
        ("thread_id", "thread_id"),
    ):
        value = _first(payload, key) or _first(entry, key)
        if value is not None and not isinstance(value, (dict, list)):
            refs[label] = _safe_string(value)
    return refs


def _event_timestamp(entry: Mapping[str, Any], payload: Mapping[str, Any] | None = None) -> str | None:
    payload = payload or {}
    return _iso(
        _first(
            payload,
            "timestamp",
            "created_at",
            "createdAt",
            "started_at",
            "completed_at",
        )
        or _first(entry, "timestamp", "created_at", "createdAt", "started_at", "completed_at")
    )


def _candidate(
    *,
    entry: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    line: int,
    role: str,
    message_type: str,
    content: Any,
    collector: _AttachmentCollector,
    native_id: Any = None,
    call_id: Any = None,
    status: Any = None,
    model: Any = None,
    mirror: bool = False,
    refs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload = payload or {}
    cleaned = _clean_content(content, collector=collector, source_line=line)
    if not _has_visible_content(cleaned) and message_type not in {"event", "unknown_event"}:
        return None
    merged_refs = dict(_refs(entry, payload))
    if refs:
        merged_refs.update(
            {
                str(key): _safe_string(value)
                for key, value in refs.items()
                if value not in (None, "") and not isinstance(value, (dict, list))
            }
        )
    return {
        "_line": line,
        "_role": _safe_string(role or "system") or "system",
        "_type": message_type,
        "_content": cleaned,
        "_native_id": _safe_string(native_id) if native_id is not None else "",
        "_call_id": _safe_string(call_id) if call_id is not None else "",
        "_status": _safe_string(status) if status is not None else "",
        "_model": _safe_string(model) if model is not None else "",
        "_timestamp": _event_timestamp(entry, payload),
        "_refs": merged_refs,
        "_mirror": mirror,
        "_fingerprint": _content_fingerprint(cleaned),
    }


def _finalize(
    *,
    source: str,
    path: Path,
    project: Any,
    session_id: Any,
    host_id: Any,
    started_at: Any,
    candidates: Sequence[dict[str, Any]],
    collector: _AttachmentCollector,
    updated_at: Any = None,
    ordinal_start: int = 0,
    audit_unknown: Iterable[str] = (),
    source_cwd: Any = None,
    source_project_id: Any = None,
) -> dict[str, Any]:
    project_payload = _project_dict(project)
    sid = _safe_string(session_id) or path.stem
    unknown = sorted({str(item) for item in audit_unknown if item})

    # event_msg user/agent records mirror response_item messages in Codex.
    # Counter-based removal keeps repeated user messages intact.
    response_counts = Counter(
        item["_fingerprint"] for item in candidates if not item.get("_mirror")
    )
    visible: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: int(value.get("_line", 0))):
        if item.get("_mirror"):
            fingerprint = item["_fingerprint"]
            # Any event_msg with a response_item counterpart is a mirror;
            # dropping all of them also handles a source that emitted the
            # same mirror more than once.  Repeated user turns remain intact
            # because the response_item records themselves are retained.
            if fingerprint in response_counts:
                continue
        visible.append(item)

    # Codex emits large internal lifecycle/telemetry records whose provider
    # type is not part of the conversation.  The legacy collector deliberately
    # ignored these records; retaining their full payload made otherwise valid
    # current sessions exceed the outbox limit (several megabytes) and blocked
    # the 30-minute path.  Their type names remain in ``audit.unknown_types``
    # below, so coverage keeps the diagnostic without copying the payload.
    visible = [item for item in visible if item.get("_type") != "unknown_event"]

    messages: list[dict[str, Any]] = []
    for ordinal, item in enumerate(visible, max(0, int(ordinal_start)) + 1):
        content = _bound_message_content(item["_content"], item["_type"])
        native_id = item.get("_native_id") or ""
        if native_id:
            message_id = native_id
        else:
            message_id = "msg-" + hashlib.sha256(
                _canonical_json(
                    {
                        "session_id": sid,
                        "ordinal": ordinal,
                        "role": item["_role"],
                        "type": item["_type"],
                        "content": content,
                    }
                ).encode("utf-8")
            ).hexdigest()[:24]
        message: dict[str, Any] = {
            "id": message_id,
            "ordinal": ordinal,
            "role": item["_role"],
            "type": item["_type"],
            "timestamp": item.get("_timestamp"),
            "content": content,
        }
        for internal, public in (
            ("_call_id", "call_id"),
            ("_status", "status"),
            ("_model", "model"),
            ("_refs", "refs"),
        ):
            value = item.get(internal)
            if value:
                message[public] = value
        messages.append(message)

    if not messages:
        raise EmptyTranscriptError(f"no visible records in {path}")

    attachments = collector.as_list()
    stable = {"session_id": sid, "messages": messages, "attachments": attachments}
    revision = hashlib.sha256(_canonical_json(stable).encode("utf-8")).hexdigest()
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project_payload,
        "source": source,
        "session_id": sid,
        "revision": revision,
        "source_path": str(path.resolve()),
        "host_id": _safe_string(host_id),
        "started_at": _iso(started_at) or _file_mtime(path),
        "updated_at": _iso(updated_at) or _file_mtime(path),
        "messages": messages,
        "attachments": attachments,
    }
    # Keep provider routing evidence separate from the destination project.
    # This is especially important for a Hermes DB that contains sessions from
    # more than one checkout: callers can route each session before parsing or
    # verify that a supplied project matches this source metadata.
    if source_cwd not in (None, ""):
        envelope["source_cwd"] = _safe_string(source_cwd)
    if source_project_id not in (None, ""):
        envelope["source_project_id"] = _safe_string(source_project_id)
    if unknown:
        # Unknown records are represented in messages above and listed here
        # for a cheap audit without retaining a second raw copy of the event.
        envelope["audit"] = {"unknown_types": unknown}
    return envelope


def _read_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise TranscriptError(f"cannot read transcript: {source}") from exc
    records: list[tuple[int, dict[str, Any]]] = []
    with handle:
        # Keep the line boundaries so a malformed final record can be
        # distinguished from corruption in the middle.  A writer may flush a
        # final partial JSON object with or without a trailing newline.
        raw_lines = list(handle)
        for index, raw in enumerate(raw_lines):
            line_number = index + 1
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                has_following_content = any(item.strip() for item in raw_lines[index + 1 :])
                # A newline-terminated malformed line is a complete record
                # from the writer's perspective.  Only an unterminated final
                # line is safely classifiable as a deferred partial write.
                if not has_following_content and not raw.endswith(("\n", "\r")):
                    raise IncompleteTranscriptError(
                        f"truncated final JSONL record at line {line_number}"
                    ) from exc
                raise MalformedTranscriptError(
                    f"malformed JSONL record at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise MalformedTranscriptError(f"JSONL record {line_number} is not an object")
            records.append((line_number, value))
    return records


def _read_jsonl_from_offset(
    path: str | Path, offset: int
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Read only complete JSONL records appended after ``offset``."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if offset < 0 or offset > size:
                offset = 0
            handle.seek(offset)
            if offset:
                # A stored offset may point into a record after a crash. Drop
                # that partial record and resume at the next newline.
                handle.seek(offset - 1)
                previous = handle.read(1)
                handle.seek(offset)
                if previous not in {b"\n", b"\r"}:
                    handle.readline()
            start = handle.tell()
            records: list[tuple[int, dict[str, Any]]] = []
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if not raw.endswith((b"\n", b"\r")):
                        raise IncompleteTranscriptError(
                            f"truncated final JSONL record at byte {start}"
                        ) from exc
                    raise MalformedTranscriptError(
                        f"malformed JSONL record near byte {start}"
                    ) from exc
                if not isinstance(value, dict):
                    raise MalformedTranscriptError("JSONL record is not an object")
                records.append((line_number, value))
            return records, handle.tell()
    except (OSError, UnicodeError) as exc:
        raise TranscriptError(f"cannot read transcript increment: {source}") from exc


def _visible_assistant(payload: Mapping[str, Any]) -> bool:
    for marker in (payload.get("channel"), payload.get("phase")):
        if marker is None:
            continue
        lowered = str(marker).strip().lower()
        if lowered in _HIDDEN_BLOCK_TYPES:
            return False
        if lowered not in {"final", "commentary", "output", "result"}:
            return False
    return True


def _text_or_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "content" in value:
            return value["content"]
        if "text" in value:
            return value["text"]
    return value


def parse_codex(
    path: str | Path,
    project: Any,
    *,
    host_id: str | None = None,
    session_id: str | None = None,
    _records: Sequence[tuple[int, dict[str, Any]]] | None = None,
    _ordinal_start: int = 0,
) -> dict[str, Any]:
    """Parse one Codex JSONL rollout into one filtered envelope."""

    source_path = Path(path)
    records = list(_records) if _records is not None else _read_jsonl(source_path)
    meta: dict[str, Any] = {}
    current_model: Any = None
    first_timestamp: str | None = None
    latest_timestamp: str | None = None
    candidates: list[dict[str, Any]] = []
    unknown: list[str] = []
    collector = _AttachmentCollector(str(source_path.resolve()))

    for line, entry in records:
        event_type = str(entry.get("type") or "")
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            payload = {}
        event_timestamp = _event_timestamp(entry, payload)
        first_timestamp = first_timestamp or event_timestamp
        latest_timestamp = event_timestamp or latest_timestamp
        if event_type == "session_meta":
            if not meta:
                meta.update(payload)
            current_model = _first(payload, "model") or current_model
            continue
        if event_type == "turn_context":
            current_model = _first(
                payload, "model", "model_name", "model_reasoning_effort", "effort"
            ) or current_model
            continue
        if event_type == "response_item":
            item_type = str(payload.get("type") or "")
            role = str(payload.get("role") or "")
            if item_type == "message":
                if role == "assistant" and not _visible_assistant(payload):
                    continue
                item = _candidate(
                    entry=entry,
                    payload=payload,
                    line=line,
                    role=role or "system",
                    message_type="message",
                    content=payload.get("content"),
                    collector=collector,
                    native_id=_first(payload, "id") or _first(entry, "id"),
                    model=_first(payload, "model") or current_model,
                )
                if item:
                    candidates.append(item)
                continue
            if item_type in {"function_call", "custom_tool_call", "tool_call"}:
                item = _candidate(
                    entry=entry,
                    payload=payload,
                    line=line,
                    role=role or "assistant",
                    message_type="tool_call",
                    content={
                        "name": payload.get("name"),
                        "arguments": payload.get("arguments", payload.get("input")),
                    },
                    collector=collector,
                    native_id=_first(payload, "id"),
                    call_id=_first(payload, "call_id", "tool_call_id", "id"),
                    model=_first(payload, "model") or current_model,
                )
                if item:
                    candidates.append(item)
                continue
            if item_type in {
                "function_call_output",
                "custom_tool_call_output",
                "tool_result",
            }:
                output = payload.get("output", payload.get("content", payload.get("result")))
                status = payload.get("status")
                if status in (None, "") and payload.get("is_error", payload.get("isError")):
                    status = "error"
                item = _candidate(
                    entry=entry,
                    payload=payload,
                    line=line,
                    role=role or "tool",
                    message_type="tool_result",
                    content=output,
                    collector=collector,
                    native_id=_first(payload, "id"),
                    call_id=_first(payload, "call_id", "tool_call_id", "id"),
                    status=status,
                    model=_first(payload, "model") or current_model,
                )
                if item:
                    candidates.append(item)
                continue
            if item_type in {"reasoning", "analysis"}:
                continue
            unknown_type = item_type or "response_item"
            unknown.append(unknown_type)
            item = _candidate(
                entry=entry,
                payload=payload,
                line=line,
                role="system",
                message_type="unknown_event",
                content=payload,
                collector=collector,
                native_id=_first(payload, "id"),
                refs={"unknown_type": unknown_type},
            )
            if item:
                candidates.append(item)
            continue
        if event_type == "event_msg":
            subtype = str(payload.get("type") or "")
            if subtype in {"user_message", "agent_message"}:
                role = "user" if subtype == "user_message" else "assistant"
                content = _text_or_content(
                    payload.get("message", payload.get("text", payload.get("content")))
                )
                if role == "assistant" and not _visible_assistant(payload):
                    continue
                item = _candidate(
                    entry=entry,
                    payload=payload,
                    line=line,
                    role=role,
                    message_type="message",
                    content=content,
                    collector=collector,
                    native_id=_first(payload, "id"),
                    model=_first(payload, "model") or current_model,
                    mirror=True,
                )
                if item:
                    candidates.append(item)
                continue
            if subtype in {"task_started", "task_complete", "turn_started", "turn_complete", "status"}:
                item = _candidate(
                    entry=entry,
                    payload=payload,
                    line=line,
                    role="system",
                    message_type="event",
                    content=payload,
                    collector=collector,
                    native_id=_first(payload, "id", "turn_id"),
                    status=_first(payload, "status"),
                    refs={"event_type": subtype},
                )
                if item:
                    candidates.append(item)
                continue
            unknown_type = subtype or "event_msg"
            unknown.append(unknown_type)
            item = _candidate(
                entry=entry,
                payload=payload,
                line=line,
                role="system",
                message_type="unknown_event",
                content=payload,
                collector=collector,
                native_id=_first(payload, "id"),
                refs={"unknown_type": unknown_type},
            )
            if item:
                candidates.append(item)
            continue
        if event_type == "agent_message":
            content = _text_or_content(
                entry.get("message", entry.get("content", payload.get("message", payload.get("content"))))
            )
            item = _candidate(
                entry=entry,
                payload=payload,
                line=line,
                role="assistant",
                message_type="message",
                content=content,
                collector=collector,
                native_id=_first(entry, "id") or _first(payload, "id"),
                model=_first(entry, "model") or _first(payload, "model") or current_model,
                mirror=True,
            )
            if item:
                candidates.append(item)
            continue
        if event_type in {"", "response_item", "turn_context"}:
            continue
        # Keep useful hooks/status records and make all other event kinds
        # visible as an explicit audit message.
        if event_type in {"hook", "status", "task_started", "task_complete"}:
            item_type = "event"
        else:
            item_type = "unknown_event"
            unknown.append(event_type)
        item = _candidate(
            entry=entry,
            payload=payload,
            line=line,
            role="system",
            message_type=item_type,
            content=payload or entry,
            collector=collector,
            native_id=_first(entry, "id") or _first(payload, "id"),
            status=_first(payload, "status"),
            refs={"unknown_type": event_type} if item_type == "unknown_event" else {"event_type": event_type},
        )
        if item:
            candidates.append(item)

    detected_session_id = _first(meta, "id", "session_id") or session_id or source_path.stem
    if (
        session_id is not None
        and detected_session_id not in (None, "")
        and str(detected_session_id) != str(session_id)
    ):
        raise EmptyTranscriptError(
            f"Codex session {session_id!r} is absent from {source_path}"
        )
    return _finalize(
        source="codex",
        path=source_path,
        project=project,
        session_id=detected_session_id,
        host_id=host_id or _first(meta, "host_id", "hostname"),
        started_at=_first(meta, "timestamp", "started_at") or first_timestamp,
        updated_at=latest_timestamp,
        candidates=candidates,
        collector=collector,
        ordinal_start=_ordinal_start,
        audit_unknown=unknown,
        source_cwd=_first(meta, "cwd", "working_directory", "project_root", "repo_root"),
    )


def parse_codex_incremental(
    path: str | Path,
    project: Any,
    *,
    offset: int,
    ordinal_start: int = 0,
    host_id: str | None = None,
    session_id: str | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Parse only Codex JSONL records appended after a durable byte offset."""

    source_path = Path(path)
    records, next_offset = _read_jsonl_from_offset(source_path, offset)
    if not records:
        return None, next_offset
    return (
        parse_codex(
            source_path,
            project,
            host_id=host_id,
            session_id=session_id,
            _records=records,
            _ordinal_start=max(0, int(ordinal_start)),
        ),
        next_offset,
    )


def _claude_role(entry: Mapping[str, Any], message: Mapping[str, Any]) -> str:
    value = _first(message, "role") or _first(entry, "role")
    if value:
        return str(value)
    event_type = str(entry.get("type") or "")
    return event_type if event_type in {"user", "assistant", "system", "tool"} else ""


def parse_claude(
    path: str | Path,
    project: Any,
    *,
    host_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Parse Claude Code JSONL, including tool blocks and hook/status events."""

    source_path = Path(path)
    records = _read_jsonl(source_path)
    candidates: list[dict[str, Any]] = []
    collector = _AttachmentCollector(str(source_path.resolve()))
    unknown: list[str] = []
    detected_session_id: Any = None
    cwd: Any = None
    started_at: Any = None
    updated_at: Any = None
    current_model: Any = None

    for line, entry in records:
        detected_session_id = detected_session_id or _first(
            entry, "sessionId", "session_id", "conversation_id"
        )
        cwd = cwd or _first(entry, "cwd", "working_directory", "project_root", "repo_root")
        timestamp = _event_timestamp(entry)
        started_at = started_at or timestamp
        updated_at = timestamp or updated_at
        current_model = _first(entry, "model", "model_name") or current_model
        raw_message = entry.get("message")
        message = raw_message if isinstance(raw_message, Mapping) else {}
        cwd = cwd or _first(message, "cwd", "working_directory", "project_root", "repo_root")
        role = _claude_role(entry, message)
        if role in {"user", "assistant", "developer", "system"}:
            current_model = _first(message, "model") or current_model
            content = message.get("content", entry.get("content"))
            if content is None and raw_message is not None and not isinstance(raw_message, Mapping):
                content = raw_message
            blocks = content if isinstance(content, list) else [content]
            ordinary: list[Any] = []
            for block in blocks:
                if not isinstance(block, Mapping):
                    ordinary.append(block)
                    continue
                block_type = str(block.get("type") or "").lower()
                if block_type in {"tool_use", "tool_call", "function_call"}:
                    item = _candidate(
                        entry=entry,
                        payload=block,
                        line=line,
                        role="assistant",
                        message_type="tool_call",
                        content={
                            "name": block.get("name"),
                            "input": block.get("input", block.get("arguments")),
                        },
                        collector=collector,
                        native_id=_first(block, "id"),
                        call_id=_first(block, "id", "call_id", "tool_use_id"),
                        model=current_model,
                    )
                    if item:
                        candidates.append(item)
                    continue
                if block_type in {"tool_result", "function_call_output", "tool_response"}:
                    status = "error" if block.get("is_error", block.get("isError")) else block.get("status")
                    item = _candidate(
                        entry=entry,
                        payload=block,
                        line=line,
                        role="tool",
                        message_type="tool_result",
                        content=block.get("content", block.get("output", block.get("result"))),
                        collector=collector,
                        native_id=_first(block, "id"),
                        call_id=_first(block, "tool_use_id", "call_id", "id"),
                        status=status,
                        model=current_model,
                    )
                    if item:
                        candidates.append(item)
                    continue
                if block_type in _HIDDEN_BLOCK_TYPES:
                    continue
                ordinary.append(block)
            ordinary_content: Any = ordinary if isinstance(content, list) else (ordinary[0] if ordinary else None)
            if _has_visible_content(ordinary_content):
                item = _candidate(
                    entry=entry,
                    payload=message,
                    line=line,
                    role=role,
                    message_type="message",
                    content=ordinary_content,
                    collector=collector,
                    native_id=_first(message, "id") or _first(entry, "id"),
                    model=current_model,
                )
                if item:
                    candidates.append(item)
            continue

        event_type = str(entry.get("type") or "")
        if event_type in {"tool_use", "tool_call", "function_call"}:
            item = _candidate(
                entry=entry,
                payload=entry,
                line=line,
                role="assistant",
                message_type="tool_call",
                content={"name": entry.get("name"), "input": entry.get("input", entry.get("arguments"))},
                collector=collector,
                native_id=_first(entry, "id"),
                call_id=_first(entry, "id", "call_id"),
                model=current_model,
            )
            if item:
                candidates.append(item)
            continue
        if event_type in {"tool_result", "tool_response", "function_call_output"}:
            item = _candidate(
                entry=entry,
                payload=entry,
                line=line,
                role="tool",
                message_type="tool_result",
                content=entry.get("content", entry.get("output", entry.get("result"))),
                collector=collector,
                native_id=_first(entry, "id"),
                call_id=_first(entry, "tool_use_id", "call_id", "id"),
                status="error" if entry.get("is_error", entry.get("isError")) else entry.get("status"),
                model=current_model,
            )
            if item:
                candidates.append(item)
            continue
        if event_type in {"hook", "hook_event", "status", "progress", "session_start", "session_end"}:
            item = _candidate(
                entry=entry,
                payload=entry,
                line=line,
                role="system",
                message_type="event",
                content=entry,
                collector=collector,
                native_id=_first(entry, "id"),
                status=_first(entry, "status"),
                model=current_model,
                refs={"event_type": event_type},
            )
            if item:
                candidates.append(item)
            continue
        if not event_type:
            # A message-less metadata line is still useful only when it has a
            # status/hook payload; otherwise do not invent a transcript turn.
            continue
        unknown.append(event_type)
        item = _candidate(
            entry=entry,
            payload=entry,
            line=line,
            role="system",
            message_type="unknown_event",
            content=entry,
            collector=collector,
            native_id=_first(entry, "id"),
            refs={"unknown_type": event_type},
        )
        if item:
            candidates.append(item)

    detected_session_id = detected_session_id or source_path.stem
    if (
        session_id is not None
        and detected_session_id not in (None, "")
        and str(detected_session_id) != str(session_id)
    ):
        raise EmptyTranscriptError(
            f"Claude session {session_id!r} is absent from {source_path}"
        )
    return _finalize(
        source="claude",
        path=source_path,
        project=project,
        session_id=detected_session_id,
        host_id=host_id,
        started_at=started_at,
        updated_at=updated_at,
        candidates=candidates,
        collector=collector,
        audit_unknown=unknown,
        source_cwd=cwd,
    )


def _quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise HermesAdapterUnavailable("unsupported Hermes identifier")
    return '"' + value.replace('"', '""') + '"'


def inspect_hermes_schema(path: str | Path) -> dict[str, Any]:
    """Inspect only SQLite schema through a read-only URI connection."""

    source = Path(path)
    if not source.exists():
        raise HermesAdapterUnavailable(f"Hermes database is absent: {source}")
    try:
        connection = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise HermesAdapterUnavailable("Hermes SQLite adapter unavailable") from exc
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        tables: dict[str, list[dict[str, Any]]] = {}
        for (name,) in rows:
            if not isinstance(name, str) or name.startswith("sqlite_"):
                continue
            columns = connection.execute(f"PRAGMA table_info({_quote_identifier(name)})").fetchall()
            tables[name] = [
                {"name": row[1], "type": row[2], "notnull": bool(row[3]), "pk": bool(row[5])}
                for row in columns
            ]
        return {"tables": tables}
    except sqlite3.Error as exc:
        raise HermesAdapterUnavailable("Hermes SQLite schema inspection failed") from exc
    finally:
        connection.close()


def _hermes_table(schema: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    tables = schema.get("tables", {})
    for table_name, columns in tables.items() if isinstance(tables, Mapping) else ():
        if table_name not in {"messages", "conversation_messages"}:
            continue
        mapping = {
            str(column.get("name", "")).lower(): str(column.get("name", ""))
            for column in columns
            if isinstance(column, Mapping)
        }
        if "role" not in mapping:
            continue
        content_key = next(
            (key for key in ("content", "body", "text", "payload") if key in mapping), None
        )
        if content_key is None:
            continue
        return table_name, mapping
    raise HermesAdapterUnavailable(
        "Hermes SQLite schema unsupported; expected messages(role, content)"
    )


def _hermes_column(columns: Mapping[str, str], names: Sequence[str]) -> str | None:
    """Return the first supported lower-case column key."""

    return next((name for name in names if name in columns), None)


def _hermes_row_value(row: Mapping[str, Any], columns: Mapping[str, str], names: Sequence[str]) -> Any:
    key = _hermes_column(columns, names)
    return row.get(key) if key else None


def _normalised_root(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(Path(str(value)).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return str(value).strip()


def _project_descriptors(project: Any) -> list[Any]:
    """Normalise one descriptor or a caller-provided project collection."""

    if project is None:
        return []
    if isinstance(project, Mapping):
        nested = project.get("projects")
        if isinstance(nested, (list, tuple)):
            return list(nested)
        return [project]
    if isinstance(project, (list, tuple)):
        return list(project)
    return [project]


def _project_descriptor_value(project: Any, names: Sequence[str]) -> Any:
    if isinstance(project, Mapping):
        return _first(project, *names)
    for name in names:
        try:
            value = getattr(project, name)
        except AttributeError:
            continue
        if value not in (None, ""):
            return value
    return None


def _select_hermes_project(
    project: Any,
    *,
    source_root: str,
    source_project_id: Any = None,
) -> Any:
    """Select the project matching one Hermes session's routing metadata.

    A single project argument is safe only when the DB row has no routing
    metadata or names that exact project.  If a DB contains another checkout,
    fail closed instead of silently attributing it to whichever project was
    resolved first by the caller.
    """

    descriptors = _project_descriptors(project)
    if not descriptors:
        raise HermesAdapterUnavailable("Hermes session has no registered project identity")
    if not source_root and source_project_id in (None, ""):
        if len(descriptors) == 1:
            return descriptors[0]
        raise HermesAdapterUnavailable("Hermes session has no project routing metadata")

    root_key = _normalised_root(source_root)
    id_key = str(source_project_id).strip() if source_project_id not in (None, "") else ""
    matches: list[Any] = []
    for descriptor in descriptors:
        descriptor_root = _normalised_root(
            _project_descriptor_value(descriptor, ("root", "project_root", "repo_root"))
        )
        descriptor_id = _project_descriptor_value(descriptor, ("id", "project_id", "uuid"))
        # When both routing hints are present, they must describe the same
        # registered project.  Matching an ID while silently ignoring a
        # conflicting root would reintroduce cross-project attribution.
        if root_key and descriptor_root != root_key:
            continue
        if id_key and descriptor_id not in (None, "") and str(descriptor_id) != id_key:
            continue
        if root_key or (id_key and descriptor_id not in (None, "")):
            matches.append(descriptor)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise HermesAdapterUnavailable(
            "Hermes session project does not match a registered project identity"
        )
    raise HermesAdapterUnavailable("Hermes session matches multiple project identities")


def _hermes_query_parts(
    source_path: Path,
) -> tuple[str, dict[str, str], list[str], str, sqlite3.Connection]:
    """Open Hermes read-only and return the safe, known-column query shape."""

    schema = inspect_hermes_schema(source_path)
    table_name, columns = _hermes_table(schema)
    wanted_keys = [
        key
        for key in (
            "id",
            *_HERMES_SESSION_KEYS,
            "role",
            "content",
            "body",
            "text",
            "payload",
            "type",
            *_HERMES_ROOT_KEYS,
            *_HERMES_PROJECT_ID_KEYS,
            *_HERMES_PROJECT_NAME_KEYS,
            "created_at",
            "timestamp",
            "call_id",
            "tool_call_id",
            "status",
            "model",
            "is_error",
        )
        if key in columns
    ]
    if "role" not in wanted_keys:
        raise HermesAdapterUnavailable("Hermes messages table has no role column")
    content_key = _hermes_column(columns, ("content", "body", "text", "payload"))
    if content_key is None:
        raise HermesAdapterUnavailable("Hermes messages table has no supported content column")
    try:
        connection = sqlite3.connect(f"file:{source_path.resolve()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise HermesAdapterUnavailable("Hermes SQLite adapter unavailable") from exc
    return table_name, columns, wanted_keys, content_key, connection


def _read_hermes_rows(
    path: str | Path,
) -> tuple[Path, dict[str, str], list[str], str, dict[str, list[dict[str, Any]]]]:
    source_path = Path(path)
    table_name, columns, wanted_keys, content_key, connection = _hermes_query_parts(source_path)
    select = ", ".join(_quote_identifier(columns[key]) for key in wanted_keys)
    query = f"SELECT {select} FROM {_quote_identifier(table_name)} ORDER BY rowid"
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        for row in connection.execute(query):
            raw = {key: row[index] for index, key in enumerate(wanted_keys)}
            session = _hermes_row_value(raw, columns, _HERMES_SESSION_KEYS)
            if session not in (None, ""):
                session_key = str(session)
            else:
                root = _hermes_row_value(raw, columns, _HERMES_ROOT_KEYS)
                session_key = (
                    f"{source_path.stem}:{_normalised_root(root)}"
                    if root not in (None, "")
                    else source_path.stem
                )
            grouped.setdefault(session_key, []).append(raw)
    except sqlite3.Error as exc:
        raise HermesAdapterUnavailable("Hermes messages read failed") from exc
    finally:
        connection.close()
    return source_path, columns, wanted_keys, content_key, grouped


def iter_hermes(
    path: str | Path,
    project: Any,
    *,
    host_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one filtered envelope per Hermes session.

    ``session_id`` is an optional metadata target used by bounded collection.
    Without it, a multi-session DB still yields one envelope per session; a
    project-root column is matched against the supplied registered project so
    another project's rows can never be silently attributed to this one.
    """

    source_path, columns, _wanted_keys, content_key, grouped = _read_hermes_rows(path)
    if session_id is not None:
        target = str(session_id)
        grouped = {key: rows for key, rows in grouped.items() if key == target}
        if not grouped:
            raise EmptyTranscriptError(f"Hermes session {target!r} is absent from {source_path}")

    for current_session_id, rows in grouped.items():
        roots = {
            _normalised_root(_hermes_row_value(row, columns, _HERMES_ROOT_KEYS))
            for row in rows
            if _hermes_row_value(row, columns, _HERMES_ROOT_KEYS) not in (None, "")
        }
        if len(roots) > 1:
            raise HermesAdapterUnavailable("Hermes session contains multiple project roots")
        source_root = next(iter(roots), "")
        source_project_id = next(
            (
                _hermes_row_value(row, columns, _HERMES_PROJECT_ID_KEYS)
                for row in rows
                if _hermes_row_value(row, columns, _HERMES_PROJECT_ID_KEYS) not in (None, "")
            ),
            None,
        )
        try:
            selected_project = _select_hermes_project(
                project,
                source_root=source_root,
                source_project_id=source_project_id,
            )
        except HermesAdapterUnavailable:
            # An un-targeted call may be given one registered project while
            # the database also contains sessions from other projects.  Skip
            # those sessions rather than attributing them to the first
            # project.  A targeted call must surface the mismatch so the
            # caller can repair its routing candidate.
            if session_id is None:
                continue
            raise
        collector = _AttachmentCollector(str(source_path.resolve()))
        candidates: list[dict[str, Any]] = []
        timestamps: list[str] = []
        for ordinal, row in enumerate(rows, 1):
            timestamp = _iso(_hermes_row_value(row, columns, ("created_at", "timestamp")))
            if timestamp:
                timestamps.append(timestamp)
            role = str(row.get("role") or "system")
            row_type = str(row.get("type") or "message")
            if role.lower() in _HIDDEN_BLOCK_TYPES or row_type.lower() in _HIDDEN_BLOCK_TYPES:
                continue
            if row_type in {"tool_call", "function_call"}:
                message_type = "tool_call"
            elif row_type in {"tool_result", "function_call_output", "tool_response"}:
                message_type = "tool_result"
            else:
                message_type = "message"
            item = _candidate(
                entry=row,
                payload=row,
                line=ordinal,
                role=role,
                message_type=message_type,
                content=row.get(content_key),
                collector=collector,
                native_id=row.get("id"),
                call_id=row.get("call_id") or row.get("tool_call_id"),
                status=("error" if row.get("is_error") else row.get("status")),
                model=row.get("model"),
            )
            if item:
                candidates.append(item)
        if not candidates:
            continue
        yield _finalize(
            source="hermes",
            path=source_path,
            project=selected_project,
            session_id=current_session_id,
            host_id=host_id,
            started_at=timestamps[0] if timestamps else None,
            updated_at=timestamps[-1] if timestamps else None,
            candidates=candidates,
            collector=collector,
            source_cwd=source_root,
            source_project_id=source_project_id,
        )


def parse_hermes(
    path: str | Path,
    project: Any,
    *,
    host_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    envelopes = list(iter_hermes(path, project, host_id=host_id, session_id=session_id))
    if len(envelopes) == 1:
        return envelopes[0]
    return envelopes


def parse_transcript(
    path: str | Path,
    source: str,
    project: Any,
    *,
    host_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Dispatch to a source adapter; Hermes may yield multiple sessions."""

    normalized = str(source).lower().strip()
    if normalized == "codex":
        return parse_codex(path, project, host_id=host_id, session_id=session_id)
    if normalized == "claude":
        return parse_claude(path, project, host_id=host_id, session_id=session_id)
    if normalized == "hermes":
        return parse_hermes(path, project, host_id=host_id, session_id=session_id)
    raise UnsupportedSourceError(f"unsupported transcript source: {source}")


def _metadata_snapshot_id(source: str, path: Path, session_id: str) -> str:
    seed = f"{source}|{path.resolve()}|{session_id}"
    return f"{source}-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _metadata_jsonl(
    path: str | Path,
    source: str,
    *,
    host_id: str | None,
    session_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Read only a bounded prefix needed to route a JSONL source.

    The body parser remains responsible for the complete transcript.  This
    function intentionally reads at most 128 KiB and 32 records, so a source
    cannot consume the collection budget merely because it appears first in a
    directory listing.
    """

    source_path = Path(path)
    resolved = source_path.resolve()
    try:
        with source_path.open("rb") as handle:
            prefix = handle.read(128 * 1024)
    except OSError as exc:
        raise TranscriptError(f"cannot read transcript metadata: {source_path}") from exc
    text = prefix.decode("utf-8", errors="replace")
    detected_session: Any = None
    detected_cwd: Any = None
    first_timestamp: Any = None
    latest_timestamp: Any = None
    for raw in text.splitlines()[:32]:
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            # A large first prompt or a partial prefix is not a routing
            # failure.  The conservative fallback is the stable file stem;
            # the full parser will report structural errors later.
            continue
        if not isinstance(entry, Mapping):
            continue
        payload = entry.get("payload") if isinstance(entry.get("payload"), Mapping) else {}
        message = entry.get("message") if isinstance(entry.get("message"), Mapping) else {}
        detected_session = detected_session or _first(
            entry,
            "sessionId",
            "session_id",
            "conversation_id",
            "thread_id",
            "id" if source == "codex" and entry.get("type") == "session_meta" else "",
        )
        payload_session_keys = ("session_id", "conversation_id", "thread_id")
        if source == "codex" and entry.get("type") == "session_meta":
            payload_session_keys = ("id", *payload_session_keys)
        detected_session = detected_session or _first(payload, *payload_session_keys)
        for candidate in (entry, payload, message):
            detected_cwd = detected_cwd or _first(
                candidate, "cwd", "working_directory", "project_root", "repo_root"
            )
            event_timestamp = _event_timestamp(entry, candidate)
            first_timestamp = first_timestamp or event_timestamp
            latest_timestamp = event_timestamp or latest_timestamp
    current_session = str(session_id or detected_session or source_path.stem)
    if session_id is not None and detected_session not in (None, "") and str(detected_session) != str(session_id):
        return
    try:
        mtime = source_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    yield {
        "metadata_only": True,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "provider": source,
        "path": resolved,
        "source_path": str(resolved),
        "mtime": mtime,
        "source_mtime": mtime,
        "session_id": current_session,
        "snapshot_id": _metadata_snapshot_id(source, source_path, current_session),
        "host_id": _safe_string(host_id),
        "project_root": str(detected_cwd) if detected_cwd not in (None, "") else None,
        "started_at": _iso(first_timestamp),
        "updated_at": _iso(latest_timestamp),
    }


def _metadata_hermes(
    path: str | Path,
    *,
    host_id: str | None,
    session_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Enumerate Hermes sessions from metadata columns without reading body."""

    source_path = Path(path)
    schema = inspect_hermes_schema(source_path)
    table_name, columns = _hermes_table(schema)
    session_key = _hermes_column(columns, _HERMES_SESSION_KEYS)
    root_key = _hermes_column(columns, _HERMES_ROOT_KEYS)
    project_id_key = _hermes_column(columns, _HERMES_PROJECT_ID_KEYS)
    project_name_key = _hermes_column(columns, _HERMES_PROJECT_NAME_KEYS)
    timestamp_key = _hermes_column(columns, ("created_at", "timestamp"))
    select_parts = []
    for key in (session_key, root_key, project_id_key, project_name_key):
        if key:
            select_parts.append(_quote_identifier(columns[key]))
        else:
            select_parts.append("NULL")
    if timestamp_key:
        select_parts.extend(
            [
                f"MIN({_quote_identifier(columns[timestamp_key])})",
                f"MAX({_quote_identifier(columns[timestamp_key])})",
            ]
        )
    else:
        select_parts.extend(["NULL", "NULL"])
    select_parts.append("COUNT(*)")
    group_parts = [part for part in select_parts[:4] if part != "NULL"]
    query = (
        f"SELECT {', '.join(select_parts)} FROM {_quote_identifier(table_name)}"
        + (f" GROUP BY {', '.join(group_parts)}" if group_parts else "")
    )
    try:
        connection = sqlite3.connect(f"file:{source_path.resolve()}?mode=ro", uri=True)
        rows = connection.execute(query).fetchall()
    except sqlite3.Error as exc:
        raise HermesAdapterUnavailable("Hermes metadata read failed") from exc
    finally:
        with contextlib.suppress(UnboundLocalError):
            connection.close()
    try:
        mtime = source_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    for row in rows:
        raw_session, raw_root, raw_project_id, raw_project_name, started, updated, count = row
        if raw_session in (None, "") and raw_root not in (None, ""):
            current_session = f"{source_path.stem}:{_normalised_root(raw_root)}"
        else:
            current_session = str(raw_session or source_path.stem)
        if session_id is not None and current_session != str(session_id):
            continue
        yield {
            "metadata_only": True,
            "schema_version": SCHEMA_VERSION,
            "source": "hermes",
            "provider": "hermes",
            "path": source_path.resolve(),
            "source_path": str(source_path.resolve()),
            "mtime": mtime,
            "source_mtime": mtime,
            "session_id": current_session,
            "snapshot_id": _metadata_snapshot_id("hermes", source_path, current_session),
            "host_id": _safe_string(host_id),
            "project_root": str(raw_root) if raw_root not in (None, "") else None,
            "project_id": str(raw_project_id) if raw_project_id not in (None, "") else None,
            "project_name": str(raw_project_name) if raw_project_name not in (None, "") else None,
            "started_at": _iso(started),
            "updated_at": _iso(updated),
            "row_count": int(count or 0),
        }


def _metadata_snapshots(
    source: str,
    path: str | Path,
    *,
    host_id: str | None,
    session_id: str | None,
) -> Iterator[dict[str, Any]]:
    normalized = str(source).lower().strip()
    if normalized in {"codex", "claude"}:
        yield from _metadata_jsonl(path, normalized, host_id=host_id, session_id=session_id)
        return
    if normalized == "hermes":
        yield from _metadata_hermes(path, host_id=host_id, session_id=session_id)
        return
    raise UnsupportedSourceError(f"unsupported transcript source: {source}")


def iter_snapshots(
    sources: Iterable[Any],
    project: Any | None = None,
    *,
    host_id: str | None = None,
    parse: bool = False,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield bounded metadata candidates, or parsed snapshots when requested.

    A record can be a two-item tuple/list or a mapping containing
    ``source``/``path`` (and optional ``host_id``).  The function does not
    discover arbitrary files, which keeps collection bounded and explicit.
    Metadata mode is the default: the full transcript is parsed only by the
    caller after it has applied its collection limit.  ``parse=True`` keeps a
    compatibility path for callers that intentionally want complete parsing.
    """

    yielded = 0
    for configured in sources:
        local_host = host_id
        target_session: str | None = None
        if isinstance(configured, Mapping):
            source = configured.get("source")
            path = configured.get("path") or configured.get("source_path")
            local_host = configured.get("host_id") or local_host
            raw_target = configured.get("session_id") or configured.get("target_session_id")
            target_session = str(raw_target) if raw_target not in (None, "") else None
        elif isinstance(configured, (list, tuple)) and len(configured) >= 2:
            source, path = configured[0], configured[1]
            if len(configured) >= 3:
                local_host = configured[2]
        else:
            raise TranscriptError("configured source must contain source and path")
        if not source or not path:
            raise TranscriptError("configured source must contain source and path")
        metadata = _metadata_snapshots(
            str(source), path, host_id=local_host, session_id=target_session
        )
        for candidate in metadata:
            if limit is not None and limit >= 0 and yielded >= limit:
                return
            if not parse:
                yield candidate
                yielded += 1
                continue
            if project is None:
                raise TranscriptError("parsed snapshots require a registered project identity")
            result = parse_transcript(
                candidate["path"],
                candidate["source"],
                project,
                host_id=local_host,
                session_id=candidate.get("session_id"),
            )
            if isinstance(result, list):
                for envelope in result:
                    if limit is not None and limit >= 0 and yielded >= limit:
                        return
                    yield envelope
                    yielded += 1
            else:
                yield result
                yielded += 1


iter_source_snapshots = iter_snapshots


def _cli() -> int:
    parser = argparse.ArgumentParser(description="ACE transcript adapters")
    subparsers = parser.add_subparsers(dest="command")
    parse_parser = subparsers.add_parser("schema", help="inspect a Hermes SQLite schema")
    parse_parser.add_argument("path")
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(inspect_hermes_schema(args.path), ensure_ascii=False, sort_keys=True))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI smoke/help only
    raise SystemExit(_cli())
