"""Render a private, deterministic daily ACE status report.

The report reads existing collection, audit, and incident state. It never calls
an LLM and it never changes the incident registry. A daily report is scoped to
Europe/Paris and labels cumulative state as such.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from utils import redact_sensitive_text


PARIS = ZoneInfo("Europe/Paris")
DEFAULT_COLLECTION_STATE = Path.home() / ".codex" / "ace" / "collection-state.json"
DEFAULT_INCIDENT_STATE = Path.home() / ".codex" / "ace" / "incident-tracking.json"
DEFAULT_AUDIT_DIR = Path.home() / ".agents" / "private" / "ace" / "overengineering"
DEFAULT_REPORT_DIR = Path(
    os.environ.get(
        "ACE_DAILY_REPORT_DIR",
        str(Path.home() / ".agents" / "private" / "ace" / "daily"),
    )
)
REPORT_BRAND = os.environ.get("ACE_REPORT_BRAND", "ACE").strip() or "ACE"
WINDOW_KEYS = ("candidates", "ingested", "unexamined", "failed")
PENDING_COVERAGE_KEYS = (
    "pending_count",
    "pending_oldest_mtime",
    "pending_oldest_age_seconds",
    "freshest_candidate_mtime",
    "freshest_candidate_age_seconds",
)
ATTEMPT_SUFFIX = ".attempt.json"
FAILED_ANALYSIS_STATUSES = frozenset(
    {"model-error", "error", "failed", "failure", "degraded"}
)
STAGE_NAMES = ("source", "extraction", "analysis", "compile", "query")
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
)
_CLAIM_STATE_FIELDS = ("accepted", "refused", "applied", "verified", "effective")
_SENSITIVE_VALUE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "cookie",
        "set_cookie",
    }
)


def parse_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp and return it in Europe/Paris."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(PARIS)


def day_window(day: date | None = None) -> tuple[datetime, datetime, date]:
    local_day = day or datetime.now(PARIS).date()
    start = datetime.combine(local_day, time.min, tzinfo=PARIS)
    return start, start + timedelta(days=1), local_day


def load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    """Load a JSON object without exposing its contents in an error message."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "absent"
    except (OSError, UnicodeError):
        return {}, "lecture impossible"
    except json.JSONDecodeError:
        return {}, "JSON invalide"
    if not isinstance(value, dict):
        return {}, "racine JSON inattendue"
    return value, None


def in_window(value: Any, start: datetime, end: datetime) -> bool:
    parsed = parse_timestamp(value)
    return parsed is not None and start <= parsed < end


def as_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def optional_metric(value: Any) -> str:
    """Render a new collector metric without turning absent data into zero."""
    if value is None or value == "":
        return "inconnu"
    return str(value)


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Bound report data and redact secret-bearing keys at every level."""
    if _depth > 12:
        return "<REDACTED: depth limit>"
    if isinstance(value, str):
        return redact_sensitive_text(value).replace("\n", " ").strip()[:16000]
    if isinstance(value, (list, tuple)):
        return [redact_value(item, _depth + 1) for item in value[:500]]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in list(value.items())[:500]:
            safe_key = str(key)[:200]
            normalized_key = safe_key.strip().casefold().replace("-", "_")
            compact_key = normalized_key.replace("_", "")
            sensitive = normalized_key in _SENSITIVE_VALUE_KEYS or compact_key in {
                "credentials", "authorization", "privatekey", "apikey", "apitoken",
                "accesstoken", "refreshtoken", "clientsecret", "sessioncookie",
            }
            redacted[safe_key] = "<REDACTED>" if sensitive else redact_value(item, _depth + 1)
        return redacted
    return value


def _dated_path_value(path: Path, suffix: str = ".json") -> date | None:
    """Read a YYYY-MM-DD date from a report filename when one is present."""
    name = path.name
    if not name.endswith(suffix):
        return None
    value = name[: -len(suffix)]
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _analysis_status(report: dict[str, Any]) -> str:
    """Return the strongest explicit status, including per-conversation failures."""
    values: list[str] = []
    for key in ("analysis_status", "status"):
        value = str(report.get(key) or "").strip().lower()
        if value:
            values.append(value)
    conversations = report.get("conversations")
    if isinstance(conversations, list):
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            for key in ("analysis_status", "status"):
                value = str(conversation.get(key) or "").strip().lower()
                if value:
                    values.append(value)
    for value in values:
        if value in FAILED_ANALYSIS_STATUSES:
            return value
    return values[0] if values else ""


def _error_labels(report: dict[str, Any]) -> list[str]:
    """Expose only bounded error categories, never raw error payloads."""
    raw = report.get("errors")
    if raw in (None, "", [], {}):
        return []
    values = raw if isinstance(raw, list) else [raw]
    labels: list[str] = []
    for item in values:
        label = "error"
        if isinstance(item, dict):
            for key in ("kind", "type", "error_type", "code"):
                candidate = str(item.get(key) or "").strip()
                if candidate:
                    label = candidate
                    break
        label = redact_sensitive_text(label).replace("\n", " ").strip()[:160] or "error"
        if label not in labels:
            labels.append(label)
    return labels


def _identity_scalar(value: Any) -> str:
    """Return a bounded, non-placeholder identity component."""
    if value is None or isinstance(value, bool):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "inconnue", "none", "null"}:
        return ""
    return text


def _identity_record(value: Any, fallback_source: str = "") -> dict[str, str] | None:
    """Extract the session/revision identity used by retry filtering."""
    if not isinstance(value, dict):
        return None
    source = _identity_scalar(value.get("source") or value.get("origin") or fallback_source)
    session_id = _identity_scalar(value.get("session_id") or value.get("session"))
    conversation_id = _identity_scalar(value.get("conversation_id") or value.get("conversation"))
    revision = _identity_scalar(value.get("revision") or value.get("source_revision"))
    project_id = _identity_scalar(value.get("project_id"))
    snapshot_id = _identity_scalar(value.get("snapshot_id") or value.get("snapshot"))
    if conversation_id:
        if not session_id and ":" in conversation_id:
            parsed_source, parsed_session = conversation_id.split(":", 1)
            # A mixed/unknown report-level fallback must not override the
            # source encoded by the per-conversation key.  Keeping the
            # prefix is what lets a scoped failure suppress only its own
            # session when a failed batch contains several sources.
            if not source or source in {"unknown", "mixed"}:
                source = _identity_scalar(parsed_source)
            session_id = _identity_scalar(parsed_session)
        elif not session_id:
            session_id = conversation_id
    if not any((source, session_id, revision, project_id, snapshot_id, conversation_id)):
        return None
    return {
        "source": source,
        "session_id": session_id,
        "revision": revision,
        "project_id": project_id,
        "snapshot_id": snapshot_id,
        "conversation_id": conversation_id,
    }


def _identity_records(value: Any, fallback_source: str = "") -> list[dict[str, str]]:
    """Read explicit nested identities without treating arbitrary IDs as sessions."""
    if not isinstance(value, dict):
        return []
    found: list[dict[str, str]] = []

    def add(candidate: Any, source: str = fallback_source) -> None:
        identity = _identity_record(candidate, source)
        if identity and identity not in found:
            found.append(identity)

    add(value, fallback_source)
    for key in ("identity", "analysis_identity", "snapshot_identity"):
        add(value.get(key), fallback_source)
    for key in ("snapshot_identities", "identities"):
        nested = value.get(key)
        if isinstance(nested, list):
            for item in nested:
                add(item, fallback_source)
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        add(metadata, fallback_source)
        for key in ("identity", "analysis_identity"):
            add(metadata.get(key), fallback_source)
        for key in ("record_hashes", "record_dates", "completeness"):
            nested = metadata.get(key)
            if isinstance(nested, dict):
                for record_key in nested:
                    if isinstance(record_key, str) and record_key.strip():
                        add({"conversation_id": record_key}, fallback_source)
    return found


def _report_identities(report: dict[str, Any], fallback_source: str = "") -> list[dict[str, str]]:
    """Collect report identities while retaining multi-session boundaries."""
    found: list[dict[str, str]] = []

    def add_many(value: Any, source: str = fallback_source) -> None:
        for identity in _identity_records(value, source):
            if identity not in found:
                found.append(identity)

    add_many(report, fallback_source)
    for conversation in report.get("conversations", []):
        add_many(conversation, fallback_source)
    nested_reports = report.get("reports")
    if isinstance(nested_reports, list):
        for nested in nested_reports:
            if isinstance(nested, dict):
                add_many(nested, fallback_source)
    return found


def _failure_identities(report: dict[str, Any]) -> list[dict[str, str]]:
    """Return only identities attributed to the failed part of an analysis."""
    found: list[dict[str, str]] = []

    def add_many(value: Any, fallback_source: str = "") -> None:
        for identity in _identity_records(value, fallback_source):
            if identity not in found:
                found.append(identity)

    # The pipeline attaches session_id to per-session errors.  Those are the
    # authoritative scope for a degraded batch; the other sessions remain
    # eligible for the report.
    raw_errors = report.get("errors")
    error_values = raw_errors if isinstance(raw_errors, list) else [raw_errors]
    for error in error_values:
        if isinstance(error, dict):
            add_many(error, report_source(report))

    # A per-report/per-conversation failure can carry its own identity when no
    # separate error row exists.
    nested_reports = report.get("reports")
    if isinstance(nested_reports, list):
        for nested in nested_reports:
            if not isinstance(nested, dict):
                continue
            status = str(nested.get("analysis_status") or nested.get("status") or "").lower()
            if status in FAILED_ANALYSIS_STATUSES:
                add_many(nested, report_source(report))
    conversations = report.get("conversations")
    if isinstance(conversations, list):
        for conversation in conversations:
            if not isinstance(conversation, dict):
                continue
            status = str(conversation.get("analysis_status") or conversation.get("status") or "").lower()
            if status in FAILED_ANALYSIS_STATUSES:
                add_many(conversation, report_source(report))

    # Single-report failures commonly put source/session/revision in metadata.
    # Do not add every identity from a multi-session attempt: without an
    # attributed error that would recreate the old global masking bug.
    if not found:
        direct = _identity_records(report, report_source(report))
        if len(direct) == 1:
            found.extend(direct)
    return found


def _identity_compatible(left: dict[str, str], right: dict[str, str]) -> bool:
    for field in ("source", "project_id", "session_id"):
        a = left.get(field, "")
        b = right.get(field, "")
        if a and b and a != b:
            return False
    return True


def _identity_matches(left: dict[str, str], right: dict[str, str]) -> bool:
    """Match a failed revision/session without crossing source boundaries."""
    if not _identity_compatible(left, right):
        return False
    if left.get("snapshot_id") and right.get("snapshot_id"):
        return left["snapshot_id"] == right["snapshot_id"]
    if left.get("conversation_id") and right.get("conversation_id"):
        if left["conversation_id"] == right["conversation_id"]:
            return True
    if left.get("revision") and right.get("revision"):
        if left["revision"] == right["revision"]:
            return True
    if left.get("session_id") and right.get("session_id"):
        return left["session_id"] == right["session_id"]
    return False


def _failure_masks_report(failure: dict[str, Any], report: dict[str, Any]) -> bool:
    """Return whether a failed attempt applies to this report identity/time."""
    failure_time = failure.get("_sort_generated")
    generated = report.get("generated")
    if isinstance(failure_time, datetime) and isinstance(generated, datetime) and generated > failure_time:
        return False
    failure_ids = failure.get("identities")
    report_ids = report.get("identities")
    if not isinstance(failure_ids, list) or not isinstance(report_ids, list):
        return False
    return any(
        isinstance(left, dict)
        and isinstance(right, dict)
        and _identity_matches(left, right)
        for left in failure_ids
        for right in report_ids
    )


def _identity_context(identity: dict[str, str]) -> tuple[str, str, str]:
    return (
        identity.get("source", ""),
        identity.get("project_id", ""),
        identity.get("session_id", "") or identity.get("conversation_id", ""),
    )


def nonempty_refs(value: Any) -> list[str]:
    """Normalize explicit evidence refs without turning absence into proof."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _incident_has_source_proof(incident: dict[str, Any]) -> bool:
    """Only an incident with explicit source refs can drive a KPI/action."""
    return _success_supported_by_report(incident, incident)


def _success_supported_by_report(success: dict[str, Any], report: dict[str, Any]) -> bool:
    refs = nonempty_refs(success.get("evidence_refs"))
    if not refs:
        return False
    evidence = report.get("evidence")
    # A claim-level ref is only usable when the report carries the evidence
    # collection that resolves it.  Legacy reports without that collection
    # remain visible as unknown, never as proven success.
    if not isinstance(evidence, list):
        return False
    evidence_refs = {
        str(item.get("ref")).strip()
        for item in evidence
        if isinstance(item, dict) and str(item.get("ref") or "").strip()
    }
    return bool(evidence_refs) and set(refs).issubset(evidence_refs)


def _claim_inferred_states(item: dict[str, Any], *, refusal: bool = False) -> dict[str, bool]:
    status = str(item.get("status") or item.get("state") or "").strip().lower()
    inferred: dict[str, bool] = {field: False for field in _CLAIM_STATE_FIELDS}
    if status in {"accepted", "accept"} or item.get("accepted_at"):
        inferred["accepted"] = True
    if refusal or status in {"refused", "rejected", "declined"}:
        inferred["refused"] = True
    # Each status describes its own state.  An effective/verified status does
    # not silently promote earlier states to applied or verified.
    if status == "applied" or item.get("applied_at"):
        inferred["applied"] = True
    if (
        status == "verified"
        or item.get("verified_at")
        or str(item.get("test_status") or item.get("test_result") or "").strip().lower()
        in {"passed", "verified"}
    ):
        inferred["verified"] = True
    if status == "effective" or item.get("effective_at"):
        inferred["effective"] = True
    return inferred


def _claim_status_conflicted(item: dict[str, Any], *, refusal: bool = False) -> bool:
    inferred = _claim_inferred_states(item, refusal=refusal)
    return any(
        field in item
        and (
            not isinstance(item.get(field), bool)
            or (item.get(field) is False and inferred[field])
        )
        for field in _CLAIM_STATE_FIELDS
    )


def _claim_status_flags(item: dict[str, Any], *, refusal: bool = False) -> dict[str, bool | None]:
    """Read independent workflow states; contradictions remain unknown."""
    inferred = _claim_inferred_states(item, refusal=refusal)

    flags: dict[str, bool | None] = {}
    for field in _CLAIM_STATE_FIELDS:
        if field in item:
            explicit = item.get(field)
            if not isinstance(explicit, bool):
                flags[field] = None
            elif explicit is False and inferred[field]:
                # `accepted: false` + `status: accepted`, for example, is a
                # contradiction rather than evidence of either state.
                flags[field] = None
            else:
                flags[field] = explicit
        else:
            flags[field] = True if inferred[field] else None
    return flags


def _claim_state_counts(items: list[dict[str, Any]], *, available: bool) -> dict[str, Any]:
    """Count proposal states while preserving missing versus explicit zero."""
    result: dict[str, Any] = {"proposed": None, **{field: None for field in _CLAIM_STATE_FIELDS}}
    result["available"] = available
    counts: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    for item in items:
        if not isinstance(item, dict):
            continue
        flags = _claim_status_flags(item, refusal=bool(item.get("refused")))
        for field, enabled in flags.items():
            if enabled is not None:
                observed[field] += 1
                if enabled:
                    counts[field] += 1
        status = str(item.get("status") or item.get("state") or "").strip().lower()
        contradiction = _claim_status_conflicted(
            item, refusal=bool(item.get("refused"))
        )
        if not contradiction and not any(enabled is True for enabled in flags.values()):
            if status in {"proposed", "open", "pending"} or item.get("text") or item.get("recommendation"):
                observed["proposed"] += 1
                counts["proposed"] += 1
    if not available:
        for field in ("proposed", *_CLAIM_STATE_FIELDS):
            result[field] = None
        return result
    for field in ("proposed", *_CLAIM_STATE_FIELDS):
        if observed[field]:
            result[field] = counts[field]
    return result


def _numeric_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    return None


def _report_business_state_counts(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Read learning's state aggregates without treating unknown as no."""
    totals: Counter[str] = Counter()
    unknown_totals: Counter[str] = Counter()
    seen_fields: set[str] = set()
    for report in reports:
        sources: list[dict[str, Any]] = []
        business = report.get("business_metrics")
        if isinstance(business, dict):
            states = business.get("states")
            if isinstance(states, dict):
                sources.append(states)
        for key in ("status_counts", "state_counts"):
            value = report.get(key)
            if isinstance(value, dict):
                sources.append(value)
        for field in ("accepted", "applied", "effective", "refused", "proposed", "verified"):
            for source in sources:
                raw = source.get(field)
                if isinstance(raw, dict):
                    count = _numeric_count(
                        raw.get("yes")
                        if raw.get("yes") is not None
                        else raw.get("count")
                        if raw.get("count") is not None
                        else raw.get(f"{field}_count")
                    )
                    unknown = _numeric_count(raw.get("unknown"))
                    if unknown and count == 0 and not _numeric_count(raw.get("no")):
                        # Learning's yes=0/no=0/unknown=N describes no measured
                        # state, not N measured negative answers.
                        count = None
                else:
                    count = _numeric_count(raw)
                    unknown = None
                if count is None and unknown is None:
                    continue
                if count is not None:
                    totals[field] += count
                if unknown is not None:
                    unknown_totals[field] += unknown
                seen_fields.add(field)
                break
    return {
        "available": bool(seen_fields),
        **{
            field: totals[field] if field in seen_fields and field in totals else None
            for field in ("proposed", "accepted", "refused", "applied", "verified", "effective")
        },
        "unknown": dict(unknown_totals),
    }


def _request_acceptance_from_reports(
    reports: list[dict[str, Any]], fallback: dict[str, Any]
) -> dict[str, Any]:
    for report in reports:
        for container in (report.get("business_metrics"), report.get("metrics")):
            if not isinstance(container, dict):
                continue
            value = container.get("elapsed_request_to_accepted")
            if not isinstance(value, dict):
                continue
            average = value.get("average_seconds")
            if isinstance(average, (int, float)) and not isinstance(average, bool) and average >= 0:
                return {
                    "status": "known",
                    "count": _numeric_count(value.get("count")) or 0,
                    "average_seconds": round(float(average), 2),
                }
            if value.get("status") == "unknown":
                return {"status": "unknown", "count": 0, "average_seconds": None}
    return fallback


def _timestamp_from(item: dict[str, Any], names: tuple[str, ...]) -> datetime | None:
    for name in names:
        parsed = parse_timestamp(item.get(name))
        if parsed is not None:
            return parsed
    return None


def _request_acceptance_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    durations: list[float] = []
    for item in items:
        requested = _timestamp_from(item, ("requested_at", "proposed_at", "created_at", "request_at"))
        accepted = _timestamp_from(item, ("accepted_at", "acceptance_at"))
        if requested is None or accepted is None:
            continue
        seconds = (accepted - requested).total_seconds()
        if seconds >= 0:
            durations.append(seconds)
    if not durations:
        return {"status": "unknown", "count": 0, "average_seconds": None}
    return {
        "status": "known",
        "count": len(durations),
        "average_seconds": round(sum(durations) / len(durations), 2),
    }


def _usage_values(value: Any) -> dict[str, int] | None:
    """Extract actual token counters from a stage usage mapping."""
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    for key in TOKEN_KEYS:
        raw = value.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
            result[key] = int(raw)
    if result:
        if "total_tokens" not in result and "input_tokens" in result and "output_tokens" in result:
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
        return result
    for key in ("token_usage", "usage", "metrics"):
        nested = _usage_values(value.get(key))
        if nested:
            return nested
    return None


def _stage_usage(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate stage token usage, reporting unknown when counters are absent."""
    totals: dict[str, dict[str, int]] = {}
    for report in reports:
        seen_candidates: set[str] = set()
        candidates: list[tuple[str, Any]] = []
        for container_name in (
            "stage_usage",
            "usage_by_stage",
            "stages",
            "tokens_by_stage",
            "token_usage_by_stage",
        ):
            container = report.get(container_name)
            if isinstance(container, dict):
                candidates.extend((str(stage), value) for stage, value in container.items())
        for container_name in ("business_metrics", "metrics"):
            container = report.get(container_name)
            if not isinstance(container, dict):
                continue
            for stage_key in ("tokens_by_stage", "token_usage_by_stage", "token_usage", "usage"):
                nested = container.get(stage_key)
                if isinstance(nested, dict):
                    stage_values = nested.get("stages") or nested.get("by_stage") or nested.get("tokens_by_stage")
                    if isinstance(stage_values, dict):
                        candidates.extend((str(stage), value) for stage, value in stage_values.items())
                    elif stage_key in {"tokens_by_stage", "token_usage_by_stage"}:
                        candidates.extend((str(stage), value) for stage, value in nested.items())
        for stage in STAGE_NAMES:
            for key in (stage, f"{stage}_usage", f"{stage}_metrics"):
                if key in report:
                    candidates.append((stage, report.get(key)))
        usage = report.get("usage")
        if isinstance(usage, dict):
            stage_values = usage.get("stages") or usage.get("by_stage") or usage.get("tokens_by_stage")
            if isinstance(stage_values, dict):
                candidates.extend((str(stage), value) for stage, value in stage_values.items())
            elif any(key in usage for key in STAGE_NAMES):
                candidates.extend((stage, usage.get(stage)) for stage in STAGE_NAMES)
        for stage, value in candidates:
            parsed = _usage_values(value)
            if parsed is None:
                continue
            marker = json.dumps(
                [str(stage), parsed], ensure_ascii=False, sort_keys=True, default=str
            )
            if marker in seen_candidates:
                continue
            seen_candidates.add(marker)
            bucket = totals.setdefault(stage, {})
            for key, number in parsed.items():
                bucket[key] = bucket.get(key, 0) + number
    return {
        stage: {
            "status": "known" if stage in totals else "unknown",
            **(totals.get(stage) or {}),
        }
        for stage in STAGE_NAMES
    }


def _stage_metrics(
    reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    collection: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Project source and processing stages from fields that actually exist."""
    result = {stage: {"status": "unknown"} for stage in STAGE_NAMES}
    if collection is not None:
        result["source"] = {
            "status": "measured",
            **{key: collection.get(key) for key in WINDOW_KEYS if collection.get(key) is not None},
        }
    if reports:
        result["analysis"] = {
            "status": "ok" if not failures else "partial",
            "reports": len(reports),
        }
    elif failures:
        result["analysis"] = {"status": "failed", "reports": 0}
    for report in reports:
        coverage = report.get("coverage")
        if isinstance(coverage, dict):
            result["source"].update(
                {
                    key: coverage.get(key)
                    for key in (
                        "sessions",
                        "complete",
                        "partial",
                        "unavailable",
                        "unknown",
                    )
                    if coverage.get(key) is not None
                }
            )
            if result["source"].get("status") == "unknown":
                result["source"]["status"] = "measured"
        explicit_analysis = report.get("analysis_status") or report.get("status")
        if explicit_analysis not in (None, ""):
            result["analysis"]["status"] = str(explicit_analysis)
        for container_name in ("pipeline", "stages"):
            container = report.get(container_name)
            if isinstance(container, dict):
                for stage, value in container.items():
                    if stage not in result or not isinstance(value, dict):
                        continue
                    if value.get("status") not in (None, ""):
                        result[stage]["status"] = str(value["status"])
                    for key in ("records", "total", "completed", "failed", "pending", "available"):
                        if value.get(key) is not None:
                            result[stage][key] = value[key]
        for stage in STAGE_NAMES:
            direct_status = report.get(f"{stage}_status")
            if direct_status not in (None, ""):
                result[stage]["status"] = str(direct_status)
            direct = report.get(stage)
            if isinstance(direct, dict):
                if direct.get("status") not in (None, ""):
                    result[stage]["status"] = str(direct.get("status"))
                for key in ("records", "total", "completed", "failed", "pending", "available"):
                    if direct.get(key) is not None:
                        result[stage][key] = direct.get(key)
    return result


def _trend(current: Any, previous: Any) -> str:
    if isinstance(current, (int, float)) and not isinstance(current, bool) and isinstance(previous, (int, float)) and not isinstance(previous, bool):
        delta = current - previous
        return f"{delta:+g} ({'hausse' if delta > 0 else 'baisse' if delta < 0 else 'stable'})"
    return "inconnu"


def _claim_linked(claim: dict[str, Any], identifiers: set[str]) -> bool:
    values = {
        str(claim.get(key) or "").strip()
        for key in ("conversation_id", "session_id", "conversation")
    }
    return bool(values.intersection(identifiers))


def _selected_claims(item: dict[str, Any], field: str) -> list[dict[str, Any]]:
    """Return claims attributable to one selected conversation."""
    report = item.get("report") if isinstance(item.get("report"), dict) else {}
    conversation = item.get("conversation") if isinstance(item.get("conversation"), dict) else {}
    key = str(item.get("key") or "").strip()
    identifiers = {
        value
        for value in (
            key,
            str(conversation.get("conversation_id") or "").strip(),
            str(conversation.get("id") or "").strip(),
        )
        if value
    }
    if ":" in key:
        identifiers.add(key.split(":", 1)[1])
    report_conversations = report.get("conversations")
    allow_unscoped = isinstance(report_conversations, list) and sum(
        isinstance(value, dict) for value in report_conversations
    ) <= 1
    values: list[dict[str, Any]] = []
    raw = report.get(field)
    if isinstance(raw, list):
        values.extend(value for value in raw if isinstance(value, dict))
    nested = conversation.get(field)
    if isinstance(nested, list):
        values.extend(value for value in nested if isinstance(value, dict))
    # Native reports may keep the claim only in the nested per-snapshot row.
    nested_reports = report.get("reports")
    if isinstance(nested_reports, list):
        for nested_report in nested_reports:
            if not isinstance(nested_report, dict):
                continue
            nested_conversations = nested_report.get("conversations")
            nested_ids = {
                str(value.get("conversation_id") or value.get("id") or "").strip()
                for value in (nested_conversations if isinstance(nested_conversations, list) else [])
                if isinstance(value, dict)
            }
            if nested_ids and not nested_ids.intersection(identifiers):
                continue
            nested_values = nested_report.get(field)
            if isinstance(nested_values, list):
                values.extend(value for value in nested_values if isinstance(value, dict))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not _claim_linked(value, identifiers) and not (
            allow_unscoped
            and not any(str(value.get(key) or "").strip() for key in ("conversation_id", "session_id", "conversation"))
        ):
            continue
        marker = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if marker not in seen:
            selected.append(value)
            seen.add(marker)
    return selected


def _report_failed(report: dict[str, Any]) -> bool:
    """Identify an unvalidated analysis without treating ordinary limitations as failure."""
    return _analysis_status(report) in FAILED_ANALYSIS_STATUSES or bool(_error_labels(report))


def _failure_entry(
    path: Path,
    report: dict[str, Any],
    *,
    read_error: str | None = None,
    attempt_day: date | None = None,
    generated: datetime | None = None,
    is_attempt: bool,
) -> dict[str, Any]:
    status = _analysis_status(report)
    labels = _error_labels(report)
    identities = _failure_identities(report)
    if read_error:
        labels = [read_error, *labels]
        status = "read-error"
    if is_attempt and not labels and status not in FAILED_ANALYSIS_STATUSES:
        labels = ["attempt_non_valide"]
    if not status:
        status = "failed" if not is_attempt else "attempt_non_valide"
    report_date = attempt_day or (generated.date() if generated is not None else None)
    sort_generated = generated
    if sort_generated is None and report_date is not None:
        sort_generated = datetime.combine(report_date, time.max, tzinfo=PARIS)
    raw_errors = report.get("errors")
    if isinstance(raw_errors, list):
        error_count = len(raw_errors)
    elif raw_errors not in (None, "", {}):
        error_count = 1
    else:
        error_count = 1 if read_error else 0
    conversations = report.get("conversations")
    conversation_count = (
        sum(1 for item in conversations if isinstance(item, dict))
        if isinstance(conversations, list)
        else 0
    )
    return {
        "path": path,
        "date": report_date.isoformat() if report_date is not None else "inconnue",
        "status": status,
        "error_labels": list(dict.fromkeys(labels)),
        "error_count": error_count,
        "conversation_count": conversation_count,
        "identities": identities,
        "scope_status": "scoped" if identities else "unscoped",
        "is_attempt": is_attempt,
        "generated": generated.isoformat() if generated is not None else None,
        # Internal ordering key; never rendered or persisted.
        "_sort_generated": sort_generated or datetime.min.replace(tzinfo=PARIS),
    }


def attempt_summary_lines(audit: dict[str, Any]) -> list[str]:
    """Render nonvalidated attempts separately from the validated report view."""
    attempts = audit.get("attempts") if isinstance(audit.get("attempts"), list) else []
    latest = audit.get("latest_attempt")
    if not isinstance(latest, dict):
        latest = audit.get("latest_failure")
    if not isinstance(latest, dict):
        return []
    labels = latest.get("error_labels") if isinstance(latest.get("error_labels"), list) else []
    errors = ", ".join(str(item) for item in labels if str(item).strip()) or "aucune erreur typée"
    kind = "attempt d'audit" if latest.get("is_attempt") else "rapport d'audit"
    lines = [
        f"Dernier {kind} non validé: date={latest.get('date', 'inconnue')}; "
        f"statut={latest.get('status', 'inconnu')}; erreurs={errors}; "
        f"portée={latest.get('scope_status', 'inconnue')}; "
        f"fichier=`{latest.get('path', 'inconnu')}`.",
    ]
    if attempts:
        lines.append(f"Attempts non validés dans la fenêtre: {len(attempts)}.")
    validated = audit.get("latest_validated_file")
    if validated:
        lines.append(
            f"Dernière analyse validée conservée séparément: `{validated}`; "
            "elle n'est pas comptée comme réussite de l'analyse échouée."
        )
    else:
        lines.append("Dernière analyse validée conservée séparément: aucune dans la fenêtre.")
    if latest.get("scope_status") == "unscoped":
        lines.append(
            "Portée de l'échec non attribuée à une session/révision: les analyses valides d'autres sessions restent conservées."
        )
    return lines


def collection_snapshot(
    path: Path, start: datetime, end: datetime, project_id: str | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    state, error = load_json(path)
    errors = [f"{path}: {error}"] if error else []
    selected = state
    if project_id is not None:
        projects = state.get("projects")
        selected = projects.get(str(project_id), {}) if isinstance(projects, dict) else {}
        if not isinstance(selected, dict):
            selected = {}
    last_run = selected.get("last_run_at")
    coverage = selected.get("coverage")
    if not in_window(last_run, start, end) or not isinstance(coverage, dict):
        return None, {
            "last_run_at": last_run,
            "project_id": project_id,
            "backlog": len(selected.get("backlog", [])) if isinstance(selected.get("backlog"), list) else None,
            "selection_cursor": selected.get("selection_cursor")
            if selected.get("selection_cursor") is not None
            else None,
        }, errors
    snapshot = {
        key: as_nonnegative_int(coverage[key]) if coverage.get(key) is not None else None
        for key in WINDOW_KEYS
    }
    for key in ("calls", "unchanged", "deferred", "active"):
        snapshot[key] = as_nonnegative_int(coverage[key]) if coverage.get(key) is not None else None
    for key in PENDING_COVERAGE_KEYS:
        value = coverage.get(key)
        if key in {"pending_count"} or key.endswith("_age_seconds"):
            snapshot[key] = as_nonnegative_int(value) if value is not None else None
        elif isinstance(value, str) and value.strip():
            snapshot[key] = value.strip()
        else:
            snapshot[key] = None
    return snapshot, {
        "last_run_at": last_run,
        "project_id": project_id,
        "backlog": len(selected.get("backlog", [])) if isinstance(selected.get("backlog"), list) else None,
        "selection_cursor": selected.get("selection_cursor")
        if selected.get("selection_cursor") is not None
        else None,
    }, errors


def report_source(report: dict[str, Any]) -> str:
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    sources = metadata.get("sources")
    if isinstance(sources, list):
        values = sorted({str(item).strip() for item in sources if str(item).strip()})
        if len(values) == 1:
            return values[0]
        if values:
            return "mixed"
    return "unknown"


def _date_dimension_values(report: dict[str, Any], dimension: str) -> set[str]:
    """Read date metadata while keeping absent dimensions explicitly unknown."""
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    dimensions = metadata.get("date_dimensions")
    values: set[str] = set()
    if isinstance(dimensions, dict):
        raw = dimensions.get(dimension) or dimensions.get(f"{dimension}_dates")
        if isinstance(raw, list):
            values.update(str(item).strip() for item in raw if str(item).strip() and str(item) != "unknown")
    top_level_field = {
        "source": "source_date",
        "ingestion": "ingestion_date",
        "audit": "audit_date",
    }.get(dimension)
    if top_level_field:
        value = str(metadata.get(top_level_field) or "").strip()
        if value and value != "unknown":
            values.add(value)
    record_dates = metadata.get("record_dates")
    if isinstance(record_dates, dict):
        field = {
            "source": "source_date",
            "ingestion": "ingestion_date",
            "audit": "audit_date",
        }.get(dimension, dimension)
        for record in record_dates.values():
            if isinstance(record, dict):
                value = str(record.get(field) or "").strip()
                if value and value != "unknown":
                    values.add(value)
    return values


def _date_in_window(value: Any, start: datetime, end: datetime) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value_date = date.fromisoformat(value.strip())
    except ValueError:
        return False
    return start.date() <= value_date < end.date()


def _report_activity_in_window(report: dict[str, Any], start: datetime, end: datetime) -> bool:
    return any(
        _date_in_window(value, start, end)
        for dimension in ("source", "ingestion")
        for value in _date_dimension_values(report, dimension)
    )


def _freshness_label(values: set[str], local_day: date) -> str:
    if not values:
        return "inconnue"
    try:
        latest = max(date.fromisoformat(value) for value in values)
    except ValueError:
        return "inconnue"
    delta = (local_day - latest).days
    if delta == 0:
        return "du jour"
    if delta > 0:
        suffix = "jour" if delta == 1 else "jours"
        return f"ancienne de {delta} {suffix}"
    suffix = "jour" if delta == -1 else "jours"
    return f"future de {-delta} {suffix}"


def conversation_source(conversation: dict[str, Any], fallback: str) -> str:
    for key in ("source", "origin"):
        value = conversation.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def conversation_key(source: str, conversation_id: Any) -> str | None:
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        return None
    return f"{source}:{conversation_id.strip()}"


def severity_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1, "none": 0}.get(
        str(value or "").strip().lower(), 0
    )


def audit_reports(
    audit_dir: Path, start: datetime, end: datetime
) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {
        "files": [],
        "validated_files": [],
        "selected_files": [],
        "report_count": 0,
        "selected_report_count": 0,
        "attempts": [],
        "attempt_count": 0,
        "failed_report_count": 0,
        "latest_attempt": None,
        "latest_failure": None,
        "latest_validated_file": None,
        "record_count": 0,
        "observed_conversation_count": 0,
        "reaudit_observations": 0,
        "conversation_statuses": Counter(),
        "incident_count": 0,
        "incident_proven_count": 0,
        "incidents_without_proof": 0,
        "incident_observations": 0,
        "incident_reaudit_observations": 0,
        "incident_types": Counter(),
        "success_count": 0,
        "limitation_count": 0,
        "recommendations": [],
        "evidence_windows": 0,
        "incident_entries": [],
        # Kept as an in-memory projection for the weekly ACE renderer.  The
        # daily report does not render these fields, but retaining the latest
        # selected conversation here avoids a second, subtly different
        # date/deduplication pass in the weekly report.
        "selected_conversations": [],
        "date_dimensions": {"source": set(), "ingestion": set(), "audit": set()},
        "unknown_date_records": 0,
        "partial_completeness": 0,
        "unavailable_completeness": 0,
        "coverage_limited": False,
        "success_without_proof": 0,
        "success_without_conversation": 0,
        "claim_state_counts": {},
        "request_acceptance": {"status": "unknown", "count": 0, "average_seconds": None},
        "preferences": [],
        "recurrences": [],
        "escalations": [],
        "priorities": Counter(),
        "risks": Counter(),
        "external_research": set(),
        "stage_metrics": {},
        "usage_by_stage": {},
    }
    errors: list[str] = []
    if not audit_dir.is_dir():
        return summary, [f"{audit_dir}: absent"]

    valid_reports: list[dict[str, Any]] = []
    failure_entries: list[dict[str, Any]] = []
    attempt_entries: list[dict[str, Any]] = []
    loaded_reports: list[dict[str, Any]] = []

    def add_valid_report(
        path: Path,
        report: dict[str, Any],
        generated: datetime | None,
        activity_in_window: bool,
        source_fallback: str = "",
    ) -> None:
        source = report_source(report)
        if source == "unknown" and source_fallback:
            source = source_fallback
        sort_generated = generated or datetime.min.replace(tzinfo=PARIS)
        valid_reports.append(
            {
                "path": path,
                "report": report,
                "generated": sort_generated,
                "source": source,
                "activity_in_window": activity_in_window,
                "identities": _report_identities(report, source),
            }
        )
        for dimension in ("source", "ingestion", "audit"):
            summary["date_dimensions"][dimension].update(
                _date_dimension_values(report, dimension)
            )
        if generated is not None:
            summary["date_dimensions"]["audit"].add(generated.date().isoformat())
        metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
        for field in ("priorities", "risks"):
            values = report.get(field)
            if isinstance(values, dict):
                for label, count in values.items():
                    numeric = _numeric_count(count)
                    if numeric is not None:
                        summary[field][str(label)] += numeric
            elif isinstance(values, list):
                for value in values:
                    label = str(value or "").strip()
                    if label:
                        summary[field][label] += 1
        external_research = report.get("external_research")
        if external_research not in (None, ""):
            summary["external_research"].add(str(external_research))
        completeness = metadata.get("completeness")
        record_dates = metadata.get("record_dates") if isinstance(metadata.get("record_dates"), dict) else {}
        if isinstance(completeness, dict):
            for record_id, value in completeness.items():
                if not isinstance(value, dict):
                    continue
                observation = str(value.get("observation") or "unknown").lower()
                if observation == "partial":
                    summary["partial_completeness"] += 1
                elif observation == "unavailable":
                    summary["unavailable_completeness"] += 1
                dates = record_dates.get(record_id) if isinstance(record_dates.get(record_id), dict) else {}
                if dates.get("source_date") in {None, "", "unknown"} and dates.get("ingestion_date") in {
                    None,
                    "",
                    "unknown",
                }:
                    summary["unknown_date_records"] += 1
        limitations = report.get("limitations")
        if isinstance(limitations, list) and limitations:
            summary["coverage_limited"] = True
        if isinstance(metadata.get("coverage_limited"), bool) and metadata["coverage_limited"]:
            summary["coverage_limited"] = True

    for path in sorted(audit_dir.glob("*.json")):
        is_attempt = path.name.endswith(ATTEMPT_SUFFIX)
        attempt_day = _dated_path_value(path, ATTEMPT_SUFFIX) if is_attempt else None
        if path.name == "latest.json":
            continue
        report, error = load_json(path)
        if error:
            if is_attempt and attempt_day is not None and start.date() <= attempt_day < end.date():
                entry = _failure_entry(
                    path,
                    report,
                    read_error=error,
                    attempt_day=attempt_day,
                    is_attempt=True,
                )
                failure_entries.append(entry)
                attempt_entries.append(entry)
            elif error != "absent":
                errors.append(f"{path}: {error}")
            continue
        generated = parse_timestamp(report.get("generated_at"))
        loaded_reports.append(report)
        activity_in_window = _report_activity_in_window(report, start, end)
        if is_attempt:
            attempt_in_window = (
                attempt_day is not None and start.date() <= attempt_day < end.date()
            )
            if not attempt_in_window and generated is not None:
                attempt_in_window = start <= generated < end or activity_in_window
            if not attempt_in_window:
                continue
            entry = _failure_entry(
                path,
                report,
                attempt_day=attempt_day,
                generated=generated,
                is_attempt=True,
            )
            failure_entries.append(entry)
            attempt_entries.append(entry)
            nested_reports = report.get("reports")
            if isinstance(nested_reports, list):
                for nested in nested_reports:
                    if not isinstance(nested, dict) or _report_failed(nested):
                        continue
                    nested_report = dict(nested)
                    nested_generated = parse_timestamp(nested_report.get("generated_at")) or generated
                    if nested_generated is not None:
                        nested_report.setdefault("generated_at", nested_generated.isoformat())
                    nested_source = report_source(report)
                    loaded_reports.append(nested_report)
                    add_valid_report(
                        path,
                        nested_report,
                        nested_generated,
                        True,
                        nested_source,
                    )
            continue
        if generated is None and not activity_in_window:
            file_day = _dated_path_value(path)
            if _report_failed(report) and file_day is not None and start.date() <= file_day < end.date():
                entry = _failure_entry(
                    path,
                    report,
                    attempt_day=file_day,
                    is_attempt=False,
                )
                failure_entries.append(entry)
                nested_reports = report.get("reports")
                if isinstance(nested_reports, list):
                    for nested in nested_reports:
                        if not isinstance(nested, dict) or _report_failed(nested):
                            continue
                        nested_report = dict(nested)
                        nested_generated = parse_timestamp(nested_report.get("generated_at"))
                        if nested_generated is not None:
                            nested_report.setdefault("generated_at", nested_generated.isoformat())
                        loaded_reports.append(nested_report)
                        add_valid_report(
                            path,
                            nested_report,
                            nested_generated,
                            True,
                            report_source(report),
                        )
                continue
            errors.append(f"{path}: generated_at absent ou invalide")
            continue
        if generated is not None and not (start <= generated < end) and not activity_in_window:
            continue
        if _report_failed(report):
            entry = _failure_entry(
                path,
                report,
                generated=generated,
                is_attempt=False,
            )
            failure_entries.append(entry)
            nested_reports = report.get("reports")
            if isinstance(nested_reports, list):
                for nested in nested_reports:
                    if not isinstance(nested, dict) or _report_failed(nested):
                        continue
                    nested_report = dict(nested)
                    nested_generated = parse_timestamp(nested_report.get("generated_at")) or generated
                    if nested_generated is not None:
                        nested_report.setdefault("generated_at", nested_generated.isoformat())
                    loaded_reports.append(nested_report)
                    add_valid_report(
                        path,
                        nested_report,
                        nested_generated,
                        activity_in_window,
                        report_source(report),
                    )
            continue
        add_valid_report(path, report, generated, activity_in_window)

    summary["files"] = [item["path"] for item in valid_reports]
    summary["validated_files"] = list(summary["files"])
    summary["report_count"] = len(valid_reports)
    summary["attempts"] = sorted(
        attempt_entries,
        key=lambda item: (item["_sort_generated"], str(item["path"])),
    )
    summary["attempt_count"] = len(attempt_entries)
    summary["failed_report_count"] = len(failure_entries)
    if attempt_entries:
        summary["latest_attempt"] = max(
            attempt_entries,
            key=lambda item: (item["_sort_generated"], str(item["path"])),
        )
    if failure_entries:
        summary["latest_failure"] = max(
            failure_entries,
            key=lambda item: (item["_sort_generated"], str(item["path"])),
        )
        summary["coverage_limited"] = True
    if valid_reports:
        latest_validated = max(
            valid_reports,
            key=lambda item: (item["generated"], str(item["path"])),
        )
        summary["latest_validated_file"] = latest_validated["path"]

    # A failed attempt must not promote its partial/empty output, nor make the
    # previous validated report look like a current success.  Scope the filter
    # to the failed revision/session.  The filtering is done per conversation,
    # rather than per JSON file, because one validated batch can contain
    # several independent sessions.  A global error with no identity cannot
    # safely suppress several independent conversations; retain those reports
    # and expose the unscoped failure separately.  For the legacy one-session
    # shape only, the sole known context remains suppressible.
    known_contexts = {
        _identity_context(identity)
        for item in valid_reports
        for identity in item.get("identities", [])
        if isinstance(identity, dict) and _identity_context(identity)[2]
    }
    sole_context = next(iter(known_contexts)) if len(known_contexts) == 1 else None
    selected_reports = valid_reports
    latest_by_key: dict[str, dict[str, Any]] = {}
    for item in selected_reports:
        conversations = item["report"].get("conversations")
        if not isinstance(conversations, list):
            continue
        for index, conversation in enumerate(conversations):
            if not isinstance(conversation, dict):
                continue
            source = conversation_source(conversation, item["source"])
            key = conversation_key(source, conversation.get("conversation_id") or conversation.get("id"))
            if key is None:
                key = f"report:{item['path']}:{index}"
            candidate = {
                "key": key,
                "path": item["path"],
                "report": item["report"],
                "conversation": conversation,
                "generated": item["generated"],
                "source": source,
                "identities": _identity_records(conversation, source),
            }
            if not candidate["identities"] and len(item.get("identities", [])) == 1:
                candidate["identities"] = [
                    identity
                    for identity in item.get("identities", [])
                    if isinstance(identity, dict)
                ]
            suppressed = False
            for failure in failure_entries:
                failure_time = failure.get("_sort_generated")
                if (
                    isinstance(failure_time, datetime)
                    and isinstance(candidate.get("generated"), datetime)
                    and candidate["generated"] > failure_time
                ):
                    continue
                failure_ids = failure.get("identities")
                if isinstance(failure_ids, list) and failure_ids:
                    if any(
                        isinstance(left, dict)
                        and isinstance(right, dict)
                        and _identity_matches(left, right)
                        for left in failure_ids
                        for right in candidate.get("identities", [])
                    ):
                        suppressed = True
                        break
                elif sole_context is not None and any(
                    isinstance(identity, dict)
                    and _identity_context(identity) == sole_context
                    for identity in candidate.get("identities", [])
                ):
                    suppressed = True
                    break
            if suppressed:
                continue
            prior = latest_by_key.get(key)
            if prior is None or (candidate["generated"], str(candidate["path"])) > (
                prior["generated"], str(prior["path"])
            ):
                latest_by_key[key] = candidate

    selected_conversations = list(latest_by_key.values())
    selected_paths = {item["path"] for item in selected_conversations}

    def selected_successes(item: dict[str, Any]) -> list[Any]:
        report_successes = item["report"].get("successes")
        if not isinstance(report_successes, list):
            return []
        conversation = item["conversation"]
        identifiers = {
            str(value).strip()
            for value in (
                item["key"],
                conversation.get("conversation_id"),
                conversation.get("id"),
            )
            if value is not None and str(value).strip()
        }
        key = item["key"]
        if ":" in key:
            identifiers.add(key.split(":", 1)[1])
        return [
            success
            for success in report_successes
            if isinstance(success, dict)
            and str(success.get("conversation_id") or "").strip() in identifiers
        ]

    def proven_successes(item: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            success
            for success in selected_successes(item)
            if _success_supported_by_report(success, item["report"])
        ]

    def selected_recurrences(item: dict[str, Any]) -> list[dict[str, Any]]:
        values = _selected_claims(item, "recurrences")
        if values:
            return values
        raw = item["report"].get("recurrences")
        if isinstance(raw, dict):
            return [raw]
        return [value for value in raw if isinstance(value, dict)] if isinstance(raw, list) else []

    summary["selected_conversations"] = [
        {
            "key": item["key"],
            "path": item["path"],
            "generated": item["generated"].isoformat(),
            "source": item["source"],
            "conversation": redact_value(item["conversation"]),
            "metadata": redact_value(
                item["report"].get("metadata")
                if isinstance(item["report"].get("metadata"), dict)
                else {}
            ),
            "successes": redact_value(
                selected_successes(item)
            ),
            "observations": redact_value(_selected_claims(item, "observations")),
            "preferences": redact_value(_selected_claims(item, "preferences")),
            "recommendations_detail": redact_value(_selected_claims(item, "recommendations")),
            "decisions": redact_value(_selected_claims(item, "decisions")),
            "refusals": redact_value(_selected_claims(item, "refusals")),
            "recurrences": redact_value(selected_recurrences(item)),
            "usage": redact_value(item["report"].get("usage")),
            "stage_usage": redact_value(
                item["report"].get("stage_usage")
                or item["report"].get("usage_by_stage")
                or {}
            ),
            "evidence": redact_value(
                item["report"].get("evidence")
                if isinstance(item["report"].get("evidence"), list)
                else None
            ),
            "limitations": redact_value(
                item["report"].get("limitations")
                if isinstance(item["report"].get("limitations"), list)
                else []
            ),
        }
        for item in selected_conversations
    ]
    summary["selected_files"] = sorted(selected_paths)
    summary["selected_report_count"] = len(selected_paths)
    summary["record_count"] = len(selected_conversations)
    summary["observed_conversation_count"] = sum(
        1
        for item in valid_reports
        for conversation in (item["report"].get("conversations") or [])
        if isinstance(conversation, dict)
    )
    summary["reaudit_observations"] = max(
        0, summary["observed_conversation_count"] - summary["record_count"]
    )

    selected_by_key = latest_by_key
    incident_entries: list[dict[str, Any]] = []
    raw_incident_count = 0
    for item in valid_reports:
        report = item["report"]
        conversations = report.get("conversations") if isinstance(report.get("conversations"), list) else []
        conversation_by_id = {
            str(conversation.get("conversation_id") or conversation.get("id")): conversation
            for conversation in conversations
            if isinstance(conversation, dict)
            and (conversation.get("conversation_id") or conversation.get("id"))
        }
        incidents = report.get("incidents")
        if not isinstance(incidents, list):
            continue
        raw_incident_count += sum(1 for incident in incidents if isinstance(incident, dict))
        if item["path"] not in selected_paths:
            continue
        for index, incident in enumerate(incidents):
            if not isinstance(incident, dict):
                continue
            conversation_id = incident.get("conversation_id")
            conversation = conversation_by_id.get(str(conversation_id))
            source = conversation_source(conversation or {}, item["source"])
            key = conversation_key(source, conversation_id)
            if key is not None and key in selected_by_key and selected_by_key[key].get("path") != item["path"]:
                continue
            level = incident.get("severity") or (conversation or {}).get("level")
            incident_entries.append(
                {
                    "path": item["path"],
                    "index": index,
                    "incident": {
                        **redact_value(incident),
                        # Resolve this claim against its parent report; never
                        # accept a catalog supplied by the incident itself.
                        "evidence": [
                            {"ref": proof["ref"]}
                            for proof in (report.get("evidence") or [])
                            if isinstance(proof, dict)
                            and isinstance(proof.get("ref"), str)
                            and proof["ref"] in nonempty_refs(incident.get("evidence_refs"))
                        ] if isinstance(report.get("evidence"), list) else [],
                    },
                    "generated": item["generated"],
                    "severity": severity_rank(level),
                    "conversation_key": key,
                }
            )

    incident_entries.sort(
        key=lambda entry: (-entry["severity"], -entry["generated"].timestamp(), str(entry["path"]), entry["index"])
    )
    summary["incident_entries"] = incident_entries
    summary["incident_count"] = len(incident_entries)
    summary["incident_proven_count"] = sum(
        1
        for entry in incident_entries
        if isinstance(entry.get("incident"), dict)
        and _incident_has_source_proof(entry["incident"])
    )
    summary["incidents_without_proof"] = (
        summary["incident_count"] - summary["incident_proven_count"]
    )
    summary["incident_observations"] = raw_incident_count
    summary["incident_reaudit_observations"] = max(0, raw_incident_count - len(incident_entries))
    summary["conversation_statuses"].update(
        str(item["conversation"].get("status"))
        for item in selected_conversations
        if item["conversation"].get("status")
    )
    summary["incident_types"].update(
        redact_sensitive_text(str(entry["incident"].get("type"))).strip()
        for entry in incident_entries
        if entry["incident"].get("type") and _incident_has_source_proof(entry["incident"])
    )
    for entry in incident_entries:
        if not _incident_has_source_proof(entry["incident"]):
            continue
        recommendation = str(entry["incident"].get("recommendation") or "").strip()
        if recommendation and recommendation not in summary["recommendations"]:
            summary["recommendations"].append(recommendation)

    selected_claims = summary["selected_conversations"]
    recommendation_items = [
        item
        for selected in selected_claims
        for item in selected.get("recommendations_detail", [])
        if isinstance(item, dict)
    ]
    refusal_items = [
        item
        for selected in selected_claims
        for item in selected.get("refusals", [])
        if isinstance(item, dict)
    ]
    summary["claim_state_counts"] = _claim_state_counts(
        [*recommendation_items, *refusal_items],
        available=bool(recommendation_items or refusal_items),
    )
    business_reports = [
        item["report"] for item in valid_reports if item["path"] in selected_paths
    ] or [item["report"] for item in valid_reports]
    business_states = _report_business_state_counts(business_reports)
    if business_states.get("available"):
        current_states = summary["claim_state_counts"]
        current_states["available"] = True
        current_states["unknown"] = business_states.get("unknown", {})
        for field in ("proposed", "accepted", "refused", "applied", "verified", "effective"):
            if business_states.get(field) is not None:
                current_states[field] = business_states[field]
    summary["request_acceptance"] = _request_acceptance_from_reports(
        business_reports,
        _request_acceptance_metrics(recommendation_items),
    )
    preference_values: list[dict[str, Any]] = []
    preference_seen: set[str] = set()
    for selected in selected_claims:
        for field in ("observations", "preferences"):
            values = selected.get(field, [])
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or item.get("type") or "").strip().lower()
                if field == "preferences" and not kind:
                    kind = "preference"
                if kind != "preference":
                    continue
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in preference_seen:
                    preference_values.append(redact_value(item))
                    preference_seen.add(marker)
    summary["preferences"] = preference_values
    summary["recurrences"] = [
        redact_value(item)
        for selected in selected_claims
        for item in selected.get("recurrences", [])
        if isinstance(item, dict)
    ]
    summary["escalations"] = [
        redact_value(item)
        for item in [*recommendation_items, *refusal_items]
        if item.get("escalation") is True
        or item.get("escalate") is True
        or item.get("requires_authorization") is True
    ]
    summary["stage_metrics"] = _stage_metrics(
        [item["report"] for item in valid_reports if item["path"] in selected_paths]
        or [item["report"] for item in valid_reports],
        failure_entries,
    )
    usage_reports = [
        item["report"] for item in valid_reports if item["path"] in selected_paths
    ] or [item["report"] for item in valid_reports] or loaded_reports
    summary["usage_by_stage"] = _stage_usage(usage_reports)

    for report_item in valid_reports:
        if report_item["path"] not in selected_paths:
            # An orphan claim must remain visible even when no conversation
            # was selected.  Do not count older, valid linked successes from
            # superseded reports as orphaned merely because they were deduped.
            declared_ids = {
                str(conv.get("conversation_id") or conv.get("id") or "").strip()
                for conv in (report_item["report"].get("conversations") or [])
                if isinstance(conv, dict)
            } - {""}
            orphan_successes = report_item["report"].get("successes") or []
            summary["success_without_conversation"] += sum(
                1 for claim in orphan_successes
                if isinstance(claim, dict)
                and str(claim.get("conversation_id") or "").strip() not in declared_ids
            ) if isinstance(orphan_successes, list) else 0
            continue
        metadata = report_item["report"].get("metadata") if isinstance(report_item["report"].get("metadata"), dict) else {}
        summary["evidence_windows"] += as_nonnegative_int(metadata.get("evidence_window_count"))
        if as_nonnegative_int(metadata.get("evidence_window_count")) >= as_nonnegative_int(
            metadata.get("evidence_window_limit")
        ) > 0:
            summary["coverage_limited"] = True
        limitations = report_item["report"].get("limitations")
        if isinstance(limitations, list):
            summary["limitation_count"] += len(limitations)
        successes = report_item["report"].get("successes")
        if isinstance(successes, list):
            selected = []
            proven = []
            selected_ids: set[str] = set()
            for selected_item in selected_conversations:
                if selected_item["path"] != report_item["path"]:
                    continue
                selected_ids.update(
                    str(value).strip()
                    for value in (
                        selected_item["key"],
                        selected_item["conversation"].get("conversation_id"),
                        selected_item["conversation"].get("id"),
                    )
                    if value is not None and str(value).strip()
                )
                if ":" in selected_item["key"]:
                    selected_ids.add(selected_item["key"].split(":", 1)[1])
            selected = [
                value
                for value in successes
                if isinstance(value, dict)
                and str(value.get("conversation_id") or "").strip() in selected_ids
            ]
            proven = [
                value
                for value in selected
                if _success_supported_by_report(value, report_item["report"])
            ]
            summary["success_count"] += len(proven)
            summary["success_without_proof"] += max(0, len(selected) - len(proven))
            summary["success_without_conversation"] += sum(
                1
                for value in successes
                if isinstance(value, dict)
                and str(value.get("conversation_id") or "").strip()
                not in selected_ids
            )
    summary["date_dimensions"] = {
        key: sorted(values) for key, values in summary["date_dimensions"].items()
    }
    summary["priorities"] = dict(summary["priorities"])
    summary["risks"] = dict(summary["risks"])
    summary["external_research"] = sorted(summary["external_research"])
    if errors:
        summary["coverage_limited"] = True
    return summary, errors


_INCIDENT_CORRECTION_REF_KEYS = (
    "correction_evidence_refs",
    "application_evidence_refs",
    "applied_evidence_refs",
    "correction_proof_refs",
)
_INCIDENT_VERIFICATION_REF_KEYS = (
    "verification_evidence_refs",
    "test_evidence_refs",
    "verification_proof_refs",
)


def _refs_from_incident(incident: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    refs: list[str] = []
    for name in names:
        refs.extend(nonempty_refs(incident.get(name)))
    return list(dict.fromkeys(refs))


def incident_tracking(
    path: Path, start: datetime, end: datetime
) -> tuple[dict[str, Any], list[str]]:
    state, error = load_json(path)
    errors = [f"{path}: {error}"] if error else []
    registry_available = error is None and isinstance(state.get("incidents"), dict)
    registry = state.get("incidents") if registry_available else {}
    counts = Counter()
    proof_counts = Counter()
    state_seen = Counter()
    state_unknown = Counter()
    today = 0
    correction_events: list[dict[str, Any]] = []

    def has_refs(incident: dict[str, Any], names: tuple[str, ...]) -> bool:
        for name in names:
            value = incident.get(name)
            if isinstance(value, list) and any(str(item).strip() for item in value):
                return True
            if isinstance(value, str) and value.strip():
                return True
        return False

    for registry_id, incident in registry.items():
        if not isinstance(incident, dict):
            continue
        flags = _claim_status_flags(incident)
        accepted = flags["accepted"] is True
        refused = flags["refused"] is True
        applied = flags["applied"] is True
        verified = flags["verified"] is True
        effective = flags["effective"] is True
        for field, enabled in (
            ("accepted", accepted),
            ("refused", refused),
            ("applied", applied),
            ("verified", verified),
            ("effective", effective),
        ):
            if flags[field] is None:
                state_unknown[field] += 1
            else:
                state_seen[field] += 1
            if enabled:
                counts[field] += 1
        if verified or applied or effective:
            pass
        elif not accepted and not refused and not _claim_status_conflicted(incident):
            counts["proposed"] += 1
        cause = incident.get("cause") if isinstance(incident.get("cause"), dict) else {}
        cause_proof = has_refs(
            incident,
            ("cause_evidence_refs", "cause_proof_refs", "cause_proof"),
        ) or has_refs(cause, ("evidence_refs", "proof_refs", "proof"))
        correction_proof = has_refs(
            incident,
            (
                "correction_evidence_refs",
                "application_evidence_refs",
                "applied_evidence_refs",
                "correction_proof_refs",
            ),
        )
        verification_proof = has_refs(
            incident,
            (
                "verification_evidence_refs",
                "test_evidence_refs",
                "verification_proof_refs",
            ),
        )
        if cause.get("status") == "verified" and cause_proof:
            proof_counts["cause_proven"] += 1
        elif cause.get("status") == "verified":
            proof_counts["cause_without_proof"] += 1
        if applied:
            if correction_proof:
                proof_counts["correction_proven"] += 1
            else:
                proof_counts["applied_without_proof"] += 1
        if verified:
            if verification_proof:
                proof_counts["verification_proven"] += 1
            else:
                proof_counts["verified_without_proof"] += 1
        if in_window(incident.get("last_seen_at"), start, end):
            today += 1
        correction_time = _timestamp_from(
            incident,
            ("effective_at", "verified_at", "applied_at", "updated_at"),
        )
        if applied or verified or effective:
            correction_events.append(
                {
                    "id": str(incident.get("id") or registry_id or "").strip(),
                    "type": str(incident.get("type") or "").strip(),
                    "signature": str(
                        incident.get("signature")
                        or incident.get("fingerprint")
                        or incident.get("dedupe_key")
                        or ""
                    ).strip(),
                    "conversation_id": str(
                        incident.get("conversation_id") or incident.get("session_id") or ""
                    ).strip(),
                    "correction_at": correction_time.isoformat() if correction_time else None,
                    "correction_refs": _refs_from_incident(incident, _INCIDENT_CORRECTION_REF_KEYS),
                    "verification_refs": _refs_from_incident(incident, _INCIDENT_VERIFICATION_REF_KEYS),
                }
            )
    values: dict[str, Any] = {
        "proposed": counts["proposed"] if registry_available else None,
        "accepted": counts["accepted"] if registry_available and state_seen["accepted"] else None,
        "refused": counts["refused"] if registry_available and state_seen["refused"] else None,
        "applied": counts["applied"] if registry_available and state_seen["applied"] else None,
        "verified": counts["verified"] if registry_available and state_seen["verified"] else None,
        "effective": counts["effective"] if registry_available and state_seen["effective"] else None,
        "registry_count": sum(
            1 for incident in registry.values() if isinstance(incident, dict)
        )
        if registry_available
        else None,
        "registry_available": registry_available,
        "state_available": {
            field: bool(registry_available and state_seen[field])
            for field in _CLAIM_STATE_FIELDS
        },
        "state_unknown": {
            field: state_unknown[field] if registry_available and state_unknown[field] else None
            for field in _CLAIM_STATE_FIELDS
        },
        "observed_today": today if registry_available else None,
        "closed_automatically": 0 if registry_available else None,
        "correction_events": correction_events,
        "cause_proven": proof_counts["cause_proven"] if registry_available else None,
        "cause_without_proof": proof_counts["cause_without_proof"] if registry_available else None,
        "correction_proven": proof_counts["correction_proven"] if registry_available else None,
        "applied_without_proof": proof_counts["applied_without_proof"] if registry_available else None,
        "verification_proven": proof_counts["verification_proven"] if registry_available else None,
        "verified_without_proof": proof_counts["verified_without_proof"] if registry_available else None,
    }
    return values, errors


def post_correction_recurrences(
    audit: dict[str, Any], tracking: dict[str, Any]
) -> dict[str, Any]:
    """Compare later incident observations with explicitly evidenced fixes."""
    events = [
        event
        for event in tracking.get("correction_events", [])
        if isinstance(event, dict)
    ]
    if not events:
        return {
            "status": "unknown",
            "count": None,
            "observations": 0,
            "reason": "aucune correction enregistrée",
        }
    evidenced = [event for event in events if nonempty_refs(event.get("correction_refs"))]
    if not evidenced:
        return {
            "status": "unknown",
            "count": None,
            "observations": 0,
            "reason": "correction enregistrée sans preuve d'application",
        }
    timed = [event for event in evidenced if parse_timestamp(event.get("correction_at"))]
    if not timed:
        return {
            "status": "unknown",
            "count": None,
            "observations": 0,
            "reason": "date d'application vérifiable indisponible",
        }
    recurrences: list[dict[str, Any]] = []
    for entry in audit.get("incident_entries", []):
        if not isinstance(entry, dict):
            continue
        incident = entry.get("incident") if isinstance(entry.get("incident"), dict) else {}
        observed_at = entry.get("generated")
        if not isinstance(observed_at, datetime):
            observed_at = parse_timestamp(observed_at)
        if observed_at is None:
            continue
        incident_id = str(incident.get("id") or "").strip()
        conversation_id = str(
            incident.get("conversation_id") or incident.get("session_id") or ""
        ).strip()
        for event in timed:
            correction_at = parse_timestamp(event.get("correction_at"))
            if correction_at is None or observed_at <= correction_at:
                continue
            # A recurrence is a sourced observation, not a repeated label.
            # Without an incident evidence ref there is no post-fix signal to
            # compare, even when identifiers happen to match.
            if not _incident_has_source_proof(incident):
                continue
            same_id = bool(incident_id and event.get("id") and incident_id == event.get("id"))
            same_signature = bool(
                incident.get("signature")
                and event.get("signature")
                and str(incident.get("signature")) == str(event.get("signature"))
            )
            same_conversation = bool(
                conversation_id
                and event.get("conversation_id")
                and conversation_id == event.get("conversation_id")
            )
            if same_id or (same_signature and same_conversation):
                recurrences.append(
                    {
                        "type": str(incident.get("type") or "incident"),
                        "observed_at": observed_at.isoformat(),
                        "correction_at": correction_at.isoformat(),
                    }
                )
                break
    status = "known" if len(timed) == len(evidenced) else "partial"
    return {
        "status": status,
        "count": len(recurrences),
        "observations": len(recurrences),
        "corrections_considered": len(timed),
        "reason": "comparaison limitée aux incidents datés et aux corrections avec preuve d'application",
        "items": recurrences,
    }


def reference(path: Path, anchor: str) -> str:
    return f"`{path}` (JSON pointer `{anchor}`)"


def collection_coverage_pointer(project_id: str | None = None) -> str:
    if project_id is None:
        return "/coverage"
    escaped = str(project_id).replace("~", "~0").replace("/", "~1")
    return f"/projects/{escaped}/coverage"


def json_proof_link(path: Path, index: int) -> str:
    return f"[preuve JSON](<{path}#/incidents/{index}>)"


def incident_priority(entry: dict[str, Any]) -> dict[str, Any]:
    incident = entry.get("incident") if isinstance(entry.get("incident"), dict) else {}
    cause = incident.get("cause") if isinstance(incident.get("cause"), dict) else {}
    path = entry.get("path")
    index = entry.get("index")
    if not isinstance(path, Path) or not isinstance(index, int):
        return {"kind": "incident", "title": "Incident non référencé", "detail": "Preuve de source indisponible."}
    cause_proof = cause.get("evidence_refs") or incident.get("cause_evidence_refs") or incident.get("cause_proof_refs")
    correction_proof = (
        incident.get("correction_evidence_refs")
        or incident.get("application_evidence_refs")
        or incident.get("applied_evidence_refs")
        or incident.get("correction_proof_refs")
    )
    return {
        "kind": "incident",
        "title": str(incident.get("type") or "Incident sans type"),
        "priority": str(incident.get("priority") or "inconnue"),
        "risk": str(incident.get("risk") or "inconnu"),
        "expected": str(incident.get("expected") or "non établi"),
        "observed": str(incident.get("observed") or "non établi"),
        "cause": f"{cause.get('status', 'unknown')}: {cause.get('summary', 'non établie')}",
        "recommendation": str(incident.get("recommendation") or "aucune correction proposée"),
        "test": str(incident.get("test") or "aucun test proposé"),
        "cause_proof": "présente" if cause_proof else "non fournie",
        "correction_proof": "présente" if correction_proof else "non fournie",
        "proof": json_proof_link(path, index),
    }


def build_priorities(
    coverage: dict[str, Any] | None,
    audit: dict[str, Any],
    tracking: dict[str, Any],
    collection_path: Path,
    audit_paths: list[Path],
    incident_path: Path,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    proven_entries = [
        entry
        for entry in audit.get("incident_entries", [])
        if isinstance(entry, dict)
        and isinstance(entry.get("incident"), dict)
        and _incident_has_source_proof(entry["incident"])
    ]
    priorities = [incident_priority(entry) for entry in proven_entries[:3]]
    coverage_ref = reference(collection_path, collection_coverage_pointer(project_id))
    incident_ref = reference(incident_path, "/incidents")
    if len(priorities) < 3:
        if coverage and coverage["unexamined"]:
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Couverture non examinée",
                    "detail": f"Examiner les {coverage['unexamined']} contenus non examinés du dernier passage daté ({coverage_ref}).",
                }
            )
        elif coverage and coverage["failed"]:
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Captures en échec",
                    "detail": f"Reprendre les {coverage['failed']} captures en échec ({coverage_ref}).",
                }
            )
        else:
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Mesure de couverture",
                    "detail": f"Conserver une mesure de couverture datée ({coverage_ref}).",
                }
            )
    if len(priorities) < 3:
        if coverage and (coverage["failed"] or coverage["deferred"]):
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Captures à reprendre",
                    "detail": (
                        f"Traiter les captures en échec ou différées: {coverage['failed']} en échec, "
                        f"{coverage['deferred']} différées ({coverage_ref})."
                    ),
                }
            )
        elif audit.get("limitation_count") and audit_paths:
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Limites documentées",
                    "detail": (
                        f"Réduire les limites documentées dans les rapports du jour "
                        f"({reference(audit_paths[-1], '/limitations')})."
                    ),
                }
            )
        else:
            priorities.append(
                {
                    "kind": "coverage",
                    "title": "Nouvelles preuves",
                    "detail": f"Vérifier les nouvelles preuves avant toute conclusion ({coverage_ref}).",
                }
            )
    if len(priorities) < 3:
        priorities.append(
            {
                "kind": "tracking",
                "title": "Suivi des corrections",
                "detail": f"Revoir les {optional_metric(tracking.get('proposed'))} corrections proposées sans clôture automatique ({incident_ref}).",
            }
        )
    return priorities[:3]


def render_stage_metrics(
    stage_metrics: dict[str, Any], usage_by_stage: dict[str, Any]
) -> list[str]:
    """Render pipeline stages with explicit unknowns and observed counters."""
    lines = ["| Étape | Statut | Mesures observées |", "|---|---|---|"]
    for stage in STAGE_NAMES:
        metric = stage_metrics.get(stage) if isinstance(stage_metrics, dict) else None
        metric = metric if isinstance(metric, dict) else {}
        status = optional_metric(metric.get("status"))
        observed = [
            f"{key}={optional_metric(metric.get(key))}"
            for key in ("candidates", "ingested", "unexamined", "failed", "records", "total", "completed", "pending")
            if metric.get(key) is not None
        ]
        lines.append(f"| {stage} | {status} | {', '.join(observed) if observed else 'inconnu'} |")
    lines.append("")
    lines.append("| Tokens par étape | Statut | Compteurs observés |")
    lines.append("|---|---|---|")
    for stage in STAGE_NAMES:
        usage = usage_by_stage.get(stage) if isinstance(usage_by_stage, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        status = optional_metric(usage.get("status"))
        counters = [
            f"{key}={optional_metric(usage.get(key))}"
            for key in TOKEN_KEYS
            if usage.get(key) is not None
        ]
        lines.append(f"| {stage} | {status} | {', '.join(counters) if counters else 'inconnu'} |")
    return lines


def render_claim_states(audit: dict[str, Any]) -> list[str]:
    """Render recommendation states independently, including refusals."""
    counts = audit.get("claim_state_counts")
    counts = counts if isinstance(counts, dict) else {}
    unknown = counts.get("unknown") if isinstance(counts.get("unknown"), dict) else {}
    lines = ["| État des propositions | Nombre |", "|---|---:|"]
    for field, label in (
        ("proposed", "proposées"),
        ("accepted", "acceptées"),
        ("refused", "refusées"),
        ("applied", "appliquées"),
        ("verified", "vérifiées"),
        ("effective", "effectives"),
    ):
        value = optional_metric(counts.get(field))
        unknown_count = _numeric_count(unknown.get(field))
        if unknown_count:
            value += f" (inconnues={unknown_count})"
        lines.append(f"| {label} | {value} |")
    request = audit.get("request_acceptance")
    request = request if isinstance(request, dict) else {}
    average = request.get("average_seconds")
    lines.append(
        "Temps demande → accepted: "
        + (f"{average} secondes (n={request.get('count')})" if average is not None else "inconnu")
        + "."
    )
    return lines


def render_signal_counts(audit: dict[str, Any], post_correction: dict[str, Any]) -> list[str]:
    preferences = audit.get("preferences") if isinstance(audit.get("preferences"), list) else []
    recurrences = audit.get("recurrences") if isinstance(audit.get("recurrences"), list) else []
    escalations = audit.get("escalations") if isinstance(audit.get("escalations"), list) else []
    recurrence_count = post_correction.get("count") if isinstance(post_correction, dict) else None
    priorities = audit.get("priorities") if isinstance(audit.get("priorities"), dict) else {}
    risks = audit.get("risks") if isinstance(audit.get("risks"), dict) else {}
    external = audit.get("external_research") if isinstance(audit.get("external_research"), list) else []
    lines = [
        "Signaux métier explicites: "
        f"préférences={len(preferences) if preferences else 'inconnu'}, "
        f"récurrences déclarées={len(recurrences) if recurrences else 'inconnu'}, "
        f"escalades={len(escalations) if escalations else 'inconnu'}, "
        f"récidives après correction={optional_metric(recurrence_count)}.",
    ]
    lines.append(
        "Incidents: "
        f"prouvés={optional_metric(audit.get('incident_proven_count'))}, "
        f"sans preuve, hors KPI/priorités={optional_metric(audit.get('incidents_without_proof'))}."
    )
    lines.append(
        "Priorités observées: "
        + (", ".join(f"{key}={value}" for key, value in sorted(priorities.items())) or "inconnues")
        + "; risques observés: "
        + (", ".join(f"{key}={value}" for key, value in sorted(risks.items())) or "inconnus")
        + "."
    )
    lines.append(
        "Recherche externe: "
        + (", ".join(str(value) for value in external) if external else "inconnue")
        + "."
    )
    return lines


def render_auto_improvement(
    audit: dict[str, Any],
    tracking: dict[str, Any],
    post_correction: dict[str, Any],
) -> list[str]:
    """Render the evidence-backed auto-improvement work in one visible block.

    Suggestions, their references, recorded workflow states, and verification evidence are kept
    separate so a proposal cannot look like an applied fix.  This renderer is
    intentionally read-only and never changes the incident registry.
    """
    lines = [
        "## Auto-amélioration",
        "",
        "Ce bloc sépare les signaux détectés, les suggestions produites et les actions réellement enregistrées.",
        "",
        "### Signaux détectés",
        f"- Conversations analysées: {optional_metric(audit.get('record_count'))}; observations: {optional_metric(audit.get('observed_conversation_count'))}.",
        f"- Incidents retenus: {optional_metric(audit.get('incident_count'))}; prouvés: {optional_metric(audit.get('incident_proven_count'))}; sans preuve: {optional_metric(audit.get('incidents_without_proof'))}.",
    ]
    incident_types = audit.get("incident_types")
    if isinstance(incident_types, dict) and incident_types:
        rendered_types = ", ".join(
            f"{str(label)}={optional_metric(count)}"
            for label, count in sorted(incident_types.items(), key=lambda item: str(item[0]))
        )
    else:
        rendered_types = "inconnus"
    lines.append(f"- Types détectés: {rendered_types}.")

    suggestions: list[tuple[str, str, int, int]] = []
    suggestion_seen: set[str] = set()
    selected = audit.get("selected_conversations")
    if isinstance(selected, list):
        for conversation in selected:
            if not isinstance(conversation, dict):
                continue
            details = conversation.get("recommendations_detail")
            if not isinstance(details, list):
                continue
            for item in details:
                if not isinstance(item, dict):
                    continue
                text = str(
                    item.get("text")
                    or item.get("recommendation")
                    or item.get("summary")
                    or ""
                ).strip()
                if text and text not in suggestion_seen:
                    suggestion_seen.add(text)
                    suggestions.append(
                        (
                            text,
                            str(item.get("type") or "suggestion").strip() or "suggestion",
                            len(nonempty_refs(item.get("evidence_refs"))),
                            len(nonempty_refs(item.get("message_ids"))),
                        )
                    )
    for item in audit.get("recommendations", []):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("recommendation") or "").strip()
            kind = str(item.get("type") or "suggestion").strip() or "suggestion"
            evidence_count = len(nonempty_refs(item.get("evidence_refs")))
            message_count = len(nonempty_refs(item.get("message_ids")))
        else:
            text = str(item or "").strip()
            kind = "suggestion"
            evidence_count = 0
            message_count = 0
        if text and text not in suggestion_seen:
            suggestion_seen.add(text)
            suggestions.append((text, kind, evidence_count, message_count))

    lines.extend(["", "### Suggestions proposées"])
    if suggestions:
        for text, kind, evidence_count, message_count in suggestions[:10]:
            lines.append(
                f"- [{kind}] {text} (preuves={evidence_count}; messages={message_count})."
            )
        if len(suggestions) > 10:
            lines.append(f"- … {len(suggestions) - 10} suggestion(s) supplémentaire(s) non affichée(s).")
    else:
        lines.append("- Aucune suggestion exploitable avec preuve conservée.")

    lines.extend(
        [
            "",
            "### Actions enregistrées",
            f"- Proposées: {optional_metric(tracking.get('proposed'))}; acceptées: {optional_metric(tracking.get('accepted'))}; appliquées: {optional_metric(tracking.get('applied'))}; vérifiées: {optional_metric(tracking.get('verified'))}; effectives: {optional_metric(tracking.get('effective'))}.",
        ]
    )
    events = tracking.get("correction_events")
    events = events if isinstance(events, list) else []
    evidenced_events = [
        event
        for event in events
        if isinstance(event, dict)
        and (nonempty_refs(event.get("correction_refs")) or nonempty_refs(event.get("verification_refs")))
    ]
    if not tracking.get("registry_available", True):
        lines.append("- Registre des actions indisponible; les états appliqué/vérifié/effectif restent inconnus.")
    elif evidenced_events:
        for event in evidenced_events[:10]:
            label = str(event.get("type") or event.get("id") or "correction").strip()
            refs = nonempty_refs(event.get("verification_refs")) or nonempty_refs(event.get("correction_refs"))
            lines.append(f"- {label}: preuve enregistrée ({len(refs)} référence(s)).")
        if len(evidenced_events) > 10:
            lines.append(f"- … {len(evidenced_events) - 10} action(s) supplémentaire(s) non affichée(s).")
    else:
        lines.append("- Aucune correction appliquée ou vérifiée avec preuve dans cette fenêtre.")

    recurrence_count = post_correction.get("count") if isinstance(post_correction, dict) else None
    recurrence_reason = (
        post_correction.get("reason", "preuve indisponible")
        if isinstance(post_correction, dict)
        else "preuve indisponible"
    )
    lines.append(
        f"- Récidive après correction: {optional_metric(recurrence_count)} ({recurrence_reason})."
    )
    lines.append("- Aucune suggestion n'est appliquée automatiquement; une autorisation et une preuve restent nécessaires.")
    return lines


def render_trends(
    audit: dict[str, Any],
    previous_audit: dict[str, Any],
    coverage: dict[str, Any] | None,
    previous_coverage: dict[str, Any] | None,
) -> str:
    current_available = bool(
        audit.get("files")
        or audit.get("attempt_count")
        or audit.get("failed_report_count")
    )
    previous_available = bool(
        previous_audit.get("files")
        or previous_audit.get("attempt_count")
        or previous_audit.get("failed_report_count")
    )
    return (
        "Tendances par rapport à la fenêtre précédente: "
        f"sessions={_trend(audit.get('record_count') if current_available else None, previous_audit.get('record_count') if previous_available else None)}; "
        f"succès prouvés={_trend(audit.get('success_count') if current_available else None, previous_audit.get('success_count') if previous_available else None)}; "
        f"incidents retenus={_trend(audit.get('incident_count') if current_available else None, previous_audit.get('incident_count') if previous_available else None)}; "
        f"candidats source={_trend(coverage.get('candidates') if coverage else None, previous_coverage.get('candidates') if previous_coverage else None)}."
    )


def build_report(
    *,
    report_date: date | None = None,
    collection_state_path: Path = DEFAULT_COLLECTION_STATE,
    incident_state_path: Path = DEFAULT_INCIDENT_STATE,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    project_id: str | None = None,
) -> str:
    start, end, local_day = day_window(report_date)
    coverage, collection_meta, collection_errors = collection_snapshot(
        collection_state_path, start, end, project_id=project_id
    )
    audit, audit_errors = audit_reports(audit_dir, start, end)
    tracking, incident_errors = incident_tracking(incident_state_path, start, end)
    previous_start = start - timedelta(days=1)
    previous_audit, _ = audit_reports(audit_dir, previous_start, start)
    previous_coverage, _, _ = collection_snapshot(
        collection_state_path, previous_start, start, project_id=project_id
    )
    post_correction = post_correction_recurrences(audit, tracking)
    errors = collection_errors + audit_errors + incident_errors
    audit_paths = audit["files"]
    stage_metrics = {
        stage: dict(value) if isinstance(value, dict) else {"status": "unknown"}
        for stage, value in (audit.get("stage_metrics") or {}).items()
    }
    if coverage is not None:
        stage_metrics["source"] = {
            **(stage_metrics.get("source") or {}),
            "status": "measured",
            **{key: coverage.get(key) for key in WINDOW_KEYS if coverage.get(key) is not None},
        }

    lines = [
        f"# Rapport {REPORT_BRAND} quotidien — {local_day.isoformat()}",
        "",
        f"Fenêtre: {start.isoformat()} à {end.isoformat()} ({PARIS.key})",
        "Génération: agrégation locale des états et rapports existants.",
        "Analyse LLM supplémentaire: non.",
        "",
        "## État de la mission agent",
        "",
    ]
    blockers: list[str] = []
    if coverage is None:
        blockers.append("aucun passage de collecte avec couverture daté dans la fenêtre")
    else:
        if coverage["unexamined"]:
            blockers.append(f"{coverage['unexamined']} contenus non examinés")
        if coverage["failed"]:
            blockers.append(f"{coverage['failed']} captures en échec")
        if coverage["deferred"]:
            blockers.append(f"{coverage['deferred']} captures différées")
    if audit["limitation_count"]:
        blockers.append(f"{audit['limitation_count']} limites déclarées par les audits du jour")
    if audit.get("attempt_count"):
        blockers.append(f"{audit['attempt_count']} attempt(s) d'audit non validé(s)")
    elif audit.get("failed_report_count"):
        blockers.append(f"{audit['failed_report_count']} audit(s) non validé(s)")
    blocker_text = "; ".join(blockers) if blockers else "aucun blocage observé dans les états chargés"
    lines.append(f"Bloc mission: {blocker_text}.")
    lines.append("Ce statut est limité aux états chargés; il ne signifie pas que toute la file d’analyse est traitée.")
    lines.append("Le rapport ne marque aucune mission ni correction comme résolue.")
    lines.extend(["", "## Bloc mission agent", ""])
    proven_incidents = [
        entry
        for entry in audit["incident_entries"]
        if isinstance(entry, dict)
        and isinstance(entry.get("incident"), dict)
        and _incident_has_source_proof(entry["incident"])
    ]
    if proven_incidents:
        lines.append("Mission: traiter au maximum trois incidents prouvés, puis vérifier le test prévu.")
        lines.append(
            "Instruction: vérifier le diagnostic et proposer un changement ciblé. N'appliquer ce changement "
            "qu'après autorisation explicite pour les ressources concernées. Exécuter ensuite le test indiqué."
        )
        for entry in proven_incidents[:3]:
            incident = entry.get("incident") if isinstance(entry.get("incident"), dict) else {}
            recommendation = str(incident.get("recommendation") or "aucune correction proposée")
            test = str(incident.get("test") or "aucun test proposé")
            lines.append(f"- Action préparatoire: vérifier «{recommendation}». Test requis: «{test}».")
    else:
        lines.append("Mission: aucune mission d'incident prouvée dans un rapport daté de la fenêtre.")
        if audit.get("incidents_without_proof"):
            lines.append(
                f"{audit['incidents_without_proof']} incident(s) sans preuve de source restent visibles hors KPI et hors priorités."
            )
        lines.append("Instruction: attendre une preuve ou un échec enregistré; ne pas inventer une correction.")
    lines.extend(["", "## Couverture du jour", ""])
    if coverage is None:
        lines.append("Couverture: non mesurée dans la fenêtre. L'état cumulatif n'est pas présenté comme daily.")
        lines.append("Âge du plus ancien pending, fraîcheur du candidat récent et curseur de sélection: inconnus.")
    else:
        lines.append(
            "Les mesures suivantes décrivent le dernier passage daté dans la fenêtre; elles ne représentent pas le total quotidien."
        )
        lines.append("| Mesure | Valeur |")
        lines.append("|---|---:|")
        for key in WINDOW_KEYS:
            lines.append(f"| {key} | {optional_metric(coverage.get(key))} |")
        lines.append(f"| traitements locaux de collecte (sans modèle) | {optional_metric(coverage.get('calls'))} |")
        lines.append(f"| inchangés | {optional_metric(coverage.get('unchanged'))} |")
        lines.append(f"| différés | {optional_metric(coverage.get('deferred'))} |")
        lines.append(f"| actifs | {optional_metric(coverage.get('active'))} |")
        lines.append(f"| pending courant | {optional_metric(coverage.get('pending_count'))} |")
        lines.append(
            f"| âge du plus ancien pending (secondes) | {optional_metric(coverage.get('pending_oldest_age_seconds'))} |"
        )
        lines.append(
            f"| date du plus ancien pending | {optional_metric(coverage.get('pending_oldest_mtime'))} |"
        )
        lines.append(
            f"| âge du candidat le plus récent (secondes) | {optional_metric(coverage.get('freshest_candidate_age_seconds'))} |"
        )
        lines.append(
            f"| date du candidat le plus récent | {optional_metric(coverage.get('freshest_candidate_mtime'))} |"
        )
        lines.append(
            f"| curseur de sélection | {optional_metric(collection_meta.get('selection_cursor'))} |"
        )
        lines.append("")
        lines.append(
            f"Source: {reference(collection_state_path, collection_coverage_pointer(project_id))}."
        )
    if collection_meta.get("backlog") is not None:
        lines.append(f"Backlog courant hors fenêtre: {collection_meta['backlog']} éléments; il ne constitue pas une mesure daily.")

    lines.extend(["", "## Indicateurs de chaîne et KPI métier", ""])
    lines.extend(render_stage_metrics(stage_metrics, audit.get("usage_by_stage") or {}))
    lines.append(render_trends(audit, previous_audit, coverage, previous_coverage))
    lines.extend(["", *render_claim_states(audit)])
    lines.extend(render_signal_counts(audit, post_correction))
    if audit.get("success_without_proof") or audit.get("success_without_conversation"):
        lines.append(
            "Succès exclus du KPI: "
            f"sans preuve={audit.get('success_without_proof') or 0}; "
            f"sans conversation/session liée={audit.get('success_without_conversation') or 0}."
        )
    lines.append(
        "Récidives après correction: "
        f"{optional_metric(post_correction.get('count'))} "
        f"({post_correction.get('reason', 'preuve indisponible')})."
    )

    lines.extend(["", "## Trois priorités", ""])
    for index, item in enumerate(
        build_priorities(
            coverage,
            audit,
            tracking,
            collection_state_path,
            audit_paths,
            incident_state_path,
            project_id=project_id,
        ),
        start=1,
    ):
        lines.append(f"### {index}. {item.get('title', 'Priorité')}")
        if item.get("kind") == "incident":
            lines.append("")
            lines.append(f"- Attendu : {item.get('expected', 'non établi')}")
            lines.append(f"- Observé : {item.get('observed', 'non établi')}")
            lines.append(f"- Priorité : {item.get('priority', 'inconnue')}")
            lines.append(f"- Risque : {item.get('risk', 'inconnu')}")
            lines.append(f"- Cause : {item.get('cause', 'non établie')}")
            lines.append(f"- Preuve de cause : {item.get('cause_proof', 'non fournie')}")
            lines.append(f"- Correction proposée : {item.get('recommendation', 'aucune')}")
            lines.append(f"- Preuve d'application : {item.get('correction_proof', 'non fournie')}")
            lines.append(f"- Test : {item.get('test', 'aucun')}")
            lines.append(f"- Preuve : {item.get('proof', 'source indisponible')}")
        else:
            lines.append("")
            lines.append(f"- {item.get('detail', 'Détail indisponible.')}")
        lines.append("")

    lines.extend(["", *render_auto_improvement(audit, tracking, post_correction)])

    lines.extend(["", "## Connaissances, tâches et signaux", ""])
    if audit["files"]:
        statuses = ", ".join(f"{key}: {value}" for key, value in sorted(audit["conversation_statuses"].items())) or "aucun statut"
        lines.append(
            f"Audits du jour: {audit['report_count']} rapport(s), {audit['record_count']} conversation(s) retenue(s) "
            f"après déduplication; {audit['observed_conversation_count']} observations, dont "
            f"{audit['reaudit_observations']} réaudit(s); {audit['success_count']} succès documentés, "
            f"{audit['incident_count']} incidents retenus sur {audit['incident_observations']} observations, "
            f"{audit['evidence_windows']} fenêtres de preuve. Statuts: {statuses}."
        )
        if audit["incident_types"]:
            types = ", ".join(f"{key}: {value}" for key, value in sorted(audit["incident_types"].items()))
            lines.append(f"Signaux frustration, suringénierie ou outils: {types}.")
        if audit["recommendations"]:
            lines.append("Tâches proposées par preuve:")
            lines.extend(f"- {item}" for item in audit["recommendations"][:5])
        lines.append("Rapports examinés: " + ", ".join(f"`{path}`" for path in audit["selected_files"]) + ".")
        date_dimensions = audit.get("date_dimensions", {})
        source_dates = set(date_dimensions.get("source", []))
        ingestion_dates = set(date_dimensions.get("ingestion", []))
        audit_dates = set(date_dimensions.get("audit", []))
        lines.append(
            "Dates distinctes: source/activité = "
            + (", ".join(sorted(source_dates)) if source_dates else "inconnue")
            + "; ingestion = "
            + (", ".join(sorted(ingestion_dates)) if ingestion_dates else "inconnue")
            + "; audit = "
            + (", ".join(sorted(audit_dates)) if audit_dates else "inconnue")
            + "."
        )
        lines.append(
            "Fraîcheur: source/activité = "
            + _freshness_label(source_dates, local_day)
            + "; ingestion = "
            + _freshness_label(ingestion_dates, local_day)
            + "; audit = "
            + _freshness_label(audit_dates, local_day)
            + "."
        )
        coverage_notes: list[str] = []
        if audit.get("partial_completeness"):
            coverage_notes.append(f"{audit['partial_completeness']} source(s) partielle(s)")
        if audit.get("unavailable_completeness"):
            coverage_notes.append(f"{audit['unavailable_completeness']} source(s) indisponible(s)")
        if audit.get("unknown_date_records"):
            coverage_notes.append(f"{audit['unknown_date_records']} date(s) source/ingestion inconnue(s)")
        if audit.get("coverage_limited"):
            coverage_notes.append("limites ou échecs présents")
        lines.append(
            "Couverture audit: "
            + ("limitée — " + "; ".join(coverage_notes) if coverage_notes else "bornée mais sans limite supplémentaire observée")
            + "."
        )
        if audit["reaudit_observations"] or audit["incident_reaudit_observations"]:
            lines.append(
                "Les observations de réaudit restent comptées séparément; les priorités utilisent le rapport valide le plus récent par conversation et source."
            )
    else:
        if audit.get("attempt_count") or audit.get("failed_report_count"):
            lines.append(
                "Aucun rapport d'audit validé retenu dans la fenêtre; un attempt échoué est exposé séparément."
            )
        else:
            lines.append("Aucun rapport d'audit daté dans la fenêtre. Cette absence ne prouve pas l'absence d'incident.")
        lines.append("Dates source/activité, ingestion et audit: inconnues faute de rapport exploitable.")
        lines.append("Fraîcheur et couverture audit: inconnues; la collecte ou l'audit peut être en échec ou hors fenêtre.")

    failure_lines = attempt_summary_lines(audit)
    if failure_lines:
        lines.extend(["", "## Échecs d'audit", "", *failure_lines])

    lines.extend(["", "## Suivi des corrections", ""])
    lines.append("Le registre est courant. Les nombres ci-dessous ne sont pas additionnés aux observations daily.")
    lines.append("| État | Nombre |")
    lines.append("|---|---:|")
    lines.append(f"| proposées | {optional_metric(tracking.get('proposed'))} |")
    lines.append(f"| acceptées explicitement | {optional_metric(tracking.get('accepted'))} |")
    lines.append(f"| refusées explicitement | {optional_metric(tracking.get('refused'))} |")
    lines.append(f"| appliquées explicitement | {optional_metric(tracking.get('applied'))} |")
    lines.append(f"| vérifiées explicitement | {optional_metric(tracking.get('verified'))} |")
    lines.append(f"| effectives explicitement | {optional_metric(tracking.get('effective'))} |")
    lines.append(f"| observées dans la fenêtre | {optional_metric(tracking.get('observed_today'))} |")
    lines.append(f"| clôturées automatiquement | {optional_metric(tracking.get('closed_automatically'))} |")
    lines.append(f"| causes vérifiées avec preuve dédiée | {optional_metric(tracking.get('cause_proven'))} |")
    lines.append(f"| causes marquées vérifiées sans preuve dédiée | {optional_metric(tracking.get('cause_without_proof'))} |")
    lines.append(f"| corrections appliquées sans preuve d'application | {optional_metric(tracking.get('applied_without_proof'))} |")
    lines.append(f"| vérifications sans preuve de test | {optional_metric(tracking.get('verified_without_proof'))} |")
    lines.append(f"Source: {reference(incident_state_path, 'incidents')}.")
    lines.append(
        "Les statuts applied/verified ne constituent pas à eux seuls une preuve réelle; "
        "aucune correction n'est appliquée ou clôturée automatiquement."
    )

    lines.extend(["", "## Limites", ""])
    lines.append("- Le rapport lit les fichiers existants et ne relance aucune analyse de conversation.")
    lines.append("- Une couverture daily exige un `last_run_at` et une section `coverage` dans la fenêtre locale.")
    lines.append("- Les mesures de couverture décrivent un passage daté, pas le total de la journée.")
    lines.append("- Le backlog, le registre des incidents et les rapports antérieurs restent des états cumulatifs.")
    lines.append("- L'absence de rapport d'audit ne prouve pas l'absence d'incident.")
    lines.append("- Un échec d'audit non enregistré dans un état existant ne peut pas être compté.")
    lines.append("- Une correction reste proposée, appliquée ou vérifiée seulement si le registre porte cette preuve.")
    if errors:
        lines.append("- Erreurs de lecture:")
        lines.extend(f"  - {error}" for error in errors)
    else:
        lines.append("- Erreurs de lecture: aucune.")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report_dir: Path, content: str, report_date: date | None = None) -> Path:
    """Write dated and latest Markdown files atomically with private permissions."""
    local_day = report_date or datetime.now(PARIS).date()
    report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        report_dir.chmod(0o700)
    except OSError:
        pass
    dated = report_dir / f"{local_day.isoformat()}.md"
    for path in (dated, report_dir / "latest.md"):
        fd, temp_name = tempfile.mkstemp(dir=report_dir, prefix=".ace-daily-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temp_name, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        finally:
            Path(temp_name).unlink(missing_ok=True)
    return dated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-state", default=str(DEFAULT_COLLECTION_STATE))
    parser.add_argument("--incident-state", default=str(DEFAULT_INCIDENT_STATE))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--project",
        "--project-id",
        dest="project_id",
        help="Registered ACE project id",
    )
    parser.add_argument("--date", help="Local date YYYY-MM-DD in Europe/Paris")
    parser.add_argument("--stdout", action="store_true", help="Print the report without writing files")
    args = parser.parse_args(argv)
    report_date = date.fromisoformat(args.date) if args.date else None
    content = build_report(
        report_date=report_date,
        collection_state_path=Path(args.collection_state).expanduser(),
        incident_state_path=Path(args.incident_state).expanduser(),
        audit_dir=Path(args.audit_dir).expanduser(),
        project_id=args.project_id,
    )
    if args.stdout:
        print(content, end="")
    else:
        path = write_report(Path(args.report_dir).expanduser(), content, report_date)
        print(f"WROTE {REPORT_BRAND} daily report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
