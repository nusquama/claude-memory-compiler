#!/usr/bin/env python3
"""Render a deterministic rolling-seven-day ACE synthesis.

The renderer only reads the existing audit JSON files and incident registry.
It does not call a model, update incident status, or infer success from the
absence of an incident.  Audit/date/deduplication work is delegated to the
daily report helpers so the daily and weekly views share one source-selection
contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from ace_daily_report import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_INCIDENT_STATE,
    PARIS,
    STAGE_NAMES,
    TOKEN_KEYS,
    audit_reports,
    attempt_summary_lines,
    incident_tracking,
    _incident_has_source_proof,
    load_json,
    optional_metric,
    render_claim_states,
    render_stage_metrics,
    render_signal_counts,
    render_trends,
    parse_timestamp,
    post_correction_recurrences,
    redact_value,
    _claim_status_flags,
    _success_supported_by_report,
    write_report as write_private_report,
)


DEFAULT_REPORT_DIR = Path(
    os.environ.get(
        "ACE_WEEKLY_REPORT_DIR",
        str(Path.home() / ".agents" / "private" / "ace" / "weekly"),
    )
)
ROLLING_DAYS = 7
MIN_PATTERN_SESSIONS = 3
_LABEL_RE = re.compile(r"[^\w]+", re.UNICODE)
_EXPLICIT_EFFORT_KEYS = (
    "retry_count",
    "attempt_count",
    "attempts",
    "retries",
    "recovery_attempts",
    "repeated_attempts",
    "repeated_effort",
    "recovery_steps",
)
_SUCCESS_LINK_KEYS = (
    "comparable_to",
    "comparable_to_incident",
    "comparison_type",
    "incident_type",
    "linked_pattern",
    "pattern_type",
)
_CORRECTION_REF_KEYS = (
    "correction_evidence_refs",
    "application_evidence_refs",
    "applied_evidence_refs",
    "correction_proof_refs",
)
_VERIFICATION_REF_KEYS = (
    "verification_evidence_refs",
    "test_evidence_refs",
    "verification_proof_refs",
)


def week_window(week_end: date | None = None) -> tuple[datetime, datetime, date, date]:
    """Return an inclusive seven-local-date window as a half-open interval."""
    end_day = week_end or datetime.now(PARIS).date()
    start_day = end_day - timedelta(days=ROLLING_DAYS - 1)
    start = datetime.combine(start_day, time.min, tzinfo=PARIS)
    end = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=PARIS)
    return start, end, start_day, end_day


def _normal_label(value: Any) -> str:
    """Normalize only spelling/spacing; never merge semantic labels."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = _LABEL_RE.sub(" ", text)
    return " ".join(text.split())


def _display(value: Any, fallback: str = "inconnu") -> str:
    if value is None or value == "":
        return fallback
    safe = redact_value(value)
    if isinstance(safe, (dict, list)):
        safe = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return str(safe).replace("\n", " ").replace("|", "\\|").strip() or fallback


def _nonempty_refs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _refs_from(item: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    refs: list[str] = []
    for key in keys:
        for ref in _nonempty_refs(item.get(key)):
            if ref not in refs:
                refs.append(ref)
    return refs


def _record_ids(record: dict[str, Any]) -> set[str]:
    conversation = record.get("conversation") if isinstance(record.get("conversation"), dict) else {}
    values = {
        str(record.get("key") or "").strip(),
        str(conversation.get("conversation_id") or "").strip(),
        str(conversation.get("id") or "").strip(),
    }
    key = str(record.get("key") or "").strip()
    if ":" in key:
        values.add(key.split(":", 1)[1])
    return {value for value in values if value}


def _date_values(record: dict[str, Any], dimension: str) -> list[str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    record_dates = metadata.get("record_dates") if isinstance(metadata.get("record_dates"), dict) else {}
    values: list[str] = []
    for record_id in _record_ids(record):
        dates = record_dates.get(record_id)
        if not isinstance(dates, dict):
            continue
        value = str(dates.get(f"{dimension}_date") or "").strip()
        if value and value != "unknown" and value not in values:
            values.append(value)
    if dimension == "audit" and not values:
        generated = parse_timestamp(record.get("generated"))
        if generated is not None:
            values.append(generated.date().isoformat())
    return values


def _in_window(value: str, start: datetime, end: datetime) -> bool:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return start.date() <= parsed < end.date()


def _record_window_state(record: dict[str, Any], start: datetime, end: datetime) -> dict[str, Any]:
    source_dates = _date_values(record, "source")
    ingestion_dates = _date_values(record, "ingestion")
    audit_dates = _date_values(record, "audit")
    # A source date is the work/activity date.  Ingestion is retained as a
    # separate dimension and never substitutes for missing source activity.
    work_dates = list(source_dates)
    audit_in = any(_in_window(value, start, end) for value in audit_dates)
    work_in = any(_in_window(value, start, end) for value in work_dates)
    if work_in:
        work_status = "dans la fenêtre"
    elif work_dates:
        work_status = "hors fenêtre"
    else:
        work_status = "inconnue"
    return {
        "source_dates": source_dates,
        "ingestion_dates": ingestion_dates,
        "audit_dates": audit_dates,
        "work_dates": work_dates,
        "audit_in": audit_in,
        "work_in": work_in,
        "work_status": work_status,
        "stale": audit_in and bool(work_dates) and not work_in,
    }


def _weekly_records(
    audit: dict[str, Any], start: datetime, end: datetime
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    excluded = 0
    for raw in audit.get("selected_conversations", []):
        if not isinstance(raw, dict):
            continue
        state = _record_window_state(raw, start, end)
        if not state["audit_in"] and not state["work_in"]:
            excluded += 1
            continue
        records.append({**raw, "window": state})
    if records:
        records[0]["_success_without_conversation"] = audit.get(
            "success_without_conversation", 0
        )
    return records, excluded


def _quality_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count missing dimensions on the selected records, not report totals."""
    counts = {
        "partial": 0,
        "unavailable": 0,
        "unknown": 0,
        "source_unknown": 0,
        "ingestion_unknown": 0,
    }
    for record in records:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        completeness = metadata.get("completeness") if isinstance(metadata.get("completeness"), dict) else {}
        observations = []
        for record_id in _record_ids(record):
            value = completeness.get(record_id)
            if isinstance(value, dict):
                observations.append(str(value.get("observation") or "unknown").lower())
        observation = observations[0] if observations else "unknown"
        if observation == "partial":
            counts["partial"] += 1
        elif observation == "unavailable":
            counts["unavailable"] += 1
        elif observation != "complete":
            counts["unknown"] += 1
        if not _date_values(record, "source"):
            counts["source_unknown"] += 1
        if not _date_values(record, "ingestion"):
            counts["ingestion_unknown"] += 1
    return counts


def _incident_entries(audit: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {str(record.get("key")): record for record in records if record.get("key")}
    entries: list[dict[str, Any]] = []
    for raw in audit.get("incident_entries", []):
        if not isinstance(raw, dict):
            continue
        incident = raw.get("incident") if isinstance(raw.get("incident"), dict) else {}
        key = str(raw.get("conversation_key") or "").strip()
        record = by_key.get(key)
        if record is None:
            continue
        entries.append({**raw, "incident": redact_value(incident), "record": record})
    return entries


def _proof_ref(path: Any, index: Any) -> str:
    if isinstance(path, Path) and isinstance(index, int):
        return f"`{path}` (JSON pointer `#/incidents/{index}`)"
    return "preuve JSON indisponible"


def _patterns(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        incident = entry.get("incident") if isinstance(entry.get("incident"), dict) else {}
        if not _incident_has_source_proof(incident):
            continue
        label = _normal_label(incident.get("type"))
        if label:
            grouped[label].append(entry)

    patterns: list[dict[str, Any]] = []
    for normalized, grouped_entries in grouped.items():
        session_keys = {
            str(entry.get("record", {}).get("key") or "").strip()
            for entry in grouped_entries
            if str(entry.get("record", {}).get("key") or "").strip()
            and not str(entry.get("record", {}).get("key")).startswith("report:")
        }
        if len(session_keys) < MIN_PATTERN_SESSIONS:
            continue
        representative = next(
            _display((entry.get("incident") or {}).get("type"), normalized)
            for entry in grouped_entries
            if (entry.get("incident") or {}).get("type")
        )
        references = [_proof_ref(entry.get("path"), entry.get("index")) for entry in grouped_entries[:3]]
        incident_ids = [
            _display((entry.get("incident") or {}).get("id"), "sans id")
            for entry in grouped_entries
        ]
        recommendations = [
            _display((entry.get("incident") or {}).get("recommendation"), "")
            for entry in grouped_entries
            if (entry.get("incident") or {}).get("recommendation")
        ]
        tests = [
            _display((entry.get("incident") or {}).get("test"), "")
            for entry in grouped_entries
            if (entry.get("incident") or {}).get("test")
        ]
        patterns.append(
            {
                "normalized": normalized,
                "label": representative,
                "entries": grouped_entries,
                "sessions": len(session_keys),
                "occurrences": len(grouped_entries),
                "max_severity": max((int(entry.get("severity") or 0) for entry in grouped_entries), default=0),
                "references": references,
                "incident_ids": incident_ids,
                "recommendation": recommendations[0] if recommendations else "",
                "test": tests[0] if tests else "",
            }
        )
    return sorted(
        patterns,
        key=lambda pattern: (-pattern["sessions"], -pattern["max_severity"], pattern["normalized"]),
    )


def _effort_evidence(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for entry in entries:
        incident = entry.get("incident") if isinstance(entry.get("incident"), dict) else {}
        evidence_refs = _nonempty_refs(incident.get("evidence_refs"))
        if not evidence_refs:
            continue
        for key in _EXPLICIT_EFFORT_KEYS:
            value = incident.get(key)
            if value is None or value == "" or value == []:
                continue
            found.append(
                {
                    "label": _display(incident.get("type"), "incident sans type"),
                    "field": key,
                    "value": _display(value),
                    "sessions": str(entry.get("record", {}).get("key") or "inconnue"),
                    "references": [_proof_ref(entry.get("path"), entry.get("index"))],
                }
            )
    return found


def _success_records(
    records: list[dict[str, Any]], success_without_conversation: int = 0
) -> tuple[list[dict[str, Any]], int]:
    evidenced: list[dict[str, Any]] = []
    without_proof = int(success_without_conversation or 0)
    if not without_proof:
        without_proof = sum(
            int(record.get("_success_without_conversation") or 0)
            for record in records[:1]
            if isinstance(record, dict)
        )
    for record in records:
        for raw in record.get("successes", []):
            if not isinstance(raw, dict):
                continue
            conversation_id = str(raw.get("conversation_id") or "").strip()
            if not conversation_id or conversation_id not in _record_ids(record):
                without_proof += 1
                continue
            refs = _nonempty_refs(raw.get("evidence_refs"))
            if not refs or not _success_supported_by_report(raw, record):
                without_proof += 1
                continue
            evidenced.append(
                {
                    "record": record,
                    "conversation_id": _display(conversation_id, record.get("key", "inconnue")),
                    "summary": _display(raw.get("summary"), "résultat non décrit"),
                    "evidence_refs": refs,
                    "links": _success_links(raw),
                }
            )
    return evidenced, without_proof


def _success_links(success: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in _SUCCESS_LINK_KEYS:
        value = success.get(key)
        if isinstance(value, dict):
            value = value.get("type") or value.get("incident_type") or value.get("pattern")
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for candidate in candidates:
            normalized = _normal_label(candidate)
            if normalized and normalized not in values:
                values.append(normalized)
    comparison = success.get("comparison")
    if isinstance(comparison, dict):
        for key in ("type", "incident_type", "pattern"):
            normalized = _normal_label(comparison.get(key))
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def _counterexamples(patterns: list[dict[str, Any]], successes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pattern in patterns:
        for success in successes:
            if pattern["normalized"] in success.get("links", []):
                result[pattern["normalized"]].append(success)
    return result


def _incident_is_applied(incident: dict[str, Any]) -> bool:
    return _claim_status_flags(incident)["applied"] is True


def _incident_is_verified(incident: dict[str, Any]) -> bool:
    return _claim_status_flags(incident)["verified"] is True


def _correction_results(
    incident_state: dict[str, Any], entries: list[dict[str, Any]], patterns: list[dict[str, Any]]
) -> dict[str, Any]:
    registry = incident_state.get("incidents") if isinstance(incident_state.get("incidents"), dict) else {}
    ids = {
        str((entry.get("incident") or {}).get("id") or "").strip()
        for entry in entries
        if str((entry.get("incident") or {}).get("id") or "").strip()
    }
    types = {pattern["normalized"] for pattern in patterns}
    results: list[dict[str, Any]] = []
    marked_without_proof = 0
    for registry_id, raw in registry.items():
        if not isinstance(raw, dict):
            continue
        incident_id = str(raw.get("id") or registry_id).strip()
        raw_type = _normal_label(raw.get("type"))
        if incident_id not in ids and raw_type not in types:
            continue
        applied = _incident_is_applied(raw)
        verified = _incident_is_verified(raw)
        correction_refs = _refs_from(raw, _CORRECTION_REF_KEYS)
        verification_refs = _refs_from(raw, _VERIFICATION_REF_KEYS)
        if applied and not correction_refs:
            marked_without_proof += 1
        if verified and not verification_refs:
            marked_without_proof += 1
        if correction_refs or verification_refs:
            results.append(
                {
                    "id": incident_id,
                    "type": _display(raw.get("type"), "incident sans type"),
                    "applied": applied and bool(correction_refs),
                    "verified": verified and bool(verification_refs),
                    "correction_refs": correction_refs,
                    "verification_refs": verification_refs,
                }
            )
    return {"results": results, "marked_without_proof": marked_without_proof}


def _priority(pattern: dict[str, Any]) -> dict[str, Any]:
    first_action = pattern.get("recommendation") or "Rejouer un cas comparable et capturer le résultat observable."
    criterion = (
        "Après application explicitement prouvée, vérifier un test référencé avec statut `passed` "
        f"et compter les occurrences de «{pattern['normalized']}» dans les trois prochaines sessions comparables; "
        "la correction n'est démontrée que si le test est passé et si le compteur vaut zéro, sinon le résultat reste inconnu."
    )
    return {
        "title": pattern["label"],
        "first_action": first_action,
        "criterion": criterion,
        "references": pattern["references"],
    }


def _quality_priority(records: list[dict[str, Any]], audit_paths: list[Path]) -> dict[str, Any] | None:
    quality = _quality_counts(records)
    stale = [record for record in records if record.get("window", {}).get("stale")]
    incomplete_count = quality["partial"] + quality["unavailable"] + quality["unknown"]
    if not incomplete_count and not stale:
        return None
    reason = []
    if incomplete_count:
        reason.append(f"{incomplete_count} source(s) partielle(s), indisponible(s) ou inconnue(s)")
    if stale:
        reason.append(f"{len(stale)} travail(aux) hors fenêtre malgré un audit récent")
    return {
        "title": "Qualité des dates et de la complétude",
        "first_action": "Reprendre les sources partielles et renseigner séparément date de travail et date d'audit.",
        "criterion": "Chaque session retenue porte une observation complète et une date de travail explicite, ou reste marquée inconnue.",
        "references": [f"`{path}`" for path in audit_paths[:3]],
        "reason": "; ".join(reason),
    }


def build_report(
    *,
    report_date: date | None = None,
    incident_state_path: Path = DEFAULT_INCIDENT_STATE,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> str:
    start, end, start_day, end_day = week_window(report_date)
    audit, audit_errors = audit_reports(audit_dir, start, end)
    records, excluded = _weekly_records(audit, start, end)
    entries = _incident_entries(audit, records)
    patterns = _patterns(entries)
    efforts = _effort_evidence(entries)
    successes, success_without_proof = _success_records(
        records, audit.get("success_without_conversation", 0)
    )
    counterexamples = _counterexamples(patterns, successes)
    incident_state, incident_errors = load_json(incident_state_path)
    tracking, tracking_errors = incident_tracking(incident_state_path, start, end)
    previous_start = start - timedelta(days=ROLLING_DAYS)
    previous_audit, _ = audit_reports(audit_dir, previous_start, start)
    post_correction = post_correction_recurrences(audit, tracking)
    corrections = _correction_results(incident_state, entries, patterns)
    errors = audit_errors + ([f"{incident_state_path}: {incident_errors}"] if incident_errors else []) + tracking_errors
    audit_paths = audit.get("selected_files", [])
    quality = _quality_counts(records)

    lines = [
        f"# Rapport ACE hebdomadaire — rolling7days — {start_day.isoformat()} à {end_day.isoformat()}",
        "",
        f"Période examinée: {start.isoformat()} à {end.isoformat()} ({PARIS.key}).",
        "Fenêtre d'audit: les audits générés dans la fenêtre ou liés à une date de source/ingestion dans la fenêtre.",
        "Génération: agrégation locale des audits et du registre existants.",
        "Analyse LLM supplémentaire: non.",
        "",
        "## Couverture et qualité des sources",
        "",
    ]
    if records:
        audit_in = sum(1 for record in records if record["window"]["audit_in"])
        work_in = sum(1 for record in records if record["window"]["work_in"])
        work_unknown = sum(1 for record in records if record["window"]["work_status"] == "inconnue")
        stale = sum(1 for record in records if record["window"]["stale"])
        lines.append(
            f"Sessions distinctes retenues: {len(records)} (dernier audit valide par source/session); "
            f"audit dans la fenêtre: {audit_in}, travail dans la fenêtre: {work_in}, "
            f"travail inconnu: {work_unknown}, travail possiblement ancien: {stale}."
        )
        lines.append(
            f"Observations d'audit: {audit.get('observed_conversation_count', 0)}; "
            f"réaudits écartés de la sélection: {audit.get('reaudit_observations', 0)}; "
            f"sessions hors fenêtre après déduplication: {excluded}."
        )
        lines.append(
            f"Complétude: partielle={quality['partial']}, "
            f"indisponible={quality['unavailable']}, "
            f"inconnue={quality['unknown']}; dates source inconnues={quality['source_unknown']}, "
            f"ingestion inconnues={quality['ingestion_unknown']}."
        )
        lines.append(
            "Dates explicites: travail = date source; ingestion et audit restent séparées; "
            "une date d'ingestion ou d'audit récente ne remplace pas une date source inconnue ou ancienne."
        )
        lines.append("Rapports retenus: " + ", ".join(f"`{path}`" for path in audit_paths[:8]) + ".")
    else:
        if audit.get("attempt_count") or audit.get("failed_report_count"):
            lines.append(
                "Aucune session d'audit validée retenue dans la fenêtre; un attempt échoué est exposé séparément."
            )
        else:
            lines.append("Aucune session d'audit exploitable dans la fenêtre; cette absence ne prouve pas l'absence de problème.")
        lines.append("Dates de travail, d'ingestion et d'audit: inconnues faute de session retenue.")

    lines.extend(["", "## Chaîne de traitement et états métier", ""])
    lines.extend(render_stage_metrics(audit.get("stage_metrics") or {}, audit.get("usage_by_stage") or {}))
    lines.append(render_trends(audit, previous_audit, None, None))
    lines.extend(["", *render_claim_states(audit)])
    lines.extend(render_signal_counts(audit, post_correction))
    lines.append(
        "Récidives après correction: "
        f"{optional_metric(post_correction.get('count'))} "
        f"({post_correction.get('reason', 'preuve indisponible')})."
    )

    failure_lines = attempt_summary_lines(audit)
    if failure_lines:
        lines.extend(["", "## Échecs d'audit", "", *failure_lines])

    lines.extend(["", "## Problèmes récurrents", ""])
    lines.append("Un problème récurrent exige au moins trois sessions source/conversation distinctes; les événements et réaudits seuls ne suffisent pas.")
    unproved_incidents = sum(
        1
        for entry in entries
        if isinstance(entry.get("incident"), dict)
        and not _incident_has_source_proof(entry["incident"])
    )
    if unproved_incidents:
        lines.append(
            f"{unproved_incidents} incident(s) sans preuve de source restent visibles hors KPI et hors regroupements récurrents."
        )
    if patterns:
        for pattern in patterns:
            lines.append(
                f"- **{pattern['label']}** — {pattern['sessions']} sessions distinctes, "
                f"{pattern['occurrences']} incident(s); regroupement limité au même type normalisé exactement `" 
                f"{pattern['normalized']}`."
            )
            lines.append("  Preuves: " + "; ".join(pattern["references"]) + ".")
    else:
        lines.append("Aucun type normalisé n'atteint trois sessions distinctes dans les preuves retenues.")
    lines.append("Les libellés différents ne sont pas regroupés par ressemblance sémantique.")

    lines.extend(["", "## Effort répété", ""])
    if efforts:
        for effort in efforts[:8]:
            lines.append(
                f"- {effort['label']}: champ explicite `{effort['field']}` = {effort['value']} "
                f"(session {effort['sessions']}); preuve: {'; '.join(effort['references'])}."
            )
    else:
        lines.append("Effort répété: inconnu; aucune preuve explicite de tentatives, reprises ou étapes répétées n'est portée par les incidents retenus.")

    lines.extend(["", "## Résultats durables et contre-exemples", ""])
    if successes:
        lines.append(f"Résultats explicitement réussis avec preuve: {len(successes)}.")
        for success in successes[:8]:
            lines.append(
                f"- {success['summary']} (session {success['conversation_id']}; "
                f"preuves: {', '.join(success['evidence_refs'])})."
            )
    else:
        lines.append("Aucun résultat durable explicitement prouvé dans les sessions retenues.")
    if success_without_proof:
        lines.append(
            "Succès déclarés sans preuve ou conversation/session liée, exclus des résultats: "
            f"{success_without_proof}."
        )
    if patterns:
        any_counterexample = False
        for pattern in patterns:
            matches = counterexamples.get(pattern["normalized"], [])
            if not matches:
                continue
            any_counterexample = True
            lines.append(f"Contre-exemple comparable explicitement lié à `{pattern['normalized']}`:")
            for success in matches[:3]:
                lines.append(
                    f"- {success['summary']} (session {success['conversation_id']}; "
                    f"preuves: {', '.join(success['evidence_refs'])})."
                )
        if not any_counterexample:
            lines.append("Aucun contre-exemple comparable explicitement lié à un problème récurrent avec preuve.")
    lines.append("L'absence d'incident n'est pas comptée comme une réussite.")

    lines.extend(["", "## Corrections suivies", ""])
    lines.append(
        "Registre courant: "
        f"proposées={optional_metric(tracking.get('proposed'))}, "
        f"acceptées={optional_metric(tracking.get('accepted'))}, "
        f"refusées={optional_metric(tracking.get('refused'))}, "
        f"appliquées={optional_metric(tracking.get('applied'))}, "
        f"vérifiées={optional_metric(tracking.get('verified'))}, "
        f"effectives={optional_metric(tracking.get('effective'))}; "
        "aucune modification du registre et aucune clôture automatique."
    )
    proved_results = corrections["results"]
    if proved_results:
        for result in proved_results[:8]:
            states = []
            if result["applied"]:
                states.append("application prouvée")
            if result["verified"]:
                states.append("test/vérification prouvé")
            lines.append(
                f"- {result['type']} ({result['id']}): {', '.join(states) or 'preuve de correction sans statut revendiqué'}; "
                f"références application={result['correction_refs'] or ['inconnue']}, "
                f"test={result['verification_refs'] or ['inconnue']}."
            )
    else:
        lines.append("Aucun résultat de correction ne peut être revendiqué sans preuve explicite d'application ou de test.")
    if corrections["marked_without_proof"]:
        lines.append(
            f"Statuts appliqué/vérifié sans preuve dédiée, non revendiqués comme résultats: {corrections['marked_without_proof']}."
        )

    lines.extend(["", "## Trois priorités", ""])
    priorities = [_priority(pattern) for pattern in patterns[:3]]
    quality = _quality_priority(records, audit_paths)
    if quality and len(priorities) < 3:
        priorities.append(quality)
    if not priorities:
        priorities.append(
            {
                "title": "Maintenir une preuve comparable",
                "first_action": "Conserver les identifiants de session, dates travail/audit et evidence_refs lors du prochain audit.",
                "criterion": "Le prochain rapport retient au moins trois sessions distinctes ou indique explicitement pourquoi le seuil n'est pas atteint.",
                "references": [f"`{path}`" for path in audit_paths[:3]] or ["preuve indisponible"],
            }
        )
    for index, priority in enumerate(priorities[:3], start=1):
        lines.append(f"### {index}. {priority['title']}")
        lines.append(f"- Première action : {priority['first_action']}")
        lines.append(f"- Critère mesurable : {priority['criterion']}")
        if priority.get("reason"):
            lines.append(f"- Motif de qualité : {priority['reason']}")
        lines.append(f"- Références : {'; '.join(priority['references'])}.")

    lines.extend(["", "## Limites", ""])
    lines.append("- Le rapport lit les audits JSON et le registre existants; il ne relance aucune analyse de conversation.")
    lines.append("- La sélection utilise le dernier audit valide par source/session; les réaudits ne créent pas de nouvelles sessions.")
    lines.append("- Un type n'est regroupé qu'après normalisation déterministe de son libellé exact; aucune équivalence sémantique n'est inférée.")
    lines.append("- Un succès exige un objet `successes` et des `evidence_refs`; un statut de conversation ou une absence d'incident ne suffit pas.")
    lines.append("- L'effort répété et les corrections restent inconnus quand la preuve explicite manque.")
    if errors:
        lines.append("- Erreurs de lecture:")
        lines.extend(f"  - {error}" for error in errors)
    else:
        lines.append("- Erreurs de lecture: aucune.")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report_dir: Path, content: str, report_date: date | None = None) -> Path:
    """Reuse the daily report's private atomic writer for weekly output."""
    return write_private_report(report_dir, content, report_date)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident-state", default=str(DEFAULT_INCIDENT_STATE))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--date", help="End date YYYY-MM-DD in Europe/Paris")
    parser.add_argument("--stdout", action="store_true", help="Print the report without writing files")
    args = parser.parse_args(argv)
    report_date = date.fromisoformat(args.date) if args.date else None
    content = build_report(
        report_date=report_date,
        incident_state_path=Path(args.incident_state).expanduser(),
        audit_dir=Path(args.audit_dir).expanduser(),
    )
    if args.stdout:
        print(content, end="")
    else:
        path = write_report(Path(args.report_dir).expanduser(), content, report_date)
        print(f"WROTE ACE weekly report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
