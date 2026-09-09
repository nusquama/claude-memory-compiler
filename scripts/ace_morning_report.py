"""Consolidated ACE morning report across every registered project.

The renderer reads the per-project analysis reports and audits that the
pipeline already wrote under the private root.  It never calls a model, never
touches the database and never mutates any state.  Its single output is one
readable Markdown file per day plus a ``latest.md`` alias.

Layout of the report, in reading order:

1. what broke, sorted by priority then by risk;
2. recurrences, by exact signature;
3. what worked;
4. what should become a durable rule or preference;
5. the analysed conversations;
6. token consumption per stage;
7. limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
DEFAULT_PRIVATE_ROOT = Path(
    os.environ.get("ACE_PRIVATE_DIR", str(Path.home() / ".agents" / "private" / "ace"))
)
_PRIORITY_ORDER = {"high": 0, "élevée": 0, "elevee": 0, "normal": 1, "medium": 1, "moyenne": 1, "low": 2, "faible": 2}
_RISK_ORDER = {"low": 0, "faible": 0, "medium": 1, "moyen": 1, "high": 2, "élevé": 2, "eleve": 2}
_MAX_TEXT = 320
# Signals about the exchange itself, the ones that improve the agent. Tool
# failures are kept but never allowed to bury them.
_AGENT_SIGNAL_TYPES = frozenset(
    {
        "frustration",
        "correction_utilisateur",
        "demande_repetee",
        "fausse_completion",
        "perte_de_contexte",
        "preference_recurrente",
    }
)
_SIGNAL_ORDER = {
    "frustration": 0,
    "correction_utilisateur": 1,
    "demande_repetee": 2,
    "fausse_completion": 3,
    "perte_de_contexte": 4,
    "preference_recurrente": 5,
    "tool_error": 6,
}


def _clip(value: Any, limit: int = _MAX_TEXT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _project_names(private_root: Path) -> dict[str, str]:
    payload = _load_json(private_root / "projects.json") or {}
    projects = payload.get("projects")
    names: dict[str, str] = {}
    if isinstance(projects, Mapping):
        for project_id, item in projects.items():
            if isinstance(item, Mapping):
                names[str(project_id)] = str(item.get("name") or project_id)
    return names


def _analysis_for(private_root: Path, project_id: str, day: str) -> tuple[Mapping[str, Any] | None, str]:
    analysis_dir = private_root / "reports" / project_id / "analysis"
    dated = analysis_dir / f"daily-{day}.json"
    if dated.exists():
        return _load_json(dated), "dated"
    # The pipeline labels an analysis either by its run day or by its source
    # day.  Accept the latest report when it was written within the last
    # 36 hours so the morning report never hides a fresh analysis.
    latest = analysis_dir / "latest-daily.json"
    if latest.exists():
        try:
            age = datetime.now(PARIS).timestamp() - latest.stat().st_mtime
        except OSError:
            age = None
        if age is not None and 0 <= age <= 36 * 3600:
            return _load_json(latest), "latest"
    return None, "missing"


def _audit_for(private_root: Path, project_id: str, day: str) -> Mapping[str, Any] | None:
    return _load_json(private_root / "audits" / project_id / f"{day}.json")


def _sort_key(item: Mapping[str, Any]) -> tuple[int, int, str]:
    priority = str(item.get("priority") or "").lower()
    risk = str(item.get("risk") or "").lower()
    return (_PRIORITY_ORDER.get(priority, 1), _RISK_ORDER.get(risk, 3), str(item.get("signature") or ""))


def _tokens(stage_metrics: Any) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    if not isinstance(stage_metrics, Mapping):
        return result
    for stage, metrics in stage_metrics.items():
        if not isinstance(metrics, Mapping):
            continue
        usage = metrics.get("token_usage")
        if not isinstance(usage, Mapping):
            continue
        row = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = int(value)
        calls = metrics.get("call_count")
        if isinstance(calls, (int, float)) and not isinstance(calls, bool):
            row["calls"] = int(calls)
        result[str(stage)] = row
    return result


def _signals_for(private_root: Path, project_id: str, day: str) -> list[dict[str, Any]]:
    """Read the signals the extractor observed in the raw transcript.

    Signal files are named after the conversation's own day, which is often an
    earlier date than the day the extraction ran. Selecting by file name alone
    silently hid every signal captured today from a conversation started
    yesterday. Select on ``recorded_at`` instead: the day ACE saw the signal is
    the day the report must show it.
    """
    directory = private_root / "signals" / project_id
    if not directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
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
            recorded = str(row.get("recorded_at") or "")
            if recorded:
                try:
                    seen = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
                except ValueError:
                    seen = None
                if seen is not None and seen.astimezone(PARIS).date().isoformat() != day:
                    continue
            elif path.stem != day:
                continue
            rows.append(dict(row))
    return rows


def _render_signals(signals: list[dict[str, Any]]) -> list[str]:
    """Render the raw capture signals, verbatim quote included.

    The daily log deliberately neutralises tone; these rows keep the user's
    exact wording, which is the primary evidence of a friction.
    """
    lines = [
        "## 2. Signaux captés à la capture",
        "",
        "Relevés pendant la lecture du transcript, avant toute reformulation. La citation est verbatim.",
        "",
    ]
    if not signals:
        lines.extend(["Aucun signal capté. Les conversations du jour sont antérieures à cette capture, ou sans signal.", ""])
        return lines
    counts: dict[tuple[str, str], int] = {}
    for row in signals:
        key = (str(row.get("type")), str(row.get("signature")))
        counts[key] = counts.get(key, 0) + 1
    # The goal is to improve the agent, not to list tool failures. Signals
    # about the exchange itself come first; tool errors are grouped last.
    agent_rows = [row for row in signals if str(row.get("type")) in _AGENT_SIGNAL_TYPES]
    tool_rows = [row for row in signals if str(row.get("type")) not in _AGENT_SIGNAL_TYPES]

    def table(rows: list[dict[str, Any]]) -> list[str]:
        out = ["| Type | Signature | Projet | Occurrences |", "|---|---|---|---:|"]
        seen: set[tuple[str, str]] = set()
        for row in sorted(rows, key=lambda item: _SIGNAL_ORDER.get(str(item.get("type")), 9)):
            key = (str(row.get("type")), str(row.get("signature")))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"| {key[0]} | {_clip(key[1], 60)} | {row.get('_project', '-')} | {counts[key]} |")
        return out

    lines.append(f"### Sur l'agent ({len(agent_rows)})")
    lines.append("")
    if agent_rows:
        lines.extend(table(agent_rows))
        lines.extend(["", "Ce que tu as dit, mot pour mot :", ""])
        for row in sorted(agent_rows, key=lambda item: _SIGNAL_ORDER.get(str(item.get("type")), 9))[:12]:
            quote = _clip(row.get("quote"), 180)
            if not quote:
                continue
            observed = _clip(row.get("observed"), 160)
            lines.append(f"- **{row.get('type')}** « {quote} »")
            if observed:
                lines.append(f"    Avant cela : {observed}")
    else:
        lines.append("Aucun signal sur l'échange lui-même.")
    lines.extend(["", f"### Sur les outils ({len(tool_rows)})", ""])
    if tool_rows:
        lines.extend(table(tool_rows))
    else:
        lines.append("Aucune erreur d'outil captée.")
    lines.append("")
    return lines


def _ace_health(private_root: Path, day: str, names: Mapping[str, str]) -> list[str]:
    """Report the pipeline's own errors from local state only. No model, no DB."""
    lines: list[str] = []
    problems = 0

    # 1. Outbox: what never reached the database.
    db_path = private_root / "outbox.sqlite3"
    if db_path.exists():
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            rows = con.execute(
                "select status, count(*), coalesce(substr(last_error,1,60),'') from ace_outbox "
                "where status != 'acknowledged' group by status, 3"
            ).fetchall()
            con.close()
        except sqlite3.Error as error:
            rows = []
            lines.append(f"- File de sortie illisible : {type(error).__name__}.")
            problems += 1
        for status, count, error in rows:
            problems += int(count)
            detail = f" ({error})" if error else ""
            lines.append(f"- File de sortie : {count} conversation(s) en état `{status}`{detail}.")
    else:
        lines.append("- File de sortie absente.")

    # 2. Collection: unrouted, unexamined, failed per project.
    collection = _load_json(private_root / "collection.json") or {}
    projects = collection.get("projects") if isinstance(collection.get("projects"), Mapping) else {}
    for project_id, item in projects.items():
        coverage = item.get("coverage") if isinstance(item, Mapping) else None
        if not isinstance(coverage, Mapping):
            continue
        failed = int(coverage.get("failed") or 0)
        unexamined = int(coverage.get("unexamined") or 0)
        if failed or unexamined > 50:
            problems += 1
            lines.append(
                f"- Collecte {names.get(str(project_id), str(project_id))} : {failed} échec(s) de lecture, {unexamined} fichier(s) non examiné(s)."
            )
    sessions = collection.get("sessions") if isinstance(collection.get("sessions"), Mapping) else {}
    failed_sessions = [k for k, v in sessions.items() if isinstance(v, Mapping) and v.get("status") == "failed"]
    if failed_sessions:
        problems += len(failed_sessions)
        kinds: dict[str, int] = {}
        for key in failed_sessions:
            kind = str(sessions[key].get("error_type") or "inconnu")
            kinds[kind] = kinds.get(kind, 0) + 1
        lines.append("- Transcripts en échec de lecture : " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) + ".")
    automation = collection.get("automation_daily") if isinstance(collection.get("automation_daily"), Mapping) else {}
    today = automation.get(day) if isinstance(automation.get(day), Mapping) else None
    if today and today.get("status") == "failed":
        problems += 1
        lines.append(
            f"- Cycle du matin du {day} : échec après {today.get('attempts')} tentative(s), motif « {_clip(today.get('last_error'), 120)} »."
        )

    # 3. Extraction: snapshots still pending with an error type.
    extraction = _load_json(private_root / "extraction.json") or {}
    snapshots = extraction.get("snapshots") if isinstance(extraction.get("snapshots"), Mapping) else {}
    pending_kinds: dict[str, int] = {}
    for record in snapshots.values():
        if isinstance(record, Mapping) and record.get("status") == "pending":
            kind = str(record.get("error_type") or "en attente")
            pending_kinds[kind] = pending_kinds.get(kind, 0) + 1
    if pending_kinds:
        problems += sum(pending_kinds.values())
        lines.append("- Extraction non terminée : " + ", ".join(f"{k}={v}" for k, v in sorted(pending_kinds.items())) + ".")

    # 4. Compile / analysis state for the day, per project.
    for state_name, label in (("compile", "Compilation"), ("analysis", "Analyse")):
        state = _load_json(private_root / f"{state_name}.json") or {}
        by_project = state.get("projects") if isinstance(state.get("projects"), Mapping) else {}
        for project_id, item in by_project.items():
            days = item.get("days") if isinstance(item, Mapping) else None
            record = days.get(day) if isinstance(days, Mapping) else None
            if not isinstance(record, Mapping):
                continue
            status = str(record.get("status") or "")
            if status in {"failed", "pending"} or record.get("error_type") or record.get("analysis_status") in {"failed", "pending"}:
                problems += 1
                reason = record.get("reason") or record.get("pending_reason") or record.get("error_type") or ""
                lines.append(
                    f"- {label} {names.get(str(project_id), str(project_id))} le {day} : `{status or record.get('analysis_status')}` {reason}."
                )
                # The compiler stores its exact reason beside the day record.
                # Surface it: a generic status hides a fixable knowledge-base
                # defect, such as a broken link that blocks every compilation.
                diagnostics = item.get("diagnostics") if isinstance(item, Mapping) else None
                entry = diagnostics.get(day) if isinstance(diagnostics, Mapping) else None
                detail = entry.get("diagnostic") if isinstance(entry, Mapping) else None
                if detail:
                    lines.append(f"    Motif : {_clip(detail, 240)}")

    # 5. Model reports rejected today.
    for project_id, name in names.items():
        attempt = private_root / "audits" / project_id / f"{day}.attempt.json"
        if attempt.exists():
            payload = _load_json(attempt) or {}
            if str(payload.get("analysis_status") or "") == "model-error":
                problems += 1
                lines.append(f"- Analyse {name} : au moins un rapport du modèle refusé (preuve introuvable ou JSON invalide), reprise automatique.")

    # 6. Native service log.
    error_log = private_root / "launchd.error.log"
    if error_log.exists():
        try:
            age_hours = (datetime.now(PARIS).timestamp() - error_log.stat().st_mtime) / 3600
            text = error_log.read_text(encoding="utf-8", errors="replace")
            tracebacks = text.count("Traceback (most recent call last)")
            if age_hours <= 24 and tracebacks:
                problems += 1
                last = [line for line in text.splitlines() if line.strip().startswith(("ace_", "json.", "Supabase", "WARNING"))]
                tail = _clip(last[-1], 140) if last else "voir le journal"
                lines.append(f"- Service natif : {tracebacks} trace(s) d'erreur dans le journal, dernière : « {tail} ».")
        except OSError:
            pass
    tick_log = private_root / "launchd.log"
    if tick_log.exists():
        try:
            last_line = tick_log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1]
            tick = json.loads(last_line)
            summary = []
            for stage in ("collect", "sync", "process", "daily"):
                block = tick.get(stage) if isinstance(tick.get(stage), Mapping) else {}
                failed = int(block.get("failed") or 0)
                if failed:
                    problems += 1
                summary.append(f"{stage} échec={failed}")
            if tick.get("error"):
                summary.append(f"erreur={tick.get('error')}")
            lines.append("- Dernier tick natif : " + ", ".join(summary) + ".")
        except (OSError, ValueError, IndexError):
            lines.append("- Dernier tick natif : journal illisible.")

    header = ["## 9. Santé de ACE", ""]
    if problems == 0:
        return header + ["Aucune erreur de la chaîne détectée dans l'état local.", *lines, ""]
    return header + [f"{problems} point(s) à regarder.", "", *lines, ""]


def build_report(private_root: Path, day: str) -> str:
    names = _project_names(private_root)
    incidents: list[dict[str, Any]] = []
    recurrences: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    conversations: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    tokens: dict[str, dict[str, dict[str, int]]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    limits: list[str] = []

    for project_id in sorted(names, key=lambda item: names[item]):
        name = names[project_id]
        for row in _signals_for(private_root, project_id, day):
            row["_project"] = name
            signals.append(row)
        analysis, origin = _analysis_for(private_root, project_id, day)
        audit = _audit_for(private_root, project_id, day)
        if analysis is None and audit is None:
            coverage[name] = {"status": "aucun rapport"}
            continue
        if analysis is None:
            limits.append(f"{name} : audit présent mais aucun rapport d'analyse daté du {day}.")
        source = analysis or {}
        coverage[name] = {
            "status": str(source.get("status") or (audit or {}).get("status") or "inconnu"),
            "sessions": (source.get("coverage") or {}).get("sessions") if isinstance(source.get("coverage"), Mapping) else None,
            "origin": origin,
        }
        for item in source.get("incidents") or []:
            if isinstance(item, Mapping):
                row = dict(item)
                row["_project"] = name
                incidents.append(row)
        for item in source.get("recurrences") or []:
            if isinstance(item, Mapping):
                row = dict(item)
                row["_project"] = name
                recurrences.append(row)
        for item in source.get("successes") or []:
            if isinstance(item, Mapping):
                row = dict(item)
                row["_project"] = name
                successes.append(row)
        for item in source.get("preferences") or []:
            if isinstance(item, Mapping):
                row = dict(item)
                row["_project"] = name
                preferences.append(row)
        if audit is not None:
            for item in audit.get("conversations") or []:
                if isinstance(item, Mapping):
                    row = dict(item)
                    row["_project"] = name
                    conversations.append(row)
            stage_tokens = _tokens(audit.get("stage_metrics"))
            if stage_tokens:
                tokens[name] = stage_tokens
            for error in audit.get("errors") or []:
                limits.append(f"{name} : erreur d'analyse enregistrée : {_clip(error, 200)}")

    incidents.sort(key=_sort_key)
    recurrences.sort(key=lambda item: (-int(item.get("session_count") or 0), str(item.get("signature") or "")))

    lines: list[str] = [
        f"# Rapport du matin ACE — {day}",
        "",
        f"Généré le {datetime.now(PARIS).strftime('%Y-%m-%d %H:%M')} Europe/Paris, sans appel modèle, à partir des analyses déjà produites.",
        "",
        "## Résumé",
        "",
        "| Projet | État | Conversations analysées |",
        "|---|---|---:|",
    ]
    for name, item in coverage.items():
        sessions = item.get("sessions")
        lines.append(f"| {name} | {item.get('status')} | {sessions if sessions is not None else '-'} |")
    lines.extend(
        [
            "",
            f"Incidents : {len(incidents)}. Récurrences : {len(recurrences)}. Succès prouvés : {len(successes)}. Préférences détectées : {len(preferences)}.",
            "",
            "## 1. Ce qui casse",
            "",
            "Tri : priorité décroissante, puis risque croissant. Aucune correction n'est appliquée automatiquement.",
            "",
        ]
    )
    if not incidents:
        lines.append("Aucun incident retenu avec preuve.")
    for index, item in enumerate(incidents, start=1):
        cause = item.get("cause") if isinstance(item.get("cause"), Mapping) else {}
        evidence = item.get("evidence_refs") if isinstance(item.get("evidence_refs"), list) else []
        lines.extend(
            [
                f"### 1.{index} {item.get('signature') or item.get('type') or 'incident'} — {item['_project']}",
                "",
                f"- Priorité : {item.get('priority') or 'inconnue'}. Risque : {item.get('risk') or 'inconnu'}. Preuves : {len(evidence)}.",
                f"- Conversation : `{item.get('conversation_id') or 'inconnue'}`.",
                f"- Attendu : {_clip(item.get('expected'))}",
                f"- Observé : {_clip(item.get('observed'))}",
                f"- Cause ({cause.get('status') or 'inconnue'}) : {_clip(cause.get('summary')) or 'non établie'}",
                f"- Correction proposée : {_clip(item.get('recommendation')) or 'aucune'}",
                f"- Test de vérification : {_clip(item.get('test')) or 'aucun'}",
                "",
            ]
        )
    lines.extend(_render_signals(signals))
    lines.extend(["## 3. Récurrences", ""])
    if not recurrences:
        lines.append("Aucune récurrence : un même problème n'est pas revenu dans plusieurs sessions.")
    else:
        lines.extend(["| Signature | Projet | Occurrences | Sessions distinctes |", "|---|---|---:|---:|"])
        for item in recurrences:
            lines.append(
                f"| {item.get('signature')} | {item['_project']} | {item.get('occurrences') or 0} | {item.get('session_count') or 0} |"
            )
        lines.extend(["", "Le regroupement porte sur la signature exacte. Deux libellés différents restent deux lignes."])
    lines.extend(["", "## 4. Ce qui marche", ""])
    if not successes:
        lines.append("Aucun succès prouvé par une preuve explicite dans cette fenêtre.")
    for item in successes:
        text = item.get("summary") or item.get("text") or item.get("observed") or item
        lines.append(f"- {item['_project']} : {_clip(text)}")
    lines.extend(["", "## 5. À capitaliser", ""])
    if not preferences:
        lines.append("Aucune préférence récurrente détectée dans cette fenêtre.")
    for item in preferences:
        text = item.get("text") or item.get("summary") or item.get("preference") or item
        lines.append(f"- {item['_project']} : {_clip(text)}")
    lines.extend(["", "## 6. Conversations analysées", ""])
    if not conversations:
        lines.append("Aucune conversation décrite dans les audits du jour.")
    else:
        lines.extend(["| Projet | Sujet | État | Résumé |", "|---|---|---|---|"])
        for item in conversations:
            lines.append(
                f"| {item['_project']} | {_clip(item.get('subject'), 80)} | {item.get('status') or '-'} | {_clip(item.get('summary'), 160)} |"
            )
    lines.extend(["", "## 7. Consommation", ""])
    if not tokens:
        lines.append("Aucune mesure de tokens disponible. Une mesure absente n'est pas zéro.")
    else:
        lines.extend(["| Projet | Étape | Appels | Entrée | Entrée en cache | Sortie |", "|---|---|---:|---:|---:|---:|"])
        total = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        for name, stages in tokens.items():
            for stage, row in stages.items():
                lines.append(
                    f"| {name} | {stage} | {row.get('calls', '-')} | {row.get('input_tokens', '-')} | {row.get('cached_input_tokens', '-')} | {row.get('output_tokens', '-')} |"
                )
                for key in total:
                    total[key] += int(row.get(key, 0) or 0)
        lines.append(
            f"| **Total** | | | {total['input_tokens']} | {total['cached_input_tokens']} | {total['output_tokens']} |"
        )
    lines.extend(["", "## 8. Limites", ""])
    lines.append("- Le rapport lit les analyses existantes. Il ne relance aucune analyse.")
    lines.append("- Un projet sans rapport daté n'a pas été analysé ce jour, ou son analyse a échoué.")
    for item in limits:
        lines.append(f"- {item}")
    lines.append("")
    lines.extend(_ace_health(private_root, day, names))
    return "\n".join(lines)


def write_report(output_dir: Path, content: str, day: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dated = output_dir / f"{day}.md"
    temp = output_dir / f".{day}.md.tmp"
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, dated)
    latest = output_dir / "latest.md"
    temp_latest = output_dir / ".latest.md.tmp"
    temp_latest.write_text(content, encoding="utf-8")
    os.replace(temp_latest, latest)
    for path in (dated, latest):
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return dated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", default=str(DEFAULT_PRIVATE_ROOT))
    parser.add_argument("--date", help="Local day YYYY-MM-DD in Europe/Paris; default today")
    parser.add_argument("--stdout", action="store_true", help="Print without writing files")
    args = parser.parse_args(argv)
    day = args.date or datetime.now(PARIS).date().isoformat()
    date.fromisoformat(day)
    private_root = Path(args.private_root).expanduser()
    content = build_report(private_root, day)
    if args.stdout:
        print(content, end="")
        return 0
    path = write_report(private_root / "reports" / "morning", content, day)
    print(f"WROTE ACE morning report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
