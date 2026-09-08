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


def build_report(private_root: Path, day: str) -> str:
    names = _project_names(private_root)
    incidents: list[dict[str, Any]] = []
    recurrences: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    conversations: list[dict[str, Any]] = []
    tokens: dict[str, dict[str, dict[str, int]]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    limits: list[str] = []

    for project_id in sorted(names, key=lambda item: names[item]):
        name = names[project_id]
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
    lines.extend(["## 2. Récurrences", ""])
    if not recurrences:
        lines.append("Aucune récurrence : un même problème n'est pas revenu dans plusieurs sessions.")
    else:
        lines.extend(["| Signature | Projet | Occurrences | Sessions distinctes |", "|---|---|---:|---:|"])
        for item in recurrences:
            lines.append(
                f"| {item.get('signature')} | {item['_project']} | {item.get('occurrences') or 0} | {item.get('session_count') or 0} |"
            )
        lines.extend(["", "Le regroupement porte sur la signature exacte. Deux libellés différents restent deux lignes."])
    lines.extend(["", "## 3. Ce qui marche", ""])
    if not successes:
        lines.append("Aucun succès prouvé par une preuve explicite dans cette fenêtre.")
    for item in successes:
        text = item.get("summary") or item.get("text") or item.get("observed") or item
        lines.append(f"- {item['_project']} : {_clip(text)}")
    lines.extend(["", "## 4. À capitaliser", ""])
    if not preferences:
        lines.append("Aucune préférence récurrente détectée dans cette fenêtre.")
    for item in preferences:
        text = item.get("text") or item.get("summary") or item.get("preference") or item
        lines.append(f"- {item['_project']} : {_clip(text)}")
    lines.extend(["", "## 5. Conversations analysées", ""])
    if not conversations:
        lines.append("Aucune conversation décrite dans les audits du jour.")
    else:
        lines.extend(["| Projet | Sujet | État | Résumé |", "|---|---|---|---|"])
        for item in conversations:
            lines.append(
                f"| {item['_project']} | {_clip(item.get('subject'), 80)} | {item.get('status') or '-'} | {_clip(item.get('summary'), 160)} |"
            )
    lines.extend(["", "## 6. Consommation", ""])
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
    lines.extend(["", "## 7. Limites", ""])
    lines.append("- Le rapport lit les analyses existantes. Il ne relance aucune analyse.")
    lines.append("- Un projet sans rapport daté n'a pas été analysé ce jour, ou son analyse a échoué.")
    for item in limits:
        lines.append(f"- {item}")
    lines.append("")
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
