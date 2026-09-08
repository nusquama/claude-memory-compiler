from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ace_weekly_report import build_report, write_report  # noqa: E402


def _dates(session_id: str, source: str = "2026-09-03") -> dict[str, object]:
    return {
        "record_dates": {
            f"codex:{session_id}": {
                "source_date": source,
                "ingestion_date": "2026-09-03",
                "audit_date": "2026-09-05",
            }
        },
        "completeness": {
            f"codex:{session_id}": {"observation": "complete", "terminal_evidence": True}
        },
        "date_dimensions": {
            "source_dates": [source],
            "ingestion_dates": ["2026-09-03"],
            "audit_dates": ["2026-09-05"],
        },
    }


def _report(
    session_id: str,
    generated_at: str,
    *,
    incident_type: str | None = None,
    source_date: str = "2026-09-03",
    successes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    conversation_id = f"codex:{session_id}"
    incidents: list[dict[str, object]] = []
    if incident_type:
        incidents.append(
            {
                "id": f"incident-{session_id}",
                "conversation_id": conversation_id,
                "type": incident_type,
                "expected": "livrable mesurable",
                "observed": "écart observable",
                "cause": {"status": "not_established", "summary": "non établi"},
                "evidence_refs": [f"ev-{session_id}"],
                "recommendation": "Capturer le résultat et le test.",
                "test": "Rejouer le cas comparable.",
            }
        )
    values = {
        "generated_at": generated_at,
        "metadata": _dates(session_id, source_date),
        "conversations": [
            {
                "conversation_id": conversation_id,
                "status": "incident" if incidents else "success",
                "level": "medium" if incidents else "none",
                "subject": "Cas comparable",
                "incidents": [item["id"] for item in incidents],
            }
        ],
        "incidents": incidents,
        "successes": successes or [],
        "limitations": [],
    }
    evidence_refs: list[str] = []
    for incident in values["incidents"]:
        if not isinstance(incident, dict):
            continue
        evidence_refs.extend(
            str(ref).strip()
            for ref in incident.get("evidence_refs", [])
            if str(ref).strip()
        )
    for success in values["successes"]:
        if not isinstance(success, dict):
            continue
        evidence_refs.extend(
            str(ref).strip()
            for ref in success.get("evidence_refs", [])
            if str(ref).strip()
        )
    values["evidence"] = [{"ref": ref} for ref in dict.fromkeys(evidence_refs)]
    return values


class AceWeeklyReportTests(unittest.TestCase):
    def test_recurrence_uses_three_sessions_and_latest_reaudit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            # The older audit for session-a must not count as a second session
            # or leak its stale type into the weekly synthesis.
            fixtures = [
                (
                    "old-a.json",
                    _report("a", "2026-09-01T10:00:00+02:00", incident_type="stale_label"),
                ),
                (
                    "new-a.json",
                    _report("a", "2026-09-02T10:00:00+02:00", incident_type="repeatable_gap"),
                ),
                (
                    "b.json",
                    _report("b", "2026-09-03T10:00:00+02:00", incident_type="repeatable_gap"),
                ),
                (
                    "c.json",
                    _report("c", "2026-09-04T10:00:00+02:00", incident_type="repeatable_gap"),
                ),
            ]
            for name, value in fixtures:
                (audit_dir / name).write_text(json.dumps(value), encoding="utf-8")

            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("repeatable_gap", report)
        self.assertIn("3 sessions distinctes", report)
        self.assertNotIn("stale_label", report)
        self.assertIn("réaudits écartés de la sélection: 1", report)
        self.assertIn("statut `passed`", report)
        self.assertIn("compteur vaut zéro", report)

    def test_counterexample_requires_explicit_link_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            for session_id in ("a", "b", "c"):
                value = _report(
                    session_id,
                    f"2026-09-0{int(session_id == 'a') + 2}T10:00:00+02:00",
                    incident_type="repeatable_gap",
                )
                (audit_dir / f"{session_id}.json").write_text(json.dumps(value), encoding="utf-8")
            success_report = _report(
                "success",
                "2026-09-05T10:00:00+02:00",
                successes=[
                    {
                        "conversation_id": "codex:success",
                        "summary": "succès déclaré sans preuve",
                        "evidence_refs": [],
                    },
                    {
                        "conversation_id": "codex:success",
                        "summary": "succès prouvé mais non comparable",
                        "evidence_refs": ["ev-success-1"],
                    },
                    {
                        "conversation_id": "codex:success",
                        "summary": "contre-exemple comparable",
                        "evidence_refs": ["ev-success-2"],
                        "comparable_to": "repeatable_gap",
                    },
                ],
            )
            (audit_dir / "success.json").write_text(json.dumps(success_report), encoding="utf-8")
            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("Résultats explicitement réussis avec preuve: 2", report)
        self.assertIn("contre-exemple comparable", report)
        self.assertNotIn("succès déclaré sans preuve", report)
        self.assertIn("succès prouvé mais non comparable", report)
        self.assertIn("L'absence d'incident n'est pas comptée comme une réussite.", report)

    def test_failed_attempt_is_exposed_without_promoting_previous_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T10:00:00+02:00",
                        "metadata": _dates("same"),
                        "conversations": [
                            {"conversation_id": "codex:same", "status": "success", "level": "none"}
                        ],
                        "incidents": [],
                        "successes": [
                            {
                                "conversation_id": "codex:same",
                                "summary": "ancien succès validé",
                                "evidence_refs": ["ev-old"],
                            }
                        ],
                        "limitations": [],
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "2026-09-05.attempt.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T12:00:00+02:00",
                        "status": "model-error",
                        "analysis_status": "model-error",
                        "conversations": [],
                        "incidents": [],
                        "successes": [],
                        "errors": [{"kind": "audit_runner_error"}],
                        "limitations": ["last analysis attempt was not complete"],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("Dernier attempt d'audit non validé", report)
        self.assertIn("date=2026-09-05", report)
        self.assertIn("statut=model-error", report)
        self.assertIn("audit_runner_error", report)
        self.assertIn("Dernière analyse validée conservée séparément", report)
        self.assertIn("2026-09-05.json", report)
        self.assertIn("Aucun résultat durable explicitement prouvé", report)
        self.assertNotIn("ancien succès validé", report)

    def test_failed_report_success_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T12:00:00+02:00",
                        "status": "degraded",
                        "analysis_status": "degraded",
                        "metadata": _dates("failed"),
                        "conversations": [
                            {"conversation_id": "codex:failed", "status": "success", "level": "none"}
                        ],
                        "incidents": [],
                        "successes": [
                            {
                                "conversation_id": "codex:failed",
                                "summary": "succès issu d'une analyse échouée",
                                "evidence_refs": ["ev-failed"],
                            }
                        ],
                        "errors": [{"kind": "store_error"}],
                    }
                ),
                encoding="utf-8",
            )

            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("Dernier rapport d'audit non validé", report)
        self.assertIn("statut=degraded", report)
        self.assertIn("store_error", report)
        self.assertIn("Aucun résultat durable explicitement prouvé", report)
        self.assertNotIn("succès issu d'une analyse échouée", report)

    def test_success_requires_both_evidence_and_conversation_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            value = _report(
                "linked",
                "2026-09-05T10:00:00+02:00",
                successes=[
                    {
                        "conversation_id": "codex:linked",
                        "summary": "succès lié",
                        "evidence_refs": ["ev-linked"],
                    },
                    {
                        "conversation_id": "codex:other",
                        "summary": "succès d'une autre session",
                        "evidence_refs": ["ev-linked"],
                    },
                ],
            )
            value["evidence"] = [{"ref": "ev-linked"}]
            (audit_dir / "linked.json").write_text(json.dumps(value), encoding="utf-8")
            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("Résultats explicitement réussis avec preuve: 1", report)
        self.assertIn(
            "Succès déclarés sans preuve ou conversation/session liée, exclus des résultats: 1.",
            report,
        )
        self.assertNotIn("succès d'une autre session", report)

    def test_unlinked_success_is_reported_when_no_session_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            value = {
                "generated_at": "2026-09-05T10:00:00+02:00",
                "metadata": {"sources": ["codex"]},
                "conversations": [],
                "successes": [
                    {
                        "conversation_id": "codex:missing-session",
                        "summary": "succès sans session retenue",
                        "evidence_refs": ["ev-unlinked"],
                    }
                ],
                "evidence": [{"ref": "ev-unlinked"}],
                "incidents": [],
            }
            (audit_dir / "unlinked.json").write_text(json.dumps(value), encoding="utf-8")
            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertIn("Succès déclarés sans preuve ou conversation/session liée, exclus des résultats: 1.", report)
        self.assertNotIn("succès sans session retenue", report)

    def test_unproved_incidents_are_excluded_from_patterns_and_priorities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            for index, session_id in enumerate(("a", "b", "c"), start=1):
                value = _report(
                    session_id,
                    f"2026-09-0{index}T10:00:00+02:00",
                    incident_type="unproved_gap",
                )
                value["incidents"][0]["evidence_refs"] = ["ev-unresolved"]
                value["evidence"] = []
                (audit_dir / f"{session_id}.json").write_text(json.dumps(value), encoding="utf-8")

            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=root / "missing-registry.json",
            )

        self.assertNotIn("unproved_gap", report)
        self.assertIn("3 incident(s) sans preuve de source restent visibles hors KPI", report)
        self.assertIn("Aucun type normalisé n'atteint trois sessions distinctes", report)

    def test_source_date_is_not_replaced_by_ingestion_and_registry_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            old_work = _report(
                "old",
                "2026-09-05T10:00:00+02:00",
                incident_type="single_gap",
                source_date="2026-08-20",
            )
            (audit_dir / "old.json").write_text(json.dumps(old_work), encoding="utf-8")
            unknown_dates = _report("unknown", "2026-09-05T11:00:00+02:00")
            unknown_dates["metadata"] = {}
            (audit_dir / "unknown.json").write_text(json.dumps(unknown_dates), encoding="utf-8")
            registry = root / "incident-tracking.json"
            registry_value = {
                "version": 1,
                "incidents": {
                    "incident-old": {
                        "id": "incident-old",
                        "type": "single_gap",
                        "status": "verified",
                        "test_status": "passed",
                        "last_seen_at": "2026-09-05T10:00:00+02:00",
                    }
                },
            }
            registry.write_text(json.dumps(registry_value, sort_keys=True), encoding="utf-8")
            before = registry.read_bytes()
            report = build_report(
                report_date=date(2026, 9, 6),
                audit_dir=audit_dir,
                incident_state_path=registry,
            )
            after = registry.read_bytes()

        self.assertEqual(before, after)
        self.assertIn("travail dans la fenêtre: 0", report)
        self.assertIn("travail possiblement ancien: 1", report)
        self.assertIn("inconnue=1", report)
        self.assertIn("dates source inconnues=1", report)
        self.assertIn("Aucun résultat de correction ne peut être revendiqué", report)
        self.assertIn("Statuts appliqué/vérifié sans preuve dédiée", report)
        self.assertIn("Effort répété: inconnu", report)

    def test_write_report_creates_dated_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "weekly"
            dated = write_report(directory, "# test\n", date(2026, 9, 6))
            self.assertEqual(dated.name, "2026-09-06.md")
            self.assertEqual((directory / "latest.md").read_text(), "# test\n")
            self.assertEqual(dated.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
