from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from ace_daily_report import (  # noqa: E402
    _claim_status_flags,
    audit_reports,
    build_report,
    day_window,
    incident_tracking,
    post_correction_recurrences,
    redact_value,
    render_auto_improvement,
    write_report,
)


class CmcDailyReportTests(unittest.TestCase):
    def test_auto_improvement_section_lists_suggestions_and_recorded_actions(self) -> None:
        lines = render_auto_improvement(
            {
                "record_count": 1,
                "observed_conversation_count": 1,
                "incident_count": 2,
                "incident_proven_count": 1,
                "incidents_without_proof": 1,
                "incident_types": {"frustration_mismatch": 2},
                "selected_conversations": [
                    {
                        "recommendations_detail": [
                            {
                                "text": "Répondre plus court.",
                                "type": "concision",
                                "evidence_refs": ["ev-1"],
                                "message_ids": ["msg-1"],
                            },
                        ]
                    }
                ],
                "recommendations": [],
            },
            {
                "proposed": 1,
                "accepted": 1,
                "applied": 0,
                "verified": 0,
                "effective": 0,
                "correction_events": [],
            },
            {"count": None, "reason": "aucune correction enregistrée"},
        )

        body = "\n".join(lines)
        self.assertIn("## Auto-amélioration", body)
        self.assertIn("### Suggestions proposées", body)
        self.assertIn("[concision] Répondre plus court. (preuves=1; messages=1).", body)
        self.assertIn("Aucune correction appliquée ou vérifiée", body)
        self.assertIn("Aucune suggestion n'est appliquée automatiquement", body)

    def test_daily_report_uses_current_window_and_existing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection-state.json"
            collection.write_text(
                json.dumps(
                    {
                        "last_run_at": "2026-09-05T11:00:00+02:00",
                        "coverage": {
                            "candidates": 10,
                            "ingested": 2,
                            "unexamined": 6,
                            "failed": 1,
                            "calls": 2,
                            "unchanged": 1,
                            "deferred": 0,
                            "active": 0,
                        },
                        "backlog": [{"path": "old.jsonl", "source": "codex"}],
                    }
                )
            )
            incidents = root / "incident-tracking.json"
            incidents.write_text(
                json.dumps(
                    {
                        "incidents": {
                            "one": {"status": "open", "last_seen_at": "2026-09-05T10:00:00+02:00"},
                            "two": {"status": "applied", "applied_at": "2026-09-05T10:01:00+02:00"},
                            "three": {"status": "verified", "test_status": "passed"},
                        }
                    }
                )
            )
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05T110000.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T11:30:00+02:00",
                        "metadata": {"record_count": 1, "evidence_window_count": 2},
                        "conversations": [{"status": "incident"}],
                        "incidents": [
                            {
                                "type": "outil",
                                "expected": "réponse courte",
                                "observed": "détour inutile",
                                "cause": {"status": "verified", "summary": "périmètre ignoré"},
                                "recommendation": "Tester le correctif.",
                                "test": "Rejouer le cas contrôlé.",
                                "conversation_id": "fixture-conversation",
                                "evidence_refs": ["ev-fixture"],
                            }
                        ],
                        "evidence": [{"ref": "ev-fixture"}],
                        "successes": [],
                        "limitations": ["Échantillon borné."],
                    }
                )
            )

            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=collection,
                incident_state_path=incidents,
                audit_dir=audit_dir,
            )

            self.assertIn("# Rapport ACE quotidien — 2026-09-05", report)
            self.assertIn("| candidates | 10 |", report)
            self.assertIn("| ingested | 2 |", report)
            self.assertIn("| unexamined | 6 |", report)
            self.assertIn("| failed | 1 |", report)
            self.assertIn("### 2. Couverture non examinée", report)
            self.assertIn("- Examiner les 6 contenus non examinés", report)
            self.assertIn("- Attendu : réponse courte", report)
            self.assertIn("[preuve JSON]", report)
            self.assertIn("#/incidents/0", report)
            self.assertIn("Bloc mission agent", report)
            self.assertIn("Les mesures suivantes décrivent le dernier passage daté", report)
            self.assertIn("N'appliquer ce changement qu'après autorisation explicite", report)
            self.assertIn("proposées | 1", report)
            self.assertIn("appliquées explicitement | 1", report)
            self.assertIn("vérifiées explicitement | 1", report)
            self.assertIn("clôturées automatiquement | 0", report)
            self.assertIn("Analyse LLM supplémentaire: non.", report)

    def test_missing_daily_collection_does_not_claim_cumulative_state_is_daily(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection-state.json"
            collection.write_text(
                json.dumps(
                    {
                        "last_run_at": "2026-09-04T23:59:00+02:00",
                        "coverage": {"candidates": 99, "ingested": 99},
                        "backlog": [{"path": "old.jsonl"}],
                    }
                )
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=collection,
                incident_state_path=root / "missing-incidents.json",
                audit_dir=root / "missing-audit",
            )
            self.assertIn("Couverture: non mesurée dans la fenêtre", report)
            self.assertIn("état cumulatif n'est pas présenté comme daily", report)
            self.assertNotIn("| candidates | 99 |", report)

    def test_reaudit_uses_latest_valid_report_for_priorities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            old = {
                "generated_at": "2026-09-05T10:00:00+02:00",
                "metadata": {"sources": ["codex"], "record_count": 1},
                "conversations": [{"conversation_id": "same", "status": "incident", "level": "high"}],
                "incidents": [
                    {
                        "conversation_id": "same",
                        "type": "old-false-positive",
                        "expected": "ancienne attente",
                        "observed": "ancien écart",
                        "cause": {"status": "verified", "summary": "ancienne cause"},
                        "recommendation": "ancienne correction",
                        "test": "ancien test",
                        "evidence_refs": ["old-ref"],
                    }
                ],
                "successes": [],
                "limitations": [],
            }
            new = {
                "generated_at": "2026-09-05T12:00:00+02:00",
                "metadata": {"sources": ["codex"], "record_count": 1},
                "conversations": [{"conversation_id": "same", "status": "success", "level": "none"}],
                "incidents": [],
                "successes": [{"conversation_id": "same", "summary": "corrigé", "evidence_refs": []}],
                "limitations": [],
            }
            (audit_dir / "old.json").write_text(json.dumps(old))
            (audit_dir / "new.json").write_text(json.dumps(new))
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )
            self.assertNotIn("old-false-positive", report)
            self.assertIn("1 conversation(s) retenue(s) après déduplication", report)
            self.assertIn("1 réaudit(s)", report)
            self.assertIn("0 incidents retenus sur 1 observations", report)

    def test_failed_attempt_is_exposed_without_promoting_previous_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T10:00:00+02:00",
                        "metadata": {"sources": ["codex"]},
                        "conversations": [
                            {"conversation_id": "same", "status": "success", "level": "none"}
                        ],
                        "incidents": [],
                        "successes": [
                            {
                                "conversation_id": "same",
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
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )

        self.assertIn("Dernier attempt d'audit non validé", report)
        self.assertIn("date=2026-09-05", report)
        self.assertIn("statut=model-error", report)
        self.assertIn("audit_runner_error", report)
        self.assertIn("Dernière analyse validée conservée séparément", report)
        self.assertIn("2026-09-05.json", report)
        self.assertIn("0 succès documentés", report)
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
                        "conversations": [
                            {"conversation_id": "failed", "status": "success", "level": "none"}
                        ],
                        "incidents": [],
                        "successes": [
                            {
                                "conversation_id": "failed",
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
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )

        self.assertIn("Dernier rapport d'audit non validé", report)
        self.assertIn("statut=degraded", report)
        self.assertIn("store_error", report)
        self.assertNotIn("succès issu d'une analyse échouée", report)

    def test_write_report_creates_dated_and_latest_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "daily"
            dated = write_report(directory, "# test\n", date(2026, 9, 5))
            self.assertEqual(dated.name, "2026-09-05.md")
            self.assertEqual(dated.read_text(), "# test\n")
            self.assertEqual((directory / "latest.md").read_text(), "# test\n")
            self.assertEqual(dated.stat().st_mode & 0o777, 0o600)

    def test_activity_date_is_not_replaced_by_audit_generation_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "historical-audit.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-04T23:00:00+02:00",
                        "metadata": {
                            "record_count": 1,
                            "record_dates": {
                                "codex:active": {
                                    "source_date": "2026-09-05",
                                    "ingestion_date": "2026-09-05",
                                    "audit_date": "2026-09-04",
                                }
                            },
                            "date_dimensions": {
                                "source_dates": ["2026-09-05"],
                                "ingestion_dates": ["2026-09-05"],
                                "audit_dates": ["2026-09-04"],
                            },
                            "completeness": {
                                "codex:active": {"observation": "partial", "terminal_evidence": False}
                            },
                        },
                        "conversations": [
                            {
                                "conversation_id": "active",
                                "status": "insufficient_evidence",
                                "level": "none",
                            }
                        ],
                        "incidents": [],
                        "successes": [],
                        "limitations": ["source partielle"],
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )
        self.assertIn("source/activité = 2026-09-05", report)
        self.assertIn("ingestion = 2026-09-05", report)
        self.assertIn("audit = 2026-09-04", report)
        self.assertIn("Couverture audit: limitée", report)

    def test_correction_statuses_show_missing_real_proof_and_no_auto_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incidents = root / "incident-tracking.json"
            incidents.write_text(
                json.dumps(
                    {
                        "incidents": {
                            "one": {
                                "status": "applied",
                                "applied_at": "2026-09-05T10:00:00+02:00",
                                "cause": {"status": "verified", "summary": "cause"},
                            },
                            "two": {
                                "status": "verified",
                                "test_status": "passed",
                                "cause": {"status": "verified", "summary": "cause"},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=incidents,
                audit_dir=root / "missing-audit",
            )
        self.assertIn("causes marquées vérifiées sans preuve dédiée | 2", report)
        self.assertIn("corrections appliquées sans preuve d'application | 1", report)
        self.assertIn("vérifications sans preuve de test | 1", report)
        self.assertIn("clôturées automatiquement | 0", report)
        self.assertIn("aucune correction n'est appliquée ou clôturée automatiquement", report)

    def test_collection_pending_age_and_cursor_are_visible_or_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection-state.json"
            collection.write_text(
                json.dumps(
                    {
                        "last_run_at": "2026-09-05T11:00:00+02:00",
                        "selection_cursor": 2,
                        "coverage": {
                            "candidates": 3,
                            "ingested": 1,
                            "unexamined": 1,
                            "failed": 0,
                            "calls": 1,
                            "unchanged": 0,
                            "deferred": 0,
                            "active": 0,
                            "pending_count": 4,
                            "pending_oldest_mtime": "2026-09-04T10:00:00+02:00",
                            "pending_oldest_age_seconds": 90000,
                            "freshest_candidate_mtime": "2026-09-05T10:59:00+02:00",
                            "freshest_candidate_age_seconds": 60,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=collection,
                incident_state_path=root / "missing-incidents.json",
                audit_dir=root / "missing-audit",
            )
        self.assertIn("| pending courant | 4 |", report)
        self.assertIn("| âge du plus ancien pending (secondes) | 90000 |", report)
        self.assertIn("| âge du candidat le plus récent (secondes) | 60 |", report)
        self.assertIn("| curseur de sélection | 2 |", report)

    def test_collection_pending_age_is_unknown_for_legacy_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection-state.json"
            collection.write_text(
                json.dumps(
                    {
                        "last_run_at": "2026-09-05T11:00:00+02:00",
                        "coverage": {"candidates": 1, "ingested": 1},
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=collection,
                incident_state_path=root / "missing-incidents.json",
                audit_dir=root / "missing-audit",
            )
        self.assertIn("| pending courant | inconnu |", report)
        self.assertIn("| âge du plus ancien pending (secondes) | inconnu |", report)
        self.assertIn("| curseur de sélection | inconnu |", report)


    def test_failed_attempt_masks_only_the_attributed_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            valid = {
                "generated_at": "2026-09-05T10:00:00+02:00",
                "metadata": {
                    "sources": ["codex"],
                    "snapshot_identities": [
                        {"source": "codex", "session_id": "a", "revision": "r1"},
                        {"source": "codex", "session_id": "b", "revision": "r1"},
                    ],
                },
                "conversations": [
                    {"conversation_id": "codex:a", "status": "success"},
                    {"conversation_id": "codex:b", "status": "success"},
                ],
                "successes": [
                    {"conversation_id": "codex:a", "summary": "A", "evidence_refs": ["ev-a"]},
                    {"conversation_id": "codex:b", "summary": "B", "evidence_refs": ["ev-b"]},
                ],
                "evidence": [{"ref": "ev-a"}, {"ref": "ev-b"}],
                "incidents": [],
            }
            (audit_dir / "2026-09-05.json").write_text(json.dumps(valid), encoding="utf-8")
            (audit_dir / "2026-09-05.attempt.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T12:00:00+02:00",
                        "status": "model-error",
                        "analysis_status": "model-error",
                        "errors": [
                            {
                                "kind": "analysis_failed",
                                "source": "codex",
                                "session_id": "a",
                                "revision": "r1",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            start, end, _ = day_window(date(2026, 9, 5))
            summary, errors = audit_reports(audit_dir, start, end)

        self.assertEqual(errors, [])
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(
            [item["conversation"]["conversation_id"] for item in summary["selected_conversations"]],
            ["codex:b"],
        )
        self.assertEqual(summary["latest_failure"]["scope_status"], "scoped")
        self.assertIn("a", json.dumps(summary["latest_failure"]["identities"]))

    def test_failed_batch_keeps_valid_nested_session_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T10:00:00+02:00",
                        "metadata": {
                            "sources": ["codex"],
                            "snapshot_identities": [
                                {"source": "codex", "session_id": "a", "revision": "r1"}
                            ],
                        },
                        "conversations": [{"conversation_id": "codex:a", "status": "success"}],
                        "successes": [
                            {
                                "conversation_id": "codex:a",
                                "summary": "ancienne session masquée",
                                "evidence_refs": ["ev-a"],
                            }
                        ],
                        "evidence": [{"ref": "ev-a"}],
                        "incidents": [],
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
                        "metadata": {"sources": ["claude", "codex"]},
                        "errors": [
                            {
                                "kind": "analysis_failed",
                                "source": "codex",
                                "session_id": "a",
                                "revision": "r1",
                            }
                        ],
                        "reports": [
                            {
                                "generated_at": "2026-09-05T11:30:00+02:00",
                                "metadata": {},
                                "conversations": [{"conversation_id": "codex:b", "status": "success"}],
                                "successes": [
                                    {
                                        "conversation_id": "codex:b",
                                        "summary": "session valide imbriquée",
                                        "evidence_refs": ["ev-b"],
                                    }
                                ],
                                "evidence": [{"ref": "ev-b"}],
                                "incidents": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            start, end, _ = day_window(date(2026, 9, 5))
            summary, errors = audit_reports(audit_dir, start, end)

        self.assertEqual(errors, [])
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(
            [item["conversation"]["conversation_id"] for item in summary["selected_conversations"]],
            ["codex:b"],
        )
        self.assertEqual(summary["latest_failure"]["scope_status"], "scoped")

    def test_success_kpi_requires_evidence_resolving_and_conversation_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            value = {
                "generated_at": "2026-09-05T10:00:00+02:00",
                "metadata": {"sources": ["codex"]},
                "conversations": [
                    {"conversation_id": "codex:one", "status": "success"},
                    {"conversation_id": "codex:two", "status": "success"},
                ],
                "successes": [
                    {"conversation_id": "codex:one", "summary": "vrai", "evidence_refs": ["ev-one"]},
                    {"conversation_id": "codex:two", "summary": "référence absente", "evidence_refs": ["ev-two"]},
                    {"conversation_id": "codex:other", "summary": "session absente", "evidence_refs": ["ev-one"]},
                ],
                "evidence": [{"ref": "ev-one"}],
                "incidents": [],
            }
            (audit_dir / "2026-09-05.json").write_text(json.dumps(value), encoding="utf-8")
            start, end, _ = day_window(date(2026, 9, 5))
            summary, _ = audit_reports(audit_dir, start, end)
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )

        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["success_without_proof"], 1)
        self.assertEqual(summary["success_without_conversation"], 1)
        self.assertIn("Succès exclus du KPI: sans preuve=1; sans conversation/session liée=1.", report)

    def test_chain_kpis_show_known_values_and_unknown_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = root / "collection-state.json"
            collection.write_text(
                json.dumps(
                    {
                        "last_run_at": "2026-09-05T11:00:00+02:00",
                        "coverage": {"candidates": 4, "ingested": 3, "unexamined": 1, "failed": 0},
                    }
                ),
                encoding="utf-8",
            )
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "known.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T11:00:00+02:00",
                        "metadata": {"sources": ["codex"]},
                        "conversations": [{"conversation_id": "codex:k", "status": "success"}],
                        "incidents": [],
                        "successes": [],
                        "extraction": {"status": "ok", "records": 3},
                        "compile": {"status": "ok", "records": 2},
                        "query": {"status": "ok", "records": 1},
                        "stage_usage": {
                            "analysis": {"token_usage": {"input_tokens": 10, "output_tokens": 4}},
                            "compile": {"input_tokens": 2, "output_tokens": 3},
                        },
                        "business_metrics": {
                            "tokens_by_stage": {
                                "analysis": {"input_tokens": 10, "output_tokens": 4},
                                "compile": {"input_tokens": 2, "output_tokens": 3},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=collection,
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )
            unknown_report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=root / "missing-audit",
            )

        self.assertIn("| extraction | ok | records=3 |", report)
        self.assertIn("| compile | ok | records=2 |", report)
        self.assertIn("| analysis | known | input_tokens=10, output_tokens=4, total_tokens=14 |", report)
        self.assertIn("| query | ok | records=1 |", report)
        self.assertIn("| source | measured | candidates=4, ingested=3, unexamined=1, failed=0 |", report)
        self.assertIn("| extraction | inconnu | inconnu |", unknown_report)
        self.assertIn("| analysis | inconnu | inconnu |", unknown_report)

    def test_refusal_and_registry_state_are_separate_and_missing_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "known.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T11:00:00+02:00",
                        "conversations": [{"conversation_id": "codex:k", "status": "success"}],
                        "recommendations": [
                            {
                                "text": "faire",
                                "status": "accepted",
                                "requested_at": "2026-09-05T10:00:00+02:00",
                                "accepted_at": "2026-09-05T10:01:30+02:00",
                            }
                        ],
                        "refusals": [{"text": "ne pas faire", "status": "refused"}],
                        "incidents": [],
                        "successes": [],
                    }
                ),
                encoding="utf-8",
            )
            registry = root / "incidents.json"
            registry.write_text(
                json.dumps(
                    {
                        "incidents": {
                            "accepted": {"status": "accepted"},
                            "refused": {"status": "refused"},
                            "effective": {"status": "effective", "effective_at": "2026-09-05T11:00:00+02:00"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            start, end, _ = day_window(date(2026, 9, 5))
            tracking, _ = incident_tracking(registry, start, end)
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=registry,
                audit_dir=audit_dir,
            )
            missing, _ = incident_tracking(root / "missing.json", start, end)

        self.assertEqual(tracking["accepted"], 1)
        self.assertEqual(tracking["refused"], 1)
        self.assertEqual(tracking["effective"], 1)
        self.assertIn("Temps demande → accepted: 90.0 secondes", report)
        self.assertIn("acceptées | 1", report)
        self.assertIn("refusées | 1", report)
        self.assertIn("effectives explicitement | 1", report)
        self.assertIsNone(missing["accepted"])
        self.assertIsNone(missing["registry_count"])

    def test_sensitive_nested_fields_are_redacted_without_hiding_usage_metrics(self) -> None:
        value = redact_value(
            {
                "credentials": {
                    "password": "short-secret",
                    "api_key": "key-x",
                    "token": "token-x",
                },
                "token_usage": {"input_tokens": 4, "output_tokens": 2},
            }
        )

        self.assertEqual(value["credentials"], "<REDACTED>")
        serialized = json.dumps(value, ensure_ascii=False)
        self.assertNotIn("short-secret", serialized)
        self.assertNotIn("key-x", serialized)
        self.assertNotIn("token-x", serialized)
        self.assertEqual(value["token_usage"]["input_tokens"], 4)

    def test_redaction_bounds_nested_depth_and_collection_size(self) -> None:
        deep: dict[str, object] = {}
        cursor = deep
        for _ in range(14):
            child: dict[str, object] = {}
            cursor["next"] = child
            cursor = child

        bounded_deep = redact_value(deep)
        self.assertIn("REDACTED: depth limit", json.dumps(bounded_deep))

        bounded_wide = redact_value({"items": list(range(600))})
        self.assertEqual(len(bounded_wide["items"]), 500)

    def test_claim_contradictions_and_absent_states_stay_unknown(self) -> None:
        self.assertIsNone(
            _claim_status_flags({"status": "accepted", "accepted": False})["accepted"]
        )
        self.assertIsNone(
            _claim_status_flags({"status": "effective", "effective": False})["effective"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / "incidents.json"
            registry.write_text(
                json.dumps(
                    {
                        "incidents": {
                            "open": {"status": "open", "type": "unmeasured"},
                            "accepted_conflict": {"status": "accepted", "accepted": False},
                            "effective_conflict": {"status": "effective", "effective": False},
                        }
                    }
                ),
                encoding="utf-8",
            )
            start, end, _ = day_window(date(2026, 9, 5))
            tracking, errors = incident_tracking(registry, start, end)

        self.assertEqual(errors, [])
        self.assertIsNone(tracking["accepted"])
        self.assertIsNone(tracking["effective"])
        self.assertIsNone(tracking["applied"])
        self.assertIsNone(tracking["verified"])
        self.assertGreaterEqual(tracking["state_unknown"]["accepted"], 1)
        self.assertGreaterEqual(tracking["state_unknown"]["effective"], 1)

    def test_post_correction_recurrence_requires_identity_and_observed_proof(self) -> None:
        audit = {
            "incident_entries": [
                {
                    "generated": "2026-09-05T11:00:00+02:00",
                    "incident": {
                        "id": "other-id",
                        "type": "same_label_only",
                        "signature": "same-signature",
                        "conversation_id": "codex:b",
                        "evidence_refs": ["ev-b"],
                        "evidence": [{"ref": "ev-b"}],
                    },
                },
                {
                    "generated": "2026-09-05T11:05:00+02:00",
                    "incident": {
                        "id": "fixed-id",
                        "type": "same_label_only",
                        "signature": "same-signature",
                        "conversation_id": "codex:b",
                    },
                },
                {
                    "generated": "2026-09-05T11:10:00+02:00",
                    "incident": {
                        "id": "fixed-id",
                        "type": "same_label_only",
                        "signature": "same-signature",
                        "conversation_id": "codex:b",
                        "evidence_refs": ["ev-id"],
                        "evidence": [{"ref": "ev-id"}],
                    },
                },
                {
                    "generated": "2026-09-05T11:15:00+02:00",
                    "incident": {
                        "id": "new-id",
                        "type": "same_label_only",
                        "signature": "other-signature",
                        "conversation_id": "codex:a",
                        "evidence_refs": ["ev-wrong-session"],
                        "evidence": [{"ref": "ev-wrong-session"}],
                    },
                },
                {
                    "generated": "2026-09-05T11:20:00+02:00",
                    "incident": {
                        "id": "new-id-2",
                        "type": "same_label_only",
                        "signature": "same-signature",
                        "conversation_id": "codex:a",
                        "evidence_refs": ["ev-same-session"],
                        "evidence": [{"ref": "ev-same-session"}],
                    },
                },
                {
                    "generated": "2026-09-05T11:25:00+02:00",
                    "incident": {
                        "id": "fixed-id",
                        "type": "same_label_only",
                        "signature": "same-signature",
                        "conversation_id": "codex:a",
                        "evidence_refs": ["ev-unresolved"],
                    },
                },
            ]
        }
        tracking = {
            "correction_events": [
                {
                    "id": "fixed-id",
                    "type": "same_label_only",
                    "signature": "same-signature",
                    "conversation_id": "codex:a",
                    "correction_at": "2026-09-05T10:00:00+02:00",
                    "correction_refs": ["fix-ref"],
                }
            ]
        }

        result = post_correction_recurrences(audit, tracking)

        self.assertEqual(result["count"], 2)

    def test_unproved_daily_incident_is_visible_but_not_a_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "2026-09-05.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-09-05T10:00:00+02:00",
                        "metadata": {"sources": ["codex"]},
                        "conversations": [{"conversation_id": "codex:one", "status": "incident"}],
                        "incidents": [
                            {
                                "type": "unproved_daily_gap",
                                "priority": "high",
                                "risk": "high",
                                "expected": "preuve",
                                "observed": "absence de preuve",
                                "evidence_refs": ["ev-unresolved"],
                            }
                        ],
                        "successes": [],
                    }
                ),
                encoding="utf-8",
            )
            report = build_report(
                report_date=date(2026, 9, 5),
                collection_state_path=root / "missing-collection.json",
                incident_state_path=root / "missing-incidents.json",
                audit_dir=audit_dir,
            )

        priority_section = report.split("## Trois priorités", 1)[1].split(
            "## Connaissances, tâches et signaux", 1
        )[0]
        self.assertNotIn("unproved_daily_gap", report)
        self.assertNotIn("unproved_daily_gap", priority_section)
        self.assertNotIn("Tâches proposées par preuve:", report)
        self.assertIn("Mission: aucune mission d'incident prouvée", report)
        self.assertIn("sans preuve, hors KPI/priorités=1", report)


if __name__ == "__main__":
    unittest.main()
