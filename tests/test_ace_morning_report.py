"""Tests for the consolidated, model-free ACE morning report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ace_morning_report as morning  # noqa: E402


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def private_root(tmp_path: Path) -> Path:
    root = tmp_path / "ace"
    _write(
        root / "projects.json",
        {
            "projects": {
                "p1": {"name": "alpha", "project_id": "p1"},
                "p2": {"name": "beta", "project_id": "p2"},
            }
        },
    )
    _write(
        root / "reports" / "p1" / "analysis" / "daily-2026-09-08.json",
        {
            "status": "ok",
            "coverage": {"sessions": 2},
            "incidents": [
                {
                    "signature": "low_first",
                    "priority": "normal",
                    "risk": "low",
                    "conversation_id": "codex:a",
                    "expected": "e1",
                    "observed": "o1",
                    "cause": {"status": "unknown", "summary": ""},
                    "recommendation": "r1",
                    "test": "t1",
                    "evidence_refs": ["ev-1"],
                },
                {
                    "signature": "high_first",
                    "priority": "high",
                    "risk": "high",
                    "conversation_id": "codex:b",
                    "expected": "e2",
                    "observed": "o2",
                    "cause": {"status": "verified", "summary": "c2"},
                    "recommendation": "r2",
                    "test": "t2",
                    "evidence_refs": ["ev-2", "ev-3"],
                },
            ],
            "recurrences": [{"signature": "tool_error", "occurrences": 3, "session_count": 2}],
            "successes": [{"summary": "s1"}],
            "preferences": [{"text": "pref1"}],
        },
    )
    _write(
        root / "audits" / "p1" / "2026-09-08.json",
        {
            "status": "ok",
            "conversations": [{"subject": "Sujet", "status": "incident", "summary": "Résumé"}],
            "stage_metrics": {
                "extraction": {"call_count": 2, "token_usage": {"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 5}},
                "analysis": {"call_count": 1, "token_usage": {"input_tokens": 50, "output_tokens": 7}},
            },
            "errors": [],
        },
    )
    return root


def test_report_orders_incidents_by_priority_then_risk(private_root: Path) -> None:
    content = morning.build_report(private_root, "2026-09-08")
    assert content.index("### 1.1 high_first") < content.index("### 1.2 low_first")
    assert "| alpha | ok | 2 |" in content
    assert "| beta | aucun rapport | - |" in content
    assert "| tool_error | alpha | 3 | 2 |" in content
    assert "- alpha : s1" in content
    assert "- alpha : pref1" in content
    assert "| alpha | Sujet | incident | Résumé |" in content
    assert "| **Total** | | | 150 | 10 | 12 |" in content
    assert "Cause (verified) : c2" in content


def test_report_falls_back_to_recent_latest(private_root: Path) -> None:
    latest = private_root / "reports" / "p2" / "analysis" / "latest-daily.json"
    _write(latest, {"status": "ok", "coverage": {"sessions": 1}, "incidents": []})
    content = morning.build_report(private_root, "2026-09-08")
    assert "| beta | ok | 1 |" in content


def test_write_report_creates_dated_and_latest(private_root: Path, tmp_path: Path) -> None:
    content = morning.build_report(private_root, "2026-09-08")
    out = tmp_path / "morning"
    path = morning.write_report(out, content, "2026-09-08")
    assert path == out / "2026-09-08.md"
    assert (out / "latest.md").read_text(encoding="utf-8") == content
    assert not list(out.glob(".*tmp"))


def test_report_without_any_analysis_is_explicit(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    _write(root / "projects.json", {"projects": {"p1": {"name": "alpha"}}})
    content = morning.build_report(root, "2026-09-08")
    assert "Aucun incident retenu avec preuve." in content
    assert "Aucune mesure de tokens disponible." in content


def test_health_section_reports_pipeline_errors(private_root: Path) -> None:
    import sqlite3

    con = sqlite3.connect(private_root / "outbox.sqlite3")
    con.execute(
        "create table ace_outbox (key text primary key, source text, project_id text, session_id text, revision text,"
        " payload text, payload_bytes integer, status text, attempts integer, next_attempt_at real, lease_until real,"
        " receipt text, last_error text, created_at real, updated_at real)"
    )
    con.execute(
        "insert into ace_outbox values ('k1','claude','p1','s1','r1','{}',2,'retry',3,0,NULL,NULL,'SupabaseStoreError',0,0)"
    )
    con.commit()
    con.close()
    _write(
        private_root / "collection.json",
        {
            "projects": {"p1": {"coverage": {"failed": 2, "unexamined": 0}}},
            "sessions": {"/tmp/a.jsonl": {"status": "failed", "error_type": "EmptyTranscriptError"}},
            "automation_daily": {"2026-09-08": {"status": "failed", "attempts": 3, "last_error": "daily stages failed"}},
        },
    )
    _write(private_root / "extraction.json", {"snapshots": {"x": {"status": "pending", "error_type": "PipelineError"}}})
    _write(private_root / "analysis.json", {"projects": {"p1": {"days": {"2026-09-08": {"status": "pending", "reason": "stage_failed"}}}}})
    content = morning.build_report(private_root, "2026-09-08")
    assert "## 8. Santé de ACE" in content
    assert "1 conversation(s) en état `retry` (SupabaseStoreError)" in content
    assert "Collecte alpha : 2 échec(s) de lecture" in content
    assert "EmptyTranscriptError=1" in content
    assert "Cycle du matin du 2026-09-08 : échec après 3 tentative(s)" in content
    assert "Extraction non terminée : PipelineError=1" in content
    assert "Analyse alpha le 2026-09-08 : `pending` stage_failed" in content


def test_health_section_is_quiet_when_state_is_clean(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    _write(root / "projects.json", {"projects": {"p1": {"name": "alpha"}}})
    content = morning.build_report(root, "2026-09-08")
    assert "Aucune erreur de la chaîne détectée" in content


def test_health_section_surfaces_the_compiler_reason(private_root: Path) -> None:
    _write(
        private_root / "compile.json",
        {
            "projects": {
                "p1": {
                    "days": {"2026-09-08": {"status": "failed", "error_type": "PipelineError"}},
                    "diagnostics": {
                        "2026-09-08": {
                            "diagnostic": "incomplete knowledge build; broken internal link: concepts/gtm-x",
                            "returncode": 1,
                        }
                    },
                }
            }
        },
    )
    content = morning.build_report(private_root, "2026-09-08")
    assert "Compilation alpha le 2026-09-08 : `failed` PipelineError." in content
    assert "Motif : incomplete knowledge build; broken internal link: concepts/gtm-x" in content
