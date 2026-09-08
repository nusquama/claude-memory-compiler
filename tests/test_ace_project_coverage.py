from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_health as health  # noqa: E402
import config  # noqa: E402


def _project(vault: Path, name: str) -> Path:
    project = vault / name
    (project / "knowledge").mkdir(parents=True, exist_ok=True)
    return project


def _write_jsonl(path: Path, rows: list[dict[str, object]], *, age: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    timestamp = time.time() - age
    os.utime(path, (timestamp, timestamp))
    return path


def _claude_rollout(path: Path, session_id: str, cwd: str, text: str) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "sessionId": session_id,
                "cwd": cwd,
                "message": {"role": "user", "content": "Question"},
            },
            {"message": {"role": "assistant", "content": text}},
        ],
    )


def _codex_rollout(path: Path, session_id: str, cwd: str, text: str) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": session_id,
                    "timestamp": "2026-09-05T10:00:00Z",
                    "cwd": cwd,
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            },
        ],
    )


def _health_args(
    *,
    codex_root: Path,
    claude_roots: list[Path],
    state_file: Path | None,
) -> dict[str, object]:
    return {
        "codex_root": codex_root,
        "claude_roots": claude_roots,
        "days": 7,
        "fallback_project": "Conversations",
        "skip_active_seconds": 0,
        "max_context_chars": 100_000,
        "collection_state_file": state_file,
    }


def test_dot_agents_is_a_valid_project_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["git", "-C", "/fixture/.agents", "rev-parse"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "/fixture/.agents\n"
                if command[-1] == "--show-toplevel"
                else "/fixture/.agents/.git\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    assert config.canonical_project_name("/fixture/.agents") == ".agents"


def test_worktree_identity_resolves_to_main_checkout_without_live_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = (
            "/fixture/repo/.claude/worktrees/feature\n"
            if command[-1] == "--show-toplevel"
            else "/fixture/repo/.git\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    assert config.canonical_git_root("/fixture/repo/.claude/worktrees/feature") == Path(
        "/fixture/repo"
    )


def test_dot_agents_source_requires_its_own_initialized_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    _project(vault, ".agents")
    monkeypatch.setattr(config, "canonical_project_name", lambda _cwd: ".agents")

    route = config.resolve_project_route(
        "/fixture/.agents",
        fallback_project="Conversations",
        vault_root=vault,
    )

    assert route.source_project == ".agents"
    assert route.destination_project == ".agents"
    assert route.destination_dir == vault / ".agents"
    assert route.used_fallback is False
    assert route.source_project == route.destination_project


def test_all_sources_count_exact_success_hashes_and_group_source_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    agents = _project(vault, ".agents")
    repo = _project(vault, "repo")
    codex_root = tmp_path / "codex"
    claude_root = tmp_path / "claude"
    claude = _claude_rollout(
        claude_root / "claude.jsonl", "claude-id", "/fixture/.agents", "Claude result"
    )
    codex = _codex_rollout(
        codex_root / "2026" / "09" / "codex.jsonl",
        "codex-id",
        "/fixture/repo",
        "Codex result",
    )
    claude_hash = health.file_hash(claude)
    codex_hash = health.file_hash(codex)
    collection_state = tmp_path / "collection-state.json"
    collection_state.write_text(
        json.dumps(
            {
                "sessions": {
                    "claude:claude-id": {
                        "status": "ingested",
                        "source_hash": claude_hash,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (repo / ".state").mkdir()
    (repo / ".state" / "codex-backfill.json").write_text(
        json.dumps(
            {
                "sessions": {
                    "codex-id": {"status": "ingested", "source_hash": codex_hash}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "VAULT_ROOT", vault)
    monkeypatch.setattr(
        config,
        "canonical_project_name",
        lambda cwd: ".agents" if str(cwd) == "/fixture/.agents" else "repo",
    )

    report = health.build_all_sources_report(
        **_health_args(
            codex_root=codex_root,
            claude_roots=[claude_root],
            state_file=collection_state,
        )
    )

    assert report["ingested"] == 2
    assert report["coverage_percent"] == 100.0
    assert report["status"] == "ok"
    assert (
        report["projects"][".agents"]["sources"]["claude"]["destinations"]
        [".agents"]["ingested"]
        == 1
    )
    assert (
        report["projects"]["repo"]["sources"]["codex"]["destinations"]["repo"]
        ["ingested"]
        == 1
    )
    assert agents.is_dir()


def test_exact_duplicate_is_counted_but_same_session_with_new_hash_remains_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    _project(vault, ".agents")
    claude_root = tmp_path / "claude"
    first = _claude_rollout(claude_root / "a.jsonl", "same-id", "/fixture/.agents", "first")
    _claude_rollout(claude_root / "b.jsonl", "same-id", "/fixture/.agents", "first")
    _claude_rollout(claude_root / "c.jsonl", "same-id", "/fixture/.agents", "changed")
    state_file = tmp_path / "collection-state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "claude:same-id": {
                        "status": "ingested",
                        "source_hash": health.file_hash(first),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "VAULT_ROOT", vault)
    monkeypatch.setattr(config, "canonical_project_name", lambda _cwd: ".agents")

    report = health.build_source_report(
        "claude",
        **_health_args(
            codex_root=tmp_path / "codex",
            claude_roots=[claude_root],
            state_file=state_file,
        ),
    )

    assert report["rollouts"] == 3
    assert report["duplicates"] == 1
    assert report["ingested"] == 1
    assert report["pending"] == 1
    assert report["projects"][".agents"]["destinations"][".agents"]["rollouts"] == 2


@pytest.mark.parametrize("state_kind", ["missing", "corrupt"])
def test_empty_source_or_unavailable_state_never_claims_full_coverage(
    tmp_path: Path, state_kind: str
) -> None:
    state_file = tmp_path / "collection-state.json"
    if state_kind == "corrupt":
        state_file.write_text("{not-json", encoding="utf-8")

    report = health.build_all_sources_report(
        **_health_args(
            codex_root=tmp_path / "codex",
            claude_roots=[tmp_path / "claude"],
            state_file=state_file,
        )
    )

    assert report["rollouts"] == 0
    assert report["coverage_percent"] is None
    assert report["status"] == "unknown"
    assert report["coverage_percent"] != 100.0


def test_json_error_is_visible_as_structural_failure_without_raw_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    _project(vault, "Conversations")
    claude_root = tmp_path / "claude"
    broken = claude_root / "broken.jsonl"
    _write_jsonl(
        broken,
        [{"sessionId": "broken", "message": {"role": "user", "content": "ok"}}],
    )
    with broken.open("a", encoding="utf-8") as output:
        output.write("{malformed-raw-fixture\n")
    monkeypatch.setattr(health, "VAULT_ROOT", vault)
    monkeypatch.setattr(config, "canonical_project_name", lambda _cwd: ".agents")

    report = health.build_source_report(
        "claude",
        **_health_args(
            codex_root=tmp_path / "codex",
            claude_roots=[claude_root],
            state_file=None,
        ),
    )

    assert report["parse_errors"] == 1
    assert report["coverage_percent"] == 0.0
    assert report["status"] == "attention"
    assert "malformed-raw-fixture" not in json.dumps(report)


def test_health_report_is_read_only_for_fixture_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault = tmp_path / "vault"
    _project(vault, "Conversations")
    claude_root = tmp_path / "claude"
    source = _claude_rollout(
        claude_root / "session.jsonl", "readonly", "/fixture/.agents", "result"
    )
    state_file = tmp_path / "collection-state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "claude:readonly": {
                        "status": "ingested",
                        "source_hash": health.file_hash(source),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(health, "VAULT_ROOT", vault)
    monkeypatch.setattr(config, "canonical_project_name", lambda _cwd: ".agents")

    def snapshot(root: Path) -> dict[str, tuple[int, bytes, int]]:
        return {
            str(path.relative_to(root)): (
                path.stat().st_mode,
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot(tmp_path)
    health.build_all_sources_report(
        **_health_args(
            codex_root=tmp_path / "codex",
            claude_roots=[claude_root],
            state_file=state_file,
        )
    )
    after = snapshot(tmp_path)

    assert after == before
