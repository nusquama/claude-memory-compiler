from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ace_projects import AmbiguousProjectError, ProjectRegistry  # noqa: E402
from ace_transcripts import (  # noqa: E402
    HermesAdapterUnavailable,
    iter_snapshots,
    parse_hermes,
    parse_codex_incremental,
)


def _project(registry: ProjectRegistry, root: Path):
    root.mkdir(parents=True)
    return registry.register(root)


def test_strict_resolution_rejects_name_only_legacy_folder(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "repo" / "knowledge").mkdir(parents=True)
    root = tmp_path / "repo"
    root.mkdir()

    assert ProjectRegistry(vault).resolve(root, strict=True) is None


def test_legacy_cmc_root_evidence_is_exact_and_read_only(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    project_dir = vault / "repo"
    (project_dir / "knowledge").mkdir(parents=True)
    root = tmp_path / "repo"
    root.mkdir()
    state_dir = project_dir / ".state"
    state_dir.mkdir()
    (state_dir / "codex-backfill.json").write_text(
        json.dumps({"sessions": {"s": {"cwd": str(root)}}}), encoding="utf-8"
    )

    registry = ProjectRegistry(vault)
    resolved = registry.resolve(root, strict=True)
    assert resolved is not None
    assert resolved.root == root.resolve()
    assert not registry.marker_path.exists()

    other = tmp_path / "other" / "repo"
    other.mkdir(parents=True)
    with pytest.raises(AmbiguousProjectError):
        registry.resolve(other, strict=True)


def test_hermes_metadata_is_per_session_and_does_not_assign_first_cwd(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    first = _project(registry, tmp_path / "first")
    second = _project(registry, tmp_path / "second")
    database = tmp_path / "hermes.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE messages(id TEXT, session_id TEXT, cwd TEXT, role TEXT, content TEXT, created_at TEXT)"
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("m1", "session-a", str(first.root), "user", "first", "2026-09-07T10:00:00Z"),
            ("m2", "session-b", str(second.root), "user", "second", "2026-09-07T10:01:00Z"),
        ],
    )
    connection.commit()
    connection.close()

    metadata = list(iter_snapshots([{"source": "hermes", "path": database}], first))
    assert {(item["session_id"], Path(item["project_root"])) for item in metadata} == {
        ("session-a", first.root.resolve()),
        ("session-b", second.root.resolve()),
    }
    assert all(item["metadata_only"] is True and "messages" not in item for item in metadata)

    second_envelope = parse_hermes(database, second, session_id="session-b")
    assert second_envelope["project"]["root"] == str(second.root.resolve())
    assert second_envelope["messages"][0]["content"] == "second"
    with pytest.raises(HermesAdapterUnavailable):
        parse_hermes(database, first, session_id="session-b")


def test_jsonl_snapshot_metadata_is_bounded_before_parse(tmp_path: Path) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "bounded", "cwd": str(tmp_path / "repo")},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": "x" * 300_000,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = next(iter_snapshots([{"source": "codex", "path": transcript}], None))
    assert candidate["session_id"] == "bounded"
    assert candidate["metadata_only"] is True
    assert "messages" not in candidate


def test_codex_incremental_parser_reads_only_appended_records(tmp_path: Path) -> None:
    transcript = tmp_path / "codex.jsonl"
    first = json.dumps({
        "type": "session_meta",
        "payload": {"id": "incremental", "cwd": str(tmp_path / "repo")},
    }) + "\n"
    transcript.write_text(first, encoding="utf-8")
    offset = transcript.stat().st_size
    transcript.open("a", encoding="utf-8").write(
        json.dumps({
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "id": "new-message",
                "content": "only the appended turn",
            },
        }) + "\n"
    )
    registry = ProjectRegistry(tmp_path / "vault")
    project = _project(registry, tmp_path / "repo")
    envelope, next_offset = parse_codex_incremental(
        transcript,
        project,
        offset=offset,
        session_id="incremental",
    )
    assert envelope is not None
    assert next_offset == transcript.stat().st_size
    assert len(envelope["messages"]) == 1
    assert envelope["messages"][0]["ordinal"] == 1
    assert envelope["messages"][0]["content"] == "only the appended turn"
