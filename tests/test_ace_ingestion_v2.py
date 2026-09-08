from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ace_outbox import Outbox  # noqa: E402
from ace_projects import (  # noqa: E402
    AmbiguousProjectError,
    ProjectRegistry,
    ProjectRegistryError,
)
from ace_transcripts import (  # noqa: E402
    IncompleteTranscriptError,
    MalformedTranscriptError,
    HermesAdapterUnavailable,
    inspect_hermes_schema,
    parse_claude,
    parse_codex,
    parse_hermes,
)


def _legacy_project(vault: Path, name: str) -> Path:
    project = vault / name
    (project / "knowledge").mkdir(parents=True)
    return project


def _project(registry: ProjectRegistry, root: Path, *, name: str | None = None):
    return registry.register(root, name=name)


def _write_jsonl(path: Path, rows: list[dict[str, object]], *, trailing_newline: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if trailing_newline:
        encoded += "\n"
    path.write_text(encoded, encoding="utf-8")
    return path


def _codex_rows(session_id: str = "codex-session", *, extra: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "cwd": "/synthetic/repo",
                "timestamp": "2026-09-07T10:00:00Z",
                "model": "synthetic-model",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "id": "u-1",
                "content": [{"type": "input_text", "text": "repeat"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": "repeat"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "id": "u-2",
                "content": [{"type": "input_text", "text": "repeat"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "analysis",
                "content": [{"type": "output_text", "text": "hidden"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "role": "assistant",
                "id": "call-1",
                "name": "full_tool",
                "arguments": {"value": "A" * 300},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "role": "tool",
                "call_id": "call-1",
                "output": "tool output " + "B" * 500,
                "status": "error",
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_complete", "status": "observed"},
        },
    ]
    if extra:
        rows.append(
            {
                "type": "agent_message",
                "message": "updated",
                "id": "a-update",
            }
        )
    return rows


def test_uninitialised_project_is_excluded_without_state_write(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    root = tmp_path / "repo"
    root.mkdir()
    registry = ProjectRegistry(vault)

    assert registry.resolve(root) is None
    assert not (vault / ".state" / "ace-projects.json").exists()


def test_legacy_resolution_is_deterministic_and_same_name_collision_fails_closed(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    legacy = _legacy_project(vault, "repo")
    first = tmp_path / "one" / "repo"
    second = tmp_path / "two" / "repo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    # A marker-less CMC project is eligible only when its old state contains
    # the exact canonical checkout root.  A knowledge directory by basename
    # alone is intentionally not an authorization signal.
    legacy_state = legacy / ".state"
    legacy_state.mkdir()
    (legacy_state / "codex-backfill.json").write_text(
        json.dumps({"project_root": str(first)}), encoding="utf-8"
    )
    registry = ProjectRegistry(vault)

    resolved = registry.resolve(first)
    assert resolved is not None
    assert resolved.id == registry.resolve(first).id
    with pytest.raises(AmbiguousProjectError):
        registry.resolve(second)


def test_explicit_marker_binds_custom_name_and_dot_prefixed_listing(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    root = tmp_path / ".agents"
    root.mkdir()
    registry = ProjectRegistry(vault)
    info = _project(registry, root, name="named-project")

    resolved = registry.resolve(root)
    assert resolved == info
    assert resolved.name == "named-project"
    assert registry.marker_path.exists()
    assert registry.list_initialized(include_legacy=False) == [info]


def test_codex_envelope_preserves_repeated_user_and_full_tool_result(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    path = _write_jsonl(tmp_path / "codex.jsonl", _codex_rows())

    envelope = parse_codex(path, project)
    assert envelope["schema_version"] == 1
    assert envelope["source"] == "codex"
    user_messages = [item for item in envelope["messages"] if item["role"] == "user"]
    assert len(user_messages) == 2
    assert not any(item["content"] == [{"type": "output_text", "text": "hidden"}] for item in envelope["messages"])
    tool_result = next(item for item in envelope["messages"] if item["type"] == "tool_result")
    assert len(tool_result["content"]) == 512
    assert tool_result["status"] == "error"
    assert not any(item["type"] == "unknown_event" for item in envelope["messages"])


def test_codex_drops_large_unknown_event_payloads_from_transport(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    path = _write_jsonl(
        tmp_path / "codex-large-event.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "large-event", "cwd": str(root)}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "keep this conversation"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "provider_internal_state",
                    "state": "x" * 5_000_000,
                },
            },
        ],
    )

    envelope = parse_codex(path, project)
    assert [item["role"] for item in envelope["messages"]] == ["user"]
    assert envelope["audit"]["unknown_types"] == ["provider_internal_state"]
    assert len(json.dumps(envelope, ensure_ascii=False).encode("utf-8")) < 1_000_000


def test_codex_bounds_tool_evidence_for_transport(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    path = _write_jsonl(
        tmp_path / "codex-large-tools.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "large-tools", "cwd": str(root)}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "role": "assistant",
                    "id": "call-1",
                    "name": "run",
                    "arguments": {"command": "x" * 100_000},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "role": "tool",
                    "call_id": "call-1",
                    "output": "y" * 100_000,
                },
            },
        ],
    )

    envelope = parse_codex(path, project)
    call = next(item for item in envelope["messages"] if item["type"] == "tool_call")
    result = next(item for item in envelope["messages"] if item["type"] == "tool_result")
    assert len(call["content"]["arguments"]) <= 1_200
    assert len(result["content"]) <= 2_000


def test_secret_and_base64_are_removed_with_attachment_reference(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    raw = bytes(range(64))
    encoded = base64.b64encode(raw).decode()
    path = _write_jsonl(
        tmp_path / "codex.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "secret-session", "cwd": str(root)}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "api_key=sk-test-0123456789012345"},
                        {"type": "image", "media_type": "image/png", "data": encoded},
                    ],
                },
            },
        ],
    )

    envelope = parse_codex(path, project)
    dumped = json.dumps(envelope, ensure_ascii=False)
    assert "sk-test-0123456789012345" not in dumped
    assert encoded not in dumped
    assert envelope["attachments"][0]["sha256"]
    assert envelope["attachments"][0]["bytes"] == len(raw)
    assert "attachment_id" in json.dumps(envelope["messages"])


def test_jsonl_truncation_defers_and_middle_corruption_fails(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    incomplete = tmp_path / "incomplete.jsonl"
    _write_jsonl(incomplete, _codex_rows(), trailing_newline=False)
    with incomplete.open("a", encoding="utf-8") as handle:
        handle.write('{"type":"response_item","payload":{"x":')
    with pytest.raises(IncompleteTranscriptError) as incomplete_error:
        parse_codex(incomplete, project)
    assert incomplete_error.value.defer is True
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"type":"session_meta"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(MalformedTranscriptError):
        parse_codex(malformed, project)


def test_claude_tools_hidden_thinking_and_hooks(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    path = _write_jsonl(
        tmp_path / "claude.jsonl",
        [
            {"sessionId": "claude-session", "cwd": str(root), "type": "user", "message": {"role": "user", "content": "hello"}},
            {
                "sessionId": "claude-session",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {"path": "x"}},
                    ],
                },
            },
            {
                "sessionId": "claude-session",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "result"}],
                },
            },
            {"sessionId": "claude-session", "type": "hook", "status": "done"},
        ],
    )
    envelope = parse_claude(path, project)
    assert any(item["type"] == "tool_call" for item in envelope["messages"])
    assert any(item["type"] == "tool_result" for item in envelope["messages"])
    assert any(item["type"] == "event" for item in envelope["messages"])
    assert "hidden" not in json.dumps(envelope["messages"])


def test_hermes_schema_adapter_is_explicit_and_read_only(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    unsupported = tmp_path / "unsupported.sqlite"
    sqlite3.connect(unsupported).execute("CREATE TABLE other(value TEXT)")
    with pytest.raises(HermesAdapterUnavailable):
        parse_hermes(unsupported, project)

    db = tmp_path / "hermes.sqlite"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE messages(id TEXT, session_id TEXT, role TEXT, content TEXT, created_at TEXT, status TEXT)"
    )
    connection.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
        ("m1", "h1", "user", "hello", "2026-09-07T10:00:00Z", None),
    )
    connection.commit()
    connection.close()
    schema = inspect_hermes_schema(db)
    assert "messages" in schema["tables"]
    envelope = parse_hermes(db, project)
    assert isinstance(envelope, dict)
    assert envelope["source"] == "hermes"
    assert envelope["messages"][0]["content"] == "hello"


def test_outbox_replay_update_restart_retry_and_ack(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    registry = ProjectRegistry(vault)
    root = tmp_path / "repo"
    root.mkdir()
    project = _project(registry, root)
    transcript = _write_jsonl(tmp_path / "codex.jsonl", _codex_rows())
    first = parse_codex(transcript, project)
    second = parse_codex(
        _write_jsonl(tmp_path / "codex-updated.jsonl", _codex_rows(extra=True)), project
    )
    outbox_path = tmp_path / "outbox.sqlite"
    queue = Outbox(outbox_path, lease_seconds=0)
    first_key = queue.enqueue(first)
    assert queue.enqueue(first) == first_key
    second_key = queue.enqueue(second)
    assert second_key != first_key
    claim = queue.pending(limit=2)
    assert {item.key for item in claim} == {first_key, second_key}
    queue.fail(first_key, "temporary error", retry_at=0)
    queue.close()

    restarted = Outbox(outbox_path, lease_seconds=0)
    retry = restarted.pending(limit=1)
    assert retry
    retry_key = retry[0].key
    restarted.ack(retry_key, {"database": "confirmed"})
    assert restarted.summary()["statuses"]["acknowledged"]["count"] == 1
    restarted.close()
