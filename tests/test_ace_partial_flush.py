from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ace_collect as collect  # noqa: E402
import flush  # noqa: E402


def _map_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(flush, "FLUSH_SINGLE_PASS_THRESHOLD", 1)
    monkeypatch.setattr(flush, "FLUSH_CHUNK_SIZE", 20)


def test_partial_chunk_failure_is_a_flush_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _map_reduce(monkeypatch)
    calls: list[str] = []

    async def fake_llm_call(prompt: str, _stderr: list[str]) -> tuple[str, Exception | None]:
        calls.append(prompt)
        if len(calls) == 1:
            return "partial extraction", None
        return "", RuntimeError("fixture chunk failure")

    monkeypatch.setattr(flush, "_llm_call", fake_llm_call)

    response, _stderr = asyncio.run(flush.run_flush("x" * 55))

    assert response.startswith("FLUSH_ERROR: partial chunk 2/")
    assert "partial extraction" not in response
    assert len(calls) == 2


def test_consolidation_failure_is_not_returned_as_partial_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _map_reduce(monkeypatch)
    context = "y" * 55
    expected_chunks = len(flush._chunk_at_turn_boundaries(context, 20))
    calls: list[str] = []

    async def fake_llm_call(prompt: str, _stderr: list[str]) -> tuple[str, Exception | None]:
        calls.append(prompt)
        if "Tu reçois" in prompt:
            return "", RuntimeError("fixture consolidation failure")
        return "partial extraction", None

    monkeypatch.setattr(flush, "_llm_call", fake_llm_call)

    response, _stderr = asyncio.run(flush.run_flush(context))

    assert response.startswith("FLUSH_ERROR: consolidation failed:")
    assert "partial extraction" not in response
    assert len(calls) == expected_chunks + 1


def _collection_fixture(tmp_path: Path) -> tuple[Path, Path, argparse.Namespace]:
    vault = tmp_path / "vault"
    (vault / "Conversations" / "knowledge").mkdir(parents=True)
    source = tmp_path / "claude" / "session.jsonl"
    source.parent.mkdir()
    source.write_text(
        "".join([
            json.dumps(
                {
                    "sessionId": "partial-session",
                    "message": {"role": "user", "content": "u" * 90},
                }
            )
            + "\n",
            json.dumps(
                {"message": {"role": "assistant", "content": "a" * 90}}
            )
            + "\n",
        ]),
        encoding="utf-8",
    )
    old = time.time() - 300
    source.touch()
    source = source.resolve()
    os.utime(source, (old, old))
    args = argparse.Namespace(
        state_file=str(tmp_path / "collection-state.json"),
        codex_root=str(tmp_path / "codex"),
        claude_root=str(source.parent),
        ccs_root=str(tmp_path / "ccs"),
        days=7,
        limit=2,
        stable_seconds=0,
        max_chars=500000,
        dry_run=False,
        fallback_project="Conversations",
        path=None,
        source=None,
        compile=False,
    )
    return vault, source, args


def test_collect_retries_full_source_then_deduplicates_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _map_reduce(monkeypatch)
    vault, source, args = _collection_fixture(tmp_path)
    monkeypatch.setattr(collect, "VAULT_ROOT", vault)
    # The fixture represents an explicitly initialized Conversations project.
    # Keep the source JSONL minimal while making the authorization boundary
    # explicit; ACE must never infer this destination as a fallback.
    monkeypatch.setattr(
        collect,
        "resolve_project_route",
        lambda source_cwd, **_kwargs: collect.ProjectRoute(
            source_project="Conversations",
            source_cwd=str(source_cwd or "/fixture/Conversations"),
            destination_project="Conversations",
            destination_dir=vault / "Conversations",
            used_fallback=False,
            reason="source_project",
        ),
    )

    failed_calls: list[str] = []

    async def fail_second_chunk(
        prompt: str, _stderr: list[str]
    ) -> tuple[str, Exception | None]:
        failed_calls.append(prompt)
        if len(failed_calls) == 2:
            return "", RuntimeError("fixture chunk failure")
        return "partial extraction", None

    monkeypatch.setattr(flush, "_llm_call", fail_second_chunk)
    first = collect.collect(args)

    assert first["failed"] == 1
    assert first["ingested"] == 0
    assert not (vault / "Conversations" / "daily").exists()
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    record = state["sessions"]["claude:partial-session"]
    assert record["status"] == "failed"
    assert "daily_file" not in record

    successful_calls: list[str] = []

    async def succeed_full_source(
        prompt: str, _stderr: list[str]
    ) -> tuple[str, Exception | None]:
        successful_calls.append(prompt)
        if "Tu reçois" in prompt:
            return "consolidated extraction", None
        return "partial extraction", None

    monkeypatch.setattr(flush, "_llm_call", succeed_full_source)
    second = collect.collect(args)

    assert second["ingested"] == 1
    assert successful_calls
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    record = state["sessions"]["claude:partial-session"]
    assert record["status"] == "ingested"
    daily_path = vault / "Conversations" / "daily" / record["daily_file"]
    daily = daily_path.read_text(encoding="utf-8")
    assert "consolidated extraction" in daily

    call_count = len(successful_calls)
    third = collect.collect(args)

    assert third["unchanged"] == 1
    assert len(successful_calls) == call_count
    assert successful_calls[0] == failed_calls[0]

    daily_before_change = daily
    with source.open("a", encoding="utf-8") as output:
        output.write(
            json.dumps(
                {"message": {"role": "assistant", "content": "n" * 120}}
            )
            + "\n"
        )
    changed_hash = collect.file_hash(source)

    failed_after_change_calls: list[str] = []

    async def fail_changed_second_chunk(
        prompt: str, _stderr: list[str]
    ) -> tuple[str, Exception | None]:
        failed_after_change_calls.append(prompt)
        if len(failed_after_change_calls) == 2:
            return "", RuntimeError("fixture changed-source chunk failure")
        return "changed partial extraction", None

    monkeypatch.setattr(flush, "_llm_call", fail_changed_second_chunk)
    fourth = collect.collect(args)

    assert fourth["failed"] == 1
    assert fourth["ingested"] == 0
    assert daily_path.read_text(encoding="utf-8") == daily_before_change
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    record = state["sessions"]["claude:partial-session"]
    assert record["status"] == "failed"
    assert record["source_hash"] == changed_hash

    retry_after_change_calls: list[str] = []

    async def succeed_changed_full_source(
        prompt: str, _stderr: list[str]
    ) -> tuple[str, Exception | None]:
        retry_after_change_calls.append(prompt)
        if "Tu reçois" in prompt:
            return "consolidated changed extraction", None
        return "changed partial extraction", None

    monkeypatch.setattr(flush, "_llm_call", succeed_changed_full_source)
    fifth = collect.collect(args)

    assert fifth["ingested"] == 1
    assert retry_after_change_calls[0] == failed_after_change_calls[0]
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    record = state["sessions"]["claude:partial-session"]
    assert record["status"] == "ingested"
    assert record["source_hash"] == changed_hash
    updated_daily = (vault / "Conversations" / "daily" / record["daily_file"]).read_text(
        encoding="utf-8"
    )
    assert updated_daily.count("<!-- ace-claude-session:") == 1
    assert "consolidated changed extraction" in updated_daily
    assert "consolidated extraction" not in updated_daily
    assert source.exists()
