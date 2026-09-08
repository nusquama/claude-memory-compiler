import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import codex_runner
import flush


def test_json_stream_retains_usage_but_never_returns_tool_or_reasoning_text(tmp_path, monkeypatch):
    events = [
        {'type': 'item.completed', 'item': {'type': 'reasoning', 'text': 'private reasoning'}},
        {'type': 'item.completed', 'item': {'type': 'command_execution', 'aggregated_output': 'private tool'}},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Verified answer'}},
        {'type': 'turn.completed', 'usage': {'input_tokens': 120, 'cached_input_tokens': 60, 'output_tokens': 18, 'secret': 'DO_NOT_RETAIN'}},
    ]
    commands = []

    class Process:
        returncode = 0

        async def communicate(self, prompt):
            return '\n'.join(map(json.dumps, events)).encode(), b''

    async def spawn(*command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.delenv('ACE_LLM_CHILD', raising=False)
    monkeypatch.setattr(codex_runner.asyncio, 'create_subprocess_exec', spawn)
    answer, diagnostics = asyncio.run(codex_runner.run_codex('synthetic', cwd=tmp_path))
    assert answer == 'Verified answer'
    assert '--json' in commands[0]
    assert isinstance(diagnostics, list)
    assert diagnostics.as_metrics()['token_usage'] == {'input_tokens': 120, 'cached_input_tokens': 60, 'output_tokens': 18}
    assert diagnostics.as_metrics()['usage_status'] == 'available'
    assert diagnostics.as_metrics()['call_count'] == 1
    assert 'private' not in json.dumps(diagnostics.as_metrics())
    assert 'DO_NOT_RETAIN' not in json.dumps(diagnostics.as_metrics())


def test_missing_or_invalid_counts_are_unavailable_not_zero():
    for usage in ({}, {'input_tokens': True, 'output_tokens': 2}, {'input_tokens': -1, 'output_tokens': 2}):
        _answer, counts, _ = codex_runner.parse_codex_events(json.dumps({'type': 'turn.completed', 'usage': usage}))
        assert counts is None
    assert codex_runner.RunDiagnostics(calls=1).as_metrics()['token_usage'] is None


def test_json_without_a_final_answer_cannot_promote_tool_output(tmp_path, monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self, prompt):
            return b'{"type":"item.completed","item":{"type":"command_execution","aggregated_output":"not an answer"}}', b''

    async def spawn(*args, **kwargs):
        return Process()

    monkeypatch.delenv('ACE_LLM_CHILD', raising=False)
    monkeypatch.setattr(codex_runner.asyncio, 'create_subprocess_exec', spawn)
    with pytest.raises(RuntimeError, match='no final message'):
        asyncio.run(codex_runner.run_codex('synthetic', cwd=tmp_path))


def test_extraction_attempts_aggregate_success_and_failure_usage(monkeypatch):
    attempts = 0

    async def fake_run(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        measured = codex_runner.RunDiagnostics(usage={'input_tokens': 10, 'output_tokens': 3}, duration_seconds=2, calls=1)
        if attempts == 1:
            error = RuntimeError('fixture failure')
            error.diagnostics = measured
            raise error
        return 'result', measured

    monkeypatch.setattr(flush, 'run_codex', fake_run)
    diagnostics = codex_runner.RunDiagnostics()
    with pytest.raises(RuntimeError):
        asyncio.run(flush._run_codex_query('synthetic', diagnostics))
    assert asyncio.run(flush._run_codex_query('synthetic', diagnostics)) == 'result'
    assert diagnostics.as_metrics() == {
        'call_count': 2, 'duration_seconds': 4,
        'token_usage': {'input_tokens': 20, 'output_tokens': 6}, 'usage_status': 'available',
    }
    diagnostics.merge(codex_runner.RunDiagnostics(calls=1))
    assert diagnostics.as_metrics()['usage_status'] == 'partial'
    assert diagnostics.as_metrics()['call_count'] == 3


def test_extraction_result_keeps_measured_metadata_for_pipeline(monkeypatch):
    async def fake_run(*args, **kwargs):
        return 'documented result', codex_runner.RunDiagnostics(usage={'input_tokens': 11, 'output_tokens': 4}, calls=1)

    monkeypatch.delenv('ACE_LLM_CHILD', raising=False)
    monkeypatch.setattr(flush, 'run_codex', fake_run)
    answer, diagnostics = asyncio.run(flush.run_flush('synthetic user correction'))
    assert answer == 'documented result'
    assert diagnostics.as_metrics()['token_usage']['output_tokens'] == 4


def test_fidelity_instructions_survive_map_and_consolidation():
    prompts = [flush._build_single_pass_prompt('fixture'), flush._build_partial_prompt('fixture', 1, 2), flush._build_consolidation_prompt(['one', 'two'], 2)]
    for prompt in prompts:
        assert "Conserve les corrections de l'utilisateur" in prompt
        assert "Respecte l'étendue de la preuve" in prompt
