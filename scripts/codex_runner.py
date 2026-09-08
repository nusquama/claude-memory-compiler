"""Small, non-recursive Codex runner used by every ACE LLM stage.

ACE deliberately keeps one execution engine.  The caller chooses whether the
Codex child may write inside the vault; user config, project rules, and the
notify hook are always disabled so a ACE run cannot recurse into itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

from config import CODEX_EXEC_PATH, CODEX_MODEL, CODEX_REASONING_EFFORT, VAULT_ROOT
from utils import safe_codex_diagnostic_lines


class RunDiagnostics(list):
    """Safe diagnostic lines plus measured usage; compatible with old callers."""

    def __init__(self, lines=(), *, usage=None, duration_seconds=0.0, calls=0):
        super().__init__(lines)
        self.usage = dict(usage) if usage is not None else None
        self.duration_seconds = duration_seconds
        self.calls = calls
        self.measured_calls = calls if usage is not None else 0

    def merge(self, other):
        self.extend(other)
        self.calls += getattr(other, "calls", 0)
        self.duration_seconds += getattr(other, "duration_seconds", 0.0)
        self.measured_calls += getattr(other, "measured_calls", 0)
        usage = getattr(other, "usage", None)
        if usage is not None:
            if self.usage is None:
                self.usage = {}
            for key, value in usage.items():
                self.usage[key] = self.usage.get(key, 0) + value

    def as_metrics(self):
        status = "unavailable"
        if self.measured_calls:
            status = "available" if self.measured_calls == self.calls else "partial"
        return {
            "call_count": self.calls,
            "duration_seconds": round(self.duration_seconds, 3),
            "token_usage": dict(self.usage) if self.usage is not None else None,
            "usage_status": status,
        }


def parse_codex_events(stdout):
    """Extract only final assistant text and allowlisted numeric usage."""
    final = ""
    usage = None
    saw_events = False
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            continue
        saw_events = True
        if event["type"] == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message":
                if isinstance(item.get("text"), str):
                    final = item["text"].strip()
        if event["type"] == "turn.completed" and isinstance(event.get("usage"), dict):
            numbers = {}
            for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
                value = event["usage"].get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    numbers[key] = value
            # Missing required counts are unavailable, never an invented zero.
            if "input_tokens" in numbers and "output_tokens" in numbers:
                if usage is None:
                    usage = {}
                for key, value in numbers.items():
                    usage[key] = usage.get(key, 0) + value
    return final, usage, saw_events


async def run_codex(
    prompt: str,
    *,
    cwd: Path | None = None,
    sandbox: str = "read-only",
    timeout: int = 600,
    output_schema: dict | None = None,
) -> tuple[str, RunDiagnostics]:
    """Return final text, safe diagnostics and measured usage from one child."""

    if os.environ.get("ACE_LLM_CHILD") == "1":
        raise RuntimeError("nested ACE LLM run refused (ACE_LLM_CHILD=1)")

    if sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
        raise ValueError(f"unsupported Codex sandbox: {sandbox!r}")

    run_cwd = (cwd or VAULT_ROOT).resolve()
    if not run_cwd.is_dir():
        raise RuntimeError(f"Codex working directory does not exist: {run_cwd}")

    with tempfile.TemporaryDirectory(prefix="ace-codex-") as temp_dir:
        output_path = Path(temp_dir) / "last-message.md"
        sandbox_args = [] if sandbox == "workspace-write" else ["--sandbox", sandbox]
        command = [
            CODEX_EXEC_PATH,
            "exec",
            "--ephemeral",
            "--json",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            *sandbox_args,
            "--model",
            CODEX_MODEL,
            "--config",
            f"model_reasoning_effort={CODEX_REASONING_EFFORT}",
            "--config",
            "project_doc_max_bytes=0",
            "--cd",
            str(run_cwd),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if sandbox == "workspace-write":
            # Compilation and file-back are intentionally writable, but they
            # run detached/non-interactively. Codex's automatic reviewer both
            # supplies the workspace-write sandbox and prevents a terminal
            # approval prompt from hanging the child.
            command.insert(-1, "--approve-for-me")
        if output_schema is not None:
            schema_path = Path(temp_dir) / 'output-schema.json'
            schema_path.write_text(json.dumps(output_schema), encoding='utf-8')
            command[-1:-1] = ['--output-schema', str(schema_path)]
        child_env = os.environ.copy()
        child_env["CODEX_ACE_BACKFILL_ENABLED"] = "0"
        child_env["CODEX_TURN_ENDED_FORWARD_SKY"] = "0"
        child_env["CLAUDE_INVOKED_BY"] = "ace_codex"
        child_env["ACE_FLUSH_ENGINE"] = "codex"
        child_env["ACE_LLM_CHILD"] = "1"

        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(run_cwd),
            env=child_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=max(30, timeout),
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            error = TimeoutError(f"Codex ACE run timed out after {timeout}s")
            error.diagnostics = RunDiagnostics(duration_seconds=time.monotonic() - started, calls=1)
            raise error from exc

        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        event_response, usage, saw_events = parse_codex_events(stdout_text)
        stderr_lines = RunDiagnostics(
            safe_codex_diagnostic_lines(stderr_text),
            usage=usage, duration_seconds=time.monotonic() - started, calls=1,
        )
        if process.returncode != 0:
            detail = " | ".join(stderr_lines[-10:])
            error = RuntimeError(
                f"codex exec failed with exit code {process.returncode}"
                + (f": {detail[-1000:]}" if detail else "; no safe diagnostic available")
            )
            error.diagnostics = stderr_lines
            raise error

        response = ""
        if output_path.exists():
            response = output_path.read_text(encoding="utf-8").strip()
        if not response:
            response = event_response if saw_events else stdout_text
        if not response:
            error = RuntimeError("codex exec returned no final message")
            error.diagnostics = stderr_lines
            raise error
        return response, stderr_lines
