"""
Memory flush agent - extracts important knowledge from conversation context.

Spawned by session-end.py or pre-compact.py as a background process. Reads
pre-extracted conversation context from a .md file, uses the Claude Agent SDK
to decide what's worth saving, and appends the result to today's daily log.

Usage:
    uv run python flush.py <context_file.md> <session_id>
"""

from __future__ import annotations

# Recursion prevention: set this BEFORE any imports that might trigger Claude
import os
os.environ["CLAUDE_INVOKED_BY"] = "memory_flush"

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    DAILY_DIR,
    FLUSH_LOG as LOG_FILE,
    FLUSH_MODEL,
    FLUSH_STATE_FILE as STATE_FILE,
    PROJECT_DIR,
    SCRIPTS_DIR,
    STATE_DIR,
    TOOL_DIR as ROOT,
)

# No project detected → nothing to flush. Exit silently before any I/O.
if PROJECT_DIR is None:
    sys.exit(0)

PROJECT_DIR.mkdir(parents=True, exist_ok=True)
DAILY_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

# Set up file-based logging so we can verify the background process ran.
# The parent process sends stdout/stderr to DEVNULL (to avoid the inherited
# file handle bug on Windows), so this is our only observability channel.
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_flush_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_flush_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state), encoding="utf-8")


def append_to_daily_log(content: str, section: str = "Session") -> None:
    """Append content to today's daily log."""
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    entry = f"### {section} ({time_str})\n\n{content}\n\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def detect_failure_cause(stderr_lines: list[str], exc_message: str) -> str:
    """Best-effort one-line label for why the bundled CLI failed."""
    text = (" ".join(stderr_lines) + " " + exc_message).lower()
    if "401" in text or "invalid x-api-key" in text or "authentication_error" in text:
        return "authentication failed (likely v2.1.92 intermittent 401 bug)"
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return "rate limited"
    if "shell is already running" in text:
        return "concurrent CLI invocation"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "not a valid model" in text or "invalid model" in text:
        return "invalid model name"
    return "see flush.log for details"


def append_error_marker_to_daily(session_id: str, cause: str) -> None:
    """Write a brief, lisible error marker to today's daily log.

    Replaces the previous behaviour of dumping the full FLUSH_ERROR string
    (with traceback) into the daily log. The detail still lives in
    flush.log; the daily log just gets a one-paragraph signal so the
    failure stays visible without polluting the knowledge base.
    """
    today = datetime.now(timezone.utc).astimezone()
    log_path = DAILY_DIR / f"{today.strftime('%Y-%m-%d')}.md"

    if not log_path.exists():
        DAILY_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"# Daily Log: {today.strftime('%Y-%m-%d')}\n\n## Sessions\n\n## Memory Maintenance\n\n",
            encoding="utf-8",
        )

    time_str = today.strftime("%H:%M")
    short_id = (session_id or "unknown")[:8]
    entry = (
        f"### [ERROR] Memory Flush Failed ({time_str})\n\n"
        f"Session `{short_id}`: {cause}\n\n"
        f"Full details: `{LOG_FILE}`\n\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


RETRY_DELAYS = (5, 15, 45)  # seconds before retries 2, 3, 4

# Knobs (overridable via env). Backfill sets MAX_RETRIES=1 to avoid
# multiplying the per-session wait; SessionEnd hooks keep the default.
MAX_ATTEMPTS = max(1, int(os.environ.get("CMC_FLUSH_MAX_RETRIES", "3")))
ATTEMPT_TIMEOUT = max(30, int(os.environ.get("CMC_FLUSH_ATTEMPT_TIMEOUT", "180")))


def _is_transient_error(exc: Exception, stderr_text: str) -> bool:
    """Decide if a query() failure is worth retrying.

    Transient: auth races (401), rate limits (429), concurrent CLI lockfile
    collisions, generic "exit code 1" with no further detail.
    Non-transient: model validation errors, malformed prompts, etc.
    """
    msg = str(exc).lower()
    text = stderr_text.lower()
    # Hard non-transient signals — don't waste time retrying
    if "not a valid model" in text or "invalid model" in text:
        return False
    if "invalid api key" in text or "authentication is currently not supported" in text:
        return False
    # Transient signals
    if "401" in text or "auth" in text:
        return True
    if "429" in text or "rate limit" in text:
        return True
    if "shell is already running" in text:
        return True
    # Default: retry generic "exit code 1" failures (most common)
    if "exit code 1" in msg or "command failed" in msg:
        return True
    return False


async def run_flush(context: str) -> tuple[str, list[str]]:
    """Use Claude Agent SDK to extract important knowledge from conversation context.

    Returns (response_text, captured_stderr_lines). On success the response
    is the LLM output; on failure it's a "FLUSH_ERROR: ..." marker. The
    captured stderr from the bundled CLI is returned so callers can detect
    the failure cause (auth/rate/concurrency/etc.).
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    prompt = f"""Review the conversation context below and extract only what is worth preserving long-term.
Do NOT use any tools — just return plain text.
Do NOT reproduce raw conversation text or quotes from the conversation.
Write in your own words, in a structured, encyclopedia-style format.

Format your response as a structured daily log entry with ONLY the sections that have real content:

**Context:** [One line about what the user was working on]

**Key Exchanges:**
- [Summarize important discussions, decisions discovered, or answers — no raw quotes]

**Decisions Made:**
- [Technical or architectural decisions with their rationale]

**Lessons Learned:**
- [Non-obvious gotchas, patterns, or insights worth remembering]

**Action Items:**
- [Explicit follow-ups or TODOs mentioned]

Rules:
- Never reproduce raw user/assistant dialogue
- Skip routine tool calls, file reads, trivial clarifications
- Skip anything that's obvious from the code or already well-known
- If the session has nothing worth preserving, respond with exactly: FLUSH_OK

## Conversation Context

{context}"""

    captured_stderr: list[str] = []

    def stderr_callback(line: str) -> None:
        # Per-line callback fired by the SDK transport (subprocess_cli.py).
        # Without this, stderr from the bundled CLI is irretrievable —
        # the SDK hardcodes "Check stderr output for details" as the
        # exception's stderr field (anthropics/claude-agent-sdk-python#800).
        text = line.rstrip()
        if text:
            captured_stderr.append(text)
            logging.warning("[bundled CLI] %s", text)

    last_exc: Exception | None = None
    response = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = RETRY_DELAYS[min(attempt - 2, len(RETRY_DELAYS) - 1)]
            logging.info("Retry attempt %d/%d after %ds", attempt, MAX_ATTEMPTS, delay)
            await asyncio.sleep(delay)

        attempt_stderr_start = len(captured_stderr)
        response = ""

        async def _run_query() -> str:
            local = ""
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=str(ROOT),
                    model=FLUSH_MODEL,
                    allowed_tools=[],
                    max_turns=2,
                    stderr=stderr_callback,
                ),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            local += block.text
                elif isinstance(message, ResultMessage):
                    pass
            return local

        try:
            response = await asyncio.wait_for(_run_query(), timeout=ATTEMPT_TIMEOUT)
            return response, captured_stderr
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"bundled CLI hung for >{ATTEMPT_TIMEOUT}s — killed")
            logging.warning(
                "Attempt %d/%d timed out after %ds (bundled CLI hung)",
                attempt, MAX_ATTEMPTS, ATTEMPT_TIMEOUT,
            )
            # Timeouts are always transient — keep retrying within MAX_ATTEMPTS
            continue
        except Exception as e:
            last_exc = e
            this_attempt_stderr = "\n".join(captured_stderr[attempt_stderr_start:])
            logging.warning(
                "Attempt %d/%d failed: %s",
                attempt, MAX_ATTEMPTS, e,
            )
            if not _is_transient_error(e, this_attempt_stderr):
                logging.info("Error is non-transient — skipping further retries")
                break

    import traceback
    logging.error(
        "Agent SDK error after %d attempts: %s\n%s",
        attempt, last_exc, traceback.format_exc(),
    )
    response = f"FLUSH_ERROR: {type(last_exc).__name__}: {last_exc}"
    return response, captured_stderr


COMPILE_AFTER_HOUR = 18  # 6 PM local time


def maybe_trigger_compilation() -> None:
    """If it's past the compile hour and today's log hasn't been compiled, run compile.py."""
    import subprocess as _sp

    now = datetime.now(timezone.utc).astimezone()
    if now.hour < COMPILE_AFTER_HOUR:
        return

    # Check if today's log has already been compiled
    today_log = f"{now.strftime('%Y-%m-%d')}.md"
    from config import STATE_FILE as compile_state_file
    if compile_state_file.exists():
        try:
            compile_state = json.loads(compile_state_file.read_text(encoding="utf-8"))
            ingested = compile_state.get("ingested", {})
            if today_log in ingested:
                # Already compiled today - check if the log has changed since
                from hashlib import sha256
                log_path = DAILY_DIR / today_log
                if log_path.exists():
                    current_hash = sha256(log_path.read_bytes()).hexdigest()[:16]
                    if ingested[today_log].get("hash") == current_hash:
                        return  # log unchanged since last compile
        except (json.JSONDecodeError, OSError):
            pass

    compile_script = SCRIPTS_DIR / "compile.py"
    if not compile_script.exists():
        return

    logging.info("End-of-day compilation triggered (after %d:00)", COMPILE_AFTER_HOUR)

    cmd = ["uv", "run", "--directory", str(ROOT), "python", str(compile_script)]

    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    try:
        log_handle = open(str(STATE_DIR / "compile.log"), "a")
        _sp.Popen(cmd, stdout=log_handle, stderr=_sp.STDOUT, cwd=str(ROOT), **kwargs)
    except Exception as e:
        logging.error("Failed to spawn compile.py: %s", e)


# Concurrency lock: prevents two flush.py instances racing for the bundled
# Claude CLI. Both SessionEnd hooks firing across multiple projects and
# /cmc-scan running while a Claude Code session is active can lead to
# concurrent invocations of the bundled CLI. The CLI v2.1.92 has an open
# auth race bug (anthropics/claude-code#44100) that surfaces as `exit code
# 1, Check stderr output for details`. The lock is VAULT-WIDE (lives in
# _config/.state/, not in per-project .state/) so that flushes from
# different projects also serialise.
LOCK_FILE = ROOT / ".state" / "flush.lock"
LOCK_STALE_SECONDS = 600  # 10 min — beyond any realistic flush duration
LOCK_WAIT_TIMEOUT = 90    # max seconds to wait for another flush to finish
LOCK_POLL_INTERVAL = 2    # check every 2s


def acquire_flush_lock() -> bool:
    """Acquire an exclusive flush lock for the whole vault.

    Waits up to LOCK_WAIT_TIMEOUT seconds for another flush to finish
    before giving up. Returns True if the lock was acquired, False if
    another flush.py is still holding it after the wait window.

    Caller can bypass the lock by setting CMC_FLUSH_SKIP_LOCK=1 — used
    by backfill.py's --parallel mode where the caller manages the
    concurrency budget itself.
    """
    if os.environ.get("CMC_FLUSH_SKIP_LOCK") == "1":
        return True
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    waited = 0
    while LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except OSError:
            age = 0
        if age >= LOCK_STALE_SECONDS:
            logging.warning("Stale lock found (%.1fs old) — overriding", age)
            break
        if waited >= LOCK_WAIT_TIMEOUT:
            logging.info(
                "Another flush.py held the lock for >%ds, skipping (lock %.1fs old)",
                waited, age,
            )
            return False
        time.sleep(LOCK_POLL_INTERVAL)
        waited += LOCK_POLL_INTERVAL
    try:
        LOCK_FILE.write_text(f"{os.getpid()} {time.time()}", encoding="utf-8")
    except OSError as e:
        logging.error("Failed to write lock file: %s", e)
        return False
    if waited > 0:
        logging.info("Acquired flush lock after waiting %ds", waited)
    return True


def release_flush_lock() -> None:
    if os.environ.get("CMC_FLUSH_SKIP_LOCK") == "1":
        return
    LOCK_FILE.unlink(missing_ok=True)


def main():
    if len(sys.argv) < 3:
        logging.error("Usage: %s <context_file.md> <session_id>", sys.argv[0])
        sys.exit(1)

    context_file = Path(sys.argv[1])
    session_id = sys.argv[2]

    logging.info("flush.py started for session %s, context: %s", session_id, context_file)

    if not context_file.exists():
        logging.error("Context file not found: %s", context_file)
        return

    # Deduplication: skip if same session was flushed within 60 seconds
    state = load_flush_state()
    if (
        state.get("session_id") == session_id
        and time.time() - state.get("timestamp", 0) < 60
    ):
        logging.info("Skipping duplicate flush for session %s", session_id)
        context_file.unlink(missing_ok=True)
        return

    # Acquire concurrency lock — only one flush.py may invoke the Agent SDK
    # across the entire vault at a time. See comment near LOCK_FILE definition.
    if not acquire_flush_lock():
        # Don't unlink the context file — let the running instance finish
        # with it, or leave it for the next manual run.
        return

    try:
        # Read pre-extracted context
        context = context_file.read_text(encoding="utf-8").strip()
        if not context:
            logging.info("Context file is empty, skipping")
            context_file.unlink(missing_ok=True)
            return

        logging.info("Flushing session %s: %d chars", session_id, len(context))

        # Run the LLM extraction (now retries internally + captures stderr)
        response, captured_stderr = asyncio.run(run_flush(context))

        if "FLUSH_OK" in response:
            logging.info("Result: FLUSH_OK (skipped)")
        elif "FLUSH_ERROR" in response:
            logging.error("Result: %s", response)
            cause = detect_failure_cause(captured_stderr, response)
            logging.error("Detected cause: %s", cause)
            append_error_marker_to_daily(session_id, cause)
        else:
            logging.info("Result: saved to daily log (%d chars)", len(response))
            append_to_daily_log(response, "Session")

        # Update dedup state
        save_flush_state({"session_id": session_id, "timestamp": time.time()})

        # Clean up context file
        context_file.unlink(missing_ok=True)

        # End-of-day auto-compilation: if it's past the compile hour and today's
        # log hasn't been compiled yet, trigger compile.py in the background.
        maybe_trigger_compilation()

        logging.info("Flush complete for session %s", session_id)
    finally:
        release_flush_lock()


if __name__ == "__main__":
    main()
