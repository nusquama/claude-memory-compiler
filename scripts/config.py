"""Path constants and configuration — project-aware via Obsidian vault.

Vault layout:
    <VAULT_ROOT>/
        _config/             ← this clone (TOOL_DIR)
        <project-A>/         ← per-project KB
            daily/
            knowledge/
            .state/
        <project-B>/
            ...

Project resolution: $CLAUDE_PROJECT_DIR (or cwd) → git rev-parse --show-toplevel.
If no git repo is found, PROJECT_DIR is None and consumers must skip silently.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone


# ── Vault layout (always defined) ──────────────────────────────────────
TOOL_DIR = Path(__file__).resolve().parent.parent       # …/_config
VAULT_ROOT = TOOL_DIR.parent                            # …/Claude_code
AGENTS_FILE = TOOL_DIR / "AGENTS.md"
SCRIPTS_DIR = TOOL_DIR / "scripts"
HOOKS_DIR = TOOL_DIR / "hooks"
ROOT_DIR = TOOL_DIR  # back-compat alias for any leftover reference


# ── Project resolution ────────────────────────────────────────────────
def resolve_project_dir() -> Path | None:
    """Return the per-project folder inside the vault, or None.

    Cascade:
    1. Read $CLAUDE_PROJECT_DIR (set by Claude Code when launching hooks).
       Fallback to os.getcwd() for manual `uv run python …` invocations.
    2. Run `git -C <start> rev-parse --show-toplevel`. If it fails, return None
       (no project detected → skip silently).
    3. Project name = basename of git toplevel. Reject names that start with "."
       or are empty (defensive).
    """
    start = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    toplevel = Path(result.stdout.strip())
    project_name = toplevel.name
    if not project_name or project_name.startswith("."):
        return None

    # Opt-in: only consider this an active KB project if the folder exists in
    # the vault. Users explicitly enable a project by running the cmc-init
    # skill (which creates the folder structure). Without that, hooks skip
    # silently — preventing accidental capture in random git repos.
    candidate = VAULT_ROOT / project_name
    if not candidate.is_dir():
        return None
    return candidate


PROJECT_DIR = resolve_project_dir()


# ── Per-project paths (only defined when PROJECT_DIR is not None) ─────
if PROJECT_DIR is not None:
    DAILY_DIR = PROJECT_DIR / "daily"
    KNOWLEDGE_DIR = PROJECT_DIR / "knowledge"
    CONCEPTS_DIR = KNOWLEDGE_DIR / "concepts"
    CONNECTIONS_DIR = KNOWLEDGE_DIR / "connections"
    QA_DIR = KNOWLEDGE_DIR / "qa"
    REPORTS_DIR = PROJECT_DIR / "reports"
    STATE_DIR = PROJECT_DIR / ".state"
    INDEX_FILE = KNOWLEDGE_DIR / "index.md"
    LOG_FILE = KNOWLEDGE_DIR / "log.md"
    STATE_FILE = STATE_DIR / "state.json"
    FLUSH_STATE_FILE = STATE_DIR / "last-flush.json"
    FLUSH_LOG = STATE_DIR / "flush.log"
else:
    DAILY_DIR = None
    KNOWLEDGE_DIR = None
    CONCEPTS_DIR = None
    CONNECTIONS_DIR = None
    QA_DIR = None
    REPORTS_DIR = None
    STATE_DIR = None
    INDEX_FILE = None
    LOG_FILE = None
    STATE_FILE = None
    FLUSH_STATE_FILE = None
    FLUSH_LOG = None


# ── Model selection ───────────────────────────────────────────────────
# Cheap models for the right job. Override via env vars if needed.
# Note: explicit model IDs bypass any global $ANTHROPIC_DEFAULT_*_MODEL
# aliasing the user may have set in ~/.claude/settings.json.
FLUSH_MODEL = os.environ.get("CMC_FLUSH_MODEL", "claude-haiku-4-7")
COMPILE_MODEL = os.environ.get("CMC_COMPILE_MODEL", "claude-sonnet-4-7")
QUERY_MODEL = os.environ.get("CMC_QUERY_MODEL", "claude-sonnet-4-7")


# ── Timezone ──────────────────────────────────────────────────────────
TIMEZONE = "Europe/Paris"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
