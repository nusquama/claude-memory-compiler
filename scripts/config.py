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
    0. If $CMC_PROJECT_DIR or $CMC_PROJECT is set, use that vault project
       directly. This is for maintenance/backfill jobs that are not running
       from a matching git checkout, such as the shared `Conversations` vault.
    1. Read $CLAUDE_PROJECT_DIR (set by Claude Code when launching hooks).
       Fallback to os.getcwd() for manual `uv run python …` invocations.
    2. Resolve the canonical project root.
       - In a normal checkout: `git rev-parse --show-toplevel`.
       - In a git worktree: `--show-toplevel` returns the worktree path
         (e.g. `<repo>/.claude/worktrees/foo`), not the main repo. We use
         `--git-common-dir` to find the main `.git/` and take its parent.
       Both cases unify to "the directory whose name is the project name in
       the vault".
    3. Project name = basename of canonical root. Reject names that start
       with "." or are empty (defensive).
    """
    explicit_dir = os.environ.get("CMC_PROJECT_DIR")
    if explicit_dir:
        candidate = Path(explicit_dir).expanduser().resolve()
        return candidate if candidate.is_dir() else None

    explicit_project = os.environ.get("CMC_PROJECT")
    if explicit_project:
        candidate = VAULT_ROOT / explicit_project
        return candidate if candidate.is_dir() else None

    start = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        toplevel_res = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
        common_res = subprocess.run(
            ["git", "-C", start, "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if toplevel_res.returncode != 0:
        return None

    toplevel = Path(toplevel_res.stdout.strip())

    # Worktree detection: --git-common-dir points to the MAIN .git/ even from
    # inside a worktree. If common_dir != toplevel/.git, we're in a worktree
    # and the canonical project root is the parent of the common dir.
    if common_res.returncode == 0:
        common_dir = Path(common_res.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (Path(start) / common_dir).resolve()
        if common_dir.name == ".git" and common_dir.parent != toplevel:
            toplevel = common_dir.parent

    project_name = toplevel.name
    if not project_name or project_name.startswith("."):
        return None

    # Opt-in: only consider this an active KB project if the folder exists in
    # the vault. Users explicitly enable a project by running the cmc-init
    # skill (which creates the folder structure). Without that, hooks skip
    # silently — preventing accidental capture in random git repos.
    candidate = VAULT_ROOT / project_name
    if not candidate.is_dir():
        # Make silent failures visible: when we ARE in a git repo but the
        # project is not vaulted, write a single line to stderr so debug
        # logs show why hooks no-op. Claude Code surfaces hook stderr in
        # debug mode without blocking the session.
        if os.environ.get("CMC_DEBUG_RESOLUTION"):
            print(
                f"[cmc] PROJECT_DIR=None: '{project_name}' not in vault "
                f"({VAULT_ROOT}). Run /cmc-init to enable.",
                file=__import__("sys").stderr,
            )
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
#
# Defaults verified live against the Claude Agent SDK on 2026-04-27.
# Latest available per tier: Haiku 4.5, Sonnet 4.6, Opus 4.7. Versions
# are NOT synchronized across tiers — there is no Haiku 4.7 or 4.6, and
# no Opus 4.6.
# The daily log is a lossy bottleneck: whatever the extractor drops at this
# stage is gone for good. compile.py and query.py operate on the daily log,
# not the original source, so they cannot recover lost nuance. Both
# extractors (flush, scan_md) therefore default to Sonnet to preserve the
# rationale, gotchas, and abandoned paths that lower-tier models tend to
# strip. Haiku is still available via env var for cost-bounded use cases
# (massive sessions, repos with hundreds of low-density .md files).
FLUSH_MODEL = os.environ.get("CMC_FLUSH_MODEL", "claude-sonnet-4-6")
SCAN_MODEL = os.environ.get("CMC_SCAN_MODEL", "claude-sonnet-4-6")
COMPILE_MODEL = os.environ.get("CMC_COMPILE_MODEL", "claude-sonnet-4-6")
QUERY_MODEL = os.environ.get("CMC_QUERY_MODEL", "claude-sonnet-4-6")


# ── Conversation extraction limits & chunking ────────────────────────
# Architecture: map-reduce. flush.py never truncates. If a conversation
# exceeds FLUSH_SINGLE_PASS_THRESHOLD, it is split into chunks of
# FLUSH_CHUNK_SIZE chars at turn boundaries, each chunk gets its own LLM
# call (partial extraction), then a single consolidation call merges the
# partial summaries into the final daily log entry. No content is dropped
# regardless of session length.
#
# Sonnet 4.6 has a 200K-token context window (~800K chars at 4 chars/token).
# Per-chunk budget = 120K chars, leaving comfortable headroom for the
# prompt itself (~10K), retry buffer, and output (~5K).
#
# FLUSH_MAX_CHARS is now a hard safety cap, not the working budget — it
# only protects against pathological inputs (corrupted transcripts, etc.).
# Real conversations of any realistic length flow through the chunking
# pipeline without loss.
FLUSH_MAX_TURNS = int(os.environ.get("CMC_FLUSH_MAX_TURNS", "10000"))
FLUSH_MAX_CHARS = int(os.environ.get("CMC_FLUSH_MAX_CHARS", "5000000"))  # 5M — hard safety cap
FLUSH_SINGLE_PASS_THRESHOLD = int(os.environ.get("CMC_FLUSH_SINGLE_PASS_THRESHOLD", "200000"))
FLUSH_CHUNK_SIZE = int(os.environ.get("CMC_FLUSH_CHUNK_SIZE", "120000"))


# ── Timezone ──────────────────────────────────────────────────────────
TIMEZONE = "Europe/Paris"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
