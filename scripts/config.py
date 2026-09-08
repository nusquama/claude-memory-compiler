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
from dataclasses import dataclass
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
@dataclass(frozen=True)
class ProjectRoute:
    """Stable source/destination identity for a captured conversation."""

    source_project: str | None
    source_cwd: str
    destination_project: str | None
    destination_dir: Path | None
    used_fallback: bool = False
    reason: str = ""


def _valid_project_name(value: str | None) -> bool:
    """Accept dot-prefixed project names (notably ``.agents``) safely."""
    if not value or value in {".", ".."}:
        return False
    # Project names originate from a basename. Reject separators so an
    # explicit env value cannot escape the vault root.
    return Path(value).name == value and "/" not in value and "\\" not in value


def canonical_git_root(start: str | Path) -> Path | None:
    """Return the main checkout root for a normal checkout or git worktree."""
    start_path = Path(start).expanduser()
    try:
        toplevel_res = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
        common_res = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if toplevel_res.returncode != 0:
        return None

    toplevel = Path(toplevel_res.stdout.strip()).resolve()
    if common_res.returncode == 0:
        common_dir = Path(common_res.stdout.strip())
        if not common_dir.is_absolute():
            common_dir = (start_path / common_dir).resolve()
        # A linked worktree reports its own top-level but shares the main
        # checkout's .git directory. Use the main checkout for identity.
        if common_dir.name == ".git" and common_dir.parent != toplevel:
            toplevel = common_dir.parent
    return toplevel


def canonical_project_name(cwd: str | Path | None) -> str | None:
    """Return the shared project identity for a source checkout."""
    if not cwd:
        return None
    root = canonical_git_root(cwd)
    if root is None:
        return None
    name = root.name
    return name if _valid_project_name(name) else None


def initialized_project_dir(
    vault_root: str | Path,
    project_name: str | None,
    *,
    require_knowledge: bool = True,
) -> Path | None:
    """Resolve a vault project using one shared initialization contract."""
    if not _valid_project_name(project_name):
        return None
    candidate = Path(vault_root) / str(project_name)
    if not candidate.is_dir():
        return None
    if require_knowledge and not (candidate / "knowledge").is_dir():
        return None
    return candidate


def resolve_project_route(
    source_cwd: str | Path | None,
    *,
    fallback_project: str | None = None,
    vault_root: str | Path = VAULT_ROOT,
    target_project: str | None = None,
) -> ProjectRoute:
    """Map source identity to an initialized destination without fallback routing."""
    source_cwd_text = str(source_cwd or "")
    source_project = canonical_project_name(source_cwd)

    if target_project:
        destination = initialized_project_dir(vault_root, target_project)
        return ProjectRoute(
            source_project=source_project,
            source_cwd=source_cwd_text,
            destination_project=destination.name if destination else None,
            destination_dir=destination,
            used_fallback=False,
            reason="forced_target" if destination else "forced_target_missing",
        )

    if source_project:
        destination = initialized_project_dir(vault_root, source_project)
        if destination is not None:
            return ProjectRoute(
                source_project=source_project,
                source_cwd=source_cwd_text,
                destination_project=destination.name,
                destination_dir=destination,
                used_fallback=False,
                reason="source_project",
            )

    return ProjectRoute(
        source_project=source_project,
        source_cwd=source_cwd_text,
        destination_project=None,
        destination_dir=None,
        used_fallback=False,
        reason="no_initialized_destination",
    )


def resolve_project_dir() -> Path | None:
    """Return the per-project folder inside the vault, or None.

    Cascade:
    0. If $ACE_PROJECT_DIR or $ACE_PROJECT is set, use that initialized vault
       project directly for an explicit maintenance or target operation.
    1. Read $CLAUDE_PROJECT_DIR (set by Claude Code when launching hooks).
       Fallback to os.getcwd() for manual `uv run python …` invocations.
    2. Resolve the canonical project root.
       - In a normal checkout: `git rev-parse --show-toplevel`.
       - In a git worktree: `--show-toplevel` returns the worktree path
         (e.g. `<repo>/.claude/worktrees/foo`), not the main repo. We use
         `--git-common-dir` to find the main `.git/` and take its parent.
       Both cases unify to "the directory whose name is the project name in
       the vault".
    3. Project name = basename of canonical root. Dot-prefixed names such as
       `.agents` are valid; only path traversal names are rejected.
    """
    explicit_dir = os.environ.get("ACE_PROJECT_DIR")
    if explicit_dir:
        candidate = Path(explicit_dir).expanduser().resolve()
        return candidate if candidate.is_dir() else None

    explicit_project = os.environ.get("ACE_PROJECT")
    if explicit_project:
        return initialized_project_dir(VAULT_ROOT, explicit_project)

    start = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    toplevel = canonical_git_root(start)
    if toplevel is None:
        return None
    project_name = toplevel.name
    if not _valid_project_name(project_name):
        return None

    # Opt-in: only consider this an active KB project if the folder exists in
    # the vault. Users explicitly enable a project by running the ace-init
    # skill (which creates the folder structure). Without that, hooks skip
    # silently — preventing accidental capture in random git repos.
    candidate = initialized_project_dir(VAULT_ROOT, project_name)
    if candidate is None:
        # Make silent failures visible: when we ARE in a git repo but the
        # project is not vaulted, write a single line to stderr so debug
        # logs show why hooks no-op. Claude Code surfaces hook stderr in
        # debug mode without blocking the session.
        if os.environ.get("ACE_DEBUG_RESOLUTION"):
            print(
                f"[ace] PROJECT_DIR=None: '{project_name}' not in vault "
                f"({VAULT_ROOT}). Run /ace-init to enable.",
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
# ACE has one execution engine and one model contract.  The Claude hooks are
# capture-only launchers; every extraction, scan, compile, query, and lint
# decision is made by this Codex child.  Keep these values fixed so an inherited
# Claude environment cannot silently switch the processor back to another model.
CODEX_MODEL = "gpt-5.6-luna"
CODEX_REASONING_EFFORT = "medium"
FLUSH_ENGINE = "codex"
FLUSH_MODEL = CODEX_MODEL
SCAN_MODEL = CODEX_MODEL
COMPILE_MODEL = CODEX_MODEL
QUERY_MODEL = CODEX_MODEL
CODEX_EXEC_PATH = os.environ.get("ACE_CODEX_EXEC_PATH", "codex")


# ── Conversation extraction limits & chunking ────────────────────────
# Architecture: map-reduce. flush.py never truncates. If a conversation
# exceeds FLUSH_SINGLE_PASS_THRESHOLD, it is split into chunks of
# FLUSH_CHUNK_SIZE chars at turn boundaries, each chunk gets its own LLM
# call (partial extraction), then a single consolidation call merges the
# partial summaries into the final daily log entry. No content is dropped
# regardless of session length.
#
# The Codex child has a bounded context window. Per-chunk budget = 120K chars,
# leaving comfortable headroom for the
# prompt itself (~10K), retry buffer, and output (~5K).
#
# FLUSH_MAX_CHARS is now a hard safety cap, not the working budget — it
# only protects against pathological inputs (corrupted transcripts, etc.).
# Real conversations of any realistic length flow through the chunking
# pipeline without loss.
FLUSH_MAX_TURNS = int(os.environ.get("ACE_FLUSH_MAX_TURNS", "10000"))
FLUSH_MAX_CHARS = int(os.environ.get("ACE_FLUSH_MAX_CHARS", "5000000"))  # 5M — hard safety cap
FLUSH_SINGLE_PASS_THRESHOLD = int(os.environ.get("ACE_FLUSH_SINGLE_PASS_THRESHOLD", "200000"))
FLUSH_CHUNK_SIZE = int(os.environ.get("ACE_FLUSH_CHUNK_SIZE", "120000"))


# ── Timezone ──────────────────────────────────────────────────────────
TIMEZONE = "Europe/Paris"


def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def today_iso() -> str:
    """Current date in ISO 8601 format."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
