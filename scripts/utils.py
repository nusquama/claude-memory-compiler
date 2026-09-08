"""Shared utilities for the personal knowledge base."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from config import (
    CONCEPTS_DIR,
    CONNECTIONS_DIR,
    DAILY_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
    LOG_FILE,
    QA_DIR,
    STATE_FILE,
)


# Conservative secret filters applied before content reaches the LLM or vault.
# They target explicit credential labels and well-known token formats. Ordinary
# technical identifiers such as task_id, model, and status remain unchanged.
_SECRET_NAME = (
    r"authorization|proxy-authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|client[_-]?secret|password|passwd|secret|token|"
    r"cookie|set-cookie"
)
_QUOTED_SECRET_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?(?:{_SECRET_NAME})[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
# The central ``ace.jsonb_is_clean`` check accepts an optional quote around
# the label (``'token': value``). Mirror it here so a message the adapter
# considers clean is never rejected by the database afterwards.
_PLAIN_SECRET_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?\b(?:{_SECRET_NAME})\b[\"']?\s*[:=]\s*)"
    r"(?![\"']?<REDACTED>)"
    r"(?P<value>(?:Bearer\s+)?[^\s,;&]+)"
)
_PROVIDER_TOKEN_RE = re.compile(
    r"\b(?:sk-(?:live-|test-|proj-)?[A-Za-z0-9_-]{16,}|"
    r"sbp_[A-Za-z0-9_]{20,}|sb_secret_[A-Za-z0-9_-]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[A-Z0-9]{16})\b"
)
_CLI_SECRET_RE = re.compile(
    rf"(?i)(?P<prefix>--(?:{_SECRET_NAME})\s+)(?P<value>[^\s]+)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
    r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)


# Exact mirror of the central ``ace.jsonb_is_clean`` string rules. The local
# redactor above is permissive by design; this final pass guarantees that a
# message accepted here is never rejected by the database afterwards.
_DB_MARKER_PAIR_RE = re.compile(
    rf"(?i)(^|[^A-Za-z0-9_])[\"']?(?:{_SECRET_NAME})[\"']?\s*[:=]\s*(?:<REDACTED>|'<REDACTED>'|\"<REDACTED>\")([^A-Za-z0-9_]|$)"
)
_DB_ASSIGN_RE = re.compile(
    rf"(?i)(?P<prefix>(?:^|[^A-Za-z0-9_])[\"']?(?:{_SECRET_NAME})[\"']?\s*[:=]\s*)(?P<value>[^\s,;&]+)"
)
_DB_FLAG_RE = re.compile(
    rf"(?i)(?P<prefix>--(?:{_SECRET_NAME})\s+)(?P<value>\S+)"
)
_DB_TOKEN_RE = re.compile(
    r"(^|[^A-Za-z0-9_])(sk-(?:live-|test-|proj-)?[A-Za-z0-9_-]{16,}|sbp_[A-Za-z0-9_]{20,}|"
    r"sb_secret_[A-Za-z0-9_-]{16,}|github_pat_[A-Za-z0-9_]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[A-Z0-9]{16})(?=[^A-Za-z0-9_-]|$)"
)


def _database_clean_pass(content: str) -> str:
    """Apply the database's own secret grammar so both sides agree."""
    # Postgres jsonb rejects NUL escapes outright.
    content = content.replace("\x00", "")
    probe = _DB_MARKER_PAIR_RE.sub(r"\1\2", content)
    if not (_DB_ASSIGN_RE.search(probe) or _DB_FLAG_RE.search(probe) or _DB_TOKEN_RE.search(probe)):
        return content

    def keep_marker(match: "re.Match[str]") -> str:
        value = match.group("value")
        if value.strip("\"'") == "<REDACTED>":
            return match.group(0)
        return f"{match.group('prefix')}<REDACTED>"

    content = _DB_ASSIGN_RE.sub(keep_marker, content)
    content = _DB_FLAG_RE.sub(keep_marker, content)
    content = _DB_TOKEN_RE.sub(r"\1<REDACTED>", content)
    # The database strips each ``name=<REDACTED>`` pair together with the
    # character that follows it, so two adjacent pairs can leave a bare
    # ``name =`` that still matches. When the mirrored check still fails,
    # drop the separator after the label: no assignment, nothing to leak.
    if not _database_string_is_clean(content):
        content = _DB_LABEL_SEPARATOR_RE.sub(r"\1 ", content)
    return content


_DB_LABEL_SEPARATOR_RE = re.compile(
    rf"(?i)([\"']?(?:{_SECRET_NAME})[\"']?)\s*[:=]\s*"
)
_DB_PROBE_STRIP_FLAG_RE = re.compile(
    rf"(?i)(^|[^A-Za-z0-9_])--(?:{_SECRET_NAME})\s+(?:<REDACTED>|'<REDACTED>'|\"<REDACTED>\")([^A-Za-z0-9_]|$)"
)
_DB_PROBE_STRIP_MARKER_RE = re.compile(r"(?i)(<REDACTED>|'<REDACTED>'|\"<REDACTED>\")")


def _database_string_is_clean(content: str) -> bool:
    """Mirror ``ace.jsonb_is_clean`` for one string, marker stripping included."""
    probe = _DB_MARKER_PAIR_RE.sub(r"\1\2", content)
    probe = _DB_PROBE_STRIP_FLAG_RE.sub(r"\1\2", probe)
    probe = _DB_PROBE_STRIP_MARKER_RE.sub("", probe)
    if _PRIVATE_KEY_RE.search(probe) or re.search(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----", probe):
        return False
    return not (_DB_ASSIGN_RE.search(probe) or _DB_FLAG_RE.search(probe) or _DB_TOKEN_RE.search(probe))


def redact_sensitive_text(content: str) -> str:
    """Replace explicit credentials with a stable non-secret marker."""
    redacted = _PRIVATE_KEY_RE.sub("<REDACTED>", content)
    redacted = _QUOTED_SECRET_RE.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}"
            f"<REDACTED>{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _PLAIN_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}<REDACTED>",
        redacted,
    )
    redacted = _CLI_SECRET_RE.sub(lambda match: f"{match.group('prefix')}<REDACTED>", redacted)
    redacted = _PROVIDER_TOKEN_RE.sub("<REDACTED>", redacted)
    return _database_clean_pass(redacted)


def sensitive_text_findings(content: str) -> dict[str, int]:
    """Count potential secrets by class without returning their values."""
    findings = {
        "private_key": len(_PRIVATE_KEY_RE.findall(content)),
        "quoted_credential": len(_QUOTED_SECRET_RE.findall(content)),
        "plain_credential": len(_PLAIN_SECRET_RE.findall(content)),
        "provider_token": len(_PROVIDER_TOKEN_RE.findall(content)),
        "cli_credential": len(_CLI_SECRET_RE.findall(content)),
    }
    return {name: count for name, count in findings.items() if count}


def safe_codex_diagnostic_lines(stderr_text: str) -> list[str]:
    """Keep system diagnostics while dropping echoed prompts and responses."""
    safe: list[str] = []
    seen: set[str] = set()
    for raw in stderr_text.splitlines():
        line = raw.strip()
        lowered = line.lower()
        timestamped = bool(
            re.match(r"^\d{4}-\d{2}-\d{2}t\S+\s+(?:warn|error)\b", line, re.IGNORECASE)
        )
        explicit_error = lowered.startswith(("error:", "fatal:"))
        known_failure = any(
            marker in lowered
            for marker in (
                "authentication_error",
                "invalid x-api-key",
                "rate limit",
                "timed out",
            )
        )
        if timestamped or explicit_error or known_failure:
            redacted = redact_sensitive_text(line)
            signature = re.sub(
                r"^\d{4}-\d{2}-\d{2}t\S+\s+",
                "",
                redacted,
                flags=re.IGNORECASE,
            )
            if signature not in seen:
                safe.append(redacted)
                seen.add(signature)
    return safe


# ── State management ──────────────────────────────────────────────────

def load_state() -> dict:
    """Load persistent state from state.json."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"ingested": {}, "query_count": 0, "last_lint": None, "total_cost": 0.0}


def save_state(state: dict) -> None:
    """Atomically save private state, including for a newly initialized vault."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=STATE_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, STATE_FILE)
    finally:
        Path(temporary).unlink(missing_ok=True)


# ── File hashing ──────────────────────────────────────────────────────

def file_hash(path: Path) -> str:
    """SHA-256 hash of a file (first 16 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ── Slug / naming ─────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


# ── Wikilink helpers ──────────────────────────────────────────────────

def extract_wikilinks(content: str) -> list[str]:
    """Extract internal article links as normalized concept IDs (e.g. `concepts/foo`).

    Supports two forms:
      - OKF markdown links `[label](/concepts/foo.md)` or `[label](concepts/foo.md)`
        (current, OKF v0.1 §5 — bundle-relative, `.md` suffix).
      - Legacy Obsidian `[[concepts/foo|alias#anchor]]` links (backward compat).
    Both normalize to `concepts/foo`. Only links into internal dirs
    (concepts/connections/qa/daily) are returned.
    """
    out: list[str] = []
    for target in re.findall(r"\]\(([^)]+)\)", content):          # OKF markdown links
        t = target.split("#", 1)[0].strip().lstrip("/")
        m = re.match(r"((?:concepts|connections|qa|daily)/.+)", t)
        if m:
            out.append(m.group(1).removesuffix(".md"))
    for link in re.findall(r"\[\[([^\]]+)\]\]", content):         # legacy wikilinks
        out.append(link.split("|", 1)[0].split("#", 1)[0].strip())
    return out


def wiki_article_exists(link: str) -> bool:
    """Check if a wikilinked article exists on disk."""
    path = KNOWLEDGE_DIR / f"{link}.md"
    return path.exists()


# ── Wiki content helpers ──────────────────────────────────────────────

def read_wiki_index() -> str:
    """Read the knowledge base index file."""
    if INDEX_FILE.exists():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "# Knowledge Base Index\n\n| Article | Summary | Compiled From | Updated |\n|---------|---------|---------------|---------|"


def read_all_wiki_content() -> str:
    """Read index + all wiki articles into a single string for context."""
    parts = [f"## INDEX\n\n{read_wiki_index()}"]

    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if not subdir.exists():
            continue
        for md_file in sorted(subdir.glob("*.md")):
            rel = md_file.relative_to(KNOWLEDGE_DIR)
            content = md_file.read_text(encoding="utf-8")
            parts.append(f"## {rel}\n\n{content}")

    return "\n\n---\n\n".join(parts)


def list_wiki_articles() -> list[Path]:
    """List all wiki article files."""
    articles = []
    for subdir in [CONCEPTS_DIR, CONNECTIONS_DIR, QA_DIR]:
        if subdir.exists():
            articles.extend(sorted(subdir.glob("*.md")))
    return articles


def list_raw_files() -> list[Path]:
    """List all daily log files."""
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("*.md"))


# ── Index helpers ─────────────────────────────────────────────────────

def count_inbound_links(target: str, exclude_file: Path | None = None) -> int:
    """Count how many wiki articles link to a given target.

    Uses extract_wikilinks so aliased/anchored links count correctly
    (`[[target|alias]]` and `[[target#section]]` both match `target`).
    """
    count = 0
    for article in list_wiki_articles():
        if article == exclude_file:
            continue
        content = article.read_text(encoding="utf-8")
        if target in extract_wikilinks(content):
            count += 1
    return count


def get_article_word_count(path: Path) -> int:
    """Count words in an article, excluding YAML frontmatter."""
    content = path.read_text(encoding="utf-8")
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:]
    return len(content.split())


def build_index_entry(rel_path: str, summary: str, sources: str, updated: str) -> str:
    """Build a single index table row."""
    link = rel_path.replace(".md", "")
    return f"| [[{link}]] | {summary} | {sources} | {updated} |"
