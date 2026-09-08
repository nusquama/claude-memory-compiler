"""
Scan Markdown files in the current git repo and feed summaries into the
project's daily log.

A second ingestion source for the knowledge base, alongside session
transcripts. Walks the repo's *.md files, filters by date / hash dedup,
asks Codex Luna Medium to summarise each one, then appends the result to today's
daily log under a `### MD Scan: <path> (HH:MM)` section. The existing
compile.py pipeline picks it up from there.

Usage:
    uv run python scripts/scan_md.py                    # changed since last scan
    uv run python scripts/scan_md.py --init             # full rescan, clears state
    uv run python scripts/scan_md.py --days 7           # files modified in last 7 days
    uv run python scripts/scan_md.py --since 2026-04-01 # files modified since date
    uv run python scripts/scan_md.py --all              # ignore hash dedup
    uv run python scripts/scan_md.py --cwd /repo        # source checkout to scan
    uv run python scripts/scan_md.py --path /repo       # override repo root (legacy alias)
    uv run python scripts/scan_md.py --dry-run          # preview, no LLM calls
"""

from __future__ import annotations

# Recursion prevention: set BEFORE any import that might trigger another ACE child
import os
os.environ.setdefault("CLAUDE_INVOKED_BY", "scan_md")

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import (
    DAILY_DIR,
    PROJECT_DIR,
    STATE_DIR,
    TOOL_DIR as ROOT,
    VAULT_ROOT,
    now_iso,
)
from codex_runner import run_codex
from utils import file_hash, load_state, redact_sensitive_text, save_state

# Reuse the daily-log writer, lock, and retry policy from flush.py rather
# than duplicate them. flush.py is import-safe (its main() is gated under
# __name__ == "__main__").
from flush import (
    MAX_ATTEMPTS,
    RETRY_DELAYS,
    _is_transient_error,
    acquire_flush_lock,
    append_to_daily_log,
    release_flush_lock,
)


# File-type detection — drives the extraction policy per file. A CHANGELOG
# is mostly noise to a knowledge base ("fix typo", "bump deps") with a few
# real architectural decisions; a "drafts/note.md" is exploratory and must
# default to [Hypothèse]; a README is established. We branch the prompt on
# this type rather than running one-size-fits-all.
CHANGELOG_NAMES = {
    "CHANGELOG.md", "CHANGELOG", "CHANGES.md", "HISTORY.md", "RELEASES.md",
    "RELEASE_NOTES.md", "NEWS.md",
}
PLAN_NAMES = {"ROADMAP.md", "TODO.md", "PLAN.md", "PLANS.md", "BACKLOG.md"}
PLAN_DIR_SEGMENTS = {"plans", "roadmap", "todo", "todos"}
REFLECTION_DIR_SEGMENTS = {
    "drafts", "draft", "notes", "note", "journal", "journals",
    "brainstorm", "brainstorms", "reflections", "reflection",
    "thoughts", "ideas", "scratch", "scratchpad", "wip",
}


def detect_md_type(rel_path: str) -> str:
    """Classify an .md file into one of: doc / changelog / reflection / plan.

    Heuristics are filename- and path-segment-based. Any path containing a
    reflection segment wins over plan/changelog/doc, then plan, then
    changelog. Default = doc. Detection is conservative: when uncertain,
    return doc (standard policy applies).
    """
    p = rel_path.replace("\\", "/")
    parts = set(p.split("/")[:-1])
    base = p.rsplit("/", 1)[-1]

    if parts & REFLECTION_DIR_SEGMENTS:
        return "reflection"
    if base in PLAN_NAMES or (parts & PLAN_DIR_SEGMENTS):
        return "plan"
    if base in CHANGELOG_NAMES:
        return "changelog"
    return "doc"


EXCLUDED_DIR_SEGMENTS = {
    ".git", "node_modules", "vendor", "dist", "build",
    ".next", ".venv", "venv", "__pycache__", ".turbo", ".cache",
    "target", ".gradle", ".idea", ".vscode", "worktrees",
}

MAX_FILE_BYTES = 100 * 1024  # 100 KB; truncate longer files (keep head)
ATTEMPT_TIMEOUT = max(30, int(os.environ.get("ACE_SCAN_ATTEMPT_TIMEOUT", "120")))


def find_repo_root(override: str | None, cwd: str | None = None) -> Path | None:
    """Resolve the git repo root. Refuse the vault itself as a target.

    Cascade for the search starting point:
        1. --path argument (resolved as-is, no git lookup needed)
        2. --cwd argument (the explicit source checkout selected by ACE)
        3. $ACE_PROJECT_CWD (an explicit source checkout from a caller)
        4. $CLAUDE_PROJECT_DIR (legacy hook context)
        5. $PWD (preserved by the shell across `uv run --directory`)
        6. os.getcwd()

    The fallbacks matter because the typical UX is `uv run --directory _config
    python scripts/scan_md.py`, which changes cwd to _config before Python
    starts — `Path.cwd()` alone would always return the vault.  ``ACE_PROJECT_DIR``
    intentionally is not used here: it identifies the destination knowledge
    directory, while ``--cwd`` identifies the source checkout to scan.
    """
    if override:
        candidate = Path(override).resolve()
        if not candidate.is_dir():
            print(f"error: --path {override} does not exist or is not a directory", file=sys.stderr)
            return None
        return candidate

    start = (
        cwd
        or os.environ.get("ACE_PROJECT_CWD")
        or os.environ.get("CLAUDE_PROJECT_DIR")
        or os.environ.get("PWD")
        or str(Path.cwd())
    )
    try:
        result = subprocess.run(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def is_vault_internal(root: Path) -> bool:
    """True if the resolved root is the vault, the tool dir, or any
    per-project KB folder inside the vault. Prevents the scanner from
    self-ingesting daily/ and knowledge/."""
    try:
        root.relative_to(VAULT_ROOT.resolve())
    except ValueError:
        return False
    return True


def discover_md_files(root: Path) -> list[Path]:
    """Recursively list .md files, skipping noise directories."""
    results: list[Path] = []
    for path in root.rglob("*.md"):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(root).parts)
        if rel_parts & EXCLUDED_DIR_SEGMENTS:
            continue
        results.append(path)
    return sorted(results)


def parse_since(value: str) -> float:
    """Return a unix timestamp at midnight (local) for YYYY-MM-DD."""
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as e:
        raise SystemExit(f"error: --since expects YYYY-MM-DD, got {value!r} ({e})")
    local = d.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return local.timestamp()


def filter_files(
    files: list[Path],
    root: Path,
    state: dict,
    *,
    since_ts: float | None,
    ignore_hash: bool,
) -> list[tuple[Path, str, float]]:
    """Apply mtime + hash filters. Returns (path, rel, mtime) for each
    file that should be (re)scanned."""
    scanned = state.get("scanned_md", {})
    out: list[tuple[Path, str, float]] = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        mtime = stat.st_mtime
        if since_ts is not None and mtime < since_ts:
            continue
        rel = path.relative_to(root).as_posix()
        if not ignore_hash:
            prev = scanned.get(rel)
            if prev and prev.get("hash") == file_hash(path):
                continue
        out.append((path, rel, mtime))
    return out


TYPE_POLICIES = {
    "doc": (
        "Type 'doc' (README, docs/, ADR, spec). Politique standard:\n"
        "- Préserve les décisions [Décidé] avec rationale, les faits [Établi],\n"
        "  les contraintes, les gotchas [Découvert].\n"
        "- Filtre les généralités, descriptions méta, faits universels."
    ),
    "changelog": (
        "Type 'changelog'. Politique restrictive:\n"
        "- Extrais UNIQUEMENT les décisions architecturales, breaking changes,\n"
        "  deprecations annoncées, rationale derrière un refactor majeur,\n"
        "  changements de comportement non-évidents.\n"
        "- SKIP: 'fix typo', 'bump deps', 'update version', 'rename X to Y' sans\n"
        "  rationale, 'improve perf 5%', changements purement cosmétiques.\n"
        "- Si une entrée n'a pas d'impact externe ou architectural, omets-la.\n"
        "- Si tout le changelog est routinier, réponds: SKIP."
    ),
    "reflection": (
        "Type 'reflection' (drafts/, notes/, journal/, brainstorm/). Le contenu\n"
        "est exploratoire par nature. Politique:\n"
        "- TOUT est [Hypothèse] par défaut, SAUF si le texte dit explicitement\n"
        "  'j'ai décidé X', 'we chose X', 'décision actée' (alors [Décidé]).\n"
        "- NE promote PAS une note exploratoire en fait établi. Si le texte\n"
        "  dit 'on pourrait essayer X', c'est [Hypothèse], pas [Décidé].\n"
        "- Capture agressivement les chemins abandonnés (idée explorée puis\n"
        "  rejetée) — c'est le signal principal d'un brouillon.\n"
        "- Si la note est trop incomplète/vague pour extraire un signal\n"
        "  cohérent, réponds: SKIP."
    ),
    "plan": (
        "Type 'plan' (ROADMAP, TODO, plans/). Le contenu est intention future.\n"
        "Politique:\n"
        "- TOUT est [Hypothèse] (intention non encore réalisée), SAUF si\n"
        "  marqué explicitement 'fait', 'livré', 'shipped', 'done', '[x]'\n"
        "  (alors [Décidé] ou [Établi] selon le contexte).\n"
        "- Capture les contraintes mentionnées (deadlines, dépendances,\n"
        "  blockers) comme [Établi] seulement si présentées comme acquises\n"
        "  ('on doit livrer avant X', 'X dépend de Y' factuellement).\n"
        "- N'invente pas de rationale pour une intention non justifiée."
    ),
}


async def summarise_file(rel: str, mtime_iso: str, content: str, file_type: str) -> tuple[str, list[str]]:
    """Call Codex Luna Medium to summarise a single .md file.

    Returns (text, stderr_lines).
    Mirrors flush.run_flush retry/transient logic."""
    content = redact_sensitive_text(content)
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        # Keep the head — title, intro, key sections usually live there.
        # Encode/decode to avoid splitting a multi-byte char.
        encoded = content.encode("utf-8")[:MAX_FILE_BYTES]
        content = encoded.decode("utf-8", errors="ignore")
        content += "\n\n[...truncated...]"

    type_policy = TYPE_POLICIES.get(file_type, TYPE_POLICIES["doc"])

    prompt = f"""Tu extrais d'un fichier Markdown TOUT ce qu'il contient comme atomes
d'information distincts. Le résultat alimente `compile.py` (Codex Luna Medium aval)
qui croise toutes les sources, déduplique, et structure en concept articles.

**Tu es un EXTRACTEUR, pas un CURATEUR.** À ton étage, le job est de NE
RIEN PERDRE. Si tu hésites à inclure quelque chose, INCLUS-LE. compile.py
fera le tri. Une info incluse en trop est éliminable au compile ; une info
manquée n'est pas récupérable.

**Sortie pour LLM agent uniquement** — pas pour humains. Optimise pour
densité d'information, pas lisibilité. **Grammaire compressée (caveman):**
drop articles (le/la/un/une/the/a), drop politesses, drop sujets pronoms
(I/we/the user/nous/on), utilise → pour causation, | pour séparation, @
pour localisation/timing. Code, paths, identifiers, error strings, IDs,
versions, nombres restent **BYTE-EXACT**. Une atom par ligne.

**Pointeur > duplication.** Le fichier source existe sous git. Ne
RECOPIE PAS le contenu — capture les atomes (décisions, valeurs, gotchas)
qui ne sont pas évidents depuis le filename + headers de section. Pour
les sections triviales (intro, formatting, exemples génériques), une
ligne `structure:` suffit.

N'utilise AUCUN outil. Réponds en texte brut.

## Type de fichier détecté

{type_policy}

## Règles de capture (par-dessus la politique de type)

1. **Exhaustivité.** Capture chaque atome distinct: décisions, valeurs,
   structures, exemples cités, références à d'autres fichiers/services.
   Pas de quota maximum, pas de filtre "déjà connu". Si tu hésites,
   inclus.

2. **Marque la provenance** entre crochets au début de chaque bullet:
   - `[Établi]` — fait vérifiable, présenté comme acquis
   - `[Décidé]` — choix explicite ("nous avons choisi X")
   - `[Hypothèse]` — supposition explicite non vérifiée
   - `[Découvert]` — gotcha, post-mortem, observation

3. **N'invente RIEN.** Si une info n'est pas dans le fichier, ne la déduis
   pas. Pas de rationale deviné.

4. **Préserve VERBATIM** les valeurs (IDs, paths, URLs, snippets cités,
   noms de champs, valeurs de config, versions, line numbers).

5. **Single source de filtrage.** Tu skippes UNIQUEMENT:
   - Headers de sections sans contenu propre
   - Phrases de transition pures ("Voyons maintenant…")
   - Répétitions littérales d'une info déjà capturée plus haut
   - Texte boilerplate (license headers, install standards, ToC)

6. **Ne reproduis pas le markdown brut.** Reformule en caveman grammar,
   sans paraphraser les valeurs (cf règle 4).

## Format de sortie

Ta réponse DOIT commencer directement par `ctx:`. Pas de préface, pas de
header de section ajouté, pas de répétition des instructions, pas de
sentinel SKIP dans le corps. Omets toute section sans contenu réel.
N'inclus PAS le path ou le type — le wrapper les ajoute déjà en en-tête.

```
ctx: <une ligne — ce que le fichier traite>

structure:
- <section ou item significatif> → <ce qu'il établit ou décide>
- ...

facts:
- [Établi] <contenu> — <contexte si nécessaire>
- [Décidé] <contenu> → <rationale si donné>

abandoned-paths:
- <approche essayée> → <raison du rejet>

gotchas:
- [Découvert] <comportement non-évident ou contrainte cachée>

hypotheses:
- [Hypothèse] <supposition explicite du document>

entities:
- <nom> — <rôle dans le contexte>

action-items:
- <TODO explicite ou follow-up dans le texte>
```

Règles de structure:
- 1 atom par ligne. Pas de prose paragraphes.
- IDs/paths/numbers/versions verbatim.
- Pour un changelog: 1 ligne par changement architectural retenu.
- Pour une doc/spec: 1 ligne par section utile (ignore "Introduction").
- Pour une réflexion: 1 ligne par hypothèse ou pivot de pensée.
- Pour un plan: 1 ligne par étape ou contrainte.
- 3-15 items typiquement. Pas de quota.

## Mode silence

Si le fichier ne contient rien d'utile à long terme (TODO vide, contenu
trivial, doc générique sans décisions ni contraintes, fichier auto-généré),
réponds exactement: SKIP

## Langue

Écris dans la langue dominante du fichier, en grammaire compressée
(caveman). Termes techniques en VO.

Fichier: {rel}
Type: {file_type}
Dernière modification: {mtime_iso}

## Contenu du fichier

{content}"""

    captured: list[str] = []

    last_exc: Exception | None = None
    response = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            delay = RETRY_DELAYS[min(attempt - 2, len(RETRY_DELAYS) - 1)]
            logging.info("Retry attempt %d/%d after %ds", attempt, MAX_ATTEMPTS, delay)
            await asyncio.sleep(delay)

        try:
            response, stderr_lines = await run_codex(
                prompt,
                cwd=VAULT_ROOT,
                sandbox="read-only",
                timeout=ATTEMPT_TIMEOUT,
            )
            captured.extend(stderr_lines)
            return response, captured
        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"bundled CLI hung for >{ATTEMPT_TIMEOUT}s")
            logging.warning("Attempt %d/%d timed out", attempt, MAX_ATTEMPTS)
            continue
        except TimeoutError as e:
            last_exc = e
            logging.warning("Attempt %d/%d timed out", attempt, MAX_ATTEMPTS)
            continue
        except Exception as e:
            last_exc = e
            this_stderr = "\n".join(captured)
            logging.warning("Attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, e)
            if not _is_transient_error(e, this_stderr):
                logging.info("Non-transient error — aborting retries")
                break

    return f"SCAN_ERROR: {type(last_exc).__name__}: {last_exc}", captured


def build_daily_section(rel: str, mtime: float, file_type: str, body: str) -> str:
    """Compose the daily-log section body (without the heading).

    Header is dense single-line `key: value | key: value` — agent-readable,
    machine-parsable, matches the caveman grammar of the body.
    """
    mtime_local = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone()
    return (
        f"src: `{rel}` | type: {file_type} | mod: {mtime_local.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{body.strip()}"
    )


async def process_one(path: Path, rel: str, mtime: float, file_type: str, dry_run: bool) -> tuple[str, str | None]:
    """Process a single file. Returns (status, error-message-or-None).
    status is one of: 'ok', 'skip', 'error'."""
    if dry_run:
        return "ok", None

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return "error", f"read failed: {e}"

    if not content.strip():
        return "skip", "empty file"

    mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone().isoformat(timespec="minutes")
    response, _stderr = await summarise_file(rel, mtime_iso, content, file_type)

    if response.strip() == "SKIP":
        return "skip", "LLM said SKIP"
    if response.startswith("SCAN_ERROR"):
        return "error", response

    section_body = build_daily_section(rel, mtime, file_type, response)
    append_to_daily_log(section_body, section=f"MD Scan ({file_type}): {rel}")
    return "ok", None


MENU_TEXT = """
=========================================================
 scan_md — ingest the current repo's Markdown files
=========================================================
  1) Incremental scan (only files changed since last scan)
  2) Full rescan from scratch (--init)
  3) Files modified in last N days (--days N)
  4) Files modified since a specific date (--since YYYY-MM-DD)
  5) Force rescan all files (--all, ignore hash dedup)
  6) Preview only — no LLM calls (--dry-run)
  q) Quit
---------------------------------------------------------
""".strip("\n")


def run_menu() -> argparse.Namespace:
    """Interactive menu — returns a populated argparse.Namespace.

    Avoids stale terminal state by always producing a fresh Namespace per
    invocation. Defaults match the CLI: every flag is False/None unless
    the user picks an option that sets it.
    """
    ns = argparse.Namespace(
        init=False, all=False, days=None, since=None, path=None, cwd=None, dry_run=False,
    )
    print(MENU_TEXT)
    while True:
        try:
            choice = input("Pick an option (1-6, q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice in ("q", "quit", "exit"):
            print("Cancelled.")
            sys.exit(0)
        if choice == "1":
            return ns
        if choice == "2":
            ns.init = True
            return ns
        if choice == "3":
            raw = input("How many days back? ").strip()
            try:
                ns.days = int(raw)
                if ns.days < 0:
                    raise ValueError
            except ValueError:
                print("error: needs a non-negative integer")
                continue
            return ns
        if choice == "4":
            raw = input("Since when? (YYYY-MM-DD): ").strip()
            try:
                datetime.strptime(raw, "%Y-%m-%d")
            except ValueError:
                print("error: format must be YYYY-MM-DD")
                continue
            ns.since = raw
            return ns
        if choice == "5":
            ns.all = True
            return ns
        if choice == "6":
            ns.dry_run = True
            return ns
        print(f"unknown option: {choice!r}")


def main() -> int:
    if PROJECT_DIR is None:
        print("error: no project detected. Run from inside a git repo with a vault folder.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="Scan Markdown files in the current git repo into the daily log")
    parser.add_argument("-m", "--menu", action="store_true", help="Interactive menu (recommended for one-off manual runs)")
    parser.add_argument("--init", action="store_true", help="Reset scanned_md state and rescan everything")
    parser.add_argument("--all", action="store_true", help="Scan all files (ignore hash dedup, but still respect --days/--since)")
    parser.add_argument("--days", type=int, help="Only scan files modified in the last N days (by mtime)")
    parser.add_argument("--since", type=str, help="Only scan files modified on/after YYYY-MM-DD (by mtime)")
    parser.add_argument("--path", type=str, help="Override repo root (default: git rev-parse --show-toplevel)")
    parser.add_argument(
        "--cwd",
        type=str,
        help="Explicit source checkout to scan (takes precedence over hook and shell cwd)",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files that would be scanned, no LLM calls")
    args = parser.parse_args()

    # Explicit --menu only. Bare invocation keeps the prior behaviour
    # (incremental scan) so cron jobs and CI runs aren't broken.
    if args.menu:
        path_keep = args.path  # path is environmental, preserved across menu
        cwd_keep = args.cwd
        args = run_menu()
        args.path = path_keep
        args.cwd = cwd_keep

    # File-based logging — same channel as flush.py for consistency.
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(STATE_DIR / "scan-md.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = find_repo_root(args.path, args.cwd)
    if root is None:
        print("error: could not resolve a git repo root. Pass --path explicitly.", file=sys.stderr)
        return 1
    if is_vault_internal(root):
        print(
            f"error: refusing to scan inside the vault ({root}). "
            "Run from your code repo, or pass --path /elsewhere.",
            file=sys.stderr,
        )
        return 1

    print(f"Repo root: {root}")
    print(f"Project KB: {PROJECT_DIR}")

    files = discover_md_files(root)
    if not files:
        print("No .md files found.")
        return 0
    print(f"Discovered {len(files)} .md files")

    state = load_state()
    if args.init:
        prev_count = len(state.get("scanned_md", {}))
        state["scanned_md"] = {}
        if not args.dry_run:
            save_state(state)
            print(f"--init: cleared scanned_md state ({prev_count} entries)")
        else:
            print(f"--init [dry-run]: would clear {prev_count} entries from scanned_md state")

    since_ts: float | None = None
    if args.days is not None:
        if args.days < 0:
            print("error: --days must be non-negative", file=sys.stderr)
            return 2
        since_ts = (datetime.now().astimezone() - timedelta(days=args.days)).timestamp()
    elif args.since:
        since_ts = parse_since(args.since)

    to_scan = filter_files(
        files, root, state,
        since_ts=since_ts,
        ignore_hash=args.all or args.init,
    )

    if not to_scan:
        print("Nothing to scan — all .md files are up to date.")
        return 0

    # Annotate each candidate with its detected type so user sees which
    # extraction policy will apply before the LLM is invoked.
    typed = [(p, rel, mtime, detect_md_type(rel)) for (p, rel, mtime) in to_scan]

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to scan ({len(typed)}):")
    type_counts: dict[str, int] = {}
    for _, rel, mtime, ftype in typed:
        mt = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone().strftime("%Y-%m-%d")
        print(f"  - [{ftype:>10s}]  {rel}  (modified {mt})")
        type_counts[ftype] = type_counts.get(ftype, 0) + 1
    type_summary = ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items()))
    print(f"  Types: {type_summary}")

    if args.dry_run:
        return 0

    if not acquire_flush_lock():
        print("error: another flush.py / scan is holding the vault lock. Try again later.", file=sys.stderr)
        return 3

    try:
        ok = skipped = errored = 0
        for i, (path, rel, mtime, ftype) in enumerate(typed, 1):
            print(f"\n[{i}/{len(typed)}] Scanning [{ftype}] {rel}...")
            try:
                status, msg = asyncio.run(process_one(path, rel, mtime, ftype, dry_run=False))
            except Exception as e:
                status, msg = "error", str(e)

            if status == "ok":
                ok += 1
                state.setdefault("scanned_md", {})[rel] = {
                    "hash": file_hash(path),
                    "mtime": mtime,
                    "type": ftype,
                    "scanned_at": now_iso(),
                }
                save_state(state)
                print(f"  appended to daily log")
            elif status == "skip":
                skipped += 1
                state.setdefault("scanned_md", {})[rel] = {
                    "hash": file_hash(path),
                    "mtime": mtime,
                    "type": ftype,
                    "scanned_at": now_iso(),
                    "skipped": True,
                }
                save_state(state)
                print(f"  skipped: {msg}")
            else:
                errored += 1
                logging.error("scan failed for %s: %s", rel, msg)
                print(f"  ERROR: {msg}")
    finally:
        release_flush_lock()

    print(
        f"\nScan complete. ok={ok} skipped={skipped} errors={errored}\n"
        f"Daily log: {DAILY_DIR / (datetime.now().astimezone().strftime('%Y-%m-%d') + '.md')}\n"
        f"Run: uv run python scripts/compile.py  to fold this into knowledge/"
    )
    return 0 if errored == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
