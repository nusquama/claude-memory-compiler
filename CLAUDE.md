# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Claude Memory Compiler (CMC) — a personal knowledge base that automatically captures Claude Code conversations and compiles them into structured wiki articles. Raw material flows: conversation transcript → daily log → compiled concepts/connections → injected back into future sessions.

## Quick Access — Slash Commands in Claude Code

The primary entry points are slash commands invoked from inside a Claude Code session. They auto-resolve the current project (`git rev-parse --show-toplevel`), export `CLAUDE_PROJECT_DIR`, and dispatch to the right underlying script.

| Slash command | Purpose |
|---|---|
| `/cmc` | Top-level menu — picks among all CMC operations |
| `/cmc-init` | Initialise the KB for the current project (one-time) |
| `/cmc-scan` | Backfill past Claude Code transcripts into daily logs |
| `/cmc-scan-md` | Scan the `.md` files of the current repo (READMEs, docs, ADRs) |
| `/cmc-compile` | Compile daily logs → knowledge articles |
| `/cmc-query` | Ask a question against the KB |

For `/cmc-scan-md`, natural-language invocations work: "scan md des 7 derniers jours", "rescanne les md depuis le début", "scan les docs", "scan markdown depuis 2026-04-01". The skill parses the date filter and the rescan mode from the user's wording.

## Key Commands (terminal, manual invocation)

```bash
# Compile daily logs into knowledge articles (only changed files)
uv run python scripts/compile.py

# Force recompile everything
uv run python scripts/compile.py --all

# Compile a specific daily log
uv run python scripts/compile.py --file daily/2026-04-01.md

# Query the knowledge base
uv run python scripts/query.py "your question here"

# Query and file the answer back as a qa/ article
uv run python scripts/query.py "your question" --file-back

# Health checks (broken links, orphans, stale articles, contradictions)
uv run python scripts/lint.py

# Skip LLM-powered checks (free, structural only)
uv run python scripts/lint.py --structural-only

# Backfill historical transcripts
uv run python scripts/backfill.py

# Scan the .md files of the current git repo into today's daily log
uv run python scripts/scan_md.py --menu             # interactive menu (recommended)
uv run python scripts/scan_md.py                    # changed since last scan
uv run python scripts/scan_md.py --init             # full rescan from scratch
uv run python scripts/scan_md.py --days 7           # files modified in last 7 days
uv run python scripts/scan_md.py --since 2026-04-01 # files modified since date
uv run python scripts/scan_md.py --all              # ignore hash dedup
uv run python scripts/scan_md.py --dry-run          # preview without LLM calls

# Backfill Codex Desktop/CLI conversations from ~/.codex/sessions
uv run python scripts/backfill_codex.py --dry-run --fallback-project Conversations
uv run python scripts/backfill_codex.py --fallback-project Conversations --compile
```

> `scan_md.py` requires being run from the user's project repo (or with
> `--path /repo`). When invoked through `uv run --directory _config`, set
> `CLAUDE_PROJECT_DIR=$(pwd)` so the script knows the original cwd
> (`uv run --directory` changes cwd to `_config` before exec).

## Architecture

### Data Flow

```
Session start  →  hooks/session-start.py  →  reads knowledge/index.md + recent daily log
                                           →  returns additionalContext JSON to Claude Code

Conversation ends  →  hooks/session-end.py   →  reads JSONL transcript
                   →  hooks/pre-compact.py   →  safety net before auto-compaction
                                              →  spawns flush.py as detached background process

[Async] flush.py  →  Claude Haiku extracts what's worth saving
                  →  appends to daily/YYYY-MM-DD.md
                  →  triggers compile.py automatically if past 6 PM

[Async] compile.py  →  Claude Sonnet reads daily log
                    →  writes/updates concept and connection articles
                    →  updates knowledge/index.md and knowledge/log.md

[Manual] git repo .md files  →  scan_md.py (Haiku summarises each file)
                              →  appends `### MD Scan: <path>` sections to today's daily log
                              →  compile.py picks them up next run

[Automatic] Codex turn-ended notify  →  /Users/franck/.agents/bin/codex-turn-ended
                                    →  /Users/franck/.agents/bin/codex-cmc-backfill
                                    →  backfill_codex.py --compile after idle delay
```

### Key Design Decisions

**No vector database.** At personal scale (50–500 articles), the LLM reads the full index and selects relevant articles. This outperforms cosine similarity because the model understands intent, not just word overlap.

**Opt-in per project.** A project gets knowledge capture only when its vault folder already exists (created via `cmc-init` skill). Projects without a folder are silently skipped. Detection: `git rev-parse --show-toplevel` → maps to `Claude_code/<project>/`.

**Hooks never block.** `flush.py` is spawned as a fully detached subprocess (Unix: `start_new_session=True`, Windows: `CREATE_NEW_PROCESS_GROUP`). Context is passed via temp file, cleaned up after. No pipe deadlocks.

**Recursion prevention.** `CLAUDE_INVOKED_BY=memory_flush` is set before any imports in hooks. Any hook that detects this env var exits immediately — prevents infinite loops when flush.py calls the Agent SDK, which would fire hooks again.

**Incremental compilation.** SHA-256 hashes of daily logs are stored in `scripts/state.json`. Only files with changed hashes are recompiled.

**Codex capture.** Codex does not expose Claude-style hooks, so capture uses
Codex `notify` in `/Users/franck/.codex/config.toml`. Computer Use may keep
its own notify command in front and pass the CMC wrapper through
`--previous-notify`; this is expected and was verified with a dummy payload.
The previous notify points to `/Users/franck/.agents/bin/codex-turn-ended`,
which launches a detached delayed backfill worker. The worker skips active
rollouts, deduplicates by session hash, skips contexts above 120k characters
by default, marks failed sessions, and compiles affected CMC projects when
entries are written. Large historical sessions should be backfilled manually
with explicit limits. Rollback config backup:
`/Users/franck/.codex/backups/config.toml.pre-cmc-codex-notify-20260601`.

**Tiered models for cost control.**
- Flush (lightweight extraction): `claude-haiku-4-5` (~$0.02–0.05 per session)
- Compile/Query (complex reasoning, large context): `claude-sonnet-4-6` (~$0.45–0.65 per compile)
- Override via env: `CMC_FLUSH_MODEL`, `CMC_COMPILE_MODEL`, `CMC_QUERY_MODEL`

**Deduplication.** `scripts/last-flush.json` tracks session IDs + timestamps; duplicate flushes within 60 seconds are skipped.

### Knowledge Storage Layout

```
daily/              Append-only source logs (immutable, one file per day)
knowledge/
  index.md          Master catalog (table of all articles + one-line summaries)
  log.md            Build log (timestamped compilations, costs)
  concepts/         Atomic knowledge articles
  connections/      Cross-cutting insights linking 2+ concepts
  qa/               Filed query answers
scripts/
  state.json        SHA hashes + compile timestamps + cost tracking (gitignored)
  last-flush.json   Session deduplication state (gitignored)
```

### Hook Configuration

Hooks live in `.claude/settings.json` and fire automatically in Claude Code. Three hooks:
- `SessionStart` → `hooks/session-start.py` (timeout 15s)
- `PreCompact` → `hooks/pre-compact.py` (timeout 10s)
- `PostToolUse/Stop` → `hooks/session-end.py` (timeout 10s)

### Article Format

Concept and connection articles use YAML frontmatter:
```yaml
---
title: "Article Title"
type: concept           # or connection, qa
tags: [tag1, tag2]
related: [[other-article]]
updated: 2026-04-29
---
```

Wikilinks (`[[article-name]]`) are first-class — `utils.py` parses them for link validation in `lint.py`.

## Dependencies

- Python 3.12+
- `uv` (package manager)
- `claude-agent-sdk>=0.1.29` — LLM calls in compile/flush/query
- `python-dotenv`, `tzdata`

`ANTHROPIC_API_KEY` must be set in environment or `.env`.
