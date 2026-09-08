# AGENTS.md - Agent Continuous Evolution Schema

This runtime is branded Agent Continuous Evolution (ACE).
The canonical entry point is `/Users/franck/.agents/bin/ace`.
Old CMC paths and states remain historical data only. They are not active aliases.

> Adapted from [Andrej Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture.
> Instead of ingesting external articles, this system compiles knowledge from your own AI conversations.

## The Compiler Analogy

```
daily/          = source code    (filtered conversation extracts - the input)
LLM             = compiler       (extracts and organizes knowledge)
knowledge/      = executable     (structured, queryable knowledge base)
lint            = test suite     (health checks for consistency)
queries         = runtime        (using the knowledge)
```

You don't manually organize your knowledge. You have conversations, and the LLM handles the synthesis, cross-referencing, and maintenance.

---

## Architecture

### Layer 1: `daily/` - Approved Conversation Extracts

Daily logs contain filtered extracts from database-acquitted snapshots.
They are durable project sources, not a raw transcript archive.

```
daily/
├── 2026-04-01.md
├── 2026-04-02.md
├── ...
```

Each file follows this format. Sections without real content are omitted —
no padding, no quota.

```markdown
# Daily Log: YYYY-MM-DD

## Sessions

### Session (HH:MM)

**Contexte:** What the user was working on (one line).

**Décisions prises:**
- `[Décidé]` Chose library X over Y — rationale extracted from the conversation.

**Chemins abandonnés:**
- Tried approach Z → rejected because...

**Découvertes / gotchas:**
- `[Découvert]` The gotcha with W is that...

**Entités mentionnées:**
- LibraryX — auth layer for the API.
- ServiceY — third-party SaaS used for Z.

**Action items:**
- [ ] Follow up on X.
```

**Provenance markers** — every claim is tagged with how strongly it is
established:

- `[Établi]` — verified in code, docs, or by execution
- `[Décidé]` — choice made in this session
- `[Hypothèse]` — unverified guess (must not be promoted to fact downstream)
- `[Découvert]` — gotcha/behaviour observed during the session

`[Hypothèse]` markers preserve epistemic uncertainty across compilation. The
compiler must not silently turn an `[Hypothèse]` into a stated fact in a
concept article — it stays flagged until a later session confirms (then
upgraded to `[Établi]`) or contradicts (flagged as a contradiction).

### Layer 2: `knowledge/` - Compiled Knowledge (Local Vault Master)

The local vault is the master for compiled knowledge.
The LLM writes articles through the documented compiler contract.
The database may keep a read-only compiled snapshot.

```
knowledge/
├── index.md              # Master catalog - every article with one-line summary
├── log.md                # Append-only chronological build log
├── concepts/             # Atomic knowledge articles
├── connections/          # Cross-cutting insights linking 2+ concepts
└── qa/                   # Filed query answers (compounding knowledge)
```

### Layer 3: This File (AGENTS.md)

The schema that tells the LLM how to compile and maintain the knowledge base. This is the "compiler specification."

---

## Structural Files

### `knowledge/index.md` - Master Catalog

An OKF §6 bullet listing of every knowledge article. This is the primary retrieval mechanism - the LLM reads this FIRST when answering any query, then selects relevant articles to read in full.

Format:

OKF v0.1 §6: sections + bullets, NO table. `okf_version` frontmatter allowed only
here (the bundle-root index). Description comes from each article's `description:`;
the `_(MAJ …)_` date is its `updated:`. **Sort each group by `updated` DESCENDING —
most recently updated article first.**

```markdown
---
okf_version: "0.1"
---

# Knowledge Base Index

# Concepts

* [Supabase Auth](concepts/supabase-auth.md) - Row-level security patterns and JWT gotchas _(MAJ 2026-04-03)_
* [Next.js Project Structure](concepts/nextjs-project-structure.md) - App-router layout conventions _(MAJ 2026-04-01)_

# Connections

* [Auth and Webhooks](connections/auth-and-webhooks.md) - Token verification patterns shared across Supabase auth and Stripe webhooks _(MAJ 2026-04-04)_
```

### `knowledge/log.md` - Build Log

Append-only chronological record of every compile, query, and lint operation.

Format:

OKF v0.1 §7: `## YYYY-MM-DD` date headings, `* ` bullets, markdown links.

```markdown
# Directory Update Log

## 2026-04-02
* **Query** ("How do I handle auth redirects?") — consulted [supabase-auth](/concepts/supabase-auth.md), [nextjs-middleware](/concepts/nextjs-middleware.md); filed to [auth-redirect-handling](/qa/auth-redirect-handling.md)

## 2026-04-01
* **Compile** (daily/2026-04-01.md) — created [nextjs-project-structure](/concepts/nextjs-project-structure.md), [tailwind-setup](/concepts/tailwind-setup.md)
```

---

## Article Formats

### Concept Articles (`knowledge/concepts/`)

One article per atomic piece of knowledge — facts, patterns, decisions,
gotchas extracted from your conversations.

**No length minimums. No bullet quotas. No forced wikilinks.** Every
section except `Sources` and frontmatter is optional. Add a section only
when there is real content for it. A 100-word article with one solid Key
Point and no Related Concepts is fine; a sparse article beats a padded
one.

```markdown
---
type: Gotchas                          # REQUIRED (OKF v0.1 §4.1). Derive from the topic:
                                       # Gotchas | API Reference | Data Model | Pattern |
                                       # Config Reference | Playbook | Architecture | Brand |
                                       # Marketing Reference | Business Reference | Concept
title: "Concept Name"
description: "One-line summary — feeds the index.md entry."   # OKF recommended
aliases: [alternate-name, abbreviation]
tags: [domain, topic]
sources:
  - "daily/2026-04-01.md"
  - "daily/2026-04-03.md"
created: 2026-04-01
updated: 2026-04-03
---

# Concept Name

[Core explanation — as short as the topic warrants. No meta-description like
"X is a thing that does Y"; explain the thing.]

## Key Points  (optional — only if there are distinct takeaways beyond Details)

- [Self-contained bullets. Use provenance prefixes when relevant: "Décidé:",
  "Hypothèse non vérifiée:", "Comportement observé:".]

## Details  (optional)

[Deeper explanation only if needed.]

## Contradictions  (only if a daily log contradicts an earlier claim)

- 2026-04-15: daily/2026-04-15.md indicates [non-X], contradicts [X]
  (original source: daily/2026-04-01.md). Resolution: pending.

## Related Concepts  (optional — only if real, non-trivial links exist)

- [related-concept](/concepts/related-concept.md) — How it actually connects (one line).

## Sources  (required)

- [2026-04-01](/daily/2026-04-01.md) — Initial discovery during project setup.
- [2026-04-03](/daily/2026-04-03.md) — Updated after debugging session; added gotcha Z.
```

### Connection Articles (`knowledge/connections/`)

Cross-cutting synthesis linking 2+ concepts. Created when a conversation reveals a non-obvious relationship.

```markdown
---
type: Connection                       # REQUIRED (OKF v0.1 §4.1)
title: "Connection: X and Y"
description: "One-line summary of the non-obvious relationship."
connects:
  - "concepts/concept-x"
  - "concepts/concept-y"
sources:
  - "daily/2026-04-04.md"
created: 2026-04-04
updated: 2026-04-04
---

# Connection: X and Y

## The Connection

[What links these concepts]

## Key Insight

[The non-obvious relationship discovered]

## Evidence

[Specific examples from conversations]

## Related Concepts

- [concept-x](/concepts/concept-x.md)
- [concept-y](/concepts/concept-y.md)
```

### Q&A Articles (`knowledge/qa/`)

Filed answers from queries. Every complex question answered by the system can be permanently stored, making future queries smarter.

```markdown
---
type: Q&A                              # REQUIRED (OKF v0.1 §4.1)
title: "Q: Original Question"
description: "One-line gist of the question."
question: "The exact question asked"
consulted:
  - "concepts/article-1"
  - "concepts/article-2"
filed: 2026-04-05
---

# Q: Original Question

## Answer

[The synthesized answer with markdown links [x](/concepts/x.md) to sources]

## Sources Consulted

- [article-1](/concepts/article-1.md) - Relevant because...
- [article-2](/concepts/article-2.md) - Provided context on...

## Follow-Up Questions

- What about edge case X?
- How does this change if Y?
```

---

## Core Operations

### 1. Compile (daily/ -> knowledge/)

When processing a daily log:

1. Read the daily log file.
2. Read `knowledge/index.md` to understand current state.
3. Read existing articles that may need updating.
4. For each piece of knowledge in the log, apply the merge policy below.
5. If the log reveals a non-obvious connection between 2+ existing concepts,
   create a `connections/` article.
6. Update `knowledge/index.md`.
7. Append to `knowledge/log.md`.

**Filtering policy — no quotas:**
- Extract as many concepts as there are distinct signals. Mono-thematic log
  → 1 article. Touffu log → as many articles as concepts. Never fragment
  artificially. Never aggregate unrelated concepts in one article.
- Don't invent. If the rationale isn't in the log, don't reconstruct it. In
  doubt, omit. An omission is recoverable next cycle; a hallucination
  self-reinforces.
- No padding. No minimum bullet count, paragraph count, or wikilink count.

**Merge policy — when an existing article already covers the topic:**

| Situation | Action |
|---|---|
| Compatible (new info, no conflict) | Integrate, add daily log to `sources:`, bump `updated:` |
| Contradictory (log says non-X, article says X) | Do NOT overwrite. Add a `## Contradictions` section flagging both claims with their sources. Resolution is left to the user. |
| Redundant (same claim, already present) | Don't touch the body. Add the daily log to `sources:` to trace confirmation. |

**Provenance preservation:**
The daily log's `[Établi]`/`[Décidé]`/`[Hypothèse]`/`[Découvert]` markers
must survive compilation. An `[Hypothèse]` becomes "Hypothèse non vérifiée:"
in the article — never a stated fact. A later confirmation upgrades it to
`[Établi]`; a later contradiction triggers the contradiction policy above.

**Link policy (OKF v0.1 §5):**
- Use bundle-relative markdown links `[label](/concepts/slug.md)` /
  `[label](/connections/slug.md)` — leading slash, `.md` suffix. NOT `[[wikilinks]]`.
- Add a link only when the target exists AND is genuinely relevant.
- An isolated article with zero outbound links is acceptable.
- No padding links to hit a count.

**Language:** match the dominant language of the daily log. No mixing
languages within a single article. Technical terms (lib names, errors,
commands) stay verbatim.

### 2. Query (Ask the Knowledge Base)

1. Read `knowledge/index.md` (the master catalog)
2. Based on the question, identify 3-10 relevant articles from the index
3. Read those articles in full
4. Synthesize an answer with markdown-link `[x](/concepts/x.md)` citations
5. If `--file-back` is specified: create a `knowledge/qa/` article and update index.md and log.md

**Why this works without RAG:** At personal knowledge base scale (50-500 articles), the LLM reading a structured index outperforms cosine similarity. The LLM understands what the question is really asking and selects pages accordingly. Embeddings find similar words; the LLM finds relevant concepts.

### 3. Lint (Health Checks)

Seven checks, run periodically:

1. **Broken links** - markdown links `[x](/concepts/x.md)` (or legacy `[[wikilinks]]`) pointing to non-existent articles
2. **Orphan pages** - Articles with zero inbound links from other articles
3. **Orphan sources** - Daily logs that haven't been compiled yet
4. **Stale articles** - Source daily log changed since article was last compiled
5. **Contradictions** - Conflicting claims across articles (requires LLM judgment)
6. **Missing backlinks** - A links to B but B doesn't link back to A
7. **Sparse articles** - Below 200 words (suggestion only — sparse can be correct under the no-padding policy; the lint flags them so the user can confirm)

Output: a markdown report with severity levels (error, warning, suggestion).

---

## Conventions

- **Links:** OKF v0.1 §5 markdown links `[label](/path/to/article.md)` (bundle-relative, leading slash, `.md`). Not `[[wikilinks]]`. Configure Obsidian → Files & Links → "Use [[Wikilinks]]" OFF + "New link format: Absolute path in vault" so it emits and resolves this form.
- **Writing style:** Encyclopedia-style, factual, third-person where appropriate
- **Dates:** ISO 8601 (YYYY-MM-DD for dates, full ISO for timestamps in log.md)
- **File naming:** lowercase, hyphens for spaces (e.g., `supabase-row-level-security.md`)
- **Frontmatter:** Every article must have YAML frontmatter with at minimum: **type** (OKF-required), title, description, sources, created, updated
- **Sources:** Always link back to the daily log(s) that contributed to an article

---

## Full Project Structure

```
llm-personal-kb/
|-- AGENTS.md                        # Article schema and merge policy
|-- README.md                        # Runtime overview
|-- CLAUDE.md                        # Runtime operating notes
|-- pyproject.toml                   # Python dependencies
|-- daily/                           # Filtered, database-acquitted extracts
|-- knowledge/                       # Local compiled knowledge master
|   |-- index.md                     #   Master catalog
|   |-- log.md                       #   Append-only build log
|   |-- concepts/                    #   Atomic knowledge articles
|   |-- connections/                 #   Cross-cutting insights
|   |-- qa/                          #   Filed query answers
|-- scripts/                         # ACE runtime and delegated operations
|   |-- ace_pipeline.py              #   Collection, sync, extraction, daily, tick
|   |-- ace_transcripts.py           #   Source adapters and filtering
|   |-- ace_database.py              #   Supabase stdin transport
|   |-- ace_learning.py              #   Evidence and improvement reports
|   |-- ace_schedule.py              #   Native scheduler plan and service
|   |-- compile.py                   #   Compile daily logs -> knowledge articles
|   |-- query.py                     #   Index-guided retrieval
|   |-- lint.py                      #   Knowledge health checks
|-- migrations/                      # Supabase schema and functions
|-- docs/                            # Runtime notes and local changelog
```

---

## Native Capture and Processing

`/Users/franck/.agents/bin/ace` is the only active entry point.
It runs the bounded pipeline for Claude, Codex and Hermes sources.
The native service calls `ace tick`; it does not depend on a Claude hook or a
Codex heartbeat.

`scripts/ace_transcripts.py` reads source files in read-only mode.
It discovers metadata and selects an explicitly opt-in project before parsing a
bounded source prefix.
It masks secrets, replaces encoded media with references and removes internal
reasoning blocks before it calculates the revision.

The filtered envelope enters a private SQLite outbox.
`scripts/ace_database.py` sends SQL through the Supabase wrapper on stdin.
The profile is `amastuces` and the schema is `ace`.
The processor reads only snapshots that the database has acquitted.

`scripts/ace_pipeline.py` then performs the following steps:

1. extract a bounded snapshot with the fixed Luna contract;
2. on the memory branch, write the result to the local `daily/YYYY-MM-DD.md`
   file and compile it into the local `knowledge/` vault;
3. on the improvement branch, analyze evidence from the same acquitted
   revision and write private observations, decisions, corrections and reports.

The two branches share the database acknowledgement. Their implementation does
not prove independent schedules or concurrent execution. The compile/analysis
independence and retry-fairness guards are delivered; the scheduler remains one
best-effort tick.

The local vault remains the master for compiled knowledge.
The database may keep a compiled version for read-only retrieval.

### Native schedule

`scripts/ace_schedule.py` plans `launchd` with `ace tick` every 1 800 seconds.
The tick owns persistent catch-up state and the single-processor lock.
The active plist is `/Users/franck/Library/LaunchAgents/com.agentcentral.ace.plist`
with mode `600`; the current launch reports `runs=1` and PID `54145`.
The lock was observed held, and a concurrent attempt was rejected by `fcntl`.
Daily compilation and analysis start at 07:00 Europe/Paris and target an 08:00
report for the previous day.
Sleep, shutdown and failed stages can delay the target.
The first native cycle is still running and has not returned an exit code; the
target remains best-effort when the Mac is awake.

`ace schedule plan --json` previews the service.
`ace schedule status --json` checks installation without changing it.
The service is installed and active at the date of this document.

There is no recursive LLM path. The strict acknowledgement marker gates
processing; source-hash and daily-state retries preserve failed work. Claims
retain verified message references, and coverage records retain timestamps and
project scope. Empty model invocations are rejected.

---

## Script Details

### compile.py - The Compiler

Runs one Codex CLI child through `scripts/codex_runner.py` with
`gpt-5.6-luna`, `medium` reasoning, and `workspace-write` access limited to
the local vault. User config and notify forwarding are disabled for the child.

- Builds a prompt with: AGENTS.md schema, current index, all existing articles, and the daily log
- Codex reads the daily log, decides what concepts to extract, and writes files directly
- The wrapper records the final response and marks the daily log compiled only after Codex exits successfully
- Incremental: tracks SHA-256 hashes of daily logs in `state.json`, skips unchanged files
- Usage: Codex account consumption varies with prompt size and model output.

The normal daily path calls this compiler through `ace daily`.
Do not infer a successful daily run from the presence of the script alone.

**CLI:**
```bash
uv run python scripts/compile.py              # compile new/changed only
uv run python scripts/compile.py --all        # force recompile everything
uv run python scripts/compile.py --file daily/2026-04-01.md
uv run python scripts/compile.py --dry-run
```

### query.py - Index-Guided Retrieval

Loads the entire knowledge base into context (index + all articles). No RAG.

At personal KB scale (50-500 articles), the LLM reading a structured index outperforms vector similarity. The LLM understands what you're really asking; cosine similarity just finds similar words.

**CLI:**
```bash
uv run python scripts/query.py "What auth patterns do I use?"
uv run python scripts/query.py "What's my error handling strategy?" --file-back
```

With `--file-back`, creates a Q&A article in `knowledge/qa/` and updates the index and log. This is the compounding loop - every question makes the KB smarter.

### lint.py - Health Checks

Seven checks:

| Check | Type | Catches |
|-------|------|---------|
| Broken links | Structural | markdown links (or legacy `[[wikilinks]]`) to non-existent articles |
| Orphan pages | Structural | Articles with zero inbound links |
| Orphan sources | Structural | Daily logs not yet compiled |
| Stale articles | Structural | Source logs changed since compilation |
| Missing backlinks | Structural | A links to B but B doesn't link back |
| Sparse articles | Structural | Under 200 words |
| Contradictions | LLM | Conflicting claims across articles |

**CLI:**
```bash
uv run python scripts/lint.py                    # all checks
uv run python scripts/lint.py --structural-only  # skip LLM check (free)
```

Reports saved to `reports/lint-YYYY-MM-DD.md`.

---

## State Tracking

ACE keeps private, atomic state for each pipeline stage under the configured
private directory:

- collection candidates and source revisions ;
- synchronization and database acknowledgements ;
- extraction status and daily file ;
- compilation and analysis dates ;
- scheduler catch-up and processor locks.

The local SQLite outbox uses the tuple `source`, project, session and revision
for idempotence. Failed and deferred records remain retryable.
State files and reports use private permissions.

At the 2026-09-07 measurement, four enabled and initialized projects were
registered (`.agents`, `_config`, `hermes-agents`, `jiang`), with 10 sessions,
10 revisions and 1,905 messages. The same measure counted 6 observations,
12 recommendations, including 3 verified real frustration recommendations, and
9 snapshots pending processing before the first native run. These are dated
counts, not exhaustive coverage guarantees; the live state may evolve.

An observed daily dated 2026-08-31 produced five articles; a resumed
`ace compile` returned `OK`, and a new-process `ace query` returned a response
with a verified source. The v3 publication and read-only copy were checked: 6
articles, 8 files, 6 valid index links, and a deterministic index.
The corrected learning report covers 10 sessions and preserves 18 attempts:
4 are `OK`, 6 model errors remain to retry, and `A` was validated by replaying
real JSON without a new model call. The four `OK` reports are distinct from the
single current ACK and the 9 pending-processing snapshots. Two outbox entries
are currently visible as `pending` while the first cycle runs.
Native lot-1 conversation/context analysis preserves evidence; later passages
remain under retry, the normalizer and history are corrected, and collection is
unchanged. Final compile/analysis independence and retry fairness are delivered.
The canonical quality report is readable at
`private/ace/reports/8d9ed0fc-8485-51ed-9c0f-10826b15acbb/analysis/latest-daily.md`
for 10 sources with 4 `OK` and 6 model errors; the operational daily report
represents only valid loaded audits. The first cycle remains in progress, so
these observations do not establish E2E completion.

---

## Dependencies

`pyproject.toml` (at project root):
- Codex CLI (`codex`) - fixed Luna execution engine
- `python-dotenv>=1.0.0` - Environment variable management
- `tzdata>=2024.1` - Timezone data
- Python 3.12+, managed by [uv](https://docs.astral.sh/uv/)

The Supabase wrapper resolves its profile credentials outside this runtime.
Do not add credentials to source files, prompts or logs.
SQL is transported over stdin, including payloads larger than 2 MB; it is not
placed in process arguments.

---

## Usage

| Operation | Accounting |
|-----------|------------|
| Compile / query / semantic lint | Codex usage; varies with prompt and output size |
| Structural lint | Local only; no model call |
| Collection, extraction and analysis | Codex usage when the bounded Luna stage runs |
| Native scheduler tick | Local orchestration; no model for scheduling |

---

## Customization

### Additional Article Types

Add directories like `people/`, `projects/`, `tools/` to `knowledge/`. Define the article format in this file (AGENTS.md) and update `utils.py`'s `list_wiki_articles()` to include them.

### Obsidian Integration

The knowledge base is an OKF v0.1 bundle: pure markdown with `type` frontmatter and bundle-relative markdown links `[x](/concepts/x.md)`. Point an Obsidian vault at `knowledge/` for graph view, backlinks, and search — set Files & Links → "Use [[Wikilinks]]" OFF + "New link format: Absolute path in vault" so Obsidian emits and resolves the OKF link form. Any OKF-aware agent can also mount and traverse it directly.

### Scaling Beyond Index-Guided Retrieval

At ~2,000+ articles / ~2M+ tokens, the index becomes too large for the context window. At that point, add hybrid RAG (keyword + semantic search) as a retrieval layer before the LLM. See Karpathy's recommendation of `qmd` by Tobi Lutke for search at scale.
