# ACE Runtime Documentation Changelog

## 2026-09-08 - Daily audit completion

- Daily reports now include a dedicated `Auto-amélioration` section with
  detected signals, suggestions, recorded actions and verification state;
  suggestions remain approval-gated and are never applied automatically.
- The explicit daily source-window path now resolves selected snapshot deltas
  in one bounded database-wrapper call, removing the per-conversation secret
  lookup delay without changing the evidence contract.
- A live `.agents` run for `2026-09-07` returned
  `analyzed=1, compiled=0, failed=0, pending=0`.
- The persisted audit is `status=ok`, covers 9 sessions, and contains no
  errors. The targeted ACE suite passes 108 tests.
- No schema migration, historical replay, daily-log rewrite, or automatic
  skill/rule change was performed.

## 2026-09-07 - ACE contract and measured runtime state

### Revision examined

- Runtime HEAD: `83aed624af373b7462a961bf1f784a364e5700c3`.
- ACE scripts and the migration were already present in the checkout. This
  documentation pass updated `README.md`, `CLAUDE.md`, `AGENTS.md`, and `docs/`.
- Root changelog entries were updated without rewriting older entries.

### Scope examined

- `README.md`, `CLAUDE.md`, and `AGENTS.md`: active architecture, commands, and
  article editorial schema.
- `scripts/ace_pipeline.py`: pipeline order, acknowledgement, daily run, and tick.
- `scripts/ace_transcripts.py`: secret, media, and internal-reasoning filtering.
- `scripts/ace_database.py` and `migrations/001_ace.sql`: `amastuces` transport,
  `ace` schema, and SQL surface.
- `scripts/ace_learning.py`: evidence windows, recommendations, refusals,
  corrections, evaluations, and reports.
- `scripts/ace_schedule.py`: native service, cadence, and Europe/Paris times.
- Root `CHANGELOG.md`: stdin transport, CLI await/on-notice handling, migration
  and state-preservation history.

### Documentation changes

- Obsolete sections about hooks, `flush`, compilation after 18:00, and CMC
  routes were replaced with the `bin/ace` pipeline.
- The Luna model contract, Astra orchestration, project opt-in, and local master
  vault are documented.
- The article formats and merge policy in `AGENTS.md` remain unchanged.
- The post-acknowledgement flow is documented as separate memory and improvement
  branches without claiming independent cadences.
- The collector selects the project before reading and filtering the bounded
  source; downstream processing accepts only strict database acknowledgements.
- Recent guards are recorded: no recursive LLM, stdin SQL above 2 MB, strict ACK,
  metadata and token usage retention, secret/base64/reasoning guards, hash/daily
  retries, verified message references, no empty model calls, and timestamped
  project-scoped coverage.
- The link to the canonical Agent Central source remains in `docs/INDEX.md` and
  `docs/ace.md`.

### Checks

- `ace status` was run without a model.
- `ace schedule plan --json` was run read-only.
- `ace schedule status --json` returns `installed: false`.
- Read-only Supabase query on 2026-09-07: four enabled and initialized projects,
  10 sessions, 10 revisions, 1,905 messages, 14 ACE tables, and RLS on all 14.
- The same DB measure counted 6 observations, 12 recommendations, including 3
  verified real frustration recommendations, and 9 snapshots pending processing.
- Latest validation: 220 runtime tests and 5 Agent Central tests passed.
- The additive migration was applied after a private 21,802,182-byte backup;
  45 states were migrated and original copies were retained.
- A 2026-08-31 daily produced five articles; resumed `ace compile` returned `OK`,
  and a new-process `ace query` returned a response with a verified source.
- The v3 publication and read-only copy were checked: 6 articles, 8 files, 6
  valid index links, and a deterministic index.
- The corrected learning report covers 10 sessions and preserves 18 attempts:
  4 are `OK`, 6 model errors remain to retry, and `A` was validated by replaying
  real JSON without a new model call. The four `OK` reports are distinct from
  the single current ACK and the 9 pending-processing snapshots.
- Native lot-1 conversation/context analysis preserves evidence; later passages
  remain under retry and collection is unchanged. Final compile/analysis
  independence and retry fairness remain under completion.
- The normalizer remains under targeted correction; collection is unchanged.
- The corrected real-report rerun remained to be validated.
- No scheduler is installed and no E2E flow is claimed.

### Open risks

- The corrected real-report rerun and the full E2E flow still need validation.
- `launchd` installation requires an explicit decision and a separate check.
