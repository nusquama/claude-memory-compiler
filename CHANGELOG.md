# Changelog

All notable changes to this project are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed — 2026-09-08 (ACE morning report and Claude payload unblock, uncommitted)

- Memory first (`DEFAULT_EXTRACTION_MODE = "local"`): `process()` now runs
  `process_local()` before the database path. The local path reads the
  sanitized envelopes from `outbox.sqlite3` (`Outbox.snapshots`, read-only, no
  claim), applies the same session cursor, chunking, extractor and daily
  writer, and records each snapshot with `path="local"`. The database path
  only closes the remote stage for locally extracted revisions
  (`_reconcile_local_extraction`) and never replays or retries a snapshot the
  local path owns; it still extracts snapshots that exist only remotely.
  `ACE_EXTRACTION_MODE=database` restores the previous DB-first contract.
- `ace collect --extract` writes the daily log right after capture; the Claude
  session hooks pass it (`hooks/ace_capture.py`), restoring the pre-ACE
  session-end flush without any database or Bitwarden access.
- `_PLAIN_SECRET_RE` ignores the `<REDACTED>` marker as a value so marker
  contexts stay accepted by `normalize_envelope`.
- `redact_sensitive_text` ends with `_database_clean_pass`, an exact mirror of
  the `ace.jsonb_is_clean` string grammar (assignment, CLI flag, provider
  token, marker stripping) plus NUL removal, with a separator-dropping
  fallback when the mirrored check still fails. A message accepted locally is
  therefore never rejected by the database ("ACE message is not sanitized",
  "unsupported Unicode escape sequence"). Eight historical rows were
  re-sanitized in place and ingested.
- `ace collect --since YYYY-MM-DD`: explicit date bound for a historical
  replay, replacing both the activation cutoff and `--days`; implies
  `--all-history` for that window. Restores the `backfill_codex.py --since`
  capability.
- Analysis salvage: `_normalise_model_report` no longer discards a whole
  conversation report because one claim cites a proof outside the supplied
  evidence windows. Claims whose proof resolves are kept, the discarded count
  is exposed as `dropped_claims` plus a `limitations` line, and the report is
  rejected only when nothing verifiable remains. One bad citation used to
  cost a full model call and hide every good finding of that conversation.
- `ANALYSIS_BATCH_LIMIT` 1 → 6 conversations per daily analysis call, and the
  native tick chains up to `DAILY_CONTINUATIONS_PER_TICK = 4` pending
  continuations instead of one batch per 30-minute tick.
- Native tick budgets: `AUTOMATION_COLLECT_LIMIT = 40` for model-free
  discovery/parsing (285 sessions were `deferred` because each tick examined
  only 4 candidates) and `AUTOMATION_PROCESS_LIMIT = 4` extractions per project
  per tick now that a database call no longer costs a Bitwarden read.
- Tests: `test_end_to_end_collect_process_writes_daily_before_db_ack`,
  `test_database_mode_keeps_extraction_after_db_ack`,
  `test_process_local_works_without_any_store`; hook command test expects
  `--extract`.
- Add `scripts/ace_morning_report.py` and the `ace report [--day]` subcommand:
  one model-free Markdown report across every registered project, written to
  `private/ace/reports/morning/<day>.md` plus `latest.md`. Incidents are sorted
  by priority then risk; recurrences, successes, preferences, analysed
  conversations, token usage per stage and limits follow. The native tick
  writes it after the due daily work (`result["report"]`).
- Raise `AUTOMATION_MAX_PAYLOAD_BYTES` from 500 000 to 4 000 000 bytes. Every
  Claude transcript above 500 KB was deferred for 24 hours on each pass and
  never reached the database (17 rows in `retry`, all with
  `automatic payload deferred`). Deferred rows were reset to `pending` after a
  backup under `private/ace/backups/repair-20260908T160648Z/`.
- Observation: each database call through the Supabase wrapper costs 45 to
  60 seconds because the Bitwarden secret is resolved per process behind a
  12-second cross-process gate. Manual batches export the secret once via
  `agent_export_secret`; the native tick still pays the cost on every call.
- Align the local redactor with `ace.jsonb_is_clean`: `_PLAIN_SECRET_RE` in
  `scripts/utils.py` now accepts an optional quote around the credential label
  (`'token': value`). A 3.2 MB Codex snapshot was rejected by the database with
  "ACE message is not sanitized" because one tool-call argument contained
  Python source matching that shape locally unredacted. Its stale `retry` row
  stays in the outbox until an explicit cleanup; the next collection of that
  session enqueues a clean revision.
- Strengthen the analysis prompt in `build_snapshot_prompt`: incidents and
  recommendations must carry a stable `signature`, a `priority` (high/normal/
  low by user cost) and a `risk` (low/medium/high by change scope);
  `cause.summary` starts with one root-cause category among
  `[regle_absente] [regle_non_suivie] [information_manquante]
  [information_introuvable] [outil] [inconnue]`; recommendations name the
  target component and give the exact wording, "Avant/Après" when a text
  changes; repeated user preferences become `rule` or `memory`
  recommendations. Generic "reproduce the calls" advice is only allowed when no
  rule, skill or information can prevent the problem. No schema field added.
- Validation: `tests/test_ace_pipeline.py` and `tests/test_ace_pipeline_runtime.py`
  pass (62 tests); `tests/test_ace_morning_report.py` added (4 tests, pass);
  learning, contract, safety, evidence, ingestion and identity suites pass
  (75 tests).

### Fixed — 2026-09-08 (ACE audit repairs, uncommitted)

- Add a dedicated `Auto-amélioration` section to the daily report. It exposes
  detected signals, suggestion references, recorded action states and
  correction verification without applying suggestions automatically.
- Preserve the first messages of new incremental Codex sessions and retry failed/deferred reads without advancing their cursor. Baseline sessions predating activation without a history replay. Serialize collection state and hook callback claims; expire and bound the callback ledger.
- Rotate the durable outbox selection to prevent starvation, correct its configured payload/lot limits, and preserve strict database acknowledgements.
- Add migration `002_ace_stage_leases.sql`: atomic stage claims, owner/host checks, renewable leases, expiration, stale-worker rejection and source-date windows applied before the database limit. Apply only unapplied numbered migrations, without replaying old grants.
- Apply migration 002 to `amastuces` after Franck's explicit authorization. Preserve the pre-migration backup at `/Users/franck/.agents/private/ace/backups/20260907T221250928246Z.json` (33,515,982 bytes; SHA-256 `fbb4a17b466fee9711d4e5a8bc855dc9668421ecf360131f240eb1172cf19c73`). Versions 1 and 2 are present; no migration remains pending; RLS remains enabled on all 15 ACE tables.
- Verify the real database's atomic rollback, idempotence, latest revision selection, project isolation, role permissions, compiled reading and analysis retry/fairness. The transactional verification leaves zero fixture rows.
- Share the native `tick.lock` with manual process/daily/compile/scan entry points. Start the daily work after 07:00, resume after sleep, bound actual failure retries, and keep successful pending continuations eligible until all current batches finish. Keep explicit `--day` scoped to that local source day and protect the native activation boundary.
- Pass `scan-md` its explicit verified source root separately from the vault destination. Migrate legacy CMC state to the canonical ACE private state root.
- Use the canonical query-context helper, rank the whole candidate corpus before the context limit, and keep ordinary knowledge queries free of state writes.
- Compile in an isolated temporary workspace. Validate index/article/source/build-log relationships before committing, preserve concurrent edits, and keep the prior corpus after a failed stage. Validate knowledge bundles before publication/materialization.
- Repair the existing `.agents` knowledge links and the build log's inaccurate missing-article reference without rewriting daily sources. Preserve the three originals under `private/ace/backups/repair-20260907T212414Z/knowledge-links/`.
- Keep the Luna `gpt-5.6-luna` / `medium` contract. Collect measured Codex JSON usage and duration, including retries, and persist extraction/compile/analysis stage metrics. Missing usage remains unknown.
- Replace the twelve-section summary template with four focused sections; retain user corrections, actor/message references, exact tool results, and verification limits. A synthetic eight-message check produces 16 lines with the scope correction and all decisive values preserved.
- Reject weak terminal markers, negative/missing event acknowledgements, proofless recurrences and unknown session identities. Persist refusal semantics through the existing decision RPC and suppress refused proposals across sessions.
- Preserve nested model-report structure and strict evidence validation. Scope rejected attempts to their session/revision. Keep successes, incidents, correction recurrence, and independent accepted/refused/applied/verified/effective metrics tied to evidence; exclude unresolved claims from actions and proven KPIs. Bound recursive report sanitization and redact secret-bearing keys.
- Validation: 309 runtime tests passed, including three tests against an isolated PostgreSQL instance, plus four Agent Central entry-point tests. The temporary PostgreSQL server was stopped after verification.
- A real Luna run on an entirely synthetic Codex conversation completed collection, synchronization, extraction, compilation, analysis and sourced knowledge query: `daily={days:1,compiled:1,analyzed:1,failed:0,pending:0}`. This is separate from the production conversation check.
- Fix the manual daily source-window handoff: fetch actual snapshot messages without an incremental cutoff, unwrap storage envelopes before learning, preserve flat project identities, and retain a safe persistence error reason. The real replay remains scoped to the activation cutoff; final result is recorded below. No automatic skill/rule changes, no historical backfill and no Git publication.
- Batch the bounded `snapshot_delta` reads for an explicit daily source window into one Supabase wrapper call, avoiding one Bitwarden throttle per conversation while preserving the same evidence and fail-closed validation. The live `.agents` run for `2026-09-07` now returns `analyzed=1, compiled=0, failed=0, pending=0`; the persisted audit is `status=ok` with 9 sessions and no errors. Targeted ACE tests: 108 passed.
- Source rollback backup: `/Users/franck/.agents/private/ace/backups/repair-20260907T212414Z`. Operator edits predating this repair remain intact.

### Correction du 2026-09-07 : diagnostic des rapports ACE rejetés (non commité)

- Conservation du champ, de l’index du claim et du motif précis de rejet.
- Références et messages identifiés sans enregistrer le texte brut du modèle.
- Conservation des diagnostics initiaux et de la reprise unique en échec.
- Transmission des diagnostics à la reprise, sans assouplir la validation.
- Les 53 tests du contrat d’analyse et du runtime passent.
- Rejeu local du seul snapshot acquitté courant : un appel Luna medium, une conversation, statut `ok`, aucune erreur.
- Aucune règle de preuve assouplie : le rejet précédent ne s’est pas reproduit.
- Aucun historique relu, aucune écriture Supabase, aucun daily log modifié.
- Les compteurs natifs `ace daily` ne sont pas validés par ce rejeu isolé.

### Fixed — 2026-09-07 (ACE bounded automation, uncommitted)
- Restore the CMC-style incremental boundary for the native 30-minute pass:
  Supabase returns compact revision references, then only the per-session
  message delta is eligible for extraction.
- Ignore historical pending revisions whose source update predates the
  automation cutoff; explicit/manual processing and `--all-history` retain
  their requested history semantics.
- Bound automatic extraction to one revision per project per tick and run the
  daily report/compilation only once during the configured morning window
  (`ACE_DAILY_REPORT_TARGET`, default `08:00`).
- Add a Codex JSONL byte cursor: the first automatic observation records only
  the end position, later observations parse appended records instead of
  rebuilding a multi-megabyte transcript; oversized automatic outbox retries
  are deferred for 24 hours without deletion.
- Fix the automatic Supabase reference query so a one-item processing limit
  scans past historical pending rows before applying the incremental cutoff;
  this restores the current conversation to the daily-log path.
- Fetch the first bounded delta for a session created after the cutoff;
  existing sessions still baseline without replaying historical content.
- Validation: 231 tests pass; a live incremental tick scoped to `.agents`
  returned `process.candidates=1`, `process.processed=1`, and
  `sync.failed=0`, updating the project daily log. No history backfill was
  performed; the morning report window was not due.

### Changed — 2026-09-07 (ACE measured integration, uncommitted)
- Document the two post-acknowledgement branches: memory (`daily/` and
  `knowledge/`) and evidence-based improvement. The documentation does not
  claim independent cadences or E2E completion.
- Keep the canonical `/Users/franck/.agents/bin/ace` entry point, explicit
  project opt-in, local vault master, and read-only compiled database versions.
- Correct collection semantics: select the project before reading and filtering
  the bounded source; process only strict database ACK revisions.
- Record the stdin SQL transport (including payloads above 2 MB), CLI await and
  on-notice handling, strict ACK, metadata and token-usage retention, secret,
  base64 and reasoning guards, hash/daily retries, verified message references,
  no recursive LLM, no empty model calls, and timestamped project-scoped coverage.
- Preserve 45 migrated states with original copies, the recoverable CMC archive,
  and the additive migration backup at
  `/Users/franck/.agents/private/ace/backups/20260907T141813217439Z.json`
  (21,802,182 bytes).
- Measured on 2026-09-07: four opt-in projects, 10 sessions, 10 revisions,
  1,905 messages, 14 ACE tables, and RLS on all 14. An observed daily dated
  2026-08-31 produced five articles; resumed `ace compile` returned `OK`, and a
  new-process `ace query` returned a response with a verified source.
- The same DB measure counted 6 observations, 12 recommendations, including 3
  verified real frustration recommendations, and 9 snapshots pending processing.
- Latest validation passed 220 runtime tests and 5 Agent Central tests.
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
- The corrected real-report rerun remained in progress and required validation.
  The native service is not installed. No commit, push, or completed E2E claim.

### Fixed — 2026-09-06 (ACE collection reliability, lot 1, uncommitted)
- Abort extraction when any map-reduce chunk or consolidation fails. Return
  `FLUSH_ERROR` rather than validating the full source hash with partial output.
  Retry the complete source on a later eligible collection attempt.
- Share project routing across the collector, backfill and runtime config;
  retain the source project separately from the vault destination.
- Extend read-only capture coverage to Claude/CCS and Codex while retaining
  the historical Codex entry point. No scheduler, model, project initialization,
  transcript migration or historical re-extraction is included.
- Add synthetic regression tests for partial failure, complete retry,
  deduplication, project routing and multi-source coverage.
- Validation: 91 tests pass with `python3 -B -m pytest -q -p no:cacheprovider tests`;
  scoped diff checks pass. No live transcript re-extraction was performed.

### Added — 2026-09-06 (ACE weekly synthesis)
- Add `scripts/ace_weekly_report.py`, a deterministic rolling-seven-day
  Europe/Paris synthesis over existing audit JSON and incident-tracking state.
- Reuse the daily report's source/session selection and private atomic writer;
  report recurring exact-label types only after three distinct sessions, keep
  work/ingestion/audit dates separate, require evidence for successes and
  explicitly linked counterexamples, and leave correction statuses untouched.
- Invoke the weekly renderer after the independent daily renderer in the
  existing Codex worker; a weekly failure does not suppress the daily report.
- Add synthetic weekly and worker-isolation tests.

### Fixed — 2026-09-05 (ACE pre-mortem)
- The conditional Codex `turn-ended` path now documents and preserves its
  360-second delay, 30-minute per-session guard, stable-rollout cooldown, and
  ten-conversation audit batch without `--force`; no scheduler was added.
- Collection fairness now rotates new, retry, and older pending candidates and
  records pending/freshest age and mtime coverage fields.
- Query retrieval ranks before the Codex call and bounds the index to 12,000
  characters, each article to 12,000, eight articles, and the combined context
  to 24,000; fallback relevance is explicitly unverified.
- Audit completeness is independent from bounded evidence windows, rejects a
  success claim without current terminal evidence, keeps observable incidents
  on partial sources, validates same-conversation evidence references, and
  preserves separate source, ingestion, and audit dates without auto-closing
  incident workflow fields.
- Backfill redacts before daily/state writes and keeps state directories/files
  private. Final runtime validation: 74 `test_cmc*.py` tests passed.

### Changed — 2026-09-05
- Daily CMC reports now default to the ignored Agent Central private directory
  `/Users/franck/.agents/private/cmc/daily`; existing `~/.codex/reports/cmc-daily/`
  reports are preserved.

### Agent improvement cycle — 2026-09-05 (uncommitted)
- Add bounded Claude/Codex collection with durable hash checkpoints and a compile queue.
- Preserve tool evidence and distinguish observed outcomes from agent claims.
- Track structured incidents and successes against changing conversation versions.
- Add a private synthetic model evaluation harness without changing production models.
- Propagate query/compile failures; preserve the compiled input hash if a daily log changes.
- Extend tests for collection recovery, evidence retention, incidents and verified outcomes.

### Fixed
- The automatic Codex worker now preserves rollout modification times, so a
  stable copied rollout no longer becomes `SKIP active`.
- A worker checkpoint reaches `completed` only after a successful write or a
  terminal skip. Failed attempts remain eligible for retry.
- Codex subagent rollouts are excluded by default. Use `--include-subagents`
  only for an intentional historical import.
- Conversation, Markdown, compile, and query prompts redact common credentials
  before any model call or new vault write.
- Codex child diagnostics no longer echo prompts or responses into CMC logs.
- Repeated Codex system diagnostics are deduplicated within each child run.
- `backfill_codex.py --limit` now counts actual model calls. Active, subagent,
  oversized, and already ingested rollouts do not consume the requested limit.

### Added
- `scripts/cmc_health.py` reports seven-day Codex capture coverage without
  modifying the vault.
- `scripts/cmc_secret_audit.py` reports potential credentials in historical
  vault files without modifying them.
- `scripts/cmc_overengineering_audit.py` creates a private report after ten
  newly ingested Codex conversations, or after seven days when unaudited
  conversations exist. Reports stay outside the CMC vault under
  `~/.codex/reports/overengineering/`.
- First activation audits the latest batch and records older ingested sessions
  as the baseline, so future automatic runs process only new conversations.
- Frustration metadata from the shared Agent Central detector now triggers an
  immediate audit. Raw vulgar or private user text is never stored in metrics,
  and agent fault remains unverified until the report proves a concrete gap.
- Audit batches are ordered by conversation time instead of ingestion time.
  Reports must contain every requested conversation and the frustration section;
  one bounded repair pass runs before an incomplete report can be written.
- `tests/test_cmc_safety.py` covers credential redaction and subagent detection.

### Changed
- **CMC is now Codex-only end to end.** `flush.py`, `scan_md.py`, `compile.py`,
  `query.py`, and semantic `lint.py` all use the fixed `codex exec` runner with
  `gpt-5.6-luna` and `medium` reasoning. Claude Code hooks remain capture-only
  launchers; the Claude Agent SDK dependency and fallback path were removed.

### Added
- `scripts/scan_md.py` — second ingestion source alongside session transcripts. Scans `.md` files in the current git repo (README, docs/, plans/, design docs, etc.) and appends Sonnet-extracted summaries to today's daily log under `### MD Scan: <path>` sections, identifying the source file. Filters: `--init` (full rescan, clears state), `--days N` (modified in last N days), `--since YYYY-MM-DD`, `--all` (ignore hash dedup), `--path` (override repo root), `--dry-run`. Default behavior re-scans only files whose hash changed since the last scan. State stored under `state.json["scanned_md"]`.
- New `CMC_SCAN_MODEL` env var (default `claude-sonnet-4-6`).
- **Type-aware extraction in `scan_md.py`.** Files are classified into `doc` / `changelog` / `reflection` / `plan` via filename + path-segment heuristics, and the prompt branches its extraction policy per type. CHANGELOGs no longer pollute the KB with "fix typo" / "bump deps" entries — only architectural decisions and breaking changes are extracted. Drafts/notes default everything to `[Hypothèse]` to prevent exploratory thoughts from being promoted to facts. Plans tag intent as `[Hypothèse]` unless explicitly marked done. The detected type is stored in `state.json["scanned_md"][rel].type` and surfaced in the daily-log section heading (`### MD Scan (changelog): CHANGELOG.md`) so `compile.py` and humans can see which policy was applied. Detection segments are conservative: when uncertain, defaults to `doc` (standard policy).
- Interactive menu in `scan_md.py`: invoke with `-m` / `--menu` to pick an action (incremental, full rescan, last N days, since date, force-all, dry-run preview, quit). Bare invocation keeps the prior behaviour (incremental scan) so cron jobs and CI runs are not affected.
- **`/cmc-scan-md` slash command** (`~/.claude/skills/cmc-scan-md/SKILL.md`). Auto-resolves the current git repo via `git rev-parse --show-toplevel`, exports `CLAUDE_PROJECT_DIR`, dispatches to `scan_md.py` with the right flags. Parses natural-language filters from the user's invocation: "scan md depuis 7 jours" → `--days 7`, "rescanne depuis le début" → `--init`, "scan md depuis 2026-04-01" → `--since 2026-04-01`, "scan docs" → incremental default. Echoes the candidate file count before the LLM-billed scan and asks for confirmation when >20 files.
- **`/cmc` top-level slash command** (`~/.claude/skills/cmc/SKILL.md`). Single menu that lists all CMC operations (init, scan transcripts, scan md, compile, query, lint) and dispatches to the right sub-skill based on the user's pick or natural-language reply. Detects when the project isn't init'd yet and surfaces option 0 (init) prominently. Inline natural-language requests bypass the menu and dispatch directly.
- Graceful EOFError handling in `scan_md.py`'s interactive menu so the parent menu loop can recover when stdin closes.
- Vault-internal guard in `scan_md.py`: refuses to scan the `_config/` tool directory or any per-project KB folder, to avoid self-ingestion of `daily/` and `knowledge/`.
- Documented `scan_md.py` commands and a new data-flow branch in `CLAUDE.md`.

### Changed
- **All session flushes now extract through Codex by default.** Claude
  SessionEnd/PreCompact hooks, Claude transcript backfill, and Codex backfill
  use an ephemeral, read-only `codex exec` child with `gpt-5.6-luna` and
  medium reasoning. User config, project rules, and notify recursion are
  disabled for the child; the Claude Agent SDK remains available only through
  the explicit `CMC_FLUSH_ENGINE=claude` fallback.
- **Inversion philosophique flush/compile : `flush` est EXHAUSTIF, `compile` est CURATEUR.** Le flush n'est plus un filtre intelligent ; il devient un extracteur exhaustif. Les anciennes règles "filter the obvious", "skip the routine", "anti-padding" ont été remplacées par : "capture chaque atome distinct, ne filtre que les répétitions littérales et fillers (greetings, chain-of-thought, Read tool calls sans valeur)". Rationale : le flush écrit pour l'agent compile aval (pas pour un humain) — l'agent peut éliminer du signal en trop, mais ne peut pas reconstruire un signal manqué. Une session jiang de 2h51 produit maintenant ~17K chars de daily log (vs ~4K avant truncation, ~6K en mode "filtre intelligent") — tous les IDs, sub-task Asana, valeurs verbatim, citations Thomas, artefacts produits sont préservés pour que `compile.py` ait de la matière à curer.
- **Format daily log restructuré.** Nouveau format : `Contexte` (1-2 lignes) + `Déroulé` (liste numérotée chronologique, 1 événement causal par ligne avec flèches `→`) + `Décisions prises` (macro et micro) + `Chemins abandonnés` + `Découvertes / gotchas / observations` (toutes les surprises et valeurs vues) + `Entités mentionnées` (verbatim) + `Citations notables` (verbatim utilisateur/tiers) + `Artefacts produits` (avec statut) + `Action items` + `Questions ouvertes`. Le `Déroulé` permet à `compile.py` de reconstruire la séquence et les pivots de la session ; les sections atomiques fournissent les briques queryables. Pas de quota par section : autant d'items que la session a d'atomes distincts.
- **`compile.py` prompt mis à jour** pour parser la nouvelle structure des daily logs : politique explicite sur comment incorporer Déroulé / Citations / Artefacts / Questions ouvertes dans les concept articles. Indication explicite que `compile` est le curateur du signal exhaustif livré par flush.
- **Bug fix critique** : `if "FLUSH_OK" in response` était un check substring qui matchait quand le modèle echoait des fragments du prompt (le prompt contient lui-même la chaîne "FLUSH_OK" comme sentinel). Conséquence : extractions complètes silencieusement jetées. Fixé en `response.strip() == "FLUSH_OK"` (exact match). Idem pour `FLUSH_ERROR` → `startswith`. Le prompt a aussi été renforcé pour interdire l'echo des sentinels et headers de sections du prompt dans la sortie.
- **`scan_md.py` aligné** sur la philosophie d'exhaustivité. Plus de filtre "déjà connu", plus de quota maximum. Le détail par-type (doc/changelog/reflection/plan) reste, mais opère en complément de l'exhaustivité, pas en remplacement.
- **Map-reduce chunking in `flush.py` — no content is dropped, regardless of session length.** Previously the hooks truncated the conversation context to a fixed char budget (150K → 400K via head+tail) before flushing, which silently dropped the middle of multi-hour sessions. Audit of `jiang/daily/2026-05-04.md` confirmed truncation lost the Asana task ID, the 25% KPI target, the 5-lever cancel-rate strategy, and several technical specifics from the first 75 minutes of a 2h51 session. New architecture: `flush.run_flush(context)` is now a dispatcher. Below `FLUSH_SINGLE_PASS_THRESHOLD` (200K chars), one LLM call as before. Above it, the context is split at turn boundaries into chunks of `FLUSH_CHUNK_SIZE` (120K chars), each chunk gets its own partial-extraction LLM call, then a single consolidation call merges all partials into the final daily-log entry. Cost is linear in conversation size; no content is ever truncated under the 5M-char hard safety cap. New env vars: `CMC_FLUSH_SINGLE_PASS_THRESHOLD`, `CMC_FLUSH_CHUNK_SIZE`. Hooks (`session-end.py`, `pre-compact.py`) and `backfill.py` now write the full conversation to the temp file (only the absolute-safety cap of 5M applies).
- **Verbatim-preservation rule added to the flush prompt.** Specific values (Zap IDs, deal IDs, file paths, URLs, Sheet IDs, latencies, KPIs, regex formats, event names like `deal.added`) must be preserved word-for-word, not paraphrased. Reformulating sentences is fine; reformulating values is forbidden. Same rule replicated in the partial and consolidation prompts so verbatim values survive every stage of map-reduce.
- **Partial and consolidation prompts** instruct the model to deduplicate cross-chunk claims, resolve cross-chunk references ("the Zap mentioned" → `Zap 174442936`), preserve provenance markers (taking the most-cautious one when chunks disagree), and surface contradictions explicitly rather than silently overwriting.
- **`flush` now defaults to Sonnet 4.6** (was Haiku 4.5). Reasoning: the daily log is a lossy bottleneck — whatever the extractor drops at this stage is gone for good. `compile.py` and `query.py` operate on the daily log, not the original source, so they cannot recover lost nuance. Conversations contain the most subtle signal (tradeoffs explored, paths abandoned, gotchas hit during debug) — precisely what a lower-tier model strips. Override via `CMC_FLUSH_MODEL=claude-haiku-4-5` if you need cost-bounded operation on massive session loads.
- **Rewrote the `flush` prompt.** Switched to French (matches the user's daily-log language). Added explicit anti-hallucination rule, anti-padding rule, "rationale-or-omit" policy, and a section for `Chemins abandonnés` (paths explored then rejected — high-signal content previously discarded). Introduced provenance markers on every bullet (`[Établi]` / `[Décidé]` / `[Hypothèse]` / `[Découvert]`) to preserve epistemic uncertainty across compilation. Added an `Entités mentionnées` section to capture libs/services/projects systematically.
- **Rewrote the `compile` prompt.** Removed rigid quotas (3-7 concepts, 3-5 bullets, 2+ paragraphs, 2+ wikilinks) — they forced padding and fragmentation. Added explicit merge policy (compatible / contradictory / redundant) with mandatory `## Contradictions` section instead of silent overwrite. Wikilinks now strictly opt-in (no minimum). Provenance markers preserved end-to-end: `[Hypothèse]` cannot be promoted to fact without a confirming session.
- **Updated `AGENTS.md`** to match the new policies: no length minimums, no bullet quotas, optional sections (only `Sources` and frontmatter required), explicit merge policy table, provenance marker spec, contradiction-handling section in concept articles. Lint check #7 (sparse articles) downgraded to suggestion-only — sparse is now valid under the no-padding policy.
- `scripts/flush.py` is now import-safe: module-level side-effects (`sys.exit`, `mkdir`, `logging.basicConfig`) moved into `_bootstrap_for_main()` and called from `main()`. Allows `scan_md.py` to reuse `append_to_daily_log`, the vault-wide flush lock, and the retry constants without triggering early exits or hijacking the logging configuration.

## [2026-05-03]

### Added
- Capture Claude Code's native `/compact` summaries.

## [2026-04-28]

### Changed
- Flush captures the full conversation by default (`CMC_FLUSH_MAX_TURNS=1000`, `CMC_FLUSH_MAX_CHARS=150000`). Truncation keeps the last N chars where conclusions live.

## [2026-04-27]

### Added
- Tiered models: Haiku 4.5 for `flush` (cheap extraction), Sonnet 4.6 for `compile` and `query` (complex reasoning, large context). Override via `CMC_FLUSH_MODEL` / `CMC_COMPILE_MODEL` / `CMC_QUERY_MODEL`.
- Project-aware paths: `$CLAUDE_PROJECT_DIR` + `git rev-parse --show-toplevel` resolve which per-project KB folder to use.

### Fixed
- Use verified model IDs (Haiku 4.5, Sonnet 4.6) after live SDK validation.

### Changed
- Project routing is opt-in: a project gets KB storage only when its vault folder already exists. Prevents accidental capture in random git repos.

## [2026-04-07]

### Added
- Backfill picks the transcripts directory with the most recent file when both CCS (`~/.ccs/...`) and standard (`~/.claude/projects/...`) paths exist.
- Context extraction now captures tool calls (truncated to 200 chars) and subagent transcripts.

### Fixed
- Flush silences `FLUSH_OK` entries and tightens the prompt to avoid raw dialogue dumps in the daily log.
- Backfill keeps the leading dash in the cwd slug so it correctly matches the Claude Code transcripts directory.

## [2026-04-06]

### Added
- Initial release: hooks (`session-start`, `session-end`, `pre-compact`), `flush.py`, `compile.py`, `query.py`, `lint.py`, `backfill.py`. Knowledge base layout (`daily/`, `knowledge/concepts/`, `connections/`, `qa/`). Index-guided retrieval, no RAG.
