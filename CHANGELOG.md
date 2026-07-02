# Changelog

All notable changes to this project are documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
