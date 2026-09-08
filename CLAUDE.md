# CLAUDE.md

This checkout contains the Agent Continuous Evolution (ACE) runtime.
It provides two functions: project memory and evidence-based agent improvement.

The canonical entry point is `/Users/franck/.agents/bin/ace`.
The complete process is documented in the
[canonical ACE process](/Users/franck/.agents/docs/ace/processus-ace.md).
The article editing contract remains in [AGENTS.md](AGENTS.md).

## Main commands

```bash
# Read-only checks
/Users/franck/.agents/bin/ace status
/Users/franck/.agents/bin/ace schedule plan --json
/Users/franck/.agents/bin/ace schedule status --json

# Bounded pipeline
/Users/franck/.agents/bin/ace init --cwd /path/to/project
/Users/franck/.agents/bin/ace collect --cwd /path/to/project
/Users/franck/.agents/bin/ace sync --cwd /path/to/project
/Users/franck/.agents/bin/ace process --cwd /path/to/project
/Users/franck/.agents/bin/ace daily --cwd /path/to/project
/Users/franck/.agents/bin/ace tick

# Explicit retained delegations
/Users/franck/.agents/bin/ace compile --cwd /path/to/project
/Users/franck/.agents/bin/ace query --cwd /path/to/project "question"
/Users/franck/.agents/bin/ace scan-md --cwd /path/to/project --dry-run
```

Use `ace init` to explicitly register a project.
Do not treat a vault directory name as an implicit authorization.
Do not use a `Conversations` fallback directory.

## Architecture

```text
Claude, Codex, or Hermes source
    -> metadata discovery and explicit project opt-in
    -> read-only adapter, normalization, and filtering
    -> private SQLite outbox
    -> Supabase stdin: amastuces profile, ace schema
    -> database-acquitted revision
         |-> memory branch: Luna extraction -> daily/YYYY-MM-DD.md
         |                  -> daily compilation -> knowledge/
         `-> improvement branch: evidence analysis -> reports
```

The pipeline keeps a strict boundary between source, transport, and processing.
The collector selects the project before reading the bounded source body,
normalizes and filters it, and places it in the outbox.
The processor processes only snapshots acquitted by Supabase.
A failure leaves records in the outbox for retry.

### Filtering

`scripts/ace_transcripts.py` produces one JSON envelope for Claude, Codex, and
Hermes.
The filter masks sensitive fields and token candidates.
It replaces bytes, images, and encoded data with references.
It removes `analysis`, `thinking`, `reasoning`, and equivalent blocks.
It keeps visible messages and useful tool references.
It retains non-sensitive metadata and `token_usage` needed for coverage and
diagnostics.

The revision is calculated after this cleanup.
Secrets and media therefore do not enter message or report copies.

### Supabase

`scripts/ace_database.py` calls only the wrapper
`/Users/franck/.agents/bin/supabase`.
The profile is `amastuces` and the schema is `ace`.
Queries go through stdin, including SQL payloads larger than 2 MB.
The wrapper resolves credentials; the runtime never reads them.

The database separates ingestion, processing, and read roles.
It stores filtered snapshots, processing stages, and improvement tracking.
Compiled versions are readable in the database, but the local vault remains master.

### Project and vault

The vault registry binds each project to its Git root and UUID.
A project without explicit registration remains outside collection.
The 2026-09-07 measurement found four enabled and initialized projects:
`.agents`, `_config`, `hermes-agents`, and `jiang`. This is a dated measure,
not a permanent coverage guarantee.
The latest validation passed 220 runtime tests and 5 Agent Central tests.

```text
<vault>/<project>/
├── daily/
└── knowledge/
    ├── concepts/
    ├── connections/
    ├── qa/
    ├── index.md
    └── log.md
```

`daily/` contains approved extractions.
`knowledge/` contains compiled articles.
`AGENTS.md` defines formats, provenance, and merge policy.

## Model contract and improvement

Model-backed stages call `gpt-5.6-luna` with fixed `medium` reasoning.
Deterministic stages can run without a model.
Astra orchestrates substantial work and verifies the results.

There is no recursive LLM path and no empty-model invocation.
The strict acknowledgement marker gates processing. Retries use source hashes
and daily state, and coverage records retain timestamps and project scope.

`scripts/ace_learning.py` builds evidence windows linked to messages.
It distinguishes expectations, preferences, refusals, frustrations, tool errors,
overengineering, false completion, and observable outcomes.

Reports retain:

- evidence and limitations;
- incidents and causes;
- proposed recommendations;
- refusals and decisions;
- explicit corrections;
- acceptance, application, and effectiveness states.

A recommendation never changes a rule or skill automatically.
Silence and absence of an incident do not prove resolution.
An incomplete conversation cannot become `success`.

## Native scheduling

`scripts/ace_schedule.py` describes the Mac `launchd` service.
The service calls `ace tick` every 1,800 seconds.
The tick owns the processor lock and persistent catch-up state.
The active plist is `/Users/franck/Library/LaunchAgents/com.agentcentral.ace.plist`
with mode `600`; the current launch reports `runs=1` and PID `54145`.
The lock was observed held, and a concurrent attempt was rejected by `fcntl`.

Daily compilation and analysis start at 07:00 Europe/Paris.
The report targets 08:00 for the previous day.
An asleep or powered-off Mac can delay the report.
There is no Codex heartbeat.
The first native cycle is still running and has not returned an exit code.
The 07:00 to 08:00 target remains best-effort when the Mac is awake.

Use `ace schedule plan --json` to inspect the plan.
Use `ace schedule status --json` to inspect installation without changing it.

## Limits and verified state

The additive migration contains 14 tables under `ace`, with RLS enabled on all
14 at the 2026-09-07 measurement. That measurement counted 10 sessions, 10
revisions, 1,905 messages, and the four opt-in projects listed above.
A private latest backup preceded migration application:
`/Users/franck/.agents/private/ace/backups/20260907T152703561667Z.json`
(21,848,234 bytes, SHA-256
`3366ebcc0723d2df1f4d6b22826d20e744739ce12b7805bfeda6fb24e4ba6ecc`). Forty-five
states were migrated while original copies were retained.
The same DB measure counted 6 observations, 12 recommendations, including 3
verified real frustration recommendations, and 9 snapshots pending processing.
An observed 2026-08-31 daily produced five articles; resumed `ace compile`
returned `OK`, and a new-process `ace query` returned a response with a verified
source. The v3 publication and read-only copy were checked: 6 articles, 8 files,
6 valid index links, and a deterministic index. The complete E2E flow remains
unfinished.
The corrected learning report covers 10 sessions and preserves 18 attempts:
4 are `OK`, 6 model errors remain to retry, and `A` was validated by replaying
real JSON without a new model call. The four `OK` reports are distinct from the
single current ACK and the 9 pending-processing snapshots, which are the
pre-first-run state. Two outbox entries are currently visible as `pending` while
the first cycle runs.
Native lot-1 conversation/context analysis preserves evidence; later passages
remain under retry, the normalizer and history are corrected, and collection is
unchanged. Final compile/analysis independence and retry fairness are delivered.
The canonical quality report is readable at
`private/ace/reports/8d9ed0fc-8485-51ed-9c0f-10826b15acbb/analysis/latest-daily.md`
for 10 sources with 4 `OK` and 6 model errors; the operational daily report
represents only valid loaded audits.
These observations do not prove exhaustive collection or a successful final
report.

Historical CMC traces remain available for old-state reading.
They are not active routes and must not be used as aliases.

## Local documentation

- [Index](docs/INDEX.md)
- [Runtime ACE contract](docs/ace.md)
- [Documentation changelog](docs/CHANGELOG.md)
- [Article schema and merge policy](AGENTS.md)
