# ACE Runtime Contract

This page describes the local runtime.
The canonical source is the
[ACE process](/Users/franck/.agents/docs/ace/processus-ace.md).

## Implementation

| Function | Main file | Contract |
|---|---|---|
| Memory branch | `scripts/ace_pipeline.py` | DB-acquitted extraction, `daily/`, compile, `knowledge/` |
| Improvement branch | `scripts/ace_learning.py` | Evidence, incidents, recommendations, corrections, and effectiveness |
| Source filtering | `scripts/ace_transcripts.py` | Secrets masked, media referenced, internal reasoning excluded |
| Database transport | `scripts/ace_database.py` | Supabase stdin wrapper, `amastuces` profile, `ace` schema |
| Scheduling | `scripts/ace_schedule.py` | `launchd`, 1,800-second tick, Europe/Paris |
| Schema | `migrations/001_ace.sql` | 14 tables and `ace.*` functions |

The processor uses `gpt-5.6-luna` with `medium` reasoning.
Astra remains the orchestrator for substantial work.
The local vault is the master source for compiled knowledge.
Compiled database versions remain read-only views.

After a database acknowledgement, the flow branches logically:

```text
DB-acquitted revision
    |-> memory: extraction -> daily/ -> compile -> knowledge/
    `-> improvement: evidence windows -> analysis -> reports
```

The branches share the acknowledgement and processor state. The documented
schedule does not prove independent cadences or concurrent execution.

## State checked on 2026-09-07

- The `amastuces` profile and `ace` schema respond to a read-only query.
- The 2026-09-07 measurement found four enabled and initialized projects:
  `.agents`, `_config`, `hermes-agents`, and `jiang`.
- The same measurement counted 10 sessions, 10 revisions, 1,905 messages, and
  14 ACE tables with RLS enabled on all 14.
- The same DB measure counted 6 observations, 12 recommendations, including 3
  verified real frustration recommendations, and 9 snapshots pending processing.
- Latest validation passed 220 runtime tests and 5 Agent Central tests.
- The additive migration was applied after a private 21,802,182-byte backup;
  45 states were migrated and original copies were retained.
- A 2026-08-31 daily produced five articles; resumed `ace compile` returned
  `OK`, and a new-process `ace query` returned a response with a verified source.
- The v3 publication and read-only copy were checked: 6 articles, 8 files, 6
  valid index links, and a deterministic index.
- The corrected learning report covers 10 sessions and preserves 18 attempts:
  4 are `OK`, 6 model errors remain to retry, and `A` was validated by replaying
  real JSON without a new model call.
- The four `OK` reports are distinct from the single current ACK and the 9 pending
  processing snapshots.
- Native lot-1 conversation/context analysis preserves evidence; later passages
  remain under retry, the normalizer remains under targeted correction, and
  collection is unchanged.
- Final compile/analysis independence and retry fairness remain under completion.
- The corrected real-report rerun remained in progress and required validation.
- `ace schedule status --json` returns `installed: false`.
- The E2E flow still needs to be run.

These observations are dated and partial. They do not prove scheduler
installation, independent branch cadences, or a successful full cycle.

## Non-mutating commands

```bash
/Users/franck/.agents/bin/ace status
/Users/franck/.agents/bin/ace schedule plan --json
/Users/franck/.agents/bin/ace schedule status --json
```

Old CMC traces remain readable for history.
They are not active runtime routes.
