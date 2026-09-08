# Agent Continuous Evolution

ACE turns authorized conversations into project memory and evidence-based
agent-improvement reports.

The canonical entry point is `/Users/franck/.agents/bin/ace`.
The complete process is documented in the
[canonical ACE process](/Users/franck/.agents/docs/ace/processus-ace.md).
The local index is [docs/INDEX.md](docs/INDEX.md).

## Two functions

ACE has two distinct functions:

1. capture authorized sources, write daily logs, and compile the previous day;
2. analyze evidence and produce observations, recommendations, refusals,
   corrections, effectiveness evaluations, and reports.

The local vault is the master for compiled knowledge.
The database stores filtered snapshots and read-only compiled versions.

## Flow

```text
Claude, Codex, and Hermes sources
    -> project opt-in and metadata discovery
    -> read-only adapter and filtering
    -> private outbox
    -> Supabase wrapper on stdin: amastuces / ace
    -> database-acquitted revision
         |-> memory branch: Luna extraction -> daily/ -> knowledge/
         `-> improvement branch: evidence -> observations and reports
```

Projects must be explicitly registered with `ace init`.
A vault folder alone does not grant authorization.
A transport failure leaves records in the outbox for retry.
The collector filters after project selection; downstream processing accepts only
database-acquitted revisions.

## Model contract

Model-backed stages use fixed `gpt-5.6-luna` with `medium` reasoning.
Astra orchestrates substantial work and verifies the results.
The scheduler does not start a model to wait or count.

## Scheduling

The `launchd` service on the central Mac is active since 2026-09-07 17:33
Europe/Paris.
It calls `ace tick` every 1,800 seconds.
The current launch has `runs=1` and PID `54145`.
The plist is `/Users/franck/Library/LaunchAgents/com.agentcentral.ace.plist`
with mode `600`.
The plan targets collection ticks and daily processing from 07:00 Europe/Paris.
The report targets 08:00 for the previous day.
Sleep, failure, or delay can move that target.
There is no Codex heartbeat.
The first native cycle is still running and has not returned an exit code.
The schedule remains best-effort when the Mac is asleep.

The plan is visible without writing:

```bash
/Users/franck/.agents/bin/ace status
/Users/franck/.agents/bin/ace schedule plan --json
/Users/franck/.agents/bin/ace schedule status --json
```

At the 2026-09-07 read-only measurement, schema `ace` contained 14 tables with
RLS enabled on all 14. Four opt-in projects were registered: `.agents`,
`_config`, `hermes-agents`, and `jiang`; the same measurement counted 10
sessions, 10 revisions, and 1,905 messages.
The latest validation passed 220 runtime tests and 5 Agent Central tests.
The additive migration had been applied after the latest private backup
`/Users/franck/.agents/private/ace/backups/20260907T152703561667Z.json`
(21,848,234 bytes, SHA-256
`3366ebcc0723d2df1f4d6b22826d20e744739ce12b7805bfeda6fb24e4ba6ecc`).
The measured state is dated and is not an exhaustive-coverage guarantee.
The same DB measure counted 6 observations, 12 recommendations, including 3
verified real frustration recommendations, and 9 snapshots pending processing.
Two outbox entries are currently visible as `pending` while the first native
cycle runs.
An observed 2026-08-31 daily produced five articles; a resumed `ace compile`
returned `OK`, and a new-process `ace query` returned a response with a verified
source. The v3 publication and read-only copy were checked: 6 articles, 8 files,
6 valid index links, and a deterministic index. These are partial observations,
not an E2E completion claim.
The corrected learning report covers 10 sessions and preserves 18 attempts:
4 are `OK`, 6 model errors remain to retry, and `A` was validated by replaying
real JSON without a new model call. The four `OK` reports are distinct from the
single current ACK and the pending-processing snapshots.
Native lot-1 conversation/context analysis preserves evidence; later passages
remain under retry, the normalizer and history are corrected, collection is
unchanged, and final compile/analysis independence plus retry fairness are
delivered.
The canonical quality report is readable at
`private/ace/reports/8d9ed0fc-8485-51ed-9c0f-10826b15acbb/analysis/latest-daily.md`
for 10 sources with 4 `OK` and 6 model errors; the operational daily report
represents only valid loaded audits.
The 4 `OK` reports are distinct from the single current ACK. The 9 pending
snapshots are the pre-first-run state and may change live.
The end-to-end flow is not yet complete because the first cycle is still running.

## Local documentation

- [Runtime index](docs/INDEX.md): local pages and canonical links.
- [Runtime ACE note](docs/ace.md): checkout-specific contract.
- [AGENTS.md](AGENTS.md): article formats and merge policy.
- [CLAUDE.md](CLAUDE.md): runtime operating notes.
- [CHANGELOG.md](CHANGELOG.md): the runtime's declared general history.

Historical CMC traces and paths remain available for old-state reading, with a
recoverable archive and preserved original state copies.
They are not active commands or aliases.
