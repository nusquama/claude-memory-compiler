"""Manage additive ACE schema changes and private, lossless data backups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ace_database import SupabaseStore, _sql_json

TABLES = ("schema_migrations", "projects", "sessions", "revisions", "messages", "attachments",
          "processing_runs", "observations", "recommendations", "results", "decisions",
          "corrections", "evaluations", "knowledge_versions")
# Processing leases are transient coordination, not durable conversation data.
# Do not restore an old live owner after restoring the durable tables.
MAX_BACKUP_BYTES = 128 * 1024 * 1024


def status(store: SupabaseStore) -> dict[str, Any]:
    rows = store._rows("SELECT count(*)::integer AS tables, bool_and(rowsecurity) AS rls_enabled "
                       "FROM pg_tables WHERE schemaname = 'ace'")
    return rows[0]


def migration_plan(store: SupabaseStore, directory: Path | None = None) -> dict[str, Any]:
    """Select unapplied, numbered migrations without replaying old grants."""
    directory = directory or Path(__file__).parents[1] / "migrations"
    initialized = bool(status(store)["tables"])
    applied = {
        int(row["version"])
        for row in store._rows("SELECT version FROM ace.schema_migrations ORDER BY version")
    } if initialized else set()
    migrations = []
    versions = set()
    for path in sorted(directory.glob("*.sql")):
        match = re.fullmatch(r"([0-9]+)_[a-z0-9_]+\.sql", path.name)
        if not match or path.is_symlink():
            raise ValueError("invalid migration filename")
        version = int(match[1])
        if version in versions:
            raise ValueError("duplicate migration version")
        versions.add(version)
        if version not in applied:
            migrations.append({"version": version, "path": str(path)})
    return {"initialized": initialized, "applied_versions": sorted(applied), "pending": sorted(migrations, key=lambda item: item["version"])}


def apply_migrations(store: SupabaseStore, directory: Path | None = None) -> dict[str, Any]:
    plan = migration_plan(store, directory)
    previous = backup(store) if plan["initialized"] and plan["pending"] else None
    applied = []
    for migration in plan["pending"]:
        store._rows(Path(migration["path"]).read_text())
        rows = store._rows("SELECT version FROM ace.schema_migrations ORDER BY version")
        if migration["version"] not in {int(row["version"]) for row in rows}:
            raise RuntimeError("migration did not record its successful application")
        applied.append(migration["version"])
    return {"applied": bool(applied), "applied_versions": applied, "backup": previous, **status(store)}


def backup(store: SupabaseStore, destination: Path | None = None) -> dict[str, Any]:
    target = destination or (Path.home() / ".agents/private/ace/backups" /
                            (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"))
    fields = ", ".join(f"'{table}', (SELECT coalesce(jsonb_agg(to_jsonb(t)), '[]'::jsonb) FROM ace.{table} t)" for table in TABLES)
    row = store._rows(f"SELECT jsonb_build_object({fields}) AS tables")[0]
    payload = {"schema_version": 1, "tables": row["tables"]}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    if len(data) > MAX_BACKUP_BYTES:
        raise ValueError("backup exceeds bound; use the database administrator backup")
    if target.exists() or target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError("backup target exists or uses a symlink")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".backup-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"path": str(target), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "rows": {table: len(row["tables"][table]) for table in TABLES}}


def restore(store: SupabaseStore, source: Path, *, apply: bool = False) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_BACKUP_BYTES:
        raise ValueError("invalid backup file")
    payload = json.loads(source.read_text(encoding="utf-8"))
    tables = payload.get("tables")
    if payload.get("schema_version") != 1 or not isinstance(tables, dict) or set(tables) != set(TABLES):
        raise ValueError("invalid backup schema")
    if not all(isinstance(tables[name], list) for name in TABLES):
        raise ValueError("invalid backup rows")
    result: dict[str, Any] = {"applied": False, "rows": {name: len(tables[name]) for name in TABLES}, "mode": "insert_missing_only"}
    if not apply:
        return result
    # One transaction restores FK parents first. Existing unequal rows cause
    # the entire transaction to fail; current data is never overwritten.
    statements = ["BEGIN"]
    for name in TABLES:
        data = _sql_json(tables[name])
        statements.append(f"INSERT INTO ace.{name} SELECT * FROM jsonb_populate_recordset(NULL::ace.{name}, {data}) ON CONFLICT DO NOTHING")
        statements.append(f"DO $verify$ BEGIN IF EXISTS (SELECT * FROM jsonb_populate_recordset(NULL::ace.{name}, {data}) EXCEPT SELECT * FROM ace.{name}) THEN RAISE EXCEPTION 'ACE restore conflicts with current {name}'; END IF; END $verify$")
    statements.append("COMMIT")
    store._rows(";\n".join(statements) + ";")
    return {**result, "applied": True}


def verify(store: SupabaseStore) -> dict[str, Any]:
    """Exercise atomicity, roles, revisions and restore in a rolled-back fixture."""
    sql = """BEGIN;
DO $proof$
DECLARE
 p uuid := gen_random_uuid();
 saved jsonb;
 n integer;
 v_receipt_project uuid;
 v_receipt_source text;
 v_receipt_session text;
 v_receipt_revision text;
 v_inserted boolean;
 v_status text;
 v_latest_revision text;
 v_rejected boolean;
 base jsonb;
 marker jsonb;
 retry_a jsonb;
 retry_b jsonb;
 v_first_session text;
 analysis jsonb;
 unsafe jsonb;
BEGIN
 base := jsonb_build_object('schema_version',1,
  'project',jsonb_build_object('id',p,'name','ACE verification','root','/ace/fixture','vault_dir','/ace/fixture/vault'),
  'source','codex','session_id','ace-verification','revision',repeat('a',64),
  'started_at','2026-09-01T01:00:00Z','updated_at','2026-09-01T01:00:00Z',
  'messages',jsonb_build_array(jsonb_build_object('id','m1','ordinal',0,'role','user','type','message','content','ACE_ROUNDTRIP')), 'attachments','[]'::jsonb);
 BEGIN
  PERFORM ace.register_project(p,'ACE verification','/ace/fixture','/ace/fixture/vault',true,true);
  PERFORM ace.ingest_snapshot(base);
  SELECT snapshot INTO saved FROM ace.revisions WHERE project_id=p;
  RAISE no_data_found;
 EXCEPTION WHEN no_data_found THEN NULL;
 END;
 IF EXISTS (SELECT 1 FROM ace.revisions WHERE project_id=p) THEN RAISE EXCEPTION 'fixture rollback failed'; END IF;
 PERFORM ace.register_project(p,'ACE verification','/ace/fixture','/ace/fixture/vault',true,true);
 SELECT i.project_id, i.source, i.session_id, i.revision, i.inserted, i.status
   INTO v_receipt_project, v_receipt_source, v_receipt_session, v_receipt_revision, v_inserted, v_status
   FROM ace.ingest_snapshot(saved) AS i;
 IF v_receipt_project<>p OR v_receipt_source<>'codex' OR v_receipt_session<>'ace-verification'
    OR v_receipt_revision<>repeat('a',64) OR v_inserted IS DISTINCT FROM true
    OR v_status<>'accepted' THEN
  RAISE EXCEPTION 'ingest acknowledgement failed';
 END IF;
 SELECT i.inserted, i.status INTO v_inserted, v_status
   FROM ace.ingest_snapshot(saved) AS i;
 IF v_inserted IS DISTINCT FROM false OR v_status<>'accepted' THEN
  RAISE EXCEPTION 'ingest retry acknowledgement failed';
 END IF;
 SELECT count(*) INTO n FROM ace.messages WHERE project_id=p;
 IF n<>1 THEN RAISE EXCEPTION 'restore or idempotence failed'; END IF;
 PERFORM ace.ingest_snapshot(saved || jsonb_build_object('revision',repeat('b',64),'updated_at',NULL));
 SELECT latest_revision INTO v_latest_revision
 FROM ace.sessions
 WHERE project_id=p AND source='codex' AND session_id='ace-verification';
 IF v_latest_revision<>repeat('a',64) THEN RAISE EXCEPTION 'NULL latest ordering failed'; END IF;
 SELECT count(*) INTO n FROM ace.pending_snapshots(10,'extraction',p);
 IF n<>1 THEN RAISE EXCEPTION 'pending project isolation failed'; END IF;
 PERFORM ace.claim_stage(p,'codex','ace-verification',repeat('a',64),'extraction','fixture-owner','fixture-host',1800);
 PERFORM ace.mark_stage('codex','ace-verification',repeat('a',64),p,'extraction','succeeded','fixture-owner','fixture-host',NULL);
 SELECT count(*) INTO n FROM ace.pending_snapshots(10,'extraction',p);
 IF n<>0 THEN RAISE EXCEPTION 'latest revision processing failed'; END IF;
 retry_a := saved || jsonb_build_object(
    'session_id','retry-a', 'revision', repeat('1',64),
    'updated_at','2026-09-01T02:00:00Z');
 retry_b := saved || jsonb_build_object(
    'session_id','retry-b', 'revision', repeat('2',64),
    'updated_at','2026-09-01T03:00:00Z');
 PERFORM ace.ingest_snapshot(retry_a);
 PERFORM ace.ingest_snapshot(retry_b);
 PERFORM ace.claim_stage(p,'codex','retry-a',repeat('1',64),'extraction','fixture-owner','fixture-host',1800);
 PERFORM ace.mark_stage('codex','retry-a',repeat('1',64),p,'extraction','failed','fixture-owner','fixture-host','fixture failure');
 SELECT session_id INTO v_first_session FROM ace.pending_snapshots(1,'extraction',p);
 IF v_first_session<>'retry-b' THEN RAISE EXCEPTION 'unattempted snapshot did not precede retry'; END IF;
 PERFORM ace.claim_stage(p,'codex','retry-b',repeat('2',64),'extraction','fixture-owner','fixture-host',1800);
 PERFORM ace.mark_stage('codex','retry-b',repeat('2',64),p,'extraction','failed','fixture-owner','fixture-host','fixture failure');
 SELECT count(*) INTO n FROM ace.pending_snapshots(10,'extraction',p)
 WHERE session_id IN ('retry-a','retry-b');
 IF n<>2 THEN RAISE EXCEPTION 'failed snapshots were not retryable'; END IF;
 PERFORM ace.claim_stage(p,'codex','retry-a',repeat('1',64),'extraction','fixture-owner','fixture-host',1800);
 PERFORM ace.mark_stage('codex','retry-a',repeat('1',64),p,'extraction','running','fixture-owner','fixture-host',NULL);
 SELECT session_id INTO v_first_session FROM ace.pending_snapshots(1,'extraction',p);
 IF v_first_session<>'retry-b' THEN RAISE EXCEPTION 'retry timestamp was not refreshed'; END IF;
 marker := saved || jsonb_build_object(
    'revision', repeat('f',64),
    'updated_at', NULL,
    'messages', jsonb_build_array(jsonb_build_object(
      'id','marker','ordinal',0,'role','assistant','type','tool',
      'content',jsonb_build_object('text','TOKEN=<REDACTED>' || chr(10) || 'nextword'))));
 SELECT i.inserted, i.status INTO v_inserted, v_status
 FROM ace.ingest_snapshot(marker) AS i;
 IF v_inserted IS DISTINCT FROM true OR v_status<>'accepted' THEN
  RAISE EXCEPTION 'redacted marker context was rejected';
 END IF;
 SELECT count(*) INTO n FROM ace.search_history(p,'ACE_ROUNDTRIP',10)
 WHERE kind='message' AND payload->>'session_id'='ace-verification';
 IF n<>1 THEN RAISE EXCEPTION 'message history or latest revision failed'; END IF;
 analysis := jsonb_build_object('observations',jsonb_build_array(jsonb_build_object('problem_signature','fixture','evidence',jsonb_build_object('message_id','m1'))),
    'recommendations',jsonb_build_array(jsonb_build_object('recommendation','Test the retry','evidence',jsonb_build_object('message_id','m1'))));
 PERFORM ace.save_analysis(p,'codex','ace-verification',repeat('b',64),analysis);
 PERFORM ace.save_analysis(p,'codex','ace-verification',repeat('b',64),analysis);
 SELECT count(*) INTO n FROM ace.observations WHERE project_id=p;
 IF n<>1 THEN RAISE EXCEPTION 'analysis observation duplicated'; END IF;
 SELECT count(*) INTO n FROM ace.recommendations WHERE project_id=p;
 IF n<>1 THEN RAISE EXCEPTION 'analysis recommendation duplicated'; END IF;
 unsafe := saved || jsonb_build_object(
    'revision', repeat('c',64),
    'messages', jsonb_build_array(jsonb_build_object(
      'id','unsafe','ordinal',0,'role','user','type','message',
      'content',jsonb_build_object('reasoning','hidden','data','QUJDREVGR0hJSktMTU5PUA==',
                                    'text','API_KEY=ACE_TEST_MARKER'))));
 v_rejected := false;
 BEGIN
  PERFORM ace.ingest_snapshot(unsafe);
 EXCEPTION WHEN OTHERS THEN
  v_rejected := true;
 END;
 IF NOT v_rejected OR EXISTS (SELECT 1 FROM ace.revisions WHERE project_id=p AND revision=repeat('c',64)) THEN
  RAISE EXCEPTION 'unsanitized snapshot was accepted';
 END IF;
 IF has_function_privilege('ace_reader','ace.ingest_snapshot(jsonb)','EXECUTE')
  OR has_table_privilege('ace_reader','ace.messages','SELECT')
  OR has_function_privilege('public','ace.pending_snapshots(integer,text,uuid)','EXECUTE')
  OR has_function_privilege('public','ace.claim_stage(uuid,text,text,text,text,text,text,integer)','EXECUTE')
  OR has_function_privilege('ace_processor','ace.mark_processed(text,text,text,uuid,text,text,text)','EXECUTE') THEN
  RAISE EXCEPTION 'permission separation failed';
 END IF;
 PERFORM ace.publish_compiled_snapshot(p,1,jsonb_build_object('schema_version',1,'project_id',p,'files',jsonb_build_object('index.md','# ACE')),repeat('c',64));
 SELECT count(*) INTO n FROM ace.read_compiled_snapshot(p,1);
 IF n<>1 THEN RAISE EXCEPTION 'compiled reader failed'; END IF;
 PERFORM ace.register_project(p,'ACE verification','/ace/fixture','/ace/fixture/vault',false,false);
 IF EXISTS (SELECT 1 FROM ace.search_history(p,'ACE_ROUNDTRIP',10)) THEN
  RAISE EXCEPTION 'disabled history search leaked content';
 END IF;
 IF EXISTS (SELECT 1 FROM ace.read_compiled_snapshot(p,1)) THEN
  RAISE EXCEPTION 'disabled compiled reader leaked content';
 END IF;
 v_rejected := false;
 BEGIN
  PERFORM ace.publish_compiled_snapshot(p,2,jsonb_build_object('schema_version',1),repeat('d',64));
 EXCEPTION WHEN OTHERS THEN
  v_rejected := true;
 END;
 IF NOT v_rejected THEN RAISE EXCEPTION 'disabled compiled publication was accepted'; END IF;
 v_rejected := false;
 BEGIN
  PERFORM ace.ingest_snapshot(saved || jsonb_build_object('revision',repeat('e',64)));
 EXCEPTION WHEN OTHERS THEN
  v_rejected := true;
 END;
 IF NOT v_rejected OR EXISTS (SELECT 1 FROM ace.revisions WHERE project_id=p AND revision=repeat('e',64)) THEN
  RAISE EXCEPTION 'disabled project accepted ingestion';
 END IF;
END $proof$;
SET LOCAL ROLE ace_reader;
SELECT count(*) >= 0 AS reader_function_allowed FROM ace.list_projects();
ROLLBACK;"""
    store._rows(sql)
    return {"atomic_restore": "pass", "idempotence": "pass", "latest_revision": "pass",
            "project_isolation": "pass", "message_history": "pass", "permissions": "pass",
            "compiled_read": "pass", "analysis_retry": "pass", "retry_fairness": "pass",
            "persistent_fixture_rows": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "plan", "apply", "backup", "restore", "verify"), nargs="?", default="status")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--apply", action="store_true", help="apply an additive restore")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    store = SupabaseStore(timeout=120)
    try:
        if args.action == "plan":
            result = migration_plan(store)
        elif args.action == "apply":
            result = apply_migrations(store)
        elif args.action == "backup":
            result = backup(store, args.file)
        elif args.action == "restore":
            if args.file is None:
                raise ValueError("restore requires --file")
            result = restore(store, args.file, apply=args.apply)
        elif args.action == "verify":
            result = verify(store)
        else:
            result = status(store)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
