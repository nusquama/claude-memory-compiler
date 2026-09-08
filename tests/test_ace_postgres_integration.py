"""Real PostgreSQL proofs; run only against an explicitly isolated local socket."""
import json
import os
from pathlib import Path
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture
def pg():
    socket = os.environ.get('ACE_TEST_POSTGRES_SOCKET')
    if not socket:
        pytest.skip('isolated local PostgreSQL socket not configured')
    assert Path(socket).resolve().is_relative_to(Path('/private/tmp'))
    command = ['/opt/homebrew/opt/postgresql@15/bin/psql', '-h', socket, '-p', os.environ.get('ACE_TEST_POSTGRES_PORT','55439'), '-d','postgres','-X','-qAt','-v','ON_ERROR_STOP=1']

    def query(sql, allow_error=False):
        result = subprocess.run(command,input=sql,capture_output=True,text=True,timeout=15)
        if not allow_error:
            assert result.returncode == 0,result.stderr
        return result

    assert query("SELECT current_setting('listen_addresses') = ''").stdout.strip() == 't'
    project = str(uuid.uuid4())
    query(f"SELECT * FROM ace.register_project('{project}','Synthetic PG proof','/ace/synthetic','/ace/synthetic/vault',true,true)")
    return command,query,project


def add_snapshot(pg, session='fixture', day='2026-09-07', revision=None):
    _,query,project=pg
    revision=revision or uuid.uuid4().hex*2
    payload={'schema_version':1,'project':{'id':project,'name':'Synthetic PG proof','root':'/ace/synthetic','vault_dir':'/ace/synthetic/vault'},'source':'codex','session_id':session,'revision':revision,'started_at':day+'T09:00:00Z','updated_at':day+'T09:01:00Z','messages':[{'id':'m1','ordinal':1,'role':'user','type':'message','content':'Synthetic message'}],'attachments':[]}
    query("SELECT * FROM ace.ingest_snapshot('"+json.dumps(payload).replace("'","''")+"'::jsonb)")
    return revision


def claim_sql(project, revision, owner, session='fixture'):
    return f"SELECT row_to_json(t) FROM ace.claim_stage('{project}','codex','{session}','{revision}','analysis','{owner}','fixture-host',1800) t"


def mark_sql(project, revision, owner, state='succeeded'):
    return f"SELECT * FROM ace.mark_stage('codex','fixture','{revision}','{project}','analysis','{state}','{owner}','fixture-host',NULL)"


def test_concurrent_claim_expiry_recovery_and_stale_owner_rejection(pg):
    _,query,project=pg
    revision=add_snapshot(pg)
    def claim(owner):
        return json.loads(query('BEGIN; SET LOCAL ROLE ace_processor; '+claim_sql(project,revision,owner)+'; COMMIT;').stdout)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(claim,['first','second']))
    assert sum(result['claimed'] for result in results)==1
    owner=next(result['lease_owner'] for result in results if result['claimed'])
    assert claim(owner)['claimed']
    query(f"UPDATE ace.processing_leases SET lease_until=clock_timestamp()-interval '1 second' WHERE project_id='{project}'")
    assert claim('replacement')['claimed']
    old=query('BEGIN; SET LOCAL ROLE ace_processor; '+mark_sql(project,revision,owner)+'; COMMIT;',allow_error=True)
    assert old.returncode!=0 and 'missing or expired' in old.stderr
    query('BEGIN; SET LOCAL ROLE ace_processor; '+mark_sql(project,revision,'replacement')+'; COMMIT;')
    assert claim('late')['claimed'] is False
    assert query("SELECT has_function_privilege('ace_processor','ace.mark_processed(text,text,text,uuid,text,text,text)','EXECUTE')").stdout.strip()=='f'


def test_claim_waiting_on_terminal_completion_cannot_reopen_success(pg):
    command,query,project=pg
    revision=add_snapshot(pg)
    assert json.loads(query(claim_sql(project,revision,'owner')).stdout)['claimed']
    writer=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    writer.stdin.write('BEGIN; '+mark_sql(project,revision,'owner')+"; SELECT 'stage-lock-held'; SELECT pg_sleep(0.35); COMMIT;\n")
    writer.stdin.close()
    while True:
        line=writer.stdout.readline()
        assert line, 'writer failed before holding the stage lock'
        if line.strip()=='stage-lock-held':
            break
    waiting=json.loads(query(claim_sql(project,revision,'waiter')).stdout)
    assert waiting['claimed'] is False
    assert writer.wait(timeout=5)==0,writer.stderr.read()


def test_source_day_filter_precedes_limit_and_preserves_project_boundary(pg):
    _,query,project=pg
    for index in range(4):
        add_snapshot(pg,session=f'old-{index}',day='2026-09-06')
    add_snapshot(pg,session='wanted',day='2026-09-07')
    add_snapshot(pg,session='next-day',day='2026-09-08')
    result=query(f"BEGIN; SET LOCAL ROLE ace_processor; SELECT session_id FROM ace.pending_snapshot_refs_window(1,'analysis','{project}','2026-09-07T00:00:00Z','2026-09-08T00:00:00Z'); COMMIT;")
    assert result.stdout.strip()=='wanted'
