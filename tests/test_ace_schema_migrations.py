from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ace_schema


class Store:
    def __init__(self, versions=()):
        self.versions=set(versions)
        self.executed=[]

    def _rows(self, sql):
        if 'FROM pg_tables' in sql:
            return [{'tables':14 if self.versions else 0,'rls_enabled':bool(self.versions)}]
        if sql.startswith('SELECT version'):
            return [{'version':v} for v in sorted(self.versions)]
        self.executed.append(sql)
        if sql.startswith('APPLY '):
            self.versions.add(int(sql.split()[1]))
        return []


def test_existing_schema_applies_only_missing_migrations(tmp_path,monkeypatch):
    (tmp_path/'001_base.sql').write_text('APPLY 1')
    (tmp_path/'002_leases.sql').write_text('APPLY 2')
    store=Store([1])
    backups=[]
    monkeypatch.setattr(ace_schema,'backup',lambda store:backups.append(True) or {'fixture':True})
    assert [row['version'] for row in ace_schema.migration_plan(store,tmp_path)['pending']]==[2]
    assert store.executed==[]
    result=ace_schema.apply_migrations(store,tmp_path)
    assert result['applied_versions']==[2]
    assert store.executed==['APPLY 2']
    assert backups==[True]
    assert ace_schema.apply_migrations(store,tmp_path)['applied'] is False
    assert backups==[True]
    assert store.executed==['APPLY 2']


def test_unrecorded_migration_cannot_report_success(tmp_path,monkeypatch):
    (tmp_path/'002_broken.sql').write_text('does not record success')
    monkeypatch.setattr(ace_schema,'backup',lambda store:None)
    with pytest.raises(RuntimeError,match='did not record'):
        ace_schema.apply_migrations(Store([1]),tmp_path)


def test_duplicate_versions_are_rejected_before_mutation(tmp_path):
    (tmp_path/'002_first.sql').write_text('APPLY 2')
    (tmp_path/'002_second.sql').write_text('APPLY 2')
    store=Store([1])
    with pytest.raises(ValueError,match='duplicate'):
        ace_schema.migration_plan(store,tmp_path)
    assert store.executed==[]
