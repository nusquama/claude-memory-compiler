import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ace_collect as collect


class CollectionTests(unittest.TestCase):
    def fixture(self, root):
        vault = root / 'vault'
        (vault / 'Conversations/knowledge').mkdir(parents=True)
        source_cwd = root / 'initialized-source'
        source_cwd.mkdir()
        source = root / 'claude/session.jsonl'
        source.parent.mkdir()
        source.write_text(json.dumps({'sessionId': 'session', 'cwd': str(source_cwd), 'message': {
            'role': 'user', 'content': 'Keep this useful decision'}}) + '\n')
        os.utime(source, (time.time()-300, time.time()-300))
        args = argparse.Namespace(state_file=str(root / 'state.json'), codex_root=str(root/'codex'),
            claude_root=str(source.parent), ccs_root=str(root/'ccs'), days=7, limit=2,
            stable_seconds=0, max_chars=500000, dry_run=False,
            path=None, source=None, compile=False)
        route = collect.ProjectRoute(
            source_project='fixture-project',
            source_cwd=str(source_cwd),
            destination_project='Conversations',
            destination_dir=vault / 'Conversations',
            used_fallback=False,
            reason='fixture_initialized',
        )
        return vault, source, args, route

    def test_dry_run_does_not_create_state_or_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault, source, args, route = self.fixture(Path(tmp))
            args.dry_run = True
            with patch.object(collect, 'VAULT_ROOT', vault), patch.object(
                collect, 'project_route', return_value=route
            ), patch.object(collect, 'run_flush') as model:
                result = collect.collect(args)
            self.assertEqual(result['calls'], 1)
            model.assert_not_called()
            self.assertFalse(Path(args.state_file).exists())
            self.assertFalse((vault/'Conversations/daily').exists())

    def test_failure_retry_then_hash_dedup_then_changed_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault, source, args, route = self.fixture(Path(tmp))
            with patch.object(collect, 'VAULT_ROOT', vault), patch.object(
                collect, 'project_route', return_value=route
            ), patch.object(collect, 'run_flush',
                new=AsyncMock(side_effect=[('FLUSH_ERROR timeout', []), ('useful decision', []),
                                          ('updated decision', [])])) as model:
                self.assertEqual(collect.collect(args)['failed'], 1)
                self.assertEqual(collect.collect(args)['ingested'], 1)
                self.assertEqual(collect.collect(args)['unchanged'], 1)
                with source.open('a') as out:
                    out.write(json.dumps({'message': {'role':'assistant','content':'New outcome'}})+'\n')
                self.assertEqual(collect.collect(args)['ingested'], 1)
                self.assertEqual(model.await_count, 3)
            daily = next((vault/'Conversations/daily').glob('*.md')).read_text()
            self.assertEqual(daily.count('<!-- ace-claude-session:'), 1)
            self.assertIn('updated decision', daily)
            self.assertNotIn('useful decision', daily)

    def test_limit_retains_backlog(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault, source, args, route = self.fixture(Path(tmp))
            second = source.with_name('second.jsonl')
            second.write_text(json.dumps({'sessionId':'second','message':{'role':'user','content':'Another decision'}})+'\n')
            args.limit = 1
            with patch.object(collect, 'VAULT_ROOT', vault), patch.object(
                collect, 'project_route', return_value=route
            ), patch.object(collect, 'run_flush', new=AsyncMock(return_value=('record',[]))):
                self.assertEqual(collect.collect(args)['unexamined'], 1)
                os.utime(second, (time.time()-10*86400, time.time()-10*86400))
                self.assertEqual(collect.collect(args)['ingested'], 1)

    def test_fair_order_retries_pending_between_fresh_sessions_and_reports_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / 'vault'
            (vault / 'Conversations/knowledge').mkdir(parents=True)
            claude_root = root / 'claude'
            claude_root.mkdir()

            def write_session(name, session_id, content, age):
                path = claude_root / name
                path.write_text(json.dumps({'sessionId': session_id, 'message': {
                    'role': 'user', 'content': content}}) + '\n')
                stamp = time.time() - age
                os.utime(path, (stamp, stamp))
                return path

            fresh = write_session('fresh.jsonl', 'fresh', 'new evidence', 120)
            second_fresh = write_session('second-fresh.jsonl', 'second', 'second evidence', 180)
            old_failed = write_session('old-failed.jsonl', 'old', 'retry evidence', 10 * 86400)
            state_path = root / 'state.json'
            state_path.write_text(json.dumps({
                'version': 1,
                'sessions': {
                    'claude:old': {
                        'source': 'claude', 'path': str(old_failed),
                        'status': 'failed', 'source_mtime': old_failed.stat().st_mtime,
                    }
                },
                'backlog': [],
            }))
            args = argparse.Namespace(
                state_file=str(state_path), codex_root=str(root / 'codex'),
                claude_root=str(claude_root), ccs_root=str(root / 'ccs'), days=7,
                limit=2, stable_seconds=0, max_chars=500000, dry_run=False,
                path=None, source=None, compile=False,
            )
            route = collect.ProjectRoute(
                source_project='fixture-project',
                source_cwd=str(root / 'initialized-source'),
                destination_project='Conversations',
                destination_dir=vault / 'Conversations',
                used_fallback=False,
                reason='fixture_initialized',
            )
            with patch.object(collect, 'VAULT_ROOT', vault), patch.object(
                collect, 'project_route', return_value=route
            ), patch.object(collect, 'run_flush', new=AsyncMock(side_effect=[('fresh result', []), ('retry result', [])])
            ) as model:
                result = collect.collect(args)

            self.assertEqual(result['ingested'], 2)
            contexts = [call.args[0] for call in model.await_args_list]
            self.assertIn('fresh', contexts[0])
            self.assertIn('old-failed.jsonl', contexts[1])
            saved = json.loads(state_path.read_text())
            self.assertEqual(saved['coverage']['pending_count'], 1)
            self.assertIsNotNone(saved['coverage']['pending_oldest_mtime'])
            self.assertGreater(saved['coverage']['pending_oldest_age_seconds'], 0)
            self.assertIsNotNone(saved['coverage']['freshest_candidate_mtime'])
            self.assertEqual(saved['backlog'][0]['path'], str(second_fresh))


if __name__ == '__main__':
    unittest.main()
