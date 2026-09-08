import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import query
import codex_runner


class OutcomeTests(unittest.TestCase):
    def test_runner_passes_explicit_output_schema(self):
        captured = {}
        class Process:
            returncode = 0
            async def communicate(self, prompt):
                return b'{"ok":true}', b''
        async def spawn(*command, **kwargs):
            import json
            captured['schema'] = json.loads(Path(command[command.index('--output-schema')+1]).read_text())
            captured['sandbox'] = command[command.index('--sandbox')+1]
            return Process()
        schema = {'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(codex_runner.asyncio, 'create_subprocess_exec', side_effect=spawn):
            answer, _ = asyncio.run(codex_runner.run_codex('synthetic', cwd=Path(temp), output_schema=schema))
        self.assertEqual(captured['schema'], schema)
        self.assertEqual(captured['sandbox'], 'read-only')
        self.assertEqual(answer, '{"ok":true}')

    def test_query_failure_does_not_record_success(self):
        with patch.object(query, 'read_all_wiki_content', return_value='synthetic knowledge'), \
             patch.object(query, 'run_codex', new=AsyncMock(side_effect=RuntimeError('failure'))), \
             patch.object(query, 'save_state') as save:
            with self.assertRaises(RuntimeError):
                asyncio.run(query.run_query('synthetic question'))
            save.assert_not_called()

    def test_file_back_requires_changed_article(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(query, 'QA_DIR', Path(temp)), \
             patch.object(query, 'KNOWLEDGE_DIR', Path(temp)), \
             patch.object(query, 'read_all_wiki_content', return_value='synthetic'), \
             patch.object(query, 'run_codex', new=AsyncMock(return_value=('done', []))), \
             patch.object(query, 'save_state') as save:
            with self.assertRaisesRegex(RuntimeError, 'verified Q&A'):
                asyncio.run(query.run_query('question', file_back=True))
            save.assert_not_called()

    def test_query_context_is_index_guided_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            knowledge = Path(temp) / 'knowledge'
            concepts = knowledge / 'concepts'
            concepts.mkdir(parents=True)
            entries = []
            for index in range(12):
                slug = f'article-{index:02d}'
                entries.append(f'* [Article {index}](/concepts/{slug}.md) - topic {index}')
                (concepts / f'{slug}.md').write_text(
                    f'# Article {index}\n\nunique-topic-{index} details\n',
                    encoding='utf-8',
                )
            (knowledge / 'index.md').write_text(
                '# Knowledge Base Index\n\n# Concepts\n\n' + '\n'.join(entries) + '\n',
                encoding='utf-8',
            )
            query._load_query_context_helper.cache_clear()
            with patch.object(query, 'KNOWLEDGE_DIR', knowledge):
                context = query._bounded_wiki_content('topic 3')

        selected = [line for line in context.splitlines() if line.startswith('## concepts/')]
        self.assertLessEqual(len(selected), query.MAX_QUERY_ARTICLES)
        self.assertIn('Article 11', context)  # The full index remains available.
        self.assertLessEqual(len(context), query.MAX_QUERY_CONTEXT_CHARS)

    def test_query_fallback_marks_relevance_as_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            knowledge = Path(temp) / 'knowledge'
            (knowledge / 'concepts').mkdir(parents=True)
            (knowledge / 'index.md').write_text('# Knowledge Base Index\n', encoding='utf-8')
            (knowledge / 'concepts' / 'one.md').write_text('# One\n', encoding='utf-8')
            query._load_query_context_helper.cache_clear()
            with patch.object(query, 'KNOWLEDGE_DIR', knowledge), patch.object(
                query, 'QUERY_CONTEXT_HELPER', Path(temp) / 'missing-helper.py'
            ):
                context = query._bounded_wiki_content('question')

        self.assertIn('Article relevance is not verified.', context)
        self.assertLessEqual(len(context), query.MAX_QUERY_CONTEXT_CHARS)

    def test_query_uses_canonical_context_helper_by_default(self):
        self.assertEqual(
            query.QUERY_CONTEXT_HELPER,
            Path.home() / '.agents/skills/ace/scripts/query_context.py',
        )
        self.assertTrue(query.QUERY_CONTEXT_HELPER.is_file())

    def test_query_fallback_can_select_a_relevant_ninth_article(self):
        with tempfile.TemporaryDirectory() as temp:
            knowledge = Path(temp) / 'knowledge'
            concepts = knowledge / 'concepts'
            concepts.mkdir(parents=True)
            entries = []
            for index in range(9):
                slug = f'article-{index:02d}'
                entries.append(f'* [Article {index}](concepts/{slug}.md) - generic topic')
                text = 'generic background\n' if index < 8 else 'needle-specific evidence\n'
                (concepts / f'{slug}.md').write_text(text, encoding='utf-8')
            (knowledge / 'index.md').write_text(
                '# Knowledge Base Index\n\n# Concepts\n\n' + '\n'.join(entries) + '\n',
                encoding='utf-8',
            )
            query._load_query_context_helper.cache_clear()
            with patch.object(query, 'KNOWLEDGE_DIR', knowledge), patch.object(
                query, 'QUERY_CONTEXT_HELPER', Path(temp) / 'missing-helper.py'
            ):
                context = query._bounded_wiki_content('needle')

        self.assertIn('## concepts/article-08', context)

    def test_query_without_file_back_does_not_write_state(self):
        diagnostics = codex_runner.RunDiagnostics(
            usage={'input_tokens': 4, 'output_tokens': 2},
            duration_seconds=0.25,
            calls=1,
        )
        with patch.object(query, '_bounded_wiki_content', return_value='synthetic context'), \
             patch.object(query, 'run_codex', new=AsyncMock(return_value=('answer', diagnostics))), \
             patch.object(query, 'save_state') as save:
            answer = asyncio.run(query.run_query('synthetic question'))

        self.assertEqual(answer, 'answer')
        save.assert_not_called()
        self.assertEqual(query.run_query.last_metrics, diagnostics.as_metrics())

    def test_compile_checkpoints_input_snapshot(self):
        spec = importlib.util.spec_from_file_location('cmc_compile_test', SCRIPTS/'compile.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            daily = root/'2026-09-05.md'
            daily.write_text('original evidence')
            initial = module.file_hash(daily)
            knowledge = root/'knowledge'
            concepts = knowledge/'concepts'
            concepts.mkdir(parents=True)
            schema = root/'schema.md'
            schema.write_text('synthetic schema')
            async def changing_model(*args, **kwargs):
                daily.write_text('original evidence plus later event')
                stage_concepts = kwargs['cwd'] / 'knowledge' / 'concepts'
                stage_concepts.mkdir(parents=True, exist_ok=True)
                (stage_concepts/'evidence.md').write_text(
                    '---\nsources:\n  - "daily/2026-09-05.md"\n---\n# Evidence\n',
                    encoding='utf-8',
                )
                return ('compiled original evidence', [])

            def write_build_log(log_path, *_args, **kwargs):
                target_log = kwargs.get('log_file', knowledge / 'log.md')
                target_log.write_text(
                    f'* Compile (compile | {log_path.name}) — source daily/{log_path.name}\n',
                    encoding='utf-8',
                )

            with patch.object(module, 'AGENTS_FILE', schema), \
                 patch.object(module, 'KNOWLEDGE_DIR', knowledge), \
                 patch.object(module, 'read_wiki_index', return_value=''), \
                 patch.object(module, 'list_wiki_articles',
                              side_effect=lambda: sorted(concepts.glob('*.md'))), \
                 patch.object(module, '_ensure_build_log_entry', side_effect=write_build_log), \
                 patch.object(module, 'save_state'), \
                 patch.object(module, 'run_codex', side_effect=changing_model):
                state = {}
                asyncio.run(module.compile_daily_log(daily, state))
            self.assertEqual(state['ingested'][daily.name]['hash'], initial)
            self.assertNotEqual(initial, module.file_hash(daily))


if __name__ == '__main__':
    unittest.main()
