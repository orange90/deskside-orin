import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import assistant
import store


class IdeaCaptureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        no_inference = patch.object(assistant, 'chat', side_effect=AssertionError('capture invoked the model'))
        no_inference.start()
        self.addCleanup(no_inference.stop)

    def notes(self, kind):
        with store.database() as db:
            return [r['body'] for r in db.execute('SELECT body FROM notes WHERE kind=? ORDER BY id', (kind,))]

    def test_save_preference_preserves_the_idea_and_first_opted_in_query(self):
        body = '核对 [[工位助手|项目笔记]]。\n\n#选题\n- 保留句尾！'
        first, second = '灵感：' + body, '请记下灵感，' + body
        answer = assistant.respond(first, 'capture', False)
        self.assertEqual(answer, '已记下灵感。')
        self.assertEqual(self.notes('idea'), [body])
        self.assertEqual(self.notes('dialogue'), [])
        self.assertEqual(store.snapshot()['tasks'], [])
        store.initialize()
        for query, save in [(second, True), (first, True), (second, False)]:
            self.assertEqual(assistant.respond(query, 'capture', save), answer)
        self.assertEqual(self.notes('idea'), [body])
        self.assertEqual(self.notes('dialogue'), [f'你：{second}\n助手：{answer}'])
        for edited in (body[:-1], body.replace('选题', '构想'), body.replace('\n\n', '\n')):
            with self.subTest(edited=edited), self.assertRaises(ValueError):
                assistant.respond('记个灵感：' + edited, 'capture', False)
        self.assertEqual(self.notes('idea'), [body])

    def test_transcribed_voice_separators_keep_punctuation_in_the_idea(self):
        self.assertEqual(assistant.idea_body('记下灵感。文章可以从一把空椅子开始。'), '文章可以从一把空椅子开始。')
        self.assertEqual(assistant.idea_body('记个灵感!下次写文章先描述一个具体的场景。'), '下次写文章先描述一个具体的场景。')
        self.assertEqual(assistant.idea_body('记个灵感！！原话'), '！原话')
        self.assertIsNone(assistant.idea_body('灵感!只是提到了这个词'))

    def test_mixed_concurrent_retries_commit_one_idea_and_only_one_opted_in_dialogue(self):
        body = '#观察 读完 [[旧周记]]。'
        requests = [('灵感：' + body, False), ('记个灵感：' + body, True),
                    ('请记下灵感，' + body, False), ('灵感: ' + body, True)]
        gate = Barrier(len(requests))

        def capture(args):
            query, save = args
            gate.wait(timeout=5)
            return assistant.respond(query, 'retry-id', save)

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            self.assertEqual(list(pool.map(capture, requests)), ['已记下灵感。'] * len(requests))
        self.assertEqual(self.notes('idea'), [body])
        saved, = self.notes('dialogue')
        self.assertIn(saved, [f'你：{q}\n助手：已记下灵感。' for q, enabled in requests if enabled])

    def test_empty_commands_and_requests_for_inspiration_do_not_create_ideas(self):
        for query in ('记个灵感', '灵感： \n'):
            self.assertIn('写下', assistant.respond(query, 'empty', True))
        with patch.object(assistant, 'chat', return_value='{"action":"answer","body":"看看旧笔记。"}') as model:
            self.assertEqual(assistant.respond('给我几个灵感', 'general', False), '看看旧笔记。')
            model.assert_called_once()
        with patch.object(assistant, 'chat', return_value='{"action":"idea","body":"猜出来的收藏"}'):
            with self.assertRaises(ValueError):
                assistant.respond('今天写什么好？', 'model-attempt', False)
        self.assertEqual(self.notes('idea'), [])
        self.assertEqual(self.notes('dialogue'), [])

    def test_failed_dialogue_commit_rolls_back_the_idea_and_receipt(self):
        with store.database() as db:
            db.execute("""CREATE TRIGGER fail_receipt BEFORE INSERT ON saved_replies
                          BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
        query = '灵感：用一张图解释计时口径。'
        with self.assertRaises(sqlite3.IntegrityError):
            assistant.respond(query, 'retry', True)
        with store.database() as db:
            for table in ('notes', 'idea_requests', 'saved_replies'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0)
            db.execute('DROP TRIGGER fail_receipt')
        answer = assistant.respond(query, 'retry', True)
        self.assertEqual(self.notes('idea'), ['用一张图解释计时口径。'])
        self.assertEqual(self.notes('dialogue'), [f'你：{query}\n助手：{answer}'])

    def test_legacy_notes_and_other_request_ids_are_not_reassigned(self):
        body = '同一句旧想法。'
        query = '灵感：' + body
        store.note('idea', body)  # Historical note has no request ID.
        store.task('used-task', 'todo', '原有待办')
        with patch.object(assistant, 'chat', return_value='{"action":"answer","body":"旧答复"}'):
            assistant.respond('此前的问题', 'used-dialogue', True)
        for request_id in ('used-task', 'used-dialogue'):
            for save in (False, True):
                with self.subTest(request_id=request_id, save=save), self.assertRaises(ValueError):
                    assistant.respond(query, request_id, save)
        self.assertEqual(self.notes('idea'), [body])
        assistant.respond(query, 'unused', False)
        self.assertEqual(self.notes('idea'), [body, body])
        for save in (False, True):
            with self.assertRaises(ValueError):
                assistant.respond('添加待办：误占编号', 'unused', save)
        with patch.object(assistant, 'chat', return_value='{"action":"answer","body":"新答复"}'):
            with self.assertRaises(ValueError):
                assistant.respond('另一个问题', 'unused', True)
        self.assertEqual(self.notes('dialogue'), ['你：此前的问题\n助手：旧答复'])
        self.assertEqual([t['id'] for t in store.snapshot()['tasks']], ['used-task'])
