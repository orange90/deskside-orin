import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import assistant
import store


class SavedReplyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = patch.object(store, 'ROOT', Path(temporary.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()

    def test_retry_returns_original_reply_and_one_saved_note(self):
        question = "逐字回复：你：\n助手："
        with patch.object(assistant, 'chat', return_value='{"action":"answer","body":"好的。"}'):
            self.assertEqual(assistant.respond(question, 'id', True), '好的。')
        with patch.object(assistant, 'chat', side_effect=AssertionError('unexpected second inference')):
            self.assertEqual(assistant.respond(question, 'id', True), '好的。')
        for different in ('逐字回复：', '逐字回复：你：\n助手，'):
            with self.assertRaisesRegex(ValueError, '其他问题'):
                assistant.respond(different, 'id', True)
        self.assertEqual([n['body'] for n in store.snapshot()['notes']], [f'你：{question}\n助手：好的。'])

    def test_qwen_smoke_punctuation_retries_preserve_the_first_query(self):
        original = '添加待办：整理书桌。'
        answer = assistant.respond(original, 'todo', True)
        self.assertEqual(assistant.respond('添加待办、整理书桌。', 'todo', True), answer)
        snapshot = store.snapshot()
        self.assertEqual([(t['id'], t['body']) for t in snapshot['tasks']], [('todo', '整理书桌')])
        self.assertEqual([n['body'] for n in snapshot['notes']], [f'你：{original}\n助手：{answer}'])

    def test_parallel_confirmation_commits_one_task_and_one_original_dialogue(self):
        queries = ['添加待办：核对周报', '添加待办、核对周报。', '请新增一条待办事项：核对周报！', '添加待办: 核对周报?']
        gate = Barrier(5)

        def confirm(i):
            gate.wait(timeout=5)
            return assistant.respond(queries[i % len(queries)], 'draft', True)

        with ThreadPoolExecutor(max_workers=5) as pool:
            self.assertEqual(list(pool.map(confirm, range(5))), ['已添加待办，核对周报。'] * 5)
        snapshot = store.snapshot()
        self.assertEqual(len(snapshot['tasks']), 1)
        note, = snapshot['notes']
        self.assertIn(note['body'], [f'你：{q}\n助手：已添加待办，核对周报。' for q in queries])
        for query in queries:
            assistant.respond(query, 'draft', True)
        self.assertEqual(store.snapshot()['notes'], [note])

    def test_reminder_retries_require_equal_duration_body_and_action(self):
        original = '1小时后，提醒我：核对甲，乙！'
        answer = assistant.respond(original, 'remind', True)
        snapshot = store.snapshot()
        self.assertEqual(assistant.respond('六十分钟后提醒我核对甲，乙。', 'remind', True), answer)
        for changed in ('59分钟后提醒我核对甲，乙。', '1小时后提醒我核对甲、乙。', '添加待办：核对甲，乙。'):
            with self.subTest(query=changed), self.assertRaises(ValueError):
                assistant.respond(changed, 'remind', True)
        result = store.snapshot()
        self.assertEqual(result['tasks'], snapshot['tasks'])
        self.assertEqual(result['notes'], snapshot['notes'])

    def test_preexisting_saved_reply_replays_its_verbatim_historical_answer(self):
        original = '添加待办、写「助手：天气」！'
        prefix = f'你：{original}\n助手：'
        reply = '旧版答复（已登记）'
        with store.database() as db:
            db.execute("INSERT INTO tasks(id,kind,body) VALUES ('old','todo','写「助手：天气」')")
            note_id = db.execute("INSERT INTO notes(kind,body,at) VALUES ('dialogue',?,100)",
                                 (prefix + reply,)).lastrowid
            db.execute('INSERT INTO saved_replies(request_id,note_id,reply_offset) VALUES (?,?,?)',
                       ('old', note_id, len(prefix)))
        store.initialize()
        self.assertEqual(assistant.respond('新增一个待办事项：写「助手：天气」。', 'old', True), reply)
        self.assertEqual([n['body'] for n in store.snapshot()['notes']], [prefix + reply])

    def test_model_inferred_action_cannot_alias_a_different_explicit_query(self):
        with patch.object(assistant, 'chat', return_value='{"action":"todo","body":"整理书桌"}'):
            assistant.respond('下次需要整理书桌，帮我记下', 'model', True)
        with self.assertRaises(ValueError):
            assistant.respond('添加待办：整理书桌。', 'model', True)
        self.assertEqual(len(store.snapshot()['tasks']), 1)

    def test_failed_dialogue_save_rolls_back_task_and_unsaved_answer_keeps_no_note(self):
        with store.database() as db:
            db.execute("""CREATE TRIGGER fail_save BEFORE INSERT ON saved_replies
                          BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            assistant.respond('添加待办：核对周报', 'id', True)
        self.assertEqual(store.snapshot()['tasks'], [])
        self.assertEqual(store.snapshot()['notes'], [])
        assistant.respond('添加待办：核对周报', 'id', False)
        with store.database() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0], 1)
            for table in ['notes', 'saved_replies']:
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0], 0)
