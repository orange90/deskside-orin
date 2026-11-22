import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
        with self.assertRaisesRegex(ValueError, '其他问题'):
            assistant.respond('逐字回复：', 'id', True)
        self.assertEqual([n['body'] for n in store.snapshot()['notes']], [f'你：{question}\n助手：好的。'])

    def test_parallel_confirmation_commits_one_task_and_one_dialogue(self):
        def confirm(_):
            return assistant.respond('添加待办：核对周报', 'draft', True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            self.assertEqual(list(pool.map(confirm, range(5))), ['已添加待办，核对周报。'] * 5)
        snapshot = store.snapshot()
        self.assertEqual(len(snapshot['tasks']), 1)
        self.assertEqual(len(snapshot['notes']), 1)

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
