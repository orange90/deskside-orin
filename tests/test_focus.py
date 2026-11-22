import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import store
import focus_sessions as focus


class FocusTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        self.start = datetime(2026, 11, 22, 23, 59, 45, tzinfo=store.TZ).timestamp()
        with store.database() as legacy:
            legacy.execute('''CREATE TABLE tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                due REAL, fired INTEGER NOT NULL DEFAULT 0)''')
            legacy.execute('''CREATE TABLE notes (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL)''')
            legacy.execute("INSERT INTO tasks VALUES ('旧 / todo', 'todo', '整理笔记', NULL, 0)")
        store.initialize()

    def reminders(self):
        with store.database() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, body, at FROM notes WHERE kind='reminder' ORDER BY id")]

    def test_legacy_todo_can_be_timed_idempotently_without_completing_it(self):
        key = str(uuid4())
        first = focus.start(key, 25, task_id='旧 / todo', now=self.start)
        self.assertEqual(first['title'], '整理笔记')
        self.assertEqual(first['due_at'], self.start + 25 * 60)
        self.assertEqual(first, focus.start(key, 25, task_id='旧 / todo', now=self.start + 5))
        with self.assertRaises(focus.FocusConflict):
            focus.start(key, 26, task_id='旧 / todo', now=self.start + 5)
        stopped = focus.finish(key, now=self.start + 10)
        self.assertEqual((stopped['state'], stopped['ended_at']), ('stopped', self.start + 10))
        store.initialize()
        self.assertEqual(focus.start(key, 25, task_id='旧 / todo', now=self.start + 30), stopped)
        self.assertIsNone(focus.snapshot(now=self.start + 30)['active'])
        self.assertEqual(store.list_todos()['items'],
                         [{'id': '旧 / todo', 'body': '整理笔记', 'completed': False}])
        self.assertEqual(self.reminders(), [])

    def test_concurrent_different_requests_only_start_one_timer(self):
        n = 3
        gate = Barrier(n)

        def attempt(key):
            gate.wait(timeout=5)
            try:
                return focus.start(key, 1, title='写周记', now=self.start)
            except focus.FocusConflict:
                return None

        keys = [str(uuid4()) for _ in range(n)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            winners = [entry for entry in pool.map(attempt, keys) if entry is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0]['title'], '写周记')
        self.assertEqual(focus.snapshot(now=self.start + 1)['active'], winners[0])

    def test_unattended_timer_and_regular_reminder_fire_once_after_restart(self):
        key = str(uuid4())
        focus.start(key, 1, title='读书', now=self.start)
        due = self.start + 60  # The next local date.
        with store.database() as db:
            db.execute("INSERT INTO tasks VALUES ('scheduled','reminder','既有的定时提醒',?,0)", (due + 2,))
        store.record_due_reminders(now=due - 1)
        self.assertEqual(self.reminders(), [])
        store.initialize()
        delivery = due + 300
        gate = Barrier(2)

        def wake_scheduler():
            gate.wait(timeout=5)
            store.record_due_reminders(now=delivery)

        with ThreadPoolExecutor(max_workers=2) as pool:
            work = [pool.submit(wake_scheduler) for _ in range(2)]
            for result in work:
                result.result(timeout=10)
        notes = self.reminders()
        self.assertCountEqual([n['body'] for n in notes], ['既有的定时提醒', '专注计时结束：读书（1 分钟）'])
        self.assertEqual([n['at'] for n in notes], [delivery, delivery])
        snapshot = focus.snapshot(now=delivery)
        self.assertIsNone(snapshot['active'])
        self.assertEqual(len(snapshot['recent']), 1)
        finished = snapshot['recent'][0]
        self.assertEqual((finished['state'], finished['ended_at']), ('finished', due))
        self.assertEqual(focus.start(key, 1, title='读书', now=self.start - 15), finished)
        self.assertIsNone(focus.snapshot(now=self.start - 15)['active'])
        self.assertEqual(focus.finish(key, now=self.start), finished)
        store.initialize()
        store.record_due_reminders(now=delivery + 1)
        self.assertEqual(self.reminders(), notes)

    def test_early_stop_clamps_backwards_time_and_old_stop_cannot_cancel_new_timer(self):
        first, second = str(uuid4()), str(uuid4())
        focus.start(first, 1, now=self.start)
        stopped = focus.finish(first, now=self.start - 100)
        self.assertEqual(stopped['ended_at'], self.start)
        self.assertEqual(stopped['state'], 'stopped')
        new = focus.start(second, 180, now=self.start + 10)
        self.assertEqual(new['title'], '专注')
        self.assertEqual(focus.finish(first, now=self.start + 20), stopped)
        with self.assertRaises(KeyError):
            focus.finish(str(uuid4()), now=self.start + 20)
        store.record_due_reminders(now=self.start + 61)
        self.assertEqual(focus.snapshot(now=self.start + 61)['active'], new)
        self.assertEqual(self.reminders(), [])

    def test_rejects_invalid_duration_title_id_and_unavailable_todos(self):
        for minutes in (0, 181, True, 1.5):
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                focus.start(str(uuid4()), minutes, now=self.start)
        with self.assertRaises(ValueError):
            focus.start('not-a-uuid', 1, now=self.start)
        with self.assertRaises(ValueError):
            focus.start(str(uuid4()), 1, title='长' * 121, now=self.start)
        store.set_todo_completed('旧 / todo', True)
        store.task('timer', 'reminder', '休息', minutes=5)
        for task_id in ('旧 / todo', 'timer'):
            with self.subTest(task_id=task_id), self.assertRaises(focus.FocusConflict):
                focus.start(str(uuid4()), 1, task_id=task_id, now=self.start)
        with self.assertRaises(KeyError):
            focus.start(str(uuid4()), 1, task_id='missing', now=self.start)
        result = focus.snapshot(now=self.start)
        self.assertIsNone(result['active'])
        self.assertEqual(result['recent'], [])

    def test_notification_write_failure_is_recoverable(self):
        key = str(uuid4())
        focus.start(key, 1, title='读书', now=self.start)
        with store.database() as db:
            db.execute("""CREATE TRIGGER fail_notification BEFORE INSERT ON notes
                          WHEN NEW.kind='reminder'
                          BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            store.record_due_reminders(now=self.start + 60)
        self.assertEqual(self.reminders(), [])
        with store.database() as db:
            db.execute('DROP TRIGGER fail_notification')
        store.record_due_reminders(now=self.start + 65)
        self.assertEqual(len(self.reminders()), 1)
        self.assertEqual(focus.snapshot(now=self.start + 65)['recent'][0]['ended_at'], self.start + 60)
