import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import focus_sessions
import reminder_queue
import store


class ReminderQueueTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        self.now = 1795360010.125
        self.due = self.now + 60
        self.legacy = '旧 / 17'
        with store.database() as db:
            db.execute('''CREATE TABLE tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                due REAL, fired INTEGER NOT NULL DEFAULT 0)''')
            db.execute('''CREATE TABLE notes (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL)''')
            db.execute("INSERT INTO tasks VALUES (?, 'reminder', '复核草稿', ?, 0)", (self.legacy, self.due))
            db.execute("INSERT INTO tasks VALUES ('old-cancelled', 'reminder', '旧取消正文', ?, 2)", (self.due,))
        store.initialize()

    def delivered(self):
        with store.database() as db:
            return [row['body'] for row in db.execute("SELECT body FROM notes WHERE kind='reminder' ORDER BY id")]

    def test_pending_pages_ignore_other_records_and_keep_tied_deadlines_in_order(self):
        tied = [f'tie-{n:03}' for n in reversed(range(44))]
        with store.database() as db:
            db.executemany('INSERT INTO tasks VALUES (?,?,?,?,?)',
                           [(i, 'reminder', '同时到期', self.due, 0) for i in tied] + [
                               ('soon', 'reminder', '快到期', self.now + 1, 0),
                               ('later', 'reminder', '晚一点', self.due + 1, 0),
                               ('expired', 'reminder', '到期未补发', self.now, 0),
                               ('sent', 'reminder', '已发正文', self.due, 1),
                               ('todo', 'todo', '私有待办', self.due, 0),
                               ('unscheduled', 'reminder', '未定时', None, 0)])
            db.execute("INSERT INTO notes(kind,body,at) VALUES('reminder','私有通知历史',?)", (self.now,))
            db.execute("INSERT INTO notes(kind,body,at) VALUES('dialogue','私有对话',?)", (self.now,))
        focus_sessions.start(str(uuid4()), 1, title='私有专注标题', now=self.now)
        first = reminder_queue.list_pending(now=self.now)
        self.assertEqual(set(first), {'items', 'next_cursor', 'server_now'})
        self.assertEqual(first['server_now'], self.now)
        self.assertEqual(len(first['items']), 20)
        self.assertTrue(all(set(item) == {'id', 'body', 'due'} for item in first['items']))
        items = list(first['items'])
        cursor = first['next_cursor']
        # A previously returned row can disappear while its cursor is still in the client.
        anchor = items[-1]
        reminder_queue.change(anchor['id'], str(uuid4()), 'cancel', anchor['due'], now=self.now)
        while cursor is not None:
            page = reminder_queue.list_pending(cursor, now=self.now)
            items.extend(page['items'])
            self.assertNotEqual(cursor, page['next_cursor'])
            cursor = page['next_cursor']
        self.assertEqual([row['id'] for row in items], ['soon', *sorted([self.legacy, *tied]), 'later'])
        text = json.dumps(items, ensure_ascii=False)
        for private in ('旧取消正文', '到期未补发', '已发正文', '私有待办', '未定时', '私有通知历史', '私有对话', '私有专注标题'):
            self.assertNotIn(private, text)
        self.assertEqual(self.delivered(), ['私有通知历史'])
        for bad_cursor in ('***', ' ', 'A' * 4096):
            with self.subTest(cursor=bad_cursor[:10]), self.assertRaises(ValueError):
                reminder_queue.list_pending(bad_cursor, now=self.now)

    def test_snooze_retries_add_exactly_ten_minutes_and_compare_previous_deadline(self):
        key = str(uuid4())
        gate = Barrier(3)

        def retry():
            gate.wait(timeout=5)
            return reminder_queue.change(self.legacy, key, 'snooze', self.due, now=self.now + 17)

        with ThreadPoolExecutor(max_workers=3) as pool:
            work = [pool.submit(retry) for _ in range(3)]
            results = [f.result(timeout=10) for f in work]
        new_due = self.due + 600
        expected = {'id': self.legacy, 'action': 'snooze', 'due': new_due}
        self.assertEqual(results, [expected] * 3)
        for action_key, action, old_due in [(str(uuid4()), 'cancel', self.due),
                                            (key, 'cancel', self.due), (key, 'snooze', new_due)]:
            with self.assertRaises(reminder_queue.ReminderConflict):
                reminder_queue.change(self.legacy, action_key, action, old_due, now=self.now + 18)
        store.initialize()
        store.record_due_reminders(now=self.due)
        self.assertEqual(self.delivered(), [])
        self.assertEqual(reminder_queue.list_pending(now=self.due)['items'][0]['due'], new_due)
        store.record_due_reminders(now=new_due)
        store.record_due_reminders(now=new_due + 1)
        self.assertEqual(self.delivered(), ['复核草稿'])
        self.assertEqual(reminder_queue.change(self.legacy, key, 'snooze', self.due, now=new_due + 1), expected)
        self.assertEqual(reminder_queue.list_pending(now=new_due + 1)['items'], [])

    def test_cancel_survives_restart_and_creation_retry_preserving_focus_and_other_reminders(self):
        key = str(uuid4())
        focus_sessions.start(str(uuid4()), 1, title='读书', now=self.now)
        with store.database() as db:
            db.execute("INSERT INTO tasks VALUES('other','reminder','正常提醒',?,0)", (self.due,))
        cancelled = reminder_queue.change(self.legacy, key, 'cancel', self.due, now=self.now)
        store.initialize()
        store.task(self.legacy, 'reminder', '复核草稿', minutes=1)
        with self.assertRaises(reminder_queue.ReminderConflict):
            reminder_queue.change(self.legacy, str(uuid4()), 'snooze', self.due, now=self.now)
        store.record_due_reminders(now=self.due + 1)
        self.assertCountEqual(self.delivered(), ['正常提醒', '专注计时结束：读书（1 分钟）'])
        self.assertEqual(reminder_queue.change(self.legacy, key, 'cancel', self.due, now=self.due + 10), cancelled)
        with store.database() as db:
            self.assertEqual(tuple(db.execute("SELECT due,fired FROM tasks WHERE id=?", (self.legacy,)).fetchone()),
                             (self.due, 2))

    def test_expired_and_fired_reminders_reject_new_actions(self):
        with store.database() as db:
            db.execute("INSERT INTO tasks VALUES ('todo','todo','普通待办',NULL,0)")
        for at in (self.due, self.due + 1):
            with self.assertRaises(reminder_queue.ReminderConflict):
                reminder_queue.change(self.legacy, str(uuid4()), 'cancel', self.due, now=at)
        self.assertEqual(self.delivered(), [])
        store.record_due_reminders(now=self.due + 1)
        for task_id in (self.legacy, 'todo', 'old-cancelled'):
            with self.subTest(task_id=task_id), self.assertRaises(reminder_queue.ReminderConflict):
                reminder_queue.change(task_id, str(uuid4()), 'cancel', self.due, now=self.now)
        with self.assertRaises(KeyError):
            reminder_queue.change('missing', str(uuid4()), 'cancel', self.due, now=self.now)
        with self.assertRaises(ValueError):
            reminder_queue.change(self.legacy, 'bad-uuid', 'snooze', self.due, now=self.now)
        for action, expected_due in [('delete', self.due), ('cancel', float('nan')), ('snooze', float('inf'))]:
            with self.assertRaises(ValueError):
                reminder_queue.change(self.legacy, str(uuid4()), action, expected_due, now=self.now)
        self.assertEqual(self.delivered(), ['复核草稿'])

    def test_failure_to_save_request_does_not_leave_the_due_time_changed(self):
        key = str(uuid4())
        with store.database() as db:
            db.execute("""CREATE TRIGGER fail_request BEFORE INSERT ON reminder_changes
                          BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            reminder_queue.change(self.legacy, key, 'snooze', self.due, now=self.now)
        self.assertEqual(reminder_queue.list_pending(now=self.now)['items'][0]['due'], self.due)
        with store.database() as db:
            db.execute('DROP TRIGGER fail_request')
        changed = reminder_queue.change(self.legacy, key, 'snooze', self.due, now=self.now)
        self.assertEqual(changed['due'], self.due + 600)

    def test_scheduler_and_queue_change_do_not_both_win_the_deadline(self):
        for action in ('snooze', 'cancel'):
            with self.subTest(action=action):
                due = self.now + 2
                with store.database() as db:
                    db.execute("INSERT INTO tasks VALUES (?, 'reminder', ?, ?, 0)", (action, action, due))
                gate = Barrier(2)

                def deliver():
                    gate.wait(timeout=5)
                    store.record_due_reminders(now=due)

                def manage():
                    gate.wait(timeout=5)
                    try:
                        reminder_queue.change(action, str(uuid4()), action, due, now=due - 0.25)
                    except reminder_queue.ReminderConflict:
                        return False
                    return True

                with ThreadPoolExecutor(max_workers=2) as pool:
                    delivery, change = pool.submit(deliver), pool.submit(manage)
                    delivery.result(timeout=10)
                    changed = change.result(timeout=10)
                self.assertEqual(self.delivered().count(action), 0 if changed else 1)
                store.record_due_reminders(now=due + 600)
                expected = 0 if changed and action == 'cancel' else 1
                self.assertEqual(self.delivered().count(action), expected)
