import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import store
from assistant import respond


class TodoTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)

    def test_legacy_task_survives_upgrade_retry_restart_and_restore(self):
        task_id, body = '现场 / 测试 17', '整理草稿'
        with store.database() as db:
            db.execute('''CREATE TABLE tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                due REAL, fired INTEGER NOT NULL DEFAULT 0)''')
            db.execute('INSERT INTO tasks VALUES (?, ?, ?, ?, ?)', (task_id, 'todo', body, None, 0))
        store.initialize()
        item = {'id': task_id, 'body': body, 'completed': False}
        self.assertEqual(store.list_todos()['items'], [item])
        done = {**item, 'completed': True}
        self.assertEqual(store.set_todo_completed(task_id, True), done)
        self.assertEqual(respond('添加待办：整理草稿。', task_id, False), '已添加待办，整理草稿。')
        store.initialize()
        self.assertEqual(store.list_todos()['items'], [])
        self.assertEqual(store.list_todos(completed=True)['items'], [done])
        self.assertEqual(store.set_todo_completed(task_id, False), item)
        self.assertEqual(store.set_todo_completed(task_id, False), item)
        store.initialize()
        self.assertEqual(store.list_todos()['items'], [item])
        self.assertEqual(store.list_todos(completed=True)['items'], [])
        old_payload = store.snapshot()['tasks']
        self.assertEqual([(r['id'], r['kind'], r['body'], r['due'], r['fired']) for r in old_payload],
                         [(task_id, 'todo', body, None, 0)])

    def test_setting_the_same_state_concurrently_is_idempotent(self):
        store.initialize()
        task_id = 'todo-benchmark'
        respond('添加待办：归档。', task_id, False)
        for completed in [True, False]:
            barrier = Barrier(3)

            def set_state():
                barrier.wait(timeout=5)
                return store.set_todo_completed(task_id, completed)

            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = [pool.submit(set_state) for _ in range(3)]
                items = [future.result(timeout=10) for future in pending]
            self.assertEqual(items, [{'id': task_id, 'body': '归档', 'completed': completed}] * 3)
            self.assertEqual(store.list_todos(completed=completed)['items'], items[:1])
            self.assertEqual(store.list_todos(completed=not completed)['items'], [])

    def test_selection_does_not_fire_or_dismiss_reminders(self):
        store.initialize()
        with store.database() as db:
            db.executemany('INSERT INTO tasks(id,kind,body,due,fired) VALUES (?,?,?,?,?)', [
                ('todo', 'todo', '整理草稿', None, 0), ('due', 'reminder', '该开会了', 10, 0),
                ('future', 'reminder', '稍后开会', 1000, 0), ('old', 'reminder', '已提醒过', 5, 1),
            ])
        for task_id in ('due', 'future', 'old'):
            with self.assertRaises(store.TodoConflict):
                store.set_todo_completed(task_id, True)
        with self.assertRaises(KeyError):
            store.set_todo_completed('不存在', False)
        store.set_todo_completed('todo', True)
        with patch('store.time.time', return_value=100):
            store.record_due_reminders()
            store.record_due_reminders()
        self.assertEqual([r['id'] for r in store.list_todos(completed=True)['items']], ['todo'])
        store.set_todo_completed('todo', False)
        with store.database() as db:
            rows = db.execute("SELECT id,due,fired FROM tasks ORDER BY id").fetchall()
            self.assertEqual([tuple(row) for row in rows],
                             [('due', 10, 1), ('future', 1000, 0), ('old', 5, 1), ('todo', None, 0)])
            self.assertEqual([tuple(row) for row in db.execute('SELECT kind,body FROM notes')],
                             [('reminder', '该开会了')])

    def test_paging_filters_before_limit_and_ignores_newer_insertions(self):
        store.initialize()
        open_ids = [f'open-{i}' for i in range(45)]
        done_ids = [f'done-{i}' for i in range(53)]
        with store.database() as db:
            db.executemany('INSERT INTO tasks(id,kind,body) VALUES (?,?,?)',
                           [(i, 'todo', i) for i in open_ids + done_ids] +
                           [(f'reminder-{n}', 'reminder', 'check') for n in range(55)])
        for task_id in done_ids:
            store.set_todo_completed(task_id, True)
        for completed, expected in [(False, open_ids), (True, done_ids)]:
            page = store.list_todos(completed=completed)
            self.assertEqual(len(page['items']), 20)
            found = [row['id'] for row in page['items']]
            if not completed:
                store.task('added-later', 'todo', 'added during pagination')
            while page['next_cursor'] is not None:
                previous = page['next_cursor']
                page = store.list_todos(completed=completed, before=previous)
                if page['next_cursor'] is not None:
                    self.assertLess(int(page['next_cursor']), int(previous))
                found.extend(row['id'] for row in page['items'])
            self.assertEqual(found, list(reversed(expected)))

    def test_invalid_mutations_and_cursors_leave_todos_unchanged(self):
        store.initialize()
        store.task('todo', 'todo', '检查')
        for task_id, value in [('  ', True), (None, True), ('todo', 1), ('todo', 'false')]:
            with self.subTest(task_id=task_id, value=value), self.assertRaises(ValueError):
                store.set_todo_completed(task_id, value)
        for invalid in (True, 0, -1, 'nan', str(2**63)):
            with self.subTest(before=invalid), self.assertRaises(ValueError):
                store.list_todos(before=invalid)
        self.assertEqual(store.list_todos()['items'], [{'id': 'todo', 'body': '检查', 'completed': False}])
