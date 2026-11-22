import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import dashboard_data
import focus_handoffs as handoffs
import focus_sessions as focus
import store


class FocusHandoffTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = patch.object(store, 'ROOT', Path(directory.name))
        root.start()
        self.addCleanup(root.stop)
        self.now = datetime(2026, 11, 23, 11, tzinfo=store.TZ).timestamp()
        self.legacy_id = str(uuid4())
        with store.database() as db:
            db.execute('''CREATE TABLE focus_sessions (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, minutes INTEGER NOT NULL,
                task_id TEXT, started_at REAL NOT NULL, due_at REAL NOT NULL,
                ended_at REAL, state TEXT NOT NULL)''')
            db.execute('INSERT INTO focus_sessions VALUES (?,?,1,NULL,?,?,?,?)',
                       (self.legacy_id, '升级前专注', self.now - 80, self.now - 20, self.now - 20, 'finished'))
        store.initialize()

    def start(self, *, task_id=None, offset=0):
        session = focus.start(str(uuid4()), 1, title='校订第二章', task_id=task_id, now=self.now + offset)
        return session['id']

    def session(self, session_id):
        with store.database() as db:
            return dict(db.execute('SELECT * FROM focus_sessions WHERE id=?', (session_id,)).fetchone())

    def test_end_and_save_are_atomic_exact_text_is_only_stored_in_handoff(self):
        source = self.start()
        text = '  独有草稿片段 #待查\n核实第三段。  '
        with store.database() as db:
            db.execute('''CREATE TRIGGER fail_save BEFORE INSERT ON focus_handoffs
                          BEGIN SELECT RAISE(ABORT, 'injected failure'); END''')
        with self.assertRaises(sqlite3.IntegrityError):
            handoffs.save(source, text, now=self.now + 1)
        self.assertEqual(self.session(source)['state'], 'active')
        self.assertEqual(handoffs.pending()['items'], [])
        with store.database() as db:
            db.execute('DROP TRIGGER fail_save')
        result = handoffs.save(source, text, now=self.now + 2)
        self.assertEqual((result['id'], result['body'], result['status']), (source, text, 'pending'))
        self.assertEqual(self.session(source)['ended_at'], self.now + 2)
        self.assertEqual(self.session(source)['state'], 'stopped')
        self.assertEqual(handoffs.save(source, text, now=self.now + 4), result)
        with self.assertRaises(handoffs.HandoffConflict):
            handoffs.save(source, text.rstrip(), now=self.now + 5)
        store.initialize()
        self.assertEqual(handoffs.pending()['items'], [result])
        resumed = handoffs.resume(source, str(uuid4()), now=self.now + 6)
        live = focus.snapshot(now=self.now + 6)
        self.assertEqual(live['active']['resume_note'], text)
        self.assertNotIn('独有草稿片段', str(live['recent']))
        focus.finish(resumed['id'], now=self.now + 7)
        self.assertNotIn('独有草稿片段', str(focus.snapshot(now=self.now + 8)))
        with store.database() as db:
            copies = [line for line in db.iterdump() if '独有草稿片段' in line]
        self.assertEqual(len(copies), 1)
        self.assertIn('focus_handoffs', copies[0])
        _, markdown = dashboard_data.weekly_markdown(now=datetime.fromtimestamp(self.now + 8, store.TZ))
        self.assertNotIn('独有草稿片段', markdown)

    def test_due_at_save_finishes_naturally_and_notifies_once(self):
        source = self.start()
        due = self.now + 60
        gate = Barrier(2)

        def tick():
            gate.wait(timeout=5)
            store.record_due_reminders(now=due)

        with ThreadPoolExecutor(max_workers=1) as pool:
            scheduler = pool.submit(tick)
            gate.wait(timeout=5)
            saved = handoffs.save(source, '从插图继续', now=due)
            scheduler.result(timeout=10)
        self.assertEqual((self.session(source)['state'], self.session(source)['ended_at']), ('finished', due))
        store.initialize()
        self.assertEqual(handoffs.save(source, '从插图继续', now=due + 5), saved)
        store.record_due_reminders(now=due + 6)
        with store.database() as db:
            notes = [tuple(row) for row in db.execute("SELECT kind,body FROM notes")]
        self.assertEqual(notes, [('reminder', '专注计时结束：校订第二章（1 分钟）')])

    def test_stale_page_can_attach_to_legacy_session_while_another_is_running(self):
        other = self.start()
        with self.assertRaises(KeyError):
            handoffs.save(str(uuid4()), '来源已不存在', now=self.now + 1)
        previous = self.session(self.legacy_id)
        note = handoffs.save(self.legacy_id, '给旧页补记', now=self.now + 2)
        self.assertEqual((note['id'], note['title']), (self.legacy_id, previous['title']))
        self.assertEqual(self.session(self.legacy_id), previous)
        self.assertEqual(focus.snapshot(now=self.now + 2)['active']['id'], other)
        with self.assertRaises((focus.FocusConflict, handoffs.HandoffConflict)):
            handoffs.resume(self.legacy_id, other, now=self.now + 3)
        self.assertEqual(handoffs.pending()['items'], [note])

    def test_resume_preserves_todo_identity_only_while_original_is_pending(self):
        for index, mode in enumerate(('pending', 'completed', 'deleted')):
            with self.subTest(mode=mode):
                task_id = '旧任务 / ' + mode
                title = '修改片头 ' + mode
                store.task(task_id, 'todo', title)
                offset = 60 * index
                source = self.start(task_id=task_id, offset=offset)
                handoffs.save(source, '再处理结尾', now=self.now + offset + 1)
                if mode == 'completed':
                    store.set_todo_completed(task_id, True)
                elif mode == 'deleted':
                    with store.database() as db:
                        db.execute('DELETE FROM tasks WHERE id=?', (task_id,))
                new_id = str(uuid4())
                resumed = handoffs.resume(source, new_id, now=self.now + offset + 2)
                self.assertEqual((resumed['id'], resumed['minutes'], resumed['title']), (new_id, 1, title))
                self.assertEqual(resumed['task_id'], task_id if mode == 'pending' else None)
                with store.database() as db:
                    task = db.execute('SELECT body FROM tasks WHERE id=?', (task_id,)).fetchone()
                    completed = db.execute('SELECT 1 FROM completed_todos WHERE task_id=?', (task_id,)).fetchone()
                self.assertEqual(task['body'] if task else None, None if mode == 'deleted' else title)
                self.assertEqual(completed is not None, mode == 'completed')
                focus.finish(new_id, now=self.now + offset + 3)

    def test_resume_race_lost_response_and_stale_page_cannot_affect_later_session(self):
        source = self.start()
        saved = handoffs.save(source, '请从脚注继续', now=self.now + 1)
        other = self.start(offset=2)
        with self.assertRaises((focus.FocusConflict, handoffs.HandoffConflict)):
            handoffs.resume(source, str(uuid4()), now=self.now + 3)
        self.assertEqual(handoffs.save(source, saved['body'], now=self.now + 3), saved)
        self.assertEqual(handoffs.pending()['items'], [saved])
        self.assertEqual(focus.snapshot(now=self.now + 3)['active']['id'], other)
        focus.finish(other, now=self.now + 4)

        gate = Barrier(3)

        def attempt(key):
            gate.wait(timeout=5)
            try:
                return handoffs.resume(source, key, now=self.now + 5)['id']
            except (focus.FocusConflict, handoffs.HandoffConflict):
                return None

        candidate = str(uuid4())
        keys = [candidate, candidate, str(uuid4())]
        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            outcomes = list(pool.map(attempt, keys))
        winner = focus.snapshot(now=self.now + 6)['active']['id']
        self.assertEqual(outcomes, [key if key == winner else None for key in keys])
        with store.database() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM focus_sessions WHERE id IN (?,?)',
                                        (candidate, keys[-1])).fetchone()[0], 1)
        self.assertEqual(handoffs.pending()['items'], [])
        with self.assertRaises(handoffs.HandoffConflict):
            handoffs.dismiss(source, now=self.now + 7)
        focus.finish(winner, now=self.now + 8)
        latest = self.start(offset=9)
        store.initialize()
        replay = handoffs.resume(source, winner, now=self.now + 10)
        self.assertEqual((replay['id'], replay['state']), (winner, 'stopped'))
        self.assertEqual(handoffs.save(source, saved['body'], now=self.now + 10)['status'], 'resumed')
        self.assertEqual(focus.snapshot(now=self.now + 10)['active']['id'], latest)

    def test_archive_and_resume_are_mutually_exclusive(self):
        source = self.start()
        handoffs.save(source, '检查尾注', now=self.now + 1)
        new_id = str(uuid4())
        gate = Barrier(2)

        def compete(action):
            gate.wait(timeout=5)
            try:
                if action == 'dismiss':
                    handoffs.dismiss(source, now=self.now + 2)
                else:
                    handoffs.resume(source, new_id, now=self.now + 2)
                return action
            except handoffs.HandoffConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            winners = [action for action in pool.map(compete, ('dismiss', 'resume')) if action]
        self.assertEqual(len(winners), 1)
        self.assertEqual(handoffs.pending()['items'], [])
        active = focus.snapshot(now=self.now + 3)['active']
        if winners == ['dismiss']:
            self.assertIsNone(active)
            self.assertEqual(handoffs.dismiss(source, now=self.now + 3)['status'], 'dismissed')
        else:
            self.assertEqual(active['id'], new_id)

    def test_existing_sessions_pagination_and_content_boundaries(self):
        self.assertEqual(handoffs.pending()['items'], [])
        self.assertEqual(self.session(self.legacy_id)['state'], 'finished')
        self.assertEqual(focus.snapshot(now=self.now)['recent'][0]['id'], self.legacy_id)
        source = self.start()
        for body in (None, 123, '', ' \n\t', '句' * 501):
            with self.subTest(body=str(body)[:8]), self.assertRaises(ValueError):
                handoffs.save(source, body, now=self.now + 1)
        self.assertEqual(self.session(source)['state'], 'active')
        exact = ' 行\n' + '字' * 496 + '。'
        self.assertEqual(len(exact), 500)
        self.assertEqual(handoffs.save(source, exact, now=self.now + 2)['body'], exact)
        keys = [source]
        for index in range(1, 23):
            key = self.start(offset=index * 3)
            handoffs.save(key, f'继续第{index}段', now=self.now + index * 3 + 1)
            keys.append(key)
        first = handoffs.pending()
        self.assertEqual([item['id'] for item in first['items']], list(reversed(keys[3:])))
        self.assertTrue(first['next_cursor'])
        handoffs.dismiss(keys[1], now=self.now + 70)
        self.assertEqual(handoffs.dismiss(keys[1], now=self.now + 70)['status'], 'dismissed')
        with self.assertRaises(handoffs.HandoffConflict):
            handoffs.resume(keys[1], str(uuid4()), now=self.now + 71)
        late = self.start(offset=72)
        handoffs.save(late, '翻页间新增', now=self.now + 73)
        store.initialize()
        second = handoffs.pending(before=first['next_cursor'])
        self.assertEqual([item['id'] for item in second['items']], [keys[2], keys[0]])
        self.assertIsNone(second['next_cursor'])
        for invalid in ('-1', 'x', '0'):
            with self.subTest(cursor=invalid), self.assertRaises(ValueError):
                handoffs.pending(before=invalid)
