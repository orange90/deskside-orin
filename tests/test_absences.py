import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import store
from dashboard_data import dashboard_snapshot


class AbsenceTests(unittest.TestCase):
    first = '2024-02-29'
    last = '2024-03-01'
    # March 2 at 00:01 in Beijing. March 1 is already a complete local day.
    now = datetime(2024, 3, 1, 16, 1, tzinfo=timezone.utc)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2024, 2, 29, tzinfo=store.TZ).timestamp()

    def confirm(self, start=None, end=None, request_id=None):
        return store.create_absence(start or self.first, end or self.last,
                                    request_id=request_id, now=self.now)

    def annotations(self):
        with store.database() as db:
            return [dict(row) for row in db.execute('SELECT * FROM activity_annotations ORDER BY id')]

    def insert_intervals(self, rows):
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)', [
                (kind, state, self.midnight + start, self.midnight + end)
                for kind, state, start, end in rows
            ])

    def test_two_days_keep_raw_rows_privacy_and_gaps(self):
        self.insert_intervals([
            ('person', 'sit', -30, 30), ('person', 'privacy', 30, 45),
            ('person', 'stand', 45, 60), ('presence', 'present', -30, 60),
            ('desk', 'high', -30, 60),
            ('person', 'estimated_sit', 86370, 86430), ('presence', 'present', 86370, 86430),
            ('person', 'stand', 172770, 172830), ('presence', 'present', 172770, 172830),
        ])
        with store.database() as db:
            raw = db.execute('SELECT * FROM intervals ORDER BY kind,start').fetchall()
        days = ['2024-02-28', self.first, self.last, '2024-03-02']
        before = {day: dashboard_snapshot(day, self.now) for day in days}
        batch = self.confirm()
        self.assertEqual([(r['start'], r['end'], r['state']) for r in self.annotations()], [
            (self.midnight, self.midnight + 86400, 'away'),
            (self.midnight + 86400, self.midnight + 172800, 'away'),
        ])
        self.assertEqual(store.list_absences(self.first), store.list_absences(self.last))
        self.assertEqual([b['batch_id'] for b in store.list_absences(self.last)], [batch['batch_id']])
        first = dashboard_snapshot(self.first, self.now)
        self.assertEqual(first['totals'], {'away': 75, 'privacy': 15})
        self.assertEqual(sum(h['away'] for h in first['hourly']), 75)
        self.assertEqual(sum(r['end'] - r['start'] for r in first['intervals']
                             if r['kind'] == 'presence' and r['state'] == 'privacy'), 15)
        self.assertEqual([r['state'] for r in first['intervals'] if r['kind'] == 'desk'], ['high'])
        self.assertEqual(dashboard_snapshot(self.last, self.now)['totals'], {'away': 60})
        for neighbor in [days[0], days[-1]]:
            self.assertEqual(store.list_absences(neighbor), [])
            self.assertEqual(dashboard_snapshot(neighbor, self.now), before[neighbor])
        with store.database() as db:
            self.assertEqual(raw, db.execute('SELECT * FROM intervals ORDER BY kind,start').fetchall())
        store.undo_absence(batch['batch_id'])
        self.assertEqual({day: dashboard_snapshot(day, self.now) for day in days}, before)

    def test_conflict_on_last_day_does_not_partially_confirm_first_day(self):
        mark = self.midnight + 86400
        store.annotate_activity(mark + 120, mark + 240, 'stand', 'verified')
        before = self.annotations()
        with self.assertRaises(store.AbsenceConflict):
            self.confirm()
        self.assertEqual(self.annotations(), before)
        self.assertEqual(store.list_absences(self.first), [])

    def test_undo_only_removes_its_own_annotations(self):
        t = self.midnight
        previous = store.annotate_activity(t, t + 60, 'away', 'older correction')
        batch = self.confirm()
        later = store.annotate_activity(t + 30, t + 90, 'sit', 'later correction')
        with self.assertRaises(store.AbsenceConflict):
            self.confirm(self.last, self.last)
        store.undo_absence(batch['batch_id'])
        self.assertEqual([r['id'] for r in self.annotations()], [previous, later])
        self.assertEqual(store.list_absences(self.first), [])

    def test_concurrent_retries_and_replay_after_undo(self):
        key = str(uuid4())
        barrier = Barrier(2)

        def confirm_at_barrier():
            barrier.wait(timeout=5)
            return self.confirm(request_id=key)['batch_id']

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(confirm_at_barrier) for _ in range(2)]
            ids = [f.result(timeout=10) for f in pending]
        self.assertEqual(ids[0], ids[1])
        other_key = str(uuid4())
        self.assertEqual(self.confirm(request_id=other_key)['batch_id'], ids[0])
        with self.assertRaises(store.AbsenceConflict):
            self.confirm('2024-02-28', '2024-02-28', request_id=key)
        self.assertEqual(len(self.annotations()), 2)
        store.undo_absence(ids[0])
        store.undo_absence(ids[0])
        for old_key in (key, other_key):
            with self.assertRaises(store.AbsenceConflict):
                self.confirm(request_id=old_key)
        self.assertEqual(self.annotations(), [])
        self.assertEqual(store.list_absences(self.last), [])

    def test_write_error_rolls_back_batch_and_daily_annotations(self):
        with store.database() as db:
            db.execute(f"""CREATE TRIGGER fail_second_day BEFORE INSERT ON activity_annotations
                          WHEN NEW.start = {self.midnight + 86400}
                          BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.confirm()
        self.assertEqual(self.annotations(), [])
        self.assertEqual(store.list_absences(self.first), [])
        with store.database() as db:
            db.execute('DROP TRIGGER fail_second_day')
        self.confirm()
        self.assertEqual(len(self.annotations()), 2)

    def test_only_completed_local_dates_up_to_31_days(self):
        for start, end in [('2024-03-02', '2024-03-02'), ('2024-03-04', '2024-03-05'),
                           (self.last, self.first), ('2024-02-30', self.last),
                           ('2024-01-30', self.last), ('20240229', self.last)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.confirm(start, end)
        self.assertEqual(self.annotations(), [])
        self.confirm('2024-01-31', self.last)
        self.assertEqual(len(self.annotations()), 31)
