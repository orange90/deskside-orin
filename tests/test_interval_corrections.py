import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import store
import interval_corrections as corrections
from assistant import activity_answer
from dashboard_data import activity_week, dashboard_snapshot


class IntervalCorrectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.day = self.midnight.date().isoformat()
        self.t = self.midnight.timestamp()
        self.now = self.midnight + timedelta(hours=3)

    def insert(self, rows):
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)',
                           [(kind, state, self.t + left, self.t + right) for kind, state, left, right in rows])

    def raw(self):
        with store.database() as db:
            return [tuple(row) for row in db.execute('SELECT * FROM intervals ORDER BY kind,start')]

    def correct(self, left, right, state='stand', key=None, day=None):
        return corrections.create(key or str(uuid4()), day or self.day, self.t + left,
                                  self.t + right, state, now=self.now)

    def test_fractional_hour_boundary_updates_dashboard_week_and_answer(self):
        self.insert([('person', 'estimated_sit', 3597.25, 3604.75),
                     ('presence', 'unknown', 3597.25, 3604.75),
                     ('desk', 'high', 3597.25, 3604.75)])
        before = dashboard_snapshot(self.day, self.now)
        raw = self.raw()
        result = self.correct(3597.25, 3604.75)
        day = dashboard_snapshot(self.day, self.now)
        self.assertEqual(day['totals'], {'stand': 8, 'present': 8})
        self.assertEqual([h['stand'] for h in day['hourly'][:2]], [2.75, 4.75])
        annotation, = day['annotations']
        self.assertEqual(annotation['correction_id'], result['correction_id'])
        self.assertEqual((annotation['start'], annotation['end']), (self.t + 3597.25, self.t + 3604.75))
        week = activity_week(now=self.now)
        self.assertEqual((week['totals']['stand'], week['estimated_sit'], week['manual_seconds']), (8, 0, 8))
        self.assertIn('站了 8 秒', activity_answer('今天站了多久', self.now))
        self.assertEqual(self.raw(), raw)
        corrections.undo(result['correction_id'])
        self.assertEqual(dashboard_snapshot(self.day, self.now), before)
        self.assertEqual(activity_week(now=self.now)['estimated_sit'], 8)
        self.assertEqual(self.raw(), raw)

    def test_beijing_midnight_and_live_cutoff(self):
        current = self.now.timestamp() - self.t
        yesterday = (self.midnight - timedelta(days=1)).date().isoformat()
        tomorrow = (self.midnight + timedelta(days=1)).date().isoformat()
        self.insert([('person', 'sit', -10.125, 7.5), ('person', 'sit', current - 5, current + 5)])
        for left, right, day in [(-1, 1, yesterday), (-1, 1, self.day),
                                 (current - 5, current + 0.5, self.day), (86400, 86401, tomorrow)]:
            with self.subTest(left=left, right=right, day=day), self.assertRaises(ValueError):
                self.correct(left, right, day=day)
        self.correct(-10.125, 0, day=yesterday)
        self.correct(current - 5, current)
        self.assertEqual(dashboard_snapshot(yesterday, self.now)['hourly'][23]['stand'], 10.125)
        self.assertEqual(dashboard_snapshot(self.day, self.now)['totals'], {'sit': 8, 'stand': 5})

    def test_privacy_gaps_other_clocks_and_old_annotations_cannot_be_corrected(self):
        self.insert([('person', 'sit', 0, 10), ('person', 'privacy', 10, 20),
                     ('person', 'sit', 20, 30), ('person', 'stand', 31, 35),
                     ('desk', 'low', 40, 60), ('presence', 'present', 50, 70)])
        store.annotate_activity(self.t + 3, self.t + 4, 'stand', 'prior standalone annotation')
        before = dashboard_snapshot(self.day, self.now)
        for left, right in [(0, 10), (10, 20), (9, 21), (29, 32), (36, 39), (40, 49), (51, 61)]:
            with self.subTest(left=left, right=right), self.assertRaises(corrections.CorrectionConflict):
                self.correct(left, right)
        with self.assertRaises(ValueError):
            self.correct(22, 23, 'privacy')
        self.assertEqual(dashboard_snapshot(self.day, self.now), before)

    def test_concurrent_retries_and_replays_after_undo(self):
        self.insert([('person', 'unknown', 1, 30)])
        key = str(uuid4())
        barrier = Barrier(2)

        def retry():
            barrier.wait(timeout=5)
            return self.correct(1, 30, key=key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(retry) for _ in range(2)]
            results = [f.result(timeout=10) for f in pending]
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(dashboard_snapshot(self.day, self.now)['annotations']), 1)
        for other_key, left, state in [(key, 1, 'sit'), (key, 2, 'stand'), (str(uuid4()), 20, 'away')]:
            with self.assertRaises(corrections.CorrectionConflict):
                self.correct(left, 30, state, key=other_key)
        corrections.undo(key)
        corrections.undo(key)
        with self.assertRaises(corrections.CorrectionConflict):
            self.correct(1, 30, key=key)
        self.assertEqual(dashboard_snapshot(self.day, self.now)['annotations'], [])

    def test_daily_absence_keeps_interval_undo_identity_and_unrelated_notes(self):
        start = -86400
        day = (self.midnight - timedelta(days=1)).date().isoformat()
        self.insert([('person', 'sit', start, start + 90), ('presence', 'present', start, start + 90)])
        previous = store.annotate_activity(self.t + start, self.t + start + 1, 'away', '旧标注')
        with store.database() as db:
            db.execute("INSERT INTO notes(kind,body,at) VALUES ('dialogue','旧笔记',?)", (self.t + start + 1,))
        result = self.correct(start + 30, start + 60, 'away', day=day)
        batch = store.create_absence(day, day, now=self.now)
        with self.assertRaises(corrections.CorrectionConflict):
            self.correct(start + 65, start + 70, 'sit', day=day)
        covered = dashboard_snapshot(day, self.now)
        active = [a['correction_id'] for a in covered['annotations'] if a.get('correction_id')]
        self.assertEqual(active, [result['correction_id']])
        corrections.undo(result['correction_id'])
        after = dashboard_snapshot(day, self.now)
        self.assertEqual(after['totals'], {'away': 90})
        self.assertEqual(after['notes'], covered['notes'])
        self.assertEqual(after['notes'][0]['body'], '旧笔记')
        self.assertEqual(len(after['annotations']), 2)
        self.assertIn(previous, [a['id'] for a in after['annotations']])
        self.assertFalse(any(a.get('correction_id') for a in after['annotations']))
        self.assertEqual(store.list_absences(day)[0]['batch_id'], batch['batch_id'])

    def test_submillisecond_pose_gap_is_not_manually_added_to_presence(self):
        self.insert([('person', 'sit', 0, 1), ('person', 'sit', 1.0005, 5),
                     ('presence', 'unknown', 0, 5)])
        result = self.correct(0, 5)
        day = dashboard_snapshot(self.day, self.now)
        gaps = [r for r in day['intervals'] if r['kind'] == 'presence' and r['state'] == 'unknown']
        self.assertEqual([(r['start'], r['end']) for r in gaps], [(self.t + 1, self.t + 1.0005)])
        corrections.undo(result['correction_id'])
        self.assertEqual(dashboard_snapshot(self.day, self.now)['annotations'], [])
