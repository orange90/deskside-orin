import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import store
from dashboard_data import activity_range, activity_week


class ActivityRangeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        database_root = patch.object(store, 'ROOT', Path(temp.name))
        database_root.start()
        self.addCleanup(database_root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.now = self.midnight + timedelta(hours=12)

    def insert(self, rows):
        midnight = self.midnight.timestamp()
        with store.database() as db:
            db.executemany("INSERT INTO intervals VALUES (?,?,?,?)",
                           [(kind, state, midnight + start, midnight + end) for kind, state, start, end in rows])

    def test_cross_midnight_counts_manual_estimated_and_private_only_where_recorded(self):
        self.insert([('person', 'sit', -100, -58), ('person', 'unknown', -58, -55),
                     ('person', 'sit', -55, -50), ('person', 'privacy', -50, -40),
                     ('person', 'unknown', -40, -30), ('person', 'estimated_stand', -30, 12),
                     ('presence', 'present', -80, 20), ('presence', 'away', 20, 40)])
        t = self.midnight.timestamp()
        store.annotate_activity(t - 45, t - 35, 'away', '历史标注')
        start, end = '2026-11-21T23:59', '2026-11-22T00:01'
        result = activity_range(start, end, self.now)
        self.assertEqual(result, {
            'start': start, 'end': end, 'elapsed_seconds': 120,
            'recorded_seconds': 72, 'presence_recorded_seconds': 100,
            'presence_unknown_seconds': 0, 'presence_privacy_seconds': 10,
            'manual_seconds': 5, 'manual_presence_seconds': 5, 'estimated_sit': 3, 'estimated_stand': 42,
            'totals': {'sit': 10, 'stand': 42, 'privacy': 10, 'unknown': 5, 'away': 5, 'present': 65}})

    def test_presence_uncertainty_uses_its_own_recorded_seconds(self):
        self.insert([('person', 'unknown', 0, 60), ('presence', 'unknown', 0, 60),
                     ('person', 'privacy', 60, 90), ('presence', 'present', 72, 82),
                     ('presence', 'unknown', 97, 103.6), ('presence', 'privacy', 104, 111)])
        unknown = activity_range('2026-11-22T00:00', '2026-11-22T00:01', self.now)
        self.assertEqual((unknown['totals']['sit'], unknown['totals']['present'], unknown['totals']['unknown'],
                          unknown['presence_unknown_seconds'], unknown['presence_privacy_seconds']), (0, 0, 60, 60, 0))
        mixed = activity_range('2026-11-22T00:01', '2026-11-22T00:02', self.now)
        self.assertEqual((mixed['totals']['privacy'], mixed['presence_privacy_seconds']), (30, 17))
        self.assertEqual((mixed['totals']['unknown'], mixed['presence_unknown_seconds']), (0, 7))
        self.assertEqual((mixed['recorded_seconds'], mixed['presence_recorded_seconds']), (30, 24))

    def test_both_window_edges_use_context_but_cannot_see_beyond_as_of(self):
        self.insert([('person', 'sit', -10, -1), ('person', 'unknown', -1, 2), ('person', 'sit', 2, 10),
                     ('person', 'stand', 50, 59), ('person', 'unknown', 59, 62), ('person', 'stand', 62, 75),
                     ('presence', 'present', -10, 75)])
        start, end = '2026-11-22T00:00', '2026-11-22T00:01'
        result = activity_range(start, end, self.now)
        self.assertEqual((result['totals']['sit'], result['totals']['stand']), (10, 10))
        self.assertEqual((result['estimated_sit'], result['estimated_stand']), (2, 1))
        self.assertEqual((result['recorded_seconds'], result['presence_recorded_seconds']), (20, 60))
        right_now = self.midnight.astimezone(timezone.utc) + timedelta(minutes=1)
        live = activity_range(start, end, right_now)
        self.assertEqual((live['estimated_stand'], live['totals']['unknown']), (0, 1))
        with store.database() as db:
            db.execute("DELETE FROM intervals WHERE kind='presence'")
        self.insert([('presence', 'present', -10, 0), ('presence', 'privacy', 0, 1),
                     ('presence', 'present', 1, 75)])
        masked = activity_range(start, end, self.now)
        self.assertEqual((masked['estimated_sit'], masked['totals']['unknown'], masked['totals']['present']),
                         (0, 2, 59))

    def test_full_day_matches_week_day_and_partial_range_reads_only_activity(self):
        self.insert([('person', 'sit', -110, -90.4), ('person', 'estimated_stand', -80, -60.3),
                     ('presence', 'present', -120, 0), ('person', 'away', 0, 60)])
        day = activity_week(end_date='2026-11-21', now=self.now)['days'][-1]
        result = activity_range('2026-11-21T00:00', '2026-11-22T00:00', self.now)
        self.assertEqual({key: result[key] for key in day if key != 'date'},
                         {key: value for key, value in day.items() if key != 'date'})
        database = store.database
        statements = []

        @contextmanager
        def traced():
            with database() as db:
                db.set_trace_callback(statements.append)
                yield db

        with patch.object(store, 'database', traced), patch.object(store, 'activity_window', wraps=store.activity_window) as read:
            result = activity_range('2026-11-21T23:59', '2026-11-22T00:00', self.now)
        self.assertEqual((result['recorded_seconds'], result['presence_recorded_seconds'], result['totals']['present']),
                         (0, 60, 60))
        self.assertEqual(read.call_count, 1)
        self.assertIn('BEGIN', statements)
        self.assertNotRegex(' '.join(statements).lower(), r'\b(notes|tasks|settings|focus_handoffs)\b')
        pose_only = activity_range('2026-11-22T00:00', '2026-11-22T00:01', self.now)
        self.assertEqual((pose_only['recorded_seconds'], pose_only['presence_recorded_seconds'],
                          pose_only['totals']['away']), (60, 0, 60))

    def test_strict_boundaries_future_and_empty_ancient_dates(self):
        start, end = '2026-11-21T00:00', '2026-11-22T00:00'
        self.assertEqual(activity_range(start, end, self.now)['elapsed_seconds'], 86400)
        for bounds in ((start, start), (end, start), ('2026-11-20T23:59', end),
                       ('2026-11-22T12:00', '2026-11-22T12:01'), (None, end),
                       ('2026-11-21 00:00', end), ('2026-11-21T00:00:00', end),
                       ('2026-11-21T00:00+08:00', end), ('2026-11-21t00:00', end),
                       ('1991-04-14T02:30', '1991-04-14T04:00'),
                       ('2026-02-30T00:00', end), ('2026-11-21T24:00', end),
                       ('0000-01-01T00:00', end), ('9999-12-31T23:58', '9999-12-31T23:59')):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                activity_range(*bounds, now=self.now)
        empty = activity_range('0001-01-01T00:00', '0001-01-01T00:01', self.now)
        self.assertEqual((empty['recorded_seconds'], empty['presence_recorded_seconds'], empty['elapsed_seconds']),
                         (0, 0, 60))
        self.assertFalse(any(empty['totals'].values()))
