import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import store
from dashboard_data import activity_week, dashboard_snapshot


class WeeklyActivityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = patch.object(store, 'ROOT', Path(directory.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.now = self.midnight + timedelta(hours=12)

    def insert(self, rows):
        t = self.midnight.timestamp()
        with store.database() as db:
            db.executemany("INSERT INTO intervals VALUES (?,?,?,?)",
                           [(kind, state, t + start, t + end) for kind, state, start, end in rows])

    def test_week_matches_day_after_corrections_and_midnight_interpolation(self):
        self.insert([('person', 'stand', -90, -1), ('person', 'unknown', -1, 2),
                     ('person', 'stand', 2, 8), ('person', 'estimated_sit', 10, 40),
                     ('person', 'privacy', 40, 50), ('person', 'unknown', 50, 65),
                     ('person', 'estimated_stand', 65, 80), ('presence', 'present', -90, 90),
                     ('person', 'sit', 43200, 86400)])
        t = self.midnight.timestamp()
        store.annotate_activity(t + 35, t + 55, 'away', '本人离开')
        week = activity_week(now=self.now)
        self.assertEqual((week['start_date'], week['end_date']), ('2026-11-16', '2026-11-22'))
        self.assertEqual(len(week['days']), 7)
        yesterday, today = week['days'][-2:]
        self.assertEqual((yesterday['estimated_stand'], today['estimated_stand']), (1, 17))
        self.assertEqual(today['totals'], dict(stand=23, sit=25, privacy=10, away=10, unknown=10, present=70))
        self.assertEqual((today['recorded_seconds'], today['presence_recorded_seconds'], today['estimated_sit']),
                         (78, 90, 25))
        self.assertEqual((week['recorded_seconds'], week['estimated_stand'], week['estimated_sit']), (168, 18, 25))
        self.assertEqual(week['elapsed_seconds'], 6 * 86400 + 43200)
        for day in week['days']:
            daily = dashboard_snapshot(day['date'], self.now)['totals']
            self.assertEqual(day['totals'], {key: daily.get(key, 0) for key in day['totals']})

    def test_presence_only_and_fractional_days(self):
        self.insert([('presence', 'away', -90, -10), ('presence', 'present', -10, 10),
                     ('person', 'estimated_sit', -172800, -172799.4), ('person', 'sit', 120, 120.6)])
        week = activity_week(now=self.now)
        yesterday, today = week['days'][-2:]
        self.assertEqual(yesterday['recorded_seconds'], 0)
        self.assertEqual((yesterday['presence_recorded_seconds'], yesterday['totals']['present']), (90, 10))
        self.assertEqual((today['recorded_seconds'], today['presence_recorded_seconds']), (1, 10))
        self.assertEqual(week['totals']['sit'], 2)  # Sum of rounded daily totals.
        self.assertEqual(week['totals']['away'], 0)
        empty = week['days'][0]
        self.assertEqual((empty['recorded_seconds'], empty['presence_recorded_seconds'], empty['elapsed_seconds']),
                         (0, 0, 86400))

    def test_end_date_validation_and_beijing_midnight(self):
        # UTC is still Saturday; the local calendar has reached Sunday.
        now = datetime(2026, 11, 21, 16, 1, tzinfo=timezone.utc)
        self.assertEqual(activity_week(now=now)['days'][-1]['elapsed_seconds'], 60)
        previous_week = activity_week('2026-11-21', now)
        self.assertEqual(previous_week['elapsed_seconds'], 7 * 86400)
        for invalid in ('', '20261121', '2026-02-30', '2026-W47-7', '2026-11-23', '0001-01-06'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                activity_week(invalid, now)

    def test_week_does_not_mix_snapshots_when_a_sample_arrives(self):
        read = store.activity_window
        reads = 0

        def read_then_record(db, *args):
            nonlocal reads
            result = read(db, *args)
            if reads == 0:
                self.insert([('person', 'stand', 0, 60)])
            reads += 1
            return result

        with patch.object(store, 'activity_window', read_then_record):
            self.assertEqual(activity_week(now=self.now)['totals']['stand'], 0)
        self.assertEqual(activity_week(now=self.now)['totals']['stand'], 60)
