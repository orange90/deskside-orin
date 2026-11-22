import json
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import store
import activity_evidence
from dashboard_data import activity_week, dashboard_snapshot


class ActivityEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.t = self.midnight.timestamp()
        self.now = self.midnight + timedelta(minutes=10)
        self.day = self.midnight.date().isoformat()
        # A pre-feature installation has historical person rows but no method clock.
        with store.database() as legacy:
            legacy.execute('''CREATE TABLE intervals (
                kind TEXT NOT NULL, state TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                PRIMARY KEY(kind,start))''')
            legacy.execute('INSERT INTO intervals VALUES (?,?,?,?)',
                           ('person', 'estimated_sit', self.t - 30, self.t - 20))
        store.initialize()

    def insert(self, rows):
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)',
                           [(kind, state, self.t + start, self.t + end) for kind, state, start, end in rows])

    def read(self, start, end, day=None):
        return activity_evidence.details(day or self.day, self.t + start, self.t + end, now=self.now)

    def test_raw_displayed_and_method_clocks_remain_distinct(self):
        self.insert([('person', 'sit', 0, 10), ('person', 'unknown', 10, 12),
                     ('person', 'sit', 12, 20), ('person', 'estimated_sit', 20, 26),
                     ('person', 'stand', 26, 36), ('presence', 'present', 0, 36)])
        store.annotate_activity(self.t + 29, self.t + 32, 'away', 'verified-away')
        baseline = dashboard_snapshot(self.day, self.now), activity_week(now=self.now)
        self.insert([('pose_method', 'unknown', -10 * 86400, -10 * 86400 + 1),
                     ('pose_method', 'keypoints', 4, 15), ('pose_method', 'motion', 17.25, 20),
                     ('pose_method', 'estimate_joints', 20, 26), ('pose_method', 'cosmos', 30, 40)])
        with store.database() as db:
            saved = db.execute('SELECT * FROM intervals ORDER BY kind,start').fetchall()
        original = store.database

        @contextmanager
        def readonly():
            with original() as db:
                db.execute('PRAGMA query_only=ON')
                yield db

        with patch.object(store, 'database', readonly):
            evidence = self.read(0, 40)
        self.assertEqual(evidence['raw'], {'sit': 18, 'unknown': 2, 'estimated_sit': 6, 'stand': 10})
        self.assertEqual(evidence['displayed'], {'sit': 26, 'stand': 7, 'away': 3})
        self.assertEqual(evidence['sources'], {'recorded': 25, 'estimated': 6, 'continuity': 2, 'manual': 3})
        self.assertEqual(evidence['methods'], {'keypoints': 11, 'motion': 2.75, 'estimate_joints': 6, 'cosmos': 6})
        self.assertEqual(evidence['method_missing_seconds'], 10.25)
        self.assertEqual(evidence['missing_seconds'], 4)
        self.assertEqual(evidence['privacy_seconds'], 0)
        self.assertEqual(baseline, (dashboard_snapshot(self.day, self.now), activity_week(now=self.now)))
        self.assertFalse(any(row['kind'] == 'pose_method' for row in store.snapshot(self.now)['intervals']))
        with store.database() as db:
            self.assertEqual(saved, db.execute('SELECT * FROM intervals ORDER BY kind,start').fetchall())

    def test_privacy_on_either_clock_masks_details_and_untrusted_text(self):
        self.insert([('person', 'sit', 0, 10), ('person', 'privacy', 10, 14),
                     ('person', 'stand', 14, 24), ('person', 'private-old-label', 30, 32),
                     ('presence', 'privacy', 1, 4),
                     ('presence', 'privacy', 12, 16), ('pose_method', 'keypoints', -2, 1),
                     ('pose_method', 'cosmos', 1, 4), ('pose_method', 'unknown', 4, 10),
                     ('pose_method', 'estimate_default', 10, 16),
                     ('pose_method', 'unrecognized-frame-payload', 16, 24),
                     ('pose_method', 'motion', 24, 30)])
        store.annotate_activity(self.t, self.t + 24, 'away', 'private operator note')
        with store.database() as db:
            db.execute("INSERT INTO notes(kind,body,at) VALUES ('cosmos','private visual note',?)", (self.t + 1,))
        result = self.read(0, 30)
        self.assertEqual(result['raw'], {'sit': 7, 'stand': 8, 'privacy': 9})
        self.assertEqual(result['displayed'], {'away': 15, 'privacy': 9})
        self.assertEqual(result['sources'], {'manual': 15})
        self.assertEqual(result['methods'], {'keypoints': 1, 'unknown': 6})
        self.assertEqual((result['privacy_seconds'], result['missing_seconds'], result['method_missing_seconds']),
                         (9, 6, 8))
        self.assertEqual(set(result), {'date', 'start', 'end', 'raw', 'displayed', 'sources', 'methods',
                                       'privacy_seconds', 'missing_seconds', 'method_missing_seconds'})
        for secret in ('unrecognized-frame-payload', 'private operator note', 'private visual note'):
            self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))
        old = self.read(30, 32)
        self.assertEqual(old['raw'], {'unknown': 2})
        self.assertEqual(old['displayed'], {'unknown': 2})
        self.assertNotIn('private-old-label', json.dumps(old))

    def test_old_records_have_no_method_and_day_without_pose_has_no_evidence(self):
        yesterday = (self.midnight - timedelta(days=1)).date().isoformat()
        old = self.read(-30, 0, yesterday)
        self.assertEqual(old['raw'], {'estimated_sit': 10})
        self.assertEqual(old['displayed'], {'sit': 10})
        self.assertEqual(old['sources'], {'estimated': 10})
        self.assertEqual(old['methods'], {})
        self.assertEqual((old['method_missing_seconds'], old['missing_seconds']), (10, 20))
        self.insert([('desk', 'high', 0, 50), ('presence', 'present', 0, 30),
                     ('pose_method', 'cosmos', 5, 120), ('presence', 'privacy', 30, 40)])
        none = self.read(0, 600)
        self.assertTrue(all(none[key] == {} for key in ('raw', 'displayed', 'sources', 'methods')))
        self.assertEqual((none['method_missing_seconds'], none['privacy_seconds'], none['missing_seconds']),
                         (0, 10, 590))
        masked = self.read(30, 40)
        self.assertEqual((masked['privacy_seconds'], masked['missing_seconds']), (10, 0))
        self.assertTrue(all(masked[key] == {} for key in ('raw', 'displayed', 'sources', 'methods')))

    def test_refuses_nonfinite_future_and_cross_day_ranges(self):
        for left, right in [(float('nan'), 1), (0, float('inf')), (0, 0), (5, 4),
                            (-0.25, 3), (599, 600.25)]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                self.read(left, right)
        for bad_day in ('2026-W47-7', '2026-11-23'):
            with self.subTest(day=bad_day), self.assertRaises(ValueError):
                self.read(0, 1, bad_day)

    def test_midnight_clips_evidence_after_resolving_a_short_gap(self):
        self.insert([('person', 'stand', -4, -1), ('person', 'unknown', -1, 1),
                     ('person', 'stand', 1, 4), ('presence', 'present', -4, 4)])
        yesterday = (self.midnight - timedelta(days=1)).date().isoformat()
        for label, left, right in ((yesterday, -0.5, 0), (self.day, 0, 0.5)):
            with self.subTest(day=label):
                evidence = self.read(left, right, label)
                self.assertEqual(evidence['raw'], {'unknown': 0.5})
                self.assertEqual(evidence['displayed'], {'stand': 0.5})
                self.assertEqual(evidence['sources'], {'continuity': 0.5})
                self.assertEqual(evidence['method_missing_seconds'], 0.5)
