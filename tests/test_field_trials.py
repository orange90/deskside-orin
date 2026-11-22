import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import field_trials as trials
import store


class FieldTrialTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.session = str(uuid4())
        self.at = 1_700_000_000.

    def test_retries_survive_restart_and_never_rewind_newer_marks(self):
        first, second = str(uuid4()), str(uuid4())
        trials.start(self.session, 'sit', now=self.at)
        trials.mark(self.session, first, 'stand', now=self.at + 10)
        trials.mark(self.session, second, 'pause', now=self.at + 15)
        store.initialize()
        self.assertEqual(trials.snapshot(now=self.at + 20)['active']['current_state'], 'pause')
        self.assertEqual(trials.mark(self.session, first, 'stand', now=self.at + 20)['current_state'], 'pause')
        with self.assertRaises(trials.TrialConflict):
            trials.mark(self.session, first, 'away', now=self.at + 21)
        with self.assertRaises(trials.TrialConflict):
            trials.report(self.session, now=self.at + 22)
        finished = trials.finish(self.session, now=self.at + 30)
        self.assertEqual(finished['ended_at'], self.at + 30)
        self.assertEqual(trials.finish(self.session, now=self.at + 40)['ended_at'], self.at + 30)
        self.assertEqual(trials.start(self.session, 'sit', now=self.at + 45)['ended_at'], self.at + 30)
        with self.assertRaises(trials.TrialConflict):
            trials.mark(self.session, str(uuid4()), 'sit', now=self.at + 50)
        result = trials.report(self.session)
        self.assertEqual(result['metrics']['paused_seconds'], 15)
        self.assertEqual(result['metrics']['labeled_seconds'], 15)
        self.assertEqual(result['metrics']['matrix']['sit']['missing'], 10)
        self.assertEqual(result['metrics']['matrix']['stand']['missing'], 5)

    def test_deadline_after_service_downtime_caps_report_and_releases_slot(self):
        trial = trials.start(self.session, 'away', now=self.at)
        self.assertEqual(trial['deadline'], self.at+7200)
        next_id = str(uuid4())
        trials.start(next_id, 'stand', now=self.at+9000)
        snap = trials.snapshot(now=self.at+9100)
        self.assertEqual(snap['active']['id'], next_id)
        self.assertEqual(snap['recent'][0]['ended_at'], self.at+7200)
        self.assertEqual(trials.report(self.session)['metrics']['labeled_seconds'], 7200)
        with self.assertRaises(KeyError):
            trials.report(str(uuid4()))

    def test_two_concurrent_starts_and_two_identical_mark_retries(self):
        def begin(session_id):
            try:
                return trials.start(session_id, 'sit', now=self.at)['id']
            except trials.TrialConflict:
                return None

        mark_id = str(uuid4())
        with ThreadPoolExecutor(max_workers=2) as pool:
            started = [sid for sid in pool.map(begin, [self.session, str(uuid4())]) if sid]
            self.assertEqual(len(started), 1)
            jobs = [pool.submit(trials.mark, started[0], mark_id, 'away', self.at + 2) for _ in range(2)]
            self.assertTrue(all(task.result()['current_state'] == 'away' for task in jobs))
        trials.finish(started[0], now=self.at + 7)
        report = trials.report(started[0])['metrics']
        self.assertEqual(report['transitions']['total'], 1)
        self.assertEqual(report['matrix']['away']['missing'], 5)

    def test_labels_do_not_modify_camera_history_or_use_personal_corrections(self):
        with store.database() as db:
            db.execute("INSERT INTO intervals VALUES ('person','estimated_sit',?,?)", (self.at, self.at+30))
        store.annotate_activity(self.at, self.at+30, 'stand', 'private annotation')
        trials.start(self.session, 'stand', now=self.at)
        trials.finish(self.session, now=self.at+30)
        summary = trials.report(self.session)['metrics']
        self.assertEqual(summary['matched_seconds'], 0)
        self.assertEqual(summary['estimated_seconds'], 30)
        self.assertEqual(summary['matrix']['stand']['sit'], 30)
        with store.database() as db:
            self.assertEqual(db.execute('SELECT state FROM intervals').fetchall()[0][0], 'estimated_sit')
            self.assertEqual(db.execute('SELECT count(*) FROM activity_annotations').fetchone()[0], 1)

    def test_invalid_labels_never_create_a_session(self):
        for label in ['pause', 'estimated_sit', 'private note']:
            with self.subTest(label=label), self.assertRaises(ValueError):
                trials.start(self.session, label, now=self.at)
        self.assertIsNone(trials.snapshot(now=self.at)['active'])
