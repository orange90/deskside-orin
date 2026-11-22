import importlib.util
import math
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import privacy_pause as privacy
import store


class PrivacyExtensionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = patch.object(store, 'ROOT', Path(directory.name))
        root.start()
        self.addCleanup(root.stop)
        self.t = 1750000000.25
        self.until = self.t + 60
        self.session = str(uuid4())
        with store.database() as db:
            db.execute('''CREATE TABLE camera_pauses (
                id TEXT PRIMARY KEY, minutes INTEGER NOT NULL CHECK(minutes IN (1,15,30,60)),
                started_at REAL NOT NULL, until REAL NOT NULL CHECK(until>started_at),
                ended_at REAL CHECK(ended_at BETWEEN started_at AND until))''')
            db.execute('INSERT INTO camera_pauses VALUES (?,1,?,?,NULL)', (self.session, self.t, self.until))
        store.initialize()

    def extend(self, key, minutes, expected, offset=1):
        return privacy.extend(self.session, key, minutes, expected, now=self.t + offset)

    def test_legacy_session_extension_and_lost_response_retries_survive_restart(self):
        first = str(uuid4())
        expected = self.until
        for index, minutes in enumerate((1, 15, 30, 60)):
            key = str(uuid4()) if index else first
            previous = expected
            expected += minutes * 60
            result = self.extend(key, minutes, previous, offset=index + 1)
            self.assertEqual(result, {'active': {'id': self.session, 'started_at': self.t, 'until': expected},
                                      'server_now': self.t + index + 1})
            store.initialize()
            self.assertEqual(self.extend(key, minutes, previous, offset=index + 2)['active'], result['active'])
        self.assertEqual(self.extend(first, 1, self.until, offset=10)['active']['until'], expected)
        for changed_minutes, changed_until in ((15, self.until), (1, expected)):
            with self.subTest(minutes=changed_minutes), self.assertRaises(privacy.PrivacyConflict):
                self.extend(first, changed_minutes, changed_until)
        self.assertEqual(privacy.start(self.session, 1, now=self.t + 11)['active']['until'], expected)
        self.assertIsNone(privacy.snapshot(now=expected)['active'])
        self.assertIsNone(self.extend(first, 1, self.until, offset=12)['active'])
        fresh = privacy.start(str(uuid4()), 15, now=expected + 1)['active']
        self.assertEqual(privacy.extend(self.session, first, 1, self.until, now=expected + 2)['active'], fresh)
        with self.assertRaises(privacy.PrivacyConflict):
            privacy.extend(fresh['id'], first, 1, fresh['until'], now=expected + 2)

    def test_exact_deadline_validation_and_observed_expiry_never_revives(self):
        for expected in (None, True, '1750000060.25', math.nan, math.inf):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                self.extend(str(uuid4()), 1, expected)
        for minutes in (0, True, 15., 2, 61):
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                self.extend(str(uuid4()), minutes, self.until)
        with self.assertRaises(ValueError):
            self.extend('not-a-uuid', 1, self.until)
        with self.assertRaises(KeyError):
            privacy.extend(str(uuid4()), str(uuid4()), 15, self.until, now=self.t + 1)
        with self.assertRaises(privacy.PrivacyConflict):
            self.extend(str(uuid4()), 1, math.nextafter(self.until, math.inf))
        with self.assertRaises(privacy.PrivacyConflict):
            self.extend(str(uuid4()), 1, self.until, offset=60)
        store.initialize()
        self.assertIsNone(privacy.snapshot(now=self.t + 2)['active'])
        with self.assertRaises(privacy.PrivacyConflict):
            self.extend(str(uuid4()), 15, self.until, offset=3)

    def test_competing_requests_can_change_one_seen_deadline_only_once(self):
        repeated, other = str(uuid4()), str(uuid4())
        requests = [(repeated, 1), (repeated, 1), (other, 30)]
        gate = Barrier(len(requests))

        def try_extend(request):
            key, minutes = request
            gate.wait(timeout=5)
            try:
                return self.extend(key, minutes, self.until)['active']['until']
            except privacy.PrivacyConflict:
                return None

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            outcomes = list(pool.map(try_extend, requests))
        actual = privacy.snapshot(now=self.t + 2)['active']['until']
        self.assertIn(actual, (self.until + 60, self.until + 1800))
        self.assertEqual(outcomes, [actual if self.until + minutes * 60 == actual else None for _, minutes in requests])

    def test_resume_and_extend_never_restart_a_lease(self):
        gate = Barrier(2)
        key = str(uuid4())

        def change(action):
            gate.wait(timeout=5)
            if action == 'resume':
                return privacy.resume(self.session, now=self.t + 2)
            try:
                return self.extend(key, 15, self.until, offset=2)
            except privacy.PrivacyConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            resumed, extended = list(pool.map(change, ('resume', 'extend')))
        self.assertIsNone(resumed['active'])
        if extended is not None:
            self.assertEqual(extended['active']['until'], self.until + 900)
            self.assertIsNone(self.extend(key, 15, self.until, offset=3)['active'])
        self.assertIsNone(privacy.snapshot(now=self.t + 3)['active'])
        with self.assertRaises(privacy.PrivacyConflict):
            self.extend(str(uuid4()), 15, self.until, offset=4)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('cv2', 'fastapi', 'numpy')),
                         'requires dashboard and camera dependencies')
    def test_same_session_extends_in_place_while_capture_remains_private(self):
        import cv2
        import main
        import vision

        name = '_deskside_extension_test_app'
        spec = importlib.util.spec_from_file_location(name, main.__file__)
        app = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {name: app}):
            spec.loader.exec_module(app)
        wall = self.t
        steps = 0
        requested = False

        def finished():
            nonlocal steps
            steps += 1
            self.assertLess(steps, 200, 'capture loop stopped advancing its private heartbeat')
            return wall >= self.until + 4

        def wait(delay):
            nonlocal wall, requested
            wall += delay
            if not requested and wall >= self.until - 1:
                requested = True
                generation, scene = app.capture_generation, app.vision_epoch
                changed = app.sync_privacy(privacy.extend, self.session, str(uuid4()), 1, self.until)
                self.assertEqual(changed['active']['until'], self.until + 60)
                self.assertEqual(app.camera_view()['manual_until'], self.until + 60)
                self.assertEqual((app.capture_generation, app.vision_epoch), (generation, scene))

        with patch.object(privacy.time, 'time', side_effect=lambda: wall), \
                patch.object(app, 'time', SimpleNamespace(time=lambda: wall, monotonic=lambda: wall)), \
                patch.object(app, 'stop', SimpleNamespace(is_set=finished, wait=wait)), \
                patch.object(cv2, 'VideoCapture', side_effect=AssertionError('camera opened during extended pause')), \
                patch.object(vision, 'Vision', side_effect=AssertionError('vision loaded during extended pause')):
            app.sync_privacy(startup=True)
            app.capture_frames()
            self.assertTrue(requested)
            self.assertTrue(app.camera_view()['manual_privacy'])
            self.assertIsNone(app.preview)
        with store.database() as db:
            for clock in ('person', 'presence'):
                rows = [tuple(row) for row in db.execute('SELECT state,start,end FROM intervals WHERE kind=?', (clock,))]
                self.assertEqual(rows, [('privacy', self.t, self.until + 3)])
