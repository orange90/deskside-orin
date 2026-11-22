"""Exercise privacy leases and camera endpoints without opening a camera or a model."""
import asyncio
import importlib.util
import sys
import tempfile
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import privacy_pause
import store

try:
    import cv2
    import numpy as np
    import main
    import vision
except ModuleNotFoundError as exc:
    if exc.name not in ('cv2', 'numpy', 'fastapi', 'pydantic', 'starlette'):
        raise
    main = None


class IsolatedPrivacyDB(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.replace(store, 'ROOT', Path(temp.name))
        store.initialize()

    def replace(self, obj, attr, value):
        replacement = patch.object(obj, attr, value)
        replacement.start()
        self.addCleanup(replacement.stop)
        return value


class PrivacyLeaseTests(IsolatedPrivacyDB):
    def test_restart_expiry_idempotence_and_stale_resume(self):
        old, new = str(uuid4()), str(uuid4())
        start = 1795399200.25
        lease = privacy_pause.start(old, 1, now=start)['active']
        self.assertEqual(lease, {'id': old, 'started_at': start, 'until': start + 60})
        store.initialize()
        self.assertEqual(privacy_pause.snapshot(now=start + 30)['active'], lease)
        self.assertEqual(privacy_pause.start(old, 1, now=start + 40)['active'], lease)
        with self.assertRaises(privacy_pause.PrivacyConflict):
            privacy_pause.start(old, 15, now=start + 40)
        with self.assertRaises(privacy_pause.PrivacyConflict):
            privacy_pause.start(new, 15, now=start + 40)
        self.assertIsNone(privacy_pause.snapshot(now=start + 60)['active'])
        self.assertIsNone(privacy_pause.start(old, 1, now=start + 10)['active'])
        replacement = privacy_pause.start(new, 30, now=start + 61)['active']
        self.assertEqual(privacy_pause.resume(old, now=start + 62)['active'], replacement)
        self.assertIsNone(privacy_pause.resume(new, now=start + 63)['active'])
        self.assertIsNone(privacy_pause.start(new, 30, now=start + 64)['active'])
        with self.assertRaises(KeyError):
            privacy_pause.resume(str(uuid4()), now=start)
        for minutes in (True, 0, 5, 61, 15.5):
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                privacy_pause.start(str(uuid4()), minutes, now=start)


@unittest.skipIf(main is None, 'Run camera pipeline cases with app dependencies')
class PrivacyCameraTests(IsolatedPrivacyDB):
    def setUp(self):
        super().setUp()
        # Camera state and request receipts must start from the isolated database.
        name = '_deskside_manual_privacy_test_app'
        spec = importlib.util.spec_from_file_location(name, main.__file__)
        self.app = importlib.util.module_from_spec(spec)
        modules = patch.dict(sys.modules, {name: self.app})
        modules.start()
        self.addCleanup(modules.stop)
        spec.loader.exec_module(self.app)
        self.key = uuid4()

    def pause(self):
        return self.app.pause_camera(self.app.PrivacyStart(id=self.key, minutes=1))

    def test_manual_pause_drops_cached_frames_calibration_and_queued_review(self):
        app = self.app
        app.publish(b'old image', ready=True, person='sit', presence='present', desk_edge=.4,
                    frame_clock=time.monotonic(), overlay={'boxes': ['old']})
        queued = Future()
        app.events.put_nowait(([b'queued image'], queued, app.vision_epoch, 'person', time.monotonic() + 5))
        self.pause()
        sample = app.camera_view()
        self.assertTrue(sample['manual_privacy'])
        self.assertTrue(sample['privacy_active'])
        self.assertNotIn('overlay', sample)
        self.assertIsNone(app.preview)
        self.assertTrue(queued.cancelled())
        for get in (app.live_frame, app.frame, lambda: app.calibrate('low')):
            with self.assertRaises(app.HTTPException):
                get()
        self.assertFalse(app.CAL.exists())
        app.resume_camera(self.key)
        with self.assertRaises(app.HTTPException):
            app.live_frame()

    def test_control_boundaries_exclude_previews_and_downtime_from_accounting(self):
        app = self.app
        wall = [100.0]
        with patch.object(app.time, 'time', side_effect=lambda: wall[0]), \
                patch.object(app.time, 'monotonic', side_effect=lambda: wall[0]):
            observed = {'person': 'stand', 'presence': 'present', 'desk': 'high', 'pose_method': 'keypoints'}
            app.capture_clock.write(app.capture_generation, observed, 100, 100, False)
            wall[0] = 101.25
            self.pause()
            wall[0] = 102.25
            app.capture_clock.write(app.capture_generation, app.PRIVATE_STATES, wall[0], wall[0], True)
            # A fresh process starts partway through the same pause.
            wall[0] = 114.0
            app.sync_privacy(startup=True)
            wall[0] = 115.5
            app.resume_camera(self.key)
            wall[0] = 116.5
            app.capture_clock.write(app.capture_generation, observed, wall[0], wall[0], False)
        with store.database() as db:
            rows = [tuple(r) for r in db.execute(
                "SELECT state,start,end FROM intervals WHERE kind='person' ORDER BY start")]
        self.assertEqual(rows, [('stand', 100, 101.25), ('privacy', 101.25, 102.25),
                                ('privacy', 114.0, 115.5), ('stand', 116.5, 116.5)])

    def test_late_cosmos_summary_is_not_saved_after_pause_and_resume(self):
        app = self.app
        entered, release = Event(), Event()

        def infer(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError('test did not release mock inference')
            app.stop.set()
            return 'standing', 'late summary'

        self.replace(app, 'describe_desk_frames', infer)
        app.events.put_nowait(([b'old image'], None, app.vision_epoch, None, time.monotonic() + 15))
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(app.cosmos)
            try:
                self.assertTrue(entered.wait(5))
                self.pause()
                app.resume_camera(self.key)
            finally:
                release.set()
                app.stop.set()
            job.result(timeout=5)
        with store.database() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM notes WHERE kind='cosmos'").fetchone()[0], 0)

    def test_restart_with_active_pause_keeps_watchdog_and_capture_idle(self):
        app = self.app
        privacy_pause.start(self.key, 15)
        fake_workers = MagicMock()

        async def load_stored_pause():
            with patch.object(app.threading, 'Thread', return_value=fake_workers):
                async with app.lifespan(app.app):
                    self.assertTrue(app.camera_view()['manual_privacy'])

        asyncio.run(load_stored_pause())
        start = app.camera_privacy()['server_now']
        elapsed = 0.
        beats = []

        def wait(delay):
            nonlocal elapsed
            elapsed += delay
            beats.append((100 + elapsed, app.last_publication))

        self.replace(app, 'time', SimpleNamespace(time=lambda: start + elapsed, monotonic=lambda: 100 + elapsed))
        self.replace(app, 'stop', SimpleNamespace(is_set=lambda: elapsed >= 65, wait=wait))
        self.replace(cv2, 'VideoCapture', Mock(side_effect=AssertionError('opened camera while private')))
        self.replace(vision, 'Vision', Mock(side_effect=AssertionError('initialized vision while private')))
        self.replace(app, 'SittingSession', Mock(side_effect=AssertionError('counted sitting while private')))
        app.capture_frames()
        self.assertTrue(all(0 <= now - beat <= 1 for now, beat in beats))
        with store.database() as db:
            samples = {r['kind']: (r['state'], r['end'] - r['start']) for r in db.execute('SELECT * FROM intervals')}
            self.assertEqual(db.execute("SELECT COUNT(*) FROM notes WHERE kind='reminder'").fetchone()[0], 0)
        self.assertEqual(samples['person'], ('privacy', 64))
        self.assertEqual(samples['presence'], ('privacy', 64))
        self.assertEqual(samples['pose_method'], ('unknown', 64))
        with patch.object(app, 'stop') as stop, patch.object(app.os, 'kill') as kill:
            stop.wait.side_effect = [False, True]
            app.capture_watchdog()
            kill.assert_not_called()
            # A native read that never returned cannot renew the private heartbeat.
            # Its wedged process must still be eligible for supervised restart.
            app.last_publication -= 35
            stop.wait.side_effect = [False, True]
            app.capture_watchdog()
            kill.assert_called_once_with(app.os.getpid(), app.signal.SIGTERM)

    def test_late_read_from_before_a_rapid_pause_and_resume_is_never_processed(self):
        app = self.app
        entered, release = Event(), Event()
        old, new = MagicMock(), MagicMock()
        image = np.full((32, 48, 3), 120, dtype=np.uint8)

        def read_old():
            entered.set()
            self.assertTrue(release.wait(5))
            return True, image

        def read_new():
            app.stop.set()
            return False, None

        old.read.side_effect = read_old
        new.read.side_effect = read_new
        self.replace(cv2, 'VideoCapture', Mock(side_effect=(old, new)))
        model = MagicMock()
        self.replace(vision, 'Vision', lambda: model)
        with ThreadPoolExecutor(max_workers=1) as pool:
            job = pool.submit(app.capture_frames)
            try:
                self.assertTrue(entered.wait(5))
                # The native read cannot exit until this test releases it. Return the
                # confirmation timeout immediately to exercise the fail-closed path.
                with patch.object(app.capture_released, 'wait_for', side_effect=lambda predicate, timeout: predicate()):
                    with self.assertRaises(app.HTTPException) as error:
                        self.pause()
                self.assertEqual(error.exception.status_code, 503)
                self.assertTrue(app.camera_view()['manual_privacy'])
                app.resume_camera(self.key)
                release.set()
                job.result(timeout=5)
            finally:
                release.set()
                app.stop.set()
        old.release.assert_called_once()
        new.read.assert_called_once()
        model.sample.assert_not_called()
        self.assertIsNone(app.preview)
        self.assertTrue(app.events.empty())

    def test_successful_pause_acknowledges_camera_release(self):
        app = self.app
        reading, waiting, release = Event(), Event(), Event()
        cap = MagicMock()
        self.replace(cv2, 'VideoCapture', lambda *args: cap)
        self.replace(vision, 'Vision', MagicMock)
        wait_for = app.capture_released.wait_for

        def read():
            reading.set()
            self.assertTrue(release.wait(5))
            return True, np.full((32, 48, 3), 120, dtype=np.uint8)

        def await_release(predicate, timeout):
            waiting.set()
            return wait_for(predicate, timeout)

        cap.read.side_effect = read
        self.replace(app.capture_released, 'wait_for', await_release)
        with ThreadPoolExecutor(max_workers=2) as pool:
            capture = pool.submit(app.capture_frames)
            try:
                self.assertTrue(reading.wait(5))
                pause = pool.submit(self.pause)
                self.assertTrue(waiting.wait(5))
                self.assertTrue(app.camera_view()['manual_privacy'])
                self.assertFalse(pause.done())
                release.set()
                self.assertEqual(pause.result(timeout=5)['active']['id'], str(self.key))
                cap.release.assert_called_once()
            finally:
                release.set()
                app.stop.set()
            capture.result(timeout=5)

    def test_pending_sitting_reminder_cannot_write_after_pause_and_resume(self):
        app = self.app
        cap = MagicMock()
        cap.read.return_value = (True, np.full((32, 48, 3), 120, dtype=np.uint8))
        self.replace(cv2, 'VideoCapture', lambda *args: cap)
        model = MagicMock()
        model.overlay = {}
        model.sample.return_value = ('unknown', 'sit', None, 1, False, 'unknown', None, 'keypoints')
        self.replace(vision, 'Vision', lambda: model)

        def crossed_threshold(*args):
            # Simulate a pause/resume after a frame was analyzed and just before
            # its sitting threshold callback has an opportunity to write a reminder.
            with patch.object(app.capture_released, 'wait_for', side_effect=lambda predicate, timeout: predicate()):
                with self.assertRaises(app.HTTPException) as deferred:
                    self.pause()
                self.assertEqual(deferred.exception.status_code, 503)
            app.resume_camera(self.key)
            app.stop.set()
            return True

        self.replace(app, 'SittingSession', lambda: SimpleNamespace(update=crossed_threshold))
        app.capture_frames()
        model.sample.assert_called_once()
        self.assertIsNone(app.preview)
        with store.database() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM notes WHERE kind='reminder'").fetchone()[0], 0)
            self.assertFalse(db.execute("SELECT 1 FROM intervals WHERE state IN ('sit','present')").fetchone())

    def test_resuming_manual_mode_still_detects_a_daytime_dark_cover(self):
        app = self.app
        self.pause()
        app.resume_camera(self.key)
        dark = np.zeros((32, 48, 3), dtype=np.uint8)
        cap = MagicMock()
        cap.read.return_value = (True, dark)
        self.replace(cv2, 'VideoCapture', lambda *args: cap)
        model = MagicMock()
        self.replace(vision, 'Vision', lambda: model)
        self.replace(app, 'time', SimpleNamespace(time=lambda: 1000 + cap.read.call_count,
                                                  monotonic=lambda: 100 + cap.read.call_count))
        self.replace(app, 'datetime', SimpleNamespace(now=lambda tz: SimpleNamespace(hour=12)))
        self.replace(app, 'stop', SimpleNamespace(is_set=lambda: cap.read.call_count >= 6, wait=lambda delay: None))
        app.capture_frames()
        view = app.camera_view()
        self.assertTrue(view['privacy_active'])
        self.assertTrue(view['vision_paused'])
        self.assertFalse(view['manual_privacy'])
        self.assertIsNone(app.preview)
        model.sample.assert_not_called()
        with self.assertRaises(app.HTTPException):
            app.live_frame()
