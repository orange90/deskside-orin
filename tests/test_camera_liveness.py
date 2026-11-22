"""Exercise the same service functions used by HTTP; never open a real camera."""
import unittest
from unittest.mock import Mock, patch

try:
    import main
except ModuleNotFoundError as exc:
    if exc.name not in ('fastapi', 'pydantic', 'starlette'):
        raise
    main = None


@unittest.skipIf(main is None, 'Run on Orin with application dependencies')
class CameraLivenessTests(unittest.TestCase):
    def test_stuck_capture_stops_both_previews_and_calibration(self):
        previous = dict(ready=True, frame_clock=4.0, captured_at=21., desk_edge=.4,
                        person='sit', presence='present', overlay={'keypoints': ['old']})
        with patch.object(main, 'latest', previous), patch.object(main, 'preview', b'old'), \
             patch.object(main.time, 'monotonic', return_value=10.):
            status = main.camera_view()
            self.assertFalse(status['ready'])
            self.assertNotIn('overlay', status)
            for f in (main.live_frame, main.frame, lambda: main.calibrate('low')):
                with self.assertRaises(main.HTTPException):
                    f()
        self.assertEqual(previous['person'], 'sit')

    def test_wedged_camera_triggers_existing_supervisor(self):
        with patch.object(main,'last_publication',1.), patch.object(main.time,'monotonic',return_value=35.), \
             patch.object(main,'stop',Mock(wait=Mock(return_value=False))), patch.object(main.os,'kill') as stop_process:
            main.capture_watchdog()
            stop_process.assert_called_once_with(main.os.getpid(),main.signal.SIGTERM)
