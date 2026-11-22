import queue
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from state import FrameLight, OccupiedPosture, PersonTrackEvidence, SittingSession

try:
    import cv2
    import numpy as np
    import main
    import vision
except ModuleNotFoundError:
    main = None


class TimingTests(unittest.TestCase):
    def test_ancient_detection_does_not_renew_a_weak_box(self):
        track = PersonTrackEvidence()
        observations = [track.observe(1, t in (0,1), [], t/5) for t in range(80)]
        self.assertNotIn(True, observations)
        self.assertFalse(track.observe(1, 1, [], 16))
        self.assertTrue(track.observe(1, 1, [], 16.2))

    def test_ambiguous_knee_does_not_overwrite_a_standing_session_with_sit(self):
        posture = OccupiedPosture("stand", 1)
        joints = [dict(x=0, y=0, confidence=0.) for _ in range(17)]
        for i, y in [(6, 10), (12, 60), (14, 63)]:
            joints[i] = dict(x=100, y=y, confidence=.7)
        self.assertEqual(posture.choose("unknown", "unknown", None, True, 30,
                                        {"keypoints": joints}), ("stand", "estimate_session"))

    def test_one_weak_frame_cannot_flip_sitting_but_sustained_standing_can(self):
        posture = OccupiedPosture("sit", 9)
        joints = [dict(x=0, y=0, confidence=0.) for _ in range(17)]
        for i, y in [(6, 10), (12, 60), (14, 105)]:
            joints[i] = dict(x=100, y=y, confidence=.7)
        overlay = {"keypoints": joints}
        for t in (10, 10.3):
            self.assertFalse(posture.observe_joints(posture.joint_hint(overlay), t))
            self.assertEqual(posture.choose("unknown", "sit", None, True, t, overlay),
                             ("sit", "estimate_recent"))
        self.assertTrue(posture.observe_joints(posture.joint_hint(overlay), 10.7))
        self.assertEqual(posture.choose("unknown", "sit", None, True, 10.7, overlay),
                         ("stand", "estimate_joints"))
        self.assertEqual(posture.choose("stand", "sit", "cosmos", True, 10.8, overlay),
                         ("stand", "estimate_pending"))
        self.assertEqual(posture.confirmed, "sit")
        self.assertEqual(posture.choose("stand", "stand", "cosmos", True, 12.1, overlay),
                         ("stand", "cosmos"))

    def test_old_bright_frame_cannot_end_privacy(self):
        light = FrameLight()
        for t, brightness in [(0,0), (3,0), (4,120), (30,120)]:
            self.assertTrue(light.update(t,brightness))
        self.assertFalse(light.update(32,120))

    def test_short_and_long_uncertain_gaps_in_reminders(self):
        s = SittingSession()
        s.update("sit",0,60)
        s.update("sit",55,60)
        s.update("unknown",1,60)
        s.update("sit",1,60)
        self.assertTrue(s.update("sit",5,60))
        s.update("unknown",20,60)
        self.assertEqual((s.sent,s.seconds),(False,0))


@unittest.skipIf(main is None, "Run camera pipeline cases with app dependencies on Orin")
class CameraPipelineTests(unittest.TestCase):
    def test_inputs_produce_consistent_status_and_atomic_samples(self):
        # Simulate initial single sitter, unobserved gap, multiple occupants,
        # a lone stander, an empty ROI, and a daytime camera cover.
        steps = ([(1,1,"sit")]*20 + [(5,2,"unknown")] + [(1,2,"unknown")]*15
                 + [(1,1,"stand")]*20 + [(1,0,"away")]*86 + [(1,-1,"unknown")]*25)
        frames, writes = [], []
        i, now = -1, 1000.
        bright = np.full((64,96,3),120,np.uint8)
        cap, stop = MagicMock(), MagicMock()
        def read():
            nonlocal i, now
            i += 1
            dt, count, _ = steps[i]
            now += 4. if dt == 5 else .2
            return True, bright if count >= 0 else np.zeros_like(bright)
        cap.read.side_effect = read
        stop.is_set.side_effect = lambda:i+1>=len(steps)
        model = MagicMock()
        model.model_info = {"name":"fixture"}
        model.model.last_ms = 0.
        model.overlay = {"keypoints": [], "detections": []}
        def sample(*args):
            _, n, pose=steps[i]
            return ("unknown",pose,None,n,n,"unknown",None,
                    "keypoints" if pose in ("sit","stand") else None)
        model.sample.side_effect = sample
        def persist(states, wall, continuous):
            writes.append((i,states,continuous))
        clock=SimpleNamespace(monotonic=lambda:now,time=lambda:now)
        with ExitStack() as stack:
            def replace(obj,name,value):
                stack.enter_context(patch.object(obj,name,value))
            replace(cv2,"VideoCapture",lambda *args:cap)
            replace(vision,"Vision",lambda:model)
            replace(main,"time",clock)
            replace(main,"datetime",SimpleNamespace(now=lambda _:SimpleNamespace(hour=9)))
            replace(main,"stop",stop)
            replace(main,"CAL",Path("/dev/null/nonexistent"))
            replace(main,"events",queue.Queue(1))
            replace(main,"publish",lambda _,**values:frames.append(values))
            replace(main.store,"persist_sample",persist)
            replace(main.store,"note",MagicMock())
            main.capture_frames()
        self.assertTrue(writes)
        written_frames = {ix for ix, _, _ in writes}
        for ix, (old, new) in enumerate(zip(frames, frames[1:]), 1):
            if (old["person"], old["presence"]) != (new["person"], new["presence"]):
                self.assertIn(ix, written_frames)
        self.assertTrue(any(not cont and ix==20 for ix,_,cont in writes))
        self.assertEqual(frames[19]['person'],'sit')
        self.assertEqual(frames[20]['presence'],'unknown')
        self.assertEqual((frames[35]['presence'], frames[35]['person'], frames[35]['pose_source']),
                         ('present', 'unknown', 'multiple_people'))
        self.assertEqual(frames[54]['person'],'stand')
        self.assertEqual(frames[60]['pose_source'],'estimate_recent')
        self.assertEqual(frames[-26]['presence'],'unknown')
        self.assertTrue(frames[-1]['privacy_active'])
        for r in frames:
            self.assertEqual(r['person'] in ('sit','stand'),
                             r['presence']=='present' and r['people'] <= 1)
        for ix, sample, _ in writes:
            self.assertEqual(sample['person'] in ('sit','stand','estimated_sit','estimated_stand'),
                             sample['presence']=='present' and steps[ix][1] <= 1)

    def test_pose_change_invalidates_only_replies_from_older_frames(self):
        r = main.PostureReview()
        r.label, r.label_at, r.valid_until = "sit", 10, 17
        r.requested, r.pending = 12, Future()
        r.pending.set_result("standing")
        r.invalidate_before(11)
        self.assertIsNone(r.label)
        with patch.object(main, "events", queue.Queue(1)):
            self.assertEqual(r.update(lambda: b"frame", 12.1, empty_candidate=False, interval=1.5), "stand")
        self.assertEqual(r.label_at, 12)
        stale = r.pending = Future()
        r.invalidate_before(13)
        self.assertTrue(stale.cancelled())
        self.assertIsNone(r.pending)
        self.assertIsNone(r.label)

    def test_late_clear_empty_vote_is_discarded(self):
        r=main.PostureReview()
        r.absence.update("away",5,True)
        r.absence.update("away",8,True)
        r.requested=11
        r.pending=Future()
        r.pending.set_result("away")
        with patch.object(main,"events",queue.Queue(1)):
            self.assertIsNone(r.update(lambda:b"frame",20,empty_candidate=True,interval=3))
        self.assertEqual(r.absence.votes,0)
