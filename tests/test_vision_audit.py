import unittest
from unittest.mock import patch
from state import RelativeRise

try:
    import numpy as np
    import vision
    from yolo_pose import decode
except ImportError:
    vision = None


class MotionAudit(unittest.TestCase):
    def test_motion_expires_during_stationary_track_and_disconnected_frames(self):
        for delta in (.2, 100):
            track = RelativeRise()
            for i, y in enumerate((200, 200, 160, 160)):
                track.update((50,y,100), i*.2, True)
            self.assertEqual(track.pose, 'stand')
            for i in range(1, 18):
                result = track.update((50,160,100), .6+i*delta, True)
            self.assertIsNone(result)

    def test_one_frame_vertical_glitch_does_not_report_a_posture(self):
        track = RelativeRise()
        for i,y in enumerate((200,200,150,200,200)):
            self.assertIsNone(track.update((50,y,100),i*.2,True))

    def test_scale_jump_is_not_a_rise(self):
        track = RelativeRise()
        for i, a in enumerate([(100,100,50), (100,100,50), (100,50,100)]):
            posture = track.update(a, i*.2, True)
        self.assertIsNone(posture)


@unittest.skipIf(vision is None, 'Requires Orin NumPy/OpenCV')
class VisionAudit(unittest.TestCase):
    @staticmethod
    def legs(shoulder, hip, knee, ankle):
        points = np.zeros((17,2))
        scores = np.zeros(17)
        points[[5,11,13,15]] = [shoulder,hip,knee,ankle]
        scores[[5,11,13,15]] = .9
        return points, scores

    def test_leaning_forward_on_vertical_legs(self):
        points, scores = self.legs((15,50), (55,50), (55,90), (55,130))
        self.assertEqual(vision.leg_posture(points,scores), 'stand')
        scores[15] = 0
        self.assertIsNone(vision.leg_posture(points,scores))

    def test_genuine_seated_pose_and_degenerate_projected_thigh(self):
        points, scores = self.legs((20,20), (20,80), (60,80), (65,120))
        self.assertEqual(vision.leg_posture(points,scores), 'sit')
        points[13] = (23,80)
        self.assertIsNone(vision.leg_posture(points,scores))

    def test_landmark_in_letterbox_padding_is_invisible(self):
        raw = np.zeros((1,56,8400),np.float32)
        raw[0,:5,0]=[320,320,200,200,.9]
        raw[0,5:,0]=np.tile([300,32,.9],17)
        _, pts=decode(raw,(720,1280),(.5,0,140))
        self.assertFalse(pts[0,:,2].any())

    def test_disjoint_boxes_and_two_people_do_not_share_pose(self):
        v = object.__new__(vision.Vision)
        v.model_info={'name':'test'}
        v.boxes=v.scene=v.background=None
        v.rise=RelativeRise()
        pts, scores=self.legs((20,20), (20,80), (60,80), (65,120))
        joints=np.column_stack((pts,scores))
        left=[0,10,80,150,.8]; right=[170,15,250,156,.9]
        inputs=iter(([left],[],[right],[right,[1,10,85,160,.30]]))
        def predict(_image):
            boxes=np.asarray(next(inputs)).reshape(-1,5)
            return boxes,np.repeat(joints[None],len(boxes),axis=0)
        v.model=predict
        frame=np.zeros((190,260,3),np.uint8)
        with patch.object(vision,'background_stable',return_value=True):
            self.assertEqual(v.sample(frame,{})[1],'sit')
            self.assertEqual(v.sample(frame,{})[1],'away')
            v.sample(frame,{})
            self.assertTrue(v.overlay['subject_changed'])
            self.assertEqual(v.sample(frame,{})[1],'unknown')
            self.assertFalse(v.overlay['keypoints'])
            self.assertIsNone(v.background)
