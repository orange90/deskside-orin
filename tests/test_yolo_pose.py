import unittest
from unittest.mock import patch
try:
    import numpy as np
    from yolo_pose import letterbox, decode
    import vision
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Requires production numpy/opencv; also run on Orin')
class YoloPoseTests(unittest.TestCase):
    def test_letterbox_rgb_and_padding(self):
        image=np.zeros((720,1280,3),dtype=np.uint8);image[:]=[10,20,30]
        x,t=letterbox(image)
        self.assertEqual(t,(.5,0,140));self.assertEqual(x.shape,(1,3,640,640))
        np.testing.assert_allclose(x[0,:,140,0],[30/255,20/255,10/255],atol=1e-7)
        np.testing.assert_allclose(x[0,:,0,0],114/255,atol=1e-7)

    def test_decode_nms_and_inverse_coordinates(self):
        raw=np.zeros((1,56,8400),np.float32)
        for i,score in enumerate([.9,.8]):
            raw[0,:5,i]=[320,320,100,200,score]
            raw[0,5:,i]=np.tile([300,300,.7],17)
        boxes,points=decode(raw,(720,1280,3),(.5,0,140))
        self.assertEqual(len(boxes),1)
        np.testing.assert_allclose(boxes[0,:4],[540,160,740,560])
        np.testing.assert_allclose(points[0,0],[600,320,.7],atol=1e-6)
        raw.fill(0);self.assertEqual(decode(raw,(720,1280,3),(.5,0,140))[1].shape,(0,17,3))
        with self.assertRaises(ValueError):decode(np.zeros((1,57,300)),(720,1280,3),(.5,0,140))

    def test_roi_offset_same_frame_and_reset_on_absence(self):
        v=object.__new__(vision.Vision)
        v.model_info={'name':'YOLO26m-Pose'};v.boxes=None;v.scene=None;v.background=None
        from state import RelativeRise
        v.rise=RelativeRise();v.overlay=None
        class Fake:
            def __init__(self):self.calls=0
            def __call__(self,image):
                self.calls+=1
                if self.calls==3:return np.empty((0,5)),np.empty((0,17,3))
                return np.array([[10,20,60,100,.9]]),np.tile([10*self.calls,20,.8],(1,17,1))
        v.model=Fake();frame=np.zeros((200,400,3),np.uint8)
        with patch.object(vision,'background_stable',return_value=False):
            v.sample(frame,{'person_roi':[.25,.1,.75,.9]})
            self.assertEqual(v.overlay['keypoints'][0]['x'],110)
            self.assertEqual(v.overlay['keypoints'][0]['y'],40)
            v.sample(frame,{'person_roi':[.25,.1,.75,.9]})
            self.assertEqual(v.overlay['keypoints'][0]['x'],120)
            v.sample(frame,{'person_roi':[.25,.1,.75,.9]})
            self.assertEqual(v.overlay['keypoints'],[]);self.assertEqual(v.overlay['detections'],[])
