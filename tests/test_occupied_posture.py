import unittest
from state import OccupiedPosture
from posture_continuity import bridge_posture_gaps

class OccupiedPostureTests(unittest.TestCase):
    def test_stand_preserved_through_occlusion_but_not_after_leaving(self):
        p = OccupiedPosture()
        self.assertEqual(p.choose('stand','stand','cosmos',True,10,{}), ('stand','cosmos'))
        self.assertEqual(p.choose('unknown','unknown',None,True,16,{}), ('stand','estimate_recent'))
        self.assertEqual(p.choose('unknown','unknown',None,True,40,{}), ('stand','estimate_session'))
        self.assertEqual(p.choose('away','away','cosmos',False,41,{}), ('away','cosmos'))
        self.assertEqual(p.choose('unknown','away',None,True,42,{}), ('sit','estimate_default'))

    def test_limited_joints_on_return_can_vote_stand(self):
        pts=[dict(x=0,y=0,confidence=0) for _ in range(17)]
        for i,y in [(6,10),(12,60),(14,90)]:
            pts[i]=dict(x=100,y=y,confidence=.6)
        self.assertEqual(OccupiedPosture().choose('unknown','away',None,True,0,
                         {'keypoints':pts}), ('stand','estimate_joints'))

    def test_saved_estimate_contributes_time_and_retains_source(self):
        raw=dict(kind='person',state='estimated_stand',start=10,end=20)
        self.assertEqual(bridge_posture_gaps([raw]),[{**raw,'state':'stand','source':'estimated'}])
        self.assertEqual(raw['state'],'estimated_stand')
