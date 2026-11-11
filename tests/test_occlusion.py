import ast,sys,unittest,queue
from pathlib import Path
from concurrent.futures import Future
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from state import AbsenceEvidence,PresenceContinuity

class OcclusionTests(unittest.TestCase):
    def test_three_distinct_empty_reviews(self):
        e=AbsenceEvidence()
        self.assertFalse(e.update('away',0,True));self.assertFalse(e.update('away',3,True));self.assertTrue(e.update('away',6,True))
    def test_occlusion_resets_empty_votes(self):
        e=AbsenceEvidence();e.update('away',0,True);e.update('away',3,True)
        self.assertFalse(e.update('occluded',6,True));self.assertFalse(e.update('away',9,True))
        self.assertFalse(e.update('away',12,False));self.assertEqual(e.votes,0)
    def test_reused_and_stale_frames_not_confirmation(self):
        e=AbsenceEvidence();e.update('away',0,True)
        for _ in range(100):self.assertFalse(e.update('away',0,True))
        self.assertEqual(e.votes,1);self.assertFalse(e.update('away',20,True));self.assertEqual(e.votes,1)
    def test_visibility_gap_is_bounded(self):
        p=PresenceContinuity();self.assertEqual(p.update('unknown',0)[0],'unknown')
        self.assertEqual(p.update('present',1)[0],'present');self.assertEqual(p.update('unknown',15)[0],'present')
        self.assertEqual(p.update('unknown',17)[0],'unknown')
        self.assertEqual(p.update('away',18)[0],'away');self.assertIsNone(p.last_positive)
    def test_posture_review_cached_absence_cannot_override_detection(self):
        source=Path(sys.path[0],'main.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PostureReview')
        import logging
        context={'Future':Future,'AbsenceEvidence':AbsenceEvidence,'queue':queue,'events':queue.Queue(1),'vision_epoch':0,'LOG':logging.getLogger('test')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<PostureReview>','exec'),context)
        review=context['PostureReview']();review.label='away';review.valid_until=100;review.requested=1
        self.assertIsNone(review.update(lambda:b'frame',2,empty_candidate=False,interval=5))
    def test_complete_occluded_frame_stays_not_away(self):
        source=Path(sys.path[0],'main.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='PostureReview')
        import logging
        context={'Future':Future,'AbsenceEvidence':AbsenceEvidence,'queue':queue,'events':queue.Queue(1),'vision_epoch':0,'LOG':logging.getLogger('test')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<PostureReview>','exec'),context)
        review=context['PostureReview']()
        for t in range(3,30,3):
            review.pending=Future();review.pending.set_result('occluded');review.requested=t
            self.assertEqual(review.update(lambda:b'frame',t+.5,empty_candidate=True,interval=3),'occluded')

if __name__ == '__main__':
    unittest.main(verbosity=2)
