import unittest
from state import PersonTrackEvidence, PresenceContinuity

class EntryTests(unittest.TestCase):
    def test_single_candidate_cannot_start_hold(self):
        t, c = PersonTrackEvidence(), PresenceContinuity()
        for i in range(100):
            confirmed = t.observe(i % 15 == 0, 1, [], i/6)
            self.assertEqual(c.update('present' if confirmed else 'unknown', i/6)[0], 'unknown')

    def test_supported_track_and_stale_track(self):
        t = PersonTrackEvidence()
        for i in range(10):
            yes = t.observe(1, 0, [{'confidence':.8}]*3, i/6)
        self.assertTrue(yes)
        self.assertFalse(t.observe(1, 1, [], 10))
        self.assertFalse(t.observe(0, 0, [], 11))
        for i in range(12):
            self.assertFalse(t.observe(1, 0, [], 12+i/6))
