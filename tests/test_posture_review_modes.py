import unittest
from unittest.mock import patch
from assistant import describe_desk_frames

class ReviewModeTests(unittest.TestCase):
    def test_unclear_and_negated_descriptions_abstain(self):
        for text in ('The person is not sitting.', 'The person is standing or sitting.',
                     'The chair is standing behind the desk.', 'Their upper body is visible.'):
            with self.subTest(text=text), patch('assistant.chat', return_value=text):
                self.assertEqual(describe_desk_frames([{}], review_mode='person')[0], 'present')

    def test_empty_candidate_cannot_supply_posture(self):
        for reply, label in [('sitting', 'unknown'), ('occluded', 'occluded'), ('empty', 'away')]:
            with self.subTest(reply=reply), patch('assistant.chat', return_value=reply) as model:
                self.assertEqual(describe_desk_frames([{}], review_mode='occupancy')[0], label)
                self.assertEqual(model.call_count, 1)
