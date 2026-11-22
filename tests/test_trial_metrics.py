import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from trial_metrics import summarize


def marks(pairs, origin=0):
    return [{'at': origin + at, 'state': state} for at, state in pairs]


def intervals(rows, origin=0):
    return [{'kind': kind, 'state': state, 'start': origin + start, 'end': origin + end}
            for kind, state, start, end in rows]


class TrialMetricsTests(unittest.TestCase):
    def test_cross_midnight_matrix_estimates_privacy_and_gaps(self):
        origin = datetime(2026, 11, 22, 23, 59, 55, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
        labels = marks([(-1, 'away'), (1, 'sit'), (10, 'stand'), (19, 'away'),
                       (24, 'pause'), (27, 'sit'), (30, 'away')], origin)
        raw = intervals([('person', 'sit', -80, 4), ('person', 'estimated_sit', 4, 8),
                         ('person', 'stand', 8, 12), ('person', 'unknown', 12, 14),
                         ('person', 'estimated_stand', 14, 18), ('person', 'away', 20, 24),
                         ('person', 'estimated_sit', 24, 90), ('presence', 'privacy', 7, 10),
                         ('person', 'privacy', 9, 11), ('presence', 'present', -150, 150)], origin)
        data = summarize(labels, reversed(raw), origin, origin + 30)
        self.assertEqual(data['matrix'], {
            'sit': dict(sit=9, stand=0, away=0, unknown=0, missing=0, privacy=3),
            'stand': dict(sit=0, stand=5, away=0, unknown=2, missing=1, privacy=1),
            'away': dict(sit=0, stand=0, away=4, unknown=0, missing=1, privacy=0),
        })
        self.assertEqual(data['estimated_matrix'], {
            'sit': {'sit': 6, 'stand': 0}, 'stand': {'sit': 0, 'stand': 4}, 'away': {'sit': 0, 'stand': 0}})
        self.assertEqual((data['labeled_seconds'], data['eligible_seconds'], data['matched_seconds']), (26, 22, 18))
        self.assertEqual((data['estimated_seconds'], data['matched_estimated_seconds']), (10, 10))
        self.assertEqual((data['paused_seconds'], data['unlabeled_seconds']), (3, 1))
        self.assertEqual(data['transitions'], {
            'total': 2, 'measured': 0, 'missed': 0, 'unmeasurable': 2, 'estimated_hits': 0, 'p95_seconds': None})

    def test_switch_latency_uses_last_equal_time_mark_and_first_hit(self):
        labels = marks([(0, 'sit'), (1, 'away'), (1, 'stand'), (1.5, 'stand'), (5, 'away'),
                        (8, 'sit'), (12, 'stand'), (16, 'away')])
        raw = intervals([('person', 'sit', 0, 1), ('person', 'unknown', 1, 2.25),
                         ('person', 'estimated_stand', 2.25, 5), ('person', 'estimated_sit', 5, 8),
                         ('person', 'stand', 8, 8.5), ('person', 'sit', 8.5, 10),
                         ('person', 'unknown', 10, 13), ('person', 'stand', 13, 16),
                         ('person', 'away', 16, 20), ('presence', 'privacy', 19, 20)])
        data = summarize(labels, raw, 0, 20)
        self.assertEqual(data['transitions'], {
            'total': 5, 'measured': 4, 'missed': 1, 'unmeasurable': 0, 'estimated_hits': 1, 'p95_seconds': 1.25})
        self.assertEqual(data['estimated_matrix']['away']['sit'], 3)
        self.assertEqual(data['estimated_seconds'] - data['matched_estimated_seconds'], 3)

    def test_pause_ends_an_unmatched_switch_and_resume_is_not_a_switch(self):
        labels = marks([(1, 'sit'), (3, 'stand'), (4, 'pause'), (9, 'away')])
        raw = intervals([('person', 'sit', 1, 4), ('person', 'away', 9, 20)])
        result = summarize(labels, raw, 0, 20)
        self.assertEqual((result['unlabeled_seconds'], result['paused_seconds']), (1, 5))
        self.assertEqual(result['transitions'], {
            'total': 1, 'measured': 0, 'missed': 1, 'unmeasurable': 0, 'estimated_hits': 0, 'p95_seconds': None})

    def test_p95_is_nearest_rank_over_measured_switches(self):
        labels = [(0, 'sit')]
        raw = [('person', 'sit', 0, 30)]
        for delay in range(1, 21):
            at = delay * 30
            target = 'stand' if delay % 2 else 'sit'
            labels.append((at, target))
            raw.extend([('person', 'unknown', at, at + delay),
                        ('person', target, at + delay, at + 30)])
        transitions = summarize(marks(labels), intervals(raw), 0, 630)['transitions']
        self.assertEqual((transitions['total'], transitions['measured'], transitions['p95_seconds']), (20, 20, 19))

    def test_private_time_without_person_beats_missing_and_empty_trials_have_no_latency(self):
        only_private = intervals([('presence', 'privacy', 1, 5)])
        data = summarize(marks([(0, 'sit')]), only_private, 0, 5)
        self.assertEqual((data['matrix']['sit']['missing'], data['matrix']['sit']['privacy']), (1, 4))
        self.assertEqual((data['eligible_seconds'], data['matched_seconds']), (1, 0))
        empty = summarize([], [], 12, 12)
        self.assertEqual((empty['labeled_seconds'], empty['eligible_seconds'], empty['unlabeled_seconds']), (0, 0, 0))
        self.assertIsNone(empty['transitions']['p95_seconds'])
        with self.assertRaises(ValueError):
            summarize([], [], 13, 12)
