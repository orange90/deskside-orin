import copy
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import store
from dashboard_data import dashboard_snapshot, merge_intervals
from posture_continuity import bridge_posture_gaps


def rows(state='sit', gap=1):
    return [dict(kind='person', state=s, start=a, end=b) for s, a, b in
            [(state, 0, 1), ('unknown', 1, 1+gap), (state, 1+gap, 2+gap)]] + [
                dict(kind='presence', state='present', start=0, end=2+gap)]


class ContinuityTests(unittest.TestCase):
    def test_same_posture_bridges_and_preserves_provenance_and_input(self):
        for state in ('sit', 'stand'):
            original = rows(state)
            saved = copy.deepcopy(original)
            result = merge_intervals(bridge_posture_gaps(original))
            person = [r for r in result if r['kind'] == 'person']
            self.assertEqual(len(person), 1)
            self.assertEqual((person[0]['state'], person[0]['end']), (state, 3))
            self.assertEqual(person[0]['inferred_seconds'], 1)
            self.assertEqual(person[0]['inferred_ranges'], [{'start': 1, 'end': 2}])
            self.assertEqual(original, saved)

    def test_maximum_gap_and_no_open_ended_extrapolation(self):
        self.assertEqual(bridge_posture_gaps(rows(gap=3))[1]['state'], 'sit')
        self.assertEqual(bridge_posture_gaps(rows(gap=3.01))[1]['state'], 'unknown')
        self.assertEqual(bridge_posture_gaps(rows()[:2])[1]['state'], 'unknown')
        r = rows(); r[2]['end'] = r[2]['start']
        self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')

    def test_transition_away_privacy_and_manual_are_not_bridged(self):
        for state in ('stand', 'away', 'privacy', 'unknown'):
            r = rows(); r[2]['state'] = state
            self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')
        for state in ('away', 'privacy'):
            r = rows(); r[1]['state'] = state
            self.assertEqual(bridge_posture_gaps(r)[1]['state'], state)
        r = rows(); r[0]['source'] = 'manual'
        self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')

    def test_camera_outage_missing_presence_and_discontinuous_records(self):
        for state in ('away', 'unknown', 'privacy'):
            r = rows(); r[3]['state'] = state
            self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')
        self.assertEqual(bridge_posture_gaps(rows()[:3])[1]['state'], 'unknown')
        r = rows(); r[3]['end'] = 1.5
        self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')
        r = rows(); r[0]['end'] = .9
        self.assertEqual(bridge_posture_gaps(r)[1]['state'], 'unknown')

    def test_multiple_gaps_preserve_all_inferred_ranges(self):
        r = rows()[:3] + [dict(kind='person', state=s, start=a, end=b) for s,a,b in
                          [('unknown',3,4),('sit',4,5)]]
        r.append(dict(kind='presence',state='present',start=0,end=5))
        result = merge_intervals(bridge_posture_gaps(r))
        self.assertEqual(result[0]['inferred_seconds'], 2)
        self.assertEqual(len(result[0]['inferred_ranges']), 2)

    def test_dashboard_and_qa_agree_without_rewriting_database(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(store, 'ROOT', Path(tmp)):
            store.initialize()
            now = datetime.now(store.TZ)
            start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            with store.database() as db:
                db.executemany('INSERT INTO intervals VALUES (?,?,?,?)',
                               [(r['kind'],r['state'],start+r['start'],start+r['end']) for r in rows()])
            dashboard = dashboard_snapshot(now=now)
            self.assertEqual(dashboard['totals'], store.snapshot()['totals'])
            self.assertEqual(dashboard['totals']['sit'], 3)
            self.assertEqual(dashboard['hourly'][0]['sit'], 3)
            self.assertEqual(dashboard['continuity']['inferred_seconds'], 1)
            with store.database() as db:
                self.assertEqual(db.execute("SELECT count(*) FROM intervals WHERE state='unknown'").fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
