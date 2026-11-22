import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import store
from activity_annotations import apply_annotations
from assistant import activity_answer
from dashboard_data import dashboard_snapshot


class ActivityAccounting(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.t = self.midnight.timestamp()

    def insert(self, rows):
        with store.database() as db:
            db.executemany("INSERT INTO intervals VALUES (?,?,?,?)",
                           [(kind, state, self.t+a, self.t+b) for kind,state,a,b in rows])

    def test_gap_spans_midnight_in_both_reports(self):
        self.insert([('person','sit',-20,-1), ('person','unknown',-1,2),
                     ('person','sit',2,10), ('presence','present',-20,10)])
        now = self.midnight + timedelta(minutes=1)
        for date, seconds, filled in [('2026-11-21',20,1), ('2026-11-22',10,2)]:
            day = dashboard_snapshot(date, now)
            self.assertEqual(day['totals']['sit'], seconds)
            self.assertEqual(day['continuity']['inferred_seconds'], filled)
        self.assertEqual(store.snapshot(now)['totals'], dashboard_snapshot(now=now)['totals'])

    def test_future_or_sample_extending_past_now_never_inflates_answers(self):
        self.insert([('person','stand',-5,40), ('person','sit',80,100)])
        now = self.midnight + timedelta(seconds=8)
        self.assertEqual(store.snapshot(now)['totals'], {'stand': 8})
        self.assertEqual(dashboard_snapshot(now=now)['totals'], {'stand': 8})

    def test_privacy_overrides_annotations_on_both_timelines(self):
        self.insert([('person','unknown',0,60), ('person','privacy',60,120),
                     ('person','unknown',120,180), ('presence','unknown',0,180)])
        store.annotate_activity(self.t+20, self.t+150, 'sit', 'operator')
        day = dashboard_snapshot(now=self.midnight+timedelta(hours=1))
        self.assertEqual(day['totals']['present'], 70)
        self.assertEqual(day['totals']['sit'], 70)
        self.assertEqual(day['totals']['privacy'], 60)
        self.assertEqual(day['totals']['unknown'], 50)

    def test_presence_is_split_at_privacy_edges_in_unsorted_records(self):
        rows = [dict(kind='person', state='privacy', start=a, end=b)
                for a, b in [(13, 18), (3, 10), (5, 7), (18, 20)]]
        rows.append(dict(kind='presence', state='present', start=5, end=22))
        shown = apply_annotations(rows, [dict(id=1, state='away', start=0, end=30)])
        self.assertEqual([(r['start'], r['end'], r['state']) for r in shown if r['kind']=='presence'],
                         [(5, 7, 'privacy'), (7, 10, 'privacy'), (10, 13, 'away'),
                          (13, 18, 'privacy'), (18, 20, 'privacy'), (20, 22, 'away')])
        self.assertEqual(rows[-1]['state'], 'present')

    def test_duplicate_and_reversed_clock_never_double_count(self):
        store.persist('person','sit',self.t,False)
        store.persist('person','stand',self.t,True)
        store.persist('person','stand',self.t+5,True)
        store.persist('person','sit',self.t+2,True)
        store.persist('person','sit',self.t+5,True)
        store.persist('person','sit',self.t+9,True)
        self.assertEqual(store.snapshot(self.midnight+timedelta(hours=1))['totals'],
                         {'stand':5, 'sit':4})

    def test_live_sample_is_atomic_and_handles_wall_clock_jump(self):
        store.persist_sample({'person':'sit','presence':'present'},self.t, False)
        store.persist_sample({'person':'sit','presence':'present'},self.t+1, True)
        with self.assertRaises(sqlite3.IntegrityError):
            store.persist_sample({'person':'stand','presence':None},self.t+2, True)
        store.persist_sample({'person':'sit','presence':'present'},self.t+3600, True)
        store.persist_sample({'person':'sit','presence':'present'},self.t+3601, True)
        self.assertEqual(store.snapshot(self.midnight+timedelta(hours=2))['totals'],
                         {'sit':2, 'present':2, 'unknown':3599})

    def test_qa_uses_requested_day_and_discloses_estimated_portion(self):
        self.insert([('person','estimated_sit',-180,-60), ('person','stand',-60,0),
                     ('person','sit',0,40), ('presence','present',-180,40)])
        now = self.midnight + timedelta(minutes=1)
        answer = activity_answer('昨天坐站各多久？', now)
        self.assertIn('昨天在桌前坐了 2 分钟', answer)
        self.assertIn('2 分钟 为推算', answer)
        self.assertIn('站了 1 分钟', answer)
        self.assertIn('40 秒', activity_answer('今天坐了多久？', now))
        self.assertIn('工位检测到人员在位 3 分钟',
                      activity_answer('11月21日工位在位时长？', now))
        self.assertNotIn('40 秒', activity_answer('昨天和今天坐了多久？', now))

    def test_day_query_does_not_swallow_other_intents(self):
        now = self.midnight + timedelta(hours=2)
        for query in ('我的工位有多少个显示器？', '我坐火车到广州需要多久？',
                      '多久合适站起来活动？', '五分钟后提醒我看坐姿时长'):
            self.assertIsNone(activity_answer(query, now), query)
        self.insert([('person','sit',0,60)])
        for query in ('我今天坐了多久', '告诉我今天坐了多长时间。', '今天在桌前坐了多少分钟？',
                      '今天坐姿时长是多少？'):
            self.assertIn('1 分钟', activity_answer(query, now), query)
        self.assertNotIn('1 分钟', activity_answer('今天上午我坐了多久？', now))
        self.assertIn('还没有', activity_answer('明天坐了多久？', now))

    def test_offline_day_no_records_even_when_nearby_days_have_rows(self):
        self.insert([('person','sit',-180,-3), ('person','stand',86401,86500)])
        now = self.midnight + timedelta(days=2)
        self.assertEqual(dashboard_snapshot(self.midnight.date().isoformat(), now)['totals'], {})
        self.assertIn('没有已记录', activity_answer('2026-11-22坐了多久', now))
