from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import assistant
import store


class WeekAnswerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.now = datetime(2026, 11, 22, 1, tzinfo=store.TZ)
        self.today = self.now.replace(hour=0).timestamp()

    def test_recent_calendar_week_includes_today_and_applies_personal_corrections(self):
        start = self.today - 6 * 86400
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)', [
                ('person', 'sit', start - 500, start - 1),
                ('person', 'sit', start, start + 12),
                ('person', 'privacy', start + 12, start + 17),
                ('person', 'unknown', start + 17, start + 21),
                ('presence', 'present', start, start + 21),
                ('person', 'estimated_sit', self.today, self.today + 6),
                ('person', 'stand', self.today + 6, self.today + 15),
                ('presence', 'present', self.today, self.today + 15),
            ])
        store.annotate_activity(start + 2, start + 6, 'away', 'confirmed')
        review = assistant.activity_answer('给我看近7天工位回顾', self.now)
        for part in ('2026-11-16 至 2026-11-22', '含今天', '2 天留有姿态记录（含未知和隐私时段）',
                     '坐姿 14 秒（含推算 6 秒）', '站姿 9 秒', '工位在位 27 秒',
                     '离开 4 秒', '隐私保护 5 秒', '姿态未知 4 秒', '未采集姿态'):
            self.assertIn(part, review)
        for query in ('最近七天坐站各多久？', '我过去一周在桌前坐和站分别多久', '请问近 7 天坐站时长是多少',
                      '最近一周坐了多久、站了多久？'):
            with self.subTest(query=query):
                answer = assistant.activity_answer(query, self.now)
                self.assertIn('坐姿 14 秒', answer)
                self.assertIn('站姿 9 秒', answer)
                self.assertNotIn('在位 27 秒', answer)

    def test_missing_clocks_do_not_turn_into_measured_zero(self):
        query = '最近一个星期在位时长？'
        self.assertIn('没有采集到', assistant.activity_answer(query, self.now))
        with store.database() as db:
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)',
                       ('presence', 'present', self.today, self.today + 11))
        self.assertIn('工位在位 11 秒', assistant.activity_answer(query, self.now))
        sitting = assistant.activity_answer('最近七天坐了多久', self.now)
        self.assertIn('只有 1 天留有工位在位记录，未采集姿态', sitting)
        self.assertNotIn('坐姿 0 秒', sitting)
        with store.database() as db:
            db.execute('DELETE FROM intervals')
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)',
                       ('person', 'sit', self.today, self.today + 8))
        self.assertIn('未采集到在位记录', assistant.activity_answer(query, self.now))

    def test_unrelated_or_ambiguous_intents_do_not_read_the_week(self):
        with patch('dashboard_data.activity_week', side_effect=AssertionError('unexpected scope')):
            for query in ('给我写一篇关于最近七天坐站时长的文章', '最近七天坐站时长怎么写进周报', '最近一周工位在哪里？',
                          '提醒我查看近7天工位回顾', '添加待办：统计最近七天坐站时长',
                          '最近一周应该坐多久？'):
                self.assertIsNone(assistant.activity_answer(query, self.now), query)
            for query in ('本周工位回顾', '上周坐站各多久', '昨天和近7天坐站各多久',
                          '近7天上午坐了多久', '近3天站立时长', '近几天站了多久'):
                self.assertEqual(assistant.activity_answer(query, self.now), assistant.RANGE_HELP, query)
