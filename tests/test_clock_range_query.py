import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import assistant
import store


class ClockRangeQueryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.now = datetime(2026, 11, 22, 4, tzinfo=store.TZ)

    def insert(self, kind, state, start, duration):
        stamp = datetime.fromisoformat(start).replace(tzinfo=store.TZ).timestamp()
        with store.database() as db:
            db.execute('INSERT INTO intervals VALUES (?,?,?,?)', (kind, state, stamp, stamp + duration))

    def test_real_speech_transcript_and_activity_review_use_precise_interval(self):
        self.insert('person', 'sit', '2026-11-21T08:59:55', 68)
        self.insert('person', 'estimated_sit', '2026-11-21T09:01:03', 8)
        self.insert('person', 'stand', '2026-11-21T09:01:11', 5)
        self.insert('person', 'privacy', '2026-11-21T09:01:16', 7)
        self.insert('person', 'away', '2026-11-21T09:02:00', 2)
        self.insert('person', 'unknown', '2026-11-21T09:02:02', 3)
        self.insert('presence', 'present', '2026-11-21T09:00:00', 12)
        self.insert('person', 'sit', '2026-11-21T11:00:00', 1000)
        for query in ('昨天9点到11点。我坐了多久。', '昨天九点至十一点，我在桌前坐了多久？'):
            with self.subTest(query=query):
                answer = assistant.activity_answer(query, self.now.astimezone(timezone.utc))
                self.assertIn('2026-11-21 09:00 至 2026-11-21 11:00', answer)
                self.assertIn('坐姿 1 分钟 11 秒（含推算 8 秒）', answer)
                self.assertNotIn('站姿', answer)
        answer = assistant.activity_answer('请给我看昨天9点到11点工位回顾', self.now)
        for value in ('站姿 5 秒', '工位在位 12 秒', '姿态未知 3 秒', '离开 2 秒', '隐私保护 7 秒',
                      '姿态未采集 1 小时 58 分钟 32 秒', '在位未采集 1 小时 59 分钟 48 秒'):
            self.assertIn(value, answer)
        stamp = datetime(2026, 11, 21, 9, 2, 2, tzinfo=store.TZ).timestamp()
        store.annotate_activity(stamp, stamp + 2, 'sit', '现场确认')
        corrected = assistant.activity_answer('昨天9点到11点工位活动回顾', self.now)
        self.assertIn('姿态未知 1 秒', corrected)
        self.assertIn('2 秒 经本人校正', corrected)

    def test_cross_midnight_dayparts_and_chinese_minutes(self):
        self.insert('person', 'sit', '2026-11-21T22:00:00', 4)
        self.insert('person', 'stand', '2026-11-22T01:59:57', 8)
        for query in ('昨天22点到今天凌晨2点坐站各多久', '昨天晚上十点至次日凌晨两点坐和站分别多久',
                      '从2026-11-21 22:00到第二天02:00坐站时长'):
            with self.subTest(query=query):
                result = assistant.activity_answer(query, self.now)
                for fragment in ('2026-11-21 22:00 至 2026-11-22 02:00', '坐姿 4 秒', '站姿 3 秒'):
                    self.assertIn(fragment, result)
        answer = assistant.activity_answer('昨天下午两点半到三点零五分在位多久', self.now)
        self.assertIn('2026-11-21 14:30 至 2026-11-21 15:05', answer)

    def test_independent_recording_clocks_never_report_absence_as_measured_zero(self):
        ask = '2026-11-21 09:15 到 10:45 工位在位多久'
        with patch.object(assistant, 'chat', side_effect=AssertionError('unexpected model call')):
            self.insert('person', 'sit', '2026-11-21T10:00:00', 5)
            result = assistant.respond(ask, str(uuid4()), save=False, voice_reply=True)
            self.assertIn('未采集在位记录', result)
            self.assertNotIn('工位在位 0 秒', result)
            self.insert('presence', 'present', '2026-11-21T10:00:00', 2)
            self.assertIn('工位在位 2 秒', assistant.respond(ask, str(uuid4()), save=False))
        self.insert('presence', 'present', '2026-11-21T10:30:00', 3)
        result = assistant.activity_answer('昨天十点半到十一点我坐了多久', self.now)
        self.assertIn('未采集姿态记录', result)
        self.assertNotIn('坐姿 0 秒', result)

    def test_explicit_invalid_or_ambiguous_ranges_never_fall_back_to_today(self):
        for query, hint in (('昨天22点到2点我坐了多久', '次日'), ('9点到11点坐站各多久', '注明查询日期'),
                            ('今天三点到八点坐站各多久', '当前时间'), ('前天一点到今天两点站了多久', '24 小时'),
                            ('昨天8:70到10:20坐姿时长', '有效的北京时间'),
                            ('昨天晚上十二点到今天两点坐了多久', '有效的北京时间'),
                            ('2026-02-30 09:00 到 10:00 坐姿时长', '有效的北京时间')):
            with self.subTest(query=query):
                self.assertIn(hint, assistant.activity_answer(query, self.now))

    def test_unknown_and_private_measurements_qualify_only_the_requested_clock(self):
        self.insert('person', 'unknown', '2026-11-21T09:00:00', 7200)
        self.insert('person', 'privacy', '2026-11-21T11:00:00', 1800)
        self.insert('presence', 'unknown', '2026-11-21T09:00:00', 1200)
        self.insert('presence', 'privacy', '2026-11-21T09:20:00', 1800)
        entire_unknown = assistant.activity_answer('昨天9点到11点。我坐了多久。', self.now)
        self.assertIn('记录中未识别到坐姿', entire_unknown)
        self.assertIn('姿态未知 2 小时', entire_unknown)
        self.assertNotIn('在位状态未知', entire_unknown)
        query = '昨天9点到11点半'
        posture = assistant.activity_answer(query + '站了多久', self.now)
        self.assertIn('姿态未知 2 小时', posture)
        self.assertIn('姿态记录隐私保护 30 分钟', posture)
        presence = assistant.activity_answer(query + '工位在位多久', self.now)
        self.assertIn('记录中未检测到人员在位', presence)
        self.assertIn('在位状态未知 20 分钟', presence)
        self.assertIn('在位记录隐私保护 30 分钟', presence)
        self.assertNotIn('姿态未知', presence)
        review = assistant.activity_answer(query + '工位回顾', self.now)
        for detail in ('姿态未知 2 小时', '姿态记录隐私保护 30 分钟',
                       '在位状态未知 20 分钟', '在位记录隐私保护 30 分钟'):
            self.assertEqual(review.count(detail), 1, detail)

    def test_unrelated_queries_do_not_consult_local_activity(self):
        for query in ('翻译「昨天9点到11点坐了多久」', '昨天9点到11点我坐了多久，帮我写一篇文章。',
                      '添加待办：昨天9点到11点我坐了多久', '昨天9点到11点我站多久合适？',
                      '昨天九点到十一点在车站待了多久', '提醒我查一下昨天9点到11点工位活动回顾',
                      '昨天9点到11点车站多久到机场？', '昨天9点到11点。朋友说我站了多久。',
                      '昨天9点到11点小明坐了多久？'):
            with self.subTest(query=query), patch('dashboard_data.activity_range', side_effect=AssertionError):
                self.assertIsNone(assistant.activity_answer(query, self.now))
