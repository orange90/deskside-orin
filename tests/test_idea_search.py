import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import dashboard_data
import focus_handoffs
import focus_sessions
import idea_search
import store


class IdeaSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = patch.object(store, 'ROOT', Path(directory.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 23, tzinfo=store.TZ)
        self.stamp = self.midnight.timestamp()
        clock = patch('time.time', return_value=self.stamp + 86400)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def add(self, db, at, body, kind='idea'):
        return str(db.execute('INSERT INTO notes(kind,body,at) VALUES (?,?,?)', (kind, body, at)).lastrowid)

    def test_original_text_across_beijing_midnight_only_ideas_link_to_their_day(self):
        first_text = '  空椅子\n\n#写作  [[下个场景]]。\t'
        next_text = '午夜时分，空椅子！'
        large_id = 2**53 + 1
        with store.database() as db:
            db.execute("INSERT INTO notes(id,kind,body,at) VALUES (?,'idea',?,?)",
                       (large_id, first_text, self.stamp - .0008))
            at_midnight = self.add(db, self.stamp, next_text)
            another = self.add(db, self.stamp, next_text)
            for kind in ('dialogue', 'reminder', 'cosmos'):
                self.add(db, self.stamp, '空椅子只见于' + kind, kind)
        store.task('旧待办', 'todo', '空椅子')
        session = str(uuid4())
        focus_sessions.start(session, 5, title='专注写作', now=self.stamp)
        focus_handoffs.save(session, '空椅子出现在续做便签', now=self.stamp + 1)
        result = idea_search.search_ideas('空椅子')
        self.assertEqual(result, {'items': [
            {'id': another, 'date': '2026-11-23', 'at': self.stamp, 'body': next_text},
            {'id': at_midnight, 'date': '2026-11-23', 'at': self.stamp, 'body': next_text},
            {'id': str(large_id), 'date': '2026-11-22', 'at': self.stamp - .0008, 'body': first_text},
        ], 'next_cursor': None})
        for day in ('2026-11-22', '2026-11-23'):
            notes = dashboard_data.dashboard_snapshot(day, self.midnight + timedelta(minutes=1))['notes']
            self.assertEqual([n['idea_id'] for n in notes if n['kind'] == 'idea'],
                             [item['id'] for item in result['items'] if item['date'] == day])
            self.assertTrue(all('idea_id' not in n for n in notes if n['kind'] != 'idea'))

    def test_midnight_floats_and_future_entries_match_the_day_view_cutoff(self):
        now = self.midnight + timedelta(hours=2)
        cutoff = now.timestamp()
        self.clock.return_value = cutoff
        with store.database() as db:
            before = self.add(db, math.nextafter(self.stamp, -math.inf), '午夜边界前')
            midnight = self.add(db, self.stamp, '正好午夜边界')
            after = self.add(db, math.nextafter(self.stamp, math.inf), '午夜边界后')
            at_cutoff = self.add(db, cutoff, '边界处当前时刻')
            future = self.add(db, cutoff + 86400, '边界外的服务快进记录')
        items = idea_search.search_ideas('边界')['items']
        self.assertEqual([(item['id'], item['date']) for item in items],
                         [(after, '2026-11-23'), (midnight, '2026-11-23'), (before, '2026-11-22')])
        for day in ('2026-11-22', '2026-11-23'):
            notes = dashboard_data.dashboard_snapshot(day, now)['notes']
            self.assertEqual([note['idea_id'] for note in notes],
                             [item['id'] for item in items if item['date'] == day])
        self.clock.return_value = cutoff + 1
        ids = [item['id'] for item in idea_search.search_ideas('边界')['items']]
        self.assertIn(at_cutoff, ids)
        self.assertNotIn(future, ids)

    def test_substrings_are_literal_trimmed_and_ascii_case_insensitive(self):
        with store.database() as db:
            wanted = self.add(db, self.stamp, '图表 QwEn：增长 100%_X，保留 [[草稿]]。')
            partial = self.add(db, self.stamp, '增长 100%A，草稿')
            self.add(db, self.stamp, '增长 1000X，正文')
            self.add(db, 'legacy-invalid-time', '增长 100%_X，坏时间')
        for query, expected in [('　 \tqWEn　\n', [wanted]), ('%_', [wanted]), ('100%_x', [wanted]),
                                ('100%', [partial, wanted]), ('[[草稿]]', [wanted]), ('别的章节', [])]:
            with self.subTest(query=query):
                result = idea_search.search_ideas(query)
                self.assertEqual([item['id'] for item in result['items']], expected)
                self.assertIsNone(result['next_cursor'])

    def test_keyset_paging_survives_new_writes_with_tied_and_submillisecond_times(self):
        tied = self.stamp + .12345
        with store.database() as db:
            initial = [self.add(db, tied, f'山谷草稿 {i}') for i in range(41)]
            # Its ID is newest but its timestamp is earlier in the same millisecond.
            oldest = self.add(db, self.stamp + .12340, '山谷的旧版本')
        expected = list(reversed(initial)) + [oldest]
        first = idea_search.search_ideas('山谷')
        self.assertEqual([item['id'] for item in first['items']], expected[:20])
        self.assertIsInstance(first['next_cursor'], str)
        with self.assertRaises(ValueError):
            idea_search.search_ideas('草稿', before=first['next_cursor'])
        gate = Barrier(2)

        def insert_newer():
            gate.wait(timeout=5)
            with store.database() as db:
                self.add(db, tied, '山谷新笔记，同一时间')
                self.add(db, tied + 1, '山谷新笔记，稍后')

        with ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(insert_newer)
            gate.wait(timeout=5)
            second = idea_search.search_ideas('山谷', before=first['next_cursor'])
            writer.result(timeout=10)
        self.assertEqual([item['id'] for item in second['items']], expected[20:40])
        third = idea_search.search_ideas('山谷', before=second['next_cursor'])
        self.assertEqual([item['id'] for item in third['items']], expected[40:])
        self.assertIsNone(third['next_cursor'])
        self.assertEqual(idea_search.search_ideas('　山谷 ', before=first['next_cursor']), second)

    def test_codepoint_limits_invalid_cursors_and_search_leaves_no_history(self):
        query = '🌱' * 80
        with store.database() as db:
            key = self.add(db, self.stamp, '扉页' + query)
            original = list(db.iterdump())
        self.assertEqual(idea_search.search_ideas('   ' + query + '\n')['items'][0]['id'], key)
        self.assertEqual(idea_search.search_ideas('只在本次搜索的词')['items'], [])
        for bad in ('', ' \n　\t', query + '🌱', None, True, ['草稿']):
            with self.subTest(invalid=repr(bad)[:8]), self.assertRaises(ValueError):
                idea_search.search_ideas(bad)
        for bad in ('', '!不是游标!', -1, True, ['x']):
            with self.subTest(cursor=bad), self.assertRaises(ValueError):
                idea_search.search_ideas('扉页', before=bad)
        with store.database() as db:
            self.assertEqual(list(db.iterdump()), original)
