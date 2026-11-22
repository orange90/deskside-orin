import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import dashboard_data
import store


def duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def report_rows(document):
    table = [line.strip().strip('|').split('|') for line in document.splitlines() if line.startswith('|')]
    headings = [name.strip() for name in table[0][1:]]
    return {row[0].strip(' *'): dict(zip(headings, (value.strip() for value in row[1:])))
            for row in table[2:]}


class ActivityReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.midnight = datetime(2026, 11, 22, tzinfo=store.TZ)
        self.now = self.midnight + timedelta(minutes=5)

    def insert(self, rows):
        start = self.midnight.timestamp()
        with store.database() as db:
            db.executemany('INSERT INTO intervals VALUES (?,?,?,?)', [
                (kind, state, start + left, start + right) for kind, state, left, right in rows
            ])

    def test_report_matches_daily_clocks_and_excludes_private_contents(self):
        self.insert([('person', 'estimated_sit', -10, 30), ('person', 'estimated_stand', 40, 50),
                     ('person', 'privacy', 50, 57), ('person', 'unknown', 57, 62),
                     ('presence', 'present', -10, 62), ('person', 'sit', 295, 500)])
        start = self.midnight.timestamp()
        store.annotate_activity(start + 5, start + 10, 'away', 'private-correction-reason')
        store.annotate_activity(start + 49, start + 59, 'away', 'overlaps privacy')
        store.annotate_activity(start + 296, start + 305, 'away', 'pose only, crosses report cutoff')
        store.task('private-task-id', 'todo', 'private-task-body')
        private = ['private-dialogue', 'private-camera-summary', 'private-reminder']
        with store.database() as db:
            db.executemany('INSERT INTO notes(kind,body,at) VALUES (?,?,?)',
                           [(kind, body, start + 12) for kind, body in zip(('dialogue', 'cosmos', 'reminder'), private)])
        filename, text = dashboard_data.weekly_markdown(now=self.now)
        self.assertEqual(filename, 'deskside-week-2026-11-22.md')
        self.assertIn('北京时间', text)
        self.assertIn('00:05:00', text)
        for marker in [*private, 'private-task-id', 'private-task-body', 'private-correction-reason']:
            self.assertNotIn(marker, text)

        rows = report_rows(text)
        fields = {'坐姿': 'sit', '站姿': 'stand', '工位在位': 'present',
                  '离开': 'away', '姿态未知': 'unknown', '隐私保护': 'privacy'}
        weekly = Counter()
        days = [(self.midnight - timedelta(days=n)).date().isoformat() for n in reversed(range(7))]
        self.assertEqual(list(rows), days + ['七日合计'])
        for day in days:
            totals = dashboard_data.dashboard_snapshot(day, self.now)['totals']
            weekly.update(totals)
            for title, state in fields.items():
                expected = duration(totals.get(state, 0)) if totals else '未采集'
                self.assertEqual(rows[day][title], expected, (day, state))
        for title, state in fields.items():
            self.assertEqual(rows['七日合计'][title], duration(weekly[state]))
        self.assertEqual(rows[days[-1]]['姿态未采集'], duration(300 - 57))
        self.assertEqual(rows['七日合计']['姿态未采集'], duration(6 * 86400 + 300 - 67))
        self.assertRegex(text, r'(?:坐姿.*推算|推算.*坐姿)[^\n]*00:00:35')
        self.assertRegex(text, r'(?:站姿.*推算|推算.*站姿)[^\n]*00:00:09')
        self.assertRegex(text, r'姿态记录中的校正时长[^\n]*?00:00:12')
        self.assertRegex(text, r'在位记录中的校正时长[^\n]*?00:00:08')

    def test_empty_week_and_presence_only_preserve_absence_of_posture_data(self):
        _, empty = dashboard_data.weekly_markdown('2026-11-21', self.now)
        total = report_rows(empty)['七日合计']
        for title in ('工位在位', '离开', '隐私保护', '坐姿', '站姿', '姿态未知'):
            self.assertEqual(total[title], '未采集')
        self.assertEqual(total['姿态未采集'], '168:00:00')

        self.insert([('presence', 'present', -86400 + 120, -86400 + 132)])
        _, text = dashboard_data.weekly_markdown('2026-11-21', self.now)
        rows = report_rows(text)
        for key in ('2026-11-21', '七日合计'):
            self.assertEqual(rows[key]['工位在位'], '00:00:12')
            self.assertEqual(rows[key]['坐姿'], '未采集')
            self.assertEqual(rows[key]['离开'], '未采集')
        self.assertEqual(rows['2026-11-21']['姿态未采集'], '24:00:00')
