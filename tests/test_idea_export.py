import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import dashboard_data
import store


class IdeaExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.start = datetime(2026, 11, 22, tzinfo=store.TZ)

    def test_downloads_only_selected_days_ideas_in_capture_order(self):
        stamp = self.start.timestamp()
        first = "#标题\n\n保留  两个空格和 [[工位手记]]。"
        second = "结尾这一笔，等明天补！"
        with store.database() as db:
            db.executemany('INSERT INTO notes(kind,body,at) VALUES (?,?,?)', [
                ('idea', '上一天的灵感', stamp - 0.01),
                ('idea', first, stamp), ('dialogue', '已存对话', stamp),
                ('idea', second, stamp), ('cosmos', '私人画面描述', stamp + 20),
                ('idea', '最后一条', stamp + 86400 - .01), ('reminder', '私人提醒', stamp + 100),
                ('idea', '下一天的灵感', stamp + 86400)])
            db.execute("INSERT INTO tasks VALUES ('task','todo','私人待办',NULL,0)")
        day = self.start.date().isoformat()
        filename, md = dashboard_data.daily_ideas_markdown(day, self.start + timedelta(days=1))
        self.assertEqual(filename, 'deskside-ideas-2026-11-22.md')
        self.assertIn('## 00:00:00\n\n' + first, md)
        self.assertLess(md.index(first), md.index(second))
        self.assertIn('## 23:59:59\n\n最后一条', md)
        for outside in ('上一天','下一天','私人','已存对话'):
            self.assertNotIn(outside, md)
        _, partial = dashboard_data.daily_ideas_markdown(day, self.start + timedelta(seconds=10))
        self.assertIn(first, partial)
        self.assertNotIn('最后一条', partial)

    def test_export_rejects_invalid_or_future_dates_and_handles_an_empty_day(self):
        for day in ('2026-W47-7', '../data.sqlite3', '2026-11-23'):
            with self.subTest(day=day), self.assertRaises(ValueError):
                dashboard_data.daily_ideas_markdown(day, self.start)
        _, md = dashboard_data.daily_ideas_markdown('2026-11-21', self.start)
        self.assertIn('2026-11-21', md)
        self.assertNotIn('## ', md)

    def test_day_picker_reaches_old_ideas_when_camera_was_off(self):
        stamp = self.start.timestamp()
        with store.database() as db:
            db.execute("INSERT INTO notes(kind,body,at) VALUES ('idea','跨日续写',?)", (stamp - 100,))
            db.execute("INSERT INTO intervals VALUES ('person','sit',?,?)", (stamp, stamp + 5))
        view = dashboard_data.dashboard_snapshot(now=self.start + timedelta(seconds=10))
        self.assertEqual(view['first_date'], '2026-11-21')
