import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import assistant
import focus_sessions
import reminder_queue
import store


class NextReminderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch.object(store, 'ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.now = datetime(2026, 11, 22, 23, 59, 58, tzinfo=store.TZ).timestamp()
        clock = patch('time.time', return_value=self.now)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def seed(self, *rows):
        with store.database() as db:
            db.executemany('INSERT INTO tasks(id,kind,body,due,fired) VALUES (?,?,?,?,?)', rows)

    def contents(self):
        with store.database() as db:
            return list(db.iterdump())

    def ask(self, query='我的下一条定时提醒是什么？', *, voice=False, save=False, request_id=None):
        with patch('assistant.chat', side_effect=AssertionError('explicit reminder query called Qwen')):
            return assistant.respond(query, request_id or str(uuid4()), save, voice_reply=voice)

    def test_voice_and_text_read_only_first_future_reminder_across_midnight(self):
        due = self.now + 62
        self.seed(('z-late-tie', 'reminder', '同秒后排正文', due, 0),
                  ('a / 旧编号', 'reminder', '核对校样', due, 0),
                  ('later', 'reminder', '稍后正文', due + .25, 0),
                  ('past', 'reminder', '漏发的正文', self.now - 1, 0),
                  ('due-now', 'reminder', '此刻到期正文', self.now, 0),
                  ('fired', 'reminder', '曾投递的正文', self.now + 1, 1),
                  ('cancelled', 'reminder', '已取消的正文', self.now + 1, 2),
                  ('todo', 'todo', '私人待办正文', self.now + 1, 0),
                  ('blank-due', 'reminder', '无日期的正文', None, 0))
        focus_sessions.start(str(uuid4()), 1, title='未结算的专注正文', now=self.now - 61)
        with store.database() as db:
            db.execute("INSERT INTO notes(kind,body,at) VALUES ('reminder','历史久坐正文',?)", (self.now - 10,))
        before = self.contents()
        for query, voice in [('我的下一条定时提醒是什么？', False), ('下一个提醒是几点。', True)]:
            with self.subTest(query=query):
                answer = self.ask(query, voice=voice)
                self.assertIn('核对校样', answer)
                self.assertIn('北京时间 2026-11-23 00:01:00', answer)
                self.assertNotIn('正文', answer)
        self.assertEqual(self.contents(), before)

    def test_reads_live_queue_after_snooze_cancel_and_cutoff_without_delivering(self):
        first, second = self.now + 60, self.now + 180
        self.seed(('a', 'reminder', '修订第一稿', first, 0), ('b', 'reminder', '校对第二稿', second, 0))
        self.assertIn('修订第一稿', self.ask())
        reminder_queue.change('a', str(uuid4()), 'snooze', first, now=self.now)
        self.assertIn('校对第二稿', self.ask())
        reminder_queue.change('b', str(uuid4()), 'cancel', second, now=self.now)
        moved = self.ask()
        self.assertIn('修订第一稿', moved)
        self.assertIn('00:10', moved)
        self.clock.return_value = first + 600
        before = self.contents()
        empty = self.ask(voice=True)
        self.assertIn('提醒', empty)
        self.assertIn('没有', empty)
        self.assertNotIn('稿', empty)
        self.assertEqual(self.contents(), before)

    def test_whisper_clock_homophone_only_reads_the_queue_for_an_explicit_voice_question(self):
        self.seed(('upcoming', 'reminder', '核对样稿', self.now + 60, 0))
        transcript = '下一个提醒试几点。'
        self.assertIn('核对样稿', self.ask(transcript, voice=True))
        self.assertIsNone(assistant.next_reminder_answer(transcript))
        self.assertIsNone(assistant.next_reminder_answer('请翻译：' + transcript, voice=True))

    def test_saved_reply_replay_retains_original_answer_without_creating_another_note(self):
        self.seed(('old', 'reminder', '待发的校稿', self.now + 60, 0))
        key = str(uuid4())
        question = '我的下一条定时提醒是什么？'
        original = self.ask(question, request_id=key, save=True)
        self.assertIn('待发的校稿', original)
        reminder_queue.change('old', str(uuid4()), 'cancel', self.now + 60, now=self.now)
        self.assertEqual(self.ask(question, request_id=key, save=True), original)
        self.assertNotIn('待发的校稿', self.ask(question))
        with store.database() as db:
            rows = [tuple(row) for row in db.execute('SELECT kind,body FROM notes')]
            self.assertEqual(rows, [('dialogue', '你：' + question + '\n助手：' + original)])
            self.assertEqual(db.execute('SELECT count(*) FROM saved_replies').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM tasks').fetchone()[0], 1)

    def test_quotes_and_other_reminders_never_disclose_the_queue_or_capture_new_commands(self):
        self.seed(('personal', 'reminder', '私人队列正文', self.now + 70, 0))
        queries = ('翻译「我的下一条定时提醒是什么」', '请写一段以下一个提醒是几点开头的对话',
                   '下一个久坐提醒是几点？', '我的下一次专注提醒是什么？')
        with patch('assistant.chat', return_value='{"action":"answer","body":"常规回答"}') as model:
            for query in queries:
                self.assertIsNone(assistant.next_reminder_answer(query, now=self.now))
                self.assertEqual(assistant.respond(query, str(uuid4()), False), '常规回答', query)
            self.assertEqual(model.call_count, len(queries))
        command = '五分钟后提醒我查下一个提醒是几点'
        set_id = str(uuid4())
        with patch.object(store, 'datetime', wraps=datetime) as wall, \
                patch('assistant.chat', side_effect=AssertionError('explicit command called model')):
            wall.now.return_value = datetime.fromtimestamp(self.now, store.TZ)
            answer = assistant.respond(command, set_id, False)
        self.assertIn('已设置', answer)
        with store.database() as db:
            row = db.execute('SELECT kind,body,due,fired FROM tasks WHERE id=?', (set_id,)).fetchone()
            self.assertEqual(tuple(row), ('reminder', '查下一个提醒是几点', self.now + 300, 0))
            self.assertEqual(db.execute('SELECT count(*) FROM notes').fetchone()[0], 0)
