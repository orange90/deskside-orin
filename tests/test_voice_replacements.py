import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch
from uuid import uuid4

import assistant
import data_backup
import store
import voice_replacements as rules


class VoiceReplacementTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)
        self.source = self.workspace / '原数据'
        self.source.mkdir()
        root = patch.object(store, 'ROOT', self.source)
        root.start()
        self.addCleanup(root.stop)
        with store.database() as db:
            db.execute('CREATE TABLE notes(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL)')
            db.execute("INSERT INTO notes VALUES (1,'dialogue','老版本保留的对话',1)")
        store.initialize()

    def assert_old_notes_only(self):
        with store.database() as db:
            self.assertEqual([tuple(row) for row in db.execute('SELECT kind,body FROM notes')],
                             [('dialogue', '老版本保留的对话')])
            self.assertEqual(db.execute('SELECT count(*) FROM tasks').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT count(*) FROM saved_replies').fetchone()[0], 0)

    def test_retry_alias_delete_and_recreate_survive_restore(self):
        first, alias, new = [str(uuid4()) for _ in range(3)]
        heard, written = '估位', '工位'
        rule = {'id': first, 'heard': heard, 'written': written}
        self.assertEqual(rules.list_rules(), {'items': []})
        for key in (first, first, alias):
            self.assertEqual(rules.save_rule(key, '\n' + heard, written + '　'), dict(rule, active=True))
        self.assertEqual(rules.list_rules(), {'items': [rule]})
        with self.assertRaises(rules.RuleConflict):
            rules.save_rule(first, '其他词', written)
        with self.assertRaises(rules.RuleConflict):
            rules.save_rule(str(uuid4()), heard, '另一写法')
        self.assertEqual(rules.remove_rule(alias), {'id': first, 'active': False})
        self.assertEqual(rules.list_rules(), {'items': []})
        with store.database() as db:
            payloads = '\n'.join(line for line in db.iterdump() if line.startswith('INSERT'))
            self.assertNotIn(heard, payloads)
            self.assertNotIn(written, payloads)
        rules.save_rule(new, heard, written)
        archive = self.workspace / '快照.zip'
        destination = self.workspace / '已恢复'
        data_backup.create(archive, self.source)
        data_backup.restore(archive, destination)
        with patch.object(store, 'ROOT', destination):
            store.initialize()
            for stale_id in (first, alias):
                self.assertEqual(rules.save_rule(stale_id, heard, written), dict(rule, active=False))
                self.assertEqual(rules.remove_rule(stale_id), {'id': first, 'active': False})
            unknown = str(uuid4())
            self.assertEqual(rules.remove_rule(unknown), {'id': unknown, 'active': False})
            self.assertEqual(rules.list_rules(), {'items': [dict(rule, id=new)]})
            self.assert_old_notes_only()

    def test_phrase_boundaries_use_unicode_codepoints_and_preserve_inner_spaces(self):
        heard = '🌱' * 40
        written = '声  音。'
        key = str(uuid4())
        rules.save_rule(key, '　' + heard + ' ', ' ' + written)
        self.assertEqual(rules.list_rules()['items'], [{'id': key, 'heard': heard, 'written': written}])
        for invalid in ('', ' 　', '🌱' * 41, '甲\n乙', '甲\r乙', '甲\u2028乙', None, True):
            for pair in [(invalid, '新词'), ('识别词', invalid)]:
                with self.subTest(pair=repr(pair)[:18]), self.assertRaises(ValueError):
                    rules.save_rule(str(uuid4()), *pair)
        with self.assertRaises(ValueError):
            rules.save_rule(str(uuid4()), '原词', ' 原词　')
        for operation in (lambda: rules.save_rule('old-id', '甲', '乙'), lambda: rules.remove_rule('not-a-uuid')):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(len(rules.list_rules()['items']), 1)

    def test_concurrent_conflicting_edits_have_one_winner_and_one_active_rule(self):
        keys = [str(uuid4()) for _ in range(4)]
        targets = ['新名甲', '新名乙', '新名甲', '新名乙']
        gate = Barrier(len(keys))

        def save(pair):
            key, written = pair
            gate.wait(timeout=5)
            try:
                rules.save_rule(key, '原名', written)
                return written
            except rules.RuleConflict:
                return None

        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            outcomes = list(pool.map(save, zip(keys, targets)))
        items = rules.list_rules()['items']
        self.assertEqual(len(items), 1)
        winning = items[0]['written']
        self.assertEqual(outcomes, [target if target == winning else None for target in targets])
        self.assertIn(items[0]['id'], [k for k, target in zip(keys, targets) if target == winning])
        self.assert_old_notes_only()

    def test_only_one_concurrent_addition_can_take_last_slot(self):
        for i in range(31):
            rules.save_rule(str(uuid4()), f'识别{i}', f'改为{i}')
        gate = Barrier(2)

        def add(index):
            gate.wait(timeout=5)
            try:
                rules.save_rule(str(uuid4()), f'最后{index}', '改写')
                return True
            except rules.RuleConflict:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertCountEqual(list(pool.map(add, (0, 1))), [True, False])
        active = rules.list_rules()['items']
        self.assertEqual(len(active), 32)
        same = active[0]
        rules.save_rule(str(uuid4()), same['heard'], same['written'])
        self.assertEqual(len(rules.list_rules()['items']), 32)
        rules.remove_rule(same['id'])
        rules.save_rule(str(uuid4()), '释放名额', '替换成功')
        self.assertEqual(len(rules.list_rules()['items']), 32)

    def test_fast_voice_and_draft_both_keep_original_asr_without_executing_replacement(self):
        original = '这句话问的是什么？'
        command = '五分钟后提醒我喝水'
        key = str(uuid4())
        rules.save_rule(key, original, command)
        self.assert_old_notes_only()
        voice = assistant.Voice()

        def prepare_audio():
            temp = tempfile.TemporaryDirectory(dir=self.workspace)
            self.addCleanup(temp.cleanup)
            path = Path(temp.name) / 'query.pcm'
            path.write_bytes(bytes(3200))
            voice.recording = (temp, str(path), Mock(), 'simulated-source', False)

        with patch('assistant.model_registry.asr_path', return_value='test-model'), patch.object(
                voice, 'transcribe', return_value=original), patch('assistant.chat',
                return_value='{"action":"answer","body":"按原文回答。"}') as model:
            prepare_audio()
            draft = voice.draft()
            self.assertTrue(draft['draft'])
            self.assertEqual(draft['query'], original)
            model.assert_not_called()
            prepare_audio()
            with patch.object(rules, 'list_rules', side_effect=AssertionError('fast path read replacement table')):
                answer = voice.stop(str(uuid4()), False)
            model.assert_called_once()
            self.assertEqual(model.call_args.args[1]['messages'][-1]['content'], original)
            self.assertEqual((answer['query'], answer['answer']), (original, '按原文回答。'))
        self.assertEqual(rules.list_rules()['items'], [{'id': key, 'heard': original, 'written': command}])
        self.assert_old_notes_only()
