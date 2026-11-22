import signal
import subprocess
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import assistant
import store


class VoiceDraftTests(unittest.TestCase):
    def setUp(self):
        database_dir = tempfile.TemporaryDirectory()
        self.addCleanup(database_dir.cleanup)
        root = patch.object(store, 'ROOT', Path(database_dir.name))
        root.start()
        self.addCleanup(root.stop)
        store.initialize()
        self.voice = assistant.Voice()
        self.voice.set_mute = Mock()
        model = patch('assistant.model_registry.asr_path', return_value='test-model')
        model.start()
        self.addCleanup(model.stop)

    def record(self, pcm=b'\x10\x01' * 1600):
        self.voice.set_mute.reset_mock()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'query.pcm'
        path.write_bytes(pcm)
        proc = Mock()
        proc.poll.return_value = None
        self.voice.recording = (temp, str(path), proc, 'simulated-source', True)
        return path, proc

    def assert_released(self, path):
        self.assertIsNone(self.voice.recording)
        self.assertFalse(self.voice.busy.is_set())
        self.assertFalse(path.parent.exists())
        self.voice.set_mute.assert_any_call('simulated-source', True, value=1)

    def test_draft_converts_audio_and_never_runs_the_recognized_command(self):
        samples = b'\x10\x01' * 1700
        path, proc = self.record(samples + b'\x00')
        query = '添加待办：整理工位。'

        def recognize(wav_path, model):
            self.assertEqual(model, 'test-model')
            self.assertTrue(self.voice.busy.is_set())
            with wave.open(str(wav_path), 'rb') as recording:
                self.assertEqual((recording.getnchannels(), recording.getsampwidth(),
                                  recording.getframerate()), (1, 2, 16000))
                self.assertEqual(recording.readframes(recording.getnframes()), samples)
            return query

        with patch.object(self.voice, 'transcribe', side_effect=recognize), patch(
                'assistant.respond', side_effect=AssertionError('a draft must await submission')):
            result = self.voice.draft()
        self.assertEqual(result['query'], query)
        self.assertIs(result['draft'], True)
        self.assertGreaterEqual(result['seconds'], 0)
        self.assertNotIn('answer', result)
        self.assertNotIn('answer_ready_seconds', result)
        self.assertEqual(store.snapshot()['tasks'], [])
        self.assertEqual(store.snapshot()['notes'], [])
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        self.assert_released(path)

    def test_original_stop_executes_and_saves_the_reply_once(self):
        path, _ = self.record()
        query = '添加待办：整理工位。'
        with patch.object(self.voice, 'transcribe', return_value=query):
            result = self.voice.stop('existing-client', True)
        self.assertEqual(result['query'], query)
        self.assertEqual(result['answer'], '已添加待办，整理工位。')
        self.assertEqual(result['answer_ready_seconds'], result['seconds'])
        with self.assertRaisesRegex(ValueError, '没有活动录音'):
            self.voice.stop('duplicate', True)
        data = store.snapshot()
        self.assertEqual([r['id'] for r in data['tasks']], ['existing-client'])
        self.assertEqual([r['kind'] for r in data['notes']], ['dialogue'])
        self.assert_released(path)

    def test_empty_audio_empty_transcript_and_asr_failure_release_capture(self):
        for pcm, recognized, expected in [(b'', 'unused', ValueError),
                                          (bytes(3200), ' \n ', ValueError),
                                          (bytes(3200), OSError('ASR unavailable'), OSError)]:
            with self.subTest(recognized=recognized):
                path, _ = self.record(pcm)
                respond = Mock(side_effect=AssertionError('unexpected action'))
                transcribe = Mock(return_value=recognized)
                if isinstance(recognized, Exception):
                    transcribe.side_effect = recognized
                with patch.object(self.voice, 'transcribe', transcribe), patch('assistant.respond', respond):
                    with self.assertRaises(expected):
                        self.voice.draft()
                if not pcm:
                    transcribe.assert_not_called()
                respond.assert_not_called()
                self.assert_released(path)

    def test_draft_owns_recording_until_transcription_finishes(self):
        path, _ = self.record()
        started, release = Event(), Event()

        def recognize(_wav_path, _model):
            started.set()
            self.assertTrue(release.wait(5))
            return '今天站了多久'

        with ThreadPoolExecutor(max_workers=1) as pool, patch.object(self.voice, 'transcribe', side_effect=recognize):
            pending = pool.submit(self.voice.draft)
            try:
                self.assertTrue(started.wait(5))
                with self.assertRaisesRegex(ValueError, '没有活动录音'):
                    self.voice.stop('second-stop', True)
                with self.assertRaisesRegex(ValueError, '正在处理'):
                    self.voice.start()
                self.assertTrue(self.voice.busy.is_set())
            finally:
                release.set()
            self.assertTrue(pending.result(timeout=5)['draft'])
        self.assert_released(path)

    def test_unresponsive_recorder_is_killed_and_reaped_before_returning_draft(self):
        path, proc = self.record()
        proc.pid = 101
        proc.communicate.side_effect = [subprocess.TimeoutExpired('recorder', 5), (b'', b'')]
        with patch.object(self.voice, 'transcribe', return_value='查活动记录'), patch('assistant.os.killpg') as kill:
            self.assertTrue(self.voice.draft()['draft'])
        kill.assert_called_once_with(proc.pid, signal.SIGKILL)
        self.assertEqual(proc.communicate.call_count, 2)
        self.assert_released(path)

    def test_failed_temp_cleanup_still_unlocks_voice(self):
        path, _ = self.record()
        temp = self.voice.recording[0]
        with patch.object(self.voice, 'transcribe', return_value='测试'), patch.object(
                temp, 'cleanup', side_effect=OSError('simulated filesystem error')):
            with self.assertRaises(OSError):
                self.voice.draft()
        self.assertFalse(self.voice.busy.is_set())
        self.assertIsNone(self.voice.recording)
        temp.cleanup()
        self.assertFalse(path.parent.exists())
