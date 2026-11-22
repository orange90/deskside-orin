import hashlib
import json
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile

import focus_handoffs
import focus_sessions
import store


CLI = Path(__file__).resolve().parents[1] / 'data_backup.py'
PRIVATE_TEXT = '只在本地出现的试验草稿'


class DataBackupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / '原数据'
        self.source.mkdir()
        self.archive = self.root / '工位备份.zip'
        self.restored = self.root / '恢复结果'

    def cli(self, *args):
        result = subprocess.run([sys.executable, '-B', str(CLI), *map(str, args)],
                                capture_output=True, text=True, cwd=self.root, umask=0, timeout=20)
        self.assertNotIn(PRIVATE_TEXT, result.stdout + result.stderr)
        return result

    def ok(self, *args):
        result = self.cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)

    def initialize(self):
        with patch.object(store, 'ROOT', self.source):
            store.initialize()

    def create(self):
        self.ok('create', self.archive, '--source-dir', self.source)

    def test_online_snapshot_includes_committed_wal_and_excludes_uncommitted_work(self):
        self.initialize()
        source_db = self.source / 'data.sqlite3'
        with closing(sqlite3.connect(source_db)) as writer:
            self.assertEqual(writer.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
            writer.execute('PRAGMA wal_autocheckpoint=0')
            writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            writer.execute('CREATE TABLE new_feature_items(id PRIMARY KEY, body TEXT NOT NULL)')
            writer.execute('INSERT INTO new_feature_items VALUES (1,?)', (PRIVATE_TEXT,))
            writer.commit()
            writer.execute("INSERT INTO new_feature_items VALUES (2,'未提交')")
            bare_copy = self.root / 'main-file-alone.sqlite3'
            shutil.copyfile(source_db, bare_copy)
            with closing(sqlite3.connect(bare_copy)) as bare:
                self.assertIsNone(bare.execute("SELECT 1 FROM sqlite_master WHERE name='new_feature_items'").fetchone())
            self.create()
            self.ok('verify', self.archive)
            self.ok('restore', self.archive, self.restored)
            with closing(sqlite3.connect(self.restored / 'data.sqlite3')) as recovered:
                self.assertEqual(recovered.execute('SELECT * FROM new_feature_items').fetchall(), [(1, PRIVATE_TEXT)])
            writer.commit()
        with closing(sqlite3.connect(source_db)) as original:
            self.assertEqual(original.execute('SELECT count(*) FROM new_feature_items').fetchone(), (2,))

    def test_current_records_configuration_whitelist_and_private_modes(self):
        self.initialize()
        source_id, idea_id = str(uuid4()), str(uuid4())
        with patch.object(store, 'ROOT', self.source):
            focus_sessions.start(source_id, 5, title='校稿', now=100)
            focus_handoffs.save(source_id, PRIVATE_TEXT + '，留给下一次。', now=110)
            store.record_idea(idea_id, '灵感：' + PRIVATE_TEXT, PRIVATE_TEXT, False)
        config = {'calibration.json': '{\r\n  "坐姿": 1.250\r\n}\r\n'.encode(),
                  'model-selection.json': b'{ "asr" : "small" }\n'}
        for name, body in config.items():
            (self.source / name).write_bytes(body)
        for name in ('unused.json', 'notes.txt', 'camera.jpg', 'weights.bin'):
            (self.source / name).write_text(PRIVATE_TEXT)
        self.create()
        self.ok('verify', self.archive)
        self.assertEqual(stat.S_IMODE(self.archive.stat().st_mode), 0o600)
        payload_files = {'data.sqlite3', *config}
        with ZipFile(self.archive) as archive:
            self.assertCountEqual(archive.namelist(), payload_files | {'manifest.json'})
            self.assertTrue(all(entry.compress_type == ZIP_STORED for entry in archive.infolist()))
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(set(manifest['files']), payload_files)
            for name in payload_files:
                body = archive.read(name)
                self.assertEqual(manifest['files'][name]['bytes'], len(body))
                self.assertEqual(manifest['files'][name]['sha256'], hashlib.sha256(body).hexdigest())
        self.ok('restore', self.archive, self.restored)
        self.assertEqual({p.name for p in self.restored.iterdir()}, payload_files)
        self.assertEqual(stat.S_IMODE(self.restored.stat().st_mode), 0o700)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in self.restored.iterdir()))
        for name, body in config.items():
            self.assertEqual((self.restored / name).read_bytes(), body)
        with patch.object(store, 'ROOT', self.restored):
            self.assertEqual(focus_handoffs.pending()['items'][0]['body'], PRIVATE_TEXT + '，留给下一次。')
            with store.database() as db:
                idea = db.execute('SELECT kind,body FROM notes JOIN idea_requests ON note_id=id WHERE request_id=?',
                                  (idea_id,)).fetchone()
                self.assertEqual(tuple(idea), ('idea', PRIVATE_TEXT))

    def test_legacy_database_and_absent_optional_files_remain_usable(self):
        with closing(sqlite3.connect(self.source / 'data.sqlite3')) as db:
            db.executescript('''
                CREATE TABLE intervals(kind TEXT, state TEXT, start REAL, end REAL, PRIMARY KEY(kind,start));
                CREATE TABLE notes(id INTEGER PRIMARY KEY, kind TEXT, body TEXT, at REAL);
                CREATE TABLE tasks(id TEXT PRIMARY KEY, kind TEXT, body TEXT, due REAL, fired INTEGER DEFAULT 0);
                CREATE TABLE settings(key TEXT PRIMARY KEY, value INTEGER);
                INSERT INTO settings VALUES ('sit_minutes', 35);
                INSERT INTO tasks VALUES ('旧数据','todo','整理笔记',NULL,0);
            ''')
        self.create()
        with ZipFile(self.archive) as archive:
            self.assertCountEqual(archive.namelist(), ['manifest.json', 'data.sqlite3'])
        self.ok('restore', self.archive, self.restored)
        self.assertEqual([path.name for path in self.restored.iterdir()], ['data.sqlite3'])
        with closing(sqlite3.connect(self.restored / 'data.sqlite3')) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(names, {'intervals', 'notes', 'tasks', 'settings'})
            self.assertEqual(db.execute('SELECT value FROM settings').fetchone(), (35,))
        with patch.object(store, 'ROOT', self.restored):
            store.initialize()
            self.assertEqual(store.list_todos()['items'][0]['id'], '旧数据')

    def test_malformed_archives_fail_verification_and_leave_no_restored_files(self):
        self.initialize()
        (self.source / 'calibration.json').write_bytes(b'{"size": 1}\n')
        self.create()
        with ZipFile(self.archive) as archive:
            base = {name: archive.read(name) for name in archive.namelist()}
        for fault in ('truncated', 'unexpected-member', 'duplicate-member', 'bad-digest',
                      'invalid-database', 'missing-database'):
            with self.subTest(fault=fault):
                broken = self.root / f'{fault}.zip'
                files = base.copy()
                if fault == 'truncated':
                    broken.write_bytes(self.archive.read_bytes()[:-16])
                else:
                    if fault == 'unexpected-member':
                        files['ordinary.txt'] = b'unlisted'
                    elif fault == 'bad-digest':
                        files['calibration.json'] = b'{"size": 2}\n'
                    elif fault == 'invalid-database':
                        files['data.sqlite3'] = PRIVATE_TEXT.encode()
                        manifest = json.loads(files['manifest.json'])
                        manifest['files']['data.sqlite3'] = {'bytes': len(files['data.sqlite3']),
                                                              'sha256': hashlib.sha256(files['data.sqlite3']).hexdigest()}
                        files['manifest.json'] = json.dumps(manifest).encode()
                    elif fault == 'missing-database':
                        del files['data.sqlite3']
                    with ZipFile(broken, 'w', ZIP_STORED) as archive:
                        for name, data in files.items():
                            archive.writestr(name, data)
                        if fault == 'duplicate-member':
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', 'Duplicate name:', UserWarning)
                                archive.writestr('calibration.json', files['calibration.json'])
                before = set(self.root.iterdir())
                self.assertNotEqual(self.cli('verify', broken).returncode, 0)
                self.assertNotEqual(self.cli('restore', broken, self.restored).returncode, 0)
                self.assertEqual(set(self.root.iterdir()), before)
        self.ok('verify', self.archive)

    def test_invalid_source_and_existing_destinations_are_left_alone(self):
        def create_status():
            return self.cli('create', self.archive, '--source-dir', self.source).returncode

        self.assertNotEqual(create_status(), 0)
        self.assertEqual(list(self.source.iterdir()), [])
        database = self.source / 'data.sqlite3'
        with closing(sqlite3.connect(database)) as db:
            db.execute('CREATE TABLE unrelated(value TEXT)')
        self.assertNotEqual(create_status(), 0)
        self.assertFalse(self.archive.exists())
        database.unlink()
        self.initialize()
        with closing(sqlite3.connect(database)) as db:
            db.execute('PRAGMA foreign_keys=OFF')
            db.execute("INSERT INTO idea_requests VALUES ('orphaned',12345)")
            db.commit()
        self.assertNotEqual(create_status(), 0)
        self.assertEqual(set(self.root.iterdir()), {self.source})
        with closing(sqlite3.connect(database)) as db:
            db.execute('DELETE FROM idea_requests')
            db.commit()
        self.archive.write_bytes(b'keep original destination')
        self.assertNotEqual(create_status(), 0)
        self.assertEqual(self.archive.read_bytes(), b'keep original destination')
        self.archive.unlink()
        self.create()
        self.restored.mkdir()
        self.assertNotEqual(self.cli('restore', self.archive, self.restored).returncode, 0)
        sentinel = self.restored / 'existing'
        sentinel.write_text(PRIVATE_TEXT)
        self.assertNotEqual(self.cli('restore', self.archive, self.restored).returncode, 0)
        self.assertEqual([p.name for p in self.restored.iterdir()], ['existing'])
        self.assertEqual(sentinel.read_text(), PRIVATE_TEXT)

    def test_optional_configuration_link_cannot_collect_files_outside_source(self):
        self.initialize()
        other = self.root / '独立配置.json'
        other.write_text(json.dumps({'note': PRIVATE_TEXT}, ensure_ascii=False))
        (self.source / 'calibration.json').symlink_to(other)
        result = self.cli('create', self.archive, '--source-dir', self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(set(self.root.iterdir()), {other, self.source})
        self.assertEqual(json.loads(other.read_text()), {'note': PRIVATE_TEXT})

    def test_competing_cli_processes_publish_only_one_archive_or_restoration(self):
        self.initialize()
        for command in [('create', self.archive, '--source-dir', self.source),
                        ('restore', self.archive, self.restored)]:
            with self.subTest(command=command[0]):
                gate = Barrier(2)

                def run_once():
                    gate.wait(timeout=5)
                    return self.cli(*command).returncode

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = [pool.submit(run_once) for _ in range(2)]
                    self.assertEqual(sum(item.result() == 0 for item in results), 1)
                self.ok('verify', self.archive)
        self.assertEqual(set(self.root.iterdir()), {self.source, self.archive, self.restored})
        with closing(sqlite3.connect(self.restored / 'data.sqlite3')) as db:
            self.assertEqual(db.execute('PRAGMA quick_check').fetchone(), ('ok',))
