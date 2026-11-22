"""Create, verify, and restore local personal-data snapshots using only the standard library."""
import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import struct
import sys
import tempfile
import time
import zipfile
from contextlib import closing, nullcontext


DATABASE = "data.sqlite3"
CONFIGS = ("calibration.json", "model-selection.json")
MANIFEST = "manifest.json"
FORMAT = "deskside-orin-data"
BASE_TABLES = {"intervals", "notes", "tasks", "settings"}
BLOCK = 1024 * 1024
# SQLite's public result codes are not exported as Python constants until 3.11.
SQLITE_BUSY, SQLITE_LOCKED = 5, 6


class BackupError(Exception):
    """Messages from this class may be shown without disclosing file contents."""


def _private_file(path):
    return open(path, "xb", opener=lambda name, flags: os.open(name, flags, 0o600))


def _sync_file(file):
    file.flush()
    os.fsync(file.fileno())


def _sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _database(path, mode):
    return sqlite3.connect(path.resolve().as_uri() + "?" + mode, uri=True, timeout=5)


def _check_database(path):
    # This private staging file has no separate writer or WAL.
    with closing(_database(path, "mode=ro&immutable=1")) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?,?,?)",
                            tuple(BASE_TABLES))
        if {row[0] for row in tables} != BASE_TABLES:
            raise BackupError("数据库缺少工位助手的基础表。")
        if db.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
            raise BackupError("数据库完整性校验失败。")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise BackupError("数据库引用校验失败。")


def _copy_database(source, target):
    if not stat.S_ISREG(source.lstat().st_mode):
        raise BackupError("源数据库必须是普通文件。")
    with _private_file(target):
        pass
    busy_since = None

    def progress(status, remaining, pages):
        nonlocal busy_since
        if status in (SQLITE_BUSY, SQLITE_LOCKED):
            if busy_since is None:
                busy_since = time.monotonic()
            elif time.monotonic() - busy_since >= 5:
                raise BackupError("数据库持续繁忙，请稍后重试。")
        else:
            busy_since = None

    with closing(_database(source, "mode=ro")) as src, closing(_database(target, "mode=rw")) as dst:
        # Pin a single WAL read snapshot before chunking. Frequent concurrent
        # commits can otherwise repeatedly restart an online backup at page one.
        src.execute("BEGIN")
        src.execute("SELECT rootpage FROM sqlite_master LIMIT 1").fetchone()
        src.backup(dst, pages=256, progress=progress, sleep=.05)
        src.rollback()
        dst.execute("PRAGMA journal_mode=DELETE")
    _check_database(target)


def _transfer(source, target):
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(BLOCK):
        digest.update(chunk)
        size += len(chunk)
        if target is not None:
            target.write(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _regular_input(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    file = os.fdopen(fd, "rb")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        file.close()
        raise BackupError("数据和配置必须是普通文件。")
    return file


def _archive_member(archive, name, source):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    before = os.fstat(source.fileno())
    info.file_size = before.st_size
    with archive.open(info, "w") as out:
        entry = _transfer(source, out)
    after = os.fstat(source.fileno())
    if entry["bytes"] != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise BackupError("读取期间文件发生变化，请重试。")
    return entry


def _publish(source, destination):
    # Ordinary rename() may replace even an existing empty directory. These
    # kernel operations publish a whole staged snapshot iff the name is free.
    libc = ctypes.CDLL(None, use_errno=True)
    src, dst = os.fsencode(source), os.fsencode(destination)
    if sys.platform == "darwin":
        result = libc.renamex_np(src, dst, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        result = libc.renameat2(-100, src, -100, dst, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise BackupError("当前系统不支持安全发布到新路径。")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    try:
        _sync_directory(destination.parent)
    except OSError:
        # The validated file or directory is already visible to other users.
        raise BackupError("快照已发布，系统未能确认目标目录落盘。") from None


def _unused(path):
    if os.path.lexists(path):
        raise BackupError("目标路径已存在。")


def create(destination, source_dir):
    _unused(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        work = Path(work)
        _copy_database(source_dir / DATABASE, work / DATABASE)
        staged = work / "snapshot.zip"
        with _private_file(staged) as output:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
                with _regular_input(work / DATABASE) as db:
                    files = {DATABASE: _archive_member(archive, DATABASE, db)}
                # Each atomic config replacement yields one version of that file,
                # independently of the SQLite snapshot.
                for name in CONFIGS:
                    try:
                        source = _regular_input(source_dir / name)
                    except FileNotFoundError:
                        continue
                    with source:
                        files[name] = _archive_member(archive, name, source)
                manifest = {"format": FORMAT, "version": 1, "files": files}
                archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True) + "\n")
            _sync_file(output)
        _publish(staged, destination)
    return files


class _ArchiveReader(io.BufferedReader):
    def read(self, size=-1):
        # ZipFile normally reads the entire central directory in one call.
        # Our four short member names need far less than a streaming block.
        if size > BLOCK:
            raise BackupError("归档目录过大。")
        return super().read(size)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackupError("快照清单含重复字段。")
        result[key] = value
    return result


def _manifest(archive):
    members = archive.infolist()
    names = {item.filename for item in members}
    if (len(members) != len(names) or not {DATABASE, MANIFEST} <= names
            or names - {DATABASE, MANIFEST, *CONFIGS}):
        raise BackupError("归档成员缺失、重复或不在允许范围内。")
    for item in members:
        kind = stat.S_IFMT(item.external_attr >> 16)
        if (item.orig_filename != item.filename or kind not in (0, stat.S_IFREG)
                or item.external_attr & 0x10 or item.flag_bits & 1
                or item.compress_type != zipfile.ZIP_STORED or item.file_size != item.compress_size):
            raise BackupError("快照中存在不支持的归档成员。")
    if archive.getinfo(MANIFEST).file_size > 8192:
        raise BackupError("快照清单过大。")
    data = json.loads(archive.read(MANIFEST).decode("utf-8"), object_pairs_hook=_unique_keys)
    if (not isinstance(data, dict) or set(data) != {"format", "version", "files"}
            or data["format"] != FORMAT or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["files"], dict) or set(data["files"]) != names - {MANIFEST}):
        raise BackupError("快照清单格式或版本无效。")
    for name, entry in data["files"].items():
        if (not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"} or type(entry["bytes"]) is not int
                or entry["bytes"] != archive.getinfo(name).file_size):
            raise BackupError("快照文件信息不一致。")
    return data["files"]


def _validate(archive_path, folder, restoring):
    with _ArchiveReader(io.FileIO(archive_path, "r")) as reader, zipfile.ZipFile(reader) as archive:
        files = _manifest(archive)
        for name, expected in files.items():
            target = _private_file(folder / name) if restoring or name == DATABASE else nullcontext(None)
            with target as output, archive.open(name) as source:
                if _transfer(source, output) != expected:
                    raise BackupError("快照文件的 SHA-256 或字节数不匹配。")
                if restoring:
                    _sync_file(output)
    _check_database(folder / DATABASE)
    return files


def verify(archive):
    with tempfile.TemporaryDirectory() as work:
        return _validate(archive, Path(work), False)


def restore(archive, destination):
    _unused(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        staged = Path(work) / "restored"
        staged.mkdir(mode=0o700)
        files = _validate(archive, staged, True)
        _sync_directory(staged)
        _publish(staged, destination)
        return files


def main():
    parser = argparse.ArgumentParser(description="本地个人数据快照。数据库和每份配置各自取样。")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create", help="从运行中的服务数据创建快照")
    make.add_argument("destination", type=Path)
    make.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    check = commands.add_parser("verify", help="校验归档和数据库")
    check.add_argument("archive", type=Path)
    unpack = commands.add_parser("restore", help="校验并恢复到全新目录")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            files = create(args.destination, args.source_dir)
            action = "已创建快照，数据库和各配置分别取样"
        elif args.command == "verify":
            files = verify(args.archive)
            action = "快照校验通过"
        else:
            files = restore(args.archive, args.destination)
            action = "已校验并恢复快照"
    except BackupError as error:
        print(error, file=sys.stderr)
        return 1
    except FileExistsError:
        print("目标路径已存在。", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("操作已中断。", file=sys.stderr)
        return 130
    except (OSError, sqlite3.Error, ValueError, EOFError, OverflowError, RuntimeError, struct.error,
            zipfile.BadZipFile, zipfile.LargeZipFile):
        print("操作失败，请检查路径、可用空间及快照完整性。", file=sys.stderr)
        return 1
    print(f"{action}。数据文件 {len(files)} 项，共 {sum(item['bytes'] for item in files.values())} 字节。")
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
