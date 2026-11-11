import sqlite3
from activity_annotations import apply_annotations
from posture_continuity import bridge_posture_gaps
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")


@contextmanager
def database():
    db = sqlite3.connect(ROOT / "data.sqlite3", timeout=5)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize():
    with database() as db:
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS intervals (
                kind TEXT NOT NULL, state TEXT NOT NULL,
                start REAL NOT NULL, end REAL NOT NULL,
                PRIMARY KEY (kind, start));
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                due REAL, fired INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS activity_annotations (
                id INTEGER PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('sit','stand','away')),
                reason TEXT NOT NULL, created REAL NOT NULL,
                CHECK(end>start));
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            INSERT OR IGNORE INTO settings VALUES ('stand_minutes', 60), ('sit_minutes', 50);
        """)


def persist(kind, state, now, continuous):
    with database() as db:
        row = db.execute(
            "SELECT * FROM intervals WHERE kind=? ORDER BY start DESC LIMIT 1", (kind,)
        ).fetchone()
        if row and row["state"] == state and (continuous or state == "unknown"):
            db.execute("UPDATE intervals SET end=? WHERE kind=? AND start=?",
                       (now, kind, row["start"]))
        else:
            start = now
            if row and row["end"] < now:
                if continuous or row["state"] == "unknown":
                    db.execute("UPDATE intervals SET end=? WHERE kind=? AND start=?",
                               (now, kind, row["start"]))
                else:
                    gap = max(row["end"], row["start"] + .000001)
                    if gap < now and state == "unknown":
                        start = gap
                    elif gap < now:
                        db.execute("INSERT OR IGNORE INTO intervals VALUES (?, 'unknown', ?, ?)",
                                   (kind, gap, now))
            db.execute("INSERT INTO intervals VALUES (?,?,?,?)", (kind, state, start, now))


def note(kind, body):
    with database() as db:
        db.execute("INSERT INTO notes(kind,body,at) VALUES (?,?,?)", (kind, body, time.time()))


def snapshot():
    now = datetime.now(TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    totals = {}
    with database() as db:
        rows = [dict(r) for r in db.execute(
            "SELECT * FROM intervals WHERE end>=? ORDER BY start DESC", (midnight,)
        )]
        annotations = [dict(r) for r in db.execute(
            'SELECT * FROM activity_annotations WHERE end>? AND start<? ORDER BY id', (midnight, now.timestamp()))]
        rows = bridge_posture_gaps(apply_annotations(rows, annotations))
        for row in rows:
            row["start"] = max(midnight, row["start"])
            state = row["state"]
            if row["kind"] == "person" or (row["kind"] == "presence" and state == "present"):
                totals[state] = totals.get(state, 0) + row["end"] - row["start"]
        return {
            "totals": {key: round(value) for key, value in totals.items()},
            "settings": dict(db.execute("SELECT key,value FROM settings").fetchall()),
            "intervals": rows,
            "notes": [dict(r) for r in db.execute("SELECT * FROM notes ORDER BY id DESC LIMIT 50")],
            "tasks": [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY rowid DESC LIMIT 50")],
        }


def task(request_id, kind, body, minutes=None):
    due = (datetime.now(TZ) + timedelta(minutes=minutes)).timestamp() if minutes else None
    with database() as db:
        db.execute("INSERT OR IGNORE INTO tasks(id,kind,body,due) VALUES (?,?,?,?)",
                   (request_id, kind, body, due))
        row = db.execute("SELECT kind,body,due FROM tasks WHERE id=?", (request_id,)).fetchone()
        if row["kind"] != kind or row["body"] != body:
            raise ValueError("该请求编号已使用")
        return dict(row)


def record_due_reminders():
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT INTO notes(kind,body,at) SELECT 'reminder',body,? FROM tasks "
                   "WHERE due<=? AND fired=0 ORDER BY due", (now, now))
        db.execute("UPDATE tasks SET fired=1 WHERE due<=? AND fired=0", (now,))


def annotate_activity(start, end, state, reason):
    if state not in {'sit', 'stand', 'away'} or not start < end or not reason.strip():
        raise ValueError('Invalid activity annotation')
    with database() as db:
        cursor = db.execute('INSERT INTO activity_annotations(start,end,state,reason,created) VALUES (?,?,?,?,?)',
                            (start,end,state,reason,time.time()))
        return cursor.lastrowid
