import math
import sqlite3
from activity_annotations import apply_annotations
from posture_continuity import bridge_posture_gaps, MAX_GAP_SECONDS
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


def _record_intervals(states, now, continuous, max_gap):
    if not math.isfinite(now):
        raise ValueError("Invalid sample time")
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        latest = {kind: db.execute(
            "SELECT * FROM intervals WHERE kind=? ORDER BY start DESC LIMIT 1", (kind,)
        ).fetchone() for kind in states}
        # A backwards wall-clock correction must not create overlapping or
        # negative ranges. Resume when this clock reaches the last observation.
        if any(row and now < row["end"] for row in latest.values()):
            return
        for kind, state in states.items():
            row = latest[kind]
            start = now
            if row:
                previous_end = row["end"]
                adjacent = continuous and (max_gap is None or now - previous_end <= max_gap)
                if row["start"] == now:
                    db.execute("UPDATE intervals SET state=? WHERE kind=? AND start=?",
                               (state, kind, now))
                    continue
                if adjacent or row["state"] == "unknown":
                    db.execute("UPDATE intervals SET end=? WHERE kind=? AND start=?",
                               (now, kind, row["start"]))
                    if row["state"] == state:
                        continue
                elif previous_end < now:
                    # A solitary sample occupies no duration. Turn its marker
                    # into the missing-data range instead of fabricating time.
                    if previous_end == row["start"]:
                        db.execute("UPDATE intervals SET state='unknown',end=? WHERE kind=? AND start=?",
                                   (now, kind, row["start"]))
                        if state == "unknown":
                            continue
                    elif state == "unknown":
                        start = previous_end
                    else:
                        db.execute("INSERT INTO intervals VALUES (?, 'unknown', ?, ?)",
                                   (kind, previous_end, now))
            db.execute("INSERT INTO intervals VALUES (?,?,?,?)", (kind, state, start, now))


def persist(kind, state, now, continuous):
    """Compatibility entry point for offline replay with caller-defined cadence."""
    _record_intervals({kind: state}, now, continuous, None)


def persist_sample(states, now, continuous):
    """Commit all three 1 Hz camera timelines together, also guarding wall time."""
    _record_intervals(states, now, continuous, 3.0)


def note(kind, body):
    with database() as db:
        db.execute("INSERT INTO notes(kind,body,at) VALUES (?,?,?)", (kind, body, time.time()))


def activity_window(db, start, cutoff, as_of):
    """Read both sides of a midnight-spanning short gap before clipping it."""
    if cutoff <= start:
        return [], []
    left = start - MAX_GAP_SECONDS - 1
    right = min(as_of, cutoff + MAX_GAP_SECONDS + 1)
    rows = [dict(r) for r in db.execute(
        'SELECT * FROM intervals WHERE end>? AND start<? ORDER BY start DESC', (left, right))]
    for row in rows:
        row['end'] = min(row['end'], right)
    annotations = [dict(r) for r in db.execute(
        'SELECT * FROM activity_annotations WHERE end>? AND start<? ORDER BY id', (left, right))]
    result = []
    for row in bridge_posture_gaps(apply_annotations(rows, annotations)):
        row['start'], row['end'] = max(row['start'], start), min(row['end'], cutoff)
        if row['end'] <= row['start']:
            continue
        if row.get('source') == 'continuity':
            row['inferred_seconds'] = row['end'] - row['start']
            row['inferred_ranges'] = [{'start': row['start'], 'end': row['end']}]
        result.append(row)
    return result, [a for a in annotations if a['start'] < cutoff and a['end'] > start]


def activity_totals(rows):
    totals = {}
    for row in rows:
        state = row['state']
        if row['kind'] == 'person' or (row['kind'] == 'presence' and state == 'present'):
            totals[state] = totals.get(state, 0.) + (row['end'] - row['start'])
    return {state: round(seconds) for state, seconds in totals.items()}


def snapshot(now=None):
    now = now or datetime.now(TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with database() as db:
        db.execute("BEGIN")
        rows, _ = activity_window(db, midnight, now.timestamp(), now.timestamp())
        return {
            "totals": activity_totals(rows),
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
