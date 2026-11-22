import math
import sqlite3
from activity_annotations import apply_annotations
from posture_continuity import bridge_posture_gaps, MAX_GAP_SECONDS
import time
from contextlib import contextmanager
from datetime import date, datetime, time as day_time, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Shanghai")


@contextmanager
def database():
    db = sqlite3.connect(ROOT / "data.sqlite3", timeout=5)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA foreign_keys=ON")
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
            CREATE TABLE IF NOT EXISTS saved_replies (
                request_id TEXT PRIMARY KEY, note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                reply_offset INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
                due REAL, fired INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS tasks_by_kind ON tasks(kind);
            CREATE TABLE IF NOT EXISTS completed_todos (
                task_id TEXT PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS focus_sessions (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, minutes INTEGER NOT NULL CHECK(minutes BETWEEN 1 AND 180),
                task_id TEXT, started_at REAL NOT NULL, due_at REAL NOT NULL CHECK(due_at>started_at),
                ended_at REAL, state TEXT NOT NULL CHECK(state IN ('active','finished','stopped')),
                CHECK((state='active' AND ended_at IS NULL) OR
                      (state!='active' AND ended_at IS NOT NULL AND ended_at BETWEEN started_at AND due_at)));
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_focus ON focus_sessions(state) WHERE state='active';
            CREATE TABLE IF NOT EXISTS activity_annotations (
                id INTEGER PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('sit','stand','away')),
                reason TEXT NOT NULL, created REAL NOT NULL,
                CHECK(end>start));
            CREATE TABLE IF NOT EXISTS absence_batches (
                batch_id TEXT PRIMARY KEY, start_date TEXT NOT NULL, end_date TEXT NOT NULL,
                undone INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS absence_days (
                annotation_id INTEGER PRIMARY KEY REFERENCES activity_annotations(id) ON DELETE CASCADE,
                batch_id TEXT NOT NULL REFERENCES absence_batches(batch_id));
            CREATE INDEX IF NOT EXISTS absence_days_batch ON absence_days(batch_id);
            CREATE TABLE IF NOT EXISTS absence_requests (
                request_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES absence_batches(batch_id));
            CREATE TABLE IF NOT EXISTS interval_corrections (
                id TEXT PRIMARY KEY, date TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                state TEXT NOT NULL, undone INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS correction_parts (
                annotation_id INTEGER PRIMARY KEY REFERENCES activity_annotations(id) ON DELETE CASCADE,
                correction_id TEXT NOT NULL REFERENCES interval_corrections(id));
            CREATE INDEX IF NOT EXISTS correction_parts_owner ON correction_parts(correction_id);
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
    """Commit sensing intervals and any method tag on the same sampling clock."""
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
        "SELECT * FROM intervals WHERE kind!='pose_method' AND end>? AND start<? ORDER BY start DESC",
        (left, right))]
    for row in rows:
        row['end'] = min(row['end'], right)
    annotations = []
    for row in db.execute(
            'SELECT a.*,p.correction_id FROM activity_annotations a '
            'LEFT JOIN correction_parts p ON p.annotation_id=a.id WHERE a.end>? AND a.start<? ORDER BY a.id',
            (left, right)):
        annotation = dict(row)
        if annotation['correction_id'] is None:
            del annotation['correction_id']
        annotations.append(annotation)
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


def _insert_task(db, request_id, kind, body, minutes=None):
    due = (datetime.now(TZ) + timedelta(minutes=minutes)).timestamp() if minutes else None
    db.execute("INSERT OR IGNORE INTO tasks(id,kind,body,due) VALUES (?,?,?,?)",
               (request_id, kind, body, due))
    row = db.execute("SELECT kind,body,due FROM tasks WHERE id=?", (request_id,)).fetchone()
    if row["kind"] != kind or row["body"] != body:
        raise ValueError("该请求编号已使用")
    return dict(row)


def task(request_id, kind, body, minutes=None):
    with database() as db:
        return _insert_task(db, request_id, kind, body, minutes)


def _saved_reply(db, request_id, prefix):
    row = db.execute(
        "SELECT body,reply_offset FROM saved_replies JOIN notes ON notes.id=note_id WHERE request_id=?",
        (request_id,)).fetchone()
    if row is None:
        return None
    if row['body'][:row['reply_offset']] != prefix:
        raise ValueError("该请求编号已用于其他问题")
    return row['body'][row['reply_offset']:]


def previous_answer(request_id, query):
    with database() as db:
        return _saved_reply(db, request_id, f"你：{query}\n助手：")


def record_answer(request_id, query, answer, save, new_task):
    if not save:
        if new_task:
            task(request_id, **new_task)
        return answer
    # A confirmed draft can be retried after a lost HTTP response. Store the
    # task and the opted-in dialogue together. Keep just one copy of the text.
    prefix = f"你：{query}\n助手："
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        previous = _saved_reply(db, request_id, prefix)
        if previous is not None:
            return previous
        if new_task:
            _insert_task(db, request_id, **new_task)
        note = db.execute("INSERT INTO notes(kind,body,at) VALUES ('dialogue',?,?)",
                          (prefix + answer, time.time()))
        db.execute("INSERT INTO saved_replies(request_id,note_id,reply_offset) VALUES (?,?,?)",
                   (request_id, note.lastrowid, len(prefix)))
    return answer


def list_todos(completed=False, before=None):
    if not isinstance(completed, bool):
        raise ValueError("完成状态应为布尔值")
    upper_bound = 2 ** 63 - 1
    if before is not None:
        cursor = str(before)
        if len(cursor) > 19 or not cursor.isascii() or not cursor.isdecimal():
            raise ValueError("待办分页游标无效")
        position = int(cursor)
        if not 1 <= position <= upper_bound:
            raise ValueError("待办分页游标超出范围")
        upper_bound = position - 1
    page_size = 20
    with database() as db:
        rows = db.execute(
            "SELECT t.rowid AS position,t.id,t.body FROM tasks t "
            "LEFT JOIN completed_todos c ON t.id=c.task_id "
            "WHERE t.kind='todo' AND t.rowid<=? AND (c.task_id IS NOT NULL)=? "
            "ORDER BY t.rowid DESC LIMIT ?", (upper_bound, completed, page_size + 1)).fetchall()
    page = rows[:page_size]
    return {"items": [dict(id=row['id'], body=row['body'], completed=completed) for row in page],
            "next_cursor": str(page[-1]['position']) if len(rows) > page_size else None}


class TodoConflict(Exception):
    pass


def set_todo_completed(task_id, completed):
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("待办编号不能为空")
    if not isinstance(completed, bool):
        raise ValueError("完成状态应为布尔值")
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT kind,body FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        if row['kind'] != 'todo':
            raise TodoConflict("仅待办事项可以完成或恢复")
        if completed:
            db.execute("INSERT OR IGNORE INTO completed_todos(task_id) VALUES (?)", (task_id,))
        else:
            db.execute("DELETE FROM completed_todos WHERE task_id=?", (task_id,))
        return {"id": task_id, "body": row['body'], "completed": completed}


def settle_due_focus(db, now):
    """Finish and notify atomically within the caller's write transaction."""
    active = db.execute("SELECT id,title,minutes FROM focus_sessions WHERE state='active' AND due_at<=?",
                        (now,)).fetchone()
    if active is not None:
        db.execute("UPDATE focus_sessions SET state='finished',ended_at=due_at WHERE id=?", (active['id'],))
        body = f"专注计时结束：{active['title']}（{active['minutes']} 分钟）"
        db.execute("INSERT INTO notes(kind,body,at) VALUES ('reminder',?,?)", (body, now))


def record_due_reminders(now=None):
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = time.time() if now is None else now
        db.execute("INSERT INTO notes(kind,body,at) SELECT 'reminder',body,? FROM tasks "
                   "WHERE due<=? AND fired=0 ORDER BY due", (now, now))
        db.execute("UPDATE tasks SET fired=1 WHERE due<=? AND fired=0", (now,))
        settle_due_focus(db, now)


def annotate_activity(start, end, state, reason):
    if state not in {'sit', 'stand', 'away'} or not start < end or not reason.strip():
        raise ValueError('Invalid activity annotation')
    with database() as db:
        cursor = db.execute('INSERT INTO activity_annotations(start,end,state,reason,created) VALUES (?,?,?,?,?)',
                            (start,end,state,reason,time.time()))
        return cursor.lastrowid


class AbsenceConflict(Exception):
    pass


def calendar_date(value):
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("日期格式应为 YYYY-MM-DD") from None
    if day.isoformat() != value:
        raise ValueError("日期格式应为 YYYY-MM-DD")
    return day


def list_absences(day):
    calendar_date(day)
    with database() as db:
        return [dict(row) for row in db.execute(
            "SELECT batch_id,start_date,end_date FROM absence_batches "
            "WHERE undone=0 AND start_date<=? AND end_date>=? ORDER BY start_date", (day, day))]


def create_absence(start_date, end_date, request_id=None, now=None):
    first, last = calendar_date(start_date), calendar_date(end_date)
    now = (now or datetime.now(TZ)).astimezone(TZ)
    if last < first or (last - first).days >= 31:
        raise ValueError("请选择连续的 1 至 31 天")
    if last >= now.date():
        raise ValueError("只能校正北京时间已结束的日期")
    # Calendar midnights also handle historical offset changes in Shanghai.
    boundaries = [datetime.combine(first + timedelta(days=i), day_time.min, TZ).timestamp()
                  for i in range((last - first).days + 2)]
    request_id = str(request_id) if request_id is not None else None
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        if request_id is not None:
            previous = db.execute(
                "SELECT b.* FROM absence_requests r JOIN absence_batches b USING(batch_id) "
                "WHERE r.request_id=?", (request_id,)).fetchone()
            if previous:
                if previous['start_date'] != start_date or previous['end_date'] != end_date:
                    raise AbsenceConflict("该请求编号已用于其他日期")
                if previous['undone']:
                    raise AbsenceConflict("该次校正已撤销，请重新选择日期后提交")
                return dict(batch_id=previous['batch_id'], start_date=start_date, end_date=end_date)

        overlap = db.execute(
            "SELECT batch_id,start_date,end_date FROM absence_batches "
            "WHERE undone=0 AND start_date<=? AND end_date>=? LIMIT 1",
            (end_date, start_date)).fetchone()
        if overlap:
            if overlap['start_date'] != start_date or overlap['end_date'] != end_date:
                raise AbsenceConflict(f"与已有离开校正重叠（{overlap['start_date']} 至 {overlap['end_date']}）")
            batch_id = overlap['batch_id']
        else:
            conflict = db.execute(
                "SELECT start FROM activity_annotations WHERE start<? AND end>? AND state IN ('sit','stand') "
                "ORDER BY start LIMIT 1", (boundaries[-1], boundaries[0])).fetchone()
            if conflict:
                day = datetime.fromtimestamp(max(conflict['start'], boundaries[0]), TZ).date()
                raise AbsenceConflict(f"{day} 已有人工坐姿或站姿标注，请调整日期范围")
            batch_id = str(uuid4())
            db.execute("INSERT INTO absence_batches(batch_id,start_date,end_date) VALUES (?,?,?)",
                       (batch_id, start_date, end_date))
            for left, right in zip(boundaries, boundaries[1:]):
                annotation = db.execute(
                    "INSERT INTO activity_annotations(start,end,state,reason,created) "
                    "VALUES (?,?,'away',?,?)", (left, right, "本人确认全天未到工位", now.timestamp()))
                db.execute("INSERT INTO absence_days(annotation_id,batch_id) VALUES (?,?)",
                           (annotation.lastrowid, batch_id))
        if request_id is not None:
            db.execute("INSERT INTO absence_requests(request_id,batch_id) VALUES (?,?)", (request_id, batch_id))
        return dict(batch_id=batch_id, start_date=start_date, end_date=end_date)


def undo_absence(batch_id):
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        if not db.execute("SELECT 1 FROM absence_batches WHERE batch_id=?", (batch_id,)).fetchone():
            raise KeyError("找不到该次离开校正")
        db.execute("DELETE FROM activity_annotations WHERE id IN "
                   "(SELECT annotation_id FROM absence_days WHERE batch_id=?)", (batch_id,))
        db.execute("UPDATE absence_batches SET undone=1 WHERE batch_id=?", (batch_id,))
