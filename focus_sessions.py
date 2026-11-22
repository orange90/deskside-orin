"""Persistent, single-session focus timer shared by all dashboard clients."""
import math
import time
from uuid import UUID

import store


class FocusConflict(Exception):
    pass


SESSION_SELECT = """SELECT s.*,
    EXISTS(SELECT 1 FROM focus_handoffs h WHERE h.source_session_id=s.id) AS has_handoff,
    CASE WHEN s.state='active' THEN (SELECT h.body FROM focus_handoffs h WHERE h.resumed_session_id=s.id)
    END AS resume_note FROM focus_sessions s """


def _session(row):
    if row is None:
        return None
    result = dict(row)
    result['has_handoff'] = bool(result['has_handoff'])
    if result['resume_note'] is None:
        del result['resume_note']
    return result


def get_session(db, session_id):
    return _session(db.execute(SESSION_SELECT + "WHERE s.id=?", (session_id,)).fetchone())


def _session_id(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise ValueError("专注计时编号必须是 UUID") from None


def _time(now):
    now = time.time() if now is None else now
    if not math.isfinite(now):
        raise ValueError("计时时间无效")
    return now


def insert_session(db, session_id, minutes, title, task_id, now, *, continuing=False):
    """Insert inside the caller's settled write transaction using the shared single-timer rule."""
    if db.execute("SELECT 1 FROM focus_sessions WHERE state='active'").fetchone():
        raise FocusConflict("已有进行中的专注计时")
    if task_id is not None:
        task = db.execute(
            "SELECT t.kind,t.body,c.task_id AS done FROM tasks t "
            "LEFT JOIN completed_todos c ON t.id=c.task_id WHERE t.id=?", (task_id,)).fetchone()
        if task is None or task['kind'] != 'todo' or task['done'] is not None:
            if continuing:
                task_id = None
            elif task is None:
                raise KeyError("找不到该待办")
            else:
                raise FocusConflict("只能从未完成的待办开始专注")
        elif not continuing:
            title = task['body']
    db.execute(
        "INSERT INTO focus_sessions(id,title,minutes,task_id,started_at,due_at,state) "
        "VALUES (?,?,?,?,?,?,'active')", (session_id, title, minutes, task_id, now, now + minutes * 60))


def start(session_id, minutes, title=None, task_id=None, now=None):
    session_id = _session_id(session_id)
    if type(minutes) is not int or not 1 <= minutes <= 180:
        raise ValueError("专注时长应为 1 至 180 的整数分钟")
    if title is not None and (not isinstance(title, str) or len(title.strip()) > 120):
        raise ValueError("专注标题最多 120 字")
    title = (title or '').strip() or '专注'
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip()):
        raise ValueError("待办编号不能为空")
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _time(now)
        store.settle_due_focus(db, now)
        previous = get_session(db, session_id)
        if previous:
            if (previous['minutes'] != minutes or previous['task_id'] != task_id or
                    (task_id is None and previous['title'] != title)):
                raise FocusConflict("该请求编号已用于其他专注计时")
            return previous
        insert_session(db, session_id, minutes, title, task_id, now)
        return get_session(db, session_id)


def stop_session(db, session_id, now):
    db.execute("UPDATE focus_sessions SET state='stopped',ended_at=max(started_at,?) "
               "WHERE id=? AND state='active'", (now, session_id))
    session = get_session(db, session_id)
    if session is None:
        raise KeyError("找不到该次专注计时")
    return session


def finish(session_id, now=None):
    session_id = _session_id(session_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _time(now)
        store.settle_due_focus(db, now)
        return stop_session(db, session_id, now)


def snapshot(now=None):
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _time(now)
        store.settle_due_focus(db, now)
        active = db.execute(SESSION_SELECT + "WHERE s.state='active'").fetchone()
        recent = db.execute(SESSION_SELECT + "WHERE s.state!='active' ORDER BY s.rowid DESC LIMIT 5")
        return {'active': _session(active), 'recent': [_session(row) for row in recent], 'server_now': now}
