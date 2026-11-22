"""Persistent, single-session focus timer shared by all dashboard clients."""
import math
import time
from uuid import UUID

import store


class FocusConflict(Exception):
    pass


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
        previous = db.execute("SELECT * FROM focus_sessions WHERE id=?", (session_id,)).fetchone()
        if previous:
            if (previous['minutes'] != minutes or previous['task_id'] != task_id or
                    (task_id is None and previous['title'] != title)):
                raise FocusConflict("该请求编号已用于其他专注计时")
            return dict(previous)
        if db.execute("SELECT 1 FROM focus_sessions WHERE state='active'").fetchone():
            raise FocusConflict("已有进行中的专注计时")
        if task_id is not None:
            task = db.execute(
                "SELECT t.kind,t.body,c.task_id AS done FROM tasks t "
                "LEFT JOIN completed_todos c ON t.id=c.task_id WHERE t.id=?", (task_id,)).fetchone()
            if task is None:
                raise KeyError("找不到该待办")
            if task['kind'] != 'todo' or task['done'] is not None:
                raise FocusConflict("只能从未完成的待办开始专注")
            title = task['body']
        db.execute(
            "INSERT INTO focus_sessions(id,title,minutes,task_id,started_at,due_at,state) "
            "VALUES (?,?,?,?,?,?,'active')", (session_id, title, minutes, task_id, now, now + minutes * 60))
        return dict(db.execute("SELECT * FROM focus_sessions WHERE id=?", (session_id,)).fetchone())


def finish(session_id, now=None):
    session_id = _session_id(session_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _time(now)
        store.settle_due_focus(db, now)
        db.execute("UPDATE focus_sessions SET state='stopped',ended_at=max(started_at,?) "
                   "WHERE id=? AND state='active'", (now, session_id))
        row = db.execute("SELECT * FROM focus_sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError("找不到该次专注计时")
        return dict(row)


def snapshot(now=None):
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _time(now)
        store.settle_due_focus(db, now)
        active = db.execute("SELECT * FROM focus_sessions WHERE state='active'").fetchone()
        recent = db.execute("SELECT * FROM focus_sessions WHERE state!='active' ORDER BY rowid DESC LIMIT 5")
        return {'active': dict(active) if active else None, 'recent': [dict(row) for row in recent], 'server_now': now}
