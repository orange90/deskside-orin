"""Durable, idempotent manual camera pauses, independent of capture hardware."""
import math
import time
from uuid import UUID

import store


class PrivacyConflict(Exception):
    pass


def _id(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise ValueError("隐私暂停编号必须是 UUID") from None


def _now(value):
    value = time.time() if value is None else value
    if not math.isfinite(value):
        raise ValueError("暂停时间无效")
    return value


def _active(db):
    return db.execute("SELECT id,started_at,until FROM camera_pauses WHERE ended_at IS NULL").fetchone()


def _expire(db, now):
    db.execute("UPDATE camera_pauses SET ended_at=until WHERE ended_at IS NULL AND until<=?", (now,))


def _status(row, now):
    return {"active": dict(row) if row else None, "server_now": now}


def snapshot(now=None):
    with store.database() as db:
        row = _active(db)
        now = _now(now)
        if row and row['until'] <= now:
            db.execute("BEGIN IMMEDIATE")
            _expire(db, now)
            row = _active(db)
        return _status(row, now)


def start(session_id, minutes, now=None):
    session_id = _id(session_id)
    if type(minutes) is not int or minutes not in (1, 15, 30, 60):
        raise ValueError("请选择 1、15、30 或 60 分钟")
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _now(now)
        _expire(db, now)
        previous = db.execute("SELECT minutes FROM camera_pauses WHERE id=?", (session_id,)).fetchone()
        active = _active(db)
        if previous:
            if previous['minutes'] != minutes:
                raise PrivacyConflict("该编号已用于其他暂停时长")
        elif active:
            raise PrivacyConflict("摄像头已有进行中的隐私暂停")
        else:
            db.execute("INSERT INTO camera_pauses(id,minutes,started_at,until) VALUES (?,?,?,?)",
                       (session_id, minutes, now, now + minutes * 60))
            active = _active(db)
        return _status(active, now)


def resume(session_id, now=None):
    session_id = _id(session_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _now(now)
        _expire(db, now)
        if not db.execute("SELECT 1 FROM camera_pauses WHERE id=?", (session_id,)).fetchone():
            raise KeyError("找不到该次隐私暂停")
        db.execute("UPDATE camera_pauses SET ended_at=max(started_at,?) WHERE id=? AND ended_at IS NULL",
                   (now, session_id))
        return _status(_active(db), now)
