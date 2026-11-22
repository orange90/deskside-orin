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


def _seconds(value):
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("暂停时间无效")
        return float(value)
    except OverflowError:
        raise ValueError("暂停时间无效") from None


def _now(value):
    return _seconds(time.time() if value is None else value)


def _duration(minutes):
    if type(minutes) is not int or minutes not in (1, 15, 30, 60):
        raise ValueError("请选择 1、15、30 或 60 分钟")
    return minutes * 60


def _deadline(base, duration):
    until = base + duration
    if not math.isfinite(until) or until <= base:
        raise ValueError("暂停时间无效")
    return until


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
    duration = _duration(minutes)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _now(now)
        _expire(db, now)
        previous = db.execute("SELECT minutes FROM camera_pauses WHERE id=?", (session_id,)).fetchone()
        active = _active(db)
        if previous and previous['minutes'] != minutes:
            error = PrivacyConflict("该编号已用于其他暂停时长")
        elif not previous and active:
            error = PrivacyConflict("摄像头已有进行中的隐私暂停")
        else:
            if previous is None:
                db.execute("INSERT INTO camera_pauses(id,minutes,started_at,until) VALUES (?,?,?,?)",
                           (session_id, minutes, now, _deadline(now, duration)))
                active = _active(db)
            return _status(active, now)
    raise error


def extend(session_id, request_id, minutes, expected_until, now=None):
    session_id, request_id = _id(session_id), _id(request_id)
    duration, expected_until = _duration(minutes), _seconds(expected_until)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _now(now)
        _expire(db, now)
        pause = db.execute("SELECT until,ended_at FROM camera_pauses WHERE id=?", (session_id,)).fetchone()
        receipt = db.execute("SELECT session_id,minutes,expected_until FROM camera_pause_extensions "
                             "WHERE request_id=?", (request_id,)).fetchone()
        if pause is None:
            error = KeyError("找不到该次隐私暂停")
        elif receipt:
            if tuple(receipt) == (session_id, minutes, expected_until):
                return _status(_active(db), now)
            error = PrivacyConflict("该请求编号已用于其他延长操作")
        elif pause['ended_at'] is not None:
            error = PrivacyConflict("该次隐私暂停已结束，请刷新状态")
        elif pause['until'] != expected_until:
            error = PrivacyConflict("暂停截止时间已更新，请刷新状态后再延长")
        else:
            db.execute("UPDATE camera_pauses SET until=? WHERE id=?",
                       (_deadline(expected_until, duration), session_id))
            db.execute("INSERT INTO camera_pause_extensions(request_id,session_id,minutes,expected_until) "
                       "VALUES (?,?,?,?)", (request_id, session_id, minutes, expected_until))
            return _status(_active(db), now)
    # Observing expiry is final even if this request cannot extend the lease.
    # Commit that observation before reporting the conflict to the caller.
    raise error


def resume(session_id, now=None):
    session_id = _id(session_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = _now(now)
        _expire(db, now)
        if db.execute("SELECT 1 FROM camera_pauses WHERE id=?", (session_id,)).fetchone():
            db.execute("UPDATE camera_pauses SET ended_at=max(started_at,?) WHERE id=? AND ended_at IS NULL",
                       (now, session_id))
            return _status(_active(db), now)
    raise KeyError("找不到该次隐私暂停")
