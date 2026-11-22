"""Server-timed user labels for assessing live posture sensing on the Orin."""
from contextlib import contextmanager
import math
import time
from uuid import UUID

import store
import trial_metrics


class TrialConflict(Exception):
    pass


def _key(value):
    try:
        return str(UUID(str(value)))
    except ValueError:
        raise ValueError('核验或标记编号必须是 UUID') from None


def _clock(now):
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError('核验时间无效')
    return now


@contextmanager
def _transaction(now):
    with store.database() as db:
        db.execute('BEGIN IMMEDIATE')
        now = _clock(now)
        db.execute('UPDATE field_trials SET ended_at=deadline WHERE ended_at IS NULL AND deadline<=?', (now,))
        try:
            yield db, now
        except (TrialConflict, KeyError):
            # All business conflicts are checked before writing the requested
            # action. A rejected late mark must still leave the timer expired.
            db.commit()
            raise


def _session(db, session_id):
    row = db.execute('SELECT * FROM field_trials WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise KeyError('找不到该场现场核验')
    return dict(row)


def _replay(db, session_id, request_id, state):
    mark = db.execute('SELECT session_id,state FROM field_trial_marks WHERE request_id=?', (request_id,)).fetchone()
    if mark is None:
        return False
    if mark['session_id'] != session_id or mark['state'] != state:
        raise TrialConflict('该请求编号已用于另一项标记')
    return True


def start(session_id, state, now=None):
    key = _key(session_id)
    if state not in ('sit', 'stand', 'away'):
        raise ValueError('开始核验时请选择坐姿、站姿或离开')
    with _transaction(now) as (db, now):
        if _replay(db, key, key, state):
            return _session(db, key)
        if db.execute('SELECT 1 FROM field_trials WHERE ended_at IS NULL').fetchone():
            raise TrialConflict('已有进行中的现场核验')
        previous = db.execute('SELECT ended_at FROM field_trials ORDER BY rowid DESC LIMIT 1').fetchone()
        if previous and now < previous['ended_at']:
            raise TrialConflict('设备时钟早于上一场结束时间，请等待时钟恢复后再开始')
        db.execute('INSERT INTO field_trials(id,started_at,deadline,current_state,marked_at) VALUES (?,?,?,?,?)',
                   (key, now, now + 7200, state, now))
        db.execute('INSERT INTO field_trial_marks(request_id,session_id,at,state) VALUES (?,?,?,?)',
                   (key, key, now, state))
        return _session(db, key)


def mark(session_id, request_id, state, now=None):
    key, label_id = _key(session_id), _key(request_id)
    if state not in ('sit', 'stand', 'away', 'pause'):
        raise ValueError('标记状态须为坐姿、站姿、离开或暂停')
    with _transaction(now) as (db, now):
        trial = _session(db, key)
        if _replay(db, key, label_id, state):
            return trial
        if trial['ended_at'] is not None:
            raise TrialConflict('该场现场核验已结束')
        if now < trial['marked_at']:
            raise TrialConflict('设备时钟早于上一标记，请等待时钟恢复后重试')
        if state == trial['current_state']:
            raise TrialConflict('该状态已经标记，请在实际状态改变时标记')
        if db.execute('SELECT count(*) FROM field_trial_marks WHERE session_id=?', (key,)).fetchone()[0] >= 7200:
            raise TrialConflict('本场核验的标记数已达上限，请结束本场核验')
        db.execute('INSERT INTO field_trial_marks(request_id,session_id,at,state) VALUES (?,?,?,?)',
                   (label_id, key, now, state))
        db.execute('UPDATE field_trials SET current_state=?,marked_at=? WHERE id=?', (state, now, key))
        return _session(db, key)


def finish(session_id, now=None):
    key = _key(session_id)
    with _transaction(now) as (db, now):
        trial = _session(db, key)
        if trial['ended_at'] is None:
            db.execute('UPDATE field_trials SET ended_at=max(marked_at,?) WHERE id=?', (now, key))
            trial = _session(db, key)
        return trial


def snapshot(now=None):
    with _transaction(now) as (db, now):
        active = db.execute('SELECT * FROM field_trials WHERE ended_at IS NULL').fetchone()
        recent = db.execute('SELECT * FROM field_trials WHERE ended_at IS NOT NULL ORDER BY rowid DESC LIMIT 5')
        return {'active': dict(active) if active else None, 'recent': [dict(r) for r in recent], 'server_now': now}


def report(session_id, now=None):
    key = _key(session_id)
    with store.database() as db:
        db.execute('BEGIN')
        trial = _session(db, key)
        if trial['ended_at'] is None:
            if _clock(now) < trial['deadline']:
                raise TrialConflict('请先结束本场核验，再查看报告')
            trial['ended_at'] = trial['deadline']
        started, ended = trial['started_at'], trial['ended_at']
        labels = [dict(r) for r in db.execute('SELECT at,state FROM field_trial_marks WHERE session_id=? '
                                              'ORDER BY at,rowid', (key,))]
        raw = [dict(r) for r in db.execute(
            "SELECT kind,state,start,end FROM intervals WHERE end>? AND start<? AND end>start "
            "AND (kind='person' OR (kind='presence' AND state='privacy')) ORDER BY start", (started, ended))]
    return {'session': trial, 'metrics': trial_metrics.summarize(labels, raw, started, ended)}
