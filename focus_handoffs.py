"""A private continuation note for one focus session, consumed by one later session."""
import re

import focus_sessions as focus
import store


class HandoffConflict(Exception):
    pass


HANDOFF_SELECT = """SELECT h.sequence, h.source_session_id AS id, s.title, s.minutes, s.task_id,
    h.body, h.created_at, h.status, h.resumed_session_id
    FROM focus_handoffs h JOIN focus_sessions s ON s.id=h.source_session_id """


def _metadata(row):
    item = dict(row)
    del item['sequence']
    return item


def _get(db, session_id):
    return db.execute(HANDOFF_SELECT + "WHERE h.source_session_id=?", (session_id,)).fetchone()


def _required(db, session_id):
    row = _get(db, session_id)
    if row is None:
        raise KeyError("找不到该条续做便签")
    return row


def pending(before=None):
    upper = 2**63 - 1
    if before is not None:
        cursor = str(before)
        if not re.fullmatch(r"[1-9][0-9]{0,18}", cursor) or int(cursor) > upper:
            raise ValueError("续做便签分页游标无效")
        upper = int(cursor) - 1
    with store.database() as db:
        rows = db.execute(HANDOFF_SELECT + "WHERE h.status='pending' AND h.sequence<=? "
                          "ORDER BY h.sequence DESC LIMIT 21", (upper,)).fetchall()
    page = rows[:20]
    return {"items": [_metadata(row) for row in page],
            "next_cursor": str(page[-1]['sequence']) if len(rows) > len(page) else None}


def save(session_id, body, now=None):
    session_id = focus._session_id(session_id)
    if not isinstance(body, str) or not body.strip() or len(body) > 500:
        raise ValueError("续做便签需填写 1 至 500 字")
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = focus._time(now)
        store.settle_due_focus(db, now)
        existing = _get(db, session_id)
        if existing:
            if existing['body'] != body:
                raise HandoffConflict("这次专注已保存另一条续做便签")
            return _metadata(existing)
        source = focus.stop_session(db, session_id, now)
        db.execute("INSERT INTO focus_handoffs(source_session_id,body,created_at) VALUES (?,?,?)",
                   (session_id, body, max(now, source['ended_at'])))
        return _metadata(_get(db, session_id))


def resume(source_id, new_session_id, now=None):
    source_id, new_session_id = focus._session_id(source_id), focus._session_id(new_session_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = focus._time(now)
        store.settle_due_focus(db, now)
        handoff = _required(db, source_id)
        if handoff['resumed_session_id'] == new_session_id:
            return focus.get_session(db, new_session_id)
        if handoff['status'] != 'pending':
            raise HandoffConflict("该便签已续做或已归档")
        if db.execute("SELECT 1 FROM focus_sessions WHERE id=?", (new_session_id,)).fetchone():
            raise HandoffConflict("该请求编号已用于其他专注计时")
        focus.insert_session(db, new_session_id, handoff['minutes'], handoff['title'], handoff['task_id'],
                             now, continuing=True)
        db.execute("UPDATE focus_handoffs SET status='resumed',resumed_session_id=?,resolved_at=max(created_at,?) "
                   "WHERE source_session_id=?", (new_session_id, now, source_id))
        return focus.get_session(db, new_session_id)


def dismiss(source_id, now=None):
    source_id = focus._session_id(source_id)
    with store.database() as db:
        db.execute("BEGIN IMMEDIATE")
        now = focus._time(now)
        handoff = _required(db, source_id)
        if handoff['status'] == 'resumed':
            raise HandoffConflict("该便签已开始续做")
        if handoff['status'] == 'pending':
            db.execute("UPDATE focus_handoffs SET status='dismissed',resolved_at=max(created_at,?) "
                       "WHERE source_session_id=?", (now, source_id))
            handoff = _get(db, source_id)
        return _metadata(handoff)
