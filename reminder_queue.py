"""Inspect and change pending timed reminders before they reach the Mac feed."""
import base64
import json
import math
import sys
import time
from uuid import UUID

import store


class ReminderConflict(Exception):
    pass


def _seconds(value):
    if type(value) in (int, float):
        try:
            seconds = float(value)
        except OverflowError:
            pass
        else:
            if math.isfinite(seconds):
                return seconds
    raise ValueError('提醒时间须为有限的时间戳')


def _cursor(row):
    # JSON preserves the binary float on round-trip and supports legacy IDs
    # containing punctuation, whitespace, or non-ASCII characters.
    data = json.dumps([row['due'], row['id']], separators=(',', ':')).encode('utf-8')
    return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')


def _position(cursor):
    try:
        if not isinstance(cursor, str):
            raise ValueError()
        data = base64.b64decode(cursor + '=' * (-len(cursor) % 4), altchars=b'-_', validate=True)
        position = json.loads(data)
        if not isinstance(position, list) or len(position) != 2 or not isinstance(position[1], str):
            raise ValueError()
        return _seconds(position[0]), position[1]
    except (ValueError, RecursionError):
        raise ValueError('待发提醒分页游标无效') from None


def list_pending(cursor=None, now=None):
    now = _seconds(time.time() if now is None else now)
    # The upper numeric bound also excludes non-finite or nonnumeric dates
    # from manually imported SQLite rows.
    query = "SELECT id,body,due FROM tasks WHERE kind='reminder' AND fired=0 AND due>? AND due<=?"
    args = [now, sys.float_info.max]
    if cursor is not None:
        query += " AND (due,id)>(?,?)"
        args.extend(_position(cursor))
    with store.database() as db:
        rows = db.execute(query + ' ORDER BY due,id LIMIT 21', args).fetchall()
    page = rows[:20]
    return {'items': [dict(r) for r in page], 'next_cursor': _cursor(page[-1]) if len(rows) > 20 else None,
            'server_now': now}


def change(task_id, request_id, action, expected_due, now=None):
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError('提醒编号不能为空')
    try:
        key = str(UUID(str(request_id)))
    except ValueError:
        raise ValueError('提醒操作编号必须是 UUID') from None
    if action not in ('cancel', 'snooze'):
        raise ValueError('请选择取消或延后十分钟')
    expected_due = _seconds(expected_due)
    with store.database() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT * FROM reminder_changes WHERE request_id=?', (key,)).fetchone()
        if previous is not None:
            if (previous['task_id'], previous['action'], previous['expected_due']) != (task_id, action, expected_due):
                raise ReminderConflict('该请求编号已用于其他提醒操作')
            return {'id': task_id, 'action': action, 'due': previous['result_due']}
        clock = _seconds(time.time() if now is None else now)
        row = db.execute('SELECT kind,due,fired FROM tasks WHERE id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError('找不到该提醒')
        if row['kind'] != 'reminder' or type(row['due']) not in (int, float) or not math.isfinite(row['due']):
            raise ReminderConflict('仅支持管理有到期时间的提醒')
        if row['fired'] or row['due'] <= clock:
            raise ReminderConflict('该提醒已到期、已投递或已取消，请刷新列表')
        if row['due'] != expected_due:
            raise ReminderConflict('提醒时间已变更，请刷新列表')

        due = row['due']
        if action == 'cancel':
            # Both existing and older delivery workers select only fired=0.
            db.execute('UPDATE tasks SET fired=2 WHERE id=?', (task_id,))
        else:
            due += 600
            if not math.isfinite(due) or due <= expected_due:
                raise ValueError('该提醒时间超出可延后范围')
            db.execute('UPDATE tasks SET due=? WHERE id=?', (due, task_id))
        db.execute('INSERT INTO reminder_changes(request_id,task_id,action,expected_due,result_due) VALUES (?,?,?,?,?)',
                   (key, task_id, action, expected_due, due))
        return {'id': task_id, 'action': action, 'due': due}
