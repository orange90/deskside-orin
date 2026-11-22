"""User corrections for recorded portions of a single local calendar day."""
from datetime import datetime, time, timedelta
import math
from uuid import UUID

import store


class CorrectionConflict(Exception):
    pass


def _key(request_id):
    try:
        return str(UUID(str(request_id)))
    except ValueError:
        raise ValueError('校正编号必须是 UUID') from None


def _covered_parts(db, start, end):
    gap_message = '选中的时段包含未采集的姿态记录，请刷新后重新选择'
    if db.execute("SELECT 1 FROM intervals WHERE kind IN ('person','presence') AND state='privacy' "
                  "AND end>start AND end>? AND start<? LIMIT 1", (start, end)).fetchone():
        raise CorrectionConflict('选中的时段含隐私保护，无法校正')
    parts = []
    for row in db.execute("SELECT start,end FROM intervals WHERE kind='person' AND end>start "
                          "AND end>? AND start<? ORDER BY start", (start, end)):
        left, right = max(start, row['start']), min(end, row['end'])
        if not parts:
            if left > start:
                raise CorrectionConflict(gap_message)
            parts.append([left, right])
        elif left <= parts[-1][1]:
            parts[-1][1] = max(parts[-1][1], right)
        elif left - parts[-1][1] < .001:
            # The dashboard merges sub-millisecond seams for display. Keep
            # that seam unlabelled on both clocks, even if presence spans it.
            parts.append([left, right])
        else:
            raise CorrectionConflict(gap_message)
    if not parts or parts[-1][1] < end:
        raise CorrectionConflict(gap_message)
    return parts


def create(request_id, day, start, end, state, now=None):
    key = _key(request_id)
    chosen = store.calendar_date(day)
    if state not in ('sit', 'stand', 'away'):
        raise ValueError('校正状态须为坐姿、站姿或离开')
    if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or end <= start:
        raise ValueError('请选择有效的起止时间')
    result = {'correction_id': key, 'date': day, 'start': start, 'end': end, 'state': state}
    with store.database() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT * FROM interval_corrections WHERE id=?', (key,)).fetchone()
        if previous:
            if any(previous[k] != result[k] for k in ('date', 'start', 'end', 'state')):
                raise CorrectionConflict('该请求编号已用于其他校正')
            if previous['undone']:
                raise CorrectionConflict('该次校正已撤销，请重新选择时段')
            return result

        now = (now or datetime.now(store.TZ)).astimezone(store.TZ)
        if chosen > now.date() or end > now.timestamp():
            raise ValueError('无法校正未来的时段')
        midnight = datetime.combine(chosen, time.min, store.TZ)
        if start < midnight.timestamp() or end > (midnight + timedelta(days=1)).timestamp():
            raise ValueError('时段须位于选中的同一个北京时间自然日内')
        if db.execute('SELECT 1 FROM activity_annotations WHERE end>? AND start<? LIMIT 1',
                      (start, end)).fetchone():
            raise CorrectionConflict('选中的时段已有人工标注，请刷新后重新选择')
        parts = _covered_parts(db, start, end)
        db.execute('INSERT INTO interval_corrections(id,date,start,end,state) VALUES (?,?,?,?,?)',
                   (key, day, start, end, state))
        for left, right in parts:
            annotation = db.execute(
                'INSERT INTO activity_annotations(start,end,state,reason,created) VALUES (?,?,?,?,?)',
                (left, right, state, '本人校正选中的活动时段', now.timestamp()))
            db.execute('INSERT INTO correction_parts(annotation_id,correction_id) VALUES (?,?)',
                       (annotation.lastrowid, key))
    return result


def undo(request_id):
    key = _key(request_id)
    with store.database() as db:
        db.execute('BEGIN IMMEDIATE')
        if not db.execute('SELECT 1 FROM interval_corrections WHERE id=?', (key,)).fetchone():
            raise KeyError('找不到该次时段校正')
        db.execute('DELETE FROM activity_annotations WHERE id IN '
                   '(SELECT annotation_id FROM correction_parts WHERE correction_id=?)', (key,))
        db.execute('UPDATE interval_corrections SET undone=1 WHERE id=?', (key,))
