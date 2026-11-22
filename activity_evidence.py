"""Read-only, bounded provenance summaries for one displayed activity interval."""
from collections import Counter
from datetime import datetime, time, timedelta
import math

import store
from state import POSE_METHODS


DISPLAY_STATES = {'sit', 'stand', 'away', 'unknown', 'privacy'}
RAW_STATES = DISPLAY_STATES | {'estimated_sit', 'estimated_stand'}


def _at(clock, left, end):
    """Advance a reverse-sorted interval list and return its value and next edge."""
    while clock and clock[-1]['end'] <= left:
        clock.pop()
    if not clock:
        return None, end
    row = clock[-1]
    if row['start'] > left:
        return None, min(end, row['start'])
    return row, min(end, row['end'])


def details(day, start, end, now=None):
    chosen = store.calendar_date(day)
    if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or end <= start:
        raise ValueError('请选择有效的起止时间')
    now = (now or datetime.now(store.TZ)).astimezone(store.TZ)
    if chosen > now.date() or end > now.timestamp():
        raise ValueError('无法查询未来的时段')
    midnight = datetime.combine(chosen, time.min, store.TZ)
    if start < midnight.timestamp() or end > (midnight + timedelta(days=1)).timestamp():
        raise ValueError('时段须位于选中的同一个北京时间自然日内')

    raw, methods, private = [], [], []
    with store.database() as db:
        db.execute('BEGIN')
        projected, _ = store.activity_window(db, start, end, now.timestamp())
        for row in db.execute(
                "SELECT kind,state,start,end FROM intervals WHERE kind IN ('person','presence','pose_method') "
                "AND (kind!='presence' OR state='privacy') AND end>start AND end>? AND start<? ORDER BY start DESC",
                (start, end)):
            if row['kind'] == 'person':
                raw.append(row)
            if row['kind'] == 'pose_method':
                methods.append(row)
            elif row['state'] == 'privacy':
                private.append(row)
    current = sorted((r for r in projected if r['kind'] == 'person'), key=lambda r: r['start'], reverse=True)

    raw_totals, displayed, sources, method_totals = Counter(), Counter(), Counter(), Counter()
    privacy_seconds = missing_seconds = method_missing_seconds = 0.0
    # Each clock can change inside another clock's interval. Split precisely at
    # their next real edge so methods never span a gap or a privacy pause.
    left = start
    while left < end:
        cells = [_at(clock, left, end) for clock in (raw, current, methods, private)]
        right = min(edge for _, edge in cells)
        duration = right - left
        observed, view, method, masked = (row for row, _ in cells)
        if masked is not None:
            privacy_seconds += duration
            if observed is not None:
                raw_totals['privacy'] += duration
                displayed['privacy'] += duration
        elif observed is None:
            missing_seconds += duration
        else:
            raw_state = observed['state']
            raw_totals[raw_state if raw_state in RAW_STATES else 'unknown'] += duration
            if view is not None:
                displayed[view['state'] if view['state'] in DISPLAY_STATES else 'unknown'] += duration
                source = view.get('source')
                sources[source if source in ('manual', 'continuity', 'estimated') else 'recorded'] += duration
            if method is None or method['state'] not in POSE_METHODS:
                method_missing_seconds += duration
            else:
                method_totals[method['state']] += duration
        left = right

    return {'date': day, 'start': start, 'end': end, 'raw': dict(raw_totals), 'displayed': dict(displayed),
            'sources': dict(sources), 'methods': dict(method_totals), 'privacy_seconds': privacy_seconds,
            'missing_seconds': missing_seconds, 'method_missing_seconds': method_missing_seconds}
