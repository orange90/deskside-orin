"""Read-only dashboard projections. Raw intervals and notes are never rewritten."""
from collections import Counter
from datetime import date, datetime, time, timedelta
import store
from posture_continuity import MAX_GAP_SECONDS


def merge_intervals(rows):
    merged = []
    for row in sorted(rows, key=lambda r: (r['kind'], r['start'])):
        row = dict(row)
        if row['end'] <= row['start']:
            continue
        if (merged and merged[-1]['kind'] == row['kind']
                and merged[-1]['state'] == row['state']
                and (merged[-1].get('source') == row.get('source')
                     or {merged[-1].get('source'), row.get('source')} <= {None, 'continuity'})
                and merged[-1].get('annotation_id') == row.get('annotation_id')
                and abs(row['start'] - merged[-1]['end']) < .001):
            merged[-1]['end'] = max(merged[-1]['end'], row['end'])
            merged[-1]['samples'] += 1
            if row.get('inferred_seconds'):
                merged[-1]['inferred_seconds'] = merged[-1].get('inferred_seconds', 0) + row['inferred_seconds']
                merged[-1]['inferred_ranges'] = merged[-1].get('inferred_ranges', []) + row['inferred_ranges']
            if merged[-1].get('inferred_seconds'):
                merged[-1]['source'] = 'continuity'
        else:
            merged.append({**row, 'samples': 1})
    return merged


def group_notes(rows, intervals):
    groups = []
    # Group observations, not asserted continuous posture; privacy always splits them.
    person = sorted((r for r in intervals if r['kind'] == 'person' and r['state'] == 'privacy'), key=lambda r: r['start'])
    cursor = 0
    for row in sorted(rows, key=lambda r: (r['at'], r['id'])):
        while cursor < len(person) and person[cursor]['end'] < row['at']:
            cursor += 1
        segment = cursor if cursor >= len(person) or row['at'] < person[cursor]['start'] else None
        body = row['body'].strip()
        if (groups and row['kind'] == 'cosmos' and groups[-1]['kind'] == 'cosmos'
                and groups[-1]['body'] == body and segment is not None
                and groups[-1]['segment'] == segment and 0 <= row['at'] - groups[-1]['end'] <= 600):
            groups[-1]['end'] = row['at']
            groups[-1]['count'] += 1
        else:
            groups.append({'kind': row['kind'], 'body': body, 'start': row['at'],
                           'end': row['at'], 'count': 1, 'segment': segment})
    for group in groups:
        del group['segment']
    return list(reversed(groups))


def dashboard_snapshot(day=None, now=None):
    now = now or datetime.now(store.TZ)
    chosen = date.fromisoformat(day) if day else now.date()
    start = datetime.combine(chosen, time.min, store.TZ).timestamp()
    end = datetime.combine(chosen + timedelta(days=1), time.min, store.TZ).timestamp()
    cutoff = min(end, now.timestamp())
    with store.database() as db:
        db.execute('BEGIN')
        rows, annotations = store.activity_window(db, start, cutoff, now.timestamp())
        notes = [dict(r) for r in db.execute(
            'SELECT * FROM notes WHERE at>=? AND at<? ORDER BY at,id', (start, cutoff))]
        first = db.execute('SELECT MIN(start) FROM intervals').fetchone()[0]
        settings = dict(db.execute('SELECT key,value FROM settings'))
        today_start = datetime.combine(now.date(), time.min, store.TZ).timestamp()
        reminders = [dict(r) for r in db.execute(
            "SELECT id,body,at FROM notes WHERE kind='reminder' AND at>=? ORDER BY id DESC LIMIT 20", (today_start,))]
        tasks = [dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY rowid DESC LIMIT 50')]
    totals = store.activity_totals(rows)
    rows = merge_intervals(rows)
    inferred_seconds = sum(r.get('inferred_seconds', 0) for r in rows)
    hourly = [{'hour': h, 'sit': 0, 'stand': 0, 'away': 0, 'unknown': 0, 'privacy': 0} for h in range(24)]
    for r in rows:
        if r['kind'] == 'person':
            for h in range(24):
                overlap = max(0, min(r['end'], start + (h + 1) * 3600) - max(r['start'], start + h * 3600))
                hourly[h][r['state']] = hourly[h].get(r['state'], 0) + overlap
    return {'date': chosen.isoformat(), 'today': now.date().isoformat(), 'start': start, 'end': end,
            'as_of': now.timestamp(), 'cutoff': cutoff,
            'first_date': datetime.fromtimestamp(first, store.TZ).date().isoformat() if first else chosen.isoformat(),
            'totals': totals, 'hourly': hourly, 'intervals': rows,
            'continuity': {'max_gap_seconds': MAX_GAP_SECONDS, 'inferred_seconds': inferred_seconds},
            'annotations': annotations, 'notes': group_notes(notes, rows), 'raw_note_count': len(notes), 'active_reminders': reminders, 'settings': settings, 'tasks': tasks}


def _activity_day(db, day, now):
    start = datetime.combine(day, time.min, store.TZ).timestamp()
    cutoff = (now.timestamp() if day == now.date()
              else datetime.combine(day + timedelta(days=1), time.min, store.TZ).timestamp())
    rows, _ = store.activity_window(db, start, cutoff, now.timestamp())
    totals = dict.fromkeys(('sit', 'stand', 'away', 'unknown', 'privacy', 'present'), 0)
    totals.update(store.activity_totals(rows))
    coverage = {'person': 0.0, 'presence': 0.0}
    estimates = {'sit': 0.0, 'stand': 0.0}
    for row in rows:
        if row['kind'] not in coverage:
            continue
        duration = row['end'] - row['start']
        coverage[row['kind']] += duration
        if row['kind'] == 'person' and row.get('source') in {'estimated', 'continuity'}:
            estimates[row['state']] += duration
    return {'date': day.isoformat(), 'totals': totals,
            'estimated_sit': round(estimates['sit']), 'estimated_stand': round(estimates['stand']),
            'recorded_seconds': round(coverage['person']),
            'presence_recorded_seconds': round(coverage['presence']), 'elapsed_seconds': round(cutoff - start)}


def activity_week(end_date=None, now=None):
    """Seven local calendar days, sharing one database snapshot and wall clock."""
    now = (now or datetime.now(store.TZ)).astimezone(store.TZ)
    end = now.date() if end_date is None else store.calendar_date(end_date)
    if end > now.date():
        raise ValueError('结束日期不能晚于北京时间今天')
    if end.toordinal() < 7:
        raise ValueError('结束日期之前不足七个自然日')
    start = end - timedelta(days=6)
    with store.database() as db:
        db.execute('BEGIN')
        days = [_activity_day(db, start + timedelta(days=n), now) for n in range(7)]
    # Sum displayed daily seconds so drilling down to the daily dashboard agrees.
    totals = Counter()
    for day in days:
        totals.update(day['totals'])
    result = {'start_date': start.isoformat(), 'end_date': end.isoformat(), 'days': days, 'totals': dict(totals)}
    for key in ('estimated_sit', 'estimated_stand', 'recorded_seconds', 'presence_recorded_seconds', 'elapsed_seconds'):
        result[key] = sum(day[key] for day in days)
    return result
