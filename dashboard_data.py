"""Read-only dashboard projections. Raw intervals and notes are never rewritten."""
from collections import Counter
from datetime import date, datetime, time, timedelta
import store
from idea_search import END_IDEA_AT, MIN_IDEA_AT, beijing_day
from posture_continuity import MAX_GAP_SECONDS


def captured_idea_day(request_id):
    with store.database() as db:
        row = db.execute("SELECT n.at FROM idea_requests r JOIN notes n ON n.id=r.note_id "
                         "WHERE r.request_id=? AND n.kind='idea'", (str(request_id),)).fetchone()
    return datetime.fromtimestamp(row['at'], store.TZ).date().isoformat() if row else None


def daily_ideas_markdown(day, now=None):
    now = now or datetime.now(store.TZ)
    chosen = store.calendar_date(day)
    if chosen > now.astimezone(store.TZ).date():
        raise ValueError("无法导出未来日期的灵感")
    start = datetime.combine(chosen, time.min, store.TZ)
    cutoff = min((start + timedelta(days=1)).timestamp(), now.timestamp())
    lines = [f"# 写作灵感 · {chosen.isoformat()}", "", "记录时间为北京时间。"]
    with store.database() as db:
        for row in db.execute("SELECT at,body FROM notes WHERE kind='idea' AND at>=? AND at<? "
                              "ORDER BY at,id", (start.timestamp(), cutoff)):
            stamp = datetime.fromtimestamp(row['at'], store.TZ).strftime('%H:%M:%S')
            lines.extend(["", f"## {stamp}", "", row['body']])
    return f"deskside-ideas-{chosen.isoformat()}.md", "\n".join(lines) + "\n"


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
        body = row['body'] if row['kind'] == 'idea' else row['body'].strip()
        if (groups and row['kind'] == 'cosmos' and groups[-1]['kind'] == 'cosmos'
                and groups[-1]['body'] == body and segment is not None
                and groups[-1]['segment'] == segment and 0 <= row['at'] - groups[-1]['end'] <= 600):
            groups[-1]['end'] = row['at']
            groups[-1]['count'] += 1
        else:
            groups.append({'kind': row['kind'], 'body': body, 'start': row['at'],
                           'end': row['at'], 'count': 1, 'segment': segment})
            if row['kind'] == 'idea':
                groups[-1]['idea_id'] = str(row['id'])
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
        first = db.execute("SELECT min(day_start) FROM ("
                           "SELECT min(start) AS day_start FROM intervals WHERE kind!='pose_method' "
                           "UNION ALL SELECT min(at) FROM notes WHERE kind='idea' AND at>=? AND at<? "
                           "AND typeof(at) IN ('integer','real'))",
                           (MIN_IDEA_AT, min(END_IDEA_AT, now.timestamp()))).fetchone()[0]
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
            'first_date': beijing_day(first) if first is not None else chosen.isoformat(),
            'totals': totals, 'hourly': hourly, 'intervals': rows,
            'continuity': {'max_gap_seconds': MAX_GAP_SECONDS, 'inferred_seconds': inferred_seconds},
            'annotations': annotations, 'notes': group_notes(notes, rows), 'raw_note_count': len(notes), 'active_reminders': reminders, 'settings': settings, 'tasks': tasks}


def _activity_summary(db, start, cutoff, as_of, *, presence_breakdown=False):
    rows, _ = store.activity_window(db, start, cutoff, as_of)
    totals = dict.fromkeys(('sit', 'stand', 'away', 'unknown', 'privacy', 'present'), 0)
    totals.update(store.activity_totals(rows))
    coverage = {'person': 0.0, 'presence': 0.0}
    manual = {'person': 0.0, 'presence': 0.0}
    estimates = {'sit': 0.0, 'stand': 0.0}
    presence_states = {'unknown': 0.0, 'privacy': 0.0} if presence_breakdown else {}
    for row in rows:
        if row['kind'] not in coverage:
            continue
        duration = row['end'] - row['start']
        coverage[row['kind']] += duration
        if row.get('source') == 'manual':
            manual[row['kind']] += duration
        if row['kind'] == 'person' and row.get('source') in {'estimated', 'continuity'}:
            estimates[row['state']] += duration
        if row['kind'] == 'presence' and row['state'] in presence_states:
            presence_states[row['state']] += duration
    return {'totals': totals,
            **{f'presence_{state}_seconds': round(seconds) for state, seconds in presence_states.items()},
            'estimated_sit': round(estimates['sit']), 'estimated_stand': round(estimates['stand']),
            'manual_seconds': round(manual['person']), 'manual_presence_seconds': round(manual['presence']),
            'recorded_seconds': round(coverage['person']),
            'presence_recorded_seconds': round(coverage['presence']), 'elapsed_seconds': round(cutoff - start)}


def _activity_day(db, day, now):
    start = datetime.combine(day, time.min, store.TZ).timestamp()
    cutoff = (now.timestamp() if day == now.date()
              else datetime.combine(day + timedelta(days=1), time.min, store.TZ).timestamp())
    return {'date': day.isoformat(), **_activity_summary(db, start, cutoff, now.timestamp())}


def _minute_timestamp(text):
    try:
        value = datetime.fromisoformat(text) if isinstance(text, str) and len(text) == 16 else None
    except ValueError:
        value = None
    if value is None or value.isoformat(timespec='minutes') != text:
        raise ValueError('时间请使用北京时间 YYYY-MM-DDTHH:MM')
    local = value.replace(tzinfo=store.TZ)
    if local.replace(fold=1).utcoffset() > local.utcoffset():
        raise ValueError('该本地时间不存在')
    return local.timestamp()


def activity_range(start_text, end_text, now=None):
    """Summarize a half-open Beijing-time window on the daily report's observation clocks."""
    start, end = _minute_timestamp(start_text), _minute_timestamp(end_text)
    if not 0 < end - start <= 86400:
        raise ValueError('结束时间须晚于开始时间，且时段不能超过 24 小时')
    now = now or datetime.now(store.TZ)
    as_of = (now if now.tzinfo else now.replace(tzinfo=store.TZ)).timestamp()
    if end > as_of:
        raise ValueError('结束时间不能晚于当前时间')
    with store.database() as db:
        db.execute('BEGIN')
        return {'start': start_text, 'end': end_text,
                **_activity_summary(db, start, end, as_of, presence_breakdown=True)}


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
    for key in ('estimated_sit', 'estimated_stand', 'manual_seconds', 'manual_presence_seconds',
                'recorded_seconds', 'presence_recorded_seconds', 'elapsed_seconds'):
        result[key] = sum(day[key] for day in days)
    return result


def _clock_duration(seconds):
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def _markdown_day(label, stats):
    cells = [label]
    for state in ('sit', 'stand', 'present', 'away', 'unknown', 'privacy'):
        source = 'presence_recorded_seconds' if state == 'present' else 'recorded_seconds'
        cells.append(_clock_duration(stats['totals'][state]) if stats[source] else '未采集')
    cells.append(_clock_duration(stats['elapsed_seconds'] - stats['recorded_seconds']))
    return '| ' + ' | '.join(cells) + ' |'


def weekly_markdown(end_date=None, now=None):
    """An Obsidian-ready download made entirely from the aggregate weekly snapshot."""
    now = (now or datetime.now(store.TZ)).astimezone(store.TZ)
    week = activity_week(end_date, now)
    end = week['end_date']
    cutoff = (f'今天尚未结束，统计截至北京时间 {now:%Y-%m-%d %H:%M:%S}。' if end == now.date().isoformat()
              else '统计范围为北京时间的七个完整自然日。')
    lines = [f"# 工位活动回顾 · {week['start_date']} 至 {end}", '', cutoff, '',
             '| 日期 | 坐姿 | 站姿 | 工位在位 | 离开 | 姿态未知 | 隐私保护 | 姿态未采集 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    lines.extend(_markdown_day(day['date'], day) for day in week['days'])
    lines.append(_markdown_day('七日合计', week))

    pose = week['recorded_seconds']
    presence = week['presence_recorded_seconds']
    lines.extend(['', '## 统计口径', '',
                  f"- 时长按时、分、秒列出。姿态记录覆盖 {_clock_duration(pose)}，"
                  f"独立的在位记录覆盖 {_clock_duration(presence)}。未采集表示对应记录缺失。",
                  '- 姿态未知表示已有记录但未判断出姿态。隐私保护和缺少记录的时段分别列出。'])
    if pose:
        lines.append(f"- 坐姿中含推算 {_clock_duration(week['estimated_sit'])}，"
                     f"站姿中含推算 {_clock_duration(week['estimated_stand'])}。"
                     '推算包含在位时估计的姿态和短暂缺口补齐。')
    manual_pose = _clock_duration(week['manual_seconds']) if pose else '未采集'
    manual_presence = _clock_duration(week['manual_presence_seconds']) if presence else '未采集'
    lines.append(f"- 本人校正已计入对应状态。姿态记录中的校正时长为 {manual_pose}，"
                 f"在位记录中的校正时长为 {manual_presence}。")
    return f'deskside-week-{end}.md', '\n'.join(lines) + '\n'
