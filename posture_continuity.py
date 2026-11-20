"""Retrospective, read-only interpolation of short posture recognition gaps."""

MAX_GAP_SECONDS = 3.0


def bridge_posture_gaps(rows):
    """Require equal observed endpoints and uninterrupted presence; never extrapolate."""
    result = [dict(row) for row in rows]
    for row in result:
        if row['kind'] == 'person' and row['state'] in {'estimated_sit', 'estimated_stand'}:
            row.update(state=row['state'][10:], source='estimated')
    person = sorted((r for r in result if r['kind'] == 'person'), key=lambda r: r['start'])
    presence = sorted((r for r in result if r['kind'] == 'presence'), key=lambda r: r['start'])
    for i in range(1, len(person) - 1):
        before, gap, after = person[i-1:i+2]
        if (gap['state'] != 'unknown' or not 0 < gap['end']-gap['start'] <= MAX_GAP_SECONDS
                or before['state'] not in {'sit', 'stand'} or before['state'] != after['state']
                or before['end'] <= before['start'] or after['end'] <= after['start']
                or any(r.get('source') for r in (before, gap, after))
                or abs(before['end']-gap['start']) > .001
                or abs(gap['end']-after['start']) > .001):
            continue
        # Unknown/missing presence includes camera outages, dark frames and restarts.
        cursor = gap['start']
        covered = True
        for row in presence:
            if row['end'] <= gap['start'] or row['start'] >= gap['end']:
                continue
            if row['state'] != 'present' or row['start'] > cursor + .001:
                covered = False
                break
            cursor = max(cursor, row['end'])
        if covered and cursor >= gap['end']:
            gap.update(state=before['state'], source='continuity', original_state='unknown',
                       inferred_seconds=gap['end']-gap['start'],
                       inferred_ranges=[{'start': gap['start'], 'end': gap['end']}])
    return result
