"""Non-destructive human corrections, applied only when reading recorded intervals."""
from bisect import bisect_right


def _overlay(rows, start, end, update):
    changed = []
    for row in rows:
        left, right = max(row['start'], start), min(row['end'], end)
        if right <= left:
            changed.append(row)
            continue
        fields = update(row)
        if fields is None:
            changed.append(row)
            continue
        if row['start'] < left:
            changed.append({**row, 'end': left})
        changed.append({**row, 'start': left, 'end': right, **fields})
        if right < row['end']:
            changed.append({**row, 'start': right})
    return changed


def _mask_presence(rows):
    # Privacy is stored on the person clock. A single presence row can span
    # several pauses, especially when reviewing a full week of observations.
    events = {}
    for row in rows:
        if row['kind'] == 'person' and row['state'] == 'privacy' and row['end'] > row['start']:
            for edge, change in ((row['start'], 1), (row['end'], -1)):
                events[edge] = events.get(edge, 0) + change
    edges = sorted(events)
    depth = 0
    masked = [False]
    for edge in edges:
        depth += events[edge]
        masked.append(depth > 0)

    result = []
    for row in rows:
        left, end = row['start'], row['end']
        if not edges or row['kind'] != 'presence' or end <= left:
            result.append(dict(row))
            continue
        index = bisect_right(edges, left)
        while left < end:
            right = min(edges[index], end) if index < len(edges) else end
            result.append({**row, 'start': left, 'end': right,
                           'state': 'privacy' if masked[index] else row['state']})
            left = right
            index += 1
    return result


def apply_annotations(rows, annotations):
    result = _mask_presence(rows)
    for annotation in sorted(annotations, key=lambda item: item['id']):
        def correction(row):
            if row['kind'] not in {'person', 'presence'} or row['state'] == 'privacy':
                return None
            state = annotation['state']
            if row['kind'] == 'presence':
                state = 'away' if state == 'away' else 'present'
            return {'state': state, 'source': 'manual', 'annotation_id': annotation['id']}

        result = _overlay(result, annotation['start'], annotation['end'], correction)
    return result
