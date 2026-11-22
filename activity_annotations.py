"""Non-destructive human corrections, applied only when reading recorded intervals."""

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


def apply_annotations(rows, annotations):
    result = [dict(row) for row in rows]
    # On disk the person clock marks privacy, while the presence clock can
    # contain a single "unknown" row spanning both sides of that pause.
    for row in rows:
        if row['kind'] == 'person' and row['state'] == 'privacy':
            result = _overlay(result, row['start'], row['end'],
                              lambda r: {'state': 'privacy'} if r['kind'] == 'presence' else None)

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
