"""Non-destructive human corrections, applied only when reading recorded intervals."""

def apply_annotations(rows, annotations):
    result = [dict(row) for row in rows]
    for annotation in sorted(annotations, key=lambda item: item['id']):
        changed = []
        for row in result:
            left, right = max(row['start'], annotation['start']), min(row['end'], annotation['end'])
            if row['kind'] not in {'person', 'presence'} or row['state'] == 'privacy' or right <= left:
                changed.append(row)
                continue
            if row['start'] < left:
                changed.append({**row, 'end': left})
            state = annotation['state'] if row['kind'] == 'person' else ('away' if annotation['state'] == 'away' else 'present')
            changed.append({**row, 'start': left, 'end': right, 'state': state,
                            'source': 'manual', 'annotation_id': annotation['id']})
            if right < row['end']:
                changed.append({**row, 'start': right})
        result = changed
    return result
