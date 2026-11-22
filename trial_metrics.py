"""Compare on-device posture records with a user's timed labels in a field trial."""
import math
from collections import Counter, defaultdict


TRUTH_STATES = ('sit', 'stand', 'away')
OUTCOMES = (*TRUTH_STATES, 'unknown', 'missing', 'privacy')
ESTIMATES = {'estimated_sit': 'sit', 'estimated_stand': 'stand'}


def _raw_edges(raw, start, end):
    edges = defaultdict(list)
    for row in raw:
        kind, state = row['kind'], row['state']
        if kind != 'person' and (kind != 'presence' or state != 'privacy'):
            continue
        if not (math.isfinite(row['start']) and math.isfinite(row['end'])):
            continue
        left, right = max(start, row['start']), min(end, row['end'])
        if left >= right:
            continue
        if state not in (*TRUTH_STATES, 'privacy') and state not in ESTIMATES:
            state = 'unknown'
        edges[left].append((state, 1))
        edges[right].append((state, -1))
    return edges


def _observation(active):
    if 'privacy' in active:
        return 'privacy', False
    if not active:
        return 'missing', False
    if len(active) > 1:
        # Conflicting raw person rows in an imported database have no unique verdict.
        return 'unknown', False
    state = next(iter(active))
    return ESTIMATES.get(state, state), state in ESTIMATES


def summarize(labels, raw, started_at, ended_at):
    """Sweep half-open intervals at actual changes, including both privacy clocks.

    A transition starts at the server-received mark. Its first matching person
    observation may be estimated. Missing or private time before that first hit
    makes the latency unmeasurable. Unknown observations remain in the clock.
    """
    if not all(math.isfinite(t) for t in (started_at, ended_at)) or ended_at < started_at:
        raise ValueError('Invalid trial time range')
    # Equal timestamps take the last received label. A mark at the end has no duration.
    marks = {label['at']: label['state'] for label in labels
             if started_at <= label['at'] < ended_at and label['state'] in (*TRUTH_STATES, 'pause')}
    edges = _raw_edges(raw, started_at, ended_at)
    boundaries = sorted({started_at, ended_at, *marks, *edges})
    matrix = {state: dict.fromkeys(OUTCOMES, 0.0) for state in TRUTH_STATES}
    estimated_matrix = {state: dict.fromkeys(ESTIMATES.values(), 0.0) for state in TRUTH_STATES}
    active = Counter()
    truth = pending_at = None
    latencies = []
    missed = unmeasurable = estimated_hits = 0
    paused = unlabeled = 0.0

    for left, right in zip(boundaries, boundaries[1:]):
        if left in marks and marks[left] != truth:
            if pending_at is not None:
                missed += 1
            following = marks[left]
            pending_at = left if truth in TRUTH_STATES and following in TRUTH_STATES else None
            truth = following
        for state, delta in edges.get(left, ()):
            active[state] += delta
            if active[state] == 0:
                del active[state]
        duration = right - left
        if truth is None:
            unlabeled += duration
            continue
        if truth == 'pause':
            paused += duration
            continue
        observed, estimated = _observation(active)
        matrix[truth][observed] += duration
        if estimated:
            estimated_matrix[truth][observed] += duration
        if pending_at is not None:
            if observed in ('missing', 'privacy'):
                unmeasurable += 1
                pending_at = None
            elif observed == truth:
                latencies.append(left - pending_at)
                estimated_hits += int(estimated)
                pending_at = None

    if pending_at is not None:
        missed += 1
    latencies.sort()
    measured = len(latencies)
    return {
        'matrix': matrix, 'estimated_matrix': estimated_matrix,
        'labeled_seconds': sum(sum(row.values()) for row in matrix.values()),
        'eligible_seconds': sum(value for row in matrix.values() for outcome, value in row.items()
                                if outcome != 'privacy'),
        'matched_seconds': sum(matrix[state][state] for state in TRUTH_STATES),
        'estimated_seconds': sum(sum(row.values()) for row in estimated_matrix.values()),
        'matched_estimated_seconds': sum(estimated_matrix[state][state] for state in ESTIMATES.values()),
        'paused_seconds': paused, 'unlabeled_seconds': unlabeled,
        'transitions': {
            'total': measured + missed + unmeasurable, 'measured': measured,
            'missed': missed, 'unmeasurable': unmeasurable, 'estimated_hits': estimated_hits,
            'p95_seconds': latencies[math.ceil(0.95 * measured) - 1] if measured else None,
        },
    }
