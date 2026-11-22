"""Audit daily projections against an existing isolated SQLite snapshot."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dashboard_data
import store


def check(root):
    if not (root / 'data.sqlite3').is_file():
        raise FileNotFoundError(root / 'data.sqlite3')
    store.ROOT = root
    now = datetime.now(store.TZ)
    with store.database() as db:
        integrity = db.execute('PRAGMA quick_check').fetchone()[0]
        first, count = db.execute('SELECT MIN(start), COUNT(*) FROM intervals').fetchone()
        manual = db.execute('SELECT COUNT(*) FROM activity_annotations').fetchone()[0]
        negative = db.execute('SELECT COUNT(*) FROM intervals WHERE end<start').fetchone()[0]
        overlap = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT start, MAX(end) OVER (
                    PARTITION BY kind ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS previous_end FROM intervals
            ) WHERE previous_end > start + 0.001
        """).fetchone()[0]
    assert integrity == 'ok' and negative == overlap == 0, (integrity, negative, overlap)
    start = datetime.fromtimestamp(first, store.TZ).date() if first is not None else now.date()
    days = (now.date() - start).days + 1
    for offset in range(days):
        day = start + timedelta(days=offset)
        data = dashboard_data.dashboard_snapshot(day.isoformat(), now=now)
        for h in data['hourly']:
            total = sum(v for k, v in h.items() if k != 'hour')
            assert 0 <= total <= 3600.001, (day, h['hour'], total)
        for state in ('sit', 'stand', 'away', 'privacy', 'unknown'):
            hourly = sum(h[state] for h in data['hourly'])
            assert abs(hourly - data['totals'].get(state, 0)) <= .501, (day, state)
        if day == now.date():
            assert data['totals'] == store.snapshot(now=now)['totals']
    return dict(days=days, raw_intervals=count, annotations=manual, sqlite=integrity,
                overlap=overlap, negative=negative, through=now.date().isoformat())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot_directory', type=Path)
    print(json.dumps(check(parser.parse_args().snapshot_directory)))
