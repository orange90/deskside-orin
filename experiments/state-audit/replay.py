"""Replay private fixtures through the real camera loop, Cosmos worker, and a fresh SQLite DB.

Run in an isolated checkout staged on the Orin, using the production .venv and DLA
library paths. Example: python experiments/state-audit/replay.py --assets /path/to/orin-desk-agent
The only camera adapter is a pre-existing image sequence. No camera device is opened.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
import time
from unittest.mock import patch


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('result.json'))
    parser.add_argument('--transition-only', action='store_true', help='Repeat just the first sit-to-stand transition')
    opts = parser.parse_args()
    candidate = Path.cwd().resolve()
    assets = opts.assets.resolve()
    if candidate == assets:
        raise SystemExit('Select an isolated checkout as the working directory')
    sys.path.insert(0, str(candidate))
    import cv2
    import numpy as np
    import main
    import store
    import vision

    # Changing the camera, calibration path and database root leaves all selection,
    # timing, queue and persistence functions on their production paths.
    sources = assets/'runtime'
    plan = [
        ('sit', sources/'yolo26-eval/samples/sitting.jpg', 20, 'sit'),
        ('raw_stand', sources/'standing-fix/current.jpg', 20, 'stand'),
        ('black', None, 6, 'privacy'),
        ('reported_stand', sources/'standing-fix/reported-camera.jpg', 20, 'stand'),
        ('empty', sources/'yolo26-eval/samples/empty.png', 30, 'away'),
    ]
    if opts.transition_only:
        plan = plan[:2]
    phases = []
    for name, path, seconds, expected in plan:
        frame = cv2.imread(str(path)) if path else np.zeros_like(phases[-1]['image'])
        if frame is None:
            raise FileNotFoundError(path)
        phases.append(dict(name=name, seconds=seconds, expected=expected, image=frame,
                           counts=Counter(), tail=Counter(), sources=Counter(), fps=[], transitions=[]))
    main.CAL = assets/'calibration.json'
    vision.ROOT = assets  # Read-only YOLO engine.
    class Daytime(datetime):
        @classmethod
        def now(cls, tz=None):
            return super().now(tz).replace(hour=12)

    phase_index = 0
    phase_clock = None
    class ReplayCamera:
        def __init__(self, *_):
            pass

        def set(self, *_):
            pass

        def release(self):
            pass

        def read(self):
            nonlocal phase_clock, phase_index
            now = time.monotonic()
            if phase_clock is None:
                phase_clock = now
                phases[0]['start'] = time.time()
            elif now-phase_clock >= phases[phase_index]['seconds']:
                phases[phase_index]['end'] = time.time()
                print(phases[phase_index]['name'], dict(phases[phase_index]['counts']),
                      'last 5s', dict(phases[phase_index]['tail']), flush=True)
                if phase_index+1 == len(phases):
                    main.stop.set()
                else:
                    phase_index += 1
                    phase_clock = now
                    phases[phase_index]['start'] = time.time()
            return True, phases[phase_index]['image']

    publish_original = main.publish
    def observe(frame, **state):
        publish_original(frame, **state)
        if main.stop.is_set():
            return
        phase = phases[phase_index]
        effective = 'privacy' if state.get('privacy_active') else state['person']
        age = round(time.monotonic()-phase_clock, 3)
        state_key = [effective, state['presence'], state.get('pose_source')]
        events = phase['transitions']
        if not events or events[-1]['state_presence_source'] != state_key:
            events.append({'t':age,'state_presence_source':state_key})
        phase['counts'][effective] += 1
        phase['sources'][state.get('pose_source') or 'none'] += 1
        if time.monotonic()-phase_clock >= phase['seconds']-5:
            phase['tail'][effective] += 1
        phase['fps'].append(state['fps'])
        if state.get('vision_paused') and (frame or state.get('overlay')):
            raise AssertionError('Paused frame leaked preview or overlay')

    rootfiles = ['main.py', 'state.py', 'vision.py', 'yolo_pose.py','assistant.py',
                 'store.py', 'activity_annotations.py', 'posture_continuity.py']
    report = {'hashes': {f:hashlib.sha256((candidate/f).read_bytes()).hexdigest() for f in rootfiles},
              'conditions': 'static stills, actual sampling clock, local Cosmos and TensorRT, '
                            'daytime forced only for privacy policy', 'phases': []}
    with tempfile.TemporaryDirectory(prefix='replay-db-',dir=candidate) as dbroot:
        store.ROOT = Path(dbroot)
        store.initialize()
        worker = Thread(target=main.cosmos,daemon=True)
        worker.start()
        try:
            with patch.object(cv2,'VideoCapture',ReplayCamera), patch.object(main,'publish',observe), patch.object(main,'datetime',Daytime):
                main.capture_frames()
        finally:
            main.stop.set()
            worker.join(timeout=35)
        if worker.is_alive():
            raise RuntimeError('Replay worker did not stop')
        with store.database() as db:
            rows = [dict(x) for x in db.execute("SELECT * FROM intervals ORDER BY kind,start")]
        for phase in phases:
            accumulated = defaultdict(float)
            for row in rows:
                seconds = max(0,min(phase['end'],row['end'])-max(phase['start'],row['start']))
                if seconds and row['kind'] in {'presence','person'}:
                    accumulated[row['kind']+'/'+row['state']] += seconds
            events = phase['transitions']
            first_match = next((e['t'] for e in events if e['state_presence_source'][0] == phase['expected']), None)
            sustained_start = None
            for e in reversed(events):
                if e['state_presence_source'][0] != phase['expected']:
                    break
                sustained_start = e['t']
            report['phases'].append(dict(name=phase['name'], expected=phase['expected'],
                first_match_seconds=first_match, sustained_match_seconds=sustained_start,
                image_shape=list(phase['image'].shape[:2]),
                frames=dict(phase['counts']), final_5s=dict(phase['tail']), sources=dict(phase['sources']),
                transitions=phase['transitions'],
                median_fps=round(float(np.median(phase['fps'])),2),
                database_seconds={k:round(v,2) for k,v in accumulated.items()}))
    opts.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    run()
