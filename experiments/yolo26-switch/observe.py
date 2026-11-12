"""Bounded production observation; record metadata only, then verify preserved rows."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

stage=Path('/home/orange/Documents/orin-desk-agent/runtime/yolo26-switch')
samples=[]
for _ in range(45):
    with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=5) as response:
        camera=json.load(response)['camera']
    samples.append({key:camera.get(key) for key in ['captured_at','frame_count','fps','vision_ms','ready','vision_paused','person','presence','vision_model']})
    time.sleep(2)
(stage/'online-samples.json').write_text(json.dumps(samples,indent=2))
subprocess.run([sys.executable,str(stage/'verify-preserved.py')],check=True)
