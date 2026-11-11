import json,time
from pathlib import Path
from urllib.request import urlopen
root=Path('runtime/dashboard-redesign/away-evidence');root.mkdir(exist_ok=True)
end=time.monotonic()+120;captured=0;last=0
while time.monotonic()<end and captured<3:
    try:
        with urlopen('http://127.0.0.1:8765/api/state',timeout=3) as r: before=json.load(r)['camera']
        if before.get('person')=='away' and not before.get('vision_paused') and time.monotonic()-last>5:
            with urlopen('http://127.0.0.1:8765/api/calibration/frame',timeout=3) as r: image=r.read()
            with urlopen('http://127.0.0.1:8765/api/state',timeout=3) as r: after=json.load(r)['camera']
            if before.get('frame_count')==after.get('frame_count') and after.get('person')=='away' and not after.get('vision_paused'):
                name=time.strftime('%Y%m%d-%H%M%S');(root/(name+'.jpg')).write_bytes(image);(root/(name+'.json')).write_text(json.dumps({'at':time.time(),'before':before,'after':after},indent=2));captured+=1;last=time.monotonic();print('Captured same-frame away evidence',name,flush=True)
    except Exception as e:print(type(e).__name__,flush=True)
    time.sleep(.15)
print('Completed',captured,flush=True)
