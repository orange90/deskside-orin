import base64,importlib.util,json,time
from pathlib import Path
import cv2,numpy as np
root=Path.cwd();spec=importlib.util.spec_from_file_location('assistant_candidate',root/'runtime/dashboard-redesign/assistant_candidate.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
frame=cv2.imread(str(root/'runtime/dashboard-redesign/away-evidence/20261111-143458.jpg'))
for name,image in [('user_confirmed_occluded',frame[28:648,320:998]),('synthetic_empty',np.full((620,678,3),210,dtype=np.uint8))]:
    ok,jpeg=cv2.imencode('.jpg',image);frames=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg.tobytes()).decode()}}]
    for attempt in range(3):
        t=time.monotonic();label,body=module.describe_desk_frames(frames,posture_review=True)
        print(json.dumps({'case':name,'attempt':attempt,'label':label,'summary':body,'seconds':round(time.monotonic()-t,3)},ensure_ascii=False),flush=True)
        if name=='user_confirmed_occluded':assert label in {'sitting','present','occluded','unknown'},label
        else:assert label in {'away','unknown','occluded'},label
print('Candidate replay PASS; occluded user never marked away; synthetic empty is not a field accuracy test',flush=True)
