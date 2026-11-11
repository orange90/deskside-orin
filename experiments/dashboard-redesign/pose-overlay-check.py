import importlib.util,json
from pathlib import Path
import cv2,numpy as np
root=Path.cwd();spec=importlib.util.spec_from_file_location('vision_candidate',root/'runtime/dashboard-redesign/vision_candidate.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.ROOT=root
vision=mod.Vision()
for name in ['standing-visible.jpg','sitting-visible.jpg','away-evidence/20261111-143458.jpg']:
    frame=cv2.imread(str(root/'runtime/dashboard-redesign'/name));vision.boxes=None;result=vision.sample(frame,{'person_roi':[.25,.04,.78,.9]});overlay=vision.overlay
    assert len(overlay['keypoints']) in (0,17)
    assert all(np.isfinite(p[k]) for p in overlay['keypoints'] for k in ('x','y','confidence'))
    print(json.dumps({'file':name,'raw_posture':result[1],'people':result[3],'strong':result[4],'pose_source':result[-1],'overlay':overlay}),flush=True)
# Verify SimCC-to-image coordinate transform against known synthetic peaks.
class DummyPose:
    def __call__(self,*args):
        x=np.zeros((1,17,384));y=np.zeros((1,17,512));x[:,:,192]=1;y[:,:,256]=1;return x,y
v=object.__new__(mod.Vision);v.pose=DummyPose();_,_,points=v.body_cues(np.zeros((720,1280,3),np.uint8),[100,100,300,500])
assert all(abs(p['x']-200)<1e-6 and abs(p['y']-300)<1e-6 for p in points)
print('Overlay finite/17-joint and original-coordinate transform PASS',flush=True)
