import json,time,statistics
from pathlib import Path
import cv2,numpy as np
from vision import Vision,crop
root=Path.cwd();lab=root.parent/'yolo26-eval'
v=Vision();assert v.model_info['engine_sha256']=='074860e9a8e7d94c857e6b803832229caf9d518739127bcf1097e473b32a57b5'
ref=json.loads((lab/'results/yolo26m-pose-engine/quality.json').read_text())
cases={'standing':'standing.jpg','sitting':'sitting.jpg','occluded':'occluded.jpg','empty':'empty.png'}
frames=[]
for r in ref:
 image=cv2.imread(str(lab/'samples'/cases[r['case']]))
 config={'person_roi':[.25,.04,.78,.9]} if r['view']=='workstation' else {}
 v.boxes=None;v.sample(image,config)
 actual=v.overlay;expected=r['detections']
 assert len(actual['detections'])==len(expected),(r['case'],actual,expected)
 dx,dy=(int(image.shape[1]*.25),int(image.shape[0]*.04)) if config else (0,0)
 for a,b in zip(actual['detections'],expected):
  np.testing.assert_allclose(np.array(a['box'])-[dx,dy,dx,dy],b['box'],atol=1.1,rtol=0)
  np.testing.assert_allclose(a['confidence'],b['confidence'],atol=.003)
 for a,b in zip(actual['keypoints'],r['selected_joints']):
  np.testing.assert_allclose(a['confidence'],b['confidence'],atol=.003)
  if b['confidence']>=.55:
   np.testing.assert_allclose([a['x']-dx,a['y']-dy],[b['x'],b['y']],atol=1.1,rtol=0)
 frames.append((image,config))
 print(json.dumps({'case':r['case'],'view':r['view'],'detections':len(expected),'high_confidence_joints':sum(p['confidence']>=.55 for p in actual['keypoints']),'parity':'PASS'}),flush=True)
for i in range(12):v.sample(*frames[i%8])
elapsed=[]
for repeat in range(10):
 for image,config in frames:
  v.boxes=None;t=time.perf_counter();v.sample(image,config);elapsed.append((time.perf_counter()-t)*1000);time.sleep(.02)
print(json.dumps({'native_backend_n':len(elapsed),'median_ms':statistics.median(elapsed),'p95_ms':float(np.percentile(elapsed,95)),'model':v.model_info}),flush=True)
