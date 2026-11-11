"""Local-only static-photo comparison; never connects to a camera or changes production."""
import os
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false', MPLBACKEND='Agg')
import argparse, hashlib, json, sys, time, resource, urllib.request
from pathlib import Path
import cv2, numpy as np
APP=Path(os.environ.get('ORIN_APP_ROOT', '/home/orange/Documents/orin-desk-agent'))
ROOT=APP/'runtime/yolo26-eval'
sys.path.insert(0,str(APP))
from vision import Vision, crop, leg_posture
parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--backend',default='pt');parser.add_argument('--export',action='store_true');parser.add_argument('--review',action='store_true');args=parser.parse_args()
import torch
from ultralytics import YOLO, settings
settings.update({'sync':False})
cv2.setNumThreads(1);torch.set_num_threads(2)
if args.export:
    YOLO(ROOT/'models'/f'{args.model}.pt').export(format='engine',imgsz=640,batch=1,device=0,half=True,workspace=2,simplify=False,opset=17,dynamic=False,verbose=False)
    raise SystemExit(0)
name=args.model+'-'+args.backend+('-review' if args.review else '')
out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
ROI=[.25,.04,.78,.9]
samples=[]
for case,filename,label in [('standing','standing.jpg','stand'),('sitting','sitting.jpg','sit'),('occluded','occluded.jpg','sit'),('empty','empty.png','away')]:
    frame=cv2.imread(str(ROOT/'samples'/filename));assert frame is not None
    for view,roi in [('full',[0,0,1,1]),('workstation',ROI)]:
        image,_,_=crop(frame,roi)
        samples.append((case,view,label,image))
if args.review:
    samples=[]
    for case,filename,label in [('U01','standing-old.jpg','unconfirmed_stand'),('U02','empty.png','away')]:
        frame=cv2.imread(str(ROOT/'samples'/filename));assert frame is not None
        samples.append((case,'full',label,frame))
model_path=ROOT/'models'/f'{args.model}.{args.backend}'
base=args.model=='rtmdet-rtmpose'
model=Vision() if base else YOLO(model_path,task='pose')

def infer(image):
    if base:
        model.boxes=None;model.rise.clear();model.background=None
        values=model.sample(image,{})
        detections=model.overlay['detections']
        best=max(detections,key=lambda d:d['confidence'],default=None)
        joints=model.overlay['keypoints']
        return {'detections':detections,'selected_joints':joints,'posture':values[1],
                'selected_box':best['box'] if best else None,
                'posture_source':values[-1]}
    result=model.predict(image,imgsz=640,rect=False,device=0,half=True,conf=.28,iou=.45,verbose=False,save=False)[0]
    boxes=result.boxes.data.cpu().numpy();kpts=result.keypoints.data.cpu().numpy()
    dets=[{'box':[float(x) for x in b[:4]],'confidence':float(b[4])} for b in boxes]
    index=int(np.argmax(boxes[:,4])) if len(boxes) else None
    joints=[{'x':float(p[0]),'y':float(p[1]),'confidence':float(p[2])} for p in kpts[index]] if index is not None else []
    pose=None
    if index is not None:
        b=boxes[index];width=max(b[2]-b[0],(b[3]-b[1])*.75)*1.25
        # Same geometric thresholds as RTMPose's 192x256 normalized crop.
        pose=leg_posture(kpts[index,:,:2]*(192/max(width,1)),kpts[index,:,2])
    return {'detections':dets,'selected_joints':joints,'posture':pose or ('away' if index is None else 'unknown'),
            'selected_box':dets[index]['box'] if index is not None else None,
            'posture_source':'keypoints' if pose else None,
            'all_keypoints':kpts.tolist(),'ultralytics_speed_ms':result.speed}

bones=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
def render(image,result,caption):
    img=image.copy()
    for d in result['detections']:
        x1,y1,x2,y2=map(round,d['box']);cv2.rectangle(img,(x1,y1),(x2,y2),(0,220,130),2)
        cv2.putText(img,f"person {d['confidence']:.2f}",(x1,max(35,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,130),1)
    p=result['selected_joints']
    valid=lambda i:len(p)>i and p[i]['confidence']>=.55
    xy=lambda i:(round(p[i]['x']),round(p[i]['y']))
    for a,b in bones:
        if valid(a) and valid(b):cv2.line(img,xy(a),xy(b),(0,240,240),3)
    for i in range(len(p)):
        if valid(i):cv2.circle(img,xy(i),4,(0,180,255),-1)
    cv2.rectangle(img,(0,0),(img.shape[1],30),(20,35,25),-1)
    cv2.putText(img,caption,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    return img

for i in range(12):infer(samples[i%len(samples)][3])
quality=[]
for case,view,label,image in samples:
    r=infer(image);r.update(case=case,view=view,label=label,visible_keypoints=sum(p['confidence']>=.55 for p in r['selected_joints']))
    quality.append(r)
    cv2.imwrite(str(out/f'{case}-{view}.jpg'),render(image,r,f"{name} | {case} | {r['posture']} | kpt>=.55: {r['visible_keypoints']}/17"))
    print(json.dumps({'stage':'quality',**{k:r[k] for k in ['case','view','posture','visible_keypoints']},'detections':len(r['detections'])}),flush=True)
(out/'quality.json').write_text(json.dumps(quality,indent=2))
if args.review:raise SystemExit(0)
latency=[];health=[]
for repeat in range(25):
    for case,view,label,image in samples:
        torch.cuda.synchronize();start=time.perf_counter();r=infer(image);torch.cuda.synchronize()
        latency.append({'case':case,'view':view,'ms':(time.perf_counter()-start)*1000})
        time.sleep(.05)
    if repeat%5==0:
        try:
            with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=3) as response:c=json.load(response)['camera']
            health.append({'at':time.time(),'ready':c.get('ready'),'fps':c.get('fps'),'frame_count':c.get('frame_count')})
        except Exception as e:health.append({'error':type(e).__name__})

def stats(values):
    return {'n':len(values),'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),'max_ms':max(values)}
report={'model':name,'measurement':'synchronized wall time, decoded image -> CPU boxes/keypoints and geometric label; no disk IO/render; production services running; 30W',
        'latency':stats([r['ms'] for r in latency]),'by_view':{v:stats([r['ms'] for r in latency if r['view']==v]) for v in ['full','workstation']},
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'production_health':health,
        'file_bytes':model_path.stat().st_size if model_path.exists() else None,
        'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path.exists() else None}
(out/'benchmark.json').write_text(json.dumps(report,indent=2));(out/'timings.json').write_text(json.dumps(latency))
print(json.dumps(report),flush=True)
