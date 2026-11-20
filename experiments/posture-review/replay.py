import sys,json,base64,time
from pathlib import Path
root=Path.cwd()
stage=root/'runtime/standing-fix'
sys.path[:0]=[str(stage),str(root)]
import cv2
from assistant import describe_desk_frames
from main import desk_image
for name,path,mode in [('reported',stage/'reported-camera.jpg','person'),('camera',stage/'current.jpg','person'),('standing',root/'runtime/yolo26-eval/samples/standing.jpg','person'),('sitting',root/'runtime/yolo26-eval/samples/sitting.jpg','person'),('empty',root/'runtime/yolo26-eval/samples/empty.png','occupancy'),('occluded',root/'runtime/yolo26-eval/samples/occluded.jpg','occupancy')]:
    frame=cv2.imread(str(path))
    jpeg=desk_image(frame,[.25,.04,.78,.9],posture_detail=mode=='person')
    img={'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()}}
    t=time.monotonic()
    label,_=describe_desk_frames([img],posture_review=True,review_mode=mode)
    print(name,label,round(time.monotonic()-t,2),flush=True)
