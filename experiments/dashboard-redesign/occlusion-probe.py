import base64,json,time
from pathlib import Path
import cv2
from assistant import chat
root=Path.cwd();frame=cv2.imread(str(root/'runtime/dashboard-redesign/away-evidence/20261111-143458.jpg'))
prompts=[
 'Is any part of a human visible near the desk, such as a head, torso or arm behind a chair? Ignore furniture. Answer only yes or no.',
 'Assess whether the desk is occupied. A large chair back can completely hide a seated person. Do not equate no visible head or legs with nobody present. Answer one label: present (any human body part visible), occluded (chair or furniture prevents confirming whether a person is there), empty (work area clearly visible and nobody there).',
 'Can you clearly see that the chair at the desk is unoccupied? If a chair back blocks the seat and a seated person could be hidden, answer blocked. Answer only: occupied, empty, blocked.',
 'Describe the visible evidence of a person at this desk and whether the chair blocks the view. Do not assume the person left just because the head is hidden. Answer in one short sentence.'
]
for scope in ['roi','full']:
    image=frame if scope=='full' else frame[28:648,320:998]
    ok,jpeg=cv2.imencode('.jpg',image)
    for prompt in prompts:
        begin=time.monotonic();result=chat(8082,{'model':'cosmos','max_tokens':70,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg.tobytes()).decode()}},{'type':'text','text':prompt}]}]},40)
        print(json.dumps({'scope':scope,'prompt':prompt,'output':result,'seconds':round(time.monotonic()-begin,3)},ensure_ascii=False),flush=True)
