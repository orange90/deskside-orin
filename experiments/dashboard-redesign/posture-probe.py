import base64,json,time
from pathlib import Path
import cv2
from assistant import chat
prompts=[
 'Classify the visible person at the desk. A chair blocking the lower body is not enough to conclude sitting. Compare the visible torso and head with the seat and backrest. Choose away only if nobody is visible, or unknown if posture is unclear. Answer only one label: standing, sitting, away, unknown.',
 'Is the person sitting ON a chair, or standing IN FRONT OF the chair at a raised desk? Ignore the presence of an empty chair. Compare the torso and hips to the chair seat. If the body is hidden and posture cannot be determined, answer unknown. Answer only standing, sitting, or unknown.',
 'Describe whether the visible person is standing or sitting. Explain the visible body position relative to the chair seat and backrest in one sentence. If the person is occluded, say uncertain.',
 'Look at the person, not the chairs. Is their torso upright with hips ABOVE the chair seat, or are their hips resting ON a seat? Answer only standing, sitting, or unknown. The presence of a chair does not mean sitting.'
]
for file in ['standing-visible.jpg','sitting-visible.jpg']:
    frame=cv2.imread(str(Path('runtime/dashboard-redesign')/file))
    for scope in ['roi','full']:
        h,w=frame.shape[:2];image=frame if scope=='full' else frame[int(h*.04):int(h*.9),int(w*.25):int(w*.78)]
        ok,jpeg=cv2.imencode('.jpg',image)
        for prompt in prompts:
            t=time.monotonic();result=chat(8082,{'model':'cosmos','max_tokens':90,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg.tobytes()).decode()}},{'type':'text','text':prompt}]}]},40)
            print(json.dumps({'file':file,'scope':scope,'prompt':prompt,'output':result,'seconds':round(time.monotonic()-t,3)},ensure_ascii=False),flush=True)
