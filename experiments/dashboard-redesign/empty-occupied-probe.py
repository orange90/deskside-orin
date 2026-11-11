import base64,json,time
from pathlib import Path
import cv2
from assistant import chat
root=Path('runtime/dashboard-redesign')
# Evidence derivative only: crop the camera area out of a timestamped dashboard screenshot.
# Original screenshot is retained; this is not claimed to be an original camera JPEG.
shot=cv2.imread(str(root/'empty-desk-dashboard-source.png'));empty=shot[86:581,24:904]
cv2.imwrite(str(root/'user-confirmed-empty-derived.png'),empty)
occupied=cv2.imread(str(root/'away-evidence/20261111-143458.jpg'))
prompts=[
 'Can you clearly see that the chair at the desk is unoccupied? If a chair back blocks the seat and a seated person could be hidden, answer blocked. Answer only: occupied, empty, blocked.',
 'Ignore overlaid text. Examine the desk chair carefully, including the mesh back. Is there a human torso or body silhouette occupying the chair, even if the head is hidden, or is the chair clearly empty? Do not mistake clothing draped on a chair or a bag for a person. Answer only occupied, empty, or uncertain.',
 'Ignore overlaid text. Determine occupancy from visible human body evidence: a torso against the chair mesh, a head, an arm, or legs. Clothing draped over a chair and bags are not people. Answer only present, empty, or uncertain.'
]
for name,image in [('user_confirmed_occupied',occupied),('user_confirmed_empty_screenshot_crop',empty)]:
 ok,jpeg=cv2.imencode('.jpg',image)
 for prompt in prompts:
  t=time.monotonic();output=chat(8082,{'model':'cosmos','max_tokens':20,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(jpeg.tobytes()).decode()}},{'type':'text','text':prompt}]}]},40)
  print(json.dumps({'case':name,'prompt':prompt,'label':output,'seconds':round(time.monotonic()-t,3)},ensure_ascii=False),flush=True)
