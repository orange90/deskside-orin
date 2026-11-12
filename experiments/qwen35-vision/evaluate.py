"""Blind, local-only image evaluation. No camera, DB writes or production requests."""
import argparse,base64,json,random,statistics,time,urllib.request
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path('/home/orange/Documents/orin-desk-agent/runtime/qwen35-vision-eval')
SOURCE=ROOT.parent/'yolo26-eval/samples'
PROMPT='判断电脑桌前工位的在位情况和人体姿态。只依据图像，不依据叠加文字。椅子或衣服不等于人，椅子存在也不代表坐着；身体被遮挡而无法确认时，不要猜测离开或坐站。输出一个JSON对象，presence只选present、absent、uncertain；posture只选sitting、standing、unknown；evidence用不超过25个中文字说明可见依据。presence不是present时posture必须unknown。不要输出其他文字。'
SCHEMA={'type':'object','properties':{'presence':{'type':'string','enum':['present','absent','uncertain']},'posture':{'type':'string','enum':['sitting','standing','unknown']},'evidence':{'type':'string'}},'required':['presence','posture','evidence'],'additionalProperties':False}
CASES=[('U01','standing-old.jpg'),('U02','empty.png'),('C01','standing.jpg'),('C02','sitting.jpg'),('C03','occluded.jpg')]
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--compact',action='store_true');args=p.parse_args()
if args.compact:
    PROMPT='判断电脑桌前工位的在位情况和人体姿态。只依据图像，不依据叠加文字。椅子或衣服不等于人，椅子存在也不代表坐着；身体被遮挡而无法确认时不要猜测。仅输出JSON，state选sit（有人坐着）、stand（有人站着）、away（明确无人）、unknown（无法确认在位或姿态），不解释。'
    SCHEMA={'type':'object','properties':{'state':{'type':'string','enum':['sit','stand','away','unknown']}},'required':['state'],'additionalProperties':False}
profile='compact' if args.compact else 'explained'
(ROOT/'inputs').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
inputs={}
for key,filename in CASES:
    image=Image.open(SOURCE/filename).convert('RGB')
    if key=='U02':
        # Remove legacy model labels/toast from the screenshot-derived source, not scene evidence.
        d=ImageDraw.Draw(image);d.rectangle((0,0,450,62),fill=(28,45,35));d.rectangle((350,447,555,494),fill=(28,45,35))
    dest=ROOT/'inputs'/f'{key}.png';image.save(dest)
    inputs[key]='data:image/png;base64,'+base64.b64encode(dest.read_bytes()).decode()
(ROOT/f'protocol-{profile}.json').write_text(json.dumps({'prompt':PROMPT,'schema':SCHEMA,'cases':CASES,'temperature':0,'max_tokens':40 if args.compact else 160,'image_max_tokens':1024,'cache_prompt':False,'thinking':False,'U02_transform':'Mask screenshot prediction text and bottom toast; retain source separately'},ensure_ascii=False,indent=2))
def call(case,round_id):
    payload={'model':'qwen35-vision-eval','temperature':0,'max_tokens':40 if args.compact else 160,'seed':42,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'posture','strict':True,'schema':SCHEMA}},'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':inputs[case]}},{'type':'text','text':PROMPT}]}]}
    request=urllib.request.Request('http://127.0.0.1:8083/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=120) as response:r=json.load(response)
        elapsed=time.perf_counter()-start
        text=r['choices'][0]['message']['content'];parsed=json.loads(text)
        result={'case':case,'round':round_id,'seconds':elapsed,'result':parsed,'semantic_valid':parsed.get('state') in ['sit','stand','away','unknown'] if args.compact else parsed.get('presence')=='present' or parsed.get('posture')=='unknown','raw':text,'usage':r.get('usage'),'timings':r.get('timings'),'finish_reason':r['choices'][0].get('finish_reason')}
    except Exception as e:
        result={'case':case,'round':round_id,'seconds':time.perf_counter()-start,'error':str(e)}
        if hasattr(e,'read'):result['response']=e.read().decode()
    with (ROOT/'results'/('smoke.jsonl' if args.smoke else f'trials-{profile}.jsonl')).open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if 'error' in result:raise RuntimeError(result['error'])
    return result
if args.smoke:call('U01','first_image');raise SystemExit(0)
randomizer=random.Random(42);all_results=[]
for round_id in range(1,4):
    order=[x[0] for x in CASES];randomizer.shuffle(order)
    for case in order:
        all_results.append(call(case,round_id));time.sleep(.5)
(ROOT/'results'/f'completed-{profile}.json').write_text(json.dumps({'trials':len(all_results),'completed_at':time.time()}))
