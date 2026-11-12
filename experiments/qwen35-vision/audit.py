from pathlib import Path
import hashlib,json,sys,urllib.request
root=Path('/home/orange/Documents/orin-desk-agent');lab=root/'runtime/qwen35-vision-eval'
paths=['models/qwen.gguf','models/rtmdet.plan','models/rtmpose.plan','runtime/llama-server','main.py','vision.py','state.py','store.py','dashboard.html','launch-ready.sh','model-selection.json']
state={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).exists()}
if sys.argv[1]=='before':(lab/'production-before.json').write_text(json.dumps(state,indent=2))
else:assert state==json.loads((lab/'production-before.json').read_text())
services={}
for port,route in [(8765,'api/state'),(8081,'health'),(8082,'health')]:
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/{route}',timeout=5) as r:d=json.load(r)
 services[port]={'ready':d.get('camera',{}).get('ready'),'fps':d.get('camera',{}).get('fps')} if port==8765 else d
print(json.dumps({'phase':sys.argv[1],'production_hashes':'recorded' if sys.argv[1]=='before' else 'unchanged','services':services}))
