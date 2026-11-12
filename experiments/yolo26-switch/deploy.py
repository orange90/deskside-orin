import hashlib,json,shutil,sqlite3,subprocess,time,urllib.request
from pathlib import Path
root=Path('/home/orange/Documents/orin-desk-agent');stage=root/'runtime/yolo26-switch'
assert (stage/'validated').read_text().strip()=='PASS'
def state():
    with urllib.request.urlopen('http://127.0.0.1:8765/api/state',timeout=5) as r:return json.load(r)
assert not state()['recording'],'Microphone active; leave deployment pending'
expected={'vision.py':'99228616e6126b89f750f786f96613fe98ba6a4d64921ad493e5362406d2da3b','trt_model.py':'119396ea329f92b2de877ebd1c95ea58ca5589bd3d14aa361cd3558dadfbd24b','main.py':'f03af7bf3f378905f91887699d7c394f05557ad546135b7e2c4bd3b946596fb8','model_registry.py':'c6b1abaacb5243b028b2b3a83d9ecf3a5b6631cbdb9e9495a8686c46a808f984','dashboard.html':'c528a222a0d4fa243de739db46137f710aa55671d4cf210df42fef40913d7353','launch-ready.sh':'c31c8b1da20e460f7723166f3a1c6653df11c4dadb3c3e1c5360616bc8b009d1'}
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name+' changed since review'
backup=root/'backups'/time.strftime('yolo26m-%Y%m%d-%H%M%S');backup.mkdir(parents=True)
files=['yolo_pose.py','trt_model.py','vision.py','main.py','model_registry.py','dashboard.html','launch-ready.sh','rebuild-vision.sh','build-yolo-engine.py','fetch-models.sh','MODEL-SHA256SUMS']
existing=[]
for name in files+['calibration.json','model-selection.json']:
    if (root/name).exists():shutil.copy2(root/name,backup/name);existing.append(name)
with sqlite3.connect(root/'data.sqlite3') as src,sqlite3.connect(backup/'data.sqlite3') as dst:src.backup(dst)
print('BACKUP',backup,flush=True)
engine=root/'runtime/yolo26-eval/models/yolo26m-pose.engine'
assert hashlib.sha256(engine.read_bytes()).hexdigest()=='074860e9a8e7d94c857e6b803832229caf9d518739127bcf1097e473b32a57b5'
for suffix in ['pt','onnx','engine']:
    name='yolo26m-pose.'+suffix
    shutil.copy2(engine.parent/name,root/'models'/name)
# Prevent cron from starting a half-updated service; other service windows stay running.
paused=root/'.autostart-paused';was_paused=paused.exists();paused.touch()
def stop_vision():
    subprocess.run(['tmux','send-keys','-t','=orin-agent:=vision','C-c'],check=False)
    for _ in range(100):
        names=subprocess.check_output(['tmux','list-windows','-t','=orin-agent','-F','#{window_name}'],text=True).splitlines()
        if 'vision' not in names:return
        time.sleep(.2)
    raise RuntimeError('vision failed to stop')
try:
    stop_vision()
    for name in files:
        dest=root/name;shutil.copy2(stage/name,dest.with_name(name+'.new'));dest.with_name(name+'.new').replace(dest)
    subprocess.run(['bash','stack.sh','start'],cwd=root,check=True)
    for _ in range(90):
        try:
            d=state();c=d['camera']
            if c.get('ready') and c.get('vision_model',{}).get('name')=='YOLO26m-Pose':break
        except Exception:pass
        time.sleep(1)
    else:raise RuntimeError('New vision service did not become ready')
    print('ONLINE',json.dumps({k:c.get(k) for k in ['ready','vision_model','vision_ms','fps','vision_paused','person','presence']},ensure_ascii=False),flush=True)
    (stage/'deployed.json').write_text(json.dumps({'backup':str(backup),'time':time.time(),'model':c['vision_model']},indent=2))
except Exception:
    stop_vision()
    for name in files:
        if name in existing:shutil.copy2(backup/name,root/name)
        else:(root/name).unlink(missing_ok=True)
    subprocess.run(['bash','stack.sh','start'],cwd=root,check=True)
    print('ROLLED BACK code only; database untouched',flush=True)
    raise
finally:
    if was_paused:paused.touch()
    else:paused.unlink(missing_ok=True)
