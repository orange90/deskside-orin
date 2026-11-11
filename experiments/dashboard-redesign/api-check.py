"""Isolated HTTP regression; no camera capture, real recording or production DB writes."""
import json,tempfile,time,threading,importlib.util
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import uvicorn,store
spec=importlib.util.spec_from_file_location('main_candidate',Path.cwd()/'runtime/dashboard-redesign/main_candidate.py');main=importlib.util.module_from_spec(spec);spec.loader.exec_module(main)
def request(path,body=None,origin=None):
    headers={'Content-Type':'application/json'}
    if origin:headers['Origin']=origin
    try:
        with urlopen(Request('http://127.0.0.1:8877'+path,data=None if body is None else json.dumps(body).encode(),headers=headers),timeout=5) as response:
            return response.status,json.load(response)
    except HTTPError as error:return error.code,{}
original=store.ROOT
with tempfile.TemporaryDirectory(dir='/dev/shm') as directory:
    store.ROOT=Path(directory);store.initialize()
    server=uvicorn.Server(uvicorn.Config(main.app,host='127.0.0.1',port=8877,lifespan='off',log_level='error'))
    worker=threading.Thread(target=server.run,daemon=True);worker.start()
    try:
        for _ in range(40):
            if server.started:break
            time.sleep(.1)
        assert server.started
        assert request('/api/dashboard?date=bad')[0]==400
        main.publish(b'not-a-real-image',ready=True,vision_paused=False,privacy_active=False,overlay={'keypoints':[]},frame_count=77)
        code,data=request('/api/live/frame');assert code==200 and data['camera']['frame_count']==77
        main.publish(None,ready=True,vision_paused=True,privacy_active=True)
        code,data=request('/api/live/frame');assert code==423 and 'image' not in data
        assert request('/api/calibration/frame')[0]==503 and main.preview is None
        assert request('/api/models/select',{'component':'asr','model':'../../bad'})[0]==400
        assert request('/api/models/select',{'component':'pose','model':'x'})[0]==400
        (store.ROOT/'models').mkdir();(store.ROOT/'models/ggml-base.bin').write_bytes(b'test')
        main.voice.busy.set();assert request('/api/models/select',{'component':'asr','model':'base'})[0]==409
        main.voice.busy.clear();assert request('/api/models/select',{'component':'asr','model':'base'})[0]==200
        assert json.loads((store.ROOT/'model-selection.json').read_text())['asr']=='base'
        assert request('/api/models/select',{'component':'asr','model':'base'},'https://external.example')[0]==403
        store.note('reminder','test-only-in-temp-db');feed=request('/api/reminders')[1];assert len(feed['reminders'])==1
        assert request('/api/reminders?after='+str(feed['latest_id']))[1]['reminders']==[]
    finally:
        server.should_exit=True;worker.join(5);store.ROOT=original
print('API PASS: invalid dates, model allowlist/busy/origin, local selection persistence, same-frame preview, privacy denial, reminder cursor. Production DB untouched.')
