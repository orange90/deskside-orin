import json,sqlite3,time,urllib.request,statistics
from pathlib import Path
root=Path('/home/orange/Documents/orin-desk-agent');stage=root/'runtime/yolo26-switch'
samples=json.loads((stage/'online-samples.json').read_text())
latency=sorted(s['vision_ms'] for s in samples if s['vision_ms'] is not None)
summary={'samples':len(samples),'ready':all(s['ready'] for s in samples),'paused':sum(bool(s['vision_paused']) for s in samples),'fps_average':(samples[-1]['frame_count']-samples[0]['frame_count'])/(samples[-1]['captured_at']-samples[0]['captured_at']),'model_ms_median':statistics.median(latency),'model_ms_p95':latency[int(.95*(len(latency)-1))],'all_yolo':all(s['vision_model']['name']=='YOLO26m-Pose' for s in samples)}
backup=Path(json.loads((stage/'deployed.json').read_text())['backup'])
with sqlite3.connect(backup/'data.sqlite3') as old,sqlite3.connect(root/'data.sqlite3') as new:
    tables=[r[0] for r in old.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
    retained={}
    for table in tables:
        cols=[r[1] for r in old.execute(f'pragma table_info("{table}")')]
        before=old.execute(f'select * from "{table}"').fetchall()
        after=new.execute(f'select * from "{table}"').fetchall()
        if table=='intervals':
            print('INTERVAL_COLUMNS',cols,flush=True)
            # The last active row can extend its end while the app is running.
            immutable=[i for i,k in enumerate(cols) if k not in ('end','ended_at','end_ts')]
            after_by={tuple(row[i] for i in immutable):row for row in after}
            for row in before:
                key=tuple(row[i] for i in immutable)
                assert key in after_by,(table,key)
                curr=after_by[key]
                for i in set(range(len(cols)))-set(immutable):assert curr[i]>=row[i]
        else:
            assert set(before)<=set(after),table+' changed unexpectedly'
        retained[table]=len(before)
summary['retained_rows']=retained
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
(stage/'online-summary.json').write_text(json.dumps(summary,indent=2))
