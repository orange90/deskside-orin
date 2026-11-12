from pathlib import Path
from collections import Counter
import json,statistics
root=Path('outputs/orin-desk-agent/evidence/57-qwen35-vision')
def percentile(values,p):
 values=sorted(values);x=(len(values)-1)*p;i=int(x);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)
def label(r):
 d=r['result']
 return d.get('state') or ({'sitting':'sit','standing':'stand'}.get(d.get('posture')) if d.get('presence')=='present' else 'away' if d.get('presence')=='absent' else None) or 'unknown'
summary={}
for profile,name in [('explained','trials.jsonl'),('compact','trials-compact.jsonl')]:
 path=root/'results'/name
 if profile=='explained' and (root/'results/trials-explained.jsonl').exists():path=root/'results/trials-explained.jsonl'
 rows=[json.loads(l) for l in path.read_text().splitlines()]
 assert len(rows)==15 and all('error' not in r and r['semantic_valid'] and r['finish_reason']=='stop' for r in rows)
 assert all(r['usage']['prompt_tokens_details']['cached_tokens']==0 for r in rows)
 values=[r['seconds'] for r in rows]
 summary[profile]={'n':len(rows),'median_s':statistics.median(values),'p95_s':percentile(values,.95),'min_s':min(values),'max_s':max(values),'by_case':{}}
 for case in ['U01','U02','C01','C02','C03']:
  subset=[r for r in rows if r['case']==case]
  summary[profile]['by_case'][case]={'labels':dict(Counter(label(r) for r in subset)),'median_s':statistics.median(r['seconds'] for r in subset),'min_s':min(r['seconds'] for r in subset),'max_s':max(r['seconds'] for r in subset),'result':subset[0]['result']}
summary['first_image']=json.loads((root/'results/smoke.jsonl').read_text().splitlines()[0])
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
