#!/usr/bin/env python3
"""Read existing numeric telemetry only. Never submit inference or change a service."""
import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
p.add_argument('--seconds',type=float,default=90);p.add_argument('--interval',type=float,default=2)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if any(a.out.iterdir()):p.error('output must be new or empty')
c=json.loads(a.config.read_text());headers={'Authorization':'Bearer '+c['api_key']}
def get(path):
 with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8080'+path,headers=headers),timeout=10) as f:return json.load(f)
slotkeys=('slot','group','state','request_id','prompt_tokens','prompt_read','generated','elapsed_s','tok_s')
requestkeys=('id','time','duration_s','finish','prompt_tokens','prompt_total','prompt_read','reused','output_tokens','first_engine_token_s','prompt_ms','decode_ms','decode_tok_s','drafts_offered','drafts_accepted')
start=time.monotonic();rows=[];completed={}
with (a.out/'samples.jsonl').open('x') as f:
 while True:
  t=get('/v1/telemetry');m=get('/metrics');w=t.get('work',{});l=m.get('live',{})
  row={'epoch':time.time(),'active':w.get('active'),'queued':w.get('queued'),'waiting':w.get('waiting'),
       'latest_request_phase':l.get('phase'),'output_counter':t.get('rates',{}).get('output_tokens_server_counter'),
       'slots':[{k:x.get(k) for k in slotkeys} for x in w.get('slots',[]) if x.get('state')!='idle']}
  for x in m.get('requests',[]):completed[x['id']]={k:x.get(k) for k in requestkeys}
  rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
  if time.monotonic()-start>=a.seconds:break
  time.sleep(min(a.interval,max(0,a.seconds-(time.monotonic()-start))))
(a.out/'completed-numeric.json').write_text(json.dumps(list(completed.values()),indent=2)+'\n')
record={'sample_count':len(rows),'first_epoch':rows[0]['epoch'],'last_epoch':rows[-1]['epoch'],
        'max_active':max(x['active'] or 0 for x in rows),'inference_requests_sent':0,'service_changes':0,
        'contents_recorded':False,'limits':'Coarse telemetry only. Slot counters are not per-token timestamps; no TBT/ITL percentile claim.'}
(a.out/'observation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
