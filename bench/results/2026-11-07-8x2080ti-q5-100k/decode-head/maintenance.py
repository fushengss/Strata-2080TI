#!/usr/bin/env python3
"""Own the authorized maintenance stop/restore; never terminates unrelated GPU work."""
import argparse,hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
BASE=Path('/srv/Strata/strata-orca-q5_k_m.57.json')
ROOT=Path('/srv/bench100k-20261107-decode-head')
def query():
 c=json.loads(BASE.read_text())
 req=urllib.request.Request('http://127.0.0.1:8080/v1/status',headers={'Authorization':'Bearer '+c['api_key']})
 with urllib.request.urlopen(req,timeout=5) as f: s=json.load(f)
 return {k:s.get(k) for k in ('loaded','engine','concurrency','activity')}
def table():
 d={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit(): continue
  try:
   cmd=(p/'cmdline').read_bytes().split(b'\0')
   stat=(p/'stat').read_text().split(') ',1)[1].split()
   d[int(p.name)]={'cmd':cmd,'parent':int(stat[1]),'start':stat[19]}
  except (OSError,IndexError): pass
 return d
def stop():
 procs=table()
 roots=[pid for pid,p in procs.items() if b'/srv/Strata-0157/serve/server.py' in p['cmd'] and b'8080' in p['cmd']]
 if len(roots)!=1: raise RuntimeError('Expected one live production server, found '+str(roots))
 owned=set(roots)
 while True:
  new=owned|{pid for pid,p in procs.items() if p['parent'] in owned}
  if new==owned: break
  owned=new
 before={'config_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'status':query(),'server_pids':roots,'owned_pids':sorted(owned),'time_epoch':time.time()}
 (ROOT/'maintenance-before.json').write_text(json.dumps(before,indent=2))
 for pid in roots:
  try: os.kill(pid,signal.SIGTERM)
  except ProcessLookupError: pass
 for attempt in range(60):
  live=table()
  remain=[p for p in owned if p in live and live[p]['start']==procs[p]['start'] and live[p]['cmd']!=[b'']]
  if not remain: break
  if attempt in (5,30):
   for pid in remain:
    try: os.kill(pid,signal.SIGTERM if attempt==5 else signal.SIGKILL)
    except ProcessLookupError: pass
  time.sleep(1)
 for _ in range(30):
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if not gpu: break
  time.sleep(1)
 if gpu: raise RuntimeError('GPU processes remain: '+gpu)
 print(json.dumps({'stopped':roots,'gpus_empty':True,'config_sha256':before['config_sha256']}),flush=True)
def restore():
 with open(ROOT/'restore.log','a') as f:
  r=subprocess.run(['bash','/srv/Strata/restart57.sh'],stdout=f,stderr=subprocess.STDOUT,timeout=2600)
 if r.returncode: raise RuntimeError('restart57.sh failed: '+str(r.returncode))
 for _ in range(600):
  try:
   s=query()
   if s['loaded']:
    record={'status':s,'time_epoch':time.time(),'config_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest()}
    (ROOT/'maintenance-restored.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True);return
  except Exception: pass
  time.sleep(2)
 raise TimeoutError('Production did not load')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stop','restore','status']);a=ap.parse_args()
 if a.action=='stop': stop()
 elif a.action=='restore': restore()
 else: print(json.dumps(query()),flush=True)
