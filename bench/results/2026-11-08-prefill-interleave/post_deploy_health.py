#!/usr/bin/env python3
"""Bounded authenticated production health/routing check; default dry-run.

No service/process management, config writes, cache clear or benchmark rate claim.
Only its own two HTTP connections may be closed on timeout/error.
"""
from __future__ import annotations
import argparse, hashlib, http.client, importlib, json, math, os, re, secrets, signal, socket, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SERVER=Path('/srv/Strata-0157/serve/server.py')
RELEASE=Path('/srv/Strata-0157/engine/releases/concurrent-prefill-20261108')
BACKUP=Path('/srv/bench100k-20261108-prefill-interleave/baseline-private/pre-concurrent-prefill.json')
ENGINE_SHA='45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6'
LIB_SHA='fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
SERVER_SHA='40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
BASE_SHA='f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd'
FREEZE_SHA='75a15bb36c851d994b70eb594f49e836ef26e0dcfab06185af22e7119133fbee'
FIXED={'STRATA_DECODE_HEAD_GPU':'2,4','STRATA_MTP_HEAD_GPU':'3,5','STRATA_PW_WITH_BATCH':'1',
 'STRATA_BATCH_MTP':'1','STRATA_PIPED_MTP':'1','STRATA_BATCH_SELF_COMMIT':'0','STRATA_PIPED_MTP_HEAD':'1',
 'STRATA_PIPED_TARGET_HEAD':'1','STRATA_PIPED_ASYNC_SAMPLE':'0','STRATA_PIPED_CONCURRENT_MTP':'1'}
FORBIDDEN=('STRATA_PIPED_CONCURRENT','STRATA_PIPED_READ_PIECE','STRATA_PIPED_READ_DECODE_ONLY',
 'STRATA_PIPED_CONCURRENT_MIN','STRATA_PIPED_MTP_OBSERVE','STRATA_MTP_PROFILE','STRATA_VERIFY_PROFILE','STRATA_PARK_OVERWRITE')
PLAN={'occupancy_sample_seconds':.25,'scenario_total_timeout_seconds':150,'request_total_timeout_seconds':120,
 'request_ttft_timeout_seconds':90,'request_silence_timeout_seconds':45,'temperature':.8,'top_p':.95,'top_k':20,
 'seed':20261108,'thinking_enabled':False,'sampling_profile':'sampled','prompt_count_tolerance_fraction':.3}

def require(ok,why):
 if not ok: raise RuntimeError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
 fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
 with os.fdopen(fd,'w') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def stat_id(path):
 s=path.stat();return {'device':s.st_dev,'inode':s.st_ino,'size':s.st_size}
def log_slice(path,cursor,limit=8<<20):
 st=stat_id(path)
 require((st['device'],st['inode'])==(cursor['device'],cursor['inode']) and st['size']>=cursor['size'],'log_rotated_or_truncated')
 require(st['size']-cursor['size']<=limit,'log_slice_limit')
 with path.open('rb') as f:f.seek(cursor['size']);raw=f.read(st['size']-cursor['size'])
 # An external request can currently be writing its final line. Never parse a fragment.
 last=raw.rfind(b'\n')+1
 return raw[:last].decode(errors='replace'),{**st,'from_byte':cursor['size'],'complete_through_byte':cursor['size']+last,
  'complete_bytes_sha256':hashlib.sha256(raw[:last]).hexdigest(),'incomplete_trailing_bytes':len(raw)-last}
def modules(harness,freeze):
 require(sha(freeze)==FREEZE_SHA,'unapproved_harness_freeze')
 manifest=json.loads(freeze.read_bytes());files=manifest['files_sha256']
 require(manifest['package']=='incremental-harness-v2','wrong_harness_package')
 require({p.name for p in harness.iterdir()}==set(files),'harness_files_differ')
 for name,digest in files.items():
  p=harness/name;require(p.is_file() and not p.is_symlink() and sha(p)==digest,'harness_hash_differs')
 sys.path.insert(0,str(harness.resolve()))
 return importlib.import_module('stream_client'),importlib.import_module('interleave_evidence'),importlib.import_module('runner_interleave'),files

def config_gate(cfg,original,receipt,current_sha):
 require(receipt.get('engine_sha256')==ENGINE_SHA and receipt.get('library_sha256')==LIB_SHA and
         receipt.get('server_sha256')==SERVER_SHA and receipt.get('original_config_sha256')==BASE_SHA and
         receipt.get('new_config_sha256')==current_sha and receipt.get('only_exe_and_overlap_flag_changed') is True and
         receipt.get('release')==str(RELEASE) and receipt.get('private_rollback_config')==str(BACKUP),'deployment_receipt_identity')
 require(cfg.get('exe')==str(RELEASE/'strata') and type(cfg.get('api_key')) is str and bool(cfg['api_key']),'production_config_identity')
 require(all(cfg.get('env',{}).get(k)==v for k,v in FIXED.items()),'configured_fixed_flags')
 normalized=json.loads(json.dumps(cfg));normalized['exe']=original['exe']
 if 'STRATA_PIPED_CONCURRENT_MTP' in original.get('env',{}):normalized['env']['STRATA_PIPED_CONCURRENT_MTP']=original['env']['STRATA_PIPED_CONCURRENT_MTP']
 else:normalized['env'].pop('STRATA_PIPED_CONCURRENT_MTP')
 require(normalized==original,'unrelated_production_config_changed')

def listener_owned(pid):
 inodes=set()
 for name in ('tcp','tcp6'):
  for line in Path('/proc/net/'+name).read_text().splitlines()[1:]:
   row=line.split()
   if row[3]=='0A' and int(row[1].rsplit(':',1)[1],16)==8080:inodes.add(row[9])
 for p in (Path('/proc')/str(pid)/'fd').iterdir():
  try:target=os.readlink(p)
  except OSError:continue
  if target.startswith('socket:[') and target[8:-1] in inodes:return True
 return False

def process_evidence(pid,runner,cursor):
 info=runner.process_info(pid);require(info is not None,'server_missing')
 args=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
 require(str(SERVER).encode() in args and b'--config' in args and args[args.index(b'--config')+1]==str(BASE).encode(),'server_command_identity')
 require(listener_owned(pid),'server_does_not_own_8080')
 boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
 births=boot+int(info['start_ticks'])/os.sysconf('SC_CLK_TCK')
 require(births>=cursor['created_epoch']-2,'startup_cursor_does_not_precede_new_server')
 engines=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if (proc/'exe').resolve(strict=True)!=RELEASE/'strata':continue
   ep=int(proc.name)
   if not runner.is_descendant(ep,pid):continue
   ei=runner.process_info(ep);require(ei is not None,'engine_missing')
   env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x)
   require(all(env.get(k)==v for k,v in FIXED.items()),'actual_fixed_flags')
   require(all(env.get(k) in (None,'','0') for k in FORBIDDEN),'actual_guard_or_profiler_enabled')
   maps={line.split(None,5)[5].strip() for line in (proc/'maps').read_text().splitlines() if len(line.split(None,5))==6}
   libs={p for p in maps if Path(p).name=='libstrata_kq.so'}
   require(libs=={str(RELEASE/'libstrata_kq.so')},'actual_library_mapping')
   engines.append({**ei,'engine_sha256':sha(proc/'exe'),'library_sha256':sha(RELEASE/'libstrata_kq.so'),
    'flags':{k:env.get(k) for k in (*FIXED,*FORBIDDEN)}})
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 require(len(engines)==1 and engines[0]['engine_sha256']==ENGINE_SHA and engines[0]['library_sha256']==LIB_SHA,'unique_engine_identity')
 return {'server':info,'engine':engines[0]}

def status_good(s):
 return s.get('loaded') is True and all(s.get('concurrency',{}).get(k)==8 for k in ('serving','requested'))
def same_birth(runner,identity):
 now=runner.process_info(identity['pid'])
 return bool(now and now['start_ticks']==identity['start_ticks'])
def hash_matches(path,expected):
 try:return sha(path)==expected
 except OSError:return False
def output_gates(r):
 u=r.get('usage',{});n=r.get('native_timings',{});count=u.get('completion_tokens');cap=r.get('max_tokens')
 usage=all(type(u.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens')) and u.get('prompt_tokens',0)>0 and u.get('total_tokens')==u.get('prompt_tokens',0)+u.get('completion_tokens',0)
 normal=type(count) is int and type(cap) is int and 0<count<=cap and (r.get('finish_reason')=='stop' or (r.get('finish_reason')=='length' and count==cap))
 native=all(type(n.get(k)) is int for k in ('cache_n','prompt_n','predicted_n')) and n.get('cache_n',-1)>=0 and n.get('prompt_n',-1)>=0 and n.get('cache_n',0)+n.get('prompt_n',0)==u.get('prompt_tokens') and n.get('predicted_n')==count
 chunks=r.get('chunks',[])
 return {'complete_sse':r.get('error') is None and r.get('saw_done') is True and r.get('http_status')==200,
  'normal_finish':normal,'usage_consistent':usage,'native_usage_consistent':native,
  'visible_answer':any(c['bytes'].get('visible_content',0)>0 for c in chunks),
  'answer_only':all(c['bytes'].get('reasoning',0)==0 and c['bytes'].get('tool',0)==0 and 'tool' not in c['channels'] for c in chunks)}
def route_gate(parsed,prompt_n):
 begins=[b for b in parsed['begin'] if b['tokens']>=.9*prompt_n and b['from']+b['tokens']<=prompt_n-1]
 rows=[]
 for b in begins:
  es=[e for e in parsed['end'] if e['read_id']==b['read_id']];ss=[s for s in parsed['source'] if s['read_id']==b['read_id']]
  if len(es)==len(ss)==1:rows.append({**b,**es[0],**ss[0]})
 ambiguous_fallback=any(f['tokens']>=.9*prompt_n and f['from']+f['tokens']<=prompt_n-1 for f in parsed['fallback'])
 checks={'unique_compatible_large_read':len(begins)==len(rows)==1 and not ambiguous_fallback,
  'no_cuda_error_in_fresh_slice':not parsed['cuda_error_present'],
  'markers_parsed':parsed['all_marker_lines_parsed'] and parsed['all_source_lines_parsed'],
  'private_resident_reader':bool(rows) and all(x['active_slots']>=1 and x['stages']==x['own_buffers']==x['resident_stages']==8 for x in rows),
  'reader_and_pump_complete':bool(rows) and all(x['read_ok']==x['pump_ok']==1 for x in rows),
  'strict_in_read_progress':bool(rows) and all(x['decode_rows_before_done']>0 for x in rows),
  'no_expert_dispatch':bool(rows) and all(x['dispatch_delta']==0 and x['lookahead_idle']==1 for x in rows)}
 return {'checks':checks,'all_checks_passed':all(checks.values()),'compatible_reads':rows,
  'other_begin_count':len(parsed['begin'])-len(begins),'other_fallback_count':len(parsed['fallback']),
  'binding':'Unique large reader shape compatible with owned cold prompt inside fresh injection cursor. Native read_id has no HTTP request ID; external traffic limits exact ownership attribution.'}

def count_tokens(client,text):
 body={'model':client.model,'messages':[{'role':'user','content':text}],'thinking':{'type':'disabled'},'max_tokens':1}
 conn=http.client.HTTPConnection('127.0.0.1',8080,timeout=15)
 try:
  conn.request('POST','/v1/messages/count_tokens',json.dumps(body),client.headers);response=conn.getresponse()
  require(response.status==200,'count_http_failure');data=json.loads(response.read(1<<20));n=data.get('input_tokens')
  require(type(n) is int and n>0 and not data.get('error'),'count_response_invalid');return n
 finally:conn.close()

class TextView:
 def __init__(self,text):self.text=text
 def read_text(self,**kwargs):return self.text

def close_own(client,workers,seconds=20):
 client.abort.set();client.monitor_stop.set()
 # A worker may pass its abort check just before registering or connecting.
 # Re-snapshot only this client's states until workers finish or the bound ends.
 cancelled=set();socket_cancelled=set();deadline=time.monotonic()+seconds
 while any(t.is_alive() for t in workers) and time.monotonic()<deadline:
  with client.lock:states=list(client.active.values())
  for state in states:
   cancelled.add(id(state))
   if state.get('socket') is not None:socket_cancelled.add(id(state))
   client.cancel(state,'health_cleanup_own_connection')
  for t in workers:t.join(min(.1,max(0,deadline-time.monotonic())))
 if client.thread.ident is not None:client.thread.join(2)
 with client.lock:active=len(client.active)
 return {'own_workers_finished':all(not t.is_alive() for t in workers),'own_active_connections':active,
  'own_cancelled_request_states':len(cancelled),'own_socket_shutdowns_attempted':len(socket_cancelled),
  'monitor_finished':not client.thread.is_alive()}

def runtime(a):
 require(sys.platform.startswith('linux'),'linux_only')
 require(a.out and a.harness and a.freeze and a.cursor and a.deployment_receipt and a.server_pid,'explicit_runtime_inputs_required')
 require(not a.out.exists(),'new_output_directory_required')
 cfgraw=BASE.read_bytes();cfg=json.loads(cfgraw);key=cfg.get('api_key');before=hashlib.sha256(cfgraw).hexdigest()
 require(sha(SERVER)==SERVER_SHA and sha(BACKUP)==BASE_SHA,'server_or_backup_identity')
 receipt=json.loads(a.deployment_receipt.read_bytes());config_gate(cfg,json.loads(BACKUP.read_bytes()),receipt,before)
 clientmod,evidence,runner,frozen=modules(a.harness,a.freeze)
 cursor=json.loads(a.cursor.read_bytes());log=Path(cfg['log'])
 require(cursor.get('version')=='post-deploy-startup-cursor-1' and cursor.get('log_path')==str(log) and
         cursor.get('config_sha256')==before,'startup_cursor_identity')
 identities=process_evidence(a.server_pid,runner,cursor)
 startup_text,startup_slice=log_slice(log,cursor['log'])
 facts=runner.startup_facts(TextView(startup_text))
 require(facts['batch_slot_devices']==list(range(8)) and facts['all_experts_resident'] and facts['all_eight_have_3_80_gib_own_buffers'] and facts['piped_mtp_startup_enabled'] and facts['shared_head_workspaces']==[[8,3,5,7]] and facts['target_head_startup']==[[2,4,7]] and facts['async_sample_startup']==[] and startup_text.count(evidence.STARTUP)==1,'fresh_startup_facts')
 del startup_text
 a.out.mkdir(parents=True,mode=0o700);a.out.chmod(0o700)
 report={'version':'post-deploy-health-1','scope':'Own bounded synthetic completion and fresh routing health only; no throughput claim or global-idle requirement.',
  'created_epoch':time.time(),'config_sha256':before,'deployment_receipt_sha256':sha(a.deployment_receipt),'startup_cursor_sha256':sha(a.cursor),
  'process':identities,'startup_facts':facts,'startup_slice':startup_slice,'frozen_harness_manifest_sha256':FREEZE_SHA,'service_mutations':0,'process_signals':0}
 client=clientmod.Client(8080,cfg,a.out,dict(PLAN),'post-deploy-health');workers=[];failures=[];observations=[];first_visible=threading.Event()
 original_extract=clientmod.extract_delta;old_handlers={};injection_cursor=None
 def observe(data):
  answer=original_extract(data)
  if threading.current_thread().name=='health-peer' and answer[1]['visible_content']>0:first_visible.set()
  return answer
 def run(name,role,nominal,cap,fixture):
  try:
   if client.abort.is_set():return
   client.request(name,role,0 if role=='resident' else -1,nominal,cap,fixture)
  except BaseException as error:failures.append(type(error).__name__);client.abort.set()
 def sample():
  status=client.status()
  with client.lock:own=len(client.active)
  observed=status.get('activity',{}).get('in_flight')
  observations.append({'epoch':time.time(),'own_active':own,'status':status,'external_in_flight_lower_bound':max(0,observed-own) if type(observed) is int else None})
  require(status_good(status),'production_not_loaded_8slots')
 def interrupted(sig,frame):client.abort.set();raise KeyboardInterrupt()
 try:
  for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,interrupted)
  sample();nonce=secrets.token_hex(12);peer_fixture='posthealth-'+nonce+'-peer';inject_fixture='posthealth-'+nonce+'-injected'
  counted={'peer':count_tokens(client,clientmod.synthetic_prompt(512,peer_fixture)),
           'injected':count_tokens(client,clientmod.synthetic_prompt(5000,inject_fixture))}
  require(100<=counted['peer']<=1500 and 3000<=counted['injected']<=6000,'counted_fixture_outside_bounds');report['counted_prompt_tokens']=counted
  report['owned_window_cursor_before_requests']=stat_id(log)
  clientmod.extract_delta=observe;client.thread.start()
  t=threading.Thread(target=run,args=('health-peer','resident',512,1024,peer_fixture),name='health-peer',daemon=True);workers.append(t);t.start()
  deadline=time.monotonic()+90
  while not first_visible.wait(.25):
   require(t.is_alive() and not client.abort.is_set() and time.monotonic()<deadline,'peer_first_visible_timeout')
  with client.lock:require('health-peer' in client.active,'peer_finished_before_injection')
  injection_cursor=stat_id(log);report['injection_cursor']=injection_cursor;report['injection_start_epoch']=time.time()
  t=threading.Thread(target=run,args=('health-injected','injected',5000,64,inject_fixture),name='health-injected',daemon=True);workers.append(t);t.start()
  deadline=time.monotonic()+120
  while any(t.is_alive() for t in workers):
   require(not client.abort.is_set() and not failures and time.monotonic()<deadline,'owned_request_timeout_or_error')
   sample();time.sleep(.5)
  require(not failures,'client_worker_failure');sample()
  records={x['name']:x for x in client.records};require(set(records)=={'health-peer','health-injected'},'owned_request_set')
  report['owned_checks']={name:output_gates(r) for name,r in records.items()}
  require(all(all(c.values()) for c in report['owned_checks'].values()),'owned_completion_checks')
  for name,label in (('health-peer','peer'),('health-injected','injected')):
   require(records[name]['usage']['prompt_tokens']==counted[label],'count_api_usage_mismatch')
  injected=records['health-injected'];require(injected.get('cached_tokens')==0,'injection_not_cold')
  peer=records['health-peer'];visible=[c['epoch'] for c in peer['chunks'] if c['bytes'].get('visible_content',0)>0]
  first_injected=injected['start_epoch']+injected['first_visible_content_s']
  require(visible and visible[0]<=injected['start_epoch'] and visible[-1]>=first_injected,'peer_visible_overlap_absent')
  report['own_visible_overlap']={'peer_first_epoch':visible[0],'peer_last_epoch':visible[-1],
   'injected_first_visible_epoch':first_injected,'peer_covers_injected_first_visible':True,
   'peer_covers_injected_http_end':visible[-1]>=injected['end_epoch']}
  text,sliceinfo=log_slice(log,injection_cursor);route=route_gate(evidence.parse(text),counted['injected']);del text
  report['routing']=route;report['routing_slice']=sliceinfo;require(route['all_checks_passed'],'fresh_routing_not_proven')
  report['checks_passed']=True
 except BaseException as error:
  report['failure_kind']=type(error).__name__;report['failure_code']=str(error) if type(error) is RuntimeError and re.fullmatch('[a-z0-9_]+',str(error)) else None
 finally:
  for sig in old_handlers:signal.signal(sig,signal.SIG_IGN)
  report['cleanup']=close_own(client,workers)
  clientmod.extract_delta=original_extract
  report['status_samples']=observations;report['external_concurrency_observed']=any((s['external_in_flight_lower_bound'] or 0)>0 for s in observations)
  report['external_concurrency_note']='Sampled lower bound; races/unsampled arrivals can hide external traffic. Native read IDs are not HTTP IDs.'
  report['config_unchanged']=hash_matches(BASE,before);report['server_unchanged']=hash_matches(SERVER,SERVER_SHA)
  report['same_processes_alive']=all(same_birth(runner,x) for x in (identities['server'],identities['engine']))
  report['frozen_files_unchanged']=all(hash_matches(a.harness/name,digest) for name,digest in frozen.items())
  report['owned_requests']=[{k:r.get(k) for k in ('name','request_sha256','output_sha256','channel_output_sha256',
    'usage','finish_reason','cached_tokens','max_tokens','http_status','saw_done','error','observed_output_chunks',
    'output_bytes','start_epoch','end_epoch','first_visible_content_s')} for r in client.records]
  report['ended_epoch']=time.time()
  matches=0
  for p in a.out.iterdir():
   if p.is_file() and key.encode() in p.read_bytes():matches+=1
  report['credential_scan_matches']=matches
  report['passed']=bool(report.get('checks_passed') and not report.get('failure_kind') and all(report['cleanup'][k] for k in ('own_workers_finished','monitor_finished')) and report['cleanup']['own_active_connections']==0 and report['config_unchanged'] and report['server_unchanged'] and report['same_processes_alive'] and report['frozen_files_unchanged'] and matches==0)
  require(key not in json.dumps(report),'receipt_credential_guard')
  save(a.out/'receipt.json',report)
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
 print(json.dumps({'passed':report['passed'],'receipt':str(a.out/'receipt.json'),'external_concurrency_observed':report['external_concurrency_observed'],'no_performance_claim':True,'service_mutations':0}))
 return 0 if report['passed'] else 2

def cursor_only(a):
 require(a.out and not a.out.exists(),'new_cursor_directory_required')
 cfg=json.loads(BASE.read_bytes());log=Path(cfg['log']);require(log.is_file(),'existing_append_log_required')
 a.out.mkdir(parents=True,mode=0o700);a.out.chmod(0o700)
 save(a.out/'startup-cursor.json',{'version':'post-deploy-startup-cursor-1','created_epoch':time.time(),'log_path':str(log),'log':stat_id(log),'config_sha256':sha(BASE),'service_mutations':0})
 print(json.dumps({'cursor':str(a.out/'startup-cursor.json'),'service_mutations':0}))
 return 0

def self_test():
 import tempfile
 good={'usage':{'prompt_tokens':4000,'completion_tokens':64,'total_tokens':4064},'native_timings':{'cache_n':0,'prompt_n':4000,'predicted_n':64},'max_tokens':64,'finish_reason':'length','error':None,'saw_done':True,'http_status':200,'chunks':[{'channels':['output','visible_content'],'bytes':{'visible_content':100,'reasoning':0,'tool':0}}]}
 require(all(output_gates(good).values()),'selftest_good')
 eos=json.loads(json.dumps(good));eos['finish_reason']='stop';eos['usage'].update(completion_tokens=63,total_tokens=4063);eos['native_timings']['predicted_n']=63
 require(all(output_gates(eos).values()),'selftest_eos')
 bads=[]
 for key,val in [('saw_done',False),('finish_reason','cancel'),('error',{'kind':'failure'})]:
  r=json.loads(json.dumps(good));r[key]=val;bads.append(r)
 r=json.loads(json.dumps(eos));r['finish_reason']='length';bads.append(r)
 r=json.loads(json.dumps(good));r['native_timings']['predicted_n']=63;bads.append(r)
 r=json.loads(json.dumps(good));r['chunks'][0]['channels'].append('tool');bads.append(r)
 require(all(not all(output_gates(b).values()) for b in bads),'selftest_negative')
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'engine.log';p.write_bytes(b'old\n');cur=stat_id(p);p.write_bytes(b'old\nnew\npartial')
  text,s=log_slice(p,cur);require(text=='new\n' and s['incomplete_trailing_bytes']==7,'selftest_slice')
 # Route ownership is intentionally limited to one compatible shape in a fresh window.
 parsed={'begin':[{'read_id':7,'from':0,'tokens':3990,'chunk':8192,'active_slots':1,'stages':8,'own_buffers':8,'resident_stages':8}],
  'end':[{'read_id':7,'read_ms':750.0,'decode_rows_before_done':2,'read_ok':1,'pump_ok':1,'stopped_at':3990,'admit_stop_requested':0,'yield_slot':-1}],
  'source':[{'read_id':7,'staged':1,'lookahead_idle':1,'dispatch_delta':0}], 'fallback':[],
  'cuda_error_present':False,'all_marker_lines_parsed':True,'all_source_lines_parsed':True}
 require(route_gate(parsed,4000)['all_checks_passed'],'selftest_route')
 rejected=0
 for group,key,val in [('begin','active_slots',0),('begin','own_buffers',7),('end','decode_rows_before_done',0),
  ('end','read_ok',0),('end','pump_ok',0),('source','dispatch_delta',1),('source','lookahead_idle',0)]:
  bad=json.loads(json.dumps(parsed));bad[group][0][key]=val
  require(not route_gate(bad,4000)['all_checks_passed'],'selftest_bad_route');rejected+=1
 for group in ('end','source'):
  bad=json.loads(json.dumps(parsed));bad[group]=[]
  require(not route_gate(bad,4000)['all_checks_passed'],'selftest_missing_route');rejected+=1
 bad=json.loads(json.dumps(parsed));bad['begin'].append(dict(bad['begin'][0]))
 require(not route_gate(bad,4000)['all_checks_passed'],'selftest_ambiguous_route');rejected+=1
 bad=json.loads(json.dumps(parsed));bad['fallback']=[{'from':0,'tokens':3990,'reason':'no_active_slots'}]
 require(not route_gate(bad,4000)['all_checks_passed'],'selftest_matching_fallback');rejected+=1
 other=json.loads(json.dumps(parsed));other['begin'].append({**other['begin'][0],'read_id':8,'tokens':20000})
 require(route_gate(other,4000)['all_checks_passed'],'selftest_unrelated_external_unfinished')
 original={'exe':'old-engine','api_key':'SYNTHETIC_ONLY','env':{k:v for k,v in FIXED.items() if k!='STRATA_PIPED_CONCURRENT_MTP'},'layer_split':'unchanged','args':['original-prefill']}
 current=json.loads(json.dumps(original));current['exe']=str(RELEASE/'strata');current['env']['STRATA_PIPED_CONCURRENT_MTP']='1'
 receipt={'engine_sha256':ENGINE_SHA,'library_sha256':LIB_SHA,'server_sha256':SERVER_SHA,'original_config_sha256':BASE_SHA,
 'new_config_sha256':'fake','only_exe_and_overlap_flag_changed':True,'release':str(RELEASE),'private_rollback_config':str(BACKUP)}
 config_gate(current,original,receipt,'fake')
 for key,value in [('api_key','CHANGED'),('layer_split','CHANGED'),('args',[])]:
  bad=json.loads(json.dumps(current));bad[key]=value
  try:config_gate(bad,original,receipt,'fake')
  except RuntimeError:pass
  else:raise RuntimeError('selftest_unrelated_config_accepted')
 # Simulate a worker that passed the abort check before cleanup, then registers
 # and obtains its own socket after the first cleanup snapshot. No real network.
 class FakeClient:
  def __init__(self):
   self.abort=threading.Event();self.monitor_stop=threading.Event();self.lock=threading.Lock()
   self.active={};self.thread=threading.Thread();self.closed=threading.Event();self.foreign_closed=False
  def cancel(self,state,reason):
   if state.get('socket') is not None:self.closed.set()
 fake=FakeClient()
 def late_worker():
  time.sleep(.02);state={'socket':None}
  with fake.lock:fake.active['own']=state
  time.sleep(.02);state['socket']=object()
  fake.closed.wait(1)
  with fake.lock:fake.active.pop('own')
 worker=threading.Thread(target=late_worker);worker.start();clean=close_own(fake,[worker],1)
 require(clean['own_workers_finished'] and clean['own_active_connections']==0 and fake.closed.is_set() and
         clean['own_socket_shutdowns_attempted']==1 and not fake.foreign_closed,'selftest_late_connection_cleanup')
 print(json.dumps({'self_test':'PASS','negative_output_cases':len(bads),'negative_route_cases':rejected,
 'negative_config_preservation_cases':3,'late_registered_own_connection_cleanup':True,
 'external_unfinished_read_allowed_if_other_shape':True,'HTTP':0,'GPU':0,'service_mutations':0}))


def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--execute',action='store_true');g.add_argument('--cursor-only',action='store_true');g.add_argument('--self-test',action='store_true')
 p.add_argument('--out',type=Path);p.add_argument('--harness',type=Path);p.add_argument('--freeze',type=Path);p.add_argument('--cursor',type=Path);p.add_argument('--deployment-receipt',type=Path);p.add_argument('--server-pid',type=int)
 a=p.parse_args()
 if a.self_test:self_test();return 0
 if a.cursor_only:return cursor_only(a)
 if a.execute:return runtime(a)
 print(json.dumps({'mode':'DRY_RUN_NO_IO_NO_HTTP','port':8080,'generation_requests':2,'max_outputs':[1024,64],'injected_actual_prompt_range':[3000,6000],'stateless_count_requests':2,'requires':['new startup cursor before service start','successful deployment receipt','ready owned production server PID','frozen v2 harness and manifest'],'service_mutations':0,'global_idle_required':False,'no_performance_claim':True}))
 return 0
if __name__=='__main__':raise SystemExit(main())
