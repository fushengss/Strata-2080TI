#!/usr/bin/env python3
"""Independent finite prefix-extension benchmark; dry-run by default.

One cold~100k seed, four different cold~10k resident requests, three cumulative
~1k prompt extensions. These are synthetic user-text extensions, not recorded
agent chats. Stateless count preflight; no one-token probes or cache clears.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
sys.dont_write_bytecode=True
import finite_mixed as fm
import fixture_prompts as fp
import stream_client as base
import stall_metrics as sm
import interleave_evidence as ie
from calibrate_fixtures import post,require,sha,validate_count

PERF=re.compile(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \([^\n]*?\), (\d+) generated in (\d+) ms[^\n]*$',re.M)
RESTORE=re.compile(r'^strata serve: conversation cache: restored (\d+) tokens \((live|checkpoint)\) in ([\d.]+) ms; parked=(\d+) bytes=(\d+)$',re.M)
CONTRACT='One cold~100k seed then C4 cold~10k peers and3 deterministic cumulative user-text extensions. Normal finite EOS allowed; exact actual work/cache parity and full visible C4 coverage required. First extension must restore parked cross-conversation state. Later extensions may use live prefix.'


def plan():
    settings=fm.plan('incremental')
    settings.update(extension_count=3,extension_words_each=740,min_cached_tokens=99000,min_fresh_tokens=800,max_fresh_tokens=1300)
    return settings


def fixtures(calibration):
    rows=calibration['fixtures'];seed=copy.deepcopy(next(r for r in rows if r['name']=='long-injected-0'))
    seed.update(name='incremental-seed',case='incremental',role='seed',lane=None)
    peers=[copy.deepcopy(r) for r in rows if r['case']=='primary' and r['role']=='resident']
    for r in peers:r.update(name='incremental-resident-'+str(r['lane']),case='incremental')
    texts={}
    for r in [seed,*peers]:
        source=fp.prompt(r['words'],r['fixture'])
        require(hashlib.sha256(source.encode()).hexdigest()==r['prompt_sha256'],'Source calibration prompt changed')
        r['source_calibrated_prompt_tokens']=r['counted_prompt_tokens'];r['source_calibrated_prompt_sha256']=r['prompt_sha256']
        r['fixture']='incremental-cache-v1-'+r['fixture']
        value=fp.prompt(r['words'],r['fixture']);texts[r['fixture']]=value
        r['prompt_sha256']=hashlib.sha256(value.encode()).hexdigest();r['counted_prompt_tokens']=r['target_tokens']=None
    text=texts[seed['fixture']]
    extra=[]
    for index in range(3):
        words=' '.join(fp.WORDS[(i*7+8+index)%len(fp.WORDS)]+('.' if i%17==16 else '') for i in range(plan()['extension_words_each']))
        text+='\n\nAdditional field notes '+str(index+1)+':\n'+words+'\nContinue the detailed numbered harbor diary immediately with new entries.'
        key=calibration['fixture_id']+'-incremental-extension-'+str(index)
        row={'name':'incremental-injected-'+str(index),'case':'incremental','role':'injected','lane':None,'fixture':key,
             'target_tokens':None,'counted_prompt_tokens':None,'output_tokens':64,'extension_index':index,
             'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'prefix_parent_sha256':hashlib.sha256(texts[seed['fixture']] .encode()).hexdigest() if index==0 else extra[-1]['prompt_sha256']}
        texts[key]=text;extra.append(row)
    return [seed,*peers,*extra],texts


def read_evidence(log,cursor):
    result=fm.evidence(log,cursor)
    stat=log.stat();require(stat.st_ino==cursor[0] and stat.st_size>=cursor[1],'Owned engine log changed')
    with log.open('rb') as f:f.seek(cursor[1]);raw=f.read()
    text=raw.decode(errors='replace')
    result['slice_sha256']=hashlib.sha256(raw).hexdigest()
    result['native_prompt_records']=[dict(zip(('prompt_tokens','reused_tokens','read_tokens','read_ms','generated_tokens','decode_ms'),map(int,m))) for m in PERF.findall(text)]
    result['restore_records']=[{'tokens':int(a),'source':b,'restore_ms':float(c),'parked':int(d),'bytes':int(e)} for a,b,c,d,e in RESTORE.findall(text)]
    result['scope']='Exclusive per-request log cursor, closed before next injection; unique prompt count binds native prompt record. Native initial generated count may be1 before batch handoff; API/native predicted_n validates final output.'
    return result


def gates(record,row,arm):
    g=fm.request_gates(record,row);usage=record.get('usage',{});t=record.get('native_timings',{});expected=row['counted_prompt_tokens']
    ev=record.get('admission_read_evidence',{});native=[r for r in ev.get('native_prompt_records',[]) if r['prompt_tokens']==expected]
    g['unique_native_prompt_record']=len(native)==1
    if row['role']!='injected':return g
    g.pop('cold_cache_explicitly_zero');g.pop('native_prompt_accounting')
    cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens');fresh=expected-cached if type(cached) is int else None
    g['cached_prefix_and_fresh_delta']=type(cached) is int and cached>=99000 and type(fresh) is int and 800<=fresh<=1300
    g['api_native_cache_agree']=t.get('cache_n')==cached and t.get('prompt_n')==fresh
    g['native_prompt_accounting']=len(native)==1 and native[0]['reused_tokens']==cached and native[0]['read_tokens']==fresh and native[0]['reused_tokens']+native[0]['read_tokens']==expected
    g['normal_native_initial_output']=len(native)==1 and 0<native[0]['generated_tokens']<=usage.get('completion_tokens',0)
    if row['extension_index']==0:
        g['first_extension_cross_conversation_restore']=any(r['tokens']>=99000 and r['tokens']<=cached for r in ev.get('restore_records',[])) if type(cached) is int else False
    if arm=='on':
        pairs=ev.get('reader_pump',{}).get('pairs',[])
        matching=[r for r in pairs if r['from']>=99000 and 790<=r['tokens']<=1300 and r['from']+r['tokens']<=expected-1]
        g['incremental_reader_bound']=bool(matching) and all(r['from']>=cached and r['active_slots']>=4 and r['read_ok']==1 and r['pump_ok']==1 for r in matching) if type(cached) is int else False
        g['incremental_in_read_progress']=bool(matching) and all(r['decode_rows_before_done']>0 for r in matching)
    return g


def run(a):
    require(a.port==8092 and a.exclusive_benchmark,'Isolated exclusive8092 required')
    require(a.config and a.calibration and a.out and a.engine_log,'Explicit config/calibration/out/log required')
    require(not a.out.exists(),'Output exists')
    cfg=json.loads(a.config.read_text());frozen=json.loads(a.calibration.read_text());arm='on' if cfg.get('env',{}).get('STRATA_PIPED_CONCURRENT_MTP')=='1' else 'off'
    require(frozen['fixture_generator_sha256']==sha(Path(fp.__file__)) and frozen['thinking_enabled'] is False,'Frozen source/thinking differs')
    specs,texts=fixtures(frozen);lookup={r['fixture']:r for r in specs};settings=plan()
    def prompt(nominal,fixture):
        row=lookup[fixture];require(nominal==row['counted_prompt_tokens'],'Unexpected prompt nominal');return texts[fixture]
    base.synthetic_prompt=prompt;a.out.mkdir(parents=True)
    client=base.Client(a.port,cfg,a.out,settings,frozen['fixture_id'])
    status=client.status();require(status['loaded'] and status['activity']['in_flight']==0 and status['concurrency']['serving']==8,'Need idle loaded8slot server')
    counts=[]
    for row in specs:
        count=validate_count(post(a.port,cfg['api_key'],'/v1/messages/count_tokens',fp.count_body(texts[row['fixture']],client.model)))
        if row['role']!='injected':require(abs(count-row['source_calibrated_prompt_tokens'])<=64,'Independent namespace count differs materially from source~10k/~100k size')
        row['counted_prompt_tokens']=row['target_tokens']=count;counts.append({'name':row['name'],'counted_tokens':count})
    previous=specs[0]['counted_prompt_tokens']
    for row in (r for r in specs if r['role']=='injected'):
        require(800<=row['counted_prompt_tokens']-previous<=1200,'Stateless extension count outside~1k contract');previous=row['counted_prompt_tokens']
    before=client.status();require(before['activity']['in_flight']==0 and before['activity']['requests']==status['activity']['requests'],'Count preflight must be inference-free')
    manifest={'format':1,'case':'incremental','plan':settings,'calibration_sha256':sha(a.calibration),'fixture_generator_sha256':sha(Path(fp.__file__)),
              'client_sha256':sha(Path(__file__)),'stream_client_sha256':sha(Path(base.__file__)),'stall_metrics_sha256':sha(Path(sm.__file__)),
              'fixtures':specs,'preflight_counts':counts,'completed':False,'output_contract':CONTRACT}
    base.save(a.out/'manifest.json',manifest)
    log=a.engine_log;cursor=(log.stat().st_ino,log.stat().st_size);threads=[];errors=[];phases={};injections=[];barrier=threading.Barrier(4);started=time.time()
    client.thread.start()
    def send(row,sync=False,planned=None):
        if sync:barrier.wait(timeout=20)
        at=(log.stat().st_ino,log.stat().st_size) if row['role']!='resident' else None
        r=client.request(row['name'],row['role'],row['lane'],row['counted_prompt_tokens'],row['output_tokens'],row['fixture'],planned)
        if at:
            deadline=time.monotonic()+3
            while True:
                r['admission_read_evidence']=read_evidence(log,at)
                if any(v['prompt_tokens']==row['counted_prompt_tokens'] for v in r['admission_read_evidence']['native_prompt_records']) or time.monotonic()>=deadline:break
                time.sleep(.02)
        r['expected_prompt_tokens']=row['counted_prompt_tokens'];r['gates']=gates(r,row,arm)
        if row['role']=='resident':r['gates'].pop('unique_native_prompt_record')
        base.save(a.out/(row['name']+'.json'),r);return r
    try:
        seed=send(specs[0]);require(all(seed['gates'].values()),'Cold seed output/count/cache/native proof failed')
        for row in (r for r in specs if r['role']=='resident'):
            t=threading.Thread(target=send,args=(row,True),daemon=True);t.start();threads.append(t)
        deadline=time.monotonic()+settings['resident_ready_timeout_seconds'];stable=None
        while True:
            now=time.monotonic()
            with client.lock:ready=sum(s['role']=='resident' and s['first'] is not None and now-s['last']<=1 for s in client.active.values())==4
            stable=(stable if stable is not None else now) if ready else None
            if stable is not None and now-stable>=settings['resident_ready_stable_seconds']:break
            require(now<deadline and all(t.is_alive() for t in threads),'Four resident readiness failed');client.wait(.05)
        begin=time.time();print('READY finite C4 incremental',flush=True);client.wait(settings['baseline_seconds']);mixed=time.time();phases['baseline']=(begin,mixed)
        for index,row in enumerate(r for r in specs if r['role']=='injected'):
            planned=mixed+index*settings['injection_interval_seconds'];client.wait(planned-time.time())
            require(all(t.is_alive() for t in threads),'Resident ended before extension')
            print('INJECT '+row['name']+' actual_tokens='+str(row['counted_prompt_tokens']),flush=True)
            r=send(row,planned=planned);injections.append(r);require(all(r['gates'].values()),'Incremental output/cache/native reader contract failed')
        recovery=time.time();phases['mixed']=(mixed,recovery);client.wait(settings['recovery_seconds']);phases['recovery']=(recovery,time.time())
        require(all(t.is_alive() for t in threads),'Resident ended before recovery')
    except Exception as error:
        errors.append(str(error) if isinstance(error,RuntimeError) else type(error).__name__);client.abort.set()
    finally:
        deadline=time.monotonic()+(10 if errors else settings['drain_timeout_seconds'])
        for t in threads:t.join(max(0,deadline-time.monotonic()))
        with client.lock:active=list(client.active.values())
        for state in active:client.cancel(state,'incremental_timeout_cancel')
        for t in threads:t.join(12)
        client.monitor_stop.set();client.thread.join(2)
    require(not any(t.is_alive() for t in threads),'Own workers did not stop')
    after=client.status();deadline=time.monotonic()+15
    while after['activity']['in_flight'] and time.monotonic()<deadline:time.sleep(.25);after=client.status()
    records=sorted(client.records,key=lambda r:r['name']);expected={r['name'] for r in specs}
    checks={'fixed_eight_request_set':len(records)==8 and {r['name'] for r in records}==expected,'all_request_gates':len(records)==8 and all(r.get('gates') and all(r['gates'].values()) for r in records),
            'three_phases_complete':set(phases)=={'baseline','mixed','recovery'},'idle_after':after['activity']['in_flight']==0,
            'exclusive_generation_count':after['activity']['requests']-before['activity']['requests']==8,'no_workload_errors':not errors}
    ev=read_evidence(log,cursor);checks['reader_source_safe']=ie.validate(arm,'incremental',log.read_bytes()[cursor[1]:].decode(errors='replace'))['all_checks_passed']
    metrics={name:{ch:sm.phase(records,*bounds,channel=ch) for ch in base.CHANNELS} for name,bounds in phases.items()};coverage=fm.visible_phase_coverage(records,phases);checks['four_visible_streams_cover_every_phase']=coverage['all_checks_passed']
    checked_seed=next((r for r in records if r['role']=='seed'),None);peers=[r for r in records if r['role']=='resident']
    checks['seed_completed_before_four_different_residents']=bool(checked_seed) and len(peers)==4 and all(checked_seed['end_epoch']<=r['start_epoch'] and r['request_sha256']!=checked_seed['request_sha256'] for r in peers)
    inj=[]
    for r in injections:
        end=r['start_epoch']+r['ttft_s'] if r['ttft_s'] is not None else r['end_epoch']
        inj.append({'name':r['name'],'actual_prompt_tokens':r['actual_prompt_tokens'],'ttft_first_output_s':r['ttft_s'],'ttft_answer_s':r['first_visible_content_s'],
                    'cached_tokens':r.get('cached_tokens'),'fresh_prompt_tokens':r.get('native_timings',{}).get('prompt_n'),'engine_timings':r.get('native_timings'),
                    'admission_read_evidence':r.get('admission_read_evidence'),'resident_admission_span':sm.phase(records,r['start_epoch'],end)})
    def rate(rows):
        elapsed=max(r['end_epoch'] for r in rows)-min(r['start_epoch'] for r in rows);tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in rows)
        return {'completion_tokens':tokens,'elapsed_s':elapsed,'tokens_per_s':tokens/elapsed,'scope':'Actual API completions / first request start to last request end; includes admission and drain, not pure decode.'}
    report={'case':'incremental','all_checks_passed':all(checks.values()),'checks':checks,'errors':errors,'records':records,'phases':metrics,'visible_phase_coverage':coverage,'injections':inj,
            'status_before':before,'status_after':after,'whole_case_api_rate':rate(records),'post_seed_finite_api_rate':rate([r for r in records if r['role']!='seed']) if len(records)>1 else None,
            'engine_log_evidence':ev,'output_contract':CONTRACT,'timing_limits':['Synthetic user-text prefix extension, not a complete agent chat.','Count preflight is stateless; actual API/native count equality is mandatory.','SSE deltas are not token timing. First extension must restore; subsequent extensions may retain live state.']}
    require(cfg['api_key'] not in json.dumps(report),'Credential in report');base.save(a.out/'results.json',report);base.save(a.out/'occupancy.json',client.occupancy)
    manifest.update(completed=True,all_checks_passed=report['all_checks_passed'],end_epoch=time.time());base.save(a.out/'manifest.json',manifest)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'checks':checks,'post_seed_finite_api_rate':report['post_seed_finite_api_rate']}),flush=True)
    return 0 if report['all_checks_passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--exclusive-benchmark',action='store_true')
    for name in ('config','calibration','out','engine-log'):p.add_argument('--'+name,type=Path)
    p.add_argument('--port',type=int,default=8092);a=p.parse_args()
    if not a.execute:print(json.dumps({'mode':'DRY_RUN_NO_NETWORK','plan':plan(),'contract':CONTRACT,'requests':8,'stateless_count_calls':8,'extra_generation_probes':0},indent=2));return 0
    return run(a)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'all_checks_passed':False,'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}),flush=True);raise SystemExit(1)
