#!/usr/bin/env python3
"""Finite first/repeated long admission, HTTP cancel during read, and same-prompt reuse.

Default dry-run; no service or clock mutations. Independent prompts preserve the
calibrated performance fixtures' cold caches. Cancellation means closing our HTTP
socket: it does not claim the GPU reader was immediately stopped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
sys.dont_write_bytecode=True
import fixture_prompts as fp
import stream_client as base
import finite_mixed as finite
import interleave_evidence as evidence
import cancel_read_proof as crp
from calibrate_fixtures import post,validate_count,require,sha


def specs():
    rows=[{'name':'admission-peer-'+str(i),'role':'resident','lane':i,'words':500,'output_tokens':4096} for i in range(4)]
    rows += [{'name':'admission-first-cold','role':'injected','lane':None,'words':53180,'output_tokens':64},
             {'name':'admission-cancel-cold','role':'cancel','lane':None,'words':53180,'output_tokens':64}]
    return [{**r,'fixture':'prefill-interleave-admission-v1-'+r['name']} for r in rows]


def normal_gates(record,row,cold):
    usage=record.get('usage',{});timings=record.get('native_timings',{})
    return {'normal_complete':record.get('successful') and record.get('saw_done') and record.get('finish_reason')=='length',
            'exact_counted_prompt':usage.get('prompt_tokens')==row['counted_prompt_tokens'],
            'exact_output_count':usage.get('completion_tokens')==row['output_tokens'],
            'usage_consistent':usage.get('total_tokens')==row['counted_prompt_tokens']+row['output_tokens'],
            'cold_cache' :not cold or (usage.get('prompt_tokens_details') or {}).get('cached_tokens')==0,
            'answer_only':record.get('first_visible_content_s') is not None and all(c['bytes']['reasoning']==c['bytes']['tool']==0 for c in record['chunks']),
            'timings_accounted':type(timings.get('prompt_ms')) in (int,float) and timings.get('prompt_n',-1)+timings.get('cache_n',-1)==row['counted_prompt_tokens']}


def run(a):
    require(a.exclusive_benchmark and a.port==8092 and a.config and a.out and a.engine_log,'Explicit isolated config/out/log required')
    require(not a.out.exists(),'Output already exists')
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True)
    settings=finite.plan('admission');settings.update(request_total_timeout_seconds=360,scenario_total_timeout_seconds=420)
    client=base.Client(a.port,cfg,a.out,settings,'admission-v1');before=client.status()
    require(before['loaded'] and before['activity']['in_flight']==0 and before['concurrency']['serving']==8,'Server must be idle with8slots')
    rows=specs();texts={r['fixture']:fp.prompt(r['words'],r['fixture']) for r in rows}
    for row in rows:
        row['counted_prompt_tokens']=validate_count(post(a.port,cfg['api_key'],'/v1/messages/count_tokens',fp.count_body(texts[row['fixture']],client.model)))
        row['prompt_sha256']=hashlib.sha256(texts[row['fixture']].encode()).hexdigest()
    base.synthetic_prompt=lambda nominal,fixture:texts[fixture]
    base.save(a.out/'manifest.json',{'format':1,'plan':settings,'fixtures':rows,'script_sha256':sha(Path(__file__)),'completed':False})
    stat=a.engine_log.stat();cursor=(stat.st_ino,stat.st_size)
    def log_new():
        stat2=a.engine_log.stat();require(stat2.st_ino==cursor[0] and stat2.st_size>=cursor[1],'Owned log rotated')
        with a.engine_log.open('rb') as handle:handle.seek(cursor[1]);return handle.read().decode(errors='replace')
    threads=[];errors=[];cancel_observation={};cancel_log_window={};cancel_raw=b'';cancel_prefix='';barrier=threading.Barrier(4)
    def send(row,sync=False,cold=True):
        if sync:barrier.wait(timeout=20)
        record=client.request(row['name'],row['role'],row['lane'],row['counted_prompt_tokens'],row['output_tokens'],row['fixture'])
        if row['role']!='cancel':record['gates']=normal_gates(record,row,cold)
        base.save(a.out/(row['name']+'.json'),record);return record
    client.thread.start()
    try:
        for row in rows[:4]:
            thread=threading.Thread(target=send,args=(row,True),daemon=True);thread.start();threads.append(thread)
        deadline=time.monotonic()+120
        while True:
            with client.lock:ready=sum(s['first'] is not None and s['role']=='resident' for s in client.active.values())==4
            if ready:break
            require(time.monotonic()<deadline and all(t.is_alive() for t in threads),'Four peers failed readiness')
            client.wait(.05)
        first=send(rows[4]);require(all(first['gates'].values()),'First cold read failed normal output/count gate')
        require(all(t.is_alive() for t in threads),'Peer ended before repeated admission')
        cancel_row=rows[5];cancel_window_start=a.engine_log.stat().st_size
        cancel_thread=threading.Thread(target=send,args=(cancel_row,),daemon=True);cancel_thread.start();threads.append(cancel_thread)
        deadline=time.monotonic()+90;cancel_start=time.monotonic();observed=None
        while True:
            with client.lock:state=client.active.get(cancel_row['name']);has_socket=state is not None and state.get('socket') is not None;no_output=state is not None and state.get('first') is None
            if has_socket and no_output:
                if a.arm=='on':
                    with a.engine_log.open('rb') as handle:handle.seek(cancel_window_start);snapshot=handle.read()
                    snapshot_end=cancel_window_start+len(snapshot)
                    parsed=evidence.parse(snapshot.decode(errors='replace'));ended={r['read_id'] for r in parsed['end']}
                    open_reads=[r for r in parsed['begin'] if r['read_id'] not in ended and r['tokens']>=.9*cancel_row['counted_prompt_tokens'] and r['from']+r['tokens']<=cancel_row['counted_prompt_tokens']-1]
                    if open_reads:observed=open_reads[-1];break
                elif time.monotonic()-cancel_start>=1:break
            require(time.monotonic()<deadline and cancel_thread.is_alive(),'Could not cancel the long request before first output')
            client.wait(.01)
        cancel_observation={'epoch':time.time(),'before_first_output':no_output,'reader_open_at_observation':observed is not None,
                            'reader_id':observed['read_id'] if observed else None,'native_log_observation_offset':snapshot_end if observed else None,'observation_to_shutdown_not_atomic':True,
                            'meaning':'HTTP socket closed after observing open native reader; a boundary race is possible. No claim of immediate GPU read cancellation.'}
        client.cancel(state,'intentional_admission_cancel');cancel_thread.join(15);require(not cancel_thread.is_alive(),'Cancelled HTTP thread did not stop')
        deadline=time.monotonic()+120
        while client.status()['activity']['in_flight']>4:
            require(time.monotonic()<deadline,'Cancelled admission did not drain server-side');client.wait(.1)
        require(all(t.is_alive() for t in threads[:4]),'Peer ended before cancelled-slot reuse')
        # BADM can reach the server just before stderr's final cancelled PERF line.
        # Seal this request's own window only once that line is visible, before retry.
        deadline=time.monotonic()+20
        while True:
            with a.engine_log.open('rb') as handle:handle.seek(cancel_window_start);cancel_raw=handle.read()
            if a.arm=='off' or any(int(m.group(1))==cancel_row['counted_prompt_tokens'] for m in crp.CANCELLED.finditer(cancel_raw.decode(errors='replace'))):break
            require(time.monotonic()<deadline,'Cancelled native report did not arrive before reuse');client.wait(.02)
        cancel_log_window={'log_inode':cursor[0],'start_offset':cancel_window_start,'end_offset':cancel_window_start+len(cancel_raw),
            'observation_offset':cancel_observation.get('native_log_observation_offset'),'sha256':hashlib.sha256(cancel_raw).hexdigest(),'closed_epoch':time.time(),
            'scope':'Only this cancelled admission; snapshot before HTTP shutdown, window sealed after server drain and native cancelled report, before same-request reuse.'}
        if a.arm=='on':cancel_prefix=cancel_raw[:cancel_log_window['observation_offset']-cancel_window_start].decode(errors='replace')
        reuse={**cancel_row,'name':'admission-cancel-reuse','role':'injected'}
        record=send(reuse,cold=False);require(all(record['gates'].values()),'Cancelled-prompt reuse failed')
    except Exception as error:
        errors.append(str(error) if isinstance(error,RuntimeError) else type(error).__name__);client.abort.set()
    finally:
        deadline=time.monotonic()+(10 if errors else 300)
        for thread in threads:thread.join(max(0,deadline-time.monotonic()))
        with client.lock:active=list(client.active.values())
        for state in active:client.cancel(state,'smoke_timeout_cancel')
        for thread in threads:thread.join(12)
        client.monitor_stop.set();client.thread.join(2)
    require(not any(t.is_alive() for t in threads),'Own smoke workers remain')
    after=client.status();deadline=time.monotonic()+120
    while after['activity']['in_flight'] and time.monotonic()<deadline:time.sleep(.2);after=client.status()
    records=sorted(client.records,key=lambda r:r['name']);normal=[r for r in records if r['role']!='cancel'];cancelled=[r for r in records if r['role']=='cancel']
    parsed=evidence.parse(log_new())
    cancel_proof=crp.proof(a.arm,rows[5]['counted_prompt_tokens'],cancel_observation,records,cancel_raw.decode(errors='replace'),cancel_prefix) if a.arm=='on' else {'all_checks_passed':True,'approved_cancelled_read_ids':[],'scope':'Baseline has no concurrent reader exception.'}
    approved=cancel_proof['approved_cancelled_read_ids']
    marker_gate=evidence.validate(a.arm,'admission',log_new(),approved_cancelled_read_ids=approved)
    checks={'fixed_normal_set':{r['name'] for r in normal}=={r['name'] for r in rows[:5]}|{'admission-cancel-reuse'},
            'all_normal_gates':len(normal)==6 and all(r.get('gates') and all(r['gates'].values()) for r in normal),
            'cancelled_before_output':len(cancelled)==1 and cancelled[0].get('error')=={'kind':'intentional_admission_cancel'} and not cancelled[0]['chunks'],
            'cancel_observed_open_reader':a.arm=='off' or cancel_observation.get('reader_open_at_observation') is True,
            'idle_after':after['activity']['in_flight']==0,'exclusive_generation_count':after['activity']['requests']-before['activity']['requests']==7,
            'strict_intentional_cancel_proof':cancel_proof['all_checks_passed'],'all_native_markers_qualified':marker_gate['all_checks_passed'],
            'no_cuda_errors':not parsed['cuda_error_present'],'all_reads_paired':parsed['unique_and_paired_ids'],'no_errors':not errors}
    result={'all_checks_passed':all(checks.values()),'checks':checks,'errors':errors,'records':records,'cancel_observation':cancel_observation,
            'engine_evidence':parsed,'intentional_cancel_native':cancel_proof,'cancel_log_window':cancel_log_window,
            'approved_cancelled_read_ids':approved if all(checks.values()) else [],'status_before':before,'status_after':after}
    require(cfg['api_key'] not in json.dumps(result),'Credential in result');base.save(a.out/'results.json',result);base.save(a.out/'occupancy.json',client.occupancy)
    manifest=json.loads((a.out/'manifest.json').read_text());manifest.update(completed=True,all_checks_passed=result['all_checks_passed']);base.save(a.out/'manifest.json',manifest)
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],'checks':checks}),flush=True);return 0 if result['all_checks_passed'] else 2


def compare(left,right):
    from compare_interleave import records_equal
    results=[json.loads((p/'results.json').read_text()) for p in (left,right)];plans=[json.loads((p/'manifest.json').read_text()) for p in (left,right)]
    checks={'both_qualified':all(r.get('all_checks_passed') for r in results),'same_fixtures':plans[0]['fixtures']==plans[1]['fixtures'],
            'same_script':plans[0]['script_sha256']==plans[1]['script_sha256'],'same_plan':plans[0]['plan']==plans[1]['plan']}
    exact=records_equal(*[[r for r in report['records'] if r['role']!='cancel'] for report in results],expected=6)
    return {'all_checks_passed':all(checks.values()) and exact['all_exact'],'checks':checks,'exact_normal_outputs':exact,
            'cancel_observations':[r['cancel_observation'] for r in results],'scope':'Cancel requires no emitted content; all six normal outputs and logical usage must match exactly. HTTP cancellation does not imply immediate GPU stop.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--exclusive-benchmark',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--engine-log',type=Path);p.add_argument('--out',type=Path);p.add_argument('--port',type=int,default=8092);p.add_argument('--arm',choices=('off','on'),default='off');a=p.parse_args()
    if not a.execute:print(json.dumps({'mode':'DRY_RUN_NO_NETWORK_NO_GPU','fixtures':specs(),'cancel_before_output':True,'production_mutations':0},indent=2));return 0
    return run(a)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:print(json.dumps({'all_checks_passed':False,'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}),flush=True);raise SystemExit(1)
