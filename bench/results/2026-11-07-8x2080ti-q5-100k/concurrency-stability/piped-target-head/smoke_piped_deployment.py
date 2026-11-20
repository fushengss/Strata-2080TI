#!/usr/bin/env python3
"""Optional authenticated eight-lane synthetic startup smoke; dry-run by default.

--execute sends exactly eight short sampled requests to existing localhost:8080.
It never administers services or GPUs. Uses the frozen numeric/hash-only client;
no other requests, prompt text, output text, raw HTTP bodies or keys are saved.
This is a health/route smoke, never a performance benchmark, even when exclusive.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
import time
import uuid

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=Path('/srv/bench100k-20261107-piped-target-head')
CLIENT=ROOT/'p2-harness/client.py'
CLIENT_SHA='3c019afbcd75d7a183ce1089aea49ffd21abadf8aba85445573c369996c43b8f'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def routes(text):
    target=[[int(x) for x in r] for r in re.findall(r'strata piped target head: CUDA(\d+)\+(\d+), (\d+) windows, (\d+) verification rows offloaded in this batch',text)]
    sample=[[int(x) for x in r] for r in re.findall(r'^strata piped async sample: (\d+) windows, (\d+) sampled rows completed in this batch$',text,re.M)]
    draft=[[int(x) for x in r] for r in re.findall(r'strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch',text)]
    return {'target_head':target,'async_sample':sample,'mtp_head':draft,
            'target_verification_rows_are_not_output_tokens':True,'sampled_rows_are_not_output_tokens':True}


def route_checks(counter,flags):
    target,sample,draft=counter['target_head'],counter['async_sample'],counter['mtp_head']
    consistent=(all(r[:2]==[2,4] and r[2]>0 and r[3]>0 for r in target)
                and all((r[0]==r[1]==0) or (r[0]>0 and r[1]>0) for r in sample)
                and all(0<=r[0]<8 and r[1:3]==[3,5] and r[3]>=0 for r in draft)
                and (flags['STRATA_PIPED_TARGET_HEAD']=='1' or not target)
                and (flags['STRATA_PIPED_ASYNC_SAMPLE']=='1' or not sample))
    observed=(sum(r[3] for r in draft)>0
              and (flags['STRATA_PIPED_TARGET_HEAD']=='0' or sum(r[2] for r in target)>0)
              and (flags['STRATA_PIPED_ASYNC_SAMPLE']=='0' or sum(r[1] for r in sample)>0))
    return {'counter_devices_and_disabled_routes_consistent':consistent,'selected_routes_observed':observed}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    actions=p.add_mutually_exclusive_group();actions.add_argument('--execute',action='store_true');actions.add_argument('--self-test',action='store_true')
    p.add_argument('--out',type=Path,default=ROOT/'deployment-smoke')
    a=p.parse_args()
    if a.self_test:
        text='strata piped target head: CUDA2+4, 8 windows, 16 verification rows offloaded in this batch\nstrata piped async sample: 8 windows, 16 sampled rows completed in this batch\nstrata piped MTP head: slot 0, CUDA3+5, 8 draft steps offloaded in this batch\n'
        flags={'STRATA_PIPED_TARGET_HEAD':'1','STRATA_PIPED_ASYNC_SAMPLE':'1'}
        assert all(route_checks(routes(text),flags).values())
        flags['STRATA_PIPED_ASYNC_SAMPLE']='0';assert not route_checks(routes(text),flags)['counter_devices_and_disabled_routes_consistent']
        assert not route_checks(routes(''),flags)['selected_routes_observed']
        print(json.dumps({'offline_self_test':True,'passed':True,'checks':['numeric_route_parser','disabled_route_rejected','missing_route_not_claimed'],'http_requests':0}));return
    if not a.execute:
        print(json.dumps({'mode':'DRY_RUN_NO_REQUESTS','planned_own_requests':8,'max_output_tokens_each':128,
                          'temperature':0.8,'port':8080,'performance_claim':False,'out':str(a.out)},indent=2));return
    verify=module('piped_deployment_verifier',HERE/'verify_piped_deployment.py')
    cfg,receipt,receipt_sha=verify.installed_inputs()
    verified=verify.load(ROOT/'deployment-verified.json')
    verify.require(verified.get('all_checks_passed') is True and verified['deployment_receipt_sha256']==receipt_sha,
                   'Matching successful startup verification required before smoke')
    current=verify.actual_engine(cfg,receipt,verify.load(ROOT/'deployment-log-cursor.json'))
    verify.require([(r['pid'],r['start_ticks']) for r in current]==[(r['pid'],r['start_ticks']) for r in verified['actual_engine']],
                   'Engine changed after startup verification; verify the current process again')
    verify.require(verify.sha(CLIENT)==CLIENT_SHA,'Frozen numeric client SHA256 differs')
    verify.require(not a.out.exists(),'Smoke destination already exists')
    before=verify.status(cfg);mark=verify.mark_log(Path(cfg['log']))
    a.out.mkdir(mode=0o700);verify.save(a.out/'log-cursor.json',mark)
    clientmod=module('frozen_numeric_smoke_client',CLIENT)
    plan={'temperature':0.8,'seed':20261107,'thinking_enabled':False,'top_p':0.95,'top_k':20,
          'sampling_profile':'sampled','prompt_count_tolerance_fraction':0.1,'occupancy_sample_seconds':0.25,
          'scenario_total_timeout_seconds':120,'request_total_timeout_seconds':90,
          'request_ttft_timeout_seconds':45,'request_silence_timeout_seconds':30}
    client=clientmod.Client(8080,cfg,a.out,plan,'deployment-smoke-'+uuid.uuid4().hex)
    barrier=threading.Barrier(8);threads=[]
    def run(lane):
        barrier.wait(timeout=10)
        client.request(f'lane-{lane}','resident',lane,64,128,f'{client.fixture_id}-lane-{lane}')
    start=time.time();client.thread.start()
    try:
        for lane in range(8):
            thread=threading.Thread(target=run,args=(lane,),daemon=True);thread.start();threads.append(thread)
        for thread in threads:thread.join(timeout=125)
    finally:
        client.abort.set();client.monitor_stop.set();client.thread.join(timeout=2)
    verify.require(not any(t.is_alive() for t in threads),'Own smoke threads failed to finish')
    after=verify.status(cfg)
    # Summaries are emitted only when a batch drains. A continuing external batch may not emit one here.
    text,span=verify.log_since(Path(cfg['log']),mark);counter=routes(text);routing=route_checks(counter,receipt['flags'])
    rows=[]
    for r in sorted(client.records,key=lambda r:r['name']):
        usage=r.get('usage',{});complete=all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
        ok=(r['successful'] and r['finish_reason'] in ('stop','length') and complete
            and 0<usage['completion_tokens']<=128 and usage['prompt_tokens']+usage['completion_tokens']==usage['total_tokens'])
        rows.append({'name':r['name'],'passed':ok,'saw_done':r['saw_done'],'finish_reason':r['finish_reason'],
                     'usage':usage,'error':r['error'],'output_sha256':r['output_sha256'],
                     'channel_output_sha256':r['channel_output_sha256'],'output_bytes':r['output_bytes'],
                     'request_sha256':r['request_sha256']})
    a0=before['activity'];a1=after['activity']
    delta=a1.get('requests',0)-a0.get('requests',0) if 'requests' in a0 and 'requests' in a1 else None
    exclusive=(a0.get('in_flight')==0 and a1.get('in_flight')==0 and delta==8)
    own_pass=len(rows)==8 and all(r['passed'] for r in rows)
    result={'time_epoch':time.time(),'purpose':'Authenticated startup health smoke; no performance conclusion',
            'synthetic_only':True,'own_requests':rows,'own_request_checks_passed':own_pass,
            'deployment_receipt_sha256':receipt_sha,'client_sha256':CLIENT_SHA,
            'status_before':before,'status_after':after,'status_request_count_delta':delta,
            'exclusive_by_status_counters':exclusive,'external_traffic_possible':not exclusive,
            'route_counters_after_smoke_cursor':counter,'route_checks':routing,'log_span':span,
            'route_scope':'These are service batch summaries after this cursor; may include external work and only appear after drain.',
            'all_checks_passed':own_pass and all(routing.values()),'administrative_service_mutations':0,
            'generation_requests_sent':8,'performance_claim':False,
            'limitations':['Eight client requests do not prove eight simultaneous GPU executions.',
                          'With external traffic, route counters are not attributed solely to this smoke; no throughput or latency claim is made.',
                          'Missing route summaries while external work continues mean incomplete route observation, not necessarily an engine fault.']}
    verify.no_secret(result,cfg['api_key'])
    # Existing client persists only this cohort's numeric timestamps, bytes, usage and hashes.
    for path in a.out.iterdir():
        if path.is_file():verify.require(cfg['api_key'].encode() not in path.read_bytes(),'Credential detected in own smoke output')
    verify.save(a.out/'deployment-smoke.json',result)
    print(json.dumps(result,indent=2))
    return 0 if result['all_checks_passed'] else 2

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'all_checks_passed':False,'error_kind':type(error).__name__}));raise SystemExit(1)
