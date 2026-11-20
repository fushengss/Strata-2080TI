#!/usr/bin/env python3
"""HTTP-triggered BYIELD qualification. New synthetic fixtures; dry-run by default.

C4 peers -> cold100k reader -> short queued request -> actual park/resume -> all
six requests finish normally. No service/config/GPU mutation. Baseline has no
thread-reader markers and is a complete-output control, not a BYIELD claim.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import threading
import time
sys.dont_write_bytecode=True
import fixture_prompts as fp
import stream_client as base
import finite_mixed as finite
import admission_smoke as admission
import interleave_evidence as evidence
from calibrate_fixtures import calibrate,post,validate_count,require,sha

PARK=re.compile(r'^strata batch: the prompt read gives way at (\d+) of (\d+) tokens; slot (\d+) holds it \(copied in ([\d.]+) ms\)$',re.M)


def specs():
    result=[{'name':'byield-peer-'+str(i),'role':'resident','lane':i,'words':500,'output_tokens':4096} for i in range(4)]
    result += [{'name':'byield-long','role':'long','lane':None,'target_tokens':100000,'output_tokens':128},
               {'name':'byield-short','role':'short','lane':None,'words':0,'output_tokens':64}]
    return [{**r,'fixture':'prefill-interleave-byield-v2-'+r['name']} for r in result]


def telemetry_reading(port,headers,count):
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
    try:
        conn.request('GET','/v1/telemetry',headers=headers);response=conn.getresponse();require(response.status==200,'Telemetry HTTP error')
        raw=json.loads(response.read(2<<20));slots=(raw.get('work') or {}).get('slots',[])
        return [{'slot':s.get('slot'),'prompt_tokens':s.get('prompt_tokens'),'prompt_read':s.get('prompt_read'),
                 'state':'reading'} for s in slots if s.get('state')=='reading' and s.get('prompt_tokens')==count
                and type(s.get('slot')) is int]
    finally:conn.close()


def qualify(text,observed,count,records,arm):
    parsed=evidence.parse(text)
    parks=[{'at':int(q),'prompt_tokens':int(n),'slot':int(s),'copy_ms':float(ms)} for q,n,s,ms in PARK.findall(text)]
    checks={'no_cuda_errors':not parsed['cuda_error_present'],'all_markers_parsed':parsed['all_marker_lines_parsed'],
            'all_reads_paired':parsed['unique_and_paired_ids'],
            'all_reads_ok':all(p['read_ok']==p['pump_ok']==1 for p in parsed['pairs'])}
    proof=[]
    if arm=='on':
        first=next((p for p in parsed['pairs'] if p['read_id']==observed.get('reader_id')),None)
        checks['observed_open_reader']=observed.get('reader_open') is True and first is not None
        if first:
            checks['cold_long_read_before_short']=first['from']>=0 and first['tokens']>=.9*count and first['from']+first['tokens']<=count-1 and first['active_slots']>=4
            checks['reader_stop_for_yield']=first['yield_slot']>=0 and first['from']<first['stopped_at']<first['from']+first['tokens']
            matching=[p for p in parks if p['slot']==first['yield_slot'] and p['at']==first['stopped_at'] and p['prompt_tokens']==count]
            checks['actual_park_matches_stop']=len(matching)==1
            resumes=[p for p in parsed['pairs'] if p['read_id']>first['read_id'] and p['from']==first['stopped_at']
                     and p['chunk']==first['chunk'] and p['from']+p['tokens']==first['from']+first['tokens']]
            checks['same_read_plan_resumed']=bool(resumes)
            begin_positions={int(m.group(1)):m.start() for m in evidence.BEGIN.finditer(text)}
            end_positions={int(m.group(1)):m.start() for m in evidence.END.finditer(text)}
            park_positions=[m.start() for m in PARK.finditer(text) if int(m.group(1))==first['stopped_at'] and int(m.group(3))==first['yield_slot']]
            checks['stop_park_resume_log_order']=bool(park_positions and resumes) and any(begin_positions[first['read_id']]<end_positions[first['read_id']]<pos<begin_positions[r['read_id']]<end_positions[r['read_id']] for pos in park_positions for r in resumes)
            proof=[{'first_read':first,'park':matching,'resume':resumes}]
        else:
            checks.update(cold_long_read_before_short=False,reader_stop_for_yield=False,actual_park_matches_stop=False,same_read_plan_resumed=False,stop_park_resume_log_order=False)
        byname={r['name']:r for r in records};long=byname.get('byield-long',{});short=byname.get('byield-short',{})
        firsts=[r.get('first_visible_content_s') for r in (short,long)]
        checks['short_visible_before_long']=all(type(v) in (int,float) for v in firsts) and short['start_epoch']+firsts[0]<long['start_epoch']+firsts[1]
        checks['short_dispatched_after_open_observation']=short.get('start_epoch',0)>=observed.get('epoch',float('inf'))
    else:
        checks['baseline_overlap_disabled']=not parsed['begin'] and not parsed['fallback']
        checks['baseline_reading_observed']=bool(observed.get('telemetry_reading'))
    return {'all_checks_passed':all(checks.values()),'checks':checks,'proof':proof,'parks':parks,'native_reader_evidence':parsed,
            'scope':'Candidate needs actual native yield, parked partial state, same chunk-plan resume, and short first visible output before long. Baseline is output-equivalence control and need not yield.'}


def run(a):
    require(a.exclusive_benchmark and a.port==8092 and a.config and a.out and a.engine_log,'Explicit isolated config/out/log required')
    require(not a.out.exists(),'Output already exists')
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True);settings=finite.plan('byield')
    settings.update(request_total_timeout_seconds=420,scenario_total_timeout_seconds=480)
    client=base.Client(a.port,cfg,a.out,settings,'byield-v2');before=client.status()
    require(before['loaded'] and before['activity']['in_flight']==0 and before['concurrency']['serving']==8,'Server must be idle with8slots')
    rows=specs();texts={}
    def count(text):return validate_count(post(a.port,cfg['api_key'],'/v1/messages/count_tokens',fp.count_body(text,client.model)))
    for index,row in enumerate(rows):
        if 'target_tokens' in row:row=calibrate(row,count,2);rows[index]=row
        text=fp.prompt(row['words'],row['fixture']);texts[row['fixture']]=text
        row['counted_prompt_tokens']=count(text);row['prompt_sha256']=hashlib.sha256(text.encode()).hexdigest()
    require(abs(rows[4]['counted_prompt_tokens']-100000)<=2,'Long fixture not100k')
    require(rows[5]['counted_prompt_tokens']*2<=rows[4]['counted_prompt_tokens'],'Short request does not qualify server half-length rule')
    base.synthetic_prompt=lambda nominal,fixture:texts[fixture]
    base.save(a.out/'manifest.json',{'format':1,'plan':settings,'fixtures':rows,'script_sha256':sha(Path(__file__)),'completed':False})
    stat=a.engine_log.stat();cursor=(stat.st_ino,stat.st_size)
    def log_new():
        stat2=a.engine_log.stat();require(stat2.st_ino==cursor[0] and stat2.st_size>=cursor[1],'Owned log rotated')
        with a.engine_log.open('rb') as handle:handle.seek(cursor[1]);return handle.read().decode(errors='replace')
    threads=[];errors=[];observed={};barrier=threading.Barrier(4)
    def send(row,sync=False):
        if sync:barrier.wait(timeout=20)
        record=client.request(row['name'],row['role'],row['lane'],row['counted_prompt_tokens'],row['output_tokens'],row['fixture'])
        record['gates']=admission.normal_gates(record,row,cold=True);base.save(a.out/(row['name']+'.json'),record);return record
    client.thread.start()
    try:
        for row in rows[:4]:
            thread=threading.Thread(target=send,args=(row,True),daemon=True);thread.start();threads.append(thread)
        deadline=time.monotonic()+120
        while True:
            with client.lock:ready=sum(s['first'] is not None and s['role']=='resident' for s in client.active.values())==4
            if ready:break
            require(time.monotonic()<deadline and all(t.is_alive() for t in threads),'Four peers failed readiness');client.wait(.05)
        longrow=rows[4];thread=threading.Thread(target=send,args=(longrow,),daemon=True);thread.start();threads.append(thread)
        deadline=time.monotonic()+150
        while True:
            with client.lock:state=client.active.get(longrow['name']);no_output=state is not None and state['first'] is None
            if no_output:
                if a.arm=='on':
                    parsed=evidence.parse(log_new());ended={r['read_id'] for r in parsed['end']}
                    reads=[r for r in parsed['begin'] if r['read_id'] not in ended and r['tokens']>=.9*longrow['counted_prompt_tokens']]
                    if reads:
                        observed={'epoch':time.time(),'reader_open':True,'reader_id':reads[-1]['read_id'],'before_long_first_output':True};break
                else:
                    reading=telemetry_reading(a.port,client.headers,longrow['counted_prompt_tokens'])
                    if reading:
                        observed={'epoch':time.time(),'reader_open':False,'telemetry_reading':reading,'before_long_first_output':True};break
            require(time.monotonic()<deadline and thread.is_alive(),'Cold long admission was not observable before output');client.wait(.01)
        require(all(t.is_alive() for t in threads[:4]),'Peer completed before short request dispatch')
        short=send(rows[5]);require(all(short['gates'].values()),'Short request failed normal output/count gate')
    except Exception as error:
        errors.append(str(error) if isinstance(error,RuntimeError) else type(error).__name__);client.abort.set()
    finally:
        deadline=time.monotonic()+(10 if errors else 300)
        for thread in threads:thread.join(max(0,deadline-time.monotonic()))
        with client.lock:active=list(client.active.values())
        for state in active:client.cancel(state,'byield_timeout_cancel')
        for thread in threads:thread.join(12)
        client.monitor_stop.set();client.thread.join(2)
    require(not any(t.is_alive() for t in threads),'Own BYIELD workers remain')
    after=client.status();deadline=time.monotonic()+120
    while after['activity']['in_flight'] and time.monotonic()<deadline:time.sleep(.2);after=client.status()
    records=sorted(client.records,key=lambda r:r['name']);native=qualify(log_new(),observed,rows[4]['counted_prompt_tokens'],records,a.arm)
    checks={'fixed_normal_set':{r['name'] for r in records}=={r['name'] for r in rows},
            'all_normal_gates':len(records)==6 and all(r.get('gates') and all(r['gates'].values()) for r in records),
            'idle_after':after['activity']['in_flight']==0,'exclusive_generation_count':after['activity']['requests']-before['activity']['requests']==6,
            'native_qualification':native['all_checks_passed'],'no_errors':not errors}
    report={'all_checks_passed':all(checks.values()),'checks':checks,'errors':errors,'records':records,'open_observation':observed,
            'native_qualification':native,'status_before':before,'status_after':after}
    require(cfg['api_key'] not in json.dumps(report),'Credential in result');base.save(a.out/'results.json',report);base.save(a.out/'occupancy.json',client.occupancy)
    manifest=json.loads((a.out/'manifest.json').read_text());manifest.update(completed=True,all_checks_passed=report['all_checks_passed']);base.save(a.out/'manifest.json',manifest)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'checks':checks,'native_checks':native['checks']}),flush=True)
    return 0 if report['all_checks_passed'] else 2


def compare(left,right):
    from compare_interleave import records_equal
    results=[json.loads((p/'results.json').read_text()) for p in (left,right)];plans=[json.loads((p/'manifest.json').read_text()) for p in (left,right)]
    checks={'both_qualified':all(r.get('all_checks_passed') for r in results),'same_fixtures':plans[0]['fixtures']==plans[1]['fixtures'],
            'same_script':plans[0]['script_sha256']==plans[1]['script_sha256'],'same_plan':plans[0]['plan']==plans[1]['plan'],
            'candidate_native_byield':results[1]['native_qualification']['all_checks_passed'] and results[1]['native_qualification']['checks'].get('same_read_plan_resumed') is True}
    exact=records_equal(*[r['records'] for r in results],expected=6)
    return {'all_checks_passed':all(checks.values()) and exact['all_exact'],'checks':checks,'exact_normal_outputs':exact,
            'native_qualifications':[r['native_qualification'] for r in results],'scope':'Six complete normal request/output hashes and logical usages match. Candidate additionally must prove BYIELD park/resume, not just receipt of a yield flag.'}


def self_test():
    begin='strata piped concurrent MTP: begin read_id=1 from=0 tokens=99999 chunk=2048 active_slots=4 stages=8 own_buffers=8 resident_stages=8'
    end='strata piped concurrent MTP: end read_id=1 read_ms=1500 decode_rows_before_done=20 read_ok=1 pump_ok=1 stopped_at=4096 admit_stop_requested=0 yield_slot=4'
    park='strata batch: the prompt read gives way at 4096 of 100000 tokens; slot 4 holds it (copied in 12.3 ms)'
    resume='strata piped concurrent MTP: begin read_id=2 from=4096 tokens=95903 chunk=2048 active_slots=5 stages=8 own_buffers=8 resident_stages=8\nstrata piped concurrent MTP: end read_id=2 read_ms=8000 decode_rows_before_done=100 read_ok=1 pump_ok=1 stopped_at=-1 admit_stop_requested=0 yield_slot=-1'
    text='\n'.join((begin,end,park,resume));observed={'reader_open':True,'reader_id':1,'epoch':2}
    records=[{'name':'byield-short','start_epoch':2.1,'first_visible_content_s':.5},{'name':'byield-long','start_epoch':1,'first_visible_content_s':10}]
    assert qualify(text,observed,100000,records,'on')['all_checks_passed']
    for broken in (text.replace(park,''),text.replace('yield_slot=4','yield_slot=-1'),text.replace('from=4096','from=2048'),text.replace('tokens=95903','tokens=95000'),text.replace('copied in 12.3','copied in ERROR'),begin+'\n'+park+'\n'+end+'\n'+resume,text+'\nCUDA error: failed'):
        assert not qualify(broken,observed,100000,records,'on')['all_checks_passed']
    wrong=[records[0],{**records[1],'first_visible_content_s':.1}];assert not qualify(text,observed,100000,wrong,'on')['all_checks_passed']
    assert not qualify(text,{**observed,'reader_open':False},100000,records,'on')['all_checks_passed']
    assert qualify('',{'telemetry_reading':[{'slot':4}]},100000,records,'off')['all_checks_passed']
    print('PASS: native BYIELD park/resume and short-first positive/negative qualification; no network/GPU')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--self-test',action='store_true');p.add_argument('--exclusive-benchmark',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--engine-log',type=Path);p.add_argument('--out',type=Path);p.add_argument('--port',type=int,default=8092);p.add_argument('--arm',choices=('off','on'),default='off');a=p.parse_args()
    if a.self_test:require(not a.execute,'Self-test cannot execute');self_test();return 0
    if not a.execute:print(json.dumps({'mode':'DRY_RUN_NO_NETWORK_NO_GPU','fixtures':specs(),'long_calibrated_count_target':100000,'ordinary_chat_triggers_byield':True,'production_mutations':0},indent=2));return 0
    return run(a)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:print(json.dumps({'all_checks_passed':False,'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}),flush=True);raise SystemExit(1)
