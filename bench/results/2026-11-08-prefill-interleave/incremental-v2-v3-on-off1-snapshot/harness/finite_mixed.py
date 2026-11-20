#!/usr/bin/env python3
"""Fixed finite mixed-prefill cohort, against an existing isolated local server.

Default dry-run. Never starts/stops servers, changes config/clocks, clears caches,
or sends messages/tools. Four resident requests, followed by fixed cold injections;
no time-driven refills. Final usages and hashes bind exactly the same request set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

sys.dont_write_bytecode = True
import fixture_prompts as fp
import stream_client as base
import stall_metrics as sm
import interleave_evidence as ie
from calibrate_fixtures import post, require, sha, validate_count

PIPED = re.compile(r'strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, '
                   r'(\d+) rows in ([\d.]+) ms = ([\d.]+) rows/s[^\n]*?\((\d+) of (\d+) MTP proposals kept\)')
TRACE = re.compile(r'strata trace: admission of slot (\d+): drain ([\d.]+), prompt ([\d.]+) '
                   r'\(read ([\d.]+), windows ([\d.]+), (\d+) checkpoint\(s\) ([\d.]+) = sync ([\d.]+) '
                   r'\+ save ([\d.]+) \+ evict ([\d.]+), rest ([\d.]+)\), first window ([\d.]+), '
                   r'slot copy ([\d.]+), slot freed ([\d.]+), checkpoint copy ([\d.]+)')


def plan(case):
    return {'temperature': .8, 'seed': 20261108, 'top_p': .95, 'top_k': 20,
            'thinking_enabled': False, 'sampling_profile': 'sampled-answer',
            'prompt_count_tolerance_fraction': .0002, 'occupancy_sample_seconds': .05,
            'scenario_total_timeout_seconds': 600, 'request_total_timeout_seconds': 540,
            'request_ttft_timeout_seconds': 240, 'request_silence_timeout_seconds': 150,
            'resident_ready_timeout_seconds': 240, 'resident_ready_stable_seconds': 2,
            'baseline_seconds': 5, 'injection_interval_seconds': 8, 'recovery_seconds': 10,
            'drain_timeout_seconds': 300, 'case': case}


def evidence(log, cursor):
    if log is None:
        return {'available': False}
    stat = log.stat()
    require(stat.st_ino == cursor[0] and stat.st_size >= cursor[1], 'Owned engine log rotated during case')
    with log.open('rb') as handle:
        handle.seek(cursor[1]); text = handle.read().decode(errors='replace')
    native = [dict(zip(('groups','slots_per_group','group_steps','rows','wall_ms','rows_per_s','drafts_accepted','drafts_offered'),
                      (int(x[0]),int(x[1]),int(x[2]),int(x[3]),float(x[4]),float(x[5]),int(x[6]),int(x[7])))) for x in PIPED.findall(text)]
    traces = [dict(zip(('slot','drain_ms','prompt_ms','read_ms','windows_ms','checkpoints','checkpoint_ms','sync_ms','save_ms',
                       'evict_ms','rest_ms','first_window_ms','slot_copy_ms','slot_freed_ms','checkpoint_copy_ms'),
                      [int(v) if i in (0,5) else float(v) for i,v in enumerate(x)])) for x in TRACE.findall(text)]
    return {'available': True, 'start_offset':cursor[1], 'end_offset':stat.st_size,
            'native_piped_batches':native, 'admission_trace_rows':traces,
            'native_rate_includes_admission_pauses':True, 'reader_pump':ie.parse(text),
            'trace_binding':'Exclusive-case numeric admission sequence; no automatic slot-to-HTTP-request assignment.'}


def request_gates(record, spec):
    usage, timings = record.get('usage',{}), record.get('native_timings',{})
    actual=usage.get('completion_tokens');finish=record.get('finish_reason');budget=spec['output_tokens']
    normal_output_contract=type(actual) is int and ((finish=='length' and actual==budget) or (finish=='stop' and 0<actual<=budget))
    return {'normal_complete': record.get('successful') is True and record.get('saw_done') is True and record.get('error') is None and normal_output_contract,
            'exact_calibrated_prompt': usage.get('prompt_tokens') == spec['counted_prompt_tokens'],
            'normal_output_within_fixed_budget': normal_output_contract,
            'native_output_accounting':type(timings.get('predicted_n')) is int and timings['predicted_n']==actual,
            'usage_consistent': all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
                and usage.get('total_tokens') == usage.get('prompt_tokens',0)+usage.get('completion_tokens',0),
            'cold_cache_explicitly_zero': (usage.get('prompt_tokens_details') or {}).get('cached_tokens') == 0,
            'answer_visible': record.get('first_visible_content_s') is not None,
            'no_reasoning_or_tools': all(c['bytes']['reasoning'] == 0 and c['bytes']['tool'] == 0 and 'tool' not in c.get('channels',[]) for c in record['chunks']),
            'native_prompt_timing_available': type(timings.get('prompt_ms')) in (int,float) and timings['prompt_ms'] >= 0,
            'native_prompt_accounting': timings.get('cache_n') == 0 and timings.get('prompt_n') == spec['counted_prompt_tokens']}


def visible_phase_coverage(records,phases):
    """Extra qualification; the existing silence/phase metrics remain unchanged."""
    peers=[r for r in records if r.get('role')=='resident']
    lanes=[]
    for r in peers:
        epochs=[c['epoch'] for c in r.get('chunks',[]) if c.get('bytes',{}).get('visible_content',0)>0]
        first,last=(min(epochs),max(epochs)) if epochs else (None,None)
        coverage={name:first is not None and first<=begin and last>=end for name,(begin,end) in phases.items()}
        lanes.append({'lane':r.get('lane'),'first_visible_epoch':first,'last_visible_epoch':last,'phase_coverage':coverage})
    full=len(peers)==4 and {r.get('lane') for r in peers}==set(range(4)) and set(phases)=={'baseline','mixed','recovery'} and all(all(r['phase_coverage'].values()) for r in lanes)
    return {'all_checks_passed':full,'lanes':lanes,
            'scope':'Each of four finite resident streams has visible output before and after every measured phase. Delayed HTTP/terminal completion alone does not qualify. Existing silence metrics are unchanged.'}


def run(a):
    require(a.port == 8092, 'Timed finite benchmark uses isolated loopback8092 only')
    require(a.exclusive_benchmark, 'Explicit exclusive benchmark assertion required')
    require(a.config and a.calibration and a.out, 'Explicit config/calibration/out required')
    require(not a.out.exists(), 'Output directory already exists')
    cfg = json.loads(a.config.read_text()); frozen = json.loads(a.calibration.read_text())
    require(frozen['fixture_generator_sha256'] == sha(Path(fp.__file__)), 'Fixture generator differs from calibration')
    require(frozen['thinking_enabled'] is False and frozen['model'] == cfg.get('model_name','strata'), 'Model/thinking differs')
    specs = [r for r in frozen['fixtures'] if r['case'] == a.case]
    require([r['name'] for r in specs] == [r['name'] for r in fp.specs(frozen['fixture_id'],(a.case,))], 'Frozen finite request set differs')
    fixture_map = {r['fixture']:r for r in specs}
    def calibrated_prompt(nominal, fixture):
        row = fixture_map[fixture]; require(nominal == row['target_tokens'], 'Unexpected nominal prompt request')
        text = fp.prompt(row['words'], fixture)
        require(hashlib.sha256(text.encode()).hexdigest() == row['prompt_sha256'], 'Synthetic prompt differs from calibration')
        return text
    base.synthetic_prompt = calibrated_prompt
    settings=plan(a.case)
    a.out.mkdir(parents=True)
    client=base.Client(a.port,cfg,a.out,settings,frozen['fixture_id'])
    status0=client.status()
    require(status0['loaded'] and status0['activity']['in_flight']==0 and status0['concurrency']['serving']==8, 'Isolated server must be loaded, idle and have8slots')
    count_checks=[]
    for row in specs:
        actual=validate_count(post(a.port,cfg['api_key'],'/v1/messages/count_tokens',fp.count_body(calibrated_prompt(row['target_tokens'],row['fixture']),client.model)))
        require(actual==row['counted_prompt_tokens'], 'Frozen prompt count differs on this server')
        count_checks.append({'name':row['name'],'counted_tokens':actual})
    before=client.status()
    require(before['activity']['in_flight']==0, 'Foreign work arrived during count preflight')
    base.save(a.out/'manifest.json',{'format':1,'case':a.case,'plan':settings,'calibration_sha256':sha(a.calibration),
              'fixture_generator_sha256':sha(Path(fp.__file__)),'client_sha256':sha(Path(__file__)),
              'stream_client_sha256':sha(Path(base.__file__)),'stall_metrics_sha256':sha(Path(sm.__file__)),
              'fixtures':specs,'preflight_counts':count_checks,'completed':False,
              'output_contract':'Fixed request bodies and max output budgets. length must exhaust budget; stop may end normally at1..budget. Exact actual outputs/usage/finish reason and full measured-phase C4 coverage are required for A/B.'})
    log=a.engine_log;cursor=(log.stat().st_ino,log.stat().st_size) if log else None
    phases={};injections=[];threads=[];errors=[];started=time.time();barrier=threading.Barrier(4)
    client.thread.start()
    def send(row, sync=False, planned=None):
        if sync:barrier.wait(timeout=20)
        injection_cursor=(log.stat().st_ino,log.stat().st_size) if log and row['role']=='injected' else None
        r=client.request(row['name'],row['role'],row['lane'],row['target_tokens'],row['output_tokens'],row['fixture'],planned)
        if injection_cursor:r['admission_read_evidence']=evidence(log,injection_cursor)
        r['expected_prompt_tokens']=row['counted_prompt_tokens'];r['gates']=request_gates(r,row)
        base.save(a.out/(row['name']+'.json'),r)
        return r
    try:
        for row in [r for r in specs if r['role']=='resident']:
            thread=threading.Thread(target=send,args=(row,True),daemon=True);thread.start();threads.append(thread)
        deadline=time.monotonic()+settings['resident_ready_timeout_seconds'];stable=None
        while True:
            now=time.monotonic()
            with client.lock:
                ready=sum(s['role']=='resident' and s['first'] is not None and now-s['last']<=1 for s in client.active.values())==4
            stable=(stable if stable is not None else now) if ready else None
            if stable is not None and now-stable>=settings['resident_ready_stable_seconds']:break
            require(now<deadline,'Four finite resident streams did not become ready')
            require(not any(not t.is_alive() for t in threads),'A finite resident completed before cohort readiness')
            client.wait(.05)
        begin=time.time();print('READY finite C4 '+a.case,flush=True)
        client.wait(settings['baseline_seconds']);mixed=time.time();phases['baseline']=(begin,mixed)
        for index,row in enumerate(r for r in specs if r['role']=='injected'):
            planned=mixed+index*settings['injection_interval_seconds'];client.wait(planned-time.time())
            require(all(t.is_alive() for t in threads),'A finite resident completed before an injection')
            print('INJECT '+row['name']+' actual_tokens='+str(row['counted_prompt_tokens']),flush=True)
            r=send(row,planned=planned);injections.append(r)
            require(all(r['gates'].values()),'Injected request failed numeric/visible/cold/count gate')
        recovery=time.time();phases['mixed']=(mixed,recovery)
        client.wait(settings['recovery_seconds']);phases['recovery']=(recovery,time.time())
        require(all(t.is_alive() for t in threads),'A finite resident completed before recovery ended')
    except Exception as error:
        errors.append(str(error) if isinstance(error,RuntimeError) else type(error).__name__);client.abort.set()
    finally:
        deadline=time.monotonic()+(10 if errors else settings['drain_timeout_seconds'])
        for t in threads:t.join(max(0,deadline-time.monotonic()))
        with client.lock:active=list(client.active.values())
        for state in active:client.cancel(state,'finite_case_timeout_cancel')
        for t in threads:t.join(12)
        client.monitor_stop.set();client.thread.join(2)
    require(not any(t.is_alive() for t in threads),'Own finite workers did not stop')
    after=client.status();idle_deadline=time.monotonic()+15
    while after['activity']['in_flight'] and time.monotonic()<idle_deadline:
        time.sleep(.25);after=client.status()
    records=sorted(client.records,key=lambda r:r['name'])
    expected={r['name']:r for r in specs}
    checks={'fixed_request_set':{r['name'] for r in records}==set(expected),
            'all_request_gates':len(records)==len(specs) and all(all(r.get('gates',{}).values()) and r.get('gates') for r in records),
            'three_phases_complete':set(phases)=={'baseline','mixed','recovery'},
            'idle_after':after['activity']['in_flight']==0,
            'exclusive_generation_count':after['activity']['requests']-before['activity']['requests']==len(specs),
            'no_workload_errors':not errors}
    engine_evidence=evidence(log,cursor)
    checks['no_cuda_log_errors']=not engine_evidence.get('reader_pump',{}).get('cuda_error_present',False)
    if cfg.get('env',{}).get('STRATA_PIPED_CONCURRENT_MTP')=='1':
        long_injections=[r for r in injections if r['expected_prompt_tokens']>=60000]
        checks['long_injection_read_bound_and_progress']=bool(long_injections) and all(
            any(p['tokens']>=.9*r['expected_prompt_tokens'] and p['from']+p['tokens']<=r['expected_prompt_tokens']-1 and p['active_slots']>=4 and p['decode_rows_before_done']>0
                for p in r.get('admission_read_evidence',{}).get('reader_pump',{}).get('pairs',[]))
            for r in long_injections)
    metrics={name:{channel:sm.phase(records,*bounds,channel=channel) for channel in base.CHANNELS} for name,bounds in phases.items()}
    checks['all_four_residents_cover_every_phase']=all(metrics.get(name,{}).get('output',{}).get('all_lanes_cover_phase') for name in ('baseline','mixed','recovery'))
    visible_coverage=visible_phase_coverage(records,phases)
    checks['four_visible_streams_cover_every_phase']=visible_coverage['all_checks_passed']
    injection_metrics=[]
    for row in injections:
        end=row['start_epoch']+row['ttft_s'] if row['ttft_s'] is not None else row['end_epoch']
        injection_metrics.append({'name':row['name'],'actual_prompt_tokens':row['actual_prompt_tokens'],
             'ttft_first_output_s':row['ttft_s'],'ttft_answer_s':row['first_visible_content_s'],
             'engine_timings':row.get('native_timings'), 'admission_read_evidence':row.get('admission_read_evidence'),
             'resident_admission_span':sm.phase(records,row['start_epoch'],end),
             'scope':'HTTP request start to first injected output includes queue/admission/prefill/first decode, not pure GPU read.'})
    elapsed=max((r['end_epoch'] for r in records),default=started)-min((r['start_epoch'] for r in records),default=started)
    tokens=sum(r.get('usage',{}).get('completion_tokens',0) for r in records)
    report={'all_checks_passed':all(checks.values()),'checks':checks,'errors':errors,'case':a.case,'records':records,
            'phases':metrics,'visible_phase_coverage':visible_coverage,'injections':injection_metrics,'status_before':before,'status_after':after,
            'whole_case_api_rate':{'completion_tokens':tokens,'elapsed_s':elapsed,'tokens_per_s':tokens/elapsed if elapsed else None,
                                   'scope':'First request start to last request end; includes cold admission, injected prefill and final drain.'},
            'engine_log_evidence':engine_evidence,'output_contract':'Fixed requests and max budgets, not guaranteed output lengths; normal EOS accepted only with exact matched actual work and full measured-phase C4 coverage.',
            'timing_limits':['SSE chunks are not token boundaries.',
                'Native busy-block rates include admission pauses; no pure-decode GPU rate is inferred by subtracting overlapping work.',
                'One fixed cohort per arm does not establish a tail SLO. Use reversed/repeated matched arms for a claimed stability gain.']}
    require(cfg['api_key'] not in json.dumps(report),'Credential in report')
    base.save(a.out/'results.json',report);base.save(a.out/'occupancy.json',client.occupancy)
    manifest=json.loads((a.out/'manifest.json').read_text());manifest.update(completed=True,all_checks_passed=report['all_checks_passed'],end_epoch=time.time())
    base.save(a.out/'manifest.json',manifest)
    print(json.dumps({'all_checks_passed':report['all_checks_passed'],'checks':checks,'whole_case_api_rate':report['whole_case_api_rate']}),flush=True)
    return 0 if report['all_checks_passed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute',action='store_true');p.add_argument('--exclusive-benchmark',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--calibration',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--engine-log',type=Path);p.add_argument('--case',choices=('primary','long'),default='primary')
    p.add_argument('--port',type=int,default=8092)
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({'mode':'DRY_RUN_NO_NETWORK','case':a.case,'plan':plan(a.case),
                          'finite_requests':fp.specs('FROZEN_FIXTURE_ID',(a.case,)),
                          'calibration_required':True,'production_mutations':0},indent=2));return 0
    return run(a)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'all_checks_passed':False,'error':str(error) if isinstance(error,RuntimeError) else type(error).__name__}),flush=True)
        raise SystemExit(1)
