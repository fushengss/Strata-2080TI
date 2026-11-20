#!/usr/bin/env python3
"""Optional client-only integration suite; never starts/stops/reconfigures a server.

Run separately against idle off/on servers, then compare their result directories.
The parallel>1 API permits logprobs only with max_tokens=1; that admission-path
check is API compatibility, NOT validation of the PW2 offloaded head's logits.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench_client import Bench, save_json, log_cursor, read_log, TASKS


def hash_json(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def successful(r):
    return r.get('error') is None and r.get('saw_done') and bool(r.get('usage')) and bool(r.get('output_text') or r.get('reasoning_text'))


def lp_values(x):
    values=[]
    if isinstance(x,dict):
        for key,val in x.items():
            if key=='logprob':
                values.append(val)
            else:
                values.extend(lp_values(val))
    elif isinstance(x,list):
        for val in x:
            values.extend(lp_values(val))
    return values


def compare(a):
    left,right=[Path(x) for x in a.compare]
    b,c=[json.loads((x/'integration.json').read_text()) for x in (left,right)]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    bm={r['name']:r for r in b['requests']}
    cm={r['name']:r for r in c['requests']}
    for name in sorted(set(bm)|set(cm)):
        x,y=bm.get(name),cm.get(name)
        if x is None or y is None:
            comparisons.append({'name':name,'state':'missing_request'});continue
        same_input=x.get('request_sha256')==y.get('request_sha256')
        same_text=(x.get('output_text'),x.get('reasoning_text'))==(y.get('output_text'),y.get('reasoning_text'))
        row={'name':name,'same_input':same_input,'same_text_and_reasoning':same_text,
             'baseline_success':successful(x),'candidate_success':successful(y),
             'baseline_output_sha256':x.get('combined_output_sha256'),
             'candidate_output_sha256':y.get('combined_output_sha256')}
        if name.startswith('logprobs-'):
            xl,yl=x.get('logprobs'),y.get('logprobs')
            xv,yv=lp_values(xl),lp_values(yl)
            row['same_logprobs_exactly']=xl==yl and bool(xl) and bool(yl)
            row['logprob_value_count_baseline']=len(xv)
            row['logprob_value_count_candidate']=len(yv)
            row['max_logprob_abs_difference_by_returned_position']=max((abs(float(i)-float(j)) for i,j in zip(xv,yv)),default=None) if len(xv)==len(yv) else None
        row['state']='exact_match' if same_input and same_text and row['baseline_success'] and row['candidate_success'] and row.get('same_logprobs_exactly',True) else 'difference_or_incomplete'
        comparisons.append(row)
    report={'baseline':str(left),'candidate':str(right),'comparisons':comparisons,
            'all_completed_requests_exactly_match':bool(comparisons) and all(r['state']=='exact_match' for r in comparisons),
            'baseline_checks':b['checks'],'candidate_checks':c['checks'],
            'logprobs_scope':c['logprobs_scope'],
            'note':'Cancellation prefixes are intentionally excluded from off/on exact-output comparison: GPU progress at disconnect can differ. The two completed follow-up responses must match their controls and the other arm.'}
    save_json(out/'integration-comparison.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def run_suite(a):
    out=Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('--out must be new or empty')
    bench=Bench(a)
    before=bench.status()
    if before.get('in_flight')!=0:
        raise SystemExit('Server must be idle; no service operation was performed')
    with urllib.request.urlopen(urllib.request.Request(bench.base+'/v1/status',headers=bench.headers),timeout=10) as f:
        state=json.load(f)
    parallel=(state.get('concurrency') or {}).get('serving') or bench.cfg.get('parallel',1)
    lp_tokens=a.logprobs_tokens if a.logprobs_tokens is not None else (1 if parallel>1 else 16)
    if parallel>1 and lp_tokens!=1:
        raise SystemExit('0157 with parallel>1 only supports logprobs max_tokens=1; use a separate parallel1 server for a multi-token test')
    scope=('One-token API admission-path compatibility only; does not exercise PW2 offloaded-head logits.' if parallel>1 else
           'Multi-token API test on a parallel1 server; inspect head offload records to confirm the execution path.')
    suite={'fixture_version':'decode-head-integration-1','start_epoch':time.time(),'initial_activity':before,
           'parallel':parallel,'logprobs_scope':scope,'logprobs_max_tokens':lp_tokens,
           'config_fields':{k:bench.cfg[k] for k in ('exe','gpu','parallel','layer_split') if k in bench.cfg},
           'head_gpu':bench.cfg.get('env',{}).get('STRATA_DECODE_HEAD_GPU','-1'),
           'cohorts':[],'requests':[],'checks':{}}
    save_json(out/'integration.json',suite)
    for repeat in range(a.concurrency_repeats):
        cursor=log_cursor(bench.log)
        barrier=threading.Barrier(2)
        done=threading.Event()
        samples=[]
        def observe():
            while not done.is_set():
                samples.append({'epoch':time.time(),'activity':bench.status()})
                done.wait(0.2)
        mt=threading.Thread(target=observe,daemon=True);mt.start()
        prompts=[
            '[Concurrent integration alpha] '+TASKS['code'],
            '[Concurrent integration beta] Write a Python module for a bounded FIFO task queue with priorities, cancellation, a thread-safe wrapper, and at least eight unittest cases. Output only code.',
        ]
        def one(index):
            barrier.wait(timeout=10)
            return bench.request(f'concurrent-{repeat+1}-{index+1}',[{'role':'user','content':prompts[index]}],a.concurrent_tokens,
                                 'integration-concurrent',log_owned=False,overrides={'seed':101+index,'temperature':0})
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(one,i) for i in range(2)]
                records=[f.result(timeout=300) for f in futures]
        finally:
            done.set();mt.join(10)
        starts=[r['start_epoch']+r['ttft_s'] for r in records if r['ttft_s'] is not None]
        ends=[r['start_epoch']+r['last_content_s'] for r in records if r['last_content_s'] is not None]
        overlap=max(0,min(ends)-max(starts)) if len(starts)==2 and len(ends)==2 else None
        cohort={'repeat':repeat+1,'activity_samples':samples,'max_observed_in_flight':max((s['activity'].get('in_flight',0) for s in samples),default=0),
                'stream_content_overlap_s':overlap,'both_completed':all(successful(r) for r in records),
                'engine_lines':read_log(bench.log,cursor),'requests':[r['name'] for r in records]}
        suite['cohorts'].append(cohort);suite['requests'].extend(records)
        save_json(out/'integration.json',suite)
    suite['checks']['concurrent_pairs_completed']=all(c['both_completed'] for c in suite['cohorts'])
    suite['checks']['concurrency_observed_in_every_pair']=all(c['max_observed_in_flight']>=2 and c['stream_content_overlap_s'] and c['stream_content_overlap_s']>0 for c in suite['cohorts'])
    # Controls make stale tokens or a shifted DONE immediately visible after cancellation.
    markers=('HEAD_DRAIN_ALPHA','HEAD_DRAIN_BETA')
    controls=[]
    for i,marker in enumerate(markers):
        messages=[{'role':'user','content':'[Drain integration '+marker+'] Reply with exactly '+marker+' and nothing else.'}]
        r=bench.request(f'drain-control-{i+1}',messages,24,'integration-drain')
        controls.append((messages,r));suite['requests'].append(r)
    cursor=log_cursor(bench.log)
    body={'model':bench.cfg.get('model_name','strata'),'messages':[{'role':'user','content':'[Cancellation integration] Write a complete Python implementation of a persistent task scheduler with retries, task dependencies, a thread-safe queue, JSON checkpoints, logging and extensive unit tests. Include all implementation details and tests.'}],
          'max_tokens':4096,'temperature':0,'seed':303,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1.0,
          'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
    rawbody=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
    cancel={'request_sha256':hashlib.sha256(rawbody).hexdigest(),'start_epoch':time.time(),'target_nonempty_chunks':a.cancel_after_chunks,
            'note':'Chunk count is not a token count. Socket close is intentional; missing usage/DONE is expected.'}
    chunks=[];saw_done=False;reason=None
    req=urllib.request.Request(bench.base+'/v1/chat/completions',data=rawbody,headers=bench.headers)
    with urllib.request.urlopen(req,timeout=300) as response:
        for raw in response:
            line=raw.decode(errors='replace').strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':saw_done=True;break
            event=json.loads(payload)
            for ch in event.get('choices') or []:
                dl=ch.get('delta') or {};piece=dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning')
                if piece:chunks.append({'epoch':time.time(),'text':piece})
                reason=ch.get('finish_reason') or reason
            if len(chunks)>=a.cancel_after_chunks:
                break
    cancel.update(closed_epoch=time.time(),chunks=chunks,saw_done_before_close=saw_done,finish_reason_before_close=reason,
                  closed_mid_generation=len(chunks)>=a.cancel_after_chunks and not saw_done and reason is None)
    suite['cancelled_request']=cancel;save_json(out/'integration.json',suite)
    # Do not wait for idle: the immediate next admission must drain the canceled owner correctly.
    follow=[]
    for i,(messages,control) in enumerate(controls):
        r=bench.request(f'drain-after-{i+1}',messages,24,'integration-drain')
        follow.append(r);suite['requests'].append(r)
    suite['cancel_and_followup_engine_lines']=read_log(bench.log,cursor)
    suite['checks']['cancel_closed_mid_generation']=cancel['closed_mid_generation']
    suite['checks']['drain_controls_contain_own_markers']=all(successful(r) and r['output_text'].strip()==markers[i] for i,(_,r) in enumerate(controls))
    suite['checks']['drain_followups_match_controls']=all(successful(r) and r['output_text']==controls[i][1]['output_text'] and r.get('reasoning_text','')==controls[i][1].get('reasoning_text','') for i,r in enumerate(follow))
    save_json(out/'integration.json',suite)
    # Supported OpenAI parameters verified in serve/server.py: logprobs:true,
    # top_logprobs:0..20. Do not request unsupported long replies under parallel8.
    lp_prompts=['[Logprobs integration one] What is 2 + 2? Answer only the digit.',
                '[Logprobs integration two] Reply with the uppercase letter A only.',
                '[Logprobs integration three] Complete the phrase with one word: the opposite of cold is']
    for i,prompt in enumerate(lp_prompts):
        body={'model':bench.cfg.get('model_name','strata'),'messages':[{'role':'user','content':prompt}],
              'max_tokens':lp_tokens,'temperature':0,'seed':401+i,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1.0,
              'chat_template_kwargs':{'enable_thinking':False},'logprobs':True,'top_logprobs':5,'stream':False}
        rawbody=json.dumps(body,ensure_ascii=False,separators=(',',':')).encode();cursor=log_cursor(bench.log)
        r={'name':f'logprobs-{i+1}','category':'integration-logprobs','request_sha256':hashlib.sha256(rawbody).hexdigest(),
           'request_settings':{k:v for k,v in body.items() if k!='messages'},'error':None,'start_epoch':time.time()}
        try:
            with urllib.request.urlopen(urllib.request.Request(bench.base+'/v1/chat/completions',data=rawbody,headers=bench.headers),timeout=300) as response:
                data=json.load(response)
            choice=data['choices'][0];message=choice.get('message') or {}
            r.update(response=data,usage=data.get('usage'),output_text=message.get('content') or '',
                     reasoning_text=message.get('reasoning_content') or message.get('reasoning') or '',
                     logprobs=choice.get('logprobs'),finish_reason=choice.get('finish_reason'),saw_done=True)
            r['combined_output_sha256']=hash_json([r['output_text'],r['reasoning_text']])
            vals=lp_values(r['logprobs']);r['logprobs_finite_and_present']=bool(vals) and all(isinstance(v,(float,int)) and math.isfinite(v) for v in vals)
        except Exception as e:
            r['error']={'kind':type(e).__name__,'detail':bench.safe(str(e))}
            if isinstance(e,urllib.error.HTTPError):r['error']['body']=bench.safe(e.read().decode(errors='replace'))
        r['end_epoch']=time.time();r['engine_lines']=read_log(bench.log,cursor)
        suite['requests'].append(r);save_json(out/(r['name']+'.json'),r);save_json(out/'integration.json',suite)
    suite['checks']['logprobs_returned_finite']=all(r.get('logprobs_finite_and_present') for r in suite['requests'] if r['category']=='integration-logprobs')
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        final=bench.status()
        if final.get('in_flight')==0:break
        time.sleep(0.2)
    suite['final_activity']=final;suite['checks']['server_idle_after_suite']=final.get('in_flight')==0
    suite['end_epoch']=time.time();save_json(out/'integration.json',suite)
    print(json.dumps({'checks':suite['checks'],'logprobs_scope':scope,'request_count':len(suite['requests'])},ensure_ascii=False,indent=2))
    if not all(suite['checks'].values()):
        raise SystemExit('One or more integration checks failed or were not exercised; inspect integration.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--out',required=True)
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--concurrency-repeats',type=int,default=2)
    p.add_argument('--concurrent-tokens',type=int,default=256)
    p.add_argument('--cancel-after-chunks',type=int,default=16)
    p.add_argument('--logprobs-tokens',type=int)
    p.add_argument('--compare',nargs=2,metavar=('BASELINE_DIR','CANDIDATE_DIR'))
    a=p.parse_args()
    if a.compare:
        compare(a)
    else:
        if not a.config:p.error('--config required for live client suite')
        run_suite(a)


if __name__=='__main__':
    main()
