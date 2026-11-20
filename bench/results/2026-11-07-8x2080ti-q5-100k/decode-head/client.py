#!/usr/bin/env python3
"""Single-request correctness and 100k performance suite; never restarts servers.

All inputs are synthetic and fixed across arms. A shared reference file pins the
append's assistant message to the first arm, keeping later inputs identical even
if a candidate produces a different answer. Changes in cache reuse are recorded.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from bench_client import Bench, TASKS, filler, save_json, read_log

ERROR_RE = re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|(?:error|fail)[^\n]{0,100}CUDA|illegal memory access|device.side assert|CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits', re.I)
HEAD_RE = re.compile(r'strata decode head: CUDA(\d+), (\d+) PW2 windows offloaded in this request')


def head_devices(value):
    value=str(value).strip()
    if value in ('','-1'):
        return []
    devices=[int(part.strip()) for part in value.split(',')]
    if len(devices) not in (1,2) or min(devices)<0 or len(set(devices))!=len(devices):
        raise ValueError('STRATA_DECODE_HEAD_GPU must be -1, D, or distinct D1,D2')
    return devices


def engine_record(r):
    pn, cn = r['usage'].get('prompt_tokens'), r['usage'].get('completion_tokens')
    full = [e for e in r['engine_prompt_records'] if e['prompt_tokens']==pn and e['generated_tokens']==cn]
    if len(full)==1:
        return full[0]
    return None


def check(bench, r, cold=False, append=False, warm=False):
    # The engine normally logs DONE before SSE closes. Briefly re-read only this
    # isolated request's log if timing records raced the final flush.
    before, after = r['activity_before'], r['activity_after']
    delta = after.get('requests',0)-before.get('requests',0)
    e = engine_record(r)
    cached = (r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
    head=[{'device':int(m[1]),'windows':int(m[2])} for line in r['engine_lines'] if (m:=HEAD_RE.search(line))]
    devices=head_devices(bench.cfg.get('env',{}).get('STRATA_DECODE_HEAD_GPU','-1'))
    head_enabled=bool(devices)
    settings=r.get('request_settings',{})
    sampled_shards=len(devices)>1 and float(settings.get('temperature',0))>0
    fallback=sampled_shards or bool(settings.get('logit_bias')) or settings.get('presence_penalty',0)!=0 or settings.get('frequency_penalty',0)!=0 or settings.get('repetition_penalty',1)!=1
    head_windows=sum(h['windows'] for h in head)
    route_expected='fallback' if head_enabled and fallback else 'offload' if head_enabled else 'disabled'
    early_eos=(r['category']=='correctness' and r['finish_reason']=='stop' and r['usage'].get('completion_tokens',0)<=1)
    r['decode_head_offload_records']=head
    r['decode_head_offload_windows']=head_windows
    r['decode_head_devices']=devices
    r['checks'] = {
        'request_ok': r['error'] is None and r['saw_done'] and bool(r['usage']) and (r['ttft_s'] is not None or early_eos),
        'early_eos_correctness_case':early_eos,
        'isolated_activity': before.get('in_flight')==0 and after.get('in_flight')==0 and delta==1,
        'request_counter_delta': delta,
        'engine_full_request_record_unique': e is not None,
        'api_cached_tokens': cached,
        'cold_required': cold,
        'cold_zero_reuse': e is not None and e['reused_tokens']==0 and cached in (None,0) if cold else None,
        'append_required': append,
        'append_reused_prefix': e is not None and e['reused_tokens']>90000 and (cached is None or cached>90000) if append else None,
        'warm_required': warm,
        'warm_reused_prefix': e is not None and e['reused_tokens']>90000 and (cached is None or cached>90000) if warm else None,
        'cuda_error_lines': [ln for ln in r['engine_lines'] if ERROR_RE.search(ln)],
        'decode_head_expected_route':route_expected,
        'decode_head_sharded_sample_fallback':sampled_shards,
        'decode_head_route_applicable':not early_eos,
        'decode_head_route_confirmed':None if early_eos else (head_windows>0 and all(h['device']==devices[0] for h in head)) if route_expected=='offload' else head_windows==0,
    }
    r['matched_engine_record'] = e
    save_json(bench.out/(r['name']+'.json'),r)
    save_json(bench.out/'results.json',bench.records)
    if not r['checks']['request_ok'] or r['checks']['cuda_error_lines']:
        raise RuntimeError(f"request failed: {r['name']}; see its JSON")
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--port',type=int,default=8091)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output-tokens',type=int,default=768)
    p.add_argument('--suites',default='correctness,performance')
    a=p.parse_args()
    a.suites=a.suites.split(',')
    if set(a.suites)-{'correctness','performance'}:
        p.error('unknown suite')
    bench=Bench(a)
    reference=Path(a.reference)
    reference.parent.mkdir(parents=True,exist_ok=True)
    refs=json.loads(reference.read_text()) if reference.exists() else {}
    meta={'fixture_version':'decode-head-2-sharded','created_epoch':time.time(),'repeats':a.repeats,
          'output_tokens':a.output_tokens,'nominal_prompt_tokens':100000,
          'notes':['Exact token counts come from usage and engine logs.',
                   'A fresh process per arm empties cache; each cold fixture has a different early prefix.',
                   'Append uses the canonical first-arm assistant text, preserving identical requests.',
                   'Client throughput is an SSE timing estimate; matched engine records are the primary decode measure.']}
    save_json(bench.out/'fixture-metadata.json',meta)
    # Explicit warmup is identical in every arm and excluded from speed summaries.
    for i in range(2):
        check(bench,bench.request(f'warmup-{i}',[{'role':'user','content':f'Warmup {i}. Write a Python function to compute the sum of even numbers in a list, then add tests.'}],128,'warmup'))
    if 'correctness' in a.suites:
        cases=[
            ('short-greedy',{},'Write a Python generator that yields prime numbers below n, then provide tests.'),
            ('sample-17',{'temperature':0.8,'top_p':0.95,'top_k':20,'seed':17},'Describe five unusual uses for a paper map, with a short reason for each.'),
            ('sample-29',{'temperature':0.8,'top_p':0.95,'top_k':20,'seed':29},'Describe five unusual uses for a paper map, with a short reason for each.'),
            ('penalties',{'presence_penalty':1.1,'frequency_penalty':0.3,'repetition_penalty':1.05},'Write a coherent short story about a lighthouse, using the words sea and light several times.'),
            ('bias',{'logit_bias':{'32':1.0}},'Write a Python function to merge two sorted lists without modifying either input, and explain its complexity.'),
            ('thinking',{'temperature':0.6,'top_p':0.95,'top_k':20,'seed':79,'chat_template_kwargs':{'enable_thinking':True}},'A queue allows simultaneous producers and consumers. Explain a safe approach to close it without losing items, reasoning carefully about races.'),
        ]
        for name,overrides,prompt in cases:
            check(bench,bench.request(name,[{'role':'user','content':'[Decode head correctness '+name+'] '+prompt}],192,'correctness',overrides=overrides))
    if 'performance' in a.suites:
        long=filler(100000)
        prefixes=['Amber','Quartz','Violet','Nimbus','Falcon','Cobalt','Juniper','Copper']
        for repeat in range(a.repeats):
            for kind in ('code','prose'):
                idx=repeat*2+(kind=='prose')
                prefix=prefixes[idx%len(prefixes)]+f' {idx} deterministic decode-head fixture.'
                content=prefix+' '+long+'\n\nIgnoring the text above, '+TASKS[kind]
                messages=[{'role':'user','content':content}]
                name=f'cold-{kind}-{repeat+1}'
                r=check(bench,bench.request(name,messages,a.output_tokens,'cold-'+kind),cold=True)
                # The exact same request after a complete generation exercises the
                # long-context cache restore path without fresh prefill cost.
                check(bench,bench.request(f'warm-{kind}-{repeat+1}',messages,a.output_tokens,'warm-'+kind),warm=True)
                if kind=='code':
                    if name not in refs:
                        refs[name]={'assistant':r['output_text'],'source_output_sha256':r['output_sha256'],
                                    'source_messages_sha256':r['messages_sha256']}
                        save_json(reference,refs)
                    canonical=refs[name]
                    if canonical['source_messages_sha256']!=r['messages_sha256']:
                        raise RuntimeError('Reference file belongs to a different fixture')
                    append=messages+[{'role':'assistant','content':canonical['assistant']},
                        {'role':'user','content':filler(600)+'\nExplain how to extend the LRU cache with TTL expiry. Give Python code.'}]
                    ar=check(bench,bench.request(f'append-code-{repeat+1}',append,384,'append'),append=True)
                    ar['canonical_parent_output_matches_this_arm']=canonical['source_output_sha256']==r['output_sha256']
                    save_json(bench.out/(ar['name']+'.json'),ar)
                    save_json(bench.out/'results.json',bench.records)
    # Scan the complete arm log, including startup; keep exact evidence but no key.
    logtext=bench.log.read_text(errors='replace') if bench.log else ''
    errors=[bench.safe(ln) for ln in logtext.splitlines() if ERROR_RE.search(ln)]
    save_json(bench.out/'complete.json',{'completed_epoch':time.time(),'request_count':len(bench.records),
              'cuda_error_lines':errors,'cold_cache_checks_passed':all(r['checks']['cold_zero_reuse'] for r in bench.records if r['checks']['cold_required']),
              'append_cache_checks_passed':all(r['checks']['append_reused_prefix'] for r in bench.records if r['checks']['append_required']),
              'warm_cache_checks_passed':all(r['checks']['warm_reused_prefix'] for r in bench.records if r['checks']['warm_required']),
              'all_requests_isolated':all(r['checks']['isolated_activity'] for r in bench.records)})
    if errors:
        raise RuntimeError('CUDA errors in engine log')


if __name__=='__main__':
    main()
