#!/usr/bin/env python3
"""Offline evaluation of the frozen, same-binary decode-head ABBA.

Only reads evidence and writes evaluation.json/.txt under --out. Does not contact
the machine, start a service, touch configuration, or deploy anything.
"""
import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ERROR_RE=re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|(?:error|fail)[^\n]{0,100}CUDA|illegal memory access|device.side assert|CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits',re.I)


def load(path,default=None):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def stats(values):
    v=[float(x) for x in values if isinstance(x,(int,float))]
    return {'n':len(v),'median':statistics.median(v),'mean':statistics.mean(v),'min':min(v),'max':max(v),'values':v} if v else {'n':0}


def change(a,b):
    return 100*(b/a-1) if a is not None and b is not None and a>0 else None


def metric(record,name):
    engine=record.get('matched_engine_record') or {}
    if name.startswith('engine.'):
        return engine.get(name.split('.')[1])
    return record.get(name)


def compare_metric(samples,name):
    off=[(tag,metric(r,name)) for tag,arm,r in samples if arm=='off']
    on=[(tag,metric(r,name)) for tag,arm,r in samples if arm=='on']
    os,ns=stats([v for _,v in off]),stats([v for _,v in on])
    values={tag:v for tag,v in off+on}
    pairs=[]
    for atag,btag in (('01-off','02-on'),('04-off','03-on')):
        x,y=values.get(atag),values.get(btag)
        if x is not None and y is not None:
            pairs.append({'off':atag,'on':btag,'off_value':x,'on_value':y,'change_pct':change(x,y)})
    return {'off':os,'on':ns,'on_mean_vs_off_mean_pct':change(os.get('mean'),ns.get('mean')),
            'on_median_vs_off_median_pct':change(os.get('median'),ns.get('median')),
            'abba_adjacent_pairs':pairs,
            'baseline_first_to_last_pct':change(values.get('01-off'),values.get('04-off'))}


def output_key(r):
    return (r.get('output_text',''),r.get('reasoning_text',''))


def request_checks(samples):
    reqs={r.get('request_sha256') for _,_,r in samples}
    outputs={output_key(r) for _,_,r in samples}
    tokens={r.get('usage',{}).get('completion_tokens') for _,_,r in samples}
    prompt_tokens={r.get('usage',{}).get('prompt_tokens') for _,_,r in samples}
    issues=[]
    for tag,_,r in samples:
        ck=r.get('checks',{})
        for field in ('request_ok','isolated_activity','engine_full_request_record_unique'):
            if not ck.get(field):issues.append({'arm':tag,'check':field,'value':ck.get(field)})
        for needed,field in (('cold_required','cold_zero_reuse'),('warm_required','warm_reused_prefix'),('append_required','append_reused_prefix')):
            if ck.get(needed) and not ck.get(field):issues.append({'arm':tag,'check':field,'value':ck.get(field)})
        if ck.get('decode_head_route_applicable',True) and not ck.get('decode_head_route_confirmed'):
            issues.append({'arm':tag,'check':'decode_head_route_confirmed','value':ck.get('decode_head_route_confirmed')})
        if r.get('error'):issues.append({'arm':tag,'api_error':r['error']})
        if ck.get('cuda_error_lines'):issues.append({'arm':tag,'check':'cuda_error_lines','value':ck['cuda_error_lines']})
        if hashlib.sha256(r.get('output_text','').encode()).hexdigest()!=r.get('output_sha256'):
            issues.append({'arm':tag,'check':'saved_output_sha256_matches_text','value':False})
    correctness_issues=[q for q in issues if q.get('check')!='engine_full_request_record_unique']
    correspondence=len(samples)==4 and len(reqs)==1 and None not in reqs and len(outputs)==1 and len(tokens)==1 and len(prompt_tokens)==1
    return {'all_valid_for_correctness':correspondence and not correctness_issues,
            'engine_record_limitations':[q for q in issues if q.get('check')=='engine_full_request_record_unique'],
            'four_arm_samples':len(samples)==4,'same_request':len(reqs)==1 and None not in reqs,
            'same_text_and_reasoning':len(outputs)==1,'same_completion_tokens':len(tokens)==1,
            'same_prompt_tokens':len(prompt_tokens)==1,'issues':issues,
            'all_valid_for_performance':correspondence and not issues}


def aggregate(names,case):
    group={'cases':names,'all_valid':bool(names) and all(case[n]['checks']['all_valid_for_performance'] for n in names),'metrics':{}}
    for field in ('engine.decode_tps','decode_tps_usage_minus_one_estimate','engine.prefill_ms','engine.prefill_tps','ttft_s'):
        pooled={'off':[],'on':[]};changes=[];adjacent=[];drift=[]
        for n in names:
            m=case[n]['metrics'][field]
            for arm in pooled:pooled[arm]+=m[arm].get('values',[])
            if m['on_mean_vs_off_mean_pct'] is not None:changes.append(m['on_mean_vs_off_mean_pct'])
            adjacent += [q['change_pct'] for q in m['abba_adjacent_pairs'] if q['change_pct'] is not None]
            if m['baseline_first_to_last_pct'] is not None:drift.append(m['baseline_first_to_last_pct'])
        os,ns=stats(pooled['off']),stats(pooled['on'])
        group['metrics'][field]={'off':os,'on':ns,'pooled_median_change_pct':change(os.get('median'),ns.get('median')),
            'matched_case_change_pct':stats(changes),'adjacent_pair_change_pct':stats(adjacent),'baseline_first_to_last_pct':stats(drift),
            'cases_improved_for_throughput':sum(x>0 for x in changes),'cases_lower_latency':sum(x<0 for x in changes)}
    return group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--plan',default=str(Path(__file__).with_name('evaluation-plan.json')))
    p.add_argument('--parity-log',help='Actual low-level validation log for the final candidate build')
    p.add_argument('--out',help='defaults to --root')
    a=p.parse_args();root=Path(a.root);out=Path(a.out or a.root);out.mkdir(parents=True,exist_ok=True)
    plan=load(Path(a.plan),{});manifest=load(root/'manifest.json',{})
    arms=[];byname=defaultdict(list);runtime_errors=[];integrations=[]
    for ad in sorted(root.glob('[0-9][0-9]-*')):
        meta=load(ad/'metadata.json',{});records=load(ad/'results.json',[])
        if not meta:continue
        arm={'tag':ad.name,'arm':meta.get('arm'),'meta':meta,'complete':load(ad/'complete.json'),
             'records':records,'integration':load(ad/'integration'/'integration.json')}
        arms.append(arm)
        for r in records:
            if r.get('category')!='warmup':byname[r['name']].append((ad.name,meta.get('arm'),r))
            if r.get('error'):runtime_errors.append({'arm':ad.name,'request':r['name'],'error':r['error']})
        for logfile in ('engine.log','server.log','client.log','integration.log'):
            path=ad/logfile
            if path.exists():
                for i,line in enumerate(path.read_text(errors='replace').splitlines(),1):
                    if ERROR_RE.search(line) or 'Traceback (most recent call last)' in line:
                        runtime_errors.append({'arm':ad.name,'file':logfile,'line':i,'text':line[:1000]})
        if arm['integration']:
            ig=arm['integration'];failed={k:v for k,v in ig.get('checks',{}).items() if v is not True}
            integrations.append({'arm':ad.name,'checks':ig.get('checks',{}),'failed_checks':failed,
                                 'request_count':len(ig.get('requests',[])),'logprobs_scope':ig.get('logprobs_scope'),
                                 'cohort_overlap_s':[c.get('stream_content_overlap_s') for c in ig.get('cohorts',[])]})
            for r in ig.get('requests',[]):
                if r.get('error'):runtime_errors.append({'arm':ad.name,'integration_request':r['name'],'error':r['error']})
    case={}
    for name,samples in sorted(byname.items()):
        category=samples[0][2].get('category')
        case[name]={'category':category,'checks':request_checks(samples),'metrics':{},
                    'prompt_tokens':{tag:r.get('usage',{}).get('prompt_tokens') for tag,_,r in samples},
                    'completion_tokens':{tag:r.get('usage',{}).get('completion_tokens') for tag,_,r in samples},
                    'head_windows':{tag:r.get('decode_head_offload_windows') for tag,_,r in samples},
                    'expected_head_route':{tag:r.get('checks',{}).get('decode_head_expected_route') for tag,_,r in samples}}
        for field in ('engine.prefill_ms','engine.prefill_tps','engine.decode_tps','decode_tps_usage_minus_one_estimate','ttft_s'):
            case[name]['metrics'][field]=compare_metric(samples,field)
    report={'plan':plan,'plan_sha256':hashlib.sha256(Path(a.plan).read_bytes()).hexdigest() if Path(a.plan).exists() else None,
            'root':str(root.resolve()),'manifest':manifest,'cases':case,'workloads':{},'integration_arms':integrations,
            'runtime_errors':runtime_errors,'definitions':{
                'primary_decode_gain':'For each fixed cold/warm fixture, mean of two on-arm engine rates divided by mean of two off-arm rates; then median of the six case gains for code and separately prose.',
                'most_cases':'Strictly more than half of the corresponding fixed cases improve.',
                'adjacent_pairs':'Also report B1/A1 and B2/A2 ratios, preserving the symmetric ABBA time pairing.',
                'cold_prefill_gate':'For each workload, median of six on-arm read-ms measurements divided by median of six off-arm read-ms measurements must be <=1.02.',
                'warm_append':'Report each restoration/append case separately, including engine read time and TTFT; materiality/rerun decisions require review rather than an invented threshold.',
                'client_decode':'First-to-last SSE estimate retained separately from engine decode; it is not used as a substitute for a missing engine record.'}}
    repeats=manifest.get('repeats',3)
    for kind in ('code','prose'):
        workload={'groups':{}}
        for phase in ('cold','warm','combined'):
            names=[n for n,c in case.items() if c['category']==phase+'-'+kind or phase=='combined' and c['category'] in ('cold-'+kind,'warm-'+kind)]
            group=aggregate(names,case)
            workload['groups'][phase]=group
        combined=workload['groups']['combined'];decode=combined['metrics']['engine.decode_tps']
        cold=workload['groups']['cold'];pf=cold['metrics']['engine.prefill_ms']
        n=len(combined['cases']);gain=decode['matched_case_change_pct'].get('median');reg=pf['pooled_median_change_pct']
        workload['gates']={
            'six_expected_cold_and_warm_cases':n==2*repeats,
            'matched_cases_valid':combined['all_valid'],
            'decode_gain_at_least_1pct':gain is not None and gain>=1,
            'majority_cases_improve':decode['cases_improved_for_throughput']>n/2 if n else False,
            'cold_prefill_median_regression_at_most_2pct':reg is not None and reg<=2,
        }
        report['workloads'][kind]=workload
    report['warm_append_review']=[{'case':name,'valid':c['checks']['all_valid_for_performance'],
          'engine_read_ms':c['metrics']['engine.prefill_ms'],'ttft_s':c['metrics']['ttft_s'],
          'engine_decode_tps':c['metrics']['engine.decode_tps']}
          for name,c in case.items() if c['category']=='append' or c['category'].startswith('warm-')]
    report['append_aggregate']=aggregate([name for name,c in case.items() if c['category']=='append'],case)
    parity={'provided':bool(a.parity_log),'full_head_pass':False,'shard_merge_pass':False}
    if a.parity_log:
        path=Path(a.parity_log);text=path.read_text(errors='replace')
        parity.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text=text,
                      full_head_pass=bool(re.search(r'PASS: 128 head windows, \d+ bitwise logits',text)),
                      shard_merge_pass='PASS: 16 sharded windows; bitwise full logits' in text,
                      failure_marker_present=bool(re.search(r'\bFAIL(?:ED)?\b|ERROR|Traceback',text)))
    report['low_level_parity']=parity
    report['engine_record_limitations']={name:c['checks']['engine_record_limitations'] for name,c in case.items() if c['checks']['engine_record_limitations']}
    report['definitions']['correctness_timing']='A thinking request may span multiple engine calls. Exact API output/request/count parity remains required, but a missing single combined timing record only excludes it from performance metrics.'
    comparisons=[]
    for path in sorted((root/'integration-comparisons').glob('*/integration-comparison.json')):
        x=load(path,{})
        comparisons.append({'pair':path.parent.name,'all_completed_requests_exactly_match':x.get('all_completed_requests_exactly_match'),
            'differences':[v for v in x.get('comparisons',[]) if v.get('state')!='exact_match'],
            'logprobs_numeric_comparisons':[{k:v for k,v in q.items() if k in ('name','same_logprobs_exactly','logprob_value_count_baseline','logprob_value_count_candidate','max_logprob_abs_difference_by_returned_position')}
                                          for q in x.get('comparisons',[]) if q.get('name','').startswith('logprobs-')]})
    report['integration_pair_comparisons']=comparisons
    same_binary=len(arms)==4 and all(arm['meta'].get('engine_sha256')==manifest.get('same_binary_sha256') for arm in arms)
    hashes={arm['meta'].get('config_except_flag_log_sha256') for arm in arms}
    all_correct=bool(case) and all(c['checks']['all_valid_for_correctness'] for c in case.values())
    all_routes=bool(case) and all(not q for c in case.values() for q in c['checks']['issues'] if q.get('check')=='decode_head_route_confirmed')
    required_correctness={'short-greedy','sample-17','sample-29','penalties','bias','thinking'}
    appends=[c for c in case.values() if c['category']=='append']
    report['global_gates']={
        'complete_same_binary_abba':manifest.get('sequence')==['off','on','on','off'] and len(arms)==4 and all(arm['complete'] for arm in arms) and same_binary,
        'only_head_flag_and_log_config_differ':len(arms)==4 and len(hashes)==1 and None not in hashes,
        'governors_unchanged':len(arms)==4 and all(arm['meta'].get('governors_before')==manifest.get('governors_before') and arm['meta'].get('governors_after')==manifest.get('governors_before') for arm in arms),
        'all_corresponding_outputs_and_counts_identical':all_correct,
        'all_six_correctness_cases_run_in_every_arm':required_correctness.issubset(case) and all(case[name]['checks']['four_arm_samples'] for name in required_correctness if name in case),
        'all_append_cases_valid':len(appends)==repeats and all(c['checks']['all_valid_for_performance'] for c in appends),
        'intended_head_routes_confirmed':all_routes,
        'low_level_full_and_sharded_parity_pass':parity['full_head_pass'] and parity['shard_merge_pass'] and not parity.get('failure_marker_present',False),
        'no_observed_cuda_nonfinite_api_failures':not runtime_errors,
        'all_four_integration_suites_pass':len(integrations)==4 and all(not r['failed_checks'] and len(r['checks'])==7 and r['request_count']==11 for r in integrations),
        'all_six_integration_pair_comparisons_exact':len(comparisons)==6 and all(c['all_completed_requests_exactly_match'] for c in comparisons),
        'original_service_config_unchanged':manifest.get('original_config_unchanged') is True,
        'test_gpu_processes_released':manifest.get('remaining_gpu_pids')==[],
    }
    failed=[key for key,val in report['global_gates'].items() if not val]
    failed += [kind+'.'+key for kind,w in report['workloads'].items() for key,val in w['gates'].items() if not val]
    report['failed_or_missing_gates']=failed
    report['predeclared_quantitative_and_correctness_gates_pass']=not failed
    report['decision']='Do not deploy based on this run; resolve failed/missing gates.' if failed else 'Measured gates pass. Review warm/append read time and TTFT separately before any deployment decision.'
    report['automatic_deployment']=False
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['Final same-binary ABBA evaluation',report['decision'],'Failed/missing: '+(', '.join(failed) or 'none'),'',
           'workload\tphase\tengine_off_median\tengine_on_median\tengine_case_gain_median_pct\tclient_off_median\tclient_on_median\tclient_case_gain_median_pct\tread_ms_off_median\tread_ms_on_median\tread_ms_change_pct']
    for kind,w in report['workloads'].items():
        for phase,g in w['groups'].items():
            e=g['metrics']['engine.decode_tps'];c=g['metrics']['decode_tps_usage_minus_one_estimate'];pf=g['metrics']['engine.prefill_ms']
            lines.append('\t'.join(str(v) for v in (kind,phase,e['off'].get('median'),e['on'].get('median'),e['matched_case_change_pct'].get('median'),
                c['off'].get('median'),c['on'].get('median'),c['matched_case_change_pct'].get('median'),pf['off'].get('median'),pf['on'].get('median'),pf['pooled_median_change_pct'])))
    lines += ['','Cold prefill by fixed case:','case\toff_read_ms\ton_read_ms\tmean_change_pct\toff_ttft_s\ton_ttft_s']
    for name,c in case.items():
        if not c['category'].startswith('cold-'):continue
        pf=c['metrics']['engine.prefill_ms'];tt=c['metrics']['ttft_s']
        lines.append('\t'.join(str(v) for v in (name,pf['off'].get('values'),pf['on'].get('values'),pf['on_mean_vs_off_mean_pct'],tt['off'].get('values'),tt['on'].get('values'))))
    lines += ['','Warm/append latency review:','case\toff_read_ms\ton_read_ms\tmean_change_pct\toff_ttft_s\ton_ttft_s']
    for r in report['warm_append_review']:
        pf=r['engine_read_ms'];tt=r['ttft_s']
        lines.append('\t'.join(str(v) for v in (r['case'],pf['off'].get('values'),pf['on'].get('values'),pf['on_mean_vs_off_mean_pct'],tt['off'].get('values'),tt['on'].get('values'))))
    lines += ['','Integration comparisons:']+[r['pair']+': exact='+str(r['all_completed_requests_exactly_match'])+', differences='+str(len(r['differences'])) for r in comparisons]
    rendered='\n'.join(lines)+'\n';(out/'evaluation.txt').write_text(rendered);print(rendered)


if __name__=='__main__':main()
