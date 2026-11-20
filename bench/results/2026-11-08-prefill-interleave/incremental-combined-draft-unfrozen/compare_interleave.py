#!/usr/bin/env python3
"""Offline exact finite-output comparison with descriptive SSE silence changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
COMPARATORS={'correctness':('correctness.py','comparison.json'),'budget':('budget_correctness.py','budget-comparison.json'),
             'c8':('c8_check.py','c8-comparison.json'),'long':('long_concurrency_check.py','comparison.json'),
             'prefill':('prefill_interleave_check.py','comparison.json')}


def read(path):return json.loads(path.read_text())
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def change(a,b):
    return {'off':a,'on':b,'on_relative_change_pct':(b/a-1)*100 if type(a) in (int,float) and type(b) in (int,float) and a else None}


def records_equal(left,right,expected=None):
    maps=[{r['name']:r for r in values} for values in (left,right)]
    checks={'same_unique_request_set':set(maps[0])==set(maps[1]) and len(maps[0])==len(left) and len(maps[1])==len(right)}
    if expected is not None:checks['expected_request_count']=len(left)==len(right)==expected
    rows=[]
    for name in sorted(set(maps[0])|set(maps[1])):
        a,b=(m.get(name) for m in maps);gates={'present_in_both':a is not None and b is not None}
        if a and b:
            for key in ('request_sha256','output_sha256','channel_output_sha256','finish_reason'):
                gates[key+'_equal']=a.get(key) is not None and a[key]==b.get(key)
            counts=[{k:r.get('usage',{}).get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')} for r in (a,b)]
            gates['logical_usage_equal']=counts[0]==counts[1] and all(type(x) is int for c in counts for x in c.values())
            gates['both_normal_complete']=all(r.get('successful') and r.get('saw_done') and r.get('error') is None for r in (a,b))
        rows.append({'name':name,'checks':gates,'passed':all(gates.values())})
    return {'checks':checks,'pairs':rows,'all_exact':all(checks.values()) and bool(rows) and all(x['passed'] for x in rows)}


def finite(left,right):
    reports=[read(p/'results.json') for p in (left,right)]
    plans=[read(p/'manifest.json') for p in (left,right)]
    checks={'both_runs_qualified':all(r.get('all_checks_passed') for r in reports),
            'both_visible_coverages':all(r.get('visible_phase_coverage',{}).get('all_checks_passed') is True for r in reports),
            'both_manifests_complete':all(p.get('completed') and p.get('all_checks_passed') for p in plans)}
    for key in ('case','plan','calibration_sha256','fixture_generator_sha256','client_sha256','stream_client_sha256','stall_metrics_sha256','fixtures','preflight_counts','output_contract'):
        checks[key+'_equal']=plans[0].get(key) is not None and plans[0][key]==plans[1].get(key)
    exact=records_equal(*[r['records'] for r in reports],expected=8 if plans[0]['case']=='incremental' else (6 if plans[0]['case']=='primary' else 5))
    metrics={'whole_case_tokens_per_s':change(*[r['whole_case_api_rate']['tokens_per_s'] for r in reports]),'phases':{},'injections':[]}
    for phase in ('baseline','mixed','recovery'):
        metrics['phases'][phase]={}
        for channel in ('output','visible_content','reasoning','tool'):
            rows=[r['phases'][phase][channel] for r in reports]
            fields=('max_lane_clipped_silence_s','any_lane_stalled_over_1s_duration_s','all_lanes_stalled_over_1s_duration_s',
                    'any_lane_stalled_over_5s_duration_s','all_lanes_stalled_over_5s_duration_s','aggregate_output_bytes_per_s','jain_byte_progress_fairness')
            values={k:change(*[row[k] for row in rows]) for k in fields}
            values['lanes']=[{'lane':i,**{k:change(*[row['lanes'][i]['raw_gaps_ending_in_phase'][k] for row in rows])
                             for k in ('p95_s','p99_s','max_s')}} for i in range(4)]
            metrics['phases'][phase][channel]=values
    injected=[{r['name']:r for r in report['injections']} for report in reports]
    checks['same_injection_set']=set(injected[0])==set(injected[1])
    for name in sorted(set(injected[0])&set(injected[1])):
        a,b=(m[name] for m in injected)
        metrics['injections'].append({'name':name,'actual_prompt_tokens':a['actual_prompt_tokens'],
             'ttft_first_output_s':change(a['ttft_first_output_s'],b['ttft_first_output_s']),
             'native_prompt_ms':change(*[r['engine_timings'].get('prompt_ms') for r in (a,b)]),
             'max_resident_admission_silence_s':change(*[r['resident_admission_span']['max_lane_clipped_silence_s'] for r in (a,b)])})
    if plans[0]['case']=='incremental':
        named=[{r['name']:r for r in report['records']} for report in reports]
        for name in sorted(set(named[0])|set(named[1])):
            rows=[m.get(name,{}) for m in named]
            checks[name+'_cache_metadata_equal']=all(type(r.get('cached_tokens')) is int for r in rows) and rows[0].get('cached_tokens')==rows[1].get('cached_tokens') and all(rows[0].get('native_timings',{}).get(k)==rows[1].get('native_timings',{}).get(k) for k in ('cache_n','prompt_n'))
        metrics['post_seed_finite_tokens_per_s']=change(*[r['post_seed_finite_api_rate']['tokens_per_s'] for r in reports])
    return {'all_checks_passed':all(checks.values()) and exact['all_exact'],'checks':checks,'exact_outputs':exact,'metrics':metrics,
            'scope':'Matched finite normal requests with fixed bodies/max budgets and exact actual output hashes, finish reasons and three logical usage counts, including natural EOS. No unequal-work speedup comparison. Silence is SSE arrival gap, not token TBT. prompt_ms includes engine prompt stage. Whole-case rate includes admission and final drain. No automatic deployment/performance pass.'}


def combined_identity(metas,manifests):
    return {'explicit_combined_profile':all(m.get('experiment_profile')=='concurrent-mtp-and-park-overwrite-both-off-vs-both-on' for m in manifests),
            'park_overwrite_flag_off_on':[m.get('config',{}).get('env',{}).get('STRATA_PARK_OVERWRITE') for m in metas]==['0','1'],
            'combined_overlap_flag_off_on':[m.get('config',{}).get('env',{}).get('STRATA_PIPED_CONCURRENT_MTP') for m in metas]==['0','1']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('--out exists')
    arms=[a.baseline,a.candidate];metas=[read(x/'metadata.json') for x in arms];manifests=[read(x.parent/'manifest.json') for x in arms]
    suites=[set(m['suite_returncodes']) for m in metas]
    checks={'off_on': [m['arm'] for m in metas]==['off','on'],'same_suites':suites[0]==suites[1],
            'same_planned_suite_sequence':bool(manifests[0].get('suites')) and manifests[0].get('suites')==manifests[1].get('suites'),
            'all_suites_succeeded':all(v==0 for m in metas for v in m['suite_returncodes'].values()),
            'only_mode_config_differs':metas[0]['config_except_variables_log_key_sha256']==metas[1]['config_except_variables_log_key_sha256'],
            'overlap_flag_off_on':[m['config']['env'].get('STRATA_PIPED_CONCURRENT_MTP') for m in metas]==['0','1'],
            'fixed_existing_paths':all(m['config']['env'].get(k)==v for m in metas for k,v in
                {'STRATA_PIPED_TARGET_HEAD':'1','STRATA_PIPED_ASYNC_SAMPLE':'0','STRATA_PIPED_MTP_HEAD':'1','STRATA_BATCH_MTP':'1','STRATA_PIPED_MTP':'1'}.items())}
    checks.update(combined_identity(metas,manifests))
    checks['both_runs_finished']=all(type(m.get('finished_epoch')) in (int,float) for m in metas) and all(type(m.get('ended_epoch')) in (int,float) and not m.get('failure') for m in manifests)
    checks['all_planned_suites_completed']=all(set(meta['suite_returncodes'])==set(man.get('suites',[])) and bool(meta['suite_returncodes']) for meta,man in zip(metas,manifests))
    checks['registered_completed_arms']=all(any(row.get('tag')==arm.name and row.get('arm')==meta.get('arm') and row.get('suite_returncodes')==meta['suite_returncodes'] for row in man.get('arms',[])) for arm,meta,man in zip(arms,metas,manifests))
    for key in ('same_binary_sha256','adjacent_library_sha256','base_config_sha256','server_sha256','harness_sha256'):
        checks[key+'_equal']=manifests[0].get(key) is not None and manifests[0][key]==manifests[1].get(key)
    if suites[0]&{'primary','mixed100k','incremental'}:checks['calibration_sha256_equal']=manifests[0].get('calibration_sha256') is not None and manifests[0]['calibration_sha256']==manifests[1].get('calibration_sha256')
    a.out.mkdir(parents=True);report={'identity_checks':checks,'suites':{},'suite_checks':{},'all_checks_passed':False,'performance_requires_review':True}
    if all(checks.values()):
        for suite in sorted(suites[0]):
            if suite in ('primary','mixed100k','incremental'):
                result=finite(*[x/suite for x in arms]);passed=result['all_checks_passed']
            elif suite in ('admission','byield'):
                import importlib
                module=importlib.import_module('byield_smoke' if suite=='byield' else 'admission_smoke')
                result=module.compare(*[x/suite for x in arms]);passed=result['all_checks_passed']
            else:
                script,filename=COMPARATORS[suite];destination=a.out/suite
                with (a.out/(suite+'-comparison.log')).open('w') as handle:
                    child=subprocess.run([sys.executable,str(HERE/script),'--compare',*[str(x/suite) for x in arms],'--out',str(destination)],stdout=handle,stderr=subprocess.STDOUT,timeout=120)
                result=read(destination/filename) if (destination/filename).is_file() else {'missing_comparison':True}
                passed=child.returncode==0 and (result.get('comparison_eligible') and result.get('all_greedy_outputs_identical') if suite=='prefill' else result.get('all_checks_passed'))
            report['suites'][suite]=result;report['suite_checks'][suite]=bool(passed)
    report['all_checks_passed']=all(checks.values()) and len(report['suite_checks'])==len(suites[0]) and all(report['suite_checks'].values())
    save(a.out/'comparison.json',report);print(json.dumps({'all_checks_passed':report['all_checks_passed'],'suite_checks':report['suite_checks'],'performance_requires_review':True}))
    return 0 if report['all_checks_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
