#!/usr/bin/env python3
"""Offline visible-content SSE silence, with strict matched-work eligibility.

No service/GPU/network operations. Reads existing artifacts and writes one new
JSON result. Never labels chunk gaps as token TBT or infers pure decode throughput.
Single-case mode is diagnostic and cannot establish A/B benefit.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise RuntimeError(message)

def distribution(values):
    values=sorted(values)
    def q(p):return values[max(0,math.ceil(p*len(values))-1)] if values else None
    return {'n':len(values),'p50_s':q(.5),'p95_s':q(.95),'p99_s':q(.99),'max_s':max(values) if values else None,'method':'nearest-rank descriptive chunk gaps'}

def interval(left,right):return {'start_epoch':left,'end_epoch':right,'duration_s':right-left}

def union_intervals(values,required=1):
    events={}
    for left,right in values:
        if right>left:events[left]=events.get(left,0)+1;events[right]=events.get(right,0)-1
    result=[];count=0;begin=None
    for when,delta in sorted(events.items()):
        old=count;count+=delta
        if old<required<=count:begin=when
        if old>=required>count:result.append((begin,when));begin=None
    return result

def visible_epochs(record):
    return sorted(c['epoch'] for c in record.get('chunks',[]) if c.get('bytes',{}).get('visible_content',0)>0)

def phase(records,begin,end):
    require(end>begin,'Phase must have positive duration')
    peers=[r for r in records if r.get('role')=='resident'];require(len(peers)==4 and {r['lane'] for r in peers}==set(range(4)),'Requires exactly four finite resident lanes')
    lanes=[];threshold_intervals={t:[] for t in (.25,1,5)};all_epochs=[]
    for record in sorted(peers,key=lambda r:r['lane']):
        times=visible_epochs(record);all_epochs.extend(times)
        raw=[(a,b) for a,b in zip(times,times[1:]) if b>begin and a<end]
        clipped=[(max(a,begin),min(b,end)) for a,b in raw if min(b,end)>max(a,begin)]
        for t in threshold_intervals:
            threshold_intervals[t].extend((max(a+t,begin),min(b,end)) for a,b in raw if min(b,end)>max(a+t,begin))
        top=sorted(raw,key=lambda ab:ab[1]-ab[0],reverse=True)[:5]
        lanes.append({'lane':record['lane'],'name':record['name'],
             'visible_first_epoch':times[0] if times else None,'visible_last_epoch':times[-1] if times else None,
             'visible_covers_phase':bool(times) and times[0]<=begin and times[-1]>=end,
             'raw_gaps_ending_in_phase':distribution([b-a for a,b in zip(times,times[1:]) if begin<=b<end]),
             'phase_clipped_silence':distribution([b-a for a,b in clipped]),
             'full_gaps_touching_phase_max_s':max((b-a for a,b in raw),default=None),
             'top_touching_gaps':[{**interval(a,b),'clipped_in_phase_s':min(b,end)-max(a,begin)} for a,b in top]})
    combined=sorted(set(all_epochs));raw_collective=[(a,b) for a,b in zip(combined,combined[1:]) if b>begin and a<end]
    clipped_collective=[(max(a,begin),min(b,end)) for a,b in raw_collective if min(b,end)>max(a,begin)]
    collective_top=sorted(clipped_collective,key=lambda ab:ab[1]-ab[0],reverse=True)[:10]
    thresholds={}
    for t,values in threshold_intervals.items():
        any_lane=union_intervals(values);all_lanes=union_intervals(values,4)
        thresholds[str(t)]={'any_lane_stalled_total_s':sum(b-a for a,b in any_lane),'all_lanes_stalled_total_s':sum(b-a for a,b in all_lanes),
                            'all_lanes_stalled_spans':[interval(a,b) for a,b in all_lanes]}
    return {'begin_epoch':begin,'end_epoch':end,'duration_s':end-begin,'lanes':lanes,
            'all_four_visible_cover_phase':all(row['visible_covers_phase'] for row in lanes),
            'max_lane_phase_clipped_silence_s':max((r['phase_clipped_silence']['max_s'] or 0 for r in lanes),default=None),
            'collective_no_visible_event_gaps':distribution([b-a for a,b in clipped_collective]),
            'collective_top_spans':[interval(a,b) for a,b in collective_top],
            'stalled_after_threshold_s':thresholds,
            'scope':'Only actual SSE deltas with positive visible_content bytes. Collective gaps contain no visible event from any peer. Threshold stall time starts threshold seconds after each peer last visible event. No token timestamp inference.'}

def summarize_case(directory):
    path=directory/'results.json';result=read(path);records=result['records'];phases={}
    for name,channels in result.get('phases',{}).items():
        row=channels.get('visible_content',channels.get('output',channels));phases[name]=phase(records,row['begin_epoch'],row['end_epoch'])
    if not phases:
        peers=[r for r in records if r.get('role')=='resident'];events=[visible_epochs(r) for r in peers]
        if len(events)==4 and all(events):
            begin,end=max(t[0] for t in events),min(t[-1] for t in events)
            if end>begin:phases['common_visible_window']=phase(records,begin,end)
    admissions=[]
    for record in records:
        if record.get('role')=='resident':continue
        first=record.get('first_visible_content_s')
        if first is not None and first>0:
            admissions.append({'name':record['name'],'actual_prompt_tokens':record.get('actual_prompt_tokens'),
                'visible_ttft_s':first,'native_prompt_ms':record.get('native_timings',{}).get('prompt_ms'),
                'peers_during_request_to_first_visible':phase(records,record['start_epoch'],record['start_epoch']+first)})
    rows=[{'name':r['name'],'role':r['role'],'finish_reason':r.get('finish_reason'),
           'actual_completion_tokens':r.get('usage',{}).get('completion_tokens'),
           'actual_prompt_tokens':r.get('usage',{}).get('prompt_tokens')} for r in records]
    return {'source':str(directory.resolve()),'source_results_sha256':sha(path),'original_case_qualified':result.get('all_checks_passed') is True,
            'original_checks':result.get('checks',{}),'requests':rows,'phases':phases,'admissions':admissions,
            'whole_case_api_rate':result.get('whole_case_api_rate'),'timing_scope':'No output text retained; bytes/epochs only. Whole-case API rate, when present, includes all admissions and final drain.'}

def identity_and_work(off,on,suite):
    metas=[read(p/'metadata.json') for p in (off,on)];manifests=[read(p.parent/'manifest.json') for p in (off,on)]
    reports=[read(p/suite/'results.json') for p in (off,on)];plans=[read(p/suite/'manifest.json') for p in (off,on)]
    checks={'off_on':[m.get('arm') for m in metas]==['off','on'],
            'same_planned_suite_sequence':bool(manifests[0].get('suites')) and manifests[0].get('suites')==manifests[1].get('suites'),
            'same_config_except_mode':metas[0].get('config_except_variables_log_key_sha256') is not None and metas[0]['config_except_variables_log_key_sha256']==metas[1].get('config_except_variables_log_key_sha256'),
            'flag_off_on':[m.get('config',{}).get('env',{}).get('STRATA_PIPED_CONCURRENT_MTP') for m in metas]==['0','1'],
            'both_cases_qualified':all(r.get('all_checks_passed') for r in reports),
            'both_case_manifests_complete':all(p.get('completed') and p.get('all_checks_passed') for p in plans),
            'both_case_returncodes_zero':all(m.get('suite_returncodes',{}).get(suite)==0 for m in metas)}
    for key in ('same_binary_sha256','adjacent_library_sha256','base_config_sha256','server_sha256','harness_sha256','calibration_sha256'):
        checks[key+'_equal']=manifests[0].get(key) is not None and manifests[0][key]==manifests[1].get(key)
    for key in ('case','plan','calibration_sha256','fixture_generator_sha256','client_sha256','stream_client_sha256','stall_metrics_sha256','fixtures','preflight_counts','output_contract'):
        checks['case_'+key+'_equal']=plans[0].get(key) is not None and plans[0][key]==plans[1].get(key)
    maps=[{r['name']:r for r in report['records']} for report in reports]
    checks['same_unique_request_set']=set(maps[0])==set(maps[1]) and all(len(m)==len(r['records']) for m,r in zip(maps,reports))
    expected=6 if suite=='primary' else 5;checks['expected_request_count']=all(len(m)==expected for m in maps)
    for name in sorted(set(maps[0])|set(maps[1])):
        a,b=(m.get(name,{}) for m in maps)
        for key in ('request_sha256','output_sha256','channel_output_sha256','finish_reason'):
            checks[name+'_'+key]=a.get(key) is not None and a[key]==b.get(key)
        usages=[{k:r.get('usage',{}).get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')} for r in (a,b)]
        checks[name+'_actual_usage']=usages[0]==usages[1] and all(type(v) is int for u in usages for v in u.values())
        checks[name+'_normal_and_actual_count_valid']=all(normal_actual_output(r) for r in (a,b))
    checks['expected_phase_sets']=all(set(r.get('phases',{}))=={'baseline','mixed','recovery'} for r in reports)
    completion={'both_metadata_finished':all(type(m.get('finished_epoch')) in (int,float) for m in metas),
                'both_planned_suite_sets_completed':all(set(m.get('suite_returncodes',{}))==set(man.get('suites',[])) and all(v==0 for v in m.get('suite_returncodes',{}).values()) for m,man in zip(metas,manifests)),
                'both_arms_registered_complete':all(any(row.get('tag')==p.name and row.get('arm')==m.get('arm') and row.get('suite_returncodes')==m.get('suite_returncodes') for row in man.get('arms',[])) for p,m,man in zip((off,on),metas,manifests)),
                'no_parent_run_failure':all(not m.get('failure') for m in manifests),
                'parent_runs_ended':all(type(m.get('ended_epoch')) in (int,float) for m in manifests)}
    return {'case_pair_eligible':all(checks.values()),'case_identity_and_actual_work_checks':checks,
            'full_arm_completion_checks':completion,'full_runs_complete':all(completion.values()),
            'scope':'Case-work eligibility is separate from complete-run/deployment qualification. Missing, failed or partial suites cannot authorize deployment.'}

def normal_actual_output(record):
    usage=record.get('usage',{});values=[usage.get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')]
    if not all(type(v) is int for v in values):return False
    prompt,actual,total=values;cap=record.get('max_tokens');finish=record.get('finish_reason')
    if type(cap) is not int or not (prompt>0 and 0<actual<=cap and total==prompt+actual):return False
    return (record.get('successful') is True and record.get('saw_done') is True and record.get('error') is None
            and finish in ('stop','length') and (finish!='length' or actual==cap)
            and record.get('native_timings',{}).get('predicted_n')==actual
            and not any('tool' in c.get('channels',[]) or c.get('bytes',{}).get('reasoning',0)>0 for c in record.get('chunks',[])))

def delta(a,b):return {'off':a,'on':b,'on_minus_off':b-a if type(a) in (int,float) and type(b) in (int,float) else None,
                       'on_relative_change_pct':100*(b/a-1) if type(a) in (int,float) and type(b) in (int,float) and a else None}

def compare_phases(a,b):
    out={}
    for name in sorted(set(a['phases'])&set(b['phases'])):
        rows=[a['phases'][name],b['phases'][name]]
        out[name]={'max_lane_phase_clipped_silence_s':delta(*[r['max_lane_phase_clipped_silence_s'] for r in rows]),
            'max_collective_silence_s':delta(*[r['collective_no_visible_event_gaps']['max_s'] for r in rows]),
            'lanes':[{'lane':i,**{q:delta(*[r['lanes'][i]['raw_gaps_ending_in_phase'][q] for r in rows]) for q in ('p95_s','p99_s','max_s')}} for i in range(4)],
            'collective_stall_total_s':{threshold:delta(*[r['stalled_after_threshold_s'][threshold]['all_lanes_stalled_total_s'] for r in rows]) for threshold in ('0.25','1','5')}}
    return out

def self_test():
    records=[{'name':str(i),'role':'resident','lane':i,'chunks':[{'epoch':t,'bytes':{'visible_content':1}} for t in (0,1,11,12,24)]} for i in range(4)]
    p=phase(records,2,10);assert p['all_four_visible_cover_phase'] and p['max_lane_phase_clipped_silence_s']==8
    assert p['collective_no_visible_event_gaps']['max_s']==8
    assert p['stalled_after_threshold_s']['1']['all_lanes_stalled_total_s']==8
    assert p['stalled_after_threshold_s']['5']['all_lanes_stalled_total_s']==4
    records[0]['chunks'].append({'epoch':6,'bytes':{'visible_content':0,'reasoning':1000}})
    assert phase(records,2,10)['collective_no_visible_event_gaps']['max_s']==8
    records[0]['chunks'].append({'epoch':6,'bytes':{'visible_content':1}})
    assert phase(records,2,10)['collective_no_visible_event_gaps']['max_s']==4
    assert union_intervals([(0,2),(1,3)])==[(0,3)] and union_intervals([(0,2),(1,3)],2)==[(1,2)]
    assert distribution([.1,.2,10])['p95_s']==10
    assert not phase(records,2,26)['all_four_visible_cover_phase']
    records[0]['end_epoch']=100
    assert not phase(records,2,26)['all_four_visible_cover_phase']
    record={'name':'request-0','request_sha256':'same-request','output_sha256':'same-output','channel_output_sha256':{'visible_content':'same'},
            'usage':{'prompt_tokens':10000,'completion_tokens':2815,'total_tokens':12815},'max_tokens':4096,'native_timings':{'predicted_n':2815},
            'finish_reason':'stop','successful':True,'saw_done':True,'error':None,'chunks':[]}
    assert normal_actual_output(record)
    for update in ({'finish_reason':'length'},{'native_timings':{'predicted_n':2814}},{'usage':{'prompt_tokens':10000,'completion_tokens':2815,'total_tokens':12814}},
                   {'error':'stream aborted'},{'chunks':[{'channels':['tool'],'bytes':{'tool':0}}]}):
        bad=copy.deepcopy(record);bad.update(update);assert not normal_actual_output(bad)
    with tempfile.TemporaryDirectory(prefix='strata-visible-silence-cpu-') as tmp:
        root=Path(tmp);arms=[root/'off-run'/'01-off',root/'on-run'/'01-on']
        manifest={k:'identity' for k in ('same_binary_sha256','adjacent_library_sha256','base_config_sha256','server_sha256','harness_sha256','calibration_sha256')}
        manifest.update(suites=['primary'],ended_epoch=200)
        plan={k:'same' for k in ('case','plan','calibration_sha256','fixture_generator_sha256','client_sha256','stream_client_sha256','stall_metrics_sha256','fixtures','preflight_counts','output_contract')}
        plan.update(completed=True,all_checks_passed=True)
        report={'all_checks_passed':True,'phases':dict.fromkeys(('baseline','mixed','recovery'),{}),
                'records':[{**copy.deepcopy(record),'name':'request-'+str(i)} for i in range(6)]}
        def save(path,value):path.write_text(json.dumps(value))
        for arm,flag,d in zip(('off','on'),('0','1'),arms):
            (d/'primary').mkdir(parents=True)
            parent=copy.deepcopy(manifest);parent['arms']=[{'tag':d.name,'arm':arm,'suite_returncodes':{'primary':0}}]
            save(d.parent/'manifest.json',parent)
            save(d/'metadata.json',{'arm':arm,'config_except_variables_log_key_sha256':'same','config':{'env':{'STRATA_PIPED_CONCURRENT_MTP':flag}},'suite_returncodes':{'primary':0},'finished_epoch':100})
            save(d/'primary'/'manifest.json',plan);save(d/'primary'/'results.json',report)
        good=identity_and_work(*arms,'primary');assert good['case_pair_eligible'] and good['full_runs_complete']
        changed=copy.deepcopy(report);r=changed['records'][0];r['usage']['completion_tokens']=2814;r['usage']['total_tokens']=12814;r['native_timings']['predicted_n']=2814
        save(arms[1]/'primary'/'results.json',changed);assert not identity_and_work(*arms,'primary')['case_pair_eligible']
        save(arms[1]/'primary'/'results.json',report)
        partial=copy.deepcopy(parent);partial['suites'].append('budget');save(arms[1].parent/'manifest.json',partial)
        assert not identity_and_work(*arms,'primary')['full_runs_complete']
        partial=copy.deepcopy(parent);partial['arms']=[];save(arms[1].parent/'manifest.json',partial)
        assert not identity_and_work(*arms,'primary')['full_runs_complete']
        changed=copy.deepcopy(report);changed['phases'].pop('recovery');save(arms[1]/'primary'/'results.json',changed)
        assert not identity_and_work(*arms,'primary')['case_pair_eligible']
    print('PASS: visible-only gaps, collective silence, boundary censoring, late HTTP end rejection, exact matched EOS, count/native/tool/error/phase/unfinished-run negative cases; no HTTP/GPU')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--single-case',type=Path);p.add_argument('--suites',default='primary,mixed100k');p.add_argument('--out',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return 0
    require(a.out is not None and not a.out.exists(),'Explicit NEW output file required')
    if a.single_case:
        require(not a.baseline and not a.candidate,'Choose single diagnostic or matched-pair mode')
        result={'mode':'single_case_diagnostic_no_AB_claim','case':summarize_case(a.single_case),'performance_comparison_eligible':False}
    else:
        require(a.baseline and a.candidate,'Both arm directories required');suites=a.suites.split(',');require(suites and not set(suites)-{'primary','mixed100k'},'Use primary,mixed100k only')
        result={'mode':'matched_visible_SSE_analysis','cases':{},'performance_comparison_eligible':False}
        for suite in suites:
            qualification=identity_and_work(a.baseline,a.candidate,suite);summaries=[summarize_case(p/suite) for p in (a.baseline,a.candidate)]
            qualification['visible_all_phases']=all(all(v['all_four_visible_cover_phase'] for v in s['phases'].values()) for s in summaries)
            eligible=qualification['case_pair_eligible'] and qualification['visible_all_phases'] and qualification['full_runs_complete']
            result['cases'][suite]={'qualification':qualification,'summaries':summaries,'comparative_metrics':compare_phases(*summaries) if eligible else None,
                                   'performance_comparison_eligible':eligible,'requires_full_qualifications_before_deployment':True}
        result['performance_comparison_eligible']=all(r['performance_comparison_eligible'] for r in result['cases'].values())
    result['script_sha256']=sha(Path(__file__));result['deployment_decision']='No automatic deployment verdict. Review prefill, correctness, hardware conditions and repeated matched stability.'
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'out':str(a.out.resolve()),'mode':result['mode'],'performance_comparison_eligible':result['performance_comparison_eligible']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
