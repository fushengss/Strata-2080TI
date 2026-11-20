#!/usr/bin/env python3
"""Independent local P1 full validation audit; safe to rerun as completed arms arrive.
No service/network/GPU operations and no writes inside run directories.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics as st

B = Path(__file__).resolve().parent
RUN = B / 'p1-validation'
SOURCES = [Path(__file__)]
SUITES = ('prefill', 'c8', 'c4mixed', 'budget', 'long')
EXPECTED_ENGINE = '8b27e040d26b51a67cfef4d7c7154fa8cc83f6d0a1cf1b1826f9eb0295bd30ab'
EXPECTED_LIB = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
FLAG = 'STRATA_PIPED_TARGET_HEAD'


def read(path):
    SOURCES.append(path)
    return json.loads(path.read_text())


def digest(value):
    raw = value.encode() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def logical(row):
    return {k:row.get('usage', {}).get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}


def stats(values):
    values = sorted(x for x in values if x is not None)
    return {'n':len(values), 'min':min(values), 'median':st.median(values), 'max':max(values), 'values':values} if values else None


def native(rows):
    if not rows: return None
    tokens = sum(r['rows'] for r in rows); wall = sum(r['wall_ms'] for r in rows)
    accepted = sum(r['drafts_accepted'] for r in rows); offered = sum(r['drafts_offered'] for r in rows)
    return {'batches':len(rows), 'emitted_output_rows':tokens, 'wall_ms':wall,
            'tokens_per_s':tokens/(wall/1000), 'accepted':accepted, 'offered':offered,
            'acceptance_fraction':accepted/offered if offered else None,
            'scope':'Native piped busy blocks, including admission stalls; not whole API request duration.'}


def native_key(row):
    return tuple(row[k] for k in ('groups', 'slots_per_group', 'group_steps', 'rows', 'wall_ms', 'drafts_accepted', 'drafts_offered'))


def extract_native_c4(root, c8, budget, long):
    path = root/'engine.log'
    if not path.exists(): return {'available':False, 'reason':'No numeric native C4 batch summary in downloaded files.'}
    SOURCES.append(path)
    pattern = re.compile(r'strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, (\d+) rows in ([\d.]+) ms = ([\d.]+) rows/s .*?\((\d+) of (\d+) MTP proposals kept\)')
    fields=('groups','slots_per_group','group_steps','rows','wall_ms','rows_per_s','drafts_accepted','drafts_offered')
    rows=[]
    with path.open(errors='replace') as handle:
        for line_no,line in enumerate(handle,1):
            match=pattern.search(line)
            if match: rows.append(dict(zip(fields,map(float,match.groups())),line=line_no))
    first=c8['numeric_piped_log']['records']
    last=[r for co in budget['cohorts'] for r in co['piped_numeric_log']['records']]+long['cohort']['piped_numeric_log']['records']
    count=len(rows)-len(first)-len(last)
    matched=count>0 and [native_key(r) for r in rows[:len(first)]]==[native_key(r) for r in first] and [native_key(r) for r in rows[-len(last):]]==[native_key(r) for r in last]
    if not matched: return {'available':False,'reason':'Could not uniquely bracket C4 native batches between independently saved C8 and budget/long summaries.'}
    middle=rows[len(first):len(rows)-len(last)]
    return {'available':True,'summary':native(middle),'numeric_records':middle,
            'association':'No native wall timestamps. Sequential suite order plus exact numeric matches of all preceding C8 and following budget/long batches leave these C4 batches. Not a timestamp association.'}


def integrity(records):
    checks=[]
    for r in records:
        for text_key,hash_key in [('synthetic_request','request_sha256'),('output_text','output_sha256'),('reasoning_text','reasoning_sha256')]:
            if text_key in r: checks.append(digest(r[text_key])==r[hash_key])
    return {'recomputed_hashes':len(checks),'all_match':all(checks)}


def compare_records(left,right):
    a={r['name']:r for r in left};b={r['name']:r for r in right};out=[]
    for name in sorted(a.keys()|b.keys()):
        if name not in a or name not in b:
            out.append({'name':name,'passed':False,'missing':True});continue
        x,y=a[name],b[name]
        checks={'same_request':x.get('request_sha256')==y.get('request_sha256')}
        cancel=bool(x.get('intentional_cancel') or y.get('intentional_cancel'))
        if cancel:
            checks['both_cancelled_mid_generation']=all(r.get('intentional_cancel') and r.get('closed_mid_generation') and not r.get('saw_done') and not r.get('error') for r in (x,y))
            for key in ('output_text','reasoning_text'):
                size=min(len(x[key]),len(y[key]));checks[key+'_common_prefix_equal']=x[key][:size]==y[key][:size]
            checks['nonempty_common_prefix']=min(len(r['output_text'])+len(r['reasoning_text']) for r in (x,y))>0
        else:
            for key in ('output_sha256','reasoning_sha256','channel_output_sha256','channel_sha256','messages_sha256','finish_reason'):
                if key in x or key in y:checks[key+'_equal']=key in x and key in y and x[key]==y[key]
            checks['complete']=all(r.get('saw_done') and not r.get('error') and r.get('finish_reason') in ('length','stop') for r in (x,y))
            checks['logical_usage_equal']=logical(x)==logical(y)
            checks['logical_usage_consistent']=all(all(type(v) is int for v in logical(r).values()) and logical(r)['prompt_tokens']+logical(r)['completion_tokens']==logical(r)['total_tokens'] for r in (x,y))
        out.append({'name':name,'cancelled':cancel,'checks':checks,'passed':all(checks.values()),
                    'logical_usage_off_on':[logical(x),logical(y)],'full_usage_metadata_equal':x.get('usage')==y.get('usage')})
    return {'passed':all(r['passed'] for r in out),'cases':out}


def arm(tag):
    root=RUN/tag; path=root/'metadata.json'
    if not path.exists(): return {'status':'PARTIAL_DOWNLOAD' if root.exists() else 'NOT_DOWNLOADED'},None
    meta=read(path)
    if 'finished_epoch' not in meta or set(meta.get('suite_returncodes',{}))!=set(SUITES):
        return {'status':'INCOMPLETE','completed_suites':meta.get('suite_returncodes',{})},None
    c8=read(root/'c8/c8-check.json');pref=read(root/'prefill/summary.json')
    budget=read(root/'budget/budget-correctness.json');long=read(root/'long/long-concurrency.json')
    mix=read(root/'c4mixed/c4/summary.json');mixplan=read(root/'c4mixed/manifest.json')
    precords=[read(root/'prefill'/f'{kind}-{i}.json') for kind in ('cold','warm') for i in (1,2,3)]
    mrecords=[read(p) for p in sorted((root/'c4mixed/c4').glob('*.json')) if p.name.startswith(('resident-','injected-'))]
    credentials=read(root/'credential-scan.json')
    records=c8['records'];a=min(r['start_epoch'] for r in records);z=max(r['end_epoch'] for r in records)
    c8rate=sum(r['usage']['completion_tokens'] for r in records)/(z-a)
    checks={'all_suite_returncodes_zero':all(v==0 for v in meta['suite_returncodes'].values()),
            'identity_engine_and_library':meta['engine_sha256']==EXPECTED_ENGINE and meta['adjacent_library_sha256']==EXPECTED_LIB,
            'mapped_engine_and_library':bool(meta['actual_engine']) and all(p['engine_sha256']==EXPECTED_ENGINE and p['library_sha256']==EXPECTED_LIB and p['expected_library_mapped'] for p in meta['actual_engine']),
            'target_flag_matches_arm':all(p['flags'][FLAG]==('1' if meta['arm']=='on' else '0') for p in meta['actual_engine']),
            'all_experts_resident':meta['startup_facts']['all_experts_resident'],
            'all_eight_prefill_buffers':meta['startup_facts']['all_eight_have_3_80_gib_own_buffers'],
            'c8_checks':c8['all_checks_passed'] and all(c8['checks'].values()),
            'c8_tokens_8192':sum(r['usage']['completion_tokens'] for r in records)==8192,
            'c8_rate_recomputed':abs(c8rate-c8['timing']['whole_batch']['completed_api_tokens_per_s'])<1e-8,
            'prefill_all_valid':pref['all_valid'] and all(r['valid'] for r in precords),
            'budget_all_checks':budget['all_checks_passed'] and all(budget['checks'].values()),
            'long_all_checks':long['all_checks_passed'] and all(long['summary']['checks'].values()),
            'c4_requests_complete':mix['failed_requests']==0 and mix['successful_complete_requests']==6 and not mix['error'],
            'c4_baseline_occupancy_qualified':mix['baseline_occupancy_qualified'],
            'credential_scan_zero':credentials['credential_matches']==0,
            'prefill_no_target_head_route':meta['target_head_suite_evidence']['prefill']['windows']==0,
            'decode_target_head_runtime_route':all((meta['target_head_suite_evidence'][suite]['windows']>0)==(meta['arm']=='on') for suite in ('c8','c4mixed','budget','long'))}
    hashes=integrity(records+budget['records']);checks['saved_synthetic_hashes_match']=hashes['all_match']
    prefstats={kind:{key:stats([r[key] if key=='ttft_s' else r['engine'][key] for r in precords if r['kind']==kind]) for key in ('ttft_s','read_ms','decode_ms','decode_tps')} for kind in ('cold','warm')}
    native_c4=extract_native_c4(root,c8,budget,long)
    result={'status':'COMPLETE','checks':checks,'all_arm_checks_passed':all(checks.values()),
      'metadata':{k:meta[k] for k in ('arm','start_epoch','ready_epoch','finished_epoch','engine_sha256','adjacent_library_sha256','config_except_variables_log_key_sha256','target_head_suite_evidence','shared_head_suite_evidence')},
      'saved_hash_integrity':hashes,
      'c8':{'api_tokens':8192,'api_wall_s':z-a,'api_tokens_per_s':c8rate,'native':native(c8['numeric_piped_log']['records']),
            'common_sse_interval_s':c8['timing']['overlap']['common_interval_s'],
            'ttft_s':stats([r['first_output_epoch']-r['start_epoch'] for r in records]),
            'max_inter_chunk_gap_s':max(r['longest_inter_chunk_gap_s'] for r in c8['timing']['lanes'])},
      'prefill':{'statistics':prefstats,'requests':[dict(name=r['name'],usage=r['usage'],ttft_s=r['ttft_s'],engine=r['engine']) for r in precords]},
      'c4mixed':{'whole_scenario_api':mix['whole_scenario_completed_api_token_rate'],'native':native_c4,
                 'phases':mix['phases'],'occupancy':mix['occupancy'],'recovery':{k:v for k,v in mix['recovery'].items() if k!='bins'},
                 'recovery_bin_diagnostics':[{'offset_s':x['start_epoch']-mix['phases']['recovery']['start_epoch'],
                     'output_bytes_per_s':x['resident_output_bytes_per_s'],
                     'fraction_of_baseline':x['resident_output_bytes_per_s']/mix['phases']['baseline']['resident_output_bytes_per_s'],
                     'max_silence_s':x['max_observed_no_output_interval_s']} for x in mix['recovery']['bins']],
                 'requests':[{k:r.get(k) for k in ('name','role','actual_prompt_tokens','cached_tokens','max_tokens','usage','successful','prompt_within_nominal_tolerance','ttft_s','first_visible_content_s','elapsed_s','output_inter_chunk_gaps')} for r in mrecords],
                 'nominal_prompt_tolerance_all_pass':all(r['prompt_within_nominal_tolerance'] for r in mrecords)},
      'budget':{'checks':budget['checks'],'continuations':budget['continuations'],'recovery_usage':budget['recovery_usage_comparison'],
                'native':native([r for co in budget['cohorts'] for r in co['piped_numeric_log']['records']]),
                'cancel_common_sse_s':next(co['overlap']['common_decode_s'] for co in budget['cohorts'] if co['name']=='cancel-four')},
      'long':{'summary':long['summary'],'native':native(long['cohort']['piped_numeric_log']['records'])}}
    raw={'meta':meta,'c8':records,'prefill':precords,'budget':budget['records'],'long':long['records'],'c4mixed':mrecords,'mixplan':mixplan}
    return result,raw


def change(off,on):
    return {'off':off,'on':on,'on_relative_change_pct':100*(on/off-1) if off else None}


def main():
    results={};raw={}
    for tag in ('01-on','02-off'):results[tag],raw[tag]=arm(tag)
    pilot=read(B/'p1-pilot-analysis.json')
    result={'scope':'Independent full reverse on/off audit with earlier off/on C8 pilot; local files only.',
            'arms':results,'pilot_relative_changes':pilot['relative_changes'],
            'full_comparison':None,'deployment_recommendation':'HOLD_PENDING_COMPLETED_OFF_ARM',
            'limits':['Only one pilot off/on pair and one broader reverse on/off pair; no confidence interval or production SLO proof.',
                      '100k prefill has three cold and three warm requests per arm; 256-token solo decode is secondary evidence.',
                      'C4 phases use output bytes and usage-weighted token estimates, not per-token timestamps. Inter-chunk gaps are not TBT.',
                      'C4 fixture nominal prompt sizes miss its ±10% targets; compare matched actual token counts, not label as exact10k/60k.',
                      'Native piped blocks and whole API request/scenario rates have different boundaries and token counts.',
                      'Budget and long cases must pass exact logical output/usage, but cancellation compares a valid nonempty common prefix.',
                      'Native log has no wall timestamps; C4 native association uses sequential suite order and matching neighboring numeric batch signatures.',
                      'C4 recovery threshold is sensitive to byte composition: off 4–6s bin barely exceeds90% baseline, while on falls below; recovery gaps stay25–27ms. The2s vs6s threshold result does not mean an extra4s GPU/output stall.']}
    partial={}
    for suite in ('prefill','c8','c4mixed'):
        left=RUN/'02-off'/suite;right=RUN/'01-on'/suite
        sentinel={'prefill':'summary.json','c8':'c8-check.json','c4mixed':'c4/summary.json'}[suite]
        if not (left/sentinel).exists() or not (right/sentinel).exists(): continue
        if suite=='prefill':
            pair=[[read(root/f'{kind}-{i}.json') for kind in ('cold','warm') for i in (1,2,3)] for root in (left,right)]
            metrics={}
            for kind in ('cold','warm'):
                for metric in ('read_ms','ttft_s','decode_tps'):
                    v=[st.median([r[metric] if metric=='ttft_s' else r['engine'][metric] for r in rows if r['kind']==kind]) for rows in pair]
                    metrics[kind+'_'+metric]=change(*v)
        elif suite=='c8':
            reports=[read(root/'c8-check.json') for root in (left,right)];pair=[r['records'] for r in reports]
            metrics={'api_tokens_per_s':change(*[sum(x['usage']['completion_tokens'] for x in rows)/(max(x['end_epoch'] for x in rows)-min(x['start_epoch'] for x in rows)) for rows in pair]),
                     'native_tokens_per_s':change(*[native(r['numeric_piped_log']['records'])['tokens_per_s'] for r in reports])}
        else:
            reports=[read(root/'c4/summary.json') for root in (left,right)]
            pair=[[read(p) for p in sorted((root/'c4').glob('*.json')) if p.name.startswith(('resident-','injected-'))] for root in (left,right)]
            metrics={'whole_api_tokens_per_s':change(*[r['whole_scenario_completed_api_token_rate']['tokens_per_s'] for r in reports])}
            for phase in ('baseline','mixed','recovery'):
                for metric in ('resident_output_bytes_per_s','max_observed_no_output_interval_s','jain_fairness_of_output_byte_rates'):
                    metrics[phase+'_'+metric]=change(*[r['phases'][phase][metric] for r in reports])
            for x,y in zip(pair[0],pair[1]):
                if x['role']=='injected':metrics[x['name']+'_ttft_s']=change(x['ttft_s'],y['ttft_s'])
            metrics['first_sustained_recovery_start_s']=change(*[r['recovery']['first_sustained_recovery_start_s'] for r in reports])
        partial[suite]={'scope':'Completed suite files only; final off metadata/other suites may still be pending.',
                        'parity':compare_records(*pair),'metrics':metrics,'hash_integrity':[integrity(rows) for rows in pair]}
    result['available_suite_comparisons']=partial
    manifest_path=RUN/'manifest.json'
    manifest=read(manifest_path) if manifest_path.exists() else None
    result['run_manifest_checks']=(
        {'same_binary':manifest.get('same_binary_sha256')==EXPECTED_ENGINE,
         'library':manifest.get('adjacent_library_sha256')==EXPECTED_LIB,
         'server':manifest.get('server_sha256')=='40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9',
         'base_config':manifest.get('base_config_sha256')=='f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220',
         'reverse_sequence':manifest.get('sequence')==['on','off'],
         'original_config_unchanged':manifest.get('original_config_unchanged') is True,
         'no_remaining_gpu_pids':'remaining_gpu_pids' in manifest and not manifest['remaining_gpu_pids']}
        if manifest else None)
    if manifest:
        harness_checks={}
        for name,expected in manifest['harness_sha256'].items():
            artifact=RUN/'harness'/name
            harness_checks[name]=artifact.is_file() and hashlib.sha256(artifact.read_bytes()).hexdigest()==expected
            if artifact.is_file():SOURCES.append(artifact)
        result['harness_source_checks']=harness_checks
        result['run_manifest_checks']['all_frozen_harness_sources_match']=all(harness_checks.values())
    if all(raw.values()):
        off,on=raw['02-off'],raw['01-on'];a,b=results['02-off'],results['01-on']
        cfg=[]
        for r in (off,on):
            c=copy.deepcopy(r['meta']['config']);c['env'].pop(FLAG,None);cfg.append(c)
        identity={'only_target_head_configuration_differs':cfg[0]==cfg[1],
                  'normalized_config_hash_equal':off['meta']['config_except_variables_log_key_sha256']==on['meta']['config_except_variables_log_key_sha256'],
                  'same_c4_fixture':off['mixplan']['fixture_id']==on['mixplan']['fixture_id'],
                  'same_c4_plan':off['mixplan']['plan']==on['mixplan']['plan']}
        cases={suite:compare_records(off[suite],on[suite]) for suite in SUITES}
        changes={'c8_api':change(a['c8']['api_tokens_per_s'],b['c8']['api_tokens_per_s']),
                 'c8_native':change(a['c8']['native']['tokens_per_s'],b['c8']['native']['tokens_per_s']),
                 'c4_api':change(a['c4mixed']['whole_scenario_api']['tokens_per_s'],b['c4mixed']['whole_scenario_api']['tokens_per_s']),
                 'long_api':change(a['long']['summary']['whole_batch_tps'],b['long']['summary']['whole_batch_tps']),
                 'long_native':change(a['long']['native']['tokens_per_s'],b['long']['native']['tokens_per_s']),
                 'long_max_ttft_s':change(a['long']['summary']['ttft_s']['max'],b['long']['summary']['ttft_s']['max'])}
        if a['c4mixed']['native']['available'] and b['c4mixed']['native']['available']:
            changes['c4_native']=change(a['c4mixed']['native']['summary']['tokens_per_s'],b['c4mixed']['native']['summary']['tokens_per_s'])
        for kind in ('cold','warm'):
            for metric in ('read_ms','ttft_s','decode_tps'):
                changes['prefill_'+kind+'_'+metric]=change(a['prefill']['statistics'][kind][metric]['median'],b['prefill']['statistics'][kind][metric]['median'])
        for phase in ('baseline','mixed','recovery'):
            for metric in ('resident_output_bytes_per_s','resident_token_rate_estimate','max_observed_no_output_interval_s','jain_fairness_of_output_byte_rates'):
                changes['c4_'+phase+'_'+metric]=change(a['c4mixed']['phases'][phase][metric],b['c4mixed']['phases'][phase][metric])
        for x,y in zip(sorted(a['c4mixed']['requests'],key=lambda r:r['name']),sorted(b['c4mixed']['requests'],key=lambda r:r['name'])):
            if x['role']=='injected':
                for metric in ('ttft_s','first_visible_content_s'):
                    changes[x['name']+'_'+metric]=change(x[metric],y[metric])
        all_checks=bool(result['run_manifest_checks']) and all(result['run_manifest_checks'].values()) and all(identity.values()) and all(r['all_arm_checks_passed'] for r in (a,b)) and all(c['passed'] for c in cases.values())
        c8_positive=changes['c8_api']['on_relative_change_pct']>0 and pilot['relative_changes']['c8_api_pct']>0
        prefill_up=[key for key,row in changes.items() if key.startswith('prefill_') and not key.endswith('decode_tps') and row['on_relative_change_pct']>0]
        result['full_comparison']={'identity_checks':identity,'case_parity':cases,'changes':changes,'all_correctness_and_identity_checks':all_checks,
                                  'c8_positive_in_both_orders':c8_positive,'prefill_medians_that_increased':prefill_up}
        result['deployment_recommendation']=('HOLD_CORRECTNESS_OR_IDENTITY_FAILURE' if not all_checks else
            'REVIEW_PREFILL_REGRESSION_OR_REPEAT_NOISY_MEDIANS' if prefill_up else
            'HOLD_REPEATABLE_C8_BENEFIT_NOT_ESTABLISHED' if not c8_positive else
            'P1_ELIGIBLE_FOR_MEASURED_C8_GAIN_WITH_MIXED_STABILITY_LIMITS')
        result['deployment_gate_notes']=[
            'No blocking correctness or measured median100k prefill regression found. P1 may remain the enabled baseline for independent P2 validation.',
            'P1 alone is an eligible deployment candidate for the repeated4.6–5.0% C8 API gain, with unchanged prefill configuration.',
            'Mixed C4 API/native throughput remains essentially flat in this pair; four-cold100k whole-request throughput is1.30% lower and native0.33% lower. One pair cannot distinguish such small changes from run variability.',
            'Do not describe P1 as solving mixed-admission stability: long-prefill output stalls remain about10s. Recovery byte-threshold sensitivity is separately documented.',
            'This audit does not authorize or evaluate untested P2/P3 settings; select the final engine/flags only after their independent validation.'
        ]
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(SOURCES)}
    (B/'p1-validation-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['P1 full reverse validation independent audit', 'Recommendation: '+result['deployment_recommendation']]
    for tag,r in results.items():
        lines.append(tag+': '+r['status'])
        if r['status']!='COMPLETE':continue
        lines.append('  Arm checks='+str(r['all_arm_checks_passed'])+'; C8 API %.3f tok/s, native %.3f tok/s.'%(r['c8']['api_tokens_per_s'],r['c8']['native']['tokens_per_s']))
        for kind in ('cold','warm'):
            s=r['prefill']['statistics'][kind]
            lines.append('  100k %s: median read %.3fs, TTFT %.3fs, solo decode %.2f tok/s; n=3.'%(kind,s['read_ms']['median']/1000,s['ttft_s']['median'],s['decode_tps']['median']))
        lines.append('  C4 API %.3f tok/s; actual prompts %s; nominal tolerance all pass=%s.'%(r['c4mixed']['whole_scenario_api']['tokens_per_s'],[q['actual_prompt_tokens'] for q in r['c4mixed']['requests']],r['c4mixed']['nominal_prompt_tolerance_all_pass']))
        if r['c4mixed']['native']['available']:lines.append('  C4 native %.3f tok/s, associated by suite/log order and exact neighboring summaries, not wall timestamps.'%r['c4mixed']['native']['summary']['tokens_per_s'])
        lines.append('  C4 mixed maximum silence %.3fs; byte-fairness %.5f; sustained recovery start %s seconds.'%(r['c4mixed']['phases']['mixed']['max_observed_no_output_interval_s'],r['c4mixed']['phases']['mixed']['jain_fairness_of_output_byte_rates'],r['c4mixed']['recovery']['first_sustained_recovery_start_s']))
    lines.append('Pilot off/on C8 API change: %+.3f%%; native %+.3f%%.'%(pilot['relative_changes']['c8_api_pct'],pilot['relative_changes']['c8_native_pct']))
    for suite,partial in result['available_suite_comparisons'].items():
        lines.append('Available completed '+suite+' parity='+str(partial['parity']['passed']))
        for name,row in partial['metrics'].items():lines.append('  %s: %.6f -> %.6f (%+.3f%%)'%(name,row['off'],row['on'],row['on_relative_change_pct']))
    if result['full_comparison']:
        comp=result['full_comparison'];lines.append('Full comparison exact correctness/identity: '+str(comp['all_correctness_and_identity_checks']))
        for name,row in comp['changes'].items():lines.append('%s: %.6f -> %.6f (%+.3f%% on vs off)'%(name,row['off'],row['on'],row['on_relative_change_pct']))
        lines.append('Prefill upward medians: '+str(comp['prefill_medians_that_increased']))
    if result.get('deployment_gate_notes'):lines.extend(result['deployment_gate_notes'])
    lines.extend(result['limits'])
    (B/'p1-validation-analysis.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
