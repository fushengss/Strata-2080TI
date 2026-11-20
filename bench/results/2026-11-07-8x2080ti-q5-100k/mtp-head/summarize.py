#!/usr/bin/env python3
"""Offline decode-head ABBA summary; no GPU/service operations."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from bench_client import save_json


def distribution(values):
    v=[x for x in values if isinstance(x,(float,int))]
    return {'n':len(v),'median':statistics.median(v),'min':min(v),'max':max(v),'values':v} if v else {'n':0}


def metric(records,field):
    if field.startswith('engine.'):
        return distribution([(r.get('matched_engine_record') or {}).get(field.split('.')[1]) for r in records])
    return distribution([r.get(field) for r in records])


def delta(off,on):
    if off.get('median') and on.get('median') is not None:
        return 100*(on['median']/off['median']-1)
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    a=p.parse_args()
    root=Path(a.root)
    manifest=json.loads((root/'manifest.json').read_text())
    arms=[]
    for d in sorted(root.glob('[0-9][0-9]-*')):
        if not (d/'results.json').exists():
            continue
        meta=json.loads((d/'metadata.json').read_text())
        records=json.loads((d/'results.json').read_text())
        complete=json.loads((d/'complete.json').read_text()) if (d/'complete.json').exists() else None
        arms.append({'tag':d.name,'arm':meta['arm'],'meta':meta,'records':records,'complete':complete})
    names=sorted({r['name'] for arm in arms for r in arm['records'] if r['category']!='warmup'})
    report={'manifest':manifest,'arm_count':len(arms),
            'complete_abba':manifest.get('sequence')==['off','on','on','off'] and len(arms)==4 and all(a['complete'] for a in arms),
            'same_binary_confirmed':len({a['meta']['engine_sha256'] for a in arms})==1,
            'same_config_except_flag_and_log':len({a['meta']['config_except_flag_log_sha256'] for a in arms})==1,
            'same_governors':all(a['meta']['governors_before']==manifest['governors_before'] and a['meta'].get('governors_after')==manifest['governors_before'] for a in arms),
            'requests':{},'correctness':[],'notes':[
                'The primary speed is the unique engine record matching both prompt and completion token counts.',
                'Client first-to-last SSE throughput is an estimate; chunks may carry multiple tokens.',
                'Prefill comparison requires zero reused tokens; append requires >90k prefix reuse.',
                'ABBA supplies two server launches per flag. This is a small sample; show baseline drift and ranges.',
                'Exact generated text/reasoning equality checks execution behavior, not equality of all internal logits.']}
    for name in names:
        samples=[]
        for arm in arms:
            for r in arm['records']:
                if r['name']==name:
                    samples.append((arm['tag'],arm['arm'],r))
        off=[r for _,v,r in samples if v=='off']
        on=[r for _,v,r in samples if v=='on']
        text=lambda r:json.dumps([r.get('output_text',''),r.get('reasoning_text','')],ensure_ascii=False)
        off_outputs={text(r) for r in off}
        on_outputs={text(r) for r in on}
        requests={r.get('request_sha256') for _,_,r in samples}
        stable=len(off_outputs)==1 and len(off)>=2
        exact=bool(off and on) and len(off_outputs|on_outputs)==1 and len(requests)==1 and len({r.get('finish_reason') for _,_,r in samples})==1
        if exact:
            state='exact_match_all_available_arms'
        elif len(requests)!=1:
            state='input_mismatch_not_comparable'
        elif not stable:
            state='baseline_unstable_or_not_repeated_inconclusive'
        else:
            state='candidate_output_mismatch'
        c={'name':name,'state':state,'baseline_repeated_and_stable':stable,'same_request':len(requests)==1,
           'off_distinct_outputs':len(off_outputs),'on_distinct_outputs':len(on_outputs),
           'hashes':{tag:{'text':r['output_sha256'],'reasoning':r.get('reasoning_sha256'),
                          'combined':r.get('combined_output_sha256'),'tokens':r['usage'].get('completion_tokens')}
                     for tag,_,r in samples}}
        report['correctness'].append(c)
        item={'category':samples[0][2]['category'],'samples':[{ 'tag':tag,'flag':v,'checks':r.get('checks'),
                  'prompt_tokens':r['usage'].get('prompt_tokens'),'completion_tokens':r['usage'].get('completion_tokens'),
                  'finish_reason':r['finish_reason'],'engine':r.get('matched_engine_record'),
                  'head_offload_windows':r.get('decode_head_offload_windows'),
                  'mtp_head_offload_steps':r.get('mtp_head_offload_steps'),
                  'draft_acceptance_records':r.get('draft_acceptance_records'),
                  'matched_draft_acceptance':r.get('matched_draft_acceptance'),
                  'ttft_s':r['ttft_s'],'client_decode_tps_estimate':r.get('decode_tps_usage_minus_one_estimate')}
                  for tag,v,r in samples], 'metrics':{}}
        for field in ('engine.prefill_ms','engine.prefill_tps','engine.decode_ms','engine.decode_tps','ttft_s','decode_tps_usage_minus_one_estimate'):
            om,nm=metric(off,field),metric(on,field)
            item['metrics'][field]={'off':om,'on':nm,'on_vs_off_median_pct':delta(om,nm)}
        if len(off)==2:
            for field in ('prefill_ms','decode_tps'):
                x,y=[(r.get('matched_engine_record') or {}).get(field) for r in off]
                item['baseline_first_to_last_'+field+'_pct']=100*(y/x-1) if x and y is not None else None
        checks=[r.get('checks',{}) for _,_,r in samples]
        item['performance_comparable']=bool(off and on) and exact and all(
            ck.get('request_ok') and ck.get('isolated_activity') and ck.get('engine_full_request_record_unique') and
            ck.get('decode_head_route_confirmed') and ck.get('mtp_head_route_confirmed') and
            (not ck.get('cold_required') or ck.get('cold_zero_reuse')) and
            (not ck.get('append_required') or ck.get('append_reused_prefix')) and
            (not ck.get('warm_required') or ck.get('warm_reused_prefix')) and
            (not ck.get('performance_length_required') or (ck.get('performance_length_satisfied') and
                ck.get('draft_acceptance_record_unique') and ck.get('draft_acceptance_counts_valid'))) for ck in checks)
        report['requests'][name]=item
    report['all_outputs_identical']=bool(report['correctness']) and all(c['state']=='exact_match_all_available_arms' for c in report['correctness'])
    report['all_expected_head_routes_confirmed']=all(r.get('checks',{}).get('decode_head_route_confirmed') for arm in arms for r in arm['records'] if r['category']!='warmup' and r.get('checks',{}).get('decode_head_route_applicable',True))
    report['early_eos_cases_not_used_for_route_check']=[{'arm':arm['tag'],'name':r['name']} for arm in arms for r in arm['records'] if r.get('checks',{}).get('early_eos_correctness_case')]
    report['all_performance_requests_comparable']=all(v['performance_comparable'] for v in report['requests'].values() if v['category'].startswith(('cold-','warm-')) or v['category']=='append')
    report['all_expected_mtp_head_routes_confirmed']=all(r.get('checks',{}).get('mtp_head_route_confirmed') for arm in arms for r in arm['records'] if r.get('checks',{}).get('mtp_head_route_applicable',True))
    save_json(root/'comparison.json',report)
    lines=['Same-binary MTP-head ABBA',
           'Complete ABBA: '+str(report['complete_abba']),
           'Same binary: '+str(report['same_binary_confirmed']),
           'Only flag/log differ: '+str(report['same_config_except_flag_and_log']),
           'All outputs identical: '+str(report['all_outputs_identical']),
           'All expected main-head routes confirmed: '+str(report['all_expected_head_routes_confirmed']),
           'All expected MTP-head routes confirmed: '+str(report['all_expected_mtp_head_routes_confirmed']),
           'All performance requests comparable: '+str(report['all_performance_requests_comparable']),
           '', 'request\tcomparable\tprefill_ms_off\tprefill_ms_on\tprefill_ms_change_pct\tdecode_tps_off\tdecode_tps_on\tdecode_change_pct']
    for name,item in report['requests'].items():
        if item['category']=='correctness':
            continue
        pf=item['metrics']['engine.prefill_ms'];de=item['metrics']['engine.decode_tps']
        lines.append('\t'.join(str(x) for x in (name,item['performance_comparable'],pf['off'].get('median'),pf['on'].get('median'),pf['on_vs_off_median_pct'],de['off'].get('median'),de['on'].get('median'),de['on_vs_off_median_pct'])))
    lines+=['','Correctness:']+[c['name']+': '+c['state'] for c in report['correctness']]
    output='\n'.join(lines)+'\n'
    (root/'comparison.txt').write_text(output)
    print(output)


if __name__=='__main__':
    main()
