#!/usr/bin/env python3
"""Read local numeric snapshots only. Reconstruct historical concurrency/rates."""
import json, hashlib, bisect
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
B = Path(__file__).resolve().parent.parent
O = Path(__file__).resolve().parent
TZ = ZoneInfo('Asia/Taipei')
def local(t): return datetime.fromtimestamp(t, TZ).isoformat(timespec='milliseconds')
def read(name): return json.loads((B / name).read_text())
s, n, c = map(read, ['snapshot-initial.json', 'native-numeric.json', 'console-numeric.json'])
rs = s['requests']
byid = {r['id']:r for r in rs}
def end(r): return r['time'] + r['duration_s']
def item(r):
    return dict(r, start_local=local(r['time']), end_local=local(end(r)),
                first_token_upper_local=local(r['time']+r['first_engine_token_s']),
                reuse_fraction=r['reused']/r['prompt_tokens'])
anchors = {}
done_matches = []
for d in c['done_rows']:
    cand = [r for r in rs if r['output_tokens']==d['output_tokens'] and r['decode_tok_s']==d['decode_tok_s'] and abs(r['duration_s']-d['duration_s_rounded'])<=.55]
    if len(cand)==1:
        r=cand[0]; t=end(r)
        anchors[d['line']]={'lo':t-.06,'hi':t+.06,'request_id':r['id'],'type':'done'}
        done_matches.append(dict(d, request_id=r['id'], end_epoch=t, end_local=local(t)))
progress = [p for p in c['progress_rows'] if 590<=p['line']<=870]
mapped={}; passes=[]
for iteration in range(8):
    count=0
    for p in progress:
        if p['line'] in mapped: continue
        lines=sorted(anchors); k=bisect.bisect_left(lines,p['line'])
        lo=anchors[lines[k-1]]['lo'] if k else float('-inf')
        hi=anchors[lines[k]]['hi'] if k<len(lines) else float('inf')
        cand=[]
        for r in rs:
            t=r['time']+p['elapsed_s_rounded']
            first=p['elapsed_s_rounded']-p['generated']/p['tok_s_mean'] if p['tok_s_mean'] else -1e9
            l=max(t-.55,lo,r['time']); h=min(t+.55,hi,end(r)+.06)
            if l<=h and p['generated']<=r['output_tokens'] and -.6<=first<=r['first_engine_token_s']+.6:
                cand.append({'request_id':r['id'],'lo':l,'hi':h,'estimated_epoch':t})
        if len(cand)==1:
            m=dict(p,**cand[0]); mapped[p['line']]=m
            anchors[p['line']]=dict(cand[0],type='progress');count+=1
    passes.append(count)
    if not count: break
ids=['423f212c94bb','0641d465d3a6','c56979a2ad66','c1a7326f8806']
central=[byid[x] for x in ids]
stream=[]
for rid,a,z in [('423f212c94bb',667,769),('0641d465d3a6',668,772),('c56979a2ad66',670,766),('c1a7326f8806',669,768)]:
    p,q=mapped[a],mapped[z]
    assert p['request_id']==q['request_id']==rid
    delta=q['generated']-p['generated']; dt=q['estimated_epoch']-p['estimated_epoch']
    stream.append({'request_id':rid,'from':p,'to':q,'output_delta':delta,'estimated_s':dt,'estimated_tok_s':delta/dt,'conditional_tok_s_range':[delta/(q['hi']-p['lo']),delta/(q['lo']-p['hi'])]})
# Bound counters on two shared console-line cuts, using observed cumulative counts.
early_latest=[667,668,670,669]; early_next=[671,672,674,673]
late_latest=[769,767,766,768]; late_next=[773,772,770]
def total(lines): return sum(mapped[x]['generated'] for x in lines)
e_lo,e_hi=total(early_latest),total(early_next)
l_lo=total(late_latest); l_hi=total(late_next)+byid['c1a7326f8806']['output_tokens']
d_min,d_max=l_lo-e_hi,l_hi-e_lo
t_min=mapped[769]['lo']-mapped[670]['hi']; t_max=mapped[769]['hi']-mapped[670]['lo']
# Native batch boundaries are log-order evidence, not absolute wall timestamps.
rows=n['rows']; bi=next(i for i,x in enumerate(rows) if x['kind']=='batch' and x['values'][2]==85324)
prev=max(i for i,x in enumerate(rows[:bi]) if x['kind']=='batch')
batch_matches=[]
for i in range(prev+1,bi):
    x=rows[i]
    if x['kind']!='request': continue
    v=x['values']
    cand=[r for r in rs if r['prompt_tokens']==v[0] and r['reused']==v[1] and r['prompt_read']==v[2]]
    if len(cand)==1:
        r=cand[0];batch_matches.append({'native_index':i,'offset':x['offset'],'request_id':r['id'],'start_epoch':r['time'],'end_epoch':end(r),'start_local':local(r['time']),'end_local':local(end(r))})
start=datetime(2026,11,7,21,43,tzinfo=TZ).timestamp(); finish=start+120
edges=sorted(set([start,finish]+[t for r in rs for t in [r['time'],end(r)] if start<t<finish]))
segments=[]
for a,z in zip(edges,edges[1:]):
    m=(a+z)/2; active=[r for r in rs if r['time']<=m<end(r)]
    segments.append({'start_local':local(a),'end_local':local(z),'active_count':len(active),'past_first_token_upper_count':sum(r['time']+r['first_engine_token_s']<=a for r in active),'request_ids':[r['id'] for r in active]})
v=rows[bi]['values']
out={
 'input_sha256':{name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in ['snapshot-initial.json','native-numeric.json','console-numeric.json']},
 'server_sha256':s.get('server_sha256'),
 'target_local':'2026-11-07T21:44:00+08:00',
 'time_semantics':{'history_time':'start after ensure_loaded; duration rounded 0.1s','first_engine_token_s':'clock begins earlier than history.time, so start+first is an upper approximation, not an exact first-token timestamp','console_elapsed':'rounded whole seconds since same request start','console_rate':'cumulative mean since first token, not an instantaneous rate'},
 'limitations':['No per-token wall-clock history exists in these snapshots.','Console request assignment is reconstructed from unique done matches, elapsed rounding, first-token upper approximations and line order.','Console bounds are conditional on stable clocks and approximately ordered, undelayed stdout prints; they are not certified measurement bounds.','Whole-request and batch averages are never allocated to individual seconds.','Short sub-second peaks of 300–400 tok/s are not excluded.'],
 'history_count':len(rs),'done_unique_matches':len(done_matches),'done_match_end_order_monotonic':all(a['end_epoch']<=b['end_epoch'] for a,b in zip(done_matches,done_matches[1:])),'done_end_order_max_reversal_s':max([0]+[a['end_epoch']-b['end_epoch'] for a,b in zip(done_matches,done_matches[1:])]),'done_match_end_order_consistent_with_rounding':all(a['end_epoch']-.06<=b['end_epoch']+.06 for a,b in zip(done_matches,done_matches[1:])),
 'console_mapping_pass_counts':passes,'ambiguous_progress_lines':[p['line'] for p in progress if p['line'] not in mapped],
 'timeline_21_43_to_21_45':segments,
 'central_four_requests':[item(r) for r in central],
 'central_window':{'all_four_past_first_upper_epoch':max(r['time']+r['first_engine_token_s'] for r in central),'first_end_epoch':min(end(r) for r in central),'next_admission_epoch':byid['444425241714']['time'],'no_other_native_request_admissions_between_indices':[255,257]},
 'individual_reconstructed_rates':stream,
 'sum_individual_estimated_rates_tok_s':sum(x['estimated_tok_s'] for x in stream),
 'shared_cut_reconstruction':{'early_console_line':670,'late_console_line':769,'early_counter_range':[e_lo,e_hi],'late_counter_range':[l_lo,l_hi],'output_delta_range':[d_min,d_max],'elapsed_s_range':[t_min,t_max],'conditional_aggregate_tok_s_range':[d_min/t_max,d_max/t_min]},
 'long_batch':{'native_index':bi,'offset':rows[bi]['offset'],'groups':v[0],'slots_per_group':v[1],'group_steps':v[2],'actual_emitted_rows':v[3],'wall_ms':v[4],'aggregate_rows_s':v[5],'draft_accepted':v[6],'draft_offered':v[7],'acceptance_fraction':v[6]/v[7],'outputs_per_group_step':v[3]/v[2],'output_identity_holds':v[3]==v[2]+v[6],'covers_21_44_by_numeric_matches':all(rid in [m['request_id'] for m in batch_matches] for rid in ids),'unique_request_matches':batch_matches,'wall_timestamp_limitation':'No wall timestamps; matches place the same uninterrupted busy batch across target minute. 167.7 tok/s and 80.5% acceptance are averages of whole 918.139s block, not the target minute.'},
 'maximum_completed_request_decode_tok_s':max(r['decode_tok_s'] for r in rs),
 'snapshot_hardware':{'snapshot_epoch':s['time'],'snapshot_local':local(s['time']),'gpu_util_mean':s['hardware']['gpu_util'],'per_gpu_util':[g['util'] for g in s['hardware']['gpus']],'not_target_time':True},
 'mapped_console_rows':[mapped[k] for k in sorted(mapped)],
 'matched_done_rows':done_matches,
}
(O/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['done_unique_matches','console_mapping_pass_counts','ambiguous_progress_lines','sum_individual_estimated_rates_tok_s','shared_cut_reconstruction','snapshot_hardware']},ensure_ascii=False,indent=2))
