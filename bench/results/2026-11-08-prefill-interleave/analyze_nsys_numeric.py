#!/usr/bin/env python3
"""Offline interval/correlation analysis of private numeric-only Nsight extraction."""
from __future__ import annotations
import argparse,array,collections,gzip,hashlib,heapq,json,math,random,struct
from decimal import Decimal
from pathlib import Path

def epoch_ns(v): return int(Decimal(str(v))*Decimal(1000000000))
def merge(rows):
    result=[]
    for s,e in sorted(rows):
        if e<=s: continue
        if result and s<=result[-1][1]:
            result[-1][1]=max(result[-1][1],e)
        else: result.append([s,e])
    return result

def length(rows): return sum(e-s for s,e in rows)
def pctile(a,p):
    if not a: return None
    s=sorted(a); return s[max(0,math.ceil(len(s)*p)-1)]
def stat_ns(values):
    return {'n':len(values),'min_ms':min(values)/1e6 if values else None,'median_ms':pctile(values,.5)/1e6 if values else None,'p95_ms':pctile(values,.95)/1e6 if values else None,'max_ms':max(values)/1e6 if values else None}
def gap_bins(values):
    bounds=[0,1000,10000,100000,1000000,float('inf')]
    return [{'lower_exclusive_us':a/1e3,'upper_inclusive_us':b/1e3 if math.isfinite(b) else None,
             'count':sum(a<v<=b for v in values),'total_ms':sum(v for v in values if a<v<=b)/1e6}
            for a,b in zip(bounds,bounds[1:])]
def gaps(busy,width):
    out=[]; last=0
    for s,e in busy:
        if s>last: out.append((last,s))
        last=e
    if last<width: out.append((last,width))
    return out

def active_distribution(perdev,width):
    q=[]; positions={}
    for dev,intervals in perdev.items():
        if intervals: heapq.heappush(q,(intervals[0][0],dev,1,0))
    prev=0; active=0; durations=[0]*9; zero=[]
    while q:
        now=q[0][0]
        durations[active]+=now-prev
        if active==0 and now>prev: zero.append([prev,now])
        while q and q[0][0]==now:
            _,dev,delta,i=heapq.heappop(q); active+=delta
            if delta==1: heapq.heappush(q,(perdev[dev][i][1],dev,-1,i))
            elif i+1<len(perdev[dev]): heapq.heappush(q,(perdev[dev][i+1][0],dev,1,i+1))
        assert 0<=active<=8
        prev=now
    durations[active]+=width-prev
    if active==0 and width>prev: zero.append([prev,width])
    assert sum(durations)==width
    return durations,zero

def self_test():
    assert merge([(1,4),(3,7),(7,9),(12,13)])==[[1,9],[12,13]]
    assert gaps([[1,9],[12,13]],15)==[(0,1),(9,12),(13,15)]
    d,z=active_distribution({0:[[1,5]],1:[[3,7]]},10)
    assert d[:3]==[4,4,2] and z==[[0,1],[7,10]]
    assert epoch_ns(1794074805.0337007)==1794074805033700700
    assert pctile([-2,1,3],.5)==1
    rng=random.Random(735)
    for _ in range(100):
        rows={d:[sorted(rng.sample(range(101),2)) for _ in range(20)] for d in range(8)}
        busy={d:merge(v) for d,v in rows.items()}
        got,_=active_distribution(busy,100)
        expected=[0]*9
        for tick in range(100):
            n=sum(any(s<=tick<e for s,e in rows[d]) for d in rows)
            expected[n]+=1
        assert got==expected
        assert all(length(busy[d])==sum(any(s<=t<e for s,e in rows[d]) for t in range(100)) for d in rows)
    print('self-test PASS')

def analyze(inp,receipt,expected_counts=None):
    r=json.loads(receipt.read_text()); common=r['capture_workload_coverage']['common_sse_interval']
    lo=max(epoch_ns(r['start_returned']['epoch']),epoch_ns(common['common_start_epoch']))
    hi=min(epoch_ns(r['stop_requested']['epoch']),epoch_ns(common['common_end_epoch']))
    assert hi>lo and common['eight_streams_observed'] and r['owned_workload_qualified']
    assert r['engine_sha256']=='45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6' and r['snapshot_overwrite']=='0'
    assert r['actual_engine'][0]['flags']['STRATA_PIPED_CONCURRENT_MTP']=='1'
    assert r['actual_engine'][0]['flags']['STRATA_PARK_OVERWRITE']=='0'
    width=hi-lo
    gpu={k:{d:array.array('q') for d in range(8)} for k in (1,2,3)}
    counts=collections.Counter(); inwin=collections.Counter(); sums=collections.Counter(); api=[]; other=collections.defaultdict(list)
    corr_gpu={}; tids=collections.Counter(); bad=0; raw_min=None;raw_max=None
    with gzip.open(inp,'rb') as f:
        assert f.readline()==b'NSYSNUM1\n'; meta=json.loads(f.readline());gp=meta['engine_global_pid']
        assert meta['engine_pid']==r['actual_engine'][0]['pid']==164019
        assert sorted((d,c) for d,c,p in meta['devices_numeric'] if p==164019)==[(i,i) for i in range(8)]
        while True:
            b=f.read(64*16384)
            if not b: break
            assert len(b)%64==0
            for kind,s,e,d,p,c,g,a in struct.iter_unpack('<8q',b):
                counts[kind]+=1
                if kind in (1,2,3):
                    assert p==gp and 0<=d<8
                    raw_min=s if raw_min is None else min(raw_min,s);raw_max=e if raw_max is None else max(raw_max,e)
                if e<s: bad+=1;continue
                if kind==4:
                    api.append((s,e,p,c,g,a)); tids[p]+=1
                if kind==1:
                    rec=corr_gpu.get(c)
                    if rec is None: corr_gpu[c]=[s,e,1,{d}]
                    else: rec[0]=min(rec[0],s);rec[1]=max(rec[1],e);rec[2]+=1;rec[3].add(d)
                if e<=lo or s>=hi: continue
                cs=max(s,lo)-lo;ce=min(e,hi)-lo
                inwin[kind]+=1
                if kind in (1,2,3):
                    gpu[kind][d].extend((cs,ce,c));sums[(kind,d)]+=ce-cs
                elif kind in (5,6,7): other[kind].append((cs,ce,p,c,g,a))
    assert bad==0
    if expected_counts:
        schema=json.loads(expected_counts.read_text()); tab={x['name']:x.get('rows') for x in schema['tables']}
        for k,name in ((1,'KERNEL'),(2,'MEMCPY'),(3,'MEMSET'),(4,'RUNTIME'),(5,'SYNCHRONIZATION')):
            assert counts[k]==tab['CUPTI_ACTIVITY_KIND_'+name],(k,counts[k],tab['CUPTI_ACTIVITY_KIND_'+name])
    # Correlation IDs are examined for ambiguity; no serialized-TID bit packing is assumed.
    apimap=collections.defaultdict(list)
    for row in api: apimap[row[3]].append(row)
    names=meta['cuda_api_names']; unique={c:v[0] for c,v in apimap.items() if len(v)==1}
    corrcount=collections.Counter(); delay=[]; graph_delay=[]; linked_tids=collections.Counter()
    for c,(s,e,n,devices) in corr_gpu.items():
        v=apimap.get(c,[])
        corrcount['gpu_nodes']+=n
        if len(v)==1:
            corrcount['uniquely_matched_nodes']+=n; linked_tids[v[0][2]]+=n
            if lo<=s<hi:
                delta=s-v[0][1]; delay.append(delta)
                if names.get(str(v[0][4]),'').startswith('cudaGraphLaunch'):graph_delay.append(delta)
        elif not v:corrcount['unmatched_nodes']+=n
        else:corrcount['ambiguous_nodes']+=n
    report={'version':'nsys-observed-intervals-1','instrumented_diagnostic_only':True,'performance_comparison':False,
      'inputs':{'numeric_stream_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
                'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
                'schema_sha256':hashlib.sha256(expected_counts.read_bytes()).hexdigest() if expected_counts else None},
      'engine_pid':164019,'engine_sha256':r['engine_sha256'],'engine_global_pid':gp,
      'window':{'start_epoch_ns':lo,'end_epoch_ns':hi,'duration_s':width/1e9,'common_sse_duration_s':common['common_interval_s'],'common_coverage_fraction':width/1e9/common['common_interval_s'],'selection':'common SSE intersection with [start_returned,stop_requested]','all_activity_min_epoch_ns':raw_min,'all_activity_max_epoch_ns':raw_max},
      'raw_rows':dict(counts),'rows_intersecting_window':dict(inwin),'all_raw_cupti_row_counts_match_schema':bool(expected_counts),
      'device_cuda_mapping':[[d,c] for d,c,p in meta['devices_numeric'] if p==164019],
      'diagnostics':{'incomplete_cupti_events_dropped_at_stop':6,'engine_incomplete_cuda_warning':True,'engine_incomplete_osrt_warning':True,'cpu_sampling_or_context_switch_capture':False,'interpretation':'Observed execution coverage is a lower bound if missing events intersect window; apparent gaps are unobserved activity, not proof of hardware idle. Six incomplete events at stop lack retrieved GPU timestamps; their exact location is unknown.'},
      'devices':[]}
    kernel_busy={}; any_busy={}; all_longgaps=[]
    for d in range(8):
        bykind={}; recs=None
        for kind in (1,2,3):
            data=gpu[kind][d];it=iter(data); rows=sorted(zip(it,it,it)); gpu[kind][d]=None
            bykind[kind]=merge((s,e) for s,e,c in rows)
            if kind==1:recs=rows
        kb=bykind[1]; kernel_busy[d]=kb
        ab=merge([v for kind in (1,2,3) for v in bykind[kind]]);any_busy[d]=ab
        kg=gaps(kb,width);ag=gaps(ab,width)
        top=sorted(kg,key=lambda v:v[1]-v[0],reverse=True)[:20]
        starts={s:c for s,e,c in recs};next_details=[]
        for a,b in top:
            c=starts.get(b); v=unique.get(c); detail={'start_s':a/1e9,'end_s':b/1e9,'duration_ms':(b-a)/1e6,'touches_window_boundary':a==0 or b==width}
            if v:
                detail.update({'next_kernel_correlation':c,'next_api_name':names.get(str(v[4]),'UNKNOWN'), 'next_api_start_s':(v[0]-lo)/1e9,'next_api_end_s':(v[1]-lo)/1e9,'next_api_ended_before_gap_start':v[1]<=lo+a,'next_api_end_to_kernel_start_ms':(lo+b-v[1])/1e6})
            next_details.append(detail)
        report['devices'].append({'device':d,'kernel_rows_in_window':len(recs),'kernel_union_ms':length(kb)/1e6,'kernel_union_pct':100*length(kb)/width,'kernel_duration_sum_ms':sums[(1,d)]/1e6,'kernel_mean_node_duration_us':sums[(1,d)]/len(recs)/1e3,'kernel_union_segments':len(kb),'kernel_gaps':stat_ns([b-a for a,b in kg]),'kernel_gap_duration_bins':gap_bins([b-a for a,b in kg]),'memcpy_union_ms':length(bykind[2])/1e6,'memset_union_ms':length(bykind[3])/1e6,'all_traced_gpu_activity_union_ms':length(ab)/1e6,'all_traced_gpu_activity_union_pct':100*length(ab)/width,'all_activity_gaps':stat_ns([b-a for a,b in ag]),'top_kernel_gaps':next_details})
    hist,zero=active_distribution(kernel_busy,width);ahist,azero=active_distribution(any_busy,width)
    report['concurrency']={'kernel_devices_active_duration_ms':{str(i):v/1e6 for i,v in enumerate(hist)},'kernel_devices_active_pct':{str(i):100*v/width for i,v in enumerate(hist)},'all_activity_devices_active_pct':{str(i):100*v/width for i,v in enumerate(ahist)},'all_cards_no_observed_kernel_gaps':stat_ns([b-a for a,b in zero]),'all_cards_no_observed_any_activity_gaps':stat_ns([b-a for a,b in azero]),'longest_all_cards_no_kernel': [{'start_s':a/1e9,'end_s':b/1e9,'duration_ms':(b-a)/1e6} for a,b in sorted(zero,key=lambda v:v[1]-v[0],reverse=True)[:20]]}
    api_group=collections.defaultdict(list);thread_intervals=collections.defaultdict(list); seenlinked=set(linked_tids)
    for s,e,tid,c,name,ret in api:
        if e<=lo or s>=hi:continue
        # Threads are identified by unique observed correlations to owned-engine nodes.
        if tid not in seenlinked:continue
        cs=max(s,lo)-lo;ce=min(e,hi)-lo;api_group[name].append((cs,ce,tid,ret));thread_intervals[tid].append((cs,ce))
    report['host_api']={'attribution':'runtime threads with unique correlation-ID links to the only observed owned-engine GPU nodes; no guessed globalTid decoding. All GPU raw rows match total trace counts.','runtime_global_tids':dict(tids),'correlated_runtime_global_tids':dict(linked_tids),'per_thread_runtime_union_ms':{str(t):length(merge(v))/1e6 for t,v in thread_intervals.items()},'api_groups':[]}
    for name,v in api_group.items():
        ds=[e-s for s,e,_,_ in v]
        report['host_api']['api_groups'].append({'name':names.get(str(name),'UNKNOWN_'+str(name)),'name_id':name,'calls':len(v),'sum_ms':sum(ds)/1e6,'duration':stat_ns(ds),'nonzero_return_values':dict(collections.Counter(ret for s,e,t,ret in v if ret!=0))})
    report['host_api']['api_groups'].sort(key=lambda x:x['sum_ms'],reverse=True)
    report['correlation']={**dict(corrcount),'runtime_correlation_ids_with_multiple_rows':sum(len(v)>1 for v in apimap.values()),'first_gpu_node_minus_unique_api_end':stat_ns(delay),'graph_launch_first_node_minus_api_end':stat_ns(graph_delay),'negative_delays_retained':sum(v<0 for v in graph_delay),'semantics':'One launch maps to many graph nodes. Delays include graph dependency/queue latency and profiler overhead; no causal CPU stall claim.'}
    report['profiler']={'rows_in_window':len(other[7]),'overhead_interval_union_ms':length(merge((v[0],v[1]) for v in other[7]))/1e6,'warning':'Profiler overhead events do not measure all instrumentation overhead.'}
    report['limits']=['Observed union is not SM utilization, FLOPs, occupancy or bandwidth saturation.','Single injected node-level C8 diagnostic; no unprofiled throughput or improvement inference.','Cuda event trace off: graph dependency edge interpretation is incomplete.','Six incomplete CUPTI events dropped at stop; exact missing intervals unavailable.','CPU sampling/context switches disabled; host scheduling cause unavailable.','This C8 workload does not establish long-prefill overlap.']
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--receipt',type=Path);p.add_argument('--schema',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.self_test:self_test()
    else:
        out=analyze(a.input,a.receipt,a.schema)
        a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');a.out.chmod(0o600)
        print(json.dumps({'written':str(a.out),'window_s':out['window']['duration_s'],'device_count':len(out['devices'])}))
