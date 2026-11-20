#!/usr/bin/env python3
"""Analyze an owned, UTC-normalized Nsight SQLite export of one Strata prefill.

The request window must be captured on the profiling host. Only device/process
metadata, timing, and selected CUDA kernel/API names enter the derived JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import sqlite3
from bisect import bisect_left, bisect_right

DEVICES = tuple(range(8))
ROW_BYTES = 2560 * 4 * 4  # Flash-Next FP32 residual, four streams of width 2560.
CUPTI = 'CUPTI_ACTIVITY_KIND_'


@dataclass(frozen=True)
class Copy:
    begin: int
    end: int
    device: int
    stream: int
    count: int
    kind: str
    corr: int | None


def merge(intervals):
    result = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((begin, end))
    return result


def area(intervals):
    return sum(end - begin for begin, end in intervals)


def between(intervals, begin, end):
    return [(max(a, begin), min(b, end)) for a, b in intervals if a < end and b > begin]


def ms(ns):
    return round(ns / 1e6, 6)


def stat(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return {'count': 0}
    med = (values[(n - 1) // 2] + values[n // 2]) / 2
    return {'count': n, 'sum_ms': ms(sum(values)), 'median_ms': ms(med),
            'p95_ms': ms(values[(95 * n + 99) // 100 - 1]), 'max_ms': ms(values[-1])}


def histogram(per_device, lo, hi):
    edges = Counter({lo: 0, hi: 0})
    for intervals in per_device.values():
        for begin, end in intervals:
            edges[begin] += 1
            edges[end] -= 1
    times = [0] * 9
    active = 0
    last = lo
    for pos, delta in sorted(edges.items()):
        times[active] += pos - last
        active += delta
        last = pos
    assert active == 0 and sum(times) == hi - lo
    return [ms(t) for t in times]


@lru_cache(maxsize=None)
def category(name):
    name = name.lower()
    if 'prefill::fused' in name:
        if any(part in name for part in ('native_kernel', 'native_f16_kernel', 'native_ws_kernel')):
            return 'native_expert_matmul'
        return 'native_expert_prepare'
    if 'mul_mat_q' in name and 'mul_mat_vec_q' not in name:
        return 'mmq_matmul'
    if 'prefill::mmq' in name and any(part in name for part in ('copy16_', 'copy1_', 'strata_q2_kernel', 'prep_group_kernel')):
        return 'mmq_weight_gather'
    if 'prefill::' in name and '::route_kernel<' in name:
        return 'moe_route_marker'
    if 'prefill::' in name and '::moe_combine4_kernel(' in name:
        return 'moe_combine_marker'
    if 'moe' in name:
        return 'moe_gather_route_combine'
    if 'swiglu' in name or 'quantize_mmq' in name:
        return 'swiglu_or_mmq_quantize'
    if any(part in name for part in ('attention', '_attn_', 'qsa_', 'flash_attn', 'kv_cache', 'kv_append', 'append_kv', 'block_scores_', 'block_topk_', 'rms_rope_rows_')):
        return 'attention_qsa_kv'
    if any(part in name for part in ('gdn', 'delta_rule')):
        return 'gdn'
    if any(part in name for part in ('hyper_connection', 'gr_read', 'gr_write', 'gr_norm', 'gr_mix', 'gr_inj')):
        return 'hyper_connection'
    if any(part in name for part in ('gemm', 'gemv', 'matvec', 'mmvq', 'cublas')):
        return 'other_matmul'
    return 'other'


def direction(labels):
    value = re.sub('[^a-z0-9]', '', ' '.join(label or '' for label in labels).lower())
    if 'hosttodevice' in value or 'htod' in value or 'h2d' in value:
        return 'HtoD'
    if 'devicetohost' in value or 'dtoh' in value or 'd2h' in value:
        return 'DtoH'
    if 'devicetodevice' in value or 'dtod' in value or 'd2d' in value:
        return 'DtoD'
    if 'peertopeer' in value or 'ptop' in value or 'p2p' in value:
        return 'PtoP'
    return None


def source_options(source, cap, tokens):
    yield source
    full = [item for item in source if item.count == cap * ROW_BYTES]
    remaining = tokens - len(full) * cap
    if full and 0 < remaining <= cap:
        for tail in source:
            rows = tail.count // ROW_BYTES
            if remaining - 16 <= rows <= remaining and rows < cap:
                yield sorted(full + [tail], key=lambda item: item.begin)


def pair_handoffs(copies, cap, tokens):
    groups = defaultdict(list)
    for item in copies:
        if item.count % ROW_BYTES == 0 and 1 <= item.count // ROW_BYTES <= cap:
            groups[item.device, item.kind, item.stream].append(item)
    result = []
    report = []
    for dev in DEVICES[:-1]:
        best = None
        for (gpu, kind, ss), source_rows in groups.items():
            if gpu != dev or kind != 'DtoH':
                continue
            for source in source_options(source_rows, cap, tokens):
                rows = sum(a.count for a in source) // ROW_BYTES
                sizes = {item.count for item in source}
                for (gpu, kind, ds), dest_rows in groups.items():
                    if gpu != dev + 1 or kind != 'HtoD':
                        continue
                    dest = [item for item in dest_rows if item.count in sizes]
                    if len(source) != len(dest):
                        continue
                    if not all(a.count == b.count and a.end <= b.begin for a, b in zip(source, dest)):
                        continue
                    score = (abs(tokens - rows), -len(source), ss, ds)
                    if best is None or score < best[0]:
                        best = (score, source, dest)
        if best is None:
            result.append(None)
            report.append({'from': dev, 'to': dev + 1, 'status': 'no_matching_copy_series'})
            continue
        _, source, dest = best
        rows = [a.count // ROW_BYTES for a in source]
        result.append((source, dest))
        report.append({'from': dev, 'to': dev + 1,
                       'source_stream': source[0].stream, 'dest_stream': dest[0].stream,
                       'chunks': len(rows), 'total_rows': sum(rows), 'rows_per_chunk': rows,
                       'download': stat(a.end - a.begin for a in source),
                       'relay_gap': stat(b.begin - a.end for a, b in zip(source, dest)),
                       'upload': stat(b.end - b.begin for b in dest),
                       'handoff_span': stat(b.end - a.begin for a, b in zip(source, dest))})
    qualified = all(result) and all(
        report[i]['rows_per_chunk'] == report[0]['rows_per_chunk'] for i in range(len(result)))
    if qualified:
        qualified = all(report[i]['dest_stream'] == report[i + 1]['source_stream']
                        for i in range(len(result) - 1))
    if qualified:
        row_sum = report[0]['total_rows']
        qualified = len(report[0]['rows_per_chunk']) >= 8 and 0 <= tokens - row_sum <= 16
    return result, report, bool(qualified)


def stages_from_handoffs(edges, streams, copies, memsets, lo, hi):
    """Use ordered stage-copy milestones for an observed latest-predecessor walk.

    Stage 0's ingress and stage 7's egress use earliest/latest same-stream
    kernels, so presentation excludes exact admission/last-logit attribution.
    """
    count = len(edges[0][0])
    stage_nodes = []
    for dev in DEVICES:
        inc = edges[dev - 1][1] if dev else None
        out = edges[dev][0] if dev + 1 < len(DEVICES) else None
        stream = (inc or out)[0].stream
        kernels = streams.get((dev, stream), [])
        starts = [begin for begin, _, _ in kernels]
        ends = sorted(end for _, end, _ in kernels)
        values = []
        for chunk in range(count):
            if inc and out and inc[chunk].end > out[chunk].begin:
                return {'qualified': False, 'reason': 'stage copies overlap in the wrong order'}
            if inc:
                start = inc[chunk].begin
            else:
                floor = out[chunk - 1].end if chunk else lo
                index = bisect_left(starts, floor)
                if index == len(starts) or starts[index] >= out[chunk].begin:
                    return {'qualified': False, 'reason': 'missing stage-0 kernels between anchor copies'}
                start = starts[index]
            if out:
                end = out[chunk].end
            else:
                ceiling = inc[chunk + 1].begin if chunk + 1 < count else hi
                index = bisect_right(ends, ceiling)
                if not index or ends[index - 1] <= inc[chunk].end:
                    return {'qualified': False, 'reason': 'missing stage-7 kernels between anchor copies'}
                end = ends[index - 1]
            values.append((start, end))
        stage_nodes.append(values)
    result = []
    for dev, values in enumerate(stage_nodes):
        inc = edges[dev - 1][1] if dev else None
        out = edges[dev][0] if dev + 1 < len(DEVICES) else None
        data = {'device': dev, 'stream': (inc or out)[0].stream,
                'observed_stage_span': stat(e - s for s, e in values),
                'between_chunks': stat(s - pe for (_, pe), (s, _) in zip(values, values[1:]) if s >= pe),
                'first_stage_start_ms': ms(values[0][0] - lo), 'last_stage_end_ms': ms(values[-1][1] - lo)}
        if inc and out:
            data['between_input_and_output_copy'] = stat(o.begin - i.end for i, o in zip(inc, out))
        result.append(data)

    # End-to-start gaps are measured after the later of the two source-code
    # dependencies. The walk describes observed elapsed time, not hypothetical
    # speedup if a chunk or dependency were removed.
    dev, chunk = DEVICES[-1], count - 1
    component = Counter()
    kernel_components = Counter()
    residual_components = Counter()
    moe_windows = []
    moe_components = Counter()
    bad_moe_pairs = 0
    stage_streams = {}
    stage_other_copies = defaultdict(list)
    device_activity = defaultdict(list)
    anchors = {item for source,dest in edges for item in source + dest}
    for item in copies:
        device_activity[item.device].append((item.begin,item.end))
        if item not in anchors:
            stage_other_copies[item.device,item.stream].append((item.begin,item.end))
    for (gpu, _), rows in streams.items():
        device_activity[gpu].extend((a,b) for a,b,_ in rows)
    for gpu, rows in memsets.items():
        device_activity[gpu].extend(rows)
    device_activity = {gpu:merge(rows) for gpu,rows in device_activity.items()}
    for dev_id, items in enumerate(result):
        kernel_rows = streams[dev_id,items['stream']]
        stage_streams[dev_id] = ([a for a,_,_ in kernel_rows], kernel_rows)
    path_steps = []
    while True:
        start, end = stage_nodes[dev][chunk]
        inc = edges[dev - 1][1][chunk] if dev else None
        out = edges[dev][0][chunk] if dev + 1 < len(DEVICES) else None
        copy_ns = sum(item.end - item.begin for item in (inc, out) if item)
        copy_iv = [(item.begin,item.end) for item in (inc,out) if item]
        stream = (inc or out).stream
        starts, kernel_rows = stage_streams[dev]
        first = max(0, bisect_left(starts,start) - 1)
        last = bisect_left(starts,end)
        intervals = defaultdict(list)
        for a,b,c in kernel_rows[first:last]:
            if b > start:
                intervals[c].append((max(a,start),min(b,end)))
        kernel_iv = merge(v for group in intervals.values() for v in group)
        kernel_ns = area(kernel_iv)
        for c, group in intervals.items():
            kernel_components[c] += area(merge(group))
        if kernel_ns + copy_ns > end - start:
            return {'qualified': False, 'reason': 'handoff and kernels overlap on selected stage stream'}
        component['same_stream_kernels_ns'] += kernel_ns
        component['handoff_copy_ns'] += copy_ns
        component['other_stage_span_ns'] += end - start - copy_ns - kernel_ns
        same_stream_nonhandoff = between(stage_other_copies[dev,stream],start,end)
        same_stream_total = area(merge(kernel_iv + copy_iv + same_stream_nonhandoff))
        any_work = area(between(device_activity[dev],start,end))
        residual_components['same_stream_nonhandoff_copy'] += same_stream_total - copy_ns - kernel_ns
        residual_components['overlap_stripped_other_observed_device_work'] += any_work - same_stream_total
        residual_components['no_observed_work_on_stage_device'] += end - start - any_work
        route_end = None
        for a,b,c in kernel_rows[first:last]:
            if c == 'moe_route_marker':
                bad_moe_pairs += route_end is not None
                route_end = b
            elif c == 'moe_combine_marker':
                if route_end is None or a < route_end:
                    bad_moe_pairs += 1
                else:
                    moe_windows.append(a - route_end)
                    moe_kernel_ns = area(between(kernel_iv,route_end,a))
                    moe_visible_ns = area(between(device_activity[dev],route_end,a))
                    moe_components['same_stream_kernels'] += moe_kernel_ns
                    moe_components['overlap_stripped_other_observed_device_work'] += moe_visible_ns - moe_kernel_ns
                    moe_components['no_observed_work_on_stage_device'] += a - route_end - moe_visible_ns
                route_end = None
        bad_moe_pairs += route_end is not None
        pred = []
        if dev:
            pred.append((stage_nodes[dev - 1][chunk][1], dev - 1, chunk, 'after_previous_stage_ns'))
        if chunk:
            pred.append((stage_nodes[dev][chunk - 1][1], dev, chunk - 1, 'after_previous_chunk_ns'))
        path_steps.append([dev, chunk, ms(start - lo), ms(end - lo)])
        if not pred:
            component['before_first_stage_kernel_ns'] = start - lo
            break
        available, dev, chunk, key = max(pred)
        # Ordering can be shifted by unobserved async host work. Keep the path
        # unavailable if the visible trace contradicts this dependency model.
        if start < available:
            return {'qualified': False, 'reason': 'latest source dependency does not precede visible stage work'}
        component[key] += start - available
    component['after_last_stage_kernel_ns'] = hi - stage_nodes[-1][-1][1]
    assert sum(component.values()) == hi - lo
    assert sum(residual_components.values()) == component['other_stage_span_ns']
    assert sum(moe_components.values()) == sum(moe_windows)
    return {'qualified': True, 'by_stage': result,
            'observed_path_components_ms': {k.removesuffix('_ns'): ms(v) for k, v in component.items()},
            'path_kernel_categories_ms': {k:ms(v) for k,v in kernel_components.most_common()},
            'stage_residual_breakdown_ms': {k:ms(v) for k,v in residual_components.items()},
            'router_done_to_moe_combine_start': {'span':stat(moe_windows), 'unmatched_markers':bad_moe_pairs,
                'all_six_layers_per_path_step':len(moe_windows)==6*len(path_steps) and bad_moe_pairs==0,
                'breakdown_ms':{k:ms(v) for k,v in moe_components.items()},
                'includes':'GPU expert setup/matmul, the shared expert, routed-table transfers and dispatch gaps.'},
            'observed_path_steps': path_steps[::-1],
            'interpretation': 'Inferred from ordered stage copy milestones and the latest two known predecessors. Endpoint stages use kernel boundaries. Residual spans can contain CPU submission, runtime overhead, or work on other streams. This is not a CUDA DAG or an avoidable-time estimate.'}


def analyze(args):
    lo, hi = args.start_ns, args.end_ns
    if not 0 < lo < hi:
        raise ValueError('Invalid request epoch bounds')
    with sqlite3.connect(args.sqlite.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON')
        tables = {row[0] for row in db.execute('SELECT name FROM sqlite_master WHERE type="table"')}
        process = db.execute('SELECT globalPid FROM PROCESSES WHERE pid=?', (args.pid,)).fetchall()
        if len(process) != 1:
            raise ValueError('Trace must identify exactly one owned engine process')
        pid = process[0][0]
        mapping = dict(db.execute('SELECT gpuId,cudaId FROM TARGET_INFO_CUDA_DEVICE WHERE pid=?', (args.pid,)))
        if sorted(mapping.values()) != list(DEVICES):
            raise ValueError('Trace must map all eight CUDA stage ordinals for the engine')
        where = 'globalPid=? AND end>? AND start<?'
        params = pid, lo, hi
        kernel_range = db.execute(f'SELECT min(start),max(end) FROM {CUPTI}KERNEL WHERE globalPid=?', (pid,)).fetchone()
        if kernel_range[0] is None or not lo <= kernel_range[1] or not kernel_range[0] <= hi:
            raise ValueError('Kernel timestamps do not intersect the UTC request window')
        ks = defaultdict(list)
        names = defaultdict(Counter)
        grouped = defaultdict(list)
        for d, stream, a, b, name in db.execute(f'''SELECT deviceId,streamId,start,end,s.value
             FROM {CUPTI}KERNEL k LEFT JOIN StringIds s ON s.id=k.demangledName
             WHERE {where} ORDER BY start''', params):
            dev = mapping[d]
            a, b = max(lo, a), min(hi, b)
            c = category(name or '')
            ks[dev, stream].append((a, b, c))
            names[c][name or 'unknown'] += b - a
            grouped[dev, c].append((a, b))
        copies = []
        optional_rows = defaultdict(list)
        labels = {}
        if 'ENUM_CUDA_MEMCPY_OPER' in tables:
            labels = {i: direction((n, label)) for i, n, label in db.execute('SELECT id,name,label FROM ENUM_CUDA_MEMCPY_OPER')}
        if CUPTI + 'MEMCPY' in tables:
            for d, stream, a, b, size, code, corr in db.execute(f'''SELECT deviceId,streamId,start,end,bytes,copyKind,correlationId
                 FROM {CUPTI}MEMCPY WHERE {where} ORDER BY start''', params):
                kind = labels.get(code) or {1:'HtoD', 2:'DtoH', 8:'DtoD', 10:'PtoP'}.get(code, f'copyKind{code}')
                if d in mapping:
                    copies.append(Copy(a,b,mapping[d],stream,size,kind,corr))
        if CUPTI + 'MEMSET' in tables:
            for d, a, b in db.execute(f'SELECT deviceId,start,end FROM {CUPTI}MEMSET WHERE {where}', params):
                if d in mapping:
                    optional_rows[mapping[d]].append((a,b))
        if set(d for d,_ in ks) != set(DEVICES):
            raise ValueError('The request window lacks kernel rows on one or more mapped devices')
        diagnostics = []
        if 'DIAGNOSTIC_EVENT' in tables:
            diagnostics = list(db.execute('''SELECT severity,count(*),sum(lower(text) LIKE '%lost%'
                OR lower(text) LIKE '%dropped%' OR lower(text) LIKE '%incomplete%'
                OR lower(text) LIKE '%not all cuda events%')
                FROM DIAGNOSTIC_EVENT WHERE globalPid=? OR globalPid IS NULL GROUP BY severity''', (pid,)))
        edges, edge_stats, qualified = pair_handoffs(copies, args.chunk_tokens, args.prompt_tokens)
        if qualified and CUPTI + 'RUNTIME' in tables:
            corrs = {c.corr for edge in edges for row in edge for c in row if c.corr is not None}
            apis = defaultdict(list)
            for corr, begin, end in db.execute(f'''SELECT r.correlationId,r.start,r.end
                 FROM {CUPTI}RUNTIME r JOIN StringIds s ON s.id=r.nameId
                 WHERE r.start<? AND r.end>? AND s.value LIKE 'cudaMemcpy%' ''', (hi,lo)):
                if corr in corrs:
                    apis[corr].append((begin,end))
            for stats, (source, dest) in zip(edge_stats, edges):
                relay_to_submit = [apis[b.corr][0][0] - a.end for a,b in zip(source,dest) if len(apis[b.corr])==1]
                stats['download_end_to_unique_upload_submit'] = stat(relay_to_submit)
                stats['upload_submit_before_download_end'] = sum(v<0 for v in relay_to_submit)
    gpu_data = []
    all_kernel = {}
    all_activity = {}
    for dev in DEVICES:
        cat_stats = {}
        expert_spans = defaultdict(list)
        for (d, c), seq in sorted(grouped.items()):
            if d == dev:
                cat_stats[c] = {'count': len(seq), 'sum_ms': ms(area(seq)), 'union_ms': ms(area(merge(seq)))}
                if c in ('mmq_matmul','native_expert_matmul'):
                    expert_spans['matmul'].extend(seq)
                if c in ('native_expert_prepare','mmq_weight_gather','swiglu_or_mmq_quantize','moe_gather_route_combine',
                         'moe_route_marker','moe_combine_marker'):
                    expert_spans['prepare_or_combine'].extend(seq)
        kernel = merge([(a,b) for (d,s), items in ks.items() if d==dev for a,b,c in items])
        all_kernel[dev] = kernel
        copy = defaultdict(list)
        for x in copies:
            if x.device == dev:
                copy[x.kind].append(x)
        copy_iv = merge(between([(x.begin,x.end) for seq in copy.values() for x in seq],lo,hi))
        with_copy = merge(kernel + copy_iv + between(optional_rows[dev],lo,hi))
        all_activity[dev] = with_copy
        gaps = [(a,x) for (_,a),(x,_) in zip([(lo,lo)]+with_copy,with_copy+[(hi,hi)]) if x>a]
        gpu_data.append({'device': dev, 'kernel_union_ms': ms(area(kernel)), 'kernel_sum_ms': round(sum(c['sum_ms'] for c in cat_stats.values()),6),
                         'kernel_pct': round(100*area(kernel)/(hi-lo),4), 'copy_union_ms': ms(area(copy_iv)),
                         'any_traced_activity_ms': ms(area(with_copy)), 'any_activity_gaps': stat(b-a for a,b in gaps),
                         'longest_any_activity_gaps': [[ms(a-lo),ms(b-lo)] for a,b in sorted(gaps,key=lambda v:v[1]-v[0],reverse=True)[:8]],
                         'copy_by_kind': {c:{'count':len(seq),'bytes':sum(x.count for x in seq),
                                          'union_ms':ms(area(merge(between([(x.begin,x.end) for x in seq],lo,hi))))} for c,seq in sorted(copy.items())},
                         'categories': cat_stats,
                         'name_selected_moe_union_ms': {c:ms(area(merge(seq))) for c,seq in expert_spans.items()}})
    bounded_names = {c:[{'name':n[:220],'sum_ms':ms(t)} for n,t in seq.most_common(6)] for c,seq in names.items()}
    return {'window_start_ns':lo,'window_end_ns':hi,'window_ms':ms(hi-lo), 'engine_pid':args.pid,
            'kernel_extent_relative_ms':[ms(t-lo) for t in kernel_range], 'device_mapping':mapping,
            'diagnostics_severity_count_suspected_loss':diagnostics,'per_gpu':gpu_data,
            'kernel_active_gpu_histogram_ms':histogram(all_kernel,lo,hi),
            'any_activity_gpu_histogram_ms':histogram(all_activity,lo,hi),
            'representative_kernel_names_by_category':bounded_names,
            'handoff_series_qualified':qualified,'stage_handoffs':edge_stats,
            'pipeline_path': stages_from_handoffs(edges,ks,copies,optional_rows,lo,hi) if qualified else {'qualified':False},
            'limitations': ['Copy anchors are inferred from repeated FP32 residual bytes, order and stage stream consistency. Relay gap can include waiting for other work on the receiving stage.',
                            'Kernel categories are name based. An MMQ kernel name can occur outside routed experts. Kernel sums across GPUs overlap.',
                            'Device fractions are traced time coverage. Lost events and profiling overhead can change observed intervals and timing.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sqlite',type=Path)
    p.add_argument('--receipt',type=Path,help='Per-arm capture receipt with owned engine PID and timestamp fields')
    p.add_argument('--pid',type=int)
    p.add_argument('--start-ns',type=int)
    p.add_argument('--end-ns',type=int)
    p.add_argument('--prompt-tokens',type=int,default=100000)
    p.add_argument('--chunk-tokens',type=int,default=3072)
    p.add_argument('--out',type=Path)
    args = p.parse_args()
    if args.receipt:
        receipt = json.loads(args.receipt.read_bytes())
        request = receipt['traced']
        args.pid = args.pid or receipt['engine_pid']
        args.start_ns = args.start_ns or request['request_epoch_ns']
        args.end_ns = args.end_ns or request['first_sse_epoch_ns']
        if not receipt['capture_start_epoch_ns'] <= args.start_ns < args.end_ns <= receipt['capture_end_epoch_ns']:
            p.error('The requested window must lie inside the recorded trace window')
    if args.pid is None or args.start_ns is None or args.end_ns is None:
        p.error('Specify --receipt or all of --pid, --start-ns and --end-ns')
    result = json.dumps(analyze(args),indent=2,allow_nan=False) + '\n'
    if args.out:
        with args.out.open('x') as f:
            f.write(result)
    else:
        print(result,end='')


if __name__ == '__main__':
    main()
