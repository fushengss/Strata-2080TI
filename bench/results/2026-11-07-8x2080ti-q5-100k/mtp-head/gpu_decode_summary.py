#!/usr/bin/env python3
"""CPU-only alignment of preserved request arrivals and nvidia-smi samples.

Example: python3 gpu_decode_summary.py --root pilot35
Writes an additional JSON; never invokes CUDA, nvidia-smi, a server, or the harness.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

METRICS = {
    'utilization.gpu [%]': 'gpu_utilization_pct',
    'utilization.memory [%]': 'memory_utilization_pct',
    'clocks.current.sm [MHz]': 'sm_clock_mhz',
    'clocks.current.memory [MHz]': 'memory_clock_mhz',
    'power.draw [W]': 'power_w',
    'temperature.gpu': 'temperature_c',
}
LIMITS = [
    'Decode bounds are first-to-last nonempty content arrival at the local HTTP client, not GPU kernel timestamps.',
    'The nominal query interval is 500 ms; actual per-device intervals and sample counts are reported.',
    'nvidia-smi utilization/power readings may describe a preceding device sampling interval; their exact support is not recorded.',
    'GPU utilization is not SM occupancy, memory bandwidth, critical-path time, or proof of hardware saturation.',
    'Means/medians describe sampled readings only. No peak utilization, peak power, energy, or kernel-overlap inference is supported.',
    'An interior window excludes the configured guard at both ends to reduce phase-boundary contamination, without guaranteeing its removal.',
    'Pooled means weight each captured sample equally; devices are queried sequentially, not at one simultaneous instant.',
    'Timezone is resolved from monitor lifecycle timestamps on the same host; no clock-jump correction or interpolation is applied.',
]

def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(value):
    try:
        number = float(value.strip().split()[0])
    except (ValueError, IndexError):
        return None
    return number if math.isfinite(number) else None


def statistics_of(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'sample_count': 0, 'mean': None, 'median': None}
    return {'sample_count': len(values), 'mean': statistics.fmean(values),
            'median': statistics.median(values)}


def summarize_samples(rows):
    result = {'sample_count': len(rows),
              'first_sample_epoch': rows[0]['epoch'] if rows else None,
              'last_sample_epoch': rows[-1]['epoch'] if rows else None}
    for metric in METRICS.values():
        result[metric] = statistics_of([row[metric] for row in rows])
    return result


def read_gpu(path):
    rows = []
    with path.open(newline='') as handle:
        for lineno, source in enumerate(csv.DictReader(handle, skipinitialspace=True), 2):
            if not source.get('timestamp') or source['timestamp'].strip() == 'timestamp':
                continue  # nvidia-smi can repeat the header.
            try:
                naive = datetime.strptime(source['timestamp'].strip(), '%Y/%m/%d %H:%M:%S.%f')
                row = {'raw_epoch_as_utc': naive.replace(tzinfo=timezone.utc).timestamp(),
                       'gpu': int(source['index']), 'csv_line': lineno}
                row.update({dest: finite_number(source[src]) for src, dest in METRICS.items()})
            except (KeyError, ValueError) as error:
                raise ValueError(f'{path}:{lineno}: invalid GPU sample: {error}') from error
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no GPU rows')
    return rows


def align_clock(rows, metadata, explicit_offset):
    ready, finished = float(metadata['ready_epoch']), float(metadata['finished_epoch'])
    first_raw = min(row['raw_epoch_as_utc'] for row in rows)
    offsets = [explicit_offset] if explicit_offset is not None else range(-14 * 60, 14 * 60 + 1, 15)
    candidates = [offset for offset in offsets if 0 <= first_raw - offset * 60 - ready <= 5]
    if len(candidates) != 1:
        raise ValueError('Cannot uniquely align GPU wall time to ready_epoch within 5 s; refusing to guess timezone')
    offset = candidates[0]
    for row in rows:
        row['epoch'] = row['raw_epoch_as_utc'] - offset * 60
    first, last = min(row['epoch'] for row in rows), max(row['epoch'] for row in rows)
    if not -1 <= finished - last <= 2:
        raise ValueError('GPU monitor endpoint differs from finished_epoch by more than lifecycle tolerance')
    return {'utc_offset_minutes': offset,
            'method': 'explicit offset with lifecycle checks' if explicit_offset is not None else 'unique 15-minute UTC offset satisfying same-host monitor-start bound',
            'ready_epoch': ready, 'finished_epoch': finished,
            'first_gpu_sample_minus_ready_s': first - ready,
            'finished_minus_last_gpu_sample_s': finished - last}


def decode_bounds(request):
    if request.get('error') or not request.get('saw_done'):
        raise ValueError('Request failed or did not finish; excluded rather than treating partial output as decode')
    deltas = request.get('deltas', [])
    if any(delta.get('reasoning') for delta in deltas):
        raise ValueError('Reasoning content present: selected cold/warm workload is not a content-only window')
    content = [delta for delta in deltas if delta.get('content')]
    if len(content) < 2:
        raise ValueError('Need at least two nonempty content arrivals')
    first, last = float(content[0]['epoch']), float(content[-1]['epoch'])
    start = float(request['start_epoch'])
    if not last > first:
        raise ValueError('Nonpositive decode window')
    if abs(first - start - request['ttft_s']) > 1e-5 or abs(last - start - request['last_content_s']) > 1e-5:
        raise ValueError('Content epochs and request relative times disagree')
    previous = first
    for delta in content:
        if float(delta['epoch']) < previous or abs(float(delta['epoch']) - start - delta['elapsed_s']) > 1e-5:
            raise ValueError('Invalid/nonmonotonic content timestamps')
        previous = float(delta['epoch'])
    if not start <= first < last <= float(request['end_epoch']):
        raise ValueError('Content window is outside request lifetime')
    if request.get('checks', {}).get('isolated_activity') is not True:
        raise ValueError('Request lacks a confirmed isolated-activity check')
    return first, last


def analyze(root, guard, explicit_offset):
    output = {'schema_version': 1, 'root': str(root), 'edge_guard_s': guard,
              'nominal_query_interval_ms': 500, 'limitations': LIMITS, 'arms': [], 'requests': [], 'pooled': []}
    pools = defaultdict(list)
    gpu_paths = sorted(root.glob('*/gpu.csv'))
    if not gpu_paths:
        raise ValueError(f'{root}: no arm/gpu.csv files')
    for gpu_path in gpu_paths:
        arm_dir = gpu_path.parent
        meta_path = arm_dir / 'metadata.json'
        metadata = read_json(meta_path)
        rows = read_gpu(gpu_path)
        clock = align_clock(rows, metadata, explicit_offset)
        by_gpu = defaultdict(list)
        for row in rows:
            by_gpu[row['gpu']].append(row)
        sampling = {}
        for gpu, samples in sorted(by_gpu.items()):
            gaps = [b['epoch'] - a['epoch'] for a, b in zip(samples, samples[1:])]
            if not gaps or min(gaps) <= 0:
                raise ValueError(f'{arm_dir.name}/GPU{gpu}: invalid sampling chronology')
            sampling[str(gpu)] = {'total_sample_count': len(samples), 'median_interval_s': statistics.median(gaps),
                                  'largest_gap_s': max(gaps), 'gaps_over_2s': sum(gap > 2 for gap in gaps)}
        arm_record = {'tag': arm_dir.name, 'arm': metadata['arm'], 'clock_alignment': clock,
                      'sampling_by_gpu': sampling, 'source_gpu_csv_sha256': digest(gpu_path),
                      'source_metadata_sha256': digest(meta_path)}
        output['arms'].append(arm_record)
        intervals = []
        request_paths = sorted(path for path in arm_dir.glob('*.json') if path.name.startswith(('cold-', 'warm-')))
        if not request_paths:
            raise ValueError(f'{arm_dir}: no cold/warm request JSON')
        for path in request_paths:
            request = read_json(path)
            first, last = decode_bounds(request)
            if not clock['ready_epoch'] <= request['start_epoch'] <= first < last <= clock['finished_epoch']:
                raise ValueError(f'{path}: request falls outside arm lifecycle')
            intervals.append((first, last, path.name))
            result = {'source': str(path.relative_to(root)), 'source_sha256': digest(path),
                      'tag': arm_dir.name, 'arm': metadata['arm'], 'name': request['name'],
                      'category': request['category'], 'first_content_epoch': first,
                      'last_content_epoch': last, 'decode_window_s': last - first,
                      'completion_tokens': request['usage'].get('completion_tokens'), 'gpus': {}}
            for gpu, samples in sorted(by_gpu.items()):
                if samples[0]['epoch'] > first or samples[-1]['epoch'] < last:
                    raise ValueError(f'{path}/GPU{gpu}: monitor does not cover the full decode window')
                summary = {}
                for view, begin, end in (('full', first, last), ('interior', first + guard, last - guard)):
                    selected = [row for row in samples if begin <= row['epoch'] <= end]
                    summary[view] = summarize_samples(selected)
                    summary[view]['selection_start_epoch'] = begin
                    summary[view]['selection_end_epoch'] = end
                    for category in (request['category'], 'all-cold-warm'):
                        pools[(arm_dir.name, metadata['arm'], category, gpu, view)].extend(selected)
                result['gpus'][str(gpu)] = summary
            output['requests'].append(result)
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f'{arm_dir.name}: selected decode windows overlap; pooled samples would double count')
    for (tag, arm, category, gpu, view), rows in sorted(pools.items()):
        output['pooled'].append({'tag': tag, 'arm': arm, 'category': category, 'gpu': gpu, 'view': view,
                                 **summarize_samples(sorted(rows, key=lambda row: row['epoch']))})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Extracted pilot or ABBA run directory')
    parser.add_argument('--output', type=Path, help='Defaults to root/gpu-decode-summary.json')
    parser.add_argument('--edge-guard-s', type=float, default=1.0)
    parser.add_argument('--gpu-utc-offset-minutes', type=int, help='Optional explicit CSV timezone; still validates lifecycle')
    args = parser.parse_args()
    if not math.isfinite(args.edge_guard_s) or args.edge_guard_s < 0:
        parser.error('edge guard must be finite and nonnegative')
    root = args.root.resolve()
    result = analyze(root, args.edge_guard_s, args.gpu_utc_offset_minutes)
    result['analysis_script_sha256'] = digest(Path(__file__))
    destination = args.output or root / 'gpu-decode-summary.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(destination), 'arms': len(result['arms']), 'requests': len(result['requests']),
                      'status': 'aligned sampled readings; not peak or kernel utilization'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
