#!/usr/bin/env python3
"""Summarize nvidia-smi activity within measured HTTP phases (offline only).

The benchmark host uses UTC. Activity samples are not achieved FLOPS, memory
bandwidth or a kernel timeline, and do not establish full hardware saturation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def value(text):
    text = text.strip()
    if text in ('Active', 'Not Active'):
        return float(text == 'Active')
    try:
        return float(text.split()[0])
    except (ValueError, IndexError):
        return None


def samples(path):
    out = []
    with path.open() as f:
        for raw in csv.DictReader(f, skipinitialspace=True):
            clean = {k.strip().split(' [')[0]: v.strip() for k, v in raw.items()}
            stamp = clean.pop('timestamp')
            t = datetime.strptime(stamp, '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            data = {k: value(v) for k, v in clean.items()}
            out.append({'epoch': t, **data})
    return out


def stats(values):
    xs = sorted(x for x in values if x is not None)
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'mean': statistics.fmean(xs), 'min': xs[0], 'max': xs[-1],
            'p10': xs[max(0, math.ceil(.1 * len(xs)) - 1)],
            'p95': xs[max(0, math.ceil(.95 * len(xs)) - 1)]}


def windows(result):
    rows = result.get('records', [])
    out = {}
    if rows:
        out['whole_case'] = (min(r['start_epoch'] for r in rows), max(r['end_epoch'] for r in rows))
    for name, channels in result.get('phases', {}).items():
        p = channels.get('visible_content', channels.get('output', channels))
        if 'begin_epoch' in p:
            out[name] = (p['begin_epoch'], p['end_epoch'])
    for r in rows:
        if r.get('role') == 'injected' and r.get('ttft_s') is not None:
            out['admission_' + r['name']] = (r['start_epoch'], r['start_epoch'] + r['ttft_s'])
    return out


def summarize(rows, begin, end):
    selected = [r for r in rows if begin <= r['epoch'] < end]
    cards = []
    for gpu in range(8):
        own = [r for r in selected if r['index'] == gpu]
        fields = sorted({k for r in own for k in r} - {'index', 'epoch'})
        times = sorted(r['epoch'] for r in own)
        cards.append({'gpu': gpu, 'samples': len(own),
                      'first_sample_epoch': times[0] if times else None,
                      'last_sample_epoch': times[-1] if times else None,
                      'maximum_sampling_gap_s': max((b-a for a,b in zip(times,times[1:])), default=None),
                      'fields': {f: stats([r.get(f) for r in own]) for f in fields}})
    return {'begin_epoch': begin, 'end_epoch': end, 'duration_s': end-begin,
            'all_eight_sampled': all(c['samples'] > 0 for c in cards), 'cards': cards,
            'mean_activity_across_cards_pct': statistics.fmean(
                c['fields']['utilization.gpu']['mean'] for c in cards)
                if all(c['fields'].get('utilization.gpu', {}).get('n') for c in cards) else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu-csv', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists(), 'Refuse to overwrite analysis'
    data = samples(a.gpu_csv)
    phases = windows(json.loads(a.results.read_text()))
    assert data and phases, 'Need timestamped GPU samples and measured HTTP windows'
    report = {'gpu_csv_sha256': hashlib.sha256(a.gpu_csv.read_bytes()).hexdigest(),
              'result_sha256': hashlib.sha256(a.results.read_bytes()).hexdigest(),
              'timestamp_timezone': 'UTC (verified host clock)',
              'sampling': '500 ms nominal; arithmetic sample mean, nearest-rank percentiles',
              'limits': ['GPU utilization is sampled activity, not a fraction of achieved FLOPS or memory bandwidth.',
                         'Samples have rolling measurement windows; phase boundaries are approximate.',
                         'Clock lock is a requested target; measured SM clocks and throttle reasons describe actual behavior.'],
              'phases': {name: summarize(data, *bounds) for name, bounds in phases.items()}}
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: r['mean_activity_across_cards_pct'] for name, r in report['phases'].items()}))


if __name__ == '__main__':
    main()
