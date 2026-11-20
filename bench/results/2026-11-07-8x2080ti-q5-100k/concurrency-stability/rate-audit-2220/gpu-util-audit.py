#!/usr/bin/env python3
"""Read existing local benchmark artifacts; write only numeric audit outputs."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

HERE = Path(__file__).resolve().parent
ARM = HERE.parent / 'head-reverse' / '01-on'
UTC = dt.timezone.utc
TAIPEI = dt.timezone(dt.timedelta(hours=8))
FIELDS = {
    'gpu_active_pct': 'utilization.gpu [%]',
    'memory_active_pct': 'utilization.memory [%]',
    'memory_used_mib': 'memory.used [MiB]',
    'sm_clock_mhz': 'clocks.current.sm [MHz]',
    'memory_clock_mhz': 'clocks.current.memory [MHz]',
    'power_w': 'power.draw [W]',
    'temperature_c': 'temperature.gpu',
}


def stamp(t, tz=UTC):
    return dt.datetime.fromtimestamp(t, tz).isoformat()


def summary(values):
    return {'n': len(values), 'mean': stats.mean(values), 'min': min(values),
            'max': max(values), 'median': stats.median(values)}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    meta = json.loads((ARM / 'metadata.json').read_text())
    check = json.loads((ARM / 'c8' / 'c8-check.json').read_text())
    sources = [ARM / 'metadata.json', ARM / 'gpu.csv', ARM / 'c8' / 'c8-check.json']
    lanes = []
    for path in sorted((ARM / 'c8').glob('c8-lane-*.json')):
        source = json.loads(path.read_text())
        arrivals = [row['epoch'] for row in source['deltas'] if row.get('content')]
        assert min(arrivals) == source['first_output_epoch']
        assert max(arrivals) == source['last_output_epoch']
        assert source['usage']['completion_tokens'] == 1024
        assert source['error'] is None and source['saw_done']
        lanes.append({'lane': source['name'], 'first_content_epoch': min(arrivals),
                      'last_content_epoch': max(arrivals), 'completion_tokens': 1024})
        sources.append(path)
    assert len(lanes) == 8
    common_start = max(row['first_content_epoch'] for row in lanes)
    common_end = min(row['last_content_epoch'] for row in lanes)
    start, end = common_start + 1, common_end - 1
    assert end > start
    rows = []
    with (ARM / 'gpu.csv').open() as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            epoch = dt.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=UTC).timestamp()
            rows.append({'epoch': epoch, 'gpu': int(row['index']),
                         **{name: float(re.match(r'[-\d.]+', row[column]).group())
                            for name, column in FIELDS.items()}})
    # The CSV has no timezone marker. UTC uniquely agrees with ready/finished
    # epochs; treating it as Taipei would move the entire recording eight hours away.
    assert 0 <= rows[0]['epoch'] - meta['ready_epoch'] < 1
    assert 0 <= meta['finished_epoch'] - rows[-1]['epoch'] < 1
    assert all(meta['ready_epoch'] <= row['epoch'] <= meta['finished_epoch'] for row in rows)
    selected = [row for row in rows if start <= row['epoch'] <= end]
    devices = []
    for gpu in range(8):
        samples = [row for row in selected if row['gpu'] == gpu]
        assert len(samples) >= 25
        devices.append({'gpu': gpu, 'n': len(samples), 'first_sample_epoch': samples[0]['epoch'],
                        'last_sample_epoch': samples[-1]['epoch'],
                        **{field: summary([row[field] for row in samples]) for field in FIELDS}})
    # Each complete nvidia-smi sweep has one sample per GPU; exclude incomplete
    # boundary sweeps as a cross-check on the per-card, timestamp-filtered result.
    assert len(rows) % 8 == 0
    sweeps = [rows[i:i+8] for i in range(0, len(rows), 8)]
    assert all([row['gpu'] for row in sweep] == list(range(8)) for sweep in sweeps)
    full_sweeps = [sweep for sweep in sweeps if all(start <= row['epoch'] <= end for row in sweep)]
    snapshot_path = HERE / 'snapshot-initial.json'
    snapshot = json.loads(snapshot_path.read_text())
    sources.append(snapshot_path)
    result = {
        'scope': 'Existing head-reverse/01-on short-prompt 8x1024 benchmark, not the 21:44 business peak and not a 100k benchmark.',
        'script_sha256': file_sha(Path(__file__)),
        'source_sha256': {str(path): file_sha(path) for path in sources},
        'engine_sha256': meta['engine_sha256'], 'config_env': meta['config']['env'],
        'source_timezone': {'csv_has_timezone_marker': False, 'interpreted_as': 'UTC',
                            'evidence': 'All CSV rows lie between metadata ready_epoch and finished_epoch under UTC; Taipei interpretation is eight hours outside.',
                            'first_csv_minus_ready_s': rows[0]['epoch'] - meta['ready_epoch'],
                            'finished_minus_last_csv_s': meta['finished_epoch'] - rows[-1]['epoch']},
        'lanes': lanes,
        'window': {'common_content_start_epoch': common_start, 'common_content_end_epoch': common_end,
                   'trim_each_end_s': 1, 'selected_start_epoch': start, 'selected_end_epoch': end,
                   'selected_start_taipei': stamp(start, TAIPEI), 'selected_end_taipei': stamp(end, TAIPEI),
                   'duration_s': end-start,
                   'scope': 'Intersection of client first/last nonempty content, trimmed 1s at each end; arrival timestamps are not exact GPU execution boundaries.'},
        'devices': devices,
        'aggregate': {'equal_device_mean_gpu_active_pct': stats.mean(d['gpu_active_pct']['mean'] for d in devices),
                      'sum_device_mean_power_w': sum(d['power_w']['mean'] for d in devices),
                      'complete_sweeps': len(full_sweeps),
                      'complete_sweep_mean_gpu_active_pct': summary([stats.mean(row['gpu_active_pct'] for row in sweep) for sweep in full_sweeps])},
        'whole_benchmark_throughput': check['timing']['whole_batch'],
        'whole_native_piped_batch': check['numeric_piped_log'],
        'snapshot_at_2222': {'time_taipei': stamp(snapshot['time'], TAIPEI),
                             'hardware': snapshot['hardware'],
                             'in_flight': snapshot['telemetry']['work']['active'],
                             'occupied_slots': snapshot['telemetry']['work']['occupied_slots'],
                             'history': {key: summary(snapshot['history'][key]) for key in ('gpu_util', 'gpu_power', 'gpu_temp')},
                             'history_scope': 'Last 60 samples around snapshot time; no per-item timestamps. Cannot reconstruct 21:44.'},
        'semantics': {
            'source': 'serve/telemetry.py:87 calls NVML nvmlDeviceGetUtilizationRates; .gpu only.',
            'aggregation': 'serve/telemetry.py:619 arithmetic mean of available GPUs, :621 max temperature, :616 summed power/memory.',
            'cadence': 'serve/telemetry.py:660 sample then :673 wait 1s; 60 saved samples. serve/web/app.js:205 polls 1s after previous response.',
            'meaning': 'GPU utilization is time with one or more kernels executing, not achieved FLOPS/SM occupancy/Tensor efficiency. Memory utilization is time with device-memory activity, not achieved bandwidth fraction.',
            'official_reference': 'https://docs.nvidia.com/deploy/nvml-api/620/api/structnvmlUtilization__t.html',
        },
        'limits': ['No per-card power caps or clock throttle reasons in this arm. Power-cap or thermal throttling cannot be definitively excluded.',
                   'GPU7 high busy time, high power and lower SM clocks support inspecting its critical path; they do not alone prove the cause of lower clocks.',
                   'The 22:22 snapshot sum power limit cannot establish each card power limit during the earlier benchmark.',
                   'The total-run 272.79 API tokens/s and 298.2 native emitted rows/s have different whole-run denominators; neither is a token count for this selected interior window.'],
        'all_checks_passed': True,
    }
    (HERE / 'gpu-util-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = [
        'GPU utilization audit — existing local artifacts only',
        f"Window: {stamp(start, TAIPEI)} to {stamp(end, TAIPEI)} ({end-start:.3f}s).",
        'All eight nonempty-content streams overlap; trim 1s at each end. Not 21:44; short prompts, not 100k.',
        'CSV UTC inferred and checked against metadata ready/finished epoch anchors.',
        'GPU | n | GPU active % | memory active % | mean power W | SM MHz mean(min-max) | memory MHz | temp C mean(max)',
    ]
    for device in devices:
        g = device
        lines.append(f"{g['gpu']} | {g['n']} | {g['gpu_active_pct']['mean']:.2f} | {g['memory_active_pct']['mean']:.2f} | "
                     f"{g['power_w']['mean']:.2f} | {g['sm_clock_mhz']['mean']:.1f}({g['sm_clock_mhz']['min']:.0f}-{g['sm_clock_mhz']['max']:.0f}) | "
                     f"{g['memory_clock_mhz']['mean']:.0f} | {g['temperature_c']['mean']:.1f}({g['temperature_c']['max']:.0f})")
    lines += [f"Equal-card mean GPU active: {result['aggregate']['equal_device_mean_gpu_active_pct']:.3f}%; total mean power: {result['aggregate']['sum_device_mean_power_w']:.2f}W.",
              'GPU7 is 96.57% busy while GPUs0–6 average 54.40–65.17%. Whole-machine average hides imbalance.',
              'GPU7 averages 248.85W and 1647.5MHz; its temperature is 65–69C. No throttle reason or per-card cap recorded, so power limiting cannot be ruled out or proven.',
              'NVML busy-time percentage is not compute-capacity percentage; memory active percentage is not bandwidth percentage.',
              'The separate 22:22:14 snapshot had 2 occupied slots, 25.25% mean GPU active and 1138.40W/2000W aggregate cap. Its 60-sample history max was 36.62%; no clocks/throttle reasons saved.',
              'No service, source implementation, or GPU changes; no inference requests.',
              'Reproduce: python3 gpu-util-audit.py']
    (HERE / 'gpu-util-audit.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
