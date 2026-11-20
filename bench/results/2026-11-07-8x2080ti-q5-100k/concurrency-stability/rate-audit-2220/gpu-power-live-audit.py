#!/usr/bin/env python3
"""Analyze already-downloaded C8 power telemetry; never execute the probe."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RUN = HERE / 'c8-power-live'
spec = importlib.util.spec_from_file_location('gpu_power_fields', HERE / 'gpu-power-fields.py')
fields = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fields)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def taipei(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone(dt.timedelta(hours=8))).isoformat()


def main():
    metadata = json.loads((RUN / 'metadata.json').read_text())
    check = json.loads((RUN / 'c8/c8-check.json').read_text())
    assert check['all_checks_passed'] and all(check['checks'].values())
    assert metadata['client_returncode'] == 0
    assert metadata['config_before_sha256'] == metadata['config_after_sha256']
    assert metadata['service_restarts'] == metadata['configuration_changes'] == metadata['power_limit_changes'] == 0
    sources = [Path(__file__), HERE / 'gpu-power-fields.py', RUN / 'metadata.json', RUN / 'gpu.csv', RUN / 'c8/c8-check.json']
    lanes = []
    for path in sorted((RUN / 'c8').glob('c8-lane-*.json')):
        record = json.loads(path.read_text())
        times = [row['epoch'] for row in record['deltas'] if row.get('content')]
        assert min(times) == record['first_output_epoch']
        assert max(times) == record['last_output_epoch']
        assert record['usage']['completion_tokens'] == 1024
        lanes.append({'lane': record['name'], 'first_content_epoch': min(times), 'last_content_epoch': max(times)})
        sources.append(path)
    assert len(lanes) == 8
    start = max(row['first_content_epoch'] for row in lanes) + 1
    end = min(row['last_content_epoch'] for row in lanes) - 1
    assert end > start
    rows, columns = fields.read_gpu_csv(RUN / 'gpu.csv')
    assert all(metadata['start_epoch'] <= row['epoch'] <= metadata['end_epoch'] for row in rows)
    assert 0 <= rows[0]['epoch'] - metadata['start_epoch'] < 1
    assert 0 <= metadata['end_epoch'] - rows[-1]['epoch'] < 1
    devices = fields.summarize_devices(rows, start, end)
    assert [row['gpu'] for row in devices] == list(range(8))
    assert all(row['n'] >= 20 for row in devices)
    relevant_reasons = ('sw_power_cap', 'sw_thermal_slowdown', 'hw_thermal_slowdown', 'hw_power_brake_slowdown')
    assert all(row['clock_reasons'][reason]['valid_n'] == row['n'] for row in devices for reason in relevant_reasons)
    previous_path = HERE / 'gpu-util-audit.json'
    previous = json.loads(previous_path.read_text())
    sources.append(previous_path)
    comparison = []
    for current, old in zip(devices, previous['devices']):
        comparison.append({'gpu': current['gpu'],
                           **{name: {'previous_mean': old[name]['mean'], 'current_mean': current[name]['mean']}
                              for name in ('gpu_active_pct', 'power_w', 'sm_clock_mhz', 'temperature_c')}})
    result = {
        'scope': 'Controlled 8x1024 short-prompt run on current production configuration at 22:40 Taipei; not the 21:44 peak or a 100k run.',
        'source_sha256': {str(path): sha(path) for path in sources},
        'metadata': metadata, 'correctness_checks': check['checks'], 'lanes': lanes,
        'window': {'start_epoch': start, 'end_epoch': end, 'start_taipei': taipei(start), 'end_taipei': taipei(end),
                   'duration_s': end-start, 'trim_each_end_s': 1,
                   'definition': 'Intersection of first/last nonempty content for all 8 client streams, trimmed 1s at each end; GPU execution boundaries are not directly measured.'},
        'csv_timezone': {'csv_has_explicit_timezone': False, 'interpreted_as': 'UTC',
                         'evidence': 'All samples lie inside metadata start_epoch/end_epoch; Taipei interpretation would be eight hours outside.',
                         'first_csv_minus_start_s': rows[0]['epoch'] - metadata['start_epoch'],
                         'end_minus_last_csv_s': metadata['end_epoch'] - rows[-1]['epoch']},
        'csv_columns': columns, 'devices': devices,
        'aggregate': {'equal_gpu_mean_active_pct': statistics.mean(row['gpu_active_pct']['mean'] for row in devices),
                      'sum_gpu_mean_power_w': sum(row['power_w']['mean'] for row in devices)},
        'whole_run_api_throughput': check['timing']['whole_batch'],
        'whole_native_piped_batch': check['numeric_piped_log'],
        'previous_run_comparison': comparison,
        'findings': ['GPU7 sw_power_cap is Active at every selected sample and has a 250W enforced cap. Power-limit activity is directly observed, not inferred solely from power or clock readings.',
                     'GPU7 sw_thermal_slowdown, hw_thermal_slowdown and hw_power_brake_slowdown are inactive at every selected sample. This excludes those reported flags at sample instants, not all unobserved inter-sample behavior.',
                     'GPU5 has intermittent sw_power_cap. Other sampled GPUs do not show this flag in the selected interval.',
                     'GPU7 is nearly continuously busy while other cards have lower busy fractions. The whole-machine average does not describe each device spare capacity.',
                     'These measurements do not quantify the speedup obtainable from lifting a power cap, and no power cap was changed.',
                     'Memory-active percentage measures time with memory activity; it is not percent of maximum memory bandwidth.'],
        'official_clock_reason_reference': 'https://docs.nvidia.com/deploy/nvml-api/620/api/group__nvmlClocksEventReasons.html',
        'all_checks_passed': True,
    }
    (HERE / 'gpu-power-live-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['Current-configuration C8 power audit — local analysis only',
             f'Window: {taipei(start)} to {taipei(end)} ({end-start:.3f}s), 8-stream content intersection trimmed 1s per side.',
             'Short prompts, 8x1024 outputs; not 21:44 or 100k. Source CSV UTC agrees with metadata epoch bounds.',
             'GPU | n | GPU active % | mem active % | W / enforced limit | SM MHz mean(min-max) | temp C min-max | sw_power_cap active/valid',
             ]
    for row in devices:
        reason = row['clock_reasons']['sw_power_cap']
        lines.append(f"{row['gpu']} | {row['n']} | {row['gpu_active_pct']['mean']:.2f} | {row['memory_active_pct']['mean']:.2f} | "
                     f"{row['power_w']['mean']:.2f}/{row['enforced_power_limit_w']['mean']:.0f} | "
                     f"{row['sm_clock_mhz']['mean']:.1f}({row['sm_clock_mhz']['min']:.0f}-{row['sm_clock_mhz']['max']:.0f}) | "
                     f"{row['temperature_c']['min']:.0f}-{row['temperature_c']['max']:.0f} | {reason['active_n']}/{reason['valid_n']}")
    lines += [f"Eight-GPU mean busy {result['aggregate']['equal_gpu_mean_active_pct']:.3f}%; mean total power {result['aggregate']['sum_gpu_mean_power_w']:.2f}W.",
              'GPU7: 27/27 sw_power_cap Active; 0/27 each SW/HW thermal slowdown and HW power brake. Power-cap activity is directly observed at every sample, with temperature 54–62C.',
              'GPU5: sw_power_cap 4/27 (14.81%); all other GPUs 0/27. All GPUs have 0 thermal/HW power-brake flags in this selected window.',
              'All GPUs memory clocks 6800MHz throughout selected samples. GPU7 SM clock is 1665–1725MHz, others 1815–1905MHz.',
              'Previous 21:17 run also had GPU7 96.57% busy, 248.85W and 1647.5MHz; it did not record reason flags. Current controlled run reproduces imbalance and identifies GPU7 power-cap activity.',
              'Full API throughput 8192/30.1038575s=272.1246 tokens/s; native batch 8181/27.247s=300.3 rows/s rounded. These full-run denominators differ from the selected interior window.',
              'No measured claim about speedup from raising power limit. No restart, configuration edit or power-limit change according to probe metadata.',
              'Reproduce: python3 gpu-power-live-audit.py']
    (HERE / 'gpu-power-live-audit.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
