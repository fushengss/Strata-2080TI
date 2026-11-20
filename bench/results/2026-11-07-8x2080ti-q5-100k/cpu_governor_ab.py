#!/usr/bin/env python3
"""Client-only CPU governor ABBA. Restores every original governor in finally.

Requires root for scaling_governor writes. Does not restart the service, change
frequency limits, or change model/server settings. ab_client.py must be adjacent.
"""
import argparse
import csv
import json
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ab_client import Bench, TASKS, filler, save_json


class Tee:
    def __init__(self, original, file):
        self.original, self.file = original, file
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.original.write(text)
            self.file.write(text)
            self.file.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.original.flush()
            self.file.flush()


def read_freq():
    for path in (Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
                 Path('/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq')):
        try:
            return int(path.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


class Monitor:
    def __init__(self, bench, name):
        self.bench, self.name = bench, name
        self.samples, self.activity = [], []
        self.stop = threading.Event()
        self.freq_thread = threading.Thread(target=self.frequency_loop, daemon=True)
        self.status_thread = threading.Thread(target=self.status_loop, daemon=True)

    def frequency_loop(self):
        with (self.bench.out / (self.name + '-cpu.csv')).open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('epoch', 'cpu0_scaling_cur_freq_khz'))
            while not self.stop.is_set():
                row = {'epoch': time.time(), 'khz': read_freq()}
                self.samples.append(row)
                writer.writerow((row['epoch'], row['khz']))
                f.flush()
                self.stop.wait(0.2)
            f.flush()
            os.fsync(f.fileno())

    def status_loop(self):
        while not self.stop.is_set():
            self.activity.append({'epoch': time.time(), 'activity': self.bench.status()})
            self.stop.wait(0.5)

    def __enter__(self):
        self.freq_thread.start()
        self.status_thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.freq_thread.join(6)
        self.status_thread.join(6)
        save_json(self.bench.out / (self.name + '-monitor.json'),
                  {'frequency_samples': self.samples, 'activity_samples': self.activity})


def frequency_stats(samples, begin, end):
    values = [s['khz'] for s in samples if s['khz'] is not None and begin <= s['epoch'] <= end]
    return {'sample_count': len(values),
            'mean_mhz': statistics.mean(values) / 1000 if values else None,
            'min_mhz': min(values) / 1000 if values else None,
            'max_mhz': max(values) / 1000 if values else None}


def measured_request(bench, name, messages, limit, category):
    before = bench.status()
    with Monitor(bench, name) as monitor:
        r = bench.request(name, messages, limit, category)
    after = bench.status()
    first = r['start_epoch'] + (r['ttft_s'] or 0)
    last = r['start_epoch'] + (r['last_content_s'] or r['total_s'])
    activities = [s['activity'] for s in monitor.activity]
    count_before, count_after = before.get('requests'), after.get('requests')
    delta = count_after - count_before if isinstance(count_before, int) and isinstance(count_after, int) else None
    errors = [s for s in activities if 'error' in s or 'in_flight' not in s]
    competing = any(s.get('in_flight', 0) > 1 for s in activities)
    uncontaminated = (before.get('in_flight') == 0 and after.get('in_flight') == 0 and
                     delta == 1 and not errors and not competing and r.get('error') is None)
    r['cpu_frequency_full_request'] = frequency_stats(monitor.samples, r['start_epoch'], r['end_epoch'])
    r['cpu_frequency_decode_window'] = frequency_stats(monitor.samples, first, last)
    r['contamination_check'] = {
        'activity_before': before, 'activity_after': after, 'request_counter_delta': delta,
        'observed_other_inflight': competing, 'status_error_samples': len(errors),
        'no_observed_competitor': uncontaminated,
        'note': 'Counter and sampled activity guard only; sampling cannot prove absence of every external workload.'}
    r['cold_zero_reuse_confirmed'] = any(
        e.get('prompt_tokens') == r['usage'].get('prompt_tokens') and e.get('reused_tokens') == 0
        for e in r['engine_prompt_records'])
    save_json(bench.out / (name + '.json'), r)
    save_json(bench.out / 'results.json', bench.records)
    print('CPU_REQUEST ' + json.dumps({
        'name': name, 'cpu_decode': r['cpu_frequency_decode_window'],
        'no_observed_competitor': uncontaminated,
        'cold_zero_reuse_confirmed': r['cold_zero_reuse_confirmed'],
        'client_decode_tps_estimate': r['decode_tps_usage_minus_one_estimate'],
        'engine_prompt_records': r['engine_prompt_records'], 'output_sha256': r['output_sha256']}, ensure_ascii=False), flush=True)
    return r


def set_governors(original, performance):
    for name, old in original.items():
        target = 'performance' if performance else old
        path = Path(name)
        path.write_text(target + '\n')
        got = path.read_text().strip()
        if got != target:
            raise RuntimeError(f'Governor write verification failed for {path}: {got} != {target}')


def build_report(records, original, restoration, aborted):
    stages = []
    reference = next((r for r in records if r.get('category') == 'governor-decode'), None)
    for phase, mode in (('A1', 'original'), ('B1', 'performance'), ('B2', 'performance'), ('A2', 'original')):
        decode = next((r for r in records if r['name'] == phase + '-decode'), None)
        cold = next((r for r in records if r['name'] == phase + '-fresh'), None)
        item = {'phase': phase, 'mode': mode, 'complete': decode is not None and cold is not None}
        if decode:
            full = [e for e in decode['engine_prompt_records']
                    if e.get('prompt_tokens') == decode['usage'].get('prompt_tokens') and
                    e.get('generated_tokens') == decode['usage'].get('completion_tokens')]
            item['decode'] = {k: decode[k] for k in ('usage', 'ttft_s', 'total_s', 'output_sha256', 'finish_reason', 'error',
                             'decode_tps_usage_minus_one_estimate', 'cpu_frequency_decode_window', 'contamination_check')}
            item['decode']['engine_full_request'] = full[0] if len(full) == 1 else None
            item['decode']['same_output_as_A1'] = reference is not None and decode['output_sha256'] == reference['output_sha256']
            item['decode']['same_request_as_A1'] = reference is not None and decode['request_sha256'] == reference['request_sha256']
        if cold:
            item['fresh_prefill'] = {k: cold[k] for k in ('usage', 'ttft_s', 'engine_prompt_records', 'cold_zero_reuse_confirmed',
                                    'contamination_check', 'cpu_frequency_full_request', 'error')}
        stages.append(item)
    return {'sequence': 'A1(original), B1(performance), B2(performance), A2(original)',
            'original_governors': original, 'restoration': restoration, 'aborted': aborted, 'stages': stages,
            'interpretation': 'Only compare uncontaminated requests with matching fixed-request hashes. CPU0 scaling_cur_freq is a sampled driver report, not actual APERF/MPERF effective clock. Fresh prompt differences are phase tags; confirm zero reused tokens.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = out / 'original-governors.json'
    if saved.exists():
        ap.error('output already contains original-governors.json; choose a fresh --out to retain recovery evidence')
    bench = Bench(SimpleNamespace(config=args.config, port=args.port, out=args.out))
    initial = bench.status()
    if initial.get('in_flight') != 0:
        raise RuntimeError('Service is not confirmed idle; no governor was changed: ' + json.dumps(initial))
    governors = sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    if not governors:
        raise RuntimeError('No cpufreq governor policy files found; no changes made')
    original = {str(p): p.read_text().strip() for p in governors}
    save_json(saved, original)
    for path in governors:
        available_path = path.with_name('scaling_available_governors')
        if available_path.exists() and 'performance' not in available_path.read_text().split():
            raise RuntimeError(f'performance governor unavailable: {path}; no changes made')
    stdout_file = (out / 'run.stdout.log').open('w')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_stdout, stdout_file), Tee(old_stderr, stdout_file)
    aborted, restoration = None, []
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
    fixed = [{'role': 'user', 'content': '[CPU governor fixed decode benchmark 20261107] ' + filler(100000) +
              '\n\nIgnoring the text above, ' + TASKS['code']}]
    save_json(out / 'synthetic-prompts.json', {'fixed_messages': fixed})
    try:
        print('START_CPU_ABBA ' + json.dumps({'policies': len(original), 'governors': sorted(set(original.values())), 'activity': initial}), flush=True)
        set_governors(original, False)
        time.sleep(2)
        warm = measured_request(bench, 'warmup', fixed, 32, 'warmup')
        if warm.get('error'):
            raise RuntimeError('Warmup API request failed')
        for phase, perf in (('A1', False), ('B1', True), ('B2', True), ('A2', False)):
            set_governors(original, perf)
            print('PHASE ' + phase + (' performance' if perf else ' original'), flush=True)
            time.sleep(2)
            measured_request(bench, phase + '-decode', fixed, 768, 'governor-decode')
            fresh = [{'role': 'user', 'content': '[' + phase + ' fresh CPU-governor prefill 20261107] ' + filler(100000) +
                      '\n\nIgnoring the text above, ' + TASKS['code']}]
            measured_request(bench, phase + '-fresh', fresh, 128, 'governor-fresh')
            save_json(out / 'cpu-governor-report.json', build_report(bench.records, original, [], None))
    except BaseException as e:
        aborted = {'kind': type(e).__name__, 'detail': str(e)}
        print('ABORT ' + json.dumps(aborted), flush=True)
        raise
    finally:
        # Restore each policy independently: one failure must not skip others.
        for path, value in original.items():
            result = {'path': path, 'expected': value}
            try:
                Path(path).write_text(value + '\n')
                result['actual'] = Path(path).read_text().strip()
                result['restored'] = result['actual'] == value
            except BaseException as e:
                result.update(restored=False, error=type(e).__name__ + ': ' + str(e))
            restoration.append(result)
        report = build_report(bench.records, original, restoration, aborted)
        save_json(out / 'cpu-governor-report.json', report)
        print('RESTORE ' + json.dumps({'policies': len(restoration), 'all_restored': all(r['restored'] for r in restoration)}), flush=True)
        print('REPORT ' + json.dumps(report, ensure_ascii=False), flush=True)
        sys.stdout.flush()
        os.fsync(stdout_file.fileno())
        sys.stdout, sys.stderr = old_stdout, old_stderr
        stdout_file.close()
    if not all(r['restored'] for r in restoration):
        raise RuntimeError('Some CPU governors could not be restored; see cpu-governor-report.json')


if __name__ == '__main__':
    main()
