#!/usr/bin/env python3
"""Observe numeric live telemetry with a common wall clock; no inference or service changes.

Default dry-run. Execution reads only two authenticated localhost GET endpoints.
Credentials, prompts, output text, tool data, headers and raw responses are never saved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def object_(value):
    return value if isinstance(value, dict) else {}


def numbers(value, keys):
    value = object_(value)
    return {key: number(value.get(key)) for key in keys}


def identity(value):
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value):
        return value
    return number(value)


STATES = {'idle', 'reading', 'decoding', 'generating', 'queued', 'waiting', 'admitting', 'ready',
          'draining', 'unloaded', 'loading', 'running', 'done', 'completed', 'cancelled', 'error'}


def state(value):
    return value if value in STATES else None


SLOT_KEYS = ('slot', 'group', 'replica', 'held_tokens', 'prompt_tokens', 'prompt_read', 'generated', 'elapsed_s', 'tok_s')
WORK_KEYS = ('slots_requested', 'slots_effective', 'active', 'occupied_slots', 'queued', 'waiting', 'outside_slots',
             'groups', 'effective_rows_window', 'batch_active_rows_window', 'batch_sample_windows',
             'batch_sample_rows', 'batch_sample_active_rows', 'batch_sample_span_s', 'batch_sample_age_s',
             'stage_sample_span_s', 'stage_sample_age_s', 'requests_omitted', 'max_queue', 'admission_limit',
             'admission_reserved', 'admission_rejected', 'oldest_queue_s', 'queue_timeout_s', 'queue_timeouts',
             'client_disconnect_cancels')
RATE_KEYS = ('answer_tok_s', 'answer_window_s', 'prefill_tok_s', 'output_tokens_server_counter')
LIVE_KEYS = ('queued', 'prompt_tokens', 'prompt_read', 'prompt_total', 'generated', 'max_tokens', 'elapsed_s',
             'tok_s', 'tok_s_mean', 'prefill_tok_s_mean', 'tok_s_window_s', 'parallel', 'running',
             'outside_slots', 'waiting', 'admitting')
HARDWARE_KEYS = ('cpu', 'ram_used', 'ram_total', 'gpu_util', 'gpu_mem_used', 'gpu_mem_total', 'gpu_temp',
                 'gpu_power', 'gpu_power_limit', 'gpu_pcie_rx_mb', 'gpu_pcie_tx_mb', 'disk_read_mb',
                 'disk_write_mb', 'tok_s', 'tok_s_mean', 'prefill_tok_s_mean')
GPU_KEYS = ('index', 'util', 'mem_used', 'mem_total', 'temp', 'power', 'power_limit', 'pcie_rx_mb', 'pcie_tx_mb')
HISTORY_KEYS = ('gpu_util', 'gpu_mem_used', 'gpu_temp', 'gpu_power', 'gpu_pcie_rx_mb', 'cpu', 'ram_used',
                'disk_read_mb', 'tok_s', 'prefill_tok_s_mean')
TOTAL_KEYS = ('since', 'requests', 'prompt_tokens', 'reused', 'output_tokens', 'prompt_ms', 'decode_ms')
REQUEST_KEYS = ('time', 'duration_s', 'prompt_tokens', 'prompt_total', 'prompt_read', 'reused', 'output_tokens',
                'first_engine_token_s', 'prompt_ms', 'decode_ms', 'decode_tok_s', 'drafts_offered', 'drafts_accepted')


def slots(raw):
    return [{**numbers(row, SLOT_KEYS), 'request_id': identity(object_(row).get('request_id')),
             'state': state(object_(row).get('state'))} for row in raw[:64]] if isinstance(raw, list) else []


def telemetry(raw):
    raw = object_(raw)
    work, instance = object_(raw.get('work')), object_(raw.get('instance'))
    requests = work.get('requests') if isinstance(work.get('requests'), list) else []
    return {'sampled_at': number(raw.get('sampled_at')),
            'instance': {'id': identity(instance.get('id')), 'boot_time': number(instance.get('boot_time')),
                         'state': state(instance.get('state'))},
            'rates': numbers(raw.get('rates'), RATE_KEYS),
            'work': {**numbers(work, WORK_KEYS), 'slots': slots(work.get('slots')),
                     'requests': [{**numbers(row, ('slot', 'group', 'replica', 'generated', 'prompt_read', 'queued_at',
                                                   'started_at', 'first_token_at', 'last_token_at')),
                                   'id': identity(object_(row).get('id')), 'state': state(object_(row).get('state'))}
                                  for row in requests[:64]]},
            'totals': numbers(raw.get('totals'), TOTAL_KEYS)}


def metrics(raw):
    raw = object_(raw)
    live, hardware, history = (object_(raw.get(key)) for key in ('live', 'hardware', 'history'))
    gpus = hardware.get('gpus') if isinstance(hardware.get('gpus'), list) else []
    requests = raw.get('requests') if isinstance(raw.get('requests'), list) else []
    return {'sampled_at': number(raw.get('time')),
            'live': {**numbers(live, LIVE_KEYS), 'state': state(live.get('state')), 'slots': slots(live.get('slots'))},
            'hardware': {**numbers(hardware, HARDWARE_KEYS), 'gpus': [numbers(row, GPU_KEYS) for row in gpus[:32]]},
            'history': {key: [number(value) for value in history[key][-600:]]
                        for key in HISTORY_KEYS if isinstance(history.get(key), list)},
            'totals': numbers(raw.get('totals'), TOTAL_KEYS),
            'recent_requests': [{**numbers(row, REQUEST_KEYS), 'id': identity(object_(row).get('id'))}
                                for row in requests[:50]]}


def fetch(path, headers, sanitize, timeout):
    before_epoch, before_mono = time.time(), time.monotonic()
    row = {'before_epoch': before_epoch, 'before_monotonic': before_mono, 'error': None}
    try:
        request = urllib.request.Request('http://127.0.0.1:8080' + path, headers=headers, method='GET')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
        row['snapshot'] = sanitize(raw)
    except Exception as error:
        row['error'] = type(error).__name__
    row.update(after_epoch=time.time(), after_monotonic=time.monotonic())
    row['round_trip_s'] = row['after_monotonic'] - before_mono
    row['midpoint_monotonic'] = (before_mono + row['after_monotonic']) / 2
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--config', type=Path, default=Path('/srv/Strata/strata-orca-q5_k_m.57.json'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--timeout', type=float, default=5)
    args = parser.parse_args()
    if not (0 < args.seconds <= 3600 and .1 <= args.interval <= 60 and .1 <= args.timeout <= 30):
        parser.error('Use seconds in(0,3600], interval in[.1,60], timeout in[.1,30]')
    if not args.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_NETWORK_NO_CONFIG_READ', 'seconds': args.seconds,
                          'interval': args.interval, 'port': 8080,
                          'endpoints': ['/v1/telemetry', '/metrics'], 'inference_requests': 0, 'service_changes': 0}))
        return
    if args.out is None:
        parser.error('--execute requires --out')
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error('--out must be new or empty')
    try:
        cfg = json.loads(args.config.read_text())
        key = cfg['api_key']
        if not isinstance(key, str) or not key:
            raise ValueError()
    except Exception as error:
        parser.exit(2, 'Private config could not be read: ' + type(error).__name__ + '\n')
    headers = {'Authorization': 'Bearer ' + key, 'Accept': 'application/json'}
    args.out.mkdir(parents=True, exist_ok=True)
    start, start_epoch = time.monotonic(), time.time()
    count, failures = 0, {'telemetry': 0, 'metrics': 0}
    manifest = {'created_epoch': start_epoch, 'seconds_requested': args.seconds, 'interval_requested': args.interval,
                'timeout_s': args.timeout, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'endpoints': ['/v1/telemetry', '/metrics'], 'host': '127.0.0.1', 'port': 8080,
                'inference_requests_sent': 0, 'service_mutations': 0,
                'limits': ['Two sequential GET calls are not an atomic snapshot; each has epoch and monotonic RTT brackets.',
                           'Counter differences require matching instance.id, nondecreasing counters and actual elapsed time.',
                           'Use 2s/5s/10s common-time windows; legacy live rates can sum different per-request windows.',
                           'History contains no per-item timestamps; the sampler nominally records once per second.',
                           'Numeric counters are not token-level arrival timestamps; do not claim TBT from them.']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    next_at = start
    try:
        with (args.out / 'samples.jsonl').open('x') as handle:
            while True:
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                row = {'sequence': count, 'loop_before_epoch': time.time(), 'loop_before_monotonic': time.monotonic(),
                       'telemetry': fetch('/v1/telemetry', headers, telemetry, args.timeout),
                       'metrics': fetch('/metrics', headers, metrics, args.timeout)}
                row['loop_after_monotonic'] = time.monotonic()
                for name in failures:
                    failures[name] += row[name]['error'] is not None
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                handle.flush()
                count += 1
                if time.monotonic() - start >= args.seconds:
                    break
                next_at = min(start + args.seconds, max(next_at + args.interval, time.monotonic()))
    finally:
        summary = {'finished_epoch': time.time(), 'elapsed_s': time.monotonic() - start,
                   'sample_count': count, 'get_failures': failures, 'inference_requests_sent': 0,
                   'service_mutations': 0, 'raw_content_saved': False}
        (args.out / 'observation.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
