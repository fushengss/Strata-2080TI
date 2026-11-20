#!/usr/bin/env python3
"""Synthetic concurrency client. Dry-run by default; never manages a server.

No per-token timestamps are available: measured gaps are SSE output-chunk gaps.
No prompts, generated text, tool arguments, raw HTTP bodies or API keys are saved.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import math
import os
import re
import socket
import statistics
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

WORDS = ('the river carried old maps past quiet towns where people traded stories about storms bridges lanterns '
         'harvests letters clocks ships and distant mountains while children counted stars above the harbor').split()
CHANNELS = ('output', 'visible_content', 'reasoning', 'tool')


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temp, path)


def distribution(values):
    values = sorted(values)
    def percentile(p):
        return values[max(0, math.ceil(len(values) * p) - 1)] if values else None
    return {'n': len(values), 'p50_s': percentile(.5), 'p95_s': percentile(.95),
            'p99_s': percentile(.99), 'max_s': max(values) if values else None,
            'quantile_method': 'nearest rank; small n does not establish a tail SLO'}


def extract_delta(data):
    """Return channel byte counts and a hash input, never persist strings."""
    values = {'visible_content': [], 'reasoning': [], 'tool': []}
    tool_seen = False
    for choice in data.get('choices') or []:
        delta = choice.get('delta') or {}
        for key, name in [('content', 'visible_content'), ('reasoning_content', 'reasoning')]:
            value = delta.get(key) or (delta.get('reasoning') if key == 'reasoning_content' else '')
            if isinstance(value, str) and value:
                values[name].append(value)
        calls = delta.get('tool_calls') or []
        if delta.get('function_call'):
            calls = list(calls) + [{'function': delta['function_call']}]
        for call in calls:
            function = call.get('function') or {}
            strings = [function.get('name'), function.get('arguments')]
            tool_seen |= bool(call.get('id') or any(strings))
            values['tool'].extend(value for value in strings if isinstance(value, str) and value)
    parts = {key: ''.join(value).encode() for key, value in values.items()}
    counts = {key: len(value) for key, value in parts.items()}
    counts['output'] = sum(counts.values())
    channels = [key for key in values if counts[key] or (key == 'tool' and tool_seen)]
    if channels:
        channels.insert(0, 'output')
    return channels, counts, parts


def synthetic_prompt(nominal_tokens, fixture):
    text = ' '.join(WORDS[(i * 7 + 8) % len(WORDS)] + ('.' if i % 17 == 16 else '')
                    for i in range(int(nominal_tokens / 1.3)))
    return (f'{fixture}: independent synthetic benchmark fixture.\n' + text +
            '\nWrite a very long English field diary describing many different harbor towns, their history, '
            'weather and daily work. Use numbered entries with substantial detail. Continue for thousands of '
            'entries rather than ending with a summary. Do not quote the source text.')


def summarize_request(record):
    chunks = record['chunks']
    for channel in CHANNELS:
        events = [event for event in chunks if channel in event['channels']]
        record['first_' + ('nonempty_delta' if channel == 'output' else channel) + '_s'] = events[0]['elapsed_s'] if events else None
        gaps = [b['elapsed_s'] - a['elapsed_s'] for a, b in zip(events, events[1:])]
        record[channel + '_inter_chunk_gaps'] = distribution(gaps)
    record['ttft_s'] = record['first_nonempty_delta_s']
    record['first_tool_delta_s'] = record['first_tool_s']
    record['ttft_definition'] = 'first nonempty reasoning/content/tool delta; visible-content latency is separate'
    total_bytes = sum(event['bytes']['output'] for event in chunks)
    tokens = record.get('usage', {}).get('completion_tokens')
    method = 'final_usage_proportional_to_output_bytes' if isinstance(tokens, int) and total_bytes else 'output_bytes_div4_heuristic'
    for event in chunks:
        event['token_count_estimate'] = tokens * event['bytes']['output'] / total_bytes if method.startswith('final_usage') else event['bytes']['output'] / 4
        event['token_estimate_method'] = method
    record['token_estimate_note'] = 'Fractional allocation, not tokenizer boundaries or per-token timing; usage may include non-text/control tokens.'
    record['output_bytes'] = total_bytes
    record['observed_output_chunks'] = len(chunks)
    last = chunks[-1]['elapsed_s'] if chunks else None
    record['terminal_silence_lower_bound_s'] = record['elapsed_s'] - last if last is not None and record.get('error') else None
    prompt = record.get('usage', {}).get('prompt_tokens')
    cached = record.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens')
    record['cached_tokens'] = cached
    record['verified_zero_cache'] = cached == 0 if cached is not None else None
    record['actual_prompt_tokens'] = prompt
    record['prompt_within_nominal_tolerance'] = abs(prompt - record['nominal_prompt_tokens']) <= record['nominal_prompt_tokens'] * record['prompt_tolerance'] if isinstance(prompt, int) else None
    record['successful'] = record.get('error') is None and record['saw_done'] and bool(chunks)


class Client:
    def __init__(self, port, config, out, plan, fixture_id):
        self.port, self.out, self.plan, self.fixture_id = port, out, plan, fixture_id
        self.model = config.get('model_name', 'strata')
        self.headers = {'Content-Type': 'application/json'}
        if config.get('api_key'):
            self.headers['Authorization'] = 'Bearer ' + config['api_key']
        self.lock = threading.Lock()
        self.active, self.records, self.occupancy = {}, [], []
        self.abort = threading.Event()
        self.monitor_stop = threading.Event()
        self.resident_stop = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.monitor, daemon=True)

    def status(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        try:
            connection.request('GET', '/v1/status', headers=self.headers)
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f'status HTTP {response.status}')
            data = json.loads(response.read(2 << 20))
            result = {'loaded': data.get('loaded') is True}
            for group, fields in [('activity', ('requests', 'in_flight')),
                                  ('concurrency', ('serving', 'requested')),
                                  ('context', ('native', 'max_positions'))]:
                value = data.get(group) or {}
                result[group] = {key: value[key] for key in fields if isinstance(value.get(key), (int, float))}
            return result
        finally:
            connection.close()

    def cancel(self, state, reason):
        with self.lock:
            state['cancel_reason'] = state.get('cancel_reason') or reason
            sock = state.get('socket')
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def monitor(self):
        while not self.monitor_stop.wait(self.plan['occupancy_sample_seconds']):
            now = time.monotonic()
            with self.lock:
                active = list(self.active.values())
                self.occupancy.append({'epoch': time.time(), 'own_in_flight': len(active),
                    'resident_in_flight': sum(s['role'] == 'resident' for s in active),
                    'resident_streams_with_first_output': sum(s['role'] == 'resident' and s['first'] is not None for s in active),
                    'resident_streams_output_within_1s': sum(s['role'] == 'resident' and s['last'] is not None and now - s['last'] <= 1 for s in active),
                    'resident_streams_stalled_over_1s': sum(s['role'] == 'resident' and s['last'] is not None and now - s['last'] > 1 for s in active),
                    'resident_streams_stalled_over_5s': sum(s['role'] == 'resident' and s['last'] is not None and now - s['last'] > 5 for s in active),
                    'resident_awaiting_first_output': sum(s['role'] == 'resident' and s['first'] is None for s in active)})
            if now - self.started > self.plan['scenario_total_timeout_seconds']:
                self.abort.set()
            for state in active:
                elapsed = now - state['start']
                reason = None
                if self.abort.is_set():
                    reason = 'scenario_aborted'
                elif elapsed > self.plan['request_total_timeout_seconds']:
                    reason = 'request_total_timeout'
                elif state['first'] is None and elapsed > self.plan['request_ttft_timeout_seconds']:
                    reason = 'first_output_timeout'
                elif state['last'] is not None and now - state['last'] > self.plan['request_silence_timeout_seconds']:
                    reason = 'output_silence_timeout'
                if reason:
                    self.cancel(state, reason)

    def request(self, name, role, lane, nominal, output_tokens, fixture, planned_epoch=None):
        prompt = synthetic_prompt(nominal, fixture)
        body = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': output_tokens,
                'temperature': self.plan['temperature'], 'seed': self.plan['seed'],
                'chat_template_kwargs': {'enable_thinking': self.plan.get('thinking_enabled', True)}}
        for key in ('top_p', 'top_k'):
            if key in self.plan:
                body[key] = self.plan[key]
        raw = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        start_epoch, start = time.time(), time.monotonic()
        record = {'name': name, 'role': role, 'lane': lane, 'nominal_prompt_tokens': nominal,
                  'prompt_tolerance': self.plan['prompt_count_tolerance_fraction'], 'max_tokens': output_tokens,
                  'sampling_profile': self.plan['sampling_profile'],
                  'request_sha256': hashlib.sha256(raw).hexdigest(), 'start_epoch': start_epoch,
                  'planned_epoch': planned_epoch, 'schedule_delay_s': max(0, start_epoch - planned_epoch) if planned_epoch else None,
                  'chunks': [], 'usage': {}, 'saw_done': False, 'error': None, 'finish_reason': None}
        state = {'role': role, 'lane': lane, 'start': start, 'first': None, 'last': None, 'socket': None}
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        output_hashes = {channel: hashlib.sha256() for channel in CHANNELS if channel != 'output'}
        with self.lock:
            self.active[name] = state
        try:
            connection.connect()
            with self.lock:
                state['socket'] = connection.sock
            connection.sock.settimeout(self.plan['request_total_timeout_seconds'])
            connection.request('POST', '/v1/chat/completions', body=raw, headers=self.headers)
            response = connection.getresponse()
            record['http_status'] = response.status
            if response.status != 200:
                raise RuntimeError('HTTP_error')  # Never save the response body or exception message.
            with (self.out / (name + '.chunks.jsonl')).open('w') as journal:
                pending = []
                for raw_line in response:
                    line = raw_line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        pending.append(line[5:].lstrip())
                        continue
                    if line or not pending:
                        continue
                    received = time.monotonic()
                    payload, pending = '\n'.join(pending), []
                    if payload == '[DONE]':
                        record['saw_done'] = True
                        break
                    data = json.loads(payload)
                    if data.get('error'):
                        raise RuntimeError('API_error')
                    if isinstance(data.get('timings'), dict):
                        record['native_timings'] = {k: v for k, v in data['timings'].items()
                            if k in ('cache_n', 'prompt_n', 'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
                                     'predicted_n', 'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second',
                                     'draft_n', 'draft_n_accepted') and (v is None or type(v) in (int, float))}
                    if isinstance(data.get('usage'), dict):
                        usage = data['usage']
                        record['usage'] = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if isinstance(usage.get(key), int)}
                        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                        if isinstance(cached, int):
                            record['usage']['prompt_tokens_details'] = {'cached_tokens': cached}
                    for choice in data.get('choices') or []:
                        if choice.get('finish_reason'):
                            reason = choice['finish_reason']
                            record['finish_reason'] = reason if reason in ('stop', 'length', 'tool_calls', 'function_call', 'cancel', 'content_filter') else 'unknown'
                    channels, counts, parts = extract_delta(data)
                    if not channels:
                        continue
                    now = received
                    event = {'elapsed_s': now - start, 'epoch': start_epoch + now - start,
                             'channels': channels, 'bytes': counts,
                             'previous_output_gap_s': now - state['last'] if state['last'] is not None else None}
                    record['chunks'].append(event)
                    for channel, value in parts.items():
                        output_hashes[channel].update(value)
                    with self.lock:
                        state['first'] = state['first'] if state['first'] is not None else now
                        state['last'] = now
                    journal.write(json.dumps(event, separators=(',', ':')) + '\n')
        except Exception as error:
            record['error'] = {'kind': type(error).__name__}  # No server text, credentials, or generated content.
        finally:
            connection.close()
            record['elapsed_s'] = time.monotonic() - start
            record['end_epoch'] = start_epoch + record['elapsed_s']
            record['channel_output_sha256'] = {channel: digest.hexdigest() for channel, digest in output_hashes.items()}
            record['output_sha256'] = hashlib.sha256(json.dumps(record['channel_output_sha256'], sort_keys=True).encode()).hexdigest()
            record['hash_note'] = 'Per-channel text hashes do not depend on SSE chunk boundaries; combined hash covers these three digests.'
            if state.get('cancel_reason'):
                record['error'] = {'kind': state['cancel_reason']}
            elif not record['saw_done'] and record['error'] is None:
                record['error'] = {'kind': 'incomplete_stream'}
            summarize_request(record)
            save(self.out / (name + '.json'), record)
            with self.lock:
                self.records.append(record)
                self.active.pop(name, None)
        return record

    def resident(self, lane):
        generation, failures = 0, 0
        fixture = f'{self.fixture_id}-resident-{lane}'
        while not self.resident_stop.is_set() and not self.abort.is_set():
            record = self.request(f'resident-{lane}-{generation}', 'resident', lane,
                                  self.plan['resident_context_nominal_tokens'], self.plan['resident_output_tokens'], fixture)
            failures = 0 if record['successful'] else failures + 1
            generation += 1
            if failures >= 2:
                self.abort.set()
                break

    def wait(self, seconds):
        if self.abort.wait(max(0, seconds)):
            raise RuntimeError('scenario_aborted')


def phase_summary(records, begin, end, lanes):
    duration = max(0, end - begin)
    resident = [record for record in records if record['role'] == 'resident']
    lane_bytes = [sum(chunk['bytes']['output'] for record in resident if record['lane'] == lane
                      for chunk in record['chunks'] if begin <= chunk['epoch'] < end) for lane in range(lanes)]
    selected = [chunk for record in resident for chunk in record['chunks'] if begin <= chunk['epoch'] < end]
    rates = [value / duration if duration else 0 for value in lane_bytes]
    squares = sum(value * value for value in rates)
    gaps = [chunk['previous_output_gap_s'] for chunk in selected if chunk['previous_output_gap_s'] is not None]
    silent_bounds = []
    for record in resident:
        if record['first_nonempty_delta_s'] is None:
            continue
        first = record['start_epoch'] + record['first_nonempty_delta_s']
        left, right = max(begin, first), min(end, record['end_epoch'])
        if right > left:
            anchors = [left] + [c['epoch'] for c in record['chunks'] if left < c['epoch'] < right] + [right]
            silent_bounds.extend(b - a for a, b in zip(anchors, anchors[1:]))
    return {'start_epoch': begin, 'end_epoch': end, 'duration_s': duration,
            'resident_output_bytes_per_s': sum(rates), 'lane_output_bytes_per_s': rates,
            'resident_visible_content_bytes_per_s': sum(c['bytes']['visible_content'] for c in selected) / duration if duration else None,
            'resident_token_rate_estimate': sum(c['token_count_estimate'] for c in selected) / duration if duration else None,
            'chunks_using_byte4_fallback': sum(c['token_estimate_method'] == 'output_bytes_div4_heuristic' for c in selected),
            'jain_fairness_of_output_byte_rates': sum(rates) ** 2 / (lanes * squares) if squares else None,
            'output_inter_chunk_gaps_ending_in_window': distribution(gaps),
            'max_observed_no_output_interval_s': max(silent_bounds) if silent_bounds else None,
            'gap_note': 'Gaps ending here may start before the phase; clipped silence includes censored phase/end boundaries. No token TBT.'}


def summarize(records, phases, injections, occupancy, lanes):
    result = {'requests': len(records),
              'failed_requests': sum(not r['successful'] and (r.get('error') or {}).get('kind') != 'bounded_drain_cancel' for r in records),
              'expected_drain_cancellations': sum((r.get('error') or {}).get('kind') == 'bounded_drain_cancel' for r in records),
              'successful_complete_requests': sum(r['successful'] for r in records),
              'phases': {name: phase_summary(records, *bounds, lanes) for name, bounds in phases.items()},
              'groups': {}, 'injections': [], 'recovery': {}, 'occupancy': {}}
    groups = defaultdict(list)
    for record in records:
        groups[f"{record['role']}-{record['nominal_prompt_tokens']}"].append(record)
    for name, rows in groups.items():
        result['groups'][name] = {'requests': len(rows), 'successful': sum(r['successful'] for r in rows),
            'ttft_first_output': distribution([r['ttft_s'] for r in rows if r['ttft_s'] is not None]),
            'ttft_first_visible_content': distribution([r['first_visible_content_s'] for r in rows if r['first_visible_content_s'] is not None]),
            'requests_with_no_first_output': sum(r['ttft_s'] is None for r in rows),
            'censored_ttft_lower_bounds': distribution([r['elapsed_s'] for r in rows if r['ttft_s'] is None]),
            'verified_zero_cache': sum(r['verified_zero_cache'] is True for r in rows),
            'unknown_cache_count': sum(r['verified_zero_cache'] is None for r in rows),
            'nominal_size_checks_passed': sum(r['prompt_within_nominal_tolerance'] is True for r in rows),
            'output_inter_chunk_gaps': distribution([c['previous_output_gap_s'] for r in rows for c in r['chunks'] if c['previous_output_gap_s'] is not None]),
            'max_terminal_silence_lower_bound_s': max((r['terminal_silence_lower_bound_s'] or 0 for r in rows), default=None)}
    for record in injections:
        end = record['start_epoch'] + record['ttft_s'] if record['ttft_s'] is not None else record['end_epoch']
        result['injections'].append({'name': record['name'], 'nominal_tokens': record['nominal_prompt_tokens'],
            'prefill_admission_proxy': phase_summary(records, record['start_epoch'], end, lanes),
            'proxy_end_is_first_output': record['ttft_s'] is not None,
            'note': 'Includes queue/admission/first-token latency, not GPU prefill duration.'})
    for name, bounds in phases.items():
        samples = [s for s in occupancy if bounds[0] <= s['epoch'] < bounds[1]]
        result['occupancy'][name] = {'samples': len(samples),
            'fraction_with_target_resident_streams': sum(s['resident_streams_with_first_output'] == lanes for s in samples) / len(samples) if samples else None,
            'fraction_with_all_residents_output_in_last_1s': sum(s['resident_streams_output_within_1s'] == lanes for s in samples) / len(samples) if samples else None,
            'fraction_with_any_resident_stalled_over_1s': sum(s['resident_streams_stalled_over_1s'] > 0 for s in samples) / len(samples) if samples else None,
            'fraction_with_any_resident_stalled_over_5s': sum(s['resident_streams_stalled_over_5s'] > 0 for s in samples) / len(samples) if samples else None,
            'max_resident_streams_stalled_over_1s': max((s['resident_streams_stalled_over_1s'] for s in samples), default=None),
            'note': 'Client stream counters, not confirmed GPU occupancy; no-new-output counter includes reasoning/content/tool deltas.'}
    return result


def run_level(args, plan, config, lanes, fixture_id):
    out = args.out / f'c{lanes}'
    out.mkdir()
    client = Client(args.port, config, out, plan, fixture_id + f'-c{lanes}')
    before = client.status()
    if before.get('loaded') is not True or (before.get('activity') or {}).get('in_flight') != 0:
        raise RuntimeError('Service must already be loaded and idle; no synthetic request sent for this level')
    serving = (before.get('concurrency') or {}).get('serving')
    if not isinstance(serving, int) or lanes > serving:
        raise RuntimeError('Target resident concurrency exceeds reported serving slots')
    native_context = (before.get('context') or {}).get('native')
    estimated_need = max(plan['resident_context_nominal_tokens'], *plan['injected_prefill_nominal_tokens']) * (1 + plan['prompt_count_tolerance_fraction']) + max(plan['resident_output_tokens'], plan['injected_output_tokens'])
    if not isinstance(native_context, (int, float)) or estimated_need > native_context:
        raise RuntimeError('Reported context is too small for planned prompt estimate plus output')
    save(out / 'status-before.json', before)
    workers = [threading.Thread(target=client.resident, args=(lane,), daemon=True) for lane in range(lanes)]
    phases, injections, error = {}, [], None
    scenario_start = time.time()
    client.thread.start()
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + plan['resident_ready_timeout_seconds']
        stable_since = None
        while True:
            now = time.monotonic()
            with client.lock:
                ready = sum(s['role'] == 'resident' and s['first'] is not None and now - s['last'] <= 1
                            for s in client.active.values()) == lanes
            stable_since = (stable_since if stable_since is not None else now) if ready else None
            if stable_since is not None and now - stable_since >= plan['resident_ready_stable_seconds']:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('target_resident_streams_not_established')
            client.wait(.1)
        begin = time.time()
        print(f'READY C{lanes}: stable active output, starting baseline', flush=True)
        client.wait(plan['baseline_seconds'])
        mixed = time.time()
        phases['baseline'] = [begin, mixed]
        for index, nominal in enumerate(plan['injected_prefill_nominal_tokens']):
            planned = mixed + index * plan['injection_interval_seconds']
            client.wait(planned - time.time())
            print(f'INJECT C{lanes}: {index + 1}, nominal {nominal} tokens', flush=True)
            record = client.request(f'injected-{index}-{nominal}', 'injected', None, nominal,
                                    plan['injected_output_tokens'], f'{client.fixture_id}-cold-{index}', planned)
            injections.append(record)
            if not record['successful']:
                error = 'injected_request_failed; further injections skipped'
                break
        recovery = time.time()
        print(f'RECOVERY C{lanes}', flush=True)
        phases['mixed'] = [mixed, recovery]
        client.wait(plan['recovery_seconds'])
        phases['recovery'] = [recovery, time.time()]
    except (Exception, KeyboardInterrupt) as exc:
        error = type(exc).__name__  # Do not log arbitrary server text.
        client.abort.set()
    finally:
        client.resident_stop.set()
        deadline = time.monotonic() + (5 if client.abort.is_set() else plan['resident_drain_timeout_seconds'])
        for worker in workers:
            if worker.ident is not None:
                worker.join(max(0, deadline - time.monotonic()))
        with client.lock:
            active = list(client.active.values())
        for state in active:
            client.cancel(state, 'bounded_drain_cancel')
        for worker in workers:
            if worker.ident is not None:
                worker.join(12)
        client.monitor_stop.set()
        client.thread.join(2)
    if any(worker.is_alive() for worker in workers):
        raise RuntimeError('Own client worker did not terminate; no further workload may be started')
    result = summarize(client.records, phases, injections, client.occupancy, lanes)
    result.update({'resident_workers': lanes, 'error': error, 'scenario_start_epoch': scenario_start,
                   'scenario_end_epoch': time.time(), 'sampling_profile': plan['sampling_profile'],
                   'profile': plan['profile'], 'admission_case': 'capacity_queue' if lanes == serving else 'mixed_admission'})
    baseline_occupancy = result['occupancy'].get('baseline', {}).get('fraction_with_target_resident_streams')
    result['baseline_occupancy_qualified'] = baseline_occupancy is not None and baseline_occupancy >= plan['minimum_baseline_occupancy_fraction']
    result['baseline_occupancy_requirement'] = plan['minimum_baseline_occupancy_fraction']
    if not result['error'] and result['failed_requests']:
        result['error'] = 'unexpected_request_failure'
    if not result['error'] and not result['baseline_occupancy_qualified']:
        result['error'] = 'baseline_target_occupancy_below_requirement'
    duration = result['scenario_end_epoch'] - scenario_start
    tokens = sum(r['usage'].get('completion_tokens', 0) for r in client.records)
    result['whole_scenario_completed_api_token_rate'] = {'tokens': tokens, 'seconds_including_start_and_drain': duration,
        'tokens_per_s': tokens / duration, 'requests_missing_usage': sum('completion_tokens' not in r['usage'] for r in client.records),
        'note': 'Completed API usage only, lower bound when requests lack final usage; not steady-state decode speed.'}
    if 'recovery' in phases and 'baseline' in result['phases']:
        begin, end = phases['recovery']; width = plan['recovery_bin_seconds']; bins = []
        while begin + width <= end:
            bins.append(phase_summary(client.records, begin, begin + width, lanes)); begin += width
        threshold = result['phases']['baseline']['resident_output_bytes_per_s'] * plan['recovery_baseline_fraction']
        need = plan['recovery_consecutive_bins']; recovered = None
        for index in range(len(bins) - need + 1):
            if threshold > 0 and all(b['resident_output_bytes_per_s'] >= threshold for b in bins[index:index + need]):
                recovered = bins[index]['start_epoch'] - phases['recovery'][0]; break
        result['recovery'] = {'bins': bins, 'baseline_fraction': plan['recovery_baseline_fraction'],
            'consecutive_bins': need, 'first_sustained_recovery_start_s': recovered,
            'confirmed_after_s': recovered + width * need if recovered is not None else None,
            'note': 'Descriptive aggregate output-byte-rate recovery including reasoning/tool text; not a GPU recovery metric.'}
    save(out / 'occupancy.json', client.occupancy)
    save(out / 'summary.json', result)
    # Read only the aggregate status, never logs or request history. Do not start the next level while draining.
    after = client.status()
    idle_deadline = time.monotonic() + 10
    while (after.get('activity') or {}).get('in_flight') != 0 and time.monotonic() < idle_deadline:
        time.sleep(.25)
        after = client.status()
    save(out / 'status-after.json', after)
    if (after.get('activity') or {}).get('in_flight') != 0:
        raise RuntimeError('Service has requests in flight after this level; refusing the next workload')
    return result


def self_test():
    import tempfile
    reasoning = {'choices': [{'delta': {'reasoning_content': 'think'}}]}
    content = {'choices': [{'delta': {'content': 'answer'}}]}
    tools = {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"x":1}'}}]}}]}
    assert extract_delta(reasoning)[0] == ['output', 'reasoning']
    assert extract_delta(content)[0] == ['output', 'visible_content']
    assert extract_delta(tools)[0] == ['output', 'tool']
    assert extract_delta({'choices': [{'delta': {'role': 'assistant', 'content': ''}}]})[0] == []
    chunks = []
    for elapsed, data in [(1, reasoning), (2, reasoning), (3, tools), (4, content)]:
        channels, counts, _ = extract_delta(data)
        chunks.append({'elapsed_s': elapsed, 'epoch': 100 + elapsed, 'channels': channels, 'bytes': counts,
                       'previous_output_gap_s': 1 if chunks else None})
    record = {'chunks': chunks, 'usage': {'completion_tokens': 12, 'prompt_tokens': 10000,
              'prompt_tokens_details': {'cached_tokens': 0}}, 'elapsed_s': 4.1, 'saw_done': True,
              'nominal_prompt_tokens': 10000, 'prompt_tolerance': .1, 'error': None}
    summarize_request(record)
    assert record['ttft_s'] == 1 and record['first_visible_content_s'] == 4
    assert record['first_reasoning_s'] == 1 and record['first_tool_s'] == 3
    assert record['output_inter_chunk_gaps']['max_s'] == 1
    assert abs(sum(c['token_count_estimate'] for c in chunks) - 12) < 1e-9
    assert record['verified_zero_cache'] is True
    assert distribution([.1, .2, 3])['p99_s'] == 3
    fixtures = [reasoning, tools, content,
                {'choices': [{'delta': {}, 'finish_reason': 'length'}],
                 'usage': {'prompt_tokens': 1000, 'completion_tokens': 12,
                           'prompt_tokens_details': {'cached_tokens': 0}}}]
    stream = b''.join(b'data: ' + json.dumps(value).encode() + b'\n\n' for value in fixtures) + b'data: [DONE]\n\n'
    sent = []
    class FakeSocket:
        def settimeout(self, _): pass
    class FakeResponse:
        status = 200
        def __iter__(self): return iter(stream.splitlines(keepends=True))
    class FakeConnection:
        def __init__(self, host, port, timeout):
            assert host == '127.0.0.1' and port == 12345
            self.sock = FakeSocket()
        def connect(self): pass
        def request(self, method, path, body, headers):
            assert method == 'POST' and path == '/v1/chat/completions'
            sent.append(json.loads(body))
        def getresponse(self): return FakeResponse()
        def close(self): pass
    original = http.client.HTTPConnection
    try:
        http.client.HTTPConnection = FakeConnection
        with tempfile.TemporaryDirectory() as directory:
            plan = json.loads(Path(__file__).with_name('workload-plan.json').read_text())
            for mode in ('greedy', 'sampled'):
                plan.update(plan['sampling_profiles'][mode]); plan['sampling_profile'] = mode
                client = Client(12345, {'api_key': 'offline-only-secret-never-save'}, Path(directory), plan, 'offline')
                rec = client.request(mode, 'injected', None, 1000, 12, 'offline')
                assert rec['successful'] and rec['first_reasoning_s'] is not None and rec['first_tool_delta_s'] is not None
                assert rec['first_visible_content_s'] >= rec['first_nonempty_delta_s']
                assert rec['output_inter_chunk_gaps']['n'] == 2
            assert sent[0]['temperature'] == 0 and sent[1]['temperature'] == .8 and sent[1]['top_k'] == 20
            for path in Path(directory).iterdir():
                text = path.read_text()
                assert 'offline-only-secret-never-save' not in text and 'harbor towns' not in text
                assert '"content":' not in text and '"arguments":' not in text
    finally:
        http.client.HTTPConnection = original
    print('OFFLINE PASS: SSE framing, reasoning/content/tool clocks, greedy/sampled fixtures, gap/token estimates, sanitized artifacts; HTTP transport was replaced by an in-memory fake')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('workload-plan.json'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--exclusive-benchmark', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--levels', help='Comma-separated subset of 1,2,4,8; 4 is the main mixed-admission case')
    parser.add_argument('--profile', choices=('short', 'full'), default='short')
    parser.add_argument('--sampling-profile', choices=('greedy', 'sampled'), default='greedy')
    parser.add_argument('--resident-context-tokens', type=int, help='Explicit nominal resident context override')
    parser.add_argument('--fixture-id', help='Public synthetic fixture label; use the same label across fresh-process A/B arms')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if args.execute:
            parser.error('--self-test cannot execute HTTP requests')
        self_test(); return
    plan = json.loads(args.plan.read_text())
    plan.update(plan['profiles'][args.profile])
    plan.update(plan['sampling_profiles'][args.sampling_profile])
    plan['profile'], plan['sampling_profile'] = args.profile, args.sampling_profile
    if args.resident_context_tokens is not None:
        if args.resident_context_tokens < 100:
            parser.error('resident context must be at least 100 nominal tokens')
        plan['resident_context_nominal_tokens'] = args.resident_context_tokens
    levels = [int(x) for x in args.levels.split(',')] if args.levels else plan['resident_concurrency_levels']
    if not levels or len(set(levels)) != len(levels) or set(levels) - {1, 2, 4, 8}:
        parser.error('levels must be a unique subset of 1,2,4,8')
    if not args.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_NETWORK', 'levels': levels, 'plan': plan}, ensure_ascii=False, indent=2))
        return
    if not (args.exclusive_benchmark and args.port is not None and args.config and args.out):
        parser.error('execution requires --exclusive-benchmark --port --config --out')
    if not 1 <= args.port <= 65535:
        parser.error('invalid port')
    if args.out.exists():
        parser.error('output must be a new directory')
    config = json.loads(args.config.read_text())
    args.out.mkdir(parents=True)
    if args.fixture_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', args.fixture_id):
        parser.error('fixture id must be a public ASCII alphanumeric/dash/underscore label, at most 96 characters')
    fixture_id = args.fixture_id or 'synthetic-' + uuid.uuid4().hex
    # No config contents/hash/path, headers, API key or model output are included in artifacts.
    save(args.out / 'manifest.json', {'plan': plan, 'levels': levels, 'fixture_id': fixture_id,
         'port': args.port, 'exclusive_benchmark_acknowledged': True,
         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(), 'started_epoch': time.time(),
         'privacy': 'Only synthetic request/response hashes, byte counts and timings; no text, credentials, logs or tools executed.'})
    results = []
    try:
        for lanes in levels:
            print(f'START synthetic level C{lanes}', flush=True)
            results.append(run_level(args, plan, config, lanes, fixture_id))
            save(args.out / 'results.json', results)
            print(f'FINISH synthetic level C{lanes}', flush=True)
            if results[-1]['error']:
                raise RuntimeError('level failed; subsequent levels skipped')
    except Exception as error:
        save(args.out / 'failure.json', {'kind': type(error).__name__, 'completed_levels': len(results),
                                        'note': 'No server lifecycle operation performed; inspect own saved timings.'})
        raise SystemExit('Benchmark failed; saved sanitized failure record. No subsequent workload was started.')


if __name__ == '__main__':
    main()
