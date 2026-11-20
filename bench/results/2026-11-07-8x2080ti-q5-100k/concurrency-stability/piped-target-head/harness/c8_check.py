#!/usr/bin/env python3
"""Eight simultaneous synthetic streams, no injections or service mutations.

Dry-run by default. Reuses the frozen correctness.py transport and its numeric
metrics/log filtering. Full text is saved only for this client's synthetic SSEs.
Whole-batch API token throughput includes small-prompt admission and final drain;
SSE arrival overlap is not a claim about simultaneous GPU kernel execution.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import threading
import time
from pathlib import Path

VERSION = 'piped-mtp-c8-check-1'
CORE_SHA256 = 'b3e32b596882bad35c9b6dd18f7c995615a7b4cc1ac1674cbf1bf3d06ed13a94'
FLAGS = ('STRATA_BATCH_MTP', 'STRATA_PIPED_MTP', 'STRATA_BATCH_SELF_COMMIT',
         'STRATA_DECODE_HEAD_GPU', 'STRATA_MTP_HEAD_GPU', 'STRATA_PW_WITH_BATCH',
         'STRATA_PIPED_MTP_HEAD', 'STRATA_PIPED_TARGET_HEAD')


def core_module():
    path = Path(__file__).with_name('correctness.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA256:
        raise RuntimeError('Frozen correctness.py dependency hash differs')
    spec = importlib.util.spec_from_file_location('c8_frozen_correctness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(core, maximum):
    return [core.fixture('c8-lane-' + str(i), core.CODE if i % 2 == 0 else core.PROSE,
                         seed, maximum, temperature=.8 if i % 4 < 2 else 0,
                         chat_template_kwargs={'enable_thinking': False})
            for i, seed in enumerate((17, 29, 101, 102, 113, 127, 139, 151))]


def timing(records):
    lane = []
    for record in records:
        events = [d['epoch'] for d in record['deltas']]
        first, last = record['first_output_epoch'], record['last_output_epoch']
        start, end = record['start_epoch'], record['end_epoch']
        lane.append({'name': record['name'], 'start_epoch': start, 'end_epoch': end,
                     'first_output_epoch': first, 'last_output_epoch': last,
                     'request_duration_s': end - start,
                     'first_output_s': first - start if first is not None else None,
                     'output_span_s': last - first if first is not None and last is not None else None,
                     'terminal_drain_s': end - last if last is not None else None,
                     'longest_inter_chunk_gap_s': max((b-a for a, b in zip(events, events[1:])), default=None),
                     'output_chunks': len(events), 'completion_tokens': record['usage'].get('completion_tokens')})
    begin = min((r['start_epoch'] for r in records), default=None)
    end = max((r['end_epoch'] for r in records), default=None)
    complete = [r for r in records if r.get('saw_done') and not r.get('error')
                and r.get('finish_reason') in ('stop', 'length')
                and isinstance(r['usage'].get('completion_tokens'), int)]
    total = sum(r['usage']['completion_tokens'] for r in complete)
    elapsed = end - begin if begin is not None and end is not None else None
    starts = [r['first_output_epoch'] for r in records]
    ends = [r['last_output_epoch'] for r in records]
    cb = max(starts) if starts and all(v is not None for v in starts) else None
    ce = min(ends) if ends and all(v is not None for v in ends) else None
    duration = max(0.0, ce-cb) if cb is not None and ce is not None else 0.0
    counts = {r['name']: sum(cb <= d['epoch'] <= ce for d in r['deltas']) if duration else 0 for r in records}
    return {'lanes': lane,
            'overlap': {'common_start_epoch': cb, 'common_end_epoch': ce, 'common_interval_s': duration,
                        'chunks_per_lane_in_common_interval': counts,
                        'eight_streams_observed': len(records) == 8 and duration >= 1.0 and all(v >= 2 for v in counts.values()),
                        'scope': 'SSE first/last arrival intersection; chunks are not token boundaries or GPU occupancy.'},
            'whole_batch': {'start_epoch': begin, 'last_request_end_epoch': end, 'elapsed_s': elapsed,
                            'completed_api_tokens': total, 'completed_requests': len(complete),
                            'total_requests': len(records),
                            'completed_api_tokens_per_s': total / elapsed if total is not None and elapsed and elapsed > 0 else None,
                            'definition': 'Sum of final API completion_tokens / (last request end - first request start). Includes small-prompt admission, TTFT, decoding and terminal drain. Not decode-only engine TPS.'}}


def run(args, core):
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise RuntimeError('--out must be new or empty')
    client = core.Client(args)
    specs = fixtures(core, args.max_tokens)
    env = {k: str(client.cfg.get('env', {}).get(k, '<absent>')) for k in FLAGS}
    env = {k: v for k, v in env.items() if v == '<absent>' or re.fullmatch(r'[-0-9,]+', v)}
    report = {'version': VERSION, 'synthetic_only': True, 'service_mutations': 0, 'records': [], 'checks': {},
              'expected_piped_mtp': args.expect_piped_mtp, 'config_env_flags': env,
              'max_tokens_per_request': args.max_tokens, 'correctness_dependency_sha256': CORE_SHA256,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    monitor = None
    try:
        before = client.idle()
        report['reported_parallel_slots'] = before['live'].get('parallel')
        if not core.number(report['reported_parallel_slots']) or report['reported_parallel_slots'] < 8:
            raise RuntimeError('Existing server does not report eight slots')
        cursor, begin = client.log_cursor(), time.time()
        monitor = threading.Thread(target=client.monitor, daemon=True)
        monitor.start()
        barrier = threading.Barrier(8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(client.request, spec, barrier) for spec in specs]
            records = [f.result() for f in futures]
        after = client.idle(wait=30)
        time.sleep(.2)  # stderr summary follows the terminal protocol message.
        end = time.time()
        new = [row for key, row in after['rows'].items() if key not in before['rows']]
        delta = {k: after['totals'][k] - before['totals'][k]
                 for k in ('requests', 'drafts_offered', 'drafts_accepted')
                 if k in before['totals'] and k in after['totals']}
        log = client.log_numbers(cursor)
        offered = sum(row['drafts_offered'] for row in log['records'])
        report.update(records=records, timing=timing(records), numeric_samples=client.samples,
                      numeric_metrics_before=before, numeric_metrics_after=after,
                      metrics_new_requests=new, metrics_totals_delta=delta,
                      numeric_piped_log=log, piped_drafts_offered=offered,
                      piped_drafts_accepted=sum(row['drafts_accepted'] for row in log['records']),
                      max_active=max((s.get('work', {}).get('active', 0) for s in client.samples), default=0),
                      max_own_in_flight=max((s.get('own_in_flight', 0) for s in client.samples), default=0))
        report['checks'] = {
            'eight_requests_complete': len(records) == 8 and all(core.completed(r) for r in records),
            'all_exact_requested_output_length': len(records) == 8 and all(
                r['finish_reason'] == 'length' and r['usage'].get('completion_tokens') == args.max_tokens
                for r in records),
            'four_sampled_four_greedy': sorted(s['overrides']['temperature'] for s in specs) == [0, 0, 0, 0, .8, .8, .8, .8],
            'eight_stream_common_interval_at_least_one_second': report['timing']['overlap']['eight_streams_observed'],
            'exclusive_request_identity': len(new) == 8 and delta.get('requests') == 8
                and all(begin - 1 <= row.get('time', -1) <= end + 1 for row in new),
            'piped_route_confirmed': offered > 0 if args.expect_piped_mtp == 'on' else offered == 0,
        }
        report['route_note'] = 'Positive numeric pipelined stderr proposals prove the on route. BDONE draft counts are absent in /metrics; zero proposals on off is consistency, not proof of absence.'
        report['all_checks_passed'] = all(report['checks'].values())
    except Exception as error:
        report.update(fatal_error_kind=type(error).__name__, records=client.records, all_checks_passed=False)
    finally:
        client.monitor_stop.set()
        if monitor:
            monitor.join(timeout=35)
    core.save(args.out / 'c8-check.json', report)
    print(json.dumps({'checks': report['checks'], 'all_checks_passed': report['all_checks_passed'],
                      'whole_batch': report.get('timing', {}).get('whole_batch')}))
    return 0 if report['all_checks_passed'] else 2


def compare(args, core):
    left, right = [json.loads((path / 'c8-check.json').read_text()) for path in args.compare]
    a, b = ({r['name']: r for r in report['records']} for report in (left, right))
    rows = []
    for name in sorted(set(a) | set(b)):
        x, y = a.get(name), b.get(name)
        if x is None or y is None:
            rows.append({'name': name, 'passed': False, 'missing_request': True}); continue
        usage = core.usage_comparison(x['usage'], y['usage'])
        checks = {key + '_equal': x[key] == y[key] for key in
                  ('request_sha256', 'output_sha256', 'reasoning_sha256', 'finish_reason')}
        checks['both_complete'] = core.completed(x) and core.completed(y)
        checks['logical_usage_equal'] = usage['logical_counts_equal']
        rows.append({'name': name, 'checks': checks, 'passed': all(checks.values()), 'usage_comparison': usage})
    report = {'version': VERSION, 'comparisons': rows,
              'same_version': left['version'] == right['version'] == VERSION,
              'both_runs_passed': left['all_checks_passed'] and right['all_checks_passed'],
              'left_flags': left.get('config_env_flags'), 'right_flags': right.get('config_env_flags'),
              'left_timing': left.get('timing'), 'right_timing': right.get('timing'),
              'usage_metadata_qualifications': [r['name'] for r in rows if r.get('usage_comparison') and not r['usage_comparison']['metadata_equal']]}
    report['all_checks_passed'] = len(rows) == 8 and all(r['passed'] for r in rows) and report['same_version'] and report['both_runs_passed']
    args.out.mkdir(parents=True, exist_ok=True)
    core.save(args.out / 'c8-comparison.json', report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'requests_exact': sum(r['passed'] for r in rows)}))
    return 0 if report['all_checks_passed'] else 2


def self_test(core):
    records = []
    for i in range(8):
        records.append({'name': str(i), 'start_epoch': 0.1 * i, 'end_epoch': 5 + .1 * i,
                        'first_output_epoch': 1 + .1 * i, 'last_output_epoch': 4 + .1 * i,
                        'saw_done': True, 'error': None, 'finish_reason': 'length',
                        'deltas': [{'epoch': 2}, {'epoch': 3}, {'epoch': 4}], 'usage': {'completion_tokens': 1024}})
    result = timing(records)
    assert result['overlap']['eight_streams_observed']
    assert abs(result['overlap']['common_interval_s'] - 2.3) < 1e-9
    assert abs(result['whole_batch']['completed_api_tokens_per_s'] - 8192 / 5.7) < 1e-9
    records[0]['last_output_epoch'] = 2
    assert not timing(records)['overlap']['eight_streams_observed']
    specs = fixtures(core, 1024)
    assert len(specs) == 8 and len({s['seed'] for s in specs}) == 8
    assert sum(s['overrides']['temperature'] == 0 for s in specs) == 4
    assert all(s['overrides']['chat_template_kwargs']['enable_thinking'] is False for s in specs)
    usage = {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
    assert core.usage_comparison(usage, {**usage, 'prompt_tokens_details': {'cached_tokens': 9}})['logical_counts_equal']
    assert not core.usage_comparison(usage, {**usage, 'completion_tokens': 21})['logical_counts_equal']
    print(json.dumps({'offline_self_test': 'PASS', 'network_requests': 0, 'gpu_operations': 0}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path); p.add_argument('--out', type=Path)
    p.add_argument('--port', type=int, default=8092); p.add_argument('--engine-log', type=Path)
    p.add_argument('--execute', action='store_true'); p.add_argument('--exclusive-benchmark', action='store_true')
    p.add_argument('--expect-piped-mtp', choices=('on', 'off'), default='on')
    p.add_argument('--max-tokens', type=int, choices=(768, 1024), default=1024)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--compare', type=Path, nargs=2); p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        return self_test(core_module())
    if args.compare:
        if args.out is None:
            p.error('--compare requires --out')
        return compare(args, core_module())
    if not args.execute:
        print(json.dumps({'version': VERSION, 'dry_run': True, 'requests': 8, 'max_tokens': args.max_tokens,
                          'thinking': False, 'sampled': 4, 'greedy': 4, 'simultaneous_start_barrier': 8,
                          'required_common_sse_interval_s': 1.0, 'injections': 0,
                          'execution_guard': '--execute --exclusive-benchmark --config --out; existing idle loopback server'}, indent=2))
        return 0
    if not args.exclusive_benchmark or args.config is None or args.out is None:
        p.error('--execute requires --exclusive-benchmark --config --out')
    if not 1 <= args.port <= 65535 or args.timeout < 1:
        p.error('invalid numeric option')
    return run(args, core_module())


if __name__ == '__main__':
    raise SystemExit(main())
