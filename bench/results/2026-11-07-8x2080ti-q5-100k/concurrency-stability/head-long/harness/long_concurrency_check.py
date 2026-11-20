#!/usr/bin/env python3
"""Four cold ~100k synthetic requests, two sampled/two greedy; dry-run default.

Client only: never starts/stops/configures a service. Explicit --execute and
--exclusive-benchmark are required, on loopback :8092 only. Full output hashes,
numeric usage and chunk arrival times are saved; no prompts, output text, keys or
raw server logs. An external runner must establish actual process/config identity.
"""
import argparse
import copy
import hashlib
import http.client
import importlib.util
import json
import math
import re
import tempfile
import time
from pathlib import Path

VERSION = 'long-concurrency-check-1'
DEPENDENCIES = {'correctness.py': 'b3e32b596882bad35c9b6dd18f7c995615a7b4cc1ac1674cbf1bf3d06ed13a94',
                'prefill_check.py': '6822ff7f3cd58fcf3a4e58a237ba7f23d41b68c8f27ca4a84297a3e4c77cff15'}
VARIABLES = ('STRATA_BATCH_MTP', 'STRATA_PIPED_MTP', 'STRATA_BATCH_SELF_COMMIT')
COMPARABLE_VARIABLES = (*VARIABLES, 'STRATA_PIPED_MTP_HEAD')
SEEDS = (17, 29, 101, 102)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_dependency(name):
    path = Path(__file__).with_name(name)
    require(sha(path) == DEPENDENCIES[name], 'Frozen dependency hash differs: ' + name)
    spec = importlib.util.spec_from_file_location('long_check_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(core, prefill, fixture_id):
    # Different first words, not just a suffix: no lane may reuse another lane's
    # long prefix. Reuse the known ~101k filler; actual usage is the hard gate.
    text = prefill.fixture_text(fixture_id, 0).rsplit(prefill.TASK, 1)[0] + core.CODE
    return [{'name': 'lane-' + str(i), 'prompt': prefix + ' ' + fixture_id + '\n' + text,
             'seed': seed, 'max_tokens': 512,
             'overrides': {'temperature': .8 if i % 2 == 0 else 0.0}}
            for i, (prefix, seed) in enumerate(zip(('Copper', 'Ivory', 'Saffron', 'Cobalt'), SEEDS))]


def stats(values):
    values = sorted(values)
    def percentile(f):
        if not values:
            return None
        p = (len(values) - 1) * f
        lo, hi = math.floor(p), math.ceil(p)
        return values[lo] + (values[hi] - values[lo]) * (p - lo)
    return {'count': len(values), 'p50': percentile(.5), 'p95': percentile(.95),
            'p99': percentile(.99), 'max': max(values) if values else None}


def client_type(core):
    class LongClient(core.Client):
        def request(self, spec, barrier=None, cancel=False, cancel_ready=None, name=None):
            require(not cancel, 'This workload has no cancelled requests')
            name = name or spec['name']
            body = {'model': self.model, 'messages': [{'role': 'user', 'content': spec['prompt']}],
                    'stream': True, 'stream_options': {'include_usage': True}, **core.BASE,
                    'seed': spec['seed'], 'max_tokens': 512, **spec['overrides']}
            raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
            if barrier:
                barrier.wait(timeout=30)
            started, epoch = time.monotonic(), time.time()
            conn = http.client.HTTPConnection('127.0.0.1', self.args.port, timeout=self.args.timeout)
            record = {'name': name, 'request_sha256': hashlib.sha256(raw).hexdigest(),
                      'messages_sha256': core.digest(body['messages']), 'seed': spec['seed'],
                      'temperature': body['temperature'], 'max_tokens': 512, 'thinking': False,
                      'start_epoch': epoch, 'first_output_epoch': None, 'last_output_epoch': None,
                      'usage': {}, 'finish_reason': None, 'saw_done': False, 'error': None, 'deltas': []}
            hashes = {k: hashlib.sha256() for k in ('content', 'reasoning', 'tool')}
            counts = dict.fromkeys(hashes, 0)
            with self.lock:
                self.active.add(name); self.connections[name] = conn
            try:
                conn.request('POST', '/v1/chat/completions', body=raw, headers=self.headers)
                # Keep the underlying socket even if HTTP/1.0 detaches it, so the
                # cohort watchdog can interrupt a blocked response.readline().
                with self.lock:
                    self.connections[name] = conn.sock
                response = conn.getresponse()
                require(response.status == 200, 'HTTP error')
                pending = []
                while not self.abort.is_set():
                    line = response.readline()
                    if not line:
                        break
                    line = line.decode('utf-8').rstrip('\r\n')
                    if line.startswith('data:'):
                        pending.append(line[5:].lstrip()); continue
                    if line or not pending:
                        continue
                    payload, pending = '\n'.join(pending), []
                    if payload == '[DONE]':
                        record['saw_done'] = True; break
                    event = json.loads(payload)
                    require(not event.get('error'), 'API error')
                    if event.get('usage'):
                        record['usage'] = core.numeric_tree(event['usage'])
                    parts = dict.fromkeys(hashes, '')
                    for choice in event.get('choices') or []:
                        delta = choice.get('delta') or {}
                        parts['content'] += delta.get('content') or ''
                        parts['reasoning'] += delta.get('reasoning_content') or delta.get('reasoning') or ''
                        for call in delta.get('tool_calls') or []:
                            function = call.get('function') or {}
                            parts['tool'] += (function.get('name') or '') + (function.get('arguments') or '')
                        reason = choice.get('finish_reason')
                        if reason:
                            record['finish_reason'] = reason if reason in ('length', 'stop', 'tool_calls') else 'other'
                    if any(parts.values()):
                        now = time.time()
                        if record['first_output_epoch'] is None:
                            record['first_output_epoch'] = now
                            with self.lock:
                                self.first[name] = now
                        record['last_output_epoch'] = now
                        record['deltas'].append({'epoch': now, 'elapsed_s': time.monotonic() - started,
                                                'bytes': {k: len(v.encode()) for k, v in parts.items()}})
                    for key, value in parts.items():
                        encoded = value.encode(); hashes[key].update(encoded); counts[key] += len(encoded)
                response.close()
                if self.abort.is_set():
                    record['error'] = {'kind': 'cohort_timeout'}
            except Exception as error:
                record['error'] = {'kind': type(error).__name__}
            finally:
                conn.close()
                record['end_epoch'] = time.time()
                record['total_s'] = time.monotonic() - started
                with self.lock:
                    self.active.discard(name); self.connections.pop(name, None)
            record['channel_sha256'] = {k: h.hexdigest() for k, h in hashes.items()}
            record['output_sha256'] = core.digest(record['channel_sha256'])
            record['output_bytes'] = counts
            record['ttft_s'] = record['deltas'][0]['elapsed_s'] if record['deltas'] else None
            times = [d['elapsed_s'] for d in record['deltas']]
            record['inter_output_chunk_gap_s'] = stats([b - a for a, b in zip(times, times[1:])])
            record['gap_scope'] = 'Nonempty content/reasoning/tool SSE arrivals; not per-token TBT. First output delay is TTFT.'
            with self.lock:
                self.records.append(record)
                core.save(self.args.out / (name + '.json'), record)
            print(json.dumps({'request': name, 'done': record['saw_done'], 'error': record['error'],
                              'usage': record['usage'], 'ttft_s': record['ttft_s']}), flush=True)
            return record
    return LongClient


def record_checks(record):
    usage = record.get('usage', {})
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
    logical = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'completed': not record.get('error') and record.get('saw_done') is True,
            'actual_95k_to_105k': integer(logical[0]) and 95000 <= logical[0] <= 105000,
            'all_512_tokens': record.get('finish_reason') == 'length' and logical[1] == 512,
            'complete_consistent_usage': all(integer(v) for v in logical) and logical[2] == logical[0] + logical[1],
            'explicit_zero_cached_tokens': integer(cached) and cached == 0,
            'visible_code_only': record.get('output_bytes', {}).get('content', 0) > 0
                and record.get('output_bytes', {}).get('reasoning') == 0
                and record.get('output_bytes', {}).get('tool') == 0}


def summarize(core, records, cohort, expected):
    overlap = core.overlap(records) if len(records) == 4 else None
    checks = {r['name']: record_checks(r) for r in records}
    start = min((r['start_epoch'] for r in records), default=None)
    end = max((r['end_epoch'] for r in records), default=None)
    whole = end - start if start is not None and end is not None else None
    skew = max((r['start_epoch'] for r in records), default=0) - (start or 0)
    total = sum(r.get('usage', {}).get('completion_tokens', 0) for r in records)
    logs = cohort.get('piped_numeric_log', {})
    offered = cohort.get('piped_drafts_offered', 0)
    gates = {'four_complete_cold_100k_requests': len(records) == 4 and all(all(c.values()) for c in checks.values()),
             'two_sampled_two_greedy': len(records) == 4 and sorted(r['temperature'] for r in records) == [0, 0, .8, .8],
             'synchronized_start_within_one_second': len(records) == 4 and skew <= 1,
             'four_common_output_at_least_one_second': bool(overlap and overlap['common_decode_s'] >= 1
                 and all(n >= 2 for n in overlap['deltas_per_request_in_common_window'].values())),
             'metrics_exclusive_four_requests': cohort.get('metrics_request_identity_and_exclusivity_check') is True,
             'numeric_piped_log_available': logs.get('available') is True,
             'piped_route_evidence': offered > 0 if expected == 'on' else offered == 0}
    return {'request_checks': checks, 'checks': gates, 'all_checks_passed': all(gates.values()),
            'whole_batch_s': whole, 'whole_batch_completion_tokens': total,
            'whole_batch_tps': total / whole if whole and whole > 0 else None,
            'request_start_skew_s': skew, 'ttft_s': stats([r['ttft_s'] for r in records if r['ttft_s'] is not None]),
            'overlap': overlap,
            'limits': ['Whole batch TPS includes all cold prefill/admission and final drain; not pure decode TPS.',
                       'Common SSE arrival span proves concurrent live output, not simultaneous GPU execution.',
                       'Chunk gaps cannot establish per-token TBT; MTP may co-emit multiple tokens.',
                       'One cohort per arm has no confidence interval; cached_tokens must be explicitly zero.',
                       'Positive numeric piped draft summaries confirm the on route; zero off counts are only consistent with off.']}


def normalized_config(config):
    value = copy.deepcopy(config)
    for key in ('api_key', 'log'):
        value.pop(key, None)
    for key in COMPARABLE_VARIABLES:
        value.get('env', {}).pop(key, None)
    return value


def run(args, core, prefill):
    require(args.exclusive_benchmark, '--execute requires --exclusive-benchmark')
    require(not args.out.exists() or not any(args.out.iterdir()), '--out must be new or empty')
    client = client_type(core)(args)
    require(client.log and client.log.is_file(), 'A readable isolated engine log is required')
    flags = {key: str(client.cfg.get('env', {}).get(key, '<absent>')) for key in COMPARABLE_VARIABLES}
    expected = args.expect_piped_mtp
    if expected == 'auto':
        expected = 'on' if flags['STRATA_BATCH_MTP'] == flags['STRATA_PIPED_MTP'] == '1' else 'off'
    require(all(flags[k] == ('1' if expected == 'on' else '0') for k in VARIABLES[:2]), 'Config flags do not match expected arm')
    initial = prefill.read_status(args.port, client.headers)
    require(initial['loaded'] and initial['in_flight'] == 0, 'Server must be loaded and idle')
    require(isinstance(initial['serving'], int) and initial['serving'] >= 4
            and isinstance(initial['context'], int) and initial['context'] >= 105512,
            'Four serving slots and sufficient native context are required')
    args.out.mkdir(parents=True, exist_ok=True)
    binary = Path(client.cfg['exe'])
    manifest = {'version': VERSION, 'fixture_id': args.fixture_id, 'expected_piped_mtp': expected,
                'config_flags': flags, 'port': args.port, 'cohort_timeout_s': args.timeout,
                'source_sha256': sha(Path(__file__)), 'dependencies_sha256': DEPENDENCIES,
                'engine_sha256': sha(binary), 'adjacent_library_sha256': sha(binary.parent / 'libstrata_kq.so'),
                'comparable_config_sha256': core.digest(normalized_config(client.cfg)),
                'initial_status': initial, 'started_epoch': time.time(), 'service_mutations': 0,
                'identity_scope': 'Target-file hashes only. External runner manifest proves actual process, server source and environment.',
                'privacy': 'Only synthetic hashes, numeric usage and arrival times; no prompt/output text, API key or raw logs.'}
    core.save(args.out / 'manifest.json', manifest)
    report = {'version': VERSION, 'records': [], 'all_checks_passed': False}
    try:
        specs = fixtures(core, prefill, args.fixture_id)
        cohort = client.cohort('long-four', specs)
        records = sorted(client.records, key=lambda r: r['name'])
        report.update(records=records, cohort=cohort, summary=summarize(core, records, cohort, expected))
        report['all_checks_passed'] = report['summary']['all_checks_passed']
    except Exception as error:
        report.update(fatal_error_kind=type(error).__name__, records=client.records)
    manifest.update(ended_epoch=time.time(), completed=report['all_checks_passed'])
    core.save(args.out / 'manifest.json', manifest)
    core.save(args.out / 'long-concurrency.json', report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'summary': report.get('summary')}), flush=True)
    return 0 if report['all_checks_passed'] else 2


def compare(args, core):
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in args.compare]
    reports = [json.loads((p / 'long-concurrency.json').read_text()) for p in args.compare]
    maps = [{r['name']: r for r in report['records']} for report in reports]
    pairs = []
    for name in sorted(set(maps[0]) | set(maps[1])):
        a, b = maps[0].get(name), maps[1].get(name)
        checks = {'present_in_both': bool(a and b)}
        usage = None
        if a and b:
            checks.update({key + '_equal': a[key] == b[key] for key in
                           ('request_sha256', 'output_sha256', 'channel_sha256', 'finish_reason')})
            usage = core.usage_comparison(a['usage'], b['usage'])
            checks['three_logical_usage_counts_equal'] = usage['logical_counts_equal']
            checks['both_records_qualified'] = all(record_checks(a).values()) and all(record_checks(b).values())
        pairs.append({'name': name, 'checks': checks, 'passed': all(checks.values()), 'usage_comparison': usage})
    identity = {key + '_equal': manifests[0][key] == manifests[1][key] for key in
                ('version', 'fixture_id', 'source_sha256', 'dependencies_sha256', 'engine_sha256',
                 'adjacent_library_sha256', 'comparable_config_sha256')}
    result = {'pairs': pairs, 'identity_checks': identity, 'both_runs_qualified': all(r['all_checks_passed'] for r in reports),
              'expected_piped_modes': [m['expected_piped_mtp'] for m in manifests],
              'allowed_config_variable_differences': {
                  key: [m.get('config_flags', {}).get(key, '<absent>') for m in manifests]
                  for key in COMPARABLE_VARIABLES
                  if manifests[0].get('config_flags', {}).get(key) != manifests[1].get('config_flags', {}).get(key)},
              'summaries': [r.get('summary') for r in reports],
              'scope': 'Offline exact output/channel hashes and prompt/completion/total usage; no network and no automatic deployment verdict.'}
    result['all_checks_passed'] = (len(pairs) == 4 and all(p['passed'] for p in pairs) and all(identity.values())
                                  and result['both_runs_qualified'])
    if args.out:
        require(not args.out.exists() or not any(args.out.iterdir()), 'Comparison output must be new or empty')
        args.out.mkdir(parents=True, exist_ok=True); core.save(args.out / 'comparison.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['all_checks_passed'] else 2


def self_test(core, prefill):
    import contextlib
    import io
    specs = fixtures(core, prefill, 'offline-v1')
    assert specs == fixtures(core, prefill, 'offline-v1')
    assert len({s['prompt'] for s in specs}) == 4 and all(len(s['prompt']) > 400000 for s in specs)
    assert sorted(s['overrides']['temperature'] for s in specs) == [0, 0, .8, .8]
    rows = []
    for i in range(4):
        rows.append({'name': 'lane-' + str(i), 'start_epoch': 10 + i * .001, 'end_epoch': 100 + i,
                     'first_output_epoch': 80 + i, 'last_output_epoch': 99 + i,
                     'deltas': [{'epoch': t} for t in range(80 + i, 100 + i)], 'ttft_s': 70 + i,
                     'error': None, 'saw_done': True, 'finish_reason': 'length', 'temperature': .8 if i % 2 == 0 else 0,
                     'usage': {'prompt_tokens': 101139, 'completion_tokens': 512, 'total_tokens': 101651,
                               'prompt_tokens_details': {'cached_tokens': 0}},
                     'output_bytes': {'content': 2048, 'reasoning': 0, 'tool': 0}})
    cohort = {'piped_numeric_log': {'available': True}, 'piped_drafts_offered': 33,
              'metrics_request_identity_and_exclusivity_check': True}
    assert summarize(core, rows, cohort, 'on')['all_checks_passed']
    assert not summarize(core, rows, cohort, 'off')['all_checks_passed']
    for mutate in ('cached', 'missing_cached', 'short', 'eos', 'reasoning', 'inconsistent_usage', 'no_overlap'):
        changed = copy.deepcopy(rows)
        if mutate == 'cached': changed[0]['usage']['prompt_tokens_details']['cached_tokens'] = 1
        elif mutate == 'missing_cached': changed[0]['usage'].pop('prompt_tokens_details')
        elif mutate == 'short': changed[0]['usage']['prompt_tokens'] = 8793
        elif mutate == 'eos': changed[0]['finish_reason'] = 'stop'
        elif mutate == 'reasoning': changed[0]['output_bytes']['reasoning'] = 1
        elif mutate == 'inconsistent_usage': changed[0]['usage']['total_tokens'] += 1
        else: changed[0]['last_output_epoch'] = 82
        assert not summarize(core, changed, cohort, 'on')['all_checks_passed'], mutate
    original = http.client.HTTPConnection
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'api_key': 'FAKE_KEY_MUST_NOT_SAVE'}))
            args = argparse.Namespace(config=root / 'config.json', port=8092, engine_log=None, out=root, timeout=600)
            events = [{'choices': [{'delta': {'role': 'assistant'}}]},
                      {'choices': [{'delta': {'content': 'SYNTHETIC_CODE_MUST_NOT_SAVE'}}]},
                      {'choices': [{'delta': {}, 'finish_reason': 'length'}], 'usage': rows[0]['usage']}]
            lines = iter((b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events)
                          + b'data: [DONE]\n\n').splitlines(keepends=True))
            class FakeResponse:
                status = 200
                def readline(self): return next(lines, b'')
                def close(self): pass
            class FakeConnection:
                sock = None
                def __init__(self, host, port, timeout): assert host == '127.0.0.1' and port == 8092 and timeout == 600
                def request(self, method, path, body, headers):
                    value = json.loads(body)
                    assert method == 'POST' and path == '/v1/chat/completions'
                    assert value['chat_template_kwargs'] == {'enable_thinking': False} and value['max_tokens'] == 512
                def getresponse(self): return FakeResponse()
                def close(self): pass
            http.client.HTTPConnection = FakeConnection
            client = client_type(core)(args)
            record = client.request(specs[0])
            assert all(record_checks(record).values())
            saved = (root / 'lane-0.json').read_text()
            assert 'FAKE_KEY_MUST_NOT_SAVE' not in saved and 'SYNTHETIC_CODE_MUST_NOT_SAVE' not in saved
            assert specs[0]['prompt'] not in saved
    finally:
        http.client.HTTPConnection = original
    # Run the real offline comparison against synthetic arm artifacts, then
    # independently corrupt text hash, token accounting and source identity.
    with tempfile.TemporaryDirectory() as directory:
        roots = [Path(directory) / arm for arm in ('off', 'on')]
        compare_rows = copy.deepcopy(rows)
        for row in compare_rows:
            row.update(request_sha256='fixed-request-' + row['name'], output_sha256='fixed-output',
                       channel_sha256={'content': 'fixed-content', 'reasoning': 'empty', 'tool': 'empty'})
        for root, arm in zip(roots, ('off', 'on')):
            root.mkdir()
            core.save(root / 'manifest.json', {'version': VERSION, 'fixture_id': 'self-test',
                      'source_sha256': 'self-test-source', 'dependencies_sha256': DEPENDENCIES,
                      'engine_sha256': 'fixed-engine', 'adjacent_library_sha256': 'fixed-library',
                      'comparable_config_sha256': 'fixed-config', 'expected_piped_mtp': arm})
            core.save(root / 'long-concurrency.json', {'records': compare_rows, 'all_checks_passed': True})
        compare_args = argparse.Namespace(compare=roots, out=None)
        with contextlib.redirect_stdout(io.StringIO()):
            assert compare(compare_args, core) == 0
            # Head-only A/B keeps piped MTP on in both arms. The actual allowed
            # env differences stay explicit rather than treating route=arm.
            for root, flag in zip(roots, ('0', '1')):
                manifest = json.loads((root / 'manifest.json').read_text())
                manifest.update(expected_piped_mtp='on', config_flags={'STRATA_PIPED_MTP_HEAD': flag})
                core.save(root / 'manifest.json', manifest)
            assert compare(compare_args, core) == 0
            for key in ('output_sha256', 'usage'):
                changed = copy.deepcopy(compare_rows)
                if key == 'usage': changed[0]['usage']['total_tokens'] += 1
                else: changed[0][key] = 'different-output'
                core.save(roots[1] / 'long-concurrency.json', {'records': changed, 'all_checks_passed': True})
                assert compare(compare_args, core) == 2
            core.save(roots[1] / 'long-concurrency.json', {'records': compare_rows, 'all_checks_passed': True})
            manifest = json.loads((roots[1] / 'manifest.json').read_text())
            manifest['source_sha256'] = 'different-source'
            core.save(roots[1] / 'manifest.json', manifest)
            assert compare(compare_args, core) == 2
    print('OFFLINE PASS: fixed fixtures, cold/length/overlap negative gates, long socket timeout, sanitized fake SSE and exact comparison failures; no network/GPU')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--exclusive-benchmark', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--port', type=int, choices=(8092,))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--engine-log', type=Path)
    parser.add_argument('--expect-piped-mtp', choices=('auto', 'off', 'on'), default='auto')
    parser.add_argument('--fixture-id', default='long-concurrency-100k-v1')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('OFF_DIR', 'ON_DIR'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.execute and (args.compare or args.self_test):
        parser.error('Offline actions cannot combine with --execute')
    core, prefill = load_dependency('correctness.py'), load_dependency('prefill_check.py')
    if args.self_test:
        return self_test(core, prefill)
    if args.compare:
        return compare(args, core)
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,96}', args.fixture_id) is not None, 'Use a public ASCII fixture label')
    if not args.execute:
        specs = fixtures(core, prefill, args.fixture_id)
        print(json.dumps({'version': VERSION, 'dry_run': True, 'network_requests': 0,
                          'fixture_id': args.fixture_id, 'requests': [{k: s[k] for k in ('name', 'seed', 'max_tokens', 'overrides')} for s in specs],
                          'prompt_sha256': [core.digest(s['prompt']) for s in specs],
                          'qualification': 'actual 95–105k, explicit cached0, 512 complete tokens, common output >=1s',
                          'timeout_s': args.timeout,
                          'required': '--execute --exclusive-benchmark --config PATH --port 8092 --out EMPTY_DIR --engine-log LOG --expect-piped-mtp off|on'}, indent=2))
        return 0
    if not args.exclusive_benchmark or not args.config or args.port != 8092 or not args.out or not 300 <= args.timeout <= 3600:
        parser.error('Execution needs both gates, --config, --port 8092, --out, timeout in [300,3600]')
    return run(args, core, prefill)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'failed': True, 'error_kind': type(error).__name__}))
        raise SystemExit(2)
