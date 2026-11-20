#!/usr/bin/env python3
"""Standalone 100k cold/warm prefill guard. Dry-run unless explicitly executed.

Only sends synthetic requests to an already-loaded, idle loopback :8092 service.
Never starts/stops/reconfigures a service or clears its cache. Saves hashes and
numeric timings only, never prompts, generations, SSE bodies, API keys or log lines.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
import re
import statistics
import time
from pathlib import Path

ENGINE_SHA = None
LIB_SHA = None
FLAGS = ('STRATA_PIPED_CONCURRENT_MTP',)
WORDS = ('the river carried old maps past quiet towns where people traded stories about storms, bridges, lanterns, '
         'harvests, letters, clocks, ships and distant mountains while children counted stars above the harbor').split()
TASK = ('Write a Python module with a class LRUCache (get, put, delete, __len__) built on an OrderedDict, '
        'a thread-safe wrapper class using a lock, and a unittest suite with at least six tests. Output only code.')
PERF = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), '
                  r'(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)')
CUDA_ERROR = re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|illegal memory access|device.side assert|'
                        r'CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits', re.I)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def json_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def comparable_config(config):
    value = json.loads(json.dumps(config))
    for key in ('api_key', 'log'):
        value.pop(key, None)
    for key in FLAGS:
        value.get('env', {}).pop(key, None)
    return value


def fixture_text(fixture_id, repeat):
    filler = ' '.join(WORDS[(i * 7 + 8) % len(WORDS)] + ('.' if i % 17 == 16 else '')
                      for i in range(int(100000 / 1.3)))
    prefix = ('Amber', 'Quartz', 'Violet')[repeat] + f' {fixture_id} {repeat} fixed synthetic prefill fixture.'
    return prefix + ' ' + filler + '\n\nIgnoring the text above, ' + TASK


def make_body(model, text, max_tokens):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}], 'temperature': 0,
            'seed': 20261107, 'max_tokens': max_tokens, 'stream': True,
            'stream_options': {'include_usage': True}, 'presence_penalty': 0, 'frequency_penalty': 0,
            'repetition_penalty': 1.0, 'chat_template_kwargs': {'enable_thinking': False}}


def read_status(port, headers):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request('GET', '/v1/status', headers=headers)
        response = connection.getresponse()
        require(response.status == 200, 'status HTTP error')
        data = json.loads(response.read(2 << 20))
        activity, concurrency = data.get('activity') or {}, data.get('concurrency') or {}
        def number(value):
            return value if isinstance(value, int) and not isinstance(value, bool) else None
        return {'loaded': data.get('loaded') is True,
                'requests': number(activity.get('requests')), 'in_flight': number(activity.get('in_flight')),
                'serving': number(concurrency.get('serving')), 'requested': number(concurrency.get('requested')),
                'context': number((data.get('context') or {}).get('native'))}
    finally:
        connection.close()


def read_numeric_log(path, cursor):
    stat = path.stat()
    require((stat.st_dev, stat.st_ino) == cursor[:2] and stat.st_size >= cursor[2], 'engine log rotated/truncated')
    with path.open('rb') as handle:
        handle.seek(cursor[2]); raw = handle.read((4 << 20) + 1)
    require(len(raw) <= (4 << 20), 'unexpected log volume; refusing ambiguous request evidence')
    text = raw.decode(errors='replace')
    records = []
    for match in PERF.finditer(text):
        values = match.groups()
        records.append(dict(zip(('prompt_tokens', 'reused_tokens', 'read_tokens', 'read_ms', 'prefill_tps',
                                 'generated_tokens', 'decode_ms', 'decode_tps'),
                                [int(value) if i not in (4, 7) else float(value) for i, value in enumerate(values)])))
    return records, bool(CUDA_ERROR.search(text))


def run_request(name, kind, body, port, headers, engine_log):
    before = read_status(port, headers)
    require(before['loaded'] and before['in_flight'] == 0, 'service not loaded and idle before request')
    stat = engine_log.stat(); cursor = (stat.st_dev, stat.st_ino, stat.st_size)
    raw = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    epoch, started = time.time(), time.monotonic()
    result = {'name': name, 'kind': kind, 'request_sha256': hashlib.sha256(raw).hexdigest(),
              'messages_sha256': json_sha(body['messages']), 'start_epoch': epoch, 'max_tokens': body['max_tokens'],
              'usage': {}, 'ttft_s': None, 'first_visible_content_s': None, 'last_output_s': None,
              'saw_done': False, 'finish_reason': None, 'error': None, 'output_chunks': 0}
    hashes = {key: hashlib.sha256() for key in ('visible_content', 'reasoning', 'tool')}
    counts = dict.fromkeys(hashes, 0)
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
    try:
        connection.request('POST', '/v1/chat/completions', body=raw, headers=headers)
        response = connection.getresponse()
        result['http_status'] = response.status
        require(response.status == 200, 'request HTTP error')
        pending = []
        for line in response:
            line = line.decode('utf-8').rstrip('\r\n')
            if line.startswith('data:'):
                pending.append(line[5:].lstrip()); continue
            if line or not pending:
                continue
            now = time.monotonic() - started
            payload, pending = '\n'.join(pending), []
            if payload == '[DONE]':
                result['saw_done'] = True; break
            data = json.loads(payload)
            require(not data.get('error'), 'API error')
            usage = data.get('usage') or {}
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if isinstance(usage.get(key), int):
                    result['usage'][key] = usage[key]
            cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
            if isinstance(cached, int):
                result['usage']['cached_tokens'] = cached
            parts = dict.fromkeys(hashes, '')
            for choice in data.get('choices') or []:
                delta = choice.get('delta') or {}
                parts['visible_content'] += delta.get('content') or ''
                parts['reasoning'] += delta.get('reasoning_content') or delta.get('reasoning') or ''
                for call in delta.get('tool_calls') or []:
                    function = call.get('function') or {}
                    parts['tool'] += (function.get('name') or '') + (function.get('arguments') or '')
                reason = choice.get('finish_reason')
                if reason:
                    result['finish_reason'] = reason if reason in ('stop', 'length', 'tool_calls', 'cancel') else 'unknown'
            if any(parts.values()):
                result['ttft_s'] = now if result['ttft_s'] is None else result['ttft_s']
                result['last_output_s'] = now
                result['output_chunks'] += 1
            if parts['visible_content'] and result['first_visible_content_s'] is None:
                result['first_visible_content_s'] = now
            for key, value in parts.items():
                encoded = value.encode(); hashes[key].update(encoded); counts[key] += len(encoded)
    except Exception as error:
        result['error'] = {'kind': type(error).__name__}  # Never include exception/server text.
    finally:
        connection.close()
    result['total_s'] = time.monotonic() - started
    result['end_epoch'] = epoch + result['total_s']
    result['channel_output_sha256'] = {key: value.hexdigest() for key, value in hashes.items()}
    result['output_sha256'] = json_sha(result['channel_output_sha256'])
    result['output_bytes'] = counts
    result['hash_note'] = 'Per-channel text bytes; independent of SSE chunk boundaries. No text is saved.'
    result['decode_arrival_span_s'] = result['last_output_s'] - result['ttft_s'] if result['ttft_s'] is not None else None
    matches, detected = [], False
    after = {'in_flight': None, 'requests': None}
    result['evidence_error'] = None
    try:
        deadline = time.monotonic() + 3
        while True:
            records, detected = read_numeric_log(engine_log, cursor)
            matches = [record for record in records if record['prompt_tokens'] == result['usage'].get('prompt_tokens')
                       and record['generated_tokens'] == result['usage'].get('completion_tokens')]
            if matches or time.monotonic() >= deadline:
                break
            time.sleep(.05)
        after = read_status(port, headers)
        deadline = time.monotonic() + 3
        while after['in_flight'] != 0 and time.monotonic() < deadline:
            time.sleep(.05); after = read_status(port, headers)
    except Exception as error:
        result['evidence_error'] = type(error).__name__
    result['engine_match_count'] = len(matches)
    result['engine'] = matches[0] if len(matches) == 1 else None
    engine = result['engine']; cached = result['usage'].get('cached_tokens')
    result['checks'] = {
        'isolated_request': before['in_flight'] == after['in_flight'] == 0 and
            isinstance(before['requests'], int) and after['requests'] == before['requests'] + 1,
        'stream_completed': result['error'] is None and result['saw_done'] and result['ttft_s'] is not None,
        'max_output_length': result['finish_reason'] == 'length' and result['usage'].get('completion_tokens') == body['max_tokens'],
        'prompt_approximately_100k': 95000 <= result['usage'].get('prompt_tokens', 0) <= 105000,
        'unique_engine_record': len(matches) == 1,
        'numeric_evidence_complete': result['evidence_error'] is None,
        'no_cuda_error_detected': not detected,
        'cold_zero_reuse': bool(engine and engine['reused_tokens'] == 0 and cached in (None, 0)) if kind == 'cold' else None,
        'warm_reuses_over_95pct': bool(engine and engine['reused_tokens'] > .95 * engine['prompt_tokens'] and
                                         (cached is None or cached > .95 * engine['prompt_tokens'])) if kind == 'warm' else None,
    }
    result['valid'] = all(value for value in result['checks'].values() if value is not None)
    return result


def summarize(records):
    result = {'request_count': len(records), 'all_valid': len(records) == 6 and all(r['valid'] for r in records), 'by_kind': {}}
    for kind in ('cold', 'warm'):
        rows = [r for r in records if r['kind'] == kind]
        metrics = {}
        for metric in ('ttft_s', 'read_ms', 'decode_ms', 'decode_tps'):
            values = [r.get(metric) if metric == 'ttft_s' else (r.get('engine') or {}).get(metric) for r in rows]
            values = [value for value in values if value is not None]
            metrics[metric] = {'n': len(values), 'median': statistics.median(values) if values else None,
                               'min': min(values) if values else None, 'max': max(values) if values else None}
        result['by_kind'][kind] = metrics
    result['limits'] = ['Three observations per kind/arm; report variability rather than claiming a stable tail distribution.',
                        'TTFT includes API/admission; read_ms is matched engine prefill/read time, including warm restore.',
                        'Decode is secondary evidence with only 192/256 output tokens; no full long-decode claim.',
                        'Fixed single-request greedy fixtures; this guard does not measure concurrent prefill admission.']
    return result


def compare(directories, out):
    roots = [Path(directory) for directory in directories]
    manifests = [json.loads((root / 'manifest.json').read_text()) for root in roots]
    rows = [json.loads((root / 'results.json').read_text()) for root in roots]
    summaries = [summarize(value) for value in rows]
    maps = [{row['name']: row for row in value} for value in rows]
    pairs = []
    for name in sorted(set(maps[0]) | set(maps[1])):
        left, right = maps[0].get(name), maps[1].get(name)
        pairs.append({'name': name, 'present_in_both': left is not None and right is not None,
                      'same_request': bool(left and right and left['request_sha256'] == right['request_sha256']),
                      'same_greedy_output': bool(left and right and left['output_sha256'] == right['output_sha256']),
                      'same_prompt_tokens': bool(left and right and left['usage'].get('prompt_tokens') == right['usage'].get('prompt_tokens'))})
    comparisons = {}
    for kind in ('cold', 'warm'):
        comparisons[kind] = {}
        for metric in ('ttft_s', 'read_ms', 'decode_ms', 'decode_tps'):
            a, b = [s['by_kind'][kind][metric]['median'] for s in summaries]
            comparisons[kind][metric] = {'off_median': a, 'on_median': b,
                                         'on_relative_change_pct': (b / a - 1) * 100 if a and b is not None else None}
    result = {'off_directory': str(roots[0]), 'on_directory': str(roots[1]), 'pairs': pairs,
              'same_binary_and_library': all(manifests[0][k] == manifests[1][k] for k in ('engine_sha256', 'library_sha256')),
              'same_config_except_mtp_commit_budget_flags_key_log': manifests[0]['comparable_config_sha256'] == manifests[1]['comparable_config_sha256'],
              'correct_arm_order': [m['arm'] for m in manifests] == ['off', 'on'],
              'both_explicit_flags_match_arm': all(m.get('expected_config_flags')=={'STRATA_PIPED_CONCURRENT_MTP':value,'STRATA_PARK_OVERWRITE':'0'} for m,value in zip(manifests,('0','1'))),
              'both_arms_valid': all(s['all_valid'] for s in summaries), 'metrics': comparisons,
              'all_greedy_outputs_identical': len(pairs) == 6 and all(pair['same_greedy_output'] for pair in pairs),
              'note': 'No automatic speed/deployment verdict. Negative latency change is faster; positive throughput change is faster.'}
    result['comparison_eligible'] = (result['same_binary_and_library'] and result['same_config_except_mtp_commit_budget_flags_key_log'] and
        result['correct_arm_order'] and result['both_explicit_flags_match_arm'] and result['both_arms_valid'] and len(pairs) == 6 and
        all(pair['same_request'] and pair['same_prompt_tokens'] for pair in pairs))
    if out:
        require(not out.exists(), 'comparison output already exists'); out.mkdir(parents=True)
        save(out / 'comparison.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def self_test():
    import tempfile
    assert make_body('fixture', 'not saved', 256)['temperature'] == 0
    line = 'strata serve: prompt 101130 tokens = 0 reused + 101130 read in 15345 ms (6591.4 tok/s), 256 generated in 1810 ms (141.4 tok/s)\n'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'engine.log'; path.write_text('old line\n')
        stat = path.stat(); cursor = (stat.st_dev, stat.st_ino, stat.st_size)
        with path.open('a') as handle: handle.write(line)
        records, bad = read_numeric_log(path, cursor)
        assert not bad and len(records) == 1 and records[0]['read_ms'] == 15345 and records[0]['decode_ms'] == 1810
    body1 = make_body('fixture', fixture_text('fixed-test', 0), 256)
    body2 = make_body('fixture', fixture_text('fixed-test', 0), 256)
    body3 = make_body('fixture', fixture_text('fixed-test', 1), 256)
    assert json_sha(body1) == json_sha(body2) and json_sha(body1) != json_sha(body3)
    original_connection, original_status = http.client.HTTPConnection, read_status
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine.log'; path.write_text('')
            for kind, reused in [('cold', 0), ('warm', 101129)]:
                pieces = [{'choices': [{'delta': {'role': 'assistant', 'content': ''}}]},
                          {'choices': [{'delta': {'content': 'SYNTHETIC_OUTPUT_NOT_SAVED'}}]},
                          {'choices': [{'delta': {}, 'finish_reason': 'length'}],
                           'usage': {'prompt_tokens': 101130, 'completion_tokens': 256,
                                     'prompt_tokens_details': {'cached_tokens': reused}}}]
                stream = b''.join(b'data: ' + json.dumps(piece).encode() + b'\n\n' for piece in pieces) + b'data: [DONE]\n\n'
                class FakeResponse:
                    status = 200
                    def __iter__(self): return iter(stream.splitlines(keepends=True))
                class FakeConnection:
                    def __init__(self, host, port, timeout): assert host == '127.0.0.1' and port == 8092
                    def request(self, method, endpoint, body, headers): assert method == 'POST'
                    def getresponse(self):
                        with path.open('a') as handle:
                            handle.write(line.replace('0 reused + 101130', f'{reused} reused + {101130 - reused}'))
                        return FakeResponse()
                    def close(self): pass
                counter = iter((0, 1))
                def fake_status(port, headers): return {'loaded': True, 'in_flight': 0, 'requests': next(counter)}
                http.client.HTTPConnection = FakeConnection
                globals()['read_status'] = fake_status
                row = run_request(kind, kind, body1, 8092, {'Authorization': 'Bearer FAKE_SECRET_NOT_SAVED'}, path)
                assert row['valid'] and row['engine']['read_ms'] == 15345 and row['usage']['completion_tokens'] == 256
                encoded = json.dumps(row)
                assert 'SYNTHETIC_OUTPUT_NOT_SAVED' not in encoded and 'FAKE_SECRET_NOT_SAVED' not in encoded
    finally:
        http.client.HTTPConnection = original_connection
        globals()['read_status'] = original_status
    print('OFFLINE PASS: fixed prompts, greedy parameters, fake SSE cold/warm, exact numeric log match and sanitized records; no network/GPU')


def main():
    global ENGINE_SHA, LIB_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--engine-sha256')
    parser.add_argument('--target-head', choices=('0', '1'), default='1')
    parser.add_argument('--library-sha256')
    parser.add_argument('--exclusive-benchmark', action='store_true')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--engine-log', type=Path, help='Defaults to config.log; must be the isolated engine log')
    parser.add_argument('--port', type=int, choices=(8092,))
    parser.add_argument('--arm', choices=('off', 'on'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fixture-id', default='batch-mtp-prefill-v1')
    parser.add_argument('--max-tokens', type=int, choices=(192, 256), default=256)
    parser.add_argument('--compare', nargs=2, metavar=('OFF_DIR', 'ON_DIR'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test or args.compare:
        require(not args.execute, 'offline actions cannot execute network requests')
        if args.self_test: self_test()
        else: compare(args.compare, args.out)
        return
    if not args.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_NETWORK', 'arm': args.arm, 'fixture_id': args.fixture_id,
                          'requests': ['cold-1', 'warm-1', 'cold-2', 'warm-2', 'cold-3', 'warm-3'],
                          'nominal_prompt_tokens': 100000, 'max_tokens': args.max_tokens, 'temperature': 0,
                          'seed': 20261107, 'port': 8092, 'engine_sha256': ENGINE_SHA,
                          'required': '--execute --exclusive-benchmark --port 8092 --arm --config --out'}, indent=2))
        return
    if not (args.exclusive_benchmark and args.port == 8092 and args.arm and args.config and args.out):
        parser.error('execution requires --exclusive-benchmark --port 8092 --arm --config --out')
    require(all(isinstance(x, str) and re.fullmatch(r'[0-9a-f]{64}', x) for x in (args.engine_sha256, args.library_sha256)), 'Explicit engine/library SHA256 required')
    ENGINE_SHA, LIB_SHA = args.engine_sha256, args.library_sha256
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,96}', args.fixture_id) is not None, 'fixture id must be a public ASCII label')
    require(not args.out.exists(), 'output must be a new directory')
    config = json.loads(args.config.read_text())
    require(isinstance(config.get('reasoning_budget_engine_cap'), bool), 'config budget cap must be explicit')
    expected = '1' if args.arm == 'on' else '0'
    expected_flags = {'STRATA_PIPED_CONCURRENT_MTP':'1' if args.arm == 'on' else '0','STRATA_PARK_OVERWRITE':'0'}
    fixed_flags = {'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1', 'STRATA_PIPED_TARGET_HEAD': args.target_head, 'STRATA_PIPED_ASYNC_SAMPLE': '0'}
    require(all(config.get('env', {}).get(k) == v for k, v in fixed_flags.items()), 'fixed batch MTP flags differ')
    require(config.get('reasoning_budget_engine_cap') is True, 'budget cap must be on')
    require(all(str(config.get('env', {}).get(flag)) == expected_flags[flag] for flag in expected_flags), 'config does not match requested env arm')
    binary = Path(config['exe']); library = binary.parent / 'libstrata_kq.so'
    require(file_sha(binary) == ENGINE_SHA and file_sha(library) == LIB_SHA, 'configured binary/library differs from fixed b4')
    engine_log = args.engine_log or Path(config['log'])
    require(engine_log.is_file(), 'isolated engine log is required for exact read_ms/decode evidence')
    headers = {'Content-Type': 'application/json'}
    if config.get('api_key'):
        headers['Authorization'] = 'Bearer ' + config['api_key']
    initial = read_status(args.port, headers)
    require(initial['loaded'] and initial['in_flight'] == 0, 'service must already be loaded and idle')
    require(initial['serving'] == initial['requested'] == 8 and isinstance(initial.get('context'), int) and initial['context'] >= 120000,
            'expected eight slots and sufficient existing context')
    args.out.mkdir(parents=True)
    manifest = {'arm': args.arm, 'expected_config_flags': expected_flags, 'fixture_id': args.fixture_id,
                'engine_sha256': ENGINE_SHA, 'library_sha256': LIB_SHA, 'script_sha256': file_sha(Path(__file__)),
                'comparable_config_sha256': json_sha(comparable_config(config)),
                'max_tokens': args.max_tokens, 'seed': 20261107, 'started_epoch': time.time(), 'initial_status': initial,
                'validation_scope': 'Hashes validate config target files; external runner must establish the actual process and env.',
                'privacy': 'Synthetic hashes and numeric counters only; no credentials, prompt/output text or raw logs.'}
    save(args.out / 'manifest.json', manifest)
    records = []
    try:
        for repeat in range(3):
            body = make_body(config.get('model_name', 'strata'), fixture_text(args.fixture_id, repeat), args.max_tokens)
            for kind in ('cold', 'warm'):
                name = f'{kind}-{repeat + 1}'; print('START', name, flush=True)
                record = run_request(name, kind, body, args.port, headers, engine_log)
                records.append(record); save(args.out / (name + '.json'), record)
                save(args.out / 'results.json', records); save(args.out / 'summary.json', summarize(records))
                require(record['valid'], 'request validation failed; subsequent workload skipped')
                print('FINISH', name, 'read_ms', record['engine']['read_ms'], flush=True)
        manifest['completed'] = True
    except (Exception, KeyboardInterrupt) as error:
        manifest['completed'] = False; manifest['failure_kind'] = type(error).__name__
        raise SystemExit('Prefill check failed; inspect numeric saved checks. No further request sent.')
    finally:
        manifest['ended_epoch'] = time.time(); save(args.out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
