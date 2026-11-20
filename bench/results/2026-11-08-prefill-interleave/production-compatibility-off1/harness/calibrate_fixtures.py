#!/usr/bin/env python3
"""Freeze synthetic word counts using the serving tokenizer, outside timed runs.

Default dry-run. --execute sends only count_tokens POSTs unless --probe-one-token
is also given. A probe sends one max_tokens=1 non-streaming chat request for each
fixture and verifies chat prompt usage equals the count endpoint. No service/GPU
administration and no cache clear. Private configs/keys/responses are not saved.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import fixture_prompts as fp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, why):
    if not value:
        raise RuntimeError(why)


def post(port, key, path, body, timeout=180):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        raw = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        conn.request('POST', path, raw, {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        response = conn.getresponse()
        require(response.status == 200, 'Local API returned HTTP ' + str(response.status))
        value = json.loads(response.read(4 << 20))
        require(not value.get('error'), 'Local API returned an error')
        return value
    finally:
        conn.close()


def calibrate(spec, count, tolerance):
    memo = {}
    def at(words):
        if words not in memo:
            memo[words] = count(fp.prompt(words, spec['fixture']))
        return memo[words]
    target = spec['target_tokens']
    lo, hi = 0, target
    require(at(lo) < target, 'Target shorter than template/task overhead')
    while at(hi) < target:
        hi *= 2
        require(hi <= 1000000, 'Cannot bracket synthetic prompt size')
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if at(mid) <= target:
            lo = mid
        else:
            hi = mid
    words = min((lo, hi), key=lambda n: (abs(at(n) - target), n))
    actual = at(words)
    require(abs(actual - target) <= tolerance, 'Calibrated prompt outside absolute token tolerance')
    return {**spec, 'words': words, 'counted_prompt_tokens': actual,
            'target_error_tokens': actual - target,
            'prompt_sha256': hashlib.sha256(fp.prompt(words, spec['fixture']).encode()).hexdigest(),
            'calibration_calls': [{'words': n, 'input_tokens': v} for n, v in memo.items()]}


def validate_count(value):
    n = value.get('input_tokens')
    require(type(n) is int and n > 0, 'Token-count response missing a positive integer')
    return n


def self_test():
    spec = fp.specs('offline', ('primary',))[0]
    seen = []
    # Synthetic strictly monotone counter exercises bracket and exact selection.
    def count(text):
        words = text.split('\n')[1].split()
        seen.append(len(words)); return 23 + 2 * len(words)
    result = calibrate(spec, count, 1)
    assert abs(result['counted_prompt_tokens'] - 10000) == 1
    assert len(seen) == len(set(seen))
    assert result['prompt_sha256'] == hashlib.sha256(fp.prompt(result['words'], result['fixture']).encode()).hexdigest()
    assert len(fp.specs('offline')) == 11
    for bad in ({}, {'input_tokens': True}, {'input_tokens': -1}):
        try: validate_count(bad)
        except RuntimeError: pass
        else: raise AssertionError('Invalid count accepted')
    print(json.dumps({'offline_checks_passed': True, 'http_requests': 0, 'gpu_operations': 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--probe-one-token', action='store_true')
    p.add_argument('--config', type=Path)
    p.add_argument('--config-sha256')
    p.add_argument('--server', type=Path)
    p.add_argument('--server-sha256')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--fixture-id', default='prefill-interleave-20261108-v1')
    p.add_argument('--cases', default='primary,long')
    p.add_argument('--token-tolerance', type=int, default=2)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.self_test:
        require(not a.execute, 'Self-test cannot execute'); self_test(); return
    cases = a.cases.split(',')
    require(cases and not set(cases) - {'primary', 'long'} and len(set(cases)) == len(cases), 'Invalid unique cases')
    items = fp.specs(a.fixture_id, cases)
    if not a.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_HTTP', 'fixtures': items,
                          'probe_max_one_token_each': a.probe_one_token,
                          'count_endpoint': '/v1/messages/count_tokens',
                          'note': 'Actual OpenAI prompt usage must later equal the frozen count; endpoint equality is not assumed.'}, indent=2)); return
    require(a.port in (8080, 8092), 'Only approved local service ports supported')
    require(a.config is not None and a.server is not None and a.out is not None, 'Explicit config/server/out required')
    require(not a.out.exists(), 'Output already exists; refuse overwrite')
    require(a.config_sha256 and sha(a.config) == a.config_sha256, 'Private configuration SHA differs')
    require(a.server_sha256 and sha(a.server) == a.server_sha256, 'Server source SHA differs')
    cfg = json.loads(a.config.read_text())
    key, model = cfg.get('api_key'), cfg.get('model_name', 'strata')
    require(isinstance(key, str) and key, 'Authentication key missing')
    before_config, before_server = sha(a.config), sha(a.server)
    report = {'format': 1, 'fixture_id': a.fixture_id, 'cases': cases, 'created_epoch': time.time(),
              'server_sha256': before_server, 'source_config_sha256': before_config,
              'model': model, 'thinking_enabled': False, 'token_tolerance': a.token_tolerance,
              'fixture_generator_sha256': sha(Path(fp.__file__)), 'calibrator_sha256': sha(Path(__file__)),
              'probed_one_token_each': a.probe_one_token, 'fixtures': [],
              'scope': 'Preflight synthetic token calibration; not a performance measurement. Count endpoint is inference-free.'}
    for spec in items:
        row = calibrate(spec, lambda text: validate_count(post(a.port, key, '/v1/messages/count_tokens', fp.count_body(text, model))), a.token_tolerance)
        if a.probe_one_token:
            response = post(a.port, key, '/v1/chat/completions', fp.chat_body(fp.prompt(row['words'], row['fixture']), model, 1))
            usage = response.get('usage') or {}
            numeric = {k: usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            require(all(type(v) is int for v in numeric.values()), 'Probe usage incomplete')
            require(numeric['prompt_tokens'] == row['counted_prompt_tokens'], 'Count endpoint and chat prompt usage differ')
            require(0 < numeric['completion_tokens'] <= 1 and numeric['total_tokens'] == sum(numeric[k] for k in ('prompt_tokens', 'completion_tokens')), 'Probe usage invalid')
            row['probe_usage'] = numeric
            row['probe_count_matches_chat'] = True
        report['fixtures'].append(row)
        print(json.dumps({'fixture': row['name'], 'target': row['target_tokens'], 'actual': row['counted_prompt_tokens'], 'words': row['words'], 'probe_ok': row.get('probe_count_matches_chat')}), flush=True)
    require(sha(a.config) == before_config and sha(a.server) == before_server, 'Frozen inputs changed during calibration')
    report['completed_epoch'] = time.time()
    require(key not in json.dumps(report), 'Credential detected in report')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'calibration_complete': True, 'out': str(a.out), 'fixtures': len(items), 'performance_claim': False}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'calibration_complete': False, 'error': str(error) if isinstance(error, RuntimeError) else type(error).__name__}), flush=True)
        raise SystemExit(1)
