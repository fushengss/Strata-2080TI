#!/usr/bin/env python3
"""Execute the frozen server's actual live-rate function on known token timelines.

CPU-only audit: no imports of the server, network, model, GPU, service mutation,
or implementation edit. The timelines are illustrative, not production data.
"""
from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

EXPECTED_SERVER_SHA = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
DEFAULT_SERVER = Path('/Users/fu/project/HAI/Strata-decode-head/serve/server.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(path):
    raw = path.read_bytes()
    if sha(raw) != EXPECTED_SERVER_SHA:
        raise RuntimeError('Server bytes differ from the deployed, frozen 40c5 version')
    source = raw.decode()
    tree = ast.parse(source)
    constants, constant_lines = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('RATE_WINDOW_S', 'RATE_MIN_SPAN_S'):
                    constants[target.id] = ast.literal_eval(node.value)
                    constant_lines[target.id] = node.lineno
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Service')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    rate_fn = methods['_rate_of']
    run = methods['_run']
    rate_assignment = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'rate' for t in n.targets)
                           and isinstance(n.value, ast.IfExp))
    deque_call = rate_assignment.value.body
    assert isinstance(deque_call, ast.Call)
    assert ast.unparse(deque_call.func) == 'collections.deque'
    maxlen = next(ast.literal_eval(k.value) for k in deque_call.keywords if k.arg == 'maxlen')
    assert constants == {'RATE_WINDOW_S': 2.0, 'RATE_MIN_SPAN_S': 0.25}
    assert maxlen == 32
    fn_source = ast.get_source_segment(source, rate_fn)
    executable = copy.deepcopy(rate_fn)
    executable.decorator_list = []  # execute the same function body without binding it to a class
    namespace = {**constants, 'time': SimpleNamespace(time=lambda: 100.0)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[executable], type_ignores=[])),
                 str(path) + ':unchanged_rate_of', 'exec'), namespace)
    provenance = {
        'server_path': str(path.resolve()), 'server_sha256': sha(raw),
        'function': 'Service._rate_of', 'function_line': rate_fn.lineno,
        'function_end_line': rate_fn.end_lineno, 'function_source': fn_source,
        'function_source_sha256': sha(fn_source.encode()),
        'constants': constants, 'constant_lines': constant_lines,
        'per_request_deque_line': rate_assignment.lineno,
        'per_request_deque_source': ast.get_source_segment(source, rate_assignment),
        'per_request_deque_maxlen': maxlen,
        'append_per_token_line': methods['_note'].lineno,
        'aggregation_line': methods['_tok_s'].lineno,
        'execution': 'AST extraction; original _rate_of body unchanged; only time.time is a fixed clock',
    }
    return namespace['_rate_of'], constants, maxlen, provenance


def known_timeline(stopped_at, maxlen):
    first, now, tokens_per_second = 70.0, 100.0, 50
    last_tick = int(round((stopped_at - first) * tokens_per_second))
    # First generated token at t=70, then exactly one token every 20ms.
    all_events = [(first + tick / tokens_per_second, tick + 1) for tick in range(last_tick + 1)]
    ring = collections.deque(maxlen=maxlen)
    for event in all_events:
        ring.append(event)
    state = {'busy': True, 'first_token': first, 'generated': len(all_events)}
    new_in_common_window = sum(now - 2.0 < timestamp <= now for timestamp, _ in all_events)
    return now, state, list(ring), new_in_common_window


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, default=DEFAULT_SERVER)
    p.add_argument('--out', type=Path, default=Path(__file__).with_suffix('.json'))
    a = p.parse_args()
    rate_of, constants, maxlen, provenance = extract(a.server)
    cases = []
    for name, stopped_at in [('steady_50_tps', 100.0), ('one_second_without_output', 99.0),
                             ('three_seconds_without_output', 97.0)]:
        now, state, rate, common_tokens = known_timeline(stopped_at, maxlen)
        oldest = next(((t, n) for t, n in rate if now - t <= constants['RATE_WINDOW_S']), None)
        newest = rate[-1]
        use_window = bool(oldest and newest[0] - oldest[0] >= constants['RATE_MIN_SPAN_S'])
        displayed = rate_of(state, rate)
        true_rate = common_tokens / 2.0
        cases.append({
            'name': name, 'now': now, 'seconds_without_output': now - stopped_at,
            'state': state, 'rate_samples': rate,
            'samples_kept': len(rate), 'kept_samples_span_s': rate[-1][0] - rate[0][0],
            'oldest_eligible_sample': oldest, 'newest_sample': newest,
            'branch': 'last_sample_minus_first_sample' if use_window else 'whole_request_mean_fallback',
            'denominator_s': newest[0] - oldest[0] if use_window
                else max(constants['RATE_MIN_SPAN_S'], now - state['first_token']),
            'displayed_per_request_tps': displayed,
            'actual_common_window_s': 2.0, 'actual_common_window_new_tokens': common_tokens,
            'actual_common_window_per_request_tps': true_rate,
            'eight_identical_requests_displayed_sum_tps': displayed * 8,
            'eight_identical_requests_actual_common_window_tps': true_rate * 8,
        })
    assert abs(cases[0]['displayed_per_request_tps'] - 50) < 1e-8
    assert cases[0]['actual_common_window_per_request_tps'] == 50
    assert abs(cases[1]['displayed_per_request_tps'] - 50) < 1e-8
    assert cases[1]['actual_common_window_per_request_tps'] == 25
    assert cases[2]['actual_common_window_per_request_tps'] == 0
    assert cases[2]['displayed_per_request_tps'] > 45
    result = {
        'purpose': 'Reproduce the semantics of the deployed webpage Decode now / legacy ANSWER rate',
        'synthetic_timeline_only': True, 'network_or_gpu_used': False, 'source_modified': False,
        'provenance': provenance, 'script_sha256': sha(Path(__file__).read_bytes()),
        'all_checks_passed': True, 'cases': cases,
        'scope': [
            'The webpage live.tok_s and ANSWER sum each active request rate; they are not one-request rates.',
            'The 2s cutoff is also capped at32 per-token observations, about0.62s at50 tokens/s.',
            'The sampled-window denominator stops at the newest token, not the observation time.',
            'When no eligible recent samples remain, the code returns the whole-request mean rather than zero.',
            'Rejected MTP drafts and cached prompt tokens are not added to these counters.',
            'Server-inserted wrap/tool-opening tokens also enter the API generated count.',
            'The newer Terminal ALL REQUESTS counter-difference metric uses a common time interval; this reproduction does not apply to it.',
            'This proves that300–400 displayed tokens/s can coexist with lower or zero current output. It does not establish what happened at21:44 without corresponding production observations.',
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': True, 'out': str(a.out.resolve()),
                      'cases': [{k: c[k] for k in ('name', 'branch', 'kept_samples_span_s',
                         'eight_identical_requests_displayed_sum_tps',
                         'eight_identical_requests_actual_common_window_tps')} for c in cases]}, indent=2))


if __name__ == '__main__':
    main()
