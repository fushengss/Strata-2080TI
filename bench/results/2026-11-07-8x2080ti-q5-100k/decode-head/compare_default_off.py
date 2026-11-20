#!/usr/bin/env python3
"""Offline original-engine versus final flag-off compatibility check; stdlib only.

Reads saved artifacts only. It never contacts a server, starts a process or changes
an engine. Pass a run root or an individual arm directory to either input option.
"""
import argparse
import copy
import hashlib
import json
import re
import shlex
import statistics
from pathlib import Path

ORIGINAL_SHA = '12ceada2901a3f59d72f21aca4f29617d17a74efce8acff7ecce1e310262c135'
FINAL_SHA = '372da958cd4b1ba25b2ab42a7e92480c4cb31973994ebc8d8355e5644766fc22'
ORIGINAL_CASES = {'short-greedy', 'sample-17', 'sample-29', 'penalties', 'bias', 'thinking',
                  'cold-code-1', 'warm-code-1', 'append-code-1', 'cold-prose-1', 'warm-prose-1'}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def changes(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            loc = path + '/' + key
            if key not in a or key not in b:
                result.append({'path': loc, 'original': a.get(key), 'final': b.get(key),
                               'missing_in': 'original' if key not in a else 'final'})
            else:
                result.extend(changes(a[key], b[key], loc))
        return result
    return [] if a == b else [{'path': path or '/', 'original': a, 'final': b}]


def normalized_config(config):
    result = copy.deepcopy(config)
    for key in ('exe', 'log', 'api_key'):
        result.pop(key, None)
    return result


def first_difference(a, b):
    i = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
    return {'character_offset': i, 'line': a[:i].count('\n') + 1,
            'column': i - a.rfind('\n', 0, i), 'original_length': len(a), 'final_length': len(b),
            'original_excerpt': a[max(0, i-60):i+100], 'final_excerpt': b[max(0, i-60):i+100]}


def distribution(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return ({'n': len(values), 'median': statistics.median(values), 'min': min(values),
             'max': max(values), 'values': values} if values else {'n': 0})


def metric(record, field):
    if field.startswith('engine.'):
        return (record.get('matched_engine_record') or {}).get(field.split('.')[1])
    return record.get(field)


def load_arm(path):
    meta = read(path / 'metadata.json')
    records = read(path / 'results.json')
    if not isinstance(meta, dict) or not isinstance(records, list):
        raise ValueError('Missing metadata/results: ' + str(path))
    if meta.get('arm') != 'off':
        raise ValueError('Only off arms may be compared: ' + str(path))
    manifest = read(path.parent / 'manifest.json') or {}
    text = (path / 'engine.log').read_text(errors='replace') if (path / 'engine.log').exists() else ''
    starts = re.findall(r'engine started: (.+)', text)
    command = shlex.split(starts[0]) if len(starts) == 1 else []
    selected = [r for r in records if r.get('category') != 'warmup']
    names = [r['name'] for r in selected]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate request names: ' + str(path))
    return {'path': str(path.resolve()), 'tag': path.name, 'meta': meta, 'manifest': manifest,
            'complete': read(path / 'complete.json'), 'record_count': len(records),
            'records': {r['name']: r for r in selected},
            'startup_command': command, 'startup_command_count': len(starts)}


def expand(paths):
    result = []
    for value in paths:
        p = Path(value)
        candidates = [p] if (p / 'metadata.json').exists() else sorted(p.glob('[0-9][0-9]-off'))
        result.extend(load_arm(d) for d in candidates)
    unique = {a['path']: a for a in result}
    return list(unique.values())


def record_integrity(r):
    text, reasoning = r.get('output_text'), r.get('reasoning_text')
    if not isinstance(text, str) or not isinstance(reasoning, str):
        return False
    return (r.get('output_sha256') == digest(text) and r.get('reasoning_sha256') == digest(reasoning)
            and r.get('combined_output_sha256') == digest(json.dumps([text, reasoning], ensure_ascii=False)))


def request_checks(r):
    ck = r.get('checks') or {}
    failures = []
    for name in ('request_ok', 'isolated_activity', 'decode_head_route_confirmed'):
        if ck.get(name) is not True:
            failures.append(name)
    if r.get('error') or ck.get('cuda_error_lines'):
        failures.append('request_or_cuda_error')
    if r.get('decode_head_offload_windows', 0) != 0:
        failures.append('unexpected_head_offload')
    for required, passed in (('cold_required', 'cold_zero_reuse'), ('warm_required', 'warm_reused_prefix'),
                             ('append_required', 'append_reused_prefix')):
        if ck.get(required) and ck.get(passed) is not True:
            failures.append(passed)
    perf = r.get('category', '').startswith(('cold-', 'warm-')) or r.get('category') == 'append'
    if perf and ck.get('engine_full_request_record_unique') is not True:
        failures.append('engine_full_request_record_unique')
    return failures


def compare_pair(original, final):
    om, fm = original['meta'], final['meta']
    oc, fc = om.get('config') or {}, fm.get('config') or {}
    config_diff = changes(normalized_config(oc), normalized_config(fc))
    manifest_fields = ('base_config_sha256', 'server_sha256', 'harness_sha256', 'flag', 'off_value', 'output_tokens')
    manifest_checks = {k: bool(original['manifest'].get(k)) and
                       original['manifest'].get(k) == final['manifest'].get(k) for k in manifest_fields}
    oready, fready = om.get('ready_status') or {}, fm.get('ready_status') or {}
    flags_disabled = all((m.get('config') or {}).get('env', {}).get('STRATA_DECODE_HEAD_GPU') == '-1'
                         for m in (om, fm))
    gates = {'both_complete': bool(original['complete'] and final['complete']),
             'record_counts_match_complete': all(bool(x['complete']) and x['complete'].get('request_count') == x['record_count']
                                                 for x in (original, final)),
             'same_config_except_executable_and_artifact_log': bool(oc and fc) and not config_diff,
             'same_startup_argv_except_executable': bool(original['startup_command'] and final['startup_command']) and
                 original['startup_command'][1:] == final['startup_command'][1:],
             'same_engine_version': bool(oready.get('engine')) and oready.get('engine') == fready.get('engine'),
             'same_concurrency': bool(oready.get('concurrency')) and oready.get('concurrency') == fready.get('concurrency'),
             'both_flags_disabled': flags_disabled,
             'same_governors': bool(om.get('governors_before')) and
                 om.get('governors_before') == om.get('governors_after') == fm.get('governors_before') == fm.get('governors_after'),
             'no_complete_cuda_errors': bool(original['complete'] and final['complete']) and
                 not original['complete'].get('cuda_error_lines') and not final['complete'].get('cuda_error_lines')}
    pairs = []
    for name, o in original['records'].items():
        f = final['records'].get(name)
        if f is None:
            pairs.append({'name': name, 'exact': False, 'state': 'missing_final_request'})
            continue
        same_input = all(bool(o.get(k)) and o.get(k) == f.get(k) for k in ('request_sha256', 'messages_sha256'))
        same_input = same_input and o.get('request_settings') == f.get('request_settings')
        same_output = all(o.get(k) == f.get(k) for k in ('output_text', 'reasoning_text', 'output_sha256',
                          'reasoning_sha256', 'combined_output_sha256', 'finish_reason', 'max_tokens', 'category'))
        same_tokens = all(o.get('usage', {}).get(k) == f.get('usage', {}).get(k)
                          and o.get('usage', {}).get(k) is not None for k in ('prompt_tokens', 'completion_tokens'))
        failures = {'original': request_checks(o), 'final': request_checks(f)}
        integrity = record_integrity(o) and record_integrity(f)
        exact = same_input and same_output and same_tokens and integrity and not any(failures.values())
        item = {'name': name, 'category': o.get('category'), 'exact': exact, 'same_input': same_input,
                'same_output_and_finish': same_output, 'same_token_counts': same_tokens,
                'stored_output_hashes_verified': integrity, 'check_failures': failures,
                'hashes': {key: {'original': o.get(key), 'final': f.get(key)} for key in
                           ('request_sha256', 'messages_sha256', 'output_sha256', 'reasoning_sha256', 'combined_output_sha256')},
                'checks': {'original': o.get('checks'), 'final': f.get('checks')},
                'usage': {'original': o.get('usage'), 'final': f.get('usage')}}
        for field in ('output_text', 'reasoning_text'):
            if o.get(field) != f.get(field):
                item['first_difference'] = {'field': field, **first_difference(o.get(field, ''), f.get(field, ''))}
                break
        pairs.append(item)
    return {'original': original['path'], 'final': final['path'], 'gates': gates,
            'manifest_checks': manifest_checks, 'config_differences': config_diff,
            'ignored_config_paths': {'exe': [oc.get('exe'), fc.get('exe')], 'artifact_log': [oc.get('log'), fc.get('log')]},
            'startup': {'original': original['startup_command'], 'final': final['startup_command']},
            'requests': pairs, 'final_extra_names_not_covered': sorted(final['records'].keys() - original['records'].keys()),
            'all_requests_exact': bool(pairs) and all(p.get('exact') for p in pairs),
            'all_gates_pass': all(gates.values()) and all(manifest_checks.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', required=True)
    p.add_argument('--final', action='append', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--expected-original-sha', default=ORIGINAL_SHA)
    p.add_argument('--expected-final-sha', default=FINAL_SHA)
    p.add_argument('--require-final-off-arms', type=int, default=2)
    a = p.parse_args()
    originals, finals = expand([a.original]), expand(a.final)
    if len(originals) != 1:
        p.error('--original must contain exactly one off arm')
    original = originals[0]
    pairs = [compare_pair(original, arm) for arm in finals]
    binary_checks = {'original_expected_sha': original['meta'].get('engine_sha256') == a.expected_original_sha,
                     'final_expected_sha': bool(finals) and all(f['meta'].get('engine_sha256') == a.expected_final_sha for f in finals),
                     'different_original_and_final': bool(finals) and all(original['meta'].get('engine_sha256') != f['meta'].get('engine_sha256') for f in finals)}
    coverage = {'all_expected_requests_present': ORIGINAL_CASES.issubset(original['records']),
                'original_repeats_one': original['manifest'].get('repeats') == 1,
                'original_suites_correctness_and_performance': {'correctness', 'performance'}.issubset(
                    set(original['manifest'].get('suites', '').split(',')))}
    complete = (len(finals) == a.require_final_off_arms and bool(pairs) and all(coverage.values())
                and all(x['all_gates_pass'] for x in pairs))
    compatible = complete and all(binary_checks.values()) and all(x['all_requests_exact'] for x in pairs)
    performance = {}
    for name, o in original['records'].items():
        if not (o.get('category', '').startswith(('cold-', 'warm-')) or o.get('category') == 'append'):
            continue
        matched = [f['records'][name] for f in finals if name in f['records']]
        values = {}
        for field in ('engine.prefill_ms', 'engine.prefill_tps', 'engine.decode_ms', 'engine.decode_tps',
                      'ttft_s', 'decode_tps_usage_minus_one_estimate'):
            left, right = distribution([metric(o, field)]), distribution([metric(r, field) for r in matched])
            change = 100 * (right['median'] / left['median'] - 1) if left.get('median') and right.get('median') is not None else None
            values[field] = {'original': left, 'final_flag_off': right, 'final_vs_original_median_pct': change}
        performance[name] = {'comparable': compatible, 'metrics': values}
    report = {'default_off_compatible_for_tested_cases': compatible, 'complete_comparison': complete,
              'expected_final_off_arms': a.require_final_off_arms, 'found_final_off_arms': len(finals),
              'binary_checks': binary_checks, 'coverage_checks': coverage,
              'missing_original_requests': sorted(ORIGINAL_CASES - original['records'].keys()),
              'expected_original_sha': a.expected_original_sha,
              'expected_final_sha': a.expected_final_sha, 'pairs': pairs, 'performance': performance,
              'limits': ['Exact synthetic output/reasoning equality and API token counts do not prove every internal logit is equal.',
                         'Original repeats=1 covers only matching request names; final repeats 2/3 are explicitly not covered.',
                         'One original launch versus two later launches cannot establish a precise default-off speed regression rate.',
                         'Model paths and runtime arguments are compared; artifacts do not independently hash every model file.',
                         'Credentials are excluded; executable and artifact log paths are the only allowed runtime config differences.']}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'default-off-compatibility.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['Default-off compatibility: ' + ('PASS' if compatible else 'FAIL OR INCOMPLETE'),
             f'Final off arms: {len(finals)}/{a.require_final_off_arms}',
             'Expected binary hashes: ' + str(all(binary_checks.values())),
             'request\tcomparable\toriginal_decode_tps\tfinal_off_decode_tps\tchange_pct\toriginal_prefill_ms\tfinal_off_prefill_ms']
    for name, item in performance.items():
        de, pf = item['metrics']['engine.decode_tps'], item['metrics']['engine.prefill_ms']
        lines.append('\t'.join(str(x) for x in (name, item['comparable'], de['original'].get('median'),
                     de['final_flag_off'].get('median'), de['final_vs_original_median_pct'],
                     pf['original'].get('median'), pf['final_flag_off'].get('median'))))
    for pair in pairs:
        lines.append(pair['final'] + ': gates=' + str(pair['all_gates_pass']) + ', all requests exact=' + str(pair['all_requests_exact']))
        lines.extend('  ' + r['name'] + ': mismatch/missing/check failure' for r in pair['requests'] if not r.get('exact'))
    (out / 'default-off-compatibility.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if compatible else 1


if __name__ == '__main__':
    raise SystemExit(main())
