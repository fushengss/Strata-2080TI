#!/usr/bin/env python3
"""Offline reviewed-previous-binary vs new-P2-off correctness only; never compare speed."""
import argparse
import hashlib
import json
from pathlib import Path


FIXED = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
         'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_PW_WITH_BATCH': '1',
         'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1'}


def read(path):
    return json.loads(path.read_text())


def compare_records(left, right, expected, prefill=False):
    maps = [{r['name']: r for r in records} for records in (left, right)]
    pairs = []
    for name in sorted(set(maps[0]) | set(maps[1])):
        a, b = (mapping.get(name) for mapping in maps)
        checks = {'present_in_both': bool(a and b)}
        usage = None
        if a and b:
            keys = ('request_sha256', 'output_sha256', 'finish_reason')
            keys += ('channel_output_sha256', 'messages_sha256') if prefill else ('reasoning_sha256',)
            checks.update({k + '_equal': a.get(k) is not None and a[k] == b.get(k) for k in keys})
            counts = [{k: r['usage'].get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                      for r in (a, b)]
            checks['logical_usage_equal'] = counts[0] == counts[1] and all(
                type(v) is int for counts_row in counts for v in counts_row.values())
            checks['logical_usage_consistent'] = all(r['prompt_tokens'] + r['completion_tokens'] == r['total_tokens']
                                                    for r in counts if all(type(v) is int for v in r.values()))
            checks['both_completed'] = all(r.get('saw_done') and not r.get('error') for r in (a, b))
            if prefill:
                checks['both_qualified'] = a.get('valid') is True and b.get('valid') is True
            usage = {'reference': counts[0], 'candidate': counts[1], 'metadata_equal': a['usage'] == b['usage']}
        pairs.append({'name': name, 'checks': checks, 'passed': all(checks.values()), 'usage': usage})
    return {'pairs': pairs, 'all_exact': len(pairs) == expected and all(r['passed'] for r in pairs)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True, help='Previous reviewed binary arm with matching fixed P1 setting')
    p.add_argument('--candidate', type=Path, required=True, help='New candidate P2 off arm directory')
    p.add_argument('--reference-engine-sha256', required=True)
    p.add_argument('--suites', default='c8', help='c8,prefill subset; default c8')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    suites = a.suites.split(',')
    if not suites or set(suites) - {'c8', 'prefill'} or len(set(suites)) != len(suites):
        p.error('--suites accepts a unique c8,prefill subset')
    if a.out.exists():
        p.error('--out must not exist')
    roots = [a.reference, a.candidate]
    metas = [read(root / 'metadata.json') for root in roots]
    configs = [meta['config'] for meta in metas]
    gates = {'reference_engine_identity': len(a.reference_engine_sha256) == 64 and metas[0]['engine_sha256'] == a.reference_engine_sha256,
             'candidate_p2_off': metas[1]['arm'] == 'off' and configs[1]['env'].get('STRATA_PIPED_ASYNC_SAMPLE') == '0',
             'p1_target_head_fixed_equal': configs[0]['env'].get('STRATA_PIPED_TARGET_HEAD', '0') == configs[1]['env'].get('STRATA_PIPED_TARGET_HEAD'),
             'reference_async_absent_or_off': configs[0]['env'].get('STRATA_PIPED_ASYNC_SAMPLE') in (None, '0'),
             'fixed_existing_paths': all(cfg['env'].get(key) == value for cfg in configs for key, value in FIXED.items()),
             'selected_suites_qualified': all(meta['suite_returncodes'].get(suite) == 0 for meta in metas for suite in suites)}
    for key in ('gpu', 'parallel', 'layer_split', 'engine_args_sha256', 'reasoning_budget_engine_cap'):
        gates[key + '_equal'] = configs[0].get(key) == configs[1].get(key)
    result = {'scope': 'Cross-version behavioral compatibility only; no throughput, latency or speedup comparison.',
              'reference_engine_sha256': metas[0]['engine_sha256'], 'candidate_engine_sha256': metas[1]['engine_sha256'],
              'gates': gates, 'suites': {}, 'all_checks_passed': False, 'source_sha256': {}}
    inputs = [root / 'metadata.json' for root in roots]
    if all(gates.values()):
        for suite in suites:
            if suite == 'c8':
                paths = [root / suite / 'c8-check.json' for root in roots]
                reports = [read(path) for path in paths]
                compared = compare_records(*[r['records'] for r in reports], expected=8)
                compared['both_runs_qualified'] = all(r.get('all_checks_passed') for r in reports)
            else:
                paths = [root / suite / 'results.json' for root in roots]
                compared = compare_records(*[read(path) for path in paths], expected=6, prefill=True)
                compared['both_runs_qualified'] = all(read(root / suite / 'manifest.json').get('completed') for root in roots)
                inputs += [root / suite / 'manifest.json' for root in roots]
            inputs += paths
            compared['passed'] = compared['all_exact'] and compared['both_runs_qualified']
            result['suites'][suite] = compared
    result['all_checks_passed'] = all(gates.values()) and len(result['suites']) == len(suites) and all(
        value['passed'] for value in result['suites'].values())
    result['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    a.out.mkdir(parents=True)
    (a.out / 'cross-compatibility.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': result['all_checks_passed'], 'gates': gates,
                      'suites': {name: row['passed'] for name, row in result['suites'].items()}, 'performance_compared': False}))
    return 0 if result['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
