#!/usr/bin/env python3
"""Synthetic thinking-budget boundary checks; dry-run, no service mutations.

Uses the frozen correctness.py SSE/cancel client. Only this client's synthetic
requests and outputs are saved. An appended engine-log slice is reduced to
numeric prompt/cache/read rows; no raw log or API key is copied. Run the same
fixture against each independently managed cap/batch configuration, then compare.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import re
from pathlib import Path

VERSION = 'thinking-budget-correctness-1'
BUDGET_MAX_TOKENS = 1024
CORE_SHA256 = 'b5a7786792e6632f426e3c2fe664cfaf7b37fed8931de0f230960e37fa8a0865'
WRAP_MARKER = 'I have thought about this long enough; time to give my answer.'
WORDS = ('the river carried old maps past quiet towns where people traded stories about storms bridges lanterns '
         'harvests letters clocks ships and distant mountains while children counted stars above the harbor').split()
PROMPT_RE = re.compile(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms '
                       r'\(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms')
FLAG_KEYS = ('STRATA_BATCH_MTP', 'STRATA_PIPED_MTP', 'STRATA_BATCH_SELF_COMMIT',
             'STRATA_DECODE_HEAD_GPU', 'STRATA_MTP_HEAD_GPU', 'STRATA_PW_WITH_BATCH')


def core_module():
    path = Path(__file__).with_name('correctness.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA256:
        raise RuntimeError('Frozen correctness.py dependency hash differs')
    spec = importlib.util.spec_from_file_location('frozen_budget_correctness_core', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def long_fixtures(core, fixture_id):
    specs = []
    for lane, (seed, temp, budget) in enumerate(((17, .8, 64), (29, 0, 128), (101, .8, 128), (102, 0, 64))):
        # Widely spaced prompt lengths make numeric log-to-request matching unambiguous.
        nominal = 9000 + 500 * lane
        text = ' '.join(WORDS[(i * 7 + 8) % len(WORDS)] + ('.' if i % 17 == 16 else '')
                        for i in range(int(nominal / 1.3)))
        prompt = (f'{fixture_id}-lane-{lane}: independent synthetic benchmark fixture.\n' + text +
                  '\nPlan a comprehensive numbered field diary covering thirty different harbor towns. '
                  'First reason carefully through at least thirty distinct consistency checks of geography, '
                  'seasonal weather, trade, chronology, architecture and daily work. Then write all thirty '
                  'detailed diary entries, each with at least eighty words. Do not quote the source text. '
                  'Continue until all entries are complete rather than summarizing the remaining entries.')
        spec = core.fixture('budget-lane-' + str(lane), prompt, seed, BUDGET_MAX_TOKENS, temperature=temp,
                            reasoning_budget_tokens=budget, chat_template_kwargs={'enable_thinking': True})
        spec['budget'] = budget
        specs.append(spec)
    return specs


def numeric_prompt_rows(path, cursor):
    if path is None or cursor is None:
        return {'available': False, 'rows': []}
    try:
        st = path.stat()
        if st.st_ino != cursor[0] or st.st_size < cursor[1]:
            return {'available': False, 'rotated': True, 'rows': []}
        rows = []
        with path.open('rb') as f:
            f.seek(cursor[1])
            for line in f:
                m = PROMPT_RE.search(line.decode(errors='replace'))
                if m:
                    v = m.groups()
                    rows.append(dict(zip(('prompt_tokens', 'reused', 'read', 'prompt_ms', 'prefill_tps',
                                          'generated', 'decode_ms'),
                                         (int(v[0]), int(v[1]), int(v[2]), float(v[3]), float(v[4]),
                                          int(v[5]), float(v[6])))))
        return {'available': True, 'rows': rows}
    except OSError:
        return {'available': False, 'rows': []}


def continuation_evidence(records, specs, logs, wrap_tokens, slack):
    rows = []
    for record, spec in zip(records, specs):
        p = record.get('usage', {}).get('prompt_tokens')
        low = p + spec['budget'] + wrap_tokens if isinstance(p, int) else None
        candidates = [r for r in logs['rows'] if low is not None and low <= r['prompt_tokens'] <= low + slack]
        # A later solo hand-back may add another row. The smallest eligible prompt is
        # the earliest post-wrap continuation, without assuming admission order.
        minimum = min((r['prompt_tokens'] for r in candidates), default=None)
        first = [r for r in candidates if r['prompt_tokens'] == minimum]
        unique = len(first) == 1
        chosen = first[0] if unique else None
        limit = wrap_tokens + 2
        rows.append({'name': record['name'], 'input_prompt_tokens': p, 'budget_tokens': spec['budget'],
                     'wrap_marker_present': WRAP_MARKER in record.get('reasoning_text', ''),
                     'expected_wrap_token_count': wrap_tokens, 'clean_boundary_slack_tokens': slack,
                     'matching_prompt_range': [low, low + slack] if low is not None else None,
                     'matching_rows': candidates, 'earliest_match_unique': unique, 'continuation': chosen,
                     'read_limit_tokens': limit,
                     'bounded_continuation_read': bool(chosen and chosen['read'] <= limit
                                                       and chosen['reused'] >= p + spec['budget'] - 2),
                     'note': 'Numeric length association, not a request ID. Missing/ambiguous evidence fails the cap-on gate; no raw logs are saved.'})
    return rows


def budget_output_accounted(records, specs):
    named={r['name']:r for r in records}
    def valid(spec):
        record=named.get(spec['name'],{});usage=record.get('usage',{})
        return (record.get('finish_reason') in ('length','stop')
                and all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens','total_tokens'))
                and usage['prompt_tokens']>0 and 0<usage['completion_tokens']<=spec['max_tokens']
                and (record['finish_reason']!='length' or usage['completion_tokens']==spec['max_tokens'])
                and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'])
    return len(specs)==4 and all(valid(s) for s in specs)


def checks_for(core, report, specs):
    named = {r['name']: r for r in report['records']}
    full = [r for r in report['records'] if not r['intentional_cancel']]
    cancelled = named.get('cancel-stream', {})
    control, after = named.get('recovery-control', {}), named.get('recovery-after', {})
    usage = core.usage_comparison(control.get('usage', {}), after.get('usage', {}))
    report['recovery_usage_comparison'] = usage
    report['budget_output_lengths']=[{'name':s['name'],'max_tokens':s['max_tokens'],
        'actual_completion_tokens':named.get(s['name'],{}).get('usage',{}).get('completion_tokens'),
        'finish_reason':named.get(s['name'],{}).get('finish_reason')} for s in specs]
    checks = {'budget_four_actual_output_accounted':budget_output_accounted(report['records'],specs),
              'all_nine_complete_requests_succeeded': len(full) == 9 and all(core.completed(r) for r in full),
              'four_budget_requests_reached_wrap': len(report['continuations']) == 4
                  and all(r['wrap_marker_present'] for r in report['continuations']),
              'both_four_stream_cohorts_overlap': all(c['overlap']['four_streams_observed'] for c in report['cohorts'] if c['overlap']),
              'no_unrelated_requests_or_missing_metrics': all(c['metrics_request_identity_and_exclusivity_check'] for c in report['cohorts']),
              'cancelled_after_four_stream_overlap': bool(cancelled.get('closed_mid_generation')
                  and not cancelled.get('saw_done') and not cancelled.get('error')),
              'recovery_same_request': control.get('request_sha256') == after.get('request_sha256') and bool(control),
              'recovery_same_output': bool(control) and all(control.get(k) == after.get(k) for k in
                  ('output_sha256', 'reasoning_sha256', 'finish_reason')) and usage['logical_counts_equal'],
              'recovery_marker_correct': after.get('output_text', '').strip() == 'MTP_RECOVERY_731'}
    offered = sum(c['piped_drafts_offered'] for c in report['cohorts'])
    checks['piped_route_confirmed'] = offered > 0 if report['expected_piped_mtp'] == 'on' else offered == 0
    report['piped_drafts_offered'] = offered
    report['bounded_continuation_read_all_four'] = len(report['continuations']) == 4 and all(
        r['bounded_continuation_read'] for r in report['continuations'])
    # An off arm is a diagnostic control: rereading is reported but is not a run failure.
    if report['expected_cap'] == 'on':
        checks['cap_continuations_reuse_thinking_prefix'] = report['bounded_continuation_read_all_four']
    return checks


def run(args, core):
    if not args.exclusive_benchmark:
        raise RuntimeError('--execute requires --exclusive-benchmark')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise RuntimeError('--out must be new or empty')
    client = core.Client(args)
    specs = long_fixtures(core, args.fixture_id)
    flags = {k: str(client.cfg.get('env', {}).get(k, '<absent>')) for k in FLAG_KEYS}
    flags = {k: v for k, v in flags.items() if v == '<absent>' or re.fullmatch(r'[-0-9,]+', v)}
    report = {'version': VERSION, 'fixture_id': args.fixture_id, 'synthetic_only': True,
              'service_mutations': 0, 'expected_cap': args.expect_cap,
              'expected_piped_mtp': args.expect_piped_mtp, 'config_env_flags': flags,
              'config_reasoning_budget_engine_cap': client.cfg.get('reasoning_budget_engine_cap') is True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'correctness_dependency_sha256': CORE_SHA256, 'records': [], 'cohorts': [], 'continuations': [],
              'checks': {}, 'scope': 'Synthetic four-way SSE overlap. Cap selection is orchestrator supplied; /settings may override config. Runtime route needs numeric draft evidence.'}
    try:
        client.idle()
        cursor = client.log_cursor()
        report['cohorts'].append(client.cohort('budget-four', specs))
        logs = numeric_prompt_rows(client.log, cursor)
        report['budget_engine_numeric_log'] = logs
        # Thread completion order differs from fixture order: resolve by name before association.
        ordered = {r['name']: r for r in client.records}
        report['continuations'] = continuation_evidence([ordered[s['name']] for s in specs], specs,
                                                       logs, args.wrap_tokens, args.clean_slack)
        _, _, cancel, recovery = core.fixtures()
        report['cohorts'].append(client.cohort('recovery-control', [recovery]))
        report['cohorts'].append(client.cohort('cancel-four', cancel, recovery))
        report['records'] = client.records
        report['checks'] = checks_for(core, report, specs)
        report['all_checks_passed'] = all(report['checks'].values())
    except Exception as e:
        report['fatal_error_kind'] = type(e).__name__
        report['records'] = client.records
        report['all_checks_passed'] = False
    core.save(args.out / 'budget-correctness.json', report)
    print(json.dumps({'checks': report['checks'], 'all_checks_passed': report['all_checks_passed']}))
    return 0 if report['all_checks_passed'] else 2


def difference(a, b):
    n = min(len(a), len(b))
    i = next((i for i in range(n) if a[i] != b[i]), n)
    return {'equal': a == b, 'common_prefix_chars': i, 'left_chars': len(a), 'right_chars': len(b)}


def compare(args, core):
    reports = [json.loads((p / 'budget-correctness.json').read_text()) for p in args.compare]
    pairs = []
    for i, j in itertools.combinations(range(len(reports)), 2):
        left, right = reports[i], reports[j]
        a, b = ({r['name']: r for r in x['records']} for x in (left, right))
        rows = []
        for name in sorted(set(a) | set(b)):
            x, y = a.get(name), b.get(name)
            if x is None or y is None:
                rows.append({'name': name, 'passed': False, 'missing': True}); continue
            check = {'request_hash_equal': x['request_sha256'] == y['request_sha256']}
            differences = {k: difference(x[k], y[k]) for k in ('output_text', 'reasoning_text')}
            usage = None
            if x['intentional_cancel'] or y['intentional_cancel']:
                check['both_intentionally_cancelled'] = x['closed_mid_generation'] and y['closed_mid_generation']
                check['nonempty_prefix'] = min(len(x['output_text']) + len(x['reasoning_text']),
                                               len(y['output_text']) + len(y['reasoning_text'])) > 0
                check['cancelled_shorter_prefix_equal'] = all(d['common_prefix_chars'] == min(d['left_chars'], d['right_chars']) for d in differences.values())
            else:
                usage = core.usage_comparison(x['usage'], y['usage'])
                check.update({k + '_equal': x[k] == y[k] for k in ('output_sha256', 'reasoning_sha256', 'finish_reason')})
                check['logical_usage_equal'] = usage['logical_counts_equal']
                check['both_complete'] = core.completed(x) and core.completed(y)
            rows.append({'name': name, 'checks': check, 'passed': all(check.values()),
                         'differences': differences, 'usage_comparison': usage})
        pairs.append({'left': str(args.compare[i]), 'right': str(args.compare[j]),
                      'left_flags': left.get('config_env_flags'), 'right_flags': right.get('config_env_flags'),
                      'left_expected_cap': left['expected_cap'], 'right_expected_cap': right['expected_cap'],
                      'both_runs_passed': left['all_checks_passed'] and right['all_checks_passed'],
                      'same_version_fixture': left['version'] == right['version'] == VERSION and left['fixture_id'] == right['fixture_id'],
                      'requests': rows, 'all_requests_exact': bool(rows) and all(r['passed'] for r in rows),
                      'left_continuations': left['continuations'], 'right_continuations': right['continuations']})
    result = {'version': VERSION, 'comparisons': pairs,
              'note': 'All pairwise combinations are descriptive. Cap-off can expose the original continuation divergence. Compare cap-on batch-off versus cap-on batch-on for the corrected-path identity gate; do not relabel a mismatch as numerical noise.'}
    result['all_checks_passed'] = bool(pairs) and all(p['all_requests_exact'] and p['both_runs_passed'] and p['same_version_fixture'] for p in pairs)
    args.out.mkdir(parents=True, exist_ok=True)
    core.save(args.out / 'budget-comparison.json', result)
    print(json.dumps({'all_checks_passed': result['all_checks_passed'],
                      'pairs': [{'left': p['left'], 'right': p['right'], 'exact': p['all_requests_exact']} for p in pairs]}))
    return 0 if result['all_checks_passed'] else 2


def self_test(core):
    import tempfile
    specs = long_fixtures(core, 'offline-fixture')
    assert [s['budget'] for s in specs] == [64, 128, 128, 64]
    assert [s['max_tokens'] for s in specs] == [1024]*4
    complete=[{'name':s['name'],'finish_reason':'length','usage':{'prompt_tokens':8000,'completion_tokens':1024,'total_tokens':9024}} for s in specs]
    assert budget_output_accounted(complete,specs)
    eos={**complete[0],'finish_reason':'stop','usage':{'prompt_tokens':8000,'completion_tokens':900,'total_tokens':8900}}
    assert budget_output_accounted([eos]+complete[1:],specs)
    assert not budget_output_accounted([{**eos,'finish_reason':'length'}]+complete[1:],specs)
    assert not budget_output_accounted(complete[:3],specs)
    for usage in ({'prompt_tokens':8000,'completion_tokens':0,'total_tokens':8000},
                  {'prompt_tokens':8000,'completion_tokens':1025,'total_tokens':9025},
                  {'prompt_tokens':8000,'completion_tokens':900,'total_tokens':9999}):
        assert not budget_output_accounted([{**eos,'usage':usage}]+complete[1:],specs)
    assert not budget_output_accounted([{**eos,'finish_reason':'tool_calls'}]+complete[1:],specs)
    # Four real SSE streams must share >=0.1s and at least2deltas per lane.
    streams=[{'name':s['name'],'first_output_epoch':i*6,'last_output_epoch':48+i,
              'deltas':[{'epoch':25},{'epoch':30}]} for i,s in enumerate(specs)]
    assert core.overlap(streams)['four_streams_observed']
    assert not core.overlap([{**streams[0],'last_output_epoch':17.6}]+streams[1:])['four_streams_observed']
    assert not core.overlap([{**streams[0],'deltas':[{'epoch':25}]}]+streams[1:])['four_streams_observed']
    assert [s['overrides']['temperature'] for s in specs] == [.8, 0, .8, 0]
    records = [{'name': s['name'], 'usage': {'prompt_tokens': 8000 + 400 * i},
                'reasoning_text': WRAP_MARKER} for i, s in enumerate(specs)]
    rows = [{'prompt_tokens': r['usage']['prompt_tokens'] + s['budget'] + 18,
             'reused': r['usage']['prompt_tokens'] + s['budget'] - 1, 'read': 19}
            for r, s in zip(records, specs)]
    good = continuation_evidence(records, specs, {'rows': rows}, 18, 16)
    assert all(r['bounded_continuation_read'] and r['earliest_match_unique'] for r in good)
    bad = [dict(r) for r in rows]
    bad[0].update(reused=8000, read=82)
    assert not continuation_evidence(records, specs, {'rows': bad}, 18, 16)[0]['bounded_continuation_read']
    assert not continuation_evidence(records, specs, {'rows': rows + [rows[0]]}, 18, 16)[0]['earliest_match_unique']
    assert not continuation_evidence(records, specs, {'rows': []}, 18, 16)[0]['bounded_continuation_read']
    assert difference('same output', 'same output')['equal']
    assert difference('prefix-a', 'prefix-b')['common_prefix_chars'] == 7
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'engine.log'
        path.write_text('PRIVATE_LOG_SENTINEL\n')
        st = path.stat(); cursor = st.st_ino, st.st_size
        with path.open('a') as f:
            f.write('unrelated input PRIVATE_LOG_SENTINEL\n')
            f.write('strata serve: prompt 8082 tokens = 8063 reused + 19 read in 10 ms (1900.0 tok/s), 1 generated in 3 ms\n')
        numeric = numeric_prompt_rows(path, cursor)
        assert numeric['available'] and len(numeric['rows']) == 1
        assert numeric['rows'][0]['read'] == 19 and 'PRIVATE' not in json.dumps(numeric)
    print(json.dumps({'offline_self_test': 'PASS', 'network_requests': 0, 'gpu_operations': 0}))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path); p.add_argument('--port', type=int, default=8092)
    p.add_argument('--out', type=Path); p.add_argument('--engine-log', type=Path)
    p.add_argument('--execute', action='store_true'); p.add_argument('--exclusive-benchmark', action='store_true')
    p.add_argument('--expect-cap', choices=('on', 'off'), default='on')
    p.add_argument('--expect-piped-mtp', choices=('on', 'off'), default='on')
    p.add_argument('--fixture-id', default='budget-boundary-20261107')
    p.add_argument('--wrap-tokens', type=int, default=18, help='Known current Qwen wrap-up token length; numeric matcher assumption')
    p.add_argument('--clean-slack', type=int, default=16)
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--compare', type=Path, nargs='+')
    args = p.parse_args()
    if args.self_test:
        return self_test(core_module())
    if args.compare:
        if len(args.compare) < 2 or args.out is None:
            p.error('--compare needs at least two directories and --out')
        return compare(args, core_module())
    if not args.execute:
        print(json.dumps({'version': VERSION, 'dry_run': True, 'requests': 10,
                          'four_long_context_requests': {'nominal_tokens': [9000, 9500, 10000, 10500],
                              'seeds': [17, 29, 101, 102], 'temperature': [.8, 0, .8, 0], 'budget': [64, 128, 128, 64], 'max_tokens': BUDGET_MAX_TOKENS},
                          'then': ['recovery control', 'cancel plus three peers', 'recovery after cancel'],
                          'guards': '--execute --exclusive-benchmark; existing idle loopback server only',
                          'continuation_read_limit': args.wrap_tokens + 2}, indent=2))
        return 0
    if not args.exclusive_benchmark or args.config is None or args.out is None:
        p.error('--execute requires --exclusive-benchmark --config --out')
    if not 1 <= args.port <= 65535 or args.wrap_tokens < 1 or args.clean_slack < 0 or args.timeout < 1:
        p.error('invalid numeric option')
    return run(args, core_module())


if __name__ == '__main__':
    raise SystemExit(main())
