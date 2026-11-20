#!/usr/bin/env python3
"""Independent local P2 audit. No network, inference, service or GPU operations.

Rerun as complete arm files arrive. Missing/unfinalized evidence stays PENDING;
the analysis never writes into raw run directories or imports their programs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

B = Path(__file__).resolve().parent
SUITES = ('prefill', 'c8', 'correctness', 'c4mixed', 'budget', 'long')
EXPECTED = {
    'same_binary_sha256': 'ff2cc241def6a4636fea6ce20d4b7ba5ca7554bf4025304e5ce903af4079e778',
    'adjacent_library_sha256': 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661',
    'server_sha256': '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9',
    'base_config_sha256': 'f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220',
}
FIXED = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
         'STRATA_PW_WITH_BATCH': '1', 'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1',
         'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1', 'STRATA_PIPED_TARGET_HEAD': '1'}
FLAG = 'STRATA_PIPED_ASYNC_SAMPLE'
LOGICAL = ('prompt_tokens', 'completion_tokens', 'total_tokens')
SOURCE_HASHES = {}
HEX = re.compile(r'[0-9a-f]{64}\Z')
HISTORY = {
    'mixed-greedy': {'temperature': 0, 'penalty_last_n': 128, 'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repetition_penalty': 1.05},
    'mixed-sampled': {'temperature': 1.0, 'penalty_last_n': 32, 'presence_penalty': .7, 'frequency_penalty': .1, 'repetition_penalty': 1.0},
    'mixed-penalties': {'temperature': 1.0, 'penalty_last_n': 257, 'presence_penalty': 1.1, 'frequency_penalty': .3, 'repetition_penalty': 1.05},
}
CORRECT_NAMES = {*(f'seed-{n}' for n in (17, 29, 101, 102)), *HISTORY, 'mixed-bias',
                 'cancel-stream', *(f'cancel-peer-{n}' for n in (1, 2, 3)), 'recovery-control', 'recovery-after'}
BUDGET_NAMES = {*(f'budget-lane-{n}' for n in range(4)), 'cancel-stream',
                *(f'cancel-peer-{n}' for n in (1, 2, 3)), 'recovery-control', 'recovery-after'}
LIMITS = [
    'Pending files, incomplete metadata or missing cleanup/comparison evidence are not a successful test.',
    'One off/on pair is descriptive; no confidence interval, stable SLO, automatic deployment or promised speedup.',
    'History coverage proves actual sent 32/128/257 parameters and suite-level asynchronous sampling; there is no independent per-history-row GPU counter.',
    'Cancellation compares nonempty common prefixes, not equal truncation lengths or cancelled usage. Recovery proves post-cancel correctness, not reuse of a specific physical slot.',
    'C8 and long API rates use completed logical tokens divided by earliest request start to latest end. Long includes cold prefill/admission and drain.',
    'Native piped rows are emitted output rows, but native busy-block boundaries differ from API boundaries. Target verification rows and async sampled rows are work, not accepted tokens.',
    'SSE gaps are chunk-arrival gaps, not per-token TBT. Mixed phase byte rates/fairness and usage-weighted token estimates are progress proxies.',
    'Mixed resident refills/cancellations may differ by scheduling. Unmatched records are disclosed and never treated as exact output equality.',
    'Prefill/mixed/long raw text is intentionally absent; channel-digest composition can be verified, but original text bytes cannot be independently rehashed.',
    'Prefill has only three cold and three warm samples per arm. A small median increase requires review/repetition, not an automatic assertion of regression or no regression.',
]


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    SOURCE_HASHES[str(path.resolve())] = value
    return value


def no_duplicates(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate_json_key')
        out[key] = value
    return out


def read(path):
    raw = path.read_bytes()
    SOURCE_HASHES[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw, object_pairs_hook=no_duplicates,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def digest(value, spaced=False):
    data = value.encode() if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=None if spaced else (',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def usage(record):
    result = {key: record.get('usage', {}).get(key) for key in LOGICAL}
    require(all(type(v) is int and v >= 0 for v in result.values()), 'invalid_logical_usage')
    require(result['completion_tokens'] > 0 and result['total_tokens'] == result['prompt_tokens'] + result['completion_tokens'], 'inconsistent_logical_usage')
    return result


def named(records):
    require(isinstance(records, list) and records, 'empty_records')
    rows = {}
    for row in records:
        name = row.get('name')
        require(isinstance(name, str) and name and name not in rows, 'duplicate_or_missing_record_name')
        rows[name] = row
    return rows


def completed(record):
    return (record.get('saw_done') is True and record.get('error') is None
            and not record.get('intentional_cancel') and not record.get('closed_mid_generation')
            and record.get('finish_reason') in ('length', 'stop'))


def audit_record(record, suite):
    for key in ('request_sha256', 'output_sha256'):
        require(isinstance(record.get(key), str) and HEX.fullmatch(record[key]), 'missing_or_invalid_' + key)
    checks = {}
    if suite in ('c8', 'correctness', 'budget'):
        require(isinstance(record.get('synthetic_request'), dict) and isinstance(record.get('output_text'), str)
                and isinstance(record.get('reasoning_text'), str) and isinstance(record.get('deltas'), list), 'missing_synthetic_text_fields')
        require(bool(record['output_text'] or record['reasoning_text']), 'empty_output_text')
    for content, hashed in (('synthetic_request', 'request_sha256'), ('output_text', 'output_sha256'), ('reasoning_text', 'reasoning_sha256')):
        if content in record:
            checks[content + '_rehashed'] = digest(record[content]) == record.get(hashed)
    if 'output_text' in record:
        checks['sse_content_reassembled'] = ''.join(d['content'] for d in record['deltas']) == record['output_text']
        checks['sse_reasoning_reassembled'] = ''.join(d['reasoning'] for d in record['deltas']) == record['reasoning_text']
    for key in ('channel_output_sha256', 'channel_sha256'):
        if key in record:
            channels = record[key]
            expected_channels = {'content', 'reasoning', 'tool'} if suite == 'long' else {'visible_content', 'reasoning', 'tool'}
            require(isinstance(channels, dict) and set(channels) == expected_channels and all(isinstance(v, str) and HEX.fullmatch(v) for v in channels.values()), 'invalid_channel_digests')
            checks['combined_channel_digest_rehashed'] = digest(channels, spaced=suite == 'c4mixed') == record['output_sha256']
    require(checks, 'no_hash_integrity_evidence')
    cancel = record.get('intentional_cancel') is True
    if cancel:
        checks['valid_cancelled_stream'] = (record.get('closed_mid_generation') is True and record.get('saw_done') is False
            and record.get('error') is None and record.get('finish_reason') is None
            and len(record.get('deltas', [])) >= 16 and bool(record.get('output_text') or record.get('reasoning_text')))
        logical = None
    elif suite == 'c4mixed' and (record.get('error') or {}).get('kind') == 'bounded_drain_cancel':
        checks['declared_bounded_drain'] = record.get('saw_done') is False and record.get('finish_reason') is None
        logical = None
    else:
        checks['normal_complete'] = completed(record)
        logical = usage(record)
        maximum = (record.get('synthetic_request') or {}).get('max_tokens', record.get('max_tokens'))
        require(type(maximum) is int and maximum > 0, 'missing_requested_limit')
        checks['logical_completion_within_limit'] = logical['completion_tokens'] <= maximum
        if record.get('finish_reason') == 'length':
            checks['length_equals_requested_limit'] = logical['completion_tokens'] == maximum
    require(all(checks.values()), 'record_integrity_or_completion_failed')
    return {'name': record['name'], 'checks': checks, 'logical_usage': logical,
            'request_sha256': record['request_sha256'], 'output_sha256': record['output_sha256'],
            'finish_reason': record.get('finish_reason'), 'cancelled': cancel,
            'text_rehashed': 'output_text' in record}


def compare_records(left, right, suite):
    a, b = named(left), named(right)
    rows = []
    for name in sorted(a.keys() & b.keys()):
        x, y = a[name], b[name]
        checks = {'same_request': x['request_sha256'] == y['request_sha256']}
        cancel = x.get('intentional_cancel') or y.get('intentional_cancel')
        if cancel:
            checks['both_intentionally_cancelled'] = x.get('intentional_cancel') is True and y.get('intentional_cancel') is True
            for key in ('output_text', 'reasoning_text'):
                n = min(len(x[key]), len(y[key])); checks[key + '_common_prefix'] = x[key][:n] == y[key][:n]
            checks['common_output_nonempty'] = min(len(r['output_text']) + len(r['reasoning_text']) for r in (x, y)) > 0
            logical = None
        elif completed(x) and completed(y):
            for key in ('output_sha256', 'reasoning_sha256', 'channel_output_sha256', 'channel_sha256', 'messages_sha256', 'finish_reason'):
                if key in x or key in y: checks[key + '_equal'] = key in x and key in y and x[key] == y[key]
            checks['logical_usage_equal'] = usage(x) == usage(y)
            logical = [usage(x), usage(y)]
        else:
            rows.append({'name': name, 'status': 'NOT_EXACTLY_COMPARABLE_DRAIN', 'passed': None})
            continue
        rows.append({'name': name, 'status': 'COMPARED', 'checks': checks, 'passed': all(checks.values()),
                     'logical_usage_off_on': logical, 'full_usage_metadata_equal': x.get('usage') == y.get('usage')})
    missing = {'off_only': sorted(a.keys() - b.keys()), 'on_only': sorted(b.keys() - a.keys())}
    exact_set = not any(missing.values()) and all(r['passed'] is not None for r in rows)
    return {'passed': bool(rows) and exact_set and all(r['passed'] for r in rows),
            'compared_cases_passed': bool(rows) and all(r['passed'] is not False for r in rows),
            'same_complete_case_set': exact_set, 'unmatched': missing, 'cases': rows}


def stats(values):
    require(values and all(finite(v) for v in values), 'invalid_statistical_values')
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values), 'values': values}


def api_rate(records):
    require(records and all(completed(r) for r in records), 'incomplete_api_rate_records')
    start = min(r['start_epoch'] for r in records); end = max(r['end_epoch'] for r in records)
    require(finite(start) and finite(end) and end > start, 'invalid_api_wall')
    tokens = sum(usage(r)['completion_tokens'] for r in records)
    return {'completed_tokens': tokens, 'wall_s': end - start, 'tokens_per_s': tokens / (end - start)}


def native(rows):
    if not rows: return None
    require(all(r['wall_ms'] > 0 and r['rows'] >= 0 and 0 <= r['drafts_accepted'] <= r['drafts_offered'] for r in rows), 'invalid_native_rows')
    tokens = sum(r['rows'] for r in rows); wall = sum(r['wall_ms'] for r in rows)
    offered = sum(r['drafts_offered'] for r in rows); accepted = sum(r['drafts_accepted'] for r in rows)
    return {'batches': len(rows), 'emitted_output_rows': tokens, 'wall_ms': wall, 'tokens_per_s': tokens * 1000 / wall,
            'drafts_accepted': accepted, 'drafts_offered': offered, 'acceptance_fraction': accepted / offered if offered else None}


def all_checks(report, key='checks'):
    checks = report.get(key)
    return isinstance(checks, dict) and bool(checks) and all(v is True for v in checks.values())


def recovery(records):
    rows = named(records); a, b = rows['recovery-control'], rows['recovery-after']
    checks = {'same_request': a['request_sha256'] == b['request_sha256'],
              'same_output_reasoning_finish': all(a[k] == b[k] for k in ('output_sha256', 'reasoning_sha256', 'finish_reason')),
              'same_logical_usage': usage(a) == usage(b), 'marker_exact': b['output_text'].strip() == 'MTP_RECOVERY_731',
              'both_completed': completed(a) and completed(b)}
    require(all(checks.values()), 'recovery_check_failed')
    return {'checks': checks, 'full_usage_metadata_equal': a['usage'] == b['usage'],
            'scope': 'Post-cancel request correctness; physical slot reuse is not identified.'}


def history(report):
    rows = named(report['records']); coverage = named(report['history_request_coverage']['requests'])
    require(set(coverage) == set(HISTORY), 'history_fixture_set_changed')
    result = []
    for name, values in HISTORY.items():
        actual = rows[name]['synthetic_request']
        require(all(actual.get(k) == v and coverage[name].get(k) == v for k, v in values.items()), 'history_sent_parameters_changed')
        count = usage(rows[name])['completion_tokens']
        result.append({'name': name, 'actual_sent_parameters': values, 'completion_tokens': count,
                       'completed_output_exceeds_history_window': count >= values['penalty_last_n'],
                       'request_hash_recomputed': digest(actual) == rows[name]['request_sha256']})
    require(rows['mixed-bias']['synthetic_request'].get('logit_bias') == {'32': 1.0}, 'logit_bias_fixture_changed')
    return {'fixtures': result, 'all_three_reached_window_length': all(r['completed_output_exceeds_history_window'] for r in result),
            'scope': 'Actual request parameters and generated length; suite async counters do not identify individual history GPU copies.'}


def budget_continuations(report):
    records = named(report['records']); saved = named(report['continuations'])
    require(set(saved) == {f'budget-lane-{i}' for i in range(4)}, 'budget_continuation_set')
    require(report['expected_cap'] == 'on' and report['config_reasoning_budget_engine_cap'] is True
            and report['budget_engine_numeric_log']['available'] is True, 'budget_cap_or_log_missing')
    rows = report['budget_engine_numeric_log']['rows']; result = []
    for i, budget in enumerate((64, 128, 128, 64)):
        name = f'budget-lane-{i}'; record = records[name]; old = saved[name]
        prompt = usage(record)['prompt_tokens']; low = prompt + budget + 18
        candidates = [r for r in rows if low <= r['prompt_tokens'] <= low + 16]
        minimum = min((r['prompt_tokens'] for r in candidates), default=None)
        first = [r for r in candidates if r['prompt_tokens'] == minimum]
        require(len(first) == 1, 'budget_continuation_not_unique')
        chosen = first[0]
        require(all(type(chosen[k]) is int and chosen[k] >= 0 for k in ('prompt_tokens', 'read', 'reused'))
                and chosen['prompt_tokens'] == chosen['read'] + chosen['reused']
                and chosen['read'] <= 20 and chosen['reused'] >= prompt + budget - 2, 'budget_prefix_not_reused')
        require(old['continuation'] == chosen and old['matching_rows'] == candidates and old['matching_prompt_range'] == [low, low + 16]
                and old['budget_tokens'] == budget and old['input_prompt_tokens'] == prompt
                and old['earliest_match_unique'] is True and old['bounded_continuation_read'] is True
                and old['wrap_marker_present'] is True and 'I have thought about this long enough; time to give my answer.' in record['reasoning_text'], 'budget_saved_evidence_mismatch')
        result.append({'name': name, 'input_prompt_tokens': prompt, 'budget_tokens': budget,
                       'candidate_range': [low, low + 16], 'recomputed_earliest_unique': True, 'continuation': chosen})
    return {'records': result, 'scope': 'Numeric prompt-length association, not request-ID evidence.'}


def suite_data(root, suite, harness):
    extra = {}; report = None
    if suite == 'prefill':
        report = read(root / suite / 'summary.json'); sm = read(root / suite / 'manifest.json')
        records = [read(root / suite / f'{kind}-{n}.json') for kind in ('cold', 'warm') for n in (1, 2, 3)]
        require(report['request_count'] == 6 and report['all_valid'] is True and all(r['valid'] is True for r in records), 'prefill_suite_invalid')
        require(sm['engine_sha256'] == EXPECTED['same_binary_sha256'] and sm['library_sha256'] == EXPECTED['adjacent_library_sha256'], 'prefill_binary_identity')
        require(sm['script_sha256'] == harness['prefill_async_sample_check.py'], 'prefill_script_identity')
        extra['statistics'] = {kind: {metric: stats([r[metric] if metric == 'ttft_s' else r['engine'][metric] for r in records if r['kind'] == kind])
            for metric in ('read_ms', 'ttft_s', 'decode_tps')} for kind in ('cold', 'warm')}
        extra['actual_prompts'] = [usage(r)['prompt_tokens'] for r in records]
        require(all(95000 <= v <= 105000 for v in extra['actual_prompts']), 'prefill_not_100k')
        for r in records:
            require(all(v is True for v in r['checks'].values() if v is not None), 'prefill_record_checks')
    elif suite == 'c4mixed':
        report = read(root / suite / 'c4/summary.json'); sm = read(root / suite / 'manifest.json')
        records = [read(p) for p in sorted((root / suite / 'c4').glob('*.json')) if p.name.startswith(('resident-', 'injected-'))]
        require(records and report['resident_workers'] == 4 and report['error'] is None and report['failed_requests'] == 0
                and report['baseline_occupancy_qualified'] is True and len(report['injections']) >= 2, 'mixed_suite_invalid')
        require(sm['script_sha256'] == harness['client.py'] and sm['plan_sha256'] == harness['workload-plan.json'], 'mixed_script_identity')
        saved_rate = report['whole_scenario_completed_api_token_rate']
        total = sum(usage(r)['completion_tokens'] for r in records if completed(r))
        wall = report['scenario_end_epoch'] - report['scenario_start_epoch']
        require(wall > 0 and saved_rate['tokens'] == total and abs(saved_rate['seconds_including_start_and_drain'] - wall) < 1e-8
                and abs(saved_rate['tokens_per_s'] - total / wall) < 1e-8, 'mixed_saved_rate_mismatch')
        extra.update(whole_scenario_api=report['whole_scenario_completed_api_token_rate'], phases=report['phases'],
                     occupancy=report['occupancy'], recovery={k: v for k, v in report['recovery'].items() if k != 'bins'},
                     actual_requests=[{k: r.get(k) for k in ('name', 'role', 'actual_prompt_tokens', 'cached_tokens', 'prompt_within_nominal_tolerance', 'ttft_s', 'first_visible_content_s')} for r in records],
                     same_synthetic_plan_sha256=digest(sm['plan']), fixture_id=sm['fixture_id'])
    else:
        filename = {'c8': 'c8-check.json', 'correctness': 'correctness.json', 'budget': 'budget-correctness.json', 'long': 'long-concurrency.json'}[suite]
        report = read(root / suite / filename); records = report['records']
        checks_source = report['summary'] if suite == 'long' else report
        require(report['all_checks_passed'] is True and all_checks(checks_source), 'suite_checks_failed_or_missing')
        if suite != 'long':
            require(report['synthetic_only'] is True and report['service_mutations'] == 0, 'suite_not_synthetic')
            script = {'c8': 'c8_check.py', 'correctness': 'correctness.py', 'budget': 'budget_correctness.py'}[suite]
            require(report['source_sha256'] == harness[script], 'suite_source_identity')
            if suite != 'correctness': require(report['correctness_dependency_sha256'] == harness['correctness.py'], 'correctness_dependency_identity')
        if suite == 'c8':
            require(set(named(records)) == {f'c8-lane-{i}' for i in range(8)}, 'c8_names')
            require(report['version'] == 'piped-mtp-c8-check-1' and len(report['checks']) == 6, 'c8_version_checks')
            require(sorted(r['synthetic_request']['temperature'] for r in records) == [0] * 4 + [.8] * 4, 'c8_sampling_mix')
            require(all(usage(r)['completion_tokens'] == 1024 and r['finish_reason'] == 'length' for r in records), 'c8_length')
            extra['api'] = api_rate(records); extra['native'] = native(report['numeric_piped_log']['records'])
            require(abs(extra['api']['tokens_per_s'] - report['timing']['whole_batch']['completed_api_tokens_per_s']) < 1e-8, 'c8_saved_rate_mismatch')
            extra['ttft_s'] = stats([r['first_output_epoch'] - r['start_epoch'] for r in records])
            extra['common_output_interval_s'] = report['timing']['overlap']['common_interval_s']
            extra['max_inter_chunk_gap_s'] = max(r['longest_inter_chunk_gap_s'] for r in report['timing']['lanes'])
        elif suite in ('correctness', 'budget'):
            require(set(named(records)) == (CORRECT_NAMES if suite == 'correctness' else BUDGET_NAMES), 'correctness_record_set')
            require([r['name'] for r in records if r.get('intentional_cancel') is True] == ['cancel-stream'], 'cancel_count_or_identity')
            extra['recovery'] = recovery(records)
            expected_cohorts = {'seeded-four': 4, 'mixed-four': 4, 'recovery-control': 1, 'cancel-four': 5} if suite == 'correctness' else {'budget-four': 4, 'recovery-control': 1, 'cancel-four': 5}
            cohorts = named(report['cohorts']); require(set(cohorts) == set(expected_cohorts), 'cohort_set')
            for name, size in expected_cohorts.items():
                c = cohorts[name]; require(len(c['requests']) == size and len(set(c['requests'])) == size and c['metrics_request_identity_and_exclusivity_check'] is True, 'cohort_identity')
                if name != 'recovery-control': require(c['overlap']['four_streams_observed'] is True, 'cohort_no_overlap')
            extra['native'] = native([r for c in report['cohorts'] for r in c['piped_numeric_log']['records']])
            if suite == 'correctness':
                require(report['version'] == 'piped-async-history-correctness-1', 'history_fixture_version')
                extra['history_coverage'] = history(report)
            else:
                extra['continuations'] = budget_continuations(report)
                extra['recovery_usage'] = {'logical_counts_equal': report['recovery_usage_comparison']['logical_counts_equal'], 'metadata_equal': report['recovery_usage_comparison']['metadata_equal']}
        else:
            require(set(named(records)) == {f'lane-{i}' for i in range(4)}, 'long_record_set')
            sm = read(root / suite / 'manifest.json')
            require(sm.get('completed') is True, 'long_manifest_not_completed')
            require(sm['engine_sha256'] == EXPECTED['same_binary_sha256'] and sm['adjacent_library_sha256'] == EXPECTED['adjacent_library_sha256'], 'long_binary_identity')
            require(sm['source_sha256'] == harness['long_concurrency_check.py'] and all(sm['dependencies_sha256'][k] == harness[k] for k in ('correctness.py', 'prefill_check.py')), 'long_source_identity')
            for r in records:
                require(95000 <= usage(r)['prompt_tokens'] <= 105000 and usage(r)['completion_tokens'] == 512 and r['finish_reason'] == 'length', 'long_100k_length')
                cached = (r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                require(type(cached) is int and cached == 0, 'long_not_cold')
                require(r['output_bytes']['content'] > 0 and r['output_bytes']['reasoning'] == 0 and r['output_bytes']['tool'] == 0, 'long_not_visible_only')
            require(sorted(r['temperature'] for r in records) == [0, 0, .8, .8], 'long_sampling_mix')
            extra['api'] = api_rate(records); extra['native'] = native(report['cohort']['piped_numeric_log']['records'])
            require(abs(extra['api']['tokens_per_s'] - report['summary']['whole_batch_tps']) < 1e-8, 'long_saved_rate_mismatch')
            extra['ttft_s'] = stats([r['ttft_s'] for r in records]); extra['overlap'] = report['summary']['overlap']
    audits = [audit_record(r, suite) for r in records]
    return {'status': 'COMPLETE', 'records': audits, 'metrics': extra}, records, report


def route_evidence(meta, suite, arm):
    a = meta['async_sample_suite_evidence'][suite]; t = meta['target_head_suite_evidence'][suite]; h = meta['shared_head_suite_evidence'][suite]
    require(a['windows'] == sum(r[0] for r in a['rows']) and a['sampled_rows'] == sum(r[1] for r in a['rows']), 'async_counter_sum')
    require(all(len(r) == 2 and type(r[0]) is int and type(r[1]) is int and ((r[0] == r[1] == 0) or (r[0] > 0 and r[1] >= r[0])) for r in a['rows']), 'async_counter_shape')
    require(bool(a['rows']) == (arm == 'on' and suite != 'prefill'), 'async_route_not_observed')
    if arm == 'on' and suite != 'prefill': require(a['windows'] > 0 and a['sampled_rows'] > 0, 'async_no_completed_work')
    require(t['windows'] == sum(r[2] for r in t['rows']) and t['verification_rows'] == sum(r[3] for r in t['rows']), 'target_counter_sum')
    require(all(len(r) == 4 and r[:2] == [2, 4] and r[2] > 0 and r[3] >= r[2] for r in t['rows']), 'target_route_shape')
    require(bool(t['rows']) == (suite != 'prefill'), 'fixed_target_route')
    require(h['steps'] == sum(r[3] for r in h['rows']) and all(len(r) == 4 and 0 <= r[0] < 8 and r[1:3] == [3, 5] and type(r[3]) is int and r[3] >= 0 for r in h['rows']), 'shared_head_counter_shape')
    require(bool(h['rows']) == (suite != 'prefill'), 'shared_head_route')
    if suite != 'prefill': require(h['steps'] > 0, 'shared_head_no_completed_work')
    return {'async_windows': a['windows'], 'async_sampled_rows': a['sampled_rows'],
            'target_windows': t['windows'], 'target_verification_rows': t['verification_rows'], 'mtp_draft_steps': h['steps'],
            'scope': 'Work counters, not accepted output tokens; no per-history-row counter.'}


def raw_log_evidence(root, meta, reports):
    path = root / 'engine.log'; sha(path)
    patterns = {
        'async_sample_suite_evidence': re.compile(r'^strata piped async sample: (\d+) windows, (\d+) sampled rows completed in this batch$'),
        'target_head_suite_evidence': re.compile(r'^strata piped target head: CUDA(\d+)\+(\d+), (\d+) windows, (\d+) verification rows offloaded in this batch$'),
        'shared_head_suite_evidence': re.compile(r'^strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch$'),
    }
    batch = re.compile(r'strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, (\d+) rows in ([\d.]+) ms = ([\d.]+) rows/s .*?\((\d+) of (\d+) MTP proposals kept\)')
    fields = ('groups', 'slots_per_group', 'group_steps', 'rows', 'wall_ms', 'rows_per_s', 'drafts_accepted', 'drafts_offered')
    observed = {k: [] for k in patterns}; batches = []; startups = []
    with path.open(errors='replace') as handle:
        for line in handle:
            for key, pattern in patterns.items():
                match = pattern.fullmatch(line.rstrip('\n'))
                if match: observed[key].append([int(v) for v in match.groups()])
            match = batch.search(line)
            if match: batches.append(dict(zip(fields, map(float, match.groups()))))
            match = re.fullmatch(r'strata serve: STRATA_PIPED_ASYNC_SAMPLE: source CUDA(\d+), event-polled slot sampling with persistent history\n?', line)
            if match: startups.append(int(match[1]))
    checks = {key + '_raw_log_exact': observed[key] == [row for suite in SUITES for row in meta[key][suite]['rows']] for key in patterns}
    checks['async_startup_raw_log_exact'] = startups == meta['startup_facts']['async_sample_startup']
    require(all(checks.values()), 'raw_log_route_evidence_mismatch')
    before = reports['c8']['numeric_piped_log']['records'] + [r for c in reports['correctness']['cohorts'] for r in c['piped_numeric_log']['records']]
    after = [r for c in reports['budget']['cohorts'] for r in c['piped_numeric_log']['records']] + reports['long']['cohort']['piped_numeric_log']['records']
    signature = lambda row: tuple(row[k] for k in fields if k != 'rows_per_s')
    matched = (len(batches) > len(before) + len(after) and bool(after)
               and [signature(r) for r in batches[:len(before)]] == [signature(r) for r in before]
               and [signature(r) for r in batches[-len(after):]] == [signature(r) for r in after])
    c4 = {'available': matched, 'association': 'Sequential suite order plus exact preceding/following numeric summaries; native logs have no wall timestamps.'}
    if matched: c4['summary'] = native(batches[len(before):-len(after)])
    return {'checks': checks, 'c4_native': c4}


def audit_arm(root, arm, harness):
    result = {'status': 'PENDING', 'suites': {}}; raw = {}; reports = {}
    path = root / 'metadata.json'
    if not path.exists(): return dict(result, reason='metadata_not_downloaded'), raw, None
    meta = read(path); checks = {}
    checks['engine_library'] = meta.get('engine_sha256') == EXPECTED['same_binary_sha256'] and meta.get('adjacent_library_sha256') == EXPECTED['adjacent_library_sha256']
    checks['arm_label'] = meta.get('arm') == arm
    expected_flags = {**FIXED, FLAG: '1' if arm == 'on' else '0'}
    checks['safe_configuration_flags'] = all(meta.get('config', {}).get('env', {}).get(k) == v for k, v in expected_flags.items()) and meta.get('config', {}).get('reasoning_budget_engine_cap') is True
    checks['no_observation_timing_enabled'] = all(meta.get('config', {}).get('env', {}).get(k, '0') == '0' for k in ('STRATA_PIPED_MTP_OBSERVE', 'STRATA_MTP_PROFILE'))
    actual = meta.get('actual_engine', [])
    if actual:
        checks['actual_mapped_binary_library_flags'] = len(actual) == 1 and all(p.get('engine_sha256') == EXPECTED['same_binary_sha256'] and p.get('library_sha256') == EXPECTED['adjacent_library_sha256'] and p.get('expected_library_mapped') is True and all(p.get('flags', {}).get(k) == v for k, v in expected_flags.items()) for p in actual)
        facts = meta['startup_facts']
        checks['same_prefill_buffers_residency'] = facts['all_eight_have_3_80_gib_own_buffers'] is True and facts['all_experts_resident'] is True
        checks['startup_routes'] = facts['batch_slot_devices'] == list(range(8)) and facts['piped_mtp_startup_enabled'] is True and facts['shared_head_workspaces'] == [[8, 3, 5, 7]] and facts['target_head_startup'] == [[2, 4, 7]] and facts['async_sample_startup'] == ([7] if arm == 'on' else [])
        checks['eight_serving_slots'] = meta['ready_status']['concurrency'].get('serving') == 8 and meta['ready_status']['concurrency'].get('requested') == 8
    for suite in SUITES:
        rc = meta.get('suite_returncodes', {}).get(suite)
        if rc is None: result['suites'][suite] = {'status': 'PENDING', 'reason': 'suite_not_finalized_in_metadata'}; continue
        if rc != 0: result['suites'][suite] = {'status': 'FAILED', 'reason': 'nonzero_suite_returncode', 'returncode': rc}; continue
        try:
            s, records, report = suite_data(root, suite, harness)
            s['route'] = route_evidence(meta, suite, arm)
            result['suites'][suite] = s; raw[suite] = records; reports[suite] = report
        except FileNotFoundError:
            result['suites'][suite] = {'status': 'PENDING', 'reason': 'suite_artifacts_not_fully_downloaded'}
        except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            result['suites'][suite] = {'status': 'FAILED', 'error_kind': type(error).__name__,
                                       'reason': str(error) if isinstance(error, ValueError) and re.fullmatch(r'[a-z0-9_]+', str(error)) else 'schema_or_evidence_invalid'}
    finalized = finite(meta.get('finished_epoch')) and set(meta.get('suite_returncodes', {})) == set(SUITES)
    cleanup = root / 'credential-scan.json'
    if cleanup.exists(): checks['credential_scan_zero'] = read(cleanup).get('credential_matches') == 0
    if finalized and len(reports) == len(SUITES) and (root / 'engine.log').exists():
        try:
            log_evidence = raw_log_evidence(root, meta, reports)
            result['raw_engine_log_evidence'] = log_evidence
            result['suites']['c4mixed']['metrics']['native'] = log_evidence['c4_native']
            checks['raw_engine_log_route_matches'] = True
        except (ValueError, KeyError, TypeError, IndexError):
            checks['raw_engine_log_route_matches'] = False
    if any(v is False for v in checks.values()) or any(s['status'] == 'FAILED' for s in result['suites'].values()): result['status'] = 'FAILED'
    elif finalized and actual and cleanup.exists() and checks.get('raw_engine_log_route_matches') is True and all(s['status'] == 'COMPLETE' for s in result['suites'].values()): result['status'] = 'COMPLETE'
    result.update(checks=checks, metadata_finalized=finalized,
                  metadata={k: meta.get(k) for k in ('arm', 'start_epoch', 'ready_epoch', 'finished_epoch', 'engine_sha256', 'adjacent_library_sha256', 'config_except_variables_log_key_sha256')})
    return result, raw, meta


def change(off, on):
    if not finite(off) or not finite(on): return {'off': off, 'on': on, 'on_relative_change_pct': None}
    return {'off': off, 'on': on, 'on_relative_change_pct': 100 * (on / off - 1) if off else None}


def compare_metrics(a, b, suite):
    result = {}
    if suite in ('c8', 'long'):
        for kind in ('api', 'native'):
            if a[kind] and b[kind]: result[kind + '_tokens_per_s'] = change(a[kind]['tokens_per_s'], b[kind]['tokens_per_s'])
        for key in ('median', 'max'): result['ttft_' + key + '_s'] = change(a['ttft_s'][key], b['ttft_s'][key])
        if suite == 'c8': result['max_inter_chunk_gap_s'] = change(a['max_inter_chunk_gap_s'], b['max_inter_chunk_gap_s'])
    elif suite == 'prefill':
        for kind in ('cold', 'warm'):
            for metric in ('read_ms', 'ttft_s', 'decode_tps'):
                result[kind + '_' + metric + '_median'] = change(a['statistics'][kind][metric]['median'], b['statistics'][kind][metric]['median'])
    elif suite == 'c4mixed':
        result['whole_scenario_api_tokens_per_s'] = change(a['whole_scenario_api']['tokens_per_s'], b['whole_scenario_api']['tokens_per_s'])
        if a.get('native', {}).get('available') and b.get('native', {}).get('available'):
            result['native_tokens_per_s'] = change(a['native']['summary']['tokens_per_s'], b['native']['summary']['tokens_per_s'])
        for phase in ('baseline', 'mixed', 'recovery'):
            for metric in ('resident_output_bytes_per_s', 'resident_token_rate_estimate', 'max_observed_no_output_interval_s', 'jain_fairness_of_output_byte_rates'):
                result[phase + '_' + metric] = change(a['phases'][phase][metric], b['phases'][phase][metric])
        result['first_sustained_recovery_start_s'] = change(a['recovery']['first_sustained_recovery_start_s'], b['recovery']['first_sustained_recovery_start_s'])
        ar, br = named(a['actual_requests']), named(b['actual_requests'])
        for name in sorted(ar.keys() & br.keys()):
            if ar[name]['role'] == 'injected':
                result[name + '_ttft_s'] = change(ar[name]['ttft_s'], br[name]['ttft_s'])
    return result


def analyze(run):
    result = {'scope': 'Independent P2 off/on audit with fixed P1=1; local files only.', 'status': 'PENDING',
              'deployment_recommendation': 'HOLD_PENDING_COMPLETED_EVIDENCE', 'run': str(run.resolve()),
              'expected_identities': EXPECTED, 'arms': {}, 'available_suite_comparisons': {}, 'limits': LIMITS}
    manifest_available = (run / 'manifest.json').exists()
    manifest = read(run / 'manifest.json') if manifest_available else {}
    if not manifest_available: result['pending_reason'] = 'run_manifest_not_downloaded; partial suites use locally reviewed harness identities'
    identity = {k: manifest.get(k) == v for k, v in EXPECTED.items()} if manifest_available else {}
    if manifest_available:
        identity['fixed_p1_and_head_flags'] = all(manifest.get('fixed_env', {}).get(k) == v for k, v in FIXED.items())
        identity['only_async_variable'] = manifest.get('variables') == {FLAG: {'off': '0', 'on': '1'}}
        identity['off_on_full_suite_sequence'] = manifest.get('sequence') == ['off', 'on'] and manifest.get('suites') == list(SUITES)
    reviewed = {p.name: sha(p) for p in (B / 'p2-harness').iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.txt')}
    hashes = manifest.get('harness_sha256', reviewed); source_checks = {}; missing = []
    for name, expected in hashes.items():
        require(Path(name).name == name and HEX.fullmatch(expected), 'invalid_harness_inventory')
        path = run / 'harness' / name
        if path.exists(): source_checks[name] = sha(path) == expected and (B / 'p2-harness' / name).exists() and sha(B / 'p2-harness' / name) == expected
        else: missing.append(name)
    if manifest_available: identity['harness_inventory_matches_reviewed'] = set(hashes) == set(reviewed)
    result.update(run_identity_checks=identity, frozen_source_checks=source_checks, missing_frozen_sources=missing)
    raw = {}; metas = {}
    for tag, arm in (('01-off', 'off'), ('02-on', 'on')):
        result['arms'][arm], raw[arm], metas[arm] = audit_arm(run / tag, arm, hashes)
    for suite in SUITES:
        if suite not in raw['off'] or suite not in raw['on']: continue
        a = result['arms']['off']['suites'][suite]['metrics']; b = result['arms']['on']['suites'][suite]['metrics']
        result['available_suite_comparisons'][suite] = {'scope': 'Both suite files completed; final run qualification is separately required.',
            'parity': compare_records(raw['off'][suite], raw['on'][suite], suite), 'performance': compare_metrics(a, b, suite)}
        if suite == 'c4mixed':
            result['available_suite_comparisons'][suite]['same_fixture_and_plan'] = a['fixture_id'] == b['fixture_id'] and a['same_synthetic_plan_sha256'] == b['same_synthetic_plan_sha256']
    if all(metas.values()):
        configs = []
        for meta in metas.values():
            cfg = copy.deepcopy(meta['config']); cfg.get('env', {}).pop(FLAG, None); configs.append(cfg)
        identity['only_async_safe_config_differs'] = configs[0] == configs[1]
        identity['full_normalized_config_hash_matches'] = metas['off']['config_except_variables_log_key_sha256'] == metas['on']['config_except_variables_log_key_sha256']
    cleanup_keys = ('ended_epoch', 'original_config_unchanged', 'remaining_gpu_pids')
    finalized = all(k in manifest for k in cleanup_keys) and len(manifest.get('arms', [])) == 2 and bool(manifest.get('comparisons'))
    result['run_finalized'] = finalized
    if finalized:
        identity['original_config_unchanged'] = manifest['original_config_unchanged'] is True
        identity['no_remaining_gpu_pids'] = manifest['remaining_gpu_pids'] == []
        identity['all_frozen_comparisons_succeeded'] = all(r.get('returncode') == 0 for r in manifest['comparisons'])
        identity['completed_arm_entries_exact'] = ([(r.get('tag'), r.get('arm')) for r in manifest['arms']] == [('01-off', 'off'), ('02-on', 'on')]
            and all(r.get('suite_returncodes') == {s: 0 for s in SUITES} for r in manifest['arms']))
        identity['comparison_entries_exact'] = (len(manifest['comparisons']) == 1 and manifest['comparisons'][0].get('baseline') == '01-off'
            and manifest['comparisons'][0].get('candidate') == '02-on')
        identity['no_runner_failure'] = 'failure' not in manifest
    failure = not all(identity.values()) or any(v is False for v in source_checks.values()) or any(a['status'] == 'FAILED' for a in result['arms'].values())
    failure = failure or any(not c['parity']['compared_cases_passed'] or c.get('same_fixture_and_plan') is False for c in result['available_suite_comparisons'].values())
    all_complete = manifest_available and finalized and not missing and bool(source_checks) and all(a['status'] == 'COMPLETE' for a in result['arms'].values()) and set(result['available_suite_comparisons']) == set(SUITES)
    exact = all(c['parity']['passed'] for c in result['available_suite_comparisons'].values())
    if failure: result.update(status='FAILED', deployment_recommendation='HOLD_FAILED_CORRECTNESS_OR_IDENTITY')
    elif all_complete and not exact: result.update(status='REVIEW', deployment_recommendation='HOLD_UNMATCHED_OR_NONCOMPARABLE_COMPLETED_CASES')
    elif all_complete:
        result['status'] = 'COMPLETE_CORRECTNESS_IDENTITY_PASS'
        pref = result['available_suite_comparisons']['prefill']['performance']
        upward = [k for k, v in pref.items() if not k.endswith('decode_tps_median') and finite(v['on_relative_change_pct']) and v['on_relative_change_pct'] > 0]
        result['prefill_upward_medians_require_review'] = upward
        result['deployment_recommendation'] = 'REVIEW_PERFORMANCE_AND_PREFILL_SPREAD_NO_AUTOMATIC_DEPLOYMENT'
    return result


def self_test():
    row = {'name': 'case', 'synthetic_request': {'max_tokens': 2}, 'output_text': 'ab', 'reasoning_text': '',
           'deltas': [{'content': 'a', 'reasoning': ''}, {'content': 'b', 'reasoning': ''}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
           'finish_reason': 'length', 'saw_done': True, 'error': None, 'max_tokens': 2}
    for a, b in (('synthetic_request', 'request_sha256'), ('output_text', 'output_sha256'), ('reasoning_text', 'reasoning_sha256')): row[b] = digest(row[a])
    audit_record(row, 'c8'); other = copy.deepcopy(row); other['deltas'] = [{'content': 'ab', 'reasoning': ''}]
    other['usage']['cached_tokens'] = 2
    audit_record(other, 'c8'); assert compare_records([row], [other], 'c8')['passed']
    rejected = 0
    for field, value in (('output_text', 'xx'), ('output_text', None), ('synthetic_request', None), ('reasoning_text', None), ('saw_done', False), ('finish_reason', None), ('usage', {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 5}), ('output_sha256', None)):
        bad = copy.deepcopy(row); bad[field] = value
        try: audit_record(bad, 'c8')
        except (ValueError, KeyError, TypeError): rejected += 1
        else: raise AssertionError(field)
    for rows in ([], [row, row]):
        try: named(rows)
        except ValueError: rejected += 1
        else: raise AssertionError('bad_record_set')
    assert not compare_records([row], [dict(other, name='missing')], 'c8')['passed']
    assert analyze(B / '__nonexistent_p2_audit_selftest__')['status'] == 'PENDING'
    print(json.dumps({'self_test': 'PASS', 'rejected_cases': rejected, 'pending_not_pass': True, 'network_gpu_operations': 0}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=B / 'p2-validation')
    p.add_argument('--out-prefix', type=Path, default=B / 'p2-validation-analysis')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test: self_test(); return 0
    require(not a.out_prefix.resolve().is_relative_to(a.run.resolve()), 'output_must_not_modify_raw_run')
    result = analyze(a.run)
    sha(Path(__file__))
    result['source_sha256'] = SOURCE_HASHES
    a.out_prefix.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['Independent P2 validation audit', 'Status: ' + result['status'], 'Recommendation: ' + result['deployment_recommendation']]
    for arm, r in result['arms'].items():
        lines.append(arm + ': ' + r['status'])
        for suite, s in r['suites'].items():
            lines.append('  ' + suite + ': ' + s['status'] + (' (' + s['reason'] + ')' if 'reason' in s else ''))
            if 'route' in s: lines.append('    async windows=%d, sampled rows=%d; work counters, not accepted outputs.' % (s['route']['async_windows'], s['route']['async_sampled_rows']))
            metrics = s.get('metrics', {})
            if suite in ('c8', 'long') and metrics:
                lines.append('    API %.3f tok/s including request admission/drain; native %.3f emitted rows/s.' % (metrics['api']['tokens_per_s'], metrics['native']['tokens_per_s']))
            if suite == 'prefill' and metrics:
                for kind in ('cold', 'warm'):
                    m = metrics['statistics'][kind]
                    lines.append('    %s median read %.3f s, TTFT %.3f s, solo decode %.3f tok/s; n=3.' % (kind, m['read_ms']['median'] / 1000, m['ttft_s']['median'], m['decode_tps']['median']))
            if suite == 'c4mixed' and metrics:
                lines.append('    Whole scenario API %.3f tok/s; mixed maximum silence %.3f s; byte-rate Jain %.6f.' % (metrics['whole_scenario_api']['tokens_per_s'], metrics['phases']['mixed']['max_observed_no_output_interval_s'], metrics['phases']['mixed']['jain_fairness_of_output_byte_rates']))
            if 'history_coverage' in s.get('metrics', {}):
                for h in s['metrics']['history_coverage']['fixtures']:
                    lines.append('    %s: history=%d, completed=%d, reaches window=%s.' % (h['name'], h['actual_sent_parameters']['penalty_last_n'], h['completion_tokens'], h['completed_output_exceeds_history_window']))
    for suite, r in result['available_suite_comparisons'].items():
        lines.append(suite + ': exact matched cases=' + str(r['parity']['passed']))
        for key, v in r['performance'].items():
            pct = 'n/a' if v['on_relative_change_pct'] is None else '%+.3f%%' % v['on_relative_change_pct']
            lines.append('  %s: %s -> %s (%s on vs off)' % (key, v['off'], v['on'], pct))
    lines.extend(LIMITS)
    a.out_prefix.with_suffix('.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
