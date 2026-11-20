#!/usr/bin/env python3
"""Independent local audit of the first P1 pilot; no service or GPU operations."""
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUN = HERE / 'p1-pilot'
HELPER = HERE.parent / 'rate-audit-2220/gpu-power-fields.py'
spec = importlib.util.spec_from_file_location('gpu_fields', HELPER)
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
SOURCES = [Path(__file__), HELPER]


def read(path):
    SOURCES.append(path)
    return json.loads(path.read_text())


def digest(value):
    raw = value.encode() if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def taipei(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone(dt.timedelta(hours=8))).isoformat()


def native_summary(rows):
    tokens, ms = sum(r['rows'] for r in rows), sum(r['wall_ms'] for r in rows)
    accepted, offered = sum(r['drafts_accepted'] for r in rows), sum(r['drafts_offered'] for r in rows)
    return {'batches': len(rows), 'emitted_rows': tokens, 'wall_ms': ms,
            'sum_rows_over_sum_wall_seconds': tokens / (ms / 1000),
            'arithmetic_mean_reported_batch_rows_per_s': statistics.mean(r['rows_per_s'] for r in rows),
            'drafts_accepted': accepted, 'drafts_offered': offered,
            'acceptance_pct': 100 * accepted / offered if offered else None,
            'scope': 'Native piped batches only, including admission stalls; different scope from whole-request API tokens.'}


def verify_saved_hashes(records):
    for record in records:
        assert record['request_sha256'] == digest(record['synthetic_request'])
        assert record['output_sha256'] == digest(record['output_text'])
        assert record['reasoning_sha256'] == digest(record['reasoning_text'])
    return {'records': len(records), 'hashes_recomputed': 3 * len(records), 'all_match': True}


def compare_records(left, right):
    maps = [{r['name']: r for r in records} for records in (left, right)]
    assert set(maps[0]) == set(maps[1])
    result = []
    for name in sorted(maps[0]):
        a, b = maps[0][name], maps[1][name]
        checks = {'same_request': a['request_sha256'] == b['request_sha256']}
        cancelled = a['intentional_cancel'] or b['intentional_cancel']
        counts = [{k: row['usage'].get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')} for row in (a, b)]
        if cancelled:
            checks['both_intentionally_cancelled_mid_generation'] = all(row['intentional_cancel'] and row['closed_mid_generation'] and not row['saw_done'] for row in (a, b))
            for field in ('output_text', 'reasoning_text'):
                n = min(len(a[field]), len(b[field]))
                checks[field + '_common_prefix_equal'] = a[field][:n] == b[field][:n]
            checks['nonempty_common_prefix'] = min(len(row['output_text']) + len(row['reasoning_text']) for row in (a, b)) > 0
        else:
            checks.update({field + '_equal': a[field] == b[field] for field in ('output_sha256', 'reasoning_sha256', 'finish_reason')})
            checks['both_complete'] = all(row['saw_done'] and not row['error'] and row['finish_reason'] in ('stop', 'length') for row in (a, b))
            checks['logical_usage_equal'] = counts[0] == counts[1]
            checks['logical_usage_consistent'] = all(all(type(v) is int for v in row.values()) and row['prompt_tokens'] + row['completion_tokens'] == row['total_tokens'] for row in counts)
        result.append({'name': name, 'cancelled': cancelled, 'checks': checks, 'passed': all(checks.values()),
                       'logical_usage': counts, 'output_characters': [len(row['output_text']) for row in (a, b)],
                       'full_output_hash_equal': a['output_sha256'] == b['output_sha256']})
    assert all(row['passed'] for row in result)
    return result


def main():
    manifest = read(RUN / 'manifest.json')
    assert manifest['sequence'] == ['off', 'on'] and manifest['suites'] == ['c8', 'correctness']
    assert manifest['original_config_unchanged'] and not manifest['remaining_gpu_pids']
    for name, expected in manifest['harness_sha256'].items():
        artifact = RUN / 'harness' / name
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
    arms, raw_reports = [], []
    for tag in ('01-off', '02-on'):
        root = RUN / tag
        meta = read(root / 'metadata.json')
        c8 = read(root / 'c8/c8-check.json')
        correctness = read(root / 'correctness/correctness.json')
        assert all(meta['suite_returncodes'][suite] == 0 for suite in ('c8', 'correctness'))
        assert all(report['all_checks_passed'] and all(report['checks'].values()) for report in (c8, correctness))
        for process in meta['actual_engine']:
            assert process['engine_sha256'] == manifest['same_binary_sha256']
            assert process['expected_library_mapped'] and process['library_sha256'] == manifest['adjacent_library_sha256']
            assert process['flags']['STRATA_PIPED_TARGET_HEAD'] == ('0' if meta['arm'] == 'off' else '1')
            assert all(process['flags'][key] == value for key, value in manifest['fixed_env'].items())
        controls = {record['name']: record for record in correctness['records']}
        for field in ('request_sha256', 'output_sha256', 'reasoning_sha256', 'finish_reason'):
            assert controls['recovery-control'][field] == controls['recovery-after'][field]
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            assert controls['recovery-control']['usage'][field] == controls['recovery-after']['usage'][field]
        records = c8['records']
        assert len(records) == 8 and all(r['usage']['completion_tokens'] == 1024 for r in records)
        assert sorted(r['synthetic_request']['temperature'] for r in records) == [0, 0, 0, 0, .8, .8, .8, .8]
        for row in records:
            arrivals = [event['epoch'] for event in row['deltas'] if event.get('content')]
            assert min(arrivals) == row['first_output_epoch'] and max(arrivals) == row['last_output_epoch']
        start, end = max(r['first_output_epoch'] for r in records) + 1, min(r['last_output_epoch'] for r in records) - 1
        whole_start, whole_end = min(r['start_epoch'] for r in records), max(r['end_epoch'] for r in records)
        api_tokens = sum(r['usage']['completion_tokens'] for r in records)
        api_rate = api_tokens / (whole_end - whole_start)
        assert abs(api_rate - c8['timing']['whole_batch']['completed_api_tokens_per_s']) < 1e-9
        path = root / 'gpu.csv'; SOURCES.append(path)
        rows, columns = gpu.read_gpu_csv(path)
        assert all(meta['ready_epoch'] <= r['epoch'] <= meta['finished_epoch'] for r in rows)
        devices = gpu.summarize_devices(rows, start, end)
        assert len(devices) == 8
        assert all(d['enforced_power_limit_w']['min'] == d['enforced_power_limit_w']['max'] == 250 for d in devices)
        assert all(d['memory_clock_mhz']['min'] == d['memory_clock_mhz']['max'] == 6800 for d in devices)
        for device in devices:
            for reason in ('sw_power_cap', 'sw_thermal_slowdown', 'hw_thermal_slowdown', 'hw_power_brake_slowdown'):
                assert device['clock_reasons'][reason]['valid_n'] == device['n']
                if reason != 'sw_power_cap':
                    assert device['clock_reasons'][reason]['active_n'] == 0
        all_native = [row for cohort in correctness['cohorts'] for row in cohort['piped_numeric_log']['records']]
        arms.append({'tag': tag, 'metadata': {'engine_sha256': meta['engine_sha256'],
                     'config_except_variable_hash': meta['config_except_variables_log_key_sha256'],
                     'target_head_evidence': meta['target_head_suite_evidence'],
                     'shared_mtp_head_evidence': meta['shared_head_suite_evidence']},
                     'api_c8': {'output_tokens': api_tokens, 'whole_request_wall_s': whole_end - whole_start, 'tokens_per_s': api_rate},
                     'native_c8': native_summary(c8['numeric_piped_log']['records']),
                     'native_correctness_diagnostic': native_summary(all_native),
                     'c8_gpu_window': {'start_epoch': start, 'end_epoch': end, 'duration_s': end-start,
                                      'start_taipei': taipei(start), 'end_taipei': taipei(end),
                                      'definition': 'All eight client nonempty-content intervals intersected, then trim1s each side. Not exact GPU kernel boundaries.'},
                     'gpu_csv_timezone': {'interpreted_as': 'UTC', 'explicit_marker': False,
                                          'first_csv_minus_ready_s': rows[0]['epoch'] - meta['ready_epoch'],
                                          'finished_minus_last_csv_s': meta['finished_epoch'] - rows[-1]['epoch'],
                                          'all_rows_inside_metadata_epoch_bounds': True},
                     'devices': devices, 'mean_gpu_active_pct': statistics.mean(d['gpu_active_pct']['mean'] for d in devices),
                     'sum_mean_power_w': sum(d['power_w']['mean'] for d in devices),
                     'saved_hash_integrity': verify_saved_hashes(c8['records'] + correctness['records'])})
        raw_reports.append({'c8': c8, 'correctness': correctness, 'metadata': meta})
    normalized = []
    for row in raw_reports:
        config = copy.deepcopy(row['metadata']['config']); config['env'].pop('STRATA_PIPED_TARGET_HEAD')
        normalized.append(config)
    assert normalized[0] == normalized[1]
    c8_cases = compare_records(*[r['c8']['records'] for r in raw_reports])
    correctness_cases = compare_records(*[r['correctness']['records'] for r in raw_reports])
    assert len(c8_cases) == 8 and len(correctness_cases) == 14
    automatic = read(RUN / 'comparisons/01-off-vs-02-on/comparison.json')
    assert automatic['all_checks_passed'] and all(automatic['identity_checks'].values())
    old = read(HERE.parent / 'head-reverse/01-on/c8/c8-check.json')
    compatibility = compare_records(old['records'], raw_reports[0]['c8']['records'])
    cross = read(HERE / 'p1-cross-compatibility/cross-compatibility.json')
    assert cross['all_checks_passed'] and len(compatibility) == 8
    clocks = read(HERE / 'clock-lock.json')
    assert clocks['lock']['returncode'] == 0 and clocks['lock']['argv'][-2:] == ['-lgc', '2000,2000']
    assert clocks['power_limit_changed'] is False
    changes = {key: 100 * (arms[1][group][metric] / arms[0][group][metric] - 1)
               for key, group, metric in [('c8_api_pct', 'api_c8', 'tokens_per_s'),
                                          ('c8_native_pct', 'native_c8', 'sum_rows_over_sum_wall_seconds'),
                                          ('correctness_native_diagnostic_pct', 'native_correctness_diagnostic', 'sum_rows_over_sum_wall_seconds')]}
    result = {'scope': 'First P1 off->on pilot, one observation per arm; no stable speedup or deployment claim.',
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dict.fromkeys(SOURCES)},
              'runtime_identity': {key: manifest[key] for key in ('same_binary_sha256', 'adjacent_library_sha256', 'server_sha256', 'base_config_sha256')},
              'arms': arms, 'relative_changes': changes,
              'independent_case_checks': {'c8': c8_cases, 'correctness': correctness_cases,
                  'complete_requests_exact': sum(not row['cancelled'] for row in c8_cases + correctness_cases),
                  'cancelled_prefix_cases_passed': sum(row['cancelled'] and row['passed'] for row in correctness_cases),
                  'reference_c9305_to_new_off_c8': compatibility,
                  'note': '21 complete cases exact; one intentional cancellation is compared by valid nonempty common prefix, not full output equality.'},
              'clock_policy': {'requested_sm_lock_mhz': 2000, 'lock_command_succeeded': True,
                               'power_limit_changed': False, 'selected_window_enforced_cap_w_each_gpu': 250,
                               'note': 'Requested clock is not achieved clock. Actual sampled clocks are reported per GPU.'},
              'prefill_conclusion': None, 'prefill_tested': False,
              'limits': ['Only one off/on pair, short prompts, 8x1024 C8; not a100k or mixed-injection performance result.',
                         'No solo100k prefill, C4mixed injection, budget suite or long suite in this pilot. Existing correctness includes cancellation and a penalty case only.',
                         'Native correctness cohorts include intentional cancellation and timing-dependent admissions; their aggregate rate is diagnostic, not a controlled performance endpoint.',
                         'GPU CSV is sampled data and no SM occupancy/FLOPS/bandwidth percentage is measured. Memory activity is a time fraction.',
                         'Reverse-order broader validation is still required before claiming repeatable benefit or prefill nonregression.'],
              'all_audit_checks_passed': True}
    (HERE / 'p1-pilot-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['P1 first-pilot independent audit — one off/on pair, not a stable-benefit conclusion',
             f"C8 API: {arms[0]['api_c8']['tokens_per_s']:.6f} -> {arms[1]['api_c8']['tokens_per_s']:.6f} tokens/s ({changes['c8_api_pct']:+.3f}%). Both8192 tokens; full API durations28.6530s ->27.2959s.",
             f"C8 native: {arms[0]['native_c8']['sum_rows_over_sum_wall_seconds']:.6f} -> {arms[1]['native_c8']['sum_rows_over_sum_wall_seconds']:.6f} emitted rows/s ({changes['c8_native_pct']:+.3f}%). Both8180 rows;27.470s ->26.169s; MTP3499/4675 identical.",
             'Independent correctness:8 C8 +13 normal correctness requests have exact request/output/reasoning hashes, finish and logical usage. One cancelled request has183 ->215 output characters with the same valid shorter prefix; full cancelled hashes differ as allowed. Recovery controls pass in both arms.',
             'Recomputed132 saved hashes across44 records. All match. Old c9305 ->new off independently passes all8 C8 cases; no cross-version performance comparison.',
             f"Correctness native diagnostic: {arms[0]['native_correctness_diagnostic']['sum_rows_over_sum_wall_seconds']:.3f} -> {arms[1]['native_correctness_diagnostic']['sum_rows_over_sum_wall_seconds']:.3f} rows/s ({changes['correctness_native_diagnostic_pct']:+.3f}%). Includes cancellation; do not generalize C8 gain to this workload.",
             f"Off common decode window: {arms[0]['c8_gpu_window']['start_taipei']} to {arms[0]['c8_gpu_window']['end_taipei']}.",
             f"On common decode window: {arms[1]['c8_gpu_window']['start_taipei']} to {arms[1]['c8_gpu_window']['end_taipei']}.",
             'Windows are eight-stream content intersections trimmed1s at each end. GPU CSV UTC verified against metadata epochs.',
             'GPU | n off/on | active% off/on | W off/on | SM MHz off/on | mean C off/on | sw_power_cap active/valid off/on']
    for a, b in zip(arms[0]['devices'], arms[1]['devices']):
        ca, cb = a['clock_reasons']['sw_power_cap'], b['clock_reasons']['sw_power_cap']
        lines.append(f"{a['gpu']} | {a['n']}/{b['n']} | {a['gpu_active_pct']['mean']:.2f}/{b['gpu_active_pct']['mean']:.2f} | "
                     f"{a['power_w']['mean']:.2f}/{b['power_w']['mean']:.2f} | {a['sm_clock_mhz']['mean']:.2f}/{b['sm_clock_mhz']['mean']:.2f} | "
                     f"{a['temperature_c']['mean']:.2f}/{b['temperature_c']['mean']:.2f} | {ca['active_n']}/{ca['valid_n']} ->{cb['active_n']}/{cb['valid_n']}")
    lines += [f"Equal-GPU mean active: {arms[0]['mean_gpu_active_pct']:.2f}% ->{arms[1]['mean_gpu_active_pct']:.2f}%; total mean power{arms[0]['sum_mean_power_w']:.2f}W ->{arms[1]['sum_mean_power_w']:.2f}W.",
              'GPU7 work shifts to GPU2/4: GPU7 busy96.89% ->80.46%, while GPU2/4 rise to74.88%/74.27%. On-arm target route proves4681 windows/9356 verification rows offloaded to2+4; verification rows are not outputs.',
              'All sampled cards retain250W enforced caps and6800MHz memory clocks. All selected SW/HW thermal and HW power-brake flags are inactive. GPU7 remains power-capped at every sample; GPUs2/4 also become power-capped at every on-arm sample.',
              'The recorded2000MHz lock request succeeded, but achieved clocks remain lower under the observed constraints. Do not report an achieved fixed2000MHz frequency.',
              'No prefill result exists in this pilot. No deployment/nonregression conclusion; next reverse order should include100k prefill, C4mixed injections, budget and long suites.',
              'Reproduce locally: python3 p1-pilot-analysis.py']
    (HERE / 'p1-pilot-analysis.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
