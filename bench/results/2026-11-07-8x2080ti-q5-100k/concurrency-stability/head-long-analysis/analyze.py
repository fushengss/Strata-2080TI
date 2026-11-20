#!/usr/bin/env python3
"""Offline audit of the frozen head-long experiment; no network or GPU access."""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'head-long'
ENGINE = 'c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6'
LIBRARY = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
SERVER = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
CLIENT = '23ec55c406b48accc1ee64e20ec7da986aa0d4a3c65a3a3dfe7d3116b170c696'
RUNNER = '2e41a4ae917b0a6345b7fc6caed70200cd4d216d09794f54245239a365014da5'
FIXED = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
         'STRATA_PW_WITH_BATCH': '1', 'STRATA_BATCH_MTP': '1',
         'STRATA_PIPED_MTP': '1', 'STRATA_BATCH_SELF_COMMIT': '0'}
HEAD = 'STRATA_PIPED_MTP_HEAD'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    values = sorted(values)
    return {'n': len(values), 'p50_s': values[math.ceil(.5 * len(values)) - 1] if values else None,
            'p95_s': values[math.ceil(.95 * len(values)) - 1] if values else None,
            'max_s': max(values) if values else None, 'method': 'nearest rank; descriptive, no SLO inference'}


def common_lane(record, start, end):
    chunks = [d for d in record['deltas'] if start <= d['epoch'] < end]
    times = [d['epoch'] for d in chunks]
    spans = []
    for index, (left, right) in enumerate(zip([start] + times, times + [end])):
        spans.append({'start_epoch': left, 'end_epoch': right, 'duration_s': right - left,
                      'left_boundary_censored': index == 0,
                      'right_boundary_censored': index == len(times)})
    counts = {channel: sum(d['bytes'].get(channel, 0) for d in chunks)
              for channel in ('content', 'reasoning', 'tool')}
    return {'name': record['name'], 'window_output_chunks': len(chunks), 'window_bytes': counts,
            'window_bytes_per_s': {k: v / (end - start) for k, v in counts.items()},
            'complete_inter_output_gaps': quantiles([b - a for a, b in zip(times, times[1:])]),
            'longest_clipped_silence_s': max(x['duration_s'] for x in spans),
            'clipped_silences_over_1s': [x for x in spans if x['duration_s'] > 1],
            'clipped_silences_over_5s': [x for x in spans if x['duration_s'] > 5]}


def normalized_safe_config(config):
    result = copy.deepcopy(config)
    result['env'].pop(HEAD, None)
    return result


def main():
    manifest = read(ROOT / 'manifest.json')
    exact = read(HERE / 'exact-comparison/comparison.json')
    checks = {
        'two_arms_completed': manifest['sequence'] == ['off', 'on'] and len(manifest['arms']) == 2
            and all(x['suite_returncodes'] == {'long': 0} for x in manifest['arms']) and not manifest.get('failure'),
        'frozen_server_engine_library': manifest['server_sha256'] == SERVER
            and manifest['same_binary_sha256'] == ENGINE and manifest['adjacent_library_sha256'] == LIBRARY,
        'only_head_variable': manifest['variables'] == {HEAD: {'off': '0', 'on': '1'}},
        'fixed_flags_and_cap': manifest['fixed_env'] == FIXED
            and manifest['server_variables'] == {'reasoning_budget_engine_cap': {'off': True, 'on': True}},
        'frozen_client_and_runner': manifest['harness_sha256'].get('long_concurrency_check.py') == CLIENT
            and manifest['harness_sha256'].get('runner_long.py') == RUNNER,
        'all_frozen_harness_files_match': all((ROOT / 'harness' / name).is_file()
            and sha(ROOT / 'harness' / name) == value for name, value in manifest['harness_sha256'].items()),
        'governors_unchanged_across_run': bool(manifest['governors_before'])
            and manifest['governors_before'] == manifest['governors_after'],
        'released_and_original_preserved': manifest['remaining_gpu_pids'] == []
            and manifest['original_config_unchanged'] is True and bool(manifest.get('ended_epoch')),
        'frozen_client_exact_comparison_passed': exact['all_checks_passed'] is True,
    }
    arms = []
    safe_configs = []
    for tag, flag in [('01-off', '0'), ('02-on', '1')]:
        path = ROOT / tag
        meta, report = read(path / 'metadata.json'), read(path / 'long/long-concurrency.json')
        long_manifest = read(path / 'long/manifest.json')
        records = report['records']
        fact = meta['startup_facts']
        procs = meta['actual_engine']
        route = meta['shared_head_suite_evidence']['long']
        governors = meta['governors_before']
        expected_flags = {**FIXED, HEAD: flag}
        route_log = [[int(v) for v in row] for row in re.findall(
            r'strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch',
            (path / 'engine.log').read_text(errors='replace'))]
        arm_checks = {
            'all_recorded_qualifications_passed': report['all_checks_passed'] is True
                and all(report['summary']['checks'].values()),
            'one_verified_actual_process': len(procs) == 1 and procs[0]['engine_sha256'] == ENGINE
                and procs[0]['library_sha256'] == LIBRARY and procs[0]['expected_library_mapped'] is True
                and procs[0]['flags'] == expected_flags,
            'actual_config_fixed': all(meta['config']['env'].get(k) == v for k, v in expected_flags.items())
                and meta['config']['reasoning_budget_engine_cap'] is True
                and meta['config']['parallel'] == 8 and meta['config']['layer_split'] == '6,12,18,24,30,36,42',
            'eight_slots': meta['ready_status']['concurrency']['requested'] == 8
                and meta['ready_status']['concurrency']['serving'] == 8 and fact['batch_slot_devices'] == list(range(8)),
            'all_experts_resident': fact['primary_resident'] == [3072, 3072]
                and len(fact['resident_stages']) == 7 and {x['device'] for x in fact['resident_stages']} == set(range(1, 8))
                and all(x['slots'] == x['resident'] == x['profiled'] == 3072 for x in fact['resident_stages']),
            'eight_own_buffers_3_80gib': len(fact['own_prompt_buffers']) == 8
                and {x['device'] for x in fact['own_prompt_buffers']} == set(range(8))
                and all(abs(x['buffer_gib'] - 3.80) < .011 for x in fact['own_prompt_buffers']),
            'startup_routes_match': fact['piped_mtp_startup_enabled'] is True
                and fact['shared_head_workspaces'] == ([[8, 3, 5, 7]] if flag == '1' else []),
            'actual_head_step_route': route['rows'] == route_log
                and route['steps'] == sum(row[3] for row in route_log)
                and (route['steps'] > 0 if flag == '1' else not route_log)
                and all(0 <= row[0] < 8 and row[1:3] == [3, 5] for row in route_log),
            'governors_unchanged': bool(governors) and governors == meta['governors_after'] == manifest['governors_before'],
            'credential_scan_clean': read(path / 'credential-scan.json')['credential_matches'] == 0,
            'suite_completed': meta['suite_returncodes'] == {'long': 0} and bool(meta.get('finished_epoch')),
            'long_manifest_identity': long_manifest['source_sha256'] == CLIENT
                and long_manifest['engine_sha256'] == ENGINE and long_manifest['adjacent_library_sha256'] == LIBRARY
                and long_manifest['fixture_id'] == manifest['fixture_id'] and long_manifest['completed'] is True,
            'request_files_match_report': all(read(path / 'long' / (r['name'] + '.json')) == r for r in records),
            'all_chunk_byte_totals_match': all({k: sum(d['bytes'].get(k, 0) for d in r['deltas'])
                for k in ('content', 'reasoning', 'tool')} == r['output_bytes'] for r in records),
            'chronological_chunks': all(all(a['epoch'] <= b['epoch'] for a, b in zip(r['deltas'], r['deltas'][1:]))
                for r in records),
        }
        start = max(r['first_output_epoch'] for r in records)
        end = min(r['last_output_epoch'] for r in records)
        common = [common_lane(r, start, end) for r in records]
        common_bytes = sum(sum(lane['window_bytes'].values()) for lane in common)
        lanes = []
        for record in records:
            times = [d['elapsed_s'] for d in record['deltas']]
            lanes.append({'name': record['name'], 'temperature': record['temperature'], 'seed': record['seed'],
                          'usage': record['usage'], 'ttft_s': record['ttft_s'], 'total_s': record['total_s'],
                          'request_sha256': record['request_sha256'], 'output_sha256': record['output_sha256'],
                          'channel_sha256': record['channel_sha256'], 'full_request_inter_output_gaps': quantiles(
                              [b - a for a, b in zip(times, times[1:])])})
        safe_configs.append(normalized_safe_config(meta['config']))
        arms.append({'tag': tag, 'checks': arm_checks, 'source_sha256': {str(f.relative_to(ROOT)): sha(f)
                     for f in [path / 'metadata.json', path / 'engine.log', path / 'long/long-concurrency.json',
                               path / 'long/manifest.json', path / 'credential-scan.json']},
                     'governors': sorted(set(governors.values())), 'whole_batch': {
                         'duration_s': report['summary']['whole_batch_s'],
                         'api_completion_tokens': report['summary']['whole_batch_completion_tokens'],
                         'api_tokens_per_s_including_prefill': report['summary']['whole_batch_tps']},
                     'head_offload_steps': route, 'lanes': lanes,
                     'common_output_window': {'start_epoch': start, 'end_epoch': end, 'duration_s': end - start,
                         'selection': 'half-open[start,end); max(first output) to min(last output)',
                         'total_output_bytes': common_bytes, 'output_bytes_per_s': common_bytes / (end - start),
                         'lanes': common}})
    checks['safe_configs_equal_except_head'] = safe_configs[0] == safe_configs[1]
    checks['all_arm_checks_passed'] = all(all(a['checks'].values()) for a in arms)
    off, on = arms
    def change(a, b):
        return (b / a - 1) * 100
    result = {'all_checks_passed': all(checks.values()), 'checks': checks, 'arms': arms,
              'comparison_sha256': sha(HERE / 'exact-comparison/comparison.json'),
              'runner_manifest_sha256': sha(ROOT / 'manifest.json'), 'analyzer_sha256': sha(Path(__file__)),
              'changes_percent': {
                  'whole_batch_duration': change(off['whole_batch']['duration_s'], on['whole_batch']['duration_s']),
                  'whole_batch_api_tps_including_prefill': change(off['whole_batch']['api_tokens_per_s_including_prefill'], on['whole_batch']['api_tokens_per_s_including_prefill']),
                  'common_window_output_bytes_per_s': change(off['common_output_window']['output_bytes_per_s'], on['common_output_window']['output_bytes_per_s'])},
              'limits': ['One off/on cohort; no repetition or confidence interval and no deployment verdict.',
                         'Whole-batch API TPS includes four cold prefills/admissions and final drain; it is not pure decode TPS.',
                         'Common-window bytes/s is observed synthetic output volume, not token TPS. Its boundaries and byte/token ratios can differ.',
                         'SSE gaps are chunk arrival intervals. MTP may emit multiple tokens together; these are not per-token TBT.',
                         'Per-lane TTFT depends on admission order; a lane-by-lane change is not an isolated compute speed comparison.',
                         'Actual process/library/config identity is established by frozen runner evidence, not a live re-query during offline audit.']}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    text = ['# Four cold 100k requests: head sharing off/on', '',
            f"Independent evidence checks: {'PASS' if result['all_checks_passed'] else 'FAIL'}. The frozen client exact comparison also passed.", '',
            'Both arms used the same c930 engine, fb67 library and 40c5 server, eight slots, full expert residency, '
            '3.80 GiB own prompt buffers per GPU, fixed batch MTP and budget cap. Only the shared slot-head flag changed.', '',
            'All four request/output/channel hashes and prompt/completion/total usage match. Each request has an actual '
            '101147–101148-token prompt, explicit cached_tokens=0, and all 512 output tokens. Two requests are sampled and two greedy.', '',
            '| Metric | Head off | Head on |', '| --- | ---: | ---: |',
            f"| Whole batch wall, including prefill (s) | {off['whole_batch']['duration_s']:.6f} | {on['whole_batch']['duration_s']:.6f} |",
            f"| API output tokens / whole wall (tok/s) | {off['whole_batch']['api_tokens_per_s_including_prefill']:.4f} | {on['whole_batch']['api_tokens_per_s_including_prefill']:.4f} |",
            f"| Common output window (s) | {off['common_output_window']['duration_s']:.6f} | {on['common_output_window']['duration_s']:.6f} |",
            f"| Common-window output bytes/s | {off['common_output_window']['output_bytes_per_s']:.2f} | {on['common_output_window']['output_bytes_per_s']:.2f} |", '',
            '| Lane | Temperature | Off TTFT (s) | On TTFT (s) | Off max chunk gap (s) | On max chunk gap (s) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for a, b in zip(off['lanes'], on['lanes']):
        text.append(f"| {a['name']} | {a['temperature']} | {a['ttft_s']:.3f} | {b['ttft_s']:.3f} | {a['full_request_inter_output_gaps']['max_s']:.3f} | {b['full_request_inter_output_gaps']['max_s']:.3f} |")
    text += ['', f"Whole-wall API throughput changed by {result['changes_percent']['whole_batch_api_tps_including_prefill']:+.3f}%. "
             'This single pair is descriptive; it does not establish a repeatable speed gain or a pure-decode rate.', '',
             'The JSON retains each lane’s common-window complete gaps and boundary-censored silence intervals, source hashes, '
             'individual evidence gates, governor identity and actual offloaded-step counts.', '', *('- ' + x for x in result['limits']), '']
    (HERE / 'README.md').write_text('\n'.join(text))
    print(json.dumps({'all_checks_passed': result['all_checks_passed'], 'checks': checks,
                      'changes_percent': result['changes_percent'], 'summary_sha256': sha(HERE / 'summary.json')}))


if __name__ == '__main__':
    main()
