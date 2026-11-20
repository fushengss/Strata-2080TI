#!/usr/bin/env python3
"""Offline read-guard A/B audit. No service, network, GPU, or deployment actions."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import tempfile

ENGINE = 'c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6'
SERVER = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
LIBRARY = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
BASE_CONFIG = '93492533633510b37d8ec25a825a2c2313eb03dd94dfecbf2e8f84b4b6772be0'
BUILD_RECEIPT = 'ae9d47daa66c4f7c4eb6312e90186e3505c0efdc7cca89d61be762a115ee3cc3'
BUILD_SOURCE = 'e8ea2cb8570aa9d659357b2df8286406d5b8f49e0a77ae54ca3c8eef932bc77c'
HELPER_SHA = '1f8b20065928caf60e4330198a0676ab90fd101179344be1a24292d51445cd4e'
PINNED_HARNESS = {
    'runner_guard.py': '8a28fb1a1901edb879185ab04357b01b65e98b0647c378622cf06175e9ac1c49',
    'prefill_guard_check.py': '12c8e691c87cf2ee74ed0846d36f6247aafdadbaaa0702d2bd781ee962adf0a1',
    'client.py': '3c019afbcd75d7a183ce1089aea49ffd21abadf8aba85445573c369996c43b8f',
    'workload-plan.json': '7cff12d7353c1fdf3d5847bd8e50debe16318ca8a90449535a080fe16a35f426',
    'correctness.py': 'b3e32b596882bad35c9b6dd18f7c995615a7b4cc1ac1674cbf1bf3d06ed13a94',
}
FLAGS = ('STRATA_PIPED_READ_PIECE', 'STRATA_PIPED_READ_DECODE_ONLY', 'STRATA_PIPED_DECODE_MAX_MS')
FIXED = {'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_BATCH_SELF_COMMIT': '0',
         'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5', 'STRATA_PW_WITH_BATCH': '1',
         'STRATA_PIPED_MTP_HEAD': '1'}
USAGE = ('prompt_tokens', 'completion_tokens', 'total_tokens')
CHANNELS = {'visible_content', 'reasoning', 'tool'}
HEAD_RE = re.compile(r'strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch')
READ_RE = re.compile(r'strata batch: piped read (\d+) tokens in (\d+) ms \(to (\d+) of (\d+)\)'
                     r'(?:, then the slots decoded (\d+) rows in (\d+) ms)?')
LEAD_RE = re.compile(r'strata batch: a read right after a read: the slots decoded (\d+) rows in (\d+) ms first')
PERF_RE = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), '
                     r'(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)')
ERROR_RE = re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|illegal memory access|device.side assert|'
                      r'CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits', re.I)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def change(a, b):
    return (b / a - 1) * 100 if finite(a) and finite(b) and a != 0 else None


def cache(row):
    usage = row.get('usage') or {}
    return (usage.get('prompt_tokens_details') or {}).get('cached_tokens', usage.get('cached_tokens'))


def arm_flags(arm):
    return dict(zip(FLAGS, ('16384', '1', '400') if arm == 'on' else ('0', '0', '0')))


def complete(row):
    u = row.get('usage') or {}
    return (not row.get('error') and row.get('saw_done') is True and row.get('finish_reason') in ('length', 'stop')
            and all(isinstance(u.get(k), int) and not isinstance(u[k], bool) and u[k] >= 0 for k in USAGE)
            and u['prompt_tokens'] + u['completion_tokens'] == u['total_tokens'])


def valid_hashes(row, compact):
    channels = row.get('channel_output_sha256') or {}
    digest = canonical(channels) if compact else hashlib.sha256(json.dumps(channels, sort_keys=True).encode()).hexdigest()
    return (set(channels) == CHANNELS and all(re.fullmatch('[0-9a-f]{64}', v or '') for v in channels.values())
            and re.fullmatch('[0-9a-f]{64}', row.get('request_sha256', '')) is not None
            and row.get('output_sha256') == digest)


def pair_rows(left, right, compact=False):
    maps = [{r['name']: r for r in rows} for rows in (left, right)]
    pairs = []
    for name in sorted(set(maps[0]) | set(maps[1])):
        a, b = maps[0].get(name), maps[1].get(name)
        checks = {'present_in_both': bool(a and b)}
        result = {'name': name, 'checks': checks}
        if a and b:
            checks.update(both_complete=complete(a) and complete(b), both_hash_encodings_valid=valid_hashes(a, compact) and valid_hashes(b, compact))
            for key in ('request_sha256', 'output_sha256', 'channel_output_sha256', 'finish_reason'):
                checks[key + '_equal'] = a.get(key) == b.get(key)
            checks['three_logical_usage_equal'] = all(a['usage'].get(k) == b['usage'].get(k) for k in USAGE)
            checks['full_usage_cache_equal'] = a['usage'] == b['usage']
            result.update(usage={'off': a['usage'], 'on': b['usage']}, cached={'off': cache(a), 'on': cache(b)},
                request_conditions={label: {key: value.get(key) for key in ('nominal_prompt_tokens', 'actual_prompt_tokens',
                    'prompt_tolerance', 'prompt_within_nominal_tolerance', 'max_tokens', 'sampling_profile') if key in value}
                    for label, value in (('off', a), ('on', b))})
        result['passed'] = all(checks.values())
        pairs.append(result)
    return pairs


class Audit:
    def __init__(self, root):
        self.root, self.sources, self.checks = root, {}, {}

    def text(self, relative):
        path = (self.root / relative).resolve()
        self.sources[str(relative)] = sha(path)
        return path.read_text(errors='replace')

    def read(self, relative):
        return json.loads(self.text(relative))

    def check(self, key, value):
        self.checks[key] = bool(value)


def load_helper(path):
    if sha(path) != HELPER_SHA:
        raise RuntimeError('Offline timeline helper source differs from frozen SHA')
    spec = importlib.util.spec_from_file_location('frozen_concurrency_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_evidence(text):
    rows = [[int(v) if v else None for v in m] for m in READ_RE.findall(text)]
    interleaved = [r for r in rows if r[4] is not None]
    leads = [[int(v) for v in r] for r in LEAD_RE.findall(text)]
    return {'rows': rows, 'interleaved_rows': interleaved, 'piece_count': len(rows), 'lead_in_rows': leads,
            'decode_rows': sum(r[4] for r in interleaved), 'decode_ms': sum(r[5] for r in interleaved),
            'piece_read_ms': sum(r[1] for r in rows), 'piece_tokens': sum(r[0] for r in rows),
            'columns': ['piece_tokens', 'piece_ms', 'to_token', 'prompt_tokens', 'decode_rows', 'decode_ms']}


def prompt_evidence(text):
    keys = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'read_ms', 'prefill_tps', 'generated_tokens', 'decode_ms', 'decode_tps')
    return [dict(zip(keys, [float(v) if i in (4, 7) else int(v) for i, v in enumerate(m)])) for m in PERF_RE.findall(text)]


def startup_valid(facts, text):
    own = [(int(d), float(b)) for d, b in re.findall(r'CUDA(\d+) keeps its own prompt buffers \(([0-9.]+) GiB of [0-9.]+ GiB free\): no loan', text)]
    resident = [tuple(map(int, x)) for x in re.findall(r'layer split: CUDA(\d+) runs layers \d+-\d+, expert cache (\d+) slots '
                      r'\([0-9.]+ GiB\), (\d+) of its (\d+) profiled pairs', text)]
    return (facts.get('all_experts_resident') is True and facts.get('all_eight_have_3_80_gib_own_buffers') is True
            and facts.get('batch_slot_devices') == list(range(8)) and facts.get('piped_mtp_startup_enabled') is True
            and facts.get('shared_head_workspaces') == [[8, 3, 5, 7]]
            and len(own) == 8 and {x[0] for x in own} == set(range(8)) and all(abs(x[1] - 3.80) < .011 for x in own)
            and len(resident) == 7 and {x[0] for x in resident} == set(range(1, 8))
            and all(x[1:] == (3072, 3072, 3072) for x in resident)
            and 'pre-filled 3072 of 3072 slots from the profile' in text
            and sorted({int(x) for x in re.findall(r'8 slot sessions on CUDA(\d+)', text)}) == list(range(8)))


def phase_metrics(helper, rows, samples, phase):
    begin, end = phase['start_epoch'], phase['end_epoch']
    result = helper.phase_report(rows, samples, 4, begin, end, 1.0)
    ending_gaps = [c['previous_output_gap_s'] for r in rows if r['role'] == 'resident'
                   for c in r['chunks'] if begin <= c['epoch'] < end and finite(c.get('previous_output_gap_s'))]
    result['pooled_gaps_ending_in_phase'] = helper.distribution(ending_gaps)
    result['pooled_gap_counts'] = {f'over_{n}s': sum(v > n for v in ending_gaps) for n in (1, 5)}
    result['reported_client_phase'] = phase
    for lane in result['lanes']:
        intervals = lane.pop('all_clipped_no_progress_intervals')
        lane['clipped_stalls'] = {f'over_{n}s': {'count': sum(x['duration_s'] > n for x in intervals),
                  'summed_lane_seconds': sum(x['duration_s'] for x in intervals if x['duration_s'] > n)} for n in (1, 5)}
        lane['long_no_progress_intervals'] = [x for x in intervals if x['duration_s'] > 1]
    result['worst_lane_fully_observed_p99_s'] = max((x['fully_observed_inter_output_gaps']['p99_s']
        for x in result['lanes'] if x['fully_observed_inter_output_gaps']['p99_s'] is not None), default=None)
    result['clipped_stall_counts'] = {f'over_{n}s': sum(x['clipped_stalls'][f'over_{n}s']['count'] for x in result['lanes']) for n in (1, 5)}
    return result


def match_injected_reads(rows, prompts):
    residents = [r for r in rows if r['role'] == 'resident']
    injected = sorted((r for r in rows if r['role'] == 'injected'), key=lambda r: r['start_epoch'])
    counts = {r['usage']['prompt_tokens'] for r in rows}
    cold = [p for p in prompts if p['prompt_tokens'] in counts and p['reused_tokens'] == 0 and p['read_tokens'] == p['prompt_tokens']]
    # The 10k injector can have exactly the same token count as all residents.
    # Match by a strict six-cold-read sequence, not by prompt length alone.
    valid = (len(residents) == 4 and len(injected) == 2 and len(cold) == 6
        and Counter(p['prompt_tokens'] for p in cold[:4]) == Counter(r['usage']['prompt_tokens'] for r in residents)
        and all(p['prompt_tokens'] == r['usage']['prompt_tokens'] for p, r in zip(cold[4:], injected))
        and all(r['chunks'][0]['epoch'] < injected[0]['start_epoch'] for r in residents)
        and injected[0]['end_epoch'] <= injected[1]['start_epoch'])
    return valid, [{'name': r['name'], 'actual_prompt_tokens': r['usage']['prompt_tokens'],
                    'ttft_s': r['ttft_s'], 'elapsed_s': r['elapsed_s'], 'usage': r['usage'],
                    'nominal_prompt_tokens': r['nominal_prompt_tokens'], 'prompt_tolerance': r['prompt_tolerance'],
                    'prompt_within_nominal_tolerance': r['prompt_within_nominal_tolerance'],
                    'engine_initial_read': cold[i + 4] if valid else None,
                    'read_match': 'last two of exactly six cold C4 reads after all four residents first emitted; serial injector order' if valid else 'ambiguous'}
                   for i, r in enumerate(injected)]


def analyze(root, helper, build):
    audit = Audit(root)
    manifest = audit.read('manifest.json')
    tags = {'off': Path('01-off'), 'on': Path('02-on')}
    metas = {arm: audit.read(tag / 'metadata.json') for arm, tag in tags.items()}
    expected_variables = {key: {arm: arm_flags(arm)[key] for arm in tags} for key in FLAGS}
    audit.check('same_expected_engine_server_library_base', all(manifest.get(k) == v for k, v in {
        'same_binary_sha256': ENGINE, 'server_sha256': SERVER, 'adjacent_library_sha256': LIBRARY, 'base_config_sha256': BASE_CONFIG}.items()))
    audit.check('exact_pair_suites_and_guard_variables', manifest.get('sequence') == ['off', 'on']
                and manifest.get('suites') == ['prefill', 'performance'] and manifest.get('variables') == expected_variables
                and manifest.get('server_variables') == {'reasoning_budget_engine_cap': {'off': True, 'on': True}})
    audit.check('runner_completed_cleanly', len(manifest.get('arms', [])) == 2 and not manifest.get('failure')
                and manifest.get('original_config_unchanged') is True and manifest.get('remaining_gpu_pids') == [])
    audit.check('same_normalized_config', metas['off']['config_except_variables_log_key_sha256'] == metas['on']['config_except_variables_log_key_sha256'])
    audit.check('fixed_profile_and_fixture', manifest.get('profile') == 'short' and manifest.get('levels') == ['4']
                and manifest.get('sampling_profile') == 'sampled' and re.fullmatch(r'[A-Za-z0-9_-]{1,88}', manifest.get('fixture_id', '')) is not None)
    sources = {}
    for name, expected in manifest['harness_sha256'].items():
        p = Path('harness') / name
        observed = sha(root / p)
        audit.sources[str(p)] = observed
        sources[name] = observed == expected
    audit.check('all_frozen_sources_match_manifest', bool(sources) and all(sources.values()))
    audit.check('required_sources_match_known_freeze', all(manifest['harness_sha256'].get(k) == v for k, v in PINNED_HARNESS.items()))
    receipt = json.loads((build / 'build-result.json').read_text())
    build_sources = json.loads((build / 'source-manifest.json').read_text())
    audit.sources['build-evidence/build-result.json'] = sha(build / 'build-result.json')
    audit.sources['build-evidence/source-manifest.json'] = sha(build / 'source-manifest.json')
    audit.check('known_build_receipt_and_core_sources', audit.sources['build-evidence/build-result.json'] == BUILD_RECEIPT
                and audit.sources['build-evidence/source-manifest.json'] == BUILD_SOURCE
                and receipt.get('engine_sha256') == ENGINE and receipt.get('library_sha256') == LIBRARY
                and receipt.get('source_commit') == '94fa51b605395e4dff741dfae4d2ba67d31bdbf3')
    envs = {arm: m['config']['env'] for arm, m in metas.items()}
    changes = {k: {arm: envs[arm].get(k) for arm in tags} for k in sorted(set(envs['off']) | set(envs['on']))
               if envs['off'].get(k) != envs['on'].get(k)}
    audit.check('only_three_config_env_differences', changes == expected_variables)
    safe_configs = {arm: copy.deepcopy(m['config']) for arm, m in metas.items()}
    for cfg in safe_configs.values():
        for key in FLAGS:
            cfg['env'].pop(key, None)
    audit.check('all_other_safe_config_fields_identical', safe_configs['off'] == safe_configs['on'])
    prefill, c4, summaries, phases, injectors, head, guard, logs, perf_manifests = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for arm, tag in tags.items():
        m = metas[arm]; prefix = arm + ':'
        audit.check(prefix + 'fixed_config_and_only_guard_values', all(envs[arm].get(k) == v for k, v in {**FIXED, **arm_flags(arm)}.items())
                    and m['config'].get('reasoning_budget_engine_cap') is True)
        audit.check(prefix + 'only_two_successful_suites', m.get('suite_returncodes') == {'prefill': 0, 'performance': 0})
        audit.check(prefix + 'actual_process_identity_and_flags', len(m.get('actual_engine', [])) == 1
            and m['actual_engine'][0].get('engine_sha256') == ENGINE and m['actual_engine'][0].get('library_sha256') == LIBRARY
            and m['actual_engine'][0].get('expected_library_mapped') is True
            and m['actual_engine'][0].get('flags') == {**FIXED, **arm_flags(arm)}
            and m.get('engine_sha256') == ENGINE and m.get('adjacent_library_sha256') == LIBRARY)
        concurrency = (m.get('ready_status') or {}).get('concurrency') or {}
        audit.check(prefix + 'eight_ready_serving_slots', concurrency.get('requested') == concurrency.get('serving') == 8)
        audit.check(prefix + 'governors_unchanged_48_schedutil', len(m['governors_before']) == 48
                    and m['governors_before'] == m['governors_after'] and set(m['governors_before'].values()) == {'schedutil'})
        audit.check(prefix + 'no_temporary_credential_match', audit.read(tag / 'credential-scan.json').get('credential_matches') == 0)
        log = audit.text(tag / 'engine.log'); logs[arm] = log
        audit.check(prefix + 'no_engine_error_sentinel', ERROR_RE.search(log) is None)
        audit.check(prefix + 'raw_startup_3_80_buffers_residency_slots', startup_valid(m['startup_facts'], log))
        raw_head = [[int(v) for v in r] for r in HEAD_RE.findall(log)]
        suites = m.get('shared_head_suite_evidence', {})
        head[arm] = {'rows': raw_head, 'steps': sum(r[3] for r in raw_head), 'suites': suites}
        audit.check(prefix + 'real_head_steps_two_suites_consistent', set(suites) == {'prefill', 'performance'}
            and all(s.get('steps') == sum(r[3] for r in s.get('rows', [])) for s in suites.values())
            and [r for key in ('prefill', 'performance') for r in suites.get(key, {}).get('rows', [])] == raw_head
            and all(0 <= r[0] < 8 and r[1:3] == [3, 5] for r in raw_head)
            and suites.get('prefill', {}).get('steps') == 0 and suites.get('performance', {}).get('steps', 0) > 0)
        evidence = read_evidence(log); guard[arm] = evidence
        recorded = m.get('piped_read_suite_evidence', {})
        audit.check(prefix + 'recorded_read_evidence_matches_raw', set(recorded) == {'prefill', 'performance'}
            and [r for key in ('prefill', 'performance') for r in recorded.get(key, {}).get('interleaved_read_rows', [])] == evidence['interleaved_rows']
            and sum(s.get('all_read_piece_log_count', 0) for s in recorded.values()) == evidence['piece_count']
            and [r for key in ('prefill', 'performance') for r in recorded.get(key, {}).get('lead_in_rows', [])] == evidence['lead_in_rows']
            and all(s.get('interleaved_decode_rows') == sum(r[4] for r in s.get('interleaved_read_rows', [])) for s in recorded.values()))
        audit.check(prefix + 'solo_prefill_never_split', recorded.get('prefill', {}).get('all_read_piece_log_count') == 0
                    and recorded.get('prefill', {}).get('lead_in_rows') == [])
        audit.check(prefix + 'actual_guard_route', (evidence['piece_count'] == 0 and not evidence['lead_in_rows']) if arm == 'off'
                    else evidence['piece_count'] > 0 and evidence['decode_rows'] > 0)
        prefill[arm] = audit.read(tag / 'prefill/results.json')
        pm = audit.read(tag / 'prefill/manifest.json')
        audit.check(prefix + 'prefill_manifest_identity', pm.get('completed') is True and pm.get('arm') == arm
                    and pm.get('engine_sha256') == ENGINE and pm.get('library_sha256') == LIBRARY
                    and pm.get('script_sha256') == PINNED_HARNESS['prefill_guard_check.py']
                    and pm.get('fixture_id') == manifest['fixture_id'] + '-prefill' and pm.get('expected_config_flags') == arm_flags(arm))
        audit.check(prefix + 'prefill_six_cold_warm_cache_read_qualified', len(prefill[arm]) == 6
            and {r['name'] for r in prefill[arm]} == {f'{kind}-{i}' for kind in ('cold', 'warm') for i in range(1, 4)}
            and all(r.get('valid') is True and complete(r) and 95000 <= r['usage']['prompt_tokens'] <= 105000
                    and r['usage']['completion_tokens'] == 256 and r['engine']['generated_tokens'] == 256
                    and r['engine']['prompt_tokens'] == r['usage']['prompt_tokens']
                    and r['engine']['reused_tokens'] + r['engine']['read_tokens'] == r['engine']['prompt_tokens']
                    and (r['engine']['reused_tokens'] == cache(r) == 0 if r['kind'] == 'cold' else
                         r['engine']['reused_tokens'] == cache(r) == r['usage']['prompt_tokens'] - 7
                         and r['engine']['read_tokens'] == 7) for r in prefill[arm]))
        raw_prefills = [x for x in prompt_evidence(log) if 95000 <= x['prompt_tokens'] <= 105000 and x['generated_tokens'] == 256]
        audit.check(prefix + 'prefill_engine_metrics_match_raw_log', raw_prefills == [r['engine'] for r in prefill[arm]])
        perf_manifest = audit.read(tag / 'performance/manifest.json'); perf_manifests[arm] = perf_manifest
        audit.check(prefix + 'performance_fixture_plan', perf_manifest.get('fixture_id') == manifest['fixture_id']
                    and perf_manifest.get('levels') == [4] and perf_manifest['plan'].get('sampling_profile') == 'sampled'
                    and perf_manifest['plan'].get('thinking_enabled') is True
                    and perf_manifest.get('script_sha256') == PINNED_HARNESS['client.py']
                    and perf_manifest.get('plan_sha256') == PINNED_HARNESS['workload-plan.json'])
        directory = root / tag / 'performance/c4'
        rows = [audit.read(p.relative_to(root)) for p in sorted(directory.glob('*.json')) if p.name.startswith(('resident-', 'injected-'))]
        c4[arm] = rows
        summary = audit.read(tag / 'performance/c4/summary.json'); summaries[arm] = summary
        samples = audit.read(tag / 'performance/c4/occupancy.json')
        audit.check(prefix + 'c4_six_complete_cold_requests', len(rows) == 6 and len({r['name'] for r in rows}) == 6
            and sum(r['role'] == 'resident' for r in rows) == 4 and sum(r['role'] == 'injected' for r in rows) == 2
            and all(complete(r) and r.get('successful') is True and cache(r) == 0 and r['finish_reason'] == 'length'
                    and r['usage']['completion_tokens'] == (2048 if r['role'] == 'resident' else 64) for r in rows)
            and summary.get('error') is None and summary.get('baseline_occupancy_qualified') is True)
        journals_valid = chunks_valid = True
        for r in rows:
            journal = [json.loads(line) for line in audit.text(tag / 'performance/c4' / (r['name'] + '.chunks.jsonl')).splitlines() if line]
            journals_valid &= [helper.canonical_chunk(c) for c in journal] == [helper.canonical_chunk(c) for c in r['chunks']]
            chunks_valid &= bool(r['chunks']) and all(helper.valid_chunk(c) and r['start_epoch'] <= c['epoch'] <= r['end_epoch'] for c in r['chunks'])
            chunks_valid &= all(a['epoch'] <= b['epoch'] and abs((b['epoch'] - a['epoch']) - b['previous_output_gap_s']) < 1e-4 for a, b in zip(r['chunks'], r['chunks'][1:]))
        audit.check(prefix + 'chunk_journals_match_final_and_valid_clocks', journals_valid and chunks_valid)
        whole = summary['whole_scenario_completed_api_token_rate']
        tokens = sum(r['usage']['completion_tokens'] for r in rows)
        seconds = summary['scenario_end_epoch'] - summary['scenario_start_epoch']
        audit.check(prefix + 'whole_api_tps_recomputed', whole['tokens'] == tokens == 8320 and whole['requests_missing_usage'] == 0
                    and abs(whole['seconds_including_start_and_drain'] - seconds) < .01
                    and abs(whole['tokens_per_s'] - tokens / seconds) < .001)
        phases[arm] = {k: phase_metrics(helper, rows, samples, p) for k, p in summary['phases'].items()}
        audit.check(prefix + 'all_three_phase_metrics_and_samples', set(phases[arm]) == {'baseline', 'mixed', 'recovery'}
                    and all(p['occupancy']['samples'] > 0 for p in phases[arm].values()))
        match, injectors[arm] = match_injected_reads(rows, prompt_evidence(log))
        audit.check(prefix + 'injector_initial_read_ms_unambiguously_matched', match)
    audit.check('c4_full_effective_plan_identical', perf_manifests['off']['plan'] == perf_manifests['on']['plan'])
    prefill_pairs = pair_rows(prefill['off'], prefill['on'], compact=True)
    c4_pairs = pair_rows(c4['off'], c4['on'])
    audit.check('prefill_six_exact_body_output_channel_usage_pairs', len(prefill_pairs) == 6 and all(p['passed'] for p in prefill_pairs))
    audit.check('c4_six_exact_body_output_channel_usage_pairs', len(c4_pairs) == 6 and all(p['passed'] for p in c4_pairs))
    for pair in prefill_pairs:
        rows = [next(r for r in prefill[a] if r['name'] == pair['name']) for a in tags]
        pair['engine_reuse_read_token_counts_equal'] = all(rows[0]['engine'][k] == rows[1]['engine'][k] for k in ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens'))
    audit.check('prefill_exact_engine_reuse_and_read_counts', all(p['engine_reuse_read_token_counts_equal'] for p in prefill_pairs))
    medians = {}
    for kind in ('cold', 'warm'):
        medians[kind] = {}
        for metric in ('read_ms', 'ttft_s', 'decode_ms', 'decode_tps'):
            values = {a: [r['ttft_s'] if metric == 'ttft_s' else r['engine'][metric] for r in prefill[a] if r['kind'] == kind] for a in tags}
            centers = {a: statistics.median(v) for a, v in values.items()}
            medians[kind][metric] = {'values': values, **centers, 'change_pct': change(centers['off'], centers['on'])}
    costs = []
    maps = {a: {r['name']: r for r in injectors[a]} for a in tags}
    for name in sorted(set(maps['off']) | set(maps['on'])):
        row = {'name': name, 'arms': {a: maps[a].get(name) for a in tags}, 'changes_pct': {}}
        if all(row['arms'].values()):
            for key in ('ttft_s', 'elapsed_s'):
                row['changes_pct'][key] = change(maps['off'][name][key], maps['on'][name][key])
            reads = [maps[a][name]['engine_initial_read'] for a in tags]
            row['changes_pct']['read_ms'] = change(reads[0]['read_ms'], reads[1]['read_ms']) if all(reads) else None
        costs.append(row)
    whole = {a: s['whole_scenario_completed_api_token_rate'] for a, s in summaries.items()}
    return {'scope': 'One fixed-head guard off/on pair; cold/warm solo100k and short sampled C4, no deployment decision.',
        'checks': audit.checks, 'all_checks_passed': all(audit.checks.values()), 'env_changes': changes,
        'engine_sha256': ENGINE, 'server_sha256': SERVER, 'library_sha256': LIBRARY,
        'frozen_sources_match_manifest': sources, 'build_core_sources': build_sources, 'source_sha256': audit.sources,
        'prefill_pairs': prefill_pairs, 'prefill_medians': medians,
        'prefill_details': {a: [{'name': r['name'], 'usage': r['usage'], 'engine': r['engine'], 'ttft_s': r['ttft_s']} for r in rows] for a, rows in prefill.items()},
        'c4_pairs': c4_pairs, 'c4_phases': phases, 'c4_injector_costs': costs, 'c4_whole': whole,
        'c4_whole_tps_change_pct': change(whole['off']['tokens_per_s'], whole['on']['tokens_per_s']),
        'head_actual_steps': head, 'guard_actual_pieces': guard,
        'limits': ['One sequential pair and three cold/warm samples; no confidence interval or automatic performance pass.',
                   'SSE chunk gaps are not per-token TBT. Report phase-clipped silence separately from fully observed gaps.',
                   'Pooled gap quantiles weight faster streams more; worst-lane and clipped stalls are also included.',
                   'Occupancy stall fractions count sampled client observations, not GPU occupancy or a continuous-time fraction.',
                   'Engine read_ms includes interleaved decode and related waiting; it is not pure prefill kernel time.',
                   '400ms caps requested decode budget; one pump and drain can exceed it, so it is not a gap guarantee.',
                   'Whole API TPS includes admission, prefills, reasoning/output and final drain.',
                   'C4 actual prompt counts and nominal-tolerance qualifications are retained. Equal actual work does not imply nominal10k/60k qualification.',
                   'Stored build receipts bind known source/binary hashes; offline analysis cannot independently reconstruct a remote build.']}


def fmt(v):
    return f'{v:.3f}' if finite(v) else 'n/a'


def markdown(r):
    lines = ['# 并发 read guard 离线审计', '',
        '仅比较三个 guard 参数：off=0/0/0，on=16384/1/400；两臂共享 head、batch/piped MTP、self_commit=0 和预算 cap=true。', '',
        f"完整性检查：{'全部通过' if r['all_checks_passed'] else '有失败，不能据此认可等价工作量的收益'}。尚无自动部署结论。", '',
        '| 单独100k | off | on | 变化% |', '|---|---:|---:|---:|']
    for kind in ('cold', 'warm'):
        for key in ('read_ms', 'ttft_s'):
            x=r['prefill_medians'][kind][key]
            lines.append(f"| {kind} {key} 中位数 | {fmt(x['off'])} | {fmt(x['on'])} | {fmt(x['change_pct'])} |")
    lines += ['', '| C4阶段 | pooled p99 gap off/on s | pooled max gap off/on s | 最坏lane截断停顿 off/on s | >1s/>5s lane停顿次数 off→on |', '|---|---:|---:|---:|---:|']
    for name in ('baseline','mixed','recovery'):
        a,b=[r['c4_phases'][arm][name] for arm in ('off','on')]
        lines.append(f"| {name} | {fmt(a['pooled_gaps_ending_in_phase']['p99_s'])}/{fmt(b['pooled_gaps_ending_in_phase']['p99_s'])} | "
            f"{fmt(a['pooled_gaps_ending_in_phase']['max_s'])}/{fmt(b['pooled_gaps_ending_in_phase']['max_s'])} | "
            f"{fmt(a['worst_lane_max_no_progress_s'])}/{fmt(b['worst_lane_max_no_progress_s'])} | "
            f"{a['clipped_stall_counts']['over_1s']}/{a['clipped_stall_counts']['over_5s']} → {b['clipped_stall_counts']['over_1s']}/{b['clipped_stall_counts']['over_5s']} |")
    lines += ['', '| 新prefill请求 | TTFT off/on s | engine read off/on ms |', '|---|---:|---:|']
    for item in r['c4_injector_costs']:
        a,b=[item['arms'][k] for k in ('off','on')]
        reads=[x.get('engine_initial_read') or {} for x in (a,b)]
        lines.append(f"| {item['name']} | {fmt(a['ttft_s'])}/{fmt(b['ttft_s'])} | {fmt(reads[0].get('read_ms'))}/{fmt(reads[1].get('read_ms'))} |")
    lines += ['', f"C4整场 API tokens/s：{fmt(r['c4_whole']['off']['tokens_per_s'])} → {fmt(r['c4_whole']['on']['tokens_per_s'])}（{fmt(r['c4_whole_tps_change_pct'])}%）。包含 admission/prefill 与最终 drain。", '',
        f"精确配对：solo prefill {sum(x['passed'] for x in r['prefill_pairs'])}/6，C4 {sum(x['passed'] for x in r['c4_pairs'])}/6。", '',
        'C4按实际prompt_tokens精确配对；名义10k/60k的tolerance资格另行保留，不能把相同实际输入说成名义长度合格。', '',
        '完整 JSON 保留逐请求usage/reuse、两种gap口径、每lane截断边界及>1/>5秒计数、采样stall比例、实际piece/decode行数和全部输入SHA。', '', '限制：', '']
    lines += ['- '+x for x in r['limits']]
    failed=[k for k,v in r['checks'].items() if not v]
    if failed:
        lines += ['', '失败检查：', '']+['- '+x for x in failed]
    return '\n'.join(lines)+'\n'


def self_test(helper):
    channels={k:hashlib.sha256(k.encode()).hexdigest() for k in CHANNELS}
    row={'name':'x','request_sha256':'a'*64,'output_sha256':canonical(channels),'channel_output_sha256':channels,
         'error':None,'saw_done':True,'finish_reason':'length','usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12,'cached_tokens':0}}
    assert pair_rows([row],[copy.deepcopy(row)],compact=True)[0]['passed']
    for key in ('completion_tokens','cached_tokens'):
        other=copy.deepcopy(row);other['usage'][key]+=1
        assert not pair_rows([row],[other],compact=True)[0]['passed']
    other=copy.deepcopy(row);other['channel_output_sha256']['reasoning']='b'*64
    assert not pair_rows([row],[other],compact=True)[0]['passed']
    text='strata batch: piped read 16384 tokens in 120 ms (to 16384 of 52080), then the slots decoded 28 rows in 410 ms\nstrata batch: piped read 16384 tokens in 130 ms (to 32768 of 52080)\nstrata batch: a read right after a read: the slots decoded 4 rows in 100 ms first\n'
    e=read_evidence(text);assert e['piece_count']==2 and e['decode_rows']==28 and e['decode_ms']==410 and e['lead_in_rows']==[[4,100]]
    assert read_evidence('synthetic unrelated log')['piece_count']==0
    rows=[]
    for lane in range(4):
        chunks=[{'epoch':t,'channels':['output'],'bytes':{'output':1,'visible_content':1,'reasoning':0,'tool':0},
                 'previous_output_gap_s':None if i==0 else t-[0.,2.,8.,9.][i-1]} for i,t in enumerate([0.,2.,8.,9.])]
        rows.append({'name':f'resident-{lane}-0','role':'resident','lane':lane,'start_epoch':0.,'end_epoch':10.,'chunks':chunks})
    p=phase_metrics(helper,rows,[],{'start_epoch':1.,'end_epoch':8.5})
    assert p['pooled_gaps_ending_in_phase']['p99_s']==6 and p['clipped_stall_counts']=={'over_1s':4,'over_5s':4}
    assert p['lanes'][0]['long_no_progress_intervals'][0]['duration_s']==6
    print('OFFLINE PASS: exact hashes/usage/cache negative gates, numeric read/lead parser, full versus clipped phase gaps; no service/GPU')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path(__file__).resolve().parent.parent
    p.add_argument('--input',type=Path,default=base/'guard-pilot-v1')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--build-evidence',type=Path,default=base/'head-build')
    p.add_argument('--helper',type=Path,default=base/'analyze.py')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args(); helper=load_helper(a.helper)
    if a.self_test:
        self_test(helper);return 0
    root=a.input.resolve(); out=a.out.resolve()
    if out==root or root in out.parents:
        p.error('Analysis output must be outside input evidence directory')
    result=analyze(root,helper,a.build_evidence.resolve())
    result['analysis_source_sha256']=sha(Path(__file__))
    result['timeline_helper_sha256']=sha(a.helper)
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text(markdown(result))
    print(json.dumps({'all_checks_passed':result['all_checks_passed'],
                     'failed_checks':[k for k,v in result['checks'].items() if not v],
                     'c4_whole_tps_change_pct':result['c4_whole_tps_change_pct']},ensure_ascii=False))
    return 0 if result['all_checks_passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
