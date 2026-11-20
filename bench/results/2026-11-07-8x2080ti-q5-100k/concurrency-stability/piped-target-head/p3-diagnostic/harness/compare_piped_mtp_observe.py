#!/usr/bin/env python3
"""Offline C8 behavioral comparison of a P3 diagnostic arm against a selected P2 arm.

Never launches a benchmark or imports its runner/client. Never computes speedup,
throughput or latency comparisons. Inputs are completed arm directories, each
with its own parent manifest and frozen harness.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

VERSION = 'piped-mtp-observe-comparison-1'
C8_VERSION = 'piped-mtp-c8-check-1'
C8_CHECKS = ('eight_requests_complete', 'all_exact_requested_output_length', 'four_sampled_four_greedy',
             'eight_stream_common_interval_at_least_one_second', 'exclusive_request_identity', 'piped_route_confirmed')
OBSERVE = 'STRATA_PIPED_MTP_OBSERVE'
P1, P2 = 'STRATA_PIPED_TARGET_HEAD', 'STRATA_PIPED_ASYNC_SAMPLE'
FIXED = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
         'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_PW_WITH_BATCH': '1',
         'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1'}
IDENTITIES = ('same_binary_sha256', 'adjacent_library_sha256', 'server_sha256', 'base_config_sha256')
USAGE = ('prompt_tokens', 'completion_tokens', 'total_tokens')


class QualificationError(ValueError):
    """Messages use fixed labels rather than input text, output or credentials."""


def require(condition, message):
    if not condition:
        raise QualificationError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    data = value.encode('utf-8') if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return sha_bytes(data)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON object key')
        result[key] = value
    return result


def read(path, sources):
    raw = path.read_bytes()
    sources[str(path.resolve())] = sha_bytes(raw)
    value = json.loads(raw, object_pairs_hook=unique_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(QualificationError('Nonfinite JSON value')))
    require(isinstance(value, dict), 'Expected a JSON object')
    return value


def normalized_config(config):
    require(isinstance(config, dict) and isinstance(config.get('env'), dict), 'Missing safe configuration')
    result = copy.deepcopy(config)
    result.pop('exe', None)  # Different commits may use different absolute binary paths.
    result['env'].pop(OBSERVE, None)
    return result


def qualify_arm(root, diagnostic, sources):
    meta = read(root / 'metadata.json', sources)
    manifest = read(root.parent / 'manifest.json', sources)
    report = read(root / 'c8' / 'c8-check.json', sources)
    require(all(is_sha(manifest.get(key)) for key in IDENTITIES), 'Missing four manifest SHA256 identities')
    require(meta.get('engine_sha256') == manifest['same_binary_sha256'] and
            meta.get('adjacent_library_sha256') == manifest['adjacent_library_sha256'], 'Metadata identities not bound to manifest')
    require(meta.get('tag') == root.name and meta.get('arm') in ('off', 'on'), 'Arm directory identity differs')
    require(finite(meta.get('finished_epoch')), 'Arm did not finish')
    require(isinstance(meta.get('suite_returncodes'), dict) and type(meta['suite_returncodes'].get('c8')) is int
            and meta['suite_returncodes']['c8'] == 0, 'C8 suite did not succeed')
    members = [row for row in manifest.get('arms', []) if isinstance(row, dict) and row.get('tag') == root.name]
    require(len(members) == 1 and members[0].get('arm') == meta['arm'] and
            members[0].get('suite_returncodes', {}).get('c8') == 0, 'Completed arm absent from manifest')
    config = meta.get('config')
    normalized_config(config)
    env = config['env']
    require(all(env.get(key) == value for key, value in FIXED.items()), 'Fixed head, MTP or commit paths differ')
    require(env.get(P1) in ('0', '1') and env.get(P2) in ('0', '1'), 'P1/P2 flags not explicit')
    require(env[P2] == ('1' if meta['arm'] == 'on' else '0'), 'P2 arm and flag differ')
    require(env.get(OBSERVE) == '1' if diagnostic else env.get(OBSERVE) in (None, '0'), 'Observation flag differs from selected role')
    require(env.get('STRATA_MTP_PROFILE') in (None, '0'), 'Global MTP profiler is enabled or ambiguous')
    require(config.get('parallel') == 8 and config.get('reasoning_budget_engine_cap') is True,
            'Parallel slots or reasoning budget cap differs')
    actual = meta.get('actual_engine')
    require(isinstance(actual, list) and len(actual) == 1 and isinstance(actual[0], dict), 'Actual engine evidence missing')
    actual = actual[0]
    require(actual.get('engine_sha256') == manifest['same_binary_sha256'] and
            actual.get('library_sha256') == manifest['adjacent_library_sha256'] and
            actual.get('expected_library_mapped') is True, 'Actual engine/library identity differs')
    flags = actual.get('flags', {})
    require(all(flags.get(key) == env[key] for key in (*FIXED, P1, P2)), 'Actual fixed P1/P2 flags differ')
    require(flags.get(OBSERVE) == '1' if diagnostic else flags.get(OBSERVE) in (None, '0'), 'Actual observation flag differs')
    facts = meta.get('startup_facts', {})
    require(facts.get('piped_mtp_startup_enabled') is True and facts.get('shared_head_workspaces') == [[8, 3, 5, 7]],
            'Shared-head startup route missing')
    require(facts.get('target_head_startup') == ([[2, 4, 7]] if env[P1] == '1' else []) and
            facts.get('async_sample_startup') == ([7] if env[P2] == '1' else []), 'P1/P2 startup routes differ')
    if diagnostic:
        require(manifest.get('experiment_type') == 'diagnostic' and manifest.get('diagnostic_only') is True and
                manifest.get('p3_observation_enabled') is True and manifest.get('entrypoint') == 'runner_piped_mtp_observe.py',
                'Candidate is not a P3 diagnostic manifest')
        require(manifest.get('fixed_p1_target_head') == env[P1] and manifest.get('fixed_p2_async_sampling') == env[P2]
                and manifest.get('variables') == {}, 'Candidate fixed diagnostic flags differ')
        require(facts.get('piped_mtp_observe_startup') == [8], 'Candidate observation startup not proven')
    else:
        require(not manifest.get('p3_observation_enabled') and not manifest.get('diagnostic_only'), 'Baseline is not a P2 arm')
    hashes = manifest.get('harness_sha256')
    required = ['c8_check.py', 'correctness.py', 'runner_async_sample.py']
    if diagnostic:
        required += ['runner_piped_mtp_observe.py', 'parse_piped_mtp_observe.py', 'compare_piped_mtp_observe.py']
    require(isinstance(hashes, dict), 'Frozen harness hashes missing')
    if diagnostic:
        require(manifest.get('entrypoint_sha256') == hashes.get('runner_piped_mtp_observe.py') and
                is_sha(manifest.get('entrypoint_sha256')), 'Diagnostic entrypoint not bound to frozen source')
    for name in required:
        require(is_sha(hashes.get(name)), 'Frozen harness hash missing')
        path = root.parent / 'harness' / name
        actual_hash = sha_bytes(path.read_bytes())
        sources[str(path.resolve())] = actual_hash
        require(actual_hash == hashes[name], 'Frozen harness file differs from manifest')
    require(report.get('source_sha256') == hashes['c8_check.py'] and
            report.get('correctness_dependency_sha256') == hashes['correctness.py'], 'C8 report not bound to frozen source')
    require(report.get('version') == C8_VERSION and report.get('all_checks_passed') is True and
            isinstance(report.get('checks'), dict) and set(report['checks']) == set(C8_CHECKS) and
            all(value is True for value in report['checks'].values()), 'C8 report did not qualify')
    require(report.get('synthetic_only') is True and report.get('service_mutations') == 0 and
            report.get('expected_piped_mtp') == 'on' and report.get('max_tokens_per_request') in (768, 1024),
            'C8 workload scope differs')
    report_flags = report.get('config_env_flags', {})
    require(all(report_flags.get(key) == env[key] for key in (*FIXED, P1, P2)), 'C8 report flags differ from arm')
    rows = report.get('records')
    require(isinstance(rows, list) and len(rows) == 8 and all(isinstance(row, dict) for row in rows), 'Expected eight C8 records')
    names = [row.get('name') for row in rows]
    require(all(isinstance(name, str) for name in names) and set(names) == {'c8-lane-' + str(i) for i in range(8)},
            'C8 record names missing or duplicated')
    return {'meta': meta, 'manifest': manifest, 'report': report, 'records': {row['name']: row for row in rows}}


def record_facts(record):
    require(record.get('saw_done') is True and not record.get('error') and record.get('intentional_cancel') is False and
            record.get('closed_mid_generation') is False and record.get('finish_reason') in ('stop', 'length'),
            'C8 request was incomplete or cancelled')
    request = record.get('synthetic_request')
    require(isinstance(request, dict) and is_sha(record.get('request_sha256')) and digest(request) == record['request_sha256'],
            'Request hash does not match captured synthetic request')
    output, reasoning = record.get('output_text'), record.get('reasoning_text')
    require(isinstance(output, str) and isinstance(reasoning, str) and bool(output or reasoning), 'Missing captured C8 text')
    hashes = {'visible_content': digest(output), 'reasoning': digest(reasoning)}
    require(record.get('output_sha256') == hashes['visible_content'] and record.get('reasoning_sha256') == hashes['reasoning'],
            'Output/reasoning hash does not match captured text')
    deltas = record.get('deltas')
    require(isinstance(deltas, list) and all(isinstance(row, dict) and isinstance(row.get('content'), str) and
                                           isinstance(row.get('reasoning'), str) for row in deltas), 'Captured delta channels missing')
    require(''.join(row['content'] for row in deltas) == output and ''.join(row['reasoning'] for row in deltas) == reasoning,
            'Captured delta channels do not reconstruct full text')
    # Frozen C8 records have no tool-channel capture. Do not invent a verified empty tool hash.
    if 'channel_output_sha256' in record:
        require(record['channel_output_sha256'] == hashes, 'Optional channel hashes differ from captured C8 channels')
    usage = record.get('usage')
    require(isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0 for key in USAGE),
            'Logical usage counts missing or noninteger')
    logical = {key: usage[key] for key in USAGE}
    require(logical['prompt_tokens'] + logical['completion_tokens'] == logical['total_tokens'] and
            logical['completion_tokens'] > 0, 'Logical usage is inconsistent')
    return {'request_sha256': record['request_sha256'], 'output_sha256': record['output_sha256'],
            'reasoning_sha256': record['reasoning_sha256'], 'channel_output_sha256': hashes,
            'finish_reason': record['finish_reason'], 'logical_usage': logical,
            'usage_metadata_sha256': digest({key: value for key, value in usage.items() if key not in USAGE})}


def compare(baseline, candidate):
    result = {'version': VERSION, 'scope': 'Normal completed C8 behavioral compatibility only; no performance comparison.',
              'performance_compared': False, 'all_checks_passed': False, 'pairs': [], 'identity_checks': {},
              'source_sha256': {}, 'channel_scope': 'Captured visible_content and reasoning, independent of SSE chunks. '
                  'Frozen C8 does not capture tool calls; no tool-channel compatibility claim.',
              'usage_scope': 'Prompt/completion/total token counts are strict. Cache/accounting metadata differences are reported separately.'}
    try:
        left = qualify_arm(baseline, False, result['source_sha256'])
        right = qualify_arm(candidate, True, result['source_sha256'])
        a, b = left['meta'], right['meta']
        ma, mb = left['manifest'], right['manifest']
        gates = {'each_arm_bound_to_own_four_sha_manifest': True,
                 'same_safe_config_except_binary_path_and_observation': normalized_config(a['config']) == normalized_config(b['config']),
                 'same_fixed_p1': a['config']['env'][P1] == b['config']['env'][P1],
                 'same_fixed_p2': a['config']['env'][P2] == b['config']['env'][P2],
                 'same_server_sha256': ma['server_sha256'] == mb['server_sha256'],
                 'same_base_config_sha256': ma['base_config_sha256'] == mb['base_config_sha256'],
                 'same_frozen_c8_client': ma['harness_sha256']['c8_check.py'] == mb['harness_sha256']['c8_check.py'] and
                                          ma['harness_sha256']['correctness.py'] == mb['harness_sha256']['correctness.py'],
                 'same_c8_max_tokens': left['report']['max_tokens_per_request'] == right['report']['max_tokens_per_request']}
        result['identity_checks'] = gates
        result['bound_identities'] = {'baseline': {key: ma[key] for key in IDENTITIES},
                                      'candidate': {key: mb[key] for key in IDENTITIES}}
        if not all(gates.values()):
            return result
        for name in sorted(left['records']):
            facts = [record_facts(side['records'][name]) for side in (left, right)]
            require(all(fact['finish_reason'] == 'length' and fact['logical_usage']['completion_tokens'] == side['report']['max_tokens_per_request']
                        for fact, side in zip(facts, (left, right))), 'C8 requested output length differs')
            checks = {key + '_equal': facts[0][key] == facts[1][key] for key in
                      ('request_sha256', 'output_sha256', 'reasoning_sha256', 'channel_output_sha256', 'finish_reason', 'logical_usage')}
            checks['captured_text_equal'] = all(left['records'][name][key] == right['records'][name][key]
                                                for key in ('output_text', 'reasoning_text'))
            result['pairs'].append({'name': name, 'checks': checks, 'passed': all(checks.values()),
                                    'baseline': facts[0], 'candidate': facts[1],
                                    'usage_metadata_equal': facts[0]['usage_metadata_sha256'] == facts[1]['usage_metadata_sha256']})
        result['usage_metadata_qualifications'] = [row['name'] for row in result['pairs'] if not row['usage_metadata_equal']]
        result['all_checks_passed'] = len(result['pairs']) == 8 and all(row['passed'] for row in result['pairs'])
    except (QualificationError, OSError, ValueError, TypeError, KeyError) as error:
        result['error_kind'] = type(error).__name__
        if isinstance(error, QualificationError):
            result['error'] = str(error)
    return result


def synthetic_arm(root, diagnostic):
    """Local self-test fixtures only; no inference or production artifacts."""
    root.mkdir(parents=True)
    (root / 'c8').mkdir()
    (root.parent / 'harness').mkdir()
    env = {**FIXED, P1: '0', P2: '1', **({OBSERVE: '1'} if diagnostic else {})}
    hashes = {}
    for name in ('c8_check.py', 'correctness.py', 'runner_async_sample.py', 'runner_piped_mtp_observe.py',
                 'parse_piped_mtp_observe.py', 'compare_piped_mtp_observe.py'):
        data = ('synthetic frozen ' + name).encode()
        (root.parent / 'harness' / name).write_bytes(data)
        hashes[name] = sha_bytes(data)
    identities = {key: digest(key + (str(diagnostic) if key in IDENTITIES[:2] else 'fixed')) for key in IDENTITIES}
    manifest = {**identities, 'harness_sha256': hashes,
                'arms': [{'tag': root.name, 'arm': 'on', 'suite_returncodes': {'c8': 0}}]}
    if diagnostic:
        manifest.update(experiment_type='diagnostic', diagnostic_only=True, p3_observation_enabled=True,
                        entrypoint='runner_piped_mtp_observe.py', entrypoint_sha256=hashes['runner_piped_mtp_observe.py'],
                        fixed_p1_target_head='0', fixed_p2_async_sampling='1', variables={})
    config = {'exe': '/different/' + str(diagnostic), 'parallel': 8, 'gpu': '0,1,2,3,4,5,6,7', 'layer_split': 'fixed',
              'env': env, 'reasoning_budget_engine_cap': True, 'engine_args_sha256': digest([])}
    meta = {'tag': root.name, 'arm': 'on', 'finished_epoch': 1, 'suite_returncodes': {'c8': 0}, 'config': config,
            'engine_sha256': identities['same_binary_sha256'], 'adjacent_library_sha256': identities['adjacent_library_sha256'],
            'actual_engine': [{'engine_sha256': identities['same_binary_sha256'], 'library_sha256': identities['adjacent_library_sha256'],
                               'expected_library_mapped': True, 'flags': env}],
            'startup_facts': {'piped_mtp_startup_enabled': True, 'shared_head_workspaces': [[8, 3, 5, 7]],
                              'target_head_startup': [], 'async_sample_startup': [7],
                              **({'piped_mtp_observe_startup': [8]} if diagnostic else {})}}
    records = []
    for i in range(8):
        request, output = {'seed': i, 'max_tokens': 768}, 'synthetic output ' + str(i)
        records.append({'name': 'c8-lane-' + str(i), 'request_sha256': digest(request), 'synthetic_request': request,
                        'output_text': output, 'reasoning_text': '', 'output_sha256': digest(output), 'reasoning_sha256': digest(''),
                        'saw_done': True, 'error': None, 'intentional_cancel': False, 'closed_mid_generation': False,
                        'finish_reason': 'length', 'usage': {'prompt_tokens': 10, 'completion_tokens': 768, 'total_tokens': 778},
                        'deltas': [{'content': output, 'reasoning': ''}]})
    report = {'version': C8_VERSION, 'source_sha256': hashes['c8_check.py'], 'correctness_dependency_sha256': hashes['correctness.py'],
              'all_checks_passed': True, 'checks': dict.fromkeys(C8_CHECKS, True), 'synthetic_only': True, 'service_mutations': 0,
              'expected_piped_mtp': 'on', 'max_tokens_per_request': 768, 'config_env_flags': env, 'records': records}
    for path, value in ((root / 'metadata.json', meta), (root.parent / 'manifest.json', manifest),
                        (root / 'c8' / 'c8-check.json', report)):
        path.write_text(json.dumps(value))


def self_test():
    rejected = 0
    with tempfile.TemporaryDirectory(prefix='strata-p3-offline-') as temporary:
        base = Path(temporary)
        baseline, candidate = base / 'p2' / '01-on', base / 'p3' / '01-on'
        synthetic_arm(baseline, False)
        synthetic_arm(candidate, True)
        assert compare(baseline, candidate)['all_checks_passed']  # Different engine/library commits allowed.
        path = candidate / 'c8' / 'c8-check.json'
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed['records'][0]['usage']['prompt_tokens_details'] = {'cached_tokens': 9}
        changed['records'][0]['deltas'] = [{'content': 'synthetic ', 'reasoning': ''}, {'content': 'output 0', 'reasoning': ''}]
        path.write_text(json.dumps(changed))
        qualified = compare(baseline, candidate)
        assert qualified['all_checks_passed'] and qualified['usage_metadata_qualifications'] == ['c8-lane-0']
        mutations = [lambda r: r['records'][0].update(output_text='changed'),
                     lambda r: r['records'][0].update(saw_done=False),
                     lambda r: r['records'][0].update(intentional_cancel=True),
                     lambda r: r['records'][0]['usage'].update(total_tokens=999),
                     lambda r: r['records'][0].update(request_sha256='0' * 64),
                     lambda r: r['records'][0].update(name='c8-lane-1'),
                     lambda r: r.update(all_checks_passed=False),
                     lambda r: r['records'][0].update(channel_output_sha256={'tool': digest('')}),
                     lambda r: r['checks'].pop('piped_route_confirmed'),
                     lambda r: r['records'][0].update(finish_reason='stop')]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            path.write_text(json.dumps(value))
            assert not compare(baseline, candidate)['all_checks_passed']
            rejected += 1
        path.write_text(json.dumps(original))
        meta_path = candidate / 'metadata.json'
        meta = json.loads(meta_path.read_text())
        changed = copy.deepcopy(meta)
        changed['config']['env'][P1] = '1'
        meta_path.write_text(json.dumps(changed))
        assert not compare(baseline, candidate)['all_checks_passed']
        rejected += 1
        meta_path.write_text(json.dumps(meta))
        (candidate.parent / 'harness' / 'c8_check.py').write_text('changed frozen source')
        assert not compare(baseline, candidate)['all_checks_passed']
        rejected += 1
    print(json.dumps({'offline_self_test': 'PASS', 'invalid_cases_rejected': rejected,
                      'performance_compared': False, 'network_requests': 0, 'gpu_operations': 0}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, help='Completed selected P2 arm directory')
    parser.add_argument('--candidate', type=Path, help='Completed P3 diagnostic arm directory')
    parser.add_argument('--out', type=Path, help='New directory; writes comparison.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if any(value is None for value in (args.baseline, args.candidate, args.out)):
        parser.error('--baseline, --candidate and --out are required')
    if args.out.exists():
        parser.error('--out must not exist')
    result = compare(args.baseline, args.candidate)
    result['comparator_sha256'] = sha_bytes(Path(__file__).read_bytes())
    args.out.mkdir(parents=True)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'all_checks_passed': result['all_checks_passed'], 'requests_exact': sum(row['passed'] for row in result['pairs']),
                      'performance_compared': False}))
    return 0 if result['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
