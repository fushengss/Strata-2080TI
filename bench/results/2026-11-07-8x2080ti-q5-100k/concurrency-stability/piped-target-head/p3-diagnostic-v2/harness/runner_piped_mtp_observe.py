#!/usr/bin/env python3
"""Single-arm P3 diagnostic; dry-run unless --execute. Inherits the frozen isolated P2 runner."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import runner_async_sample as base

HERE = Path(__file__).resolve().parent
ENTRY = Path(__file__).name
EXTRA_FILES = (ENTRY, 'parse_piped_mtp_observe.py', 'compare_piped_mtp_observe.py',
               'README_P3.txt', 'p3-source-provenance.json', 'p3-frozen-files.json')
STARTUP = re.compile(r'^strata serve: STRATA_PIPED_MTP_OBSERVE: (\d+) independent slot event sets; '
                     r'source-stream body/head-return/tail intervals, no packing or scheduling changes$', re.M)
LIMITATIONS = [
    'Instrumented diagnostic only; elapsed times or throughput are not optimization gains.',
    'Source-stream phase intervals include contention and dependencies; head_return includes mapped transfers.',
    'Per-slot phase sums may overlap; do not sum them into whole-system wall time.',
    'Pending pump samples mean submitted but unreaped chains, not a ready batch or walltime shares.',
    'Unsubmitted jobs at a verdict have not run their body and are not a post-attention ready queue.',
    'No packing, extra inference, frequency setting or production deployment is performed by postprocessing.',
]


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--config', type=Path, default=Path('/srv/Strata/strata-orca-q5_k_m.57.json'))
    p.add_argument('--exe', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--server', type=Path, default=Path('/srv/Strata-0157/serve/server.py'))
    p.add_argument('--python', default='/srv/Strata/.venv/bin/python')
    p.add_argument('--port', type=int, default=8092)
    p.add_argument('--suites', default='c8', help='c8, optionally c4mixed; no other suites or automatic extra benchmarks')
    p.add_argument('--target-head', choices=('0', '1'), default='1', help='Fixed selected P1 mode')
    p.add_argument('--async-sample', choices=('0', '1'), default='1', help='Fixed selected P2 mode; one arm only')
    for component in ('engine', 'library', 'server', 'config'):
        p.add_argument('--' + component + '-sha256', help='Explicit reviewed SHA256 required for execution')
    p.add_argument('--profile', choices=('short', 'full'), default='short')
    p.add_argument('--sampling-profile', choices=('sampled', 'greedy'), default='sampled')
    p.add_argument('--ready-timeout', type=int, default=1200)
    p.add_argument('--suite-timeout', type=int, default=1800)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args(argv)
    suites = a.suites.split(',')
    base.require('c8' in suites and set(suites) <= {'c8', 'c4mixed'} and len(suites) == len(set(suites)),
                 '--suites must include c8 and may additionally include c4mixed, without duplicates')
    base.require(a.port == 8092, 'P3 remains isolated on loopback port8092')
    base.require(not (a.execute and a.self_test), '--self-test cannot execute')
    return a


def forwarded(a):
    values = {'config': a.config, 'exe': a.exe, 'out': a.out, 'server': a.server, 'python': a.python,
              'port': a.port, 'arms': 'on' if a.async_sample == '1' else 'off', 'suites': a.suites,
              'target-head': a.target_head, 'profile': a.profile, 'sampling-profile': a.sampling_profile,
              'levels': '4', 'ready-timeout': a.ready_timeout, 'suite-timeout': a.suite_timeout}
    values.update({component + '-sha256': getattr(a, component + '_sha256')
                   for component in ('engine', 'library', 'server', 'config')})
    result = []
    for name, value in values.items():
        if value is not None:
            result += ['--' + name, str(value)]
    if a.execute:
        result.append('--execute')
    return result


def install(a):
    """Patch only this process's imported copied runner, never its frozen source files."""
    import parse_piped_mtp_observe as observation
    base.FIXED_ENV.update(STRATA_PIPED_MTP_OBSERVE='1', STRATA_PIPED_TARGET_HEAD=a.target_head,
                          STRATA_PIPED_ASYNC_SAMPLE=a.async_sample)
    old_freeze, old_save = base.freeze, base.save
    old_startup, old_async_rows = base.startup_facts, base.async_sample_rows
    suites = a.suites.split(',')
    suite_diagnostics = {}

    def all_files(source):
        return sorted(p for p in source.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.txt'))

    def freeze(source):
        for name in EXTRA_FILES:
            base.require((source / name).is_file(), 'Missing P3 diagnostic file: ' + name)
        provenance = json.loads((source / 'p3-source-provenance.json').read_text())
        for name, expected in provenance['copied_p2_sha256'].items():
            base.require(base.sha(source / name) == expected, 'Copied frozen P2 source changed: ' + name)
        reviewed = json.loads((source / 'p3-frozen-files.json').read_text())['files']
        actual_names = {p.name for p in all_files(source)} - {'p3-frozen-files.json'}
        base.require(actual_names == set(reviewed), 'P3 reviewed file inventory changed')
        for name, expected in reviewed.items():
            base.require(base.sha(source / name) == expected, 'Reviewed P3 diagnostic source changed: ' + name)
        result = old_freeze(source)
        base.require(all(name in result for name in EXTRA_FILES), 'P3 wrapper/parser/comparator not frozen')
        return result

    def startup(path):
        result = old_startup(path)
        rows = [int(value) for value in STARTUP.findall(path.read_text(errors='replace'))]
        base.require(rows == [8], 'P3 observation startup did not confirm eight independent slot event sets')
        result['piped_mtp_observe_startup'] = rows
        return result

    def async_rows(text):
        index = len(suite_diagnostics)
        base.require(index < len(suites), 'Unexpected extra diagnostic suite callback')
        result = observation.parse_text(text, slots=8)
        suite_diagnostics[suites[index]] = result
        return old_async_rows(text)

    def save(path, value):
        if path.name == 'manifest.json' and 'same_binary_sha256' in value:
            value.update(experiment_type='diagnostic', diagnostic_only=True, entrypoint=ENTRY,
                         entrypoint_sha256=value['harness_sha256'][ENTRY], p3_observation_enabled=True,
                         fixed_p1_target_head=a.target_head, fixed_p2_async_sampling=a.async_sample,
                         variables={}, diagnostic_limitations=LIMITATIONS,
                         arm_label_note='off/on is only the fixed selected P2 mode; this run has one diagnostic arm')
            base.require(len(value['sequence']) == 1, 'P3 must never run an A/B pair')
            if 'comparison_note' in value:
                value['comparison_note'] = 'Single instrumented diagnostic; no performance comparison or optimization claim.'
        elif path.name == 'metadata.json' and 'config_except_variables_log_key_sha256' in value:
            value.update(diagnostic_only=True, entrypoint=ENTRY,
                         diagnostic_suite_evidence=copy.deepcopy(suite_diagnostics))
        old_save(path, value)

    base.harness_files = all_files
    base.freeze, base.save = freeze, save
    base.startup_facts, base.async_sample_rows = startup, async_rows
    return observation


def self_test(a):
    before = {p.name: base.sha(p) for p in HERE.iterdir() if p.is_file()}
    observation = install(a)
    hashes = base.freeze(HERE)
    assert all(name in hashes for name in EXTRA_FILES)
    source = {'exe': '/old', 'env': {'STRATA_MTP_PROFILE': '0'}, 'api_key': 'do-not-export'}
    untouched = copy.deepcopy(source)
    arm = 'on' if a.async_sample == '1' else 'off'
    cfg = base.make_config(source, arm, '/same', 8092, '/arm/engine.log', 'private-key')
    assert source == untouched
    assert cfg['env']['STRATA_PIPED_MTP_OBSERVE'] == '1'
    assert cfg['env']['STRATA_PIPED_ASYNC_SAMPLE'] == a.async_sample
    assert cfg['env']['STRATA_PIPED_TARGET_HEAD'] == a.target_head
    assert 'private-key' not in json.dumps(base.safe_config(cfg))
    args = forwarded(a)
    assert args.count('--arms') == 1 and args[args.index('--arms') + 1] == arm
    with tempfile.TemporaryDirectory(prefix='p3-offline-self-test-') as d:
        folder = Path(d)
        log = folder / 'engine.log'
        log.write_text('strata serve: STRATA_PIPED_MTP_OBSERVE: 8 independent slot event sets; '
                       'source-stream body/head-return/tail intervals, no packing or scheduling changes\n')
        assert base.startup_facts(log)['piped_mtp_observe_startup'] == [8]
        log.write_text('strata serve: STRATA_PIPED_MTP_OBSERVE: disabled\n')
        try:
            base.startup_facts(log)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Missing observation startup was accepted')
        manifest = {'same_binary_sha256': 'a' * 64, 'harness_sha256': hashes, 'sequence': [arm]}
        base.save(folder / 'manifest.json', manifest)
        saved = json.loads((folder / 'manifest.json').read_text())
        assert saved['diagnostic_only'] and saved['variables'] == {}
        assert saved['entrypoint_sha256'] == hashes[ENTRY]
    assert before == {p.name: base.sha(p) for p in HERE.iterdir() if p.is_file()}
    assert callable(observation.parse_text)
    print(json.dumps({'self_test': 'PASS', 'network_gpu_or_inference': False,
                      'copied_sources_unchanged': True, 'wrapper_frozen': True, 'diagnostic_manifest': True}))


def main(argv=None):
    a = arguments(argv)
    if a.self_test:
        self_test(a)
        return 0
    if not a.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_NETWORK_NO_GPU', 'experiment_type': 'diagnostic',
                          'diagnostic_only': True, 'entrypoint': ENTRY, 'suites': a.suites.split(','),
                          'single_arm': 'on' if a.async_sample == '1' else 'off', 'port': a.port,
                          'fixed_p1_target_head': a.target_head, 'fixed_p2_async_sampling': a.async_sample,
                          'fixed_observation': '1', 'four_explicit_sha256_required_for_execute': True,
                          'limitations': LIMITATIONS}, indent=2))
        return 0
    observation = install(a)
    old_argv = sys.argv
    try:
        sys.argv = [str(HERE / 'runner_async_sample.py'), *forwarded(a)]
        base.main()
    finally:
        sys.argv = old_argv
    # Offline only, after the inherited runner has released its owned model. No extra suite or comparison runs.
    manifest = json.loads((a.out / 'manifest.json').read_text())
    for row in manifest['arms']:
        arm = a.out / row['tag']
        report = observation.parse_text((arm / 'engine.log').read_text(errors='replace'), slots=8)
        report.update(diagnostic_only=True, source_log_sha256=base.sha(arm / 'engine.log'),
                      parser_sha256=manifest['harness_sha256']['parse_piped_mtp_observe.py'])
        base.save(arm / 'p3-observation.json', report)
    print('P3 DIAGNOSTIC COMPLETE; compare C8 output compatibility separately; no optimization gain inferred', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
