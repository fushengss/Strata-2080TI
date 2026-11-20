#!/usr/bin/env python3
"""CPU fake-path qualification/backup/release gates; never executes deployment."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('deploy_reviewed', HERE/'deploy_concurrent_prefill.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

def check(value, message):
    if not value:
        raise RuntimeError(message)

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def rejects(call, label):
    try:
        call()
    except (RuntimeError, FileNotFoundError):
        return
    raise RuntimeError('Accepted negative: '+label)

def main():
    freeze = HERE/'incremental-harness-v1-freeze.json'; frozen = d.read(freeze)
    checks = ['off_on', 'same_suites', 'same_planned_suite_sequence', 'all_suites_succeeded',
              'only_mode_config_differs', 'overlap_flag_off_on', 'fixed_existing_paths',
              'explicit_overlap_only_profile', 'park_overwrite_fixed_off', 'overlap_flag_off_on_explicit',
              'both_runs_finished', 'all_planned_suites_completed', 'registered_completed_arms',
              'same_binary_sha256_equal', 'adjacent_library_sha256_equal', 'base_config_sha256_equal',
              'server_sha256_equal', 'harness_sha256_equal', 'calibration_sha256_equal']
    with tempfile.TemporaryDirectory(prefix='deployment-review-') as temporary:
        root = Path(temporary); run = root/'fake-run'; run.mkdir()
        shutil.copytree(HERE/'incremental-harness-v1', run/'harness')
        shutil.copyfile(HERE/'calibration.json', run/'calibration.json')
        suites = sorted(d.PACKAGES['incremental-harness-v1']['suites']); codes = dict.fromkeys(suites, 0)
        arms = [{'tag': '01-on', 'arm': 'on', 'suite_returncodes': codes},
                {'tag': '02-off', 'arm': 'off', 'suite_returncodes': codes}]
        manifest = {'same_binary_sha256': d.ENGINE, 'adjacent_library_sha256': d.LIBRARY,
                    'base_config_sha256': d.CONFIG, 'server_sha256': d.SERVER,
                    'calibration_sha256': d.CALIBRATION, 'harness_sha256': frozen['files_sha256'],
                    'experiment_profile': d.PROFILE, 'fixed_env': d.FIXED,
                    'variables': {'STRATA_PIPED_CONCURRENT_MTP': {'off': '0', 'on': '1'}},
                    'suites': suites, 'sequence': ['on', 'off'], 'arms': arms,
                    'ended_epoch': 100., 'original_config_unchanged': True, 'remaining_gpu_pids': [],
                    'comparisons': [{'baseline': '02-off', 'candidate': '01-on', 'returncode': 0}]}
        save(run/'manifest.json', manifest)
        metas = {}
        for arm in arms:
            flags = {**d.FIXED, 'STRATA_PIPED_CONCURRENT_MTP': '1' if arm['arm'] == 'on' else '0'}
            meta = {**arm, 'start_epoch': 1., 'ready_epoch': 2., 'finished_epoch': 90.,
                    'engine_sha256': d.ENGINE, 'adjacent_library_sha256': d.LIBRARY, 'config': {'env': flags},
                    'actual_engine': [{'engine_sha256': d.ENGINE, 'library_sha256': d.LIBRARY,
                                       'expected_library_mapped': True, 'flags': flags}],
                    'ready_status': {'loaded': True, 'concurrency': {'requested': 8, 'serving': 8}},
                    'startup_facts': {'batch_slot_devices': list(range(8)), 'all_experts_resident': True,
                                      'all_eight_have_3_80_gib_own_buffers': True},
                    'interleave_suite_evidence': {s: {'all_checks_passed': True} for s in suites},
                    'config_except_variables_log_key_sha256': 'a'*64}
            metas[arm['tag']] = meta; save(run/arm['tag']/'metadata.json', meta)
        comparison = {'identity_checks': dict.fromkeys(checks, True), 'suite_checks': dict.fromkeys(suites, True),
                      'all_checks_passed': True, 'suites': {s: ({'comparison_eligible': True, 'all_greedy_outputs_identical': True}
                          if s == 'prefill' else {'all_checks_passed': True}) for s in suites}}
        cp = run/'comparisons/02-off-vs-01-on/comparison.json'; save(cp, comparison)
        check(d.validate_qualification(run, freeze)['profile'] == d.PROFILE, 'Complete fake pair failed')
        total = 0
        for field, value in [('ended_epoch', None), ('ended_epoch', float('nan')), ('failure', {'kind': 'interrupted'}),
                             ('remaining_gpu_pids', [123]), ('original_config_unchanged', False), ('server_sha256', 'wrong'),
                             ('calibration_sha256', 'wrong'), ('experiment_profile', 'old'), ('sequence', ['off','on']),
                             ('suites', suites[:-1]), ('comparisons', []), ('arms', arms[:1])]:
            bad = copy.deepcopy(manifest); bad[field] = value; save(run/'manifest.json', bad)
            rejects(lambda: d.validate_qualification(run, freeze), 'parent '+field); total += 1
        save(run/'manifest.json', manifest)
        base = metas['01-on']; mp = run/'01-on/metadata.json'
        mutations = [lambda m: m.pop('finished_epoch'),
                     lambda m: m['suite_returncodes'].update(incremental=2),
                     lambda m: m.update(suite_returncodes=dict(reversed(list(m['suite_returncodes'].items())))),
                     lambda m: m['actual_engine'][0]['flags'].update(STRATA_PARK_OVERWRITE='1'),
                     lambda m: m['actual_engine'][0].update(expected_library_mapped=False),
                     lambda m: m['ready_status']['concurrency'].update(serving=7),
                     lambda m: m['startup_facts'].update(all_eight_have_3_80_gib_own_buffers=False),
                     lambda m: m['interleave_suite_evidence']['primary'].update(all_checks_passed=False),
                     lambda m: m.update(config_except_variables_log_key_sha256='b'*64)]
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(base); mutate(bad); save(mp, bad)
            rejects(lambda: d.validate_qualification(run, freeze), 'arm '+str(i)); total += 1
        save(mp, base)
        mutations = [lambda c: c['identity_checks'].pop('both_runs_finished'),
                     lambda c: c['identity_checks'].update(registered_completed_arms=False),
                     lambda c: c['suite_checks'].update(primary=False),
                     lambda c: c['suites']['prefill'].update(comparison_eligible=False),
                     lambda c: c['suites']['primary'].update(all_checks_passed=False)]
        for i, mutate in enumerate(mutations):
            bad = copy.deepcopy(comparison); mutate(bad); save(cp, bad)
            rejects(lambda: d.validate_qualification(run, freeze), 'comparison '+str(i)); total += 1
        save(cp, comparison)
        changed = run/'harness/README.txt'; payload = changed.read_bytes(); changed.write_bytes(payload+b'wrong')
        rejects(lambda: d.validate_qualification(run, freeze), 'harness mutation'); total += 1; changed.write_bytes(payload)
        check(d.validate_qualification(run, freeze)['profile'] == d.PROFILE, 'Restored fake pair failed')
        # The deployment union must come from two independent complete pairs, not partial arms.
        union_tested = False
        if d.PACKAGES['incremental-harness-v2']['freeze'] is not None:
            freeze2 = HERE/'incremental-harness-v2-freeze.json'; frozen2 = d.read(freeze2)
            run2 = root/'fake-incremental'; shutil.copytree(run, run2)
            shutil.rmtree(run2/'harness'); shutil.copytree(HERE/'incremental-harness-v2', run2/'harness')
            second = copy.deepcopy(manifest); second['harness_sha256'] = frozen2['files_sha256']; second['suites'] = ['incremental']
            for registered in second['arms']:
                registered['suite_returncodes'] = {'incremental': 0}
                meta = copy.deepcopy(metas[registered['tag']]); meta['suite_returncodes'] = {'incremental': 0}
                meta['interleave_suite_evidence'] = {'incremental': {'all_checks_passed': True}}
                save(run2/registered['tag']/'metadata.json', meta)
            save(run2/'manifest.json', second)
            c2 = copy.deepcopy(comparison); c2['suite_checks'] = {'incremental': True}; c2['suites'] = {'incremental': {'all_checks_passed': True}}
            save(run2/'comparisons/02-off-vs-01-on/comparison.json', c2)
            check(len(d.validate_qualifications([run, run2], [freeze, freeze2])) == 2, 'Complete two-pair union failed')
            rejects(lambda: d.validate_qualifications([run, run], [freeze, freeze]), 'duplicate run'); total += 1
            rejects(lambda: d.validate_qualifications([run], [freeze]), 'missing incremental pair'); total += 1
            for registered in second['arms']:
                path = run2/registered['tag']/'metadata.json'; meta = d.read(path)
                meta['config_except_variables_log_key_sha256'] = 'b'*64; save(path, meta)
            rejects(lambda: d.validate_qualifications([run, run2], [freeze, freeze2]), 'cross-run fixed config mismatch'); total += 1
            union_tested = True
        # Only synthetic local bytes: no real private config is read.
        original_config = d.CONFIG; secret = b'{"api_key":"SYNTHETIC_TEST_ONLY"}\n'
        d.CONFIG = hashlib.sha256(secret).hexdigest()
        try:
            backup = root/'private/backup.json'; backup.parent.mkdir(mode=0o755)
            d.private_backup(backup, secret); d.private_backup(backup, secret)
            check(stat.S_IMODE(backup.stat().st_mode) == 0o600 and stat.S_IMODE(backup.parent.stat().st_mode) == 0o700,
                  'Private backup permissions differ')
            rejects(lambda: d.private_backup(backup, b'wrong'), 'wrong rollback bytes'); total += 1
        finally:
            d.CONFIG = original_config
        src = root/'source.bin'; src.write_bytes(b'synthetic immutable release')
        dst = root/'release.bin'; d.install_release(src, dst, d.sha(src)); d.install_release(src, dst, d.sha(src))
        check(stat.S_IMODE(dst.stat().st_mode) == 0o555, 'Release not readonly')
        rejects(lambda: d.install_release(src, dst, '0'*64), 'release overwrite'); total += 1
        print(json.dumps({'passed': True, 'negative_cases': total, 'complete_fake_pair': True,
                          'two_complete_pair_union_tested': union_tested,
                          'private_backup_and_immutable_release': True, 'real_network': 0, 'gpu_operations': 0,
                          'deployment_execute_calls': 0}))

if __name__ == '__main__':
    main()
