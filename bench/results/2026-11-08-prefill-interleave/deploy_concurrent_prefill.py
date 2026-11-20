#!/usr/bin/env python3
"""Install a qualified frozen engine during maintenance; default is dry-run.

Does not start or stop services. The caller restores the production service and
verifies its actual binary after this atomic configuration change.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

ROOT = Path('/srv/bench100k-20261108-prefill-interleave')
BASE = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SOURCE = Path('/srv/Strata-decode-head-20261107/engine/strata.prefill-interleave3')
RELEASE = Path('/srv/Strata-0157/engine/releases/concurrent-prefill-20261108')
ENGINE = '45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6'
LIBRARY = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
CONFIG = 'f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd'
SERVER = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
SERVER_PATH = Path('/srv/Strata-0157/serve/server.py')
CALIBRATION = 'b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd'
FREEZE = '1a47c446562cb61f21e3a07d8906262c6f406288cb39b45f53c2c1ad128e7eb3'
PROFILE = 'concurrent-mtp-only-park-overwrite-fixed-off'
FIXED = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
         'STRATA_PW_WITH_BATCH': '1', 'STRATA_PARK_OVERWRITE': '0',
         'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_BATCH_SELF_COMMIT': '0',
         'STRATA_PIPED_MTP_HEAD': '1', 'STRATA_PIPED_TARGET_HEAD': '1', 'STRATA_PIPED_ASYNC_SAMPLE': '0'}
REQUIRED = {'incremental', 'admission', 'byield', 'budget', 'prefill',
            'primary', 'mixed100k', 'correctness', 'c8'}
PACKAGES = {
    'incremental-harness-v1': {'freeze': FREEZE, 'suites': REQUIRED - {'incremental'}},
    'incremental-harness-v2': {'freeze': '75a15bb36c851d994b70eb594f49e836ef26e0dcfab06185af22e7119133fbee',
                             'suites': {'incremental'}},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def read(path):
    return json.loads(path.read_bytes())

def epoch(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0

def all_true(values):
    return isinstance(values, dict) and bool(values) and all(v is True for v in values.values())

def validate_qualification(run, freeze_path):
    """Read-only gates. No inference, subprocess, deployment or performance verdict."""
    freeze_sha = sha(freeze_path)
    recognized = [(name, item) for name, item in PACKAGES.items() if item['freeze'] == freeze_sha]
    require(len(recognized) == 1, 'Frozen harness manifest not independently approved')
    package, definition = recognized[0]; required = definition['suites']
    frozen = read(freeze_path)
    files = frozen['files_sha256']
    require(frozen.get('package') == package and bool(files) and frozen['experiment_profile'] == PROFILE,
            'Wrong harness package/profile/file set')
    require(all(Path(name).name == name for name in files), 'Invalid harness path')
    require({p.name for p in (run/'harness').iterdir()} == set(files), 'Run harness file set differs')
    for name, expected in files.items():
        path = run/'harness'/name
        require(path.is_file() and not path.is_symlink() and sha(path) == expected, 'Run harness file differs: '+name)
    manifest = read(run/'manifest.json')
    identities = {'same_binary_sha256': ENGINE, 'adjacent_library_sha256': LIBRARY,
                  'base_config_sha256': CONFIG, 'server_sha256': SERVER,
                  'calibration_sha256': CALIBRATION, 'harness_sha256': files,
                  'experiment_profile': PROFILE, 'fixed_env': FIXED,
                  'variables': {'STRATA_PIPED_CONCURRENT_MTP': {'off': '0', 'on': '1'}}}
    require(all(manifest.get(k) == v for k, v in identities.items()), 'Run identities/profile differ')
    require(sha(run/'calibration.json') == CALIBRATION, 'Run calibration differs')
    require(not manifest.get('failure') and epoch(manifest.get('ended_epoch')) and
            manifest.get('original_config_unchanged') is True and manifest.get('remaining_gpu_pids') == [],
            'Parent run is incomplete, failed, or not cleaned up')
    suites, sequence, arms = manifest.get('suites', []), manifest.get('sequence', []), manifest.get('arms', [])
    require(len(suites) == len(required) and set(suites) == required, 'Planned suite set differs from approved package scope')
    require(sequence in (['off', 'on'], ['on', 'off']) and len(arms) == 2, 'Exactly two completed arms required')
    paths = [run/'manifest.json', run/'calibration.json', freeze_path]
    named, normalized = {}, []
    for index, (arm, registered) in enumerate(zip(sequence, arms), 1):
        tag = f'{index:02d}-{arm}'
        require(registered.get('tag') == tag and registered.get('arm') == arm, 'Arm registration/order differs')
        path = run/tag/'metadata.json'; meta = read(path); paths.append(path)
        require(meta.get('tag') == tag and meta.get('arm') == arm, 'Arm metadata identity differs')
        require(all(epoch(meta.get(k)) for k in ('start_epoch', 'ready_epoch', 'finished_epoch')) and
                meta['start_epoch'] <= meta['ready_epoch'] <= meta['finished_epoch'] <= manifest['ended_epoch'],
                'Arm is not completely finished')
        codes = meta.get('suite_returncodes', {})
        require(list(codes) == suites and all(type(v) is int and v == 0 for v in codes.values()) and
                registered.get('suite_returncodes') == codes, 'Own planned suite successes/order/registration differ')
        require(meta.get('engine_sha256') == ENGINE and meta.get('adjacent_library_sha256') == LIBRARY,
                'Arm binary/library differs')
        cfg = meta.get('config', {}); flags = {**FIXED, 'STRATA_PIPED_CONCURRENT_MTP': '1' if arm == 'on' else '0'}
        require(all(cfg.get('env', {}).get(k) == v for k, v in flags.items()), 'Arm config flags differ')
        actual = meta.get('actual_engine', [])
        require(len(actual) == 1 and actual[0].get('engine_sha256') == ENGINE and
                actual[0].get('library_sha256') == LIBRARY and actual[0].get('expected_library_mapped') is True and
                all(actual[0].get('flags', {}).get(k) == v for k, v in flags.items()), 'Actual tested engine differs')
        state = meta.get('ready_status', {}); facts = meta.get('startup_facts', {})
        require(state.get('loaded') is True and all(state.get('concurrency', {}).get(k) == 8 for k in ('requested', 'serving')) and
                facts.get('batch_slot_devices') == list(range(8)) and facts.get('all_experts_resident') is True and
                facts.get('all_eight_have_3_80_gib_own_buffers') is True, 'Tested slots/residency/prefill buffers differ')
        require(set(meta.get('interleave_suite_evidence', {})) == required and
                all(v.get('all_checks_passed') is True for v in meta['interleave_suite_evidence'].values()),
                'Actual reader/source suite proofs failed or missing')
        identity = meta.get('config_except_variables_log_key_sha256', '')
        require(isinstance(identity, str) and re.fullmatch(r'[0-9a-f]{64}', identity), 'Missing full config identity')
        normalized.append(identity); named[arm] = tag
    require(normalized[0] == normalized[1], 'Other tested configuration changed')
    expected_comparisons = [{'baseline': named['off'], 'candidate': named['on'], 'returncode': 0}]
    require(manifest.get('comparisons') == expected_comparisons, 'Completed paired comparison not registered')
    comparison_path = run/'comparisons'/(named['off']+'-vs-'+named['on'])/'comparison.json'
    require(list((run/'comparisons').glob('*/comparison.json')) == [comparison_path], 'Unexpected comparison set')
    comparison = read(comparison_path); paths.append(comparison_path)
    expected_keys = {'off_on', 'same_suites', 'same_planned_suite_sequence', 'all_suites_succeeded',
                     'only_mode_config_differs', 'overlap_flag_off_on', 'fixed_existing_paths',
                     'explicit_overlap_only_profile', 'park_overwrite_fixed_off', 'overlap_flag_off_on_explicit',
                     'both_runs_finished', 'all_planned_suites_completed', 'registered_completed_arms',
                     'same_binary_sha256_equal', 'adjacent_library_sha256_equal', 'base_config_sha256_equal',
                     'server_sha256_equal', 'harness_sha256_equal', 'calibration_sha256_equal'}
    checks = comparison.get('identity_checks', {})
    require(set(checks) == expected_keys and all_true(checks), 'Comparison identity checks missing or failed')
    require(comparison.get('all_checks_passed') is True and set(comparison.get('suite_checks', {})) == required and
            all_true(comparison['suite_checks']) and set(comparison.get('suites', {})) == required,
            'Own complete paired suite comparison not passed')
    for suite, result in comparison['suites'].items():
        require((result.get('comparison_eligible') is True and result.get('all_greedy_outputs_identical') is True)
                if suite == 'prefill' else result.get('all_checks_passed') is True,
                'Underlying suite comparison failed: '+suite)
    return {'comparison_path': comparison_path, 'reviewed_artifacts_sha256': {str(p): sha(p) for p in paths},
            'harness_freeze_sha256': freeze_sha, 'profile': PROFILE, 'package': package,
            'suites': sorted(required), 'full_normalized_config_sha256': normalized[0]}

def validate_qualifications(runs, freezes):
    require(len(runs) == len(freezes) == 2 and len({p.resolve() for p in runs}) == 2,
            'Two distinct complete paired runs and their approved freezes are required')
    results = [validate_qualification(run, frozen) for run, frozen in zip(runs, freezes)]
    require({r['package'] for r in results} == set(PACKAGES), 'One complete v1 core run and one complete v2 incremental run required')
    sets = [set(r['suites']) for r in results]
    require(not sets[0] & sets[1] and sets[0] | sets[1] == REQUIRED, 'Suite union must be exactly nine without duplicates')
    require(results[0]['full_normalized_config_sha256'] == results[1]['full_normalized_config_sha256'],
            'Full normalized engine configuration differs between qualification runs')
    return results

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def private_backup(path, payload):
    require(hashlib.sha256(payload).hexdigest() == CONFIG, 'Rollback payload differs')
    require(not path.parent.is_symlink() and not path.is_symlink(), 'Private backup must not be a symlink')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True); path.parent.chmod(0o700)
    if path.exists():
        require(sha(path) == CONFIG, 'Existing rollback backup differs'); path.chmod(0o600)
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    require(sha(path) == CONFIG, 'Rollback backup verification failed')
    sync_dir(path.parent)

def install_release(source, destination, expected):
    """Publish a fully verified file once; never overwrite an existing release."""
    require(not destination.is_symlink(), 'Release must not be a symlink')
    if destination.exists():
        require(sha(destination) == expected, 'Refuse to overwrite a different release'); return
    fd, name = tempfile.mkstemp(prefix='.release-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as handle, source.open('rb') as src:
            for data in iter(lambda: src.read(1024*1024), b''):
                handle.write(data)
            handle.flush(); os.fchmod(handle.fileno(), 0o555); os.fsync(handle.fileno())
        require(sha(name) == expected, 'Copied release differs')
        os.link(name, destination)  # Exclusive publication; refuses a racing replacement.
        sync_dir(destination.parent)
    finally:
        Path(name).unlink(missing_ok=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--qualified-run', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--freeze', type=Path, action='append', help='One reviewed freeze per --qualified-run, in matching order')
    a = ap.parse_args()
    if not a.execute:
        print(json.dumps({'execute': False, 'release': str(RELEASE),
                          'changes': {'exe': str(RELEASE/'strata'),
                                      'env.STRATA_PIPED_CONCURRENT_MTP': '1'},
                          'required_suites': sorted(REQUIRED),
                          'requires': 'Complete v1 core8 pair plus complete v2 incremental pair, reviewed freezes, manual performance review, empty GPUs and stopped production.'}))
        return 0
    require(not a.out.exists(), 'Deployment receipt already exists')
    qualifications = validate_qualifications(a.qualified_run, a.freeze or [])
    require(sha(BASE) == CONFIG and sha(SOURCE) == ENGINE and sha(SERVER_PATH) == SERVER, 'Runtime source/config/server differs')
    require(sha(SOURCE.parent/'libstrata_kq.so') == LIBRARY, 'Adjacent library differs')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip(), 'GPUs still occupied')
    for port in (8080, 8092):
        with socket.socket() as sock:
            sock.settimeout(1)
            require(sock.connect_ex(('127.0.0.1', port)) != 0, 'Service still listening')
    original = BASE.read_bytes(); require(hashlib.sha256(original).hexdigest() == CONFIG, 'Config changed before preparation')
    old = json.loads(original)
    require(old.get('api_key'), 'A public listener requires its existing API key')
    new = json.loads(original)
    new['exe'] = str(RELEASE/'strata')
    new.setdefault('env', {})['STRATA_PIPED_CONCURRENT_MTP'] = '1'
    # Prove no prefill geometry, model, authentication or server setting changed.
    normalized = json.loads(json.dumps(new))
    normalized['exe'] = old['exe']
    if 'STRATA_PIPED_CONCURRENT_MTP' in old.get('env', {}):
        normalized['env']['STRATA_PIPED_CONCURRENT_MTP'] = old['env']['STRATA_PIPED_CONCURRENT_MTP']
    else:
        normalized['env'].pop('STRATA_PIPED_CONCURRENT_MTP')
    require(normalized == old, 'Unrelated configuration changed')
    backup = ROOT/'baseline-private'/'pre-concurrent-prefill.json'
    private_backup(backup, original)
    require(not RELEASE.is_symlink(), 'Release directory must not be a symlink')
    RELEASE.mkdir(parents=True, exist_ok=True)
    for name, src, expected in [('strata', SOURCE, ENGINE),
                                ('libstrata_kq.so', SOURCE.parent/'libstrata_kq.so', LIBRARY)]:
        dst = RELEASE/name
        install_release(src, dst, expected)
        require(sha(dst) == expected, 'Published release differs')
    # Prove the receipt location writable before changing the production config.
    a.out.mkdir(parents=True)
    (a.out/'pending.json').write_text(json.dumps({'config_replacement_pending': True,
        'original_config_sha256': CONFIG, 'private_rollback_config': str(backup)})+'\n')
    fd, temp = tempfile.mkstemp(prefix='.concurrent-prefill-', suffix='.json', dir=BASE.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(new, handle, ensure_ascii=False, indent=2); handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        require(sha(BASE) == CONFIG, 'Configuration changed during deployment')
        os.replace(temp, BASE)
        sync_dir(BASE.parent)
    finally:
        Path(temp).unlink(missing_ok=True)
    receipt = {'time_epoch': time.time(), 'release': str(RELEASE),
               'engine_sha256': ENGINE, 'library_sha256': LIBRARY,
               'original_config_sha256': CONFIG, 'new_config_sha256': sha(BASE),
               'private_rollback_config': str(backup),
               'qualified_runs': [str(p) for p in a.qualified_run],
               'qualification_receipts': [{**q, 'comparison_path': str(q['comparison_path']),
                                           'comparison_sha256': sha(q['comparison_path'])} for q in qualifications],
               'server_sha256': SERVER, 'calibration_sha256': CALIBRATION,
               'only_exe_and_overlap_flag_changed': True, 'service_restart_pending': True,
               'manual_performance_acceptance': 'Separate operator decision; this helper validates correctness/identity only.'}
    (a.out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
