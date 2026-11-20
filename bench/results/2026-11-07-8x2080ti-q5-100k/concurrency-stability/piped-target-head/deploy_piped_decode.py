#!/usr/bin/env python3
"""Install one verified piped-decode engine during maintenance; dry-run by default.

Usage (replace values with the candidate and SHA selected after validation):
  python3 deploy_piped_decode.py --candidate /absolute/path/to/strata.CANDIDATE \
    --engine-sha256 SHA256 --target-head 1 --async-sample 0
  # Add --check for read-only host/hash checks; add --execute to install.
  python3 deploy_piped_decode.py --self-test  # local temporary fixtures only

Never starts/stops services or changes GPU clocks/power. Only CONFIG.exe and the
specified two environment entries change. Server bytes remain untouched. A caught
failure after config replacement atomically restores the original config bytes
(mode 0600). SIGINT/SIGTERM also enter rollback; SIGKILL/power loss cannot run Python
cleanup. Original config is durably backed up before replacement. A failed install
may leave an unreferenced immutable release; inspect it instead of overwriting it.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import tempfile
import time

ROOT = Path('/srv/bench100k-20261107-piped-target-head')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SERVER = Path('/srv/Strata-0157/serve/server.py')
RELEASE = Path('/srv/Strata-0157/engine/releases/piped-decode-20261107')
RECEIPT = ROOT / 'deployment-receipt.json'
CONFIG_SHA = 'f5376cc83215fc54b46d20094373f20a41b93daa8bf07262d2ac67dc13297220'
SERVER_SHA = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
LIBRARY_SHA = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
FLAG_KEYS = ('STRATA_PIPED_TARGET_HEAD', 'STRATA_PIPED_ASYNC_SAMPLE')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, data, permissions):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            os.fchmod(handle.fileno(), permissions)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def stopped():
    # Linux all-address listener table, including non-loopback and IPv6 binds.
    for name in ('/proc/net/tcp', '/proc/net/tcp6'):
        path = Path(name)
        if name.endswith('tcp6') and not path.exists():
            continue
        require(path.is_file(), 'Linux TCP listener table unavailable')
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            require(len(fields) >= 4, 'Cannot parse TCP listener table')
            port = int(fields[1].rsplit(':', 1)[1], 16)
            require(not (fields[3] == '0A' and port in (8080, 8092)),
                    'Port 8080 or 8092 still listens; no process was stopped')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
    require(result.returncode == 0, 'GPU process query failed')
    require(not result.stdout.strip(), 'GPUs are occupied; no process was stopped')


def parse_config(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate configuration key')
            result[key] = value
        return result
    try:
        config = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError('Configuration is not valid JSON') from error
    require(isinstance(config, dict) and isinstance(config.get('env'), dict), 'Invalid configuration shape')
    return config


def assert_preserved(old, new, flags):
    require(set(flags) == set(FLAG_KEYS) and all(v in ('0', '1') for v in flags.values()),
            'Only the two explicit binary decode flags are allowed')
    require(set(old) == set(new), 'Unexpected configuration key added/removed')
    for key in old:
        if key not in ('env', 'exe'):
            require(old[key] == new[key], 'A protected configuration field changed')
    for key in old['env'].keys() | new['env'].keys():
        if key not in FLAG_KEYS:
            require(key in old['env'] and key in new['env'] and old['env'][key] == new['env'][key],
                    'A protected environment field changed')
    require(all(new['env'].get(k) == v for k, v in flags.items()), 'Selected decode flags differ')
    require(new['exe'] == str(RELEASE / 'strata'), 'Unexpected engine destination')
    require(isinstance(old.get('api_key'), str) and bool(old['api_key']) and new['api_key'] == old['api_key'],
            'Authentication must be preserved')


def new_config(data, flags):
    old = parse_config(data)
    require(old.get('parallel') == 8 and old.get('layer_split') == '6,12,18,24,30,36,42',
            'Existing slot/layer layout differs')
    require(old['env'].get('STRATA_DECODE_HEAD_GPU') == '2,4' and
            old['env'].get('STRATA_MTP_HEAD_GPU') == '3,5', 'Existing head placement differs')
    new = copy.deepcopy(old)
    new['exe'] = str(RELEASE / 'strata')
    new['env'].update(flags)
    assert_preserved(old, new, flags)
    encoded = (json.dumps(new, ensure_ascii=False, indent=2) + '\n').encode()
    assert_preserved(old, parse_config(encoded), flags)
    return encoded


def prepare(candidate, engine_sha, flags):
    require(candidate.is_absolute() and candidate not in (CONFIG, SERVER, RELEASE / 'strata') and
            candidate.name != 'libstrata_kq.so', 'Candidate needs a separate absolute engine path')
    expected = {CONFIG: CONFIG_SHA, SERVER: SERVER_SHA, candidate: engine_sha,
                candidate.parent / 'libstrata_kq.so': LIBRARY_SHA}
    contents = {}
    for path, digest in expected.items():
        require(path.is_file() and not path.is_symlink(), 'A frozen input is not a regular file')
        contents[path] = path.read_bytes()
        require(sha(contents[path]) == digest, 'A frozen input SHA256 differs: ' + str(path))
    require(os.access(candidate, os.X_OK), 'Candidate engine is not executable')
    require(ROOT.is_dir() and RELEASE.parent.is_dir(), 'Deployment parent directory is missing')
    require(not os.path.lexists(RELEASE), 'Release path already exists; will not reuse or overwrite it')
    require(not os.path.lexists(RECEIPT), 'Deployment receipt already exists; inspect it before another install')
    encoded = new_config(contents[CONFIG], flags)
    return contents, encoded


def install(candidate, engine_sha, flags, contents, encoded):
    stopped()
    require(not os.path.lexists(RELEASE) and not os.path.lexists(RECEIPT), 'A deployment destination appeared')
    # Input recheck before creating artifacts, followed by another original-file check before commit.
    require(all(path.read_bytes() == data for path, data in contents.items()), 'Frozen inputs changed during preparation')
    original_mode = mode(CONFIG)
    backup = Path(tempfile.mkdtemp(prefix='rollback-piped-decode-', dir=ROOT))
    os.chmod(backup, 0o700)
    atomic(backup / CONFIG.name, contents[CONFIG], 0o600)
    atomic(backup / SERVER.name, contents[SERVER], 0o600)
    sync_dir(ROOT)
    targets = {RELEASE / 'strata': (contents[candidate], 0o755),
               RELEASE / 'libstrata_kq.so': (contents[candidate.parent / 'libstrata_kq.so'], 0o644)}
    config_attempted = False
    receipt_attempted = False
    try:
        RELEASE.mkdir(mode=0o755)
        sync_dir(RELEASE.parent)
        for path, (data, permissions) in targets.items():
            atomic(path, data, permissions)
            require(path.read_bytes() == data and mode(path) == permissions, 'Installed release verification failed')
        stopped()
        require(CONFIG.read_bytes() == contents[CONFIG] and SERVER.read_bytes() == contents[SERVER],
                'Original configuration or server changed during installation')
        require(not os.path.lexists(RECEIPT), 'Receipt appeared during installation')
        config_attempted = True
        atomic(CONFIG, encoded, 0o600)
        require(CONFIG.read_bytes() == encoded and mode(CONFIG) == 0o600, 'Installed configuration verification failed')
        require(SERVER.read_bytes() == contents[SERVER], 'Server changed during configuration commit')
        assert_preserved(parse_config(contents[CONFIG]), parse_config(CONFIG.read_bytes()), flags)
        receipt = {'mode': 'INSTALLED_SERVICE_NOT_STARTED', 'time_epoch': time.time(),
                   'backup_directory': str(backup), 'original_config_mode': original_mode,
                   'config_mode': 0o600, 'rollback_config_mode': 0o600,
                   'candidate': str(candidate), 'release': str(RELEASE),
                   'original_sha256': {str(CONFIG): CONFIG_SHA, str(SERVER): SERVER_SHA},
                   'installed_sha256': {str(CONFIG): sha(encoded), str(SERVER): SERVER_SHA,
                                        str(RELEASE / 'strata'): engine_sha,
                                        str(RELEASE / 'libstrata_kq.so'): LIBRARY_SHA},
                   'flags': flags, 'only_config_fields_changed': ['exe', 'env.' + FLAG_KEYS[0], 'env.' + FLAG_KEYS[1]],
                   'all_other_fields_preserved': True, 'api_key_unchanged': True,
                   'all_prefill_env_and_layer_split_unchanged': True, 'server_unchanged': True,
                   'gpu_clock_and_power_untouched': True, 'service_started': False}
        receipt_attempted = True
        atomic(RECEIPT, (json.dumps(receipt, indent=2) + '\n').encode(), 0o644)
        return receipt
    except BaseException as install_error:
        # No server rewrite: it is an immutable prerequisite, not part of this transaction.
        if config_attempted:
            try:
                atomic(CONFIG, contents[CONFIG], 0o600)
                require(CONFIG.read_bytes() == contents[CONFIG] and mode(CONFIG) == 0o600, 'Restored config differs')
            except BaseException as restore_error:
                raise RuntimeError('Install failed; CONFIG restoration failed; private original: ' + str(backup)) from restore_error
        if receipt_attempted:
            RECEIPT.unlink(missing_ok=True)
            sync_dir(ROOT)
        raise RuntimeError('Install failed; original configuration retained/restored; private backup: ' + str(backup)) from install_error


def self_test():
    # No /srv reads, sockets, GPU queries or network; temporary fake files only.
    global ROOT, CONFIG, SERVER, RELEASE, RECEIPT, CONFIG_SHA, SERVER_SHA, LIBRARY_SHA, stopped, atomic
    names = ('ROOT', 'CONFIG', 'SERVER', 'RELEASE', 'RECEIPT', 'CONFIG_SHA', 'SERVER_SHA', 'LIBRARY_SHA', 'stopped', 'atomic')
    saved = {name: globals()[name] for name in names}
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix='piped-decode-installer-check-') as directory:
            base = Path(directory); ROOT = base / 'audit'; ROOT.mkdir()
            CONFIG = base / 'config.json'; SERVER = base / 'server.py'
            RELEASE = base / 'releases' / 'piped-decode-20261107'; RELEASE.parent.mkdir()
            RECEIPT = ROOT / 'deployment-receipt.json'
            candidate = base / 'strata.candidate'; candidate.write_bytes(b'fake engine'); candidate.chmod(0o755)
            library = base / 'libstrata_kq.so'; library.write_bytes(b'fake library')
            old = {'exe': '/old/strata', 'api_key': 'DO_NOT_LEAK_OFFLINE_SECRET', 'parallel': 8,
                   'layer_split': '6,12,18,24,30,36,42', 'reasoning_budget_engine_cap': True,
                   'env': {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
                           'PREFILL_TEST': {'nested': [1, 'unchanged']}}}
            CONFIG.write_text(json.dumps(old)); SERVER.write_bytes(b'unchanged server')
            CONFIG_SHA = sha(CONFIG.read_bytes()); SERVER_SHA = sha(SERVER.read_bytes()); LIBRARY_SHA = sha(library.read_bytes())
            digest = sha(candidate.read_bytes()); flags = dict(zip(FLAG_KEYS, ('1', '0')))
            stopped = lambda: None
            contents, encoded = prepare(candidate, digest, flags)
            require(CONFIG.read_bytes() == contents[CONFIG] and not RELEASE.exists(), 'Prepare mutated files')
            results.append('read_only_prepare_preserves_files')
            changed = parse_config(encoded); changed['env']['PREFILL_TEST'] = 'wrong'
            try:
                assert_preserved(old, changed, flags)
            except RuntimeError:
                results.append('reject_protected_prefill_change')
            else:
                raise RuntimeError('Protected field mutation was accepted')
            try:
                prepare(candidate, '0' * 64, flags)
            except RuntimeError:
                results.append('reject_candidate_hash_mismatch')
            else:
                raise RuntimeError('Wrong engine digest was accepted')
            real_atomic = atomic
            def fail_receipt(path, data, permissions):
                if path == RECEIPT:
                    raise OSError('injected post-config receipt failure')
                return real_atomic(path, data, permissions)
            atomic = fail_receipt
            try:
                install(candidate, digest, flags, contents, encoded)
            except RuntimeError:
                require(CONFIG.read_bytes() == contents[CONFIG] and mode(CONFIG) == 0o600,
                        'Failure did not restore private original config')
                require(SERVER.read_bytes() == contents[SERVER] and not RECEIPT.exists(), 'Failure changed server or left success receipt')
                results.append('post_config_failure_rolls_back_atomically')
            else:
                raise RuntimeError('Injected failure did not fail')
            atomic = real_atomic
            try:
                prepare(candidate, digest, flags)
            except RuntimeError:
                results.append('reject_existing_release')
            else:
                raise RuntimeError('Existing release was accepted')
            for path in RELEASE.iterdir(): path.unlink()
            RELEASE.rmdir()
            contents, encoded = prepare(candidate, digest, flags)
            receipt = install(candidate, digest, flags, contents, encoded)
            require('DO_NOT_LEAK_OFFLINE_SECRET' not in json.dumps(receipt), 'Receipt leaked secret')
            require(CONFIG.read_bytes() == encoded and SERVER.read_bytes() == contents[SERVER], 'Success bytes differ')
            require(mode(Path(receipt['backup_directory'])) == 0o700 and
                    mode(Path(receipt['backup_directory']) / CONFIG.name) == 0o600, 'Backup is not private')
            results += ['successful_install_preserves_server_and_protected_fields', 'receipt_contains_no_secret', 'backup_0700_config_0600']
    finally:
        globals().update(saved)
    print(json.dumps({'mode': 'OFFLINE_SELF_TEST', 'passed': True, 'checks': results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--check', action='store_true', help='read-only host, input SHA and configuration checks')
    actions.add_argument('--self-test', action='store_true')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--engine-sha256')
    parser.add_argument('--target-head', choices=('0', '1'))
    parser.add_argument('--async-sample', choices=('0', '1'))
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.candidate is None or args.engine_sha256 is None or args.target_head is None or args.async_sample is None:
        parser.error('--candidate, --engine-sha256, --target-head and --async-sample must be explicit')
    require(args.candidate.is_absolute(), 'Candidate path must be absolute')
    require(re.fullmatch('[0-9a-f]{64}', args.engine_sha256) is not None, 'Engine SHA256 must be 64 lowercase hex characters')
    flags = dict(zip(FLAG_KEYS, (args.target_head, args.async_sample)))
    plan = {'mode': 'DRY_RUN_NO_MUTATION', 'candidate': str(args.candidate), 'engine_sha256': args.engine_sha256,
            'release': str(RELEASE), 'flags': flags,
            'required_sha256': {'original_config': CONFIG_SHA, 'unchanged_server': SERVER_SHA, 'adjacent_library': LIBRARY_SHA},
            'checks_executed': False, 'service_started': False, 'gpu_clock_and_power_untouched': True}
    if not (args.execute or args.check):
        print(json.dumps(plan, indent=2)); return
    stopped()
    contents, encoded = prepare(args.candidate, args.engine_sha256, flags)
    if args.check:
        plan.update(mode='READ_ONLY_CHECK_PASSED', checks_executed=True, proposed_config_sha256=sha(encoded))
        print(json.dumps(plan, indent=2)); return
    def interrupted(signum, frame):
        raise InterruptedError('Deployment interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(json.dumps(install(args.candidate, args.engine_sha256, flags, contents, encoded), indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Configuration contents, subprocess output and traceback locals are never printed.
        print(json.dumps({'status': 'FAILED', 'error': str(error) if isinstance(error, RuntimeError) else type(error).__name__}))
        raise SystemExit(1)
