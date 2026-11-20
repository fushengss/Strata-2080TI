#!/usr/bin/env python3
"""Install the verified concurrency candidate during the authorized maintenance window.

Dry-run by default. Never starts/stops a process or prints configuration contents.
Both current files are backed up privately; failure restores their original bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

ROOT = Path('/srv/bench100k-20261107-concurrency-mtp')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
SERVER = Path('/srv/Strata-0157/serve/server.py')
NEW_SERVER = SERVER.with_name('server_concurrency_mtp_20261107.py')
CANDIDATE = Path('/srv/Strata-decode-head-20261107/engine/strata.piped-head1')
RELEASE = Path('/srv/Strata-0157/engine/releases/concurrent-mtp-20261107')
EXPECTED = {
    CONFIG: '93492533633510b37d8ec25a825a2c2313eb03dd94dfecbf2e8f84b4b6772be0',
    SERVER: 'd6d51fac600e60508360ba92876511c52e9706d55e6eedcff940f624d0f0a448',
    NEW_SERVER: '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9',
    CANDIDATE: 'c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6',
    CANDIDATE.parent / 'libstrata_kq.so': 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, data, mode):
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def unused(port):
    with socket.socket() as sock:
        sock.settimeout(.5)
        return sock.connect_ex(('127.0.0.1', port)) != 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--head', choices=('on', 'off'), default='on')
    args = parser.parse_args()
    flags = {'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1',
             'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1' if args.head == 'on' else '0'}
    if not args.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_MUTATION', 'flags': flags,
                          'reasoning_budget_engine_cap': True,
                          'release': str(RELEASE), 'guard': 'unchanged; not enabled'}))
        return
    require(unused(8080) and unused(8092), 'A service still listens; no process was stopped')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    require(not gpu.strip(), 'GPUs are occupied; no process was stopped')
    contents = {path: path.read_bytes() for path in EXPECTED}
    require(all(sha(contents[p]) == s for p, s in EXPECTED.items()), 'A frozen source or original file changed')
    old = json.loads(contents[CONFIG])
    require(old.get('parallel') == 8 and old.get('layer_split') == '6,12,18,24,30,36,42', 'Existing slot/layer layout differs')
    require(old.get('env', {}).get('STRATA_DECODE_HEAD_GPU') == '2,4' and
            old.get('env', {}).get('STRATA_MTP_HEAD_GPU') == '3,5', 'Existing heads differ')
    new = json.loads(contents[CONFIG])
    new['env'].update(flags)
    new['reasoning_budget_engine_cap'] = True
    new['exe'] = str(RELEASE / 'strata')
    require(new.get('api_key') == old.get('api_key') and bool(new.get('api_key')), 'Authentication must be preserved')
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    require(changed == {'env', 'exe', 'reasoning_budget_engine_cap'}, 'Unexpected top-level config change')
    require({key for key in old['env'].keys() | new['env'].keys() if old['env'].get(key) != new['env'].get(key)} == set(flags),
            'Unexpected environment change')
    modes = {p: p.stat().st_mode & 0o777 for p in (CONFIG, SERVER)}
    backup = ROOT / ('rollback-' + time.strftime('%Y%m%d-%H%M%S', time.gmtime()))
    backup.mkdir(mode=0o700)
    for path in (CONFIG, SERVER):
        atomic(backup / path.name, contents[path], 0o600)
    RELEASE.mkdir(parents=True, exist_ok=True)
    targets = {RELEASE / 'strata': (contents[CANDIDATE], 0o755),
               RELEASE / 'libstrata_kq.so': (contents[CANDIDATE.parent / 'libstrata_kq.so'], 0o644)}
    for path, (data, mode) in targets.items():
        if path.exists():
            require(sha(path.read_bytes()) == sha(data), 'Existing release differs; will not overwrite it')
        else:
            atomic(path, data, mode)
        require(sha(path.read_bytes()) == sha(data), 'Installed release bytes differ')
    require(os.access(RELEASE / 'strata', os.X_OK), 'Installed engine is not executable')
    encoded = (json.dumps(new, ensure_ascii=False, indent=2) + '\n').encode()
    receipt_path = ROOT / 'deployment-receipt.json'
    require(not receipt_path.exists(), 'Deployment receipt already exists; inspect it before another install')
    require(unused(8080) and unused(8092), 'A service started during preparation')
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(),
            'A GPU process started during preparation')
    require(all(p.read_bytes() == contents[p] for p in (CONFIG, SERVER)), 'Original service files changed during preparation')
    try:
        atomic(SERVER, contents[NEW_SERVER], modes[SERVER])
        atomic(CONFIG, encoded, modes[CONFIG])
        require(sha(SERVER.read_bytes()) == EXPECTED[NEW_SERVER] and CONFIG.read_bytes() == encoded,
                'Post-write verification failed')
        receipt = {'time_epoch': time.time(), 'backup_directory': str(backup), 'original_modes': {str(p): m for p, m in modes.items()},
                   'original_sha256': {str(p): EXPECTED[p] for p in (CONFIG, SERVER)},
                   'installed_sha256': {str(p): sha(p.read_bytes()) for p in (CONFIG, SERVER, *targets)},
                   'flags': flags, 'reasoning_budget_engine_cap': True, 'api_key_unchanged': True,
                   'prefill_config_and_layer_split_unchanged': True, 'service_started': False}
        atomic(receipt_path, (json.dumps(receipt, indent=2) + '\n').encode(), 0o644)
    except BaseException as install_error:
        restore_failures = []
        for path in (SERVER, CONFIG):
            try:
                atomic(path, contents[path], modes[path])
                require(path.read_bytes() == contents[path], 'Restored bytes differ')
            except BaseException as error:
                restore_failures.append({'path': str(path), 'kind': type(error).__name__})
        if restore_failures:
            raise RuntimeError('Install failed; restoration failures: ' + json.dumps(restore_failures)
                               + '; private originals: ' + str(backup)) from install_error
        raise
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
