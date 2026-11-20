#!/usr/bin/env python3
"""Prepare/install the exact measured MTP-head candidate. Default is read-only.

--apply installs artifacts and atomically changes only exe and STRATA_MTP_HEAD_GPU.
Never starts/stops services. Credentials stay in the existing config and a 0600
remote backup. deployment-decision.json must contain deploy=true,
warm_append_review_pass=true, source_commit, source_manifest_sha256,
evaluation_sha256 and compatibility_sha256, bound to the reviewed evidence.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path('/srv/bench100k-20261107-mtp-head')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
BUILD = Path('/srv/Strata-decode-head-20261107/engine')
RELEASE = Path('/srv/Strata-0157/engine/releases/mtp-head-20261107')
BASE_SHA = '4605e1ae5f9bc9dee6ecdee787f950464707e31abf06e9fbb61890897ce3e76a'
ORIGINAL_SHA = '372da958cd4b1ba25b2ab42a7e92480c4cb31973994ebc8d8355e5644766fc22'
ENGINE_SHA = '113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f'
LIB_SHA = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
SOURCE_MANIFEST_SHA = '316e7ca1fa73a492f6b572f63d954e46a641588ff55d87471f087fc6af71a9fb'
FILES = {'strata.mtp2': ('strata', ENGINE_SHA), 'libstrata_kq.so': ('libstrata_kq.so', LIB_SHA)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def atomic_private_json(path, value, mode=0o600):
    tmp = path.with_name(path.name + '.new-' + str(os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n'); f.flush(); os.fchmod(f.fileno(), mode); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()


def validate_gates(evaluation, compatibility, decision, source_commit, digests):
    require(re.fullmatch(r'[0-9a-f]{40}', source_commit) is not None, 'Supply the full reviewed source commit')
    require(evaluation.get('predeclared_quantitative_and_correctness_gates_pass') is True, 'Final ABBA gates have not passed')
    require(not evaluation.get('failed_or_missing_gates'), 'Final ABBA has failed/missing gates')
    manifest = evaluation.get('manifest', {})
    require(manifest.get('same_binary_sha256') == ENGINE_SHA and manifest.get('base_config_sha256') == BASE_SHA,
            'Evaluation is for a different candidate or base configuration')
    require(manifest.get('sequence') == ['off', 'on', 'on', 'off'] and manifest.get('repeats') == 3,
            'Deployment requires the complete three-repeat ABBA')
    require(manifest.get('flag') == 'STRATA_MTP_HEAD_GPU' and manifest.get('off_value') == '-1' and
            manifest.get('on_value') == '3,5', 'Evaluation used different MTP-head settings')
    require(compatibility.get('default_off_compatible_for_tested_cases') is True and
            compatibility.get('complete_comparison') is True, 'Original-binary compatibility has not passed completely')
    require(compatibility.get('expected_original_sha') == ORIGINAL_SHA and
            compatibility.get('expected_final_sha') == ENGINE_SHA and compatibility.get('found_final_off_arms') == 2,
            'Compatibility evidence does not cover both final off arms and expected binaries')
    require(decision.get('deploy') is True and decision.get('warm_append_review_pass') is True,
            'Explicit deployment and warm/append review decision is missing')
    require(decision.get('source_commit') == source_commit, 'Decision source commit does not match')
    for key in ('evaluation_sha256', 'compatibility_sha256', 'source_manifest_sha256'):
        require(decision.get(key) == digests[key], 'Decision does not bind the current ' + key)
    require(digests['source_manifest_sha256'] == SOURCE_MANIFEST_SHA, 'Not the tested mtp2 source manifest')


def changed_config(original):
    cfg = json.loads(original)
    require(cfg.get('env', {}).get('STRATA_DECODE_HEAD_GPU') == '2,4', 'Existing main head must remain on GPU2,4')
    require(cfg.get('env', {}).get('STRATA_MTP_HEAD_GPU') in (None, '', '-1'), 'Baseline MTP head is already enabled')
    updated = copy.deepcopy(cfg)
    updated['exe'] = str(RELEASE / 'strata')
    updated.setdefault('env', {})['STRATA_MTP_HEAD_GPU'] = '3,5'
    restored = copy.deepcopy(updated)
    restored['exe'] = cfg['exe']
    if 'STRATA_MTP_HEAD_GPU' in cfg.get('env', {}):
        restored['env']['STRATA_MTP_HEAD_GPU'] = cfg['env']['STRATA_MTP_HEAD_GPU']
    else:
        restored['env'].pop('STRATA_MTP_HEAD_GPU')
    require(restored == cfg, 'Unexpected configuration modification')
    return cfg, updated


def require_maintenance():
    require(not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                        text=True).strip(), 'GPU processes are still running')
    for port in (8080, 8091):
        with socket.socket() as s:
            s.settimeout(0.5)
            require(s.connect_ex(('127.0.0.1', port)) != 0, 'A server still listens on port ' + str(port))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--evaluation', type=Path, default=ROOT / 'abba-final/evaluation.json')
    p.add_argument('--compatibility', type=Path, default=ROOT / 'default-off-compatibility/default-off-compatibility.json')
    p.add_argument('--decision', type=Path, default=ROOT / 'deployment-decision.json')
    p.add_argument('--source-manifest', type=Path, default=ROOT / 'source-manifest-mtp2.json')
    p.add_argument('--apply', action='store_true', help='Apply after all gates; otherwise print a read-only installation preview')
    a = p.parse_args()
    digests = {'evaluation_sha256': sha(a.evaluation.read_bytes()),
               'compatibility_sha256': sha(a.compatibility.read_bytes()),
               'source_manifest_sha256': sha(a.source_manifest.read_bytes())}
    validate_gates(read_json(a.evaluation), read_json(a.compatibility), read_json(a.decision), a.source_commit, digests)
    require_maintenance()
    original = CONFIG.read_bytes()
    require(sha(original) == BASE_SHA, 'Live configuration changed since the tested baseline')
    config_mode = stat.S_IMODE(CONFIG.stat().st_mode)
    cfg, updated = changed_config(original)
    original_exe = Path(cfg['exe'])
    require(sha(original_exe.read_bytes()) == ORIGINAL_SHA, 'Original executable is not the tested deployed baseline')
    installed = {}
    for source, (name, digest) in FILES.items():
        require(sha((BUILD / source).read_bytes()) == digest, 'Build artifact changed: ' + source)
        dst = RELEASE / name
        require(not dst.exists() or sha(dst.read_bytes()) == digest, 'Existing release artifact differs: ' + name)
        installed[name] = {'path': str(dst), 'sha256': digest}
    preview = {'apply': a.apply, 'source_commit': a.source_commit, **digests,
               'installed': installed, 'changed_config_paths': ['/exe', '/env/STRATA_MTP_HEAD_GPU'],
               'preserved_main_head': '2,4', 'mtp_head': '3,5', 'service_operation': 'none'}
    if not a.apply:
        print(json.dumps(preview, indent=2)); return 0
    require(not (ROOT / 'deployment.json').exists(), 'Deployment receipt already exists; inspect it before retrying')
    RELEASE.mkdir(parents=True, exist_ok=True)
    for source, (name, digest) in FILES.items():
        src, dst = BUILD / source, RELEASE / name
        if not dst.exists():
            tmp = RELEASE / (name + '.new-' + str(os.getpid()))
            try:
                with tmp.open('xb') as out, src.open('rb') as inp:
                    shutil.copyfileobj(inp, out); out.flush(); os.fsync(out.fileno())
                os.chmod(tmp, 0o755 if name == 'strata' else 0o644)
                require(sha(tmp.read_bytes()) == digest, 'Artifact copy verification failed: ' + name)
                os.link(tmp, dst)  # atomically publish without overwriting an existing release
            finally:
                if tmp.exists(): tmp.unlink()
        require(sha(dst.read_bytes()) == digest, 'Installed artifact verification failed: ' + name)
    # Recheck immediately before the only live mutation. No service is stopped or restarted here.
    require_maintenance()
    require(CONFIG.read_bytes() == original, 'Configuration changed during artifact installation')
    backup = CONFIG.with_name(CONFIG.name + '.bak-mtp-head-' + time.strftime('%Y%m%d-%H%M%S'))
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(original); f.flush(); os.fsync(f.fileno())
    changed = json.dumps(updated, ensure_ascii=False, indent=2).encode() + b'\n'
    record = {'time_epoch': time.time(), **preview, 'decision_sha256': sha(a.decision.read_bytes()),
              'config_sha256_before': BASE_SHA, 'config_sha256_after': sha(changed),
              'private_backup_remote': str(backup), 'source_manifest_path': str(a.source_manifest),
              'original_executable': str(original_exe), 'original_executable_sha256': ORIGINAL_SHA,
              'prefill_and_service_options_unchanged': True, 'main_head_preserved': True,
              'rollback_copy_argv': ['cp', '--', str(backup), str(CONFIG)],
              'restart_argv': ['bash', '/srv/Strata/restart57.sh']}
    atomic_private_json(ROOT / 'deployment-prepared.json', record)
    atomic_private_json(CONFIG, updated, mode=config_mode)
    require(sha(CONFIG.read_bytes()) == record['config_sha256_after'], 'Installed config hash differs')
    require(sha(original_exe.read_bytes()) == ORIGINAL_SHA, 'Original executable changed')
    atomic_private_json(ROOT / 'deployment.json', record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
