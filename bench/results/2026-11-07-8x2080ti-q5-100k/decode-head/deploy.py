#!/usr/bin/env python3
"""Install the measured head-only candidate after all recorded checks pass.

Run only on the maintenance host; keeps credentials in the existing config and
its private remote backup. Does not start/stop a service or run benchmarks.
"""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path('/srv/bench100k-20261107-decode-head')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
BUILD = Path('/srv/Strata-decode-head-20261107/engine')
RELEASE = Path('/srv/Strata-0157/engine/releases/decode-head-20261107')
BASE_SHA = '1edb2161bf111ad04b38e25aa024a99decd5440c9f1489c4b63457baf8ecd48a'
FILES = {
    'strata.final': ('strata', '372da958cd4b1ba25b2ab42a7e92480c4cb31973994ebc8d8355e5644766fc22'),
    'libstrata_kq.so': ('libstrata_kq.so', 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    evaluation = json.loads((ROOT / 'abba-final/evaluation.json').read_text())
    compatibility_path = ROOT / 'default-off-compatibility/default-off-compatibility.json'
    compatibility = json.loads(compatibility_path.read_text())
    decision = json.loads((ROOT / 'deployment-decision.json').read_text())
    assert evaluation['predeclared_quantitative_and_correctness_gates_pass']
    assert compatibility['default_off_compatible_for_tested_cases']
    assert decision['deploy'] is True and decision['warm_append_review_pass'] is True
    assert decision['evaluation_sha256'] == sha((ROOT / 'abba-final/evaluation.json').read_bytes())
    assert decision['compatibility_sha256'] == sha(compatibility_path.read_bytes())
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    original = CONFIG.read_bytes()
    assert sha(original) == BASE_SHA, 'Live config changed since maintenance began'
    config = json.loads(original)
    original_exe = Path(config['exe'])
    original_exe_sha = sha(original_exe.read_bytes())
    RELEASE.mkdir(parents=True, exist_ok=True)
    installed = {}
    for source, (name, digest) in FILES.items():
        src, dst = BUILD / source, RELEASE / name
        assert sha(src.read_bytes()) == digest, source + ' hash changed'
        if dst.exists():
            assert sha(dst.read_bytes()) == digest, str(dst) + ' already differs'
        else:
            shutil.copy2(src, dst)
        assert sha(dst.read_bytes()) == digest
        installed[name] = {'path': str(dst), 'sha256': digest}
    backup = CONFIG.with_name(CONFIG.name + '.bak-decode-head-' + time.strftime('%Y%m%d-%H%M%S'))
    with backup.open('xb') as f:
        os.chmod(backup, 0o600)
        f.write(original)
    config['exe'] = str(RELEASE / 'strata')
    config.setdefault('env', {})['STRATA_DECODE_HEAD_GPU'] = '2,4'
    changed = json.dumps(config, ensure_ascii=False, indent=2).encode() + b'\n'
    restored = json.loads(changed)
    restored['exe'] = str(original_exe)
    restored['env'].pop('STRATA_DECODE_HEAD_GPU')
    assert restored == json.loads(original), 'Unexpected config modification'
    tmp = CONFIG.with_name(CONFIG.name + '.decode-head-new')
    with tmp.open('xb') as f:
        os.chmod(tmp, stat.S_IMODE(CONFIG.stat().st_mode))
        f.write(changed)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG)
    assert sha(original_exe.read_bytes()) == original_exe_sha
    record = {
        'time_epoch': time.time(), 'source_commit': '235e2021',
        'config_sha256_before': BASE_SHA, 'config_sha256_after': sha(CONFIG.read_bytes()),
        'private_backup_remote': str(backup), 'installed': installed,
        'changed_config_paths': ['/exe', '/env/STRATA_DECODE_HEAD_GPU'],
        'flag': '2,4', 'original_executable': str(original_exe),
        'original_executable_sha256': original_exe_sha,
        'prefill_and_service_options_unchanged': True,
        'rollback': 'cp -- ' + str(backup) + ' ' + str(CONFIG) + ' && bash /srv/Strata/restart57.sh',
    }
    (ROOT / 'deployment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
