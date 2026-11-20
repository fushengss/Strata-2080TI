#!/usr/bin/env python3
"""Run a separate, untimed PARK_TRACE diagnostic through the frozen v7 runner.

Only a private /dev/shm copy receives the diagnostic flag. Production config is
never written; runner owns only its isolated server. No API key is printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path('/srv/bench100k-20261108-prefill-interleave')
BASE = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
EXPECTED = 'f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--snapshot-overwrite', action='store_true',
                    help='Use frozen v4 and enable its synchronous K/V overwrite experiment')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps({'execute': False, 'arms': ['on'],
                          'suites': ['admission', 'byield'],
                          'diagnostic_flag': {'STRATA_PARK_TRACE': '1',
                                              'STRATA_PARK_OVERWRITE': '1' if args.snapshot_overwrite else '0'},
                          'scope': 'Attribution only; not a performance comparison.'}))
        return 0
    raw = BASE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Production config changed'
    assert not args.out.exists(), 'Output already exists'
    cfg = json.loads(raw)
    cfg.setdefault('env', {})['STRATA_PARK_TRACE'] = '1'
    cfg['env']['STRATA_PARK_OVERWRITE'] = '1' if args.snapshot_overwrite else '0'
    binary = '/srv/Strata-decode-head-20261107/engine/strata.prefill-interleave' + ('4' if args.snapshot_overwrite else '3')
    binary_sha = ('92822929f1f60ecd80c64ce04fa47a44a6d789b0e4ca028ef5be5e39fecab20c'
                  if args.snapshot_overwrite else '45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6')
    fd, name = tempfile.mkstemp(prefix='strata-park-trace-base-', suffix='.json', dir='/dev/shm')
    child = None
    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()  # runner handles SIGTERM and cleans its owned server
        raise KeyboardInterrupt('diagnostic interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write('\n')
        digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        command = [sys.executable, str(ROOT/'harness-v7/runner_interleave.py'), '--execute',
                   '--arms', 'on', '--suites', 'admission,byield', '--config', name,
                   '--exe', binary, '--engine-sha256', binary_sha,
                   '--library-sha256', 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661',
                   '--server-sha256', '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9',
                   '--config-sha256', digest, '--out', str(args.out)]
        child = subprocess.Popen(command)
        return child.wait()
    finally:
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=150)
        finally:
            Path(name).unlink(missing_ok=True)
            assert hashlib.sha256(BASE.read_bytes()).hexdigest() == EXPECTED, 'Production config changed'

if __name__ == '__main__':
    raise SystemExit(main())
