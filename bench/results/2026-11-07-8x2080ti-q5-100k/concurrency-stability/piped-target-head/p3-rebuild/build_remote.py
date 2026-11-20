#!/usr/bin/env python3
"""Build frozen P3 instrumentation only between GPU benchmark arms/runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path('/srv/bench100k-20261107-piped-target-head/p3-rebuild')
SRC = Path('/srv/Strata-decode-head-20261107')
DEST = SRC / 'engine/strata.piped-mtp-observe2'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not DEST.exists(), 'Frozen binary already exists'
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    assert not pids.strip(), 'Do not build during GPU benchmark or production'
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    assert digest(ROOT / 'source.tar') == manifest['archive_sha256']
    with tarfile.open(ROOT / 'source.tar') as archive:
        assert {m.name for m in archive.getmembers() if m.isfile()} == set(manifest['files'])
        for member in archive.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
        archive.extractall(SRC, filter='data')
    full = json.loads((ROOT / 'full-source-manifest.json').read_text())
    assert full['commit'] == manifest['source_commit']
    mismatches = []
    for name, expected in full['files'].items():
        data = (SRC / name).read_bytes()
        # Tracked ps1 checkout has CRLF; compare canonical git bytes.
        if name.endswith('.ps1'):
            data = data.replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected:
            mismatches.append(name)
    assert not mismatches, 'Source mismatch: ' + repr(mismatches)
    # Archive mtimes can precede newer object files. Content changed: refresh all input mtimes.
    for name in manifest['files']:
        os.utime(SRC / name, None)
    preserved = {name: digest(SRC / ('engine/' + name)) for name in
                 ('strata.piped-target-head1', 'strata.piped-async-sample1')}
    result = {'source_commit': full['commit'], 'start_epoch': time.time(),
              'verified_source_files': len(full['files']), 'preserved_binaries': preserved}
    with (ROOT / 'build.log').open('w') as log:
        run = subprocess.run(['nice', '-n', '19', '/srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake',
                              '--build', 'build', '--target', 'strata', '-j16'],
                             cwd=SRC, stdout=log, stderr=subprocess.STDOUT)
    result['build_returncode'] = run.returncode
    if run.returncode == 0:
        for relative in ('build/CMakeFiles/strata.dir/src/program/generate.cpp.o', 'build/CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o', 'build/CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'):
            assert (SRC / relative).stat().st_mtime >= result['start_epoch'], 'Required object not rebuilt: ' + relative
        result['required_objects_rebuilt'] = True
        shutil.copy2(SRC / 'build/strata', DEST)
        result.update(engine_path=str(DEST), engine_sha256=digest(DEST),
                      library_sha256=digest(DEST.parent / 'libstrata_kq.so'))
    assert preserved == {name: digest(SRC / ('engine/' + name)) for name in preserved}
    result['end_epoch'] = time.time()
    (ROOT / 'build-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return run.returncode

if __name__ == '__main__':
    raise SystemExit(main())
