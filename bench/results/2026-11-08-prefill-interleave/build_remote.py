#!/usr/bin/env python3
"""Build a content-bound source archive without reusing stale changed objects.

Production may stay online. Never run this alongside a timed :8092 experiment.
The caller supplies a new artifact directory containing source.tar and manifest.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

SRC = Path('/srv/Strata-decode-head-20261107')
CM = '/srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake'


def digest(path):
    data = path.read_bytes()
    if path.suffix == '.ps1':
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--name', required=True)
    args = p.parse_args()
    assert args.bundle.is_absolute() and args.bundle.is_dir()
    assert args.name.startswith('strata.prefill-interleave') and '/' not in args.name
    dest = SRC / 'engine' / args.name
    assert not dest.exists(), 'Refuse to overwrite a frozen binary'
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        path = Path(table)
        if path.exists():
            assert not any(int(f[1].rsplit(':', 1)[1], 16) == 8092 and f[3] == '0A'
                           for line in path.read_text().splitlines()[1:] if (f := line.split())), 'Timed test port is occupied'
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    assert digest(args.bundle / 'source.tar') == manifest['archive_sha256']
    changed = [name for name, expected in manifest['files'].items()
               if not (SRC / name).is_file() or digest(SRC / name) != expected]
    unchanged_times = {name: ((SRC / name).stat().st_atime_ns, (SRC / name).stat().st_mtime_ns)
                       for name in manifest['files'] if name not in changed}
    with tarfile.open(args.bundle / 'source.tar') as archive:
        assert {m.name for m in archive.getmembers() if m.isfile()} == set(manifest['files'])
        archive.extractall(SRC, filter='data')
    assert all(digest(SRC / name) == expected for name, expected in manifest['files'].items())
    # Do not rebuild every untouched CUDA kernel merely because a new git archive
    # gives all files the new commit timestamp. Content changes still invalidate
    # their own objects and every header-dependent object below.
    for name, ns in unchanged_times.items():
        os.utime(SRC / name, ns=ns)
    # Git archives preserve commit time, which can precede a newer but incompatible
    # object. Refresh every changed file/header after extraction, not just .cpp files.
    for name in changed:
        os.utime(SRC / name, None)
    forced = manifest.get('force_fresh_sources', [])
    for name in forced:
        assert name in manifest['files'] and name.endswith('.cpp')
        os.utime(SRC / name, None)
    preserved = {name: digest(SRC / 'engine' / name) for name in ('strata.piped-target-head1', 'strata.piped-async-sample1')}
    result = {'source_commit': manifest['source_commit'], 'verified_source_files': len(manifest['files']),
              'changed_inputs_touched': changed, 'force_fresh_sources': forced,
              'preserved_binaries': preserved, 'start_epoch': time.time()}
    with (args.bundle / 'build.log').open('w') as log:
        run = subprocess.run(['nice', '-n', '19', CM, '--build', 'build', '--target', 'strata', '-j16'],
                             cwd=SRC, stdout=log, stderr=subprocess.STDOUT)
    result['returncode'] = run.returncode
    if run.returncode == 0:
        fresh = {}
        for relative in manifest['required_fresh_objects']:
            obj = SRC / relative
            fresh[relative] = obj.stat().st_mtime
            assert fresh[relative] >= result['start_epoch'], 'Required dependent object was not rebuilt: ' + relative
        result['rebuilt_object_mtimes'] = fresh
        shutil.copy2(SRC / 'build/strata', dest)
        result.update(engine_path=str(dest), engine_sha256=digest(dest),
                      library_sha256=digest(dest.parent / 'libstrata_kq.so'))
    assert preserved == {name: digest(SRC / 'engine' / name) for name in preserved}
    result['end_epoch'] = time.time()
    (args.bundle / 'build-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
