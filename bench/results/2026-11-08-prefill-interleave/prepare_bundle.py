#!/usr/bin/env python3
"""Freeze committed engine sources and the required dependent-object rebuild gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path('/Users/fu/project/HAI/Strata-decode-head')


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--parent', default='ad3ac67c')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    commit = git('rev-parse', args.commit + '^{commit}').decode().strip()
    parent = git('rev-parse', args.parent + '^{commit}').decode().strip()
    args.out.mkdir(parents=True, exist_ok=False)
    archive = args.out / 'source.tar'
    with archive.open('wb') as handle:
        subprocess.run(['git', '-C', str(REPO), 'archive', '--format=tar', commit,
                        'CMakeLists.txt', 'src', 'include', 'tests'], stdout=handle, check=True)
    files = {}
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if member.isfile():
                data = source.extractfile(member).read()
                if member.name.endswith('.ps1'):
                    data = data.replace(b'\r\n', b'\n')
                files[member.name] = hashlib.sha256(data).hexdigest()
            else:
                assert member.isdir(), 'Unexpected non-regular archived source: ' + member.name
    manifest = {'source_commit': commit, 'parent_commit': parent,
                'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': files,
                'force_fresh_sources': ['src/program/generate.cpp', 'src/core/verify.cpp',
                                        'src/core/mtp.cpp', 'src/prefill/prefill.cpp',
                                        'src/core/expert_source.cpp',
                                        'src/core/conversation_snapshot.cpp',
                                        'src/core/conversation_state.cpp'],
                'required_fresh_objects': [
                    'build/CMakeFiles/strata.dir/src/program/generate.cpp.o',
                    'build/CMakeFiles/strata_engine.dir/src/core/verify.cpp.o',
                    'build/CMakeFiles/strata_engine.dir/src/core/mtp.cpp.o',
                    'build/CMakeFiles/strata_engine.dir/src/core/expert_source.cpp.o',
                    'build/CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o',
                    'build/CMakeFiles/strata_engine.dir/src/core/conversation_snapshot.cpp.o',
                    'build/CMakeFiles/strata_engine.dir/src/core/conversation_state.cpp.o']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out / 'change.patch').write_bytes(git('diff', '--binary', parent, commit, '--',
                                              'CMakeLists.txt', 'src', 'include', 'tests', 'docs'))
    print(json.dumps({'source_commit': commit, 'files': len(files), 'archive_bytes': archive.stat().st_size,
                      'archive_sha256': manifest['archive_sha256']}))


if __name__ == '__main__':
    main()
