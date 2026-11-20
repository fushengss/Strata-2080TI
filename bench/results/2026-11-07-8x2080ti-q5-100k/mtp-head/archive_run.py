#!/usr/bin/env python3
"""Archive synthetic benchmark evidence only, after scanning actual credentials."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/srv/bench100k-20261107-mtp-head')
ALLOWED = {'pilot35', 'abba-final', 'compatibility-original', 'default-off-compatibility', 'production-smoke'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', choices=sorted(ALLOWED))
    name = parser.parse_args().run
    source = ROOT / name
    key = json.loads(Path('/srv/Strata/strata-orca-q5_k_m.57.json').read_text())['api_key'].encode()
    assert key, 'Expected authenticated service'
    files = sorted(p for p in source.rglob('*') if p.is_file())
    assert files, 'No evidence found'
    hits = [str(p.relative_to(source)) for p in files if key in p.read_bytes()]
    assert not hits, 'Credential found in ' + repr(hits)
    archive = ROOT / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    record = {'file': archive.name, 'bytes': archive.stat().st_size,
              'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'remote': str(archive), 'file_count': len(files), 'credential_matches': 0}
    (ROOT / (name + '-archive.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
