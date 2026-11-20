#!/usr/bin/env python3
"""Offline check of per-read zero decode expert-pool dispatch evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BEGIN = re.compile(r'^strata piped concurrent MTP: begin read_id=(\d+) ', re.M)
SOURCE = re.compile(r'^strata resident read source: read_id=(\d+) staged=([01]) lookahead_idle=([01]) dispatch_delta=(-?\d+)$', re.M)


def check(text, arm):
    ids = [int(x) for x in BEGIN.findall(text)]
    rows = [dict(zip(('read_id', 'staged', 'lookahead_idle', 'dispatch_delta'), map(int, row)))
            for row in SOURCE.findall(text)]
    source_ids = [r['read_id'] for r in rows]
    checks = {'all_begun_reads_have_source_evidence': sorted(ids) == sorted(source_ids),
              'source_evidence_unique': len(set(source_ids)) == len(source_ids),
              'all_source_lines_parsed': len(rows) == sum(line.startswith('strata resident read source:') for line in text.splitlines()),
              'all_decode_dispatch_deltas_zero': all(r['dispatch_delta'] == 0 for r in rows),
              'lookahead_always_idle_at_admission': all(r['lookahead_idle'] == 1 for r in rows),
              'disabled_has_no_reads': arm != 'off' or not ids}
    return {'passed': all(checks.values()), 'checks': checks, 'reads': rows,
            'scope': 'Expert pool dispatch entries counted from before reader launch until after reader join, before final drain. Zero is required; this is not proof that background I/O workers did no work.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    assert not a.out.exists()
    results = {}
    for log in sorted(a.run.glob('*/engine.log')):
        meta = json.loads((log.parent / 'metadata.json').read_text())
        text = log.read_text(errors='replace')
        results[log.parent.name] = {**check(text, meta['arm']),
                                   'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()}
    assert results, 'No arm engine logs'
    all_passed = all(r['passed'] for r in results.values())
    a.out.write_text(json.dumps({'all_checks_passed': all_passed, 'arms': results}, indent=2) + '\n')
    print(json.dumps({'all_checks_passed': all_passed, 'read_counts': {k: len(v['reads']) for k,v in results.items()}}))
    return 0 if all_passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
