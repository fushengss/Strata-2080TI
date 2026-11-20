#!/usr/bin/env python3
"""Record the completed measured review; no service or config operations."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/srv/bench100k-20261107-mtp-head')
paths = {
    'evaluation': ROOT / 'abba-final/evaluation.json',
    'compatibility': ROOT / 'default-off-compatibility/default-off-compatibility.json',
    'source_manifest': ROOT / 'source-manifest-mtp2.json',
    'warm_append_review': ROOT / 'warm-append-review.json',
}
digest = {k + '_sha256': hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
e, c, w = (json.loads(paths[k].read_text()) for k in ('evaluation', 'compatibility', 'warm_append_review'))
assert e['predeclared_quantitative_and_correctness_gates_pass'] and not e['failed_or_missing_gates']
assert c['default_off_compatible_for_tested_cases'] and c['complete_comparison']
assert w['warm_append_review_pass'] and w['evaluation_sha256'] == digest['evaluation_sha256']
record = {
    'created_epoch': time.time(), 'deploy': True, 'warm_append_review_pass': True,
    'source_commit': 'b4c45a6cc8167ac61407f07c1171b6b4995eec10',
    'source_commit_matches_tested_production_manifest': True,
    **digest,
    'reason': 'Complete ABBA, internal parity, integration, warm/append review and original-binary default-off compatibility passed.',
    'scope': 'Greedy solo PW2 MTP head on GPU3,5; main head stays GPU2,4; original prefill layers, buffers and eight slots retained.',
    'local_tp_decision': 'Do not deploy the decomposed TP2/4 expert prototype: slower for T1-3 and not production-bitwise.',
    'remaining_after_installation': 'Restart and verify exact running executable/library, both head routes, resident experts, eight slots and synthetic output.',
}
(ROOT / 'deployment-decision.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
