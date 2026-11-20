#!/usr/bin/env python3
"""Short, serial microbench; no service/config writes. Needs an explicit GPU-test window.

Uses real Q5 model weights, four-token activations, cap=40, and cross-process raw
output comparison. Timings during live traffic are NOT standalone GPU performance.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--phase', choices=['validate', 'abba'], default='validate')
p.add_argument('--iters', type=int, default=3)
p.add_argument('--rounds', type=int, default=1)
a = p.parse_args()
out = Path('/srv/bench100k-20261107-kernels')
c = json.loads(Path('/srv/Strata/strata-orca-q5_k_m.57.json').read_text())
model = '/models/orcarouter/Qwen3.8-Flash-Next-Uncensored-Q5/Qwen3.8-Flash-Next-Uncensored-Q5_K_M-00002-of-00003.gguf'

def activity():
    try:
        req = urllib.request.Request('http://127.0.0.1:8080/v1/status', headers={'Authorization': 'Bearer ' + c['api_key']})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)['activity']
    except Exception as e:
        return {'error': str(type(e).__name__)}

results = []
shapes = [(10, 4)] if a.phase == 'validate' else [(40, 1), (20, 2), (10, 4)]
arms = ['base', 'v2k', 'kq'] if a.phase == 'validate' else ['base', 'v2k', 'v2k', 'base', 'base', 'kq', 'kq', 'base']
for g, t in shapes:
    reference = {}
    for i, arm in enumerate(arms):
        name = f'{a.phase}-g{g}-t{t}-{i}-{arm}'
        env = {**os.environ, **c.get('env', {}), 'CUDA_VISIBLE_DEVICES': '0',
               'BENCH_ROUNDS': str(a.rounds), 'BENCH_CAP': '40', 'BENCH_SEED': '17',
               'BENCH_DUMP_PREFIX': str(out / name),
               'STRATA_EXPERT_V2K': str(int(arm == 'v2k')), 'STRATA_KQ_SPLIT': str(int(arm == 'kq'))}
        before, start = activity(), time.time()
        r = subprocess.run([str(out / 'native_expert_bench_t4'), model, '3,6', str(g), str(t), '0', '0', str(a.iters)],
                           env=env, capture_output=True, text=True, timeout=120)
        (out / f'{name}.log').write_text(r.stdout + r.stderr)
        row = {'name': name, 'arm': arm, 'groups': g, 'tokens_per_group': t,
               'activity_before': before, 'activity_after': activity(), 'wall_s': time.time()-start,
               'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr, 'outputs': {}}
        for layer in (3, 6):
            path = out / f'{name}.l{layer}.g{g}.t{t}.bin'
            if path.exists():
                data = path.read_bytes()
                if layer not in reference and arm == 'base':
                    reference[layer] = data
                row['outputs'][str(layer)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                                             'byte_equal_to_baseline': data == reference.get(layer)}
        results.append(row)
        (out / f'{a.phase}.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('stdout','stderr')}), flush=True)
        print('\n'.join(ln for ln in r.stdout.splitlines() if ln.startswith(('layer','finite_check','hash'))), flush=True)
        if r.returncode or len(row['outputs']) != 2 or not all(v['byte_equal_to_baseline'] for v in row['outputs'].values()):
            raise SystemExit('correctness check failed; stop before further timing')
