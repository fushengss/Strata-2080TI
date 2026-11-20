#!/usr/bin/env python3
"""Read running-process/config evidence and make one synthetic greedy smoke request."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from bench_client import Bench, save_json
from client import check

ROOT = Path('/srv/bench100k-20261107-decode-head')
CONFIG = Path('/srv/Strata/strata-orca-q5_k_m.57.json')


def main():
    cfg = json.loads(CONFIG.read_text())
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    expected = deployment['installed']['strata']
    expected_kq = deployment['installed']['libstrata_kq.so']
    processes = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            exe = (p / 'exe').resolve(strict=True)
            if str(exe) != expected['path']:
                continue
            argv = (p / 'cmdline').read_bytes().decode().split('\0')
            env = dict(x.split('=', 1) for x in (p / 'environ').read_bytes().decode().split('\0') if '=' in x)
            processes.append({'pid': int(p.name), 'exe': str(exe),
                              'sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
                              'head_flag': env.get('STRATA_DECODE_HEAD_GPU'),
                              'tested_kq_library_mapped': expected_kq['path'] in (p / 'maps').read_text(),
                              'argv': argv})
        except (OSError, UnicodeError):
            pass
    headers = {'Authorization': 'Bearer ' + cfg['api_key']}
    request = urllib.request.Request('http://127.0.0.1:8080/v1/status', headers=headers)
    with urllib.request.urlopen(request, timeout=10) as f:
        state = json.load(f)
    concurrency = state.get('concurrency')
    eight_slots = concurrency == 8 or isinstance(concurrency, dict) and concurrency.get('serving') == 8 and concurrency.get('requested') == 8
    args = SimpleNamespace(config=str(CONFIG), port=8080, out=str(ROOT / 'production-smoke'))
    bench = Bench(args)
    prompt = '[Decode head correctness short-greedy] Write a Python generator that yields prime numbers below n, then provide tests.'
    result = check(bench, bench.request('short-greedy', [{'role': 'user', 'content': prompt}], 192, 'correctness'))
    reference = json.loads((ROOT / 'abba-final/02-on/short-greedy.json').read_text())
    same = all(result[k] == reference[k] for k in ('request_sha256', 'output_text', 'reasoning_text', 'finish_reason'))
    gov = sorted(p.read_text().strip() for p in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    log = Path(cfg['log'])
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size - 2000000))
        tail = f.read().decode(errors='replace')
    startup = tail.rsplit('engine started:', 1)[-1]
    selected = [bench.safe(line) for line in startup.splitlines()
                if any(x in line for x in ('decode head:', 'decode head on', 'split ownbuf', 'KQ MMQ', 'K-quant', 'all experts', '3072', 'prefill chunks'))]
    checks = {
        'loaded_eight_slots': state.get('loaded') is True and eight_slots,
        'one_exact_tested_engine': len(processes) == 1 and processes[0]['sha256'] == expected['sha256'],
        'head_flag_enabled': len(processes) == 1 and processes[0]['head_flag'] == '2,4',
        'tested_kq_library_loaded': len(processes) == 1 and processes[0]['tested_kq_library_mapped'] and hashlib.sha256(Path(expected_kq['path']).read_bytes()).hexdigest() == expected_kq['sha256'],
        'two_shard_startup_confirmed': 'decode head on CUDA2+4:' in startup,
        'all_experts_still_resident': startup.count('3072 of its 3072 profiled pairs') == 7 and 'pre-filled 3072 of 3072 slots' in startup,
        'config_matches_deployment': hashlib.sha256(CONFIG.read_bytes()).hexdigest() == deployment['config_sha256_after'],
        'original_prefill_service_options_preserved': deployment['prefill_and_service_options_unchanged'],
        'synthetic_request_matches_abba': same,
        'actual_head_offload_seen': result['decode_head_offload_windows'] > 0,
        'no_smoke_request_or_cuda_errors': result['error'] is None and not result['checks']['cuda_error_lines'],
        'cpu_governors_preserved': len(gov) == 48 and set(gov) == {'schedutil'},
    }
    record = {'time_epoch': time.time(), 'checks': checks, 'pass': all(checks.values()),
              'status': {k: state.get(k) for k in ('loaded', 'engine', 'concurrency', 'activity')},
              'processes': processes, 'startup_evidence': selected,
              'smoke_head_windows': result['decode_head_offload_windows'],
              'smoke_output_sha256': result['output_sha256'], 'governors': gov}
    save_json(ROOT / 'production-verification.json', record)
    print(json.dumps({k: record[k] for k in ('checks', 'pass', 'status', 'smoke_head_windows')}, indent=2))
    return 0 if record['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
