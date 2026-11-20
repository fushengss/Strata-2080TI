#!/usr/bin/env python3
"""Verify the installed running candidate, then issue one fixed synthetic 192-token request.

Never starts/stops/reconfigures a service. Requires an idle server while the
maintenance owner still controls traffic. Persists only synthetic response data,
numeric engine/request evidence and selected startup facts, never the config,
environment, API key, arbitrary process arguments or production log contents.
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import bench_client
from bench_client import Bench, save_json
from client import check, ERROR_RE
from deploy import ROOT, CONFIG, ENGINE_SHA, LIB_SHA, ORIGINAL_SHA, require, changed_config, sha


def processes_for(expected, expected_lib):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            exe = (p / 'exe').resolve(strict=True)
            if str(exe) != expected['path']: continue
            env = dict(x.split('=', 1) for x in (p / 'environ').read_bytes().decode().split('\0') if '=' in x)
            argv = (p / 'cmdline').read_bytes().decode().split('\0')
            maps = (p / 'maps').read_text()
            mapped = {line.split(None, 5)[5].strip() for line in maps.splitlines() if len(line.split(None, 5)) == 6}
            found.append({'pid': int(p.name), 'exe': str(exe), 'sha256': sha(exe.read_bytes()),
                          'main_head_flag': env.get('STRATA_DECODE_HEAD_GPU'),
                          'mtp_head_flag': env.get('STRATA_MTP_HEAD_GPU'),
                          'tested_kq_library_mapped': expected_lib['path'] in mapped,
                          'batch8_argument': any(argv[i:i+2] == ['--batch', '8'] for i in range(len(argv)-1))})
        except (OSError, UnicodeError):
            pass
    return found


def startup_facts(log):
    # Read only in memory; retain booleans/counts, not production log text.
    with log.open('rb') as f:
        f.seek(max(0, log.stat().st_size - 4000000))
        tail = f.read().decode(errors='replace')
    startup = tail.rsplit('engine started:', 1)[-1]
    return {'main_head_two_shards': 'decode head on CUDA2+4:' in startup,
            'mtp_head_two_shards': 'MTP draft head on CUDA3+5:' in startup,
            'mtp_complete_logits_source': '106299 Q6_K subset rows, exact full logits returned to CUDA7' in startup,
            'resident_stage_count': startup.count('3072 of its 3072 profiled pairs'),
            'resident_primary_complete': 'pre-filled 3072 of 3072 slots' in startup,
            'batch_slot_devices': sorted(set(int(x) for x in re.findall(r'8 slot sessions on CUDA(\d+)', startup)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=ROOT / 'abba-final/02-on/short-greedy.json')
    p.add_argument('--out', type=Path, default=ROOT / 'production-smoke')
    a = p.parse_args()
    require(not a.out.exists() or not any(a.out.iterdir()), 'Smoke output must be new or empty')
    cfg = json.loads(CONFIG.read_text())
    deployment = json.loads((ROOT / 'deployment.json').read_text())
    expected, lib = deployment['installed']['strata'], deployment['installed']['libstrata_kq.so']
    require(expected['sha256'] == ENGINE_SHA and lib['sha256'] == LIB_SHA, 'Wrong deployment artifact receipt')
    require(sha(CONFIG.read_bytes()) == deployment['config_sha256_after'], 'Configuration differs from deployment')
    require(cfg['exe'] == expected['path'] and cfg.get('env', {}).get('STRATA_DECODE_HEAD_GPU') == '2,4' and
            cfg.get('env', {}).get('STRATA_MTP_HEAD_GPU') == '3,5', 'Installed flags or executable differ')
    # The private backup stays remote and is never copied into benchmark evidence.
    original = Path(deployment['private_backup_remote']).read_bytes()
    require(sha(original) == deployment['config_sha256_before'], 'Private baseline backup changed')
    _, recomputed = changed_config(original)
    require(recomputed == cfg, 'More than exe and MTP flag changed')
    processes = processes_for(expected, lib)
    require(len(processes) == 1 and processes[0]['sha256'] == ENGINE_SHA, 'Need exactly one tested engine process')
    require(processes[0]['tested_kq_library_mapped'] and sha(Path(lib['path']).read_bytes()) == LIB_SHA,
            'Tested KQ library is not loaded with the expected bytes')
    require(processes[0]['main_head_flag'] == '2,4' and processes[0]['mtp_head_flag'] == '3,5', 'Process flags differ')
    headers = {'Authorization': 'Bearer ' + cfg['api_key']}
    req = urllib.request.Request('http://127.0.0.1:8080/v1/status', headers=headers)
    with urllib.request.urlopen(req, timeout=10) as f: state = json.load(f)
    concurrency = state.get('concurrency')
    eight_slots = concurrency == 8 or isinstance(concurrency, dict) and concurrency.get('serving') == 8 and concurrency.get('requested') == 8
    require(state.get('loaded') is True and eight_slots and (state.get('activity') or {}).get('in_flight') == 0,
            'Server must be loaded with eight slots and idle')
    startup = startup_facts(Path(cfg['log']))
    reference = json.loads(a.reference.read_text())
    require(reference.get('name') == 'short-greedy' and reference.get('category') == 'correctness' and
            reference.get('max_tokens') == 192, 'Reference must be the non-warmup 192-token ABBA correctness case')
    # Only retain strict numeric performance/routing lines from this request. Drop all other production text.
    allowed = re.compile(r'^strata (?:serve: prompt \d+ tokens =|decode head: CUDA\d+, \d+ PW2 windows|MTP head: CUDA\d+\+\d+, \d+ draft steps)')
    def numeric_log(path, cursor):
        result = []
        if path is None or cursor is None: return result
        st = path.stat()
        with path.open('rb') as f:
            if st.st_ino == cursor[0] and st.st_size >= cursor[1]: f.seek(cursor[1])
            lines = f.read().decode(errors='replace').splitlines()
        for line in lines:
            if ERROR_RE.search(line): result.append('CUDA failure [details omitted from deployment evidence]')
            elif allowed.search(line): result.append(line)
        return result
    bench_client.read_log = numeric_log
    bench = Bench(SimpleNamespace(config=str(CONFIG), port=8080, out=str(a.out)))
    prompt = '[Decode head correctness short-greedy] Write a Python generator that yields prime numbers below n, then provide tests.'
    result = check(bench, bench.request('short-greedy', [{'role': 'user', 'content': prompt}], 192, 'correctness'))
    exact_fields = ('request_sha256', 'messages_sha256', 'output_text', 'reasoning_text', 'output_sha256',
                    'reasoning_sha256', 'combined_output_sha256', 'finish_reason')
    same = all(result.get(k) == reference.get(k) for k in exact_fields) and all(
        result.get('usage', {}).get(k) == reference.get('usage', {}).get(k) for k in ('prompt_tokens', 'completion_tokens'))
    gov = sorted(x.read_text().strip() for x in Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    checks = {'loaded_eight_slots': eight_slots and processes[0]['batch8_argument'],
              'one_exact_tested_engine': processes[0]['sha256'] == ENGINE_SHA,
              'main_head_preserved': processes[0]['main_head_flag'] == '2,4' and startup['main_head_two_shards'],
              'mtp_head_enabled': processes[0]['mtp_head_flag'] == '3,5' and startup['mtp_head_two_shards'] and startup['mtp_complete_logits_source'],
              'tested_kq_library_loaded': processes[0]['tested_kq_library_mapped'],
              'all_experts_still_resident': startup['resident_stage_count'] == 7 and startup['resident_primary_complete'],
              'eight_slots_on_all_devices': startup['batch_slot_devices'] == list(range(8)),
              'only_exe_and_mtp_flag_changed': recomputed == cfg,
              'original_executable_preserved': sha(Path(deployment['original_executable']).read_bytes()) == ORIGINAL_SHA,
              'synthetic_request_matches_abba': same,
              'actual_main_head_offload_seen': result['decode_head_offload_windows'] > 0 and result['checks']['decode_head_route_confirmed'] is True,
              'actual_mtp_head_offload_seen': result['mtp_head_offload_steps'] > 0 and result['checks']['mtp_head_route_confirmed'] is True,
              'isolated_complete_smoke': result['checks']['request_ok'] is True and result['checks']['isolated_activity'] is True,
              'smoke_engine_record_unique': result['checks']['engine_full_request_record_unique'] is True,
              'smoke_acceptance_record_valid': result['checks']['draft_acceptance_record_unique'] is True and result['checks']['draft_acceptance_counts_valid'] is True,
              'smoke_192_tokens': result['usage'].get('completion_tokens') == 192 and result['finish_reason'] == 'length',
              'no_smoke_cuda_errors': not result['checks']['cuda_error_lines'],
              'cpu48_schedutil_preserved': len(gov) == 48 and set(gov) == {'schedutil'}}
    record = {'time_epoch': time.time(), 'checks': checks, 'pass': all(checks.values()),
              'deployment_sha256': sha((ROOT / 'deployment.json').read_bytes()), 'reference_sha256': sha(a.reference.read_bytes()),
              'status': {k: state.get(k) for k in ('loaded', 'engine', 'concurrency', 'activity')},
              'processes': processes, 'startup_facts': startup, 'governors': gov,
              'smoke_main_head_windows': result['decode_head_offload_windows'],
              'smoke_mtp_head_steps': result['mtp_head_offload_steps'], 'smoke_output_sha256': result['output_sha256'],
              'smoke_draft_acceptance': result.get('matched_draft_acceptance')}
    save_json(ROOT / 'production-verification.json', record)
    print(json.dumps({k: record[k] for k in ('checks', 'pass', 'smoke_main_head_windows', 'smoke_mtp_head_steps')}, indent=2))
    return 0 if record['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
