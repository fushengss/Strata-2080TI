#!/usr/bin/env python3
"""Isolated same-binary resident piped-MTP prefill overlap A/B with selectable validation suites; dry-run unless --execute.

Never stops or starts production :8080. Execution requires no GPU compute process
and no listener on :8080, and owns only the server/children it starts on :8092.
Credential-bearing temporary configs are 0600 in /dev/shm and always removed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
import interleave_evidence as interleave

ENGINE_SHA = None  # Required explicit CLI identity binding on --execute
LIB_SHA = None
CONFIG_SHA = None
SERVER_SHA = None
VARIABLES = ('STRATA_PIPED_CONCURRENT_MTP',)

def arm_values(arm):
    return {'STRATA_PIPED_CONCURRENT_MTP': '1' if arm == 'on' else '0'}

BASE_FIXED_ENV = {'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5',
                  'STRATA_PW_WITH_BATCH': '1'}
FIXED_ENV = {**BASE_FIXED_ENV, 'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1',
             'STRATA_BATCH_SELF_COMMIT': '0', 'STRATA_PIPED_MTP_HEAD': '1', 'STRATA_PIPED_TARGET_HEAD': '1', 'STRATA_PIPED_ASYNC_SAMPLE': '0'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def gpu_pids():
    result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                     text=True, timeout=15)
    rows = [line.strip() for line in result.splitlines() if line.strip()]
    require(all(row.isdigit() for row in rows), 'Cannot establish GPU process inventory')
    return sorted({int(row) for row in rows})


def process_info(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'ppid': int(fields[1]), 'pgrp': int(fields[2]), 'start_ticks': fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def is_descendant(pid, parent):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == parent:
            return True
        seen.add(pid)
        info = process_info(pid)
        if info is None:
            return False
        pid = info['ppid']
    return False


def own_gpu_processes(server, known):
    pids = gpu_pids()
    foreign = []
    for pid in pids:
        info = process_info(pid)
        if info is not None and is_descendant(pid, server.pid):
            known[pid] = info
        else:
            foreign.append(pid)
    require(not foreign, 'An unrelated GPU process appeared; this runner will not terminate it')
    return pids


def stop(server, known):
    if server is None:
        return
    # The engine has its own process group (serve/lifeline.py). Ask its server to
    # close it first; fallback signals only recorded descendants with matching birth times.
    if server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait(timeout=10)
    for old in known.values():
        now = process_info(old['pid'])
        if now and now['start_ticks'] == old['start_ticks'] and now['pgrp'] == old['pgrp']:
            try:
                if now['pgrp'] == now['pid']:
                    os.killpg(now['pgrp'], signal.SIGKILL)
                else:
                    os.kill(now['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not gpu_pids():
            return
        time.sleep(1)
    raise RuntimeError('GPU processes remain after own-arm cleanup; inspect manually')


def governors():
    return {str(p): p.read_text().strip()
            for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))}


def stop_monitor(monitor):
    if monitor is not None:
        monitor.terminate()
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill(); monitor.wait(timeout=10)


def port_unused(port):
    with socket.socket() as s:
        s.settimeout(.5)
        return s.connect_ex(('127.0.0.1', port)) != 0


def status(port, key):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/status',
                                     headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(request, timeout=5) as f:
        value = json.load(f)
    # No /status output tail, request bodies, headers or credentials are retained.
    return {k: value.get(k) for k in ('loaded', 'engine', 'concurrency', 'activity', 'vram', 'machine')}


def harness_files(source):
    return sorted({p for p in source.glob('*.py')} | {p for p in source.glob('*plan*.json')}
                  | {source / name for name in ('README.txt', 'source-provenance.json', 'offline-checks.json')
                     if (source / name).is_file()})


def freeze(source):
    for name in ('runner_interleave.py', 'finite_mixed.py', 'stream_client.py', 'fixture_prompts.py', 'stall_metrics.py', 'calibrate_fixtures.py', 'correctness.py', 'workload-plan.json',
                 'c8_check.py', 'prefill_interleave_check.py', 'budget_correctness.py', 'long_concurrency_check.py',
                 'compare_interleave.py', 'interleave_evidence.py', 'admission_smoke.py', 'cross_interleave_compatibility.py'):
        require((source / name).is_file(), 'Missing required harness file: ' + name)
    return {p.name: sha(p) for p in harness_files(source)}


def make_config(base, arm, binary, port, log, key, cap_both=False):
    require(arm in ('off', 'on'), 'Unknown arm')
    cfg = copy.deepcopy(base)
    cfg.update(exe=str(binary), host='127.0.0.1', port=port, log=str(log), api_key=key,
               reasoning_budget_engine_cap=True)
    require(all(base.get('env', {}).get(name) in (None,'0','') for name in ('STRATA_PIPED_CONCURRENT','STRATA_PIPED_READ_PIECE','STRATA_PIPED_READ_DECODE_ONLY','STRATA_PIPED_CONCURRENT_MIN','STRATA_PIPED_MTP_OBSERVE','STRATA_MTP_PROFILE')), 'Rejected read guards/observation must stay disabled')
    cfg.setdefault('env', {}).update(FIXED_ENV)
    for name in VARIABLES:
        cfg['env'][name] = arm_values(arm)[name]
    return cfg


def comparable_config(cfg):
    value = copy.deepcopy(cfg)
    for name in ('log', 'api_key'):
        value.pop(name, None)
    for name in VARIABLES:
        value.get('env', {}).pop(name, None)
    return value


def safe_config(cfg):
    # Only configuration needed to review this experiment. Hash the full original
    # separately; never export arbitrary inherited config/env values.
    keys = ('exe', 'host', 'port', 'gpu', 'parallel', 'layer_split', 'backend', 'arch', 'reasoning_budget_engine_cap')
    result = {k: cfg[k] for k in keys if k in cfg}
    result['env'] = {k: v for k, v in cfg.get('env', {}).items()
                     if k.startswith('STRATA_') and not any(s in k.upper()
                     for s in ('KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL'))}
    result['engine_args_sha256'] = digest(cfg.get('args', []))
    return result


def startup_facts(path):
    text = path.read_text(errors='replace')
    own = [{'device': int(d), 'buffer_gib': float(b), 'free_before_gib': float(f)}
           for d, b, f in re.findall(r'CUDA(\d+) keeps its own prompt buffers \(([0-9.]+) GiB of ([0-9.]+) GiB free\): no loan', text)]
    stages = [{'device': int(d), 'slots': int(n), 'resident': int(r), 'profiled': int(p)}
              for d, n, r, p in re.findall(r'layer split: CUDA(\d+) runs layers \d+-\d+, expert cache (\d+) slots '
                                          r'\([0-9.]+ GiB\), (\d+) of its (\d+) profiled pairs', text)]
    primary = re.findall(r'pre-filled (\d+) of (\d+) slots from the profile', text)
    target_startup = [[int(v) for v in row] for row in re.findall(
        r'STRATA_PIPED_TARGET_HEAD: CUDA(\d+)\+(\d+), source CUDA(\d+), full logits return before original slot sampling; prefill placement unchanged', text)]
    async_startup = [int(device) for device in re.findall(
        r'^strata serve: STRATA_PIPED_ASYNC_SAMPLE: source CUDA(\d+), event-polled slot sampling with persistent history$', text, re.M)]
    return {'async_sample_startup': async_startup, 'target_head_startup': target_startup, 'own_prompt_buffers': own, 'resident_stages': stages,
            'primary_resident': [int(x) for x in primary[-1]] if primary else None,
            'batch_slot_devices': sorted(set(int(x) for x in re.findall(r'8 slot sessions on CUDA(\d+)', text))),
            'shared_head_workspaces': [[int(v) for v in row] for row in re.findall(r'STRATA_PIPED_MTP_HEAD: (\d+) private slot workspaces, CUDA(\d+)\+(\d+), 1 shared weight bank, source CUDA(\d+)', text)],
            'piped_mtp_startup_enabled': bool(re.search(r'STRATA_PIPED_MTP: 8 pipelined groups of 1 slot draft up to 1 MTP proposal', text)),
            'all_eight_have_3_80_gib_own_buffers': len(own) == 8 and {x['device'] for x in own} == set(range(8))
                and all(abs(x['buffer_gib'] - 3.80) < .011 for x in own),
            'all_experts_resident': primary[-1:] == [('3072', '3072')] and len(stages) == 7
                and {x['device'] for x in stages} == set(range(1, 8))
                and all(x['slots'] == x['resident'] == x['profiled'] == 3072 for x in stages)}


def async_sample_rows(text):
    return [[int(v) for v in row] for row in re.findall(
        r'^strata piped async sample: (\d+) windows, (\d+) sampled rows completed in this batch$', text, re.M)]


def engine_evidence(known, binary, library, arm):
    rows = []
    for pid in known:
        proc = Path('/proc') / str(pid)
        exe = (proc / 'exe').resolve(strict=True)
        if exe != binary:
            continue
        env = dict(s.split('=', 1) for s in (proc / 'environ').read_bytes().decode().split('\0') if '=' in s)
        maps = {line.split(None, 5)[5].strip() for line in (proc / 'maps').read_text().splitlines()
                if len(line.split(None, 5)) == 6}
        libraries = sorted(p for p in maps if Path(p).name == 'libstrata_kq.so')
        rows.append({'pid': pid, 'exe': str(exe), 'engine_sha256': sha(exe),
                     'kq_library_paths': libraries, 'expected_library_mapped': str(library) in libraries,
                     'library_sha256': sha(library),
                     'flags': {k: env.get(k) for k in (*FIXED_ENV, *VARIABLES)}})
    require(len(rows) == 1, 'Expected exactly one own tested engine process')
    row = rows[0]
    require(row['engine_sha256'] == ENGINE_SHA and row['expected_library_mapped'] and row['library_sha256'] == LIB_SHA,
            'Actual engine or loaded KQ library differs')
    require(all(row['flags'][k] == v for k, v in FIXED_ENV.items()) and
            all(row['flags'][k] == arm_values(arm)[k] for k in VARIABLES), 'Actual engine flags differ')
    return rows


def credential_scan(directory, key):
    matches = []
    needle = key.encode()
    for path in directory.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        # Match across block boundaries without reading large GPU/client artifacts at once.
        tail = b''
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                if needle in tail + block:
                    matches.append(str(path.relative_to(directory))); break
                tail = block[-len(needle):]
    return {'credential_matches': len(matches), 'matching_files': matches,
            'note': 'Scanned only for this arm temporary API key; key is not retained.'}


def validate_interleave_startup(arm, path):
    text=path.read_text(errors='replace')
    count=text.count(interleave.STARTUP)
    require(count==(1 if arm=='on' else 0),'Exact overlap startup marker differs from arm')


def validate_interleave_run(arm, suite, text):
    result=interleave.validate(arm,suite,text)
    require(result['all_checks_passed'],'Reader/pump evidence qualification failed: '+json.dumps(result['checks']))
    return result


def self_test():
    sample={'env':{},'reasoning_budget_engine_cap':False,'exe':'old','api_key':'OFFLINE_SECRET','log':'old'}
    left=make_config(sample,'off',Path('/new/strata'),8092,Path('/tmp/off.log'),'new-key')
    right=make_config(sample,'on',Path('/new/strata'),8092,Path('/tmp/on.log'),'other-key')
    require(left['env']['STRATA_PIPED_CONCURRENT_MTP']=='0' and right['env']['STRATA_PIPED_CONCURRENT_MTP']=='1','Mode flags incorrect')
    require(comparable_config(left)==comparable_config(right),'Other configuration changed')
    require(all(cfg['env']['STRATA_PIPED_TARGET_HEAD']=='1' and cfg['env']['STRATA_PIPED_ASYNC_SAMPLE']=='0' for cfg in (left,right)),'Fixed flags changed')
    require(sample['api_key']=='OFFLINE_SECRET' and sample['env']=={},'Original configuration mutated')
    require('OFFLINE_SECRET' not in json.dumps(safe_config(left)),'Secret in safe configuration')
    print('OFFLINE PASS: single overlap flag, P1 enabled, async off, immutable base; no network/GPU')


def main():
    global ENGINE_SHA, LIB_SHA, CONFIG_SHA, SERVER_SHA
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('/srv/Strata/strata-orca-q5_k_m.57.json'))
    p.add_argument('--exe', type=Path, help='defaults to the deployed config executable; exact SHA required')
    p.add_argument('--out', type=Path)
    p.add_argument('--server', type=Path, default=Path('/srv/Strata-0157/serve/server.py'))
    p.add_argument('--python', default='/srv/Strata/.venv/bin/python')
    p.add_argument('--port', type=int, default=8092)
    p.add_argument('--arms', default='off,on')
    p.add_argument('--suites', default='admission,correctness,primary')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--calibration-sha256')
    p.add_argument('--target-head', choices=('0', '1'), default='1', help='Fixed P1 setting in both overlap arms')
    for component in ('engine', 'library', 'server', 'config'):
        p.add_argument('--' + component + '-sha256', help='Required exact SHA256 for execution; no auto-trust')
    p.add_argument('--profile', choices=('short', 'full'), default='short')
    p.add_argument('--sampling-profile', choices=('sampled', 'greedy'), default='sampled')
    p.add_argument('--levels', default='4')
    p.add_argument('--ready-timeout', type=int, default=1200)
    p.add_argument('--suite-timeout', type=int, default=1800)
    p.add_argument('--cap-both', action='store_true', help='Use the budget fix in both arms to isolate batch MTP and verify identical budget continuation')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    require(a.target_head == '1', 'P1 target head stays enabled in this experiment')
    FIXED_ENV['STRATA_PIPED_TARGET_HEAD'] = a.target_head
    if a.self_test:
        require(not a.execute, '--self-test cannot execute'); self_test(); return
    arms, suites = a.arms.split(','), a.suites.split(',')
    require(bool(arms) and not set(arms) - {'off', 'on'}, '--arms accepts off,on')
    require(bool(suites) and not set(suites) - {'admission', 'correctness', 'primary', 'mixed100k', 'prefill', 'budget', 'c8', 'long'} and len(set(suites)) == len(suites),
            '--suites accepts admission,c8,primary,mixed100k,prefill,correctness,budget,long without duplicates')
    require(a.port == 8092, 'This experiment is fixed to isolated loopback port8092')
    levels = a.levels.split(',')
    require(bool(levels) and not set(levels) - {'1', '2', '4', '8'} and len(set(levels)) == len(levels),
            '--levels must be a unique subset of 1,2,4,8')
    if 'primary' in suites or 'mixed100k' in suites:
        require(levels == ['4'], 'Finite mixed tests require --levels 4')
    if not a.execute:
        print(json.dumps({'mode': 'DRY_RUN_NO_NETWORK_NO_GPU', 'arms': arms, 'suites': suites,
                          'profile': a.profile, 'sampling_profile': a.sampling_profile, 'levels': levels,
                          'port': a.port, 'variables': VARIABLES, 'fixed': FIXED_ENV,
                          'cap_both': a.cap_both, 'p1_fixed_target_head': a.target_head, 'requires': 'GPUs empty and production8080 stopped by operator'}, indent=2)); return
    identities = [a.engine_sha256, a.library_sha256, a.server_sha256, a.config_sha256]
    require(all(isinstance(x, str) and re.fullmatch(r'[0-9a-f]{64}', x) for x in identities),
            '--execute requires explicit --engine-sha256 --library-sha256 --server-sha256 --config-sha256')
    ENGINE_SHA, LIB_SHA, SERVER_SHA, CONFIG_SHA = identities
    require(a.out is not None, '--execute requires --out')
    out = a.out.resolve()
    require(not out.exists() or not any(out.iterdir()), '--out must be new or empty')
    source = Path(__file__).resolve().parent
    require(out != source, '--out must not be the harness directory')
    if any(name in suites for name in ('primary', 'mixed100k')):
        require(a.calibration is not None and a.calibration.is_file() and a.calibration_sha256 and sha(a.calibration)==a.calibration_sha256, 'Explicit frozen calibration and SHA required')
        calibration_data=a.calibration.read_bytes()
    else:
        calibration_data=None
    require(not gpu_pids(), 'GPUs occupied; no existing process was stopped')
    require(port_unused(8080), 'Production8080 still listens; no existing service was stopped')
    require(port_unused(a.port), 'Isolated port8092 already in use')
    original = a.config.read_bytes()
    require(hashlib.sha256(original).hexdigest() == CONFIG_SHA, 'Base config differs from explicitly bound SHA256')
    base = json.loads(original)
    require(all(str(base.get('env', {}).get(k)) == v for k, v in BASE_FIXED_ENV.items()), 'Fixed deployed head/PW flags differ')
    require(base.get('env', {}).get('STRATA_BATCH_SELF_COMMIT') == '0', 'Expected deployed self-commit disabled')
    require(base.get('parallel') == 8, 'This experiment requires the existing eight slots')
    require('--batch-mtp' not in base.get('args', []), 'A command-line batch-MTP flag would override the off arm')
    binary = (a.exe or Path(base['exe'])).resolve()
    library = binary.parent / 'libstrata_kq.so'
    require(sha(binary) == ENGINE_SHA and sha(library) == LIB_SHA, 'Executable or adjacent KQ library differs')
    hashes = freeze(source)
    server_hash = sha(a.server)
    require(server_hash == SERVER_SHA, 'Server differs from explicitly bound SHA256')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('harness',):
        (out / name).mkdir()
    for name in hashes:
        (out / 'harness' / name).write_bytes((source / name).read_bytes())
    require(all(sha(out / 'harness' / name) == value for name, value in hashes.items()),
            'Harness changed while its frozen copy was made')
    fixture = 'prefill-interleave-' + digest({'plan': hashes['workload-plan.json'], 'profile': a.profile,
                                    'sampling': a.sampling_profile, 'levels': levels})[:20]
    manifest = {'created_epoch': time.time(), 'sequence': arms, 'suites': suites, 'profile': a.profile,
                'sampling_profile': a.sampling_profile, 'levels': levels, 'fixture_id': fixture,
                'same_binary_sha256': ENGINE_SHA, 'adjacent_library_sha256': LIB_SHA,
                'base_config_sha256': CONFIG_SHA, 'server_sha256': server_hash, 'harness_sha256': hashes,
                'variables': {k: {'off': arm_values('off')[k], 'on': arm_values('on')[k]} for k in VARIABLES},
                'server_variables': {'reasoning_budget_engine_cap': {'off': True, 'on': True}}, 'fixed_env': FIXED_ENV,
                'governors_before': governors(), 'port': a.port, 'arms': [],
                'ownership': 'Only own isolated process groups/recorded descendants; production8080 untouched'}
    if calibration_data is not None:
        (out / 'calibration.json').write_bytes(calibration_data)
        manifest['calibration_sha256']=a.calibration_sha256
    save(out / 'manifest.json', manifest)
    server = monitor = None
    config_path = server_log = monitor_log = None
    known = {}
    ad = None
    key = ''
    def interrupted(signum, frame):
        raise KeyboardInterrupt('runner interrupted')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    try:
        for index, arm in enumerate(arms, 1):
            require(not gpu_pids() and port_unused(8080) and port_unused(a.port), 'Maintenance exclusivity lost before arm')
            require(freeze(source) == hashes, 'Harness changed after freeze')
            require(sha(binary) == ENGINE_SHA and sha(library) == LIB_SHA and sha(a.server) == server_hash,
                    'Engine, library or server changed after freeze')
            require(a.config.read_bytes() == original, 'Original service config changed during experiment')
            tag = f'{index:02d}-{arm}'; ad = out / tag; ad.mkdir()
            key = secrets.token_urlsafe(32)
            cfg = make_config(base, arm, binary, a.port, ad / 'engine.log', key, a.cap_both)
            meta = {'tag': tag, 'arm': arm, 'start_epoch': time.time(), 'config': safe_config(cfg),
                    'config_except_variables_log_key_sha256': digest(comparable_config(cfg)),
                    'engine_sha256': ENGINE_SHA, 'adjacent_library_sha256': LIB_SHA,
                    'governors_before': governors(), 'suite_returncodes': {}}
            save(ad / 'metadata.json', meta)
            fd, name = tempfile.mkstemp(prefix='strata-batch-mtp-', suffix='.json', dir='/dev/shm')
            config_path = Path(name)
            with os.fdopen(fd, 'w') as f:
                os.fchmod(f.fileno(), 0o600); json.dump(cfg, f); f.flush(); os.fsync(f.fileno())
            env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_')}
            server_log = (ad / 'server.log').open('w')
            server = subprocess.Popen([a.python, str(a.server), '--engine', 'strata', '--config', str(config_path),
                                       '--host', '127.0.0.1', '--port', str(a.port)],
                                      cwd=str(a.server.resolve().parents[1]), env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            known = {}
            save(out / 'active.json', {'tag': tag, 'phase': 'loading', 'server_pid': server.pid})
            print('LOADING', tag, 'server pid', server.pid, flush=True)
            deadline = time.monotonic() + a.ready_timeout
            while time.monotonic() < deadline:
                require(server.poll() is None, 'Isolated server exited while loading')
                own_gpu_processes(server, known)
                try:
                    state = status(a.port, key)
                    if state.get('loaded'):
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError('Isolated server readiness timed out')
            require((state.get('activity') or {}).get('in_flight') == 0, 'Server is not idle before benchmark')
            concurrency = state.get('concurrency') or {}
            require(isinstance(concurrency, dict) and concurrency.get('serving') == 8 and concurrency.get('requested') == 8,
                    'Server reduced or changed the eight requested/serving slots')
            facts = startup_facts(ad / 'engine.log')
            actual = engine_evidence(known, binary, library, arm)
            meta.update(ready_epoch=time.time(), ready_status=state, owned_gpu_processes=list(known.values()),
                        startup_facts=facts, actual_engine=actual)
            save(ad / 'metadata.json', meta)
            require(facts['batch_slot_devices'] == list(range(8)), 'Eight slots are not present on all eight GPUs')
            require(facts['all_experts_resident'], 'Expert residency changed from fully resident baseline')
            require(facts['all_eight_have_3_80_gib_own_buffers'], 'Original own prompt buffer allocation changed')
            require(facts['piped_mtp_startup_enabled'] is True, 'Pipelined MTP startup route differs from arm')
            require(facts['shared_head_workspaces'] == [[8, 3, 5, 7]], 'Shared head startup workspace/device route differs')
            require(facts['target_head_startup'] == ([[2, 4, 7]] if a.target_head == '1' else []),
                    'Fixed P1 target-head startup route differs')
            require(facts['async_sample_startup'] == [], 'Rejected async sampler must stay disabled')
            validate_interleave_startup(arm, ad / 'engine.log')
            monitor_log = (ad / 'gpu.csv').open('w')
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,power.limit,enforced.power.limit,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown',
                                        '--format=csv', '-lms', '500'], stdout=monitor_log, stderr=subprocess.DEVNULL)
            for suite in suites:
                own_gpu_processes(server, known)
                require(freeze(source) == hashes, 'Harness changed before suite')
                cmd = [sys.executable, str(source / {'admission':'admission_smoke.py', 'correctness': 'correctness.py', 'primary': 'finite_mixed.py', 'mixed100k': 'finite_mixed.py', 'prefill': 'prefill_interleave_check.py', 'budget': 'budget_correctness.py', 'c8': 'c8_check.py', 'long': 'long_concurrency_check.py'}[suite]),
                       '--execute', '--exclusive-benchmark', '--config', str(config_path), '--port', str(a.port),
                       '--out', str(ad / suite)]
                if suite == 'admission':
                    cmd += ['--arm',arm,'--engine-log',str(ad/'engine.log')]
                elif suite == 'correctness':
                    cmd += ['--expect-piped-mtp', 'on', '--engine-log', str(ad / 'engine.log')]
                elif suite in ('c8', 'long'):
                    cmd += ['--expect-piped-mtp', 'on', '--engine-log', str(ad / 'engine.log')]
                elif suite == 'budget':
                    cmd += ['--expect-cap', 'on' if cfg['reasoning_budget_engine_cap'] else 'off',
                            '--expect-piped-mtp', 'on', '--engine-log', str(ad / 'engine.log')]
                elif suite == 'prefill':
                    cmd += ['--arm', arm, '--engine-sha256', ENGINE_SHA, '--library-sha256', LIB_SHA, '--target-head', a.target_head]
                elif suite in ('primary', 'mixed100k'):
                    cmd += ['--case', 'primary' if suite == 'primary' else 'long',
                            '--calibration', str(out / 'calibration.json'), '--engine-log', str(ad / 'engine.log')]
                head_cursor = (ad / 'engine.log').stat().st_size
                print('RUNNING', tag, suite, flush=True)
                save(out / 'active.json', {'tag': tag, 'phase': suite, 'server_pid': server.pid})
                with (ad / (suite + '.log')).open('w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.suite_timeout)
                with (ad / 'engine.log').open('rb') as head_log:
                    head_log.seek(head_cursor)
                    head_text = head_log.read().decode(errors='replace')
                head_rows = [[int(v) for v in row] for row in re.findall(r'strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch', head_text)]
                target_rows = [[int(v) for v in row] for row in re.findall(
                    r'strata piped target head: CUDA(\d+)\+(\d+), (\d+) windows, (\d+) verification rows offloaded in this batch', head_text)]
                async_rows = async_sample_rows(head_text)
                meta.setdefault('shared_head_suite_evidence', {})[suite] = {'rows': head_rows, 'steps': sum(row[3] for row in head_rows)}
                meta.setdefault('target_head_suite_evidence', {})[suite] = {
                    'rows': target_rows, 'windows': sum(row[2] for row in target_rows),
                    'verification_rows': sum(row[3] for row in target_rows),
                    'note': 'Verification rows include proposals and are not accepted output tokens.'}
                meta.setdefault('async_sample_suite_evidence', {})[suite] = {
                    'rows': async_rows, 'windows': sum(row[0] for row in async_rows),
                    'sampled_rows': sum(row[1] for row in async_rows),
                    'note': 'Rejected asynchronous sampler remains disabled in every arm.'}
                meta['suite_returncodes'][suite] = result.returncode
                save(ad / 'metadata.json', meta)
                require(all(row[:2] == [2, 4] and row[2] > 0 and row[3] > 0 for row in target_rows),
                        'Unexpected piped target-head counters/devices')
                if a.target_head == '0' or suite == 'prefill':
                    require(not target_rows, 'Target-head offload observed in disabled arm or solo prefill guard')
                else:
                    require(bool(target_rows), 'No actual piped target-head windows in enabled concurrent suite')
                require(not async_rows, 'Rejected async sampler executed')
                meta.setdefault('interleave_suite_evidence', {})[suite] = validate_interleave_run(arm, suite, head_text)
                save(ad / 'metadata.json', meta)
                require(all(0 <= row[0] < 8 and row[1:3] == [3, 5] for row in head_rows), 'Unexpected shared MTP head slot/device counters')
                if suite != 'prefill':
                    require(sum(row[3] for row in head_rows) > 0, 'No actual concurrent head offload steps in suite')
                require(result.returncode == 0, 'Suite failed; subsequent workload skipped: ' + suite)
                require((status(a.port, key).get('activity') or {}).get('in_flight') == 0,
                        'Requests still in flight after suite')
            own_gpu_processes(server, known)
            require(freeze(source) == hashes, 'Harness changed during arm')
            require(sha(binary) == ENGINE_SHA and sha(library) == LIB_SHA and sha(a.server) == server_hash,
                    'Executable, library or server changed during arm')
            meta.update(finished_epoch=time.time(), governors_after=governors(), owned_gpu_processes=list(known.values()))
            save(ad / 'metadata.json', meta)
            manifest['arms'].append({'tag': tag, 'arm': arm, 'suite_returncodes': meta['suite_returncodes']})
            save(out / 'manifest.json', manifest)
            stop_monitor(monitor); monitor = None
            monitor_log.close(); monitor_log = None
            stop(server, known); server = None
            server_log.close(); server_log = None
            scan = credential_scan(ad, key)
            save(ad / 'credential-scan.json', scan)
            require(scan['credential_matches'] == 0, 'Temporary key found in own artifacts; do not archive')
            config_path.unlink(); config_path = None
            print('FINISHED', tag, 'GPUs released', flush=True)
    except BaseException as error:
        manifest['failure'] = {'kind': type(error).__name__, 'message': str(error) if isinstance(error, RuntimeError) else 'See own arm logs'}
        raise
    finally:
        # Signals during cleanup must not bypass credential deletion.
        signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                stop_monitor(monitor)
            finally:
                stop(server, known)
        finally:
            for handle in (monitor_log, server_log):
                if handle is not None:
                    handle.close()
            if config_path is not None:
                config_path.unlink(missing_ok=True)
            if ad is not None and key:
                save(ad / 'credential-scan.json', credential_scan(ad, key))
            manifest.update(ended_epoch=time.time(), governors_after=governors(),
                            original_config_unchanged=a.config.read_bytes() == original)
            try:
                remaining = gpu_pids()
            except Exception:
                remaining = None
            manifest['remaining_gpu_pids'] = remaining
            save(out / 'manifest.json', manifest)
            save(out / 'active.json', {'phase': 'stopped', 'remaining_gpu_pids': remaining})
    completed = manifest['arms']
    baseline = next((row for row in completed if row['arm'] == 'off'), None)
    candidates = [row for row in completed if row['arm'] == 'on']
    manifest['comparisons'] = []
    if baseline and candidates:
        comparisons = out / 'comparisons'; comparisons.mkdir()
        for candidate in candidates:
            label = baseline['tag'] + '-vs-' + candidate['tag']
            with (comparisons / (label + '.log')).open('w') as handle:
                child = subprocess.run([sys.executable, str(source / 'compare_interleave.py'),
                                        '--baseline', str(out / baseline['tag']), '--candidate', str(out / candidate['tag']),
                                        '--out', str(comparisons / label)], stdout=handle, stderr=subprocess.STDOUT, timeout=600)
            manifest['comparisons'].append({'baseline': baseline['tag'], 'candidate': candidate['tag'],
                                            'returncode': child.returncode})
            save(out / 'manifest.json', manifest)
            require(child.returncode == 0, 'Exact matched-arm comparison failed; candidate does not qualify')
    else:
        manifest['comparison_note'] = 'Only one arm type selected; no A/B correctness or performance conclusion.'
        save(out / 'manifest.json', manifest)
    print('COMPLETE', len(manifest['arms']), 'arms; no deployment decision made', flush=True)


if __name__ == '__main__':
    main()
