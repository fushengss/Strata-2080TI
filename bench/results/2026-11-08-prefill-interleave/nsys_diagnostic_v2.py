#!/usr/bin/env python3
"""Separate owned Nsight diagnostic v2; explicit frozen engine/overwrite selection, default dry-run."""
from __future__ import annotations
import argparse
import atexit
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ENGINE = '45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6'
V4_ENGINE = '92822929f1f60ecd80c64ce04fa47a44a6d789b0e4ca028ef5be5e39fecab20c'
LIBRARY = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
CONFIG = 'f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd'
SERVER = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
CALIBRATION = 'b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd'
FROZEN = {
    'runner_interleave.py': '85b0ecd56209f420b8781360ead2bb858a87b2673e953de090b3aed9cf90ca04',
    'correctness.py': 'b5a7786792e6632f426e3c2fe664cfaf7b37fed8931de0f230960e37fa8a0865',
    'c8_check.py': '5a8648e8324b719bc264e0ef2501e01d638f7c0203bb6c68d470ec91cf4b706d',
    'finite_mixed.py': 'a797d6aba9077b8b77bae396d520556b786036077ca7b3ebb98f2b0d8e5b9b39',
    'stream_client.py': '1f3c62913ed9a6fe7cd53aaded7b4990289c093079243906620a88a04987d38a',
    'fixture_prompts.py': 'ae83903025468d01fa383848499cb419434d2bbba812dbd40b8c7344587a1f34',
    'stall_metrics.py': '09b664e45c5bc417ad1d590f314dee82edbbb964b99d9ff4819c6901607ae005',
    'calibrate_fixtures.py': '010b1e7e9e07aaef6117c4d64f2acd5ed042f8f8643fd898b9f36afddc291d16',
    'interleave_evidence.py': 'e5874612fbbfef511797cb2ccf62b442370b05e058141a0b264e5fc3b669386e',
    'cancel_read_proof.py': '9d7ab1638eb11e96b46206d3940a99f0b7f2149540bdc1ccc24fd2291dc15a8a',
}
# This is a separate diagnostic, with no added NVTX instrumentation. CUDA+OSRT
# exposes submission, synchronization and runtime calls; ctxswitch sampling is OFF.
def commands(nsys, session, wrapper, prefix):
    return {
        'launch': [str(nsys), 'launch', '--session-new='+session, '--trace=cuda,osrt',
                   '--cuda-graph-trace=node', '--cuda-event-trace=false', '--wait=all',
                   '--show-output=false', '/bin/sh', str(wrapper)],
        'start': [str(nsys), 'start', '--session='+session, '--sample=none',
                  '--cpuctxsw=none', '--output='+str(prefix)],
        'stop': [str(nsys), 'stop', '--session='+session],
    }

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def validate_selection(engine_sha, snapshot_overwrite):
    require(engine_sha in (ENGINE,V4_ENGINE), 'Select an explicitly frozen v3/v4 engine SHA')
    require(snapshot_overwrite in ('0','1'), 'Snapshot overwrite must be0or1')
    require(snapshot_overwrite == '0' or engine_sha == V4_ENGINE, 'Overwrite1 requires frozen v4 engine')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)

def stamp():
    return {'epoch': time.time(), 'monotonic': time.monotonic()}

def exec_target(path):
    """Called only by the private shell AFTER stdout/stderr redirection."""
    b = json.loads(path.read_text())
    fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    save(Path(b['pid_file']), {'pid': os.getpid(), 'start_ticks': fields[19],
                             'nonce': b['nonce'], 'started': stamp()})
    os.chdir(b['cwd'])
    os.execv(b['argv'][0], b['argv'])

class MarkerLog:
    def __init__(self, handle, ready):
        self.handle, self.ready, self.pending = handle, ready, ''
        self.lock = threading.Lock()
    def write(self, text):
        with self.lock:
            self.handle.write(text); self.handle.flush()
            self.pending += text
            while '\n' in self.pending:
                line, self.pending = self.pending.split('\n', 1)
                if line.startswith('INJECT primary-injected-1 actual_tokens='):
                    self.ready.set()
        return len(text)
    def flush(self):
        self.handle.flush()

def runtime(a):
    require(sys.platform.startswith('linux'), 'Execution requires the remote Linux host; use --self-test locally')
    require(a.out and a.exe and a.harness, 'Explicit --out --exe --harness required')
    require(12 <= a.seconds <= 15, 'Capture must be12..15seconds')
    out = a.out.resolve(); binary = a.exe.resolve(); library = binary.parent/'libstrata_kq.so'
    require(not out.exists(), 'Diagnostic output must be new')
    for name, expected in FROZEN.items():
        require(sha(a.harness/name) == expected, 'Frozen harness mismatch: '+name)
    sys.path.insert(0, str(a.harness.resolve()))
    runner = importlib.import_module('runner_interleave')
    runner.ENGINE_SHA, runner.LIB_SHA = a.engine_sha256, LIBRARY
    require(not runner.gpu_pids(), 'GPU occupied; no existing process will be stopped')
    require(runner.port_unused(8080) and runner.port_unused(8092), '8080/8092 must be closed by operator first')
    original = a.config.read_bytes()
    require(hashlib.sha256(original).hexdigest() == CONFIG, 'Base config SHA differs')
    require(sha(binary) == a.engine_sha256 and sha(library) == LIBRARY and sha(a.server) == SERVER, 'Binary/library/server identity mismatch')
    if a.workload == 'mixed':
        require(a.calibration and sha(a.calibration) == CALIBRATION, 'Frozen mixed calibration required')
    base = json.loads(original)
    require(base.get('parallel') == 8, 'Expected8slots')
    require(all(base.get('env', {}).get(k) == v for k, v in runner.FIXED_ENV.items()), 'Fixed production flags differ')
    old_umask = os.umask(0o077)
    out.mkdir(parents=True, mode=0o700)
    key = secrets.token_urlsafe(32)
    cfg = runner.make_config(base, a.arm, binary, 8092, out/'engine.log', key)
    cfg['env']['STRATA_PARK_OVERWRITE'] = a.snapshot_overwrite
    fd, name = tempfile.mkstemp(prefix='strata-nsys-', suffix='.json', dir='/dev/shm')
    config_path = Path(name)
    atexit.register(config_path.unlink, missing_ok=True)
    with os.fdopen(fd, 'w') as handle:
        os.fchmod(handle.fileno(), 0o600); json.dump(cfg, handle)
    nonce = secrets.token_hex(16); session = 'strata_diag_'+secrets.token_hex(8)
    bootstrap = {'nonce': nonce, 'pid_file': str(out/'server-process.json'),
                 'cwd': str(a.server.resolve().parents[1]),
                 'argv': [a.python, str(a.server.resolve()), '--engine', 'strata', '--config',
                          str(config_path), '--host', '127.0.0.1', '--port', '8092']}
    save(out/'bootstrap.json', bootstrap)
    wrapper = out/'server-wrapper.sh'
    wrapper.write_text('#!/bin/sh\nset -eu\numask 077\nexec >'+shlex.quote(str(out/'server.log'))+' 2>&1\nexec '+
                       shlex.join([a.python, str(Path(__file__).resolve()), '--exec-target', str(out/'bootstrap.json')])+'\n')
    os.chmod(wrapper, 0o700)
    argv = commands(a.nsys, session, wrapper, out/'timeline')
    receipt = {'scope': 'Separate instrumented diagnostic; never a timing A/B or speedup measurement.',
               'created': stamp(), 'session': session, 'workload': a.workload, 'arm': a.arm,
               'capture_seconds_requested': a.seconds, 'command_argv': argv,
               'engine_sha256': a.engine_sha256, 'library_sha256': LIBRARY, 'server_sha256': SERVER,
               'snapshot_overwrite': a.snapshot_overwrite,
               'base_config_sha256': CONFIG, 'harness_sha256': FROZEN,
               'helper_sha256': sha(Path(__file__)), 'config': runner.safe_config(cfg),
               'launch_injected': True, 'ordinary_pid_attach': False,
               'no_performance_claim': True, 'timeline_interpretation_pending': True}
    save(out/'receipt.json', receipt)
    launcher = None; server = None; known = {}; clients = []; worker = None
    collecting = False; stop_attempted = False
    ready = threading.Event(); done = threading.Event(); result = {}
    logs = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('diagnostic interrupted')
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    def same_process(info):
        now = runner.process_info(info['pid'])
        return now is not None and now['start_ticks'] == info['start_ticks']
    def stop_capture():
        nonlocal collecting, stop_attempted
        if not collecting or stop_attempted:
            return
        stop_attempted = True; receipt['stop_requested'] = stamp()
        with (out/'nsys-stop.log').open('w') as handle:
            process = subprocess.run(argv['stop'], stdout=handle, stderr=subprocess.STDOUT, timeout=90)
        receipt['stop_returned'] = stamp(); receipt['stop_returncode'] = process.returncode
        collecting = False
        require(process.returncode == 0, 'Nsight stop failed; inspect private log')
    def remember(instance):
        clients.append(instance)
    try:
        # Avoid inheriting API/cloud credentials into Nsight's target metadata.
        env = {k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG','LC_ALL','TZ',
            'LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','XDG_RUNTIME_DIR','TMPDIR') if k in os.environ}
        handle = (out/'nsys-launch.log').open('w'); logs.append(handle)
        launcher = subprocess.Popen(argv['launch'], stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        receipt['launcher_pid'] = launcher.pid
        deadline = time.monotonic()+a.ready_timeout
        while time.monotonic() < deadline:
            require(launcher.poll() is None, 'Nsight launch exited before model readiness')
            pidfile = out/'server-process.json'
            if server is None and pidfile.exists():
                server = json.loads(pidfile.read_text())
                require(server['nonce'] == nonce and same_process(server), 'Owned wrapper process identity mismatch')
                receipt['server_process'] = {k:server[k] for k in ('pid','start_ticks','started')}
            if server:
                require(same_process(server), 'Owned server exited during load')
                runner.own_gpu_processes(types.SimpleNamespace(pid=server['pid']), known)
                try:
                    status = runner.status(8092, key)
                    if status.get('loaded'):
                        break
                except (OSError, ValueError):
                    pass
            time.sleep(1)
        else:
            raise RuntimeError('Model ready timeout')
        require(status['activity']['in_flight'] == 0 and status['concurrency']['serving'] == 8, 'Loaded server not idle with8slots')
        facts = runner.startup_facts(out/'engine.log')
        require(facts['all_experts_resident'] and facts['all_eight_have_3_80_gib_own_buffers'] and
                facts['batch_slot_devices'] == list(range(8)) and facts['target_head_startup'] == [[2,4,7]] and
                not facts['async_sample_startup'] and facts['shared_head_workspaces'] == [[8,3,5,7]] and
                facts['piped_mtp_startup_enabled'], 'Startup residency/head/buffer routes differ')
        runner.validate_interleave_startup(a.arm, out/'engine.log')
        receipt.update(ready=stamp(), startup_facts=facts,
                       actual_engine=runner.engine_evidence(known, binary, library, a.arm))
        for engine in receipt['actual_engine']:
            actual_env = dict(item.split('=',1) for item in
                (Path('/proc')/str(engine['pid'])/'environ').read_bytes().decode().split('\0') if '=' in item)
            actual_overwrite = actual_env.get('STRATA_PARK_OVERWRITE')
            require(actual_overwrite == a.snapshot_overwrite, 'Actual engine snapshot overwrite flag differs')
            engine['flags']['STRATA_PARK_OVERWRITE'] = actual_overwrite
        save(out/'receipt.json', receipt)
        work = out/'workload'; engine_cursor = (out/'engine.log').stat().st_size
        def workload():
            try:
                if a.workload == 'c8':
                    c8 = importlib.import_module('c8_check'); core = c8.core_module()
                    class Observed(core.Client):
                        def __init__(self, args):
                            super().__init__(args); remember(self)
                    proxy = types.SimpleNamespace(**{**vars(core), 'Client': Observed})
                    args = argparse.Namespace(config=config_path, port=8092, out=work,
                        engine_log=out/'engine.log', timeout=600, max_tokens=1024, expect_piped_mtp='on')
                    result['returncode'] = c8.run(args, proxy)
                else:
                    finite = importlib.import_module('finite_mixed'); client = finite.base
                    class Observed(client.Client):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs); remember(self)
                    # Isolated diagnostic process only: preserve the module's globals
                    # because Client.request resolves synthetic_prompt in that module.
                    original_client = client.Client
                    client.Client = Observed
                    args = argparse.Namespace(port=8092, exclusive_benchmark=True, config=config_path,
                        calibration=a.calibration, out=work, engine_log=out/'engine.log', case='primary')
                    try:
                        result['returncode'] = finite.run(args)
                    finally:
                        client.Client = original_client
            except BaseException as error:
                result['error_kind'] = type(error).__name__
            finally:
                done.set()
        with (out/'workload.log').open('w') as worklog, contextlib.redirect_stdout(MarkerLog(worklog, ready)):
            worker = threading.Thread(target=workload, daemon=True); worker.start()
            deadline = time.monotonic()+300
            while time.monotonic() < deadline and not done.is_set():
                if a.workload == 'c8' and clients:
                    with clients[0].lock:
                        if len(clients[0].first) == 8 and len(clients[0].active) == 8:
                            ready.set()
                if ready.is_set():
                    break
                time.sleep(.02)
            require(ready.is_set() and not done.is_set(), 'Workload ended or never reached capture trigger')
            receipt['workload_trigger'] = stamp()
            receipt['trigger_definition'] = ('All8C8streams have first output and remain active' if a.workload == 'c8'
                else 'Frozen finite primary second injection marker, immediately before its calibrated~60krequest; no workload pause')
            runner.own_gpu_processes(types.SimpleNamespace(pid=server['pid']), known)
            receipt['start_requested'] = stamp(); collecting = True
            with (out/'nsys-start.log').open('w') as startlog:
                started = subprocess.run(argv['start'], stdout=startlog, stderr=subprocess.STDOUT, timeout=10)
            receipt['start_returned'] = stamp(); receipt['start_returncode'] = started.returncode
            require(started.returncode == 0, 'Nsight start failed; inspect private log')
            # Bound stop REQUEST against start REQUEST, not delayed export completion.
            until = receipt['start_requested']['monotonic']+a.seconds
            while time.monotonic() < until and not done.is_set():
                time.sleep(min(.1, max(0, until-time.monotonic())))
            stop_capture()
            require(done.wait(600), 'Diagnostic workload drain timed out')
            worker.join(1)
        receipt['workload_result'] = result
        usable_start = receipt['start_returned']['epoch']
        usable_end = receipt['stop_requested']['epoch']
        if a.workload == 'c8' and (work/'c8-check.json').is_file():
            measured = json.loads((work/'c8-check.json').read_text())['timing']['overlap']
            lo, hi = measured['common_start_epoch'], measured['common_end_epoch']
            receipt['capture_workload_coverage'] = {'common_sse_interval':measured,
                'conservative_capture_common_intersection_s':max(0,min(usable_end,hi)-max(usable_start,lo)) if lo is not None and hi is not None else 0,
                'capture_conservative_interval_inside_common':lo is not None and hi is not None and lo<=usable_start<=usable_end<=hi}
        elif (work/'results.json').is_file():
            report = json.loads((work/'results.json').read_text())
            receipt['capture_workload_coverage'] = {'visible_phase_coverage':report.get('visible_phase_coverage'),
                'long_injections':[{'name':r['name'],'prompt_tokens':r.get('actual_prompt_tokens'),
                    'request_start_epoch':r['start_epoch'],'request_end_epoch':r['end_epoch'],
                    'first_output_epoch':r['start_epoch']+r['ttft_s'] if r.get('ttft_s') is not None else None,
                    'captured_request_intersection_s':max(0,min(usable_end,r['end_epoch'])-max(usable_start,r['start_epoch']))}
                    for r in report['records'] if r['role']=='injected' and r.get('actual_prompt_tokens',0)>=50000],
                'scope':'HTTP overlap is not the exact GPU prefill interval; classify CUDA nodes and native reader boundaries before claiming coverage.'}
        with (out/'engine.log').open('rb') as handle:
            handle.seek(engine_cursor); numeric = runner.interleave.parse(handle.read().decode(errors='replace'))
        receipt['reader_evidence'] = numeric
        receipt['owned_workload_qualified'] = result.get('returncode') == 0 and 'error_kind' not in result
        receipt['report_files'] = [{'name':p.name,'bytes':p.stat().st_size} for p in out.glob('timeline*') if p.is_file()]
        receipt['report_exists'] = any(p.suffix == '.nsys-rep' and p.stat().st_size > 0 for p in out.glob('timeline*'))
        require(receipt['report_exists'], 'No nonempty Nsight report produced')
    except BaseException as error:
        receipt['failure_kind'] = type(error).__name__
        # Do not serialize arbitrary exception strings which may contain headers/config.
    finally:
        # A repeated interrupt must not skip private-config removal or receipts.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        owned = ([server] if server else [])+list(known.values())
        try:
            try:
                stop_capture()
            except BaseException as error:
                receipt['stop_failure_kind'] = type(error).__name__
            for client in clients:
                client.abort.set()
                if hasattr(client, 'monitor_stop'):
                    client.monitor_stop.set()
            # Only recorded process birth identities are signalled. No pkill,
            # port kill, power/governor changes or production restart.
            for sig, wait_seconds in ((signal.SIGTERM,30),(signal.SIGKILL,5)):
                for info in owned:
                    if same_process(info):
                        try: os.kill(info['pid'],sig)
                        except ProcessLookupError: pass
                deadline = time.monotonic()+wait_seconds
                while time.monotonic() < deadline and any(same_process(x) for x in owned):
                    time.sleep(.1)
            if worker:
                worker.join(5)
            if launcher:
                try: launcher.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    launcher.terminate()
                    try: launcher.wait(timeout=5)
                    except subprocess.TimeoutExpired: launcher.kill(); launcher.wait(timeout=5)
        except BaseException as error:
            receipt['cleanup_failure_kind'] = type(error).__name__
        finally:
            config_path.unlink(missing_ok=True)
            for handle in logs: handle.close()
            receipt['cleanup'] = {'temporary_config_removed':not config_path.exists(),
                'owned_processes_remaining':[x['pid'] for x in owned if same_process(x)],
                'original_config_unchanged':a.config.read_bytes()==original,
                'workload_thread_finished':worker is None or not worker.is_alive()}
            receipt['credential_scan'] = runner.credential_scan(out,key)
            receipt['harness_unchanged'] = all(sha(a.harness/k)==v for k,v in FROZEN.items())
            receipt['ended'] = stamp()
            save(out/'receipt.json',receipt)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            os.umask(old_umask)
    print(json.dumps({'receipt':str(out/'receipt.json'),'report_exists':receipt.get('report_exists',False),
                      'workload_qualified':receipt.get('owned_workload_qualified',False),
                      'failure_kind':receipt.get('failure_kind'),'no_performance_claim':True}))
    return 0 if (not receipt.get('failure_kind') and not receipt.get('stop_failure_kind') and not receipt.get('cleanup_failure_kind') and
                 receipt.get('owned_workload_qualified') and not receipt['credential_scan']['credential_matches'] and
                 not receipt['cleanup']['owned_processes_remaining'] and receipt['cleanup']['original_config_unchanged'] and
                 receipt['cleanup']['workload_thread_finished'] and receipt['harness_unchanged']) else 2

def self_test():
    for engine,overwrite in ((ENGINE,'0'),(V4_ENGINE,'0'),(V4_ENGINE,'1')):
        validate_selection(engine,overwrite)
    for engine,overwrite in ((ENGINE,'1'),(V4_ENGINE,'2'),('0'*64,'0')):
        try:validate_selection(engine,overwrite)
        except RuntimeError:pass
        else:raise AssertionError('Invalid engine/overwrite selection accepted')
    c=commands('/nsys','strata_diag_test',Path('/private/wrapper'),Path('/private/report'))
    assert '--sample=none' in c['start'] and '--cpuctxsw=none' in c['start']
    assert '--cuda-event-trace=false' in c['launch'] and '--cuda-graph-trace=node' in c['launch']
    assert not any('attach' in x for cmd in c.values() for x in cmd)
    event=threading.Event(); log=MarkerLog(io.StringIO(),event)
    log.write('READY finite C4 primary\n'); assert not event.is_set()
    log.write('INJECT primary-injected-1 actual_'); assert not event.is_set()
    log.write('tokens=60000\n'); assert event.is_set()
    print(json.dumps({'self_test':'PASS','network_requests':0,'gpu_operations':0,'subprocesses':0}))
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true'); p.add_argument('--self-test',action='store_true')
    p.add_argument('--exec-target',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--out',type=Path); p.add_argument('--exe',type=Path); p.add_argument('--harness',type=Path)
    p.add_argument('--engine-sha256',default=ENGINE,choices=(ENGINE,V4_ENGINE))
    p.add_argument('--snapshot-overwrite',default='0',choices=('0','1'))
    p.add_argument('--config',type=Path,default=Path('/srv/Strata/strata-orca-q5_k_m.57.json'))
    p.add_argument('--server',type=Path,default=Path('/srv/Strata-0157/serve/server.py'))
    p.add_argument('--python',default='/srv/Strata/.venv/bin/python')
    p.add_argument('--nsys',type=Path,default=Path('/usr/local/bin/nsys'))
    p.add_argument('--calibration',type=Path); p.add_argument('--workload',choices=('c8','mixed'),default='c8')
    p.add_argument('--arm',choices=('off','on'),default='on'); p.add_argument('--seconds',type=int,default=15)
    p.add_argument('--ready-timeout',type=int,default=1800)
    a=p.parse_args()
    if a.exec_target: exec_target(a.exec_target); return 1
    if a.self_test:return self_test()
    validate_selection(a.engine_sha256,a.snapshot_overwrite)
    require(12<=a.seconds<=15,'Capture must be12..15seconds')
    if not a.execute:
        print(json.dumps({'mode':'DRY_RUN_NO_NETWORK_NO_GPU_NO_FILES','engine_sha256':a.engine_sha256,'config_sha256':CONFIG,
            'snapshot_overwrite':a.snapshot_overwrite,
            'workload':a.workload,'seconds':a.seconds,'arm':a.arm,'port':8092,
            'commands':commands(a.nsys,'strata_diag_UNIQUE',Path('/PRIVATE/server-wrapper.sh'),Path('/PRIVATE/timeline')),
            'requires':'Operator-stopped8080,empty GPUs,closed8092,exact frozen inputs; own injected launch, no existing PID attach',
            'no_performance_claim':True},indent=2));return 0
    return runtime(a)

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'failure_kind':type(error).__name__,'executed_service_actions':'See receipt if created'}))
        raise SystemExit(1)
