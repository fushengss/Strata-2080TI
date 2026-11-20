#!/usr/bin/env python3
"""Isolated 100k decode Nsight profile; run ONLY with every GPU free.

Starts an nsys-injected Python server on loopback:8091, loads the supplied config,
then captures ten seconds after the first nonempty completion delta. Never touches
the service on :8080. Credentials exist only in memory and a mode-0600 /dev/shm
config, removed on exit. Traced throughput is not a benchmark result.

python3 profile.py --config /path/to/candidate.json --out /srv/profile-candidate
"""
import argparse
import collections
import json
import os
import re
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, check=True, **kwargs)


def gpu_processes():
    return run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], timeout=15).stdout.strip()


def union_ns(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start > right:
            total += right-left
            left, right = start, end
        else:
            right = max(right, end)
    return total + (right-left if left is not None else 0)


def summarize(sqlite_file):
    with sqlite3.connect(sqlite_file) as db:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        table = 'CUPTI_ACTIVITY_KIND_KERNEL'
        if table not in names:
            raise RuntimeError('No CUDA kernel table in trace; check child-process injection')
        cols = {r[1] for r in db.execute('PRAGMA table_info(' + table + ')')}
        namecol = 'demangledName' if 'demangledName' in cols else 'shortName'
        rows = list(db.execute(f'SELECT k.deviceId,k.start,k.end,s.value FROM {table} k JOIN StringIds s ON s.id=k.{namecol} ORDER BY k.start'))
    if not rows:
        raise RuntimeError('CUDA kernel trace is empty')
    span = max(r[2] for r in rows) - min(r[1] for r in rows)
    gpu = collections.defaultdict(list)
    kernels = collections.defaultdict(lambda: [0, 0])
    for dev, start, end, name in rows:
        gpu[dev].append((start, end))
        kernels[(dev, name)][0] += 1
        kernels[(dev, name)][1] += end-start
    return {'kernel_event_span_ms': span/1e6,
            'note': 'Kernel sum can overlap across streams; busy_union merges intervals per GPU. These are traced kernel execution times, not SM occupancy or uninstrumented decode latency.',
            'per_gpu': [{'device': dev, 'kernels': len(iv), 'kernel_sum_ms': sum(b-a for a,b in iv)/1e6,
                         'kernel_busy_union_ms': union_ns(iv)/1e6, 'kernel_busy_fraction_of_event_span': union_ns(iv)/span}
                        for dev,iv in sorted(gpu.items())],
            'kernel_by_gpu': [{'device': dev, 'kernel': name, 'count': count, 'total_ms': ns/1e6, 'mean_us': ns/count/1e3}
                              for (dev,name),(count,ns) in sorted(kernels.items(), key=lambda x: -x[1][1])]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--ready-timeout', type=float, default=900)
    a = p.parse_args()
    if not 1 <= a.seconds <= 20:
        p.error('--seconds must be 1..20')
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('--out must be new or empty; avoid overwriting evidence')
    if gpu_processes():
        raise SystemExit('GPU processes are active. Stop the authorized test/live service before invoking this script.')
    # Refuse an occupied test port; never stop an existing service.
    import socket
    probe = socket.socket()
    try:
        probe.bind(('127.0.0.1', 8091))
    finally:
        probe.close()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    key = cfg['api_key']
    cfg.update(host='127.0.0.1', port=8091, log=str(out/'engine.log'))
    session = 'strata100k_' + uuid.uuid4().hex[:10]
    meta = {'session': session, 'start_epoch': time.time(), 'capture_requested_s': a.seconds,
            'nsys_version': run(['nsys','--version']).stdout.strip(),
            'config': {k:cfg[k] for k in ('exe','args','env','gpu','parallel','layer_split') if k in cfg},
            'warning': 'Profile timings include CUDA graph node tracing overhead; do not report as baseline performance.'}
    config_path, launcher, timer = None, None, None
    active = False
    launch_started = False
    lock = threading.Lock()
    server_log = None

    def stop_capture():
        nonlocal active
        with lock:
            if not active:
                return
            meta['capture_stop_called_epoch'] = time.time()
            r = subprocess.run(['nsys','stop','--session='+session], capture_output=True, text=True, timeout=180)
            (out/'nsys-stop.log').write_text(r.stdout+r.stderr)
            meta['capture_stop_returncode'] = r.returncode
            active = False
            meta['capture_stop_finished_epoch'] = time.time()

    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='strata-profile-', suffix='.json', dir='/dev/shm', delete=False) as f:
            config_path = Path(f.name)
            os.chmod(config_path, 0o600)
            json.dump(cfg, f)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(str(x) for x in cfg.get('gpu', range(8))))
        server_log = open(out/'server.log','w')
        command = ['nsys','launch','--session-new='+session,'--trace=cuda,nvtx,osrt','--cuda-graph-trace=node',
                   '--cuda-event-trace=false','--osrt-threshold=10000','--show-output=true','--wait=all',
                   '/srv/Strata/.venv/bin/python','/srv/Strata-0157/serve/server.py','--engine','strata',
                   '--config',str(config_path),'--port','8091']
        launcher = subprocess.Popen(command, cwd='/srv/Strata-0157', env=env, stdout=server_log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        launch_started = True
        deadline, next_update = time.monotonic()+a.ready_timeout, 0
        while time.monotonic() < deadline:
            if 'ready: http' in (out/'server.log').read_text(errors='replace'):
                break
            rc = launcher.poll()
            if rc not in (None, 0):
                raise RuntimeError(f'nsys launcher exited {rc}; see server.log')
            if time.monotonic() >= next_update:
                print('Waiting for injected server to load...', flush=True)
                next_update = time.monotonic()+30
            time.sleep(1)
        else:
            raise TimeoutError('Injected server did not become ready')
        meta['ready_epoch'] = time.time()
        print('Injected server ready; sending fresh ~100k prompt', flush=True)
        words = ('the river carried old maps past quiet towns where people traded stories about storms, bridges, lanterns, '
                 'harvests, letters, clocks, ships and distant mountains while children counted stars above the harbor').split()
        filler = ' '.join(words[(i*7+8)%len(words)] + ('.' if i%17 == 16 else '') for i in range(int(100000/1.3)))
        task = ('Ignoring the text above, write a complete Python module implementing a persistent, thread-safe LRU cache. '
                'Include get, put, delete, clear, TTL expiry, atomic JSON checkpoint save/load, statistics, capacity resizing, '
                'a context manager, validation of inputs, and extensive unittest coverage for at least 30 cases including '
                'concurrency and malformed checkpoints. Include type annotations and docstrings. Output all code, with no placeholders.')
        body = {'model': cfg.get('model_name','x'), 'messages':[{'role':'user','content':f'[{session}] '+filler+'\n\n'+task}],
                'max_tokens':4096,'temperature':0,'presence_penalty':0,'frequency_penalty':0,'repetition_penalty':1.0,
                'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}
        meta['request_start_epoch'] = time.time()
        req = urllib.request.Request('http://127.0.0.1:8091/v1/chat/completions',data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with urllib.request.urlopen(req, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:') or line == 'data: [DONE]':
                    continue
                event = json.loads(line[5:])
                if event.get('usage'):
                    meta['usage'] = event['usage']
                for ch in event.get('choices',[]):
                    dl = ch.get('delta',{})
                    piece = dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning')
                    if piece and 'first_content_epoch' not in meta:
                        meta['first_content_epoch'] = time.time()
                        print('First content received; starting CUDA graph node trace', flush=True)
                        started = run(['nsys','start','--session='+session,'--sample=none','--cpuctxsw=process-tree',
                                       '--output='+str(out/'decode')],timeout=60)
                        (out/'nsys-start.log').write_text(started.stdout+started.stderr)
                        meta['capture_start_finished_epoch'] = time.time()
                        active = True
                        timer = threading.Timer(a.seconds, stop_capture)
                        timer.start()
                    if ch.get('finish_reason'):
                        meta['finish_reason'] = ch['finish_reason']
        meta['request_finished_epoch'] = time.time()
        if timer is not None:
            timer.cancel()
        stop_capture()
        if timer is not None:
            timer.join()
        print('Request and capture finished; exporting trace', flush=True)
        report = out/'decode.nsys-rep'
        if not report.exists():
            raise RuntimeError('Expected decode.nsys-rep missing; see nsys-stop.log')
        exported = run(['nsys','export','--type=sqlite','--output='+str(out/'decode.sqlite'),str(report)],timeout=180)
        (out/'nsys-export.log').write_text(exported.stdout+exported.stderr)
        stats = run(['nsys','stats','--report=cuda_gpu_kern_sum,cuda_api_sum,osrt_sum','--format=csv',str(out/'decode.sqlite')],timeout=180)
        (out/'nsys-stats.csv').write_text(stats.stdout)
        (out/'nsys-stats.stderr').write_text(stats.stderr)
        summary = summarize(out/'decode.sqlite')
        (out/'gpu-summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary['per_gpu'],indent=2),flush=True)
    finally:
        if timer is not None:
            timer.cancel()
        try:
            stop_capture()
        except Exception as e:
            meta['capture_cleanup_error'] = repr(e)
        finally:
            if timer is not None:
                timer.join()
        if launch_started:
            try:
                ended = subprocess.run(['nsys','shutdown','--session='+session,'--kill=sigterm'],capture_output=True,text=True,timeout=120)
                (out/'nsys-shutdown.log').write_text(ended.stdout+ended.stderr)
                # Only this script's launcher process group is eligible for fallback cleanup.
                if launcher.poll() is None:
                    try:
                        launcher.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(launcher.pid,signal.SIGTERM)
                        launcher.wait(timeout=30)
                for _ in range(60):
                    if not gpu_processes():
                        break
                    time.sleep(1)
                meta['remaining_gpu_processes'] = gpu_processes()
            except Exception as e:
                meta['server_cleanup_error'] = repr(e)
        if server_log is not None:
            server_log.close()
        if config_path is not None:
            config_path.unlink(missing_ok=True)
        meta['end_epoch'] = time.time()
        (out/'metadata.json').write_text(json.dumps(meta,indent=2))
        print('Profile cleanup complete; remaining GPU processes:', meta.get('remaining_gpu_processes','not launched'),flush=True)
    if meta.get('remaining_gpu_processes') or meta.get('server_cleanup_error'):
        raise RuntimeError('Profile cleanup needs attention; see metadata.json')


if __name__ == '__main__':
    main()
