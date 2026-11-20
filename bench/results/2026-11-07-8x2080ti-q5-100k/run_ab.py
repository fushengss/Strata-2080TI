#!/usr/bin/env python3
"""Maintenance-only isolated A/B runner. Always restores the original live service."""
import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

p = Path('/srv/bench100k-20261107-candidates')
ap = argparse.ArgumentParser()
ap.add_argument('--arms', default='baseline,v2k,groups8,combo')
args = ap.parse_args()
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
active = None


def say(*s):
    print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), *s, flush=True)


def gpu_pids():
    s = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(x.strip()) for x in s.splitlines() if x.strip().isdigit()]


def wait_ready(config, port, timeout=1200):
    c = json.loads(Path(config).read_text())
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/status', headers={'Authorization': 'Bearer '+c['api_key']})
            with urllib.request.urlopen(r, timeout=5) as f:
                s = json.load(f)
            if s.get('loaded'):
                return s
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f'port {port} did not become ready')


def stop(pid):
    if pid is None:
        return
    say('stopping test server', pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(120):
        if not gpu_pids():
            return
        time.sleep(1)
    # Only stop the candidate session and its known GPU processes in this exclusive maintenance window.
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for q in gpu_pids():
        cmd = Path(f'/proc/{q}/cmdline').read_bytes().replace(b'\0', b' ')
        if b'/srv/Strata-0157/engine/strata' in cmd or b'/srv/Strata/engine/strata-vision' in cmd:
            os.kill(q, signal.SIGKILL)
    for _ in range(30):
        if not gpu_pids():
            return
        time.sleep(1)
    raise RuntimeError('candidate GPUs did not release')


def start(config, port, logfile):
    f = open(logfile, 'a')
    s = subprocess.Popen(['/srv/Strata/.venv/bin/python', '/srv/Strata-0157/serve/server.py', '--engine', 'strata',
                          '--config', str(config), '--port', str(port)], cwd='/srv/Strata-0157', env=env,
                         stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    f.close()
    return s.pid


try:
    for arm in args.arms.split(','):
        config = p/(arm+'.json')
        if arm == 'baseline' and (p/'baseline.pid').exists():
            active = int((p/'baseline.pid').read_text())
        else:
            assert not gpu_pids(), 'unexpected GPU processes before test start'
            active = start(config, 8091, p/(arm+'.server.log'))
        (p/'active.json').write_text(json.dumps({'arm':arm,'pid':active,'phase':'loading'}))
        say('loading', arm, 'pid', active)
        wait_ready(config, 8091)
        say('ready', arm)
        (p/'active.json').write_text(json.dumps({'arm':arm,'pid':active,'phase':'benchmark'}))
        with open(p/(arm+'.client.log'),'w') as f:
            result = subprocess.run(['python3', str(p/'ab_client.py'), '--config', str(config), '--out', str(p/arm)],
                                    stdout=f, stderr=subprocess.STDOUT, timeout=1500)
        if result.returncode:
            raise RuntimeError(f'{arm} client failed: {result.returncode}')
        say('complete', arm)
        stop(active)
        active = None
finally:
    stop(active)
    say('restoring original live config')
    live = Path('/srv/Strata/strata-orca-q5_k_m.57.json')
    assert live.read_bytes() == (p/'original-live.json').read_bytes(), 'live config changed externally; restore manually'
    pid = start(live, 8080, '/srv/Strata/serve-orca.out')
    (p/'restored.pid').write_text(str(pid))
    wait_ready(live, 8080)
    (p/'active.json').write_text(json.dumps({'phase':'restored','pid':pid}))
    say('original service restored',pid)
