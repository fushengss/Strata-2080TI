#!/usr/bin/env python3
"""Prepared only: collect a short trace from an ALREADY nsys-launched server.

The request JSON should contain a fresh ~100k prompt and a task long enough for
at least 10-20s of decode (e.g. max_tokens=4096). First actual SSE content starts
the capture; prefill/model load are excluded. No service configuration changes.
Tracing changes timings: use this for kernel attribution, not throughput claims.
"""
import argparse
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--session', required=True)
p.add_argument('--request', required=True, help='OpenAI chat request JSON, no credentials')
p.add_argument('--port', type=int, default=8091)
p.add_argument('--seconds', type=float, default=12)
p.add_argument('--out', required=True, help='new absolute report prefix')
a = p.parse_args()
if not 1 <= a.seconds <= 20:
    p.error('--seconds must be 1..20')
out = Path(a.out)
if not out.is_absolute():
    p.error('--out must be absolute')
out.parent.mkdir(parents=True, exist_ok=True)
cfg = json.loads(Path(a.config).read_text())
body = json.loads(Path(a.request).read_text())
body.update(stream=True, stream_options={'include_usage': True})
headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + cfg['api_key']}
req = urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',
                             data=json.dumps(body).encode(), headers=headers)
meta = {'session': a.session, 'requested_capture_s': a.seconds, 'request_start_epoch': time.time()}
lock = threading.Lock()
active = False
timer = None

def stop():
    global active
    with lock:
        if not active:
            return
        meta['stop_called_epoch'] = time.time()
        r = subprocess.run(['nsys', 'stop', '--session=' + a.session], capture_output=True, text=True, timeout=120)
        active = False
        meta['stop_returncode'] = r.returncode
        meta['stop_finished_epoch'] = time.time()
        out.with_suffix('.stop.log').write_text(r.stdout + r.stderr)

try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        # Deliberately iterate SSE lines, never read a fixed-size block: sparse
        # output can otherwise postpone the first-token marker until completion.
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            event = json.loads(line[5:])
            if event.get('usage'):
                meta['usage'] = event['usage']
            for ch in event.get('choices', []):
                dl = ch.get('delta', {})
                piece = dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning')
                if piece and 'first_content_epoch' not in meta:
                    meta['first_content_epoch'] = time.time()
                    started = subprocess.run(['nsys', 'start', '--session=' + a.session,
                                              '--sample=none', '--cpuctxsw=process-tree',
                                              '--output=' + str(out)], capture_output=True, text=True, timeout=60)
                    out.with_suffix('.start.log').write_text(started.stdout + started.stderr)
                    meta['start_returncode'] = started.returncode
                    if started.returncode:
                        raise RuntimeError('nsys start failed; see start.log')
                    meta['start_finished_epoch'] = time.time()
                    active = True
                    timer = threading.Timer(a.seconds, stop)
                    timer.start()
finally:
    if timer is not None:
        timer.cancel()
    stop()
    if timer is not None:
        timer.join()
    meta['request_finished_epoch'] = time.time()
    out.with_suffix('.capture.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
