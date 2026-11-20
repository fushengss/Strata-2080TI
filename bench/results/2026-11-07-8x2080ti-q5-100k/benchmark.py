#!/usr/bin/env python3
"""Non-disruptive benchmark of the existing Q5 service; never restarts or edits it.

Run on the server: python3 benchmark.py --out /srv/bench100k-20261107
Credentials are read from the config and never written to the results.
Synthetic prompts deliberately change at the beginning to avoid prefix reuse.
"""
import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='/srv/Strata/strata-orca-q5_k_m.57.json')
ap.add_argument('--out', required=True)
ap.add_argument('--port', type=int, default=8080)
ap.add_argument('--repeats', type=int, default=2)
ap.add_argument('--idle-wait', type=int, default=60)
ap.add_argument('--kinds', default='code,prose,thinking')
args = ap.parse_args()
cfg = json.loads(Path(args.config).read_text())
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{args.port}'
headers = {'Authorization': 'Bearer ' + cfg['api_key'], 'Content-Type': 'application/json'}


def status():
    try:
        req = urllib.request.Request(base + '/v1/status', headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r)['activity']
    except Exception as e:
        return {'error': type(e).__name__}


def monitor(stop, samples, timeline, log, offset):
    with log.open() as f:
        f.seek(offset)
        while not stop.is_set():
            samples.append({'time': time.time(), **status()})
            for line in f:
                if re.search(r'prompt \d+ tokens|strata batch: slot .*takes|strata batch \(pipelined', line):
                    timeline.append({'time': time.time(), 'line': line.rstrip()})
            stop.wait(0.5)


words = ('the river carried old maps past quiet towns where people traded stories about storms, bridges, lanterns, '
         'harvests, letters, clocks, ships and distant mountains while children counted stars above the harbor').split()
filler = ' '.join(words[(i * 7 + 8) % len(words)] + ('.' if i % 17 == 16 else '') for i in range(int(100000 / 1.3)))
tasks = {
    'code': 'Write a Python module with a class LRUCache (get, put, delete, __len__) built on an OrderedDict, a thread-safe wrapper class using a lock, and a unittest suite with at least six tests. Output only code.',
    'prose': 'Write a detailed 900-word essay about harbors, their history, and life in coastal towns.',
    'thinking': 'Design a thread-safe LRU cache in Python. Reason about concurrent get/put/delete operations, invariants, lock granularity, and edge cases; then provide an implementation and tests.',
}
meta = {k: cfg[k] for k in ('exe', 'args', 'gpu', 'parallel', 'layer_split', 'env', 'model_name') if k in cfg}
meta['engine_sha256'] = hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()
meta['method'] = '100000 nominal tokens of synthetic repeated text; actual token count from API usage and engine logs; 768 output tokens; fresh prefixes; no service changes'
(out / 'metadata.json').write_text(json.dumps(meta, indent=2))
gpu_file = open(out / 'gpu.csv', 'w')
gpu = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv', '-lms', '500'], stdout=gpu_file)
results = []
try:
    for rep in range(args.repeats):
        for kind in args.kinds.split(','):
            deadline = time.monotonic() + args.idle_wait
            while status().get('in_flight', 1) and time.monotonic() < deadline:
                time.sleep(2)
            before = status()
            run = f'{kind}-{rep}-{time.time_ns()}'
            prompt = f'[{run}] ' + filler + '\n\nIgnoring the text above, ' + tasks[kind]
            body = {'model': cfg['model_name'], 'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 768, 'stream': True, 'stream_options': {'include_usage': True},
                    'presence_penalty': 0, 'frequency_penalty': 0, 'repetition_penalty': 1.0,
                    'chat_template_kwargs': {'enable_thinking': kind == 'thinking'}}
            if kind != 'thinking':
                body['temperature'] = 0
            log = Path(cfg['log'])
            offset = log.stat().st_size
            samples, pieces, usage, finish, events, timeline = [], [], {}, None, [], []
            stop = threading.Event()
            th = threading.Thread(target=monitor, args=(stop, samples, timeline, log, offset), daemon=True)
            th.start()
            req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers=headers)
            start_wall, start = time.time(), time.perf_counter()
            first, last = None, None
            print('START', run, 'activity', before, flush=True)
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith('data:') or line == 'data: [DONE]':
                        continue
                    d = json.loads(line[5:])
                    if d.get('usage'):
                        usage = d['usage']
                    for ch in d.get('choices') or []:
                        dl = ch.get('delta') or {}
                        s = dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning')
                        if s:
                            last = time.perf_counter() - start
                            first = last if first is None else first
                            pieces.append(s)
                            events.append({'elapsed_s': last, 'chars': len(s)})
                        finish = ch.get('finish_reason') or finish
            elapsed = time.perf_counter() - start
            stop.set()
            th.join(6)
            after = status()
            with log.open() as f:
                f.seek(offset)
                lines = [s.rstrip() for s in f if re.search(r'prompt \d+ tokens|strata batch|pipeline|KV streaming:|suffix drafts:|resident RAM:', s)]
            n = usage.get('completion_tokens', 0)
            result = {'run': run, 'kind': kind, 'start_epoch': start_wall, 'end_epoch': time.time(),
                      'ttft_s': first, 'last_content_s': last, 'total_s': elapsed, 'usage': usage, 'finish_reason': finish,
                      'client_decode_tps': (n-1)/(last-first) if n>1 and last>first else None,
                      'activity_before': before, 'activity_after': after, 'activity_samples': samples,
                      'max_in_flight': max((s.get('in_flight', -1) for s in samples), default=-1),
                      'no_observed_competitor': before.get('in_flight') == 0 and all(s.get('in_flight', 99) <= 1 for s in samples) and after.get('requests', 0)-before.get('requests', 0) == 1,
                      'engine_lines': lines, 'chunk_events': events, 'engine_timeline': timeline,
                      'output_sha256': hashlib.sha256(''.join(pieces).encode()).hexdigest()}
            results.append(result)
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            print('DONE', json.dumps({k:v for k,v in result.items() if k not in ('activity_samples','engine_lines','chunk_events','engine_timeline')}), flush=True)
finally:
    gpu.terminate()
    gpu.wait(timeout=10)
    gpu_file.close()
