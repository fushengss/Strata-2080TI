#!/usr/bin/env python3
"""Client-only, reproducible maintenance A/B benchmark. Never manages a server.

Inputs are synthetic and identical between arms. API credentials are only read
from --config. Output includes full synthetic generations and raw SSE deltas.
Each completed request is atomically saved and fsynced. Requires only stdlib.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = ('the river carried old maps past quiet towns where people traded stories about storms, bridges, lanterns, '
         'harvests, letters, clocks, ships and distant mountains while children counted stars above the harbor').split()
TASKS = {
    'code': 'Write a Python module with a class LRUCache (get, put, delete, __len__) built on an OrderedDict, a thread-safe wrapper class using a lock, and a unittest suite with at least six tests. Output only code.',
    'prose': 'Write a detailed 900-word essay about harbors, their history, and life in coastal towns.',
}
PROMPT_RE = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)')
LOG_RE = re.compile(r'prompt \d+ tokens|strata batch|pipeline|decode head|MTP head|KV streaming:|suffix drafts:|resident RAM:|drafts accepted|ERROR|WARNING')


def filler(nominal_tokens):
    return ' '.join(WORDS[(i * 7 + 8) % len(WORDS)] + ('.' if i % 17 == 16 else '')
                    for i in range(int(nominal_tokens / 1.3)))


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_cursor(path):
    if not path:
        return None
    try:
        s = path.stat()
        return (s.st_ino, s.st_size)
    except OSError:
        return None


def read_log(path, cursor):
    if path is None or cursor is None:
        return []
    try:
        st = path.stat()
        with path.open('rb') as f:
            if st.st_ino == cursor[0] and st.st_size >= cursor[1]:
                f.seek(cursor[1])
            return [s for s in f.read().decode(errors='replace').splitlines() if LOG_RE.search(s)]
    except OSError:
        return []


def parse_prompt_lines(lines):
    records = []
    for line in lines:
        m = PROMPT_RE.search(line)
        if m:
            vals = m.groups()
            records.append(dict(zip(
                ('prompt_tokens', 'reused_tokens', 'read_tokens', 'prefill_ms', 'prefill_tps',
                 'generated_tokens', 'decode_ms', 'decode_tps'),
                (int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), float(vals[4]),
                 int(vals[5]), int(vals[6]), float(vals[7])))))
    return records


class Bench:
    def __init__(self, args):
        self.args = args
        self.cfg = json.loads(Path(args.config).read_text())
        self.secret = self.cfg.get('api_key', '')
        self.base = f'http://127.0.0.1:{args.port}'
        self.headers = {'Content-Type': 'application/json'}
        if self.secret:
            self.headers['Authorization'] = 'Bearer ' + self.secret
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = Path(self.cfg['log']) if self.cfg.get('log') else None
        self.lock = threading.Lock()
        self.records = []

    def safe(self, value):
        return value.replace(self.secret, '[REDACTED]') if self.secret else value

    def status(self):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + '/v1/status', headers=self.headers), timeout=5) as r:
                d = json.load(r)
                return d.get('activity', {})
        except Exception as e:
            return {'error': type(e).__name__}

    def request(self, name, messages, max_tokens, category, log_owned=True, overrides=None):
        body = {'model': self.cfg.get('model_name', 'strata'), 'messages': messages,
                'max_tokens': max_tokens, 'temperature': 0, 'seed': 20261107,
                'stream': True, 'stream_options': {'include_usage': True},
                'presence_penalty': 0, 'frequency_penalty': 0, 'repetition_penalty': 1.0,
                'chat_template_kwargs': {'enable_thinking': False}}
        body.update(overrides or {})
        rawbody = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
        before = self.status()
        cursor = log_cursor(self.log)
        start_epoch, start = time.time(), time.perf_counter()
        first = last = None
        chunks, pieces, reasoning, usage, finish = [], [], [], {}, None
        error = None
        saw_done = False
        print(f'START {name}', flush=True)
        event_path = self.out / (name + '.sse.jsonl')
        try:
            with event_path.open('w') as events:
                req = urllib.request.Request(self.base + '/v1/chat/completions', data=rawbody, headers=self.headers)
                with urllib.request.urlopen(req, timeout=1200) as response:
                    for raw in response:
                        line = raw.decode(errors='replace').strip()
                        if not line.startswith('data:'):
                            continue
                        now = time.perf_counter() - start
                        payload = line[5:].strip()
                        if payload == '[DONE]':
                            saw_done = True
                            break
                        d = json.loads(payload)
                        event = {'elapsed_s': now, 'epoch': start_epoch + now, 'data': d}
                        events.write(json.dumps(event, ensure_ascii=False) + '\n')
                        events.flush()
                        if d.get('error'):
                            error = {'kind': 'api_error', 'detail': self.safe(json.dumps(d['error'], ensure_ascii=False))}
                        if d.get('usage'):
                            usage = d['usage']
                        for choice in d.get('choices') or []:
                            dl = choice.get('delta') or {}
                            content = dl.get('content') or ''
                            thought = dl.get('reasoning_content') or dl.get('reasoning') or ''
                            if content or thought:
                                first = now if first is None else first
                                last = now
                                pieces.append(content)
                                reasoning.append(thought)
                                chunks.append({'elapsed_s': now, 'epoch': start_epoch + now,
                                               'content': content, 'reasoning': thought})
                            finish = choice.get('finish_reason') or finish
                events.flush()
                os.fsync(events.fileno())
        except Exception as e:
            error = {'kind': type(e).__name__, 'detail': self.safe(str(e))}
        elapsed = time.perf_counter() - start
        lines = read_log(self.log, cursor) if log_owned else []
        text = ''.join(pieces)
        thinking = ''.join(reasoning)
        n = usage.get('completion_tokens', 0)
        decode_s = last - first if last is not None and first is not None else None
        record = {
            'name': name, 'category': category, 'request_sha256': hashlib.sha256(rawbody).hexdigest(),
            'messages_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
            'max_tokens': max_tokens, 'start_epoch': start_epoch, 'end_epoch': start_epoch + elapsed,
            'ttft_s': first, 'last_content_s': last, 'total_s': elapsed, 'usage': usage,
            'finish_reason': finish, 'finished_at_length': finish == 'length',
            'output_count_matches_limit': n == max_tokens, 'saw_done': saw_done, 'error': error,
            'output_text': text, 'reasoning_text': thinking, 'output_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'reasoning_sha256': hashlib.sha256(thinking.encode()).hexdigest(),
            'combined_output_sha256': hashlib.sha256(json.dumps([text, thinking], ensure_ascii=False).encode()).hexdigest(),
            'request_settings': {k:v for k,v in body.items() if k != 'messages'},
            'delta_count': len(chunks), 'deltas': chunks,
            'decode_tps_usage_minus_one_estimate': (n - 1) / decode_s if n > 1 and decode_s and decode_s > 0 else None,
            'decode_metric_note': 'API completion tokens minus one divided by first-to-last content arrival; estimated because a chunk may contain multiple tokens.',
            'activity_before': before, 'activity_after': self.status(),
            'engine_lines': [self.safe(s) for s in lines], 'engine_prompt_records': parse_prompt_lines(lines),
        }
        if not usage:
            record['measurement_warning'] = 'No usage token counts; no token throughput inferred from text/chunks.'
        with self.lock:
            save_json(self.out / (name + '.json'), record)
            self.records.append(record)
            save_json(self.out / 'results.json', self.records)
        print('DONE ' + json.dumps({k: record[k] for k in
              ('name', 'ttft_s', 'total_s', 'usage', 'finish_reason', 'error', 'decode_tps_usage_minus_one_estimate')}, ensure_ascii=False), flush=True)
        return record

    def cohort(self, name, prompts, tokens):
        cursor = log_cursor(self.log)
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = []
            for i, messages in enumerate(prompts):
                futures.append(executor.submit(self.request, f'{name}-{i + 1}', messages, tokens, name, False))
                if i + 1 < len(prompts):
                    time.sleep(0.05)
            records = [f.result() for f in futures]
        end = max(r['end_epoch'] for r in records)
        total = sum(r['usage'].get('completion_tokens', 0) for r in records)
        firsts = [r['start_epoch'] + r['ttft_s'] for r in records if r['ttft_s'] is not None]
        lasts = [r['start_epoch'] + r['last_content_s'] for r in records if r['last_content_s'] is not None]
        common_begin = max(firsts) if len(firsts) == len(records) else None
        common_end = min(lasts) if len(lasts) == len(records) else None
        common_duration = max(0, common_end - common_begin) if common_begin is not None and common_end is not None else 0
        within = sum(1 for r in records for d in r['deltas']
                     if common_duration and common_begin < d['epoch'] <= common_end)
        lines = read_log(self.log, cursor)
        summary = {
            'name': name, 'concurrency': len(records), 'start_epoch': start, 'end_epoch': end,
            'wall_s': end - start, 'completion_tokens': total,
            'aggregate_completion_tps_over_cohort_wall': total / (end - start),
            'all_usage_present': all(bool(r['usage']) for r in records),
            'error_count': sum(r['error'] is not None for r in records),
            'length_finish_count': sum(r['finished_at_length'] for r in records),
            'common_decode_window_start_epoch': common_begin, 'common_decode_window_end_epoch': common_end,
            'common_decode_window_s': common_duration, 'common_decode_content_chunks': within,
            'common_decode_chunks_per_s': within / common_duration if common_duration else None,
            'common_decode_tokens_per_s': None,
            'common_window_note': 'SSE chunks are not tokens. Without per-delta token IDs/counts, exact common-window token throughput is unavailable; raw deltas retained.',
            'requests': [{k: r[k] for k in ('name', 'start_epoch', 'end_epoch', 'ttft_s', 'total_s', 'usage', 'finish_reason', 'output_sha256', 'error')} for r in records],
            'engine_lines': [self.safe(s) for s in lines], 'engine_prompt_records': parse_prompt_lines(lines),
        }
        save_json(self.out / (name + '-summary.json'), summary)
        print('COHORT ' + json.dumps({k: summary[k] for k in ('name', 'wall_s', 'completion_tokens', 'aggregate_completion_tps_over_cohort_wall', 'error_count')}, ensure_ascii=False), flush=True)
        return summary

    def run(self):
        long_text = filler(100000)
        cold = []
        for tag, kind in (('A', 'code'), ('B', 'code'), ('C', 'prose')):
            cold.append('[Strata deterministic benchmark ' + tag + ' 20261107] ' + long_text +
                        '\n\nIgnoring the text above, ' + TASKS[kind])
        meta = {'fixture_version': 1, 'endpoint': self.base, 'created_epoch': time.time(),
                'synthetic_only': True, 'nominal_prompt_tokens': 100000,
                'prompt_note': 'WORDS fixture targets nominal 100k; exact prompt counts from API usage/engine logs (~101k).',
                'greedy': True, 'thinking': False, 'timeout_s': 1200,
                'config_fields': {k: self.cfg[k] for k in ('exe', 'gpu', 'parallel', 'layer_split', 'model_name') if k in self.cfg},
                'fixture_sha256': [hashlib.sha256(p.encode()).hexdigest() for p in cold]}
        save_json(self.out / 'metadata.json', meta)
        save_json(self.out / 'synthetic-prompts.json', {'cold': cold, 'append': filler(600)})
        first_result = None
        if 'single' in self.args.suites:
            for i, prompt in enumerate(cold):
                r = self.request(f'cold-{i + 1}', [{'role': 'user', 'content': prompt}], 768 if i < 2 else 512,
                                 'cold-code' if i < 2 else 'cold-prose')
                if i == 0:
                    first_result = r
            append_messages = [{'role': 'user', 'content': cold[0]},
                               {'role': 'assistant', 'content': first_result['output_text']},
                               {'role': 'user', 'content': filler(600) + '\nExplain how to extend the LRU cache with TTL expiry. Give Python code.'}]
            self.request('append-600', append_messages, 256, 'append')
        cohorts = []
        if 'c6' in self.args.suites:
            prompts = [[{'role': 'user', 'content': cold[0] +
                       f'\n\nIndependent task {i + 1}: Write a complete Python implementation with extensive tests and comments. Use class name CacheVariant{i + 1}. Output code only.'}] for i in range(6)]
            cohorts.append(self.cohort('c6-long', prompts, 1024))
        if 'c8' in self.args.suites:
            prompts = [[{'role': 'user', 'content': f'Independent short benchmark {i + 1}. '
                        + TASKS['code'] + f' Name the cache class ShortCache{i + 1}; add extensive tests and comments.'}] for i in range(8)]
            cohorts.append(self.cohort('c8-short', prompts, 1024))
        save_json(self.out / 'complete.json', {'complete_epoch': time.time(), 'request_count': len(self.records),
                  'error_count': sum(r['error'] is not None for r in self.records),
                  'missing_usage_count': sum(not r['usage'] for r in self.records), 'cohorts': cohorts})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=8091)
    ap.add_argument('--suites', default='single,c6,c8', help='comma-separated single,c6,c8')
    args = ap.parse_args()
    args.suites = args.suites.split(',')
    if set(args.suites) - {'single', 'c6', 'c8'}:
        ap.error('unknown suite')
    Bench(args).run()


if __name__ == '__main__':
    main()
