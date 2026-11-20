#!/usr/bin/env python3
"""Synthetic, client-only concurrent MTP correctness checks (stdlib only).

Dry-run by default. Execution needs --execute --exclusive-benchmark and an idle
loopback server. Never starts/stops a server or changes its configuration. Metrics
and telemetry are reduced to numeric allowlists; engine logs yield only numeric
pipelined-MTP counters. Full text is saved only for this script's synthetic SSEs.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import re
import socket
import threading
import time
from pathlib import Path

VERSION = 'piped-mtp-correctness-2'
LOGICAL_USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
BASE = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
        'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'repetition_penalty': 1.0,
        'chat_template_kwargs': {'enable_thinking': False}}
CODE = ('Write a complete Python module implementing a bounded FIFO task queue with priorities, '
        'cancellation, retries, a thread-safe wrapper and at least twelve unittest cases. '
        'Include every implementation and test in full; output only code.')
PROSE = ('Describe twenty unusual practical uses for a paper map. For each numbered use write '
         'at least eighty words explaining the materials, procedure, limitations and a concrete example. '
         'Do not abbreviate or replace remaining entries with a summary.')
NUMERIC_REQUEST_KEYS = ('time', 'duration_s', 'prompt_tokens', 'prompt_total', 'prompt_read', 'reused',
                        'output_tokens', 'engine_generated', 'first_engine_token_s', 'prompt_ms',
                        'decode_ms', 'decode_tok_s', 'drafts_offered', 'drafts_accepted')
PIPED_RE = re.compile(r'strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, '
                      r'(\d+) rows in ([\d.]+) ms = ([\d.]+) rows/s[^\n]*?'
                      r'\((\d+) of (\d+) MTP proposals kept\)')
ENV_KEYS = ('STRATA_BATCH_MTP', 'STRATA_PIPED_MTP', 'STRATA_PW_WITH_BATCH',
            'STRATA_DECODE_HEAD_GPU', 'STRATA_MTP_HEAD_GPU', 'STRATA_SPLIT_MTP_BATCH')


def digest(value):
    data = value.encode() if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numbers(value, keys):
    return {k: value[k] for k in keys if number(value.get(k))}


def numeric_tree(value):
    if number(value):
        return value
    if isinstance(value, dict):
        return {k: numeric_tree(v) for k, v in value.items()
                if number(v) or isinstance(v, dict)}
    return None


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def fixture(name, prompt, seed, limit, **overrides):
    return {'name': name, 'prompt': '[Synthetic concurrency correctness: ' + name + '] ' + prompt,
            'seed': seed, 'max_tokens': limit, 'overrides': overrides}


def fixtures():
    seeded = [fixture('seed-' + str(seed), CODE if i % 2 == 0 else PROSE, seed, 384 + 64*i)
              for i, seed in enumerate((17, 29, 101, 102))]
    mixed = [fixture('mixed-greedy', CODE, 113, 384, temperature=0),
             fixture('mixed-sampled', PROSE, 127, 448),
             fixture('mixed-penalties', PROSE, 139, 512, presence_penalty=1.1,
                     frequency_penalty=0.3, repetition_penalty=1.05),
             fixture('mixed-bias', CODE, 151, 576, logit_bias={'32': 1.0})]
    cancel = [fixture('cancel-stream', CODE, 303, 1536)] + [
        fixture('cancel-peer-' + str(i+1), CODE if i % 2 == 0 else PROSE, 313+i, 640+64*i)
        for i in range(3)]
    recovery = fixture('recovery-control', 'Reply with exactly MTP_RECOVERY_731 and nothing else.',
                       701, 32, temperature=0)
    return seeded, mixed, cancel, recovery


def usage_comparison(left, right):
    """Cache reuse changes accounting metadata, not the logical generated response."""
    a = {k: left[k] for k in LOGICAL_USAGE_KEYS if number(left.get(k))}
    b = {k: right[k] for k in LOGICAL_USAGE_KEYS if number(right.get(k))}
    am = {k: v for k, v in left.items() if k not in LOGICAL_USAGE_KEYS}
    bm = {k: v for k, v in right.items() if k not in LOGICAL_USAGE_KEYS}
    valid = len(a) == len(b) == len(LOGICAL_USAGE_KEYS)
    return {'logical_counts_equal': valid and a == b, 'logical_counts_present': valid,
            'baseline_logical_counts': a, 'candidate_logical_counts': b,
            'metadata_equal': am == bm, 'baseline_metadata': am, 'candidate_metadata': bm,
            'qualification': ('usage metadata identical' if am == bm else
                              'Usage metadata differs; cache reuse/accounting is reported separately from strict logical token counts and output hashes.')}


def completed(record):
    return (not record['error'] and record['saw_done'] and bool(record['usage'])
            and all(number(record['usage'].get(k)) for k in LOGICAL_USAGE_KEYS)
            and record['finish_reason'] in ('length', 'stop')
            and bool(record['output_text'] or record['reasoning_text']))


def overlap(records):
    starts = [r['first_output_epoch'] for r in records]
    ends = [r['last_output_epoch'] for r in records]
    begin = max(starts) if all(x is not None for x in starts) else None
    end = min(ends) if all(x is not None for x in ends) else None
    duration = max(0.0, end-begin) if begin is not None and end is not None else 0.0
    counts = {r['name']: sum(begin <= d['epoch'] <= end for d in r['deltas'])
              if duration else 0 for r in records}
    return {'first_to_last_common_start_epoch': begin, 'first_to_last_common_end_epoch': end,
            'common_decode_s': duration, 'deltas_per_request_in_common_window': counts,
            'four_streams_observed': len(records) == 4 and duration >= 0.1 and all(n >= 2 for n in counts.values()),
            'scope': 'SSE arrival overlap; chunks can contain multiple tokens. No GPU concurrency inference.'}


class Client:
    def __init__(self, args):
        self.args = args
        self.cfg = json.loads(args.config.read_text())
        self.headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.cfg.get('api_key'):
            self.headers['Authorization'] = 'Bearer ' + self.cfg['api_key']
        self.model = self.cfg.get('model_name', 'strata')
        self.log = args.engine_log or (Path(self.cfg['log']) if self.cfg.get('log') else None)
        self.lock = threading.Lock()
        self.first = {}
        self.active = set()
        self.connections = {}
        self.monitor_stop = threading.Event()
        self.abort = threading.Event()
        self.samples = []
        self.records = []
        self.errors = 0

    def get(self, path):
        conn = http.client.HTTPConnection('127.0.0.1', self.args.port, timeout=10)
        try:
            conn.request('GET', path, headers=self.headers)
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError('Numeric endpoint HTTP ' + str(response.status))
            return json.loads(response.read())
        finally:
            conn.close()

    def activity(self):
        status = self.get('/v1/status')
        activity = numbers(status.get('activity', {}), ('in_flight', 'queued', 'waiting'))
        if 'in_flight' not in activity:
            raise RuntimeError('Status has no numeric in_flight; cannot establish exclusivity')
        return activity

    def metrics(self):
        raw = self.get('/metrics?requests=all')
        rows = {}
        for row in raw.get('requests', []):
            identity = digest(['id', str(row['id'])]) if row.get('id') is not None else digest(
                ['numeric-history', numbers(row, NUMERIC_REQUEST_KEYS)])
            rows[identity] = {'identity_sha256': identity, **numbers(row, NUMERIC_REQUEST_KEYS)}
        return {'server_epoch': raw.get('time') if number(raw.get('time')) else None,
                'totals': numbers(raw.get('totals', {}), ('since', 'requests', 'drafts_offered', 'drafts_accepted')),
                'live': numbers(raw.get('live', {}), ('running', 'queued', 'waiting', 'parallel')),
                'rows': rows}

    def idle(self, wait=0):
        deadline = time.monotonic() + wait
        while True:
            state = self.activity()
            metrics = self.metrics()
            values = list(state.values()) + [metrics['live'].get(k, 0) for k in ('running', 'queued', 'waiting')]
            if all(x == 0 for x in values):
                return metrics
            if time.monotonic() >= deadline:
                raise RuntimeError('Server is not idle; benchmark requires no other requests')
            time.sleep(0.2)

    def monitor(self):
        started = time.monotonic()
        while not self.monitor_stop.is_set():
            row = {'epoch': time.time()}
            with self.lock:
                row['own_in_flight'] = len(self.active)
            try:
                row['status'] = self.activity()
                tel = self.get('/v1/telemetry')
                row['work'] = numbers(tel.get('work', {}), ('active', 'queued', 'waiting'))
                row['slots'] = [numbers(s, ('slot', 'group', 'generated', 'prompt_read', 'prompt_tokens'))
                                for s in tel.get('work', {}).get('slots', [])]
            except Exception as e:
                row['error_kind'] = type(e).__name__
            self.samples.append(row)
            if time.monotonic()-started > self.args.timeout:
                self.abort.set()
                with self.lock:
                    connections = list(self.connections.values())
                for conn in connections:
                    sock = conn.sock if isinstance(conn, http.client.HTTPConnection) else conn
                    if sock:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                return
            self.monitor_stop.wait(0.4)

    def log_cursor(self):
        if self.log is None:
            return None
        try:
            st = self.log.stat()
            return st.st_ino, st.st_size
        except OSError:
            return None

    def log_numbers(self, cursor):
        if self.log is None or cursor is None:
            return {'available': False, 'records': []}
        try:
            st = self.log.stat()
            if st.st_ino != cursor[0] or st.st_size < cursor[1]:
                return {'available': False, 'rotated': True, 'records': []}
            records = []
            with self.log.open('rb') as f:
                f.seek(cursor[1])
                for raw in f:
                    match = PIPED_RE.search(raw.decode(errors='replace'))
                    if match:
                        g = match.groups()
                        records.append(dict(zip(('groups', 'slots_per_group', 'group_steps', 'rows',
                                                 'wall_ms', 'rows_per_s', 'drafts_accepted', 'drafts_offered'),
                                                (int(g[0]), int(g[1]), int(g[2]), int(g[3]), float(g[4]),
                                                 float(g[5]), int(g[6]), int(g[7])))))
            return {'available': True, 'records': records}
        except OSError:
            return {'available': False, 'records': []}

    def request(self, spec, barrier=None, cancel=False, cancel_ready=None, name=None):
        name = name or spec['name']
        body = {'model': self.model, 'messages': [{'role': 'user', 'content': spec['prompt']}],
                'stream': True, 'stream_options': {'include_usage': True}, **BASE,
                'seed': spec['seed'], 'max_tokens': spec['max_tokens'], **spec['overrides']}
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        if barrier:
            barrier.wait(timeout=30)
        start = time.time()
        conn = http.client.HTTPConnection('127.0.0.1', self.args.port, timeout=60)
        record = {'name': name, 'request_sha256': hashlib.sha256(raw).hexdigest(), 'synthetic_request': body,
                  'start_epoch': start, 'first_output_epoch': None, 'last_output_epoch': None,
                  'usage': {}, 'finish_reason': None, 'saw_done': False, 'error': None,
                  'intentional_cancel': cancel, 'closed_mid_generation': False, 'deltas': []}
        parts, thoughts = [], []
        with self.lock:
            self.active.add(name); self.connections[name] = conn
        try:
            conn.request('POST', '/v1/chat/completions', body=raw, headers=self.headers)
            # HTTP/1.0 getresponse can detach conn.sock while response still reads it.
            # Retain the actual socket so the cohort watchdog can unblock readline.
            with self.lock:
                self.connections[name] = conn.sock
            response = conn.getresponse()
            if response.status != 200:
                record['error'] = {'kind': 'http_error', 'status': response.status}
            else:
                while not self.abort.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b'data:'):
                        continue
                    payload = line[5:].strip()
                    if payload == b'[DONE]':
                        record['saw_done'] = True
                        break
                    event = json.loads(payload)
                    if event.get('error'):
                        record['error'] = {'kind': 'api_error'}
                    if event.get('usage'):
                        record['usage'] = numeric_tree(event['usage'])
                    for choice in event.get('choices', []):
                        delta = choice.get('delta') or {}
                        text = delta.get('content') or ''
                        thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if text or thought:
                            now = time.time()
                            if record['first_output_epoch'] is None:
                                record['first_output_epoch'] = now
                                with self.lock:
                                    self.first[name] = now
                            record['last_output_epoch'] = now
                            parts.append(text); thoughts.append(thought)
                            record['deltas'].append({'epoch': now, 'content': text, 'reasoning': thought})
                        reason = choice.get('finish_reason')
                        if reason:
                            record['finish_reason'] = reason if reason in ('stop', 'length', 'tool_calls', 'content_filter') else 'other'
                    if cancel and len(record['deltas']) >= 16:
                        with self.lock:
                            starts = list(self.first.values())
                        if len(starts) == 4 and time.time()-max(starts) >= 0.5 and record['finish_reason'] is None:
                            record['closed_mid_generation'] = True
                            break
                response.close()
                if self.abort.is_set():
                    record['error'] = {'kind': 'cohort_timeout'}
        except Exception as e:
            record['error'] = {'kind': type(e).__name__}
        finally:
            conn.close()
            record['end_epoch'] = time.time()
            with self.lock:
                self.active.discard(name); self.connections.pop(name, None)
            if cancel_ready:
                cancel_ready.set()
        record.update(output_text=''.join(parts), reasoning_text=''.join(thoughts))
        record['output_sha256'] = digest(record['output_text'])
        record['reasoning_sha256'] = digest(record['reasoning_text'])
        with self.lock:
            self.records.append(record)
            save(self.args.out / (name + '.json'), record)
        print(json.dumps({'request': name, 'done': record['saw_done'], 'cancelled': record['closed_mid_generation'],
                          'error': record['error'], 'completion_tokens': record['usage'].get('completion_tokens')}), flush=True)
        return record

    def cohort(self, name, specs, recovery=None):
        before = self.idle()
        cursor, begin = self.log_cursor(), time.time()
        self.first = {}; self.samples = []; self.monitor_stop.clear(); self.abort.clear()
        monitor = threading.Thread(target=self.monitor, daemon=True); monitor.start()
        records = []
        try:
            if len(specs) == 1:
                records = [self.request(specs[0])]
                primary = records
            else:
                barrier = threading.Barrier(4)
                ready = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                    futures = [pool.submit(self.request, spec, barrier, recovery is not None and i == 0,
                                           ready if recovery is not None and i == 0 else None)
                               for i, spec in enumerate(specs)]
                    if recovery is not None:
                        if not ready.wait(self.args.timeout):
                            raise RuntimeError('Cancellation stream did not terminate in time')
                        follow = pool.submit(self.request, recovery, name='recovery-after')
                    primary = [f.result() for f in futures]
                    records = primary + ([follow.result()] if recovery is not None else [])
            after = self.idle(wait=30)
            # The pipeline report is stderr following the terminal engine message.
            time.sleep(0.2)
        finally:
            self.monitor_stop.set(); monitor.join(timeout=35)
        end = time.time()
        new = [r for key, r in after['rows'].items() if key not in before['rows']]
        own_window = all(begin-1 <= r.get('time', -1) <= end+1 for r in new)
        total_delta = {k: after['totals'][k]-before['totals'][k] for k in ('requests', 'drafts_offered', 'drafts_accepted')
                       if k in before['totals'] and k in after['totals']}
        identity_ok = len(new) == len(records) and own_window and total_delta.get('requests') == len(records)
        offered = [r['drafts_offered'] for r in new if 'drafts_offered' in r]
        logs = self.log_numbers(cursor)
        result = {'name': name, 'start_epoch': begin, 'end_epoch': end,
                  'requests': [r['name'] for r in records], 'numeric_samples': self.samples,
                  'max_active': max([s.get('work', {}).get('active', 0) for s in self.samples] + [0]),
                  'max_own_in_flight': max([s['own_in_flight'] for s in self.samples] + [0]),
                  'metrics_new_requests': new, 'metrics_totals_delta': total_delta,
                  'metrics_request_identity_and_exclusivity_check': identity_ok,
                  'metrics_drafts_offered_sum': sum(offered), 'metrics_draft_field_count': len(offered),
                  'metrics_scope': 'New history IDs (hashed), or numeric fingerprints, since an idle boundary; all must start in this cohort. Requires exclusive access.',
                  'piped_numeric_log': logs,
                  'piped_drafts_offered': sum(r['drafts_offered'] for r in logs['records']),
                  'piped_drafts_accepted': sum(r['drafts_accepted'] for r in logs['records']),
                  'draft_counter_limit': 'Current BDONE does not expose batch draft counts in /metrics. Solo DONE counts cannot prove piped MTP. Numeric stderr summaries are the piped-route evidence.',
                  'overlap': overlap(primary) if len(primary) == 4 else None}
        save(self.args.out / (name + '-cohort.json'), result)
        return result


def run(args):
    if not args.exclusive_benchmark:
        raise RuntimeError('--execute requires --exclusive-benchmark; no requests were sent')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise RuntimeError('--out must be new or empty')
    client = Client(args)
    env = {k: str(client.cfg.get('env', {}).get(k, '')) for k in ENV_KEYS}
    env = {k: v for k, v in env.items() if re.fullmatch(r'[-0-9,]*', v)}
    expected = args.expect_piped_mtp
    if expected == 'auto':
        expected = 'on' if env.get('STRATA_BATCH_MTP') == '1' and env.get('STRATA_PIPED_MTP') == '1' else 'off'
    report = {'version': VERSION, 'synthetic_only': True, 'service_mutations': 0,
              'expected_piped_mtp': expected, 'env_flags': env, 'port': args.port,
              'cohorts': [], 'checks': {}, 'records': [],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    seeded, mixed, cancelled, recovery = fixtures()
    try:
        client.idle()
        for name, specs, follow in (('seeded-four', seeded, None), ('mixed-four', mixed, None),
                                    ('recovery-control', [recovery], None), ('cancel-four', cancelled, recovery)):
            report['cohorts'].append(client.cohort(name, specs, follow))
            report['records'] = client.records
            save(args.out / 'correctness.json', report)
        full = [r for r in client.records if not r['intentional_cancel']]
        cancelled_record = next(r for r in client.records if r['intentional_cancel'])
        controls = {r['name']: r for r in client.records}
        common = [c for c in report['cohorts'] if c['overlap'] is not None]
        recovery_usage = usage_comparison(controls['recovery-control']['usage'], controls['recovery-after']['usage'])
        report['recovery_usage_comparison'] = recovery_usage
        report['checks'] = {
            'all_13_complete_requests_succeeded': len(full) == 13 and all(completed(r) for r in full),
            'all_three_four_stream_cohorts_overlap': all(c['overlap']['four_streams_observed'] for c in common),
            'no_unrelated_requests_or_missing_metrics': all(c['metrics_request_identity_and_exclusivity_check'] for c in report['cohorts']),
            'cancelled_after_four_stream_overlap': cancelled_record['closed_mid_generation'] and not cancelled_record['saw_done'] and not cancelled_record['error'],
            'recovery_request_identical': controls['recovery-control']['request_sha256'] == controls['recovery-after']['request_sha256'],
            'recovery_output_matches_control': all(controls['recovery-control'][k] == controls['recovery-after'][k]
                                                   for k in ('output_sha256', 'reasoning_sha256', 'finish_reason'))
                                               and recovery_usage['logical_counts_equal'],
            'recovery_marker_correct': controls['recovery-after']['output_text'].strip() == 'MTP_RECOVERY_731',
        }
        offered = sum(c['piped_drafts_offered'] for c in report['cohorts'])
        report['checks']['piped_route_confirmed'] = (offered > 0 if expected == 'on' else offered == 0)
        report['piped_route_evidence'] = {'offered': offered, 'numeric_log_available_in_all_cohorts': all(c['piped_numeric_log']['available'] for c in report['cohorts']),
                                        'on_requires_positive_numeric_pipelined_summary': True,
                                        'off_zero_is_consistent_not_a_proof_of_absence': True}
        report['max_active'] = max(c['max_active'] for c in report['cohorts'])
        report['all_checks_passed'] = all(report['checks'].values())
    except Exception as e:
        # Error type only: an endpoint's exception body can contain unrelated data.
        report['fatal_error_kind'] = type(e).__name__
        report['records'] = client.records
        report['all_checks_passed'] = False
    save(args.out / 'correctness.json', report)
    print(json.dumps({k: report[k] for k in ('checks', 'all_checks_passed')}), flush=True)
    return 0 if report['all_checks_passed'] else 2


def compare(args):
    left, right = [json.loads((p / 'correctness.json').read_text()) for p in args.compare]
    a = {r['name']: r for r in left['records']}; b = {r['name']: r for r in right['records']}
    rows = []
    for name in sorted(set(a) | set(b)):
        x, y = a.get(name), b.get(name)
        if x is None or y is None:
            rows.append({'name': name, 'passed': False, 'missing_request': True}); continue
        check = {'request_hash_equal': x['request_sha256'] == y['request_sha256']}
        usage = None
        if x['intentional_cancel'] or y['intentional_cancel']:
            check['both_cancelled_mid_generation'] = x['closed_mid_generation'] and y['closed_mid_generation']
            for key in ('output_text', 'reasoning_text'):
                n = min(len(x[key]), len(y[key]))
                check[key + '_shorter_prefix_equal'] = x[key][:n] == y[key][:n]
            check['nonempty_cancel_prefix'] = min(len(x['output_text'])+len(x['reasoning_text']),
                                                len(y['output_text'])+len(y['reasoning_text'])) > 0
        else:
            check.update({k + '_equal': x[k] == y[k] for k in
                          ('output_sha256', 'reasoning_sha256', 'finish_reason')})
            usage = usage_comparison(x['usage'], y['usage'])
            check['usage_logical_counts_equal'] = usage['logical_counts_equal']
            check['both_complete'] = completed(x) and completed(y)
        rows.append({'name': name, 'checks': check, 'passed': all(check.values()),
                     'usage_comparison': usage,
                     'baseline_output_sha256': x['output_sha256'], 'candidate_output_sha256': y['output_sha256']})
    report = {'version': VERSION, 'comparisons': rows, 'baseline_run_passed': left.get('all_checks_passed', False),
              'candidate_run_passed': right.get('all_checks_passed', False),
              'same_fixture_version': left.get('version') == right.get('version') == VERSION,
              'note': 'Cancelled streams compare the shorter content/reasoning prefix; equal cancellation length is not required. Complete requests require identical prompt/completion/total token counts; cache and other usage metadata differences are shown as qualifications, not output mismatches.'}
    report['usage_metadata_qualifications'] = [r['name'] for r in rows
                                               if r.get('usage_comparison') and not r['usage_comparison']['metadata_equal']]
    report['all_checks_passed'] = (bool(rows) and all(r['passed'] for r in rows) and report['same_fixture_version']
                                  and report['baseline_run_passed'] and report['candidate_run_passed'])
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / 'comparison.json', report)
    print(json.dumps(report, indent=2))
    return 0 if report['all_checks_passed'] else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--port', type=int, default=8092)
    p.add_argument('--out', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--exclusive-benchmark', action='store_true')
    p.add_argument('--expect-piped-mtp', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument('--engine-log', type=Path, help='Read only newly appended numeric pipelined-MTP summaries; default config.log')
    p.add_argument('--timeout', type=float, default=600, help='Hard timeout in seconds per cohort')
    p.add_argument('--compare', type=Path, nargs=2, metavar=('BASELINE', 'CANDIDATE'))
    args = p.parse_args()
    if args.compare:
        if args.execute or not args.out:
            p.error('--compare needs --out and cannot combine with --execute')
        return compare(args)
    if not args.execute:
        seeded, mixed, cancelled, recovery = fixtures()
        print(json.dumps({'version': VERSION, 'dry_run': True, 'network_requests': 0,
                          'complete_requests': 13, 'cancelled_requests': 1,
                          'cohorts': [seeded, mixed, [recovery], cancelled],
                          'execution_requires': '--config PATH --out EMPTY_DIR --execute --exclusive-benchmark',
                          'extra_action': 'Immediately submit the identical recovery request after cancelling one of four streams.'}, indent=2))
        return 0
    if not args.config or not args.out or not 1 <= args.port <= 65535 or not 0 < args.timeout <= 3600:
        p.error('Execution needs --config, --out, a valid port and timeout in (0,3600]')
    try:
        return run(args)
    except Exception as e:
        print(json.dumps({'failed': True, 'error_kind': type(e).__name__}), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
