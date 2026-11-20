#!/usr/bin/env python3
"""Summarize local ab_client arm directories. Does not contact any service.

Example: python3 summarize_ab.py --root ./candidates --baseline baseline
Writes ab-comparison.json and ab-comparison.md beneath --out (default --root).
Partial arms are supported and explicitly identified.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ENGINE_RE = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)')
CUDA_ERROR_RE = re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|(?:error|fail)[^\n]{0,100}CUDA|illegal memory access|device.side assert|CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits', re.I)


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def engine_records(record):
    if record.get('engine_prompt_records'):
        return record['engine_prompt_records']
    parsed = []
    for line in record.get('engine_lines', []):
        m = ENGINE_RE.search(line)
        if m:
            v = m.groups()
            parsed.append(dict(zip(('prompt_tokens', 'reused_tokens', 'read_tokens', 'prefill_ms', 'prefill_tps',
                                    'generated_tokens', 'decode_ms', 'decode_tps'),
                                   (int(v[0]), int(v[1]), int(v[2]), int(v[3]), float(v[4]), int(v[5]), int(v[6]), float(v[7])))))
    return parsed


def request_summary(record):
    usage = record.get('usage') or {}
    pn, cn = usage.get('prompt_tokens'), usage.get('completion_tokens')
    candidates = [e for e in engine_records(record) if e.get('prompt_tokens') == pn]
    full = [e for e in candidates if e.get('generated_tokens') == cn]
    chosen = full[0] if len(full) == 1 else (candidates[0] if len(candidates) == 1 else None)
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    actual_hash = hashlib.sha256(record.get('output_text', '').encode()).hexdigest()
    return {'name': record.get('name'), 'prompt_tokens': pn, 'completion_tokens': cn,
            'ttft_s': record.get('ttft_s'), 'total_s': record.get('total_s'),
            'api_cached_tokens': cached,
            'engine_matching_record_count': len(candidates), 'engine_full_request_match_count': len(full),
            'engine_prefill_record': chosen,
            'zero_reuse_confirmed': chosen is not None and chosen.get('reused_tokens') == 0 and cached in (0, None),
            'engine_fullrequest_decode_tps': full[0].get('decode_tps') if len(full) == 1 else None,
            'client_decode_tps_estimate': record.get('decode_tps_usage_minus_one_estimate'),
            'finish_reason': record.get('finish_reason'), 'error': record.get('error'),
            'saw_done': record.get('saw_done'), 'output_count_matches_limit': record.get('output_count_matches_limit'),
            'messages_sha256': record.get('messages_sha256'), 'request_sha256': record.get('request_sha256'),
            'output_sha256': record.get('output_sha256'), 'output_hash_verified': actual_hash == record.get('output_sha256'),
            'reasoning_chars': len(record.get('reasoning_text', ''))}


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            break
    else:
        if len(a) == len(b):
            return None
        i = min(len(a), len(b))
    line = a.count('\n', 0, i) + 1
    prior = a.rfind('\n', 0, i)
    return {'codepoint_offset_0based': i, 'reference_line_1based': line,
            'reference_column_1based': i - prior, 'reference_length': len(a), 'candidate_length': len(b),
            'reference_context': a[max(0, i-60):i+120], 'candidate_context': b[max(0, i-60):i+120]}


def load_arm(name, directory):
    data = read_json(directory / 'results.json', [])
    if not isinstance(data, list):
        data = []
    records = {r['name']: r for r in data if isinstance(r, dict) and r.get('name')}
    # Atomic individual files can be one completed request ahead of results.json.
    for pattern in ('cold-*.json', 'append-*.json', 'c6-long-[0-9]*.json', 'c8-short-[0-9]*.json'):
        for path in directory.glob(pattern):
            r = read_json(path, {})
            if isinstance(r, dict) and r.get('name') and 'usage' in r:
                records[r['name']] = r
    cohorts = {}
    for cohort in ('c6-long', 'c8-short'):
        d = read_json(directory / (cohort + '-summary.json'))
        if d:
            group = [r for key, r in records.items() if key.startswith(cohort + '-')]
            rates = [r.get('decode_tps_usage_minus_one_estimate') for r in group]
            rates = [x for x in rates if isinstance(x, (float, int))]
            d = {k: v for k, v in d.items() if k not in ('engine_lines', 'engine_prompt_records')}
            d['per_request_decode_min_estimate'] = min(rates) if rates else None
            d['per_request_decode_max_estimate'] = max(rates) if rates else None
            cohorts[cohort] = d
    errors = []
    for r in records.values():
        for line in r.get('engine_lines', []):
            if CUDA_ERROR_RE.search(line):
                errors.append({'source': r['name'], 'line': line})
    for pattern in ('*.log', '*.out', '*.txt'):
        for path in directory.glob(pattern):
            try:
                with path.open(errors='replace') as f:
                    for lineno, line in enumerate(f, 1):
                        if CUDA_ERROR_RE.search(line):
                            errors.append({'source': path.name, 'line_number': lineno, 'line': line.rstrip()[:1000]})
            except OSError:
                pass
    arm = {'name': name, 'directory': str(directory.resolve()),
           'complete': (directory / 'complete.json').exists(), 'request_count': len(records),
           'requests': {key: request_summary(r) for key, r in sorted(records.items())},
           'cohorts': cohorts, 'cuda_or_memory_error_evidence': errors,
           'error_scan_scope': 'Available filtered engine_lines and arm-local *.log/*.out/*.txt only; absence is not proof for logs not copied.'}
    return arm, records


def compare(base, other):
    pairs = []
    for name in sorted(set(base) | set(other)):
        a, b = base.get(name), other.get(name)
        strict = name.startswith('cold-') or name.startswith('append-')
        p = {'name': name, 'strict_single_request_check': strict}
        if a is None or b is None:
            p.update(status='missing_request', reference_present=a is not None, candidate_present=b is not None)
            pairs.append(p)
            continue
        same_messages = bool(a.get('messages_sha256')) and a.get('messages_sha256') == b.get('messages_sha256')
        same_request = bool(a.get('request_sha256')) and a.get('request_sha256') == b.get('request_sha256')
        same_output = a.get('output_text') == b.get('output_text')
        same_reasoning = a.get('reasoning_text', '') == b.get('reasoning_text', '')
        if not same_messages or not same_request:
            state = 'input_or_request_mismatch_not_comparable'
        elif same_output and same_reasoning:
            state = 'exact_text_match'
        elif strict:
            state = 'strict_single_request_output_mismatch'
        else:
            state = 'concurrent_output_diff_not_a_correctness_verdict'
        p.update(status=state, same_messages=same_messages, same_request=same_request,
                 same_output_text=same_output, same_reasoning_text=same_reasoning,
                 same_output_sha256=a.get('output_sha256') == b.get('output_sha256'),
                 reference_messages_sha256=a.get('messages_sha256'), candidate_messages_sha256=b.get('messages_sha256'),
                 reference_output_sha256=a.get('output_sha256'), candidate_output_sha256=b.get('output_sha256'),
                 first_difference=first_difference(a.get('output_text', ''), b.get('output_text', '')))
        pairs.append(p)
    return pairs


def number(x, digits=2):
    return f'{x:.{digits}f}' if isinstance(x, (float, int)) else '—'


def render(report):
    lines = ['# 100k Q5 A/B comparison', '',
             'API usage gives exact token totals. Client decode is a first-to-last SSE arrival estimate; engine decode is only shown for a unique matching prompt/output-token record. Cold timings require confirmed zero reuse.', '',
             '| Arm | Request | Prompt / output | Cold zero reuse | Engine read tokens / ms | Prefill tok/s | TTFT s | Client decode est. tok/s | Engine full request decode tok/s | Finish |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|---|']
    for name, arm in report['arms'].items():
        for reqname in ('cold-1', 'cold-2', 'cold-3', 'append-600'):
            r = arm['requests'].get(reqname)
            if not r:
                continue
            e = r.get('engine_prefill_record') or {}
            zr = ('yes' if r['zero_reuse_confirmed'] else 'NO / unconfirmed') if reqname.startswith('cold') else 'append'
            lines.append(f"| {name} | {reqname} | {r['prompt_tokens']} / {r['completion_tokens']} | {zr} | {e.get('read_tokens', '—')} / {e.get('prefill_ms', '—')} | {number(e.get('prefill_tps'))} | {number(r['ttft_s'])} | {number(r['client_decode_tps_estimate'])} | {number(r['engine_fullrequest_decode_tps'])} | {r['finish_reason']} |")
    lines.extend(['', '| Arm | Cohort | Wall s | Output tokens | Aggregate tok/s over wall | Per-request decode est. range | Common decode s | Common chunks/s (not tok/s) | Errors |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for name, arm in report['arms'].items():
        for cname, c in arm['cohorts'].items():
            lines.append(f"| {name} | {cname} | {number(c.get('wall_s'))} | {c.get('completion_tokens')} | {number(c.get('aggregate_completion_tps_over_cohort_wall'))} | {number(c.get('per_request_decode_min_estimate'))}–{number(c.get('per_request_decode_max_estimate'))} | {number(c.get('common_decode_window_s'))} | {number(c.get('common_decode_chunks_per_s'))} | {c.get('error_count')} |")
    lines.extend(['', f"Text parity reference: `{report['baseline']}`. Fixed single requests require exact text equality when request hashes agree. Concurrent differences are reported without declaring a correctness failure. Text checks cannot establish finite internal logits.", '',
                  '| Candidate | Request | Same messages | Same full request | Same output hash | Result | First difference (offset / line / col) |',
                  '|---|---|---|---|---|---|---|'])
    for name, pairs in report['comparisons'].items():
        for p in pairs:
            d = p.get('first_difference')
            where = f"{d['codepoint_offset_0based']} / {d['reference_line_1based']} / {d['reference_column_1based']}" if d else '—'
            lines.append(f"| {name} | {p['name']} | {p.get('same_messages', '—')} | {p.get('same_request', '—')} | {p.get('same_output_sha256', '—')} | {p['status']} | {where} |")
    lines.extend(['', 'First-difference context and full hashes are in `ab-comparison.json`.', ''])
    for name, arm in report['arms'].items():
        er = sum(r.get('error') is not None for r in arm['requests'].values())
        badhash = sum(not r['output_hash_verified'] for r in arm['requests'].values())
        lines.append(f"- {name}: {'complete' if arm['complete'] else 'PARTIAL'}, {arm['request_count']} requests, {er} request errors, {badhash} invalid saved-output hashes, {len(arm['cuda_or_memory_error_evidence'])} CUDA/memory-error evidence lines.")
    lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--baseline', default='baseline')
    ap.add_argument('--arm', action='append', help='optional explicit NAME=PATH; repeat for each arm')
    args = ap.parse_args()
    directories = {}
    if args.arm:
        for value in args.arm:
            if '=' not in value:
                ap.error('--arm needs NAME=PATH')
            name, path = value.split('=', 1)
            directories[name] = Path(path)
    else:
        for p in sorted(args.root.rglob('results.json')):
            rows = read_json(p, [])
            if isinstance(rows, list) and rows and isinstance(rows[0], dict) and 'messages_sha256' in rows[0]:
                directories[str(p.parent.relative_to(args.root))] = p.parent
    if not directories:
        ap.error('no A/B arm results found')
    baseline = args.baseline
    if baseline not in directories:
        matches = [name for name in directories if name.endswith('/' + baseline)]
        if len(matches) == 1:
            baseline = matches[0]
        else:
            ap.error('baseline arm missing; choose --baseline from ' + ', '.join(directories))
    arms, raw = {}, {}
    for name, directory in directories.items():
        arms[name], raw[name] = load_arm(name, directory)
    report = {'baseline': baseline, 'arms': arms,
              'comparisons': {name: compare(raw[baseline], records) for name, records in raw.items() if name != baseline}}
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ab-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    markdown = render(report)
    (out / 'ab-comparison.md').write_text(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
