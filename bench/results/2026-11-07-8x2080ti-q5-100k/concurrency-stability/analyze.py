#!/usr/bin/env python3
"""Offline analysis of isolated concurrency arms. Never connects to a service.

Reads synthetic result metadata/chunk journals and numeric batch log summaries.
Writes JSON plus Markdown; no automatic deployment or performance/SLO verdict.
This file is separate from the frozen, remotely executing harness.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

CHANNELS = ('output', 'visible_content', 'reasoning', 'tool')
PHASES = ('baseline', 'mixed', 'recovery')
BATCH_RE = re.compile(r'strata batch \(pipelined, (\d+) groups of (\d+)\): (\d+) group-steps, '
                      r'(\d+) rows in ([\d.]+) ms = ([\d.]+) rows/s([^\n]*)')
MTP_RE = re.compile(r'\((\d+) of (\d+) MTP proposals kept\)')


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, issues):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        issues.append({'file': str(path), 'issue': 'missing_or_invalid_json'})
        return None


def distribution(values):
    values = sorted(float(v) for v in values if number(v) and v >= 0)
    def q(p):
        return values[max(0, math.ceil(p * len(values)) - 1)] if values else None
    return {'n': len(values), 'p50_s': q(.5), 'p95_s': q(.95), 'p99_s': q(.99),
            'max_s': max(values) if values else None,
            'method': 'nearest rank; descriptive observations, not an SLO estimate'}


def canonical_chunk(chunk):
    return {k: chunk.get(k) for k in ('epoch', 'elapsed_s', 'channels', 'bytes', 'previous_output_gap_s')}


def valid_chunk(chunk):
    counts = chunk.get('bytes') or {}
    return number(chunk.get('epoch')) and all(number(counts.get(k)) and counts[k] >= 0 for k in CHANNELS)


def output_event(chunk):
    # A tool-call identifier can be a progress event with no text bytes yet.
    return 'output' in (chunk.get('channels') or []) or chunk['bytes']['output'] > 0


def load_records(folder, issues):
    records, seen, evidence = [], set(), []
    for path in sorted(folder.glob('*.json')):
        if not path.name.startswith(('resident-', 'injected-')):
            continue
        rec = read_json(path, issues)
        if not isinstance(rec, dict) or rec.get('role') not in ('resident', 'injected'):
            issues.append({'file': str(path), 'issue': 'missing_request_role'}); continue
        rec = dict(rec)
        chunks = rec.get('chunks')
        if not isinstance(chunks, list):
            chunks = []
            issues.append({'file': str(path), 'issue': 'missing_final_chunk_array'})
        invalid = sum(not isinstance(c, dict) or not valid_chunk(c) for c in chunks)
        if invalid:
            issues.append({'file': str(path), 'issue': 'invalid_chunks', 'count': invalid})
        rec['chunks'] = [c for c in chunks if isinstance(c, dict) and valid_chunk(c)]
        records.append(rec); seen.add(path.stem)
        evidence.append({'file': str(path), 'sha256': file_sha(path)})
    # Journals preserve observations if a final request JSON was never written.
    by_name = {r.get('name'): r for r in records}
    for path in sorted(folder.glob('*.chunks.jsonl')):
        name = path.name[:-len('.chunks.jsonl')]
        chunks, invalid = [], 0
        with path.open() as f:
            for line in f:
                try:
                    c = json.loads(line)
                    if not valid_chunk(c):
                        raise ValueError()
                    chunks.append(c)
                except (ValueError, TypeError, AttributeError):
                    invalid += 1
        evidence.append({'file': str(path), 'sha256': file_sha(path)})
        if invalid:
            issues.append({'file': str(path), 'issue': 'invalid_or_truncated_journal_lines', 'count': invalid})
        if name in by_name:
            final = by_name[name]['chunks']
            if [canonical_chunk(c) for c in chunks] != [canonical_chunk(c) for c in final]:
                issues.append({'file': str(path), 'issue': 'journal_differs_from_final_request',
                               'journal_chunks': len(chunks), 'final_chunks': len(final)})
            continue
        match = re.fullmatch(r'resident-(\d+)-(\d+)', name)
        role = 'resident' if match else 'injected'
        records.append({'name': name, 'role': role, 'lane': int(match[1]) if match else None,
                        'chunks': chunks, 'usage': {}, 'successful': False,
                        'error': {'kind': 'missing_final_request_json'}, 'partial_journal_only': True})
        issues.append({'file': str(path), 'issue': 'orphan_journal_missing_final_request'})
    for rec in records:
        journal = folder / (str(rec.get('name')) + '.chunks.jsonl')
        if not journal.exists() and rec['chunks']:
            issues.append({'request': rec.get('name'), 'issue': 'missing_chunk_journal'})
    return records, evidence


def lane_timeline(records, lane, begin, end, threshold):
    """Continuous lane progress, including the wait between resident generations.

    Each clipped interval covers time with no arriving output chunk. A phase
    boundary can censor it. Adjacent observations across requests are retained.
    """
    selected = [r for r in records if r.get('role') == 'resident' and r.get('lane') == lane]
    events = sorted(({'epoch': c['epoch'], 'request': r.get('name'), 'chunk_index': i}
                     for r in selected for i, c in enumerate(r['chunks']) if output_event(c)),
                    key=lambda e: e['epoch'])
    epochs = [e['epoch'] for e in events]
    inside = [e for e in events if begin < e['epoch'] < end]
    before_i = bisect.bisect_right(epochs, begin) - 1
    after_i = bisect.bisect_left(epochs, end)
    before = events[before_i] if before_i >= 0 else None
    after = events[after_i] if after_i < len(events) else None
    anchors = [{'epoch': begin, 'event': before, 'boundary': True}]
    anchors += [{'epoch': e['epoch'], 'event': e, 'boundary': False} for e in inside]
    anchors += [{'epoch': end, 'event': after, 'boundary': True}]
    intervals = []
    for left, right in zip(anchors, anchors[1:]):
        if right['epoch'] <= left['epoch']:
            continue
        a, b = left['event'], right['event']
        intervals.append({'start_epoch': left['epoch'], 'end_epoch': right['epoch'],
                          'duration_s': right['epoch'] - left['epoch'],
                          'left_phase_censored': left['boundary'], 'right_phase_censored': right['boundary'],
                          'preceding_request': a['request'] if a else None,
                          'following_request': b['request'] if b else None,
                          'cross_request_refill': bool(a and b and a['request'] != b['request'])})
    fully_observed, refill = [], []
    for a, b in zip(events, events[1:]):
        if begin <= a['epoch'] <= b['epoch'] <= end:
            fully_observed.append(b['epoch'] - a['epoch'])
        if b['epoch'] > begin and a['epoch'] < end and a['request'] != b['request']:
            refill.append({'preceding_request': a['request'], 'following_request': b['request'],
                           'full_gap_s': b['epoch'] - a['epoch'],
                           'phase_overlap_s': max(0.0, min(end, b['epoch']) - max(begin, a['epoch']))})
    awaiting = []
    for r in selected:
        start = r.get('start_epoch')
        first = min((c['epoch'] for c in r['chunks'] if output_event(c)), default=None)
        stop = first if first is not None else r.get('end_epoch')
        if number(start) and number(stop) and min(end, stop) > max(begin, start):
            awaiting.append({'request': r.get('name'), 'phase_overlap_s': min(end, stop) - max(begin, start),
                             'received_first_output': first is not None})
    counts = {k: sum(c['bytes'][k] for r in selected for c in r['chunks'] if begin <= c['epoch'] < end)
              for k in CHANNELS}
    duration = end - begin
    return {'lane': lane, 'output_chunks_in_phase': sum(begin <= e['epoch'] < end for e in events),
            'bytes': counts, 'bytes_per_s': {k: v / duration for k, v in counts.items()},
            'fully_observed_inter_output_gaps': distribution(fully_observed),
            'all_clipped_no_progress_intervals': intervals,
            'max_clipped_no_progress_interval_s': max((r['duration_s'] for r in intervals), default=None),
            'long_no_progress_intervals': [r for r in intervals if r['duration_s'] >= threshold],
            'long_interval_threshold_s': threshold, 'refill_gaps': refill,
            'awaiting_first_output_intervals': awaiting,
            'note': 'Includes gaps between requests on this lane. Phase-edge gaps are censored; output chunks are not token boundaries.'}


def occupancy_for(samples, begin, end, lanes):
    rows = [s for s in samples if number(s.get('epoch')) and begin <= s['epoch'] < end]
    def fraction(key, test):
        known = [s[key] for s in rows if number(s.get(key))]
        return {'known_samples': len(known), 'fraction': sum(test(v) for v in known) / len(known) if known else None}
    return {'samples': len(rows),
            'target_resident_in_flight': fraction('resident_in_flight', lambda x: x == lanes),
            'target_resident_with_first_output': fraction('resident_streams_with_first_output', lambda x: x == lanes),
            'all_residents_output_within_1s': fraction('resident_streams_output_within_1s', lambda x: x == lanes),
            'any_stalled_over_1s': fraction('resident_streams_stalled_over_1s', lambda x: x > 0),
            'any_stalled_over_5s': fraction('resident_streams_stalled_over_5s', lambda x: x > 0),
            'any_awaiting_first_output': fraction('resident_awaiting_first_output', lambda x: x > 0),
            'note': 'Client observations at sampled instants; not GPU occupancy or a continuous-time fraction.'}


def phase_report(records, samples, lanes, begin, end, threshold):
    lane_rows = [lane_timeline(records, i, begin, end, threshold) for i in range(lanes)]
    rates = [x['bytes_per_s']['output'] for x in lane_rows]
    squares = sum(x*x for x in rates)
    return {'start_epoch': begin, 'end_epoch': end, 'duration_s': end-begin, 'lanes': lane_rows,
            'resident_bytes_per_s': {k: sum(x['bytes_per_s'][k] for x in lane_rows) for k in CHANNELS},
            'jain_fairness_of_lane_output_byte_rates': sum(rates)**2 / (lanes*squares) if squares else None,
            'worst_lane_max_no_progress_s': max((x['max_clipped_no_progress_interval_s'] for x in lane_rows), default=None),
            'worst_lane_observed_p95_gap_s': max((x['fully_observed_inter_output_gaps']['p95_s'] for x in lane_rows
                                                 if x['fully_observed_inter_output_gaps']['p95_s'] is not None), default=None),
            'occupancy': occupancy_for(samples, begin, end, lanes),
            'warning': 'Worst lane avoids a pooled fast-stream weighting bias. Tail quantiles from one short scenario are not an SLO.'}


def request_metadata(rec):
    usage = rec.get('usage') or {}
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return {'name': rec.get('name'), 'role': rec.get('role'), 'lane': rec.get('lane'),
            'request_sha256': rec.get('request_sha256'), 'output_sha256': rec.get('output_sha256'),
            'nominal_prompt_tokens': rec.get('nominal_prompt_tokens'), 'actual_prompt_tokens': usage.get('prompt_tokens'),
            'cached_tokens': cached, 'verified_zero_cache': cached == 0 if cached is not None else None,
            'nominal_size_within_tolerance': rec.get('prompt_within_nominal_tolerance'),
            'completion_tokens': usage.get('completion_tokens'), 'max_tokens': rec.get('max_tokens'),
            'finish_reason': rec.get('finish_reason'), 'saw_done': rec.get('saw_done'),
            'successful': rec.get('successful'), 'error_kind': (rec.get('error') or {}).get('kind'),
            'ttft_s': rec.get('ttft_s'), 'elapsed_s': rec.get('elapsed_s'),
            'start_epoch': rec.get('start_epoch'), 'end_epoch': rec.get('end_epoch'),
            'schedule_delay_s': rec.get('schedule_delay_s'), 'partial_journal_only': rec.get('partial_journal_only', False)}


def batch_summaries(path):
    rows = []
    if not path.is_file():
        return {'available': False, 'records': [], 'scope': 'Whole isolated arm engine log'}
    with path.open(errors='replace') as f:
        for index, line in enumerate(f, 1):
            match = BATCH_RE.search(line)
            if not match:
                continue
            g = match.groups(); mtp = MTP_RE.search(g[6])
            rows.append({'line': index, 'groups': int(g[0]), 'slots_per_group': int(g[1]),
                         'group_steps': int(g[2]), 'rows': int(g[3]), 'wall_ms': float(g[4]),
                         'reported_rows_per_s': float(g[5]),
                         'drafts_accepted': int(mtp[1]) if mtp else None,
                         'drafts_offered': int(mtp[2]) if mtp else None})
    offered = sum(r['drafts_offered'] or 0 for r in rows)
    accepted = sum(r['drafts_accepted'] or 0 for r in rows)
    return {'available': True, 'records': rows, 'drafts_offered': offered, 'drafts_accepted': accepted,
            'accepted_fraction': accepted / offered if offered else None,
            'valid_acceptance_counts': all(0 <= r['drafts_accepted'] <= r['drafts_offered']
                                           for r in rows if r['drafts_offered'] is not None),
            'scope': 'Whole isolated arm includes correctness and performance. Rows/s includes admissions and is not a per-request decode rate.'}


def level_report(folder, perf_manifest, meta, root_manifest, threshold):
    issues = []
    summary = read_json(folder / 'summary.json', issues) or {}
    samples = read_json(folder / 'occupancy.json', issues) or []
    before = read_json(folder / 'status-before.json', issues) or {}
    records, evidence = load_records(folder, issues)
    lanes = summary.get('resident_workers') or int(folder.name[1:])
    plan = perf_manifest.get('plan') or {}
    phases = {}
    for name in PHASES:
        phase = (summary.get('phases') or {}).get(name) or {}
        begin, end = phase.get('start_epoch'), phase.get('end_epoch')
        if not number(begin) or not number(end) or end <= begin:
            issues.append({'phase': name, 'issue': 'missing_or_invalid_phase_bounds'}); continue
        phases[name] = phase_report(records, samples, lanes, begin, end, threshold)
    requests = [request_metadata(r) for r in records]
    complete = [r for r in requests if r['successful'] is True and r['saw_done'] is True
                and isinstance(r['completion_tokens'], int) and r['completion_tokens'] >= 0]
    all_usage = [r for r in requests if isinstance(r['completion_tokens'], int) and r['completion_tokens'] >= 0]
    nominal = plan.get('resident_context_nominal_tokens')
    signature = {'sampling_profile': perf_manifest.get('plan', {}).get('sampling_profile'),
                 'fixture_id': perf_manifest.get('fixture_id'), 'plan_sha256': perf_manifest.get('plan_sha256'),
                 'effective_plan_sha256': fingerprint(plan) if plan else None, 'profile': plan.get('profile'),
                 'resident_context_nominal_tokens': nominal, 'lanes': lanes,
                 'native_context': (before.get('context') or {}).get('native'),
                 'max_positions': (before.get('context') or {}).get('max_positions'),
                 'client_sha256': perf_manifest.get('script_sha256'),
                 'engine_sha256': meta.get('engine_sha256'), 'library_sha256': meta.get('adjacent_library_sha256'),
                 'config_except_variables_sha256': meta.get('config_except_variables_log_key_sha256')}
    injected = [r for r in requests if r['role'] == 'injected']
    expected_injections = len(plan.get('injected_prefill_nominal_tokens') or [])
    qualification = {'harness_reported_error': summary.get('error'),
                     'baseline_occupancy_qualified': summary.get('baseline_occupancy_qualified'),
                     'performance_suite_returncode': meta.get('suite_returncodes', {}).get('performance'),
                     'all_three_phases_available': len(phases) == 3,
                     'expected_injections': expected_injections, 'observed_injections': len(injected),
                     'all_expected_injections_complete': bool(expected_injections) and len(injected) == expected_injections
                         and all(r['successful'] is True and r['saw_done'] is True for r in injected),
                     'all_injections_verified_cold': bool(injected) and all(r['verified_zero_cache'] is True for r in injected),
                     'all_injections_size_qualified': bool(injected) and all(r['nominal_size_within_tolerance'] is True for r in injected),
                     'raw_data_issues': issues,
                     'missing_signature_fields': [k for k, v in signature.items() if v is None]}
    qualification['comparison_evidence_complete'] = (qualification['harness_reported_error'] is None
        and qualification['baseline_occupancy_qualified'] is True and qualification['performance_suite_returncode'] == 0
        and qualification['all_three_phases_available'] and qualification['all_expected_injections_complete']
        and qualification['all_injections_verified_cold'] and qualification['all_injections_size_qualified']
        and not issues and not qualification['missing_signature_fields'])
    duration = summary.get('scenario_end_epoch', 0) - summary.get('scenario_start_epoch', 0)
    totals = {'fully_completed_requests_with_usage': len(complete),
              'fully_completed_api_completion_tokens': sum(r['completion_tokens'] for r in complete),
              'all_observed_api_completion_tokens': sum(r['completion_tokens'] for r in all_usage),
              'requests_missing_completion_usage': [r['name'] for r in requests if not isinstance(r['completion_tokens'], int)],
              'expected_bounded_drain_cancellations': sum(r['error_kind'] == 'bounded_drain_cancel' for r in requests),
              'scenario_wall_including_start_and_drain_s': duration if duration > 0 else None,
              'completed_tokens_per_s_including_start_and_drain': sum(r['completion_tokens'] for r in complete) / duration if duration > 0 else None,
              'by_role': {role: {'fully_completed_requests': sum(r['role'] == role for r in complete),
                                 'completion_tokens': sum(r['completion_tokens'] for r in complete if r['role'] == role)}
                          for role in ('resident', 'injected')},
              'note': 'Completed API counts only. Missing/cancelled streams make this a lower bound, not steady-state throughput.'}
    return {'lanes': lanes, 'signature': signature, 'qualification': qualification, 'phases': phases,
            'api_totals': totals, 'cold_injected_requests': injected, 'requests': requests,
            'original_harness_occupancy': summary.get('occupancy'), 'original_harness_recovery': summary.get('recovery'),
            'evidence': evidence, 'summary_sha256': file_sha(folder / 'summary.json')}


def arm_report(root, arm_dir, root_manifest, threshold):
    issues = []
    meta = read_json(arm_dir / 'metadata.json', issues) or {}
    perf = read_json(arm_dir / 'performance' / 'manifest.json', issues) or {}
    correct_path = arm_dir / 'correctness' / 'correctness.json'
    correct = read_json(correct_path, issues) or {}
    cohorts = []
    for cohort in correct.get('cohorts') or []:
        cohorts.append({'name': cohort.get('name'), 'piped_drafts_offered': cohort.get('piped_drafts_offered'),
                        'piped_drafts_accepted': cohort.get('piped_drafts_accepted'),
                        'metrics_request_identity_and_exclusivity_check': cohort.get('metrics_request_identity_and_exclusivity_check'),
                        'numeric_log': cohort.get('piped_numeric_log')})
    return {'id': str(root) + ':' + arm_dir.name, 'root': str(root), 'tag': arm_dir.name, 'arm': meta.get('arm'),
            'metadata_sha256': file_sha(arm_dir / 'metadata.json'), 'root_manifest_sha256': file_sha(root / 'manifest.json'),
            'source_harness_sha256': root_manifest.get('harness_sha256'), 'issues': issues,
            'startup_facts': meta.get('startup_facts'),
            'correctness': {'all_checks_passed': correct.get('all_checks_passed'), 'checks': correct.get('checks'),
                            'piped_route_evidence': correct.get('piped_route_evidence'), 'cohorts': cohorts},
            'mtp_batch_summaries': batch_summaries(arm_dir / 'engine.log'),
            'levels': {folder.name: level_report(folder, perf, meta, root_manifest, threshold)
                       for folder in sorted((arm_dir / 'performance').glob('c[1248]')) if folder.is_dir()}}


def delta(left, right):
    return {'left': left, 'right': right,
            'relative_change_percent': (right/left-1)*100 if number(left) and number(right) and left != 0 else None}


def compare_levels(a, b):
    reasons = []
    for key in sorted(set(a['signature']) | set(b['signature'])):
        left, right = a['signature'].get(key), b['signature'].get(key)
        if left is None or right is None or left != right:
            reasons.append({'field': key, 'issue': 'missing_or_mismatched_signature', 'left': left, 'right': right})
    for label, level in (('left', a), ('right', b)):
        if not level['qualification']['comparison_evidence_complete']:
            reasons.append({'arm': label, 'issue': 'incomplete_or_unqualified_evidence'})
    x = {r['name']: r for r in a['requests']}; y = {r['name']: r for r in b['requests']}
    initial = {f'resident-{i}-0' for i in range(a['lanes'])}
    injected = {r['name'] for r in a['cold_injected_requests']} | {r['name'] for r in b['cold_injected_requests']}
    matched_inputs = []
    for name in sorted(initial | injected):
        p, q = x.get(name), y.get(name)
        checks = {k: bool(p and q and p.get(k) is not None and p.get(k) == q.get(k))
                  for k in ('request_sha256', 'actual_prompt_tokens', 'cached_tokens')}
        if not all(checks.values()):
            reasons.append({'request': name, 'issue': 'missing_or_mismatched_input_cache', 'checks': checks})
        matched_inputs.append({'name': name, 'checks': checks,
                               'left': {k: p.get(k) for k in ('request_sha256', 'actual_prompt_tokens', 'cached_tokens', 'ttft_s')} if p else None,
                               'right': {k: q.get(k) for k in ('request_sha256', 'actual_prompt_tokens', 'cached_tokens', 'ttft_s')} if q else None,
                               'ttft_s': delta(p.get('ttft_s'), q.get('ttft_s')) if p and q else None})
    phases = {}
    for name in PHASES:
        p, q = a['phases'].get(name), b['phases'].get(name)
        if not p or not q:
            continue
        phases[name] = {'output_bytes_per_s': delta(p['resident_bytes_per_s']['output'], q['resident_bytes_per_s']['output']),
                        'worst_lane_observed_p95_gap_s': delta(p['worst_lane_observed_p95_gap_s'], q['worst_lane_observed_p95_gap_s']),
                        'worst_lane_max_no_progress_s': delta(p['worst_lane_max_no_progress_s'], q['worst_lane_max_no_progress_s']),
                        'jain_byte_progress_fairness': delta(p['jain_fairness_of_lane_output_byte_rates'], q['jain_fairness_of_lane_output_byte_rates']),
                        'lanes': [{'lane': i, 'bytes_per_s': delta(p['lanes'][i]['bytes_per_s']['output'], q['lanes'][i]['bytes_per_s']['output']),
                                   'max_no_progress_s': delta(p['lanes'][i]['max_clipped_no_progress_interval_s'], q['lanes'][i]['max_clipped_no_progress_interval_s'])}
                                  for i in range(min(len(p['lanes']), len(q['lanes'])))]}
    return {'matched_conditions_and_evidence': not reasons, 'disqualification_reasons': reasons,
            'matched_initial_and_cold_inputs': matched_inputs, 'phases': phases,
            'note': 'Qualification covers performance evidence only, not correctness. Deltas are descriptive even when unqualified. Performance output hashes are not correctness gates; use the separate fixed-request correctness comparison. No SLO or deployment verdict.'}


def analyze(roots, threshold):
    arms, issues = [], []
    for root in roots:
        root_manifest = read_json(root / 'manifest.json', issues) or {}
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and re.fullmatch(r'\d+-(?:off|on)', folder.name):
                arms.append(arm_report(root, folder, root_manifest, threshold))
    comparisons = []
    for a, b in itertools.combinations(arms, 2):
        if a['arm'] == 'on' and b['arm'] == 'off':
            a, b = b, a
        for level in sorted(set(a['levels']) & set(b['levels'])):
            comparisons.append({'left': a['id'], 'right': b['id'], 'level': level,
                                'kind': 'cross_flag' if a['arm'] != b['arm'] else 'same_flag_drift',
                                **compare_levels(a['levels'][level], b['levels'][level])})
    return {'analyzer_sha256': file_sha(Path(__file__)), 'source_roots': [str(p) for p in roots],
            'long_interval_threshold_s': threshold, 'issues': issues, 'arms': arms, 'comparisons': comparisons,
            'interpretation': ['SSE output-chunk arrival timing, not token TBT.',
                               'Continuous per-lane silence includes request refill waits and phase-edge censoring.',
                               'Usage totals are exact reported counts; no per-chunk token estimates are promoted to measured throughput.',
                               'Short A/B has sparse cold joins and cannot establish an SLO or deployment decision.']}


def markdown(report):
    lines = ['# Offline concurrency observations', '',
             'Output gaps are SSE chunk intervals. Phase edges are censored; figures do not establish an SLO.', '',
             '| Arm | Level | Phase | Output bytes/s | Worst lane p95 gap (s) | Longest clipped silence (s) | Performance evidence qualified | Arm correctness passed |',
             '| --- | --- | --- | ---: | ---: | ---: | --- | --- |']
    def fmt(value):
        return f'{value:.3f}' if number(value) else 'missing'
    for arm in report['arms']:
        for level, data in arm['levels'].items():
            for name, phase in data['phases'].items():
                lines.append(f"| {Path(arm['root']).name}/{arm['tag']} | {level} | {name} | "
                             f"{fmt(phase['resident_bytes_per_s']['output'])} | {fmt(phase['worst_lane_observed_p95_gap_s'])} | "
                             f"{fmt(phase['worst_lane_max_no_progress_s'])} | {data['qualification']['comparison_evidence_complete']} | "
                             f"{arm['correctness']['all_checks_passed']} |")
    lines += ['', 'The JSON retains per-lane intervals, refill gaps, occupancy, request input/cache checks, API totals, '
              'numeric MTP summaries and all missing-evidence reasons.', '', '## Pair qualification', '']
    for pair in report['comparisons']:
        lines.append(f"- {pair['left']} → {pair['right']} {pair['level']}: "
                     f"{'matched' if pair['matched_conditions_and_evidence'] else 'unqualified'}; "
                     f"{len(pair['disqualification_reasons'])} qualification issues.")
    return '\n'.join(lines) + '\n'


def self_test():
    def rec(name, times):
        return {'name': name, 'role': 'resident', 'lane': 0, 'start_epoch': times[0]-1,
                'end_epoch': times[-1]+.1,
                'chunks': [{'epoch': t, 'bytes': {'output': 4, 'visible_content': 4, 'reasoning': 0, 'tool': 0}} for t in times]}
    records = [rec('resident-0-0', [1, 2]), rec('resident-0-1', [7, 8])]
    lane = lane_timeline(records, 0, 0, 10, 1)
    assert lane['max_clipped_no_progress_interval_s'] == 5
    assert lane['refill_gaps'][0]['full_gap_s'] == 5
    assert any(x['cross_request_refill'] and x['duration_s'] == 5 for x in lane['long_no_progress_intervals'])
    clipped = lane_timeline(records, 0, 3, 6, 1)
    assert clipped['max_clipped_no_progress_interval_s'] == 3
    assert clipped['fully_observed_inter_output_gaps']['n'] == 0
    assert clipped['long_no_progress_intervals'][0]['left_phase_censored']
    assert clipped['long_no_progress_intervals'][0]['right_phase_censored']
    empty = lane_timeline(records, 1, 0, 10, 1)
    assert empty['max_clipped_no_progress_interval_s'] == 10 and empty['output_chunks_in_phase'] == 0
    assert occupancy_for([], 0, 10, 4)['target_resident_in_flight']['fraction'] is None
    a = {'signature': {'sampling_profile': 'greedy'}, 'qualification': {'comparison_evidence_complete': True},
         'requests': [], 'lanes': 0, 'cold_injected_requests': [], 'phases': {}}
    b = {**a, 'signature': {'sampling_profile': 'sampled'}}
    assert not compare_levels(a, b)['matched_conditions_and_evidence']
    b = {**a, 'qualification': {'comparison_evidence_complete': False}}
    assert not compare_levels(a, b)['matched_conditions_and_evidence']
    assert MTP_RE.search('(7 of 10 MTP proposals kept)').groups() == ('7', '10')
    print('OFFLINE PASS: cross-refill silence, censored/empty lanes, missing occupancy, incomparable sampling/evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, nargs='+', help='One or more downloaded runner result roots')
    parser.add_argument('--out', type=Path, help='New output directory outside input result roots')
    parser.add_argument('--long-gap-seconds', type=float, default=1.0)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.root or args.out is None or not math.isfinite(args.long_gap_seconds) or args.long_gap_seconds <= 0:
        parser.error('require --root --out and a positive finite --long-gap-seconds')
    roots, out = [p.resolve() for p in args.root], args.out.resolve()
    if any(not p.is_dir() for p in roots):
        parser.error('all input roots must exist')
    if any(out == p or p in out.parents for p in roots):
        parser.error('--out must be outside input artifact roots')
    if out.exists() and any(out.iterdir()):
        parser.error('--out must be new or empty')
    report = analyze(roots, args.long_gap_seconds)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (out / 'analysis.md').write_text(markdown(report))
    print(json.dumps({'arms': len(report['arms']), 'comparisons': len(report['comparisons']),
                      'qualified_comparisons': sum(p['matched_conditions_and_evidence'] for p in report['comparisons']),
                      'output': str(out), 'note': 'Offline observations only; no performance pass or deployment verdict'}))


if __name__ == '__main__':
    main()
