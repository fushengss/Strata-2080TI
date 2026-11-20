#!/usr/bin/env python3
"""Strict offline parser for drained per-slot piped MTP observation reports.

No service, network, inference or GPU operations. Other engine log lines are
ignored; a malformed observation line or incomplete observation block fails.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

VERSION = 'piped-mtp-observe-parser-1'
PREFIX = 'strata piped MTP observe'
NUMBER = r'(?:[0-9]+(?:\.[0-9]+)?)'
STAT = (r'(?P<name>[a-z_]+)_ms\(n=(?P<n>[0-9]+),mean=(?P<mean>' + NUMBER +
        r'),min=(?P<min>' + NUMBER + r'),max=(?P<max>' + NUMBER +
        r'),sum=(?P<sum>' + NUMBER + r')\)')
STAT_RE = re.compile(STAT)
SLOT_RE = re.compile(r'strata piped MTP observe: slot ([0-9]+), completed_chains=([0-9]+), '
                     r'source_stream_intervals (.+)')
HOST_RE = re.compile(r'strata piped MTP observe host: verdicts=([0-9]+), '
                     r'stage0_draft_not_ready_group_skips=([0-9]+) (.+)')
HOST_SUFFIX = (' (pump-sample counts, not walltime shares; verdict jobs are immediately submitted, '
               'not a post-attention ready batch; GPU intervals include contention and mapped head transfer)')
NOTES = {
    'units': 'All phase and verdict-gap times are milliseconds; counts are integer events or pump samples.',
    'gpu_intervals': 'Source-stream elapsed intervals include contention/dependencies, not kernel active time or FLOPS. '
                     'Head return includes mapped transfer. Body excludes pre-body residual staging and its dependency. '
                     'Slot intervals overlap; never sum them into engine wall time or token throughput.',
    'verdict_gap': 'Host last-stage verdict arrival gaps can include admission, prefill and drain/refill. '
                   'Align with route phases; this is not pure decode interarrival time.',
    'pending_histogram': 'Submitted-but-unreaped chains sampled at pump entry; GPU completion may already have occurred. '
                         'Counts are pump samples, not duration shares or a post-attention ready queue.',
    'unsubmitted_histogram': 'Live jobs of this verdict immediately before submission. With GS=1 there are 0 or 1; '
                             'these have not run their bodies and are not post-attention ready jobs.',
    'scope': 'Only completed, drained reports within the supplied log range; no packing or performance conclusion.',
}


class ObservationError(ValueError):
    """Safe diagnostic messages contain line numbers and labels, never raw logs."""


def require(condition, message):
    if not condition:
        raise ObservationError(message)


def parse_stat(text, expected):
    match = STAT_RE.fullmatch(text)
    require(match is not None and match['name'] == expected, 'Malformed ' + expected + ' summary')
    row = {'n': int(match['n']), **{k + '_ms': float(match[k]) for k in ('mean', 'min', 'max', 'sum')}}
    n = row['n']
    mean, minimum, maximum, total = (row[k + '_ms'] for k in ('mean', 'min', 'max', 'sum'))
    require(all(math.isfinite(x) and x >= 0 for x in (mean, minimum, maximum, total)),
            'Nonfinite or negative ' + expected + ' summary')
    # Native fprintf rounds each value independently to six fractional digits.
    tolerance = (n + 1) * 0.500001e-6 + 1e-9
    require(abs(total - mean * n) <= tolerance, expected + ' sum/count/mean mismatch')
    require(minimum <= maximum and minimum - 1e-6 <= mean <= maximum + 1e-6,
            expected + ' min/mean/max mismatch')
    if n == 0:
        require(all(x == 0 for x in (mean, minimum, maximum, total)), expected + ' zero count has nonzero values')
    else:
        require(n * minimum - tolerance <= total <= n * maximum + tolerance,
                expected + ' sum outside min/max bounds')
    return row


def histogram(text, slots):
    fields = text.split(',')
    require(len(fields) == slots + 1, 'Histogram bucket count differs from slots')
    result = []
    for index, field in enumerate(fields):
        match = re.fullmatch(r'([0-9]+):([0-9]+)', field)
        require(match is not None and int(match[1]) == index, 'Histogram indices missing, duplicate or unordered')
        result.append(int(match[2]))
    return result


def merge_stats(rows):
    count = sum(row['n'] for row in rows)
    total = sum(row['sum_ms'] for row in rows)
    populated = [row for row in rows if row['n']]
    return {'n': count, 'mean_ms': total / count if count else 0.0,
            'min_ms': min((row['min_ms'] for row in populated), default=0.0),
            'max_ms': max((row['max_ms'] for row in populated), default=0.0), 'sum_ms': total}


def parse_text(text, slots=8):
    """Return numeric-only report; raise ObservationError for absent/incomplete/bad observations.

    Suitable for a suite byte-range slice without startup lines. Startup route
    qualification is separate; this function accepts only complete drained blocks.
    """
    require(type(slots) is int and 1 <= slots <= 64, 'slots must be an integer in 1..64')
    require(isinstance(text, str), 'Log text must be a string')
    blocks, current = [], {}
    for line_number, original in enumerate(text.splitlines(), 1):
        start = original.find(PREFIX)
        if start < 0:
            continue
        line = original[start:]
        try:
            slot_match = SLOT_RE.fullmatch(line)
            if slot_match:
                slot, chains = int(slot_match[1]), int(slot_match[2])
                require(0 <= slot < slots and slot not in current, 'Slot missing, repeated or out of range')
                pieces = slot_match[3].split(' ')
                require(len(pieces) == 3, 'Expected exactly three slot phase summaries')
                phases = {name: parse_stat(piece, name) for name, piece in zip(('body', 'head_return', 'tail'), pieces)}
                counts = [row['n'] for row in phases.values()]
                require(len(set(counts)) == 1, 'Slot phase counts differ')
                require((chains == 0 and counts[0] == 0) or chains <= counts[0] <= 8 * chains,
                        'Completed-chain and draft-step counts differ')
                current[slot] = {'slot': slot, 'completed_chains': chains, 'phases': phases,
                                 'source_line': line_number}
                continue
            host_match = HOST_RE.fullmatch(line)
            require(host_match is not None, 'Malformed expected observation line')
            require(set(current) == set(range(slots)), 'Host summary lacks a complete slot set')
            verdicts, skips = int(host_match[1]), int(host_match[2])
            tail = host_match[3]
            require(tail.endswith(HOST_SUFFIX), 'Missing host counter interpretation suffix')
            tail = tail[:-len(HOST_SUFFIX)]
            match = re.fullmatch(r'(.+) submitted_unreaped_pump_samples=\[([^\]]*)\] '
                                 r'unsubmitted_jobs_at_verdict=\[([^\]]*)\]', tail)
            require(match is not None, 'Malformed host histograms')
            gap = parse_stat(match[1], 'verdict_gap')
            pending = histogram(match[2], slots)
            unsubmitted = histogram(match[3], slots)
            require(gap['n'] == max(verdicts - 1, 0), 'Verdict gap count differs from verdict count')
            require(sum(unsubmitted) == verdicts, 'Unsubmitted histogram count differs from verdict count')
            require(sum(pending) > 0, 'Drained summary has no pump samples')
            blocks.append({'index': len(blocks), 'slots': [current[i] for i in range(slots)],
                           'host': {'verdicts': verdicts, 'stage0_draft_not_ready_group_skips': skips,
                                    'verdict_gap': gap, 'submitted_unreaped_pump_samples': pending,
                                    'unsubmitted_jobs_at_verdict': unsubmitted, 'source_line': line_number}})
            current = {}
        except ObservationError as error:
            raise ObservationError('Line ' + str(line_number) + ': ' + str(error)) from None
    require(not current, 'Incomplete final observation block')
    require(bool(blocks), 'No complete observation blocks')
    aggregate = []
    for slot in range(slots):
        rows = [block['slots'][slot] for block in blocks]
        aggregate.append({'slot': slot, 'completed_chains': sum(row['completed_chains'] for row in rows),
                          'phases': {name: merge_stats([row['phases'][name] for row in rows])
                                     for name in ('body', 'head_return', 'tail')}})
    hosts = [block['host'] for block in blocks]
    totals = {'verdicts': sum(row['verdicts'] for row in hosts),
              'stage0_draft_not_ready_group_skips': sum(row['stage0_draft_not_ready_group_skips'] for row in hosts),
              'verdict_gap': merge_stats([row['verdict_gap'] for row in hosts]),
              'submitted_unreaped_pump_samples': [sum(row['submitted_unreaped_pump_samples'][i] for row in hosts)
                                                  for i in range(slots + 1)],
              'unsubmitted_jobs_at_verdict': [sum(row['unsubmitted_jobs_at_verdict'][i] for row in hosts)
                                             for i in range(slots + 1)]}
    return {'version': VERSION, 'valid': True, 'slot_count': slots, 'block_count': len(blocks),
            'blocks': blocks, 'aggregate_slots': aggregate, 'aggregate_host': totals, 'interpretation': NOTES,
            'performance_compared': False, 'packing_implemented': False}


def self_test():
    phase = '{}_ms(n=2,mean=1.000000,min=0.500000,max=1.500000,sum=2.000000)'
    lines = ['strata piped MTP observe: slot ' + str(i) + ', completed_chains=1, source_stream_intervals ' +
             ' '.join(phase.format(name) for name in ('body', 'head_return', 'tail')) for i in range(2)]
    host = ('strata piped MTP observe host: verdicts=2, stage0_draft_not_ready_group_skips=7 '
            'verdict_gap_ms(n=1,mean=3.000000,min=3.000000,max=3.000000,sum=3.000000) '
            'submitted_unreaped_pump_samples=[0:1,1:2,2:3] unsubmitted_jobs_at_verdict=[0:0,1:2,2:0]' + HOST_SUFFIX)
    valid = '\n'.join(['unrelated engine line'] + lines + [host])
    report = parse_text(valid + '\n' + valid, slots=2)
    assert report['block_count'] == 2 and report['aggregate_slots'][0]['phases']['body']['n'] == 4
    assert report['aggregate_host']['verdict_gap']['n'] == 2  # no artificial gap across reset reports
    broken = ['', '\n'.join(lines), '\n'.join(lines + [lines[0], host]), valid.replace('tail_ms(n=2', 'tail_ms(n=1'),
              valid.replace('sum=2.000000', 'sum=7.000000', 1), valid.replace('verdicts=2', 'verdicts=3'),
              valid.replace('1:2,2:0', '1:3,2:0'), valid.replace('max=1.500000', 'max=nan', 1),
              valid.replace('slot 1', 'slot 2'), valid.replace('source_stream_intervals', 'unknown', 1),
              valid.replace('0:1,1:2,2:3', '0:1,0:2,2:3')]
    for text in broken:
        try:
            parse_text(text, slots=2)
        except ObservationError:
            continue
        raise AssertionError('Malformed synthetic fixture unexpectedly accepted')
    rounded = parse_stat('body_ms(n=3,mean=0.333333,min=0.333333,max=0.333334,sum=1.000000)', 'body')
    assert rounded['n'] == 3
    print(json.dumps({'offline_self_test': 'PASS', 'invalid_cases_rejected': len(broken),
                      'network_requests': 0, 'gpu_operations': 0}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path, help='New directory; writes observation.json')
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.log is None or args.out is None:
        parser.error('--log and --out are required')
    if args.out.exists():
        parser.error('--out must not exist')
    report = {'version': VERSION, 'valid': False, 'performance_compared': False, 'interpretation': NOTES}
    try:
        raw = args.log.read_bytes()
        report = parse_text(raw.decode('utf-8', errors='strict'), args.slots)
        report['log_sha256'] = hashlib.sha256(raw).hexdigest()
    except (ObservationError, OSError, UnicodeError) as error:
        report['error_kind'] = type(error).__name__
        if isinstance(error, ObservationError):
            report['error'] = str(error)
    report['parser_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    (args.out / 'observation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'valid': report['valid'], 'block_count': report.get('block_count', 0),
                      'performance_compared': False}))
    return 0 if report['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
