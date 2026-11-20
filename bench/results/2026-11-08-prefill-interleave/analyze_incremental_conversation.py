#!/usr/bin/env python3
"""Offline v2 conversation evidence and visible SSE gaps; no HTTP/GPU operations.

Single-arm output is diagnostic. Comparative metrics require a complete exact
v2 pair and all four residents visibly covering each measured phase. Raw chunk
gaps are neither token latency nor pure decode throughput.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FREEZE_SHA = '75a15bb36c851d994b70eb594f49e836ef26e0dcfab06185af22e7119133fbee'
VISIBLE_SHA = '4413f7fc433d08d0b820fc9d98d5f4271cb061469209e6c6d037ccbd8f9e46c3'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def dependencies():
    frozen_path = HERE / 'incremental-harness-v2-freeze.json'
    require(sha(frozen_path) == FREEZE_SHA, 'v2 freeze identity differs')
    frozen = read(frozen_path)
    directory = HERE / 'incremental-harness-v2'
    require({p.name for p in directory.iterdir()} == set(frozen['files_sha256']), 'v2 package file set differs')
    for name, expected in frozen['files_sha256'].items():
        require(Path(name).name == name and sha(directory / name) == expected, 'v2 source identity differs: ' + name)
    path = HERE / 'analyze_visible_silence.py'
    require(sha(path) == VISIBLE_SHA, 'Visible SSE analysis dependency differs')
    spec = importlib.util.spec_from_file_location('frozen_visible_silence', path)
    visible = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(visible)
    return frozen, directory, visible


def case_summary(arm, visible):
    directory = arm / 'incremental'
    path = directory / 'results.json'
    result = read(path)
    phases = {}
    for name, channels in result.get('phases', {}).items():
        bounds = channels['visible_content']
        phases[name] = visible.phase(result['records'], bounds['begin_epoch'], bounds['end_epoch'])
    records = []
    for record in result['records']:
        timings = record.get('native_timings', {})
        ev = record.get('admission_read_evidence', {})
        reader = ev.get('reader_pump', {})
        usage = record.get('usage', {})
        dependency = record.get('continuation_dependency', {})
        previous_prompt = dependency.get('previous_actual_prompt_tokens')
        previous_turn = previous_prompt - 7 if type(previous_prompt) is int else None
        prompt = usage.get('prompt_tokens')
        cached = record.get('cached_tokens')
        row = {
            'name': record['name'], 'role': record['role'], 'request_sha256': record.get('request_sha256'),
            'actual_prompt_tokens': prompt, 'actual_completion_tokens': usage.get('completion_tokens'),
            'finish_reason': record.get('finish_reason'), 'cached_tokens': cached,
            'fresh_prompt_tokens': timings.get('prompt_n'), 'native_prompt_ms': timings.get('prompt_ms'),
            'visible_ttft_s': record.get('first_visible_content_s'),
            'count_preflight': record.get('count_preflight'),
            'previous_last_user_turn_checkpoint': previous_turn,
            'fresh_upper_bound_from_previous_turn': prompt - previous_turn if type(prompt) is int and previous_turn is not None else None,
            'conversation_messages_sha256': record.get('conversation_messages_sha256'),
            'continuation_dependency': dependency,
            'native_prompt_records': ev.get('native_prompt_records', []),
            'restore_records': ev.get('restore_records', []),
            'observed_reader_endpoints': [r['from'] + r['tokens'] for r in reader.get('pairs', []) + reader.get('fallback', [])],
            'strict_in_read_native_output_rows': sum(r['decode_rows_before_done'] for r in reader.get('pairs', [])),
            'failed_gates': [k for k, v in record.get('gates', {}).items() if v is not True],
        }
        if record['role'] == 'injected' and type(record.get('first_visible_content_s')) in (int, float) and record['first_visible_content_s'] > 0:
            begin = record['start_epoch']
            row['peers_request_to_first_visible'] = visible.phase(result['records'], begin, begin + record['first_visible_content_s'])
        records.append(row)
    return {
        'source': str(arm.resolve()), 'source_results_sha256': sha(path),
        'case_qualified': result.get('all_checks_passed') is True,
        'failed_checks': [k for k, v in result.get('checks', {}).items() if v is not True],
        'phases': phases, 'requests': records,
        'whole_case_api_rate_including_seed': result.get('whole_case_api_rate'),
        'post_seed_finite_api_rate': result.get('post_seed_finite_api_rate'),
        'all_four_visible_cover_complete_phases': set(phases) == {'baseline', 'mixed', 'recovery'} and all(p['all_four_visible_cover_phase'] for p in phases.values()),
    }


def package_run_identity(arm, frozen):
    manifest = read(arm.parent / 'manifest.json')
    require(manifest.get('harness_sha256') == frozen['files_sha256'], 'Run did not use frozen v2')
    require(manifest.get('suites') == ['incremental'], 'Use the independent incremental-only paired run')
    for name, expected in frozen['files_sha256'].items():
        require(sha(arm.parent / 'harness' / name) == expected, 'Run frozen source differs: ' + name)


def matched_metrics(summaries, visible):
    maps = [{r['name']: r for r in s['requests']} for s in summaries]
    output = {'phases': visible.compare_phases(*summaries), 'injections': []}
    for key in ('whole_case_api_rate_including_seed', 'post_seed_finite_api_rate'):
        output[key] = visible.delta(*[s[key]['tokens_per_s'] for s in summaries])
    for name in sorted(maps[0]):
        rows = [m[name] for m in maps]
        if rows[0]['role'] != 'injected':
            continue
        output['injections'].append({
            'name': name, 'actual_prompt_tokens': rows[0]['actual_prompt_tokens'],
            'actual_completion_tokens': rows[0]['actual_completion_tokens'],
            'cached_tokens': rows[0]['cached_tokens'], 'fresh_prompt_tokens': rows[0]['fresh_prompt_tokens'],
            'visible_ttft_s': visible.delta(*[r['visible_ttft_s'] for r in rows]),
            'native_prompt_ms': visible.delta(*[r['native_prompt_ms'] for r in rows]),
            'count_preflight_s': visible.delta(*[r['count_preflight']['duration_s'] for r in rows]),
            'max_lane_request_to_first_visible_gap_s': visible.delta(*[r['peers_request_to_first_visible']['max_lane_phase_clipped_silence_s'] for r in rows]),
        })
    return output


def self_test(visible):
    peers = [{'name': 'peer-' + str(i), 'lane': i, 'role': 'resident', 'chunks': [
        {'epoch': t, 'bytes': {'visible_content': 1}} for t in (1., 2., 6., 10.)]} for i in range(4)]
    report = {'records': peers, 'phases': {name: {'visible_content': {'begin_epoch': a, 'end_epoch': b}} for name, a, b in (
        ('baseline', 1., 2.), ('mixed', 2., 6.), ('recovery', 6., 9.))}, 'all_checks_passed': True, 'checks': {'test': True}}
    with tempfile.TemporaryDirectory(prefix='incremental-numeric-cpu-') as root:
        arm = Path(root) / 'arm'; (arm / 'incremental').mkdir(parents=True)
        target = arm / 'incremental/results.json'; target.write_text(json.dumps(report))
        result = case_summary(arm, visible)
        require(result['all_four_visible_cover_complete_phases'], 'Positive coverage failed')
        require(result['phases']['mixed']['collective_no_visible_event_gaps']['max_s'] == 4., 'Collective gap differs')
        bad = copy.deepcopy(report); bad['records'][0]['chunks'].pop(); bad['records'][0]['end_epoch'] = 100.
        target.write_text(json.dumps(bad))
        require(not case_summary(arm, visible)['all_four_visible_cover_complete_phases'], 'Late HTTP end covered absent visible content')
        bad = copy.deepcopy(report); bad['phases'].pop('recovery'); target.write_text(json.dumps(bad))
        require(not case_summary(arm, visible)['all_four_visible_cover_complete_phases'], 'Incomplete phases accepted')
    print('PASS: frozen dependency identities, visible-only complete-phase coverage, missing-phase and late-HTTP negatives; no network/GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single-arm', type=Path)
    parser.add_argument('--baseline', type=Path); parser.add_argument('--candidate', type=Path)
    parser.add_argument('--out', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(); frozen, directory, visible = dependencies()
    if args.self_test:
        self_test(visible); return 0
    require(args.out is not None and not args.out.exists(), 'Explicit new output path required')
    result = {'script_sha256': sha(Path(__file__)), 'v2_freeze_sha256': FREEZE_SHA, 'visible_analysis_sha256': VISIBLE_SHA,
              'performance_comparison_eligible': False, 'comparative_metrics': None,
              'scope': 'Actual visible SSE chunk arrivals; raw p95 contains full gaps ending inside a phase, clipped gaps cover phase intersections. Incomplete visibility censors tails. Counts and cache are actual API/native values. Neither token TBT nor pure decode throughput. No deployment verdict.'}
    if args.single_arm:
        require(not args.baseline and not args.candidate, 'Choose single diagnostic or complete pair')
        package_run_identity(args.single_arm, frozen)
        result.update(mode='single_arm_diagnostic', summaries=[case_summary(args.single_arm, visible)])
    else:
        require(args.baseline and args.candidate, 'Both arms required')
        for arm in (args.baseline, args.candidate):
            package_run_identity(arm, frozen)
        with tempfile.TemporaryDirectory(prefix='incremental-exact-review-') as temporary:
            out = Path(temporary) / 'comparison'
            child = subprocess.run([sys.executable, '-B', str(directory / 'compare_interleave.py'), '--baseline', str(args.baseline),
                '--candidate', str(args.candidate), '--out', str(out)], capture_output=True, text=True, timeout=120)
            comparison = read(out / 'comparison.json') if (out / 'comparison.json').is_file() else {'all_checks_passed': False, 'missing_comparison': True}
        summaries = [case_summary(arm, visible) for arm in (args.baseline, args.candidate)]
        eligible = child.returncode == 0 and comparison.get('all_checks_passed') is True and all(s['all_four_visible_cover_complete_phases'] for s in summaries)
        result.update(mode='matched_complete_v2_pair', exact_comparator=comparison, summaries=summaries, performance_comparison_eligible=eligible,
                      comparative_metrics=matched_metrics(summaries, visible) if eligible else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'out': str(args.out.resolve()), 'mode': result['mode'], 'performance_comparison_eligible': result['performance_comparison_eligible']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
