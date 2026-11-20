#!/usr/bin/env python3
"""Offline matched-arm comparison. No service/GPU/network operations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMPARATORS = {'c8': ('c8_check.py', 'c8-comparison.json'),
               'correctness': ('correctness.py', 'comparison.json'),
               'budget': ('budget_correctness.py', 'budget-comparison.json'),
               'long': ('long_concurrency_check.py', 'comparison.json'),
               'prefill': ('prefill_target_head_check.py', 'comparison.json')}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed off arm directory')
    parser.add_argument('--candidate', type=Path, required=True, help='Completed on arm directory')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not exist')
    arms = [args.baseline, args.candidate]
    meta = [read(p / 'metadata.json') for p in arms]
    manifests = [read(p.parent / 'manifest.json') for p in arms]
    suites = [set(m['suite_returncodes']) for m in meta]
    checks = {'off_then_on': [m['arm'] for m in meta] == ['off', 'on'],
              'same_suites': suites[0] == suites[1],
              'all_selected_suites_succeeded': all(v == 0 for m in meta for v in m['suite_returncodes'].values()),
              'only_target_head_config_differs': meta[0]['config_except_variables_log_key_sha256'] == meta[1]['config_except_variables_log_key_sha256'],
              'target_flag_off_on': [m['config']['env'].get('STRATA_PIPED_TARGET_HEAD') for m in meta] == ['0', '1'],
              'mtp_head_fixed_on': all(m['config']['env'].get('STRATA_PIPED_MTP_HEAD') == '1' for m in meta)}
    for key in ('same_binary_sha256', 'adjacent_library_sha256', 'base_config_sha256', 'server_sha256', 'harness_sha256'):
        checks[key + '_equal'] = manifests[0][key] == manifests[1][key]
    args.out.mkdir(parents=True)
    report = {'scope': 'Matched P1 arms; all output correctness is exact. Performance remains descriptive; no deployment action.',
              'identity_checks': checks, 'suites': {}, 'suite_checks': {}, 'all_checks_passed': False}
    if not all(checks.values()):
        save(args.out / 'comparison.json', report)
        raise SystemExit('Arm identity/configuration qualification failed; no comparisons accepted')
    for suite in sorted(suites[0]):
        if suite == 'c4mixed':
            summaries = [read(p / suite / 'results.json') for p in arms]
            plans = [read(p / suite / 'manifest.json') for p in arms]
            valid = (all(len(rows) == 1 and rows[0]['resident_workers'] == 4 and not rows[0]['error']
                         and rows[0]['baseline_occupancy_qualified'] and len(rows[0]['injections']) >= 2 for rows in summaries)
                     and plans[0]['plan'] == plans[1]['plan'] and plans[0]['fixture_id'] == plans[1]['fixture_id'])
            report['suites'][suite] = {'both_runs_qualified': valid, 'summaries': summaries,
                                        'note': 'Scheduled injections and refill/cancel timing differ by runtime. Compare TTFT, chunk gaps, occupancy, fairness and recovery; chunks/byte rates are not token timestamps.'}
            report['suite_checks'][suite] = valid
            continue
        script, filename = COMPARATORS[suite]
        destination = args.out / suite
        command = [sys.executable, str(HERE / script), '--compare', str(arms[0] / suite),
                   str(arms[1] / suite), '--out', str(destination)]
        with (args.out / (suite + '-comparison.log')).open('w') as handle:
            child = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, timeout=120)
        result_path = destination / filename
        result = read(result_path) if result_path.is_file() else {'missing_comparison_result': True}
        if suite == 'prefill':
            passed = child.returncode == 0 and result.get('comparison_eligible') and result.get('all_greedy_outputs_identical')
            changes = [result.get('metrics', {}).get(kind, {}).get(metric, {}).get('on_relative_change_pct')
                       for kind in ('cold', 'warm') for metric in ('read_ms', 'ttft_s')]
            result['observed_prefill_medians_nonincreasing'] = all(
                isinstance(value, (int, float)) and value <= 0 for value in changes)
            result['performance_note'] = 'Three samples/kind; compare read_ms/TTFT spread and repeat reverse order if ambiguous. This boolean is descriptive, not a stable SLO or deployment gate.'
        else:
            passed = child.returncode == 0 and result.get('all_checks_passed')
        report['suites'][suite] = result
        report['suite_checks'][suite] = bool(passed)
    report['all_checks_passed'] = all(report['identity_checks'].values()) and all(report['suite_checks'].values())
    save(args.out / 'comparison.json', report)
    print(json.dumps({'all_checks_passed': report['all_checks_passed'], 'suite_checks': report['suite_checks'],
                      'performance_requires_review': True}))
    return 0 if report['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
