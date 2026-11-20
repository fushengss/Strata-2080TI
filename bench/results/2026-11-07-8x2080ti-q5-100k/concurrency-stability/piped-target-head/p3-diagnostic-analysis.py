#!/usr/bin/env python3
"""Offline, numeric-only audit of the completed P3 v2 C8 observation.

Reads local frozen artifacts; makes no service/network/GPU calls. Recomputes
weighted phase totals from strict parsing of the engine log, verifies bound file
hashes and compatibility evidence, writes only this analysis JSON/text.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'p3-diagnostic-v2'
ARM = RUN / '01-off'
EXPECTED_ENGINE = 'f36c12833e900a77cdc966298863f15ded275696855405816b4b2377208683b2'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read(RUN / 'manifest.json')
    meta = read(ARM / 'metadata.json')
    observation = read(ARM / 'p3-observation.json')
    c8 = read(ARM / 'c8/c8-check.json')
    compat = read(ROOT / 'p3-compatibility/comparison.json')
    parser_path = RUN / 'harness/parse_piped_mtp_observe.py'
    spec = importlib.util.spec_from_file_location('frozen_p3_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parsed = parser.parse_text((ARM / 'engine.log').read_text(), slots=8)
    checks = {
        'same_expected_v2_engine': manifest['same_binary_sha256'] == meta['engine_sha256'] == EXPECTED_ENGINE,
        'diagnostic_only': manifest['diagnostic_only'] and meta['diagnostic_only'],
        'only_c8_measured': manifest['suites'] == ['c8'],
        'fixed_p1_on_p2_off_observation_on': all(meta['config']['env'][k] == v for k, v in {
            'STRATA_PIPED_TARGET_HEAD': '1', 'STRATA_PIPED_ASYNC_SAMPLE': '0',
            'STRATA_PIPED_MTP_OBSERVE': '1'}.items()),
        'suite_passed': meta['suite_returncodes'] == {'c8': 0} and c8['all_checks_passed']
            and all(c8['checks'].values()),
        'eight_slots_resident_buffers': meta['startup_facts']['all_experts_resident']
            and meta['startup_facts']['all_eight_have_3_80_gib_own_buffers']
            and meta['startup_facts']['batch_slot_devices'] == list(range(8)),
        'eight_observation_event_sets': meta['startup_facts']['piped_mtp_observe_startup'] == [8],
        'log_hash_matches': observation['source_log_sha256'] == sha(ARM / 'engine.log'),
        'parser_hash_matches': observation['parser_sha256'] == sha(parser_path),
        'saved_observation_recomputed': all(parsed[k] == observation[k] for k in (
            'valid', 'slot_count', 'block_count', 'aggregate_slots', 'aggregate_host')),
        'suite_observation_matches': all(meta['diagnostic_suite_evidence']['c8'][k] == parsed[k]
            for k in ('valid', 'slot_count', 'block_count', 'aggregate_slots', 'aggregate_host')),
        'frozen_harness_hashes_match': all(sha(RUN / 'harness' / name) == value
            for name, value in manifest['harness_sha256'].items()),
        'compatibility_passed': compat['all_checks_passed'] and len(compat['pairs']) == 8
            and all(p['passed'] and all(p['checks'].values()) for p in compat['pairs'])
            and all(compat['identity_checks'].values()),
        'compatibility_source_hashes_match': all(sha(Path(name)) == value
            for name, value in compat['source_sha256'].items()),
        'no_performance_comparison': observation['performance_compared'] is False
            and compat['performance_compared'] is False,
        'no_packing_implementation': observation['packing_implemented'] is False,
    }
    if not all(checks.values()):
        raise ValueError('Failed numeric audit checks: ' + ', '.join(k for k, v in checks.items() if not v))
    phases = {name: parser.merge_stats([s['phases'][name] for s in parsed['aggregate_slots']])
              for name in ('body', 'head_return', 'tail')}
    count = phases['body']['n']
    assert len({p['n'] for p in phases.values()}) == 1
    chain_count = sum(s['completed_chains'] for s in parsed['aggregate_slots'])
    host = parsed['aggregate_host']
    same_step_mean_span = sum(p['mean_ms'] for p in phases.values())
    pending = host['submitted_unreaped_pump_samples']
    live = host['unsubmitted_jobs_at_verdict']
    route = c8['numeric_piped_log']['records']
    assert len(route) == 1 and route[0]['group_steps'] == host['verdicts']
    result = {
        'version': 'p3-diagnostic-independent-analysis-1', 'all_checks_passed': True,
        'checks': checks, 'fixed_modes': {'p1': 1, 'p2': 0, 'p3_observation': 1},
        'engine_sha256': EXPECTED_ENGINE,
        'scope': {'suite': 'C8 short synthetic', 'requests': 8,
            'prompt_tokens': sorted({r['usage']['prompt_tokens'] for r in c8['records']}),
            'completion_tokens_per_request': sorted({r['usage']['completion_tokens'] for r in c8['records']}),
            'total_api_completion_tokens': sum(r['usage']['completion_tokens'] for r in c8['records']),
            'normal_completed_cases_compatible': len(compat['pairs']),
            'tool_channel_validated': False, 'long_context_diagnostic': False,
            'same_binary_observation_off_on_overhead_measured': False},
        'block_count': parsed['block_count'], 'completed_chains': chain_count,
        'completed_steps': count, 'weighted_phase_intervals': phases,
        'sum_of_phase_means_same_step_ms': same_step_mean_span,
        'mean_span_is_not_whole_engine_walltime': True,
        'slots': parsed['aggregate_slots'], 'host': host,
        'pump_histogram_total_samples': sum(pending),
        'pump_histogram_max_nonzero_bucket': max(i for i, n in enumerate(pending) if n),
        'verdict_histogram_max_nonzero_bucket': max(i for i, n in enumerate(live) if n),
        'native_route_accounting': {k:v for k,v in route[0].items()
            if k in ('groups','slots_per_group','group_steps','rows','drafts_accepted','drafts_offered')},
        'interpretation': [
            'Each phase mean is sum_ms / observed step count across slots; not an unweighted mean of means.',
            'Source-stream intervals contain contention and dependencies. Body excludes pre-body residual staging. Head-return includes remote projection and mapped logits return.',
            'The three same-step phases sum to an observed draft-chain span, not whole-engine time, GPU occupancy, or a possible speedup. Cross-slot sums can overlap.',
            'Submitted-but-unreaped chains were 0 or 1 at pump entry; some may already have finished on GPU. These are pump samples, not duration shares.',
            'Unsubmitted verdict jobs were 0 or 1 by the eight-group/one-slot structure; they have not executed their bodies and do not form a post-attention ready queue.',
            'Stage0 draft-not-ready skips are repeated scheduling attempts. They prove this condition occurred, but not how many requests stalled or how much wall time was lost.',
            'Host verdict gaps include admission, prefill, drain/refill and scheduling. Their mean/max is not pure decode interarrival or measured GPU stage latency.',
            'This short-context instrumented run observes no multi-chain backlog at the sampled points. It does not establish profitable packing; waiting for a batch can add latency and requires a new experiment.',
            'No 100k or mixed-admission diagnostic was run. Instrumentation overhead has not been isolated with an observation off/on pair on the same binary.',
            'C8 compatibility validates eight normal completed outputs/usage/finish reasons, not cancellation, tool channels, or every workload.'
        ],
        'performance_compared': False, 'packing_implemented': False,
        'recommended_deployment': 'Keep P3 disabled; use qualified P1 binary. Treat P3 as diagnostic evidence only.',
    }
    inputs = [RUN/'manifest.json', ARM/'metadata.json', ARM/'p3-observation.json', ARM/'engine.log',
              ARM/'c8/c8-check.json', ROOT/'p3-compatibility/comparison.json', parser_path]
    result['input_sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    result['analysis_script_sha256'] = sha(Path(__file__))
    (ROOT/'p3-diagnostic-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['P3 v2 independent numeric audit: PASS',
        'Scope: C8, 8 normal completed synthetic requests, 66/68 prompt tokens, 1024 output tokens each; 8/8 exact compatible with qualified P2-off run.',
        f'Observed completed chains/steps: {chain_count}/{count}; drained blocks: {parsed["block_count"]}.']
    for name,p in phases.items():
        lines.append(f'{name}: weighted mean {p["mean_ms"]:.9f} ms, n={p["n"]}, min={p["min_ms"]:.6f}, max={p["max_ms"]:.6f}.')
    lines.extend([f'Same-step sum of phase means: {same_step_mean_span:.9f} ms (not engine wall time).',
        f'Verdicts={host["verdicts"]}; gap n={host["verdict_gap"]["n"]}, mean={host["verdict_gap"]["mean_ms"]:.9f} ms, max={host["verdict_gap"]["max_ms"]:.6f} ms.',
        'Submitted-unreaped pump histogram='+str(pending),
        'Unsubmitted-at-verdict histogram='+str(live),
        f'Stage0 draft-not-ready scheduling attempts={host["stage0_draft_not_ready_group_skips"]}; repeated attempts, neither unique jobs nor time.',
        *result['interpretation'], result['recommended_deployment']])
    (ROOT/'p3-diagnostic-analysis.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'all_checks_passed':True,'checks':len(checks),'steps':count,
        'weighted_means_ms':{k:v['mean_ms'] for k,v in phases.items()},'packing_implemented':False}))


if __name__ == '__main__':
    main()
