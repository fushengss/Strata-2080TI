#!/usr/bin/env python3
"""Offline head AB/BA audit; local saved files only, no service/GPU operations."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
DEPENDENCY = HERE.parent / 'head-pilot-analysis/analyze.py'
spec = importlib.util.spec_from_file_location('head_pilot_common', DEPENDENCY)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def metrics(prefill, c8):
    result = {'prefill': {}, 'c8': {}}
    for kind in ('cold', 'warm'):
        result['prefill'][kind] = {}
        for metric in ('read_ms', 'ttft_s', 'decode_ms'):
            vals = {arm: [row['ttft_s'] if metric == 'ttft_s' else row['engine'][metric]
                          for row in rows if row['kind'] == kind] for arm, rows in prefill.items()}
            meds = {arm: statistics.median(values) for arm, values in vals.items()}
            result['prefill'][kind][metric] = {'values': vals, **meds,
                'on_change_pct': common.change(meds['off'], meds['on'])}
    result['c8'] = {arm: report['timing']['whole_batch'] for arm, report in c8.items()}
    result['c8_on_throughput_change_pct'] = common.change(result['c8']['off']['completed_api_tokens_per_s'],
                                                        result['c8']['on']['completed_api_tokens_per_s'])
    return result


def inspect_round(base, tags, expected_suites, sequence):
    audit = common.Audit(base)
    manifest = audit.read(Path('manifest.json'))
    metadata = {arm: audit.read(tag / 'metadata.json') for arm, tag in tags.items()}
    audit.check('engine_server_library', manifest['same_binary_sha256'] == common.ENGINE
                and manifest['server_sha256'] == common.SERVER and manifest['adjacent_library_sha256'] == common.LIBRARY)
    audit.check('requested_sequence_and_suites', manifest['sequence'] == sequence and manifest['suites'] == expected_suites)
    audit.check('declared_head_only_variable', manifest['variables'] == {common.HEAD_FLAG: {'off': '0', 'on': '1'}})
    audit.check('same_config_except_head', metadata['off']['config_except_variables_log_key_sha256'] == metadata['on']['config_except_variables_log_key_sha256'])
    for name, expected in manifest['harness_sha256'].items():
        path = Path('harness') / name
        actual = common.sha(base / path)
        audit.sources[str(path)] = actual
        audit.check('frozen_source:' + name, actual == expected)
    env = {arm: meta['config']['env'] for arm, meta in metadata.items()}
    delta = {key: {arm: env[arm].get(key) for arm in tags} for key in sorted(set(env['off']) | set(env['on']))
             if env['off'].get(key) != env['on'].get(key)}
    audit.check('config_head_only_difference', delta == {common.HEAD_FLAG: {'off': '0', 'on': '1'}})
    head_steps = {}
    for arm, meta in metadata.items():
        actual = meta['actual_engine']
        audit.check(arm + ':actual_binary_library_flags', len(actual) == 1
                    and actual[0]['engine_sha256'] == common.ENGINE and actual[0]['library_sha256'] == common.LIBRARY
                    and actual[0]['expected_library_mapped']
                    and all(actual[0]['flags'].get(k) == v for k, v in common.FIXED.items())
                    and actual[0]['flags'].get(common.HEAD_FLAG) == ('1' if arm == 'on' else '0'))
        audit.check(arm + ':fixed_config_cap_suites', meta['config']['reasoning_budget_engine_cap'] is True
                    and all(env[arm].get(k) == v for k, v in common.FIXED.items())
                    and set(meta['suite_returncodes']) == set(expected_suites)
                    and all(v == 0 for v in meta['suite_returncodes'].values()))
        facts = meta['startup_facts']
        audit.check(arm + ':resources_and_workspace', facts['all_experts_resident']
                    and facts['all_eight_have_3_80_gib_own_buffers'] and facts['batch_slot_devices'] == list(range(8))
                    and facts['piped_mtp_startup_enabled']
                    and facts['shared_head_workspaces'] == ([[8, 3, 5, 7]] if arm == 'on' else []))
        log_rows = [[int(v) for v in row] for row in common.HEAD_RE.findall(audit.text(tags[arm] / 'engine.log'))]
        evidence = meta['shared_head_suite_evidence']
        head_steps[arm] = {'suites': evidence, 'all_log_steps': sum(row[3] for row in log_rows)}
        audit.check(arm + ':head_steps', sum(item['steps'] for item in evidence.values()) == sum(row[3] for row in log_rows)
                    and all(item['steps'] == sum(row[3] for row in item['rows']) for item in evidence.values())
                    and evidence['prefill']['steps'] == 0
                    and (not log_rows if arm == 'off' else evidence['c8']['steps'] > 0
                         and all(0 <= row[0] < 8 and row[1:3] == [3, 5] for row in log_rows)))
    prefill = {arm: audit.read(tag / 'prefill/results.json') for arm, tag in tags.items()}
    c8 = {arm: audit.read(tag / 'c8/c8-check.json') for arm, tag in tags.items()}
    pairs = {'prefill': common.pair_rows(prefill['off'], prefill['on']),
             'c8': common.pair_rows(c8['off']['records'], c8['on']['records'])}
    audit.check('six_prefill_full_exact_pairs', len(pairs['prefill']) == 6 and all(p['passed'] for p in pairs['prefill']))
    audit.check('eight_c8_full_exact_pairs', len(pairs['c8']) == 8 and all(p['passed'] for p in pairs['c8']))
    for arm in tags:
        audit.check(arm + ':prefill_valid_exact_cache', len(prefill[arm]) == 6 and all(r['valid']
            and 95000 <= r['usage']['prompt_tokens'] <= 105000 and r['usage']['completion_tokens'] == 256
            and ((r['engine']['reused_tokens'] == 0 and common.cached(r) == 0) if r['kind'] == 'cold' else
                 r['engine']['reused_tokens'] == common.cached(r) and r['engine']['read_tokens'] == 7) for r in prefill[arm]))
        r = c8[arm]
        audit.check(arm + ':c8_qualified_8192', r['all_checks_passed'] and r['expected_piped_mtp'] == 'on'
                    and r['timing']['overlap']['eight_streams_observed']
                    and r['timing']['whole_batch']['completed_api_tokens'] == 8192
                    and all(x['usage']['completion_tokens'] == 1024 and x['finish_reason'] == 'length' for x in r['records']))
    return {'checks': audit.checks, 'all_checks_passed': all(audit.checks.values()), 'sequence': sequence,
            'suites': expected_suites, 'env_changes': delta, 'head_steps': head_steps, 'pairs': pairs,
            'timing': metrics(prefill, c8), 'sources_sha256': audit.sources,
            'normalized_config_sha256': metadata['off']['config_except_variables_log_key_sha256'],
            'measured_harness_sha256': {name: manifest['harness_sha256'][name]
                                      for name in ('prefill_head_check.py', 'c8_check.py', 'correctness.py')},
            'raw': {'prefill': prefill, 'c8': c8}}


def combine(rounds):
    pooled = {'prefill': {}, 'c8': {}}
    paired = []
    for label, r in rounds.items():
        by_arm = {arm: {row['name']: row for row in rows} for arm, rows in r['raw']['prefill'].items()}
        for name, a in by_arm['off'].items():
            b = by_arm['on'][name]
            paired.append({'round': label, 'name': name, 'kind': a['kind'],
                'read_ms': {'off': a['engine']['read_ms'], 'on': b['engine']['read_ms'],
                            'on_change_pct': common.change(a['engine']['read_ms'], b['engine']['read_ms'])},
                'ttft_s': {'off': a['ttft_s'], 'on': b['ttft_s'], 'on_change_pct': common.change(a['ttft_s'], b['ttft_s'])}})
    for kind in ('cold', 'warm'):
        pooled['prefill'][kind] = {}
        for metric in ('read_ms', 'ttft_s'):
            rows = [p for p in paired if p['kind'] == kind]
            med = {arm: statistics.median(p[metric][arm] for p in rows) for arm in ('off', 'on')}
            pooled['prefill'][kind][metric] = {**med, 'observations_per_arm': len(rows),
                'pooled_median_change_pct': common.change(med['off'], med['on']),
                'median_of_paired_relative_changes_pct': statistics.median(p[metric]['on_change_pct'] for p in rows)}
    for arm in ('off', 'on'):
        runs = [r['timing']['c8'][arm] for r in rounds.values()]
        seconds = sum(r['elapsed_s'] for r in runs)
        tokens = sum(r['completed_api_tokens'] for r in runs)
        pooled['c8'][arm] = {'completed_api_tokens': tokens, 'elapsed_s_sum': seconds,
                             'aggregate_tps': tokens / seconds,
                             'run_tps_median': statistics.median(r['completed_api_tokens_per_s'] for r in runs)}
    pooled['c8']['on_aggregate_tps_change_pct'] = common.change(pooled['c8']['off']['aggregate_tps'], pooled['c8']['on']['aggregate_tps'])
    pooled['scope'] = ('Descriptive pooled two-round AB/BA statistics only. Not a contiguous, identically ordered ABBA: '
                       'pilot runs budget before prefill/C8; reverse starts with prefill, and other tests separate rounds. '
                       'Startup, prior activity and thermal/cache state can differ; do not infer causal certainty or a confidence interval.')
    return pooled, paired


def render(result):
    lines = ['# 共享 piped MTP head 正反轮次复核', '',
             '仅离线分析已保存的 head-pilot 与 head-reverse；binary `c9305b08…`、server `40c5e9df…`。两臂 piped MTP 始终开启，只切换 `STRATA_PIPED_MTP_HEAD`。', '',
             f"身份、资源、完整输出、usage/cache 检查：{'全部通过' if result['all_checks_passed'] else '存在失败，见 summary.json checks'}。", '',
             '| 每轮指标 | 首轮 AB off | 首轮 AB on | 变化 | 反轮 BA off | 反轮 BA on | 变化 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('cold', 'warm'):
        for metric in ('read_ms', 'ttft_s'):
            a, b = [r['timing']['prefill'][kind][metric] for r in result['rounds'].values()]
            lines.append(f"| {kind} {metric} 中位数 | {a['off']:.3f} | {a['on']:.3f} | {a['on_change_pct']:+.2f}% | "
                         f"{b['off']:.3f} | {b['on']:.3f} | {b['on_change_pct']:+.2f}% |")
    a, b = [r['timing'] for r in result['rounds'].values()]
    lines.append(f"| C8 API tokens/s | {a['c8']['off']['completed_api_tokens_per_s']:.3f} | {a['c8']['on']['completed_api_tokens_per_s']:.3f} | "
                 f"{a['c8_on_throughput_change_pct']:+.2f}% | {b['c8']['off']['completed_api_tokens_per_s']:.3f} | "
                 f"{b['c8']['on']['completed_api_tokens_per_s']:.3f} | {b['c8_on_throughput_change_pct']:+.2f}% |")
    lines += ['', '每轮各有三次 cold、三次 warm；延迟正变化更慢、吞吐正变化更快。C8 每臂完成 8192 API tokens，整批 TPS 包含 admission/prefill 与尾部 drain。', '',
              '首轮顺序为 budget→prefill→C8→C4，反轮为 prefill→C8；两轮之间还有其他测试。因此以下合并值只是描述统计，不是热状态与 suite 顺序严格一致的连续 ABBA。', '',
              '| 描述性合并指标 | off | on | 变化 |', '|---|---:|---:|---:|']
    for kind in ('cold', 'warm'):
        for metric in ('read_ms', 'ttft_s'):
            r = result['pooled']['prefill'][kind][metric]
            lines.append(f"| {kind} {metric}，每臂6次的中位数 | {r['off']:.3f} | {r['on']:.3f} | {r['pooled_median_change_pct']:+.2f}% |")
    p = result['pooled']['c8']
    lines.append(f"| C8 总tokens/总耗时 | {p['off']['aggregate_tps']:.3f} | {p['on']['aggregate_tps']:.3f} | {p['on_aggregate_tps_change_pct']:+.2f}% |")
    lines += ['', '每个 fixture 的配对变化及其变化率中位数见 `summary.json`，不以合并值替代上方逐轮结果。', '',
              '两轮各检查六条 prefill、八条 C8 的完整 request/output/reasoning/channel hashes、全部 usage/cache；另外比较相同 arm 在两轮之间的输出兼容。检查实际进程 exe/映射库、冻结源码、唯一 head flag 差异、全专家驻留、8槽与每卡3.80GiB prefill buffer；确认 prefill 的 shared head steps=0、C8 on 存在实际 steps。', '',
              'C8 是短 prompt 场景，不能外推四路 100k；chunk arrivals 不能当作逐 token TBT。样本量小，顺序和热状态有差异，本报告不自动批准部署。', '',
              '复现：`python3 analyze.py --pilot ../head-pilot --reverse ../head-reverse --out .`。所有读取文件的 SHA256 和每项检查均在 `summary.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=HERE.parent / 'head-pilot')
    parser.add_argument('--reverse', type=Path, default=HERE.parent / 'head-reverse')
    parser.add_argument('--out', type=Path, default=HERE)
    args = parser.parse_args()
    rounds = {'AB': inspect_round(args.pilot.resolve(), {'off': Path('01-off'), 'on': Path('02-on')},
                                 ['budget', 'prefill', 'c8', 'performance'], ['off', 'on']),
              'BA': inspect_round(args.reverse.resolve(), {'on': Path('01-on'), 'off': Path('02-off')},
                                 ['prefill', 'c8'], ['on', 'off'])}
    cross_pairs = {}
    for arm in ('off', 'on'):
        cross_pairs[arm] = {'prefill': common.pair_rows(rounds['AB']['raw']['prefill'][arm], rounds['BA']['raw']['prefill'][arm]),
                           'c8': common.pair_rows(rounds['AB']['raw']['c8'][arm]['records'], rounds['BA']['raw']['c8'][arm]['records'])}
    checks = {'both_rounds_qualified': all(r['all_checks_passed'] for r in rounds.values()),
              'same_cross_round_fixed_config': rounds['AB']['normalized_config_sha256'] == rounds['BA']['normalized_config_sha256'],
              'same_measured_harness_sources': rounds['AB']['measured_harness_sha256'] == rounds['BA']['measured_harness_sha256'],
              'cross_round_exact_output_usage': all(p['passed'] for suites in cross_pairs.values() for rows in suites.values() for p in rows)}
    pooled, paired = combine(rounds)
    for r in rounds.values():
        del r['raw']
    result = {'checks': checks, 'all_checks_passed': all(checks.values()), 'rounds': rounds,
              'cross_round_pairs': cross_pairs, 'paired_prefill_changes': paired, 'pooled': pooled,
              'analysis_source_sha256': common.sha(Path(__file__)), 'analysis_dependency_sha256': common.sha(DEPENDENCY)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.out / 'README.md').write_text(render(result))
    print(json.dumps({'all_checks_passed': result['all_checks_passed'], 'checks': checks,
                      'round_C8_change_pct': {k: r['timing']['c8_on_throughput_change_pct'] for k, r in rounds.items()},
                      'pooled': pooled}, ensure_ascii=False))
    return 0 if result['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
