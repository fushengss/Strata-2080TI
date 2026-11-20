#!/usr/bin/env python3
"""Offline head-only pilot audit; reads local artifacts, never requests a service/GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

ENGINE = 'c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6'
SERVER = '40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9'
LIBRARY = 'fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661'
HEAD_FLAG = 'STRATA_PIPED_MTP_HEAD'
FIXED = {'STRATA_BATCH_MTP': '1', 'STRATA_PIPED_MTP': '1', 'STRATA_BATCH_SELF_COMMIT': '0',
         'STRATA_DECODE_HEAD_GPU': '2,4', 'STRATA_MTP_HEAD_GPU': '3,5', 'STRATA_PW_WITH_BATCH': '1'}
USAGE = ('prompt_tokens', 'completion_tokens', 'total_tokens')
# Actual logger prints CUDA3+5 (one CUDA prefix).
HEAD_RE = re.compile(r'strata piped MTP head: slot (\d+), CUDA(\d+)\+(\d+), (\d+) draft steps offloaded in this batch')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def change(a, b):
    return (b / a - 1) * 100 if a else None


def cached(row):
    usage = row.get('usage') or {}
    return (usage.get('prompt_tokens_details') or {}).get('cached_tokens', usage.get('cached_tokens'))


def complete(row):
    usage = row.get('usage') or {}
    return (not row.get('error') and row.get('saw_done') is True and row.get('finish_reason') in ('length', 'stop')
            and all(isinstance(usage.get(k), int) and not isinstance(usage[k], bool) for k in USAGE)
            and usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'])


def pair_rows(left, right, cancelled=False):
    a, b = ({r['name']: r for r in rows} for rows in (left, right))
    pairs = []
    for name in sorted(set(a) | set(b)):
        x, y = a.get(name), b.get(name)
        checks = {'present_in_both': bool(x and y)}
        row = {'name': name, 'checks': checks}
        if x and y:
            cancel = cancelled and bool(x.get('intentional_cancel') or y.get('intentional_cancel'))
            row.update(cancelled=cancel, usage={'off': x.get('usage'), 'on': y.get('usage')},
                       cache={'off': cached(x), 'on': cached(y)},
                       output_sha256={'off': x['output_sha256'], 'on': y['output_sha256']})
            checks['request_identical'] = x['request_sha256'] == y['request_sha256']
            if cancel:
                checks['both_cancelled_mid_generation'] = all(r.get('intentional_cancel') and r.get('closed_mid_generation')
                    and not r.get('saw_done') and not r.get('error') for r in (x, y))
                for field in ('output_text', 'reasoning_text'):
                    tx, ty = x[field], y[field]
                    checks[field + '_shorter_is_prefix'] = tx.startswith(ty) or ty.startswith(tx)
                checks['nonempty_shared_prefix'] = min(len(r['output_text']) + len(r['reasoning_text']) for r in (x, y)) > 0
                row['cancel_full_hash_equal'] = x['output_sha256'] == y['output_sha256']
                row['cancel_output_bytes'] = {'off': len(x['output_text'].encode()), 'on': len(y['output_text'].encode())}
            else:
                checks['both_complete'] = complete(x) and complete(y)
                checks['output_hash_identical'] = x['output_sha256'] == y['output_sha256']
                checks['three_usage_counts_identical'] = all(x['usage'].get(k) == y['usage'].get(k) for k in USAGE)
                checks['full_usage_identical'] = x['usage'] == y['usage']
                checks['finish_reason_identical'] = x['finish_reason'] == y['finish_reason']
                for field in ('reasoning_sha256', 'channel_output_sha256'):
                    if field in x or field in y:
                        checks[field + '_identical'] = x.get(field) == y.get(field)
            # When full synthetic output text exists, verify that its saved hash
            # really represents that complete text rather than trusting a label.
            for label, r in (('off', x), ('on', y)):
                if 'output_text' in r:
                    checks[label + '_stored_output_hash_valid'] = text_sha(r['output_text']) == r['output_sha256']
                    checks[label + '_stored_reasoning_hash_valid'] = text_sha(r['reasoning_text']) == r['reasoning_sha256']
        row['passed'] = all(checks.values())
        pairs.append(row)
    return pairs


class Audit:
    def __init__(self, base):
        self.base = base
        self.sources = {}
        self.checks = {}

    def read(self, relative):
        path = self.base / relative
        self.sources[str(relative)] = sha(path)
        return json.loads(path.read_text())

    def text(self, relative):
        path = self.base / relative
        self.sources[str(relative)] = sha(path)
        return path.read_text(errors='replace')

    def check(self, name, value):
        self.checks[name] = bool(value)


def analyze(base):
    audit = Audit(base)
    manifest = audit.read(Path('manifest.json'))
    tags = {'off': Path('01-off'), 'on': Path('02-on')}
    meta = {label: audit.read(tag / 'metadata.json') for label, tag in tags.items()}
    audit.check('expected_same_engine_server_library', manifest['same_binary_sha256'] == ENGINE
                and manifest['server_sha256'] == SERVER and manifest['adjacent_library_sha256'] == LIBRARY)
    audit.check('only_declared_head_variable', manifest['variables'] == {HEAD_FLAG: {'off': '0', 'on': '1'}})
    audit.check('one_off_on_pair', manifest['sequence'] == ['off', 'on'])
    audit.check('same_normalized_config', meta['off']['config_except_variables_log_key_sha256'] == meta['on']['config_except_variables_log_key_sha256'])
    source_matches = {}
    for name, expected in manifest['harness_sha256'].items():
        path = Path('harness') / name
        source_matches[name] = sha(base / path) == expected
        audit.sources[str(path)] = sha(base / path)
    audit.check('frozen_harness_sources_match_manifest', all(source_matches.values()) and bool(source_matches))
    env = {label: m['config']['env'] for label, m in meta.items()}
    env_changes = {k: {label: env[label].get(k) for label in tags}
                   for k in sorted(set(env['off']) | set(env['on'])) if env['off'].get(k) != env['on'].get(k)}
    audit.check('only_actual_config_head_flag_changes', env_changes == {HEAD_FLAG: {'off': '0', 'on': '1'}})
    head = {}
    for label, m in meta.items():
        prefix = label + ':'
        audit.check(prefix + 'all_four_suites_passed', set(m['suite_returncodes']) == {'budget', 'prefill', 'c8', 'performance'}
                    and all(v == 0 for v in m['suite_returncodes'].values()))
        audit.check(prefix + 'server_cap_true', m['config']['reasoning_budget_engine_cap'] is True)
        audit.check(prefix + 'fixed_configuration', all(env[label].get(k) == v for k, v in FIXED.items()))
        audit.check(prefix + 'metadata_binary_library', m['engine_sha256'] == ENGINE and m['adjacent_library_sha256'] == LIBRARY)
        actual = m['actual_engine']
        audit.check(prefix + 'actual_process_binary_library_flags', len(actual) == 1
                    and actual[0]['engine_sha256'] == ENGINE and actual[0]['library_sha256'] == LIBRARY
                    and actual[0]['expected_library_mapped'] is True
                    and actual[0]['flags'].get(HEAD_FLAG) == ('1' if label == 'on' else '0')
                    and all(actual[0]['flags'].get(k) == v for k, v in FIXED.items()))
        facts = m['startup_facts']
        audit.check(prefix + 'residency_buffers_slots', facts['all_experts_resident']
                    and facts['all_eight_have_3_80_gib_own_buffers'] and facts['batch_slot_devices'] == list(range(8))
                    and facts['piped_mtp_startup_enabled'])
        audit.check(prefix + 'shared_workspace_startup', facts['shared_head_workspaces'] == ([[8, 3, 5, 7]] if label == 'on' else []))
        log = audit.text(tags[label] / 'engine.log')
        raw_rows = [[int(v) for v in match] for match in HEAD_RE.findall(log)]
        suites = m['shared_head_suite_evidence']
        head[label] = {'log_rows': raw_rows, 'log_total_steps': sum(r[3] for r in raw_rows), 'suites': suites}
        audit.check(prefix + 'head_steps_are_real_and_consistent',
                    all(s['steps'] == sum(r[3] for r in s['rows']) for s in suites.values())
                    and sum(s['steps'] for s in suites.values()) == head[label]['log_total_steps']
                    and (not raw_rows if label == 'off' else bool(raw_rows)
                         and all(0 <= r[0] < 8 and r[1:3] == [3, 5] for r in raw_rows)
                         and all(suites[k]['steps'] > 0 for k in ('budget', 'c8', 'performance'))))
        # Solo prefill guard should never activate this concurrent-only route.
        audit.check(prefix + 'solo_prefill_no_shared_head', suites['prefill']['steps'] == 0)

    budget = {label: audit.read(tag / 'budget/budget-correctness.json') for label, tag in tags.items()}
    budget_pairs = pair_rows(budget['off']['records'], budget['on']['records'], cancelled=True)
    audit.check('budget_nine_complete_plus_one_cancel_prefix', len(budget_pairs) == 10
                and sum(not p.get('cancelled', False) for p in budget_pairs) == 9 and all(p['passed'] for p in budget_pairs))
    continuations = {}
    for label, report in budget.items():
        continuations[label] = [{'name': r['name'], 'continuation': r['continuation'],
                                 'bounded': r['bounded_continuation_read']} for r in report['continuations']]
        audit.check(label + ':budget_gates_and_read19', report['all_checks_passed'] and report['config_reasoning_budget_engine_cap']
                    and len(continuations[label]) == 4 and all(r['bounded'] and r['continuation']['read'] == 19 for r in continuations[label]))

    prefill = {label: audit.read(tag / 'prefill/results.json') for label, tag in tags.items()}
    prefill_pairs = pair_rows(prefill['off'], prefill['on'])
    audit.check('prefill_six_full_exact_pairs', len(prefill_pairs) == 6 and all(p['passed'] for p in prefill_pairs))
    prefill_detail = {}
    for label, rows in prefill.items():
        audit.check(label + ':prefill_valid_exact_cache', len(rows) == 6 and all(r['valid']
            and 95000 <= r['usage']['prompt_tokens'] <= 105000
            and r['usage']['completion_tokens'] == 256
            and ((r['engine']['reused_tokens'] == 0 and cached(r) == 0) if r['kind'] == 'cold' else
                 r['engine']['reused_tokens'] == cached(r) and r['engine']['read_tokens'] == 7) for r in rows))
        prefill_detail[label] = {r['name']: {'usage': r['usage'], 'engine': r['engine'], 'ttft_s': r['ttft_s']} for r in rows}
    medians = {}
    for kind in ('cold', 'warm'):
        medians[kind] = {}
        for metric in ('ttft_s', 'read_ms', 'decode_ms', 'decode_tps'):
            values = {label: [r['ttft_s'] if metric == 'ttft_s' else r['engine'][metric]
                             for r in rows if r['kind'] == kind] for label, rows in prefill.items()}
            med = {label: statistics.median(v) for label, v in values.items()}
            medians[kind][metric] = {'values': values, **med, 'change_pct': change(med['off'], med['on'])}

    c8 = {label: audit.read(tag / 'c8/c8-check.json') for label, tag in tags.items()}
    c8_pairs = pair_rows(c8['off']['records'], c8['on']['records'])
    audit.check('c8_eight_full_exact_pairs', len(c8_pairs) == 8 and all(p['passed'] for p in c8_pairs))
    for label, report in c8.items():
        audit.check(label + ':c8_qualified_full_8192', report['all_checks_passed'] and report['expected_piped_mtp'] == 'on'
                    and report['timing']['overlap']['eight_streams_observed']
                    and report['timing']['whole_batch']['completed_api_tokens'] == 8192
                    and all(r['usage']['completion_tokens'] == 1024 and r['finish_reason'] == 'length' for r in report['records']))
    c8_whole = {label: r['timing']['whole_batch'] for label, r in c8.items()}

    c4_summary = {label: audit.read(tag / 'performance/c4/summary.json') for label, tag in tags.items()}
    c4 = {}
    for label, tag in tags.items():
        rows = []
        for path in sorted((base / tag / 'performance/c4').glob('*.json')):
            item = json.loads(path.read_text())
            if isinstance(item, dict) and 'role' in item:
                rows.append(audit.read(path.relative_to(base)))
        c4[label] = rows
    c4_pairs = pair_rows(c4['off'], c4['on'])
    audit.check('c4_six_full_exact_pairs', len(c4_pairs) == 6 and all(p['passed'] for p in c4_pairs))
    chronology, timelines, phases = {}, {}, {}
    for label, s in c4_summary.items():
        audit.check(label + ':c4_qualified_full_8320', s['error'] is None and s['baseline_occupancy_qualified']
                    and s['whole_scenario_completed_api_token_rate']['tokens'] == 8320
                    and s['whole_scenario_completed_api_token_rate']['requests_missing_usage'] == 0
                    and len(c4[label]) == 6 and all(r['successful'] and cached(r) == 0 for r in c4[label]))
        z = s['scenario_start_epoch']
        phases[label] = {k: {**p, 'occupancy': s['occupancy'][k]} for k, p in s['phases'].items()}
        ends = sorted(r['end_epoch'] for r in c4[label] if r['role'] == 'resident')
        chronology[label] = {'startup_to_baseline_s': s['phases']['baseline']['start_epoch'] - z,
                             **{k + '_s': p['duration_s'] for k, p in s['phases'].items()},
                             'drain_s': s['scenario_end_epoch'] - s['phases']['recovery']['end_epoch'],
                             'final_single_resident_interval_s': ends[-1] - ends[-2]}
        timelines[label] = {r['name']: {'role': r['role'], 'usage': r['usage'], 'ttft_s': r['ttft_s'],
            'start_s': r['start_epoch'] - z, 'first_output_s': r['start_epoch'] + r['ttft_s'] - z,
            'last_output_s': r['chunks'][-1]['epoch'] - z, 'end_s': r['end_epoch'] - z,
            'chunk_gaps': r['output_inter_chunk_gaps']} for r in c4[label]}
    c4_whole = {label: s['whole_scenario_completed_api_token_rate'] for label, s in c4_summary.items()}

    # Historical compatibility only: do not pool or compare cross-run timings.
    legacy = {'v4': Path('../budget-v4-c4-sampled'), 'v5': Path('../c8-v5')}
    legacy_meta = {}
    for key, path in legacy.items():
        old_manifest = audit.read(path / 'manifest.json')
        old_meta = audit.read(path / '02-on/metadata.json')
        legacy_meta[key] = {'server_sha256': old_manifest['server_sha256'],
                            'engine_sha256': old_meta['engine_sha256'], 'config_env': old_meta['config']['env']}
        audit.check(key + ':legacy_compatible_source_and_fixed_flags',
            old_manifest['server_sha256'] == SERVER
            and old_meta['engine_sha256'] == '113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f'
            and old_meta['config']['reasoning_budget_engine_cap'] is True
            and all(old_meta['config']['env'].get(k) == v for k, v in FIXED.items()))
    old_budget = audit.read(legacy['v4'] / '02-on/budget/budget-correctness.json')['records']
    old_prefill = audit.read(legacy['v4'] / '02-on/prefill/results.json')
    old_c8 = audit.read(legacy['v5'] / '02-on/c8/c8-check.json')['records']
    old_c4 = []
    for path in sorted((base / legacy['v4'] / '02-on/performance/c4').glob('*.json')):
        item = json.loads(path.read_text())
        if isinstance(item, dict) and 'role' in item:
            old_c4.append(audit.read(path.relative_to(base)))
    compatibility_pairs = {'budget': pair_rows(old_budget, budget['off']['records'], cancelled=True),
                           'prefill': pair_rows(old_prefill, prefill['off']),
                           'c8': pair_rows(old_c8, c8['off']['records']), 'c4': pair_rows(old_c4, c4['off'])}
    for suite, pairs in compatibility_pairs.items():
        audit.check('new_head_off_legacy_' + suite + '_outputs_usage_compatible',
                    len(pairs) == {'budget': 10, 'prefill': 6, 'c8': 8, 'c4': 6}[suite] and all(p['passed'] for p in pairs))

    return {'scope': 'One head-only off/on pilot pair; batch/piped MTP on and server budget cap true in both.',
            'checks': audit.checks, 'all_checks_passed': all(audit.checks.values()),
            'engine_sha256': ENGINE, 'server_sha256': SERVER, 'library_sha256': LIBRARY,
            'env_changes': env_changes, 'frozen_sources_match': source_matches,
            'startup': {k: m['startup_facts'] for k, m in meta.items()}, 'head_actual_steps': head,
            'budget_pairs': budget_pairs, 'budget_continuations': continuations,
            'prefill_pairs': prefill_pairs, 'prefill_detail': prefill_detail, 'prefill_medians': medians,
            'c8_pairs': c8_pairs, 'c8_timing': {k: r['timing'] for k, r in c8.items()},
            'c8_whole_throughput_change_pct': change(c8_whole['off']['completed_api_tokens_per_s'], c8_whole['on']['completed_api_tokens_per_s']),
            'c4_pairs': c4_pairs, 'c4_whole': c4_whole,
            'c4_whole_throughput_change_pct': change(c4_whole['off']['tokens_per_s'], c4_whole['on']['tokens_per_s']),
            'c4_phases': phases, 'c4_chronology': chronology, 'c4_timelines': timelines,
            'c4_recovery': {k: {a: b for a, b in s['recovery'].items() if a != 'bins'} for k, s in c4_summary.items()},
            'legacy_output_compatibility': {'pairs': compatibility_pairs, 'source_identity': legacy_meta,
                'pair_label_note': 'In these legacy-only pairs, off means old v4/v5 MTP-on engine; on means new engine HEAD=0. No historical timing is compared.',
                'scope': 'Only matching synthetic requests, full output/reasoning/channel hashes, complete usage and cancelled shared prefixes; excludes performance.'},
            'source_sha256': audit.sources,
            'limits': ['One sequential pair; no confidence interval or stable throughput claim.',
                       'C8 uses short prompts; C4 resident contexts and injector lengths are reported from actual usage. Neither establishes four-way 100k performance.',
                       'Whole API TPS includes admission/prefill and final drain; chunk gaps are not per-token TBT.',
                       'Per-phase byte throughput/fairness is descriptive because each phase may contain different content positions.',
                       'Prefill medians have only three cold/three warm requests. Warm read time includes restore.',
                       'Cancelled streams compare their shared prefixes; full cancelled text hashes need not match.']}


def markdown(result):
    ok = result['all_checks_passed']
    lines = ['# 共享 piped MTP head 单轮对照', '',
             '本报告仅使用 head-pilot 的一次 off/on 对照；两臂 batch/piped MTP 始终开启、预算 cap=true，仅 `STRATA_PIPED_MTP_HEAD` 从 0 改为 1。', '',
             f"完整性与身份检查：{'全部通过' if ok else '存在失败，见 summary.json checks；不能把差异当作等价工作量收益'}。", '',
             f"引擎 `{ENGINE}`；server `{SERVER}`。实际进程 exe/映射库、冻结 harness、8 槽、全专家驻留及每卡 3.80 GiB prefill buffer 分别检查。", '',
             '| 检查 | 成对数量 | 全部通过 |', '|---|---:|---|']
    for label, key in [('预算完整请求及取消共同前缀', 'budget_pairs'), ('100k 冷/暖 prefill', 'prefill_pairs'),
                       ('C8 完整请求', 'c8_pairs'), ('C4 resident+injector', 'c4_pairs')]:
        rows = result[key]
        lines.append(f"| {label} | {len(rows)} | {all(r['passed'] for r in rows)} |")
    lines += ['', '预算项含 9 条完整请求和 1 条取消流；取消只要求共同前缀相同，不要求完整 hash 相同。prefill 另外检查实际输入长度、cold cache=0、warm 精确 cache/read；性能请求检查完整输出、分通道 hash、全部 usage。', '',
              '| 指标 | off | on | on 相对变化 |', '|---|---:|---:|---:|']
    for label, key, metric in [('C8 整批 API tokens/s', 'c8_timing', 'completed_api_tokens_per_s'),
                               ('C4 整场 API tokens/s', 'c4_whole', 'tokens_per_s')]:
        values = {arm: result[key][arm]['whole_batch'][metric] if key == 'c8_timing' else result[key][arm][metric] for arm in ('off', 'on')}
        lines.append(f"| {label} | {values['off']:.3f} | {values['on']:.3f} | {change(values['off'], values['on']):+.2f}% |")
    for kind in ('cold', 'warm'):
        for metric in ('read_ms', 'ttft_s'):
            r = result['prefill_medians'][kind][metric]
            lines.append(f"| 100k {kind} {metric} 中位数 | {r['off']:.3f} | {r['on']:.3f} | {r['change_pct']:+.2f}% |")
    cold = result['prefill_medians']['cold']['read_ms']['change_pct']
    lines += ['', f"冷 prefill read 中位数本轮变化 {cold:+.2f}%；仅三次观测，不能据此保证 prefill 无回退，也不能断言微小变化来自 head 开关。",
              '吞吐包含 admission/prefill、输出和尾部 drain，不是纯 decode kernel 速度。延迟负变化更快；吞吐正变化更快。', '',
              '| C4 阶段 | off/on bytes/s | off/on p99 gap (s) | off/on max gap (s) | off/on Jain |', '|---|---:|---:|---:|---:|']
    for phase in ('baseline', 'mixed', 'recovery'):
        a, b = [result['c4_phases'][arm][phase] for arm in ('off', 'on')]
        ga, gb = [r['output_inter_chunk_gaps_ending_in_window'] for r in (a, b)]
        lines.append(f"| {phase} | {a['resident_output_bytes_per_s']:.1f}/{b['resident_output_bytes_per_s']:.1f} | "
                     f"{ga['p99_s']:.4f}/{gb['p99_s']:.4f} | {ga['max_s']:.4f}/{gb['max_s']:.4f} | "
                     f"{a['jain_fairness_of_output_byte_rates']:.4f}/{b['jain_fairness_of_output_byte_rates']:.4f} |")
    lines += ['', 'SSE 出流间隔包含 reasoning/content/tool；没有逐 token 时间，不能称 TBT。阶段 bytes/s 会受文本位置影响，完整 usage/hash 才证明整场工作量一致。', '',
              '| 实际共享 head steps | off | on |', '|---|---:|---:|']
    for suite in ('budget', 'prefill', 'c8', 'performance'):
        a, b = [result['head_actual_steps'][arm]['suites'][suite]['steps'] for arm in ('off', 'on')]
        lines.append(f'| {suite} | {a} | {b} |')
    compatibility = result['legacy_output_compatibility']['pairs']
    lines += ['', '新版 head=0 与旧 v4/v5 MTP-on 的兼容核对：', '',
              '| 旧数据对应项 | 数量 | 输出/usage 或取消共同前缀通过 |', '|---|---:|---|']
    for suite, pairs in compatibility.items():
        lines.append(f"| {suite} | {len(pairs)} | {all(p['passed'] for p in pairs)} |")
    lines += ['', '这些跨轮比较仅证明固定 synthetic 请求的输出与计数兼容；suite 顺序不同，不比较历史性能。']
    lines += ['', '适用范围：', ''] + ['- ' + text for text in result['limits']]
    lines += ['', '复现：`python3 analyze.py --input ../head-pilot --out .`。所有输入 SHA256、逐请求比较、阶段占用/停顿、缓存明细与实际 steps 保存在 `summary.json`。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent.parent / 'head-pilot')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = analyze(args.input.resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    result['analysis_source_sha256'] = sha(Path(__file__))
    (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.out / 'README.md').write_text(markdown(result))
    print(json.dumps({'all_checks_passed': result['all_checks_passed'],
                      'failed_checks': [k for k, v in result['checks'].items() if not v],
                      'c8_whole_throughput_change_pct': result['c8_whole_throughput_change_pct'],
                      'c4_whole_throughput_change_pct': result['c4_whole_throughput_change_pct']}, ensure_ascii=False))
    return 0 if result['all_checks_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
