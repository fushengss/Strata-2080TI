#!/usr/bin/env python3
"""Offline production compatibility baseline; no source/remote/GPU mutation."""
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import statistics
import analyze_gpu

N = Path(__file__).resolve().parent
ARM = N / 'production-compatibility-off1' / '01-off'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(xs):
    return {'n': len(xs), 'median': statistics.median(xs),
            'mean': statistics.fmean(xs), 'min': min(xs), 'max': max(xs)}


def main():
    inputs = {'manifest': ARM.parent / 'manifest.json', 'metadata': ARM / 'metadata.json',
              'c8': ARM / 'c8/c8-check.json', 'prefill': ARM / 'prefill/results.json',
              'gpu': ARM / 'gpu.csv'}
    manifest, meta, c8, prefill = [json.loads(inputs[k].read_text()) for k in ('manifest', 'metadata', 'c8', 'prefill')]
    assert meta.get('finished_epoch') and meta['suite_returncodes'] == {'c8': 0, 'prefill': 0}
    assert manifest['original_config_unchanged'] and not manifest['remaining_gpu_pids']
    assert c8['all_checks_passed'] and len(prefill) == 6 and all(r['valid'] for r in prefill)
    overlap = c8['timing']['overlap']
    begin, end = overlap['common_start_epoch'], overlap['common_end_epoch']
    assert overlap['eight_streams_observed'] and end > begin
    rows = analyze_gpu.samples(inputs['gpu'])
    gpu = analyze_gpu.summarize(rows, begin, end)
    assert gpu['all_eight_sampled']
    for card in gpu['cards']:
        own = [r for r in rows if r['index'] == card['gpu'] and begin <= r['epoch'] < end]
        times = sorted(r['epoch'] for r in own)
        card['sampling_gap_s'] = measure([b-a for a, b in zip(times, times[1:])])
        card['event_active_samples'] = {}
        for key in ('sw_power_cap', 'sw_thermal_slowdown', 'hw_thermal_slowdown', 'hw_power_brake_slowdown'):
            xs = [r.get('clocks_event_reasons.' + key) for r in own]
            valid = [v for v in xs if v is not None]
            active = sum(v == 1 for v in valid)
            card['event_active_samples'][key] = {'active': active, 'valid': len(valid),
                'fraction': active / len(valid) if valid else None}
    assert all(c['fields']['enforced.power.limit']['min'] == c['fields']['enforced.power.limit']['max'] == 250 for c in gpu['cards'])
    assert all(c['fields']['clocks.current.sm']['max'] < 1900 for c in gpu['cards'])
    assert all(c['event_active_samples'][k]['active'] == 0 for c in gpu['cards'] for k in ('sw_thermal_slowdown', 'hw_thermal_slowdown', 'hw_power_brake_slowdown'))
    trimmed = analyze_gpu.summarize(rows, begin + 1, end - 1)
    prefill_rows = [{k: r[k] for k in ('name', 'kind', 'valid', 'ttft_s', 'engine', 'usage')} for r in prefill]
    prefill_summary = {}
    for kind in ('cold', 'warm'):
        own = [r for r in prefill_rows if r['kind'] == kind]
        assert len(own) == 3
        prefill_summary[kind] = {'ttft_s': measure([r['ttft_s'] for r in own]),
            **{key: measure([r['engine'][key] for r in own]) for key in ('read_ms', 'decode_ms', 'decode_tps')},
            'prompt_tokens': sorted({r['engine']['prompt_tokens'] for r in own}),
            'reused_tokens': sorted({r['engine']['reused_tokens'] for r in own}),
            'read_tokens': sorted({r['engine']['read_tokens'] for r in own}),
            'generated_tokens': sorted({r['engine']['generated_tokens'] for r in own})}
    report = {
        'format': 1, 'scope': 'Current production binary baseline characterization and cross-version compatibility reference only; not an overlap on/off performance comparison.',
        'source_sha256': {key: {'path': str(path), 'sha256': sha(path)} for key, path in inputs.items()},
        'identity': {key: manifest[key] for key in ('same_binary_sha256', 'adjacent_library_sha256', 'base_config_sha256', 'server_sha256', 'fixed_env')},
        'qualification': {'c8': c8['all_checks_passed'], 'prefill': all(r['valid'] for r in prefill),
                          'suite_returncodes': meta['suite_returncodes'], 'original_config_unchanged': True},
        'c8': {'common_sse_interval': overlap,
               'common_start_utc': datetime.fromtimestamp(begin, timezone.utc).isoformat(),
               'common_end_utc': datetime.fromtimestamp(end, timezone.utc).isoformat(),
               'common_start_taipei': datetime.fromtimestamp(begin, timezone(timedelta(hours=8))).isoformat(),
               'common_end_taipei': datetime.fromtimestamp(end, timezone(timedelta(hours=8))).isoformat(),
               'prompt_tokens': sorted({r['usage']['prompt_tokens'] for r in c8['records']}),
               'whole_batch': c8['timing']['whole_batch'], 'gpu_common_interval': gpu,
               'gpu_inner_interval_excluding_one_second_each_boundary': trimmed},
        'prefill': {'rows': prefill_rows, 'summary': prefill_summary},
        'limits': [
            'GPU CSV timestamps are UTC, verified on the remote host by root. The SSE intersection is a host-arrival window, not an exact GPU-kernel interval.',
            '500 ms requested sampling includes CLI/query overhead; actual per-card gaps are reported. Means and event proportions are unweighted sample statistics, not exact time fractions.',
            'GPU and memory utilization report sampled activity, not achieved FLOPS, memory bandwidth or full hardware saturation. Rolling sensor windows and sequential per-card reads affect boundaries.',
            'Power-limit event samples and power.draw use different sensor windows; a cap event with a sub-limit displayed power does not establish a counter error.',
            'The inner-window sensitivity check removes one second at each boundary; it is not a substitute for a CUDA timeline.',
            'C8 has short 66/68-token prompts and 1024 output tokens each. Whole-batch API tokens/s includes admission, TTFT and drain; it is not 100k-context or decode-only throughput.',
            'Prefill uses actual 101139-token prompts, three cold and three warm requests. Warm read_ms includes cache restoration with only seven fresh prompt tokens; do not call seven divided by restore time bulk prefill speed.',
            'The prefill decode segment has 256 output tokens. Three repetitions do not establish a stable tail latency distribution.',
            'No old/new engine speedup, overlap benefit, bottleneck causality or universal GPU saturation is inferred from this standalone baseline.'
        ]}
    out = N / 'production-baseline-analysis.json'
    assert not out.exists(), 'Refuse to overwrite analysis'
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['生产基线本地数值分析（不作为新并发模式 A/B）',
        '8b27e040 / config f752a214 / server 40c5e9df / library fb67bffc；P1=1、async=0、overlap=0。C8 与 3 cold + 3 warm prefill 均通过。',
        f"八流共同 SSE 输出区间：{report['c8']['common_start_taipei']} 至 {report['c8']['common_end_taipei']}，共 {end-begin:.3f}s。每卡29–30个采样；不将GPU活动率当FLOP饱和度。", '',
        'GPU | 活动均值% | SM均值MHz (min–max) | 功耗均值W | SW功耗限制active/样本']
    for card in gpu['cards']:
        f = card['fields']; cap = card['event_active_samples']['sw_power_cap']
        lines.append(f"{card['gpu']} | {f['utilization.gpu']['mean']:.2f} | {f['clocks.current.sm']['mean']:.1f} ({f['clocks.current.sm']['min']:.0f}–{f['clocks.current.sm']['max']:.0f}) | {f['power.draw']['mean']:.2f} | {cap['active']}/{cap['valid']} ({cap['fraction']*100:.1f}%)")
    lines += ['', f"各卡活动均值的算术平均 {gpu['mean_activity_across_cards_pct']:.2f}%；去除两端各1s后为 {trimmed['mean_activity_across_cards_pct']:.2f}%。",
        '所有卡 enforced power limit 为250W；所有 SW/HW thermal slowdown 与 HW power brake 样本均未激活。各卡时钟样本均低于1900MHz，不能把2000MHz设置当作实际运行频率。',
        '2/4/5卡全部采样、7卡29/30、0卡19/29出现SW功耗限制；同时1/6卡活动较低。说明活动分布和功耗约束不均，单靠nvidia-smi不能判定具体等待源或可提升比例。',
        f"C8 8192 API输出token / {c8['timing']['whole_batch']['elapsed_s']:.3f}s = {c8['timing']['whole_batch']['completed_api_tokens_per_s']:.2f} tok/s，包含入场与TTFT；prompt仅66/68token，不是100k decode。", '',
        'Prefill每条实际101139 prompt tokens，输出256tokens：']
    for kind in ('cold', 'warm'):
        r = prefill_summary[kind]
        lines.append(f"{kind}: read_ms median {r['read_ms']['median']:.0f} (range {r['read_ms']['min']:.0f}–{r['read_ms']['max']:.0f}); TTFT median {r['ttft_s']['median']:.3f}s; decode median {r['decode_tps']['median']:.1f}tok/s; reused {r['reused_tokens']}, fresh {r['read_tokens']}.")
    lines += ['Warm时长主要包含缓存恢复；不可将7个fresh token的商当作bulk prefill能力。三次重复只描述本轮波动，不能推定尾延迟。',
        '仅本地提取数值和哈希，不输出生成正文；未执行远程/GPU/服务操作。']
    (N / 'production-baseline-analysis.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
