#!/usr/bin/env python3
"""Summarize recorded live measurements; GPU CSV timestamps are host UTC."""
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for filename in ('live-results.json', 'timing-results.json'):
    path = root / filename
    if not path.exists():
        continue
    for r in json.loads(path.read_text()):
        n = r['usage']['prompt_tokens']
        pat = rf'prompt {n} tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\)'
        matches = [re.search(pat, line) for line in r['engine_lines']]
        matches = [m for m in matches if m]
        assert len(matches) == 1, (r['run'], len(matches))
        m = matches[0]
        row = dict(run=r['run'], kind=r['kind'], source=filename, prompt_tokens=n,
                   cached_tokens=int(m[1]), read_tokens=int(m[2]), prefill_s=int(m[3])/1000,
                   prefill_tps=float(m[4]), ttft_s=r['ttft_s'], decode_tps=r['client_decode_tps'],
                   output_tokens=r['usage']['completion_tokens'], max_in_flight=r['max_in_flight'],
                   in_flight_before=r['activity_before']['in_flight'],
                   new_other_requests=r['activity_after']['requests']-r['activity_before']['requests']-1,
                   uncontended=r['no_observed_competitor'])
        samples = [x['in_flight'] for x in r['activity_samples'] if 'in_flight' in x]
        row['in_flight_median'] = statistics.median(samples)
        ev = r.get('chunk_events', [])
        if ev:
            gaps = [{'start_s': a['elapsed_s'], 'end_s': b['elapsed_s'],
                     'gap_s': b['elapsed_s']-a['elapsed_s']}
                    for a,b in zip(ev, ev[1:])]
            row['stream_gaps_over_500ms'] = [g for g in gaps if g['gap_s'] > 0.5]
            row['max_stream_gap_s'] = max(g['gap_s'] for g in gaps)
            row['stream_gaps_over_500ms_sum_s'] = sum(g['gap_s'] for g in row['stream_gaps_over_500ms'])
            # Logs are observed at 0.5 s intervals, so these are approximate completion timestamps.
            row['prompt_timeline'] = [dict(elapsed_s=e['time']-r['start_epoch'], line=e['line'])
                                      for e in r.get('engine_timeline', []) if 'read in' in e['line']]
        gpu_path = root / ('gpu.csv' if filename == 'live-results.json' else 'timing-gpu.csv')
        if gpu_path.exists():
            metrics = {}
            for g in csv.DictReader(gpu_path.open()):
                g = {k.strip():v.strip() for k,v in g.items()}
                t = datetime.strptime(g['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                if not r['start_epoch'] + r['ttft_s'] <= t <= r['start_epoch'] + r['last_content_s']:
                    continue
                d = metrics.setdefault(g['index'], {'gpu_busy_pct':[], 'memory_busy_pct':[], 'sm_mhz':[], 'power_w':[]})
                for dst, src in [('gpu_busy_pct','utilization.gpu [%]'),('memory_busy_pct','utilization.memory [%]'),
                                 ('sm_mhz','clocks.current.sm [MHz]'),('power_w','power.draw [W]')]:
                    d[dst].append(float(g[src].split()[0]))
            row['gpu_during_response'] = {i:{k:round(statistics.mean(v),2) for k,v in d.items()} for i,d in metrics.items()}
        rows.append(row)
(root / 'summary.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False))
for r in rows:
    print(f"{r['run']}: {r['prompt_tokens']} tokens, prefill {r['prefill_s']:.3f}s / {r['prefill_tps']:.1f} tok/s; "
          f"TTFT {r['ttft_s']:.2f}s, decode {r['decode_tps']:.2f} tok/s; max inflight {r['max_in_flight']}")
