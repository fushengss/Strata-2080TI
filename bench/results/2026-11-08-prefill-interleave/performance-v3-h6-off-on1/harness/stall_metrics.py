"""Offline SSE silence metrics. No inferred token boundaries or GPU percentages."""
import math


def dist(values):
    values = sorted(values)
    def q(p):
        return values[max(0, math.ceil(len(values) * p) - 1)] if values else None
    return {'n': len(values), 'p50_s': q(.5), 'p95_s': q(.95), 'p99_s': q(.99),
            'max_s': max(values) if values else None, 'method': 'nearest rank, descriptive only'}


def union_duration(intervals, required=1):
    events = []
    for left, right in intervals:
        if right > left:
            events.extend(((left, 1), (right, -1)))
    count, previous, total = 0, None, 0.0
    for when, delta in sorted(events):
        if previous is not None and count >= required:
            total += when - previous
        count += delta
        previous = when
    return total


def phase(records, begin, end, lanes=4, channel='output'):
    assert end >= begin
    per_lane, all_stalled = [], {1: [], 5: []}
    for lane in range(lanes):
        selected = [r for r in records if r['role'] == 'resident' and r['lane'] == lane]
        # Finite cohort deliberately has exactly one request per lane.
        assert len(selected) <= 1
        epochs, active_start, active_end, events, raw_gaps, clipped = [], None, None, [], [], []
        intervals = {1: [], 5: []}
        if selected:
            record = selected[0]
            events = [c for c in record['chunks'] if c['bytes'].get(channel, 0) > 0]
            epochs = [c['epoch'] for c in events]
            if epochs:
                active_start, active_end = epochs[0], record['end_epoch']
                anchors = epochs + [active_end]
                raw_gaps = [b-a for a,b in zip(epochs,epochs[1:]) if begin <= b < end]
                for left, right in zip(anchors, anchors[1:]):
                    x, y = max(left, begin), min(right, end)
                    if y > x:
                        clipped.append(y-x)
                    for threshold in intervals:
                        x, y = max(left+threshold, begin), min(right, end)
                        if y > x:
                            intervals[threshold].append((x,y))
                            all_stalled[threshold].append((x,y))
        byte_count = sum(c['bytes'].get(channel,0) for c in events if begin <= c['epoch'] < end)
        covered = max(0, min(end, active_end)-max(begin, active_start)) if active_start is not None else 0
        per_lane.append({'lane':lane, 'active_output_coverage_s':covered,
                         'covers_entire_phase': active_start is not None and active_start <= begin and active_end >= end,
                         'output_bytes':byte_count, 'bytes_per_s':byte_count/(end-begin) if end>begin else None,
                         'raw_gaps_ending_in_phase':dist(raw_gaps), 'clipped_silence_intervals':dist(clipped),
                         'stalled_over_1s_duration_s':union_duration(intervals[1]),
                         'stalled_over_5s_duration_s':union_duration(intervals[5])})
    rates = [x['bytes_per_s'] or 0 for x in per_lane]
    squares = sum(x*x for x in rates)
    return {'begin_epoch':begin, 'end_epoch':end, 'duration_s':end-begin, 'channel':channel,
            'all_lanes_cover_phase': all(row['covers_entire_phase'] for row in per_lane), 'lanes':per_lane,
            'aggregate_output_bytes_per_s':sum(rates),
            'jain_byte_progress_fairness':sum(rates)**2/(lanes*squares) if squares else None,
            'max_lane_clipped_silence_s': max((r['clipped_silence_intervals']['max_s'] or 0 for r in per_lane),default=0),
            'any_lane_stalled_over_1s_duration_s':union_duration(all_stalled[1]),
            'all_lanes_stalled_over_1s_duration_s':union_duration(all_stalled[1],lanes),
            'any_lane_stalled_over_5s_duration_s':union_duration(all_stalled[5]),
            'all_lanes_stalled_over_5s_duration_s':union_duration(all_stalled[5],lanes),
            'scope':'SSE arrival/chunk silence; thresholds start after the last observed output plus1/5s. Not per-token TBT or GPU-idle time.'}


def self_test():
    def record(lane, times):
        return {'role':'resident','lane':lane,'end_epoch':25,
                'chunks':[{'epoch':x,'bytes':{'output':1,'visible_content':1,'reasoning':0,'tool':0}} for x in times]}
    records=[record(i,[0,1,11,12,24]) for i in range(4)]
    x=phase(records,2,10)
    assert x['all_lanes_cover_phase'] and x['max_lane_clipped_silence_s']==8
    assert x['all_lanes_stalled_over_1s_duration_s']==8
    assert x['all_lanes_stalled_over_5s_duration_s']==4
    assert phase(records,10,12)['all_lanes_stalled_over_1s_duration_s']==1
    assert union_duration([(0,2),(1,3)])==3
    assert union_duration([(0,2),(1,3)],2)==1
    assert dist([.1,.2,10])['p99_s']==10
    assert not phase(records,0,26)['all_lanes_cover_phase']
    assert not phase(records,2,10,channel='reasoning')['all_lanes_cover_phase']
    print('PASS: phase-censored gaps, threshold duration and lane overlap; no network')


if __name__=='__main__':
    self_test()
