#!/usr/bin/env python3
"""Offline numeric PARK_TRACE parser. No API/network/GPU calls; no raw text retained."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import unittest

F = r"(?:\d+(?:\.\d*)?|\.\d+)"
STAGE = re.compile(
    r"strata parktrace: joins (?P<joins_ms>"+F+r") ms, sync (?P<sync_ms>"+F+r") ms, "
    r"(?P<checkpoint_count>\d+) checkpoints copied (?P<checkpoints_ms>"+F+r") ms, live state (?P<live_ms>"+F+r") ms, "
    r"begin (?P<begin_ms>"+F+r") ms, K/V (?P<kv_mib>\d+) MB (?P<kv_wall_ms>"+F+r") ms "
    r"\(layer max (?P<kv_layer_max_ms>"+F+r"), sum (?P<kv_layer_sum_ms>"+F+r"); "
    r"attrs (?P<kv_attrs_ms>"+F+r"), resize (?P<kv_resize_ms>"+F+r"), copy (?P<kv_copy_ms>"+F+r")"
    r"(?:; overwrite_added_bytes (?P<overwrite_added_bytes>\d+))?\)")
PARK = re.compile(
    r"strata serve: conversation cache: (?P<status>parked|skipped) (?P<tokens>\d+) tokens"
    r"(?P<source>.*?) in (?P<wall_ms>"+F+r") ms; parked=(?P<cache_count>\d+) bytes=(?P<cache_bytes>\d+) "
    r"evictions=(?P<cache_evictions>\d+) snapshot_bytes=(?P<snapshot_bytes>\d+) reused_kv_bytes=(?P<reused_kv_bytes>\d+)")
BEGIN = re.compile(r"strata piped concurrent MTP: begin (?P<fields>.*)$")
END = re.compile(r"strata piped concurrent MTP: end (?P<fields>.*)$")
FIELD = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)")
BT = re.compile(r"(?:^|\s)BT (?P<slot>\d+) -?\d+\s*$")
PROMPT = re.compile(r"strata serve: prompt (?P<tokens>\d+) tokens = (?P<reused>\d+) reused \+ "
                    r"(?P<read>\d+)(?: of (?P<attempted>\d+))? read in (?P<prompt_ms>"+F+r") ms")
EXCLUSIVE = ('joins_ms','sync_ms','checkpoints_ms','live_ms','begin_ms','kv_wall_ms')
NESTED = ('kv_attrs_ms','kv_resize_ms','kv_copy_ms')
INTEGER = {'checkpoint_count','kv_mib','overwrite_added_bytes'}
BEGIN_FIELDS = {'read_id','from','tokens','chunk','active_slots','stages','own_buffers','resident_stages'}
END_FIELDS = {'read_id','read_ms','decode_rows_before_done','read_ok','pump_ok','stopped_at','admit_stop_requested','yield_slot'}
NOTES = [
    'All durations are milliseconds unless explicitly suffixed _us; snapshot/reuse/overwrite fields are bytes.',
    'The source prints K/V MB using bytes >> 20: kv_mib is rounded down MiB, not exact payload bytes.',
    'Stage IDs are inferred sequential ordinals, not explicit GPU IDs. Exactly expected_stages uninterrupted full snapshot-stat lines are required.',
    'Checkpoint save join/vectors/small-copy sublines are ignored for grouping; they are already nested in live_ms.',
    'Exclusive stage terms sum joins+sync+checkpoints+live+begin+KV wall. KV attrs/resize/copy are NESTED and must not be added again.',
    'The labels are host intervals, not isolated GPU kernels. In particular sync_ms also includes size validation and some metadata work.',
    'The trace prints before owner draft KV capture. Whole park additionally includes draft work, outer per-stage sync, and other host/publication work.',
    'Residual is whole wall minus reported exclusive stage terms; it is not automatically draft time or synchronization time.',
    'Direct main/slot temporary KV saves, including MTP slot copies, have no full snapshot-stat/wholepark pair; this parser cannot time their gaps.',
    'Snapshot bytes include host-authoritative KV/checkpoints/device state. No whole-D2H bandwidth is calculated.',
    'Reader decode_rows_before_done counts native BT outputs (including special/EOS), not raw verification rows or API visible-content tokens.',
    'Following-read associations use log order only. BT lines may be absent because an engine stderr log does not contain native stdout.',
    'Shape comparison (tokens, snapshot bytes, source and stage layout) is not proof of identical prompt/cache contents or workload identity.'
]

def pct(n, d):
    return 100.0*n/d if d > 0 else None

def numeric_fields(text, allowed):
    return {k: float(v) if '.' in v else int(v) for k,v in FIELD.findall(text) if k in allowed}

def parse_text(text: str, name='<synthetic>', expected_stages=8):
    result = {'input':name,'expected_stages':expected_stages,'parks':[],'reads':[],
              'native_prompts':[],'orphans':[],'warnings':[],'ignored_checkpoint_trace_lines':0,
              'notes':NOTES}
    pending=[]
    session=0
    active_reads={}
    nearest_park=None
    def flush(reason, line):
        nonlocal pending
        if pending:
            result['orphans'].append({'reason':reason,'boundary_line':line,
                'stage_lines':[s['line'] for s in pending],'stage_count':len(pending)})
            pending=[]
    def warning(code,line):
        result['warnings'].append({'code':code,'line':line})
    for lineno,line in enumerate(text.splitlines(),1):
        if 'engine started:' in line:
            flush('engine_restart',lineno)
            for rd in active_reads.values(): rd['complete']=False
            active_reads={}; nearest_park=None; session+=1
        sm=STAGE.search(line)
        if sm:
            s={k:(None if v is None else int(v) if k in INTEGER else float(v)) for k,v in sm.groupdict().items()}
            s.update(line=lineno, session=session)
            pending.append(s)
            continue
        if 'strata parktrace: joins ' in line:
            flush('malformed_full_snapshot_stat',lineno)
            warning('malformed_full_snapshot_stat',lineno)
            continue
        if 'strata parktrace: save ' in line or 'strata parktrace: async ckpt' in line:
            result['ignored_checkpoint_trace_lines']+=1
            continue
        pm=PARK.search(line)
        if pm:
            g=pm.groupdict(); suffix=g.pop('source').strip()
            slot=re.fullmatch(r'from slot (\d+)',suffix)
            source='main' if not suffix else 'slot' if slot else 'unknown'
            p={k:v if k=='status' else float(v) if k=='wall_ms' else int(v) for k,v in g.items()}
            p.update(park_id=len(result['parks'])+1,line=lineno,session=session,
                     source_kind=source,source_slot=int(slot.group(1)) if slot else None,
                     stages=pending,stage_count=len(pending),
                     grouping='complete' if len(pending)==expected_stages else 'ambiguous',
                     active_read_ids=sorted(active_reads),following_read_index=None)
            if source=='unknown': warning('unknown_park_source_suffix',lineno)
            if len(pending)!=expected_stages: warning('unexpected_snapshot_stage_count',lineno)
            for i,s in enumerate(pending): s['inferred_stage_ordinal']=i
            p['component_sums_ms']={k:sum(s[k] for s in pending) for k in EXCLUSIVE+NESTED}
            p['checkpoint_count_per_stage']=[s['checkpoint_count'] for s in pending]
            p['trace_kv_mib_sum']=sum(s['kv_mib'] for s in pending)
            present=sum(s['overwrite_added_bytes'] is not None for s in pending)
            p['overwrite_field_stages']=present
            p['overwrite_added_bytes_sum']=sum(s['overwrite_added_bytes'] for s in pending) if pending and present==len(pending) else None
            if p['grouping']=='complete':
                exclusive=sum(p['component_sums_ms'][k] for k in EXCLUSIVE)
                p['exclusive_trace_sum_ms']=exclusive
                p['unattributed_residual_ms']=p['wall_ms']-exclusive
                p['whole_percentages']={k:pct(v,p['wall_ms']) for k,v in p['component_sums_ms'].items()}
                p['exclusive_trace_percent']=pct(exclusive,p['wall_ms'])
                # Each 1-decimal exclusive term and the whole wall are rounded independently.
                p['rounding_tolerance_ms']=0.05*(len(EXCLUSIVE)*expected_stages+1)
                if p['unattributed_residual_ms'] < -p['rounding_tolerance_ms']:
                    warning('exclusive_sum_exceeds_whole_beyond_rounding',lineno)
            else:
                p['exclusive_trace_sum_ms']=p['unattributed_residual_ms']=None
                p['whole_percentages']=None
            result['parks'].append(p); nearest_park=p; pending=[]
            continue
        if 'strata serve: conversation cache:' in line and (
            'skip parking' in line or 'allocation failed; skip parking' in line or 'not parked before' in line):
            flush('park_abandoned_or_skipped_before_capture',lineno)
            nearest_park=None
        bm=BEGIN.search(line)
        if bm:
            flush('reader_begin_before_wholepark',lineno)
            f=numeric_fields(bm.group('fields'),BEGIN_FIELDS)
            if 'read_id' not in f: warning('reader_begin_missing_id',lineno); continue
            rid=f['read_id']
            if rid in active_reads: warning('duplicate_open_read_id',lineno)
            rd={'session':session,'begin_line':lineno,'begin':f,'end':None,'complete':False,
                'observed_bt_lines_during_read':0,'observed_bt_by_slot':{},'index':len(result['reads'])}
            result['reads'].append(rd); active_reads[rid]=rd
            if nearest_park is not None and nearest_park['following_read_index'] is None:
                nearest_park['following_read_index']=rd['index']
            continue
        em=END.search(line)
        if em:
            flush('reader_end_before_wholepark',lineno)
            f=numeric_fields(em.group('fields'),END_FIELDS); rid=f.get('read_id')
            rd=active_reads.pop(rid,None)
            if rd is None: warning('reader_end_without_begin',lineno)
            else:
                rd.update(end=f,end_line=lineno,complete=True)
                rd['native_bt_outputs_before_done']=f.get('decode_rows_before_done')
                rd['read_succeeded']=f.get('read_ok')==1 and f.get('pump_ok')==1
            continue
        bt=BT.search(line)
        if bt:
            for rd in active_reads.values():
                rd['observed_bt_lines_during_read']+=1
                slot=bt.group('slot')
                rd['observed_bt_by_slot'][slot]=rd['observed_bt_by_slot'].get(slot,0)+1
            continue
        nm=PROMPT.search(line)
        if nm:
            flush('native_prompt_finished_before_wholepark',lineno)
            f={k:None if v is None else float(v) if k=='prompt_ms' else int(v) for k,v in nm.groupdict().items()}
            f.update(line=lineno,session=session)
            result['native_prompts'].append(f)
            nearest_park=None
        if 'strata trace: r0' in line:
            flush('new_request_before_wholepark',lineno)
            nearest_park=None
    flush('end_of_file',len(text.splitlines())+1)
    result['summary']={'park_count':len(result['parks']),
        'complete_stage_groups':sum(p['grouping']=='complete' for p in result['parks']),
        'ambiguous_stage_groups':sum(p['grouping']!='complete' for p in result['parks']),
        'reader_count':len(result['reads']),'complete_readers':sum(r['complete'] for r in result['reads']),
        'orphan_group_count':len(result['orphans']),'warning_count':len(result['warnings'])}
    return result

def shape(p):
    return (p['status'],p['tokens'],p['snapshot_bytes'],p['source_kind'],p['source_slot'],
            p['stage_count'],tuple(p['checkpoint_count_per_stage']))

def compare(a,b):
    aa=defaultdict(list);bb=defaultdict(list)
    for p in a['parks']:
        if p['grouping']=='complete' and p['source_kind']!='unknown': aa[shape(p)].append(p)
    for p in b['parks']:
        if p['grouping']=='complete' and p['source_kind']!='unknown': bb[shape(p)].append(p)
    matches=[]
    for key in sorted(aa.keys() & bb.keys(),key=repr):
        x,y=aa[key],bb[key]
        ma=statistics.median(p['wall_ms'] for p in x);mb=statistics.median(p['wall_ms'] for p in y)
        matches.append({'shape':list(key),'baseline_n':len(x),'candidate_n':len(y),
            'baseline_park_ids':[p['park_id'] for p in x],'candidate_park_ids':[p['park_id'] for p in y],
            'baseline_wall_median_ms':ma,'candidate_wall_median_ms':mb,
            'candidate_wall_change_percent':pct(mb-ma,ma),
            'baseline_component_medians_ms':{k:statistics.median(p['component_sums_ms'][k] for p in x) for k in EXCLUSIVE+NESTED},
            'candidate_component_medians_ms':{k:statistics.median(p['component_sums_ms'][k] for p in y) for k in EXCLUSIVE+NESTED}})
    return {'validity':'shape-matched diagnostic only; workload/content identity must be proved independently',
            'matched_shapes':matches,'baseline_unmatched_shapes':[list(k) for k in aa.keys()-bb.keys()],
            'candidate_unmatched_shapes':[list(k) for k in bb.keys()-aa.keys()]}

def render(data):
    lines=['PARK_TRACE numeric audit','Units: ms; exact byte fields in bytes; trace KV MB is truncated MiB.',
           'KV attrs/resize/copy are nested under KV wall, never added a second time.','']
    for log in data['logs']:
        lines.extend([f"Input: {log['input']}",f"SHA256: {log.get('input_sha256','synthetic')}",json.dumps(log['summary'],sort_keys=True)])
        for p in log['parks']:
            src=p['source_kind']+(f":{p['source_slot']}" if p['source_slot'] is not None else '')
            lines.append(f"park {p['park_id']} line {p['line']}: {p['status']} tokens={p['tokens']} source={src} snapshot_bytes={p['snapshot_bytes']} reused_kv_bytes={p['reused_kv_bytes']} wall={p['wall_ms']:.3f} ms stages={p['stage_count']} grouping={p['grouping']}")
            if p['grouping']=='complete':
                for k in EXCLUSIVE:
                    pc=p['whole_percentages'][k]
                    lines.append(f"  {k}={p['component_sums_ms'][k]:.3f}"+(f" ({pc:.2f}% of whole)" if pc is not None else ''))
                lines.append('  nested KV: '+', '.join(f"{k}={p['component_sums_ms'][k]:.3f} ms" for k in NESTED))
                lines.append(f"  exclusive sum={p['exclusive_trace_sum_ms']:.3f} ms; unattributed residual={p['unattributed_residual_ms']:.3f} ms; overwrite_added_bytes={p['overwrite_added_bytes_sum']}")
            else: lines.append('  No wholepark percentages: stage grouping ambiguous.')
            ri=p['following_read_index']
            if ri is not None:
                rd=log['reads'][ri];en=rd.get('end') or {}
                lines.append(f"  next read (log order only): id={rd['begin']['read_id']} from={rd['begin'].get('from')} tokens={rd['begin'].get('tokens')} read_ms={en.get('read_ms')} native_BT_before_done={en.get('decode_rows_before_done')} read_ok={en.get('read_ok')} pump_ok={en.get('pump_ok')}")
        if log['warnings']: lines.append('Warnings: '+json.dumps(log['warnings'],sort_keys=True))
        if log['orphans']: lines.append('Orphan stages: '+json.dumps(log['orphans'],sort_keys=True))
        lines.append('')
    if 'comparison' in data:
        lines.extend(['Shape comparison (diagnostic; not workload-identity proof):',json.dumps(data['comparison'],indent=2),''])
    lines.extend(['Interpretation limits:']+['- '+n for n in NOTES])
    return '\n'.join(lines)+'\n'

SAMPLE='strata parktrace: joins 1.0 ms, sync 2.0 ms, 3 checkpoints copied 3.0 ms, live state 4.0 ms, begin 5.0 ms, K/V 67 MB 15.0 ms (layer max 15.00, sum 15.00; attrs 1.00, resize 8.00, copy 5.00{tail})'
def sample(n=8,tail='',suffix='',wall=300,tokens=59939,bytes_=1402574080):
    rows=[]
    for i in range(n):
        rows.extend(['strata parktrace: save join 0.01 ms','strata parktrace: save vectors fresh, resize 1.23 ms',SAMPLE.format(tail=tail)])
    rows.append(f'strata serve: conversation cache: parked {tokens} tokens{suffix} in {wall:.1f} ms; parked=5 bytes=2045560240 evictions=0 snapshot_bytes={bytes_} reused_kv_bytes=0')
    return '\n'.join(rows)+'\n'

class Tests(unittest.TestCase):
    def test_v3_and_nested(self):
        d=parse_text(sample());p=d['parks'][0]
        self.assertEqual(d['summary']['complete_stage_groups'],1)
        self.assertEqual(d['ignored_checkpoint_trace_lines'],16)
        self.assertEqual(p['exclusive_trace_sum_ms'],240)
        self.assertEqual(p['unattributed_residual_ms'],60)
        self.assertEqual(p['component_sums_ms']['kv_resize_ms'],64)
        self.assertIsNone(p['overwrite_added_bytes_sum'])
        self.assertEqual(p['checkpoint_count_per_stage'],[3]*8)
    def test_v4_and_source(self):
        p=parse_text(sample(tail='; overwrite_added_bytes 123',suffix=' from slot 3'))['parks'][0]
        self.assertEqual((p['source_kind'],p['source_slot']),('slot',3))
        self.assertEqual(p['overwrite_added_bytes_sum'],984)
    def test_missing_or_excess_stages_fail_closed(self):
        for n in (0,7,9,16):
            p=parse_text(sample(n))['parks'][0]
            self.assertEqual(p['grouping'],'ambiguous');self.assertIsNone(p['whole_percentages'])
    def test_checkpoint_lines_do_not_become_stages(self):
        d=parse_text('strata parktrace: save small copies 4.20 ms\n'+sample())
        self.assertEqual(d['parks'][0]['stage_count'],8)
    def test_abandoned_capture_and_restart(self):
        text=SAMPLE.format(tail='')+'\nstrata serve: conversation cache: allocation failed; skip parking\n'+sample()
        d=parse_text(text);self.assertEqual(len(d['orphans']),1);self.assertEqual(d['parks'][0]['stage_count'],8)
        d=parse_text(SAMPLE.format(tail='')+'\nengine started: secret args\n'+sample())
        self.assertEqual(d['parks'][0]['session'],1);self.assertNotIn('secret',json.dumps(d))
    def test_reader_and_no_raw_token(self):
        text=sample()+'strata piped concurrent MTP: begin read_id=5 from=0 tokens=59932 chunk=8192 active_slots=4 stages=8 own_buffers=8 resident_stages=8\nBT 1 123456789\nstrata piped concurrent MTP: end read_id=5 read_ms=5875.702 decode_rows_before_done=647 read_ok=0 pump_ok=1 stopped_at=-1 admit_stop_requested=0 yield_slot=-1\n'
        d=parse_text(text);r=d['reads'][0]
        self.assertEqual(d['parks'][0]['following_read_index'],0)
        self.assertEqual(r['native_bt_outputs_before_done'],647);self.assertFalse(r['read_succeeded'])
        self.assertEqual(r['observed_bt_lines_during_read'],1);self.assertNotIn('123456789',json.dumps(d))
    def test_compare_requires_shape(self):
        a=parse_text(sample());b=parse_text(sample(wall=250,tail='; overwrite_added_bytes 55'))
        c=compare(a,b);self.assertEqual(len(c['matched_shapes']),1)
        self.assertAlmostEqual(c['matched_shapes'][0]['candidate_wall_change_percent'],-100/6)
        for b in (parse_text(sample(tokens=100001)),parse_text(sample(bytes_=42)),parse_text(sample(suffix=' from slot 1'))):
            self.assertEqual(compare(a,b)['matched_shapes'],[])
    def test_malformed_and_zero_wall(self):
        d=parse_text('strata parktrace: joins broken\n'+sample(wall=0))
        self.assertEqual(d['warnings'][0]['code'],'malformed_full_snapshot_stat')
        self.assertIsNone(d['parks'][0]['whole_percentages']['sync_ms'])
    def test_unknown_source_and_incomplete_read(self):
        p=parse_text(sample(suffix=' unknown source'))['parks'][0];self.assertEqual(p['source_kind'],'unknown')
        d=parse_text('strata piped concurrent MTP: begin read_id=1 from=0 tokens=1\n')
        self.assertFalse(d['reads'][0]['complete'])


def resolve_inputs(items):
    paths=[]
    for item in items:
        p=Path(item)
        if p.is_dir():
            found=sorted(p.rglob('engine.log'))
            if not found: raise ValueError(f'no engine.log under {p}')
            paths.extend(found)
        elif p.is_file(): paths.append(p)
        else: raise ValueError(f'input does not exist: {p}')
    return list(dict.fromkeys(p.resolve() for p in paths))

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs',nargs='*',help='engine.log files or directories searched for engine.log')
    ap.add_argument('--expected-stages',type=int,default=8)
    ap.add_argument('--out',type=Path,help='output prefix, writes PREFIX.json and PREFIX.txt')
    ap.add_argument('--compare',action='store_true',help='shape-compare exactly two resolved logs in input order')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args(argv)
    if args.self_test:
        test=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
        return 0 if test.wasSuccessful() else 1
    if not args.inputs: ap.error('supply input logs/directories or --self-test')
    if args.expected_stages < 1: ap.error('--expected-stages must be positive')
    try: paths=resolve_inputs(args.inputs)
    except ValueError as exc: ap.error(str(exc))
    if args.compare and len(paths)!=2: ap.error('--compare requires exactly two resolved logs')
    logs=[]
    for p in paths:
        raw=p.read_bytes();r=parse_text(raw.decode('utf-8',errors='replace'),str(p),args.expected_stages)
        r['input_sha256']=hashlib.sha256(raw).hexdigest();logs.append(r)
    data={'schema':1,'diagnostic_only':True,'logs':logs}
    if args.compare: data['comparison']=compare(*logs)
    report=render(data)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        Path(str(args.out)+'.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        Path(str(args.out)+'.txt').write_text(report)
        print(json.dumps({'outputs':[str(args.out)+'.json',str(args.out)+'.txt'],'logs':[r['summary'] for r in logs]},sort_keys=True))
    else: print(report,end='')
    return 0
if __name__=='__main__': sys.exit(main())
