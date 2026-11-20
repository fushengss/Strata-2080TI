"""Narrow exception for one proven intentionally cancelled admission reader.

A stopped Prefill.run may return read_ok=0. This module does not excuse errors
based on a generic cancelled log line: it binds an open reader observation,
a closed pre-reuse log window, the one no-output HTTP cancel, a complete native
cancelled report, and a normal same-request retry. Pure CPU; no HTTP/service actions.
"""
import hashlib
import json
from pathlib import Path
import re
import interleave_evidence as ie

CANCELLED=re.compile(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) of (\d+) read in ([\d.]+) ms \(([\d.]+) tok/s\), (\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+), (\d+) checkpoints \(cancelled\)$',re.M)


def require(condition,message):
    if not condition:raise RuntimeError(message)


def proof(arm,expected_count,observation,records,window_text,observed_prefix_text):
    parsed=ie.parse(window_text);prefix=ie.parse(observed_prefix_text);byname={r['name']:r for r in records}
    cancel=byname.get('admission-cancel-cold',{});reuse=byname.get('admission-cancel-reuse',{})
    wanted=observation.get('reader_id');pairs=[p for p in parsed['pairs'] if p['read_id']==wanted]
    native=[]
    for match in CANCELLED.finditer(window_text):
        v=match.groups();native.append({'prompt_tokens':int(v[0]),'reused_tokens':int(v[1]),'read_tokens':int(v[2]),
              'fresh_tokens':int(v[3]),'read_ms':float(v[4]),'generated_tokens':int(v[6]),'log_offset':match.start()})
    checks={'candidate_arm_only':arm=='on','expected_positive_count':type(expected_count) is int and expected_count>0,
            'open_observation_before_output':observation.get('before_first_output') is True and observation.get('reader_open_at_observation') is True,
            'observed_read_open_in_snapshot':type(wanted) is int and sum(p['read_id']==wanted for p in prefix['begin'])==1 and not any(p['read_id']==wanted for p in prefix['end']),
            'unique_expected_read_pair':len(pairs)==1,'all_window_markers_parsed':parsed['all_marker_lines_parsed'],
            'all_window_reads_paired':parsed['unique_and_paired_ids'],
            'only_observed_read_failed':len([p for p in parsed['pairs'] if p['read_ok']==0])==1 and [p['read_id'] for p in parsed['pairs'] if p['read_ok']==0]==[wanted],
            'all_pumps_ok':all(p['pump_ok']==1 for p in parsed['pairs']),
            'no_cuda_errors':not parsed['cuda_error_present'],
            'http_one_intentional_cancel_no_output':sum(r.get('role')=='cancel' for r in records)==1 and cancel.get('error')=={'kind':'intentional_admission_cancel'} and cancel.get('chunks')==[] and cancel.get('saw_done') is False,
            'same_request_reused':bool(cancel.get('request_sha256')) and cancel.get('request_sha256')==reuse.get('request_sha256'),
            'reuse_normal_complete':reuse.get('successful') is True and reuse.get('saw_done') is True and reuse.get('finish_reason')=='length' and reuse.get('error') is None and bool(reuse.get('gates')) and all(reuse['gates'].values()),
            'one_cancelled_native_report':len(native)==1}
    if pairs:
        p=pairs[0]
        checks['observed_large_read_matches_prompt']=p['tokens']>=.9*expected_count and 0<=p['from'] and p['from']+p['tokens']<=expected_count-1
        checks['failed_read_only_pump_success']=p['read_ok']==0 and p['pump_ok']==1
    if len(native)==1 and pairs:
        row=native[0];endpos=next(m.end() for m in ie.END.finditer(window_text) if int(m.group(1))==wanted)
        nextbegin=min((m.start() for m in ie.BEGIN.finditer(window_text) if m.start()>endpos),default=len(window_text))
        checks['cancel_report_bound_after_end_before_next_read']=endpos<row['log_offset']<nextbegin
        checks['cancel_native_exact_cold_prompt_zero_output']=row['prompt_tokens']==expected_count and row['reused_tokens']==0 and row['fresh_tokens']==expected_count and 0<=row['read_tokens']<=row['fresh_tokens'] and row['generated_tokens']==0
    else:checks['cancel_report_bound_after_end_before_next_read']=checks['cancel_native_exact_cold_prompt_zero_output']=False
    passed=all(checks.values())
    return {'all_checks_passed':passed,'checks':checks,'approved_cancelled_read_ids':[wanted] if passed else [],
            'cancelled_native_reports':native,'read_pairs':pairs,'scope':'Only this exact observed reader may return read_ok=0; pump_ok must remain1. No allowance for another reader, suite, CUDA error, generated content, or failed retry.'}


def read_window(path,window):
    stat=path.stat();start,end,observed=window.get('start_offset'),window.get('end_offset'),window.get('observation_offset')
    require(stat.st_ino==window.get('log_inode'),'Cancellation log inode differs')
    require(all(type(x) is int for x in (start,end,observed)) and 0<=start<observed<=end<=stat.st_size,'Invalid closed cancellation log window')
    with path.open('rb') as handle:
        handle.seek(start);raw=handle.read(end-start)
    require(hashlib.sha256(raw).hexdigest()==window.get('sha256'),'Cancellation log window differs')
    return raw.decode(errors='replace'),raw[:observed-start].decode(errors='replace')


def approved_from_results(directory,engine_log,arm):
    """Runner independently reconstructs approval from its own frozen log and results."""
    result=json.loads((directory/'results.json').read_text());manifest=json.loads((directory/'manifest.json').read_text())
    require(result.get('all_checks_passed') is True and manifest.get('completed') is True and manifest.get('all_checks_passed') is True,'Admission suite did not qualify')
    require(all(result.get('checks',{}).values()) and bool(result.get('checks')),'Admission checks incomplete')
    if arm=='off':
        require(result.get('approved_cancelled_read_ids')==[],'Baseline may not authorize a failed reader');return []
    rows=[r for r in manifest['fixtures'] if r['name']=='admission-cancel-cold'];require(len(rows)==1,'Unique cancel fixture missing')
    text,prefix=read_window(engine_log,result['cancel_log_window'])
    observed=result['cancel_observation'];reuse=next((r for r in result['records'] if r['name']=='admission-cancel-reuse'),{})
    require(result['cancel_log_window']['closed_epoch']<=reuse.get('start_epoch',0),'Cancel window was not closed before retry')
    require(observed.get('epoch',0)<=result['cancel_log_window']['closed_epoch'],'Invalid cancellation observation order')
    recomputed=proof(arm,rows[0]['counted_prompt_tokens'],observed,result['records'],text,prefix)
    require(recomputed['all_checks_passed'],'Intentional-cancel reader proof failed recomputation')
    require(recomputed['approved_cancelled_read_ids']==result.get('approved_cancelled_read_ids'),'Declared cancelled read IDs differ from proof')
    return recomputed['approved_cancelled_read_ids']


def self_test():
    begin='strata piped concurrent MTP: begin read_id=5 from=0 tokens=59938 chunk=2048 active_slots=4 stages=8 own_buffers=8 resident_stages=8'
    end='strata piped concurrent MTP: end read_id=5 read_ms=5875 decode_rows_before_done=647 read_ok=0 pump_ok=1 stopped_at=-1 admit_stop_requested=0 yield_slot=-1'
    source='strata resident read source: read_id=5 staged=1 lookahead_idle=1 dispatch_delta=0'
    cancel='strata serve: prompt 59939 tokens = 0 reused + 8192 of 59939 read in 7240 ms (1131.5 tok/s), 0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 0 checkpoints (cancelled)'
    text='\n'.join((begin,end,source,cancel));obs={'reader_id':5,'before_first_output':True,'reader_open_at_observation':True}
    records=[{'name':'admission-cancel-cold','role':'cancel','error':{'kind':'intentional_admission_cancel'},'chunks':[],'saw_done':False,'request_sha256':'same'},
             {'name':'admission-cancel-reuse','role':'injected','successful':True,'saw_done':True,'finish_reason':'length','error':None,'gates':{'complete':True},'request_sha256':'same'}]
    assert proof('on',59939,obs,records,text,begin)['approved_cancelled_read_ids']==[5]
    for broken in (text.replace('read_ok=0','read_ok=1'),text.replace('pump_ok=1','pump_ok=0'),text.replace('0 generated','1 generated'),text.replace('(cancelled)',''),text.replace('59939 tokens','59940 tokens'),text.replace('0 reused','1 reused'),cancel+'\n'+begin+'\n'+end,text+'\nCUDA error failed'):
        assert not proof('on',59939,obs,records,broken,begin)['all_checks_passed']
    assert not proof('off',59939,obs,records,text,begin)['all_checks_passed']
    assert not proof('on',59939,{**obs,'reader_id':6},records,text,begin)['all_checks_passed']
    assert not proof('on',59939,obs,records,text,begin+'\n'+end)['all_checks_passed']
    for changed in ([{**records[0],'chunks':[{}]},records[1]],[records[0],{**records[1],'request_sha256':'different'}],[records[0],{**records[1],'gates':{'complete':False}}]):
        assert not proof('on',59939,obs,changed,text,begin)['all_checks_passed']
    assert not ie.validate('on','admission',text)['all_checks_passed']
    assert ie.validate('on','admission',text,approved_cancelled_read_ids=[5])['checks']['all_launched_reads_completed']
    assert not ie.validate('on','admission',text,approved_cancelled_read_ids=[6])['all_checks_passed']
    assert not ie.validate('on','primary',text,approved_cancelled_read_ids=[5])['all_checks_passed']
    assert not ie.validate('off','admission',text,approved_cancelled_read_ids=[5])['all_checks_passed']
    extra=begin.replace('read_id=5','read_id=6')+'\n'+end.replace('read_id=5','read_id=6')+'\n'+source.replace('read_id=5','read_id=6')
    assert not proof('on',59939,obs,records,text+'\n'+extra,begin)['all_checks_passed']
    assert not ie.validate('on','admission',text+'\n'+extra,approved_cancelled_read_ids=[5])['all_checks_passed']
    assert not ie.validate('on','admission',text.replace('pump_ok=1','pump_ok=0'),approved_cancelled_read_ids=[5])['all_checks_passed']
    print('PASS: exact intentional-cancel read proof + all unbound/read/pump/output/retry/suite negative gates; no HTTP/GPU')


if __name__=='__main__':self_test()
