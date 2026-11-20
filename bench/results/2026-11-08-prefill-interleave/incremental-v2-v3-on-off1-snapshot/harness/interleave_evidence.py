"""Numeric reader/pump evidence. No legacy meanwhile inference; no raw log export."""
import re

STARTUP = 'strata serve: STRATA_PIPED_CONCURRENT_MTP: enabled; resident own-buffer whole reads; prepared PLE published before batch launch'
BEGIN = re.compile(r'^strata piped concurrent MTP: begin read_id=(\d+) from=(\d+) tokens=(\d+) chunk=(\d+) active_slots=(\d+) stages=(\d+) own_buffers=(\d+) resident_stages=(\d+)$', re.M)
END = re.compile(r'^strata piped concurrent MTP: end read_id=(\d+) read_ms=([\d.]+) decode_rows_before_done=(\d+) read_ok=([01]) pump_ok=([01]) stopped_at=(-?\d+) admit_stop_requested=([01]) yield_slot=(-?\d+)$', re.M)
FALLBACK = re.compile(r'^strata piped concurrent MTP: fallback from=(\d+) tokens=(\d+) reason=([A-Za-z0-9_-]+)$', re.M)
SOURCE = re.compile(r'^strata resident read source: read_id=(\d+) staged=([01]) lookahead_idle=([01]) dispatch_delta=(-?\d+)$',re.M)
CUDA_ERROR = re.compile(r'CUDA[^\n]{0,100}(?:error|fail)|illegal memory access|device.side assert|CUBLAS_STATUS_(?!SUCCESS)|cudaError|out of memory|nan logits|non.finite logits', re.I)


def parse(text):
    begins=[dict(zip(('read_id','from','tokens','chunk','active_slots','stages','own_buffers','resident_stages'),map(int,m))) for m in BEGIN.findall(text)]
    ends=[dict(zip(('read_id','read_ms','decode_rows_before_done','read_ok','pump_ok','stopped_at','admit_stop_requested','yield_slot'),
                   [float(v) if i==1 else int(v) for i,v in enumerate(m)])) for m in END.findall(text)]
    sources=[dict(zip(('read_id','staged','lookahead_idle','dispatch_delta'),map(int,m))) for m in SOURCE.findall(text)]
    source_lines=[line for line in text.splitlines() if line.startswith('strata resident read source:')]
    reasons=[{'from':int(a),'tokens':int(b),'reason':c} for a,b,c in FALLBACK.findall(text)]
    marker_lines=[line for line in text.splitlines() if line.startswith('strata piped concurrent MTP:')]
    known=len(begins)+len(ends)+len(reasons)
    byid={r['read_id']:r for r in ends}
    pairs=[{**b,**byid[b['read_id']]} for b in begins if b['read_id'] in byid]
    return {'begin':begins,'end':ends,'fallback':reasons,'pairs':pairs,'source':sources,
            'all_source_lines_parsed':len(source_lines)==len(sources),
            'source_ids_match_reads':sorted(r['read_id'] for r in sources)==sorted(r['read_id'] for r in begins),
            'source_safe':all(r['lookahead_idle']==1 and r['dispatch_delta']==0 for r in sources),
            'startup_count':text.count(STARTUP),'cuda_error_present':bool(CUDA_ERROR.search(text)),
            'all_marker_lines_parsed':len(marker_lines)==known,
            'unique_and_paired_ids':len({r['read_id'] for r in begins})==len(begins)==len(ends)==len(byid)==len(pairs),
            'in_read_decode_rows_lower_bound':sum(r['decode_rows_before_done'] for r in ends),
            'scope':'Only pump rows committed after reader start and before observed reader done. Final drain and ambiguous boundary pumps excluded; rows count native emitted BT output tokens (including termination), not API visible-content tokens or only accepted draft tokens.'}


def validate(arm,suite,text,approved_cancelled_read_ids=()):
    result=parse(text)
    approved=set(approved_cancelled_read_ids)
    approved_scope=not approved or (arm=='on' and suite=='admission')
    allowed=approved if approved_scope else set()
    checks={'no_cuda_errors':not result['cuda_error_present'],'all_marker_lines_parsed':result['all_marker_lines_parsed'],
            'unique_and_paired_ids':result['unique_and_paired_ids'],
            'cancel_approval_scope':approved_scope,
            'approved_ids_are_exact_failed_reads':not approved or (len(approved)==len(approved_cancelled_read_ids) and all(any(p['read_id']==i and p['read_ok']==0 and p['pump_ok']==1 for p in result['pairs']) for i in approved)),
            'source_lines_parsed':result['all_source_lines_parsed'],
            'source_ids_match_reads':result['source_ids_match_reads'],
            'no_lookahead_or_expert_dispatch':result['source_safe'],
            'all_launched_reads_safe':all(r['active_slots']>0 and r['stages']==r['own_buffers']==r['resident_stages']==8 for r in result['pairs']),
            'all_launched_reads_completed':all(r['pump_ok']==1 and (r['read_ok']==1 or r['read_id'] in allowed) for r in result['pairs'])}
    if arm=='off':checks['disabled_no_route']=not result['begin'] and not result['fallback']
    elif suite=='prefill':checks['solo_original_route']=not result['begin']
    elif suite in ('primary','mixed100k'):
        target=60000 if suite=='primary' else 100000
        matching=[r for r in result['pairs'] if r['tokens']>=.9*target and r['from']+r['tokens']<=target-1 and r['active_slots']>=4]
        checks['long_injection_whole_read_exercised']=bool(matching)
        checks['long_injection_in_read_progress']=bool(matching) and all(r['decode_rows_before_done']>0 for r in matching)
    elif suite=='admission':checks['first_and_repeat_read_exercised']=len(result['pairs'])>=2
    result['approved_cancelled_read_ids']=sorted(allowed);result['checks']=checks;result['all_checks_passed']=all(checks.values())
    return result


def self_test():
    begin='strata piped concurrent MTP: begin read_id=7 from=0 tokens=59999 chunk=2048 active_slots=4 stages=8 own_buffers=8 resident_stages=8'
    end='strata piped concurrent MTP: end read_id=7 read_ms=9123.5 decode_rows_before_done=128 read_ok=1 pump_ok=1 stopped_at=59999 admit_stop_requested=0 yield_slot=-1'
    text=begin+'\n'+end+'\nstrata resident read source: read_id=7 staged=1 lookahead_idle=1 dispatch_delta=0'
    assert validate('on','primary',text)['all_checks_passed']
    assert not validate('off','primary',text)['all_checks_passed']
    assert not validate('on','primary',text.replace('decode_rows_before_done=128','decode_rows_before_done=0'))['all_checks_passed']
    assert not validate('on','primary',begin)['all_checks_passed']
    assert not validate('on','primary',text+'\n'+end)['all_checks_passed']
    assert not validate('on','primary',text+'\nstrata piped concurrent MTP: unparsed') ['all_checks_passed']
    assert not validate('on','primary',text+'\nCUDA error: illegal memory access')['all_checks_passed']
    assert not validate('on','primary',text.replace('dispatch_delta=0','dispatch_delta=1'))['all_checks_passed']
    assert not validate('on','primary',text.replace('lookahead_idle=1','lookahead_idle=0'))['all_checks_passed']
    assert validate('on','prefill','')['all_checks_passed']
    assert validate('off','primary','')['all_checks_passed']
    print('PASS: exact reader markers; strict in-read rows, pairing, disabled/solo/error/zero-progress rejection')


if __name__=='__main__':self_test()
