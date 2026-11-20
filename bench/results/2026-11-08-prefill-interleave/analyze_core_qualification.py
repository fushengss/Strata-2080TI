#!/usr/bin/env python3
"""Read-only local review of the two completed core pairs. Never runs workloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

def main():
    final=HERE/'core-qualification-v3-on-off2'; prior=HERE/'performance-v3-h6-off-on1'
    deploy=mod('deploy_review',HERE/'deploy_concurrent_prefill.py')
    strict=deploy.validate_qualification(final,HERE/'incremental-harness-v1-freeze.json')
    ie=mod('interleave_review',final/'harness/interleave_evidence.py')
    comparison=read(next((final/'comparisons').glob('*/comparison.json')))
    recomputed=read(HERE/'core-qualification-v3-independent-recomputed/comparison.json')
    require(recomputed['all_checks_passed'] and all(recomputed['suite_checks'].values()),'Fresh exact comparator failed')
    counts={'prefill':len(recomputed['suites']['prefill']['pairs']),
      **{s:len(recomputed['suites'][s]['exact_outputs']['pairs']) for s in ['primary','mixed100k']},
      **{s:len(recomputed['suites'][s]['exact_normal_outputs']['pairs']) for s in ['admission','byield']},
      'correctness':sum(x['name']!='cancel-stream' for x in recomputed['suites']['correctness']['comparisons']),
      'c8':len(recomputed['suites']['c8']['comparisons']),
      'budget':sum(x['name']!='cancel-stream' for x in recomputed['suites']['budget']['comparisons'][0]['requests'])}
    require(counts=={'prefill':6,'primary':6,'mixed100k':5,'correctness':13,'c8':8,'admission':6,'byield':6,'budget':9},'Normal exact cohort changed')
    report={'decision':'CORE8_CORRECTNESS_IDENTITY_PASS_PENDING_INCREMENTAL_PAIR_AND_OPERATOR_ACCEPTANCE',
      'source_commit':'bf5a8e5b24b1973efbae747a109f5a970b25e09d','engine_sha256':deploy.ENGINE,
      'strict_identity_receipt':{**strict,'comparison_path':str(strict['comparison_path'])},
      'independently_recomputed_comparison_sha256':sha(HERE/'core-qualification-v3-independent-recomputed/comparison.json'),
      'normal_exact_requests_by_suite':counts,
      'normal_exact_request_total':sum(counts.values()),'intentional_cancel_cases_separate':3,
      'arms':{},'performance_pairs':[],'solo_prefill_pooled':{},
      'limits':['Two ordered pairs only; not a stable tail distribution or production SLA.',
        'Native BT counts include termination; they are neither verification rows nor API-visible tokens.',
        'Whole API rate includes admission and final drain; SSE gaps are chunk-arrival intervals, not token TBT.',
        'Mixed prefill latency rises while peers continue; solo no material observed paired regression is not a zero-regression guarantee.',
        'First pair used h6 and five suites; reverse used reviewed v1 core8. Compare within each pair, not cross-harness arms.',
        '13 correctness/9 budget normal completions are exact; their deliberate cancellations use shorter-prefix parity, not full completion equality.']}
    for arm in ['01-on','02-off']:
        directory=final/arm; meta=read(directory/'metadata.json'); raw=ie.parse((directory/'engine.log').read_text(errors='replace'))
        admission=read(directory/'admission/results.json'); byield=read(directory/'byield/results.json')
        approved=admission.get('approved_cancelled_read_ids',[])
        failed=[r['read_id'] for r in raw['pairs'] if r['read_ok']!=1]
        require(raw['unique_and_paired_ids'] and raw['source_safe'] and raw['source_ids_match_reads'] and not raw['cuda_error_present'],'Whole native source/error evidence failed')
        require(failed==approved and all(r['pump_ok']==1 for r in raw['pairs']),'Unexpected failed reader/pump')
        if arm.endswith('off'):require(not raw['pairs'] and not raw['fallback'],'Disabled route ran')
        require(not meta['interleave_suite_evidence']['prefill']['pairs'],'Solo prefill used reader route')
        a={'engine_log_sha256':sha(directory/'engine.log'),'whole_native_reader_count':len(raw['pairs']),
           'source_record_count':len(raw['source']),'all_source_dispatch_delta_zero':all(x['dispatch_delta']==0 for x in raw['source']),
           'all_pumps_success':all(x['pump_ok']==1 for x in raw['pairs']),'only_failed_read_ids':failed,
           'strict_in_read_native_BT_lower_bound':raw['in_read_decode_rows_lower_bound'],
           'prefill_reader_count':0,'reader_counts_by_suite':{k:len(v['pairs']) for k,v in meta['interleave_suite_evidence'].items()},
           'target_head_windows_by_suite':{k:v['windows'] for k,v in meta['target_head_suite_evidence'].items()},
           'async_sample_windows_by_suite':{k:v['windows'] for k,v in meta['async_sample_suite_evidence'].items()},
           'admission_checks':admission['checks'],'admission_cancel_proof':admission.get('intentional_cancel_native'),
           'byield_checks':byield['checks'],'byield_native_proof':byield['native_qualification'].get('proof'),
           'cohorts':{},'fixture_history':None,'budget_continuations':None}
        for suite,filename in [('correctness','correctness.json'),('budget','budget-correctness.json')]:
            v=read(directory/suite/filename);require(v['all_checks_passed'] and all(v['checks'].values()),suite+' failed')
            a['cohorts'][suite]={'checks':v['checks'],'normal':sum(not x.get('intentional_cancel') for x in v['records']),
              'cancel':sum(bool(x.get('intentional_cancel')) for x in v['records']),
              'overlap':[{'name':x['name'],'overlap':x['overlap']} for x in v['cohorts'] if x.get('overlap')]}
            if suite=='correctness':a['fixture_history']=v['history_request_coverage']
            else:a['budget_continuations']=[{'name':x['name'],'read_tokens':x['continuation']['read'],'bounded':x['bounded_continuation_read']} for x in v['continuations']]
        report['arms'][arm]=a
    pooled={arm:{kind:[] for kind in ['cold','warm']} for arm in ['off','on']}
    for directory in [prior,final]:
        manifest=read(directory/'manifest.json');cp=next((directory/'comparisons').glob('*/comparison.json'));c=read(cp)
        require(c['all_checks_passed'] and all(c['suite_checks'].values()),'Prior/final pair failed')
        tags={a['arm']:a['tag'] for a in manifest['arms']}
        pair={'run':directory.name,'order':manifest['sequence'],'manifest_sha256':sha(directory/'manifest.json'),
              'comparison_sha256':sha(cp),'suites':manifest['suites'],'mixed':{},'solo_prefill':{},'c8_whole_api_rate':{}}
        for suite in ['primary','mixed100k']:
            metrics=c['suites'][suite]['metrics'];rows=[]
            for arm in ['off','on']:
                r=read(directory/tags[arm]/suite/'results.json')
                require(r['all_checks_passed'] and r['visible_phase_coverage']['all_checks_passed'],'Finite gate failed')
                n=sum(x['usage']['completion_tokens'] for x in r['records']);rows.append(n)
            require(rows[0]==rows[1],'Unequal actual work')
            v=metrics['phases']['mixed']['visible_content']
            pair['mixed'][suite]={'actual_output_tokens':rows[0],'whole_api_tps':metrics['whole_case_tokens_per_s'],
              'max_visible_mixed_silence_s':v['max_lane_clipped_silence_s'],
              'visible_lane_p95_range_s':{arm:[min(x['p95_s'][arm] for x in v['lanes']),max(x['p95_s'][arm] for x in v['lanes'])] for arm in ['off','on']},
              'injections':metrics['injections']}
        for arm in ['off','on']:
            rs=read(directory/tags[arm]/'prefill/results.json');require(all(r['valid'] for r in rs),'Solo request failed')
            pair['solo_prefill'][arm]={}
            for kind in ['cold','warm']:
                selected=[r for r in rs if r['kind']==kind];values=[r['engine']['read_ms'] for r in selected]
                pooled[arm][kind].extend(values)
                pair['solo_prefill'][arm][kind]={'native_read_ms':values,'median_ms':statistics.median(values),
                  'prompt_tokens':sorted({r['usage']['prompt_tokens'] for r in selected}),
                  'cached_tokens':sorted({r['usage']['cached_tokens'] for r in selected})}
            cc=read(directory/tags[arm]/'c8/c8-check.json')
            pair['c8_whole_api_rate'][arm]=cc['timing']['whole_batch']['completed_api_tokens_per_s']
        report['performance_pairs'].append(pair)
    for arm in pooled:
        report['solo_prefill_pooled'][arm]={kind:{'n':len(values),'native_read_ms':values,'min_ms':min(values),'max_ms':max(values),'median_ms':statistics.median(values)} for kind,values in pooled[arm].items()}
    out=HERE/'final-core-qualification-independent-review.json';out.write_text(json.dumps(report,indent=2)+'\n')
    lines=['Final core8 independent review: PASS (incremental pair and operator acceptance pending).',
      'Same selected45ea engine, library, server, base, calibration, flags and frozen v1 source; complete parent and both arms, all8 suites rc0.',
      'Fresh frozen comparator rerun passes all8 suites.59 normal request pairs exact;3 deliberate cancellations are separately qualified, not counted as full equal completions.',
      'Native reader/source markers reparsed from complete logs: no CUDA errors, all pumps success, every source dispatch_delta0; only approved admission cancel read may fail. Solo prefill launches0 readers.',
      'Correctness covers actual32/128/257 history fixtures, sampled/greedy/penalties/bias and cancel/recovery; budget wraps all4, each continuation rereads19 tokens.',
      'BYIELD candidate stops8192, parks matching state, resumes8192 with unchanged plan and completes short before long;4 peers span transfer/recovery.', '']
    for pair in report['performance_pairs']:
        lines.append(pair['run']+' order '+','.join(pair['order']))
        for suite,v in pair['mixed'].items():
            rate=v['whole_api_tps'];gap=v['max_visible_mixed_silence_s']
            lines.append(f"  {suite}: actualoutputs{v['actual_output_tokens']}, wholeAPI {rate['off']:.3f}->{rate['on']:.3f}tok/s ({rate['on_relative_change_pct']:+.2f}%), mixedmaxvisiblegap {gap['off']:.3f}->{gap['on']:.3f}s")
            for inj in v['injections']:
                t=inj['native_prompt_ms'];tt=inj['ttft_first_output_s'];lines.append(f"    prompt{inj['actual_prompt_tokens']}: nativeprompt {t['off']:.1f}->{t['on']:.1f}ms ({t['on_relative_change_pct']:+.2f}%), firstoutput {tt['off']:.3f}->{tt['on']:.3f}s")
        for kind in ['cold','warm']:
            a=pair['solo_prefill']['off'][kind];b=pair['solo_prefill']['on'][kind];lines.append(f"  solo{kind} native ms OFF{a['native_read_ms']} ON{b['native_read_ms']}; n3 medians{a['median_ms']}->{b['median_ms']}")
    lines.extend(['','Six observations perarm/kind are descriptive only; repeatedfirst/cold/warm order and2pairs do not establish stable tail.',
      'Across bothpairs primary throughput gain2.46–6.17%;100k −0.49–+2.02% (no clear whole-case gain). Primary worst mixed gaps~10.94–10.96s→0.659–0.662s;100k~17.13–17.34s→2.438–2.547s.',
      'Tradeoff: lane p95 chunk gaps rise, mixed prompt latency+~11%10k/+~7%60k/+~5.5–6%100k; solo paired medians and ranges overlap. Do not call this zero prefill cost, per-token latency, or full GPU utilization.',
      'No remote/GPU/service operations. No deployment verdict.'])
    (HERE/'final-core-qualification-independent-review.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'decision':report['decision'],'normal_exact_pairs':59,'intentional_cancel_separate':3,'on_native_readers':report['arms']['01-on']['whole_native_reader_count'],'all_paired_suites':True,'json':str(out)}))
if __name__=='__main__':main()
