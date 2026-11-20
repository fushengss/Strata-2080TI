#!/usr/bin/env python3
"""Overlap-only experiment identity tests; no HTTP/GPU/process launch."""
import copy,json
from pathlib import Path
import runner_interleave as runner
import compare_interleave as compare
import prefill_interleave_check as prefill
import long_concurrency_check as long_check


def main():
    base={'env':{},'api_key':'PRIVATE_TEST_KEY','log':'old','exe':'old','reasoning_budget_engine_cap':True}
    configs=[runner.make_config(base,arm,Path('/new/engine'),8092,Path('/own/log'+arm),'random-'+arm) for arm in ('off','on')]
    assert runner.VARIABLES==('STRATA_PIPED_CONCURRENT_MTP',)
    assert [c['env']['STRATA_PARK_OVERWRITE'] for c in configs]==['0','0']
    assert runner.comparable_config(configs[0])==runner.comparable_config(configs[1])
    assert prefill.comparable_config(configs[0])==prefill.comparable_config(configs[1])
    assert long_check.normalized_config(configs[0])==long_check.normalized_config(configs[1])
    metas=[{'config':c} for c in configs];manifests=[{'experiment_profile':'concurrent-mtp-only-park-overwrite-fixed-off'} for _ in range(2)]
    assert all(compare.overlap_identity(metas,manifests).values())
    for index in (0,1):
        for value in (None,'1','2'):
            bad=copy.deepcopy(metas)
            if value is None:bad[index]['config']['env'].pop('STRATA_PARK_OVERWRITE')
            else:bad[index]['config']['env']['STRATA_PARK_OVERWRITE']=value
            assert not all(compare.overlap_identity(bad,manifests).values())
    for value in (None,'0','2'):
        bad=copy.deepcopy(metas)
        if value is None:bad[1]['config']['env'].pop('STRATA_PIPED_CONCURRENT_MTP')
        else:bad[1]['config']['env']['STRATA_PIPED_CONCURRENT_MTP']=value
        assert not all(compare.overlap_identity(bad,manifests).values())
    bad=copy.deepcopy(manifests);bad[0].pop('experiment_profile');assert not all(compare.overlap_identity(metas,bad).values())
    # Only the named overlap experiment flag and random key/log may normalize away.
    for field in ('STRATA_PARK_OVERWRITE','STRATA_PIPED_TARGET_HEAD','STRATA_PIPED_ASYNC_SAMPLE','STRATA_PARK_TRACE','UNRELATED_FLAG'):
        changed=copy.deepcopy(configs[1]);changed['env'][field]='DIFFERENT'
        assert runner.comparable_config(configs[0])!=runner.comparable_config(changed)
        assert prefill.comparable_config(configs[0])!=prefill.comparable_config(changed)
        assert long_check.normalized_config(configs[0])!=long_check.normalized_config(changed)
    assert base['env']=={} and base['api_key']=='PRIVATE_TEST_KEY'
    assert 'PRIVATE_TEST_KEY' not in json.dumps(runner.safe_config(configs[0]))
    print('PASS: explicit overlap off/on and park fixed0 identity, missing/mismatched flag/profile rejection, only overlap flag normalized; park remains in equality, unrelated config preserved, private base untouched; no HTTP/GPU')
if __name__=='__main__':main()
