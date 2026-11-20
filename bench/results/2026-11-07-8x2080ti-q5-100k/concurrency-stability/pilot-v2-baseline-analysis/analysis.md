# Offline concurrency observations

Output gaps are SSE chunk intervals. Phase edges are censored; figures do not establish an SLO.

| Arm | Level | Phase | Output bytes/s | Worst lane p95 gap (s) | Longest clipped silence (s) | Performance evidence qualified | Arm correctness passed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pilot-v2-c4-sampled/01-off | c4 | baseline | 962.767 | 0.018 | 0.020 | False | True |
| pilot-v2-c4-sampled/01-off | c4 | mixed | 104.098 | 0.024 | 10.056 | False | True |
| pilot-v2-c4-sampled/01-off | c4 | recovery | 799.284 | 0.018 | 0.020 | False | True |

The JSON retains per-lane intervals, refill gaps, occupancy, request input/cache checks, API totals, numeric MTP summaries and all missing-evidence reasons.

## Pair qualification
