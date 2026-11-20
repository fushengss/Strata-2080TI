# Offline concurrency observations

Output gaps are SSE chunk intervals. Phase edges are censored; figures do not establish an SLO.

| Arm | Level | Phase | Output bytes/s | Worst lane p95 gap (s) | Longest clipped silence (s) | Performance evidence qualified | Arm correctness passed |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| pilot-v2-c4-sampled/01-off | c4 | baseline | 962.767 | 0.018 | 0.020 | False | True |
| pilot-v2-c4-sampled/01-off | c4 | mixed | 104.098 | 0.024 | 10.056 | False | True |
| pilot-v2-c4-sampled/01-off | c4 | recovery | 799.284 | 0.018 | 0.020 | False | True |
| pilot-v3-c4-sampled/01-on | c4 | baseline | 1063.971 | 0.024 | 0.025 | False | True |
| pilot-v3-c4-sampled/01-on | c4 | mixed | 95.487 | 0.032 | 10.317 | False | True |
| pilot-v3-c4-sampled/01-on | c4 | recovery | 954.183 | 0.024 | 0.026 | False | True |

The JSON retains per-lane intervals, refill gaps, occupancy, request input/cache checks, API totals, numeric MTP summaries and all missing-evidence reasons.

## Pair qualification

- /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/pilot-v2-c4-sampled:01-off → /Users/fu/project/HAI/Strata/bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/pilot-v3-c4-sampled:01-on c4: unqualified; 2 qualification issues.
