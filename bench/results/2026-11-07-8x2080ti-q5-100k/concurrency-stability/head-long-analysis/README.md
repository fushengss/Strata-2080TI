# Four cold 100k requests: head sharing off/on

Independent evidence checks: PASS. The frozen client exact comparison also passed.

Both arms used the same c930 engine, fb67 library and 40c5 server, eight slots, full expert residency, 3.80 GiB own prompt buffers per GPU, fixed batch MTP and budget cap. Only the shared slot-head flag changed.

All four request/output/channel hashes and prompt/completion/total usage match. Each request has an actual 101147–101148-token prompt, explicit cached_tokens=0, and all 512 output tokens. Two requests are sampled and two greedy.

| Metric | Head off | Head on |
| --- | ---: | ---: |
| Whole batch wall, including prefill (s) | 78.094635 | 77.594571 |
| API output tokens / whole wall (tok/s) | 26.2246 | 26.3936 |
| Common output window (s) | 7.236680 | 7.264042 |
| Common-window output bytes/s | 1132.29 | 1115.49 |

| Lane | Temperature | Off TTFT (s) | On TTFT (s) | Off max chunk gap (s) | On max chunk gap (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| lane-0 | 0.8 | 19.111 | 69.928 | 17.254 | 0.412 |
| lane-1 | 0.0 | 36.562 | 18.605 | 17.014 | 17.250 |
| lane-2 | 0.8 | 53.602 | 36.052 | 16.993 | 16.939 |
| lane-3 | 0.0 | 70.642 | 53.005 | 0.377 | 16.928 |

Whole-wall API throughput changed by +0.644%. This single pair is descriptive; it does not establish a repeatable speed gain or a pure-decode rate.

The JSON retains each lane’s common-window complete gaps and boundary-censored silence intervals, source hashes, individual evidence gates, governor identity and actual offloaded-step counts.

- One off/on cohort; no repetition or confidence interval and no deployment verdict.
- Whole-batch API TPS includes four cold prefills/admissions and final drain; it is not pure decode TPS.
- Common-window bytes/s is observed synthetic output volume, not token TPS. Its boundaries and byte/token ratios can differ.
- SSE gaps are chunk arrival intervals. MTP may emit multiple tokens together; these are not per-token TBT.
- Per-lane TTFT depends on admission order; a lane-by-lane change is not an isolated compute speed comparison.
- Actual process/library/config identity is established by frozen runner evidence, not a live re-query during offline audit.
