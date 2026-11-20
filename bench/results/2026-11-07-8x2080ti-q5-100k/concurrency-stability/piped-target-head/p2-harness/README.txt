P2 asynchronous slot-sampling validation — independent from frozen P1 harness

Entry: runner_async_sample.py. Default dry run; no config read/network/GPU calls.
Only STRATA_PIPED_ASYNC_SAMPLE changes 0/1 between arms. --target-head 1 (default)
fixes P1 STRATA_PIPED_TARGET_HEAD=1 in both arms; --target-head 0 fixes it off.
Never compare arms with different --target-head settings as a P2-only result.
Decode bank2,4, MTP head3,5, shared piped MTP head, piped MTP, deferred commit,
prefill placement/buffers, expert residency and layer split stay fixed.

Default suites c8, arms off,on. Select --suites c8,correctness for the first fast
pass. Optional suites are c4mixed,prefill,budget,long, independently selectable.
C8 is 4 sampled + 4 greedy streams, 1024 tokens each. c4mixed short has 4 nominal
10k resident streams plus nominal10k/60k injected prefills. --profile full uses
100k residents and six injections. prefill is 3 cold + 3 warm approximately100k,
greedy256 outputs. long is 4 cold approximately100k, 2 sampled + 2 greedy,512 each.
Budget/cancel/reuse checks retain the existing bounded-thinking fixtures.

CPU/dry-run examples:
  python3 runner_async_sample.py --suites c8,correctness
  python3 runner_async_sample.py --target-head 0 --suites prefill,c4mixed
  python3 runner_async_sample.py --self-test --target-head 0

Execution requires --execute plus reviewed exact identities:
  --engine-sha256 HEX --library-sha256 HEX --server-sha256 HEX --config-sha256 HEX
and --exe CANDIDATE --server SERVER_PY --python PYTHON --config PRIVATE_CONFIG
    --out NEW_RESULT_DIRECTORY
No hashes are auto-trusted. The runner copies/fingerprints the harness and checks
the actual /proc engine, mapped library and environment. The operator owns the
maintenance window: existing production8080 must already be stopped, GPUs empty,
8092 unused. One owned model is loaded at a time. Original config is unchanged;
only owned descendants are terminated. Temporary 0600 /dev/shm configs are deleted.
No power limit, clocks, governors or production deployment is changed.

P2 evidence gates:
  On startup: exact sourceCUDA7/event-polled sampling/persistent-history marker.
  This marker follows successful pinned/device history-buffer allocation.
  Runtime: actual windows and sampled rows must be positive for C8/correctness/
  budget/long and sampled c4mixed. Pure greedy c4mixed without penalties/bias can
  legitimately report0; solo prefill has no piped runtime counter. Off has none.
  P1 startup/work counters must match the fixed --target-head setting in both arms.
  Verification/sampler work counters are not accepted output token counters.

History coverage is explicit in the existing correctness mixed-four cohort,
without extra requests: greedy+repeat penalty with window128, sampled+presence/
frequency penalties with window32, and mixed penalties with window257, alongside
the bias fixture. The report records these numerical parameters and verifies the
saved actual synthetic request bodies contain them unchanged. Engine source has
no independent per-history-row counter; do not describe this as one. Completed
output/usage parity and the async runtime marker provide the behavioral evidence.

Completion/cancellation contract:
  Every normally completed request requires exact request/output/reasoning hashes,
  finish status and logical prompt/completion/total usage across arms.
  Intentional cancellation compares the nonempty common content/reasoning prefix,
  verifies both stopped during generation, and checks the subsequent recovery
  output and logical usage against its control. Cancellation length/full-text hash
  equality is deliberately not required; earlier pump handling can stop sooner.

Runner performs offline matched-arm comparison after all arms release GPUs.
Standalone: python3 compare_async_sample.py --baseline OFF_ARM --candidate ON_ARM
             --out NEW_COMPARISON_DIRECTORY
All exact correctness gates must pass. Performance remains descriptive: inspect
prefill read_ms/TTFT spread, concurrent throughput, injected TTFT, chunk gaps,
fairness, stalls and recovery. Chunk timestamps are not token-level TBT.

Cross-binary safety check, without comparing performance:
  python3 cross_async_compatibility.py --reference PREVIOUS_REVIEWED_ARM
    --reference-engine-sha256 PREVIOUS_ENGINE_SHA --candidate NEW_P2_OFF_ARM
    --suites c8,prefill --out NEW_CROSS_DIRECTORY
The reference must have async off/absent and the same fixed P1 setting. Select
only c8 for the initial fast pass. P2 off/on agreement alone cannot exclude a
common refactor regression relative to the previous reviewed binary.

GPU CSV adds per-card utilization, memory activity, power/enforced cap, clocks,
temperature and SW-power/SW-thermal/HW-thermal/HW-power-brake flags. Use actual row
timestamps to align the common decode interval. Memory activity is not bandwidth.
Only CPU/fake-transport tests were run during local harness preparation.
