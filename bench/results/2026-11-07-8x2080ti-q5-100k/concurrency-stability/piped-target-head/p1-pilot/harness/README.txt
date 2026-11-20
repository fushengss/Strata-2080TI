Piped target-head P1 validation harness

This directory is independent of all previous frozen harnesses. Default invocation
is a dry run: no configuration read, network request, GPU call or process mutation.
Only STRATA_PIPED_TARGET_HEAD differs between arms (0/1). Existing decode-head
bank 2,4, MTP-head 3,5, piped MTP, shared piped MTP head, batch groups, layer split
and prefill placement remain fixed. P2 is not implemented or varied here.

Runner: runner_target_head.py
Default: --arms off,on --suites c8 --port 8092
Optional: --arms on,off or off,on,on,off. One arm alone is supported but does not
produce an A/B conclusion. Only one model process is loaded at a time.

Selectable suites (comma separated):
  c8          8 x 1024; 4 sampled / 4 greedy, fixed seeds, nonempty output overlap,
              exact requested completion length and request-identity qualification.
              A/B compares exact output/reasoning hashes and logical token usage.
  c4mixed     Existing short mixed-admission workload: 4 resident streams, nominal
              10k contexts; injected nominal 10k + 60k prompts, baseline/recovery.
              --profile full selects 100k residents and six injections. This is
              intentionally optional; default short c4mixed is not a 100k test.
  prefill     3 cold + 3 warm approximately 100k prompts; greedy 256 output tokens.
              Exact matched engine read_ms, prompt/reuse counts, output hashes.
  correctness Existing 14-request fixtures: sampled, greedy, tool/reasoning/budget,
              cancel during four-stream overlap, recovery reuse/control identity.
  budget      Existing 10-request bounded-thinking/cancel/recovery correctness.
  long        Optional 4 cold approximately 100k contexts, 2 sampled / 2 greedy,
              512 output tokens each. Separate from C8 and solo prefill guards.

Examples that are safe to execute locally (dry runs):
  python3 runner_target_head.py
  python3 runner_target_head.py --suites c8,c4mixed
  python3 runner_target_head.py --suites prefill,correctness,budget,long

Execution requires explicit --execute and all four reviewed SHA256 bindings:
  --engine-sha256 HEX --library-sha256 HEX --server-sha256 HEX --config-sha256 HEX
plus --exe CANDIDATE_BINARY --server SERVER_PY --config PRIVATE_BASE_CONFIG
     --python SERVER_PYTHON --out NEW_RESULT_DIRECTORY
The library is CANDIDATE_BINARY's adjacent libstrata_kq.so.
There are no auto-trusted hashes. Reusing a reviewed candidate with a modified
binary/library/server/config fails before loading. Harness sources are frozen and
checked again before each arm/suite. The running /proc executable, mapped library
and actual STRATA_* flags are checked, not just paths in a JSON config.

The operator must first enter maintenance and stop production. Runner requires
zero GPU compute processes and no listeners on 8080 or 8092. It never stops an
existing service. It launches one owned localhost:8092 server, waits for readiness,
checks all eight slots, full expert residency and eight unchanged 3.80GiB own
prefill buffers, and terminates only its owned descendants after each arm.
Temporary authentication configs are 0600 in /dev/shm and deleted on exit. Original
config bytes must remain unchanged. The harness never restores/deploys production;
the operator owns that existing maintenance/restoration workflow.

P1 route evidence is mandatory: exact startup CUDA2+4/sourceCUDA7 marker, enabled
concurrent suites must emit positive target-head windows and verification rows;
disabled arms and solo prefill must emit none. Verification rows include draft
proposals and must never be interpreted as accepted output tokens. Shared MTP-head
3,5 counters must stay positive for concurrent suites in both P1 arms.

GPU telemetry records per-card utilization, memory-activity time, power/caps,
SM/memory clocks, temperature and SW-power / SW-thermal / HW-thermal / power-brake
flags every requested 500ms. Actual row timestamps, not requested cadence, define
measurement intervals. Memory activity is not bandwidth percentage.

After all arms finish and GPUs are released, runner automatically compares every
on arm to the first completed off arm using compare_target_head.py. A hash/usage,
fixed-configuration or validation mismatch fails the comparison. Standalone use:
  python3 compare_target_head.py --baseline OFF_ARM_DIRECTORY \
    --candidate ON_ARM_DIRECTORY --out NEW_COMPARISON_DIRECTORY
This is offline and can compare matching runs stored in separate result roots.

Before attributing a P1 gain, also check that the new binary's OFF arm preserves
the previous deployed c9305 engine. New off/on agreement alone cannot exclude a
shared-refactor regression. Cross-version correctness-only command:
  python3 cross_compatibility.py --reference OLD_HEAD_REVERSE_01_ON_DIRECTORY \
    --candidate NEW_TARGET_HEAD_OFF_DIRECTORY --suites c8,prefill \
    --out NEW_COMPATIBILITY_DIRECTORY
Select --suites c8 for the initial fast check before prefill exists. The old
reference is concurrency-stability/head-reverse/01-on, which has MTP head enabled
and the previous production binary. The comparator requires exact request/output
hashes, finish status and prompt/completion/total usage; all existing MTP/head
settings stay fixed. It intentionally produces no cross-version speed comparison.

Performance is descriptive, never an automatic deployment. Review cold/warm
read_ms and TTFT medians/spread, concurrent throughput, injection TTFT, per-stream
chunk gaps, fairness, stall fractions and recovery. Three prefill observations
per kind cannot establish a stable tail SLO. Byte/chunk arrival metrics are not
token-level TBT. Exact requested correctness gates must pass before accepting
speed gains. Repeat the opposite arm order when results are ambiguous.

Offline checks use only CPU/fake HTTP or existing synthetic artifacts; no GPU test
has been performed by this harness author. See offline-checks.json and the source
provenance file for input/output hashes. Current four explicit runtime hashes are
deliberately not filled in: root must bind the final compiled candidate.
