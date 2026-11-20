BYIELD ADDITION — independent v2 package, frozen v1 unchanged

Run this package's runner_interleave.py --suites byield --arms off,on with the same
explicit engine/library/server/config SHA arguments documented below. No external
calibration file is required for byield: its independent long prompt is count-only
calibrated to100000±2 on each arm, then final OpenAI usage must exactly match.
Request and complete output hashes/counts must match across arms.

Example:
python3 byield-harness-v2/runner_interleave.py --execute --arms off,on --suites byield --config /srv/Strata/strata-orca-q5_k_m.57.json --exe CANDIDATE_PATH --engine-sha256 CANDIDATE_SHA --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd --out NEW_BYIELD_OUTPUT_DIRECTORY

C4 finite peers (4096 outputs each), independent cold100k (128outputs), and short
~100-token prompt (64outputs). Candidate dispatches the short ordinary chat only
after observing the long native reader begin without end, before long first output.
Baseline dispatches after numeric telemetry reports that long prompt reading;
it is an output-equivalence control and does not claim the new BYIELD path.

Server automatically asks BYIELD when a waiting prompt is <=half of the reading
prompt (serve/server.py _shorter_waiting and admission ask). No special API,
configuration, raw engine stdin, or production request cancellation is needed.

Candidate must show: the observed read end has yield_slot>=0 and an interior
stopped_at; actual stderr gives-way park at that token/slot; a subsequent reader
starts at the parked token, retaining the same chunk and segment endpoint;
read begin/end/park/resume chronological ordering; the short visible answer starts
before the long answer. All six responses must end normally with exact max output
counts, answer bytes and no reasoning/tools, zero cold cache, and valid usage.
No reader-open or park/resume proof means qualification fails, not a claimed pass.

The standard admission smoke's BSTOP cancellation does not cover this branch.
BYIELD exercises stopped reader join/drain, partial slot KV/PLE state copy, and
resumption of the same prefill plan. This additional qualification is recommended
before deployment; it is not a throughput benchmark. Approximately1–2minutes per
arm plus model load are expected, not guaranteed.

CPU checks: byield_smoke.py --self-test and byield_offline_tests.py. The latter
uses the full client/control path with in-memory HTTP only, including off/on
output equality and corrupted-output rejection. Negative checks reject missing
park, yield flag without park, wrong resume location/extent, wrong log order,
short answered after long, missing open observation, and CUDA errors.

Native decode_rows_before_done counts emitted native BT output tokens (including
termination), not API visible-content tokens or only accepted MTP drafts. It is a
conservative in-read lower bound excluding final drain and ambiguous boundary work.

--- Inherited v1 harness documentation follows ---

Prefill interleave benchmark harness — 2026-11-08

All entry points default to dry-run. No script stops production :8080. The isolated
runner requires :8080 stopped, :8092 free, and no GPU compute process. It creates
only its own :8092 server and private 0600 /dev/shm config, records process birth
identities, and cleans up only owned descendants. No API key is retained.

Variable: STRATA_PIPED_CONCURRENT_MTP=0 versus 1, SAME candidate binary. Fixed P1
STRATA_PIPED_TARGET_HEAD=1 and rejected sampler STRATA_PIPED_ASYNC_SAMPLE=0. Old
STRATA_PIPED_CONCURRENT / READ_PIECE / READ_DECODE_ONLY and concurrent minimum guards
must stay disabled. Original PP placement, own prompt buffers, expert residency,
head GPUs, slots, base config SHA, library SHA and server SHA are verified.

Run order:
1. Fresh candidate off/on --suites admission,correctness,c8 (first/repeated cold
   long read, cancel/reuse; seeded/mixed/penalty/bias correctness; saturated C8).
2. Fresh candidate off/on --suites primary,mixed100k,prefill. Calibrated fixture
   SHA must be supplied. Reverse/repeat arms after a plausible gain; do not infer
   a stable tail SLO from one finite cohort. Automatic comparison verifies output
   equality and qualification, not a performance/deployment decision.
3. Cross-binary --arms off --suites c8,prefill with the reviewed production binary
   and new candidate. cross_interleave_compatibility.py compares only behavior,
   requires current base/server/library parity, and does NOT compare performance.

Example (replace CANDIDATE_PATH / CANDIDATE_SHA / NEW_OUTPUT_DIRECTORY):
python3 harness/runner_interleave.py --execute --arms off,on --suites admission,correctness,c8 --config /srv/Strata/strata-orca-q5_k_m.57.json --exe CANDIDATE_PATH --engine-sha256 CANDIDATE_SHA --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd --out NEW_OUTPUT_DIRECTORY

For performance suites add:
--suites primary,mixed100k,prefill --calibration calibration.json --calibration-sha256 b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd

The frozen calibration has 11 count-only calibrated prompts, all independently
verified with a max1 OpenAI response usage: exactly 10,000 / 60,000 / 100,000 prompt
tokens. The client recounts every prompt before timing and gates final OpenAI
usage against that exact frozen count. Count/template equality is never assumed.

primary: four finite 10,000-token resident prompts, 4096 output tokens each;
10,000- and 60,000-token independent cold injections with 64 outputs each.
mixed100k: four finite 100,000-token residents, 6144 output tokens each; one cold
100,000-token injection with64outputs. No time-driven refills or perf cancellations.
Both use thinking=false and verify visible answer bytes >0, reasoning/tool bytes=0,
normal length finish, complete usage and exact output budgets. Early EOS or peers
ending before recovery invalidates the cohort. Never silently alter only one arm.

admission: independent fixtures keep performance prompts cold. Four peers have
4096 outputs; first ~60k admission completes64outputs; another independent ~60k
admission is canceled before first output; the same canceled prompt is retried
normally (cache reuse is allowed/reported). Actual counts are obtained on each arm
without inference and exact normal request/output hashes and logical usage must
match. Candidate cancellation requires an open native reader observation. The
socket shutdown and observation are not atomic, and BSTOP may finish prefill before
cleanup; this suite claims cancellation/reuse safety, not immediate GPU abort.

prefill: three independent cold/warm pairs from the already-reviewed standalone
fixture, approximately101k actual tokens (historically101139; actual API+engine
counts reported, 95k–105k hard range). This is not mislabeled as exactly100000.
Greedy output equality, API counts, native serial read_ms, explicit reuse and idle
isolation are gated. Its prompt generator stays unchanged for cross-version
behavioral comparison with earlier evidence; the new mixed100k is exactly100000.

Measurements:
- Every baseline/mixed/recovery phase reports output, visible answer, reasoning,
  and tool channels; per-lane chunk-gap p95/p99/max, phase-clipped max silence,
  any/all lane time stalled >1s and >5s, byte progress and byte Jain fairness.
- SSE chunks are not token boundaries. Never report these as per-token TBT.
- Final usage gives total completion tokens / first request start to final request
  end, including admissions and natural drain. Native piped rows/s includes busy
  admission blocks and is not accepted-token goodput.
- Injected TTFT and native prompt_ms are bound to the specific request. TTFT
  includes queue/admission/first decode; prompt_ms is the engine prompt stage.
- Each injection has a separate log slice with reader begin/end/fallback numeric
  evidence. Reader range usually ends n-1; turn/checkpoint boundaries may split
  it. Do not require begin.tokens == API prompt_tokens. A large segment >=90% of
  the counted prompt, bounded within [0,n-1], with >=4 active peers and positive
  strict in-read decode rows is required for the candidate long injection.
- decode_rows_before_done is a conservative native lower bound excluding end
  drain and ambiguous boundary pumps, not an accepted-output count. No legacy
  "meanwhile" log establishes concurrent progress.
- Reader read_ms is reader-wall duration of sp.run, not isolated GPU time; baseline
  has no reader thread. No invented pure-decode rate by subtracting overlaps.
- GPU CSV is500ms activity telemetry, not SM saturation; root analyzes it separately
  against phase begin_epoch/end_epoch.

Local CPU tests: python3 harness/offline_tests.py; plus --self-test on runner,
stream_client, calibrate_fixtures, prefill_interleave_check, c8_check,
budget_correctness, long_concurrency_check, and direct stall_metrics.py. correctness.py
has no --self-test; its unchanged dependency is exercised by the C8/budget/long tests.
No real HTTP socket/GPU was used by offline tests. See offline-checks.json.

Artifacts retain only synthetic hashes, numeric SSE timestamps/byte counts, safe
config/engine evidence and bounded status fields. correctness.py additionally
retains only its own explicitly synthetic outputs for legal-prefix cancellation
checks. Existing old harness and user source/docs were not modified.
