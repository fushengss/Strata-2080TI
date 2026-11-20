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
