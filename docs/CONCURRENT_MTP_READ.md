# Concurrent prompt reads with piped MTP

`STRATA_PIPED_CONCURRENT_MTP=1` is an experimental, default-off extension of
the piped reader. A prompt runs on its existing prefill streams while the
host continues to pump decoding slot groups. The prefill layer split and
planned chunk geometry stay the same. A read is not divided into additional
decode-budget pieces. The measurements below describe the tested workloads;
they do not establish full GPU saturation or a gain for every workload.

This flag is sufficient to request the path; `STRATA_PIPED_CONCURRENT` and
`STRATA_PIPED_READ_PIECE` are not required. The existing
`STRATA_PIPED_CONCURRENT_MIN`, `_PRIO`, and `_UNPIN` controls still apply.
Verifier-owned streams request high priority unless `_PRIO=0`. An externally
supplied pipeline stream keeps its original priority: with `--pipeline-windows
2`, the main verifier graphs use those default-priority streams. MTP and output
head streams have their own creation rules. GPU stream priority does not
guarantee preemption or latency; solo and mixed prefill performance must both
be measured.

## Eligibility and lifetime

Each prompt part checks the live state after the normal pipeline drain:

- CUDA, deferred-commit piped MTP, and `STRATA_SPLIT_MTP_BATCH=1`.
- At least one active decoding slot. A solo prompt uses the existing path.
- Every prefill stage owns its prompt region, with no borrowed buffers,
  borrowed spill, active MTP room, or outstanding expert-cache loan.
- Every verifier stage reports all experts resident, the host residency
  table agrees, and the main drafter's expert weights remain resident.
- No pending tier change, asynchronous tier worker, KV growth, peer path,
  helper expert cache, remote expert work, or Foresight swap space.
- One text prompt. Images and co-reads use the existing path.
- A PLE stage with a native quantized key and no BF16 key uses the serial
  path: that key's quantization scratch belongs to the shared weight table.
- No prefill helper, DMA doorbell, helper source, or unknown source type.
- The router lookahead has no pending or running prediction. A staged file
  source is allowed only in this strict mode, when it is the same source
  for prefill and decode. Resident batch windows skip the expert dispatch
  that consumes source blobs and submits lookahead jobs. While the reader
  runs, `in_prompt_read` prevents adaptive rounds; with no pending changes
  or asynchronous tier, `apply_pending` cannot mutate the source. The file
  source's own I/O workers and prefill stagers retain their normal behavior.
  The legacy `STRATA_PIPED_CONCURRENT` source restriction is unchanged.

A failed check prints a fallback reason and uses the existing serial path.
Before starting the reader, the host reserves the drafter's input records,
the prompt upload rings, and the prompt MMQ contexts. A pinned reservation
failure also falls back; it does not start a concurrent read. The prompt
gets its own PLE fallback scratch, since the verifier uses the main session's
PLE scratch. The fallback PLE kernels and arithmetic remain unchanged.

These preparations do **not** reserve every internal MMQ/GEMM allocation.
First-use or larger-shape workspaces and weight conversions may still cause
device barriers. Cold first use, subsequent admissions, and small tails must
be validated separately.

## Why the prepared PLE flag is published early

An all-resident batch graph can still wait for the mapped PLE-ready flag.
Previously, the host set its `b_running_` flag only after launching the body,
save, and head work. A concurrent reader's allocation barrier could block
one of those CUDA calls while the body waited for PLE. The doorbell thread
saw `b_running_ == false`, so it could not release that wait.

Only during the new mode's external serving, and only for a resident PLE
stage, `stage_batch` publishes the ready flag after its actual gather and
flag reset, with the existing CPU/store fences, before any body launch.
`b_running_` is still published at its original point. Nonresident expert
waits and the default batch path are unchanged. Batch graphs do not use the
solo inter-stage mapped handoff wait. Head and draft event waits retain their
producer-before-consumer enqueue order.

The reader is joined on normal, failure, and yield exits before its stop
hook is removed. The doorbell owner then joins its thread and returns
external serving to the normal pump. `BSTOP` marks a slot to stop; it does
not promise to abort an in-progress prefill immediately. `BYIELD` retains
the existing chunk-boundary behavior.

## Runtime evidence

The startup line reports that the flag was requested, not that an eligible
read ran. Actual reads print paired `strata piped concurrent MTP: begin` and
`end` lines with a monotonic `read_id` within that engine process. A rejected
read prints `fallback ... reason=...`.

`strata resident read source` records the same `read_id`, whether the source
uses staging, the lookahead's idle observation at entry, and `dispatch_delta`.
The last field is the change in an atomic decode expert-pool call counter,
sampled before the final drain; it must be zero for the resident path. These
are decode dispatch calls, not the prompt source's own I/O or stager calls.

`begin` includes the token interval (`from`, `tokens`), planned chunk,
active-slot count, and owned/resident stage counts. Prompt parts can end at
checkpoints; the final prompt token is normally handled by the first decode
window. Do not equate one `tokens` field with the API's full prompt count.

`end.read_ms` measures the reader's `Prefill::run` wall time, before the final
decode drain. `decode_rows_before_done` is a conservative count of emitted
native `BT` output tokens, including an emitted termination token: pumping
starts only after the reader-start publication, and
only a completed pump observed before the reader's done publication is
counted. The ambiguous last pump and final drain are excluded. This count
proves output progress during the read; it is not verification rows, a count
of accepted drafts alone, API content tokens, or GPU occupancy. The old
“decoded meanwhile” line includes the drain and is not this
strict measure.

`read_ok`, `pump_ok`, `stopped_at`, `admit_stop_requested`, and `yield_slot`
describe their individual states. An `end` line alone does not prove that
the final drain, admission, or request completed successfully. Validation
must also check completed outputs and usage, cancellation and slot reuse,
absence of engine errors, solo 100k prefill, and mixed-stream output gaps.

## Measurements on 10.100.3.3, 2026-11-08

The first complete same-binary comparison used eight RTX 2080 Ti cards with
22 GiB each, two Xeon E5-2650 v4 CPUs, 377 GiB of RAM, CUDA 13, and the
orcarouter Q5_K_M model. Experts were resident on all eight cards. The layer
boundaries stayed at 6,12,18,24,30,36,42; int8 K/V kept 32768 tokens resident;
prompt chunks stayed at 8192. The existing target heads on GPUs 2/4 and MTP
heads on GPUs 3/5 stayed enabled. The requested clock target was 2000 MHz;
the 250 W power limits were unchanged, so this is not a claim that every card
held 2000 MHz.

The binary was built from `bf5a8e5b24b1973efbae747a109f5a970b25e09d`, SHA256
`45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6`.
Only `STRATA_PIPED_CONCURRENT_MTP` changed. This binary does not contain the
later overwrite experiment. Results are in
`bench/results/2026-11-08-prefill-interleave/performance-v3-h6-off-on1` in the
main checkout. Both arms completed all five planned suites and passed the
exact output, finish-reason and logical token-count comparisons.

| Workload and metric | Off | On |
| --- | ---: | ---: |
| Solo 101139-token cold prompt, median native prompt time, 3 observations | 16.451 s | 16.423 s |
| Same prompt with 101132 tokens reused, median native prompt time, 3 observations | 2.282 s | 2.213 s |
| C4 with 10k/60k cold injections, largest visible SSE gap in mixed phase | 10.938 s | 0.662 s |
| Same C4 case, 15231 actual output tokens / complete case time | 133.990 tok/s | 142.251 tok/s |
| 60k injection, native prompt time during that C4 case | 10.604 s | 11.372 s |
| Four 100k-context requests plus a cold 100k injection, largest mixed-phase visible SSE gap | 17.345 s | 2.547 s |
| Same 100k case, 24640 actual output tokens / complete case time | 128.825 tok/s | 128.195 tok/s |
| 100k injection, native prompt time during that C4 case | 16.453 s | 17.437 s |

Whole-case rates include admission and final drain; they are not pure decode
rates. Native prompt time includes cache parking, restore and other admission
work as well as reading the prompt. SSE gaps describe client-visible chunks,
not individual token timing. The maximum pause improved, but per-lane mixed
phase p95 chunk gaps increased from about 26 ms to 135–151 ms in the 10k/60k
case and from 35–38 ms to 207–250 ms in the 100k case. Solo prefill and prefill
under contention must therefore be reported separately. These are one pair's
results, with exact actual work matched.

A second comparison reversed the order, running on before off, with the same
selected binary and fixed engine configuration. Results are in
`bench/results/2026-11-08-prefill-interleave/core-qualification-v3-on-off2`.
Both arms completed all eight planned suites: solo prefill, the two mixed
workloads, correctness, C8, admission/cancellation, BYIELD, and reasoning-budget
continuation. The frozen harness differs from the first comparison; each
comparison matches work and configuration within its own pair.

| Workload and metric, second comparison | Off | On |
| --- | ---: | ---: |
| Solo 101139-token cold prompt, median native prompt time, 3 observations | 16.213 s | 16.365 s |
| Same prompt with 101132 tokens reused, median native prompt time, 3 observations | 2.158 s | 2.225 s |
| C4 with 10k/60k cold injections, largest mixed-phase visible SSE gap | 10.961 s | 0.659 s |
| Same C4 case, 15231 actual output tokens / complete case time | 134.281 tok/s | 137.585 tok/s |
| 60k injection, native prompt time during that C4 case | 10.645 s | 11.388 s |
| Four 100k-context requests plus a cold 100k injection, largest mixed-phase visible SSE gap | 17.134 s | 2.438 s |
| Same 100k case, 24640 actual output tokens / complete case time | 127.141 tok/s | 129.704 tok/s |
| 100k injection, native prompt time during that C4 case | 16.387 s | 17.286 s |

Across the two comparisons, the 10k/60k mixed case gained 2.46–6.17% in
whole-case throughput. Its largest mixed-phase visible gap fell from
10.938–10.961 s to 0.659–0.662 s. The 100k case changed by −0.49% to +2.02%
in whole-case throughput, so these runs do not show a repeatable throughput
gain for that workload. Its largest gap fell from 17.134–17.345 s to
2.438–2.547 s; multi-second admission pauses remain.

The mixed prefill cost repeated: native prompt time increased by 10.77–11.02%
for the 10k injection, 6.98–7.25% for 60k, and 5.48–5.98% for 100k. In the
second comparison, per-lane mixed-phase p95 visible chunk gaps increased from
about 26–27 ms to 135–138 ms in the 10k/60k case, and from 30–41 ms to
236–246 ms in the 100k case. The change trades long output interruptions for
continued, more widely spaced output while prefill shares the GPUs.

There are six solo observations per arm and cache condition across the two
comparisons. Cold native prompt times ranged from 14.480–16.898 s off and
14.493–16.958 s on; their combined medians were 16.332 s and 16.394 s. Warm
times ranged from 2.129–2.412 s off and 2.090–2.413 s on, with combined medians
of 2.256 s and 2.219 s. These small samples overlap and retain the same
cold/warm sequence; they do not establish stable tail latency or guarantee
unchanged prefill performance.

The second comparison passed exact output, finish-reason and logical usage
checks for 59 normal request pairs. Three intentional cancellations were
checked separately, including cancellation during an observed open reader
and successful reuse of the same request. The correctness fixtures exercised
greedy and sampled output, bias, penalties, and requested history windows of
32, 128 and 257 tokens. This is sent-parameter and output evidence, not an
independent GPU history-row counter.

The on arm produced 24 paired reader records. Every source record had zero
decode expert dispatches, every pump succeeded, and the only failed read was
the specifically observed intentional admission cancellation. The off arm
produced no concurrent-reader records; solo prefill launched none in either
arm. BYIELD stopped at token 8192, parked that partial state, resumed from
8192 with the same chunk plan, and let the short request produce visible
output before the long request. Four peers covered the transfer and recovery.
All four reasoning-budget continuations reread 19 tokens in each arm.
These results complete the core qualification. True conversation-incremental
and deployment health checks are separate from these eight suites.

### Cached agent turns

A separate complete on-then-off pair used the same engine and settings, with
the frozen `incremental-harness-v2`. Results are in
`bench/results/2026-11-08-prefill-interleave/incremental-v2-v3-on-off1`.
After a 100007-token seed completed, four different 10k-context requests
decoded while three new user messages were appended to the seed conversation.
Each turn included the preceding actual assistant response. Stateless counts
checked each full message list before submission; the two arms matched the
request hashes, actual replies, finish reasons, usage and cache counts.

The three prompts contained 100937, 101867 and 102752 tokens. They reused
100070, 101000 and 101885 tokens respectively: each read only 867 fresh tokens.
All four background streams visibly covered the baseline, mixed and recovery
phases. All eight request pairs passed the exact comparison. The short
assistant replies ended normally at 64, 19 and 12 actual output tokens; equal
actual work, rather than the requested maximum alone, determines the rate.

| Cached-turn workload and metric | Off | On |
| --- | ---: | ---: |
| Post-seed finite case, 15658 actual output tokens / elapsed time | 145.771 tok/s | 154.578 tok/s |
| Largest background-stream visible gap in the mixed phase | 3.451 s | 1.847 s |
| Largest interval without visible output from any of the four streams | 3.431 s | 1.817 s |
| First appended turn, time to first visible output | 2.454 s | 2.665 s |
| Second appended turn, time to first visible output | 2.799 s | 2.994 s |
| Third appended turn, time to first visible output | 3.132 s | 3.382 s |
| Per-stream mixed-phase p95 visible chunk gap | about 26 ms | 39–40 ms |

This single pair gained 6.04% in post-seed whole-case throughput. The rate
includes background request admission and final drain; it is not a pure
decode rate. The added turns took 0.19–0.25 s longer to produce their first
visible output, an increase of 6.9–8.6%. Native prompt time increased
7.6–9.4%, including admission and cache work. This is the measured tradeoff
accepted for concurrent agent workloads, not evidence that all latency
percentiles improve. One pair does not establish a repeatable gain for every
agent workload. The nominal 500 ms NVML samples averaged across eight cards in the
mixed phase rose from 30.48% to 35.63%; this activity measure does not show
full GPU saturation.

An unselected overwrite experiment was built separately from `e0a93a09`, SHA256
`92822929f1f60ecd80c64ce04fa47a44a6d789b0e4ca028ef5be5e39fecab20c`.
Five host/GPU snapshot tests passed. Diagnostic runs with `STRATA_PARK_TRACE`
showed no useful reduction in total K/V save time: for a matching 100000-token
park shape, resize fell from 889.90 ms to 0.96 ms while copy rose from
179.64 ms to 1083.36 ms. Complete park time was 1721.8 ms and 1839.2 ms.
These separate diagnostic runs are not a request-level performance A/B, but
they show why removing the clear alone does not justify enabling the option.
The experiment and its tests were removed from the working branch after this
measurement. Its patch and build/test records remain in the benchmark archive;
it is not part of the selected engine. The source changes retained for the
engine are the concurrent-reader changes from `bf5a8e5b`.

### Deployment on the measured host

The selected `45eaa612` binary and its tested `fb67bffc` adjacent library were
installed at
`/srv/Strata-0157/engine/releases/concurrent-prefill-20261108`.
Only the configured executable and `env.STRATA_PIPED_CONCURRENT_MTP=1`
changed. The model, layer boundaries, prompt buffers, chunk limits, head
routes, authentication and other serving settings stayed the same. The
original configuration was saved privately for rollback. The selected
binary, rather than a rebuild of the later experiment, is what was deployed.

The production service on port 8080 was restarted and reported eight serving
slots. A fresh startup-log cursor and actual process/library hashes verified
the new engine, all-resident experts and eight private prefill buffers.
The bounded production health check completed two synthetic streaming
requests and verified usage and visible output. Its compatible 4415-token
reader completed with `read_ok=1`, `pump_ok=1`, 133 conservative native output
tokens during the read, and zero decode expert dispatches. The peer visibly
continued through the injected request. Both owned connections and workers
finished, and the same production processes remained alive.

Deployment and health records are in
`bench/results/2026-11-08-prefill-interleave/deployment-v3-on1` and
`production-v3-health2`. An initial health preflight rejected an extra Python
bytecode-cache directory before issuing HTTP requests; the completed check
used the exact frozen harness copy from the successful incremental pair.
No validation gate or engine code changed for that correction. Native logs
have no HTTP request ID, so the fresh-window, unique compatible reader is
routing evidence with that attribution limit. The production health check
is not another throughput benchmark and does not prove full GPU saturation.
