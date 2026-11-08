# Piped decode measurements on eight RTX 2080 Ti cards

These measurements were made on 10.100.3.3 on 2026-11-07: eight RTX 2080 Ti cards
with 22 GiB each, two Xeon E5-2650 v4 CPUs, CUDA 13, no GPU peer-to-peer access,
and the orcarouter Qwen3.8-Flash-Next-Uncensored Q5_K_M model. Experts were resident
in VRAM. Eight service slots used eight pipeline groups and the layer split
`6,12,18,24,30,36,42`. Each card retained its own 3.80 GiB prefill buffers.

The driver accepted a 2000 MHz minimum/maximum GPU clock request on all eight
cards. The power limit stayed at 250 W per card. Actual clocks remained subject
to the power limit; a clock request is not a measurement of sustained frequency.

## Concurrent target head

`STRATA_PIPED_TARGET_HEAD=1` uses the existing GPU 2/4 target-head weights for the
deferred piped batch route. Full logits return to GPU 7 before the original slot
sampler. The existing MTP draft head remains on GPU 3/5. Prefill placement and
ordinary single-request decoding retain their previous paths.

The tested P1 source is `48f9d6d2d6fbf9487a316264e14d63a792427440`; the frozen
engine SHA256 is
`8b27e040d26b51a67cfef4d7c7154fa8cc83f6d0a1cf1b1826f9eb0295bd30ab`.
The same binary was tested with the switch off/on, then on/off.

| Measurement | Off | On |
|---|---:|---:|
| First C8 run, completed API tokens/s | 285.90 | 300.12 |
| Reverse C8 run, completed API tokens/s | 271.88 | 284.40 |
| 101,139-token cold prefill, median read time | 16.481 s | 16.453 s |
| Warm prefix restore/read, median time | 2.263 s | 2.187 s |
| Cold single-request decode, median | 122.8 tokens/s | 122.4 tokens/s |
| Four-stream mixed workload, complete API tokens/s | 105.69 | 105.53 |
| Four cold 100k requests, complete API tokens/s | 26.20 | 25.86 |

C8 was four sampled and four greedy requests, each producing 1024 tokens. API
rates include admission, first-token latency and drain. Its measured improvement
was 4.97% and 4.61% in the two run orders. Absolute rates from different run
orders should not be subtracted to estimate an additional improvement.

Cold/warm prefill each had three samples and 256 output tokens. Their medians
did not regress in the matched test. This does not establish a tail-latency SLO
or a long-running single-request decode speedup. The four cold 100k requests
include all four admissions and only 512 output tokens each.

The mixed test had four resident streams and two injected prompts. The injected
prompts were actually 8794 and 52081 tokens, below the nominal 10k/60k fixture
tolerance. Both arms used identical requests. Overall throughput was effectively
unchanged, and the longest mixed-phase output silence remained about 10.4 seconds.
The target-head change did not solve long-prefill admission pauses.

In the first C8 run's common output interval, GPU 7 activity fell from 96.89% to
80.46%; GPU 2/4 rose to about 75%. Mean GPU 7 SM clock rose from 1671 to 1852 MHz.
The eight-card mean activity rose from 64.46% to 69.89%. These are NVML activity
samples, not SM occupancy, FLOPS or memory-bandwidth utilization.

The GPU head test compared 111,247,360 logits bit for bit across 128 base windows,
plus 16 sharded and 36 full-return windows covering sampled/greedy selection,
penalty/bias, nonfinite handling and odd shard ranges. Full API verification covered budgets, cancellation/recovery,
mixed requests and long context. Normally completed outputs and logical usage
matched exactly; deliberately cancelled requests used nonempty common prefixes.

## Asynchronous slot sampling

The independently tested P2 source is `c1b2f0f0c005a8b03cb38e86f4dd5ed719aad8ec`.
With P1 fixed on, only `STRATA_PIPED_ASYNC_SAMPLE` changed. C8 API throughput changed
by -0.18%, mixed API throughput by +0.22%, and long-context native busy-block
throughput by -0.14%. These measurements do not demonstrate a useful gain, so
the selected configuration keeps this switch off.

The full P2 comparison passed 46 complete request comparisons and two intentional
cancellation prefix comparisons. Requests actually sent history windows of 128,
32 and 257 tokens and completed 384, 448 and 512 tokens respectively. Asynchronous
route counters were positive at suite level; there was no independent per-history
GPU copy counter. Cold/warm prefill medians rose by 1.17% and
0.73%, with overlapping ranges from three samples per arm; that alone does not
establish a prefill regression.

## Remaining work and records

MTP observation is a diagnostic switch, not a batching implementation. With one
slot per pipeline group, submitted-but-unreaped drafts are not an unsubmitted
batch ready for a shared GEMM. Forming a useful batch requires additional graph
and scheduling work; waiting merely to collect jobs can add latency.

The short-prompt C8 diagnostic recorded 4673 completed draft steps. Weighted
source-stream means were 0.620737 ms for the body, 0.319316 ms for head return and
0.030594 ms for the tail. They include contention/dependencies and exclude the
initial residual handoff. They do not measure pure kernel active time or a share
of whole-system elapsed time. Pump samples showed at most one submitted unreaped
chain. Repeated draft-not-ready checks did occur; their count is not waiting time.
This diagnostic did not test 100k contexts or measure observation overhead.

The selected P1 binary was installed at
`/srv/Strata-0157/engine/releases/piped-decode-20261107/strata`. On 2026-11-07 at
23:58 UTC+8, startup verification confirmed the exact binary/library, all eight
slots, resident experts and unchanged prefill buffers. Eight sampled requests
then completed 128 tokens each. Actual target-head and MTP-head route counters
were present. P2 stayed off and P3 was not deployed. The smoke verified health
and routing; it was not another throughput benchmark.

Raw matched runs, source/binary hashes, independent audit scripts, GPU samples
and deployment receipts are kept in
`bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/piped-target-head`
in the original project checkout, and under
`/srv/bench100k-20261107-piped-target-head` on the test host. Private configuration
backups remain on the host and are not included in the downloaded results.
