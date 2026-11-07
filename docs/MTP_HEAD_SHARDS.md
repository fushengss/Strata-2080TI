# Experimental MTP draft-head row sharding

`STRATA_MTP_HEAD_GPU=D1,D2` places two vocabulary-row shards of the MTP draft head on two other GPUs. It is
**off by default**. The draft layer's body, K/V state and probability calculation stay on the original MTP GPU.
This setting is independent of `STRATA_DECODE_HEAD_GPU`, which controls the main verifier head.

The supported weights are a plain, unpacked **Q6_K draft-vocabulary subset**. By itself this setting only changes
eligible greedy solo `--pipeline-windows 2` chains in a layer split. A separate experimental piped-slot option is
described below; prefill and ordinary draft graphs stay intact. See [multi-GPU settings](MULTI_GPU.md) for the
existing layer split and PW2 requirements.

## Selecting the targets

For example, with the drafter on visible CUDA device 7:

```sh
STRATA_MTP_HEAD_GPU=3,5
```

The values are CUDA device indices as seen by the process, after `CUDA_VISIBLE_DEVICES`. Both targets must be
distinct, belong to the configured layer split and differ from the drafter's device. `2,4` is also valid when
those devices meet the same requirements. Selecting devices already used by the main head may introduce
contention; the best pair must be measured on the actual split and workload.

Unset, empty or `-1` disables the feature. A single value such as `0` is not a disable switch. Malformed values,
missing PW2/MTP support, invalid target devices, an unsupported head layout, or a failed allocation/capture
cause a clear startup error when the option was requested. The engine does not silently claim that an
unsupported requested configuration is active.

The supported subset must be owned by the main drafter, contain at least 256 rows and use the original Q6_K
layout. Full-vocabulary heads, direct initialization from borrowed slot heads, `--mtp-q4` head copies, `STRATA_Q6_PACKED=1` and other
weight types/layouts are rejected. Each target owns only its row shard; the original source head remains
available for the ordinary paths.

## Request eligibility and fallback

With `STRATA_MTP_HEAD_GPU` alone, only the solo PW2 path enables the shards for a request, and only when it is greedy with no history
penalty, logit bias or active coupled draft sampling. Sampled requests, requests with penalties/bias, prefill,
serial drafts and batch execution retain their original graphs. The chain entry also checks coupled sampling
internally before selecting the split graphs.

The startup log identifies the two targets and the source. At the end of a PW2 request,
`strata MTP head: CUDA3+5, N draft steps offloaded in this request` reports the number of actually queued split
steps. A zero count is expected for a request that takes a fallback path. No MTP head shards, mapped buffers,
target streams, events or split graphs are allocated or captured when the feature is disabled.

## Preserving the draft result

The round and each chain step have a separate body and tail in the opt-in path. Their order is:

1. The original source body produces the same Q8_1 activation, including its original FP16 scale and sum.
2. A source event orders its exact bytes through portable pinned memory to two private target streams.
3. Each target runs the original single-row MMVQ over its contiguous vocabulary rows. The split starts at a
   multiple of 128 rows, preserving the original row/tile phase.
4. Both targets write their complete logits into disjoint ranges of mapped memory. Separate target events
   order a source-stream copy of all logits, in the original subset order.
5. The original source tail runs the original argmax or greedy sampler, `row_top_prob` or `row_top_prob_split`,
   subset-to-global vocabulary mapping, `mtp_select`, and teacher-forcing update.

No target-side softmax or probability reduction is used. In particular, shard-local probabilities are never
combined: the original full-row probability reduction determines `chain_prob`, preserving the draft admission
decision. Odd row counts use exact 32-bit word copies; they do not pad or read/write beyond the real logit count.

The entire chain is queued in one host call, without polling after each head. Every step has distinct producer
and completion events. The next body/activation write follows the previous source tail, so the shared pinned
buffers cannot be reused before both target heads and the source logit consumer have finished. Normal request
drain/cancellation waits for the source stream, which includes all target waits and tails. The helper's drain
and destruction also wait for source consumption before freeing buffers or events.

The implementation is in [mtp.cpp](../src/core/mtp.cpp) and [mtp_head.cpp](../src/core/mtp_head.cpp), with the
low-level interface in [mtp_head.hpp](../include/strata/core/mtp_head.hpp).

## Experimental piped-slot workspaces

`STRATA_PIPED_MTP_HEAD=1` additionally uses the already initialized solo head bank for piped MTP slot chains.
It is off by default and requires an active `STRATA_BATCH_MTP=1`, `STRATA_PIPED_MTP=1` route with
`STRATA_BATCH_SELF_COMMIT=0`, plus the existing two-target `STRATA_MTP_HEAD_GPU` setting. It does not choose new
targets or enable a missing solo head configuration. An unsupported requested combination fails at startup.

The two read-only weight shards are shared through a reference-counted bank; they are not copied once per
slot. Each slot has its own portable mapped activation/logits, target scratch, streams, graph executables,
and all eight steps' events. Equal step indices in concurrent slots therefore refer to different buffers and
events. A workspace drains its source consumer and both target streams before releasing resources and its
bank reference. Resetting the original helper leaves the other workspaces' weights alive. This does not change
the older drafter lifetime rule: the main `MtpDrafter` must still outlive clones borrowing its dense weights,
experts and subset map.

Slot proposals are greedy even when the target request samples or uses penalties/bias. These slot chains can
use the split head; coupled draft sampling is still excluded at chain entry. Admission's first draft, prefill,
ordinary drafts and target verification remain unchanged. With the new flag off, slot initialization keeps
its original ordering. With it on, chain capture waits until the existing solo head bank is ready.

Startup reports the number of private workspaces, both targets, the source and one shared weight bank. Each
completed piped batch then reports `strata piped MTP head: slot B, CUDA3+5, N draft steps offloaded in this batch`.
These per-slot deltas, rather than the configuration alone, establish whether the path ran.

On 2026-11-07, this candidate built with CUDA 13 / sm75 on the same 8×RTX 2080 Ti 22 GiB host. The additional
eight-workspace test passed 1416 cases and 150519384 complete logits bitwise, including probability bits and
mapped tokens. The original 144 cases and repeated-chain checks also passed. The command was:

```sh
engine/mtp_head_test --gguf MODEL.gguf --draft-vocab draft_vocab.bin --source 7 --target 3 --target2 5 --shared-slots 8
```

The test includes original-owner reset, partial-chain drain/reuse and one workspace's reset/rebind while
peers have queued work. Its log is `/srv/bench100k-20261107-concurrency-mtp/head-build/parity.log` on the test
host. Engine SHA256: `c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6`.
The end-to-end measurements below followed these checks. The piped-slot option remains off by default in code;
it was enabled on this host with deployment verified at about 21:37 Asia/Taipei on 2026-11-07. HIP has not been
built. Metal checks cover syntax and the unavailable stub.

### Piped-slot GPU measurements, 2026-11-07

The same 8×RTX 2080 Ti host used Q5_K_M, six layers per GPU, eight slots/groups, all experts resident and the
original 3.80 GiB prompt buffers per card. Both arms kept batch/piped MTP enabled, self-commit disabled, main-head
targets 2/4 and MTP-head targets 3/5. Only `STRATA_PIPED_MTP_HEAD` changed. Both used engine `c9305b08…` above and
server SHA256 `40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`, with the default-off
`reasoning_budget_engine_cap` explicitly enabled in both arms.

| Workload and order | Head off | Head on | API throughput change |
|---|---:|---:|---:|
| Eight short prompts, off→on | 248.752 tokens/s | 264.296 tokens/s | +6.25% |
| Eight short prompts, on→off | 256.544 tokens/s | 272.790 tokens/s | +6.33% |
| Four residents with two cold joins, one off→on pair | 106.426 tokens/s | 106.153 tokens/s | -0.26% |
| Four cold 100K prompts, one off→on pair | 26.2246 tokens/s | 26.3936 tokens/s | +0.64% |

Each eight-stream run had 8192 complete output tokens from 66/68-token prompts, four sampled and four greedy.
Across the two rounds, total tokens divided by total time increased by 6.29%. The first round ran budget checks
before prefill/C8; the reverse round started with prefill, with other tests between the rounds. This is a repeated
directional observation, not a contiguous ABBA with identical startup and thermal history.

The mixed four-resident workload had 8793-token contexts, cold joins of 8793 and 52080 tokens, and 8320 completed
output tokens. Its maximum mixed-phase SSE gap remained 10.433 s off versus 10.441 s on. It showed no whole-run
throughput improvement. The separate four-way long test used actual 101147–101148-token prompts, explicit zero
cached tokens, two sampled/two greedy requests, thinking disabled and 512 output tokens each. All four streams
had a common output interval of 7.237/7.264 seconds off/on. Whole time was 78.095/77.595 seconds. The +0.64% result
is one pair, without evidence of repeatability; per-lane TTFT also changed with admission order.

All rates above include admission/prefill and final drain. SSE arrival gaps are not per-token TBT. The complete
request/output/channel hashes and usage matched in the two C8 rounds, mixed C4 and long C4 comparisons. The
budget suite passed nine complete requests plus a cancellation shared-prefix/recovery check. With head sharing
disabled, the new executable also matched the older MTP-on executable's fixed-request outputs and usage; that
cross-run compatibility check does not compare performance.

Each round also had three cold and three warm solo 100K prefill pairs, with 256 output tokens. Their six full
outputs, usage and exact cache reuse matched within and across rounds. The cold/warm read medians were:

| Order | Cold off/on | Cold change | Warm off/on | Warm change |
|---|---:|---:|---:|---:|
| off→on | 16012 / 16294 ms | +1.76% | 2186 / 2056 ms | -5.95% |
| on→off | 16498 / 16442 ms | -0.34% | 2206 / 2179 ms | -1.22% |

Cold changes had opposite signs. The descriptive pooled cold median changed by +0.70%, while the median of six
fixture-paired relative changes was -0.22%; startup/suite order and the small sample affect these summaries.
All solo prefill logs recorded zero piped-head steps. Prefill preservation remains a deployment requirement;
these measurements do not guarantee zero regression for other conditions.

The restored service was verified at 21:37:55 Asia/Taipei on 2026-11-07 with engine `c9305b08…`, server `40c5e9df…`,
`STRATA_BATCH_MTP=1`, `STRATA_PIPED_MTP=1`, `STRATA_BATCH_SELF_COMMIT=0`, `STRATA_PIPED_MTP_HEAD=1` and
`reasoning_budget_engine_cap=true`. Fresh checks confirmed eight slots, all experts resident, the original
3.80 GiB prompt buffers on every card and eight private head workspaces on targets 3/5. Main-head targets stay
2/4. The separately tested prompt-read guard was rejected and remains absent/default-off.

Four synthetic smoke requests completed 128 output tokens each. A later, separate live-route check observed
4419 actual piped-head steps and passed; it includes resumed test traffic, so these steps are not attributed
solely to the smoke requests. The original smoke artifact is retained even though its short observation window
had not yet seen a completed shared-batch summary. Full deployment and verification receipts are listed in
[the concurrency deployment record](CONCURRENCY_STABILITY.md#restored-service-and-deployment-verification).

Evidence is under `bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/` in the benchmark checkout:
`head-pilot-analysis/`, `head-reverse-analysis/` and `head-long-analysis/`, with corresponding raw-data directories.
The reports preserve per-request checks, input hashes, actual process/library/flag evidence and resource gates.
See [concurrency measurements](CONCURRENCY_STABILITY.md) for the separate budget/batch MTP comparison and the
remaining long-prompt stall.

## Validation status, 2026-11-07

CUDA 13 / sm75 built successfully on the 8×RTX 2080 Ti 22 GiB machine. The low-level test used source CUDA7 and
targets CUDA3/5, with peer access unavailable and never enabled. It gathered the actual Q6_K draft subset from
the model: 106299 rows × 2560 columns, split into 53120 and 53179 rows.

The [manual parity test](../tests/core/mtp_head_test.cpp) passed:

- 144 cases, comparing 15307056 complete logits bitwise, plus the original probability bits and exact
  subset-to-global token mapping. This covers both argmax implementations and both probability implementations.
- 128 outputs across 16 repeated eight-step chains, queued without a host synchronization between steps,
  exercising per-step events and shared-buffer reuse.
- Default/unsupported helper contracts. Request-level fallback remains a separate engine integration check.

The log is `/srv/bench100k-20261107-mtp-head-build/mtp2-parity.log` on the test host. A manual rerun, with the
model and its matching draft-vocabulary file available and GPUs reserved for testing, uses:

```sh
engine/mtp_head_test --gguf MODEL.gguf --draft-vocab draft_vocab.bin --source 7 --target 3 --target2 5
```

HIP and SYCL have not been built or tested for this feature. Metal has only local C++ syntax checks for the
shared drafter and an explicit unavailable stub; there is no full Metal build or device validation.

## Measured 100k result

On 2026-11-07, the same 8-GPU host ran the actual Q5_K_M model with all experts resident, six layers per GPU,
eight slots, groups=8 and PW2. The existing main head stayed on GPUs 2/4. The MTP setting alone alternated
off/on/on/off, with three fixed code prompts and three prose prompts, each around 101k tokens, tested cold
and warm with 768 output tokens. All four launches used the same candidate executable.

| Workload | Cold decode off/on, token/s | Cold read off/on, seconds | Matched cold/warm decode gain |
|---|---:|---:|---:|
| Code | 143.10 / 146.50 | 16.405 / 16.352 | +2.276% |
| Prose | 81.75 / 83.20 | 16.6275 / 16.5505 | +1.681% |

Cold rates and read times are pooled medians. The gain is computed separately: for each fixed cold/warm
case, divide the two on-run mean decode rates by the two off-run means, then take the median of six case gains.
All six cases improved in each workload. The initial-to-final baseline drift median was -0.067% for code and
0% for prose. These are measurements of these fixed workloads, not a guarantee for other prompts or GPUs.

The 92 main requests had matching corresponding outputs and token counts. Four integration suites completed
44 requests plus four cancellation scenarios; all six pairs of suites had exact completed outputs. The
108 compared API logprob values covered one-token admission only. Sampled requests, penalties, bias and
concurrent work exercised the unchanged fallback paths. All required routing checks passed, with no observed
CUDA or API failure.

Cold prefill medians changed by -0.323% and -0.463%. Warm and append cases had no consistent regression:
matched read-time changes ranged from -5.265% to +1.208% for warm requests and -0.316% to +0.776% for append.
Each matching request reused the same number of tokens across all four arms. The prefill graph, weights,
K/V/GDN state, placement, buffers and batch slots were retained. Small run-to-run variation remains visible.

Raw evidence and the frozen evaluation are under `/srv/bench100k-20261107-mtp-head/abba-final` on the test host.
The candidate SHA256 is `113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f`.
Original-binary default-off compatibility and the deployment receipt are recorded separately in that task's
benchmark directory; the ABBA result alone does not establish those checks.
