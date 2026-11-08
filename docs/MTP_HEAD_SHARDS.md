# Experimental MTP draft-head row sharding

`STRATA_MTP_HEAD_GPU=D1,D2` places two vocabulary-row shards of the MTP draft head on two other GPUs. It is
**off by default**. The draft layer's body, K/V state and probability calculation stay on the original MTP GPU.
This setting is independent of `STRATA_DECODE_HEAD_GPU`, which controls the main verifier head.

The current implementation is limited to a plain, unpacked **Q6_K draft-vocabulary subset** and eligible greedy
solo `--pipeline-windows 2` chains in a layer split. It does not replace the ordinary draft graphs or the batch
and prefill paths. See [multi-GPU settings](MULTI_GPU.md) for the existing layer split and PW2 requirements.

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
layout. Full-vocabulary heads, shared slot-drafter heads, `--mtp-q4` head copies, `STRATA_Q6_PACKED=1` and other
weight types/layouts are rejected. Each target owns only its row shard; the original source head remains
available for the ordinary paths.

## Request eligibility and fallback

Only the solo PW2 path enables the shards for a request, and only when the request is greedy with no history
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
