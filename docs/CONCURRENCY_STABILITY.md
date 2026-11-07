# Prompt reads while requests decode

This candidate targets long gaps in ongoing answers when another request reads a long prompt. It is opt-in and has
not been measured on a GPU. It does not change the model, sampling, prefill chunks or MTP implementation.

On a layer split with pipelined slot groups, the default prompt read runs to completion before those slots decode
again. `STRATA_BATCH_DECODE_SHARE` alone does not change that path. The existing `STRATA_PIPED_READ_PIECE` option
inserts decode time between pieces, but also splits a prompt when no request is decoding.

`STRATA_PIPED_READ_DECODE_ONLY=1` limits that existing piece policy to reads beside active decoding slots. It requires
`STRATA_PIPED_READ_PIECE` to be positive. With no active slots, the prompt follows the original whole-read call. If
the last slot finishes during a decode budget, the remaining prompt follows one whole-read call from that boundary.
The planned chunk size stays unchanged. The same check follows the lead-in budget before the first piece.

The guard is off by default. With it unset or zero, the existing piece policy and lead-in history are unchanged.
In guarded mode, an idle remainder, cancellation, give-way or failed read clears the previous read's lead-in history.
A short completed read does not inherit an earlier long piece's timing. A new request cannot enter a decoding slot
while the current prompt owns the admission control lines.

A configuration to **test**, not a measured recommendation:

```text
STRATA_PIPED_READ_PIECE=16384
STRATA_PIPED_READ_DECODE_ONLY=1
STRATA_PIPED_DECODE_MAX_MS=400
```

Keep the existing prefill options, including `STRATA_SPLIT_MTP_BATCH=1`, and the default decode share of 0.5. The piece
size is rounded to whole configured chunks. The decode budget is half the preceding piece's time, capped at 400 ms;
finishing groups still in flight can take longer than that budget. A read shorter than one piece still runs whole.
The new guard does not alter `STRATA_PIPED_CONCURRENT`: that separate experimental path takes precedence if enabled
and remains incompatible with `STRATA_SPLIT_MTP_BATCH=1`.

Reading and decoding still compete for time. Mixed-load prompt TTFT may increase. No claim of unchanged solo
prefill speed or faster concurrent decoding is made before measurement. A request arriving during an otherwise
solo read retains the original admission behavior; this guard does not add new preemption points to that read.

Validation should compare the same build with the guard disabled and enabled, preserving all other settings.
Use four ongoing streams and cold joins of 12K, 18K and 61K tokens. Record per-stream p95 and maximum nonempty SSE chunk gaps (token gaps only when actual token timestamps exist),
rates during prompt reads, aggregate tokens per second, joining-request TTFT and errors. Also test slots finishing
during a read, cancellation and back-to-back joins. Use fixed greedy prompts for output checks and a separate sampled
workload matching the application. Test solo cold 100K prefill separately: it must not regress before deployment.

The CPU policy test covers solo whole reads, mixed reads, merging the idle remainder, the unchanged legacy policy,
chunk rounding and lead-in history at early exits. It does not test CUDA scheduling, output equality or performance.
