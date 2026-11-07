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

## Bound native generation at a known thinking budget

`reasoning_budget_engine_cap` is a separate, default-off server option. It does not enable the prompt-read policy
above or change the batch/piped MTP switches. As of 2026-11-07, its CPU tests and the GPU budget-boundary checks below
have passed. The concurrency comparisons enabled this option in both arms. It has not been deployed to the normal
service. The frozen candidate `serve/server.py` SHA256 is
`40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`.

The triggering observation was a thinking-budget continuation during sampled C4: two requests with batch MTP
enabled reused only their initial 8793-token prompt and read 1554 tokens again. The control reused 10329 tokens and
read 18. Reasoning text hashes matched; those two answer text hashes differed. Speculative tokens emitted before
the frontend's asynchronous `BSTOP` can leave the slot beyond the prefix that the frontend retains. The next
prompt inserts a thinking wrap-up at that earlier boundary, so the longer recurrent state cannot simply be
truncated. The GPU checks below exercise this observed boundary with fixed synthetic requests.

Enable it in the run config for a new server process:

```json
{"reasoning_budget_engine_cap": true}
```

It is also a validated boolean in the existing loaded `POST /settings` defaults:

```json
{"defaults": {"reasoning_budget_engine_cap": true}}
```

That endpoint replaces the shared defaults object; include any other shared defaults that should remain. A shared
`false` overrides an enabled run-config value. Clearing the shared override returns to the run-config value. Each
request snapshots the setting; changing it does not alter a request already running. Existing endpoint
authentication and own-page restrictions still apply.

The option applies only to `StrataEngine`, excluding the all-cards split backend (`split_engine`) and the separate
multi-request engine protocol (`multi`). Other backends retain their existing behavior. Requests with an effective
nonzero presence/frequency penalty or a repetition penalty other than 1 also retain the original path: extra
native segments would otherwise move the penalty-history start. The option does not introduce a new penalty
protocol. Fixed-seed comparison requests should supply an explicit positive seed; without one, a new native
generation can choose a new seed, so additional segments are not claimed to preserve the same random stream.

While the parser is in reasoning and a thinking budget is known, each native `max_new` is bounded by the remaining
budget and the remaining overall output allowance. For example, the existing answer-room policy can give a
2048-token response a 1536-token thinking budget. At the boundary, a partial tag or UTF-8 character advances through
one-token native segments until the existing parser-clean and decoder-clean checks allow the normal wrap-up.
If the model closes its thinking naturally before the cap, reaching that artificial cap continues the answer with
the remaining overall allowance. Actual EOS, stop strings, cancellation, failures and loop/repetition guards do
not trigger this artificial-length continuation. Forced tool openings and the existing wrap-up text retain their
previous order and token accounting.

When a clean wrap-up coincides with the native cap, the frontend consumes the terminating `DONE`/`BDONE` on the
request's own thread before closing the iterator. This retains the segment's generated-token and decode-time
statistics instead of leaving them to a background stop drain. A naturally capped slot may hold the prompt plus
all generated tokens except the final token, so a continuation reading 19 tokens where the asynchronous-stop
control read 18 can be correct. GPU validation must check the actual boundary, outputs and state reuse rather
than require the old read count exactly.

CPU validation completed locally: 16 new tests passed, plus 74 existing reasoning-budget, close-retry, forced-tool,
shared-settings, reasoning-loop and request-accounting tests. The new suite covers the automatic 1536/manual 64
budgets, early natural thinking closure, split tags and UTF-8, EOS exactly at the cap, stop/cancel/error paths,
forced openings, penalty/backend fallbacks and loaded settings. A local fake subprocess exercises the real
`StrataEngine` BT/BDONE reader and verifies both segment accounting and complete slot release. These tests do not
run model weights or establish GPU numerical parity or performance.

Commands used from `/Users/fu/project/HAI/Strata-decode-head`:

```sh
uv run --offline --no-project --with jinja2 --with regex python -m unittest serve.test_reasoning_budget_engine_cap -q
uv run --offline --no-project --with jinja2 --with regex python -m unittest \
  serve.test_server.ThinkingBudget serve.test_server.ReasoningCloseRetry \
  serve.test_server.ForcedToolChoice serve.test_server.SharedSettings \
  serve.test_reasoning_loop_recovery serve.test_request_accounting -q
```

The packages were first prepared in a temporary uv environment using the same command without `--offline`; no
repository dependency files were changed. The resolved environment was Python 3.13.13, Jinja2 3.1.6 and regex
2026.11.7. Its interpreter was `/Users/fu/.cache/uv/builds-v0/.tmpdgJCbU/bin/python`; uv may recreate this temporary
path. The existing-regression log is `/tmp/strata-reasoning-budget-regression.log` (74 tests, 27.826 seconds).
The independent new-suite runtime was 0.402 seconds. For a fresh machine, prepare those dependency versions before
using `--offline`, or use the interpreter of an existing environment containing them.

## GPU comparisons on eight RTX 2080 Ti cards

These 2026-11-07 measurements used Qwen3.8-Flash-Next-Uncensored Q5_K_M on eight RTX 2080 Ti cards, six layers per
card, eight slots and eight pipeline groups. All experts remained resident, and each card retained its original
3.80 GiB prompt buffers. Main-head shards stayed on GPUs 2/4 and the solo MTP-head shards on GPUs 3/5. These are
isolated synthetic tests, with one sequential off/on pair per comparison; they do not establish a production tail
latency guarantee. The prompt-read guard described above remains unmeasured and undeployed.

The v4 budget comparison used engine SHA256
`113a4ea9bf2101a9ad9a3449ae611f8a21f892cc62293fa9a779d302dc2df96f` and the server candidate identified above.
Both arms had `reasoning_budget_engine_cap=true`. The on arm enabled `STRATA_BATCH_MTP=1` and `STRATA_PIPED_MTP=1`
and disabled `STRATA_BATCH_SELF_COMMIT`; the off arm used 0, 0 and 1 respectively. All three values are part of the
tested piped route.

Nine complete budget-specific requests had identical output/reasoning hashes and complete usage across arms.
The cancelled stream passed a shared-prefix comparison and the subsequent recovery matched; its full output
length and hash were not required to match. All four budget continuations in both arms read 19 tokens. In the C4
performance workload, each thinking continuation reused 10328 tokens and read 19; none repeated the earlier
1554-token read. All six performance requests had identical complete output/channel hashes and usage.

| Batch/piped MTP comparison | Off | On | Throughput change |
|---|---:|---:|---:|
| C4, 8320 completed API tokens | 82.044 s / 101.409 tokens/s | 78.936 s / 105.402 tokens/s | +3.94% |
| C8, 8192 completed API tokens | 33.495 s / 244.572 tokens/s | 30.497 s / 268.619 tokens/s | +9.83% |

C4 used four 8793-token resident requests and two cold joins of 8793 and 52080 tokens. C8 used eight short prompts
of 66 or 68 tokens, four sampled and four greedy, with thinking disabled and 1024 output tokens each. Its eight full
outputs and usage also matched. These API rates include admission, prefill and final drain; they are not pure
decode rates, and neither workload measures four simultaneous 100K contexts.

The C4 mixed-read stall remained: maximum nonempty SSE gap was 10.111 s off and 10.527 s on; p99 was 5.778 s and
6.120 s. The sampled fraction of the mixed phase with at least one resident silent for more than one second rose
from 69.5% to 72.3%. Improved whole-run throughput therefore did not resolve the long prompt-read pause. SSE gaps
include reasoning, content and tool deltas; no per-token timestamps were collected, so these are not token TBT.

The v4 solo prefill guard had three cold and three warm pairs, each with 101139 prompt tokens and 256 output tokens.
All six request/output hashes and usage matched. Cold cached tokens were zero; warm requests reused 101132 tokens
and read seven. Cold read-time median was 16491 ms off versus 16433 ms on; warm was 2225 ms versus 2181 ms. There was
no median regression in this sample, but three observations do not establish a stable prefill guarantee.

The subsequent shared piped-slot head candidate used engine SHA256
`c9305b08b24e6ff85e363147063827760c54c8243eeec6b2362587159cda5ca6`. Both arms kept batch/piped MTP and the budget cap
enabled; only `STRATA_PIPED_MTP_HEAD` changed from 0 to 1. The on arm used one shared read-only weight bank on GPUs
3/5 with eight independent workspaces sourced from GPU 7. Actual process, mapped library, frozen scripts, full
residency and original prompt-buffer allocation were checked in both arms.

| Shared piped-slot head comparison | Off | On | Change |
|---|---:|---:|---:|
| C8 complete API throughput | 248.752 tokens/s | 264.296 tokens/s | +6.25% |
| C4 complete API throughput | 106.426 tokens/s | 106.153 tokens/s | -0.26% |
| Cold 100K read-time median | 16012 ms | 16294 ms | +1.76% |
| Warm 100K read-time median | 2186 ms | 2056 ms | -5.95% |

The head candidate passed the same nine complete budget requests plus cancellation-prefix check, six prefill
pairs, eight C8 requests and six C4 requests, with full output and usage agreement. The on-arm logs recorded 2097,
4672 and 5219 head steps for the budget, C8 and C4 suites respectively, and zero during solo prefill. C4 maximum
mixed gap remained 10.433 versus 10.441 s. The cold prefill median increased in this pair; further repetitions are
required before accepting a prefill-preservation claim or deploying this option.
With the new head disabled, its fixed-request outputs and usage also matched the older v4/v5 MTP-on engine; suite
order differed, so that compatibility check makes no cross-run performance comparison.

Reproducible local records are under
`bench/results/2026-11-07-8x2080ti-q5-100k/concurrency-stability/` in the benchmark checkout:
`budget-v4-analysis/` and `budget-v4-c4-sampled/` for v4, `c8-v5/` for the eight-stream batch comparison, and
`head-pilot-analysis/` plus `head-pilot/` for the shared-head pilot. The analysis directories include input SHA256
manifests and per-request checks. None of these candidate switches has been deployed to the normal service as of
this report. Long-context concurrency and additional prefill/head repetitions remain separate deployment gates.
