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
above or change the batch/piped MTP switches. As of 2026-11-07, its CPU tests pass; the independent GPU candidate is
running budget-specific checks, prefill checks and concurrency comparisons with the option enabled in both arms.
Those GPU checks are not yet complete, and this option has not been deployed to the normal service. The frozen
candidate `serve/server.py` SHA256 is `40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9`.

The triggering observation was a thinking-budget continuation during sampled C4: two requests with batch MTP
enabled reused only their initial 8793-token prompt and read 1554 tokens again. The control reused 10329 tokens and
read 18. Reasoning text hashes matched; those two answer text hashes differed. Speculative tokens emitted before
the frontend's asynchronous `BSTOP` can leave the slot beyond the prefix that the frontend retains. The next
prompt inserts a thinking wrap-up at that earlier boundary, so the longer recurrent state cannot simply be
truncated. This is the motivating diagnosis, not proof that the candidate fixes GPU output or latency.

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
