# Piped target head

`STRATA_PIPED_TARGET_HEAD=1` is an experimental CUDA path for the final verifier in a deferred-commit
piped MTP batch. It uses the existing `STRATA_DECODE_HEAD_GPU=D1,D2` weight copies. It is off by default.
The existing solo PW2 head setting alone does not enable this batch path.

The final stage keeps its layers and final mixer. Its exact Q8_1 rows go to the two helper GPUs, which
project separate vocabulary rows using the same MMVQ mode and token-column grouping as the original head.
The complete logits return in vocabulary order to the final stage. That stage runs its original global
greedy pick and nonfinite guard, then the existing per-slot sampler. Temperature, top-k/top-p, penalties,
logit bias and Philox seed/position remain with the original sampler; shard-local argmax cannot replace
sampled target logits.

The helpers borrow the existing read-only `NativeHead` weights. They have one private batch workspace
per helper, shared over the groups, not one weight copy per slot. The final stage stays busy until its
logits, guard and sampling are complete. Its residual, row metadata and deferred state cannot be reused by
another group while a remote head is still reading or returning results. The batch body, save and deferred
commit share a separate graph key; ordinary slot runs and admission work retain their original full-head
graphs. Prefill layer placement, buffers and chunks are unchanged by this switch.

Startup rejects a requested path without piped MTP/deferred commit or an existing helper weight bank.
Full-logit workspaces require exact multi-column MMVQ, a tile-aligned vocabulary split and matching
source/helper kernel-selection eligibility. Unsupported packed head layouts are rejected. No P2P is needed:
portable mapped activation/logit buffers and cross-device events order the transfers.

Startup reports:

```text
strata serve: STRATA_PIPED_TARGET_HEAD: CUDA2+4, source CUDA7, full logits return before original slot sampling; prefill placement unchanged
```

After a piped batch drains, a separate line proves the path actually ran:

```text
strata piped target head: CUDA2+4, N windows, M verification rows offloaded in this batch
```

These are completed verifier windows and input verification rows, including draft rows. They are not
accepted output tokens or throughput. A prefill-only run does not increment them. The stage's ordinary
head profile stamp ends with its body graph; it does not include the remote projection and returned tail.
Use end-to-end timing and GPU/stage traces when comparing this path.

Validation must compare off/on builds and actual route counters, not just the environment. The expanded
`decode_head_test` exercises full-logit return, reuse, the source tail and sampled/penalized/biased rows:

```sh
build/decode_head_test --gguf MODEL.gguf --source 7 --target 2 --target2 4 --cycles 4
```

The test includes T=1/2/3/8, exact full logits, an odd row-range tail, nonfinite inputs, and shared weights
across independent workspaces. Run the engine A/B with mixed greedy/sampled requests, MTP acceptance and
rejection, cancellation/reuse, budget boundaries and long contexts. Check full output/usage/finish identity
and 100k prefill separately. Performance and backend validation were not run on the local development host;
this document makes no speedup or prefill-regression claim.

This change deliberately retains the existing synchronous per-slot sampling completion. Making it
asynchronous is a separate change, requiring persistent penalty-history staging and buffer-lifetime checks.
