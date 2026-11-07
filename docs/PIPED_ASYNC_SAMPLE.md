# Piped asynchronous slot sampling

`STRATA_PIPED_ASYNC_SAMPLE=1` lets the piped batch pump advance other stages while the final stage's
existing per-slot sampler runs. It is off by default and requires deferred-commit piped MTP batches.
It is independent of `STRATA_PIPED_TARGET_HEAD`; evaluate each switch separately before combining them.

The sampler kernels, parameters, logits, Philox seed/position, penalties and logit bias are unchanged.
The final stage records an event after all required row draws instead of synchronizing its stream inside
`sample_rows`. `batch_poll` returns while that event is incomplete. The stage remains occupied: no next
group can overwrite its logits, residual, row positions or history, and no verdict, output or deferred
commit uses its picks until the event completes. Ordinary synchronous slot runs, solo decode and prefill
keep their existing completion path.

At initialization the final verifier reserves pinned host history and device history for `max_t` rows,
each with a fixed 4096-token stride. A temporary CPU vector computes the same history as before, then its
selected row is copied into that row's persistent pinned slice. The GPU upload reads the persistent slice,
not the temporary vector. Different row penalty lengths cannot overlap; neither history allocation grows,
is freed or is reused while sampling is in flight. Existing admission/drain boundaries protect per-slot
bias allocations; this change does not add a way to update a live slot's bias.

Startup reports:

```text
strata serve: STRATA_PIPED_ASYNC_SAMPLE: source CUDA7, event-polled slot sampling with persistent history
```

Each completed piped batch reports actual completed asynchronous work:

```text
strata piped async sample: N windows, M sampled rows completed in this batch
```

The row count includes greedy rows that require penalty or bias processing. Plain greedy rows retain the
captured pick and need no asynchronous draw. A pure-greedy batch may therefore report zero; these counters
are not accepted output tokens. Prefill-only work does not increment them.

The sampler's existing per-device/per-stream scratch allocator remains unchanged; its first call can
allocate scratch. The switch removes the explicit per-window sampling wait, not every possible driver or
allocation delay. A CUDA event query also has host overhead. No performance claim follows from the code.

Validation should compare the same binary with this switch off/on, holding target-head placement and all
other settings fixed. Require exact completed outputs, usage and finish reasons for mixed greedy/sampled
requests, different penalty lengths and penalties, logit bias, MTP acceptance/rejection, cancellation/reuse
and budget boundaries. Check nonzero runtime counts on a sampled workload, then compare concurrent decode
and 100k prefill. CUDA/GPU and end-to-end validation are performed on the test host; they were not run during
local implementation. Local syntax checking with the Metal compatibility declarations does not replace
the CUDA build or validate asynchronous device execution.
