# Piped MTP observation

`STRATA_PIPED_MTP_OBSERVE=1` adds timing and counters to the existing CUDA piped MTP shared-head
path. It is off by default. Startup rejects it without active piped MTP slot drafters and
`STRATA_PIPED_MTP_HEAD=1`, on other backends, or together with `STRATA_MTP_PROFILE`.
The latter uses one shared timestamp buffer and cannot separate concurrent slot chains.

This switch does not pack jobs, create a ready queue, wait to form a batch, or change the order of the
body, head or tail graphs. Prefill and admission's ordinary draft graphs retain their path. With the
switch off, no observation event is created or recorded and no observation clock is read.

Each slot owns four timing-enabled CUDA events per possible draft step. All four are recorded on
that slot's source stream:

1. Before the body graph, after the existing residual staging and dependency.
2. After the body graph.
3. After both remote head shards have completed and their full logits have returned.
4. After the original selection/probability/residual tail.

The events are read only after the existing chain completion event succeeds. A slot cannot launch
another chain until that completion is reaped, so neither another slot nor the next chain can overwrite
its measurements. Reading elapsed times adds no stream synchronization. Destruction retains the
existing source-stream drain before destroying the observation events.

The batch-drained report contains completed-chain counts and per-draft-step count, mean, minimum,
maximum and sum in milliseconds for `body`, `head_return` and `tail`. Step zero's body includes the
verified-window catch-up; later steps, when configured, execute a single new draft row. These are
source-stream elapsed intervals. The body starts after residual staging and its upstream dependency,
so it excludes waiting before the chain body begins. The intervals include GPU contention and dependencies; `head_return` includes
activation publication, remote projection, source waits and the mapped logit copy. They are not pure
kernel active time, FLOPS utilization or disjoint contributions to whole-system wall time. Intervals
from different slot streams can overlap and their sums must not be added into an engine throughput.

The host report contains:

- `verdicts` and `verdict_gap_ms`: arrivals of last-stage group verdicts. Gaps can include admission,
  prefill, pipeline drain/refill and other pauses within the reported span; they are neither GPU stage
  execution time nor pure decode interarrival time. Align them with the run's admission/decode phases.
- `stage0_draft_not_ready_group_skips`: scheduling attempts that found a group's draft still pending.
  Repeated attempts count repeatedly; this is neither elapsed waiting time nor a count of distinct jobs.
- `submitted_unreaped_pump_samples=[N:count,...]`: pending-chain count sampled at each pump entry.
  `pm_drafting` means submitted but not reaped, including work whose GPU event may already have
  completed. This is a distribution of pump samples, not a time-weighted occupancy distribution.
- `unsubmitted_jobs_at_verdict=[N:count,...]`: the live jobs from the current verdict immediately
  before the existing submission loop. With eight slots and eight groups this is zero or one. They
  have not run their body and are not a post-attention GPU ready batch.

Reports appear only with the existing drained-batch summary, not per chain. The per-slot aggregates
cover completed chains since the preceding report. No prompt, token ids, output text or credentials
are logged by this observation path.

Use an observation-off run as the performance baseline. Timing events, event queries, host clocks and
counters themselves add overhead. Compare full output, usage, finish reasons, MTP acceptance and route
counts before treating an observed run as representative. This code has no local GPU performance
measurement and does not claim a speedup.

These probes can establish which broad draft phase competes for GPU7 and whether draft waits are
common. They cannot establish a usable post-attention packing queue: existing chains are submitted
immediately, and this probe deliberately preserves that schedule. A later packing experiment must
show naturally queued compatible jobs and retain an immediate single-job path when only one is ready.
