# Optimization iterations published on 2026-11-08

This series preserves the fork's main history and separates the tested changes by direction.
The older main tree and the production development branch had unrelated Git roots. The
baseline commit imports the tested upstream/fork snapshot `9d71f701`, including its software
and build dependencies, while retaining this fork's README, agent instructions and historical
tuning documents. It does not fetch a newer upstream version.

| Direction | Commit | Scope |
| --- | --- | --- |
| Earlier prefill tuning and measurements | `cca2ae05` | Preserve the local split-chunk ceiling control and completed optimization roadmap; follows historical commit `25cd4ab4`. |
| Tested upstream/fork baseline | `0e132859` | Engine 0.1.57 and matching software foundation; includes earlier fork patches ported to that baseline. |
| Target output head | `ad375c60` | Opt-in PW2 output-head offload and vocabulary sharding, with CPU/backend interfaces and tests. |
| Solo MTP draft head | `a5b521a5` | Distribute draft-head vocabulary work across two GPUs. |
| Concurrent MTP and reasoning budgets | `fdaa4a48` | Share draft-head shards across piped slots, restrict optional prompt slicing, and bound known reasoning budgets. |
| Piped target heads and observations | `0cdee312` | Full-logit return, exact slot sampling, optional asynchronous sampling and timing observations. |
| Concurrent resident prefill | `e20a4087` | Keep decode advancing during eligible resident prompt reads; resource guards, private scratch and source-idle checks. |

The final runtime sources, tests, CMake files, Metal files and third-party dependencies match
`3ebdd999` from the production development branch. Its engine code matches `bf5a8e5b`, built
as SHA256 `45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6`.
Publication reorganizes commits; it does not rebuild or restart the running service.

## Selected behavior and measured results

The measured host has eight RTX 2080 Ti 22 GiB cards, Q5_K_M experts resident on all cards,
int8 K/V with 32768 resident tokens, eight slots, and the unchanged layer boundaries
6,12,18,24,30,36,42. Target heads use GPUs 2/4 and MTP heads use GPUs 3/5.
`STRATA_PIPED_CONCURRENT_MTP=1` is selected on that host and remains opt-in in the source.

Two matched cold-injection comparisons reduced the largest visible output gap from about
11 s to 0.66 s for 10k/60k injections, and from 17.1–17.3 s to 2.4–2.5 s for a 100k injection.
Mixed native prompt time increased 5.5–11%. A separate true cached-conversation pair reused
more than 100k tokens per turn, reading 867 fresh tokens each time. Post-seed finite-case
throughput rose from 145.8 to 154.6 tok/s; the largest visible gap fell from 3.45 to 1.85 s.
Added-turn first output took 0.19–0.25 s longer. Whole-case rates include admission and drain,
and the cached-conversation result is one pair. This does not establish full GPU saturation.

- [Head placement and solo MTP](MTP_HEAD_SHARDS.md)
- [Concurrent MTP and budget handling](CONCURRENCY_STABILITY.md)
- [Piped target head](PIPED_TARGET_HEAD.md)
- [Piped decode results and selection](PIPED_DECODE_2080TI_RESULTS.md)
- [Concurrent prefill implementation, measurements and deployment](CONCURRENT_MTP_READ.md)
- [Earlier optimization roadmap](OPTIMIZATION_ROADMAP.md)

## Experiments and validation limits

Asynchronous slot sampling (`STRATA_PIPED_ASYNC_SAMPLE`) and timing observations remain
optional. Asynchronous sampling was not selected for production. The read-slicing guard
was also not selected. Their documented presence is not a claim of performance benefit.
The later KV snapshot overwrite experiment had no measured benefit; its implementation
and tests are absent from this series's final source.

The selected runtime previously passed CUDA build/hardware validation, exact output and
usage comparisons across nine workload suites, cancellation/reuse, BYIELD, reasoning-budget
continuation, cached conversation turns, and postdeployment health checks on the measured
host. The publication checkout also runs the CPU prompt-read guard test and native thinking
budget tests. Other GPU backends and the entire imported app/Terminal suite were not rerun
for this publication. Imported upstream whitespace is retained to keep source identity.

Local 2026-11-07/08 run archives, live request telemetry, host configuration, credentials,
model files, binaries and the local DOCX are not included. Upstream's already tracked public
benchmark fixtures remain part of the baseline. Benchmark paths mentioned in the technical
reports refer to the private local archive, not files newly published with this series.
