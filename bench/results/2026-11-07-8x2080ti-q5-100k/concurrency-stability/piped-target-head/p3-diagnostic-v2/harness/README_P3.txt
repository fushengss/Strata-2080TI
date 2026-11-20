P3 piped MTP observation — single-arm diagnostic

Use runner_piped_mtp_observe.py, not the copied P2 entry point. The 14 original
P2 harness files remain byte-identical; p3-source-provenance.json binds those
copies. The P3 wrapper imports only its local copied runner, adds OBSERVE=1,
and inherits the isolated-process ownership, four-SHA binding, config/credential
handling, GPU/port exclusivity, actual-process evidence and suite checks.
The wrapper itself, parser, comparator, all Python/JSON/TXT files and this README
are copied and SHA256-frozen into each executed result's harness directory.
The reviewed static inventory in p3-frozen-files.json is checked before running;
its own SHA256 is included in the result's runtime harness_sha256 map.

Default is dry-run: no private config read, network, inference or GPU operation.
  python3 runner_piped_mtp_observe.py --target-head 1 --async-sample 1
  python3 runner_piped_mtp_observe.py --self-test

Select the FINAL reviewed P1/P2 settings explicitly. This wrapper permits only
one arm and requires C8; c4mixed may be added. The inherited arm directory name
01-on/01-off reflects only the chosen P2 mode, not an observation A/B comparison.
The root manifest has experiment_type=diagnostic, diagnostic_only=true, fixed
P1/P2 fields, an empty variables map and the frozen entrypoint SHA256.

Execution requires operator-established maintenance/exclusivity, the candidate
observation binary, and four explicit reviewed identities:
  python3 runner_piped_mtp_observe.py --execute --target-head 1 --async-sample 1 \
    --suites c8 --exe CANDIDATE --server SERVER_PY --python PYTHON \
    --config PRIVATE_BASE_CONFIG --out NEW_DIAGNOSTIC_DIRECTORY \
    --engine-sha256 HEX --library-sha256 HEX --server-sha256 HEX \
    --config-sha256 HEX

Do not add another arm via the copied runner. There is no --arms option here.
No power/clock setting or production deployment is performed. Existing :8080
must already be stopped; only owned :8092 descendants are started/stopped.
P1/P2 configurations, source directories and frozen results are not modified.

The wrapper requires the observation startup marker for eight independent slot
event sets and parses every suite's exact new engine-log range. A missing,
partial or malformed observation report fails qualification. It also retains
the inherited P1/P2 route-counter checks. After cleanup it writes each arm's
p3-observation.json using only offline parsing; it does not start another test.

Standalone offline tools:
  python3 parse_piped_mtp_observe.py --log ENGINE_LOG --out NEW_PARSE_DIRECTORY
  python3 compare_piped_mtp_observe.py --baseline SELECTED_P2_ARM \
    --candidate P3_DIAGNOSTIC_ARM --out NEW_COMPATIBILITY_DIRECTORY

The comparator checks C8 normally completed request/output/reasoning hashes,
usage and finish against the selected P2 arm, plus each run's frozen source and
binary/library/server/config identity records and fixed P1/P2 modes. Different
engine/library builds are individually bound, not required to match. Server
and base config must match across runs. This is a correctness compatibility
check, not a speed comparison or an automatic additional benchmark. C8 captures
content/reasoning channels, not a comprehensive tool-call stream.

Observation units and limits:
- body/head_return/tail are milliseconds measured by independent per-slot,
  per-step events on the source stream. They include contention/dependencies;
  head_return includes mapped activation/logit transfer and waits.
- The body interval starts after existing residual staging/dependencies. It is
  not the full verdict-to-draft latency or a pure kernel-active measurement.
- Streams overlap; phase sums across slots are not total engine wall time.
- verdict_gap is a host arrival interval in milliseconds and may cross new
  prefill admission, drain/refill or other pauses in the selected log range.
- submitted_unreaped_pump_samples is a histogram of pump samples, not durations
  or time-weighted occupancy. A submitted chain can already be complete on GPU.
- unsubmitted_jobs_at_verdict is the number immediately submitted by that
  verdict. GS=1 gives zero or one; it is NOT a post-attention ready queue.
- stage0_draft_not_ready_group_skips counts attempts, including repeats; it is
  not elapsed waiting time or distinct jobs.

Instrumentation can perturb timing. Do not report diagnostic throughput as an
optimization gain. These probes do not implement packing and cannot establish
a usable post-attention packing queue by themselves. Any later packing design
must demonstrate naturally ready compatible jobs and avoid waiting to form a
batch. Only CPU/synthetic checks run during local tool preparation.
