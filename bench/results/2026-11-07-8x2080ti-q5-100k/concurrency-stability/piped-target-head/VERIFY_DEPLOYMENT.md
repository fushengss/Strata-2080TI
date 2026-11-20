# Production acceptance scripts

These files were prepared and checked locally. No remote status call, request generation, GPU operation, service restart or deployment was performed by this preparation task.

`verify_piped_deployment.py` defaults to the read-only `verify` action. It writes only an audit receipt. After the selected installation and **before starting its engine**, record a new cursor:

```sh
python3 /srv/bench100k-20261107-piped-target-head/verify_piped_deployment.py cursor
```

Start the existing service through the established maintenance procedure, then:

```sh
python3 /srv/bench100k-20261107-piped-target-head/verify_piped_deployment.py verify
```

The script never starts/stops anything. It checks the deployment receipt and its immutable release; original private config hash and field-by-field preservation; API key presence/equality without disclosure; active config 0600; server/library hashes; one actual engine process and its executable image; exact library map; all configured environment values; fixed MTP baseline flags and selected P1/P2 flags; original layer split and budget cap. Read guards, concurrent-read and MTP observation/profile switches must remain disabled.

Freshness uses the pre-start cursor's boot ID/uptime and the engine's `/proc` start ticks. The cursor also records log device/inode/offset and a tail hash; stale prefixes are excluded, rotations/truncations recognized. Startup facts use the frozen `p2-harness/runner_async_sample.py` parser, SHA256 `97db26740aae5b10f65b4bd0d0aa564eda9f578eb4f8bd7171631a20521d8fa6`. All eight slot devices, all-resident experts, eight 3.80 GiB own prefill buffers and shared MTP head route must appear. P1/P2 enabled markers must match receipt flags; disabled switches need no explicit disabled marker and must have no enabled marker. The parser is imported only for its functions; no harness execution occurs.

Success writes `deployment-verified.json`. Existing receipts/cursors are not overwritten; use a new `--out` or `--cursor-file` when deliberately repeating an audit. No raw log, environment, process command line, API key or response text is saved. Verification makes only an authenticated GET to localhost `/v1/status`; it sends no generation request. GPU clocks/power are neither queried nor changed.

## Optional eight-lane smoke

`smoke_piped_deployment.py` is separate and defaults to a plan with **zero HTTP requests**:

```sh
python3 /srv/bench100k-20261107-piped-target-head/smoke_piped_deployment.py
# Only after installation and successful verification, explicitly send eight requests:
python3 /srv/bench100k-20261107-piped-target-head/smoke_piped_deployment.py --execute
```

It requires the successful verification to refer to the same current engine process. It uses the frozen `p2-harness/client.py`, SHA256 `3c019afbcd75d7a183ce1089aea49ffd21abadf8aba85445573c369996c43b8f`, for eight sampled synthetic requests, max 128 output tokens each. Only its own timestamps, byte counts, usage and hashes are stored in a new `deployment-smoke` directory. There is no configuration/cache-clear/admin request; generation naturally updates the engine's request/cache state.

The smoke saves a separate log cursor immediately before its requests and extracts numeric target-head, async-sampling and MTP-head counters afterward. These counters cover service batches, not necessarily only the smoke when external traffic exists, and drain summaries may be absent until unrelated work finishes. The result states whether status counters are consistent with exclusive access and whether selected routes were observed. Missing route summaries mean incomplete route evidence, not proof of an engine fault; disabled routes producing counters fail the check. No throughput or latency conclusion is made, even if the smoke is exclusive.

Local-only checks:

```sh
python3 verify_piped_deployment.py self-test
python3 smoke_piped_deployment.py --self-test
```

Results are `verify-offline-check.json`, `smoke-offline-check.json`, and `smoke-dry-run.json`. They exercise numeric parser combinations, missing/duplicate startup rejection, protected config checks, disabled guards, old-log exclusion/copy-truncate detection, credential-output rejection and receipt overwrite prevention. They do not replace an actual post-start acceptance run.
