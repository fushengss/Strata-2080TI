# Concurrent prompt reads with piped MTP

`STRATA_PIPED_CONCURRENT_MTP=1` is an experimental, default-off extension of
the piped reader. A prompt runs on its existing prefill streams while the
host continues to pump decoding slot groups. The prefill layer split and
planned chunk geometry stay the same. A read is not divided into additional
decode-budget pieces. This candidate has no measured performance claim yet.

This flag is sufficient to request the path; `STRATA_PIPED_CONCURRENT` and
`STRATA_PIPED_READ_PIECE` are not required. The existing
`STRATA_PIPED_CONCURRENT_MIN`, `_PRIO`, and `_UNPIN` controls still apply.
Verifier streams request high priority unless `_PRIO=0`. GPU stream priority
does not guarantee preemption or latency; solo and mixed prefill performance
must both be measured.

## Eligibility and lifetime

Each prompt part checks the live state after the normal pipeline drain:

- CUDA, deferred-commit piped MTP, and `STRATA_SPLIT_MTP_BATCH=1`.
- At least one active decoding slot. A solo prompt uses the existing path.
- Every prefill stage owns its prompt region, with no borrowed buffers,
  borrowed spill, active MTP room, or outstanding expert-cache loan.
- Every verifier stage reports all experts resident, the host residency
  table agrees, and the main drafter's expert weights remain resident.
- No pending tier change, asynchronous tier worker, KV growth, peer path,
  helper expert cache, remote expert work, or Foresight swap space.
- One text prompt. Images and co-reads use the existing path.
- A PLE stage with a native quantized key and no BF16 key uses the serial
  path: that key's quantization scratch belongs to the shared weight table.
- The old source restrictions remain: no prefill helper, DMA doorbell,
  staging/recycled source buffers, or unknown source type.

A failed check prints a fallback reason and uses the existing serial path.
Before starting the reader, the host reserves the drafter's input records,
the prompt upload rings, and the prompt MMQ contexts. A pinned reservation
failure also falls back; it does not start a concurrent read. The prompt
gets its own PLE fallback scratch, since the verifier uses the main session's
PLE scratch. The fallback PLE kernels and arithmetic remain unchanged.

These preparations do **not** reserve every internal MMQ/GEMM allocation.
First-use or larger-shape workspaces and weight conversions may still cause
device barriers. Cold first use, subsequent admissions, and small tails must
be validated separately.

## Why the prepared PLE flag is published early

An all-resident batch graph can still wait for the mapped PLE-ready flag.
Previously, the host set its `b_running_` flag only after launching the body,
save, and head work. A concurrent reader's allocation barrier could block
one of those CUDA calls while the body waited for PLE. The doorbell thread
saw `b_running_ == false`, so it could not release that wait.

Only during the new mode's external serving, and only for a resident PLE
stage, `stage_batch` publishes the ready flag after its actual gather and
flag reset, with the existing CPU/store fences, before any body launch.
`b_running_` is still published at its original point. Nonresident expert
waits and the default batch path are unchanged. Batch graphs do not use the
solo inter-stage mapped handoff wait. Head and draft event waits retain their
producer-before-consumer enqueue order.

The reader is joined on normal, failure, and yield exits before its stop
hook is removed. The doorbell owner then joins its thread and returns
external serving to the normal pump. `BSTOP` marks a slot to stop; it does
not promise to abort an in-progress prefill immediately. `BYIELD` retains
the existing chunk-boundary behavior.

## Runtime evidence

The startup line reports that the flag was requested, not that an eligible
read ran. Actual reads print paired `strata piped concurrent MTP: begin` and
`end` lines with a monotonic `read_id` within that engine process. A rejected
read prints `fallback ... reason=...`.

`begin` includes the token interval (`from`, `tokens`), planned chunk,
active-slot count, and owned/resident stage counts. Prompt parts can end at
checkpoints; the final prompt token is normally handled by the first decode
window. Do not equate one `tokens` field with the API's full prompt count.

`end.read_ms` measures the reader's `Prefill::run` wall time, before the final
decode drain. `decode_rows_before_done` is a conservative count of emitted
native `BT` output tokens, including an emitted termination token: pumping
starts only after the reader-start publication, and
only a completed pump observed before the reader's done publication is
counted. The ambiguous last pump and final drain are excluded. This count
proves output progress during the read; it is not verification rows, a count
of accepted drafts alone, API content tokens, or GPU occupancy. The old
“decoded meanwhile” line includes the drain and is not this
strict measure.

`read_ok`, `pump_ok`, `stopped_at`, `admit_stop_requested`, and `yield_slot`
describe their individual states. An `end` line alone does not prove that
the final drain, admission, or request completed successfully. Validation
must also check completed outputs and usage, cancellation and slot reuse,
absence of engine errors, solo 100k prefill, and mixed-stream output gaps.
