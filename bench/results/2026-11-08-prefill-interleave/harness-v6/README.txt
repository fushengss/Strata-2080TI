HARNESS v6 — realistic finite-request output contract

This revision changes qualification for primary/mixed100k, before running the new
paired candidate experiment. Request bodies, prompts, fixture IDs, sampling/seed,
max_tokens budgets, schedule, calibration and stream client are unchanged. It is
a fixed-request finite cohort with fixed MAXIMUM output budgets, not a promise
that a model will always use the full budget. No EOS suppression or prompt tuning
is introduced. Failed earlier runs remain failed artifacts and are not relabeled.

A finite response qualifies only when:
- It is a complete, error-free normal response with visible answer content.
- finish_reason=length has actual completion_tokens exactly equal to max_tokens.
- finish_reason=stop has an integer actual completion count in1..max_tokens.
- API prompt/completion/total counts are integers and arithmetically consistent.
- Native final timings.predicted_n exactly equals API completion_tokens.
- The frozen exact prompt count, explicit cold-cache0, native prompt accounting,
  thinking=false, no reasoning bytes, and no tool channel hold. A tool-id-only
  event is rejected even when its text/argument byte count is zero.
Cancellation, tools, content-filter/error/unknown finishes, empty output,
over-budget stop, or a truncated length finish are not normal EOS.

All four resident streams must still cover every baseline/mixed/recovery phase.
The existing phase/silence calculations are unchanged. An extra visible-output
qualification requires each peer's first visible chunk at/before each phase start
and last visible chunk at/after its end. A late terminal/HTTP close cannot hide a
peer whose content already stopped. A request naturally finishing before recovery
ends still invalidates the performance cohort.

The matched comparison remains exact per request: same request body hash,
visible/reasoning/tool output hashes, finish_reason, and all three ACTUAL logical
usage counts. Identical natural EOS across both arms is valid; different EOS
positions, different finish reasons or unequal actual token work invalidate the
comparison. No speedup claim from unequal work is accepted. Whole-case throughput
uses summed actual completion tokens and the original complete-case duration.
The comparator also binds this output_contract and visible-coverage qualification.

Fixed workloads (unchanged): primary C4 residents each exactly10000 prompt tokens
with max4096 outputs, plus independent10000/60000 injections with max64 outputs;
mixed100k C4 residents each exactly100000 prompt tokens with max6144 outputs, plus
an independent100000 injection with max64 outputs. The calibration SHA remains
b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd.

Run NEW paired arms and a NEW output directory:
python3 harness-v6/runner_interleave.py --execute --arms off,on --suites primary,mixed100k --calibration calibration.json --calibration-sha256 b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd --config /srv/Strata/strata-orca-q5_k_m.57.json --exe CANDIDATE_PATH --engine-sha256 CANDIDATE_SHA --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd --out NEW_V6_OUTPUT_DIRECTORY

All other suites and safety gates are byte-identical to v5: BYIELD still uses
fixed3072 peer outputs and strict length/exact-output/last-visible coverage;
admission retains its independently proven single intentional-cancel read_ok0
exception; normal reader/pump/source identity, residency, CUDA checks, exact build
and base-config identities, own-process cleanup and private config handling remain.
No script stops production8080. Root controls the maintenance window.

CPU tests: full offline_tests.py runs matching EOS2815 of a4096budget in both arms,
then rejects differing EOS count, finish reason, native predicted_n, invalid stop
bounds, truncated length, tools/cancel/error, phase-early end, missing lane and
last-content early with delayed HTTP end. Complete admission cancellation and
BYIELD fake-HTTP regression tests pass. AST parses all20Python files; the finite
plan AST, request generator/client/calibration, BYIELD and other actual suite
sources remain unchanged. See source-provenance.json and offline-checks.json.
CPU results do not establish GPU correctness, prefill non-regression or speedup.

SSE gaps remain chunk-arrival measurements, not per-token TBT. Actual native
in-read output rows exclude end drain and ambiguous boundaries.500ms GPU activity
samples do not establish SM saturation. Performance/deployment review stays with
root after exact paired outputs and all independent qualifications pass.
