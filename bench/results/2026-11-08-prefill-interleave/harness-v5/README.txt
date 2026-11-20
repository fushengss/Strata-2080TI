HARNESS v5 — finite BYIELD peer fixture correction

Only BYIELD peer output budgets change from4096 to3072. The same synthetic prompts,
fixture IDs, temperature/seed and request sequence are preserved. Early EOS is
still rejected: every normal request must complete its exact requested length,
and the two arms must match request/output hashes and actual logical usages.

Measured motivation (root's on-host v4 observation): native stop/park/resume and
short-first gates passed; one peer naturally stopped at4039 before its4096budget.
The transfer/recovery completed around25seconds, whereas that peer's EOS was
around80seconds. These observations motivated a lower fixed3072budget; they do
not prove v5 GPU correctness or adequate coverage before it is run.

New explicit gate: all four peers emit visible content before long admission and
continue visible output until after BOTH long and short HTTP requests finish
normally. The last visible chunk must reach that recovery boundary: a delayed
terminal/HTTP drain alone cannot qualify. The result records each peer's last
visible output and finish margins, and the minimum margins. Early completion, late
first output, missing peers or normal EOS fail. Both arms must pass this coverage
in the offline comparison. No primary/mixed100k performance-cohort requirement is
relaxed. Admission/cancel/source audit and every other suite's source remain byte
identical to v4; the only other Python edit adapts the CPU fake transport's
resident-budget threshold to3072.

Use fresh paired arms from this version for BYIELD:
python3 harness-v5/runner_interleave.py --execute --arms on,off --suites byield --config /srv/Strata/strata-orca-q5_k_m.57.json --exe CANDIDATE_PATH --engine-sha256 CANDIDATE_SHA --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd --out NEW_V5_BYIELD_OUTPUT_DIRECTORY

No external calibration file is required for BYIELD. Its independent prompt is
count-only calibrated to100000±2 on each arm, then exact final OpenAI prompt usage
must match. Long outputs128, short outputs64, peers3072 each. The long resumed
request may report parked tokens as cached; initial coldness is proved from the
first native PERF leg reused0. Native stopped_at is relative: absolute park and
resume position is begin.from+stopped_at, with the same chunk/segment endpoint.
Candidate must have actual park, ordered native legs/resume, and short first
visible output before long. Baseline is a complete-output control.

All other suites remain available: admission,correctness,c8,primary,mixed100k,
prefill,budget,long. Original runner safety remains: no production mutation,
isolated8092, empty GPUs,8080 stopped by root, exact explicit identities, only
owned process cleanup, private temporary configuration, sanitized artifacts.

The v4 cancellation exception is unchanged: only one proven intentional cancel
read ID in on/admission can have read_ok0, with pump_ok1, an exact native cancelled
report, no HTTP output, same-request normal reuse, and independent runner proof
recomputation from the sealed pre-reuse log window. Other failed reads always fail.
All reader/source pairing and lookahead_idle1/dispatch_delta0 gates remain.

CPU checks passed: byield_smoke.py --self-test; byield_offline_tests.py; full
finite/admission offline_tests.py; AST all20Python files; exact source comparison
showing only byield_smoke.py and offline_tests.py changed since v4. Coverage tests
include early end, late first output, early EOS, missing lane, missing boundaries,
and early last visible output masked by a much later HTTP end.
GPU BYIELD qualification still must be run. See source-provenance.json and
harness-v5-freeze.json. Previous v1/v2/v3/v4 packages remain immutable.
