Incremental prefix-cache qualification and overlap A/B — v1

This NEW package is separate from immutable harness-v1/v4/v5/v6/v7 and BYIELD
packages. It compares only STRATA_PIPED_CONCURRENT_MTP=0 vs1. Both arms explicitly
set STRATA_PARK_OVERWRITE=0, retain it in configuration equality, and verify it
in the actual engine environment. Manifest profile:
  concurrent-mtp-only-park-overwrite-fixed-off

The experimental variable remains one flag. The earlier two-flag draft is saved
separately as incremental-combined-draft-unfrozen and must not be executed.
Actual overwrite tracing moved resize cost into first-touch copy without a
net park benefit; this package does not enable it or attribute gains to it.

Candidate selected by operator: source bf5a8e5b, v3 engine
/srv/Strata-decode-head-20261107/engine/strata.prefill-interleave3
SHA25645eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6.
Exact engine/library/server/base-config identities are required CLI arguments;
no implicit trust or production mutation. Original prefill split, prompt chunk,
head routes, MTP settings, eight slots and own buffers are preserved.

Incremental workload

Eight normal, finite requests: one cold roughly100k seed, then four DIFFERENT
roughly10k resident prompts (max4096), then three cumulative roughly1k user-text
extensions of the seed (max64 each). No unknown assistant text is added to
later requests. This is a synthetic prefix-extension proxy, not a full agent
chat or tool workflow. Five seconds baseline after C4 is ready, injections
planned8seconds apart, then10seconds recovery and normal final drain.

Every new prompt uses a fixed independent incremental-cache-v1 namespace. The
source calibrated word counts and TASK remain fixed, but the namespace changes
tokenization slightly: report the ACTUAL counted prompt lengths, not invented
exact100000/101000 labels. All8 bodies are deterministic before execution.
Each arm makes8 stateless count_tokens calls, verifies generation counts did
not increase, saves preflight counts/body hashes in its manifest, then requires
actual API prompt usage and native prompt accounting to match exactly. The
original calibration.json stays unchanged. There are no GPU one-token probes,
cache clears, unknown chat content, timed refills or dynamic body changes.
Independent namespace prevents polluting later old primary/mixed cold fixtures.

Seed must complete cold before any of the four distinct resident requests.
First increment requires a native restored>=99000 marker in its own exclusive
log slice, API cached>=99000, native cache/read agreement, and fresh800..1300.
This proves cross-conversation restoration after other prompts displaced the
main session. Later increments may use live-prefix reuse; restore counts and
native timings are reported separately. On-arm each increment additionally
requires an actually paired reader from>=99000 with roughly1k tokens, at least
four active slots, read/pump success and positive strictly-in-read native BT
outputs. Native BT output includes control/EOS; API visible chunks establish
visible progress. Reader/source proofs remain strict for every launched read.

Normal length must exhaust max_tokens; normal stop must have0<actual<=budget.
API usage arithmetic, native predicted_n and actual visible answer are required;
errors, cancellation, reasoning or tools do not qualify. All four residents must
have visible output before and after EVERY measured phase. A late HTTP end does
not establish coverage. No length/coverage gate is relaxed after seeing results.

Paired comparison requires identical request/output/channel hashes, finish reason,
three actual API usage counts, and exact cache_n/fresh prompt_n/cached API counts
for every request, as well as same preflight/body/plan/source/config/engine
identities. Mismatched cache behavior cannot claim matched-work speedup.
Both whole-case efficiency INCLUDING seed and post-seed finite C4+increments
are reported. Both include admission/restore and final drain; neither is pure
decode. SSE gaps are client-arrival chunk gaps, not token TBT. No automatic
performance or deployment verdict is produced.

Other suites

The old finite/client/fixture/correctness/C8/budget/BYIELD/admission/stall-metric
implementations and request contracts retain their v7 hashes. In particular,
primary normal EOS remains valid only with exact actual paired work and visible
coverage; BYIELD peers are3072; budget peers cap1024 and must actually overlap.
Updated files are runner/comparator and prefill/long/cross-comparison mode
validation; only the one overlap flag is normalized away. PARK_OVERWRITE remains
fixed0 in equality. Core/C8/budget compare output/count/finish, with outer runner
actual-env and config gates establishing the experimental mode.

The copied legacy prefill_check.py is not an execution path for this runner;
use prefill_interleave_check.py via --suites prefill. Never invoke an old helper
and assume it represents this package's profile.

Manual comparison additionally requires complete parent manifests, all planned
suite returncodes0 in identical order, finished arm metadata, completed-arm
registration, no parent failure, and explicit overlap-only profile/flags. This
also closes the older standalone comparator's partial-run eligibility gap.

CPU checks (no HTTP sockets or GPU)

  python3 -B incremental_offline_tests.py
  python3 -B overlap_profile_tests.py
  python3 -B offline_tests.py
  python3 -B byield_offline_tests.py
  python3 -B prefill_interleave_check.py --self-test
  python3 -B budget_correctness.py --self-test
  python3 -B long_concurrency_check.py --self-test

Tests cover complete fake off/on control flow and cache restoration, matched
normal EOS, exact cache/usage/hash comparison, source/reader bindings, missing
restore/cold read/wrong from/count/finish/early visible completion rejection,
wrong or missing mode flags, and preservation of unrelated config fields.
Synthetic fake-test tokens/s values have no hardware performance meaning.

Execution (operator owns maintenance and production restore)

Default invocation is dry-run. Example remote exact command, after verifying
archive/freeze hashes and with production8080 stopped and GPUs empty:

python3 /srv/bench100k-20261108-prefill-interleave/incremental-harness-v1/runner_interleave.py \
  --execute --arms on,off \
  --suites incremental,prefill,primary,mixed100k,correctness,c8,admission,byield,budget \
  --calibration /srv/bench100k-20261108-prefill-interleave/calibration.json \
  --calibration-sha256 b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd \
  --exe /srv/Strata-decode-head-20261107/engine/strata.prefill-interleave3 \
  --engine-sha256 45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6 \
  --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 \
  --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 \
  --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd \
  --out /srv/bench100k-20261108-prefill-interleave/NEW-OWNED-RUN

A different suite order is permitted, but must be identical within the A/B pair.
Independent namespace makes incremental cold seed valid alongside the old suites;
its cold0 gate still rejects an unexpected prior identical incremental request.
No per-suite source modifications, service restart, cache clear or profile change
is made between suites. Full runner comparisons happen after both arms finish.
