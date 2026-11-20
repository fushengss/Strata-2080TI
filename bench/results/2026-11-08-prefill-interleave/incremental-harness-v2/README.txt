True synthetic conversation cache benchmark — incremental v2

This is a NEW immutable package; v1 and its failed real run remain unchanged.
It uses the same v3 engine and single experimental variable as v1:
  OFF STRATA_PIPED_CONCURRENT_MTP=0; ON=1.
  STRATA_PARK_OVERWRITE=0 in BOTH arms, retained in config equality and verified
  in the actual process environment. No overwrite optimization is enabled.

Why v2

The v1 proxy lengthened its original user text, invalidating the last user-turn
checkpoint. Its real first increment was100859 tokens, restored98304 (a valid
periodic checkpoint,6*default16384), and read2555 including1702 old prompt tokens.
Its64-token answer completed correctly; the harness rejected its>=99k contract
and aborted the peers. This failure is preserved, not retroactively accepted.

v2 keeps the original user message intact. Each next input is the ACTUAL history:
  original user, actual visible assistant reply, fixed new user,
  actual next assistant reply, next fixed new user, ...
Replies are synthetic and captured only in memory. They are never substituted
with an invented fixed answer, and never written to JSON/JSONL. Captured content
is checked against its visible-channel hash before it becomes a later message.
Capture is capped at1MiB per reply; requests still have max64 output tokens.
Residents are not captured. Each full request/message hash and previous response
hash/count dependency is recorded and compared exactly across arms. This is a
synthetic conversation continuation, not a measured tool-use agent workflow.

Workload and counts

Eight normal finite requests: one cold roughly100k seed(max64), four distinct
cold roughly10k resident prompts(max4096), then three new user turns, each with
740 fixed synthetic words and an instruction(max64). The independent
incremental-conversation-v2 namespace prevents cache pollution of the old cold
suites. Original calibrated word counts/TASK seed the new inputs; actual prompt
counts are reported, not assumed to be exact100000/101000.

Five stateless count calls precede the seed/residents. Each complete conversation
is counted immediately before its injection, after the prior actual assistant
reply exists. Thus the final3 request bodies are runtime-dependent, not all known
before execution. Their fixed construction algorithm, fixed new-user bytes,
previous actual reply hashes, actual full message/body hashes, preflight counts
and actual API/native usage must agree across arms. Count calls are checked to
not increase generation count. There are no GPU probes, cache clears or refills.

The3 runtime count durations are reported and INCLUDED in mixed-phase and
whole-case wall time. They do not run the model, but are client validation
overhead; do not silently subtract them or label these rates pure decode.
Five-second baseline after C4 readiness, planned8-second injection spacing,
10-second recovery, normal final drain remain unchanged.

Checkpoint contract

The actual model template is frozen as chat_template-active.jinja, SHA256
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041.
Root verified that server40c5 loads the configured tokenizer/chat_template.jinja
and no DeepSeek override applies. The package and actual configured template
must both match this SHA before execution; the actual template is rechecked at
the end. The runner includes the Jinja file in its source freeze.

The thinking-disabled generation header is7 tokens in this verified setup.
The original real seed100006 had a native read endpoint99999. The engine saves
its turn checkpoint before the last im_start generation header. On-arm every
seed/continuation must independently have a native reader/fallback endpoint at
its actual prompt_tokens minus7, checking this boundary assumption at runtime.

Each next request requires cached>=max(99000,previous_prompt_tokens-7), and
0<fresh<=current_actual_prompt_tokens-(previous_prompt_tokens-7).
API cached count, native cache_n/prompt_n and the unique native prompt report
must agree. First continuation also requires a native restored marker at or
beyond that previous turn checkpoint, after the seed completed and four distinct
other prompts were admitted. It cannot pass with the old98304 periodic replay.

This derives the fresh budget from an actual prior turn boundary. It does NOT
assume all previous output tokens or the parked live state can be reused:
template content trim, re-tokenization, hidden empty-think formatting, end tokens
and MTP rows committed beyond the emitted cap can differ. A fresh count above
1300 can be valid if it remains within the proven turn-boundary budget; a cold
read or old periodic checkpoint cannot qualify.

On-arm per-injection reader pairs must lie within that request's fresh token
range, have>=4 active slots and successful read/pump, with positive aggregate
strictly-in-read native BT progress. Multiple message segments are allowed;
small segments need not individually finish a pump. API visible chunks, not
native BT counts, establish actual visible progress. Source safety gates remain.

Correctness, comparison and scope

Normal length must exhaust its cap; normal stop must have0<actual<=cap. Actual
API arithmetic/native predicted_n, visible output, no tools/reasoning/errors and
all4 resident visible coverage through baseline/mixed/recovery remain mandatory.
A delayed HTTP end cannot establish coverage. No post-result coverage relaxation.

Both arms must have the same actual request/message/output/channel hashes,
finish reasons, three logical usage counts, cache/fresh metadata and valid exact
assistant-response dependency chains. The actual template and conversation
client source SHA must match. The comparator requires complete paired runs,
identical planned suite order, finished registered arms and exact config/engine
identities. Different generated replies mean different later work and disqualify
the performance comparison; successful output alone cannot hide cache differences.

Report both whole-case efficiency including seed and post-seed C4+turns rate.
Both include admission/restore, count preflight and final drain, not pure decode.
SSE gaps are client-arrival chunk gaps, not token TBT. No deployment decision is
made by this harness. The other eight suites can qualify in a separate complete
v1 pair; do not compare one v1 arm against one v2 arm or merge their artifact files.

Changed implementation vs v1: incremental_mixed, incremental comparator path,
new conversation_client, expanded incremental CPU tests, and runner inclusion
of the Jinja identity file. Existing finite/stream-client/fixture/core/C8/budget/
BYIELD/admission/stall metrics and their request contracts remain unchanged.
The legacy prefill_check.py is not this runner's execution path.

CPU checks

  python3 -B incremental_offline_tests.py
  python3 -B overlap_profile_tests.py
  python3 -B offline_tests.py
  python3 -B byield_offline_tests.py

The full offline-checks.json additionally records prefill/budget/C8/long/client/
calibrator/stall/cancel tests, AST and dry-run. Incremental fake transport gives
each reply unique text, checks1/3/5/7-message history and correct prior assistant,
scans JSON/JSONL for any captured text, and covers bounded capture, exact chains,
missing restore,98304 replay, bad counts/native evidence/finish, unequal cache,
late HTTP completion and valid larger fresh budgets. Fake rates are not hardware
performance measurements.

Execution — only after operator maintenance exclusivity

Default is dry-run. This package can run incremental alone as a complete on/off
pair, separately from the ongoing eight-suite v1 qualification:

python3 /srv/bench100k-20261108-prefill-interleave/incremental-harness-v2/runner_interleave.py \
  --execute --arms on,off --suites incremental \
  --calibration /srv/bench100k-20261108-prefill-interleave/calibration.json \
  --calibration-sha256 b036e3ad8523dcad87d8dce76ff236a624a6f19c9058313f74ad403bc4c091dd \
  --exe /srv/Strata-decode-head-20261107/engine/strata.prefill-interleave3 \
  --engine-sha256 45eaa612cbdda74d0bcaf6302e37807ae9cd9c4fe8b1c633282564d379cad8b6 \
  --library-sha256 fb67bffc9ece81aa90ee22aa97fed40710c4d57ece6bceeb4ae023f9728a0661 \
  --server-sha256 40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9 \
  --config-sha256 f752a21492dedb98bda4fd4af7551999ed0a30516c8a9e5271f055e17f2a04dd \
  --out /srv/bench100k-20261108-prefill-interleave/NEW-CONVERSATION-PAIR

Runner owns only isolated8092 and its descendants/private0600 configs. It does
not manage production8080 or alter clocks, power, caches or production config.
