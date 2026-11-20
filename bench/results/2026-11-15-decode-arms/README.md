# 2026-11-15 decode: where the time goes, and which switches move it

Box: 8x RTX 2080 Ti 22 GiB (sm_75, no P2P), driver 580.178.04, CUDA 13.0, 2x Xeon E5-2650 v4, Linux. Engine `park-warm4-20261115` (0.1.57 + this fork),
Qwen3.8-Flash-Next Uncensored Q5_K_M (orca pack), layer split 6 layers per card, `--pipeline-windows 2 --spec 3 --spec-min-p 0.5 --batch-groups 8`,
`parallel 8`, MTP drafter on CUDA7, decode head on CUDA2+4 and MTP head on CUDA3+5 (greedy only).
All arm numbers: 2026-11-15 03:12-04:00 CST, the service bound to 127.0.0.1 (nobody else calling), engine restarted per arm, speeds read from the
engine's own request line. Every request carries a seed, so a sampled reply is the same text in every arm that changes no arithmetic (all of them were:
the reply hashes in `raw/*.json` are equal across arms and across the values of a sweep).

## 1. The host and the clocks are not the limit (quiet box, `probes/probe.py`, `probes/probe2.py`)

| What | Measured while decoding 500 tokens |
| --- | --- |
| GPU state | P2, memory clock 6800 MHz (max 7000), SM 1860-1890 MHz (locked 2000), 11-24% busy per card |
| CPU 0 (the host thread's core) | 2.89 GHz (turbostat Bzy_MHz 2892-2895 of 2900 max), C6 0%, 100% busy, governor schedutil |
| host thread (`strata-host`, pinned to CPU 0) | 99-100% of a core; perf: 62% libcuda, 20% libcudart, 9% libc, 5% engine code |

The host thread spins in the CUDA driver waiting for the cards. A governor, C-state or clock change has nothing to win.

## 2. Anatomy of a window (`STRATA_DECODE_TIMING=1`, arm `timing`)

A window that starts after the previous verdict ("fresh") takes ~25.5 ms: eight stages of ~3.1 ms. With `--pipeline-windows 2` a window launched ahead of the
verdict ("speculative") that proves right ("on the path") adds ~12.5 ms; one that proves wrong is rolled back.

| Request (500 tokens) | windows | ms/window | tokens/window | speculative | on the path | rolled back |
| --- | --- | --- | --- | --- | --- | --- |
| code, greedy (the warm-up request) | 196 | 17.3 | 2.6 | 172 | 124 | 48 |
| code, sampled (5 seeds) | 191-209 | 17.8-18.7 | 2.4-2.6 | 166-179 | 115-128 | 40-59 |
| prose, sampled (3 of the 5 seeds were read) | 243-263 | 19.1-20.7 | 1.9-2.1 | 204-220 | 75-105 | 109-138 |

Greedy vs sampled (temperature 1.0, top_p 0.95, top_k 20): draft acceptance 83.6 vs 80.4% (code), 73.4 vs 65.0% (prose); solo speed 147.6 vs 134.7 and 117.9 vs 98.4 tok/s.
Only greedy requests use the two-shard decode head and the MTP head shards (windows 17.2 / 18.3 ms greedy, 18.5 / 19.8 ms sampled).

## 3. Arms (same 10 sampled seeds, 4 greedy runs, 3 rounds of 4 at once, one cold 19K read)

| Arm | sampled code | sampled prose | greedy code | greedy prose | 4 at once (tok/s, 3 rounds) | cold read tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| ctl (as deployed) | 134.7 | 98.4 | 147.6 | 117.9 | 226 236 235 | 3025 |
| head1: `STRATA_DECODE_HEAD_GPU=2` (one card; sampled requests now offload: 18.5 / 19.8 ms/window), no `STRATA_PIPED_TARGET_HEAD` | 134.6 | 98.3 | 144.8 | 115.7 | 226 238 239 | 3014 |
| cost: `STRATA_SPEC_COST=1` | 134.8 | 98.2 | 147.9 | 118.2 | 224 238 236 | 3024 |
| fuse: `STRATA_B6_MMVQ_ROWS=1 STRATA_GU_SWIGLU=1 STRATA_FUSE_DECODE=1` | 135.6 | 98.8 | 148.3 | 118.7 | 224 237 237 | 3018 |

`STRATA_SM75_PERF=1` (the whole Turing kernel series) is already on in production; `src/core/nv_defaults.cpp` has default stacks for sm_86, sm_60 and sm_121 but none for sm_75, and the arms above are
what those stacks add that production lacks (`STRATA_HC_UPMIX` is read only by the prompt path, `STRATA_ADAPT_*` and `STRATA_CPU_IQ_ALL_EXACT1` belong to the expert tiers, all resident here). On sampled requests
a single decode-head card and `STRATA_SPEC_COST` change nothing and the fusions give +0.4..0.7%; the one-card head is 2% slower for greedy requests.

## 4. Per-request sweeps on one server (`harness/sweep.py`, `harness/longctx.py`; 3 seeds per cell, medians)

`--pipeline-windows` gate theta and PW on/off (`STRATA_PIPELINE_DEBUG=1`, `STRATA_PIPELINE_SWITCH` file):

| setting | code tok/s | prose tok/s |
| --- | --- | --- |
| theta 0.0 / 0.1 / 0.2 (default) | 131.1 / 131.1 / 131.8 | 100.2 / 100.2 / 99.8 |
| theta 0.35 / 0.5 / 0.7 / 0.9 | 131.2 / 129.7 / 128.0 / 123.1 | 98.7 / 94.5 / 90.3 / 87.0 |
| pw=0 (the serial loop) | 97.5 | 82.5 |

The default gate is at the optimum; pipelined windows are worth +35% (code) and +21% (prose) for a sampled request.

Speculative `min_p` (request body `{"strata_tune": {"spec_min_p": v}}`, same as `--spec-min-p`; default 0.5):

| prompt | min_p | code tok/s | prose tok/s | code ms/window | prose ms/window |
| --- | --- | --- | --- | --- | --- |
| short (~50 tokens) | 0.5 | 131.6 | 99.4 | 18.30 | 19.82 |
| | 0.7 | 132.6 | 102.5 | 17.43 | 18.12 |
| | 0.9 | 133.4 | 101.9 | 16.61 | 15.97 |
| 40,050 tokens | 0.5 | 110.1 | 89.9 | 20.83 | 21.53 |
| | 0.8 | **116.5** | **93.3** | 18.45 | 18.66 |
| | 0.9 | 115.2 | 93.4 | 17.53 | 17.27 |
| | 0.95 | 113.6 | 93.1 | 16.91 | 16.43 |
| | 0.98 | 113.0 | 92.9 | 16.19 | 15.72 |

A 40K-token context costs ~2.5 ms per window (110.1 vs 131.6 tok/s for code). Lower values (0.1, 0.3) are worse than the default (127-128 / 92-94).

## 5. The live traffic, from the log of 2026-11-08 .. 11-14 (47,480 requests, engine `concurrent-prefill-20261108`; `probes/logstats*.py`)

| Requests that generated >= 300 tokens | n | share of decode time | median tok/s (p10 / p90) | acceptance | prompt median |
| --- | --- | --- | --- | --- | --- |
| pipelined windows (PW2) | 1,993 | 79% (6.24 h) | 100.1 (84.9 / 125.0) | 71.1% | 107,522 tokens |
| serial (no PW2; small prompts, ~570 tokens out; presence_penalty requests decode serially by design) | 951 | 21% (1.62 h) | 91.9 (88.3 / 96.7) | 83.3% | 3,686 tokens |

0 of 2,964 PW2 requests of >= 100 tokens offloaded a window to the decode head, so the agent traffic is sampled. Suffix (n-gram) drafts supply 5.5% of the accepted drafts (78% accepted).
Per-request fixed cost of a decode is ~10 ms (OLS over 44,542 requests), so short outputs are not slow; 80.4% of all requests generate exactly one token.

## 6. What this leaves

- `--spec-min-p 0.8`: +3.8% prose, +5.8% code at 40K tokens, +1-3% on a short prompt, same replies. **Deployed 2026-11-15 04:40 CST** (`harness/deploy_minp.py 0.8`, ready 04:42:45;
  the live config differs from the earlier copy `.final-park-warm4` in that one value). Tested on single requests when deployed; the concurrent runs of sections 7-8 came after (4 at once 238-248 tok/s
  aggregate against 226-236 with 0.5, 8 at once 270-282). Check the live effect with `python3 /srv/Strata/decode_report.py --from-line 911515` (`probes/decode_report.py`; 911515 is the engine put back
  at 05:30 CST after the lab tests, the lines 904554-911515 hold their requests) against the 11-08..14 baseline (PW2 requests: median 100.1 tok/s, acceptance 71.1%); first numbers in section 9.
- Two-shard head and MTP head shards for sampled requests: the greedy path is ~1.3-1.5 ms/window (7-8%) faster, the one-card variant gives none. Engine change.
- A third window in flight (`--pipeline-windows` is capped at 2 in `generate.cpp`): rollbacks cost almost nothing (theta sweep), so the 25 ms traversal is what to cut. Large change.
- Fewer, fused kernels per stage (3.1 ms per stage of 6 layers). The 2026-10 profile put perfect kernels at 1.4x for that configuration (roadmap section 四), +10-15% realistic. Large change.
- Penalties under PW2: +21-35% for requests that carry `presence_penalty` (they decode serially). Engine change, or a config default of 0 for non-thinking requests (changes the replies).

Rollback of the deployment (back to `--spec-min-p 0.5`): `python3 harness/setcfg2.py restore` copies `/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4` back (host 0.0.0.0), then `/srv/Strata/restart57.sh`.
Putting the deployed config (0.8) back after a test arm: `harness/restore_production.sh` (copies `.final-park-warm4-minp08`, then the same restart).

## 7. After the deployment of `--spec-min-p 0.8` (2026-11-15 05:00 CST, service bound to 127.0.0.1, `parallel 8`, engine `park-warm4-20261115`)

Standard suite `/srv/abq57.py` (live arm, one run per cell, greedy and no penalties unless noted) against the best lab arm of 2026-11-07 (`c1_final`: engine 0.1.57 candidate, before the fork's
concurrent-prefill build, the park work and `--spec-min-p 0.8`). `mix` = a greedy answer is the same text when three other requests arrive mid-stream (both rounds: True, as before).

| Test | now | 11-07 `c1_final` |
| --- | --- | --- |
| solo_c: 3 essays x 256 tokens, tok/s | 97.7 / 101.6 / 94.3 | 87.8 / 93.2 / 87.5 |
| code_c: 3 x 400 tokens, tok/s | 152.3 / 152.5 / 152.4 | 140.3 / 139.9 / 139.6 |
| agent: thinking on, model-card sampling, 700 tokens x 4, tok/s | 116.4 / 122.2 / 109.9 / 120.5 | 100.7 / 99.8 / 90.3 / 109.5 |
| four_c / eight_c at once, aggregate tok/s | 203.0 / 269.9 | 172.7 / 225.8 |
| docq (30K, 60K, 12K characters of a document): TTFT s, decode tok/s | (5.83, 113.5) (6.63, 113.2) (3.03, 117.2) | (7.13, 103.6) (7.03, 107.8) (3.45, 110.3) |
| long64c: fresh 64K-token prompt, TTFT s, decode tok/s (2 runs) | (10.87, 133.7) (11.12, 134.5) | (11.07, 125.1) (12.36, 125.1) |
| fresh 8K / 24K / 64K prompt: TTFT s (decode tok/s*) | 5.41 (68.1) / 7.00 (69.0) / 10.95 (65.7) | 5.62 (71.5) / 7.33 (67.1) / 11.65 (66.1) |
| turn2: second turn of a 24K conversation, TTFT s | 0.26 | 0.32 |
| app3k: 3K conversation + 900 new tokens, TTFT s (3 turns) | 1.08 / 1.10 / 1.11 | 1.09 / 1.18 / 1.18 |
| app120k: first read of 120K tokens, s; then +500 new tokens, TTFT s (3 turns) | 16.76; 1.07 / 1.11 / 1.14 | 17.24; 1.13 / 1.17 / 1.24 |

\* the decode after a `fresh` prompt runs with the server's non-thinking default sampling (presence_penalty 1.5), so it is the serial loop; it is not comparable with the greedy rows.
The gain is the sum of three changes (the fork's concurrent-prefill build in production since 11-08, the park work of 11-15, `--spec-min-p 0.8`), not of the last one alone.

Sampled decode and 40K context, same seeds and prompts as section 3 (`harness/armtest.py`, `harness/longctx.py`), the `ctl` arm of 03:12 CST against 05:00 CST (replies identical):

| | before (`--spec-min-p 0.5`) | after (0.8) |
| --- | --- | --- |
| sampled code, 5 seeds, tok/s median (acceptance) | 134.7 (80.4%) | 137.9 (87.1%) |
| sampled prose | 98.4 (65.0%) | 102.7 (80.1%) |
| greedy code tok/s (ms/window) | 147.6 (17.2) | 150.9 (15.8) |
| greedy prose | 117.9 (18.3) | 119.8 (15.4) |
| 4 at once, aggregate tok/s, 3 rounds | 226 / 236 / 235 | 238 / 248 / 240 |
| 40,050-token context, sampled code / prose | 110.1 / 89.9 | 116.2 / 93.4 (acceptance 85.7% / 79.0%) |
| cold 19K read, tok/s | 3025 | 3030 |

## 8. Many requests at once (`harness/conc.py`, 256-token essays, streamed; `parallel 8`, `--batch-groups 8`)

| N at once | sampling | wall s | aggregate tok/s | first token median / max s | decode per request median / min tok/s |
| --- | --- | --- | --- | --- | --- |
| 8 | greedy | 7.3 | 281.5 | 1.08 / 1.77 | 42.6 / 38.2 |
| 8 | sampled | 7.4 | 278.3 | 0.85 / 1.60 | 42.3 / 37.5 |
| 16 (8 wait for a slot) | greedy | 14.4 | 283.9 | 4.16 / 8.92 | 40.4 / 31.9 |
| 16 (8 wait for a slot) | sampled | 14.6 | 281.2 | 4.30 / 8.66 | 39.5 / 30.9 |
| 8 decodes of 400 tokens + 8 fresh 8K prompts 2 s later | sampled / greedy | 44.4 | decodes 41.9 tok/s each | reads: 26.7 / 41.9 | |

The aggregate is flat from 4 at once (203-248) through 8 (278-282) to 16 queued (281-284); the free VRAM per card never moved during these runs (918 MiB on CUDA7 at the lowest).
Free VRAM per card at 8 slots after the suite: 3066 / 2796 / 3366 / 3230 / 3358 / 3226 / 3876 / 918 MiB. Demand seen on 2026-10-28 (roadmap section 八, `parallel 8` Q8_0 service): never more than 7 requests at once in a day, 3.5 on average in the busiest hour.

`parallel 16` (`--batch-groups 8`, two slots per group; `raw/conc_par16.json`): the engine starts it. It reports 16 slot sessions (0.15 GiB on CUDA0/2/4/6, 0.21 GiB on CUDA1/3/5/7 each), 16 slot drafters on CUDA7
(49 MiB each), 16 private MTP-head workspaces and batch windows of up to 16 sequences; the experts stay 100% resident; `/v1/status` says serving 16 / requested 16; the start took 260 s (140 s at 8 slots).

| N at once | sampling | wall s | aggregate tok/s | first token median / max s | decode per request median / min tok/s |
| --- | --- | --- | --- | --- | --- |
| 8 | greedy (the first test after the start: graphs still being captured) | 8.2 | 248.8 | 1.74 / 2.76 | 41.6 / 35.2 |
| 8 | sampled | 7.4 | 277.6 | 0.85 / 1.64 | 41.7 / 37.5 |
| 16 | greedy | 12.7 | 323.1 | 2.14 / 4.28 | 25.5 / 21.3 |
| 16 | sampled | 12.3 | 332.9 | 1.44 / 3.47 | 25.3 / 22.1 |
| 8 decodes of 400 tokens + 8 fresh 8K prompts 2 s later | sampled / greedy | 39.8 | decodes 10.4 tok/s each (8 slots: 41.9) | reads: 16.3 / 29.0 (8 slots: 26.7 / 41.9) | |

Against `parallel 8`: 16 at once finish in 12.3-12.7 s instead of 14.4-14.6 s (aggregate +14% greedy, +18% sampled), the first token of the 16 comes after 1.4-2.1 s (median) instead of 4.2-4.3 s and the slowest
after 3.5-4.3 s instead of 8.7-8.9 s; 8 at once are unchanged (277.6 against 278.3 tok/s sampled). With reads beside decodes the decodes drop from 41.9 to 10.4 tok/s per request while the 8 reads run (the reads
start at once instead of waiting for the 8 slots: first token 16 s instead of 27 s) and the whole mix is 10% shorter. No request failed and the engine log has no error line.
Free VRAM per card, the lowest seen during these runs: 1994 / 3974 / 2662 / 1994 / 2660 / 1990 / 3044 / 2280 MiB (a fresh start and 8K prompts only; not comparable with the 8-slot numbers above, which were
taken after the 64K / 120K tests).
**Not measured:** 16 slots with 60-120K-token conversations (the size of the live agents' contexts). The VRAM margin there is the open question: CUDA7 had 918 MiB free at 8 slots after the long tests, and 8 more
slots take another 0.15-0.21 GiB per card. The ssh session that ran the 16-slot test was cut by a drop of the EasyConnect VPN; the results had been saved on the box and were read after the VPN came back
(memory note `feedback-test-window-watchdog`).

## 9. First live numbers after the restore (2026-11-15 10:52 CST) and the decision on 16 slots

The agents (10.100.3.26) were back: 461 requests in log lines 911515-917940 (the service put back at 05:30-05:39 CST: `--spec-min-p 0.8`, `parallel 8`, host 0.0.0.0, the live config equal to `.final-park-warm4-minp08`),
the log last written 6 minutes before the check, 0 error-like lines. `python3 /srv/Strata/decode_report.py --from-line 911515`, requests of >= 300 generated tokens, against the 2026-11-08..14 baseline
(engine `concurrent-prefill-20261108`, `--spec-min-p 0.5`, 47,480 requests; now: engine `park-warm4-20261115`):

| | now (911515-917940) | baseline 11-08..14 |
| --- | --- | --- |
| PW2 requests | 30 | 1,993 |
| PW2 tok/s median (p10 / p90) | 96.2 (91.6 / 116.8) | 100.1 (84.9 / 125.0) |
| PW2 acceptance | 79.8% | 71.1% |
| PW2 prompt median | 51,397 | 107,522 |
| PW2 tok/s by prompt size 8-32K / 32-64K / 64-110K / 110K+ (n) | 92.8 (8) / 97.3 (17) / 115.2 (2) / 101.7 (3) | 104.8 (68) / 95.1 (285) / 100.3 (671) / 101.1 (948) |
| serial requests | 56 | 951 |
| serial tok/s median | 91.1 | 91.9 |
| serial acceptance | 92.2% | 83.3% |

The acceptance went up as in the lab (section 7). The speed shows no difference yet: 30 PW2 requests are few, their contexts are younger than the baseline's (prompt median 51K against 107.5K: the restarts of
04:40-05:39 CST flushed the conversation cache), and the lab gain at 40K (+4-6%) is smaller than the spread of the live requests (p10-p90 85-125 tok/s). Run the same command again after a few days; a verdict
needs a few hundred PW2 requests. Roll back to 0.5 (section 6) if the acceptance does not hold or the batch speeds drop.

**Decision on `parallel 16`** (user, 2026-11-15 about 10:50 CST, asked whether to test 16 slots with 60-120K-token conversations: no): the service stays at `parallel 8`, the 16-slot test with live-sized
contexts is not run and no further test window is planned.
