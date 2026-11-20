#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/setcfg2.py - write the live service config for one decode test arm, always from the pristine deployed copy.
Run on the serving box:  python3 setcfg2.py <variant>      (variant = parts joined with '+', or 'restore')
  ctl     the deployed settings, only "host" becomes 127.0.0.1 (the service is reachable from this PC alone while a test runs)
  head1   STRATA_DECODE_HEAD_GPU=2 (one target card: sampled PW2 requests may use the offloaded head), STRATA_PIPED_TARGET_HEAD removed (it needs two)
  cost    STRATA_SPEC_COST=1 (cost-aware MTP window, a GB10 default upstream)
  fuse    STRATA_B6_MMVQ_ROWS=1 STRATA_GU_SWIGLU=1 STRATA_FUSE_DECODE=1 (exact decode kernels that are default on sm_86 / sm_60)
  restore copy the pristine deployed config back (host 0.0.0.0)
The pristine copy is /srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4 (the 2026-11-15 02:18 CST deployment).  Secrets are never printed."""
import json, shutil, sys
PRISTINE = "/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4"
LIVE = "/srv/Strata/strata-orca-q5_k_m.57.json"
v = sys.argv[1]
cfg = json.load(open(PRISTINE))
if v == "restore":
    shutil.copyfile(PRISTINE, LIVE)
    print("restored the deployed config: host %s, exe %s" % (cfg.get("host"), cfg["exe"].split("/")[-2]))
    sys.exit(0)
env = cfg["env"]
cfg["host"] = "127.0.0.1"
for p in v.split("+"):
    if p == "ctl":
        pass
    elif p == "head1":
        env["STRATA_DECODE_HEAD_GPU"] = "2"
        env.pop("STRATA_PIPED_TARGET_HEAD", None)
    elif p == "cost":
        env["STRATA_SPEC_COST"] = "1"
    elif p == "fuse":
        env["STRATA_B6_MMVQ_ROWS"] = "1"
        env["STRATA_GU_SWIGLU"] = "1"
        env["STRATA_FUSE_DECODE"] = "1"
    elif p == "timing":     # STRATA_DECODE_TIMING=1: one extra stderr line group per request - where the decode time goes (windows, speculative / rolled back, per-class ms)
        env["STRATA_DECODE_TIMING"] = "1"
    elif p == "theta":      # the per-request pipelined-window switch file (sweep.py theta): pw=<0|1|2> theta=<gate>, read at every request
        env["STRATA_DECODE_TIMING"] = "1"
        env["STRATA_PIPELINE_DEBUG"] = "1"
        env["STRATA_PIPELINE_SWITCH"] = "/tmp/claude-decode/pw.switch"
        open("/tmp/claude-decode/pw.switch", "w").write("pw=2 theta=0.2\n")
    else:
        raise SystemExit("unknown part " + p)
json.dump(cfg, open(LIVE, "w"), indent=2)
open(LIVE, "a").write("\n")
print("config set for %s: host %s, exe %s, extra env %s" % (v, cfg["host"], cfg["exe"].split("/")[-2],
      {k: env[k] for k in sorted(env) if k in ("STRATA_DECODE_HEAD_GPU", "STRATA_SPEC_COST", "STRATA_B6_MMVQ_ROWS", "STRATA_GU_SWIGLU", "STRATA_FUSE_DECODE", "STRATA_PIPED_TARGET_HEAD", "STRATA_DECODE_TIMING", "STRATA_PIPELINE_DEBUG", "STRATA_PIPELINE_SWITCH")}))
