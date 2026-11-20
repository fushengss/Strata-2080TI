#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/harness/deploy_minp.py - write the live service config as the deployed one (final-park-warm4) with --spec-min-p changed.
Run on the serving box:  python3 deploy_minp.py 0.8      then restart with /srv/Strata/restart57.sh (or bounded_restart.sh).
It refuses to run unless the live config is still byte-identical to the deployed copy, changes only the value after --spec-min-p (0.5 -> the argument) and keeps the result as
/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4-minp<value>.  Rollback: cp /srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4 /srv/Strata/strata-orca-q5_k_m.57.json; /srv/Strata/restart57.sh"""
import filecmp, json, shutil, sys
PRISTINE = "/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4"
LIVE = "/srv/Strata/strata-orca-q5_k_m.57.json"
v = sys.argv[1]
if not filecmp.cmp(PRISTINE, LIVE, shallow=False):
    raise SystemExit("the live config is not the deployed copy: not touching it")
cfg = json.load(open(PRISTINE))
a = cfg["args"]
i = a.index("--spec-min-p")
if a[i + 1] != "0.5":
    raise SystemExit("--spec-min-p is %s, expected 0.5" % a[i + 1])
if cfg.get("host") != "0.0.0.0":
    raise SystemExit("host is %s, expected 0.0.0.0" % cfg.get("host"))
a[i + 1] = v
json.dump(cfg, open(LIVE, "w"), indent=2)
open(LIVE, "a").write("\n")
keep = "%s-minp%s" % (PRISTINE, v.replace(".", ""))
shutil.copyfile(LIVE, keep)
print("live config: --spec-min-p %s, host %s, exe %s; copy kept as %s" % (v, cfg["host"], cfg["exe"].split("/")[-2], keep))
