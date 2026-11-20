#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/harness/setcfg3.py - the live service config for a test arm, from the config deployed on 2026-11-15 04:40 CST (--spec-min-p 0.8).
Run on the serving box:  python3 setcfg3.py <variant>        variants: ctl | par16 | restore   (joined with '+')
  ctl      the deployed settings, only "host" becomes 127.0.0.1 (the service is reachable from this PC alone while a test runs)
  par16    "parallel": 16 (the engine's --batch 16; --batch-groups stays 8, so two slots per pipeline group)
  restore  copy the deployed config back (host 0.0.0.0)
The deployed copy is /srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4-minp08; the one before it (--spec-min-p 0.5) is ...final-park-warm4 (setcfg2.py's base).
Secrets are never printed."""
import filecmp, json, shutil, sys
BASE = "/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4-minp08"
LIVE = "/srv/Strata/strata-orca-q5_k_m.57.json"
v = sys.argv[1]
cfg = json.load(open(BASE))
if v == "restore":
    shutil.copyfile(BASE, LIVE)
    print("restored the deployed config: host %s, parallel %s, --spec-min-p %s, identical to the copy: %s" % (
        cfg.get("host"), cfg.get("parallel"), cfg["args"][cfg["args"].index("--spec-min-p") + 1], filecmp.cmp(BASE, LIVE, shallow=False)))
    sys.exit(0)
cfg["host"] = "127.0.0.1"
for p in v.split("+"):
    if p == "ctl":
        pass
    elif p == "par16":
        cfg["parallel"] = 16
    else:
        raise SystemExit("unknown part " + p)
json.dump(cfg, open(LIVE, "w"), indent=2)
open(LIVE, "a").write("\n")
a = cfg["args"]
print("config set for %s: host %s, parallel %s, --batch-groups %s, --spec-min-p %s, exe %s" % (
    v, cfg["host"], cfg.get("parallel"), a[a.index("--batch-groups") + 1], a[a.index("--spec-min-p") + 1], cfg["exe"].split("/")[-2]))
