#!/usr/bin/env python3
"""read-only: how much do the suffix (n-gram lookup) drafts contribute in live traffic?  baseline 6.8 days.  run: ssh root@10.100.3.3 python3 - < logstats4.py"""
import re, statistics as st
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
SFX = re.compile(r"strata serve: suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted")
lines = open(LOG, errors="replace").read().splitlines()
reqs = []
sfx = None
for l in lines[204406 - 1:849811]:
    m = SFX.search(l)
    if m:
        sfx = tuple(int(x) for x in m.groups())
        continue
    m = PROMPT.search(l)
    if m:
        g = m.groups()
        reqs.append(dict(gen=int(g[4]), gms=int(g[5]), tps=float(g[6]), acc=int(g[7]), drf=int(g[8]), sfx=sfx))
        sfx = None
L = [r for r in reqs if r["gen"] >= 300]
S = [r for r in L if r["sfx"]]
print("requests with >= 300 generated: %d; with a suffix-drafts line: %d (%.1f%%)" % (len(L), len(S), 100.0 * len(S) / len(L)))
tw = sum(r["sfx"][0] for r in S); ta = sum(r["sfx"][1] for r in S); td = sum(r["sfx"][2] for r in S)
print("suffix drafts in those: %d windows, %d of %d drafted tokens accepted (%.1f%%); that is %.1f%% of all accepted drafts of the %d requests" %
      (tw, ta, td, 100.0 * ta / max(1, td), 100.0 * ta / max(1, sum(r["acc"] for r in S)), len(S)))
print("windows per request with suffix drafts: median %.0f, generated tokens per request median %.0f" % (st.median(r["sfx"][0] for r in S), st.median(r["gen"] for r in S)))
for name, b in (("with suffix drafts", S), ("without", [r for r in L if not r["sfx"]])):
    print("  %-20s n %5d  tok/s median %5.1f  mean(tokens/ms) %5.1f  acceptance %4.1f%%" % (name, len(b), st.median(r["tps"] for r in b),
          1000.0 * sum(r["gen"] for r in b) / sum(r["gms"] for r in b), 100.0 * sum(r["acc"] for r in b) / max(1, sum(r["drf"] for r in b))))
