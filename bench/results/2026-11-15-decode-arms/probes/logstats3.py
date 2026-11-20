#!/usr/bin/env python3
"""read-only: who decodes serially (no pipelined windows)?  prompt sizes and reuse of the two classes, baseline 6.8 days.  run: ssh root@10.100.3.3 python3 - < logstats3.py"""
import re, statistics as st
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
lines = open(LOG, errors="replace").read().splitlines()
out, keep = [], []
for l in lines[204406 - 1:849811]:
    m = PROMPT.search(l)
    if m:
        g = m.groups()
        pw = any("strata decode head: CUDA" in x for x in keep)
        out.append(dict(prompt=int(g[0]), reused=int(g[1]), new=int(g[2]), rms=int(g[3]), gen=int(g[4]), gms=int(g[5]), tps=float(g[6]), acc=int(g[7]), drf=int(g[8]), pw=pw))
        keep = []
    elif "strata decode head: CUDA" in l:
        keep.append(l)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")


for mingen in (100, 300):
    L = [r for r in out if r["gen"] >= mingen]
    for name, b in (("PW2", [r for r in L if r["pw"]]), ("serial", [r for r in L if not r["pw"]])):
        print("gen >= %d  %-7s n %5d | prompt tokens p10 %6.0f med %6.0f p90 %7.0f | new read tokens med %5.0f | generated med %5.0f p90 %5.0f | tok/s med %5.1f | acceptance %4.1f%%" % (
            mingen, name, len(b), q([r["prompt"] for r in b], .1), st.median(r["prompt"] for r in b), q([r["prompt"] for r in b], .9),
            st.median(r["new"] for r in b), st.median(r["gen"] for r in b), q([r["gen"] for r in b], .9), st.median(r["tps"] for r in b),
            100.0 * sum(r["acc"] for r in b) / max(1, sum(r["drf"] for r in b))))
# tok/s of the serial class by generated-length bucket, and the same for PW (decode speed is not flat in length)
print()
for lo, hi in ((100, 300), (300, 700), (700, 1500), (1500, 10**9)):
    row = []
    for name, flag in (("PW2", True), ("serial", False)):
        b = [r for r in out if lo <= r["gen"] < hi and r["pw"] == flag]
        row.append("%s n %4d med %5.1f tok/s" % (name, len(b), st.median(r["tps"] for r in b) if b else float("nan")))
    print("generated %5d-%-6s  %s | %s" % (lo, "inf" if hi >= 10**9 else hi - 1, row[0], row[1]))
