#!/usr/bin/env python3
"""read-only: how many live requests decode through the pipelined windows (the 'strata decode head: ... PW2 windows' line) and how fast each class is.
run: ssh root@10.100.3.3 python3 - < logstats2.py"""
import re, statistics as st
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
DEC = re.compile(r"strata decode head: CUDA(\d+), (\d+) PW2 windows offloaded")
lines = open(LOG, errors="replace").read().splitlines()


def classify(a, b):
    out, keep = [], []
    for l in lines[a - 1:b]:
        m = PROMPT.search(l)
        if m:
            g = m.groups()
            pw = None
            batch = False
            for x in keep:
                d = DEC.search(x)
                if d:
                    pw = int(d.group(2))
                if "strata batch: slot" in x:
                    batch = True
            out.append(dict(prompt=int(g[0]), reused=int(g[1]), new=int(g[2]), rms=int(g[3]), gen=int(g[4]), gms=int(g[5]), tps=float(g[6]),
                            acc=int(g[7]), drf=int(g[8]), pw=pw, batch=batch))
            keep = []
        elif "strata decode head: CUDA" in l or "strata batch: slot" in l:
            keep.append(l)
    return out


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")


def report(name, R, mingen):
    print("\n==== %s: %d requests; requests that generated >= %d tokens:" % (name, len(R), mingen))
    L = [r for r in R if r["gen"] >= mingen]
    cls = {"pipelined windows (PW2)": [r for r in L if r["pw"] is not None],
           "batch slots (no PW)": [r for r in L if r["pw"] is None and r["batch"]],
           "solo serial (no PW)": [r for r in L if r["pw"] is None and not r["batch"]]}
    tot = sum(r["gms"] for r in L)
    for k, b in cls.items():
        if not b:
            print("  %-26s n 0" % k); continue
        ms = sum(r["gms"] for r in b); tk = sum(r["gen"] for r in b)
        acc = sum(r["acc"] for r in b) / max(1, sum(r["drf"] for r in b))
        line = "  %-26s n %5d (%4.1f%% of requests, %4.1f%% of decode time %.2f h)  tok/s median %5.1f p10 %5.1f p90 %5.1f  mean(tokens/ms) %5.1f  acceptance %4.1f%%" % (
            k, len(b), 100.0 * len(b) / len(L), 100.0 * ms / tot, ms / 3.6e6, st.median(r["tps"] for r in b), q([r["tps"] for r in b], 0.1), q([r["tps"] for r in b], 0.9), 1000.0 * tk / ms, 100 * acc)
        print(line)
        if k.startswith("pipelined"):
            w = [r for r in b if r["pw"] > 0]
            if w:
                print("      per window: %.1f ms/window median, %.2f tokens/window median (n %d)" % (st.median(r["gms"] / r["pw"] for r in w), st.median(r["gen"] / r["pw"] for r in w), len(w)))
            else:
                print("      (the decode-head counter is 0 in every one of these requests: the line only marks that the pipelined path ran)")


for (name, a, b) in (("baseline 6.8 days (old engine)", 204406, 849811), ("last segment (park-warm4, my tests included)", 889110, len(lines))):
    R = classify(a, b)
    report(name, R, 100)
    report(name, R, 300)
