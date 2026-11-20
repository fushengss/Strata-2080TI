#!/usr/bin/env python3
"""read-only: decode-side statistics of the live engine log.  run: ssh root@10.100.3.3 python3 - < logstats.py"""
import re, statistics as st, collections
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
lines = open(LOG, errors="replace").read().splitlines()
print("log lines", len(lines))
starts = [i + 1 for i, l in enumerate(lines) if "file tier (Linux defaults" in l]
bounds = starts + [len(lines) + 1]
segs = []
for a, b in zip(bounds, bounds[1:]):
    n = sum(1 for l in lines[a - 1:b - 1] if "strata serve: prompt" in l)
    segs.append((a, b - 1, n))
print("engine segments (first line, last line, requests):", segs[-8:])
print("\nline vocabulary of the last 20000 lines:")
voc = collections.Counter(re.sub(r"\d+(\.\d+)?", "N", " ".join(l.split()[:5])) for l in lines[-20000:])
for k, v in voc.most_common(30):
    print("  %6d  %s" % (v, k[:100]))
print("\nsample request line:")
print([l for l in lines[-3000:] if "strata serve: prompt" in l][-1][:400])
print("\nlines mentioning accept/draft (3 samples):")
n = 0
for l in lines[-20000:]:
    if re.search(r"accept|drafted", l, re.I):
        print("  " + l[:300]); n += 1
        if n >= 3:
            break


def reqs(a, b):
    out = []
    for l in lines[a - 1:b]:
        m = PROMPT.search(l)
        if m:
            g = m.groups()
            out.append((int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]), float(g[6])))
    return out


def report(name, R):
    print("\n==== %s: %d requests" % (name, len(R)))
    long_ = [r[6] for r in R if r[4] >= 300]
    if not long_:
        print("no request with >= 300 generated tokens"); return
    steady = st.median(long_)
    print("steady decode rate (median tok/s of the %d requests with >= 300 generated): %.1f" % (len(long_), steady))
    edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 10**9]
    print("%-10s %7s %7s %9s %9s %9s %9s %10s" % ("generated", "n", "share", "med ms", "med tok/s", "p10 tok/s", "hours", "overhead h"))
    tms = ttk = 0
    for lo, hi in zip(edges, edges[1:]):
        b = [r for r in R if lo <= r[4] < hi]
        if not b:
            continue
        ms = sum(r[5] for r in b); tk = sum(r[4] for r in b)
        ov = ms - tk / steady * 1000
        tms += ms; ttk += tk
        t10 = sorted(r[6] for r in b)[len(b) // 10]
        print("%-10s %7d %6.1f%% %9.0f %9.1f %9.1f %9.2f %10.2f" % ("%d-%s" % (lo, "inf" if hi >= 10**9 else hi - 1), len(b), 100.0 * len(b) / len(R),
              st.median(r[5] for r in b), st.median(r[6] for r in b), t10, ms / 3.6e6, ov / 3.6e6))
    print("all with >= 1 generated: %.2f h generating, %.2f h would be needed at the steady rate -> %.2f h overhead (%d tokens)" %
          (tms / 3.6e6, ttk / steady / 3600, (tms - ttk / steady * 1000) / 3.6e6, ttk))
    pts = [(r[4], r[5]) for r in R if 1 <= r[4] <= 300]
    if len(pts) > 20:
        mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
        sxx = sum((p[0] - mx) ** 2 for p in pts); sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        b1 = sxy / sxx; b0 = my - b1 * mx
        print("OLS generated_ms = %.0f ms + %.2f ms/token x generated (requests with 1..300 tokens, n=%d): the per-request fixed part is %.0f ms, slope = %.1f tok/s" % (b0, b1, len(pts), b0, 1000 / b1))
    # decode rate against how much of the context was newly read in that request (a prefill of the same request ends before decode starts)
    for lo, hi in ((0, 100), (100, 2000), (2000, 10**9)):
        b = [r[6] for r in R if r[4] >= 100 and lo <= r[2] < hi]
        if len(b) >= 5:
            print("  requests with >= 100 generated and %d..%s new prompt tokens: n %d median %.1f tok/s p10 %.1f" % (lo, "inf" if hi >= 10**9 else hi, len(b), st.median(b), sorted(b)[len(b) // 10]))


big = max(segs, key=lambda s: s[2])
report("baseline segment %s (old engine)" % (big,), reqs(big[0], big[1]))
last = segs[-1]
report("last segment %s" % (last,), reqs(889697, last[1]))
