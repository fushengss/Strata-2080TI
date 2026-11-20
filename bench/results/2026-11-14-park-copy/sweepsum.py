#!/usr/bin/env python3
"""bench/results/2026-11-14-park-copy/sweepsum.py - table of read time by append size from readsweep.py runs:  python3 sweepsum.py chunk2 chunk4"""
import json, re, statistics as st, sys
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
tags = sys.argv[1:]
data = {}
for t in tags:
    reqs = json.load(open(t + ".json"))["requests"]
    lines = [m for m in (PROMPT.search(l) for l in open(t + ".log", errors="replace")) if m]
    rows = []
    for m in lines:
        rows.append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
    data[t] = (reqs, rows)
sizes = sorted({r["size"] for t in tags for r in data[t][0] if r["kind"] == "append"})
colds = sorted({r["size"] for t in tags for r in data[t][0] if r["kind"] == "cold"})
print("%-26s" % "read ms (median)" + "".join("%12s" % t for t in tags))
for size in sizes:
    row = []
    for t in tags:
        vals = [ms for reused, new, ms in data[t][1] if reused > 10000 and abs(new - size) <= 0.2 * size + 60]
        row.append("%d (n=%d)" % (st.median(vals), len(vals)) if vals else "-")
    print("append ~%-17d" % size + "".join("%12s" % x for x in row))
for size in colds:
    row = []
    for t in tags:
        vals = [(ms, new) for reused, new, ms in data[t][1] if reused == 0 and abs(new - size) <= 0.2 * size + 60 and new < 30000]
        row.append("%d (%.0f t/s)" % (st.median(v[0] for v in vals), st.median(v[1] / (v[0] / 1000) for v in vals)) if vals else "-")
    print("cold ~%-20d" % size + "".join("%12s" % x for x in row))
h = {t: [r["reply_sha1"] for r in data[t][0]] for t in tags}
same = sum(1 for i in range(min(len(v) for v in h.values())) if len({h[t][i] for t in tags}) == 1)
print("\nreplies identical across variants: %d of %d requests" % (same, min(len(v) for v in h.values())))
