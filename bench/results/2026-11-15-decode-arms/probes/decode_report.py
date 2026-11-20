#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/probes/decode_report.py - decode speed and draft acceptance of the live requests against the 2026-11-08..14 baseline.
Run on the serving box (read-only, it only reads the engine log):
    python3 /srv/Strata/decode_report.py [--from-line N]
Default: from the last engine start.  --spec-min-p 0.8 (baseline: 0.5) was deployed 2026-11-15 04:40 CST (engine start at log line 904554); lab tests then ran on the service until it was
put back at 05:30 CST (engine start at log line 911515, README section 8), so lines 904554-911515 hold benchmark requests and `--from-line 911515` is the real traffic since the deployment.
Only requests that generated >= 300 tokens count; "PW2" = the request ran the pipelined windows (the
"strata decode head" line is in its log block), the others decoded serially (a request with a presence/repetition penalty) or in batch slots.
Measured on 2026-11-15 in the lab at 40,050 tokens of context, sampled: +5.8% (code) / +3.8% (prose) tok/s and a higher acceptance with 0.8, see README.md."""
import argparse, re, statistics as st

LOG = "/srv/Strata/strata-orca-q5_k_m.log"
BASE = (204406, 849811)   # engine concurrent-prefill-20261108, 2026-11-08 03:00 CST .. 11-14, 47,480 requests
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
BUCKETS = ((0, 8000), (8000, 32000), (32000, 64000), (64000, 110000), (110000, 10**9))


def load(lines, a, b):
    out, keep = [], False
    for l in lines[a - 1:b]:
        m = PROMPT.search(l)
        if m:
            g = m.groups()
            out.append(dict(prompt=int(g[0]), gen=int(g[4]), ms=int(g[5]), tps=float(g[6]), acc=int(g[7]), drf=int(g[8]), pw=keep))
            keep = False
        elif "strata decode head: CUDA" in l:
            keep = True
    return out


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * p))] if v else float("nan")


def label(lo, hi):
    f = lambda v: "inf" if v >= 10**9 else "%dK" % (v // 1000)
    return "%s-%s" % (f(lo), f(hi))


def report(title, rows):
    L = [r for r in rows if r["gen"] >= 300]
    print("\n%s: %d requests, %d generated >= 300 tokens" % (title, len(rows), len(L)))
    for name, flag in (("PW2", True), ("no PW2", False)):
        b = [r for r in L if r["pw"] == flag]
        if not b:
            print("  %-6s n 0" % name); continue
        print("  %-6s n %5d  tok/s median %5.1f (p10 %5.1f, p90 %5.1f)  acceptance %4.1f%%  prompt median %6d  decode time %.2f h" % (
            name, len(b), st.median(r["tps"] for r in b), q([r["tps"] for r in b], .1), q([r["tps"] for r in b], .9),
            100.0 * sum(r["acc"] for r in b) / max(1, sum(r["drf"] for r in b)), st.median(r["prompt"] for r in b), sum(r["ms"] for r in b) / 3.6e6))
    pw = [r for r in L if r["pw"]]
    print("  PW2 by prompt size:  " + "   ".join("%s: %s (n %d)" % (label(lo, hi), "%.1f" % st.median(r["tps"] for r in pw if lo <= r["prompt"] < hi) if any(lo <= r["prompt"] < hi for r in pw) else "-",
                                                  sum(1 for r in pw if lo <= r["prompt"] < hi)) for lo, hi in BUCKETS))


ap = argparse.ArgumentParser()
ap.add_argument("--from-line", type=int, default=0)
a = ap.parse_args()
lines = open(LOG, errors="replace").read().splitlines()
start = a.from_line or [i + 1 for i, l in enumerate(lines) if "file tier (Linux defaults" in l][-1]
report("baseline 2026-11-08..14 (--spec-min-p 0.5)", load(lines, *BASE))
report("now (log lines %d-%d)" % (start, len(lines)), load(lines, start, len(lines)))
bad = [l for l in lines[start - 1:] if re.search(r"(^|\W)(ERR|illegal memory|Segmentation|abort\(|allocation failed|Xid)", l)]
print("\nerror-like lines since line %d: %d" % (start, len(bad)))
