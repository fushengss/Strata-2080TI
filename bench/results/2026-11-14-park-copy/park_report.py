#!/usr/bin/env python3
"""bench/results/2026-11-14-park-copy/park_report.py - the live engine log against the 2026-11-08..14 baseline (47,464 requests, engine concurrent-prefill-20261108).

Run on the serving box (read-only, it only reads the log):
    python3 park_report.py [--from-line N] [--to-line M]
By default it reads from the last engine start ("file tier (Linux defaults" line) to the end.  Lines of my own test runs are in the log too: after the
2026-11-15 02:18 CST deployment they end at line 889696, so `--from-line 889697` is real traffic only.
Medians of small samples are noise: look at the "n" columns.
"""
import argparse, re, statistics as st

LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
PARK = re.compile(r"conversation cache: parked (\d+) tokens(?: from slot (\d+))? in ([\d.]+) ms; parked=(\d+) bytes=(\d+) evictions=(\d+) snapshot_bytes=(\d+) reused_kv_bytes=(\d+)")
REST = re.compile(r"conversation cache: restored (\d+) tokens \((\w+)\) in ([\d.]+) ms")
BACK = re.compile(r"slot (\d+) gave back (\d+) tokens .*? in ([\d.]+) ms")
TAKE = re.compile(r"slot (\d+) takes (\d+) tokens \(copied in ([\d.]+) ms\)")

# medians measured on the 47,464 requests of 2026-11-08 03:00 .. 2026-11-14 (old engine, 128 GiB cache)
BASE = {
    "park": {(0, 8000): 183, (8000, 32000): 463, (32000, 64000): 717, (64000, 110000): 1148, (110000, 10**9): 1579},
    "back": {(0, 32000): 69, (32000, 64000): 146, (64000, 110000): 265, (110000, 10**9): 372},
    "take": {(0, 32000): 195, (32000, 64000): 140, (64000, 110000): 251, (110000, 10**9): 345},
    "small": {(0, 8000): 637, (8000, 32000): 952, (32000, 64000): 1044, (64000, 110000): 1290, (110000, 10**9): 1466},
    "read": {(100, 2000): 2500, (2000, 5000): 4400, (5000, 12000): 7300, (12000, 30000): 9600, (30000, 70000): 12500, (70000, 10**9): 18300},
    "dec": {(0, 8000): 92.0, (8000, 32000): 103.8, (32000, 64000): 95.1, (64000, 110000): 100.3, (110000, 10**9): 101.1},
}


def label(lo, hi):
    f = lambda v: "inf" if v >= 10**9 else ("%dK" % (v // 1000) if v >= 1000 else str(v))
    return "%s-%s" % (f(lo), f(hi))


def table(title, unit, buckets, rows, key, base):
    print("\n%s" % title)
    print("  %-12s %10s %8s   %10s" % ("bucket", "now " + unit, "n", "before"))
    for (lo, hi) in buckets:
        v = [r[1] for r in rows if lo <= r[0] < hi]
        now = "%.0f" % st.median(v) if v and unit != "tok/s" else ("%.1f" % st.median(v) if v else "-")
        print("  %-12s %10s %8d   %10s" % (label(lo, hi), now, len(v), ("%.0f" % base[(lo, hi)]) if unit != "tok/s" else ("%.1f" % base[(lo, hi)])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-line", type=int, default=0)
    ap.add_argument("--to-line", type=int, default=0)
    a = ap.parse_args()
    lines = open(LOG, errors="replace").read().splitlines()
    start = a.from_line
    if not start:
        starts = [i + 1 for i, l in enumerate(lines) if "file tier (Linux defaults" in l]
        start = starts[-1] if starts else 1
    end = a.to_line or len(lines)
    seg = lines[start - 1:end]
    parks, backs, takes, rests, reads, dec = [], [], [], [], [], []
    tot_read = tot_gen = 0.0
    nreq = 0
    for l in seg:
        m = PARK.search(l)
        if m:
            parks.append((int(m.group(1)), float(m.group(3)), int(m.group(7)), int(m.group(8)))); continue
        m = REST.search(l)
        if m: rests.append(float(m.group(3))); continue
        m = BACK.search(l)
        if m: backs.append((int(m.group(2)), float(m.group(3)))); continue
        m = TAKE.search(l)
        if m: takes.append((int(m.group(2)), float(m.group(3)))); continue
        m = PROMPT.search(l)
        if m:
            prompt, reused, new, rms, gen, gms, tps = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), float(m.group(7))
            nreq += 1; tot_read += rms; tot_gen += gms
            reads.append((prompt, reused, new, rms))
            if gen >= 300: dec.append((prompt, tps))
    print("log lines %d-%d (%d lines), %d requests, prompt read %.2f h, generate %.2f h (before: 39.5 h read, 8.7 h generate over 47,464 requests)" %
          (start, end, end - start + 1, nreq, tot_read / 3.6e6, tot_gen / 3.6e6))
    big = [(p[0], p[1], p[2] - p[3]) for p in parks]
    table("conversation park (main session and slots), median ms by conversation size", "ms", list(BASE["park"]), [(t, ms) for t, ms, _, _ in parks], "park", BASE["park"])
    gbs = [(n / 1e9) / (ms / 1000) for _, ms, snap, ru in parks for n in [snap - ru] if n > 200e6]
    print("  park speed for parks of >200 MB: median %.2f GB/s over %d parks (before 1.37 GB/s)" % (st.median(gbs) if gbs else float("nan"), len(gbs)))
    table("slot gave back, median ms", "ms", list(BASE["back"]), backs, "back", BASE["back"])
    table("slot takes, median ms", "ms", list(BASE["take"]), takes, "take", BASE["take"])
    print("\nrestore: median %s ms over %d (before 493-544 ms)" % ("%.0f" % st.median(rests) if rests else "-", len(rests)))
    small = [(reused, ms) for prompt, reused, new, ms in reads if new <= 100]
    table("read time of appends of <= 100 new tokens, by reused prefix", "ms", list(BASE["small"]), small, "small", BASE["small"])
    table("read time by number of new tokens (new > 100)", "ms", list(BASE["read"]), [(new, ms) for prompt, reused, new, ms in reads if new > 100], "read", BASE["read"])
    table("decode speed of requests that generated >= 300 tokens, by context", "tok/s", list(BASE["dec"]), dec, "dec", BASE["dec"])
    bad = [l for l in seg if re.search(r"(^|\W)(ERR|illegal memory|Segmentation|abort\(|allocation failed)", l) and "no error" not in l]
    print("\nerror-like lines in the window: %d%s" % (len(bad), ("\n  " + "\n  ".join(b[:160] for b in bad[:5])) if bad else ""))
    evict = [p for p in re.findall(r"evictions=(\d+)", "\n".join(seg))]
    print("cache: last park line says evictions=%s" % (evict[-1] if evict else "-"))


if __name__ == "__main__":
    main()
