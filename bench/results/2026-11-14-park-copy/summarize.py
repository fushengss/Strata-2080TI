#!/usr/bin/env python3
"""bench/results/2026-11-14-park-copy/summarize.py - compare park / restore / read timings of the variants a parktest.py churn run produced.

Run next to the <variant>.log / <variant>.json files:   python3 summarize.py old-small new-small newshim-small
"""
import json, re, statistics as st, sys

PARK = re.compile(r"parked (\d+) tokens(?: from slot (\d+))? in ([\d.]+) ms; parked=(\d+) bytes=(\d+) evictions=(\d+) snapshot_bytes=(\d+) reused_kv_bytes=(\d+)")
REST = re.compile(r"restored (\d+) tokens \((\w+)\) in ([\d.]+) ms")
BACK = re.compile(r"slot (\d+) gave back (\d+) tokens .*? in ([\d.]+) ms")
TAKE = re.compile(r"slot (\d+) takes (\d+) tokens \(copied in ([\d.]+) ms\)")
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
STAGE = re.compile(r"parktrace: joins [\d.]+ ms, sync [\d.]+ ms, (\d+) checkpoints copied ([\d.]+) ms, live state ([\d.]+) ms, begin [\d.]+ ms, K/V (\d+) MB ([\d.]+) ms \(layer max [\d.]+, sum [\d.]+; attrs [\d.]+, resize ([\d.]+), copy ([\d.]+)\)")


def med(v):
    return st.median(v) if v else float("nan")


def load(tag):
    parks, rests, backs, takes, reads, stages = [], [], [], [], [], []
    for line in open(tag + ".log", errors="replace"):
        m = PARK.search(line)
        if m:
            tok, slot, ms, cnt, byt, ev, snap, ru = m.groups()
            parks.append(dict(tok=int(tok), ms=float(ms), snap=int(snap), new=int(snap) - int(ru), ev=int(ev), slot=slot is not None))
            continue
        m = REST.search(line)
        if m: rests.append(float(m.group(3))); continue
        m = BACK.search(line)
        if m: backs.append(float(m.group(3))); continue
        m = TAKE.search(line)
        if m: takes.append(float(m.group(3))); continue
        m = PROMPT.search(line)
        if m: reads.append(dict(prompt=int(m.group(1)), reused=int(m.group(2)), new=int(m.group(3)), ms=float(m.group(4)))); continue
        m = STAGE.search(line)
        if m: stages.append(dict(ck=float(m.group(2)), live=float(m.group(3)), kv=float(m.group(5)), resize=float(m.group(6)), copy=float(m.group(7))))
    return parks, rests, backs, takes, reads, stages


def main():
    tags = sys.argv[1:]
    rows = {}
    for t in tags:
        parks, rests, backs, takes, reads, stages = load(t)
        client = json.load(open(t + ".json"))["requests"]
        big = [p for p in parks if p["new"] > 200e6]            # a park that copies at least 200 MB
        steady = [p for p in big if p["ev"] > 0]                 # after the cache started evicting
        gbs = lambda ps: [p["new"] / 1e9 / (p["ms"] / 1000) for p in ps]
        small_reads = [r for r in reads if r["new"] <= 700 and r["reused"] > 10000]
        rows[t] = dict(n_park=len(parks), n_big=len(big), n_steady=len(steady), park_ms=med([p["ms"] for p in big]),
                       park_ms_steady=med([p["ms"] for p in steady]), gbs=med(gbs(big)), gbs_steady=med(gbs(steady)),
                       restore=med(rests), back=med(backs), take=med(takes), read_small=med([r["ms"] for r in small_reads]),
                       n_small=len(small_reads), resize=med([s["resize"] for s in stages]), ckpt=med([s["ck"] for s in stages]),
                       live=med([s["live"] for s in stages]), kvstage=med([s["kv"] for s in stages]),
                       client_append=med([r["client_s"] for r in client if r["turn"] > 0]))
    cols = [("parks logged", "n_park", "%d"), ("parks copying >=200 MB", "n_big", "%d"), ("... after evictions began", "n_steady", "%d"),
            ("park ms, median (all big)", "park_ms", "%.0f"), ("park ms, median (after evictions)", "park_ms_steady", "%.0f"),
            ("park GB/s (all big)", "gbs", "%.2f"), ("park GB/s (after evictions)", "gbs_steady", "%.2f"),
            ("per-stage K/V save ms (median)", "kvstage", "%.1f"), ("  of it zero-fill/resize ms", "resize", "%.1f"),
            ("per-stage checkpoint copies ms", "ckpt", "%.1f"), ("per-stage live state ms", "live", "%.1f"),
            ("restore ms, median", "restore", "%.0f"), ("slot gave back ms", "back", "%.0f"), ("slot takes ms", "take", "%.0f"),
            ("read ms of small appends (<=700 new, >10K reused)", "read_small", "%.0f"), ("   requests counted", "n_small", "%d"),
            ("client latency of appends, s (median)", "client_append", "%.2f")]
    print("%-52s" % "" + "".join("%18s" % t for t in tags))
    for label, key, fmt in cols:
        print("%-52s" % label + "".join("%18s" % (fmt % rows[t][key]) for t in tags))
    # same conversation, same turn, same reply across variants (exactness)
    hashes = {t: {(r["conv"], r["turn"]): r.get("reply_sha1") for r in json.load(open(t + ".json"))["requests"]} for t in tags}
    keys = sorted(set.intersection(*[set(h) for h in hashes.values()]))
    same = sum(1 for k in keys if len({hashes[t][k] for t in tags}) == 1)
    print("\nreplies identical across all %d variants: %d of %d (conversation, turn) pairs" % (len(tags), same, len(keys)))


if __name__ == "__main__":
    main()
