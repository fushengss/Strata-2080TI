#!/usr/bin/env python3
"""Insert one cold 100k read during eight established loopback streams."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import socket
import threading
import time

import bench_decode as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.mode, args.workload, args.fixture, args.presence_penalty = "sampled", "mixed", "cold-injection", 0
    cfg = json.loads(args.config.read_bytes())
    port = int(cfg.get("port", 8080))
    base.loopback_only(port)
    server = base.Server(cfg, port)
    base.require(server.idle()["work"]["slots_effective"] >= 9, "At least nine actual slots are required")
    peers = [base.fixture(server, args, lane, "peer", 0, 3072) for lane in range(8)]
    cold = base.fixture(server, args, 8, "read", 100000, 32)
    before = server.idle()
    deadline = time.monotonic() + 240
    abort, lock, sockets = threading.Event(), threading.Lock(), {}
    gate = threading.Barrier(8)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
        futures = [pool.submit(base.stream, server, lane, item, 3072, gate, deadline, abort, sockets, lock)
                   for lane, item in enumerate(peers)]
        try:
            warm_until = time.monotonic() + 30
            established = launched = None
            while not all(f.done() for f in futures) and time.monotonic() < deadline:
                sample = server.sample()
                samples.append(sample)
                if launched is None:
                    active = sum(s["state"] == "decoding" for s in sample["slots"])
                    if active == 8:
                        if established is None:
                            established = sample["time"]
                        elif sample["time"] - established >= 1.0:
                            futures.append(pool.submit(base.stream, server, 8, cold, 32, threading.Barrier(1),
                                                       deadline, abort, sockets, lock))
                            launched = sample["time"]
                    else:
                        established = None
                        base.require(time.monotonic() < warm_until, "Eight decode streams did not become active")
                time.sleep(.25)
            base.require(launched is not None and all(f.done() for f in futures), "Cohort expired")
        finally:
            if not all(f.done() for f in futures):
                abort.set()
                with lock:
                    for sock in sockets.values():
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
        records = [f.result() for f in futures]
    after = server.idle()
    base.loopback_only(port)
    read = records[8]
    start, finish = read["start"], read["events"][0] if read["events"] else read["end"]
    peers = records[:8]
    gaps = []
    for peer in peers:
        events = peer["events"]
        during = [right - left for left, right in zip(events, events[1:]) if right >= start and left <= finish]
        gaps.append({"lane": peer["lane"], "overlapping_gap_max_s": max(during, default=None),
                     "events_during_read": sum(start <= t <= finish for t in events),
                     "covers_read": bool(events and events[0] <= start and events[-1] >= finish)})
    accounted = sum(record["usage"].get("completion_tokens", 0) for record in records)
    checks = {"same_instance": len({before["instance"], after["instance"], *(s["instance"] for s in samples)}) == 1,
              "cohort_complete": all(r["usable"] for r in records),
              "cohort_exclusive": after["completed"] - before["completed"] == len(records)
              and after["counter"] - before["counter"] == accounted,
              "cold_read": read["usage"].get("cached_tokens") == 0,
              "all_peers_cover_read": all(item["covers_read"] for item in gaps)}
    summary = {"checks": checks, "qualified": all(checks.values()), "injected_ttft_s": read["ttft_s"],
               "peers": gaps, "max_peer_gap_s": max((g["overlapping_gap_max_s"] for g in gaps
                                                       if g["overlapping_gap_max_s"] is not None), default=None)}
    with args.out.open("x") as output:
        json.dump({"summary": summary, "records": records, "samples": samples}, output)
        output.write("\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
