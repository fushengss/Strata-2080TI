#!/usr/bin/env python3
"""Repeat the warm comparison after equalizing the previous slot size."""

import argparse
import json

import bench_decode as bench
import bench_decode_warm as warm
from qsa_steady_ab import RUN, note, one_shot, options, run_arm, select


def wash_slots(server, width, arm):
    select(0)
    args = options(16, 3, 384)
    args.fixture = f"sm75-steady-reset-{width}-{arm}"
    fixtures = [one_shot(bench.fixture(server, args, lane, 0, 0, args.max_tokens))
                for lane in range(args.concurrency)]
    server.idle()
    records, samples = bench.run_cohort(server, args, fixtures, args.max_tokens, True)
    server.idle()
    slots = {slot["slot"] for row in samples for slot in row.get("slots", [])
             if "slot" in slot and slot["state"] in ("reading", "decoding")}
    bench.require(all(row["usable"] for row in records) and len(slots) == 16,
                  "The precondition did not replace all 16 slot sessions")
    note(phase="short_slots_seeded", width=width, arm=arm, slots=len(slots))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=(8, 16), default=8)
    width = parser.parse_args().width
    cfg = json.loads((RUN / "loopback.private.json").read_bytes())
    bench.loopback_only(8080)
    server = bench.Server(cfg, 8080)
    args = options(width, 3, 640)
    args.fixture = options(width, 2).fixture
    document, counted = warm.calibrate(server, 100000, args.fixture)
    fixtures = [one_shot(entry) for entry in warm.lane_fixtures(server, args, document, 0, counted)]
    for arm in ("guarded", "v3"):
        wash_slots(server, width, arm)
        bench.require(warm.seed_prefix(server, document, counted, args)["qualified"], "Shared prefix seed failed")
        run_arm(server, args, arm, 3, fixtures, counted)
    note(phase="slot_matched_recheck_complete")


if __name__ == "__main__":
    main()
