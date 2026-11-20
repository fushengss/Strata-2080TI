#!/usr/bin/env python3
"""Keep cold probes one-shot and verify short decode with equivalent prefix reuse."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import bench_decode as bench
import bench_decode_warm as warm
import prefill_ab as prefill
from qsa_ab import run_case, select


def cold_probe():
    original_fixture = bench.fixture

    def one_shot(*args, **kwargs):
        payload, counted = original_fixture(*args, **kwargs)
        request = json.loads(payload)
        request["strata_checkpoint"] = False
        return json.dumps(request, separators=(",", ":")).encode(), counted

    bench.fixture = one_shot
    try:
        return prefill.main()
    finally:
        bench.fixture = original_fixture


def verify_short(private):
    server = bench.Server(json.loads(private.read_bytes()), 8080)
    for width, order in ((8, ("fp16", "v3")), (16, ("v3", "fp16"))):
        opts = SimpleNamespace(concurrency=width, timeout=120, mode="sampled", workload="mixed",
                               fixture=f"sm75-qsa-short{width}", presence_penalty=0)
        # Prime an identical prompt turn with minimal output before each arm.
        prime = [bench.fixture(server, opts, lane, 0, 0, 1) for lane in range(width)]
        for arm in order:
            select(0 if arm == "v3" else 32)
            bench.loopback_only(8080)
            server.idle()
            primed, _ = bench.run_cohort(server, opts, prime, 1, False)
            bench.require(all(row["usable"] for row in primed), "The cache-matched short test could not be primed")
            run_case(arm, f"cached-short{width}", "bench_decode.py", private,
                     ["--concurrency", str(width), "--fixture", opts.fixture, "--max-tokens", "768",
                      "--min-steady", "7"])


def verify_warm(private):
    server = bench.Server(json.loads(private.read_bytes()), 8080)
    for width, order in ((8, ("v3", "fp16")), (16, ("fp16", "v3"))):
        opts = SimpleNamespace(concurrency=width, timeout=160, mode="sampled", workload="mixed",
                               fixture=f"sm75-qsa-warm{width}", presence_penalty=0, max_tokens=1,
                               max_fresh_tokens=256, seed_timeout=120)
        document, counted = warm.calibrate(server, 100000, opts.fixture)
        for arm in order:
            select(0 if arm == "v3" else 32)
            bench.loopback_only(8080)
            bench.require(warm.seed_prefix(server, document, counted, opts)["qualified"],
                          "The shared document could not be seeded")
            prime = warm.lane_fixtures(server, opts, document, 0, counted)
            primed, _ = bench.run_cohort(server, opts, prime, 1, False)
            bench.require(all(row["usable"] for row in primed), "The cache-matched warm test could not be primed")
            run_case(arm, f"cached-warm100k-{width}", "bench_decode_warm.py", private,
                     ["--concurrency", str(width), "--fixture", opts.fixture, "--max-tokens", "768",
                      "--context-tokens", "100000", "--min-steady", "7"])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    main_args, _ = parser.parse_known_args()
    status = cold_probe()
    if main_args.out.name == "fp16-cold100k.json":
        verify_short(main_args.config)
        verify_warm(main_args.config)
    return status


if __name__ == "__main__":
    sys.exit(main())
