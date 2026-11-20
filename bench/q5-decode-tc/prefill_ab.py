#!/usr/bin/env python3
"""Measure native prefill and first-token latency on a loopback-only Strata arm."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import bench_decode as base


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3, choices=(1, 2, 3, 4, 5))
    p.add_argument("--context", type=int, action="append", choices=(2048, 8192, 26000, 100000))
    args = p.parse_args()
    cfg = json.loads(args.config.read_bytes())
    port = int(cfg.get("port", 8080))
    base.loopback_only(port)
    api = base.Server(cfg, port)
    initial = api.idle()
    opts = SimpleNamespace(mode="greedy", workload="code", fixture="native-prefill-ab",
                           presence_penalty=0, concurrency=1, timeout=180)
    corpus = []
    for length in args.context or (2048, 8192, 26000, 100000):
        for trial in range(args.rounds):
            corpus.append((length, trial, base.fixture(api, opts, trial, str(length), length, 64)))
    warmup = base.fixture(api, opts, 0, "warmup", 3000, 32)
    warmed, _ = base.run_cohort(api, opts, [warmup], 32, False)
    base.require(warmed[0]["usable"], "Warm-up did not finish normally")
    records = []
    for length, trial, request in corpus:
        before = api.idle()
        response, _ = base.run_cohort(api, opts, [request], 64, False)
        after = api.idle()
        row = response[0]
        usage = row["usage"]
        timing = row.get("native_timings", {})
        actual = timing.get("prompt_ms")
        checks = {
            "same_instance": initial["instance"] == before["instance"] == after["instance"],
            "one_request": after["completed"] - before["completed"] == 1,
            "cold": usage.get("cached_tokens") == 0,
            "stream_complete": row["usable"],
            "native_prompt_timing": isinstance(actual, (int, float)) and actual > 0,
        }
        clean = {k: row.get(k) for k in ("request_sha256", "output_sha256", "reasoning_sha256",
                                         "counted_prompt", "finish", "ttft_s", "native_timings", "usage", "error")}
        rec = {"context": length, "trial": trial, "checks": checks, "qualified": all(checks.values()), **clean}
        records.append(rec)
        print(json.dumps({"context": length, "trial": trial, "native_prompt_ms": actual,
                          "ttft_s": row["ttft_s"], "qualified": rec["qualified"]}), flush=True)
    base.loopback_only(port)
    rollup = []
    for length in sorted({r["context"] for r in records}):
        group = [r for r in records if r["context"] == length and r["qualified"]]
        rollup.append({"context": length, "samples": len(group),
                       "native_ms_median": statistics.median(r["native_timings"]["prompt_ms"] for r in group) if group else None,
                       "ttft_s_median": statistics.median(r["ttft_s"] for r in group) if group else None})
    with args.out.open("x") as out:
        json.dump({"qualified": all(r["qualified"] for r in records), "records": records, "summary": rollup}, out, indent=2)
        out.write("\n")
    return 0 if all(r["qualified"] for r in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
