#!/usr/bin/env python3
"""Compare V3 and guarded SM75 QSA with the same shared checkpoint and requests."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from types import SimpleNamespace

import bench_decode as bench
import bench_decode_warm as warm
import run_ab as service
from benchmark_q5 import ENGINE_SHA, launch, note, preflight
from capture_run import ready


LAB = Path(__file__).resolve().parent
RUN = LAB / "steady-ab"
CTRL = RUN / "split.control"
ENGINE_LOG = RUN / "engine.log"
RESTORE = {
    "parked_restore": re.compile(r"conversation cache: restored (\d+) tokens .* in ([\d.]+) ms"),
    "slot_restore": re.compile(r"slot \d+ gave back (\d+) tokens .* in ([\d.]+) ms"),
    "slot_copy": re.compile(r"slot \d+ takes (\d+) tokens .* in ([\d.]+) ms"),
}


def publish(path, data):
    temp = path.with_suffix(".next")
    temp.write_bytes(data)
    if path.exists():
        temp.chmod(path.stat().st_mode & 0o777)
    temp.replace(path)


def select(split):
    publish(CTRL, f"{split}\n".encode())


def one_shot(entry):
    data, counted = entry
    body = json.loads(data)
    body["strata_checkpoint"] = False
    return json.dumps(body, separators=(",", ":")).encode(), counted


def options(width, repeat, maximum=768):
    return SimpleNamespace(concurrency=width, timeout=200, interval=.25, min_steady=7,
                           max_tokens=maximum, fixture=f"sm75-steady-w{width}-r{repeat}",
                           mode="sampled", workload="mixed", presence_penalty=0,
                           max_fresh_tokens=256, max_prefix_deviation=64, seed_timeout=120)


def first_times(records, samples, width):
    origin = min(row["start"] for row in records)
    last = min(row["events"][-1] for row in records)
    fully_decoding = [row["time"] for row in samples
                      if row.get("ready") and row["time"] < last
                      and sum(slot["state"] == "decoding" for slot in row["slots"]) == width]
    windows = bench.decode_windows(samples, max(row["events"][0] for row in records), last, width)
    longest = max(windows, key=lambda w: w[-1]["time"] - w[0]["time"], default=[])
    gaps = []
    if longest:
        begin, end = longest[0]["time"], longest[-1]["time"]
        for row in records:
            events = row["events"]
            gaps.extend(b - a for a, b in zip(events, events[1:]) if begin <= a <= b <= end)
    return {"whole_width_decoding_s": min(fully_decoding) - origin if fully_decoding else None,
            "steady_sse_gap_s": bench.quantiles(gaps),
            "prompt_compute_ms": bench.quantiles([v for row in records
                if isinstance(v := row.get("native_timings", {}).get("prompt_ms"), (int, float))])}


def logs(offset):
    with ENGINE_LOG.open("rb") as f:
        f.seek(offset)
        lines = f.read().decode(errors="replace").splitlines()
    trace = [row for row in lines if "QSA decode policy:" in row or ("SM75" in row and "attention" in row)]
    counts = {}
    for kind, marker in RESTORE.items():
        records = [match.groups() for row in lines if (match := marker.search(row))]
        counts[kind] = {"count": len(records), "tokens": [int(n) for n, _ in records],
                        "duration_ms": [float(ms) for _, ms in records]}
    return counts, trace


def run_arm(server, args, label, repeat, fixtures, counted):
    select(32 if label == "guarded" else 0)
    time.sleep(.12)
    bench.loopback_only(8080)
    before = server.idle()
    offset = ENGINE_LOG.stat().st_size
    note(phase="case_start", width=args.concurrency, repeat=repeat, arm=label)
    records, samples = bench.run_cohort(server, args, fixtures, args.max_tokens, True)
    after = server.idle()
    bench.loopback_only(8080)
    summary = bench.summarize(before, after, records, samples, args)
    cache = warm.cache_summary(records, counted, args)
    summary["qualified"] = summary["qualified"] and cache["all_lanes_reused_shared_prefix"]
    detail = first_times(records, samples, args.concurrency) if all(r["events"] for r in records) else {}
    source, trace = logs(offset)
    report = {"width": args.concurrency, "repeat": repeat, "arm": label, "summary": summary,
              "cache": cache, "restore": source, "policy_trace": trace, "timing": detail,
              "records": records, "samples": samples}
    path = RUN / f"w{args.concurrency}-r{repeat}-{label}.json"
    path.write_text(json.dumps(report) + "\n")
    note(phase="case_complete", width=args.concurrency, repeat=repeat, arm=label, qualified=summary["qualified"],
         steady=summary["steady_decode_tok_s"], whole=summary["whole_batch_api_tok_s"],
         ttft=summary["ttft_s"], fresh=cache["fresh_tokens"], restore_counts={k:v["count"] for k,v in source.items()})
    return report


def overlap(server, document):
    select(32)
    args = options(8, 2, 640)
    counted = warm.count(server, warm.messages(document))
    fixtures = [one_shot(entry) for entry in warm.lane_fixtures(server, args, document, 2, counted)]
    before = server.idle()
    offset = ENGINE_LOG.stat().st_size
    note(phase="overlap_start", width=8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = pool.submit(bench.run_cohort, server, args, fixtures, args.max_tokens, False)
        deadline = time.monotonic() + 30
        while True:
            sample = server.sample()
            live = sum(slot["state"] == "decoding" for slot in sample["slots"])
            if live == args.concurrency:
                break
            if work.done() or time.monotonic() > deadline:
                raise RuntimeError("Overlap test never reached eight simultaneous decoders")
            time.sleep(.15)
        time.sleep(1)
        injected_start = time.monotonic_ns()
        injected = server.json("/v1/chat/completions", {
            "model": server.model, "messages": warm.messages(document, "Reply with the single word Green."),
            "strata_prefix": {"messages": 1}, "strata_checkpoint": False, "temperature": 0, "max_tokens": 1,
            "chat_template_kwargs": warm.THINKING_DISABLED,
        }, timeout=30)
        injected_end = time.monotonic_ns()
        rows, _ = work.result(timeout=220)
    after = server.idle()
    source, trace = logs(offset)
    report = {"injected_start_ns": injected_start, "injected_end_ns": injected_end,
              "injected_usage": injected.get("usage"), "completion_delta": after["completed"] - before["completed"],
              "original_requests_complete": all(row["usable"] for row in rows), "policy_trace": trace, "restore": source}
    (RUN / "overlap.json").write_text(json.dumps(report, indent=2) + "\n")
    note(phase="overlap_complete", delta=report["completion_delta"], traces=len(trace))


def benchmarks(server):
    document8 = ""
    for width in (8, 16):
        for repeat in (1, 2):
            args = options(width, repeat)
            select(0)
            document, counted = warm.calibrate(server, 100000, args.fixture)
            bench.require(warm.seed_prefix(server, document, counted, args)["qualified"], "Shared prefix seed failed")
            if width == 8:
                document8 = document
            fixtures = [one_shot(entry) for entry in warm.lane_fixtures(server, args, document, 0, counted)]
            order = ("v3", "guarded") if (width == 8) == (repeat == 1) else ("guarded", "v3")
            for arm in order:
                run_arm(server, args, arm, repeat, fixtures, counted)
    overlap(server, document8)


def boot(config, original, activate=False):
    production = launch([str(service.LAUNCH)], service.LIVE / "serve-orca.out")
    note(phase="production_loading", pid=production.pid, guarded=activate)
    try:
        elapsed = ready(production, config["api_key"], config["host"])
    except BaseException:
        if activate:
            service.stop(production.pid)
        raise
    receipt = {"pid": production.pid, "slots": 16, "startup_s": elapsed, "guarded": activate,
               "original_config": service.CONFIG.read_bytes() == original}
    (RUN / "production-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    note(phase="production_ready", **receipt)


def promote(cfg, original, exe):
    published = False
    try:
        destination = Path(cfg["exe"]).parent.parent / "qsa-steady-20261120"
        destination.mkdir(mode=0o755)
        shutil.copy2(exe, destination / "strata")
        shutil.copy2(exe.parent / "libstrata_kq.so", destination / "libstrata_kq.so")
        (RUN / "production.before.private.json").write_bytes(original)
        new = dict(cfg, exe=str(destination / "strata"))
        new["env"] = dict(cfg["env"], STRATA_SM75_DECODE_ATTN_SPLIT="32")
        publish(service.CONFIG, json.dumps(new, indent=2).encode() + b"\n")
        published = True
        boot(new, original, activate=True)
    except BaseException:
        if published:
            publish(service.CONFIG, original)
        boot(cfg, original)
        raise


def main():
    os.umask(0o077)
    original = service.CONFIG.read_bytes()
    cfg = json.loads(original)
    exe = LAB / "steady-bin/strata"
    if not (hashlib.sha256(original).hexdigest() == service.BASE_CONFIG_SHA and cfg.get("api_key")
            and hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest() == ENGINE_SHA and os.access(exe, os.X_OK)):
        raise RuntimeError("Base production configuration and staged steady-decode engine are required")
    RUN.mkdir()
    select(0)
    preflight(cfg)
    private = RUN / "loopback.private.json"
    process = None
    activate = False
    try:
        local = dict(cfg, host="127.0.0.1", exe=str(exe), log=str(ENGINE_LOG), engine_ready_s=1100)
        local["env"] = dict(cfg["env"], STRATA_SM75_DECODE_ATTN_SPLIT_CONTROL=str(CTRL),
                            STRATA_SM75_DECODE_ATTN_TRACE="1")
        private.write_text(json.dumps(local))
        process = launch([str(service.PYTHON), str(service.APP), "--engine", "strata", "--config", str(private),
                          "--host", "127.0.0.1", "--port", "8080"], RUN / "server.log")
        note(phase="loopback_loading", pid=process.pid)
        elapsed = ready(process, cfg["api_key"], "127.0.0.1")
        note(phase="loopback_ready", pid=process.pid, startup_s=elapsed)
        benchmarks(bench.Server(local, 8080))
        note(phase="qualification_ready")
        decision = RUN / "production-decision"
        end = time.monotonic() + 100
        while not decision.exists() and time.monotonic() < end:
            time.sleep(.5)
        activate = decision.exists() and decision.read_text().strip() == "promote"
    finally:
        if process is not None:
            service.stop(process.pid)
        private.unlink(missing_ok=True)
        if activate:
            promote(cfg, original, exe)
        else:
            boot(cfg, original)


if __name__ == "__main__":
    main()
