#!/usr/bin/env python3
"""Measure incumbent decode when a cold 100k prefill enters a loopback service."""

import argparse
import concurrent.futures
import http.client
import json
import math
import re
import socket
import threading
import time
from pathlib import Path

import bench_decode as base


LONG_TOKENS = 100000
READER_OUTPUT = 32
SATURATED_SLOTS = 16


def identities(sample, state):
    slots = [slot for slot in sample.get("slots", []) if slot.get("state") == state]
    if any("request" not in slot for slot in slots):
        return ()
    ids = tuple(sorted(slot["request"] for slot in slots))
    return ids if len(set(ids)) == len(ids) else ()


def queue_depth(sample):
    work = sample.get("work", {})
    # An HTTP request waiting for a free batch slot can be in live_reqs but no
    # longer in queued/waiting. At full occupancy it is counted outside_slots.
    return max(work.get("queued", 0), work.get("waiting", 0), work.get("outside_slots", 0))


def before_read(sample, slots):
    work = sample.get("work", {})
    ids = identities(sample, "decoding")
    if (sample.get("ready") and work.get("occupied_slots") == work.get("active") == slots
            and queue_depth(sample) == 0 and len(ids) == slots):
        return ids
    return ()


def inside_read(sample, tokens, decoders, established):
    work = sample.get("work", {})
    peers = identities(sample, "decoding")
    readers = [slot for slot in sample.get("slots", []) if slot.get("state") == "reading"]
    if (sample.get("ready") and len(peers) == decoders and set(peers) <= set(established)
            and work.get("occupied_slots") == decoders + 1 and queue_depth(sample) == 0
            and len(readers) == 1 and readers[0].get("prompt_tokens") == tokens
            and readers[0].get("generated") == 0 and readers[0].get("request")
            and readers[0]["request"] not in established):
        return peers, readers[0]["request"]
    return ()


def sample_runs(samples, match):
    runs, current, prev = [], [], None
    for row in samples:
        key = match(row)
        if current and key != prev:
            runs.append(current)
            current = []
        if key:
            current.append(row)
        prev = key
    if current:
        runs.append(current)
    return runs


def rate_window(rows):
    if len(rows) < 2:
        return None
    first, last = rows[0], rows[-1]
    span = last["time"] - first["time"]
    if span <= 0:
        return None
    tokens = last["counter"] - first["counter"]
    short_rates = []
    right_index = 1
    for left_index, left in enumerate(rows):
        right_index = max(right_index, left_index + 1)
        while right_index < len(rows) and rows[right_index]["time"] - left["time"] < 1:
            right_index += 1
        if right_index == len(rows):
            break
        right = rows[right_index]
        short_rates.append((right["counter"] - left["counter"]) / (right["time"] - left["time"]))
    return {"start": first["time"], "end": last["time"], "span_s": span, "tokens": tokens,
            "tok_s": tokens / span, "samples": len(rows), "one_second_tok_s": base.quantiles(short_rates)}


def largest_window(runs):
    return max(runs, key=lambda rows: rows[-1]["time"] - rows[0]["time"], default=[])


def event_gaps(records, start, end):
    all_gaps = []
    lanes = []
    for rec in records:
        events = rec["events"]
        gaps = [right - left for left, right in zip(events, events[1:]) if right > start and left < end]
        all_gaps.extend(gaps)
        lanes.append({"lane": rec["lane"], "covers_window": bool(events and events[0] <= start <= end <= events[-1]),
                      "gaps_s": base.quantiles(gaps), "events_inside": sum(start <= t <= end for t in events)})
    longest = [lane["gaps_s"]["max"] for lane in lanes if lane["gaps_s"]]
    return {"pooled_gaps_s": base.quantiles(all_gaps), "lane_maxima_s": base.quantiles(longest),
            "gap_counts_above_s": {str(s): sum(value > s for value in all_gaps) for s in (.25, .5, 1, 2)},
            "per_lane": lanes}


def summarize(before, after, records, samples, pre_rows, established, args):
    peers = records[:args.decoders]
    sentinel = records[args.decoders] if args.saturated else None
    reader = records[-1]
    actual = reader["usage"].get("prompt_tokens")
    native_timings = reader.get("native_timings", {})
    reader_first = reader["events"][0] if reader["events"] else reader["end"]
    overlap = sample_runs(samples, lambda row: inside_read(row, actual, args.decoders, established)
                          if reader["start"] <= row.get("start", -1) and row.get("end", math.inf) <= reader_first else ())
    read_rows = largest_window(overlap)
    pre = rate_window(pre_rows)
    during = rate_window(read_rows)
    queue_rows = [row for row in samples if args.saturated and reader["start"] <= row.get("start", -1)
                  and row.get("end", math.inf) <= sentinel["end"] and queue_depth(row) == 1
                  and row["work"].get("occupied_slots") == SATURATED_SLOTS
                  and identities(row, "decoding") == established and not identities(row, "reading")]
    request_gaps = event_gaps(peers, reader["start"], reader_first)
    read_gaps = event_gaps(peers, during["start"], during["end"]) if during else None
    steady_peers = peers + ([sentinel] if sentinel else [])
    pre_gaps = event_gaps(steady_peers, pre["start"], pre["end"]) if pre else None
    checked = [before, *(row for row in samples if "counter" in row), after]
    tokens = sum(row["usage"].get("completion_tokens", 0) for row in records)
    checks = {
        "same_instance": len({row["instance"] for row in checked}) == 1,
        "cohort_complete": all(row["usable"] for row in records),
        "exclusive_requests": after["completed"] - before["completed"] == len(records),
        "exclusive_output": after["counter"] - before["counter"] == tokens,
        "counter_monotonic": all(a["counter"] <= b["counter"] for a, b in zip(checked, checked[1:])),
        "cold_100k": reader["usage"].get("cached_tokens") == 0 and actual is not None
        and abs(actual - LONG_TOKENS) <= 2 and native_timings.get("prompt_n") == actual,
        "established_decode": pre is not None and pre["span_s"] >= args.steady_s and pre["tokens"] > 0,
        "read_decode_overlap": during is not None and during["span_s"] >= 3 and during["tokens"] > 0,
        "incumbent_outputs_cover_request": all(row["covers_window"] for row in request_gaps["per_lane"]),
    }
    if sentinel is not None:
        reused_slot = False
        if queue_rows and read_rows and queue_rows[-1]["time"] < read_rows[0]["time"]:
            survivor_ids = set(identities(read_rows[0], "decoding"))
            missing_slots = [row["slot"] for row in queue_rows[-1]["slots"] if row["state"] == "decoding"
                             and row["request"] not in survivor_ids]
            reader_slots = [row["slot"] for row in read_rows[0]["slots"] if row["state"] == "reading"]
            reused_slot = missing_slots == reader_slots
        checks.update(queued_with_full_slots=bool(queue_rows), sentinel_finishes=sentinel["finish"] == "length"
                      and sentinel["usage"].get("completion_tokens") == args.sentinel_tokens
                      and reader["start"] < sentinel["end"] < reader_first,
                      reader_takes_freed_slot=reused_slot)
    return {"qualified": all(checks.values()), "checks": checks, "read_prompt_tokens": actual,
            "read_prompt_ms": native_timings.get("prompt_ms"), "read_ttft_s": reader["ttft_s"],
            "before_decoders": len(established), "simultaneous_decoders_during_read": args.decoders if read_rows else 0,
            "before_decode": pre, "during_prefill_decode": during,
            "before_sse_gaps": pre_gaps, "over_request_ttft_sse_gaps": request_gaps, "during_prefill_sse_gaps": read_gaps,
            "queued_while_full_samples": len(queue_rows),
            "queued_while_full_observed_s": queue_rows[-1]["time"] - queue_rows[0]["time"] if queue_rows else None,
            "seconds_from_post_to_first_read_sample": read_rows[0]["time"] - reader["start"] if read_rows else None,
            "maximum_waiting_outside_slots_seen": max((queue_depth(s) for s in samples), default=0)}


def observe(server):
    try:
        return server.sample()
    except (OSError, ValueError, base.BenchmarkError, http.client.HTTPException) as exc:
        return {"time": time.monotonic(), "error": type(exc).__name__}


def run_round(server, args, round_id):
    count = args.decoders + int(args.saturated)
    reader_cap = READER_OUTPUT
    caps = [args.max_tokens] * args.decoders + ([args.sentinel_tokens] if args.saturated else [])
    workload = [base.fixture(server, args, lane, f"{round_id}-peer", 0, cap) for lane, cap in enumerate(caps)]
    inject = base.fixture(server, args, count, f"{round_id}-cold", LONG_TOKENS, reader_cap)
    before = server.idle()
    deadline = time.monotonic() + args.timeout
    warm_until = time.monotonic() + min(args.timeout / 2, 75)
    abort, lock, sockets = threading.Event(), threading.Lock(), {}
    gate = threading.Barrier(count)
    samples, pre_rows, established = [], [], ()
    reader = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=count + 1) as pool:
        futures = [pool.submit(base.stream, server, lane, item, cap, gate, deadline, abort, sockets, lock)
                   for lane, (item, cap) in enumerate(zip(workload, caps))]
        try:
            while time.monotonic() < deadline:
                sample_start = time.monotonic()
                if reader and all(future.done() for future in futures):
                    break
                sample = observe(server)
                samples.append(sample)
                if reader is None:
                    ids = before_read(sample, count)
                    if ids != established:
                        pre_rows, established = [], ids
                    if ids:
                        pre_rows.append(sample)
                    if pre_rows and sample["time"] - pre_rows[0]["time"] >= args.steady_s:
                        reader = pool.submit(base.stream, server, count, inject, reader_cap, threading.Barrier(1),
                                             deadline, abort, sockets, lock)
                        futures.append(reader)
                    elif any(f.done() for f in futures) or time.monotonic() >= warm_until:
                        raise base.BenchmarkError(f"Failed to establish {count} decoding slots before injection")
                time.sleep(max(0, min(deadline - time.monotonic(), args.interval - (time.monotonic() - sample_start))))
            base.require(reader is not None and all(future.done() for future in futures), "Cohort timed out")
        finally:
            if not all(future.done() for future in futures):
                abort.set()
                with lock:
                    for sock in sockets.values():
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
        records = [future.result() for future in futures]
    after = server.idle()
    base.loopback_only(server.port)
    return {"round": round_id, "before": before, "after": after, "records": records, "samples": samples,
            "summary": summarize(before, after, records, samples, pre_rows, established, args)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--decoders", type=int, choices=(8, 15), required=True)
    parser.add_argument("--saturated", action="store_true", help="Start 16, queue the long request, overlap with 15")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sentinel-tokens", type=int, default=768)
    parser.add_argument("--steady-s", type=float, default=4.)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--interval", type=float, default=.125)
    parser.add_argument("--timeout", type=float, default=240.)
    parser.add_argument("--fixture", default="gpu-mmq-insert")
    args = parser.parse_args()
    args.mode, args.workload, args.presence_penalty = "sampled", "mixed", 0
    base.require((not args.saturated or args.decoders == SATURATED_SLOTS - 1)
                 and 128 <= args.sentinel_tokens < args.max_tokens <= 8192 and 1 <= args.rounds <= 10
                 and .1 <= args.interval <= 1 and 2 <= args.steady_s <= 15 and 60 <= args.timeout <= 600
                 and re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", args.fixture) is not None, "Invalid workload parameters")
    cfg = json.loads(args.config.read_bytes())
    port = args.port or int(cfg.get("port", 8080))
    base.loopback_only(port)
    server = base.Server(cfg, port)
    initial = server.idle()
    slots = initial["work"].get("slots_effective")
    base.require(slots == SATURATED_SLOTS, "This benchmark requires the deployed 16-slot service configuration")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.open("x").close()
    report = {"version": 1, "source_sha256": base.sha(Path(__file__).read_bytes()), "listener": f"127.0.0.1:{port}",
              "effective_slots": slots, "parameters": {key: val for key, val in vars(args).items()
                                                       if key not in ("config", "out", "port")}, "rounds": []}
    try:
        for round_id in range(args.rounds):
            row = run_round(server, args, round_id)
            report["rounds"].append(row)
            summary = row["summary"]
            read_gaps = (summary["during_prefill_sse_gaps"] or {}).get("pooled_gaps_s") or {}
            print(json.dumps({"round": round_id, **{k: v for k, v in summary.items() if not k.endswith("gaps")},
                              "max_read_sse_gap_s": read_gaps.get("max")}), flush=True)
        report["qualified"] = all(row["summary"]["qualified"] for row in report["rounds"])
    except Exception as exc:
        report.update(qualified=False, error=str(exc) if isinstance(exc, base.BenchmarkError) else type(exc).__name__)
        print(json.dumps({"error": report["error"]}), flush=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
