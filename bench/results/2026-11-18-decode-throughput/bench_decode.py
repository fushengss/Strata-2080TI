#!/usr/bin/env python3
"""Measure a synthetic decode cohort against a loopback-only Strata listener.

Each run reports final API work and steady server-counter throughput. SSE events
provide per-stream client timing and output hashes. The output file contains
numbers, labels and hashes only. Run on the machine that serves the model.
"""

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import re
import socket
import statistics
import subprocess
import threading
import time


TOPICS = ("bridges", "volcanoes", "bees", "trains", "glaciers", "printing", "chess", "rice",
          "lighthouses", "tea", "owls", "rivers", "clocks", "maps", "salt", "kites")
WORDS = ("quiet towns ship harbor weather cloud distant river mountain lantern carriage old map street "
         "farmer bridge water child market home hand song plant field garden blue clock town rain seed").split()
ESSAY = ("Write a very long illustrated field guide in English about {topic}. Number at least one hundred "
         "sections, and for each give a substantial description, an observation, and an example. "
         "Start the first section immediately and keep going. Do not summarize.")
CODE = ("Write a long standalone Python module for analyzing {topic}. Include detailed type annotations, "
        "documented classes, validation, and at least fifty realistic unittest cases. Start immediately "
        "with the code and continue for several thousand lines. Output only code.")
GPU_FIELDS = ("index", "mem_used", "mem_total", "util", "temp", "power")
WORK_FIELDS = ("active", "occupied_slots", "slots_effective", "slots_requested", "groups", "queued", "waiting",
               "outside_slots", "batch_sample_windows", "batch_sample_rows", "batch_sample_active_rows",
               "batch_sample_span_s", "batch_sample_age_s", "stage_sample_elapsed_us", "stage_sample_age_s",
               "admission_rejected", "queue_timeouts", "client_disconnect_cancels")


class BenchmarkError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BenchmarkError(message)


def numeric(row, names):
    return {key: row[key] for key in names if type(row.get(key)) in (int, float) and math.isfinite(row[key])}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {"min": values[0], "p10": values[math.ceil(.10 * len(values)) - 1],
            "median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "max": values[-1], "n": len(values)}


def loopback_only(port):
    result = subprocess.run(["ss", "-H", "-ltn", f"sport = :{port}"], check=True, capture_output=True,
                            text=True, timeout=5)
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    require(rows and all(len(row) >= 5 and row[3] == f"127.0.0.1:{port}" for row in rows),
            "Listener check failed: requested port must listen only on 127.0.0.1")


class Server:
    def __init__(self, cfg, port):
        self.port = port
        self.model = cfg.get("model_name") or "strata"
        self.headers = {"Content-Type": "application/json", "Connection": "close"}
        if cfg.get("api_key"):
            self.headers["Authorization"] = "Bearer " + cfg["api_key"]

    def json(self, path, body=None, timeout=5):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            conn.request("GET" if body is None else "POST", path,
                         body=None if body is None else json.dumps(body), headers=self.headers)
            with conn.getresponse() as response:
                require(response.status == 200, f"{path} returned HTTP {response.status}")
                return json.load(response)
        finally:
            conn.close()

    def sample(self):
        start = time.monotonic()
        data = self.json("/v1/telemetry")
        end = time.monotonic()
        instance = data.get("instance", {}).get("id")
        counter = data.get("rates", {}).get("output_tokens_server_counter")
        total = data.get("totals", {}).get("requests")
        require(isinstance(instance, str) and type(counter) is int and type(total) is int,
                "Telemetry lacks instance, output counter or completed-request counter")
        work = data.get("work", {})
        slots = []
        for source in work.get("slots", []):
            slot = numeric(source, ("slot", "group", "replica", "generated", "prompt_tokens", "prompt_read"))
            slot["state"] = source["state"] if source.get("state") in ("idle", "reading", "decoding") else None
            if isinstance(source.get("request_id"), str):
                slot["request"] = sha(source["request_id"].encode())
            slots.append(slot)
        return {"start": start, "end": end, "time": (start + end) / 2, "instance": sha(instance.encode()),
                "ready": data.get("model", {}).get("loaded") is True and data.get("instance", {}).get("state") == "ready",
                "counter": counter, "completed": total, "work": numeric(work, WORK_FIELDS), "slots": slots,
                "devices": [numeric(row, GPU_FIELDS) for row in data.get("devices", [])]}

    def idle(self, wait=20):
        deadline = time.monotonic() + wait
        while True:
            row = self.sample()
            work = row["work"]
            if row["ready"] and all(work.get(k) == 0 for k in ("active", "queued", "waiting", "occupied_slots")):
                return row
            require(time.monotonic() < deadline, "A ready, idle inference engine is required")
            time.sleep(.25)

    def count(self, prompt):
        result = self.json("/v1/messages/count_tokens", {"model": self.model, "max_tokens": 1,
                           "messages": [{"role": "user", "content": prompt}], "thinking": {"type": "disabled"}}, 20)
        n = result.get("input_tokens")
        require(type(n) is int and n > 0, "Token-count endpoint did not return a positive integer")
        return n


def fixture(server, args, lane, suffix, context, maximum):
    seed = 20261118 + lane
    rng = random.Random(seed)
    task = CODE if args.workload == "code" or args.workload == "mixed" and lane % 2 == 0 else ESSAY
    task = task.format(topic=TOPICS[lane % len(TOPICS)])
    label = f"Fixture {args.fixture} lane {lane} cohort {suffix}."
    prompt = f"{label}\n{task}"
    counted = None
    if context:
        words = rng.choices(WORDS, k=context + 32)
        size = context - 50
        for _ in range(6):
            prompt = f"{label}\nReference notes:\n{' '.join(words[:size])}\n\n{task}"
            counted = server.count(prompt)
            if abs(counted - context) <= 2:
                break
            size += context - counted
            require(0 < size <= len(words), "Unable to fit the requested token context")
        require(abs(counted - context) <= 2, "Unable to calibrate a synthetic prompt to the requested context")
    body = {"model": server.model, "max_tokens": maximum, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0 if args.mode == "greedy" else 1.0, "top_p": .95, "top_k": 20,
            "presence_penalty": args.presence_penalty, "messages": [{"role": "user", "content": prompt}]}
    return json.dumps(body, separators=(",", ":")).encode(), counted


def stream(server, lane, fixture_data, maximum, barrier, deadline, abort, sockets, lock):
    payload, counted = fixture_data
    result = {"lane": lane, "request_sha256": sha(payload), "counted_prompt": counted, "events": [],
              "finish": None, "done": False, "usage": {}, "error": None}
    content, reasoning = hashlib.sha256(), hashlib.sha256()
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=max(1, deadline - time.monotonic()))
    try:
        barrier.wait(timeout=max(1, deadline - time.monotonic()))
        result["start"] = time.monotonic()
        conn.connect()
        with lock:
            sockets[lane] = conn.sock
        require(not abort.is_set(), "Cohort expired before request write")
        conn.request("POST", "/v1/chat/completions", body=payload, headers=server.headers)
        with conn.getresponse() as response:
            if response.status != 200:
                result["error"] = f"http_{response.status}"
            else:
                while not abort.is_set() and time.monotonic() < deadline:
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(data)
                    if "error" in event:
                        result["error"] = "api_error"
                        break
                    if event.get("usage"):
                        usage = event["usage"]
                        result["usage"] = numeric(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
                        result["usage"].update(numeric(usage.get("prompt_tokens_details", {}), ("cached_tokens",)))
                    if event.get("timings"):
                        result["native_timings"] = numeric(event["timings"], ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms"))
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if text or thought:
                            result["events"].append(time.monotonic())
                            content.update(text.encode())
                            reasoning.update(thought.encode())
                        if choice.get("finish_reason"):
                            result["finish"] = choice["finish_reason"] if choice["finish_reason"] in ("stop", "length") else "other"
    except Exception as exc:
        result["error"] = type(exc).__name__
    finally:
        conn.close()
        with lock:
            sockets.pop(lane, None)
        result.setdefault("start", time.monotonic())
        result["end"] = time.monotonic()
    if abort.is_set():
        result["error"] = "cohort_timeout"
    result["output_sha256"] = content.hexdigest()
    result["reasoning_sha256"] = reasoning.hexdigest()
    usage, events = result["usage"], result["events"]
    tokens = usage.get("completion_tokens", 0)
    result["usable"] = (result["error"] is None and result["done"] and bool(events)
                        and (result["finish"] == "stop" and 0 < tokens <= maximum
                             or result["finish"] == "length" and tokens == maximum)
                        and usage.get("prompt_tokens", 0) + tokens == usage.get("total_tokens")
                        and (counted is None or counted == usage.get("prompt_tokens")))
    result["ttft_s"] = events[0] - result["start"] if events else None
    result["sse_gaps_s"] = quantiles([right - left for left, right in zip(events, events[1:])])
    result["per_stream_api_tokens_over_first_to_last_sse_s"] = (tokens / (events[-1] - events[0])
                                                               if len(events) >= 2 and events[-1] > events[0] else None)
    return result


def run_cohort(server, args, fixtures, maximum, measured):
    barrier = threading.Barrier(args.concurrency)
    abort, lock, sockets = threading.Event(), threading.Lock(), {}
    deadline = time.monotonic() + args.timeout
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(stream, server, lane, item, maximum, barrier, deadline, abort, sockets, lock)
                   for lane, item in enumerate(fixtures)]
        try:
            if measured:
                while not all(item.done() for item in futures) and time.monotonic() < deadline:
                    sample_start = time.monotonic()
                    try:
                        samples.append(server.sample())
                    except (OSError, ValueError, BenchmarkError, http.client.HTTPException) as exc:
                        samples.append({"time": sample_start, "error": type(exc).__name__})
                    time.sleep(max(0, min(deadline - time.monotonic(), args.interval - (time.monotonic() - sample_start))))
            else:
                concurrent.futures.wait(futures, timeout=max(0, deadline - time.monotonic()))
        finally:
            if not all(item.done() for item in futures):
                abort.set()
                with lock:
                    for sock in sockets.values():
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
        return [item.result() for item in futures], samples


def decode_windows(samples, begin, end, concurrency):
    runs, current, old_ids = [], [], None
    for row in samples:
        decoders = [slot for slot in row.get("slots", []) if slot["state"] == "decoding"]
        work = row.get("work", {})
        ids = tuple(sorted(slot["request"] for slot in decoders if "request" in slot))
        valid = (begin <= row.get("start", -1) and row.get("end", math.inf) <= end
                 and row.get("ready") and len(decoders) == concurrency
                 and len({slot.get("slot") for slot in decoders}) == concurrency
                 and work.get("active") == work.get("occupied_slots") == concurrency
                 and work.get("queued") == work.get("waiting") == 0
                 and (not ids or len(set(ids)) == concurrency))
        if current and (not valid or ids != old_ids):
            runs.append(current)
            current = []
        if valid:
            current.append(row)
            old_ids = ids
    if current:
        runs.append(current)
    return runs


def summarize(before, after, records, samples, args):
    checked = [before, *(row for row in samples if "counter" in row), after]
    stable = len({row["instance"] for row in checked}) == 1
    monotonic = all(left["counter"] <= right["counter"] for left, right in zip(checked, checked[1:]))
    timeline = [row["events"] for row in records]
    elapsed = max(row["end"] for row in records) - min(row["start"] for row in records)
    tokens = sum(row["usage"].get("completion_tokens", 0) for row in records)
    common_begin = max(events[0] for events in timeline) if all(timeline) else math.inf
    common_end = min(events[-1] for events in timeline) if all(timeline) else -math.inf
    runs = decode_windows(samples, common_begin, common_end, args.concurrency)
    longest = max(runs, key=lambda rows: rows[-1]["time"] - rows[0]["time"], default=[])
    span = longest[-1]["time"] - longest[0]["time"] if longest else 0.0
    output = longest[-1]["counter"] - longest[0]["counter"] if longest else 0
    rates = []
    for index, left in enumerate(longest):
        for right in longest[index + 1:]:
            delta = right["time"] - left["time"]
            if delta >= 5:
                rates.append((right["counter"] - left["counter"]) / delta)
                break
    checks = {"same_server_instance": stable, "output_counter_monotonic": monotonic,
              "all_streams_complete": all(row["usable"] for row in records),
              "completed_cohort_exclusive": after["completed"] - before["completed"] == args.concurrency,
              "output_cohort_exclusive": after["counter"] - before["counter"] == tokens,
              "simultaneous_server_decode": span >= args.min_steady}
    return {"checks": checks, "qualified": all(checks.values()), "whole_batch_tokens": tokens,
            "whole_batch_elapsed_s": elapsed, "whole_batch_api_tok_s": tokens / elapsed if elapsed else None,
            "steady_decode_tok_s": output / span if span and stable and monotonic else None,
            "steady_decode_tokens": output, "steady_decode_elapsed_s": span, "steady_decode_samples": len(longest),
            "steady_five_second_tok_s": quantiles(rates) if stable and monotonic else None,
            "simultaneous_sse_elapsed_s": max(0, common_end - common_begin),
            "distinct_decoding_slots_peak": max((len({slot.get("slot") for slot in row.get("slots", [])
                                                        if slot["state"] == "decoding"}) for row in samples), default=0),
            "ttft_s": quantiles([row["ttft_s"] for row in records if row["ttft_s"] is not None]),
            "max_sse_gap_s": max((row["sse_gaps_s"]["max"] for row in records if row["sse_gaps_s"]), default=None),
            "per_stream_api_tok_s": quantiles([row["per_stream_api_tokens_over_first_to_last_sse_s"] for row in records
                                               if row["per_stream_api_tokens_over_first_to_last_sse_s"] is not None]),
            "exact_max_output_requests": sum(row["finish"] == "length" and row["usage"].get("completion_tokens") == args.max_tokens
                                             for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=0, help="Exact synthetic target, within two tokens")
    parser.add_argument("--mode", choices=("sampled", "greedy"), default="sampled")
    parser.add_argument("--workload", choices=("essay", "code", "mixed"), default="mixed")
    parser.add_argument("--fixture", default="decode-throughput-v1")
    parser.add_argument("--presence-penalty", type=float, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0, help="Run one short-prompt full-width warmup with this output cap")
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--min-steady", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", args.fixture) is not None, "Invalid synthetic fixture label")
    require(1 <= args.max_tokens <= 8192 and 0 <= args.warmup <= 4096 and 1 <= args.rounds <= 10
            and (args.context_tokens == 0 or 128 <= args.context_tokens <= 200000)
            and .1 <= args.interval <= 5 and 0 < args.min_steady < args.timeout <= 1800
            and -2 <= args.presence_penalty <= 2, "Invalid benchmark bounds")
    cfg = json.loads(args.config.read_bytes())
    port = args.port or int(cfg.get("port", 8080))
    require(1 <= port <= 65535, "Invalid port")
    loopback_only(port)
    server = Server(cfg, port)
    initial = server.idle()
    require(initial["work"].get("slots_effective", 0) >= args.concurrency,
            "Effective slots are below requested concurrency")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as artifact:
        report = {"version": 1, "source_sha256": sha(Path(__file__).read_bytes()), "listener": f"127.0.0.1:{port}",
                  "started_epoch": time.time(), "parameters": {key: value for key, value in vars(args).items()
                                                                 if key not in ("config", "out", "port")}, "rounds": []}
        try:
            if args.warmup:
                fixtures = [fixture(server, args, lane, "warmup", 0, args.warmup) for lane in range(args.concurrency)]
                warmup, _ = run_cohort(server, args, fixtures, args.warmup, False)
                require(all(row["usable"] for row in warmup), "Warmup failed")
                report["warmup_requests_complete"] = len(warmup)
            for round_id in range(args.rounds):
                fixtures = [fixture(server, args, lane, round_id, args.context_tokens, args.max_tokens)
                            for lane in range(args.concurrency)]
                before = server.idle()
                records, samples = run_cohort(server, args, fixtures, args.max_tokens, True)
                row = {"round": round_id, "records": records, "samples": samples, "before": before}
                report["rounds"].append(row)
                row["after"] = server.idle()
                loopback_only(port)
                row["summary"] = summarize(before, row["after"], records, samples, args)
                print(json.dumps({"round": round_id, **row["summary"]}), flush=True)
            report["qualified"] = all(row["summary"]["qualified"] for row in report["rounds"])
        except Exception as exc:
            report["error"] = str(exc) if isinstance(exc, BenchmarkError) else type(exc).__name__
            report["qualified"] = False
        json.dump(report, artifact, indent=2)
        artifact.write("\n")
    if "error" in report:
        print(json.dumps({"error": report["error"]}), flush=True)
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc) if isinstance(exc, BenchmarkError) else type(exc).__name__}), flush=True)
        raise SystemExit(2)
