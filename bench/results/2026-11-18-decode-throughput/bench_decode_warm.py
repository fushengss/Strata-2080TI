#!/usr/bin/env python3
"""Seed one shared long document, then measure concurrent decode of distinct questions.

Run on the inference host with bench_decode.py beside this file. Strata's
strata_prefix.messages pins the first user message. Every benchmark lane must
report a near-full cache hit, and the original all-decoding checks also apply.
"""

import argparse
import json
from pathlib import Path
import random
import re
import time

import bench_decode as base


SEED_QUESTION = "Reply with exactly one word: Ready."
THINKING_DISABLED = {"enable_thinking": False}


def messages(document, question=None):
    result = [{"role": "user", "content": document}]
    if question is not None:
        result.append({"role": "user", "content": question})
    return result


def count(server, prompt):
    result = server.json("/v1/messages/count_tokens", {"model": server.model, "messages": prompt,
                         "thinking": {"type": "disabled"}, "max_tokens": 1}, 30)
    tokens = result.get("input_tokens")
    base.require(type(tokens) is int and tokens > 0, "Missing positive token count")
    return tokens


def calibrate(server, target, label):
    # Change the start of a fixture to distinguish independent shared prefixes.
    words = random.Random(20261118).choices(base.WORDS, k=target + 64)
    size = target - 20
    for _ in range(8):
        document = (f"Synthetic reference {label}. The following are background notes.\n\n"
                    + " ".join(words[:size]) + "\n\nEnd of reference document.")
        counted = count(server, messages(document))
        if abs(counted - target) <= 2:
            return document, counted
        size += target - counted
        base.require(0 < size <= len(words), "Shared document could not be calibrated")
    raise base.BenchmarkError("Shared document is more than two tokens off the target")


def seed_prefix(server, document, counted, args):
    prompt = messages(document, SEED_QUESTION)
    expected = count(server, prompt)
    before = server.idle()
    start = time.monotonic()
    data = server.json("/v1/chat/completions", {"model": server.model, "messages": prompt,
                       "strata_prefix": {"messages": 1}, "max_tokens": 1, "temperature": 0,
                       "presence_penalty": 0, "seed": 20261118, "chat_template_kwargs": THINKING_DISABLED},
                       args.seed_timeout)
    elapsed = time.monotonic() - start
    after = server.idle()
    usage = data.get("usage", {})
    tokens = base.numeric(usage, ("prompt_tokens", "completion_tokens", "total_tokens"))
    tokens.update(base.numeric(usage.get("prompt_tokens_details", {}), ("cached_tokens",)))
    choices = data.get("choices") or [{}]
    finish = choices[0].get("finish_reason")
    finish = finish if finish in ("stop", "length") else None
    # A warmed repeated fixture may already have the pinned document, but it
    # must still account for exactly the one seed request.
    checks = {"prompt_token_count_matches": tokens.get("prompt_tokens") == expected,
              "prompt_has_small_distinct_suffix": 0 < expected - counted <= args.max_fresh_tokens,
              "one_output_token": tokens.get("completion_tokens") == 1 and tokens.get("total_tokens") == expected + 1,
              "same_server_instance": before["instance"] == after["instance"],
              "completed_seed_exclusive": after["completed"] - before["completed"] == 1,
              "output_seed_exclusive": after["counter"] - before["counter"] == 1,
              "normal_finish": finish is not None}
    return {"elapsed_s": elapsed, "usage": tokens, "finish": finish, "counted_seed_prompt_tokens": expected,
            "checks": checks, "qualified": all(checks.values()), "instance": after["instance"]}


def lane_fixtures(server, args, document, round_id, counted):
    fixtures = []
    for lane in range(args.concurrency):
        topic = base.TOPICS[lane % len(base.TOPICS)]
        code = args.workload == "code" or args.workload == "mixed" and lane % 2 == 0
        instruction = (base.CODE if code else base.ESSAY).format(topic=topic)
        question = f"Task {lane + 1}, round {round_id + 1}. {instruction}"
        prompt = messages(document, question)
        expected = count(server, prompt)
        base.require(0 < expected - counted <= args.max_fresh_tokens, "Lane prompt has an unexpected suffix length")
        body = {"model": server.model, "messages": prompt, "strata_prefix": {"messages": 1},
                "max_tokens": args.max_tokens, "seed": 20261118 + lane, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": THINKING_DISABLED,
                "temperature": 0 if args.mode == "greedy" else 1.0, "top_p": .95, "top_k": 20,
                "presence_penalty": args.presence_penalty}
        fixtures.append((json.dumps(body, separators=(",", ":")).encode(), expected))
    return fixtures


def cache_summary(records, counted, args):
    lanes = []
    for row in records:
        usage = row["usage"]
        prompt, cached = usage.get("prompt_tokens"), usage.get("cached_tokens")
        fresh = prompt - cached if type(prompt) is int and type(cached) is int else None
        matched = (type(cached) is int and abs(cached - counted) <= args.max_prefix_deviation
                   and fresh is not None and 0 < fresh <= args.max_fresh_tokens)
        lanes.append({"lane": row["lane"], "prompt_tokens": prompt, "cached_tokens": cached, "fresh_tokens": fresh,
                      "matched_shared_prefix": matched})
    return {"lanes": lanes, "all_lanes_reused_shared_prefix": all(row["matched_shared_prefix"] for row in lanes),
            "cached_tokens": base.quantiles([row["cached_tokens"] for row in lanes if type(row["cached_tokens"]) is int]),
            "fresh_tokens": base.quantiles([row["fresh_tokens"] for row in lanes if row["fresh_tokens"] is not None])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--context-tokens", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--mode", choices=("sampled", "greedy"), default="sampled")
    parser.add_argument("--workload", choices=("essay", "code", "mixed"), default="mixed")
    parser.add_argument("--fixture", default="decode-warm-v1")
    parser.add_argument("--presence-penalty", type=float, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--interval", type=float, default=.25)
    parser.add_argument("--min-steady", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--seed-timeout", type=float, default=120)
    parser.add_argument("--max-prefix-deviation", type=int, default=64,
                        help="Allowed difference between count of the document-only request and each actual cache hit")
    parser.add_argument("--max-fresh-tokens", type=int, default=256)
    args = parser.parse_args()
    base.require(re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", args.fixture) is not None, "Invalid fixture label")
    base.require(512 <= args.context_tokens <= 200000 and 128 <= args.max_tokens <= 8192
                 and 1 <= args.rounds <= 10 and .1 <= args.interval <= 5
                 and 0 < args.min_steady < args.timeout <= 1800 and 0 < args.seed_timeout <= 600
                 and -2 <= args.presence_penalty <= 2 and 0 <= args.max_prefix_deviation <= 512
                 and 1 <= args.max_fresh_tokens <= 1024, "Invalid benchmark bounds")
    cfg = json.loads(args.config.read_bytes())
    port = args.port or int(cfg.get("port", 8080))
    base.require(1 <= port <= 65535, "Invalid port")
    base.loopback_only(port)
    server = base.Server(cfg, port)
    initial = server.idle()
    base.require(initial["work"].get("slots_effective", 0) >= args.concurrency,
                 "Effective slots are below requested concurrency")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as artifact:
        report = {"version": 1, "source_sha256": base.sha(Path(__file__).read_bytes()),
                  "base_sha256": base.sha(Path(base.__file__).read_bytes()), "listener": f"127.0.0.1:{port}",
                  "prefix_rule": {"messages": 1}, "started_epoch": time.time(),
                  "parameters": {key: value for key, value in vars(args).items() if key not in ("config", "out", "port")},
                  "rounds": []}
        try:
            started = time.monotonic()
            document, counted = calibrate(server, args.context_tokens, args.fixture)
            report.update(counted_document_prompt_tokens=counted, shared_document_sha256=base.sha(document.encode()),
                          calibration_s=time.monotonic() - started)
            report["seed"] = seed_prefix(server, document, counted, args)
            print(json.dumps({"phase": "seed", "counted_document_prompt_tokens": counted, **report["seed"]}), flush=True)
            base.require(report["seed"]["qualified"], "Seed request failed its usage or exclusivity checks")
            for round_id in range(args.rounds):
                fixtures = lane_fixtures(server, args, document, round_id, counted)
                before = server.idle()
                base.require(before["instance"] == report["seed"]["instance"], "Seeded server restarted")
                records, samples = base.run_cohort(server, args, fixtures, args.max_tokens, True)
                row = {"round": round_id, "records": records, "samples": samples, "before": before}
                report["rounds"].append(row)
                row["after"] = server.idle()
                base.loopback_only(port)
                row["summary"] = summary = base.summarize(before, row["after"], records, samples, args)
                row["cache"] = cache = cache_summary(records, counted, args)
                summary["checks"]["all_lanes_reused_shared_prefix"] = cache["all_lanes_reused_shared_prefix"]
                summary["qualified"] = all(summary["checks"].values())
                print(json.dumps({"round": round_id, **summary, "cache": cache}), flush=True)
            report["qualified"] = all(row["summary"]["qualified"] for row in report["rounds"])
        except Exception as exc:
            report["error"] = str(exc) if isinstance(exc, base.BenchmarkError) else type(exc).__name__
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
        print(json.dumps({"error": str(exc) if isinstance(exc, base.BenchmarkError) else type(exc).__name__}), flush=True)
        raise SystemExit(2)
