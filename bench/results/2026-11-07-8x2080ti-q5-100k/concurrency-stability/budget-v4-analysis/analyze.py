#!/usr/bin/env python3
"""Offline v4-only audit. Reads saved artifacts; never contacts a service or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

EXPECTED_SERVER = "40c5e9dfc96ab208b6514bd2f5f48986afb3e01b3927f873cff8d358d62528c9"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change(old, new):
    return (new / old - 1) * 100 if old else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path(__file__).resolve().parent.parent / "budget-v4-c4-sampled")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    base, dest = args.input.resolve(), args.out.resolve()
    manifest = read(base / "manifest.json")
    arms = {label: base / tag for label, tag in (("off", "01-off"), ("on", "02-on"))}
    metadata = {k: read(p / "metadata.json") for k, p in arms.items()}
    assert manifest["server_sha256"] == EXPECTED_SERVER
    for label, m in metadata.items():
        assert m["config"]["reasoning_budget_engine_cap"] is True
        assert all(v == 0 for v in m["suite_returncodes"].values())
        assert m["startup_facts"]["all_experts_resident"]
        assert m["startup_facts"]["all_eight_have_3_80_gib_own_buffers"]
        assert m["engine_sha256"] == manifest["same_binary_sha256"]
        assert m["adjacent_library_sha256"] == manifest["adjacent_library_sha256"]
    assert metadata["off"]["config_except_variables_log_key_sha256"] == metadata["on"]["config_except_variables_log_key_sha256"]
    environment = {k: m["config"]["env"] for k, m in metadata.items()}
    env_changes = {k: {a: environment[a].get(k) for a in arms}
                   for k in set(environment["off"]) | set(environment["on"])
                   if environment["off"].get(k) != environment["on"].get(k)}
    assert set(env_changes) == {"STRATA_BATCH_MTP", "STRATA_PIPED_MTP", "STRATA_BATCH_SELF_COMMIT"}
    source_files = [base / "manifest.json"]
    source_files += [p / "metadata.json" for p in arms.values()]

    summaries = {k: read(p / "performance/c4/summary.json") for k, p in arms.items()}
    requests = {}
    for label, p in arms.items():
        rows = {}
        for f in sorted((p / "performance/c4").glob("*.json")):
            item = read(f)
            if isinstance(item, dict) and "role" in item:
                rows[item["name"]] = item
                source_files.append(f)
        requests[label] = rows
        source_files.append(p / "performance/c4/summary.json")
    assert requests["off"].keys() == requests["on"].keys() and len(requests["off"]) == 6
    paired = []
    for name, a in requests["off"].items():
        b = requests["on"][name]
        channels = {key: a["channel_output_sha256"][key] == b["channel_output_sha256"][key]
                    for key in a["channel_output_sha256"]}
        row = {"name": name, "request_equal": a["request_sha256"] == b["request_sha256"],
               "request_sha256": a["request_sha256"],
               "usage_equal": a["usage"] == b["usage"], "channels_equal": channels,
               "output_equal": a["output_sha256"] == b["output_sha256"], "usage": a["usage"],
               "output_sha256": a["output_sha256"], "channel_output_sha256": a["channel_output_sha256"],
               "timeline": {}}
        for label, r in (("off", a), ("on", b)):
            z = summaries[label]["scenario_start_epoch"]
            row["timeline"][label] = {"start_s": r["start_epoch"] - z, "first_output_s": r["start_epoch"] + r["ttft_s"] - z,
                                       "last_output_s": r["chunks"][-1]["epoch"] - z, "end_s": r["end_epoch"] - z,
                                       "ttft_s": r["ttft_s"], "output_bytes": r["output_bytes"],
                                       "finish_reason": r["finish_reason"], "successful": r["successful"]}
        assert row["request_equal"] and row["usage_equal"] and row["output_equal"] and all(channels.values())
        assert a["successful"] and b["successful"] and a["cached_tokens"] == b["cached_tokens"] == 0
        paired.append(row)

    phase_data, chronology = {}, {}
    for label, s in summaries.items():
        assert s["error"] is None and s["baseline_occupancy_qualified"]
        phase_data[label] = {name: {**p, "occupancy": s["occupancy"][name]} for name, p in s["phases"].items()}
        last = sorted(r["end_epoch"] for r in requests[label].values() if r["role"] == "resident")
        chronology[label] = {"startup_to_baseline_s": s["phases"]["baseline"]["start_epoch"] - s["scenario_start_epoch"],
                             "baseline_s": s["phases"]["baseline"]["duration_s"],
                             "mixed_s": s["phases"]["mixed"]["duration_s"],
                             "recovery_s": s["phases"]["recovery"]["duration_s"],
                             "drain_s": s["scenario_end_epoch"] - s["phases"]["recovery"]["end_epoch"],
                             "final_single_resident_interval_s": last[-1] - last[-2]}
    whole = {k: s["whole_scenario_completed_api_token_rate"] for k, s in summaries.items()}
    assert all(x["tokens"] == 8320 and x["requests_missing_usage"] == 0 for x in whole.values())

    pf = {k: {r["name"]: r for r in read(p / "prefill/results.json")} for k, p in arms.items()}
    prefill_pairs = []
    for name, a in pf["off"].items():
        b = pf["on"][name]
        row = {"name": name, "request_equal": a["request_sha256"] == b["request_sha256"],
               "usage_equal": a["usage"] == b["usage"], "output_equal": a["output_sha256"] == b["output_sha256"],
               "channels_equal": a["channel_output_sha256"] == b["channel_output_sha256"],
               "usage": a["usage"], "valid": a["valid"] and b["valid"],
               "off": {"ttft_s": a["ttft_s"], **a["engine"]}, "on": {"ttft_s": b["ttft_s"], **b["engine"]}}
        assert all(row[k] for k in ("request_equal", "usage_equal", "output_equal", "channels_equal", "valid"))
        prefill_pairs.append(row)
    assert len(prefill_pairs) == 6
    prefill_medians = {}
    for kind in ("cold", "warm"):
        prefill_medians[kind] = {}
        for metric in ("ttft_s", "read_ms", "decode_ms", "decode_tps"):
            values = {label: [r[label][metric] for r in prefill_pairs if r["name"].startswith(kind)] for label in arms}
            med = {label: statistics.median(v) for label, v in values.items()}
            prefill_medians[kind][metric] = {"values": values, **med, "change_pct": change(med["off"], med["on"])}
    source_files += [p / "prefill/results.json" for p in arms.values()]

    budget = {k: read(p / "budget/budget-correctness.json") for k, p in arms.items()}
    budget_pairs = []
    by_name = {k: {r["name"]: r for r in d["records"]} for k, d in budget.items()}
    for name, a in by_name["off"].items():
        b = by_name["on"][name]
        cancelled = bool(a["intentional_cancel"])
        row = {"name": name, "intentional_cancel": cancelled,
               "request_equal": a["request_sha256"] == b["request_sha256"],
               "output_equal": a["output_sha256"] == b["output_sha256"],
               "reasoning_equal": a["reasoning_sha256"] == b["reasoning_sha256"],
               "usage_equal": a["usage"] == b["usage"],
               "cancel_completed_in_both": bool(a["closed_mid_generation"] and b["closed_mid_generation"]),
               "off_usage": a["usage"], "on_usage": b["usage"]}
        assert row["request_equal"]
        if not cancelled:
            assert row["output_equal"] and row["reasoning_equal"] and row["usage_equal"]
        else:
            row["cancel_output_bytes"] = {"off": len(a["output_text"].encode()), "on": len(b["output_text"].encode())}
            row["cancel_shorter_output_is_prefix"] = (a["output_text"].startswith(b["output_text"])
                                                       or b["output_text"].startswith(a["output_text"]))
            assert row["cancel_completed_in_both"] and row["cancel_shorter_output_is_prefix"]
        budget_pairs.append(row)
    assert sum(not r["intentional_cancel"] for r in budget_pairs) == 9
    continuation = {k: [{"name": c["name"], **c["continuation"]} for c in d["continuations"]] for k, d in budget.items()}
    assert all(d["all_checks_passed"] and d["config_reasoning_budget_engine_cap"] for d in budget.values())
    assert all(c["read"] == 19 for v in continuation.values() for c in v)
    source_files += [p / "budget/budget-correctness.json" for p in arms.values()]

    # Associate performance wrap continuations by their exact synthetic prompt length; no content is copied.
    perf_wrap = {}
    pattern = re.compile(r'prompt 10347 tokens = (\d+) reused \+ (\d+) read in (\d+) ms')
    for label, p in arms.items():
        log = p / "engine.log"
        perf_wrap[label] = [{"line": i, "reused": int(m[1]), "read": int(m[2]), "read_ms": int(m[3])}
                            for i, line in enumerate(log.read_text(errors="replace").splitlines(), 1)
                            if (m := pattern.search(line))]
        assert len(perf_wrap[label]) == 4 and all(r["read"] == 19 for r in perf_wrap[label])
        source_files.append(log)

    result = {"scope": "v4 only; one off/on C4 sampled pair, both server cap=true; no v3 results pooled",
              "server_sha256": EXPECTED_SERVER, "engine_sha256": manifest["same_binary_sha256"],
              "library_sha256": manifest["adjacent_library_sha256"], "cap_both_true": True,
              "env_changes": env_changes, "startup_gates": {k: m["startup_facts"] for k, m in metadata.items()},
              "performance_pairs": paired, "whole_scenario": whole,
              "whole_throughput_change_pct": change(whole["off"]["tokens_per_s"], whole["on"]["tokens_per_s"]),
              "phases": phase_data, "chronology": chronology,
              "recovery": {k: {a: b for a, b in s["recovery"].items() if a != "bins"} for k, s in summaries.items()},
              "prefill_pairs": prefill_pairs, "prefill_medians": prefill_medians,
              "budget_pairs": budget_pairs, "budget_continuations": continuation, "performance_wrap_reads": perf_wrap,
              "source_sha256": {str(p.relative_to(base)): sha(p) for p in source_files},
              "limits": ["One sequential off/on pair; no statistical confidence or durable speedup claim.",
                         "Resident contexts are actual 8793 tokens; injectors are 8793 and 52080, not exact 10K/60K or concurrent 100K.",
                         "SSE inter-output-chunk gaps include reasoning/content/tool and are not per-token TBT.",
                         "Per-phase byte rates/fairness are descriptive; only full usage establishes the exact 8320-token comparison.",
                         "100K controls have three cold and three warm requests per arm, 256 output tokens; no tail SLO.",
                         "Budget continuation numeric log association is by synthetic length, not engine request ID."]}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_assertions_passed": True, "performance_pairs": len(paired), "prefill_pairs": len(prefill_pairs),
                      "budget_complete_pairs": 9, "throughput_change_pct": result["whole_throughput_change_pct"],
                      "output": str(dest / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
