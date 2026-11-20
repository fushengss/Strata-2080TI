#!/usr/bin/env python3
"""Run the standalone local-TP diagnostic in an exclusive GPU window supplied by the operator.

This script does not stop/start a service, alter GPU clocks, download weights or compile anything.
Example (root arranges the maintenance window and binary):
  python3 run.py --binary /path/local_tp_bench --gguf /path/model-00001-of-00003.gguf \
    --devices 0,1 --output /path/local-tp/tp2
Repeat with --devices 0,1,2,3 for TP4. Inputs default to actual layers 3 and 6 and ten fixed experts.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import threading
import time


FIELDS = "timestamp,index,name,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,memory.free"
PERF_ENV = ("STRATA_SM75_PERF", "STRATA_EXP_MODE", "STRATA_NO_PCIE_GROUP", "STRATA_GROUPED_V1",
            "STRATA_BF16_TC", "STRATA_MMVQ_IL_ROWS", "STRATA_Q6_PACKED", "STRATA_Q8_PACKED",
            "STRATA_V2K", "STRATA_KQ_SPLIT", "STRATA_EXPERT_NIBBLE", "STRATA_NATIVE_MOE_COMBINE",
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def gpu_info(args):
    try:
        p = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def summarize(records, returncode):
    cases = [r for r in records if r.get("kind") == "routed_moe"]
    parity = [r for r in records if r.get("kind") == "parity"]
    collectives = [r for r in records if r.get("kind") == "host_allgather"]
    return {
        "returncode": returncode,
        "status": "skipped" if returncode == 77 else "completed" if returncode == 0 else "failed",
        "scope": "Single layer routed experts only; all token rows share the same ten experts. No router, shared expert, HC, GDN or attention.",
        "not_an_end_to_end_engine_tp_benchmark": True,
        "timing_limits": [
            "Wall latency includes main-thread submissions, device selection, host-mapped exchanges and all-rank synchronization.",
            "Compute-phase CUDA event times exclude host exchanges but are not a reconstructed cross-GPU critical path.",
            "Collective CUDA events include copy kernels and waits; they are not compute-only measurements.",
            "No result is an exact production candidate unless both decomposed and production comparisons are bitwise equal.",
            "No prefill performance is measured or claimed.",
            "The TP prototype uses separate per-expert GEMVs; losing to fused production does not disprove a future fused TP implementation.",
        ],
        "case_count": len(cases),
        "parity_case_count": len(parity),
        "all_tp_bitwise_decomposed": bool(cases) and all(r["tp_bitwise_decomposed"] for r in cases),
        "all_tp_bitwise_production": bool(cases) and all(r["tp_bitwise_production"] for r in cases),
        "all_collectives_bitwise": bool(collectives) and all(r["bitwise"] for r in collectives),
        "cases": [dict(r, wall_speedup_vs_production=(r["production_fused"]["wall_us_p50"] / r["row_tp"]["wall_us_p50"]),
                       exact_candidate=r["tp_bitwise_production"] and r["tp_bitwise_decomposed"]) for r in cases],
        "parity": parity,
        "collectives": collectives,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True, type=pathlib.Path)
    p.add_argument("--gguf", required=True)
    p.add_argument("--devices", required=True, choices=("0,1", "0,1,2,3"))
    p.add_argument("--layers", default="3,6")
    p.add_argument("--experts", default="5,42,79,116,153,190,227,264,301,338")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--parity-cases", type=int, default=4)
    p.add_argument("--output", required=True, type=pathlib.Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()):
        p.error("output directory must be empty; preserve earlier measurements")
    command = [str(a.binary.resolve()), "--gguf", a.gguf, "--devices", a.devices, "--layers", a.layers,
               "--experts", a.experts, "--warmup", str(a.warmup), "--iterations", str(a.iterations),
               "--parity-cases", str(a.parity_cases)]
    import os
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command,
                "binary_sha256": sha256(a.binary), "gpu_before": gpu_info(["--query-gpu=" + FIELDS, "--format=csv"]),
                "topology": gpu_info(["topo", "-m"]),
                "environment": {k: os.environ[k] for k in PERF_ENV if k in os.environ}}
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stop = threading.Event()

    def monitor():
        with (a.output / "gpu-samples.jsonl").open("w") as out:
            while not stop.is_set():
                out.write(json.dumps(gpu_info(["--query-gpu=" + FIELDS, "--format=csv"])) + "\n")
                out.flush()
                stop.wait(1.0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    start = time.monotonic()
    try:
        with (a.output / "results.jsonl").open("w") as out, (a.output / "stderr.log").open("w") as err:
            result = subprocess.run(command, stdout=out, stderr=err, check=False)
    finally:
        stop.set()
        thread.join(timeout=20)
    records = []
    for number, line in enumerate((a.output / "results.jsonl").read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid JSON on line {number}: {e}") from e
    summary = summarize(records, result.returncode)
    summary["wall_elapsed_seconds"] = time.monotonic() - start
    summary["gpu_after"] = gpu_info(["--query-gpu=" + FIELDS, "--format=csv"])
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("status", "case_count", "parity_case_count", "all_tp_bitwise_decomposed", "all_tp_bitwise_production", "all_collectives_bitwise")}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
