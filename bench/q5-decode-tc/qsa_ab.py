#!/usr/bin/env python3
"""Measure the opt-in SM75 FP16 QSA decode path on an isolated Q5 API."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_q5 import ENGINE_SHA, launch, note, preflight
from capture_run import ready
import run_ab as service


LAB = Path(__file__).resolve().parent
RESULTS = LAB / "qsa-ab"
CTRL = RESULTS / "split.control"


def select(split):
    tmp = CTRL.with_suffix(".next")
    tmp.write_text(f"{split}\n")
    tmp.replace(CTRL)


def run_case(arm, name, py, private, arguments):
    select(0 if arm == "v3" else 32)
    note(phase="started", arm=arm, case=name)
    filename = RESULTS / f"{arm}-{name}.json"
    command = [sys.executable, "-B", str(LAB / py), "--config", str(private), "--out", str(filename), *arguments]
    result = subprocess.run(command, cwd=LAB, timeout=450)
    note(phase="completed", arm=arm, case=name, exit=result.returncode)
    if result.returncode not in (0, 2) or not filename.is_file():
        raise RuntimeError(f"{name} {arm} did not produce a benchmark report")


def benchmarks(private):
    for width, order in ((8, ("v3", "fp16")), (16, ("fp16", "v3"))):
        for arm in order:
            run_case(arm, f"short{width}", "bench_decode.py", private,
                     ["--concurrency", str(width), "--fixture", f"sm75-qsa-short{width}", "--max-tokens", "768",
                      "--warmup", "128", "--min-steady", "7"])
    for width, order in ((8, ("fp16", "v3")), (16, ("v3", "fp16"))):
        for arm in order:
            run_case(arm, f"warm100k-{width}", "bench_decode_warm.py", private,
                     ["--concurrency", str(width), "--fixture", f"sm75-qsa-warm{width}", "--max-tokens", "768",
                      "--context-tokens", "100000", "--min-steady", "7"])
    for arm in ("v3", "fp16"):
        run_case(arm, "cold100k", "prefill_ab.py", private, ["--context", "100000", "--rounds", "3"])


def restore(config, original):
    production = launch([str(service.LAUNCH)], service.LIVE / "serve-orca.out")
    note(phase="production_loading", pid=production.pid)
    elapsed = ready(production, config["api_key"], config["host"])
    receipt = {"pid": production.pid, "startup_s": elapsed, "slots": 16,
               "original_configuration": service.CONFIG.read_bytes() == original}
    (RESULTS / "production-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    note(phase="production_ready", **receipt)


def main():
    os.umask(0o077)
    original = service.CONFIG.read_bytes()
    cfg = json.loads(original)
    exe = LAB / "ab-bin/strata"
    if not (hashlib.sha256(original).hexdigest() == service.BASE_CONFIG_SHA and cfg.get("api_key")
            and hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest() == ENGINE_SHA and os.access(exe, os.X_OK)):
        raise RuntimeError("Base Q5 deployment and staged A/B engine are required")
    RESULTS.mkdir()
    select(0)
    preflight(cfg)
    private = RESULTS / "loopback.private.json"
    process = None
    try:
        local = dict(cfg, host="127.0.0.1", exe=str(exe), log=str(RESULTS / "engine.log"), engine_ready_s=1100)
        local["env"] = dict(cfg["env"], STRATA_SM75_DECODE_ATTN_SPLIT_CONTROL=str(CTRL))
        private.write_text(json.dumps(local))
        process = launch([str(service.PYTHON), str(service.APP), "--engine", "strata", "--config", str(private),
                          "--host", "127.0.0.1", "--port", "8080"], RESULTS / "server.log")
        note(phase="loopback_loading", pid=process.pid)
        elapsed = ready(process, cfg["api_key"], "127.0.0.1")
        note(phase="loopback_ready", pid=process.pid, startup_s=elapsed)
        benchmarks(private)
    finally:
        if process is not None:
            service.stop(process.pid)
        private.unlink(missing_ok=True)
        restore(cfg, original)


if __name__ == "__main__":
    main()
