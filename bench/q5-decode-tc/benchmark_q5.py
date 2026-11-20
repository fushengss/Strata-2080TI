#!/usr/bin/env python3
"""Benchmark the selected Q5 deployment on loopback, then start its live API."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_ab as service


LAB = Path(__file__).resolve().parent
ENGINE_SHA = "d4cdb02e868a05c96f032d8c29993704bc72581d5af9c71269c7f8f03c58e60e"
APP_SHA = "e4a17c7565ef9f88d3e6b2752d3f0066c5b589e17511053dcde1ef3f888e0c7a"
GPU_QUERY = ("timestamp,index,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,"
             "utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,memory.used")


def note(**fields):
    print(json.dumps(fields), flush=True)


def launch(argv, log):
    with log.open("ab") as out:
        return subprocess.Popen(argv, cwd=service.APP.parent.parent, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)


def benchmark(name, script, cfg, args):
    note(phase="benchmark_start", name=name)
    with (LAB / f"{name}-gpu.csv").open("xb") as gpu:
        sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
                                    "--loop-ms=500"], stdout=gpu, stderr=subprocess.STDOUT)
        try:
            result = subprocess.run([sys.executable, "-B", str(LAB / script), "--config", str(cfg),
                                     "--out", str(LAB / f"{name}.json"), *args], cwd=LAB, timeout=780)
            note(phase="benchmark_complete", name=name, exit=result.returncode)
        finally:
            sampler.terminate()
            sampler.wait(timeout=15)


def preflight(cfg):
    sockets = service.listeners()
    if not sockets:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                         text=True, timeout=25).strip()
        if active:
            raise RuntimeError("Unexpected CUDA owner with no API listener")
        return
    found = re.search(r"\bpid=(\d+),", sockets[0]) if len(sockets) == 1 else None
    if not found or sockets[0].split()[3] != f'{cfg["host"]}:8080':
        raise RuntimeError("An unexpected process is listening on 8080")
    pid = int(found[1])
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if os.fsencode(service.APP) not in args or os.fsencode(service.CONFIG) not in args:
        raise RuntimeError("8080 does not belong to the selected Q5 production instance")
    for _ in range(90):
        if service.status("127.0.0.1", cfg["api_key"])["activity"]["in_flight"] == 0:
            service.stop(pid)
            return
        time.sleep(2)
    raise RuntimeError("The production service did not drain in 180 seconds")


def run_cases(private):
    benchmark("cold100k", "prefill_ab.py", private, ["--context", "100000", "--rounds", "3"])
    for width, fixture in ((8, "decode-throughput-width"), (16, "decode-throughput-v1")):
        benchmark(f"decode{width}", "bench_decode.py", private, ["--concurrency", str(width), "--fixture", fixture,
                  "--warmup", "128", "--max-tokens", "1024", "--rounds", "2"])


def start_production(cfg, prod_bytes):
    production = launch([str(service.LAUNCH)], service.LIVE / "serve-orca.out")
    note(phase="production_loading", pid=production.pid)
    elapsed = service.ready(production, cfg["api_key"], cfg["host"])
    receipt = {"pid": production.pid, "startup_s": elapsed, "slots": 16, "model": cfg["model_name"],
               "original_config_unchanged": service.CONFIG.read_bytes() == prod_bytes}
    (LAB / "production-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    note(phase="production_ready", **receipt)


def main():
    os.umask(0o077)
    prod_bytes = service.CONFIG.read_bytes()
    cfg = json.loads(prod_bytes)
    if hashlib.sha256(prod_bytes).hexdigest() != service.BASE_CONFIG_SHA or not cfg.get("api_key"):
        raise RuntimeError("Original authenticated Q5 config changed")
    for name, expected in ((Path(cfg["exe"]), ENGINE_SHA), (service.APP, APP_SHA)):
        if hashlib.sha256(name.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Deployment artifact changed: {name}")
    preflight(cfg)

    private = LAB / "loopback.private.json"
    test_cfg = dict(cfg, host="127.0.0.1", log=str(LAB / "loopback.engine.log"))
    test = None
    key = cfg["api_key"]
    try:
        with private.open("x") as out:
            json.dump(test_cfg, out)
        test = launch([str(service.PYTHON), str(service.APP), "--engine", "strata", "--config", str(private),
                       "--host", "127.0.0.1", "--port", "8080"], LAB / "loopback.server.log")
        note(phase="loopback_loading", pid=test.pid)
        elapsed = service.ready(test, key, "127.0.0.1")
        note(phase="loopback_ready", startup_s=elapsed, slots=16)
        run_cases(private)
    finally:
        if test:
            service.stop(test.pid)
        private.unlink(missing_ok=True)
        start_production(cfg, prod_bytes)


if __name__ == "__main__":
    main()
