#!/usr/bin/env python3
"""Compare CPU and stable GPU grouping on the same loopback-only MMQ engine."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import run_ab as service


LAB = Path(__file__).resolve().parent
CANDIDATE = LAB / "candidate" / "strata"


def note(**record):
    print(json.dumps(record), flush=True)


def workload(name, config, script, args):
    result = subprocess.run([sys.executable, "-B", str(LAB / script), "--config", str(config),
                             "--out", str(LAB / f"{name}.json"), *args], cwd=LAB, timeout=850)
    note(phase="workload_done", name=name, exit=result.returncode)
    if result.returncode not in (0, 2):
        raise RuntimeError(f"Workload {name} failed ({result.returncode})")


def arm(name, flag, src, key):
    c = json.loads(src)
    c.update(host="127.0.0.1", exe=str(CANDIDATE), log=str(LAB / f"{name}.engine.log"))
    c["env"]["STRATA_MMQ_GPU_ROUTE"] = flag
    config = LAB / f"{name}.private.json"
    with config.open("x") as f:
        json.dump(c, f)
    child = None
    try:
        with (LAB / f"{name}.server.log").open("xb") as log:
            child = subprocess.Popen([str(service.PYTHON), str(service.APP), "--engine", "strata",
                                      "--config", str(config), "--host", "127.0.0.1", "--port", "8080"],
                                     cwd=service.APP.parent.parent, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        note(phase="loading_loopback", arm=name, pid=child.pid)
        ready = service.ready(child, key, "127.0.0.1")
        note(phase="loopback_ready", arm=name, startup_s=ready)
        workload(f"{name}-prefill", config, "prefill_ab.py", ["--context", "100000", "--rounds", "3"])
        marker = re.compile(r"CUDA(\d+) resident MMQ routes grouped on GPU")
        devices = sorted({int(marker.search(line)[1]) for line in (LAB / f"{name}.engine.log").read_text().splitlines()
                          if marker.search(line)})
        expected = list(range(8)) if flag == "1" else []
        note(phase="route_coverage", arm=name, devices=devices, expected=expected)
        if devices != expected:
            raise RuntimeError("Unexpected GPU grouping coverage")
        workload(f"{name}-mixed8", config, "bench_decode_insert.py",
                 ["--decoders", "8", "--rounds", "2", "--fixture", "gpu-mmq-insert8"])
        workload(f"{name}-mixed16", config, "bench_decode_insert.py",
                 ["--decoders", "15", "--saturated", "--rounds", "2", "--fixture", "gpu-mmq-insert16"])
    finally:
        if child is not None:
            service.stop(child.pid)
        config.unlink(missing_ok=True)


def main():
    os.umask(0o077)
    src = service.CONFIG.read_bytes()
    if hashlib.sha256(src).hexdigest() != service.BASE_CONFIG_SHA:
        raise RuntimeError("Production config changed since the verified baseline")
    cfg = json.loads(src)
    key = cfg["api_key"]
    if cfg["env"].get("STRATA_MMQ_RESIDENT_SORT_NE") != "1" or not CANDIDATE.is_file():
        raise RuntimeError("Expected production resident sort and a compiled test binary")
    lines = service.listeners()
    if len(lines) != 1 or lines[0].split()[3] != f'{cfg["host"]}:8080':
        raise RuntimeError("Expected one original listener")
    match = re.search(r"\bpid=(\d+),", lines[0])
    if not match:
        raise RuntimeError("Could not identify the original listener")
    pid = int(match[1])
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if os.fsencode(service.APP) not in command or os.fsencode(service.CONFIG) not in command:
        raise RuntimeError("Original listener identity changed")
    for _ in range(90):
        if service.status("127.0.0.1", key)["activity"]["in_flight"] == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Production did not drain in 180 seconds")
    try:
        note(phase="switching_to_loopback", pid=pid)
        service.stop(pid)
        with (LAB / "route-parity.log").open("xb") as log:
            subprocess.run([str(CANDIDATE.parent / "prefill_mmq_route_test")], cwd=LAB, stdout=log,
                           stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                           check=True, timeout=180)
        note(phase="gpu_route_permutations_passed")
        arm("cpu", "0", src, key)
        arm("gpu", "1", src, key)
    finally:
        if not service.listeners():
            with (service.LIVE / "serve-orca.out").open("ab") as log:
                original = subprocess.Popen([str(service.LAUNCH)], cwd=service.LIVE, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            note(phase="production_loading", pid=original.pid)
            elapsed = service.ready(original, key, cfg["host"])
            receipt = {"pid": original.pid, "startup_s": elapsed, "slots": 16,
                       "original_config_unchanged": service.CONFIG.read_bytes() == src}
            (LAB / "restore-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            note(phase="production_ready", **receipt)


if __name__ == "__main__":
    main()
