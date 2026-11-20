#!/usr/bin/env python3
"""Time the two Q5 expert operators and trace both resident PP8 prefill paths."""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import bench_decode as bench
import run_ab as service


LAB = Path(__file__).resolve().parent
CANDIDATE = LAB.parent / "bench-20261119-native-prefill" / "candidate"
PORT = 8080


def note(**data):
    print(json.dumps(data), flush=True)


def save(name, data):
    (LAB / name).write_text(json.dumps(data, indent=2) + "\n")


def command(name, args, timeout, check=True, env=None):
    with (LAB / f"{name}.log").open("xb") as log:
        return subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=check,
                              timeout=timeout, env=env, cwd=LAB).returncode


def owns(pid, path, config):
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return os.fsencode(path) in args and os.fsencode(config) in args
    except FileNotFoundError:
        return False


def engine_pid(pid):
    tasks = Path(f"/proc/{pid}/task").glob("*")
    children = {int(v) for p in tasks for v in (p / "children").read_text().split()}
    matches = [child for child in children if Path(f"/proc/{child}/exe").resolve() == CANDIDATE / "strata"]
    if len(matches) != 1:
        raise RuntimeError("Expected one candidate engine below the owned launcher")
    return matches[0]


def check_listener(address, path, config):
    lines = service.listeners()
    if not lines:
        return None
    if len(lines) != 1 or lines[0].split()[3] != f"{address}:{PORT}":
        raise RuntimeError("Unexpected listener address during test")
    match = re.search(r"\bpid=(\d+),", lines[0])
    if not match or not owns(int(match[1]), path, config):
        raise RuntimeError("Listener is owned by a different application")
    return int(match[1])


def wait_loopback(launcher, config, key):
    start = time.monotonic()
    while time.monotonic() - start < 720:
        if launcher.poll() not in (0, None):
            raise RuntimeError(f"Nsight launch exited {launcher.returncode}")
        pid = check_listener("127.0.0.1", service.APP, config)
        if pid is not None:
            s = service.status("127.0.0.1", key)
            if s["concurrency"] != {"serving": 16, "requested": 16}:
                raise RuntimeError("Expected all 16 slots to load")
            return pid, engine_pid(pid)
        time.sleep(1)
    raise TimeoutError("Loopback server failed to start")


def operator_tests():
    baseline = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "STRATA_PF_FUSED_SM75_KQ": "1", "STRATA_MMQ_KQ": "1"}
    for dtype in ("Q5_1", "Q8_0"):
        rc = command(f"operator-{dtype}", [str(CANDIDATE / "prefill_fused_iq_test"),
                     f"--only=Q5_K / {dtype}", "--unset-pf-fused", "--no-ref", "--chunks=3072"],
                     timeout=180, check=False, env=baseline)
        text = (LAB / f"operator-{dtype}.log").read_text()
        timing = [line for line in text.splitlines() if " - timing part:" in line]
        note(phase="operator", down=dtype, exit=rc, timing=timing)
        if not timing:
            raise RuntimeError(f"Operator {dtype} produced no timing")


def connection_clock():
    lo = time.monotonic_ns()
    epoch = time.time_ns()
    hi = time.monotonic_ns()
    return epoch - (lo + hi) // 2


def workload(api, opts, fixture, cap, *, cold):
    before = api.idle()
    offset = connection_clock()
    rows, _ = bench.run_cohort(api, opts, [fixture], cap, False)
    after = api.idle()
    row, = rows
    offset_end = connection_clock()
    bench.require(row["usable"] and before["instance"] == after["instance"]
                  and after["completed"] == before["completed"] + 1, "Isolated SSE request failed")
    bench.require(abs(offset_end - offset) < 2_000_000, "System clock changed during request")
    if cold:
        bench.require(row["usage"].get("cached_tokens") == 0 and row["usage"].get("prompt_tokens") == 100000
                      and row["native_timings"]["prompt_n"] == 100000, "Expected a cold 100,000-token read")
    return {**{k: row[k] for k in ("usage", "native_timings", "ttft_s", "request_sha256", "output_sha256")},
            "request_epoch_ns": round(row["start"] * 1e9) + offset,
            "first_sse_epoch_ns": round(row["events"][0] * 1e9) + offset,
            "request_done_epoch_ns": round(row["end"] * 1e9) + offset}


def profile_arm(arm, enabled, src, key):
    config = LAB / f"{arm}.private.json"
    cfg = json.loads(src)
    cfg.update(host="127.0.0.1", port=PORT, exe=str(CANDIDATE / "strata"), log=str(LAB / f"{arm}.engine.log"))
    cfg["env"]["STRATA_PF_FUSED_SM75_KQ"] = enabled
    with config.open("x") as out:
        json.dump(cfg, out)
    session = f"stratapp8_{arm}_{os.getpid()}"
    prefix = LAB / f"{arm}-prefill"
    pid = None
    launcher = None
    capture = False
    try:
        with (LAB / f"{arm}.server.log").open("xb") as log:
            launcher = subprocess.Popen([
                "nsys", "launch", f"--session-new={session}", "--trace=cuda,osrt,nvtx",
                "--cuda-graph-trace=node", "--cuda-event-trace=false", "--osrt-threshold=10000",
                "--show-output=true", "--wait=all", str(service.PYTHON), str(service.APP),
                "--engine", "strata", "--config", str(config), "--host", "127.0.0.1", "--port", str(PORT),
            ], cwd=service.APP.parent.parent, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        note(phase="arm_loading", arm=arm, session=session, launcher=launcher.pid)
        pid, engine = wait_loopback(launcher, config, key)
        note(phase="loopback_ready", arm=arm, pid=pid, engine=engine)
        bench.loopback_only(PORT)
        api = bench.Server(cfg, PORT)
        opts = SimpleNamespace(mode="greedy", workload="code", fixture="pp8-critical-path",
                               presence_penalty=0, timeout=240, concurrency=1)
        warm = bench.fixture(api, opts, 0, "warm", 3072, 8)
        plain = bench.fixture(api, opts, 0, "plain", 100000, 8)
        traced = bench.fixture(api, opts, 0, "trace", 100000, 8)
        workload(api, opts, warm, 8, cold=False)
        plain_result = workload(api, opts, plain, 8, cold=True)
        note(phase="untraced_prefill", arm=arm, **plain_result)
        bench.loopback_only(PORT)
        command(f"{arm}-nsys-start", ["nsys", "start", f"--session={session}", "--sample=none",
                                     "--cpuctxsw=none", f"--output={prefix}"], timeout=60)
        capture = True
        capture_start = time.time_ns()
        traced_result = workload(api, opts, traced, 8, cold=True)
        bench.loopback_only(PORT)
        capture_end = time.time_ns()
        command(f"{arm}-nsys-stop", ["nsys", "stop", f"--session={session}"], timeout=180)
        capture = False
        if not prefix.with_suffix(".nsys-rep").is_file():
            raise RuntimeError("Nsight report missing")
        data = {"arm": arm, "server_pid": pid, "engine_pid": engine, "untraced": plain_result,
                "traced": traced_result, "capture_start_epoch_ns": capture_start, "capture_end_epoch_ns": capture_end}
        save(f"{arm}-request.json", data)
        note(phase="trace_saved", arm=arm, **traced_result)
    finally:
        if capture:
            command(f"{arm}-nsys-recover", ["nsys", "stop", f"--session={session}"], timeout=180, check=False)
        if launcher is not None:
            command(f"{arm}-nsys-shutdown", ["nsys", "shutdown", f"--session={session}", "--kill=sigterm"],
                    timeout=90, check=False)
        pid = check_listener("127.0.0.1", service.APP, config) or pid
        if pid and owns(pid, service.APP, config):
            service.stop(pid)
        config.unlink(missing_ok=True)


def main():
    os.umask(0o077)
    src = service.CONFIG.read_bytes()
    if hashlib.sha256(src).hexdigest() != service.BASE_CONFIG_SHA:
        raise RuntimeError("Production config drift since preceding A/B")
    cfg = json.loads(src)
    key = cfg["api_key"]
    pid = check_listener(cfg["host"], service.APP, service.CONFIG)
    if pid is None:
        raise RuntimeError("Expected existing production instance")
    for _ in range(90):
        if service.status("127.0.0.1", key)["activity"]["in_flight"] == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Production did not drain in 180 seconds")
    try:
        note(phase="switching_to_loopback", pid=pid)
        service.stop(pid)
        operator_tests()
        profile_arm("native", "1", src, key)
        profile_arm("mmq", "0", src, key)
    finally:
        if not service.listeners():
            with (service.LIVE / "serve-orca.out").open("ab") as out:
                original = subprocess.Popen([str(service.LAUNCH)], cwd=service.LIVE, stdin=subprocess.DEVNULL,
                                            stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            note(phase="production_loading", pid=original.pid)
            startup = service.ready(original, key, cfg["host"])
            verified = {"pid": original.pid, "startup_s": startup, "slots": 16,
                        "original_config_unchanged": service.CONFIG.read_bytes() == src}
            save("restore-receipt.json", verified)
            note(phase="production_ready", **verified)
        else:
            raise RuntimeError("A listener remains after the test, so the original cannot be restored")


if __name__ == "__main__":
    main()
