#!/usr/bin/env python3
"""Restart two loopback-only prefill arms, then restore the existing service."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


LAB = Path(__file__).resolve().parent
LIVE = Path("/srv/Strata")
CONFIG = LIVE / "strata-orca-q5_k_m.57.json"
LAUNCH = LIVE / "run-orca-q5_k_m.57.sh"
APP = Path("/srv/Strata-decode16-20261118/serve/server.py")
PYTHON = LIVE / ".venv/bin/python"
BASE_CONFIG_SHA = "709d283a5314bb957c56d48f7ac8eba2593ec8b9546e2a3c14575f32b66ba151"


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=8).strip()


def listeners():
    return output(["ss", "-H", "-ltnp", "sport", "=", ":8080"]).splitlines()


def status(host, key):
    c = http.client.HTTPConnection(host, 8080, timeout=4)
    try:
        c.request("GET", "/v1/status", headers={"Authorization": "Bearer " + key})
        with c.getresponse() as r:
            if r.status != 200:
                raise RuntimeError(f"Status endpoint returned {r.status}")
            return json.load(r)
    finally:
        c.close()


def birth(pid):
    try:
        s = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if s[0] == "Z" else s[19]
    except FileNotFoundError:
        return None


def stop(pid):
    pids = {pid}
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            pids.update(int(v) for v in (task / "children").read_text().split())
        except FileNotFoundError:
            pass
    snapshot = {p: birth(p) for p in pids}
    if snapshot[pid]:
        os.kill(pid, signal.SIGTERM)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and any(b and birth(p) == b for p, b in snapshot.items()):
        time.sleep(.2)
    for p, b in snapshot.items():
        if b and birth(p) == b:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for _ in range(60):
        if not listeners() and not output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
            return
        time.sleep(1)
    raise RuntimeError("Port or CUDA resources were not released")


def ready(proc, key, address):
    t0 = time.monotonic()
    while time.monotonic() - t0 < 720:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited ({proc.returncode})")
        lines = listeners()
        if lines:
            if not all(line.split()[3] == f"{address}:8080" and f"pid={proc.pid}," in line for line in lines):
                raise RuntimeError("Unexpected listener address or PID")
            try:
                obj = status("127.0.0.1", key)
            except (OSError, http.client.HTTPException, ValueError, RuntimeError):
                time.sleep(1)
                continue
            if obj.get("concurrency") != {"serving": 16, "requested": 16}:
                raise RuntimeError("Expected exactly 16 effective slots")
            return round(time.monotonic() - t0, 1)
        time.sleep(1)
    raise RuntimeError("Server did not become ready")


def test(name, cfg_path):
    cases = [
        ("prefill", "prefill_ab.py", []),
        ("decode", "bench_decode.py", ["--concurrency", "16", "--warmup", "128", "--max-tokens", "1024",
                                       "--fixture", "native-prefill-decode"]),
        ("mixed", "bench_decode_mixed.py", []),
    ]
    for tag, script, args in cases:
        code = subprocess.run([sys.executable, "-B", str(LAB / script), "--config", str(cfg_path),
                               "--out", str(LAB / f"{name}-{tag}.json"), *args], cwd=LAB, timeout=1000).returncode
        print(json.dumps({"arm": name, "test": tag, "exit": code}), flush=True)
        if code not in (0, 2):
            raise RuntimeError(f"{name} {tag} failed ({code})")


def main():
    os.umask(0o077)
    src = CONFIG.read_bytes()
    if hashlib.sha256(src).hexdigest() != BASE_CONFIG_SHA:
        raise RuntimeError("Production configuration changed since preparation")
    base = json.loads(src)
    key = base["api_key"]
    exe = LAB / "candidate" / "strata"
    if not exe.is_file() or not os.access(exe, os.X_OK):
        raise RuntimeError("Candidate executable missing")
    if not (exe.parent/"libstrata_kq.so").is_file():
        raise RuntimeError("Rollback MMQ library missing from test executable directory")
    arms = []
    for name, enabled in (("mmq", "0"), ("native", "1")):
        c = json.loads(src)
        c.update(host="127.0.0.1", log=str(LAB / f"{name}.native.log"), exe=str(exe))
        c["env"]["STRATA_PF_FUSED_SM75_KQ"] = enabled
        path = LAB / f"{name}.json"
        with path.open("x") as f:
            json.dump(c, f)
        arms.append((name, path))
    current = listeners()
    if len(current) != 1 or current[0].split()[3] != "0.0.0.0:8080":
        raise RuntimeError("Expected one original production listener")
    pid = int(re.search(r"pid=(\d+)", current[0])[1])
    a = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if os.fsencode(APP) not in a or os.fsencode(CONFIG) not in a:
        raise RuntimeError("Original listener identity changed")
    for _ in range(45):
        if status("127.0.0.1", key)["activity"]["in_flight"] == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError("Production did not drain within 90 seconds")
    print(json.dumps({"phase": "switching_to_loopback", "old_pid": pid}), flush=True)
    child = None
    try:
        stop(pid)
        parity = exe.parent / "prefill_fused_iq_test"
        with (LAB/"native-kernel-parity.log").open("xb") as log:
            subprocess.run([str(parity), "--only=Q5_K / Q", "--unset-pf-fused", "--no-timing"],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "STRATA_PF_FUSED_SM75_KQ": "1",
                                "STRATA_MMQ_KQ": "1"}, stdout=log, stderr=log, cwd=LAB, check=True, timeout=300)
        print(json.dumps({"phase": "sm75_q5_kernel_parity_passed"}), flush=True)
        for name, cfg in arms:
            with (LAB / f"{name}.server.log").open("xb") as log:
                child = subprocess.Popen([str(PYTHON), str(APP), "--engine", "strata", "--config", str(cfg),
                                          "--host", "127.0.0.1", "--port", "8080"], cwd=APP.parent.parent,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
            print(json.dumps({"phase": "arm_loading", "arm": name, "pid": child.pid}), flush=True)
            elapsed = ready(child, key, "127.0.0.1")
            print(json.dumps({"phase": "loopback_ready", "arm": name, "startup_s": elapsed}), flush=True)
            test(name, cfg)
            stop(child.pid)
            child = None
    finally:
        if child:
            stop(child.pid)
        if not listeners():
            with (LIVE / "serve-orca.out").open("ab") as stdout:
                restored = subprocess.Popen([str(LAUNCH)], cwd=LIVE, stdin=subprocess.DEVNULL, stdout=stdout,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({"phase": "production_loading", "pid": restored.pid}), flush=True)
            elapsed = ready(restored, key, base["host"])
            report = {"phase": "production_ready", "pid": restored.pid, "startup_s": elapsed,
                      "slots": 16, "original_config_unchanged": CONFIG.read_bytes() == src}
            (LAB/"restore-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
