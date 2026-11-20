#!/usr/bin/env python3
"""Capture true Q5 expert routes from synthetic C8 and C16 loopback decode."""

import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import bench_decode as bench
import run_ab as service
from benchmark_q5 import ENGINE_SHA, launch, note, preflight


LAB = Path(__file__).resolve().parent
CAPTURE = LAB / "capture"
PREFIX = CAPTURE / "routes"
CONTROL = PREFIX.with_suffix(".control")
MODEL_DIR = Path("/models/orcarouter/Qwen3.8-Flash-Next-Uncensored-Q5")


def set_tag(tag):
    pending = CONTROL.with_suffix(".tmp")
    pending.write_text(f"{tag}\n")
    pending.replace(CONTROL)


class CaptureServer(bench.Server):
    width = 0
    tag = 0

    def sample(self):
        row = super().sample()
        work = row["work"]
        full = (self.width and row["ready"] and work.get("active") == work.get("occupied_slots") == self.width
                and work.get("queued") == work.get("waiting") == 0
                and sum(slot["state"] == "decoding" for slot in row["slots"]) == self.width)
        tag = self.width if full else 0
        if tag != self.tag:
            set_tag(tag)
            self.tag = tag
        return row

    def disarm(self):
        self.width = self.tag = 0
        set_tag(0)


def capture_batch(server, width, fixture):
    bench.loopback_only(8080)
    args = SimpleNamespace(concurrency=width, interval=.25, timeout=240, min_steady=2.0,
                           max_tokens=1024, mode="sampled", workload="mixed", fixture=fixture,
                           presence_penalty=0)
    warmup = [bench.fixture(server, args, lane, "warmup", 0, 128) for lane in range(width)]
    records, _ = bench.run_cohort(server, args, warmup, 128, False)
    bench.require(all(row["usable"] for row in records), "Warmup failed")
    fixtures = [bench.fixture(server, args, lane, 0, 0, args.max_tokens) for lane in range(width)]
    before = server.idle()
    epoch_offset_ns = time.time_ns() - time.monotonic_ns()
    server.width = width
    note(phase="cohort_started", width=width)
    try:
        records, samples = bench.run_cohort(server, args, fixtures, args.max_tokens, True)
    finally:
        server.disarm()
    after = server.idle()
    bench.loopback_only(8080)
    summary = bench.summarize(before, after, records, samples, args)
    begin = max(row["events"][0] for row in records if row["events"])
    end = min(row["events"][-1] for row in records if row["events"])
    settled = bench.decode_windows(samples, begin, end, width)
    longest = max(settled, key=lambda rows: rows[-1]["time"] - rows[0]["time"])
    roi = [epoch_offset_ns + int(longest[0]["end"] * 1e9),
           epoch_offset_ns + int(longest[-1]["start"] * 1e9)]
    note(phase="cohort_complete", width=width, qualified=summary["qualified"], routes_roi_epoch_ns=roi)
    report = dict(vars(args), qualified=summary["qualified"], routes_roi_epoch_ns=roi,
                  instrumented_steady_s=summary["steady_decode_elapsed_s"],
                  matched_request_sha256=[row["request_sha256"] for row in records])
    (CAPTURE / f"cohort{width}.json").write_text(json.dumps(report, indent=2) + "\n")
    bench.require(summary["qualified"], f"C{width} did not reach an exclusive decode window")


def ready(proc, key, address):
    start = time.monotonic()
    unmatched = 0
    while time.monotonic() - start < 1100:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited ({proc.returncode})")
        sockets = service.listeners()
        if sockets:
            if len(sockets) != 1 or sockets[0].split()[3] != f"{address}:8080" or f"pid={proc.pid}," not in sockets[0]:
                unmatched += 1
                if unmatched == 3:
                    raise RuntimeError("Unexpected owner of port 8080")
                time.sleep(1)
                continue
            unmatched = 0
            try:
                status = service.status("127.0.0.1", key)
            except (OSError, ValueError, RuntimeError, http.client.HTTPException):
                status = {}
            if status.get("concurrency") == {"serving": 16, "requested": 16}:
                return round(time.monotonic() - start, 1)
        time.sleep(1)
    raise RuntimeError("Server did not become ready")


def restore(key, original):
    production = launch([str(service.LAUNCH)], service.LIVE / "serve-orca.out")
    note(phase="production_loading", pid=production.pid)
    elapsed = ready(production, key, "0.0.0.0")
    receipt = {"pid": production.pid, "startup_s": elapsed,
               "original_configuration": original == service.CONFIG.read_bytes()}
    (CAPTURE / "production-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    note(phase="production_ready", **receipt)


def replay(config):
    subprocess.run([sys.executable, "-B", str(LAB / "select_routes.py"), str(CAPTURE)], check=True)
    cuda_owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if service.listeners() or cuda_owners.strip():
        raise RuntimeError("The isolated replay requires an idle GPU and a closed API listener")
    pci = "00000000:04:00.0"
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", pci, "--query-gpu=uuid",
                                        "--format=csv,noheader,nounits"], text=True).strip()
    argv = [str(LAB / "operator-bin/q5_decode_tc_bench")]
    for shard in sorted(MODEL_DIR.glob("*.gguf")):
        argv.extend(["--model", str(shard)])
    for width in (8, 16):
        argv.extend(["--routes", str(CAPTURE / f"c{width}.routes")])
    argv.extend(["--layers", "3,6,42,47", "--samples", "36", "--rounds", "5", "--iters", "4",
                 "--phases", "0,1,3"])
    env = dict(os.environ, **{key: str(value) for key, value in config["env"].items()})
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    note(phase="replay_start", pci=pci, layers=[3, 6, 42, 47])
    with (CAPTURE / "operator-gpu.csv").open("wb") as gpu:
        sample = subprocess.Popen(["nvidia-smi", "-i", pci,
                                   "--query-gpu=timestamp,clocks.current.graphics,clocks.current.memory,"
                                   "power.draw,power.limit,temperature.gpu,utilization.gpu",
                                   "--format=csv,noheader,nounits", "--loop-ms=500"], stdout=gpu)
        try:
            with (CAPTURE / "operator.jsonl").open("wb") as out, (CAPTURE / "operator.log").open("wb") as log:
                result = subprocess.run(argv, env=env, stdout=out, stderr=log, timeout=450)
        finally:
            sample.terminate()
            sample.wait(timeout=8)
    note(phase="replay_complete", exit=result.returncode)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Replay did not complete ({result.returncode})")


def check_capture_kernel():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    with (CAPTURE / "capture-parity.log").open("wb") as log:
        subprocess.run([str(LAB / "capture-bin/resident_plan_parity")], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=45)
    note(phase="capture_kernel_verified")


def main():
    os.umask(0o077)
    original = service.CONFIG.read_bytes()
    config = json.loads(original)
    exe = LAB / "capture-bin/strata"
    if (hashlib.sha256(original).hexdigest() != service.BASE_CONFIG_SHA or not config.get("api_key")
            or hashlib.sha256(Path(config["exe"]).read_bytes()).hexdigest() != ENGINE_SHA
            or not os.access(exe, os.X_OK) or not os.access(LAB / "operator-bin/q5_decode_tc_bench", os.X_OK)):
        raise RuntimeError("Select the production Q5 deployment and build the capture engine and replay first")
    CAPTURE.mkdir()
    set_tag(0)
    temporary = CAPTURE / "loopback.private.json"
    test = None
    preflight(config)
    try:
        check_capture_kernel()
        local = dict(config, host="127.0.0.1", exe=str(exe), log=str(CAPTURE / "engine.log"), engine_ready_s=1100)
        local["env"] = dict(config["env"], STRATA_PIPED_ROUTE_CAPTURE=str(PREFIX),
                            STRATA_PIPED_ROUTE_EVERY="16", STRATA_PIPED_ROUTE_LIMIT="96")
        temporary.write_text(json.dumps(local))
        test = launch([str(service.PYTHON), str(service.APP), "--engine", "strata", "--config", str(temporary),
                       "--host", "127.0.0.1", "--port", "8080"], CAPTURE / "server.log")
        note(phase="loopback_loading", pid=test.pid)
        elapsed = ready(test, config["api_key"], "127.0.0.1")
        note(phase="loopback_ready", pid=test.pid, startup_s=elapsed)
        server = CaptureServer(local, 8080)
        capture_batch(server, 8, "decode-throughput-width")
        capture_batch(server, 16, "decode-throughput-v1")
        service.stop(test.pid)
        test = None
        temporary.unlink()
        replay(config)
    finally:
        set_tag(0)
        if test is not None:
            service.stop(test.pid)
        temporary.unlink(missing_ok=True)
        restore(config["api_key"], original)


if __name__ == "__main__":
    main()
