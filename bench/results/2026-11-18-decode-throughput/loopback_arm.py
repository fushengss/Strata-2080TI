#!/usr/bin/env python3
"""Start one isolated Strata benchmark arm from the frozen loopback config."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "baseline.json"
SERVER = Path("/srv/Strata-0157/serve/server.py")
BENCH_SERVER = ROOT / "source" / "serve" / "server.py"
PYTHON = Path("/srv/Strata/.venv/bin/python")
PIDFILE = ROOT / "server.pid"
PORT = 8080
ENV_FLAGS = {
    "STRATA_BATCH_GDN_GROUPS", "STRATA_P4_TIMING", "STRATA_PIPED_ASYNC_SAMPLE",
    "STRATA_PIPED_DIRECT_SAVE", "STRATA_PIPED_COMMIT_LAYERS", "STRATA_PIPED_EARLY_HANDOFF",
    "STRATA_PIPED_MTP_OBSERVE", "STRATA_PIPED_STREAM_ATTN_ROWS", "STRATA_SM75_IL_WIDE", "STRATA_MTP_STAGE0",
}


def checked_output(argv):
    return subprocess.check_output(argv, text=True, timeout=8).strip()


def listeners():
    sockets = checked_output(["ss", "-H", "-ltnp", "sport", "=", f":{PORT}"])
    return [line.split()[3] for line in sockets.splitlines()]


def identity(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if state[0] == "Z" else (int(state[1]), int(state[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop_current():
    pid = int(PIDFILE.read_text()) if PIDFILE.exists() else 0
    if pid and identity(pid):
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if len(argv) < 2 or argv[1] not in (os.fsencode(SERVER), os.fsencode(BENCH_SERVER)) or not any(
            arg.startswith(os.fsencode(ROOT) + b"/") and arg.endswith(b".json") for arg in argv
        ):
            raise RuntimeError("Current PID is outside the benchmark")
        expected = {pid: identity(pid)}
        for proc in Path("/proc").iterdir():
            if proc.name.isdecimal():
                child = identity(int(proc.name))
                if child and child[0] == pid:
                    expected[int(proc.name)] = child
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(identity(p) == b for p, b in expected.items()):
            time.sleep(.25)
        for p, birth in expected.items():
            current = identity(p)
            if current and current[1] == birth[1]:
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    for _ in range(45):
        if not listeners() and not checked_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]):
            return
        time.sleep(1)
    raise RuntimeError("Port or CUDA resources are still in use")


def wait_ready(config, proc):
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/status",
                                     headers={"Authorization": f"Bearer {config['api_key']}"})
    for _ in range(720):
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited with {proc.returncode}")
        addrs = listeners()
        if addrs and addrs != [f"127.0.0.1:{PORT}"]:
            raise RuntimeError("Unexpected listen address")
        if addrs:
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    doc = json.load(response)
                slots = doc.get("concurrency", {}).get("serving")
                if slots == config["parallel"]:
                    return
            except OSError:
                pass
        time.sleep(1)
    raise RuntimeError("Server did not become ready with the requested slot count")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--parallel", type=int, choices=(8, 16), default=8)
    parser.add_argument("--groups", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--drafts", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prefill", type=int, choices=(3072, 4096, 6144))
    parser.add_argument("--candidate-server", action="store_true")
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", args.name) or args.parallel % args.groups:
        parser.error("Invalid arm name or group count")
    if args.parallel // args.groups * (args.drafts + 1) > 8:
        parser.error("MTP group exceeds the narrow verifier window")
    os.umask(0o077)
    cfg = json.loads(BASE.read_bytes())
    if args.exe:
        if not args.exe.resolve().is_relative_to(ROOT) or not args.exe.is_file():
            parser.error("Candidate executable must be inside the benchmark workspace")
        cfg["exe"] = str(args.exe.resolve())
    for flag in args.env:
        key, sep, val = flag.partition("=")
        if not sep or key not in ENV_FLAGS or val not in ("0", "1"):
            parser.error("Unexpected experiment flag")
        if key == "STRATA_P4_TIMING" and val == "0":
            cfg["env"].pop(key, None)
        else:
            cfg["env"][key] = val
    cfg["env"]["STRATA_PIPED_MTP"] = str(args.drafts)
    cfg["args"][cfg["args"].index("--batch-groups") + 1] = str(args.groups)
    if args.prefill:
        cfg["args"][cfg["args"].index("--prefill") + 1] = str(args.prefill)
    server = BENCH_SERVER if args.candidate_server else SERVER
    if not server.is_file():
        parser.error("Server source is missing")
    cfg.update(host="127.0.0.1", parallel=args.parallel, log=str(ROOT / f"{args.name}.native.log"))
    conf = ROOT / f"{args.name}.json"
    with conf.open("x") as f:
        json.dump(cfg, f)
    with (ROOT / f"{args.name}.server.log").open("x") as log:
        stop_current()
        with open(os.devnull) as devnull:
            proc = subprocess.Popen([str(PYTHON), str(server), "--engine", "strata", "--config", str(conf),
                                     "--host", "127.0.0.1", "--port", str(PORT)], cwd=server.parent.parent,
                                    stdin=devnull, stdout=log, stderr=log, start_new_session=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        PIDFILE.write_text(f"{proc.pid}\n")
    wait_ready(cfg, proc)
    print(json.dumps({"arm": args.name, "pid": proc.pid, "parallel": args.parallel, "groups": args.groups,
                      "drafts": args.drafts, "listener": listeners(), "env_overrides": args.env,
                      "prefill": args.prefill or "auto", "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
                      "engine_sha256": hashlib.file_digest(Path(cfg["exe"]).open("rb"), "sha256").hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
