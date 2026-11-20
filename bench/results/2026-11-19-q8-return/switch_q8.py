#!/usr/bin/env python3
"""Test the former Orca Q8 pack on the current engine, then serve it on the original address."""

import hashlib
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import run_ab as live


LAB = Path(__file__).resolve().parent
HISTORICAL = live.LIVE / "strata-orca-q8_0.json"
DEPLOYED = live.LIVE / "strata-orca-q8_0.57.json"
LAUNCHER = live.LIVE / "run-orca-q8_0.57.sh"
HISTORICAL_SHA = "d24028fdcfefff0c59b928a21d586ce55435ee20d287c61ebf3fd8fe2eae5aaf"


def note(**data):
    print(json.dumps(data), flush=True)


def save(name, value):
    (LAB / name).write_text(json.dumps(value, indent=2) + "\n")


def private_config(path, obj):
    with path.open("x") as out:
        json.dump(obj, out, indent=2)
        out.write("\n")


def request(cfg, path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=90 if payload else 5)
    try:
        conn.request("GET" if payload is None else "POST", path,
                     body=None if payload is None else json.dumps(payload),
                     headers={"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"})
        with conn.getresponse() as out:
            if out.status != 200:
                raise RuntimeError(f"{path} returned HTTP {out.status}")
            return json.load(out)
    finally:
        conn.close()


def checked_digest(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"Config changed since preflight: {path}")
    return json.loads(raw)


def contains_args(pid, args):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return all(os.fsencode(x) in cmd for x in args)
    except FileNotFoundError:
        return False


def engine_pid(server_pid, cfg):
    path = Path(f"/proc/{server_pid}/task")
    children = {int(p) for t in path.iterdir() for p in (t / "children").read_text().split()}
    native = cfg["args"][cfg["args"].index("--native") + 1]
    owned = [pid for pid in children if contains_args(pid, [cfg["exe"], "--serve", native])]
    if len(owned) != 1:
        raise RuntimeError("Cannot prove the service owns the requested Q8 engine")
    return owned[0]


def start(config, cfg, log, *, launcher=None):
    cmd = [str(launcher)] if launcher else [str(live.PYTHON), str(live.APP), "--engine", "strata",
                                          "--config", str(config), "--host", cfg["host"], "--port", "8080"]
    with log.open("ab") as out:
        return subprocess.Popen(cmd, cwd=live.APP.parent.parent, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)


def ready(proc, cfg, timeout=1900):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited while loading {cfg['model_name']} ({proc.returncode})")
        sockets = live.listeners()
        if sockets:
            if (len(sockets) != 1 or sockets[0].split()[3] != f'{cfg["host"]}:8080'
                    or f"pid={proc.pid}," not in sockets[0]):
                raise RuntimeError("Unexpected listener or owner")
            try:
                s = request(cfg, "/v1/status")
            except (OSError, ValueError, http.client.HTTPException, RuntimeError):
                time.sleep(1)
                continue
            widths = s["concurrency"]
            if widths["requested"] != cfg["parallel"] or widths["serving"] < 1:
                raise RuntimeError("Engine did not start a usable batch")
            models = [v["id"] for v in request(cfg, "/v1/models")["data"]]
            if cfg["model_name"] not in models or any(x not in models for x in cfg.get("aliases", [])):
                raise RuntimeError("API model identity or alias missing")
            return {"pid": proc.pid, "engine_pid": engine_pid(proc.pid, cfg), "models": models,
                    "requested": widths["requested"], "slots": widths["serving"],
                    "startup_s": round(time.monotonic() - started, 1)}
        time.sleep(1)
    raise TimeoutError(f"{cfg['model_name']} startup timed out")


def smoke(cfg):
    answer = request(cfg, "/v1/chat/completions", {"model": cfg["model_name"], "temperature": 0,
                     "max_tokens": 12, "chat_template_kwargs": {"enable_thinking": False},
                     "messages": [{"role": "user", "content": "Reply with the word READY only."}]})
    if not answer.get("choices") or answer.get("usage", {}).get("completion_tokens", 0) < 1:
        raise RuntimeError("Q8 did not generate a response")


def bench(name, config, script, args, timeout):
    cmd = [sys.executable, "-B", str(LAB / script), "--config", str(config),
           "--out", str(LAB / f"{name}.json"), *args]
    try:
        result = subprocess.run(cmd, cwd=LAB, timeout=timeout)
        note(phase="benchmark_complete", name=name, exit=result.returncode)
    except subprocess.TimeoutExpired:
        note(phase="benchmark_timeout", name=name)


def benchmark(config, slots):
    bench("cold100k", config, "prefill_ab.py", ["--context", "100000", "--rounds", "3"], 740)
    for width, fixture in ((8, "decode-throughput-width"), (16, "decode-throughput-v1")):
        if width <= slots:
            bench(f"decode{width}", config, "bench_decode.py", ["--concurrency", str(width), "--warmup", "128",
                  "--max-tokens", "1024", "--fixture", fixture], 450)
    if slots == 16:
        bench("mixed8", config, "bench_decode_insert.py", ["--decoders", "8", "--rounds", "1",
              "--fixture", "gpu-mmq-insert8", "--timeout", "330"], 360)
        bench("mixed16", config, "bench_decode_insert.py", ["--decoders", "15", "--saturated", "--rounds", "1",
              "--fixture", "gpu-mmq-insert16", "--timeout", "330"], 360)
    note(phase="benchmarks_done")


def prepare_cfg(current, historic, width):
    cfg = json.loads(json.dumps(current))
    for opt in ("--pack", "--native"):
        cfg["args"][cfg["args"].index(opt) + 1] = historic["args"][historic["args"].index(opt) + 1]
    cfg["tokenizer"] = historic["tokenizer"]
    cfg["vision"]["model"] = historic["vision"]["model"]
    cfg["model_name"] = historic["model_name"]
    cfg["aliases"] = list(dict.fromkeys([current["model_name"], *current.get("aliases", [])]))
    cfg["parallel"] = width
    cfg["engine_ready_s"] = 1800
    # The Q8 complement is large enough that the NUMA pinning path released the
    # model's cached pages on this host and exceeded the old startup deadline.
    cfg["env"]["STRATA_RESIDENT_PIN"] = "0"
    # Q8 shares the draft vocabulary head. The Q5 sharding extension requires
    # an owned Q6_K subset, so keep the existing unsharded MTP head on Q8.
    cfg["env"].pop("STRATA_MTP_HEAD_GPU", None)
    cfg["env"].pop("STRATA_PIPED_MTP_HEAD", None)
    for path in (cfg["exe"], cfg["tokenizer"], cfg["vision"]["model"],
                 *(cfg["args"][cfg["args"].index(opt) + 1] for opt in ("--pack", "--native"))):
        if not Path(path).exists():
            raise RuntimeError(f"Required Q8 artifact is absent: {path}")
    return cfg


def deploy(selected):
    selected["log"] = str(live.LIVE / "strata-orca-q8_0.57.log")
    private_config(DEPLOYED, selected)
    cmd = [live.PYTHON, live.APP, "--engine", "strata", "--config", DEPLOYED, "--port", "8080"]
    with LAUNCHER.open("x") as out:
        out.write(f"#!/usr/bin/env bash\ncd {shlex.quote(str(live.APP.parent.parent))}\nexec "
                  + shlex.join(map(str, cmd)) + "\n")
    LAUNCHER.chmod(0o700)
    instance = start(DEPLOYED, selected, live.LIVE / "serve-orca-q8.57.out", launcher=LAUNCHER)
    note(phase="production_q8_loading", pid=instance.pid)
    try:
        receipt = ready(instance, selected)
        receipt.update(host=selected["host"], config=str(DEPLOYED), model=selected["model_name"])
        save("deployment-receipt.json", receipt)
        note(phase="production_q8_ready", **receipt)
    except BaseException:
        live.stop(instance.pid)
        raise


def main():
    os.umask(0o077)
    q5 = checked_digest(live.CONFIG, live.BASE_CONFIG_SHA)
    historic = checked_digest(HISTORICAL, HISTORICAL_SHA)
    cfg16 = prepare_cfg(q5, historic, 16)
    cfg8 = prepare_cfg(q5, historic, 8)
    if not q5.get("api_key") or DEPLOYED.exists() or LAUNCHER.exists():
        raise RuntimeError("Expected a protected server and unused Q8 deployment paths")
    lines = live.listeners()
    if lines:
        if len(lines) != 1 or lines[0].split()[3] != f'{q5["host"]}:8080':
            raise RuntimeError("Unexpected current server")
        found = re.search(r"\bpid=(\d+),", lines[0])
        if not found or not contains_args(int(found[1]), [live.APP, live.CONFIG]):
            raise RuntimeError("Current listener is not the verified Q5 service")
        note(phase="stopping_q5", pid=int(found[1]))
        live.stop(int(found[1]))

    instance = None
    try:
        selected = None
        for choice in (cfg16, cfg8):
            width = choice["parallel"]
            scratch = dict(choice, host="127.0.0.1", log=str(LAB / f"q8-{width}.engine.log"))
            config = LAB / f"q8-{width}.private.json"
            private_config(config, scratch)
            try:
                instance = start(config, scratch, LAB / f"q8-{width}.server.log")
                note(phase="loopback_loading", requested=width, pid=instance.pid)
                receipt = ready(instance, scratch)
                smoke(scratch)
                save("test-receipt.json", receipt)
                note(phase="loopback_ready", **receipt)
                selected = choice
                benchmark(config, receipt["slots"])
                break
            except (RuntimeError, TimeoutError) as err:
                note(phase="q8_trial_failed", requested=width, error=str(err))
                if selected is not None:
                    break
            finally:
                if instance:
                    live.stop(instance.pid)
                    instance = None
                config.unlink(missing_ok=True)
        if selected is None:
            raise RuntimeError("Neither Q8 configuration passed generation health")
        deploy(selected)
    finally:
        if instance:
            live.stop(instance.pid)


if __name__ == "__main__":
    main()
