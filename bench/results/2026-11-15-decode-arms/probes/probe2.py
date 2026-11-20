#!/usr/bin/env python3
"""where does the engine's host thread spend its time while decoding?  perf-samples the main thread (strata-host, pinned to CPU 0) for 2.5 s during a ~500-token request.
Changes no setting; the only side effect is the requests.  run: ssh root@10.100.3.3 python3 - < probe2.py"""
import json, os, subprocess, threading, time, urllib.request
CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
PID = int(subprocess.run(["pgrep", "-f", "^/srv/Strata-0157/engine/releases/[^ ]*/strata --serve"], capture_output=True, text=True).stdout.split()[0])
OUT = "/tmp/claude-probe"
os.makedirs(OUT, exist_ok=True)
PROSE = "Explain in detail how a pipelined layer-split transformer decoder works across several GPUs that cannot talk to each other directly, and why speculative decoding helps it."


def chat(prompt, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d, time.time() - t0


def run(label, kw):
    done = {}

    def work():
        d, wall = chat(PROSE, **kw)
        done["tok"], done["wall"] = d["usage"]["completion_tokens"], wall
    th = threading.Thread(target=work)
    th.start()
    time.sleep(1.3)
    data = "%s/perf-%s.data" % (OUT, label)
    subprocess.run(["perf", "record", "-F", "997", "-t", str(PID), "-o", data, "--", "sleep", "2.5"], capture_output=True)
    th.join()
    print("\n=== %s: %d tokens in %.1f s" % (label, done["tok"], done["wall"]))
    for sort in ("dso", "dso,sym"):
        rep = subprocess.run(["perf", "report", "-i", data, "--stdio", "--sort", sort, "--no-children", "--percent-limit", "1.0"], capture_output=True, text=True)
        body = [l[:170] for l in rep.stdout.splitlines() if l.strip() and not l.startswith("#")]
        print("-- by %s:" % sort)
        print("\n".join(body[:26]))
        if rep.stderr.strip():
            print("(stderr)", rep.stderr.strip()[:300])


chat(PROSE, temperature=0, presence_penalty=0)   # warm-up
run("pw-greedy", dict(temperature=0, presence_penalty=0))
run("serial-pen", dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5))
