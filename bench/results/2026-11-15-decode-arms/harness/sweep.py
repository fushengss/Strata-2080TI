#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/sweep.py - per-request A/B sweeps on one local-only server (no restart between values).
    python3 sweep.py minp          spec_min_p per request through the request body {"strata_tune": {"spec_min_p": v}}  (needs the engine started with --spec-min-p, any value)
    python3 sweep.py theta         the pipelined-window gate and PW on/off per request through STRATA_PIPELINE_SWITCH (needs the service started with
                                   STRATA_PIPELINE_DEBUG=1 STRATA_PIPELINE_SWITCH=/tmp/claude-decode/pw.switch STRATA_DECODE_TIMING=1)
Each value runs 2 prompts x 3 seeds of ~500 sampled tokens (temperature 1.0, top_p 0.95, top_k 20); replies must be the same text for every value (hash check).
The API key is read from the config and never printed."""
import hashlib, json, os, re, statistics as st, sys, time, urllib.request

CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
SWITCH = "/tmp/claude-decode/pw.switch"
CODE = "Write a complete Python module that implements an LRU cache class with get, put, delete, a TTL option and thread safety, followed by unit tests using unittest. Output only code."
PROSE = "Explain in detail how a pipelined layer-split transformer decoder works across several GPUs that cannot talk to each other directly, and why speculative decoding helps it."
SAMPLED = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0)
RE_REQ = re.compile(r"strata serve: prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RE_PL = re.compile(r"strata pipeline: (\d+) windows in (\d+) ms \(([\d.]+) ms/window\): (\d+) speculative, (\d+) on the path, (\d+) rolled back, (\d+) below the gate")


def chat(prompt, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(req, timeout=900))


def one(prompt, seed, **kw):
    size0 = os.path.getsize(LOG)
    d = chat(prompt, seed=seed, **SAMPLED, **kw)
    time.sleep(0.4)
    with open(LOG, "rb") as f:
        f.seek(size0)
        new = f.read().decode(errors="replace")
    m = list(RE_REQ.finditer(new))[-1]
    p = RE_PL.search(new)
    return dict(gen=int(m.group(2)), ms=int(m.group(3)), tps=float(m.group(4)), acc=int(m.group(5)), drf=int(m.group(6)),
                win=int(p.group(1)) if p else None, spec=int(p.group(4)) if p else None, on=int(p.group(5)) if p else None, rb=int(p.group(6)) if p else None,
                gate=int(p.group(7)) if p else None, sha=hashlib.sha1((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:8])


def table(rows, title):
    print("\n" + title)
    print("%-14s %-6s %7s %7s | %6s %6s %6s %6s %6s %6s | %s" % ("value", "prompt", "tok/s", "accept", "wins", "ms/win", "tok/win", "spec", "onpath", "rolled", "same text as first value"))
    first = {}
    for val, name, rs in rows:
        h = tuple(r["sha"] for r in rs)
        first.setdefault(name, h)
        w = [r for r in rs if r["win"]]
        print("%-14s %-6s %7.1f %6.1f%% | %6s %6s %6s %6s %6s %6s | %s" % (
            val, name, st.median(r["tps"] for r in rs), 100.0 * sum(r["acc"] for r in rs) / max(1, sum(r["drf"] for r in rs)),
            "%d" % st.mean(r["win"] for r in w) if w else "-", "%.2f" % (sum(r["ms"] for r in w) / sum(r["win"] for r in w)) if w else "-",
            "%.2f" % (sum(r["gen"] for r in w) / sum(r["win"] for r in w)) if w else "-", "%d" % st.mean(r["spec"] for r in w) if w else "-",
            "%d" % st.mean(r["on"] for r in w) if w else "-", "%d" % st.mean(r["rb"] for r in w) if w else "-", "yes" if h == first[name] else "NO " + ",".join(h)))


mode = sys.argv[1]
rows = []
one(PROSE, 99)   # warm-up
if mode == "minp":
    for v in (None, 0.1, 0.3, 0.5, 0.7, 0.9):
        for name, prompt in (("code", CODE), ("prose", PROSE)):
            kw = {} if v is None else {"strata_tune": {"spec_min_p": v}}
            rows.append(("default" if v is None else "min_p %.1f" % v, name, [one(prompt, s, **kw) for s in (101, 102, 103)]))
    table(rows, "spec_min_p per request (the engine's own default is --spec-min-p 0.5)")
elif mode == "theta":
    for v in ("pw=2 theta=0.2", "pw=2 theta=0.0", "pw=2 theta=0.1", "pw=2 theta=0.35", "pw=2 theta=0.5", "pw=2 theta=0.7", "pw=2 theta=0.9", "pw=0 theta=0.2"):
        open(SWITCH, "w").write(v + "\n")
        for name, prompt in (("code", CODE), ("prose", PROSE)):
            rows.append((v.replace("pw=", "pw").replace(" theta=", " th"), name, [one(prompt, s) for s in (101, 102, 103)]))
    open(SWITCH, "w").write("pw=2 theta=0.2\n")
    table(rows, "pipelined-window gate (theta) and PW on/off, per request")
json.dump(rows, open("/tmp/claude-decode/sweep_%s.json" % mode, "w"))
