#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/harness/conc.py <tag> - many requests at once against the local-only service, with the free VRAM of every card sampled.
    python3 conc.py <tag> [8 16 ...]        writes /tmp/claude-decode/conc_<tag>.json and prints a summary
Tests (streamed, so the first token and the decode speed of every request are its own):
  essays N greedy / sampled   N x "Write a 300-word essay about <topic>", 256 tokens, all started within 0.05 s * N
                              greedy = temperature 0 + no penalty (the pipelined windows may run), sampled = temperature 1.0, top_p 0.95, top_k 20 (what the agents send)
  mixed16                     8 sampled essays of 400 tokens, and 2 s later 8 fresh ~8K-token prompts (max_tokens 8): the reads and the decodes share the slots
Columns: aggregate tok/s = tokens of all requests / wall; first token and decode speed per request (tokens / (finish - first token)); errors; the smallest free VRAM seen per card.
The API key is read from the config and never printed."""
import json, random, re, statistics as st, subprocess, sys, threading, time, urllib.request

CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
TAG = sys.argv[1]
SIZES = [int(x) for x in sys.argv[2:]] or [8, 16]
TOPICS = ["bridges", "volcanoes", "bees", "trains", "glaciers", "printing", "chess", "rice", "lighthouses", "tea", "owls", "rivers", "clocks", "maps", "salt", "kites"]
GREEDY = dict(temperature=0, presence_penalty=0)
SAMPLED = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0)
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part place case week company system "
         "program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name president team minute idea kid body information back parent").split()


def stream(prompt, max_tokens, extra, out, i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    first, n = None, 0
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens", n)
                for c in d.get("choices") or []:
                    dl = c.get("delta") or {}
                    if (dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning")) and first is None:
                        first = time.time() - t0
        out[i] = dict(first=first, total=time.time() - t0, tokens=n, error=None)
    except Exception as e:  # noqa: BLE001
        out[i] = dict(first=first, total=time.time() - t0, tokens=n, error=repr(e)[:120])


def free_vram():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
    return [int(x) for x in r.stdout.split()]


def status():
    req = urllib.request.Request("http://127.0.0.1:%d/v1/status" % PORT, headers={"Authorization": "Bearer " + KEY})
    return json.load(urllib.request.urlopen(req, timeout=20))


def idle(limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            if status()["activity"]["in_flight"] == 0:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)


def run(jobs):
    """jobs: (delay, prompt, max_tokens, extra) -> results (same order), wall, min free VRAM per card"""
    out = [None] * len(jobs)
    low = free_vram()
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                low[:] = [min(a, b) for a, b in zip(low, free_vram())]
            except Exception:  # noqa: BLE001
                pass
            stop.wait(1.0)
    w = threading.Thread(target=watch)
    w.start()
    t0 = time.time()
    th = []
    for i, (delay, prompt, n, extra) in enumerate(jobs):
        def go(i=i, delay=delay, prompt=prompt, n=n, extra=extra):
            time.sleep(delay)
            stream(prompt, n, extra, out, i)
        t = threading.Thread(target=go)
        th.append(t)
        t.start()
        time.sleep(0.05)
    for t in th:
        t.join()
    wall = time.time() - t0
    stop.set()
    w.join()
    return out, wall, low


def summarize(name, res, wall, low, idx=None):
    rs = [r for k, r in enumerate(res) if idx is None or k in idx]
    ok = [r for r in rs if not r["error"] and r["first"] is not None]
    toks = sum(r["tokens"] for r in rs)
    firsts = sorted(r["first"] for r in ok)
    dec = [r["tokens"] / max(r["total"] - r["first"], 0.01) for r in ok if r["tokens"] > 16]
    s = dict(name=name, n=len(rs), wall=round(wall, 1), tokens=toks, agg_tps=round(toks / wall, 1), errors=len(rs) - len(ok),
             first_med=round(st.median(firsts), 2) if firsts else None, first_max=round(firsts[-1], 2) if firsts else None,
             dec_med=round(st.median(dec), 1) if dec else None, dec_min=round(min(dec), 1) if dec else None, min_free_mib=low)
    return s


res = {"tag": TAG, "status_before": None, "tests": []}
try:
    st0 = status()
    res["status_before"] = st0.get("concurrency")
except Exception as e:  # noqa: BLE001
    res["status_before"] = repr(e)
print("tag %s, concurrency per /v1/status: %s, free VRAM per card (MiB): %s" % (TAG, res["status_before"], free_vram()))
for mode, extra in (("greedy", GREEDY), ("sampled", SAMPLED)):
    for n in SIZES:
        idle()
        jobs = [(0, "Write a 300-word essay about %s." % TOPICS[i % len(TOPICS)], 256, dict(extra, seed=300 + i)) for i in range(n)]
        out, wall, low = run(jobs)
        s = summarize("essays %d %s" % (n, mode), out, wall, low)
        res["tests"].append(s)
        print(s, flush=True)
if 16 in SIZES:
    idle()
    rng = random.Random(11)
    jobs = [(0, "Write a 300-word essay about %s." % TOPICS[i], 400, dict(SAMPLED, seed=400 + i)) for i in range(8)]
    jobs += [(2.0, " ".join(rng.choice(WORDS) for _ in range(6400)) + "\n\nIgnoring the text above, say OK. " + str(i), 8, dict(GREEDY)) for i in range(8)]
    out, wall, low = run(jobs)
    res["tests"].append(dict(summarize("mixed16 decodes", out, wall, low, idx=set(range(8))), part="the 8 decodes"))
    res["tests"].append(dict(summarize("mixed16 reads", out, wall, low, idx=set(range(8, 16))), part="the 8 fresh 8K reads"))
    print(res["tests"][-2], flush=True)
    print(res["tests"][-1], flush=True)
try:
    res["status_after"] = status().get("concurrency")
except Exception as e:  # noqa: BLE001
    res["status_after"] = repr(e)
json.dump(res, open("/tmp/claude-decode/conc_%s.json" % TAG, "w"), indent=1)
