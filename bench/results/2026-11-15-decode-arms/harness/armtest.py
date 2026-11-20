#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/armtest.py - the decode test of one arm, run on the serving box against the local-only service (nothing else may call it).
    python3 armtest.py <tag>        writes /tmp/claude-decode/<tag>.json and prints a summary
What it sends (every request carries its own seed, so a sampled reply is the same text in every arm that changes no arithmetic - the reply hashes are compared):
  solo sampled  (temperature 1.0, top_p 0.95, top_k 20, no penalty: the sampling of the thinking mode and of the agents), 2 prompts x 5 seeds, ~500 tokens
  solo greedy   2 prompts x 2 runs
  4 at once     sampled, 3 rounds (the batch path)
  cold prefill  one fresh ~24K-token prompt (a prefill regression check)
Speeds come from the engine's own log line of each request ("N generated in X ms (T tok/s), drafts accepted a of d"), the API key is read from the config and never printed."""
import hashlib, json, os, random, re, statistics as st, subprocess, sys, threading, time, urllib.request

CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
OUT = "/tmp/claude-decode"
os.makedirs(OUT, exist_ok=True)
TAG = sys.argv[1]
CODE = "Write a complete Python module that implements an LRU cache class with get, put, delete, a TTL option and thread safety, followed by unit tests using unittest. Output only code."
PROSE = "Explain in detail how a pipelined layer-split transformer decoder works across several GPUs that cannot talk to each other directly, and why speculative decoding helps it."
SAMPLED = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0)
GREEDY = dict(temperature=0, presence_penalty=0)
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part place case week company system "
         "program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name president team minute idea kid body information back parent").split()
RE_REQ = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")


def chat(prompt, max_tokens=500, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=900))
    return d, time.time() - t0


def log_from(size0):
    with open(LOG, "rb") as f:
        f.seek(size0)
        return f.read().decode(errors="replace")


def parse(new):
    """the (last) request line of a log excerpt plus the decode-head window counter"""
    ms = list(RE_REQ.finditer(new))
    if not ms:
        return {}
    g = ms[-1].groups()
    head = re.findall(r"strata decode head: CUDA\d+, (\d+) PW2 windows offloaded", new)
    return dict(prompt=int(g[0]), reused=int(g[1]), read=int(g[2]), read_ms=int(g[3]), read_tps=float(g[4]), gen=int(g[5]), gen_ms=int(g[6]), tps=float(g[7]),
                acc=int(g[8]), drf=int(g[9]), windows=int(head[-1]) if head else None, pw=bool(head))


def one(label, prompt, **kw):
    size0 = os.path.getsize(LOG)
    d, wall = chat(prompt, **kw)
    time.sleep(0.5)
    r = parse(log_from(size0))
    r.update(label=label, wall=wall, tokens=d["usage"]["completion_tokens"], sha1=hashlib.sha1((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:12],
             finish=d["choices"][0].get("finish_reason"))
    return r


res = {"tag": TAG, "solo_sampled": [], "solo_greedy": [], "batch": [], "prefill": None}
one("warm1", CODE, **GREEDY)
one("warm2", PROSE, **dict(SAMPLED, seed=99))
for pname, prompt in (("code", CODE), ("prose", PROSE)):
    for seed in (101, 102, 103, 104, 105):
        res["solo_sampled"].append(one("%s s%d" % (pname, seed), prompt, seed=seed, **SAMPLED))
for pname, prompt in (("code", CODE), ("prose", PROSE)):
    for rep in (1, 2):
        res["solo_greedy"].append(one("%s g%d" % (pname, rep), prompt, **GREEDY))
for rnd in range(3):
    jobs = [("c", CODE, 201 + rnd * 10), ("p", PROSE, 202 + rnd * 10), ("c", CODE, 203 + rnd * 10), ("p", PROSE, 204 + rnd * 10)]
    out = [None] * 4
    size0 = os.path.getsize(LOG)
    t0 = time.time()

    def work(i, prompt, seed):
        d, wall = chat(prompt, max_tokens=400, seed=seed, **SAMPLED)
        out[i] = d["usage"]["completion_tokens"]
    th = [threading.Thread(target=work, args=(i, j[1], j[2])) for i, j in enumerate(jobs)]
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.time() - t0
    time.sleep(0.5)
    reqs = [m.groups() for m in RE_REQ.finditer(log_from(size0))]
    res["batch"].append(dict(round=rnd, wall=wall, tokens=sum(out), agg_tps=sum(out) / wall, per_req_tps=[float(g[7]) for g in reqs],
                             acc=sum(int(g[8]) for g in reqs) / max(1, sum(int(g[9]) for g in reqs))))
rng = random.Random(7)
big = " ".join(rng.choice(WORDS) for _ in range(19000)) + "\nReply with exactly: OK"
res["prefill"] = one("cold24k", big, max_tokens=6, **GREEDY)
json.dump(res, open("%s/%s.json" % (OUT, TAG), "w"), indent=1)


def med(v):
    return st.median(v) if v else float("nan")


print("== arm %s" % TAG)
for key in ("solo_sampled", "solo_greedy"):
    for pname in ("code", "prose"):
        b = [r for r in res[key] if r["label"].startswith(pname) and r.get("tps")]
        w = [r["gen_ms"] / r["windows"] for r in b if r.get("windows")]
        print("%-13s %-5s tok/s median %6.1f (min %6.1f max %6.1f)  acceptance %4.1f%%  pw %s  ms/window %s  sha1 %s" % (
            key, pname, med([r["tps"] for r in b]), min(r["tps"] for r in b), max(r["tps"] for r in b),
            100.0 * sum(r["acc"] for r in b) / max(1, sum(r["drf"] for r in b)), "%d/%d" % (sum(1 for r in b if r["pw"]), len(b)),
            "%.1f" % med(w) if w else "-", ",".join(r["sha1"][:6] for r in b)))
print("4 at once     aggregate tok/s %s (rounds), acceptance %.1f%%" % (", ".join("%.0f" % b["agg_tps"] for b in res["batch"]), 100 * med([b["acc"] for b in res["batch"]])))
p = res["prefill"]
print("cold read     %d tokens in %d ms = %.0f tok/s" % (p["read"], p["read_ms"], p["read_tps"]))
