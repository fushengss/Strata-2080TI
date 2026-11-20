#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/longctx.py - the speculative min-p A/B at the live traffic's context size (a ~50K-token prompt), sampled, per request.
    python3 longctx.py     (local-only service; it needs STRATA_DECODE_TIMING=1 for the window columns)"""
import hashlib, json, os, random, re, statistics as st, time, urllib.request
CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part place case week company system "
         "program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name president team minute idea kid body information back parent").split()
SAMPLED = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0)
RE_REQ = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
RE_PL = re.compile(r"strata pipeline: (\d+) windows in (\d+) ms \(([\d.]+) ms/window\): (\d+) speculative, (\d+) on the path, (\d+) rolled back")
rng = random.Random(5)
FILL = " ".join(rng.choice(WORDS) for _ in range(40000))
TASKS = (("prose", "Ignore the random words above. Explain in detail how a pipelined layer-split transformer decoder works across several GPUs that cannot talk to each other directly, and why speculative decoding helps it."),
         ("code", "Ignore the random words above. Write a complete Python module that implements an LRU cache class with get, put, delete, a TTL option and thread safety, followed by unit tests using unittest. Output only code."))


def one(task, seed, **kw):
    size0 = os.path.getsize(LOG)
    body = {"model": MODEL, "messages": [{"role": "user", "content": FILL + "\n\n" + task}], "max_tokens": 500, "stream": False, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}, **SAMPLED, **kw}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.load(urllib.request.urlopen(req, timeout=1200))
    time.sleep(0.4)
    with open(LOG, "rb") as f:
        f.seek(size0)
        new = f.read().decode(errors="replace")
    m = list(RE_REQ.finditer(new))[-1]
    p = RE_PL.search(new)
    return dict(prompt=int(m.group(1)), read=int(m.group(3)), read_ms=int(m.group(4)), gen=int(m.group(5)), ms=int(m.group(6)), tps=float(m.group(7)), acc=int(m.group(8)), drf=int(m.group(9)),
                win=int(p.group(1)) if p else None, sha=hashlib.sha1((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:8])


r0 = one(TASKS[0][1], 99)
print("cold read of the long prompt: %d tokens, %d new, %d ms" % (r0["prompt"], r0["read"], r0["read_ms"]))
print("%-12s %-6s %7s %7s %6s %7s  hashes" % ("min_p", "task", "tok/s", "accept", "wins", "ms/win"))
import sys
VALUES = [None if a == "d" else float(a) for a in sys.argv[1:]] or [None, 0.7, 0.9, None]   # "d" = the engine's default (0.5)
for v in VALUES:
    for name, task in TASKS:
        rs = [one(task, s, **({} if v is None else {"strata_tune": {"spec_min_p": v}})) for s in (101, 102, 103)]
        w = [r for r in rs if r["win"]]
        print("%-12s %-6s %7.1f %6.1f%% %6s %7s  %s" % ("default" if v is None else "%.1f" % v, name, st.median(r["tps"] for r in rs), 100.0 * sum(r["acc"] for r in rs) / max(1, sum(r["drf"] for r in rs)),
              "%d" % st.mean(r["win"] for r in w) if w else "-", "%.2f" % (sum(r["ms"] for r in w) / sum(r["win"] for r in w)) if w else "-", ",".join(r["sha"][:5] for r in rs)))
