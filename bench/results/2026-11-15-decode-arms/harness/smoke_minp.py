#!/usr/bin/env python3
"""bench/results/2026-11-15-decode-arms/harness/smoke_minp.py - three sampled ~500-token requests (the code prompt, seed 101) against the live service to see that the new --spec-min-p
is in effect.  Draft acceptance of this prompt on a short context: 80.1% at --spec-min-p 0.5, 85.0% at 0.7, 89.5% at 0.9 (3 seeds pooled, raw/sweep_minp.json), so ~87% is expected at 0.8.
Real agent requests may overlap it, so read the speed as indicative only.  The API key is read from the config and never printed."""
import json, re, time, urllib.request
CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
CODE = "Write a complete Python module that implements an LRU cache class with get, put, delete, a TTL option and thread safety, followed by unit tests using unittest. Output only code."
import os
for rep in (1, 2, 3):
    size0 = os.path.getsize(LOG)
    body = {"model": MODEL, "messages": [{"role": "user", "content": CODE}], "max_tokens": 500, "stream": False, "seed": 101, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    time.sleep(0.5)
    with open(LOG, "rb") as f:
        f.seek(size0)
        new = f.read().decode(errors="replace")
    m = [x for x in re.finditer(r"prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", new) if x.group(1) == "49" and int(x.group(2)) >= 100]   # the agents send 1-token requests with small prompts too: mine generates 500
    if m:
        g = m[-1].groups()
        print("run %d: %s tokens, %s tok/s, drafts accepted %s of %s = %.1f%%, pipelined windows %s" % (rep, g[1], g[3], g[4], g[5], 100.0 * int(g[4]) / max(1, int(g[5])),
              "yes" if "strata decode head: CUDA" in new else "no"))
    else:
        print("run %d: no request line found for a 49-token prompt (other requests overlapped?)" % rep)
