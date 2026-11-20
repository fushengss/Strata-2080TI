#!/usr/bin/env python3
"""bench/results/2026-11-14-park-copy/readsweep.py - how long a read of N new tokens takes, for a spread of N, on a conversation that is already cached.

Run on the serving box:  python3 readsweep.py --tag chunk2 --seed 41 [--base 30000] [--sizes 300,800,1500,3000,6000,12000] [--rounds 2]
One conversation is read once (--base tokens), then for each size an append of about that many tokens is sent (one request at a time, a reply of 4 tokens
at most), repeated --rounds times.  The engine log lines of the window go to <tag>.log (the "prompt ... read in X ms" line is the read), the client numbers to <tag>.json.
Cold reads of several sizes (nothing cached) follow, as the prefill check.
"""
import argparse, hashlib, json, random, subprocess, sys, time, urllib.request

CFG_PATH = "/srv/Strata/strata-orca-q5_k_m.57.json"
LOG_PATH = "/srv/Strata/strata-orca-q5_k_m.log"
URL = "http://127.0.0.1:8080/v1/chat/completions"
sys.path.insert(0, "/tmp/claude-parktest")
from parktest import WORDS, filler  # noqa: E402


def chat(key, model, messages):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": 4, "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    return d, time.time() - t0


def log_lines():
    return int(subprocess.run(["wc", "-l", LOG_PATH], capture_output=True, text=True).stdout.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", type=int, default=30000)
    ap.add_argument("--sizes", default="300,800,1500,3000,6000,12000")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cold", default="2000,8000,24000")
    a = ap.parse_args()
    cfg = json.load(open(CFG_PATH))
    key, model = cfg.get("api_key", ""), cfg.get("model_name", "")
    start = log_lines()
    out = []
    ask = "Reply with exactly: OK"

    def send(conv, name, size, kind):
        d, dt = chat(key, model, conv["messages"])
        u = d.get("usage", {}); det = u.get("prompt_tokens_details") or {}
        reply = (d["choices"][0]["message"].get("content") or "").strip()
        out.append({"kind": kind, "size": size, "client_s": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"), "cached": det.get("cached_tokens"),
                    "reply_sha1": hashlib.sha1(reply.encode()).hexdigest()[:10]})
        print("%-6s %5s  %.2f s  prompt %s cached %s" % (kind, size, dt, u.get("prompt_tokens"), det.get("cached_tokens")), flush=True)
        conv["messages"].append({"role": "assistant", "content": reply})
        time.sleep(1.5)

    rng = random.Random(a.seed)
    conv = {"messages": [{"role": "user", "content": "Conversation S-%d. %s\n%s" % (a.seed, filler(rng, int(a.base / 1.25)), ask)}]}
    send(conv, "S", a.base, "base")
    for rnd in range(a.rounds):
        for size in [int(x) for x in a.sizes.split(",")]:
            conv["messages"].append({"role": "user", "content": "Next part. %s\n%s" % (filler(rng, int(size / 1.25)), ask)})
            send(conv, "S", size, "append")
    for size in [int(x) for x in a.cold.split(",") if x]:
        r = random.Random(a.seed * 31 + size)
        c = {"messages": [{"role": "user", "content": "Cold read %d-%d. %s\n%s" % (size, a.seed, filler(r, int(size / 1.25)), ask)}]}
        send(c, "C", size, "cold")
    time.sleep(2)
    end = log_lines()
    cmd = "sed -n '%d,%dp' %s | grep -a -E 'conversation cache:|strata batch: slot|strata serve: prompt|parktrace'" % (start + 1, end, LOG_PATH)
    open(a.tag + ".log", "w").write(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout)
    json.dump({"tag": a.tag, "requests": out}, open(a.tag + ".json", "w"), indent=1)
    print("log window %d-%d -> %s.log" % (start + 1, end, a.tag))


if __name__ == "__main__":
    sys.exit(main())
