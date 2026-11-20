#!/usr/bin/env python3
"""bench/results/2026-11-14-park-copy/parktest.py - two long conversations that alternate with small appends, to time the conversation park.

Run on the serving box (it reads the API key from the service config and never prints it):
    python3 parktest.py --tag before --seed 11 [--tokens 60000] [--turns 3] [--gap 2]

Each request ends with a conversation park; the engine log then holds "conversation cache: parked N tokens in X ms" lines with the snapshot size.
The script copies the log lines of its own window into <tag>.log and the client-side numbers into <tag>.json.  Nothing but filler words is sent.
"""
import argparse, hashlib, json, random, subprocess, sys, time, urllib.request

CFG_PATH = "/srv/Strata/strata-orca-q5_k_m.57.json"
LOG_PATH = "/srv/Strata/strata-orca-q5_k_m.log"
URL = "http://127.0.0.1:8080/v1/chat/completions"
WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part place case week company system "
         "program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue side "
         "kind head house service friend father power hour game line end member law car city community name president team minute idea kid body information back parent "
         "face others level office door health person art war history party result change morning reason research girl guy moment air teacher force education foot boy "
         "age policy music market sense nation plan college interest death experience effect use class control care field development role effort rate heart drug show "
         "leader light voice wife police mind price report decision son view relationship town road arm difference value building action model season society tax director "
         "position player record paper space ground form event official matter center couple site project activity star table need court oil situation cost industry figure "
         "street image phone data picture practice piece land product doctor wall patient worker news test movie north love support technology step baby computer type "
         "attention film tree source organization hair window culture chance brother energy period course summer plant opportunity term letter condition choice").split()


def filler(rng, n_words):
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def chat(key, model, messages, max_tokens=6):
    """one non-streaming request"""
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    return d, time.time() - t0


def log_lines():
    out = subprocess.run(["wc", "-l", LOG_PATH], capture_output=True, text=True).stdout.split()
    return int(out[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=60000)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--append-words", type=int, default=550)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--concurrent", action="store_true", help="send A and B of each turn at the same time (batch slots, as with several agents)")
    ap.add_argument("--churn", type=int, default=0, help="after the A/B rounds: K cycles of a new conversation (read once, never revisited) plus an A/B append pair - the cache fills and evicts")
    ap.add_argument("--churn-tokens", type=int, default=30000)
    ap.add_argument("--count", type=int, default=0, help="ask for the numbers 1..N as the reply, so both conversations decode at once")
    a = ap.parse_args()
    cfg = json.load(open(CFG_PATH))
    key, model = cfg.get("api_key", ""), cfg.get("model_name", "")
    rng = random.Random(a.seed)
    ask = ("Reply with exactly the numbers 1 to %d separated by single spaces and nothing else." % a.count) if a.count else "Reply with exactly: OK"
    start_line = log_lines()
    # first message sizes: 1.25 tokens per filler word as a first guess, corrected after the first answer
    words0 = int(a.tokens / 1.25)
    convs = {}
    for name in ("A", "B"):
        r = random.Random(a.seed * 1000 + ord(name))
        convs[name] = {"rng": r, "messages": [{"role": "user", "content": "Conversation %s-%d. %s\n%s" % (name, a.seed, filler(r, words0), ask)}]}
    results = []

    def one(name, turn):
        c = convs[name]
        d, dt = chat(key, model, c["messages"], max_tokens=(a.count * 2 + 40) if a.count else 6)
        reply = (d["choices"][0]["message"].get("content") or "").strip()
        u = d.get("usage", {})
        details = u.get("prompt_tokens_details") or {}
        results.append({"conv": name, "turn": turn, "client_s": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"),
                        "cached_tokens": details.get("cached_tokens"), "completion_tokens": u.get("completion_tokens"), "reply": reply[:20],
                        "reply_sha1": hashlib.sha1(reply.encode()).hexdigest()[:12]})
        print("%s%d  %.2f s  prompt %s cached %s  reply %r" % (name, turn, dt, u.get("prompt_tokens"), details.get("cached_tokens"), reply[:20]), flush=True)
        c["messages"].append({"role": "assistant", "content": reply})
        c["messages"].append({"role": "user", "content": "Next part. %s\n%s" % (filler(c["rng"], a.append_words), ask)})

    import threading
    if a.concurrent:
        for turn in range(0, a.turns + 1):
            th = [threading.Thread(target=one, args=(n, turn)) for n in ("A", "B")]
            for t in th: t.start()
            for t in th: t.join()
            time.sleep(a.gap)
    else:
        for name, turn in [("A", 0), ("B", 0)] + [(n, i) for i in range(1, a.turns + 1) for n in ("A", "B")]:
            one(name, turn)
            time.sleep(a.gap)
    for k in range(1, a.churn + 1):
        r = random.Random(a.seed * 7919 + k)
        name = "N%d" % k
        convs[name] = {"rng": r, "messages": [{"role": "user", "content": "Conversation %s-%d. %s\n%s" % (name, a.seed, filler(r, int(a.churn_tokens / 1.25)), ask)}]}
        one(name, 0)
        time.sleep(a.gap)
        th = [threading.Thread(target=one, args=(n, a.turns + k)) for n in ("A", "B")]
        for t in th: t.start()
        for t in th: t.join()
        time.sleep(a.gap)
    time.sleep(3)
    end_line = log_lines()
    cmd = "sed -n '%d,%dp' %s | grep -a -E 'conversation cache:|strata batch: slot|strata serve: prompt|parktrace'" % (start_line + 1, end_line, LOG_PATH)
    lines = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout
    open(a.tag + ".log", "w").write(lines)
    json.dump({"tag": a.tag, "seed": a.seed, "target_tokens": a.tokens, "turns": a.turns, "requests": results}, open(a.tag + ".json", "w"), indent=1)
    print("log window %d-%d -> %s.log" % (start_line + 1, end_line, a.tag))


if __name__ == "__main__":
    sys.exit(main())
