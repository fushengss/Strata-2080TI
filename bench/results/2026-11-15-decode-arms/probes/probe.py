#!/usr/bin/env python3
"""decode probe on the serving box: a few ~500-token requests to the live engine while the GPU clocks / P-state, the host CPU frequency and C-states and the host
threads' CPU time are recorded.  It changes no setting (the only side effect is the requests).  run: ssh root@10.100.3.3 python3 - < probe.py
Modes (same prompt):  A greedy, no penalty  |  B sampled (temperature 1.0, top_p 0.95, top_k 20), no penalty  |  C like B with presence_penalty 1.5 (the engine decodes it serially)"""
import datetime, json, os, re, statistics as st, subprocess, threading, time, urllib.request

CFG = json.load(open("/srv/Strata/strata-orca-q5_k_m.57.json"))
KEY, MODEL, PORT = CFG.get("api_key", ""), CFG.get("model_name", ""), CFG.get("port", 8080)
LOG = "/srv/Strata/strata-orca-q5_k_m.log"
PID = int(subprocess.run(["pgrep", "-f", "^/srv/Strata-0157/engine/releases/[^ ]*/strata --serve"], capture_output=True, text=True).stdout.split()[0])
OUT = "/tmp/claude-probe"
os.makedirs(OUT, exist_ok=True)
CODE = "Write a complete Python module that implements an LRU cache class with get, put, delete, a TTL option and thread safety, followed by unit tests using unittest. Output only code."
PROSE = "Explain in detail how a pipelined layer-split transformer decoder works across several GPUs that cannot talk to each other directly, and why speculative decoding helps it."
MODES = {"A greedy": dict(temperature=0, presence_penalty=0),
         "B sampled": dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0),
         "C sampled+pen1.5": dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)}


def threads():
    out = {}
    for t in os.listdir("/proc/%d/task" % PID):
        try:
            s = open("/proc/%d/task/%s/stat" % (PID, t)).read()
            name = open("/proc/%d/task/%s/comm" % (PID, t)).read().strip()
            f = s.rsplit(") ", 1)[1].split()
            out[int(t)] = (name, int(f[11]) + int(f[12]), int(f[36]))
        except Exception:
            pass
    return out


def chat(prompt, **kw):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d, t0, time.time()


smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,utilization.gpu,power.draw", "--format=csv,noheader,nounits", "-lms", "250"],
                       stdout=open(OUT + "/gpu.csv", "w"), stderr=subprocess.DEVNULL)
have_msr = os.path.exists("/dev/cpu/0/msr")
turbo = None
turbo_t0 = None
if have_msr:
    turbo_t0 = time.time()
    turbo = subprocess.Popen(["turbostat", "--quiet", "--interval", "0.5", "--cpu", "0,24"], stdout=open(OUT + "/turbo.txt", "w"), stderr=open(OUT + "/turbo.err", "w"))
runs = []
try:
    time.sleep(1.5)
    thr_all0 = threads()
    for pname, prompt in (("code", CODE), ("prose", PROSE)):
        chat(prompt, **MODES["A greedy"])   # warm-up (graph captures, the conversation cache), not recorded
        for mname, kw in MODES.items():
            for rep in (1, 2):
                size0 = os.path.getsize(LOG)
                th0 = threads()
                d, t0, t1 = chat(prompt, **kw)
                th1 = threads()
                time.sleep(0.6)
                with open(LOG, "rb") as f:
                    f.seek(size0)
                    new = f.read().decode(errors="replace")
                runs.append(dict(label="%s %s #%d" % (pname, mname, rep), t0=t0, t1=t1, tokens=d["usage"]["completion_tokens"], th0=th0, th1=th1, log=new))
                time.sleep(1.0)
    thr_all1 = threads()
finally:
    smi.terminate()
    if turbo:
        turbo.terminate()
    time.sleep(0.5)

# ---- GPU samples
gpu = []
for l in open(OUT + "/gpu.csv"):
    p = [x.strip() for x in l.split(",")]
    if len(p) >= 7:
        try:
            ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            gpu.append((ts, int(p[1]), p[2], float(p[3]), float(p[4]), float(p[5]), float(p[6])))
        except Exception:
            pass
# ---- turbostat samples for CPU 0
tcols, tsamples = None, []
if have_msr:
    for l in open(OUT + "/turbo.txt"):
        p = l.split()
        if not p:
            continue
        if p[0] in ("Core", "CPU") or "Avg_MHz" in p:
            tcols = p
            continue
        if tcols and len(p) == len(tcols):
            row = dict(zip(tcols, p))
            if row.get("CPU") == "0":
                tsamples.append(row)
print("turbostat available: %s, samples of CPU 0: %d, columns: %s" % (have_msr, len(tsamples), " ".join(tcols or [])))
if os.path.exists(OUT + "/turbo.err"):
    e = open(OUT + "/turbo.err").read().strip()
    if e:
        print("turbostat stderr:", e[:300])

hdr = "%-24s %4s %5s %6s | %-4s %6s %5s | %-10s %5s %5s %5s %5s | %6s %5s %5s | %6s"
print(hdr % ("run", "tok", "wall", "tok/s", "PW", "gen_ms", "acc", "pstates", "smMHz", "memMH", "mem<", "util%", "bzyMHz", "busy%", "c6%", "main%"))
for r in runs:
    a, b = r["t0"] + 0.7, r["t1"] - 0.2
    wall = r["t1"] - r["t0"]
    g = [x for x in gpu if a <= x[0] <= b]
    ps = sorted(set(x[2] for x in g))
    sm = st.median(x[3] for x in g) if g else float("nan")
    mem = st.median(x[4] for x in g) if g else float("nan")
    memmin = min(x[4] for x in g) if g else float("nan")
    util = st.mean(x[5] for x in g) if g else float("nan")
    m = re.search(r"prompt (\d+) tokens = .*?, (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)", r["log"])
    pw = "yes" if "strata decode head: CUDA" in r["log"] else "no"
    genms = int(m.group(3)) if m else -1
    acc = (100.0 * int(m.group(5)) / max(1, int(m.group(6)))) if m else float("nan")
    # turbostat sample i covers [turbo_t0 + 0.5 i, turbo_t0 + 0.5 (i+1)]
    ts = [s for i, s in enumerate(tsamples) if a <= turbo_t0 + 0.5 * (i + 1) <= b] if have_msr else []
    def tv(k):
        try:
            return st.median(float(s[k]) for s in ts)
        except Exception:
            return float("nan")
    mainticks = r["th1"].get(PID, ("", 0, 0))[1] - r["th0"].get(PID, ("", 0, 0))[1]
    print(hdr % (r["label"], r["tokens"], "%.1f" % wall, "%.1f" % (r["tokens"] / wall), pw, genms, "%.0f%%" % acc, "/".join(ps), "%.0f" % sm, "%.0f" % mem, "%.0f" % memmin,
                 "%.0f" % util, "%.0f" % tv("Bzy_MHz"), "%.0f" % tv("Busy%"), "%.1f" % (tv("CPU%c6") if "CPU%c6" in (tcols or []) else float("nan")), "%.0f%%" % (mainticks / wall)))

# ---- which host threads work during decode (all runs together)
print("\nhost threads, CPU ticks (1/100 s) over the whole probe, top 14:")
dt = []
for tid, (name, ticks, cpu) in thr_all1.items():
    base = thr_all0.get(tid, (name, 0, 0))[1]
    dt.append((ticks - base, name, tid, cpu))
tot_wall = sum(r["t1"] - r["t0"] for r in runs)
for d_, name, tid, cpu in sorted(dt, reverse=True)[:14]:
    print("  %6d ticks  %-18s tid %d  last cpu %d" % (d_, name, tid, cpu))
print("sum of the runs' wall time: %.1f s (100 ticks = 1 s of one core)" % tot_wall)
print("engine log lines of the probe window that mention an error:", len(re.findall(r"ERR|illegal|abort|failed", "".join(r["log"] for r in runs))))
