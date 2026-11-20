import json, sys
p = "/srv/Strata/strata-orca-q5_k_m.57.json"
cfg = json.load(open(p))
v = sys.argv[1]
OLD = "/srv/Strata-0157/engine/releases/concurrent-prefill-20261108/strata"
NEW = "/srv/Strata-0157/engine/releases/park-warm-20261115/strata"
NEW2 = "/srv/Strata-0157/engine/releases/park-warm2-20261115/strata"
NEW3 = "/srv/Strata-0157/engine/releases/park-warm3-20261115/strata"
env = cfg["env"]
for k in ("LD_PRELOAD", "STRATA_BLOCK_CACHE_MIB", "STRATA_BLOCK_CACHE_PREWARM_MIB", "STRATA_BLOCK_CACHE_STATS", "GLIBC_TUNABLES"):
    env.pop(k, None)
env["STRATA_PARK_TRACE"] = "1"
cfg["exe"] = OLD if v.startswith("old") else NEW3 if v.startswith("new3") else NEW2 if v.startswith("new2") else NEW
env.pop("STRATA_SLOT_DRAFT_DIRECT", None)
if v.startswith("new3cd"):
    env["STRATA_SLOT_DRAFT_DIRECT"] = "1"
env.pop("STRATA_CKPT_THREADS", None)
if v.startswith("new2c") or v.startswith("new3c"):
    env["STRATA_CKPT_THREADS"] = "1"
if "shim" in v:
    env.update({"LD_PRELOAD": "/srv/Strata/libwarmalloc.so", "STRATA_BLOCK_CACHE_MIB": "8192", "STRATA_BLOCK_CACHE_STATS": "1"})
if v == "final":    # the deployed set: engine park-warm3, checkpoint parts at once, slot draft K/V card to card, no trace, the 128 GiB cache
    cfg["exe"] = NEW3
    env["STRATA_CKPT_THREADS"] = "1"
    env["STRATA_SLOT_DRAFT_DIRECT"] = "1"
    env.pop("STRATA_PARK_TRACE", None)
if v.startswith("chunk"):    # chunk<N>[m<MIN>]: engine park-warm4 with the deployed settings, --split-chunks N, STRATA_SPLIT_CHUNK_MIN=MIN
    import re
    mm = re.match(r"chunk(\d+)(?:m(\d+))?", v)
    cfg["exe"] = "/srv/Strata-0157/engine/releases/park-warm4-20261115/strata"
    env["STRATA_CKPT_THREADS"] = "1"
    env["STRATA_SLOT_DRAFT_DIRECT"] = "1"
    env.pop("STRATA_PARK_TRACE", None)
    env.pop("STRATA_SPLIT_CHUNK_MIN", None)
    if mm.group(2): env["STRATA_SPLIT_CHUNK_MIN"] = mm.group(2)
    a = cfg["args"]; a[a.index("--split-chunks") + 1] = mm.group(1)
args = cfg["args"]
i = args.index("--conversation-cache-mib")
args[i + 1] = "8192" if "small" in v else "131072"
json.dump(cfg, open(p, "w"), indent=2); open(p, "a").write("\n")
print("config set for", v, "exe", cfg["exe"].split("/")[-2], "cache", args[i + 1], "env", sorted(k for k in env if k in ("LD_PRELOAD", "STRATA_PARK_TRACE", "STRATA_BLOCK_CACHE_MIB")))
