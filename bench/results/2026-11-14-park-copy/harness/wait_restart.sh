#!/bin/bash
# restart the service once it has been idle (no active request) for 30 s; give up after 90 minutes
KEY=$(python3 -c "import json;print(json.load(open(\"/srv/Strata/strata-orca-q5_k_m.57.json\")).get(\"api_key\",\"\"))")
active() { curl -s -m 5 -H "Authorization: Bearer $KEY" localhost:8080/v1/telemetry | python3 -c "
import sys, json
d = json.load(sys.stdin)
def find(o, k):
    if isinstance(o, dict):
        for kk, v in o.items():
            if kk == k: return v
            r = find(v, k)
            if r is not None: return r
print(find(d, \"active\"))" 2>/dev/null; }
T0=$(date +%s); idle=0
while [ $(( $(date +%s) - T0 )) -lt 5400 ]; do
  a=$(active)
  if [ "$a" = "0" ]; then idle=$((idle+1)); else idle=0; fi
  if [ $idle -ge 10 ]; then
    echo "$(date -u +%T) UTC idle for 30 s, restarting"
    /srv/Strata/restart57.sh
    echo "$(date -u +%T) UTC restart script returned $?"
    exit 0
  fi
  sleep 3
done
echo "$(date -u +%T) UTC gave up: the service never idled for 30 s"
