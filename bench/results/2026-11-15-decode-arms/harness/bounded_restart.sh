#!/bin/bash
# bench/results/2026-11-15-decode-arms/bounded_restart.sh [max_wait_s] [idle_s] - restart the service (restart57.sh) once no request has been active for
# idle_s seconds, or after max_wait_s seconds whatever is running (max_wait_s 0 = restart now; use it while the service is local-only: nobody else is calling).
MAXWAIT=${1:-240}; IDLE=${2:-12}
KEY=$(python3 -c "import json;print(json.load(open(\"/srv/Strata/strata-orca-q5_k_m.57.json\")).get(\"api_key\",\"\"))")
PORT=8080
active() { curl -s -m 5 -H "Authorization: Bearer $KEY" localhost:$PORT/v1/telemetry | python3 -c "
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
while [ "$MAXWAIT" -gt 0 ] && [ $(( $(date +%s) - T0 )) -lt "$MAXWAIT" ]; do
  a=$(active)
  if [ "$a" = "0" ]; then idle=$((idle+3)); else idle=0; fi
  if [ $idle -ge "$IDLE" ]; then echo "$(date -u +%T) UTC idle for ${IDLE} s"; break; fi
  sleep 3
done
[ "$MAXWAIT" -gt 0 ] && [ $idle -lt "$IDLE" ] && echo "$(date -u +%T) UTC never idle for ${IDLE} s within ${MAXWAIT} s: restarting anyway (a request in flight is cut)"
echo "$(date -u +%T) UTC restarting"
/srv/Strata/restart57.sh
echo "$(date -u +%T) UTC restart script returned $?"
