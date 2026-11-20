#!/bin/bash
# bench/results/2026-11-15-decode-arms/harness/restore_production.sh - put the deployed production config back and restart the service.
# Run on the serving box, or from any machine that can reach it:   ssh root@10.100.3.3 'bash -s' < restore_production.sh
# The deployed config is the one of 2026-11-15 04:40 CST: --spec-min-p 0.8, "parallel": 8, "host": "0.0.0.0" (copy: .final-park-warm4-minp08).
# A test arm leaves the live config with "host": "127.0.0.1" (and, for the 16-slot arm, "parallel": 16): the agents on 10.100.3.26 then cannot connect.
set -e
BASE=/srv/Strata/strata-orca-q5_k_m.57.json.final-park-warm4-minp08
LIVE=/srv/Strata/strata-orca-q5_k_m.57.json
cp "$BASE" "$LIVE"
cmp "$BASE" "$LIVE" && echo "live config == the deployed copy"
setsid nohup /srv/Strata/restart57.sh > /srv/Strata/restart57.log 2>&1 < /dev/null &
echo "restart started (stopping ~3 min, starting ~2.5 min). Check:  tail -3 /srv/Strata/restart57.log ;  ss -ltn | grep ':8080 '   (must show 0.0.0.0:8080)"
