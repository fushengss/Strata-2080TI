#!/bin/bash
# bench/results/2026-11-15-decode-arms/runarm.sh <variant> [max_wait_s] - switch the live config to a test arm (local-only), restart, run armtest.py
# run on the box (the restart takes ~2-3 minutes, the test ~2 minutes):  /tmp/claude-decode/runarm.sh head1 0
V=$1; MW=${2:-0}
cd /tmp/claude-decode || exit 1
python3 setcfg2.py "$V" || exit 1
./bounded_restart.sh "$MW" > restart_$V.out 2>&1
tail -3 restart_$V.out
if ! grep -q "ready: http" /srv/Strata/serve-orca.out; then echo "NOT READY after the restart (see /srv/Strata/serve-orca.out)"; exit 2; fi
sleep 5
python3 armtest.py "$V" > arm_$V.out 2>&1
cat arm_$V.out
