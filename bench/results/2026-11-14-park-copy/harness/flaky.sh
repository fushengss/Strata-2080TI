#!/bin/bash
# run the snapshot test N times for the patched and the original build, count passes
N=${1:-15}
for which in patched original; do
  if [ $which = patched ]; then B=/srv/Strata-park-20261115/build/conversation_snapshot_test; else B=/srv/Strata-decode-head-20261107/build/conversation_snapshot_test; fi
  pass=0; fail=0; msgs=""
  for i in $(seq $N); do
    out=$(timeout 600 nice -n 19 $B 2>&1 | tail -1)
    if echo "$out" | grep -q "checks passed"; then pass=$((pass+1)); else fail=$((fail+1)); msgs="$msgs | $out"; fi
  done
  echo "$which: $pass passed, $fail failed $msgs"
done
