#!/bin/bash
# parallel workers, each runs the instrumented snapshot test N times (niced); prints the summary and any DIFF lines
N=${1:-30}; W=${2:-3}
B=/srv/Strata-park-20261115/build/conversation_snapshot_test
for w in $(seq $W); do
  ( for i in $(seq $N); do nice -n 19 timeout 600 $B > /tmp/claude-parktest/f3_${w}_$i.out 2>&1; done ) &
done
wait
for f in /tmp/claude-parktest/f3_*.out; do tail -1 $f | cut -c1-70; done | sort | uniq -c
grep -h "^DIFF" /tmp/claude-parktest/f3_*.out | sort | uniq -c | sort -rn | head -8
rm -f /tmp/claude-parktest/f3_*.out
