#!/bin/bash
# abvariant.sh <variant> <seed>: switch the live config, restart when idle, run the churn test, keep the outputs under /tmp/claude-parktest/<variant>.*
V=$1; SEED=$2
cd /tmp/claude-parktest
python3 setcfg.py $V
./wait_restart.sh > wait_$V.out 2>&1
tail -2 wait_$V.out
python3 parktest.py --tag $V --seed $SEED --concurrent --turns 2 --count 300 --churn 10 --churn-tokens 30000 > $V.out 2>&1
echo "$V done: $(tail -1 $V.out)"
