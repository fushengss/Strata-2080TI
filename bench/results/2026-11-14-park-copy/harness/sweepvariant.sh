#!/bin/bash
# sweepvariant.sh <variant> <seed>: set the config, restart when idle, run the read sweep
V=$1; SEED=$2
cd /tmp/claude-parktest
python3 setcfg.py $V
./wait_restart.sh > wait_$V.out 2>&1
tail -1 wait_$V.out
python3 readsweep.py --tag $V --seed $SEED > $V.out 2>&1
echo "$V done: $(tail -1 $V.out)"
