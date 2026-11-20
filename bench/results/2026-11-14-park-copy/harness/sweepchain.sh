#!/bin/bash
cd /tmp/claude-parktest
for V in "$@"; do ./sweepvariant.sh $V 41; done
echo CHAIN_DONE
