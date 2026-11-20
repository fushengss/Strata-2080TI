#!/bin/bash
cd /tmp/claude-parktest
for V in "$@"; do ./abvariant.sh $V 21; done
echo CHAIN_DONE
