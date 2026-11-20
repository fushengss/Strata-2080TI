#!/bin/sh
set -eu

experiment=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
default_status=0
"$experiment/operator-bin/q5_decode_tc_bench.fp32" "$@" || default_status=$?
if [ "$default_status" -gt 1 ]; then
    exit "$default_status"
fi

half2_status=0
Q5_TC_HALF2=1 "$experiment/half2-bin/q5_decode_tc_bench" "$@" \
    > "$experiment/capture/operator-half2.jsonl" \
    2> "$experiment/capture/operator-half2.log" || half2_status=$?
if [ "$half2_status" -ne 0 ]; then
    exit "$half2_status"
fi
exit "$default_status"
