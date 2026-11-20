#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
sha256sum engine/strata.candidate engine/strata.sharded engine/decode_head_test engine/decode_head_test.sharded engine/libstrata_kq.so > final-preserved-artifacts.sha256
nice -n 19 /srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > final-build.log 2>&1
cp build/strata engine/strata.final
sha256sum engine/strata.final > engine/final.sha256
bash build-test-final.sh > final-test-build.log 2>&1
engine/decode_head_test.final --merge-only > final-merge-test.log 2>&1
sha256sum -c final-preserved-artifacts.sha256
