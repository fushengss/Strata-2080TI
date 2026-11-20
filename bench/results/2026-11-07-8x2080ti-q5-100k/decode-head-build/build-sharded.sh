#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
sha256sum -c engine/candidate.sha256
sha256sum engine/strata.candidate engine/decode_head_test engine/libstrata_kq.so > sharded-preserved-artifacts.sha256
tar -xf strata-head-sharded-20261107.tar
nice -n 19 /srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > sharded-build.log 2>&1
cp build/strata engine/strata.sharded
sha256sum engine/strata.sharded > engine/sharded.sha256
bash build-test-sharded.sh > sharded-test-build.log 2>&1
sha256sum -c sharded-preserved-artifacts.sha256
