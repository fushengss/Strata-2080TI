#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
/srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake -S . -B build -G Ninja -C matched-cache.cmake > configure.log 2>&1
nice -n 19 /srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > build.log 2>&1
mkdir -p engine
cp build/libstrata_kq.so engine/libstrata_kq.so
cp build/strata engine/strata.baseline
sha256sum engine/strata.baseline > engine/baseline.sha256
