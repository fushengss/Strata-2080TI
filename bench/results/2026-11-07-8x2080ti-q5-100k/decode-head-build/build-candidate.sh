#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
while ! test -s engine/baseline.sha256; do
    if ! pgrep -f "^bash /srv/Strata-decode-head-20261107/build-matched.sh$" >/dev/null; then
        echo "Isolated baseline build ended without a successful artifact" >&2
        exit 1
    fi
    sleep 2
done
tar -xf strata-head-candidate-20261107.tar
/srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake -S . -B build -G Ninja -C matched-cache.cmake > candidate-configure.log 2>&1
nice -n 19 /srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > candidate-build.log 2>&1
cp build/libstrata_kq.so engine/libstrata_kq.so
cp build/strata engine/strata.candidate
sha256sum engine/strata.candidate > engine/candidate.sha256
