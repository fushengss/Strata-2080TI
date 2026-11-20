#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
sha256sum engine/strata.final engine/libstrata_kq.so /srv/Strata-0157/engine/releases/decode-head-20261107/strata /srv/Strata-0157/engine/releases/decode-head-20261107/libstrata_kq.so > /srv/bench100k-20261107-mtp-head-build/preserved.sha256
tar -xzf /srv/bench100k-20261107-mtp-head-build/candidate-source.tar.gz
python3 - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('/srv/bench100k-20261107-mtp-head-build/candidate-source-manifest.json').read_text())
for p,h in m.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, p
print('Source manifest matches')
PY
nice -n 19 /srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > /srv/bench100k-20261107-mtp-head-build/candidate-build.log 2>&1
cp build/strata engine/strata.mtp1
sha256sum engine/strata.mtp1 > /srv/bench100k-20261107-mtp-head-build/candidate.sha256
sha256sum -c /srv/bench100k-20261107-mtp-head-build/preserved.sha256
