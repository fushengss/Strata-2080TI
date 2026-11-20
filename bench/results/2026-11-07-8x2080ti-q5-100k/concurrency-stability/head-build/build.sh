#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
TASK_OUT=/srv/bench100k-20261107-concurrency-mtp/head-build
test ! -e "$TASK_OUT/source-before.tar.gz"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
sha256sum engine/strata.mtp2 engine/libstrata_kq.so /srv/Strata-0157/engine/releases/mtp-head-20261107/strata /srv/Strata-0157/engine/releases/mtp-head-20261107/libstrata_kq.so > "$TASK_OUT/preserved.sha256"
python3 - <<'PY'
import pathlib,json,hashlib,tarfile
out=pathlib.Path('/srv/bench100k-20261107-concurrency-mtp/head-build')
manifest=json.loads((out/'source-manifest.json').read_text())
before={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in manifest}
(out/'source-before-manifest.json').write_text(json.dumps(before,indent=2))
with tarfile.open(out/'source-before.tar.gz','w:gz') as t:
    for p in manifest:t.add(p,arcname=p)
PY
cp -p build/strata "$TASK_OUT/strata-before-build"
tar -xzf "$TASK_OUT/source.tar.gz"
python3 - <<'PY'
import pathlib,json,hashlib
out=pathlib.Path('/srv/bench100k-20261107-concurrency-mtp/head-build')
for p,h in json.loads((out/'source-manifest.json').read_text()).items():
    assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
print('FROZEN_SOURCE_MATCH',flush=True)
PY
/srv/Strata/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake --build build --target strata -j16 > "$TASK_OUT/build.log" 2>&1
cp -p build/strata engine/strata.piped-head1
TASK_CUDA=/usr/local/cuda-13.0/targets/x86_64-linux
TASK_DEFS=(-DSTRATA_HAVE_DFLASH2=1 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERIFY_MAX_T=8)
TASK_INCLUDES=(-Iinclude -I"$TASK_CUDA/include" -I"$TASK_CUDA/include/cccl" -I/srv/Strata/build/_deps/strata_llamacpp-src/ggml/include)
/usr/bin/c++ -O3 -DNDEBUG -std=c++20 "${TASK_DEFS[@]}" "${TASK_INCLUDES[@]}" -c tests/core/mtp_head_test.cpp -o build/mtp_head_test.piped-head1.o > "$TASK_OUT/test-build.log" 2>&1
cd build
/usr/bin/c++ -O3 -DNDEBUG mtp_head_test.piped-head1.o -o ../engine/mtp_head_test.piped-head1 -Wl,--start-group libstrata_engine.a libstrata_kernels.a libstrata_core.a libstrata_kernels_cpu.a ggml/src/libggml-cpu.a ggml/src/libggml-base.a -Wl,--end-group -L"$TASK_CUDA/lib/stubs" -L"$TASK_CUDA/lib" -Wl,-rpath,"$TASK_CUDA/lib" -lcudart -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lm >> "$TASK_OUT/test-build.log" 2>&1
cd ..
engine/mtp_head_test.piped-head1 --contract-only > "$TASK_OUT/contract.log" 2>&1
sha256sum engine/strata.piped-head1 engine/mtp_head_test.piped-head1 engine/libstrata_kq.so > "$TASK_OUT/candidate.sha256"
sha256sum -c "$TASK_OUT/preserved.sha256"
cat "$TASK_OUT/candidate.sha256"
