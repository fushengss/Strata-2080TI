#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
TASK_CUDA=/usr/local/cuda-13.0/targets/x86_64-linux
TASK_DEFS=(-DSTRATA_HAVE_DFLASH2=1 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERIFY_MAX_T=8)
TASK_INCLUDES=(-Iinclude -I"$TASK_CUDA/include" -I"$TASK_CUDA/include/cccl" -I/srv/Strata/build/_deps/strata_llamacpp-src/ggml/include)
nice -n 19 /usr/bin/c++ -O3 -DNDEBUG -std=c++20 "${TASK_DEFS[@]}" "${TASK_INCLUDES[@]}" -c tests/core/mtp_head_test.cpp -o build/mtp_head_test.o
nice -n 19 /usr/local/cuda-13.0/bin/nvcc -O3 -DNDEBUG -std=c++20 -arch=sm_75 --use_fast_math "${TASK_DEFS[@]}" "${TASK_INCLUDES[@]}" -c tests/core/local_tp_bench.cu -o build/local_tp_bench.o
cd build
for task_test in mtp_head_test local_tp_bench; do
  /usr/bin/c++ -O3 -DNDEBUG "$task_test.o" -o "../engine/$task_test.mtp1" -Wl,--start-group libstrata_engine.a libstrata_kernels.a libstrata_core.a libstrata_kernels_cpu.a ggml/src/libggml-cpu.a ggml/src/libggml-base.a -Wl,--end-group -L"$TASK_CUDA/lib/stubs" -L"$TASK_CUDA/lib" -Wl,-rpath,"$TASK_CUDA/lib" -lcudart -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lm
done
../engine/mtp_head_test.mtp1 --contract-only
sha256sum ../engine/mtp_head_test.mtp1 ../engine/local_tp_bench.mtp1
