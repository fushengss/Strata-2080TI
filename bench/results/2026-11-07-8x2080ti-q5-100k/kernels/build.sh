#!/bin/sh
set -eu
OUT=/srv/bench100k-20261107-kernels
SRC=/srv/Strata-0157
CUDA=/usr/local/cuda-13.0/targets/x86_64-linux
mkdir -p "$OUT"
cd "$OUT"
nice -n 19 /usr/bin/c++ -O3 -DNDEBUG -std=c++20 -Wall -Wextra \
  -DSTRATA_HAVE_DFLASH2=1 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERIFY_MAX_T=8 \
  -I"$SRC/include" -I/srv/Strata/build/_deps/strata_llamacpp-src/ggml/include \
  -isystem "$CUDA/include" -isystem "$CUDA/include/cccl" \
  -c "$OUT/native_expert_bench_t4.cpp" -o "$OUT/native_expert_bench_t4.o"
nice -n 19 /usr/bin/c++ -O3 -DNDEBUG "$OUT/native_expert_bench_t4.o" \
  -o "$OUT/native_expert_bench_t4" \
  -L"$CUDA/lib/stubs" -L"$CUDA/lib" -Wl,-rpath,"$CUDA/lib" \
  "$SRC/build/libstrata_kernels.a" "$SRC/build/libstrata_kernels_cpu.a" \
  "$CUDA/lib/libcudart.so" "$SRC/build/libstrata_core.a" "$CUDA/lib/libcudart.so" \
  -ldl -lrt "$SRC/build/ggml/src/libggml-cpu.a" "$SRC/build/ggml/src/libggml-base.a" \
  -lm -lcudadevrt -lcudart_static -lrt -lpthread -ldl
