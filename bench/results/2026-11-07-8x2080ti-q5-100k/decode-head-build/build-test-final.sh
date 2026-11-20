#!/bin/bash
set -euo pipefail
cd /srv/Strata-decode-head-20261107
/usr/bin/c++ -O3 -DNDEBUG -std=c++20 -DSTRATA_HAVE_DFLASH2=1 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 -DSTRATA_VERIFY_MAX_T=8 -Iinclude -I/usr/local/cuda/targets/x86_64-linux/include -I/usr/local/cuda/targets/x86_64-linux/include/cccl -c tests/core/decode_head_test.cpp -o build/decode_head_test.o
cd build
/usr/bin/c++ -O3 -DNDEBUG decode_head_test.o -o ../engine/decode_head_test.final -Wl,--start-group libstrata_engine.a libstrata_kernels.a libstrata_core.a libstrata_kernels_cpu.a ggml/src/libggml-cpu.a ggml/src/libggml-base.a -Wl,--end-group -L/usr/local/cuda/targets/x86_64-linux/lib/stubs -L/usr/local/cuda/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda/targets/x86_64-linux/lib -lcudart -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lm
sha256sum ../engine/decode_head_test.final
