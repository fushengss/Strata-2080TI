// bench/results/2026-11-14-park-copy/microbench_d2h.cu - where the 1.2-1.4 GB/s of the conversation park came from.
// Build:  nvcc -O1 -std=c++17 -arch=sm_75 -o microbench_d2h microbench_d2h.cu -lpthread      Run (service idle, nice 19):  ./microbench_d2h 512
// It times, for a 512 MiB device buffer copied to the host in 16 MiB pieces (the size of the old ConversationBuffer segments):
//   1. the first touch of fresh pageable memory against a warm write,
//   2. a D2H copy into warm pageable memory, into fresh pageable memory, into pinned memory (cudaHostAlloc costs 0.54 s per GB: pool it),
//   3. a pinned D2H copy from 1, 2, 4 and 8 cards at once.
// Measured on the 8x RTX 2080 Ti box (2026-11-14): first touch 1.76 GB/s, warm write 5.8; D2H warm pageable 7.3, fresh pageable 1.28 (the engine's
// park ran at 1.37), pinned 10.2 per card, 8 cards at once 39.6 GB/s.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("ERR %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
static const size_t SEG = 16u << 20;
static void touch(uint8_t* p, size_t n, uint8_t v) { for (size_t o = 0; o < n; o += 4096) ((volatile uint8_t*) p)[o] = v; }
static double gbs(size_t n, double s) { return n / 1e9 / s; }
int main(int argc, char** argv) {
    size_t MB = argc > 1 ? atoi(argv[1]) : 512; size_t n = MB << 20;
    CK(cudaSetDevice(0));
    void* d; CK(cudaMalloc(&d, n)); CK(cudaMemset(d, 7, n));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    auto d2h = [&](uint8_t* dst) { double t = now(); for (size_t o = 0; o < n; o += SEG) { CK(cudaMemcpyAsync(dst + o, (char*) d + o, SEG, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s)); } return now() - t; };
    uint8_t* a = (uint8_t*) calloc(n, 1);
    double t = now(); touch(a, n, 1); double ft = now() - t;
    t = now(); memset(a, 0x5A, n); double warm = now() - t;
    printf("pageable memory first touch (4 KiB stride): %6.0f ms = %5.2f GB/s;  warm memset: %6.0f ms = %5.2f GB/s\n", ft * 1e3, gbs(n, ft), warm * 1e3, gbs(n, warm));
    double w = d2h(a); printf("D2H 16 MiB pieces into WARM pageable memory:  %6.0f ms = %5.2f GB/s\n", w * 1e3, gbs(n, w));
    uint8_t* b = (uint8_t*) calloc(n, 1);
    double f = d2h(b); printf("D2H into FRESH pageable memory:               %6.0f ms = %5.2f GB/s\n", f * 1e3, gbs(n, f));
    uint8_t* pin; t = now(); CK(cudaHostAlloc((void**) &pin, n, cudaHostAllocDefault)); double pa = now() - t;
    double p1 = d2h(pin), p2 = d2h(pin);
    printf("pinned: cudaHostAlloc %5.0f ms (once);  D2H %6.0f ms = %5.2f GB/s (2nd run %5.2f GB/s)\n", pa * 1e3, p1 * 1e3, gbs(n, p1), gbs(n, p2));
    for (int T : {1, 2, 4, 8}) {
        size_t per = n / 8;
        std::vector<uint8_t*> ph(T); std::vector<void*> dv(T);
        for (int i = 0; i < T; ++i) { CK(cudaSetDevice(i)); CK(cudaHostAlloc((void**) &ph[i], per, cudaHostAllocDefault)); CK(cudaMalloc(&dv[i], per)); CK(cudaMemset(dv[i], 3, per)); }
        double best = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
            std::vector<std::thread> th; double t0 = now();
            for (int i = 0; i < T; ++i) th.emplace_back([&, i] { cudaSetDevice(i); cudaStream_t st; cudaStreamCreate(&st); for (size_t o = 0; o < per; o += SEG) { cudaMemcpyAsync(ph[i] + o, (char*) dv[i] + o, std::min(SEG, per - o), cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st); } cudaStreamDestroy(st); });
            for (auto& x : th) x.join();
            best = std::min(best, now() - t0);
        }
        printf("pinned, %d GPU(s) in parallel, %4zu MiB each: wall %6.1f ms = %6.2f GB/s aggregate\n", T, per >> 20, best * 1e3, gbs(per * T, best));
        for (int i = 0; i < T; ++i) { CK(cudaSetDevice(i)); cudaFreeHost(ph[i]); cudaFree(dv[i]); }
    }
    return 0;
}
