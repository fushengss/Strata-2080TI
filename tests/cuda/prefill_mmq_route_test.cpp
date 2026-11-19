// Verify the CUDA MMQ route tables against the production CPU ordering.
#include "strata/prefill/moe_mmq.hpp"

#include "../../src/prefill/mmq_resident_sort.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace detail = strata::prefill::detail;
namespace mmq = strata::prefill::mmq;
constexpr int kExperts = 512, kMaxRows = 30720;

void check(cudaError_t status, const char* action) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

struct DeviceMemory {
    void* data = nullptr;
    explicit DeviceMemory(size_t bytes) { check(cudaMalloc(&data, bytes), "cudaMalloc"); }
    ~DeviceMemory() { cudaFree(data); }
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    int32_t* ints() const { return static_cast<int32_t*>(data); }
};

struct HostCounts {
    int32_t* data = nullptr;
    HostCounts() { check(cudaMallocHost(reinterpret_cast<void**>(&data), kExperts * sizeof(int32_t)), "cudaMallocHost"); }
    ~HostCounts() { cudaFreeHost(data); }
    HostCounts(const HostCounts&) = delete;
    HostCounts& operator=(const HostCounts&) = delete;
};

struct Stream {
    cudaStream_t data = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&data, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() { cudaStreamDestroy(data); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

std::vector<int32_t> make_routes(int rows, int active, std::mt19937& rng) {
    std::vector<int32_t> choices(kExperts);
    std::iota(choices.begin(), choices.end(), 0);
    std::shuffle(choices.begin(), choices.end(), rng);
    std::vector<int32_t> ids(rows);
    for (auto& id : ids) id = choices[rng() % active];
    return ids;
}

void compare(const std::vector<int32_t>& ids, int topk, void* scratch, int32_t* ids_dev, int32_t* slot_dev,
             int32_t* src_dev, int32_t* counts_host, cudaStream_t stream) {
    const int rows = static_cast<int>(ids.size());
    const size_t bytes = ids.size() * sizeof(int32_t);
    std::vector<int32_t> counts(kExperts), offsets(kExperts + 1), order, slot(rows), src(rows);
    for (int e : ids) ++counts[e];
    for (int e = 0; e < kExperts; ++e)
        if (counts[e]) order.push_back(e);
    detail::mmq_resident_sort_rows(ids.data(), rows, topk, counts, offsets, order, slot.data(), src.data());
    check(cudaMemcpyAsync(ids_dev, ids.data(), bytes, cudaMemcpyHostToDevice, stream), "upload routes");
    if (!mmq::resident_route(scratch, ids_dev, rows, topk, slot_dev, src_dev, counts_host, stream))
        throw std::runtime_error("GPU route planning failed");
    std::vector<int32_t> actual_slot(rows), actual_src(rows);
    check(cudaMemcpyAsync(actual_slot.data(), slot_dev, bytes, cudaMemcpyDeviceToHost, stream), "download slots");
    check(cudaMemcpyAsync(actual_src.data(), src_dev, bytes, cudaMemcpyDeviceToHost, stream), "download rows");
    check(cudaStreamSynchronize(stream), "compare route tables");
    if (!std::equal(counts.begin(), counts.end(), counts_host) || actual_slot != slot || actual_src != src)
        throw std::runtime_error("GPU route tables differ from the CPU reference");
}
}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        Stream stream;
        const size_t scratch_bytes = mmq::resident_route_workspace(kMaxRows);
        if (!scratch_bytes) throw std::runtime_error("No CUDA route workspace");
        DeviceMemory scratch(scratch_bytes), ids(kMaxRows * sizeof(int32_t)), slot(kMaxRows * sizeof(int32_t)),
            src(kMaxRows * sizeof(int32_t));
        HostCounts counts;
        std::mt19937 rng(0x5866);
        for (int active : {1, 16, 79, 256, 512}) {
            for (int tokens : {1, 17, 513, 1689, 3072}) {
                compare(make_routes(tokens * 10, active, rng), 10, scratch.data, ids.ints(), slot.ints(), src.ints(),
                        counts.data, stream.data);
            }
        }
        std::vector<int32_t> tied(kExperts * 10);
        for (size_t i = 0; i < tied.size(); ++i) tied[i] = static_cast<int32_t>(i % kExperts);
        std::shuffle(tied.begin(), tied.end(), rng);
        compare(tied, 10, scratch.data, ids.ints(), slot.ints(), src.ints(), counts.data, stream.data);
        const int32_t invalid[] = {-1, 512};
        check(cudaMemcpyAsync(ids.data, invalid, sizeof(invalid), cudaMemcpyHostToDevice, stream.data), "upload invalid routes");
        if (mmq::resident_route(scratch.data, ids.ints(), 2, 1, slot.ints(), src.ints(), counts.data, stream.data))
            throw std::runtime_error("GPU route planning accepted invalid IDs");
        std::puts("PASS: 26 stable CUDA MMQ route permutations and invalid expert IDs");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
