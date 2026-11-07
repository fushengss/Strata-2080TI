#include "strata/core/mtp_head.hpp"

#if defined(STRATA_GPU_METAL)
namespace strata::core {
MtpHeadShards::~MtpHeadShards() = default;
bool MtpHeadShards::init(const void*, int, int, int, int, int, int, std::string& err) {
    err = "mtp head: CUDA row-shard graphs are unavailable on Metal"; return false;
}
bool MtpHeadShards::enqueue(cudaStream_t, const void*, float*, int, std::string& err) {
    err = "mtp head: unavailable on Metal"; return false;
}
bool MtpHeadShards::drain(std::string&) { return true; }
void MtpHeadShards::reset() noexcept {}
}  // namespace strata::core
#else

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/elementwise.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace strata::core {
namespace {
struct DeviceScope {
    int previous = -1;
    cudaError_t status = cudaSuccess;
    bool changed = false;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != device) {
            status = cudaSetDevice(device);
            changed = status == cudaSuccess;
        }
    }
    ~DeviceScope() { if (changed) (void) cudaSetDevice(previous); }
};
bool checked(cudaError_t status, const char* operation, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("mtp head: ") + operation + ": " + cudaGetErrorString(status);
    return false;
}
}  // namespace

MtpHeadShards::~MtpHeadShards() { reset(); }

bool MtpHeadShards::drain(std::string& err) {
    bool ok = true;
    // All target waits refer to source-ready events already recorded by enqueue(). Source waits are installed
    // only after both target completion records succeed, so an enqueue failure cannot leave an unrecorded wait.
    if (source_ >= 0 && last_source_stream_) {
        const DeviceScope source(source_);
        if (!checked(source.status, "select source to drain", err) ||
            !checked(cudaStreamSynchronize(last_source_stream_), "drain source", err)) ok = false;
    }
    for (auto& s : shards_) if (s.device >= 0 && s.stream) {
        const DeviceScope target(s.device);
        if (!checked(target.status, "select target to drain", err) ||
            !checked(cudaStreamSynchronize(s.stream), "drain target", err)) ok = false;
    }
    return ok;
}

void MtpHeadShards::reset() noexcept {
    // No allocation or CUDA call on the default-off path.
    if (source_ < 0) return;
    std::string ignored;
    (void) drain(ignored);
    for (auto& s : shards_) if (s.device >= 0) {
        const DeviceScope target(s.device);
        // A failed device selection must not free a different card's handles. During unrecoverable CUDA
        // shutdown these resources can only be reclaimed by context teardown.
        if (target.status != cudaSuccess) continue;
        if (s.graph) (void) cudaGraphExecDestroy(s.graph);
        for (auto e : s.finished) if (e) (void) cudaEventDestroy(e);
        if (s.stream) (void) cudaStreamDestroy(s.stream);
        if (s.weights) (void) cudaFree(s.weights);
        if (s.q8) (void) cudaFree(s.q8);
        if (s.logits) (void) cudaFree(s.logits);
        s = {};
    }
    {
        const DeviceScope source(source_);
        if (source.status == cudaSuccess) {
            for (auto& e : producer_) { if (e) (void) cudaEventDestroy(e); e = nullptr; }
            if (activation_host_) (void) cudaFreeHost(activation_host_);
            if (logits_host_) (void) cudaFreeHost(logits_host_);
        }
    }
    source_ = -1; n_ = vocab_ = 0; q8_bytes_ = 0; ready_ = false;
    activation_host_ = activation_source_ = nullptr;
    logits_host_ = logits_source_ = nullptr;
    last_source_stream_ = nullptr;
}

bool MtpHeadShards::init(const void* weights, int source_device, int type, int n, int vocab,
                         int target0, int target1, std::string& err) {
    using namespace strata::kernels;
    if (source_ >= 0) { err = "mtp head: already initialized"; return false; }
    if (!weights || source_device < 0 || target0 < 0 || target1 < 0 || target0 == target1 ||
        target0 == source_device || target1 == source_device || n <= 0 || n % 256 || vocab < 256) {
        err = "mtp head: invalid devices or row shape"; return false;
    }
    if (type != 14 || native_q6_k_packed_enabled()) {
        err = "mtp head: only original unpacked Q6_K rows are supported (disable Q6_PACKED / mtp-q4)";
        return false;
    }
    const DeviceScope source(source_device);
    if (!checked(source.status, "select source", err)) return false;
    int devices = 0;
    if (!checked(cudaGetDeviceCount(&devices), "count devices", err)) return false;
    if (source_device >= devices || target0 >= devices || target1 >= devices) {
        err = "mtp head: target device is outside the visible device set"; return false;
    }
    source_ = source_device; n_ = n; vocab_ = vocab;
    auto fail = [&]() { reset(); return false; };
    try {
        q8_bytes_ = native_q8_1_bytes(n, 1);
        if (!checked(cudaHostAlloc(&activation_host_, q8_bytes_, cudaHostAllocMapped | cudaHostAllocPortable), "allocate activation", err) ||
            !checked(cudaHostGetDevicePointer(&activation_source_, activation_host_, 0), "map source activation", err) ||
            !checked(cudaHostAlloc((void**) &logits_host_, (size_t) vocab * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "allocate full logits", err) ||
            !checked(cudaHostGetDevicePointer((void**) &logits_source_, logits_host_, 0), "map source logits", err)) return fail();
        std::memset(activation_host_, 0, q8_bytes_);
        std::memset(logits_host_, 0, (size_t) vocab * sizeof(float));
        for (auto& e : producer_)
            if (!checked(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create producer event", err)) return fail();
        const size_t weight_row = native_mmvq_weight_bytes(type, n, 1);
        // Preserve the original MMVQ row/tile phase at the second shard, including the odd 106299-row subset.
        const int split = (vocab / 2 / 128) * 128;
        for (int i = 0; i < 2; ++i) {
            auto& s = shards_[i];
            s.device = i == 0 ? target0 : target1;
            s.begin = i == 0 ? 0 : split;
            s.count = i == 0 ? split : vocab - split;
            // Stage only this half, preserving the gathered subset's exact bytes and row order. No P2P required.
            std::vector<unsigned char> host((size_t) s.count * weight_row);
            if (!checked(cudaMemcpy(host.data(), (const unsigned char*) weights + (size_t) s.begin * weight_row,
                                    host.size(), cudaMemcpyDeviceToHost), "read shard weights", err)) return fail();
            const DeviceScope target(s.device);
            if (!checked(target.status, "select target", err)) return fail();
            int lo = 0, hi = 0;
            const char* priority = std::getenv("STRATA_MTP_PRIORITY");
            if ((!priority || std::atoi(priority) != 0) &&
                !checked(cudaDeviceGetStreamPriorityRange(&lo, &hi), "get target priority", err)) return fail();
            if (!checked(cudaStreamCreateWithPriority(&s.stream, cudaStreamNonBlocking, hi), "create target stream", err) ||
                !checked(cudaMalloc(&s.weights, host.size()), "allocate shard weights", err) ||
                !checked(cudaMemcpy(s.weights, host.data(), host.size(), cudaMemcpyHostToDevice), "copy shard weights", err) ||
                !checked(cudaMalloc(&s.q8, q8_bytes_), "allocate target Q8_1", err) ||
                !checked(cudaMalloc((void**) &s.logits, (size_t) s.count * sizeof(float)), "allocate target logits", err) ||
                !checked(cudaHostGetDevicePointer(&s.activation, activation_host_, 0), "map target activation", err) ||
                !checked(cudaHostGetDevicePointer((void**) &s.mapped_logits, logits_host_, 0), "map target logits", err)) return fail();
            for (auto& e : s.finished)
                if (!checked(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create target event", err)) return fail();
            if (!checked(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal), "begin head capture", err)) return fail();
            bool ok = true;
            try {
                copy_from_mapped((float*) s.q8, (const float*) s.activation, (int64_t) q8_bytes_ / sizeof(float), s.stream);
                native_mmvq(type, s.weights, s.q8, s.logits, n, s.count, 1, s.stream);
                // A vocabulary/subset need not contain a multiple of four rows (106299 -> 53120/53179).
                // Copy the exact 32-bit words: no float arithmetic, padding, or access beyond this shard.
                copy_i32_from_mapped((int32_t*) (s.mapped_logits + s.begin), (const int32_t*) s.logits, s.count, s.stream);
            } catch (const std::exception& e) { err = std::string("mtp head: capture: ") + e.what(); ok = false; }
            cudaGraph_t graph = nullptr;
            const cudaError_t ended = cudaStreamEndCapture(s.stream, &graph);
            if (ok) ok = checked(ended, "end head capture", err);
            if (ok) ok = checked(cudaGraphInstantiate(&s.graph, graph, nullptr, nullptr, 0), "instantiate head", err);
            if (graph) (void) cudaGraphDestroy(graph);
            if (ok) ok = checked(cudaGraphUpload(s.graph, s.stream), "upload head", err);
            if (ok) ok = checked(cudaStreamSynchronize(s.stream), "finish head upload", err);
            if (!ok) return fail();
        }
        ready_ = true;
        return true;
    } catch (const std::exception& e) {
        err = std::string("mtp head: init: ") + e.what();
        return fail();
    }
}

bool MtpHeadShards::enqueue(cudaStream_t source_stream, const void* q8, float* logits, int step, std::string& err) {
    using namespace strata::kernels;
    if (!ready_ || !source_stream || !q8 || !logits || step < 0 || step >= 8) {
        err = "mtp head: enqueue needs initialized shards, a source stream and step 0..7"; return false;
    }
    const DeviceScope source(source_);
    if (!checked(source.status, "select enqueue source", err)) return false;
    last_source_stream_ = source_stream;
    try {
        copy_from_mapped((float*) activation_source_, (const float*) q8, (int64_t) q8_bytes_ / sizeof(float), source_stream);
    } catch (const std::exception& e) { err = std::string("mtp head: publish activation: ") + e.what(); return false; }
    if (!checked(cudaEventRecord(producer_[step], source_stream), "record producer", err)) return false;
    for (auto& s : shards_) {
        const DeviceScope target(s.device);
        if (!checked(target.status, "select enqueue target", err) ||
            !checked(cudaStreamWaitEvent(s.stream, producer_[step], 0), "wait for source", err) ||
            !checked(cudaGraphLaunch(s.graph, s.stream), "launch target head", err) ||
            !checked(cudaEventRecord(s.finished[step], s.stream), "record target completion", err)) return false;
    }
    for (auto& s : shards_)
        if (!checked(cudaStreamWaitEvent(source_stream, s.finished[step], 0), "wait for target", err)) return false;
    try {
        // The source's original tail reads precisely vocab_ logits; its existing allocation is not padded here.
        copy_i32_from_mapped((int32_t*) logits, (const int32_t*) logits_source_, vocab_, source_stream);
    }
    catch (const std::exception& e) { err = std::string("mtp head: gather full logits: ") + e.what(); return false; }
    return true;
}

}  // namespace strata::core
#endif
