#include "strata/core/decode_head.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/decode_head.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace strata::core {
namespace {
// Unlike an unchecked device scope, a failed switch must never run an allocation/launch on the caller's GPU.
struct DeviceScope {
    int previous = -1;
    bool changed = false;
    cudaError_t status;
    explicit DeviceScope(int device) : status(cudaGetDevice(&previous)) {
        if (status == cudaSuccess && previous != device) {
            status = cudaSetDevice(device);
            changed = status == cudaSuccess;
        }
    }
    ~DeviceScope() { if (changed) (void) cudaSetDevice(previous); }
};
bool checked(cudaError_t status, const char* what, std::string& err) {
    if (status == cudaSuccess) return true;
    err = std::string("decode head: ") + what + ": " + cudaGetErrorString(status);
    return false;
}
}  // namespace

int32_t decode_head_merge_argmax(const int32_t* global_ids, const float* scores, int count) {
    uint32_t best_key = 0;
    int32_t best_id = 0;
    bool found = false;
    for (int i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, scores + i, sizeof bits);
        const uint32_t magnitude = bits & 0x7fffffffu;
        if (global_ids[i] < 0 || magnitude > 0x7f800000u || bits == 0xff800000u) continue;
        if (magnitude == 0) bits = 0;   // +0 and -0 compare equal in the GPU argmax
        // IEEE binary32 order without host floating-point arithmetic or denormal/fast-math assumptions.
        const uint32_t key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        if (!found || key > best_key || (key == best_key && global_ids[i] < best_id)) {
            best_key = key; best_id = global_ids[i]; found = true;
        }
    }
    return best_id;   // the ordinary argmax returns token zero when no score exceeds -inf
}

DecodeHeadWindow::~DecodeHeadWindow() { reset(); }

void DecodeHeadWindow::reset() noexcept {
    if (device_ < 0) return;
    const DeviceScope device(device_);
    if (device.status != cudaSuccess) return;   // do not release another device's objects after a failed switch
    if (stream_) (void) cudaStreamSynchronize(stream_);
    for (auto graph : graphs_) if (graph) (void) cudaGraphExecDestroy(graph);
    graphs_.clear();
    if (finished_) (void) cudaEventDestroy(finished_);
    if (stream_) (void) cudaStreamDestroy(stream_);
    if (q8_) (void) cudaFree(q8_);
    if (interleaved_) (void) cudaFree(interleaved_);
    if (arg_scratch_) (void) cudaFree(arg_scratch_);
    if (logits_) (void) cudaFree(logits_);
    if (handoff_host_) (void) cudaFreeHost(handoff_host_);
    if (out_host_) (void) cudaFreeHost(out_host_);
    if (scores_host_) (void) cudaFreeHost(scores_host_);
    if (nf_host_) (void) cudaFreeHost(nf_host_);
    head_ = nullptr;
    weights_ = nullptr;
    device_ = -1;
    stream_ = nullptr;
    finished_ = nullptr;
    q8_ = interleaved_ = arg_scratch_ = nullptr;
    logits_ = nullptr;
    handoff_host_ = handoff_source_ = handoff_target_ = nullptr;
    out_host_ = out_device_ = nullptr;
    scores_host_ = scores_device_ = nullptr;
    nf_host_ = nf_device_ = nullptr;
    live_ = false;
    last_t_ = 0;
}

bool DecodeHeadWindow::init(const NativeHead& target, int source_device, int n, int vocab, int max_t,
                            bool use_il, bool use_argmax, bool nan_guard, std::string& err, int row_begin, int row_count) {
    if (head_ || device_ >= 0) { err = "decode head: already initialized"; return false; }
    if (!target.loaded() || target.on_host() || target.device() < 0 || source_device < 0 ||
        target.device() == source_device || n <= 0 || n % 256 || vocab <= 0 ||
        max_t < 1 || max_t > strata::kernels::kVerifyLongT || row_begin < 0 || row_begin >= vocab ||
        row_count < 0 || row_count > vocab - row_begin) {
        err = "decode head: invalid devices, head or window shape";
        return false;
    }
    const DeviceScope device(target.device());
    if (!checked(device.status, "select target device", err)) return false;
    device_ = target.device();
    n_ = n; vocab_ = row_count > 0 ? row_count : vocab - row_begin; max_t_ = max_t;
    row_begin_ = row_begin;
    partial_ = row_begin_ != 0 || vocab_ != vocab;
    use_il_ = use_il; argmax_ = use_argmax; nan_guard_ = nan_guard;
    auto fail = [&]() { reset(); return false; };
    try {
        using namespace strata::kernels;
        if (target.weight_bytes() != native_mmvq_weight_bytes(target.type(), n, vocab)) {
            err = "decode head: the target weight size differs from the requested shape";
            return fail();
        }
        if (partial_ && ((target.type() == 14 && native_q6_k_packed_enabled()) ||
                        (target.type() == 8 && (native_q8_0_packed_enabled() || std::getenv("STRATA_Q8_SM60") != nullptr)))) {
            err = "decode head: row ranges require the original, unpacked head layout";
            return fail();
        }
        weights_ = (const uint8_t*) target.weights() + (size_t) row_begin_ * target.row_bytes();
        row_bytes_ = native_q8_1_bytes(n, 1);
        const size_t activation_bytes = row_bytes_ * (size_t) max_t;
        if (!checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream", err) ||
            !checked(cudaEventCreateWithFlags(&finished_, cudaEventDisableTiming), "create event", err) ||
            !checked(cudaHostAlloc(&handoff_host_, activation_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "allocate handoff", err) ||
            !checked(cudaHostGetDevicePointer(&handoff_target_, handoff_host_, 0), "map target handoff", err)) return fail();
        {
            const DeviceScope source(source_device);
            if (!checked(source.status, "select source device", err) ||
                !checked(cudaHostGetDevicePointer(&handoff_source_, handoff_host_, 0), "map source handoff", err)) return fail();
        }
        std::memset(handoff_host_, 0, activation_bytes);
        if (!checked(cudaMalloc(&q8_, activation_bytes), "allocate Q8 rows", err) ||
            !checked(cudaMalloc(&interleaved_, native_q8_1_il_bytes(n, std::min(max_t, kFusedGrMaxT))), "allocate interleaved rows", err) ||
            !checked(cudaMalloc((void**) &logits_, (size_t) max_t * (size_t) vocab_ * sizeof(float)), "allocate logits", err) ||
            !checked(cudaMalloc(&arg_scratch_, argmax_rows_scratch_bytes(max_t)), "allocate argmax scratch", err) ||
            !checked(cudaMemset(arg_scratch_, 0, argmax_rows_scratch_bytes(max_t)), "clear argmax scratch", err) ||
            !checked(cudaHostAlloc((void**) &out_host_, (size_t) max_t * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable), "allocate picks", err) ||
            !checked(cudaHostGetDevicePointer((void**) &out_device_, out_host_, 0), "map picks", err)) return fail();
        std::memset(out_host_, 0, (size_t) max_t * sizeof(int32_t));
        if (partial_) {
            if (!checked(cudaHostAlloc((void**) &scores_host_, (size_t) max_t * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable), "allocate scores", err) ||
                !checked(cudaHostGetDevicePointer((void**) &scores_device_, scores_host_, 0), "map scores", err)) return fail();
            std::memset(scores_host_, 0, (size_t) max_t * sizeof(float));
        }
        if (nan_guard_) {
            const size_t bytes = (size_t) max_t * kNonfiniteBlocks * sizeof(uint32_t);
            if (!checked(cudaHostAlloc((void**) &nf_host_, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "allocate nonfinite flags", err) ||
                !checked(cudaHostGetDevicePointer((void**) &nf_device_, nf_host_, 0), "map nonfinite flags", err)) return fail();
            std::memset(nf_host_, 0, bytes);
        }
        graphs_.assign((size_t) max_t + 1, nullptr);
        head_ = &target;
        return true;
    } catch (const std::exception& e) {
        err = std::string("decode head: init: ") + e.what();
        return fail();
    }
}

bool DecodeHeadWindow::capture(int t, std::string& err) {
    using namespace strata::kernels;
    if (graphs_[(size_t) t]) return true;
    if (!checked(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin capture", err)) return false;
    bool ok = true;
    try {
        // Only this small activation is read over PCIe. GEMV repeatedly reads its target-local Q8 copy.
        copy_from_mapped((float*) q8_, (const float*) handoff_target_, (int64_t) (row_bytes_ * t / sizeof(float)), stream_);
        for (int h0 = 0; h0 < t; h0 += kFusedGrMaxT) {
            const int ht = std::min(t - h0, kFusedGrMaxT);
            const void* x = (const uint8_t*) q8_ + (size_t) h0 * row_bytes_;
            float* y = logits_ + (size_t) h0 * vocab_;
            if (use_il_ && native_mmvq_il_supported(head_->type(), ht, vocab_)) {
                if (ht >= 2) native_q8_1_interleave(x, interleaved_, n_, ht, stream_);
                native_mmvq_il(head_->type(), weights_, x, interleaved_, y, n_, vocab_, ht, stream_);
            } else native_mmvq(head_->type(), weights_, x, y, n_, vocab_, ht, stream_);
        }
        if (argmax_) argmax_rows(logits_, t, vocab_, arg_scratch_, out_device_, stream_);
        else {
            SamplerParams sp;
            sp.greedy = true; sp.temperature = 0.0f;
            sample_tokens(logits_, t, vocab_, nullptr, 0, sp, out_device_, stream_);
        }
        if (partial_) gather_head_scores(logits_, t, vocab_, out_device_, scores_device_, stream_);
        if (nan_guard_) logits_nonfinite_rows(logits_, t, vocab_, nf_device_, stream_);
    } catch (const std::exception& e) {
        err = std::string("decode head: record: ") + e.what();
        ok = false;
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream_, &graph);
    if (ok) ok = checked(ended, "end capture", err);
    if (ok) ok = checked(cudaGraphInstantiate(&graphs_[(size_t) t], graph, nullptr, nullptr, 0), "instantiate graph", err);
    if (graph) (void) cudaGraphDestroy(graph);
    if (ok) ok = checked(cudaGraphUpload(graphs_[(size_t) t], stream_), "upload graph", err);
    if (!ok && graphs_[(size_t) t]) {
        (void) cudaGraphExecDestroy(graphs_[(size_t) t]);
        graphs_[(size_t) t] = nullptr;
    }
    return ok;
}

bool DecodeHeadWindow::capture_all(std::string& err) {
    if (!head_ || live_) { err = "decode head: capture needs an initialized idle window"; return false; }
    const DeviceScope device(device_);
    if (!checked(device.status, "select capture device", err)) return false;
    for (int t = 1; t <= max_t_; ++t) if (!capture(t, err)) return false;
    return checked(cudaStreamSynchronize(stream_), "finish graph upload", err);
}

bool DecodeHeadWindow::launch(int t, const strata::kernels::SamplerParams* sampler, uint64_t counter, std::string& err,
                              cudaEvent_t producer_ready) {
    if (!head_ || live_ || t < 1 || t > max_t_ || !graphs_[(size_t) t]) {
        err = "decode head: launch needs a captured idle window of the requested size";
        return false;
    }
    if (sampler && (sampler->logit_bias || sampler->penalty_last_n > 0)) {
        err = "decode head: history and logit bias require the ordinary head path";
        return false;
    }
    if (partial_ && sampler && !sampler->greedy && sampler->temperature > 0.0f) {
        err = "decode head: row ranges only support greedy sampling";
        return false;
    }
    const DeviceScope device(device_);
    if (!checked(device.status, "select launch device", err)) return false;
    // Cross-device events do not need P2P. Keep the wait outside the captured graph, on this window's private
    // head stream: a target-card layer stream must remain free to produce subsequent pipeline windows.
    if (producer_ready && !checked(cudaStreamWaitEvent(stream_, producer_ready, 0), "wait for producer", err)) return false;
    if (!checked(cudaGraphLaunch(graphs_[(size_t) t], stream_), "launch graph", err)) return false;
    try {
        if (sampler && !sampler->greedy && sampler->temperature > 0.0f) {
            auto sp = *sampler;
            sp.counter = counter;
            strata::kernels::sample_tokens(logits_, t, vocab_, nullptr, 0, sp, out_device_, stream_);
        }
    } catch (const std::exception& e) {
        (void) cudaStreamSynchronize(stream_);
        err = std::string("decode head: sampling: ") + e.what();
        return false;
    }
    if (!checked(cudaEventRecord(finished_, stream_), "record completion", err)) {
        (void) cudaStreamSynchronize(stream_);
        return false;
    }
    last_t_ = t;
    live_ = true;
    return true;
}

bool DecodeHeadWindow::done(std::string& err) {
    if (!head_ || last_t_ == 0) { err = "decode head: no launched window"; return false; }
    if (!live_) return true;
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    const DeviceScope device(device_);
    if (!checked(device.status, "select completion device", err)) return false;
#endif
    // CUDA event queries accept events from a different current device (Programming Guide, multi-GPU).
    // Avoid two device switches per shard on every host poll. The query still fences mapped results.
    const cudaError_t status = cudaEventQuery(finished_);
    if (status == cudaErrorNotReady) return false;
    if (!checked(status, "head completion", err)) return false;
    live_ = false;
    return true;
}

bool DecodeHeadWindow::copy_logits(int row, float* host) const {
    if (!head_ || live_ || !host || row < 0 || row >= last_t_) return false;
    const DeviceScope device(device_);
    return device.status == cudaSuccess &&
           cudaMemcpy(host, logits_ + (size_t) row * vocab_, (size_t) vocab_ * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace strata::core
