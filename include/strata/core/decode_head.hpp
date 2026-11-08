#pragma once

#include "strata/kernels/sampler.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// Merge shard-local argmax results after converting their ids to global vocabulary ids. Scores are unchanged
/// logits, not probabilities. NaN/-inf provide no candidate; equal values (including +/-0) choose the lowest id.
int32_t decode_head_merge_argmax(const int32_t* global_ids, const float* scores, int count);

/// One verifier parity's decode-only head on another GPU. The shared NativeHead must outlive this object.
/// The producer writes exact Q8_1 rows to handoff(), then finishes its stream or records producer_ready. No P2P is required.
/// Sampling with a history or logit bias stays on the caller's ordinary path.
class DecodeHeadWindow {
public:
    DecodeHeadWindow() = default;
    ~DecodeHeadWindow();
    DecodeHeadWindow(const DecodeHeadWindow&) = delete;
    DecodeHeadWindow& operator=(const DecodeHeadWindow&) = delete;

    bool init(const NativeHead& target, int source_device, int n, int vocab, int max_t,
              bool use_il, bool use_argmax, bool nan_guard, std::string& err, int row_begin = 0, int row_count = 0,
              bool return_logits = false);
    void* handoff() const { return handoff_source_; }
    size_t row_bytes() const { return row_bytes_; }
    bool capture_all(std::string& err);
    /// A null sampler keeps the graph's greedy pick. Otherwise only history-free, bias-free sampling is accepted.
    /// producer_ready, when supplied, must already be recorded after the producer's handoff writes.
    bool launch(int t, const strata::kernels::SamplerParams* sampler, uint64_t counter, std::string& err,
                cudaEvent_t producer_ready = nullptr);
    /// Full-logit mode only: launch the projection without any target-side selection or sampling. Launch every
    /// shard before enqueue_return() so the projections run concurrently. The producer event is already recorded.
    bool launch_logits(int t, cudaEvent_t producer_ready, std::string& err);
    /// Full-logit mode only: source waits for this shard, then bit-copies its rows into [T][full_vocab] at row_begin().
    /// The source stream must outlive this workspace. done() fences this copy, not just the target projection;
    /// the caller must also finish its own source tail before overwriting/reusing the destination or this handoff.
    bool enqueue_return(cudaStream_t source_stream, float* source_logits, int full_vocab, std::string& err);
    /// False while in flight; false with err on a CUDA failure. Results remain valid until the next launch.
    bool done(std::string& err);
    const int32_t* picks() const { return out_host_; }
    /// Partial heads return local ids and their original logits; full heads do not allocate scores.
    const float* scores() const { return scores_host_; }
    int row_begin() const { return row_begin_; }
    int row_count() const { return vocab_; }
    /// T * kNonfiniteBlocks words, or nullptr when the guard is disabled.
    const uint32_t* nonfinite() const { return nf_host_; }
    bool copy_logits(int row, float* host) const;

private:
    void reset() noexcept;
    bool capture(int t, std::string& err);
    const NativeHead* head_ = nullptr;
    const void* weights_ = nullptr;
    int device_ = -1, source_device_ = -1, n_ = 0, vocab_ = 0, max_t_ = 0, last_t_ = 0;
    int row_begin_ = 0;
    bool partial_ = false;
    bool use_il_ = false, argmax_ = false, nan_guard_ = false, live_ = false;
    bool return_logits_ = false, return_enqueued_ = false;
    size_t row_bytes_ = 0;
    void *handoff_host_ = nullptr, *handoff_source_ = nullptr, *handoff_target_ = nullptr;
    void *q8_ = nullptr, *interleaved_ = nullptr, *arg_scratch_ = nullptr;
    float* logits_ = nullptr;
    float *return_host_ = nullptr, *return_source_ = nullptr, *return_target_ = nullptr;
    int32_t *out_host_ = nullptr, *out_device_ = nullptr;
    float *scores_host_ = nullptr, *scores_device_ = nullptr;
    uint32_t *nf_host_ = nullptr, *nf_device_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t finished_ = nullptr;
    cudaEvent_t returned_ = nullptr;  // source-device event after the full-logit copy
    cudaStream_t last_source_stream_ = nullptr;
    std::vector<cudaGraphExec_t> graphs_;
    std::vector<bool> source_il_;  // full-vocabulary source choice, by chunk T; full-logit mode only
};

}  // namespace strata::core
