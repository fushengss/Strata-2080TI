#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <string>

namespace strata::core {

/// Two row shards of a plain Q6_K draft head. No probability or token reduction runs on the target cards:
/// the complete logits return in their original row order, for the caller's unchanged source-card tail.
/// init() owns a copy of each half of source_weights. The source stream passed to enqueue() must outlive reset().
/// A chain may enqueue steps 0..7 without a host wait. Enqueue the source tail before the next body/enqueue;
/// that source-stream ordering protects the shared activation/logits buffers. Drain before reusing a step index.
class MtpHeadShards {
public:
    MtpHeadShards() = default;
    ~MtpHeadShards();
    MtpHeadShards(const MtpHeadShards&) = delete;
    MtpHeadShards& operator=(const MtpHeadShards&) = delete;

    bool init(const void* source_weights, int source_device, int type, int n, int vocab,
              int target0, int target1, std::string& err);
    /// Copies exact Q8_1 bytes, records source-ready, queues both head graphs, then source waits and a full-logit copy.
    /// Does not synchronize or poll. The caller can immediately enqueue the original argmax/probability tail.
    bool enqueue(cudaStream_t source_stream, const void* source_q8, float* source_logits, int step, std::string& err);
    bool drain(std::string& err);
    void reset() noexcept;
    bool initialized() const { return ready_; }
    size_t row_bytes() const { return q8_bytes_; }
    int row_begin(int shard) const { return shard >= 0 && shard < 2 ? shards_[shard].begin : -1; }
    int row_count(int shard) const { return shard >= 0 && shard < 2 ? shards_[shard].count : 0; }

private:
    struct Shard {
        int device = -1, begin = 0, count = 0;
        void *weights = nullptr, *q8 = nullptr, *activation = nullptr;
        float *logits = nullptr, *mapped_logits = nullptr;
        cudaStream_t stream = nullptr;
        cudaGraphExec_t graph = nullptr;
        cudaEvent_t finished[8] = {};
    } shards_[2];
    int source_ = -1, n_ = 0, vocab_ = 0;
    size_t q8_bytes_ = 0;
    bool ready_ = false;
    void *activation_host_ = nullptr, *activation_source_ = nullptr;
    float *logits_host_ = nullptr, *logits_source_ = nullptr;
    cudaEvent_t producer_[8] = {};
    cudaStream_t last_source_stream_ = nullptr;
};

}  // namespace strata::core
