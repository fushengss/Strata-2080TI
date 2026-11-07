#include "strata/kernels/decode_head.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

__global__ void gather_head_scores_kernel(const float* __restrict__ logits, int n_rows, int n_vocab,
                                          const int32_t* __restrict__ ids, float* __restrict__ scores) {
    for (int row = (int) threadIdx.x; row < n_rows; row += (int) blockDim.x) {
        const int id = ids[row];
        assert(id >= 0 && id < n_vocab);
        if (id < 0 || id >= n_vocab) return;   // retain the bounds guard in release builds
        const unsigned bits = __float_as_uint(logits[(size_t) row * (size_t) n_vocab + (size_t) id]);
        scores[row] = __uint_as_float(bits);
    }
}

}  // namespace

void gather_head_scores(const float* logits, int n_rows, int n_vocab, const int32_t* ids,
                        float* scores, void* stream) {
    if (n_rows == 0) return;
    if (n_rows < 0 || n_vocab <= 0 || !logits || !ids || !scores)
        throw std::invalid_argument("gather_head_scores: invalid shape or buffers");
    gather_head_scores_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(logits, n_rows, n_vocab, ids, scores);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("gather_head_scores: ") + cudaGetErrorString(status));
}

}  // namespace strata::kernels
