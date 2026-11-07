#pragma once

#include <cstdint>

namespace strata::kernels {

/// Copy logits[row][ids[row]] to scores[row], preserving every float bit.
/// IDs must be local to the supplied vocabulary. One 32-thread block serves all rows.
/// The producer of ids and this gather must be ordered on the supplied stream.
void gather_head_scores(const float* logits, int n_rows, int n_vocab, const int32_t* ids,
                        float* scores, void* stream);

}  // namespace strata::kernels
