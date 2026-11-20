#pragma once

#include "strata/kernels/iq_kernels.hpp"

namespace q5_decode_tc {

// Standalone SM75 benchmark using raw Q5_K gate/up and Q5_1/Q8_0 down weights.
// The plan and scratch match native_expert_grouped. Phases are 0 full, 1 gate/up
// with SwiGLU and Q8_1 quantization, 2 down from prepared scratch, 3 gate/up only.
void launch(const strata::kernels::NativeExpertLayout& layout, const unsigned long long* group_weights,
            const int32_t* group_starts, const int32_t* group_count, const int32_t* entry_destinations,
            const int32_t* entry_tokens, int64_t group_capacity, int64_t entry_capacity,
            const void* activations_q8_1, void* scratch, float* output, void* stream, int phase);

}  // namespace q5_decode_tc
