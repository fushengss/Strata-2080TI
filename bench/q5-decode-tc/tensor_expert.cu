#include "tensor_expert.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace q5_decode_tc {
namespace {

using strata::kernels::NativeExpertLayout;

constexpr int kQ5K = 13;
constexpr int kQ5One = 7;
constexpr int kQ8Zero = 8;
constexpr int kRowsPerWarp = 16;
constexpr int kTokensPerWarp = 8;
constexpr int kTilesPerBlock = 2;
constexpr int kSplitK = 2;
constexpr int kRowsPerBlock = kTilesPerBlock * kRowsPerWarp;
constexpr int kThreads = 32 * kTilesPerBlock * kSplitK;

__device__ __forceinline__ uint32_t pack_halves(float first, float second) {
    const __half2_raw pair = __floats2half2_rn(first, second);
    return uint32_t(pair.x) | (uint32_t(pair.y) << 16);
}

__device__ __forceinline__ uint32_t load_pair(const void* source) {
    return *static_cast<const uint16_t*>(source);
}

__device__ __forceinline__ uint32_t half2_bits(__half2 value) {
    const __half2_raw raw = value;
    return uint32_t(raw.x) | (uint32_t(raw.y) << 16);
}

__device__ __forceinline__ __half2 half2_from_bits(uint32_t value) {
    const __half2_raw raw{static_cast<uint16_t>(value), static_cast<uint16_t>(value >> 16)};
    return __half2(raw);
}

template<bool Packed>
__device__ __forceinline__ auto replicate_scale(float value) {
    if constexpr (Packed) return __float2half2_rn(value);
    else return value;
}

template<bool Packed>
__device__ __forceinline__ auto replicate_scale(__half value) {
    if constexpr (Packed) return __halves2half2(value, value);
    else return __half2float(value);
}

__device__ __forceinline__ uint32_t scaled_signed_pair(const void* codes, float delta) {
    const uint32_t pair = load_pair(codes);
    return pack_halves(float(int8_t(pair)) * delta, float(int8_t(pair >> 8)) * delta);
}

__device__ __forceinline__ uint32_t scaled_signed_pair(const void* codes, __half2 delta) {
    const uint32_t biased = __byte_perm(load_pair(codes) ^ 0x8080, 0x64646464, 0x4140);
    const __half2 integers = __hsub2(half2_from_bits(biased), half2_from_bits(0x64806480));
    return half2_bits(__hmul2(integers, delta));
}

__device__ __forceinline__ uint32_t affine_unsigned_pair(uint32_t codes, float delta, float offset) {
    return pack_halves(fmaf(delta, float(codes & 31), offset),
                       fmaf(delta, float((codes >> 8) & 31), offset));
}

__device__ __forceinline__ uint32_t affine_unsigned_pair(uint32_t codes, __half2 delta, __half2 offset) {
    // A half bit pattern 0x6400 | n represents 1024+n exactly. Insert both
    // bytes at once, then remove the bias in FP16 before the affine transform.
    const uint32_t biased = __byte_perm(codes, 0x64646464, 0x4140);
    const __half2 integers = __hsub2(half2_from_bits(biased), half2_from_bits(0x64006400));
    return half2_bits(__hfma2(integers, delta, offset));
}

template<int Type, bool Packed>
__device__ __forceinline__ void weight_fragments(const uint8_t* weight_row, int block32, int pair_index,
                                                uint32_t (&fragments)[4]) {
    if constexpr (Type == kQ5K) {
        const block_q5_K& block = reinterpret_cast<const block_q5_K*>(weight_row)[block32 / 8];
        const int subblock = block32 % 8;
        const int shift = 4 * (subblock & 1);
        const uint8_t* scales = block.scales;
        const int scale = subblock < 4 ? scales[subblock] & 63
                                      : (scales[subblock + 4] & 15) | ((scales[subblock - 4] >> 6) << 4);
        const int minimum = subblock < 4 ? scales[subblock + 4] & 63
                                        : (scales[subblock + 4] >> 4) | ((scales[subblock] >> 6) << 4);
        // The packed variant rounds each Q5_K block's reconstructed scale and
        // offset to FP16 before its fused affine transform. The reference
        // variant keeps them in FP32 until the decoded weights are packed.
        const auto delta = replicate_scale<Packed>(__low2float(block.dm) * scale);
        const auto offset = replicate_scale<Packed>(-__high2float(block.dm) * minimum);
        const uint8_t* low = block.qs + (subblock / 2) * 32;
        const uint8_t* high = block.qh;
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int index = step * 8 + pair_index;
            const uint32_t codes = ((load_pair(low + index) >> shift) & 0x0f0f)
                                   | (((load_pair(high + index) >> subblock) & 0x0101) << 4);
            fragments[step] = affine_unsigned_pair(codes, delta, offset);
        }
    } else if constexpr (Type == kQ5One) {
        const block_q5_1& block = reinterpret_cast<const block_q5_1*>(weight_row)[block32];
        const uint32_t high = load_pair(block.qh) | (load_pair(block.qh + 2) << 16);
        const auto delta = replicate_scale<Packed>(__low2half(block.dm));
        const auto offset = replicate_scale<Packed>(__high2half(block.dm));
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int index = step * 8 + pair_index;
            const uint32_t low = (load_pair(block.qs + (index & 15)) >> (4 * (step / 2))) & 0x0f0f;
            const uint32_t high_pair = high >> index;
            const uint32_t first = (low & 15) | ((high_pair & 1) << 4);
            const uint32_t second = ((low >> 8) & 15) | ((high_pair & 2) << 3);
            fragments[step] = affine_unsigned_pair(first | (second << 8), delta, offset);
        }
    } else {
        const block_q8_0& block = reinterpret_cast<const block_q8_0*>(weight_row)[block32];
        const auto delta = replicate_scale<Packed>(block.d);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            fragments[step] = scaled_signed_pair(block.qs + step * 8 + pair_index, delta);
        }
    }
}

__device__ __forceinline__ void mma(float (&acc)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                 : "r"(a0), "r"(a1), "r"(b));
#else
    __trap();
#endif
}

template<int Type, bool GateUp, bool Packed>
__global__ void __launch_bounds__(kThreads) expert_matmul(
    NativeExpertLayout layout, const unsigned long long* __restrict__ group_weights,
    const int32_t* __restrict__ group_starts, const int32_t* __restrict__ group_count,
    const int32_t* __restrict__ entry_indices, const block_q8_1* __restrict__ activations,
    float* __restrict__ gate_or_down, float* __restrict__ up) {
    const int group = blockIdx.y;
    if (group >= *group_count) return;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile = warp / kSplitK;
    const int k_part = warp % kSplitK;
    // SM75 m16n8k8: A's two registers hold rows g and g+8, B is token g,
    // and each lane supplies the K pair 2*t. C uses output columns 2*t and 2*t+1.
    const int g = lane / 4;
    const int pair_index = (lane % 4) * 2;
    const int row = blockIdx.x * kRowsPerBlock + tile * kRowsPerWarp + g;
    const bool is_up = GateUp && row >= layout.n_ff;
    const int output_row = is_up ? row - int(layout.n_ff) : row;
    const auto* blob = reinterpret_cast<const uint8_t*>(group_weights[group]);
    const size_t row_bytes = GateUp ? layout.gu_row : layout.d_row;
    const size_t offset = GateUp ? (is_up ? layout.up_off : 0) : layout.down_off;
    const auto* row0 = blob + offset + size_t(output_row) * row_bytes;
    const auto* row1 = row0 + 8 * row_bytes;
    const int inner_blocks = int(GateUp ? layout.n_embd : layout.n_ff) / 32;
    const int first_block = k_part * (inner_blocks / kSplitK);
    const int last_block = first_block + inner_blocks / kSplitK;
    const int start_entry = group_starts[group];
    const int end_entry = group_starts[group + 1];
    __shared__ float second_part[kTilesPerBlock][4][32];

    for (int first_entry = start_entry; first_entry < end_entry; first_entry += kTokensPerWarp) {
        const int input_entry = first_entry + g;
        const block_q8_1* input = nullptr;
        if (input_entry < end_entry) {
            const int input_row = GateUp ? entry_indices[input_entry] : input_entry;
            input = activations + size_t(input_row) * inner_blocks;
        }
        float acc_even[4] = {};
        float acc_odd[4] = {};
        for (int block = first_block; block < last_block; ++block) {
            uint32_t a0[4], a1[4], b[4] = {};
            weight_fragments<Type, Packed>(row0, block, pair_index, a0);
            weight_fragments<Type, Packed>(row1, block, pair_index, a1);
            if (input != nullptr) {
                const auto delta = replicate_scale<Packed>(__low2half(input[block].ds));
#pragma unroll
                for (int step = 0; step < 4; ++step) {
                    b[step] = scaled_signed_pair(input[block].qs + step * 8 + pair_index, delta);
                }
            }
            mma(acc_even, a0[0], a1[0], b[0]);
            mma(acc_odd, a0[1], a1[1], b[1]);
            mma(acc_even, a0[2], a1[2], b[2]);
            mma(acc_odd, a0[3], a1[3], b[3]);
        }
        if (k_part != 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) second_part[tile][i][lane] = acc_even[i] + acc_odd[i];
        }
        __syncthreads();
        if (k_part == 0) {
            const int output_stride = int(GateUp ? layout.n_ff : layout.n_embd);
            float* output = is_up ? up : gate_or_down;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int entry = first_entry + pair_index + (i & 1);
                if (entry < end_entry) {
                    const int target = GateUp ? entry : entry_indices[entry];
                    const int column = output_row + 8 * (i / 2);
                    output[size_t(target) * output_stride + column] =
                        (acc_even[i] + acc_odd[i]) + second_part[tile][i][lane];
                }
            }
        }
        if (first_entry + kTokensPerWarp < end_entry) __syncthreads();
    }
}

__global__ void swiglu_q8_1(const float* gate, const float* up, const int32_t* group_starts,
                           const int32_t* group_count, int width, block_q8_1* q8) {
    using namespace strata::kernels;
    const size_t first = size_t(group_starts[0]) * width;
    const size_t last = size_t(group_starts[*group_count]) * width;
    const size_t i = first + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= last) return;
    const float g = gate[i];
    const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
    float amax = fabsf(h);
    float sum = h;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    const float delta = q8_1_finite(amax / 127.0f);
    const int lane = threadIdx.x % 32;
    q8[i / 32].qs[lane] = q8_1_quant(h, delta, amax);
    if (lane == 0) q8[i / 32].ds = q8_1_ds(delta, sum);
}

void check_launch() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("Q5 Tensor Core expert launch: ") + cudaGetErrorString(status));
    }
}

template<bool Packed>
void launch_variant(const NativeExpertLayout& layout, const unsigned long long* group_weights,
                    const int32_t* group_starts, const int32_t* group_count, const int32_t* entry_destinations,
                    const int32_t* entry_tokens, int64_t group_capacity, int64_t entry_capacity,
                    const void* activations_q8_1, void* scratch, float* output, void* stream, int phase) {
    const auto queue = static_cast<cudaStream_t>(stream);
    const size_t hidden_floats = size_t(entry_capacity) * size_t(layout.n_ff);
    const size_t hidden_bytes = (hidden_floats * sizeof(float) + 255) & ~size_t(255);
    auto* memory = static_cast<uint8_t*>(scratch);
    auto* gate = reinterpret_cast<float*>(memory);
    auto* up = reinterpret_cast<float*>(memory + hidden_bytes);
    auto* hq = reinterpret_cast<block_q8_1*>(memory + 3 * hidden_bytes);
    if (phase != 2) {
        const dim3 grid(unsigned(2 * layout.n_ff / kRowsPerBlock), unsigned(group_capacity));
        expert_matmul<kQ5K, true, Packed><<<grid, kThreads, 0, queue>>>(
            layout, group_weights, group_starts, group_count, entry_tokens,
            static_cast<const block_q8_1*>(activations_q8_1), gate, up);
        check_launch();
        if (phase == 3) return;
        swiglu_q8_1<<<unsigned((hidden_floats + 255) / 256), 256, 0, queue>>>(
            gate, up, group_starts, group_count, int(layout.n_ff), hq);
        check_launch();
        if (phase == 1) return;
    }
    const dim3 grid(unsigned(layout.n_embd / kRowsPerBlock), unsigned(group_capacity));
    if (layout.d_type == kQ5One) {
        expert_matmul<kQ5One, false, Packed><<<grid, kThreads, 0, queue>>>(
            layout, group_weights, group_starts, group_count, entry_destinations, hq, output, nullptr);
    } else {
        expert_matmul<kQ8Zero, false, Packed><<<grid, kThreads, 0, queue>>>(
            layout, group_weights, group_starts, group_count, entry_destinations, hq, output, nullptr);
    }
    check_launch();
}

}  // namespace

void launch(const NativeExpertLayout& layout, const unsigned long long* group_weights,
            const int32_t* group_starts, const int32_t* group_count, const int32_t* entry_destinations,
            const int32_t* entry_tokens, int64_t group_capacity, int64_t entry_capacity,
            const void* activations_q8_1, void* scratch, float* output, void* stream, int phase) {
    if (group_capacity <= 0 || entry_capacity <= 0) return;
    if (layout.gu_type != kQ5K || (layout.d_type != kQ5One && layout.d_type != kQ8Zero) ||
        layout.n_embd != 2560 || layout.n_ff != 640 || layout.gu_nib || layout.had_block || layout.tail_off ||
        phase < 0 || phase > 3) {
        throw std::invalid_argument("Q5 Tensor Core expert benchmark requires Flash-Next Q5_K/Q5_1 or Q5_K/Q8_0");
    }
    static const auto variant = [] {
        const char* flag = std::getenv("Q5_TC_HALF2");
        return flag && flag[0] == '1' && flag[1] == '\0' ? &launch_variant<true> : &launch_variant<false>;
    }();
    variant(layout, group_weights, group_starts, group_count, entry_destinations, entry_tokens, group_capacity,
            entry_capacity, activations_q8_1, scratch, output, stream, phase);
}

}  // namespace q5_decode_tc
