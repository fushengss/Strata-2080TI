// Metal keeps its existing output-head path. The shared verifier still references this CUDA-only experiment
// when it is disabled, so provide its host interface without linking CUDA graphs or device kernels.
#include "strata/core/decode_head.hpp"

#include <cstring>

namespace strata::core {
namespace {
bool unsupported(std::string& err) {
    err = "decode head: this experiment requires CUDA and is unavailable on Metal";
    return false;
}
}  // namespace

DecodeHeadWindow::~DecodeHeadWindow() = default;

bool DecodeHeadWindow::init(const NativeHead&, int, int, int, int, bool, bool, bool, std::string& err, int, int) {
    return unsupported(err);
}
bool DecodeHeadWindow::capture_all(std::string& err) { return unsupported(err); }
bool DecodeHeadWindow::launch(int, const strata::kernels::SamplerParams*, uint64_t, std::string& err, cudaEvent_t) {
    return unsupported(err);
}
bool DecodeHeadWindow::done(std::string& err) { return unsupported(err); }
bool DecodeHeadWindow::copy_logits(int, float*) const { return false; }

// Keep the pure comparison identical to src/core/decode_head.cpp. It has no GPU dependency, and giving it the
// same behavior avoids a silent wrong result if a future host-only caller uses it outside the disabled path.
int32_t decode_head_merge_argmax(const int32_t* global_ids, const float* scores, int count) {
    uint32_t best_key = 0;
    int32_t best_id = 0;
    bool found = false;
    for (int i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, scores + i, sizeof bits);
        const uint32_t magnitude = bits & 0x7fffffffu;
        if (global_ids[i] < 0 || magnitude > 0x7f800000u || bits == 0xff800000u) continue;
        if (magnitude == 0) bits = 0;
        const uint32_t key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        if (!found || key > best_key || (key == best_key && global_ids[i] < best_id)) {
            best_key = key; best_id = global_ids[i]; found = true;
        }
    }
    return best_id;
}

}  // namespace strata::core
