// Manual two-GPU integration test; requires the model's GGUF, not a running engine.
// decode_head_test --gguf MODEL.gguf --source 7 --target 0 [--target2 4] [--cycles 4]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/decode_head.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void gpu(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
void current_is(int wanted) { int d = -1; gpu(cudaGetDevice(&d)); require(d == wanted, "helper changed the caller's CUDA device"); }
struct HeadOwner {
    strata::core::NativeHead head;
    ~HeadOwner() { const strata::core::OnDevice on(head.device()); head.unload(); }
};
struct SourceBuffers {
    int device;
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    float *mixed = nullptr, *logits = nullptr;
    void *q8 = nullptr, *il = nullptr, *scratch = nullptr;
    int32_t* picks = nullptr;
    uint32_t* nf = nullptr;
    explicit SourceBuffers(int d) : device(d) {}
    ~SourceBuffers() {
        const strata::core::OnDevice on(device);
        if (stream) cudaStreamSynchronize(stream);
        if (mixed) cudaFree(mixed);
        if (logits) cudaFree(logits);
        if (q8) cudaFree(q8);
        if (il) cudaFree(il);
        if (scratch) cudaFree(scratch);
        if (picks) cudaFree(picks);
        if (nf) cudaFree(nf);
        if (ready) cudaEventDestroy(ready);
        if (stream) cudaStreamDestroy(stream);
    }
};
struct Expected {
    std::vector<float> logits;
    std::vector<int32_t> picks;
    std::vector<uint32_t> nf;
};
float from_bits(uint32_t bits) { float f; std::memcpy(&f, &bits, sizeof f); return f; }
void test_merge() {
    const int32_t ids[2] = {41, 7};
    auto check = [&](uint32_t a, uint32_t b, int32_t want) {
        const float scores[2] = {from_bits(a), from_bits(b)};
        require(strata::core::decode_head_merge_argmax(ids, scores, 2) == want, "shard argmax merge edge case");
    };
    check(0x3f800000, 0x3f800000, 7);   // equal finite values: smallest global id
    check(0x00000000, 0x80000000, 7);   // +0 == -0
    check(0x80000000, 0x00000000, 7);
    check(0x7f800000, 0x7f800000, 7);   // +inf ties are still valid argmax candidates
    check(0x3f800000, 0x7f800000, 7);
    check(0x7fc00000, 0xbf800000, 7);   // NaN is ignored, including negative NaN
    check(0x3f800000, 0xffc00000, 41);
    check(0xff800000, 0x7fc00000, 0);   // no value above -inf: original argmax returns zero
    check(0xff800000, 0xff800000, 0);
    check(0x00000001, 0x00000000, 41);  // positive subnormal beats zero without host FTZ arithmetic
    check(0x80000001, 0x80000002, 41);  // negative subnormals retain their IEEE order
}

size_t test_shards(const strata::core::NativeHead& original, const strata::core::NativeHead& first,
                   const strata::core::NativeHead& second, SourceBuffers& b, int n, int vocab, int cycles) {
    using namespace strata::kernels;
    const int cut = vocab / 2 / 128 * 128;
    require(cut > 0 && cut < vocab, "vocabulary too small for two aligned row shards");
    const size_t row_bytes = native_q8_1_bytes(n, 1);
    std::string err;
    strata::core::DecodeHeadWindow halves[2];
    require(halves[0].init(first, b.device, n, vocab, 8, true, true, true, err, 0, cut), err);
    require(halves[1].init(second, b.device, n, vocab, 8, true, true, true, err, cut, 0), err);
    for (auto& half : halves) {
        require(half.capture_all(err), err);
        SamplerParams sampled;
        err.clear(); require(!half.launch(1, &sampled, 0, err) && !err.empty(), "sharded sampling was not rejected");
        err.clear();
    }
    size_t cases = 0;
    for (int t : {1, 2, 3, 8}) for (int cycle = 0; cycle < cycles; ++cycle) {
        const bool inject_nf = cycle + 1 == cycles;
        std::vector<float> mixed((size_t) t * n, 0.0f);
        uint32_t state = 0x56781234u + cycle * 997u + t;
        if (cycle > 0) for (float& f : mixed) {
            state = state * 1664525u + 1013904223u;
            f = (float) ((int) (state >> 16) - 32768) / 8192.0f;
        }   // cycle zero exercises a full-vocabulary tie
        Expected expected;
        expected.logits.resize((size_t) t * vocab); expected.picks.resize(t); expected.nf.resize(t * kNonfiniteBlocks);
        gpu(cudaMemcpyAsync(b.mixed, mixed.data(), mixed.size() * sizeof(float), cudaMemcpyHostToDevice, b.stream));
        native_quantize_q8_1(b.mixed, b.q8, n, t, b.stream);
        const uint16_t half_nan = 0x7e00;
        if (inject_nf) gpu(cudaMemcpyAsync((uint8_t*) b.q8 + (size_t) (t - 1) * row_bytes,
                                           &half_nan, sizeof half_nan, cudaMemcpyHostToDevice, b.stream));
        if (native_mmvq_il_supported(original.type(), t, vocab)) {
            if (t >= 2) native_q8_1_interleave(b.q8, b.il, n, t, b.stream);
            native_mmvq_il(original.type(), original.weights(), b.q8, b.il, b.logits, n, vocab, t, b.stream);
        } else native_mmvq(original.type(), original.weights(), b.q8, b.logits, n, vocab, t, b.stream);
        argmax_rows(b.logits, t, vocab, b.scratch, b.picks, b.stream);
        logits_nonfinite_rows(b.logits, t, vocab, b.nf, b.stream);
        for (auto& half : halves)
            copy_from_mapped((float*) half.handoff(), (const float*) b.q8, (int64_t) t * row_bytes / sizeof(float), b.stream);
        gpu(cudaEventRecord(b.ready, b.stream));
        // Both target streams wait on the same producer event, before any reference D2H copy can block the host.
        for (auto& half : halves) require(half.launch(t, nullptr, 0, err, b.ready), err);
        gpu(cudaMemcpyAsync(expected.logits.data(), b.logits, expected.logits.size() * sizeof(float), cudaMemcpyDeviceToHost, b.stream));
        gpu(cudaMemcpyAsync(expected.picks.data(), b.picks, t * sizeof(int32_t), cudaMemcpyDeviceToHost, b.stream));
        gpu(cudaMemcpyAsync(expected.nf.data(), b.nf, expected.nf.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, b.stream));
        gpu(cudaStreamSynchronize(b.stream));
        for (auto& half : halves) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
            while (!half.done(err)) {
                require(err.empty(), err);
                require(std::chrono::steady_clock::now() < deadline, "shard timed out");
                std::this_thread::yield();
            }
            std::vector<float> got(half.row_count());
            for (int r = 0; r < t; ++r) {
                require(half.copy_logits(r, got.data()), "shard logits copy failed");
                const float* want = expected.logits.data() + (size_t) r * vocab + half.row_begin();
                require(std::memcmp(got.data(), want, got.size() * sizeof(float)) == 0, "sharded logits differ from full logits");
                require(std::memcmp(half.scores() + r, want + half.picks()[r], sizeof(float)) == 0, "shard score is not its selected logit");
            }
        }
        for (int r = 0; r < t; ++r) {
            const int32_t ids[2] = {halves[0].row_begin() + halves[0].picks()[r], halves[1].row_begin() + halves[1].picks()[r]};
            const float scores[2] = {halves[0].scores()[r], halves[1].scores()[r]};
            require(strata::core::decode_head_merge_argmax(ids, scores, 2) == expected.picks[r], "merged shard pick differs");
            bool have_nf = false, want_nf = false;
            for (int k = 0; k < kNonfiniteBlocks; ++k) {
                have_nf |= halves[0].nonfinite()[r * kNonfiniteBlocks + k] != 0 || halves[1].nonfinite()[r * kNonfiniteBlocks + k] != 0;
                want_nf |= expected.nf[r * kNonfiniteBlocks + k] != 0;
            }
            require(have_nf == want_nf, "sharded nonfinite row guard differs");
        }
        current_is(b.device);
        ++cases;
    }
    return cases;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        test_merge();
        if (argc == 2 && std::strcmp(argv[1], "--merge-only") == 0) { std::puts("PASS: CPU shard-argmax merge edge cases"); return 0; }
        std::string gguf;
        int source = 1, target = 0, target2 = -1, cycles = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            require(i + 1 < argc, "missing value for " + a);
            if (a == "--gguf") gguf = argv[++i];
            else if (a == "--source") source = std::stoi(argv[++i]);
            else if (a == "--target") target = std::stoi(argv[++i]);
            else if (a == "--target2") target2 = std::stoi(argv[++i]);
            else if (a == "--cycles") cycles = std::stoi(argv[++i]);
            else throw std::runtime_error("unknown argument: " + a);
        }
        if (target2 < 0) target2 = target;
        require(!gguf.empty() && source >= 0 && target >= 0 && source != target && source != target2 && cycles >= 3,
                "usage: decode_head_test --gguf MODEL.gguf --source N --target N [--target2 N] [--cycles 4, at least 3]");
        int devices = 0;
        const cudaError_t count = cudaGetDeviceCount(&devices);
        if (count != cudaSuccess || source >= devices || target >= devices || target2 >= devices) {
            std::fprintf(stderr, "SKIP: requested two GPUs unavailable (count=%d, CUDA=%s)\n", devices, cudaGetErrorString(count));
            return 77;
        }
        const auto paths = strata::gguf_split_paths(gguf);
        int n = 0, vocab = 0;
        {
            const strata::GgufModel model(paths);
            const auto* tensor = model.find("output.weight");
            require(tensor && tensor->shape.size() == 2 && tensor->shape[0] <= INT_MAX && tensor->shape[1] <= INT_MAX,
                    "invalid output.weight shape");
            n = (int) tensor->shape[0]; vocab = (int) tensor->shape[1];
        }
        using namespace strata::kernels;
        std::string err;
        HeadOwner original, replica, replica2;
        gpu(cudaSetDevice(source));
        require(original.head.load(paths, n, vocab, err), err);
        require(!original.head.on_host(), "source head spilled to host");
        gpu(cudaSetDevice(target));
        require(replica.head.load(paths, n, vocab, err), err);
        require(!replica.head.on_host(), "target head spilled to host");
        if (target2 != target) {
            gpu(cudaSetDevice(target2));
            require(replica2.head.load(paths, n, vocab, err), err);
            require(!replica2.head.on_host(), "second target head spilled to host");
        }
        gpu(cudaSetDevice(source));
        constexpr int max_t = 8;
        const size_t row_bytes = native_q8_1_bytes(n, 1);
        SourceBuffers b(source);
        gpu(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
        gpu(cudaEventCreateWithFlags(&b.ready, cudaEventDisableTiming));
        gpu(cudaMalloc((void**) &b.mixed, (size_t) max_t * n * sizeof(float)));
        gpu(cudaMalloc((void**) &b.logits, (size_t) max_t * vocab * sizeof(float)));
        gpu(cudaMalloc(&b.q8, max_t * row_bytes));
        gpu(cudaMalloc(&b.il, native_q8_1_il_bytes(n, max_t)));
        gpu(cudaMalloc(&b.scratch, argmax_rows_scratch_bytes(max_t)));
        gpu(cudaMemset(b.scratch, 0, argmax_rows_scratch_bytes(max_t)));
        gpu(cudaMalloc((void**) &b.picks, max_t * sizeof(int32_t)));
        gpu(cudaMalloc((void**) &b.nf, max_t * kNonfiniteBlocks * sizeof(uint32_t)));
        size_t cases = 0, compared_logits = 0;
        for (bool il : {false, true}) for (bool argmax : {false, true}) {
            // Two independent parity windows run on separate remote streams and reuse the same read-only head.
            strata::core::DecodeHeadWindow windows[2];
            for (auto& window : windows) {
                require(window.init(replica.head, source, n, vocab, max_t, il, argmax, true, err), err);
                require(window.capture_all(err), err);
                current_is(source);
                SamplerParams bad; bad.penalty_last_n = 8;
                err.clear(); require(!window.launch(1, &bad, 0, err) && !err.empty(), "history was not rejected");
                bad.penalty_last_n = 0; bad.logit_bias = (const float*) &bad;
                err.clear(); require(!window.launch(1, &bad, 0, err) && !err.empty(), "bias was not rejected");
                err.clear(); require(!window.launch(0, nullptr, 0, err) && !err.empty(), "invalid T was not rejected");
                err.clear();
            }
            for (int t : {1, 2, 3, 8}) for (int cycle = 0; cycle < cycles; ++cycle) {
                Expected expected[2];
                const bool inject_nf = cycle + 1 == cycles;
                const bool sampled = !inject_nf && (cycle % 2 != 0);
                SamplerParams sp;
                sp.greedy = !sampled; sp.temperature = sampled ? 0.7f : 0.0f;
                sp.top_k = 20; sp.top_p = 0.9f; sp.seed = 0x68a0135;
                for (int parity = 0; parity < 2; ++parity) {
                    Expected& e = expected[parity];
                    e.logits.resize((size_t) t * vocab); e.picks.resize(t); e.nf.resize(t * kNonfiniteBlocks);
                    std::vector<float> mixed((size_t) t * n);
                    uint32_t state = 0x12435678u + 131u * cycle + 17u * parity + t;
                    for (float& f : mixed) { state = state * 1664525u + 1013904223u; f = (float) ((int) (state >> 16) - 32768) / 8192.0f; }
                    gpu(cudaMemcpyAsync(b.mixed, mixed.data(), mixed.size() * sizeof(float), cudaMemcpyHostToDevice, b.stream));
                    native_quantize_q8_1(b.mixed, b.q8, n, t, b.stream);
                    // Exercise nonfinite reporting without quantizing an invalid float input: poison one Q8 scale.
                    const uint16_t half_nan = 0x7e00;
                    if (inject_nf) gpu(cudaMemcpyAsync((uint8_t*) b.q8 + (size_t) (t - 1) * row_bytes,
                                                       &half_nan, sizeof half_nan, cudaMemcpyHostToDevice, b.stream));
                    if (il && native_mmvq_il_supported(original.head.type(), t, vocab)) {
                        if (t >= 2) native_q8_1_interleave(b.q8, b.il, n, t, b.stream);
                        native_mmvq_il(original.head.type(), original.head.weights(), b.q8, b.il, b.logits, n, vocab, t, b.stream);
                    } else native_mmvq(original.head.type(), original.head.weights(), b.q8, b.logits, n, vocab, t, b.stream);
                    if (argmax) argmax_rows(b.logits, t, vocab, b.scratch, b.picks, b.stream);
                    else { SamplerParams greedy; greedy.greedy = true; greedy.temperature = 0;
                           sample_tokens(b.logits, t, vocab, nullptr, 0, greedy, b.picks, b.stream); }
                    sp.counter = 100000 + 97 * cycle + 11 * parity + t;
                    if (sampled) sample_tokens(b.logits, t, vocab, nullptr, 0, sp, b.picks, b.stream);
                    logits_nonfinite_rows(b.logits, t, vocab, b.nf, b.stream);
                    copy_from_mapped((float*) windows[parity].handoff(), (const float*) b.q8,
                                     (int64_t) t * row_bytes / sizeof(float), b.stream);
                    gpu(cudaEventRecord(b.ready, b.stream));
                    // Launch before even the pageable reference-result copies: a pageable cudaMemcpyAsync may
                    // block the host, accidentally turning an intended cross-device-event test into a sync test.
                    if (parity == 1) require(windows[parity].launch(t, sampled ? &sp : nullptr, sp.counter, err, b.ready), err);
                    gpu(cudaMemcpyAsync(e.logits.data(), b.logits, e.logits.size() * sizeof(float), cudaMemcpyDeviceToHost, b.stream));
                    gpu(cudaMemcpyAsync(e.picks.data(), b.picks, t * sizeof(int32_t), cudaMemcpyDeviceToHost, b.stream));
                    gpu(cudaMemcpyAsync(e.nf.data(), b.nf, e.nf.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, b.stream));
                    gpu(cudaStreamSynchronize(b.stream));
                    if (parity == 0) require(windows[parity].launch(t, sampled ? &sp : nullptr, sp.counter, err), err);
                    current_is(source);
                }
                for (int parity = 0; parity < 2; ++parity) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
                    while (!windows[parity].done(err)) {
                        require(err.empty(), err);
                        require(std::chrono::steady_clock::now() < deadline, "remote head timed out");
                        std::this_thread::yield();
                    }
                    current_is(source);
                    const Expected& e = expected[parity];
                    std::vector<float> row(vocab);
                    for (int r = 0; r < t; ++r) {
                        require(windows[parity].copy_logits(r, row.data()), "remote logits copy failed");
                        require(std::memcmp(row.data(), e.logits.data() + (size_t) r * vocab, (size_t) vocab * sizeof(float)) == 0,
                                "logits differ: IL=" + std::to_string(il) + " argmax=" + std::to_string(argmax) +
                                " T=" + std::to_string(t) + " cycle=" + std::to_string(cycle) + " row=" + std::to_string(r));
                    }
                    require(std::memcmp(windows[parity].picks(), e.picks.data(), t * sizeof(int32_t)) == 0, "sampled/greedy picks differ");
                    require(std::memcmp(windows[parity].nonfinite(), e.nf.data(), e.nf.size() * sizeof(uint32_t)) == 0, "nonfinite flags differ");
                    bool any_nf = false; for (uint32_t f : e.nf) any_nf |= f != 0;
                    require(any_nf == inject_nf, "nonfinite injection did not exercise the expected guard");
                    current_is(source);
                    ++cases; compared_logits += (size_t) t * vocab;
                }
            }
        }
        std::printf("PASS: %zu head windows, %zu bitwise logits; T=1,2,3,8; both IL/argmax settings; greedy/seeded sampling; "
                    "nonfinite flags; two parity buffers; host/event handoffs; source=%d target=%d\n", cases, compared_logits, source, target);
        const size_t shard_cases = test_shards(original.head, replica.head, target2 == target ? replica.head : replica2.head,
                                               b, n, vocab, cycles);
        std::printf("PASS: %zu sharded windows; bitwise full logits, gathered scores, merged greedy picks, row NF guards; "
                    "source=%d targets=%d,%d\n", shard_cases, source, target, target2);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
