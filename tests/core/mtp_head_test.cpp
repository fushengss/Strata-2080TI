// Manual three-GPU parity test. No running engine or peer-access enablement is needed.
// mtp_head_test --gguf MODEL.gguf --draft-vocab draft_vocab.bin --source 7 --target 3 --target2 5
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/mtp_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kN = 2560, kVocab = 106299, kSteps = 8;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void gpu(cudaError_t e) { require(e == cudaSuccess, cudaGetErrorString(e)); }
uint32_t bits(float v) { uint32_t b; std::memcpy(&b, &v, sizeof b); return b; }
void current_is(int device) {
    int actual = -1; gpu(cudaGetDevice(&actual));
    require(actual == device, "helper changed the caller's current CUDA device");
}
void default_env(const char* name, const char* value) {
#if defined(_WIN32)
    if (!std::getenv(name)) _putenv_s(name, value);
#else
    if (!std::getenv(name)) setenv(name, value, 0);
#endif
}

// Source lifetime encloses the helper, including on exceptions. All result/input slots are pinned:
// pageable cudaMemcpyAsync would introduce an unintended host wait in the chain test.
struct Source {
    int device;
    cudaStream_t stream = nullptr;
    std::vector<void*> allocations, pinned;
    std::vector<cudaGraphExec_t> graphs;
    explicit Source(int d) : device(d) { gpu(cudaSetDevice(d)); gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Source() {
        const strata::core::OnDevice on(device);
        if (stream) (void) cudaStreamSynchronize(stream);
        for (auto g : graphs) (void) cudaGraphExecDestroy(g);
        for (auto p : allocations) (void) cudaFree(p);
        for (auto p : pinned) (void) cudaFreeHost(p);
        if (stream) (void) cudaStreamDestroy(stream);
    }
    template<class T> T* device_buffer(size_t n) {
        void* p = nullptr; gpu(cudaMalloc(&p, n * sizeof(T))); allocations.push_back(p); return (T*) p;
    }
    template<class T> T* host_buffer(size_t n) {
        void* p = nullptr; gpu(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocPortable)); pinned.push_back(p); return (T*) p;
    }
    cudaGraphExec_t capture(const std::function<void()>& record) {
        gpu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try { record(); }
        catch (...) {
            cudaGraph_t abandoned = nullptr; (void) cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) (void) cudaGraphDestroy(abandoned);
            throw;
        }
        cudaGraph_t graph = nullptr; gpu(cudaStreamEndCapture(stream, &graph));
        cudaGraphExec_t exec = nullptr;
        const cudaError_t instantiated = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        (void) cudaGraphDestroy(graph); gpu(instantiated);
        graphs.push_back(exec); gpu(cudaGraphUpload(exec, stream));
        gpu(cudaStreamSynchronize(stream));
        return exec;
    }
};
struct Tail {
    float *logits, *prob;
    int32_t *local, *mapped;
    void *arg_scratch, *prob_scratch;
    float *host_logits, *host_prob;
    int32_t *host_local, *host_mapped;
    explicit Tail(Source& s) {
        using namespace strata::kernels;
        logits = s.device_buffer<float>(kVocab); prob = s.device_buffer<float>(1);
        local = s.device_buffer<int32_t>(1); mapped = s.device_buffer<int32_t>(1);
        arg_scratch = s.device_buffer<uint8_t>(argmax_rows_scratch_bytes(1));
        prob_scratch = s.device_buffer<uint8_t>(std::max<uint64_t>(1, row_top_prob_scratch_bytes(1)));
        gpu(cudaMemset(arg_scratch, 0, argmax_rows_scratch_bytes(1)));
        gpu(cudaMemset(prob_scratch, 0, std::max<uint64_t>(1, row_top_prob_scratch_bytes(1))));
        host_logits = s.host_buffer<float>((size_t) kSteps * kVocab);
        host_prob = s.host_buffer<float>(kSteps);
        host_local = s.host_buffer<int32_t>(kSteps); host_mapped = s.host_buffer<int32_t>(kSteps);
    }
    void record(bool split_prob, bool rows_argmax, const int32_t* vocab, cudaStream_t stream) {
        using namespace strata::kernels;
        if (rows_argmax) argmax_rows(logits, 1, kVocab, arg_scratch, local, stream);
        else { SamplerParams sp; sp.greedy = true; sample_tokens(logits, 1, kVocab, nullptr, 0, sp, local, stream); }
        if (split_prob) row_top_prob_split(logits, 1, kVocab, local, prob, prob_scratch, stream);
        else row_top_prob(logits, 1, kVocab, local, prob, stream);
        gpu(cudaMemcpyAsync(mapped, local, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        map_ids(mapped, vocab, 1, stream);
    }
    void save(int step, cudaStream_t stream) {
        gpu(cudaMemcpyAsync(host_logits + (size_t) step * kVocab, logits, kVocab * sizeof(float), cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(host_prob + step, prob, sizeof(float), cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(host_local + step, local, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        gpu(cudaMemcpyAsync(host_mapped + step, mapped, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
};

std::vector<int32_t> read_vocab(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require((bool) f && f.tellg() == (std::streamoff) kVocab * sizeof(int32_t),
            "draft vocabulary must contain exactly 106299 raw int32 ids");
    std::vector<int32_t> v(kVocab); f.seekg(0); f.read((char*) v.data(), v.size() * sizeof(int32_t));
    require((bool) f, "cannot read draft vocabulary"); return v;
}
std::vector<uint8_t> read_subset(const std::string& path, const std::vector<int32_t>& ids) {
    using namespace strata::kernels;
    const strata::GgufModel model(strata::gguf_split_paths(path));
    size_t shard = 0; const auto* t = model.find("output.weight", &shard);
    require(t && t->type == 14 && t->shape.size() == 2 && t->shape[0] == kN && model.in_bounds(*t, shard),
            "test requires real, in-bounds Q6_K output.weight with N=2560");
    const size_t row = native_mmvq_weight_bytes(14, kN, 1);
    require(row == 2100, "unexpected Q6_K row byte count");
    const uint8_t* data = model.shard(shard).tensor_data(*t);
    std::vector<uint8_t> seen((size_t) t->shape[1], 0), subset(ids.size() * row);
    for (size_t i = 0; i < ids.size(); ++i) {
        require(ids[i] >= 0 && (uint64_t) ids[i] < t->shape[1], "draft vocab id outside full vocabulary");
        require(!seen[ids[i]], "draft vocabulary contains duplicate ids"); seen[ids[i]] = 1;
        std::memcpy(subset.data() + i * row, data + (size_t) ids[i] * row, row);
    }
    return subset;
}
void fill_inputs(float* p, int cycle, int steps) {
    for (int j = 0; j < steps; ++j) {
        uint32_t state = 0x591223a5u + 997u * cycle + 1619u * j;
        for (int i = 0; i < kN; ++i) {
            state = state * 1664525u + 1013904223u;
            p[(size_t) j * kN + i] = cycle == 0 && j == 0 ? 0.0f :
                (float) ((int) (state >> 16) - 32768) / (cycle % 2 ? 8192.0f : 32768.0f);
        }
    }
}
void compare(const Tail& reference, const Tail& candidate, int step, const std::vector<int32_t>& ids,
             const std::string& label) {
    const float* a = reference.host_logits + (size_t) step * kVocab;
    const float* b = candidate.host_logits + (size_t) step * kVocab;
    for (int i = 0; i < kVocab; ++i) {
        if (bits(a[i]) != bits(b[i])) {
            char diff[160]; std::snprintf(diff, sizeof diff, " row=%d reference=0x%08x candidate=0x%08x", i, bits(a[i]), bits(b[i]));
            throw std::runtime_error(label + " full-logit mismatch" + diff);
        }
        if (!std::isfinite(a[i])) throw std::runtime_error(label + " nonfinite logit at row " + std::to_string(i));
    }
    require(reference.host_local[step] == candidate.host_local[step], label + " local argmax mismatch");
    const int32_t local = reference.host_local[step];
    require(local >= 0 && local < kVocab, label + " invalid local argmax");
    require(reference.host_mapped[step] == candidate.host_mapped[step] && candidate.host_mapped[step] == ids[local],
            label + " draft vocabulary mapping mismatch");
    if (bits(reference.host_prob[step]) != bits(candidate.host_prob[step])) {
        char diff[120]; std::snprintf(diff, sizeof diff, " reference=0x%08x candidate=0x%08x",
                                    bits(reference.host_prob[step]), bits(candidate.host_prob[step]));
        throw std::runtime_error(label + " probability mismatch" + diff);
    }
    require(std::isfinite(candidate.host_prob[step]) && candidate.host_prob[step] > 0 && candidate.host_prob[step] <= 1,
            label + " probability outside (0,1]");
}
void default_contract() {
    strata::core::MtpHeadShards h; std::string err;
    require(!h.initialized() && h.row_bytes() == 0 && h.row_count(0) == 0, "default helper must be inert");
    require(!h.enqueue(nullptr, nullptr, nullptr, 0, err) && !err.empty(), "default helper accepted enqueue");
    err.clear(); require(h.drain(err), "default helper failed drain"); h.reset();
    // Type rejection must happen before CUDA allocation or dereferencing the placeholder address.
    err.clear(); require(!h.init((const void*) 1, 7, 13, kN, kVocab, 2, 4, err) && !err.empty() && !h.initialized(),
                         "unsupported Q5 head was accepted");
    err.clear(); require(!h.init((const void*) 1, 7, 14, kN, kVocab, 2, 2, err) && !err.empty() && !h.initialized(),
                         "duplicate target devices were accepted");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        default_env("STRATA_SM75_PERF", "1"); default_env("STRATA_ARGMAX_ROWS_SM70", "1");
        default_contract();
        if (argc == 2 && std::strcmp(argv[1], "--contract-only") == 0) {
            std::puts("PASS: MTP_HEAD_DEFAULT_UNSUPPORTED_CONTRACT"); return 0;
        }
        std::string model, vocab_file; int source = 7, target0 = 2, target1 = 4, cycles = 4;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i]; require(i + 1 < argc, "missing value for " + a);
            if (a == "--gguf") model = argv[++i];
            else if (a == "--draft-vocab") vocab_file = argv[++i];
            else if (a == "--source") source = std::stoi(argv[++i]);
            else if (a == "--target") target0 = std::stoi(argv[++i]);
            else if (a == "--target2") target1 = std::stoi(argv[++i]);
            else if (a == "--cycles") cycles = std::stoi(argv[++i]);
            else throw std::runtime_error("unknown option " + a);
        }
        require(!model.empty() && !vocab_file.empty() && cycles >= 3 && source >= 0 && target0 >= 0 && target1 >= 0 &&
                    source != target0 && source != target1 && target0 != target1,
                "usage: mtp_head_test --gguf MODEL --draft-vocab FILE --source 7 --target 2 --target2 4 [--cycles 4, minimum 3]");
        int count = 0; const cudaError_t counted = cudaGetDeviceCount(&count);
        if (counted != cudaSuccess || std::max({source, target0, target1}) >= count) {
            std::fprintf(stderr, "SKIP: requested three GPUs unavailable (count=%d, CUDA=%s)\n", count, cudaGetErrorString(counted)); return 77;
        }
        using namespace strata::kernels;
        require(!native_q6_k_packed_enabled(), "test requires original unpacked Q6_K (Q6_PACKED must be off)");
        const auto ids = read_vocab(vocab_file); auto subset = read_subset(model, ids);
        Source s(source);
        uint8_t* weights = s.device_buffer<uint8_t>(subset.size());
        gpu(cudaMemcpy(weights, subset.data(), subset.size(), cudaMemcpyHostToDevice));
        subset.clear(); subset.shrink_to_fit();
        int32_t* dvocab = s.device_buffer<int32_t>(kVocab);
        gpu(cudaMemcpy(dvocab, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        float* mixed = s.device_buffer<float>(kN);
        void* q8 = s.device_buffer<uint8_t>(native_q8_1_bytes(kN, 1));
        float* inputs = s.host_buffer<float>((size_t) kSteps * kN);
        Tail reference(s), candidate(s);
        // The body, reference head, and source tails are independent single-device captures.
        // All cross-device waits/launches happen outside stream capture in helper.enqueue().
        const auto body = s.capture([&] { native_quantize_q8_1(mixed, q8, kN, 1, s.stream); });
        const auto original = s.capture([&] { native_mmvq(14, weights, q8, reference.logits, kN, kVocab, 1, s.stream); });
        strata::core::MtpHeadShards helper; std::string err;
        require(helper.init(weights, source, 14, kN, kVocab, target0, target1, err), err);
        current_is(source);
        require(helper.row_begin(0) == 0 && helper.row_begin(1) == helper.row_count(0) &&
                    helper.row_count(0) + helper.row_count(1) == kVocab && helper.row_bytes() == native_q8_1_bytes(kN, 1),
                "shards do not cover the complete subset in row order");
        for (int target : {target0, target1}) {
            int p2p = -1; gpu(cudaDeviceCanAccessPeer(&p2p, source, target));
            std::printf("topology: CUDA%d->CUDA%d peer_capable=%d; test never enables peer access\n", source, target, p2p);
        }
        err.clear(); require(!helper.enqueue(s.stream, q8, candidate.logits, 8, err) && !err.empty(), "invalid step accepted");
        err.clear();
        std::printf("shape: Q6_K subset=%d N=%d shards=%d+%d; STRATA_SM75_PERF=%s STRATA_ARGMAX_ROWS_SM70=%s\n",
                    kVocab, kN, helper.row_count(0), helper.row_count(1), std::getenv("STRATA_SM75_PERF"), std::getenv("STRATA_ARGMAX_ROWS_SM70"));
        size_t cases = 0, chained = 0;
        std::vector<uint32_t> canonical_prob;
        std::vector<int32_t> canonical_local, canonical_mapped;
        for (bool rows_argmax : {true, false}) for (bool split_prob : {false, true}) {
            const auto ref_tail = s.capture([&] { reference.record(split_prob, rows_argmax, dvocab, s.stream); });
            const auto got_tail = s.capture([&] { candidate.record(split_prob, rows_argmax, dvocab, s.stream); });
            size_t index = 0;
            for (int steps : {1, kSteps}) for (int cycle = 0; cycle < cycles; ++cycle) {
                fill_inputs(inputs, cycle, steps);
                for (int j = 0; j < steps; ++j) {
                    gpu(cudaMemcpyAsync(mixed, inputs + (size_t) j * kN, kN * sizeof(float), cudaMemcpyHostToDevice, s.stream));
                    gpu(cudaGraphLaunch(body, s.stream));
                    gpu(cudaGraphLaunch(original, s.stream)); gpu(cudaGraphLaunch(ref_tail, s.stream));
                    reference.save(j, s.stream);
                    require(helper.enqueue(s.stream, q8, candidate.logits, j, err), err);
                    gpu(cudaGraphLaunch(got_tail, s.stream)); candidate.save(j, s.stream);
                    // No host wait here: the next body must respect the source waits installed by enqueue().
                }
                require(helper.drain(err), err); current_is(source);
                for (int j = 0; j < steps; ++j, ++index) {
                    const std::string label = "argmax=" + std::to_string(rows_argmax) + " split_prob=" + std::to_string(split_prob) +
                        " steps=" + std::to_string(steps) + " cycle=" + std::to_string(cycle) + " step=" + std::to_string(j);
                    compare(reference, candidate, j, ids, label);
                    if (rows_argmax && !split_prob) {
                        canonical_prob.push_back(bits(reference.host_prob[j])); canonical_local.push_back(reference.host_local[j]);
                        canonical_mapped.push_back(reference.host_mapped[j]);
                    } else {
                        require(canonical_prob[index] == bits(reference.host_prob[j]) && canonical_local[index] == reference.host_local[j] &&
                                    canonical_mapped[index] == reference.host_mapped[j], label + " original tail modes differ");
                    }
                    if (cycle == 0 && j == 0) require(candidate.host_local[j] == 0, "zero-input tie must pick subset row zero");
                    ++cases; if (steps > 1) ++chained;
                }
            }
            std::printf("tail: argmax_rows=%d split_probability=%d passed %zu T1 outputs\n", rows_argmax, split_prob, index);
            std::fflush(stdout);
        }
        helper.reset(); require(!helper.initialized(), "reset left helper initialized"); current_is(source);
        std::printf("PASS: MTP_HEAD_BITWISE_LOGITS_PROBABILITIES cases=%zu logits=%zu dvocab_mapping=exact\n", cases, cases * kVocab);
        std::printf("PASS: MTP_HEAD_CHAIN_REUSE outputs=%zu steps=8 repeated_chains=%d\n", chained, cycles * 4);
        std::puts("PASS: MTP_HEAD_DEFAULT_UNSUPPORTED_CONTRACT (engine fallback requires separate integration test)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: MTP head parity: %s\n", e.what()); return 1;
    }
}
