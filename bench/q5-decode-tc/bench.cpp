// Replay captured MoE routes with real compressed expert weights from the GGUF.
#include "tensor_expert.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace K = strata::kernels;
namespace cpu = strata::kernels::cpu;

constexpr int kWidth = 2560;
constexpr int kHidden = 640;
constexpr int kExperts = 512;
constexpr int kTop = 10;
constexpr int kMaxRows = 16;
constexpr int kEntries = kTop * kMaxRows;
constexpr int kMib = 1024 * 1024;
constexpr double kRmsLimit = 0.01;
constexpr double kMaxByRmsLimit = 0.15;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct Options {
    std::vector<std::string> models, route_files;
    std::vector<int> layers, phases{0, 3};
    int samples = 96, rounds = 5, iters = 2, weight_mib = 1024;
    bool check_only = false;
};

std::vector<int> int_list(const std::string& raw) {
    std::istringstream input(raw);
    std::vector<int> values;
    std::string item;
    while (std::getline(input, item, ',')) {
        size_t end = 0;
        int value = std::stoi(item, &end);
        if (end != item.size()) throw std::runtime_error("invalid integer list: " + raw);
        values.push_back(value);
    }
    return values;
}

Options parse(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-only") {
            opts.check_only = true;
            continue;
        }
        if (++i >= argc) throw std::runtime_error("missing argument to " + key);
        const std::string value = argv[i];
        if (key == "--model") opts.models.push_back(value);
        else if (key == "--routes") opts.route_files.push_back(value);
        else if (key == "--layers") opts.layers = int_list(value);
        else if (key == "--phases") opts.phases = int_list(value);
        else {
            auto integers = int_list(value);
            if (integers.size() != 1 || integers.front() < 1)
                throw std::runtime_error("expected one positive integer for " + key);
            if (key == "--samples") opts.samples = integers.front();
            else if (key == "--rounds") opts.rounds = integers.front();
            else if (key == "--iters") opts.iters = integers.front();
            else if (key == "--weight-mib") opts.weight_mib = integers.front();
            else throw std::runtime_error("unknown option: " + key);
        }
    }
    if (opts.models.empty() || opts.route_files.empty() || opts.layers.empty() || opts.phases.empty() ||
        std::any_of(opts.phases.begin(), opts.phases.end(), [](int p) { return p != 0 && p != 1 && p != 3; }))
        throw std::runtime_error("usage: q5_decode_tc_bench --model SHARD [--model SHARD] "
                                 "--routes FILE [--routes FILE] --layers 3,6 "
                                 "[--phases 0,1,3] [--samples N] [--rounds N] [--iters N] [--weight-mib N] [--check-only]");
    return opts;
}

template<class T> struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* data, size_t n) { check(cudaMemcpy(ptr, data, n * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
};

void download(void* host, const void* device, size_t bytes) {
    check(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "download");
}

struct Stream {
    cudaStream_t ptr = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&ptr, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() { cudaStreamDestroy(ptr); }
};

struct Event {
    cudaEvent_t ptr = nullptr;
    Event() { check(cudaEventCreate(&ptr), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(ptr); }
};

struct Graph {
    cudaGraphExec_t exec = nullptr;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); }
    template<class F> void capture(cudaStream_t stream, F launch) {
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
        try {
            launch();
        } catch (...) {
            cudaGraph_t invalid = nullptr;
            cudaStreamEndCapture(stream, &invalid);
            if (invalid) cudaGraphDestroy(invalid);
            throw;
        }
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        const auto status = cudaGraphInstantiate(&exec, graph, 0);
        cudaGraphDestroy(graph);
        check(status, "cudaGraphInstantiate");
        check(cudaGraphUpload(exec, stream), "cudaGraphUpload");
        check(cudaStreamSynchronize(stream), "graph upload");
    }
    void run(cudaStream_t stream) const { check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch"); }
};

struct Route {
    int layer = 0, rows = 0;
    size_t line = 0;
    std::array<int, kEntries> ids{};
};

std::vector<Route> read_routes(const std::string& path, int layer) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open route file: " + path);
    std::vector<Route> result;
    size_t line = 0;
    std::string text;
    while (std::getline(file, text)) {
        ++line;
        if (text.find_first_not_of(" \t\r") == std::string::npos || text[0] == '#') continue;
        std::istringstream s(text);
        Route route;
        route.line = line;
        if (!(s >> route.layer >> route.rows) || route.layer < 0 || route.layer >= 48 ||
            route.rows <= 0 || route.rows > kMaxRows)
            throw std::runtime_error(path + ":" + std::to_string(line) + ": invalid layer or row count");
        for (int row = 0; row < route.rows; ++row) {
            const auto begin = route.ids.begin() + row * kTop;
            for (auto pos = begin; pos != begin + kTop; ++pos)
                if (!(s >> *pos) || *pos < 0 || *pos >= kExperts || std::find(begin, pos, *pos) != pos)
                    throw std::runtime_error(path + ":" + std::to_string(line) + ": invalid or duplicate expert");
        }
        std::string extra;
        if (s >> extra) throw std::runtime_error(path + ":" + std::to_string(line) + ": trailing fields");
        if (route.layer == layer) result.push_back(route);
    }
    if (!file.eof()) throw std::runtime_error("reading route file failed: " + path);
    if (result.empty()) throw std::runtime_error(path + ": no records for layer " + std::to_string(layer));
    return result;
}

using ModelFiles = std::vector<std::unique_ptr<strata::GgufFile>>;

struct Weights {
    K::NativeExpertLayout layout;
    cpu::NativeFmt format;
    std::array<const uint8_t*, 3> tensors{};
    static constexpr std::array<int, 3> width{kWidth, kWidth, kHidden};
    static constexpr std::array<int, 3> rows{kHidden, kHidden, kWidth};

    Weights(const ModelFiles& files, int layer) {
        const std::array<const char*, 3> roles{"gate", "up", "down"};
        std::array<int, 3> types{};
        for (size_t i = 0; i < roles.size(); ++i) {
            const auto name = "blk." + std::to_string(layer) + ".ffn_" + roles[i] + "_exps.weight";
            for (const auto& file : files) {
                const auto* info = file->find(name);
                if (!info) continue;
                if (tensors[i]) throw std::runtime_error("duplicate model tensor: " + name);
                if (info->shape != std::vector<uint64_t>{uint64_t(width[i]), uint64_t(rows[i]), kExperts})
                    throw std::runtime_error("unexpected expert dimensions: " + name);
                tensors[i] = file->tensor_data(*info);
                types[i] = int(info->type);
            }
            if (!tensors[i]) throw std::runtime_error("missing model tensor: " + name);
        }
        if (types[0] != GGML_TYPE_Q5_K || types[1] != GGML_TYPE_Q5_K ||
            (types[2] != GGML_TYPE_Q5_1 && types[2] != GGML_TYPE_Q8_0))
            throw std::runtime_error("expected Q5_K gate/up and Q5_1 or Q8_0 down");
        std::string err;
        if (!cpu::native_fmt(types[0], types[2], kWidth, kHidden, format, err)) throw std::runtime_error(err);
        layout = K::native_expert_layout(types[0], types[2], kWidth, kHidden);
        if (format.bytes != layout.bytes || format.up_off != layout.up_off || format.down_off != layout.down_off ||
            format.gu_row != layout.gu_row || format.d_row != layout.d_row)
            throw std::runtime_error("CPU/GPU packed expert layouts differ");
    }
    size_t row_bytes(int kind) const { return kind == 2 ? format.d_row : format.gu_row; }
    const uint8_t* expert(int kind, int id) const { return tensors[kind] + size_t(id) * rows[kind] * row_bytes(kind); }
};

struct WeightCache {
    const Weights& weights;
    int limit, used = 0;
    DeviceBuffer<uint8_t> bytes;
    std::array<int, kExperts> slots{};

    WeightCache(const Weights& w, int mib)
        : weights(w), limit(int(std::min<size_t>(kExperts, size_t(mib) * kMib / w.format.bytes))),
          bytes(size_t(std::max(limit, 1)) * w.format.bytes) {
        if (limit <= 0) throw std::runtime_error("weight budget cannot hold one expert");
        slots.fill(-1);
    }
    void load(const std::vector<int>& experts) {
        if (experts.size() > size_t(limit)) throw std::runtime_error("weight budget cannot hold one recorded route");
        const int missing = int(std::count_if(experts.begin(), experts.end(), [&](int e) { return slots[e] < 0; }));
        if (used + missing > limit) {
            slots.fill(-1);
            used = 0;
        }
        for (int e : experts) {
            if (slots[e] >= 0) continue;
            auto* dest = bytes.ptr + size_t(used) * weights.format.bytes;
            const std::array<size_t, 3> offsets{0, weights.format.up_off, weights.format.down_off};
            for (int kind = 0; kind < 3; ++kind)
                check(cudaMemcpy(dest + offsets[kind], weights.expert(kind, e),
                                 size_t(weights.rows[kind]) * weights.row_bytes(kind), cudaMemcpyHostToDevice),
                      "upload compressed expert");
            slots[e] = used++;
        }
    }
    unsigned long long address(int expert) const {
        return reinterpret_cast<unsigned long long>(bytes.ptr + size_t(slots[expert]) * weights.format.bytes);
    }
};

struct Plan {
    std::vector<int> experts;
    std::vector<int32_t> starts{0}, dst, tok;
    std::array<int, kMaxRows + 1> multiplicity{};
    explicit Plan(const Route& route) {
        const auto end = route.ids.begin() + route.rows * kTop;
        for (auto it = route.ids.begin(); it != end; ++it)
            if (std::find(route.ids.begin(), it, *it) == it) experts.push_back(*it);
        for (int expert : experts) {
            for (int j = 0; j < route.rows * kTop; ++j) {
                if (route.ids[j] != expert) continue;
                dst.push_back(j);
                tok.push_back(j / kTop);
            }
            const int count = int(dst.size()) - starts.back();
            ++multiplicity[count];
            starts.push_back(int(dst.size()));
        }
    }
};

struct Buffers {
    DeviceBuffer<unsigned long long> ptr{kEntries};
    DeviceBuffer<int32_t> start{kEntries + 1}, groups{1}, dst{kEntries}, tok{kEntries};
    DeviceBuffer<float> x{kMaxRows * kWidth}, out{kEntries * kWidth};
    DeviceBuffer<uint8_t> xq{kMaxRows * kWidth / 32 * 36}, scratch{K::native_expert_scratch_bytes(kEntries, kHidden)};
    void plan(const Plan& host, WeightCache& cache) {
        cache.load(host.experts);
        std::vector<unsigned long long> pointers;
        for (int id : host.experts) pointers.push_back(cache.address(id));
        const int32_t count = int32_t(pointers.size());
        ptr.upload(pointers.data(), pointers.size());
        start.upload(host.starts.data(), host.starts.size());
        groups.upload(&count, 1);
        dst.upload(host.dst.data(), host.dst.size());
        tok.upload(host.tok.data(), host.tok.size());
    }
};

std::vector<float> q8_values(const uint8_t* blocks, size_t values) {
    std::vector<float> floats(values);
    for (size_t i = 0; i < values; i += 32, blocks += 36) {
        ggml_fp16_t scale;
        std::memcpy(&scale, blocks, sizeof(scale));
        const float d = ggml_fp16_to_fp32(scale);
        for (int j = 0; j < 32; ++j) floats[i + j] = d * static_cast<int8_t>(blocks[4 + j]);
    }
    return floats;
}

std::vector<float> outputs(Buffers& b, int entries, int phase) {
    if (phase == 0) {
        std::vector<float> y(size_t(entries) * kWidth);
        download(y.data(), b.out.ptr, y.size() * sizeof(float));
        return y;
    }
    const size_t values = size_t(entries) * kHidden;
    const size_t bytes = values * sizeof(float);
    const size_t stride = (bytes + 255) & ~size_t(255);
    if (phase == 1) {
        std::vector<uint8_t> hq(values / 32 * 36);
        download(hq.data(), b.scratch.ptr + 3 * stride, hq.size());
        return q8_values(hq.data(), values);
    }
    std::vector<float> y(2 * values);
    download(y.data(), b.scratch.ptr, bytes);
    download(y.data() + values, b.scratch.ptr + stride, bytes);
    return y;
}

struct Error {
    double squared = 0, norm = 0, maximum = 0;
    size_t finite = 0, nonfinite = 0;
    void add(float ref, float got) {
        if (!std::isfinite(ref) || !std::isfinite(got)) { ++nonfinite; return; }
        const double error = double(got) - ref;
        squared += error * error;
        norm += double(ref) * ref;
        maximum = std::max(maximum, std::abs(error));
        ++finite;
    }
    double rms() const { return std::sqrt(squared / std::max(norm, 1e-30)); }
    double max_by_rms() const { return maximum * std::sqrt(double(finite) / std::max(norm, 1e-30)); }
    bool accepted() const { return finite && !nonfinite && rms() <= kRmsLimit && max_by_rms() <= kMaxByRmsLimit; }
};

void print_error(const Error& error) {
    std::cout << "{\"rel_rms\":" << error.rms() << ",\"max_abs_by_ref_rms\":" << error.max_by_rms()
              << ",\"nonfinite\":" << error.nonfinite << ",\"accepted\":" << (error.accepted() ? "true" : "false") << "}";
}

std::vector<float> reference_dot(const Weights& w, int kind, int expert, const std::vector<float>& x) {
    const auto type = static_cast<ggml_type>(kind == 2 ? w.format.d_type : w.format.gu_type);
    const auto dequantize = ggml_get_type_traits(type)->to_float;
    if (!dequantize) throw std::runtime_error("no CPU dequantizer for reference");
    std::vector<float> weight(x.size()), y(w.rows[kind]);
    const auto* data = w.expert(kind, expert);
    for (size_t row = 0; row < y.size(); ++row) {
        dequantize(data + row * w.row_bytes(kind), weight.data(), x.size());
        double sum = 0;
        for (size_t i = 0; i < x.size(); ++i) sum += double(x[i]) * weight[i];
        y[row] = float(sum);
    }
    return y;
}

std::vector<float> reference(const Weights& w, int expert, const Buffers& b, int phase) {
    std::array<uint8_t, kWidth / 32 * 36> xq{};
    download(xq.data(), b.xq.ptr, xq.size());
    const auto x = q8_values(xq.data(), kWidth);
    auto gate = reference_dot(w, 0, expert, x);
    const auto up = reference_dot(w, 1, expert, x);
    if (phase == 3) {
        gate.insert(gate.end(), up.begin(), up.end());
        return gate;
    }
    for (int i = 0; i < kHidden; ++i) gate[i] = gate[i] / (1.f + std::exp(-gate[i])) * up[i];
    std::array<uint8_t, kHidden / 32 * 36> hq{};
    // Both GPU kernels consume Q8_1 after SwiGLU, including on layers with Q8_0 down weights.
    auto gpu_format = w.format;
    gpu_format.d_act = GGML_TYPE_Q8_1;
    cpu::native_quant_h(gpu_format, gate.data(), hq.data());
    const auto h = q8_values(hq.data(), kHidden);
    return phase == 1 ? h : reference_dot(w, 2, expert, h);
}

class Trial {
    Stream stream_;
    Event start_, end_;
    Buffers b_;

    std::vector<float> read(const Graph& graph, int entries, int phase) {
        // Catch skipped output rows before testing a performance-only graph replay.
        check(cudaMemsetAsync(b_.out.ptr, 0xff, size_t(entries) * kWidth * sizeof(float), stream_.ptr), "poison output");
        check(cudaMemsetAsync(b_.scratch.ptr, 0xff, K::native_expert_scratch_bytes(entries, kHidden), stream_.ptr),
              "poison scratch");
        graph.run(stream_.ptr);
        check(cudaStreamSynchronize(stream_.ptr), "validate graph");
        return outputs(b_, entries, phase);
    }

    double time(const Graph& graph, int iters) {
        check(cudaEventRecord(start_.ptr, stream_.ptr), "record start");
        for (int i = 0; i < iters; ++i) graph.run(stream_.ptr);
        check(cudaEventRecord(end_.ptr, stream_.ptr), "record finish");
        check(cudaEventSynchronize(end_.ptr), "finish measurement");
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, start_.ptr, end_.ptr), "read timing");
        return 1000.0 * elapsed / iters;
    }

public:
    void prepare(const Route& route, const Plan& plan, WeightCache& cache) {
        b_.plan(plan, cache);
        std::mt19937 seed(17 + unsigned(route.layer) * 4099 + unsigned(route.line));
        std::normal_distribution<float> normal;
        std::vector<float> x(size_t(route.rows) * kWidth);
        for (float& v : x) v = normal(seed);
        b_.x.upload(x.data(), x.size());
        K::quantize_q8_1_rows(b_.x.ptr, route.rows, kWidth, b_.xq.ptr, stream_.ptr);
        check(cudaStreamSynchronize(stream_.ptr), "quantize activation");
    }

    bool run(const Route& route, const Plan& plan, const Weights& weights, const Options& opts,
             const std::string& family, int phase, bool cpu_check) {
        const int entries = route.rows * kTop;
        Graph dp4a, tensor;
        K::native_expert_set_mode(-1, phase);
        dp4a.capture(stream_.ptr, [&] {
            K::native_expert_grouped(weights.layout, b_.ptr.ptr, b_.start.ptr, b_.groups.ptr, b_.dst.ptr, b_.tok.ptr,
                                     entries, entries, b_.xq.ptr, b_.scratch.ptr, b_.out.ptr, stream_.ptr);
        });
        K::native_expert_set_mode(-1, 0);
        tensor.capture(stream_.ptr, [&] {
            q5_decode_tc::launch(weights.layout, b_.ptr.ptr, b_.start.ptr, b_.groups.ptr, b_.dst.ptr, b_.tok.ptr,
                                 entries, entries, b_.xq.ptr, b_.scratch.ptr, b_.out.ptr, stream_.ptr, phase);
        });
        const auto base = read(dp4a, entries, phase);
        const auto got = read(tensor, entries, phase);
        Error paired, base_ref, tensor_ref;
        for (size_t i = 0; i < base.size(); ++i) paired.add(base[i], got[i]);
        if (cpu_check) {
            const auto expected = reference(weights, route.ids.front(), b_, phase);
            for (size_t i = 0; i < expected.size(); ++i) {
                const auto index = phase == 3 && i >= kHidden ? i + size_t(entries - 1) * kHidden : i;
                base_ref.add(expected[i], base[index]);
                tensor_ref.add(expected[i], got[index]);
            }
        }
        const int rounds = opts.check_only ? 0 : opts.rounds;
        std::vector<double> dp4a_us(rounds), tensor_us(rounds);
        for (int round = 0; round < rounds; ++round) {
            if ((round + route.line) % 2 == 0) {
                dp4a_us[round] = time(dp4a, opts.iters);
                tensor_us[round] = time(tensor, opts.iters);
            } else {
                tensor_us[round] = time(tensor, opts.iters);
                dp4a_us[round] = time(dp4a, opts.iters);
            }
        }
        const auto array = [](const auto& values) {
            std::cout << "[";
            for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
            std::cout << "]";
        };
        std::cout << "{\"routes\":" << std::quoted(family) << ",\"line\":" << route.line
                  << ",\"layer\":" << route.layer << ",\"down\":" << std::quoted(ggml_type_name(static_cast<ggml_type>(weights.format.d_type)))
                  << ",\"rows\":" << route.rows << ",\"groups\":" << plan.experts.size()
                  << ",\"phase\":" << phase << ",\"expert_multiplicity\":";
        array(plan.multiplicity);
        std::cout << ",\"dp4a_us\":";
        array(dp4a_us);
        std::cout << ",\"tensor_us\":";
        array(tensor_us);
        std::cout << ",\"tensor_vs_dp4a\":";
        print_error(paired);
        if (cpu_check) {
            std::cout << ",\"dp4a_vs_cpu\":";
            print_error(base_ref);
            std::cout << ",\"tensor_vs_cpu\":";
            print_error(tensor_ref);
        }
        std::cout << "}\n" << std::flush;
        return paired.accepted() && (!cpu_check || (base_ref.accepted() && tensor_ref.accepted()));
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        std::cout << std::setprecision(9);
        ModelFiles files;
        for (const auto& path : options.models) files.push_back(std::make_unique<strata::GgufFile>(path));
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0), "get GPU properties");
        if (device.major != 7 || device.minor != 5) throw std::runtime_error("this benchmark expects SM75");
        check(cudaSetDevice(0), "select GPU");
        std::cout << "{\"device\":" << std::quoted(device.name) << ",\"rounds\":" << options.rounds
                  << ",\"iters\":" << options.iters << ",\"weight_mib\":" << options.weight_mib
                  << ",\"check_only\":" << (options.check_only ? "true" : "false")
                  << ",\"rms_limit\":" << kRmsLimit << ",\"max_abs_by_ref_rms_limit\":" << kMaxByRmsLimit
                  << ",\"activations\":\"deterministic Gaussian, quantized to Q8_1 by Strata\"}\n";
        Trial trial;
        size_t failed = 0;
        for (int layer : options.layers) {
            const Weights weights(files, layer);
            WeightCache cache(weights, options.weight_mib);
            for (const auto& path : options.route_files) {
                const auto routes = read_routes(path, layer);
                const auto family = std::filesystem::path(path).filename().string();
                const size_t count = std::min(routes.size(), size_t(options.samples));
                const size_t cases = options.check_only ? 1 : count;
                for (size_t i = 0; i < cases; ++i) {
                    const auto& route = routes[((2 * i + 1) * routes.size()) / (2 * count)];
                    const Plan plan(route);
                    trial.prepare(route, plan, cache);
                    for (const int phase : options.phases)
                        failed += !trial.run(route, plan, weights, options, family, phase, i == 0);
                }
            }
        }
        std::cout << "{\"failed_checks\":" << failed << "}\n";
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "q5_decode_tc_bench: " << error.what() << "\n";
        return 2;
    }
}
