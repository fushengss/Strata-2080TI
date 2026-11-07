// Standalone diagnostic, never part of the serving engine. Root schedules GPU execution.
// local_tp_bench --gguf MODEL.gguf --devices 0,1[,2,3] [--layers 3,6]
// Compare a decomposed, complete-dot routed MoE with output-row TP and with production fused experts.
// No P2P: all communication traverses portable pinned host memory, with cross-device event ordering.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace {
constexpr int N = 2560, FF = 640, MT = 8;
constexpr size_t XQ = N / 32 * 36, HQ = FF / 32 * 36;
using Clock = std::chrono::steady_clock;
void check(bool b, const std::string& s) { if (!b) throw std::runtime_error(s); }
void cu(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
double elapsed_us(Clock::time_point a) { return std::chrono::duration<double, std::micro>(Clock::now() - a).count(); }
std::vector<int> ints(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string p;
    while (std::getline(ss, p, ',')) { size_t n = 0; const int x = std::stoi(p, &n); check(n == p.size(), "bad integer list"); v.push_back(x); }
    check(!v.empty(), "empty integer list"); return v;
}
// Raw bits only; Q8_1 bytes may resemble floating point NaNs.
__global__ void publish(const uint32_t* src, uint32_t* dst, int width, int pitch, int offset, int rows) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * rows; i += blockDim.x * gridDim.x)
        dst[(i / width) * pitch + offset + i % width] = src[i];
    __threadfence_system();
}
__global__ void acquire(const uint32_t* src, uint32_t* dst, int words) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += blockDim.x * gridDim.x) dst[i] = src[i];
}
__global__ void transpose_parts(const float* src, float* dst, int experts, int t, int width) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < experts * t * width; i += blockDim.x * gridDim.x) {
        const int c = i % width, r = i / width;
        dst[((r % t) * experts + r / t) * width + c] = src[i];
    }
}
int blocks(int words) { return std::max(1, std::min(128, (words + 255) / 256)); }
void put(const void* src, void* host, int width_bytes, int pitch_bytes, int offset_bytes, int rows, cudaStream_t s) {
    check((width_bytes | pitch_bytes | offset_bytes) % 4 == 0, "copy requires whole words");
    publish<<<blocks(width_bytes / 4 * rows),256,0,s>>>((const uint32_t*) src, (uint32_t*) host,
        width_bytes / 4, pitch_bytes / 4, offset_bytes / 4, rows); cu(cudaPeekAtLastError());
}
void get(const void* host, void* dst, size_t bytes, cudaStream_t s) {
    check(bytes % 4 == 0 && bytes / 4 <= INT_MAX, "invalid acquire size");
    acquire<<<blocks((int) bytes / 4),256,0,s>>>((const uint32_t*) host, (uint32_t*) dst, (int) bytes / 4);
    cu(cudaPeekAtLastError());
}
struct Pinned {
    void* host = nullptr;
    std::vector<void*> alias;
    Pinned(size_t bytes, const std::vector<int>& devices) {
        cu(cudaHostAlloc(&host, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(host, 0, bytes);
        for (int d : devices) { cu(cudaSetDevice(d)); void* p = nullptr; cu(cudaHostGetDevicePointer(&p, host, 0)); alias.push_back(p); }
    }
    ~Pinned() { if (host) cudaFreeHost(host); }
    Pinned(const Pinned&) = delete;
};
struct Device {
    int id;
    cudaStream_t stream = nullptr;
    std::vector<void*> allocations;
    std::vector<cudaEvent_t> events;
    explicit Device(int d) : id(d) { select(); cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    void select() const { cu(cudaSetDevice(id)); }
    void sync() const { select(); cu(cudaStreamSynchronize(stream)); }
    template<class T> T* alloc(size_t n) {
        select(); void* p = nullptr; cu(cudaMalloc(&p, n * sizeof(T))); allocations.push_back(p); return (T*) p;
    }
    template<class T> T* upload(const std::vector<T>& v) {
        T* p = alloc<T>(v.size()); cu(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
    }
    cudaEvent_t event(bool timing = false) {
        select(); cudaEvent_t e = nullptr; cu(cudaEventCreateWithFlags(&e, timing ? cudaEventDefault : cudaEventDisableTiming));
        events.push_back(e); return e;
    }
    ~Device() {
        cudaSetDevice(id); if (stream) cudaStreamSynchronize(stream);
        for (auto e : events) cudaEventDestroy(e);
        for (void* p : allocations) cudaFree(p);
        if (stream) cudaStreamDestroy(stream);
    }
    Device(const Device&) = delete;
};
struct Graph {
    int device = -1;
    cudaGraphExec_t exec = nullptr;
    void make(Device& d, const std::function<void()>& fn) {
        device = d.id; d.select(); fn(); d.sync(); // initialize lazy dispatch outside capture
        cu(cudaStreamBeginCapture(d.stream, cudaStreamCaptureModeThreadLocal));
        cudaGraph_t g = nullptr;
        try { fn(); cu(cudaStreamEndCapture(d.stream, &g)); }
        catch (...) { cudaStreamEndCapture(d.stream, &g); if (g) cudaGraphDestroy(g); throw; }
        const cudaError_t e = cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);
        cudaGraphDestroy(g); cu(e); cu(cudaGraphUpload(exec, d.stream)); d.sync();
    }
    void launch(Device& d) { d.select(); cu(cudaGraphLaunch(exec, d.stream)); }
    ~Graph() { if (exec) { cudaSetDevice(device); cudaGraphExecDestroy(exec); } }
    Graph() = default; Graph(const Graph&) = delete;
};
struct Diff { size_t words = 0, different = 0, nonfinite = 0; double max_abs = 0, max_rel = 0; };
Diff diff(const std::vector<float>& a, const std::vector<float>& b) {
    check(a.size() == b.size(), "comparison shape mismatch"); Diff d; d.words = a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        d.different += std::memcmp(&a[i], &b[i], 4) != 0;
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++d.nonfinite; continue; }
        const double z = std::abs((double) a[i] - b[i]);
        d.max_abs = std::max(d.max_abs, z); d.max_rel = std::max(d.max_rel, z / (std::abs((double) a[i]) + 1e-8));
    }
    return d;
}
void print_diff(const char* name, const Diff& d) {
    std::printf("\"%s\":{\"words\":%zu,\"different\":%zu,\"nonfinite\":%zu,\"max_abs\":%.12g,\"max_rel\":%.12g}",
        name, d.words, d.different, d.nonfinite, d.max_abs, d.max_rel);
}
template<class T> std::vector<T> read(Device& d, const T* p, size_t n) {
    d.sync(); std::vector<T> v(n); cu(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
double event_us(cudaEvent_t a, cudaEvent_t b) { float ms = 0; cu(cudaEventElapsedTime(&ms, a, b)); return ms * 1000.0; }
double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end()); return v[(size_t) std::floor(q * (v.size() - 1))];
}
struct Sample { double wall = 0, kernel_max = 0, kernel_sum = 0; };
struct Timings {
    std::vector<double> wall, maximum, sum;
    void add(Sample s) { wall.push_back(s.wall); maximum.push_back(s.kernel_max); sum.push_back(s.kernel_sum); }
    void print(const char* name) const {
        std::printf("\"%s\":{\"samples\":%zu,\"wall_us_p50\":%.6f,\"wall_us_p95\":%.6f,\"kernel_phase_max_rank_us_p50\":%.6f,\"kernel_phase_sum_ranks_us_p50\":%.6f}",
            name, wall.size(), percentile(wall,.5), percentile(wall,.95), percentile(maximum,.5), percentile(sum,.5));
    }
};
struct Weights {
    int layer, gu_type, down_type, experts;
    size_t grow, drow, gate_bytes, down_bytes;
    K::NativeExpertLayout layout;
    std::vector<uint8_t> gate, up, down, blobs;
    Weights(const strata::GgufModel& m, int l, const std::vector<int>& ids) : layer(l), experts((int) ids.size()) {
        const char* names[] = {"gate", "up", "down"}; const strata::TensorInfo* ti[3]; const uint8_t* ptr[3];
        for (int k = 0; k < 3; ++k) {
            size_t shard = 0; ti[k] = m.find("blk." + std::to_string(l) + ".ffn_" + names[k] + "_exps.weight", &shard);
            check(ti[k] && m.in_bounds(*ti[k], shard), "missing or invalid expert tensor");
            const auto& s = ti[k]->shape;
            check(s.size() == 3 && s[0] == (k == 2 ? FF : N) && s[1] == (k == 2 ? N : FF), "unexpected expert shape");
            for (int id : ids) check(id >= 0 && id < (int) s[2], "expert id out of bounds");
            ptr[k] = m.shard(shard).tensor_data(*ti[k]);
        }
        gu_type = (int) ti[0]->type; down_type = (int) ti[2]->type;
        check(gu_type == 13 && ti[1]->type == 13 && (down_type == 7 || down_type == 8), "this diagnostic requires Q5_K GU and Q5_1/Q8_0 down");
        layout = K::native_expert_layout(gu_type, down_type, N, FF);
        grow = K::native_mmvq_weight_bytes(gu_type, N, 1); drow = K::native_mmvq_weight_bytes(down_type, FF, 1);
        gate_bytes = grow * FF; down_bytes = drow * N;
        check(layout.up_off == gate_bytes && layout.down_off == 2 * gate_bytes && layout.bytes == 2 * gate_bytes + down_bytes,
            "production expert layout is not raw GU/down rows");
        gate.resize(experts * gate_bytes); up.resize(gate.size()); down.resize(experts * down_bytes); blobs.resize(experts * layout.bytes);
        for (int e = 0; e < experts; ++e) {
            std::memcpy(gate.data() + e * gate_bytes, ptr[0] + ids[e] * gate_bytes, gate_bytes);
            std::memcpy(up.data() + e * gate_bytes, ptr[1] + ids[e] * gate_bytes, gate_bytes);
            std::memcpy(down.data() + e * down_bytes, ptr[2] + ids[e] * down_bytes, down_bytes);
            auto* b = blobs.data() + e * layout.bytes;
            std::memcpy(b, gate.data() + e * gate_bytes, gate_bytes);
            std::memcpy(b + gate_bytes, up.data() + e * gate_bytes, gate_bytes);
            std::memcpy(b + 2 * gate_bytes, down.data() + e * down_bytes, down_bytes);
        }
    }
};
struct Rank {
    Device d;
    int ff0, ff, n0, n, E;
    uint8_t *wg, *wu, *wd, *xq, *hq_local, *hq_full;
    float *g, *u, *parts_et, *parts_te, *weights, *out, *full_out;
    cudaEvent_t published, finished, gu0, gu1, dn0, dn1;
    Graph gu, dn;
    Rank(int device, int rank, int nr, const Weights& w) : d(device), ff0(rank * FF / nr), ff(FF / nr),
        n0(rank * N / nr), n(N / nr), E(w.experts) {
        check(ff % 32 == 0 && n % 4 == 0, "row shard violates quantization/combine alignment");
        std::vector<uint8_t> a(E * ff * w.grow), b(a.size()), c(E * n * w.drow);
        for (int e = 0; e < E; ++e) {
            std::memcpy(a.data() + e * ff * w.grow, w.gate.data() + e * w.gate_bytes + ff0 * w.grow, ff * w.grow);
            std::memcpy(b.data() + e * ff * w.grow, w.up.data() + e * w.gate_bytes + ff0 * w.grow, ff * w.grow);
            std::memcpy(c.data() + e * n * w.drow, w.down.data() + e * w.down_bytes + n0 * w.drow, n * w.drow);
        }
        wg = d.upload(a); wu = d.upload(b); wd = d.upload(c);
        xq = d.alloc<uint8_t>(MT * XQ); hq_local = d.alloc<uint8_t>(E * MT * ff / 32 * 36); hq_full = d.alloc<uint8_t>(E * MT * HQ);
        g = d.alloc<float>(E * MT * ff); u = d.alloc<float>(E * MT * ff);
        parts_et = d.alloc<float>(E * MT * n); parts_te = d.alloc<float>(E * MT * n);
        out = d.alloc<float>(MT * n); full_out = d.alloc<float>(MT * N);
        std::vector<float> rw(E * MT);
        for (int t = 0; t < MT; ++t) for (int e = 0; e < E; ++e) rw[t * E + e] = float(e + 1) / float(E * (E + 1) / 2);
        weights = d.upload(rw);
        published = d.event(); finished = d.event(); gu0 = d.event(true); gu1 = d.event(true); dn0 = d.event(true); dn1 = d.event(true);
    }
    void gate_up(const Weights& w, int t) {
        d.select();
        for (int e = 0; e < E; ++e) {
            K::native_mmvq(w.gu_type, wg + e * ff * w.grow, xq, g + e * t * ff, N, ff, t, d.stream);
            K::native_mmvq(w.gu_type, wu + e * ff * w.grow, xq, u + e * t * ff, N, ff, t, d.stream);
            K::native_swiglu_quantize_q8_1(g + e * t * ff, u + e * t * ff, hq_local + e * t * ff / 32 * 36, ff, t, d.stream);
        }
    }
    void down_combine(const Weights& w, int t, bool full_reference = false) {
        d.select(); const auto* hidden = full_reference ? hq_local : hq_full;
        for (int e = 0; e < E; ++e)
            K::native_mmvq(w.down_type, wd + e * n * w.drow, hidden + e * t * HQ, parts_et + e * t * n, FF, n, t, d.stream);
        transpose_parts<<<blocks(E * t * n),256,0,d.stream>>>(parts_et, parts_te, E, t, n); cu(cudaPeekAtLastError());
        K::native_moe_combine_multi(parts_te, weights, nullptr, out, n, E, t, d.stream);
    }
};
struct Production {
    Device& d;
    const Weights& w;
    const uint8_t* xq;
    float *parts, *out, *weights;
    uint8_t *blob, *scratch;
    unsigned long long* ptr;
    int32_t *start, *count, *dst, *tok;
    Graph graph;
    Production(Rank& r, const Weights& weights_) : d(r.d), w(weights_), xq(r.xq), weights(r.weights) {
        blob = d.upload(w.blobs); parts = d.alloc<float>(MT * w.experts * N); out = d.alloc<float>(MT * N);
        scratch = d.alloc<uint8_t>(K::native_expert_scratch_bytes(MT * w.experts, FF));
        std::vector<unsigned long long> p(w.experts);
        for (int e = 0; e < w.experts; ++e) p[e] = (unsigned long long) (blob + e * w.layout.bytes);
        ptr = d.upload(p); start = d.alloc<int32_t>(w.experts + 1); count = d.upload(std::vector<int32_t>{w.experts});
        dst = d.alloc<int32_t>(MT * w.experts); tok = d.alloc<int32_t>(MT * w.experts);
    }
    void prepare(int t) {
        std::vector<int32_t> st(w.experts + 1), di(w.experts * t), ti(di.size());
        for (int e = 0; e < w.experts; ++e) {
            st[e] = e * t;
            for (int j = 0; j < t; ++j) { di[e * t + j] = j * w.experts + e; ti[e * t + j] = j; }
        }
        st[w.experts] = w.experts * t; d.select();
        cu(cudaMemcpy(start, st.data(), st.size()*4, cudaMemcpyHostToDevice));
        cu(cudaMemcpy(dst, di.data(), di.size()*4, cudaMemcpyHostToDevice));
        cu(cudaMemcpy(tok, ti.data(), ti.size()*4, cudaMemcpyHostToDevice));
    }
    void run(int t) {
        K::native_expert_grouped(w.layout, ptr, start, count, dst, tok, w.experts * t, w.experts * t, xq, scratch, parts, d.stream);
        K::native_moe_combine_multi(parts, weights, nullptr, out, N, w.experts, t, d.stream);
    }
};
struct Tp {
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    Pinned input, hidden, output;
    int t, E;
    cudaEvent_t input_ready;
    Tp(const std::vector<int>& devs, const Weights& w, int rows) : devices(devs), input(MT*XQ,devs),
        hidden(w.experts*MT*HQ,devs), output(MT*N*4,devs), t(rows), E(w.experts) {
        for (size_t i = 0; i < devs.size(); ++i) ranks.emplace_back(std::make_unique<Rank>(devs[i], (int)i, (int)devs.size(), w));
        input_ready = ranks[0]->d.event();
        for (auto& r : ranks) {
            // Initialize every input read during lazy warmup/capture to valid zero blocks.
            r->d.select(); cu(cudaMemset(r->xq,0,MT*XQ)); cu(cudaMemset(r->hq_full,0,E*MT*HQ));
            r->gu.make(r->d,[&]{ r->gate_up(w,t); });
            r->dn.make(r->d,[&]{ r->down_combine(w,t); });
        }
    }
    void set_input(const std::vector<uint8_t>& q) {
        auto& r = *ranks[0]; r.d.select(); cu(cudaMemcpy(r.xq,q.data(),q.size(),cudaMemcpyHostToDevice));
    }
    Sample run() {
        const auto begin = Clock::now();
        auto& source = *ranks[0]; source.d.select();
        put(source.xq,input.alias[0],(int)XQ,(int)XQ,0,t,source.d.stream); cu(cudaEventRecord(input_ready,source.d.stream));
        for (size_t i = 0; i < ranks.size(); ++i) {
            auto& r = *ranks[i]; r.d.select();
            if (i) { cu(cudaStreamWaitEvent(r.d.stream,input_ready,0)); get(input.alias[i],r.xq,t*XQ,r.d.stream); }
            cu(cudaEventRecord(r.gu0,r.d.stream)); r.gu.launch(r.d); cu(cudaEventRecord(r.gu1,r.d.stream));
            put(r.hq_local,hidden.alias[i],r.ff/32*36,(int)HQ,r.ff0/32*36,E*t,r.d.stream);
            cu(cudaEventRecord(r.published,r.d.stream));
        }
        for (size_t i = 0; i < ranks.size(); ++i) {
            auto& r = *ranks[i]; r.d.select();
            for (auto& p : ranks) cu(cudaStreamWaitEvent(r.d.stream,p->published,0));
            get(hidden.alias[i],r.hq_full,E*t*HQ,r.d.stream);
            cu(cudaEventRecord(r.dn0,r.d.stream)); r.dn.launch(r.d); cu(cudaEventRecord(r.dn1,r.d.stream));
            put(r.out,output.alias[i],r.n*4,N*4,r.n0*4,t,r.d.stream); cu(cudaEventRecord(r.finished,r.d.stream));
        }
        for (size_t i = 0; i < ranks.size(); ++i) {
            auto& r = *ranks[i]; r.d.select();
            for (auto& p : ranks) cu(cudaStreamWaitEvent(r.d.stream,p->finished,0));
            get(output.alias[i],r.full_out,t*N*4,r.d.stream);
        }
        for (auto& r : ranks) r->d.sync();
        Sample s; s.wall = elapsed_us(begin);
        for (auto& r : ranks) { r->d.select(); const double x = event_us(r->gu0,r->gu1)+event_us(r->dn0,r->dn1); s.kernel_max=std::max(s.kernel_max,x); s.kernel_sum+=x; }
        return s;
    }
    ~Tp() { for (auto& r : ranks) { cudaSetDevice(r->d.id); cudaStreamSynchronize(r->d.stream); } }
};
Sample single(Device& d, Graph& g, cudaEvent_t a, cudaEvent_t b) {
    const auto begin=Clock::now(); d.select(); cu(cudaEventRecord(a,d.stream)); g.launch(d); cu(cudaEventRecord(b,d.stream)); d.sync();
    Sample s; s.wall=elapsed_us(begin); s.kernel_max=s.kernel_sum=event_us(a,b); return s;
}
std::vector<uint8_t> activation(Rank& r, int t, int seed) {
    std::vector<float> h(t*N); uint32_t state=(uint32_t)seed;
    if (seed) for(float& x:h) { state=1664525u*state+1013904223u; x=float(int(state>>16)-32768)/32768.0f; }
    float* x=r.full_out; r.d.select(); cu(cudaMemcpy(x,h.data(),h.size()*4,cudaMemcpyHostToDevice));
    K::native_quantize_q8_1(x,r.xq,N,t,r.d.stream); return read(r.d,r.xq,t*XQ);
}
void collective(const std::vector<int>& devs, int bytes, int t, int warm, int iters) {
    Pinned h((size_t)bytes*t,devs); std::vector<std::unique_ptr<Device>> ds;
    std::vector<uint32_t*> src,dst; std::vector<cudaEvent_t> ready,a,b; const int width=bytes/(int)devs.size();
    check(bytes%((int)devs.size()*4)==0,"collective payload must divide into words");
    for(size_t i=0;i<devs.size();++i) {
        ds.emplace_back(std::make_unique<Device>(devs[i])); auto& d=*ds.back();
        std::vector<uint32_t> v((size_t)width/4*t); for(size_t j=0;j<v.size();++j) v[j]=uint32_t(i*1048576+j);
        src.push_back(d.upload(v)); dst.push_back(d.alloc<uint32_t>((size_t)bytes/4*t));
        ready.push_back(d.event()); a.push_back(d.event(true)); b.push_back(d.event(true));
    }
    Timings results;
    for(int it=-warm;it<iters;++it) {
        const auto begin=Clock::now();
        for(size_t i=0;i<ds.size();++i) { auto& d=*ds[i]; d.select(); cu(cudaEventRecord(a[i],d.stream)); put(src[i],h.alias[i],width,bytes,(int)i*width,t,d.stream); cu(cudaEventRecord(ready[i],d.stream)); }
        for(size_t i=0;i<ds.size();++i) { auto& d=*ds[i]; d.select(); for(auto e:ready)cu(cudaStreamWaitEvent(d.stream,e,0)); get(h.alias[i],dst[i],(size_t)bytes*t,d.stream); cu(cudaEventRecord(b[i],d.stream)); }
        for(auto& d:ds)d->sync(); Sample s; s.wall=elapsed_us(begin);
        for(size_t i=0;i<ds.size();++i) { ds[i]->select(); const double z=event_us(a[i],b[i]); s.kernel_max=std::max(s.kernel_max,z); s.kernel_sum+=z; }
        if(it>=0)results.add(s);
    }
    bool ok=true;
    for(size_t i=0;i<ds.size();++i) {
        const auto v=read(*ds[i],dst[i],(size_t)bytes/4*t);
        for(int row=0;row<t;++row)for(size_t p=0;p<ds.size();++p)for(int j=0;j<width/4;++j)
            ok &= v[(size_t)row*bytes/4+p*width/4+j]==uint32_t(p*1048576+row*width/4+j);
    }
    std::printf("{\"kind\":\"host_allgather\",\"ranks\":%zu,\"bytes_per_row\":%d,\"T\":%d,\"bitwise\":%s,\"device_event_includes_wait_and_copy\":true,",devs.size(),bytes,t,ok?"true":"false");
    results.print("timing"); std::puts("}"); check(ok,"collective bitwise failure");
}
} // namespace

int main(int argc,char** argv) {
    try {
        std::string gguf; auto devices=std::vector<int>{0,1}; auto layers=std::vector<int>{3,6};
        auto ids=std::vector<int>{5,42,79,116,153,190,227,264,301,338};
        int warm=10,iters=100,cases=4; bool collective_only=false;
        for(int i=1;i<argc;++i) {
            const std::string a=argv[i];
            if(a=="--collective-only"){collective_only=true;continue;}
            check(i+1<argc,"missing argument for "+a); const std::string v=argv[++i];
            if(a=="--gguf")gguf=v; else if(a=="--devices")devices=ints(v); else if(a=="--layers")layers=ints(v);
            else if(a=="--experts")ids=ints(v); else if(a=="--warmup")warm=std::stoi(v); else if(a=="--iterations")iters=std::stoi(v);
            else if(a=="--parity-cases")cases=std::stoi(v); else throw std::runtime_error("unknown argument "+a);
        }
        check((devices.size()==2||devices.size()==4)&&std::set<int>(devices.begin(),devices.end()).size()==devices.size(),"devices must be two or four distinct GPUs");
        check(warm>=1&&iters>=3&&cases>=2&&ids.size()==10,"warmup>=1, iterations>=3, parity-cases>=2, exactly ten expert ids required");
        check(collective_only||!gguf.empty(),"--gguf is required unless --collective-only");
        int count=0; const auto status=cudaGetDeviceCount(&count);
        if(status!=cudaSuccess||*std::min_element(devices.begin(),devices.end())<0||*std::max_element(devices.begin(),devices.end())>=count) {
            std::fprintf(stderr,"SKIP: requested GPUs unavailable; count=%d CUDA=%s\n",count,cudaGetErrorString(status));return 77;
        }
        setvbuf(stdout,nullptr,_IONBF,0);
        std::printf("{\"kind\":\"configuration\",\"warmup\":%d,\"iterations\":%d,\"parity_cases\":%d,\"devices\":[",warm,iters,cases);
        for(size_t i=0;i<devices.size();++i)std::printf("%s%d",i?",":"",devices[i]);
        std::puts("],\"scope\":\"routed MoE only; fixed ten experts shared by all token rows; excludes router/shared/HC/attention\",\"timing\":\"interleaved graph replays; wall includes host submissions, mapped-host communication and all-rank synchronization; compute-event sums are not critical-path times\"}");
        for(int d:devices){cudaDeviceProp p{};cu(cudaGetDeviceProperties(&p,d));std::fprintf(stderr,"device=%d name=%s cc=%d%d SM=%d memory=%zu\n",d,p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem);}
        if(!collective_only) {
            strata::GgufModel model(strata::gguf_split_paths(gguf)); bool all_tp=true,all_prod=true;
            for(int layer:layers) {
                Weights w(model,layer,ids);
                for(int t:{1,2,3,8}) {
                    Rank ref(devices[0],0,1,w); ref.d.select();cu(cudaMemset(ref.xq,0,MT*XQ));
                    Production prod(ref,w);prod.prepare(t); Graph baseline,production;
                    baseline.make(ref.d,[&]{ref.gate_up(w,t);ref.down_combine(w,t,true);});
                    production.make(ref.d,[&]{prod.run(t);}); Tp tp(devices,w,t);
                    const auto ev0=ref.d.event(true),ev1=ref.d.event(true);
                    bool tp_ok=true,prod_ok=true;
                    for(int c=0;c<cases;++c) {
                        const auto q=activation(ref,t,c==0?0:17+layer*131+c*997+t);tp.set_input(q);
                        single(ref.d,baseline,ev0,ev1);const auto expected=read(ref.d,ref.out,t*N);const auto eh=read(ref.d,ref.hq_local,w.experts*t*HQ);
                        single(ref.d,production,ev0,ev1);const auto actual=read(ref.d,prod.out,t*N);tp.run();
                        const auto got=read(tp.ranks[0]->d,tp.ranks[0]->full_out,t*N);
                        const Diff td=diff(expected,got),pd=diff(actual,got);
                        const bool hidden_ok=std::memcmp(eh.data(),tp.hidden.host,eh.size())==0;
                        tp_ok &= td.different==0&&td.nonfinite==0&&hidden_ok;prod_ok &= pd.different==0&&pd.nonfinite==0;
                        std::printf("{\"kind\":\"parity\",\"layer\":%d,\"T\":%d,\"case\":%d,\"hidden_q8_bitwise\":%s,",layer,t,c,hidden_ok?"true":"false");
                        print_diff("tp_vs_decomposed",td);std::printf(",");print_diff("tp_vs_production",pd);std::puts("}");
                    }
                    // Fixed nonzero input for timing; model weights, captures and allocations are outside timing.
                    tp.set_input(activation(ref,t,8191+layer+t));Timings a,b,c;
                    for(int i=-warm;i<iters;++i)for(int k=0;k<3;++k) {
                        const int arm=((i+warm)%3+k)%3;Sample s;
                        if(arm==0)s=single(ref.d,production,ev0,ev1);else if(arm==1)s=single(ref.d,baseline,ev0,ev1);else s=tp.run();
                        if(i>=0){if(arm==0)a.add(s);else if(arm==1)b.add(s);else c.add(s);}
                    }
                    std::printf("{\"kind\":\"routed_moe\",\"layer\":%d,\"T\":%d,\"ranks\":%zu,\"GU_type\":%d,\"down_type\":%d,\"experts\":%d,\"production_cap_groups\":%d,\"tp_bitwise_decomposed\":%s,\"tp_bitwise_production\":%s,",layer,t,devices.size(),w.gu_type,w.down_type,w.experts,w.experts*t,tp_ok?"true":"false",prod_ok?"true":"false");
                    a.print("production_fused");std::printf(",");b.print("decomposed_single_gpu");std::printf(",");c.print("row_tp");std::puts("}");
                    all_tp &= tp_ok;all_prod &= prod_ok;
                }
            }
            std::printf("{\"kind\":\"parity_summary\",\"all_tp_bitwise_decomposed\":%s,\"all_tp_bitwise_production\":%s,\"production_exact_candidate\":%s}\n",all_tp?"true":"false",all_prod?"true":"false",all_tp&&all_prod?"true":"false");
            check(all_tp,"TP differs from the complete-dot decomposed reference");
        }
        for(int bytes:{6912,7200,10240,24576,40960})for(int t:{1,2,3,8})collective(devices,bytes,t,warm,iters);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
