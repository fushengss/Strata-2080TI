// bench/results/2026-11-14-park-copy/warmalloc.cpp - keeps freed large blocks warm for the next allocation of the same size.
// Build:  g++ -O2 -std=c++17 -shared -fPIC -o libwarmalloc.so warmalloc.cpp -lpthread
// Use:    LD_PRELOAD=/path/libwarmalloc.so  (the engine's conversation cache frees and allocates 16 MiB segments and ~15 MB checkpoint
//         vectors on every park; fresh pages cost 1.76 GB/s of page faults on the 8x2080Ti box, pages that are already in run 5-7x faster)
// Env:    STRATA_BLOCK_CACHE_MIB          cache cap in MiB (default 8192; 0 turns the cache off)
//         STRATA_BLOCK_CACHE_PREWARM_MIB  fault this much in at start, on a background thread (default 0)
//         STRATA_BLOCK_CACHE_STATS=1      one line on stderr every 60 s
// It replaces operator new / delete.  Blocks of 1 MiB or more are taken from malloc in whole-MiB classes; operator delete keeps up to the cap
// of them (any block of 1 MiB or more, whoever allocated it: the class is the usable size rounded down) and hands them out again.  Everything
// else goes straight to malloc / free.  A block handed out again is not cleared, as with plain operator new.
#include <malloc.h>
#include <pthread.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>

namespace {
constexpr size_t kMiB = 1u << 20;
constexpr size_t kMinBlock = kMiB;           // smaller blocks are not cached
constexpr size_t kMaxClass = 1024;           // MiB: larger blocks go straight to malloc / free

struct Node { Node* next; };
struct Cache {
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    Node* head[kMaxClass + 1] = {};
    size_t held = 0;                         // bytes
    std::atomic<unsigned long> hits{0}, misses{0}, gives{0}, dropped{0};
};
Cache& cache() { static Cache* c = new Cache; return *c; }   // leaked on purpose: threads may free after statics are gone

size_t cap_bytes() {
    static const size_t v = [] {
        const char* e = std::getenv("STRATA_BLOCK_CACHE_MIB");
        const long long m = e != nullptr ? std::atoll(e) : 8192;
        return m > 0 ? (size_t) m * kMiB : (size_t) 0;
    }();
    return v;
}

void* take(size_t cls) {
    Cache& c = cache();
    pthread_mutex_lock(&c.mu);
    Node* n = c.head[cls];
    if (n != nullptr) { c.head[cls] = n->next; c.held -= cls * kMiB; }
    pthread_mutex_unlock(&c.mu);
    if (n != nullptr) c.hits.fetch_add(1, std::memory_order_relaxed);
    else c.misses.fetch_add(1, std::memory_order_relaxed);
    return n;
}

bool give(void* p, size_t cls) {
    Cache& c = cache();
    bool kept = false;
    pthread_mutex_lock(&c.mu);
    if (c.held + cls * kMiB <= cap_bytes()) {
        Node* n = static_cast<Node*>(p);
        n->next = c.head[cls];
        c.head[cls] = n;
        c.held += cls * kMiB;
        kept = true;
    }
    pthread_mutex_unlock(&c.mu);
    (kept ? c.gives : c.dropped).fetch_add(1, std::memory_order_relaxed);
    return kept;
}

void* big_alloc(size_t n) {
    const size_t cls = (n + kMiB - 1) / kMiB;
    if (cap_bytes() != 0)
        if (void* p = take(cls)) return p;
    return std::malloc(cls * kMiB);
}

inline void* plain_alloc(size_t n) {
    if (n >= kMinBlock && n <= kMaxClass * kMiB) return big_alloc(n);
    return std::malloc(n != 0 ? n : 1);
}

inline void plain_free(void* p) {
    if (p == nullptr) return;
    if (cap_bytes() != 0) {
        const size_t u = malloc_usable_size(p);
        if (u >= kMinBlock) {
            const size_t cls = u / kMiB;
            if (cls <= kMaxClass && give(p, cls)) return;
        }
    }
    std::free(p);
}

void prewarm() {
    const char* e = std::getenv("STRATA_BLOCK_CACHE_PREWARM_MIB");
    const long long want = e != nullptr ? std::atoll(e) : 0;
    if (want <= 0 || cap_bytes() == 0) return;
    std::thread([want] {
        const size_t step = 16;               // MiB: the size of a conversation K/V segment
        for (long long done = 0; done + (long long) step <= want; done += (long long) step) {
            void* p = std::malloc(step * kMiB);
            if (p == nullptr) return;
            for (size_t o = 0; o < step * kMiB; o += 4096) static_cast<volatile unsigned char*>(p)[o] = 0;
            if (!give(p, step)) { std::free(p); return; }
        }
    }).detach();
}

void stats_thread() {
    const char* e = std::getenv("STRATA_BLOCK_CACHE_STATS");
    if (e == nullptr || e[0] != '1') return;
    std::thread([] {
        for (;;) {
            sleep(60);
            Cache& c = cache();
            std::fprintf(stderr, "strata block cache: %lu hits %lu misses, %lu kept %lu dropped, %zu MiB held (cap %zu MiB)\n",
                         c.hits.load(), c.misses.load(), c.gives.load(), c.dropped.load(), c.held / kMiB, cap_bytes() / kMiB);
        }
    }).detach();
}

struct Init { Init() { (void) cache(); prewarm(); stats_thread(); } } init;
}  // namespace

void* operator new(size_t n) {
    void* p = plain_alloc(n);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}
void* operator new[](size_t n) { return operator new(n); }
void* operator new(size_t n, const std::nothrow_t&) noexcept { return plain_alloc(n); }
void* operator new[](size_t n, const std::nothrow_t&) noexcept { return plain_alloc(n); }
void operator delete(void* p) noexcept { plain_free(p); }
void operator delete[](void* p) noexcept { plain_free(p); }
void operator delete(void* p, size_t) noexcept { plain_free(p); }
void operator delete[](void* p, size_t) noexcept { plain_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { plain_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { plain_free(p); }

// over-aligned blocks: plain aligned allocations, not cached
void* operator new(size_t n, std::align_val_t a) {
    void* p = nullptr;
    if (posix_memalign(&p, (size_t) a < sizeof(void*) ? sizeof(void*) : (size_t) a, n != 0 ? n : 1) != 0) throw std::bad_alloc();
    return p;
}
void* operator new[](size_t n, std::align_val_t a) { return operator new(n, a); }
void* operator new(size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
    void* p = nullptr;
    return posix_memalign(&p, (size_t) a < sizeof(void*) ? sizeof(void*) : (size_t) a, n != 0 ? n : 1) == 0 ? p : nullptr;
}
void* operator new[](size_t n, std::align_val_t a, const std::nothrow_t& t) noexcept { return operator new(n, a, t); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
