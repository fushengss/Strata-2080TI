// batch_read.hpp: the pieces of a --batch prompt read, where the concurrent read runs (synthetic: no model, no GPU).
#include "strata/program/batch_read.hpp"

#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

using strata::program::batch_read_piece;
using strata::program::concurrent_read_off;
using strata::program::ConcurrentReadSetup;
using strata::program::piped_decode_budget_ms;
using strata::program::piped_read_lead_in;
using strata::program::piped_read_piece;
using strata::program::PipedReadHistory;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // no --batch, or the pipelined groups: one run
    CHECK(batch_read_piece(false, false, false, false, 2048, 8) == 0);
    CHECK(batch_read_piece(true, true, true, false, 2048, 8) == 0);
    // one GPU: a chunk per piece, whether or not a slot decodes (the yield point is the chunk boundary)
    CHECK(batch_read_piece(true, false, false, false, 2048, 8) == 2048);
    CHECK(batch_read_piece(true, false, false, true, 2048, 8) == 2048);
    // layer split with slots decoding: a chunk per piece as before
    CHECK(batch_read_piece(true, false, true, true, 2048, 8) == 2048);
    // layer split, nobody decoding: pieces of 8 chunks - the BYIELD of a shorter request can be taken (it was one run)
    CHECK(batch_read_piece(true, false, true, false, 2048, 8) == 8 * 2048);
    CHECK(batch_read_piece(true, false, true, false, 1024, 4) == 4096);
    // STRATA_BATCH_YIELD_CHUNKS=0 keeps the old single run
    CHECK(batch_read_piece(true, false, true, false, 2048, 0) == 0);
    // a degenerate chunk never gives a zero piece
    CHECK(batch_read_piece(true, false, false, false, 0, 8) == 1);
    // a piece is always a multiple of the chunk
    for (int64_t c : {256, 1000, 2048, 4096})
        for (int64_t k : {1, 3, 8, 16}) CHECK(batch_read_piece(true, false, true, false, c, k) % c == 0);
    // concurrent_read_off (STRATA_PIPED_CONCURRENT): runs with nothing set; each setting alone keeps it off, named
    CHECK(concurrent_read_off({}) == nullptr);
    bool ConcurrentReadSetup::*const each[] = {
        &ConcurrentReadSetup::prefill_help, &ConcurrentReadSetup::split_mtp_batch, &ConcurrentReadSetup::fs_slots,
        &ConcurrentReadSetup::pcie_dma,     &ConcurrentReadSetup::helper_caches,   &ConcurrentReadSetup::staged_source,
        &ConcurrentReadSetup::other_source};
    const char* named[std::size(each)] = {};
    auto same = [](const char* a, const char* b) { return a != nullptr && b != nullptr && std::strcmp(a, b) == 0; };
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        s.*each[i] = true;
        named[i] = concurrent_read_off(s);
        CHECK(named[i] != nullptr);
        for (size_t j = 0; j < i; ++j) CHECK(!same(named[i], named[j]));   // a reason of its own
    }
    CHECK(same(named[0], "STRATA_PREFILL_HELP"));
    CHECK(same(named[3], "--pcie-mode dma"));
    // with several, the first in the struct's order is named
    for (size_t i = 0; i < std::size(each); ++i) {
        ConcurrentReadSetup s;
        for (size_t j = i; j < std::size(each); ++j) s.*each[j] = true;
        CHECK(same(concurrent_read_off(s), named[i]));
    }
    // piped_read_piece (#1906): off -> one run; whole chunks, at least one, at most 32
    CHECK(piped_read_piece(0, 8192) == 0);
    CHECK(piped_read_piece(-5, 8192) == 0);
    CHECK(piped_read_piece(32768, 8192) == 32768);
    CHECK(piped_read_piece(20000, 8192) == 16384);      // rounded down to whole chunks
    CHECK(piped_read_piece(4096, 8192) == 8192);        // never below one chunk
    CHECK(piped_read_piece(1 << 30, 8192) == 32 * 8192);
    CHECK(piped_read_piece(5000, 0) == 32);             // chunk 0 guarded as 1 (and the cap applies)
    for (int64_t c : {1024, 2048, 6144, 8192})
        for (int64_t p : {1, 1000, 9000, 50000, 300000}) CHECK(piped_read_piece(p, c) % c == 0);   // the header's rule
    // Decode-only keeps a solo prompt whole and rechecks at every drained boundary. The simulated slot counts
    // are those left after each decode budget (or the lead-in before the first piece), not GPU work itself.
    auto spans = [](bool decode_only, std::vector<int> active) {
        std::vector<int64_t> out;
        int64_t remaining = 61000;
        size_t boundary = 0;
        while (remaining > 0) {
            const int count = active[std::min(boundary, active.size() - 1)];
            const int64_t piece = piped_read_piece(16384, 8192, decode_only, count > 0);
            const int64_t n = piece > 0 ? std::min(piece, remaining) : remaining;
            out.push_back(n);
            remaining -= n;
            ++boundary;
        }
        return out;
    };
    CHECK(spans(true, {0}) == std::vector<int64_t>({61000}));   // no slots, or the lead-in finished them all
    CHECK(spans(true, {3, 2, 1, 1}) == std::vector<int64_t>({16384, 16384, 16384, 11848}));
    CHECK(spans(true, {3, 0}) == std::vector<int64_t>({16384, 44616}));   // merge the entire idle remainder
    CHECK(spans(true, {3, 1, 0}) == std::vector<int64_t>({16384, 16384, 28232}));
    CHECK(spans(false, {0}) == std::vector<int64_t>({16384, 16384, 16384, 11848}));   // original PIECE behavior
    CHECK(piped_read_piece(0, 8192, true, true) == 0);   // the guard alone never enables slicing
    CHECK(piped_read_piece(20000, 8192, true, true) == 16384);   // the original chunk grid
    // A completed mixed read may give the next read a lead-in. An idle remainder or any early exit clears it,
    // even if a later admission arrives immediately. Legacy mode keeps its prior history semantics.
    for (bool complete : {false, true}) {
        for (bool active : {false, true}) {
            PipedReadHistory h{true, 900.0};
            h.finish(true, complete, active);
            CHECK(h.have_previous == (complete && active));
            CHECK(h.piece_ms == (complete && active ? 900.0 : 0.0));
            PipedReadHistory legacy{true, 900.0};
            legacy.finish(false, complete, active);
            CHECK(legacy.have_previous && legacy.piece_ms == 900.0);
        }
    }
    PipedReadHistory idle;
    idle.finish(true, true, true);
    CHECK(!idle.have_previous && idle.piece_ms == 0.0);   // finishing alone cannot invent a prior read
    // piped_decode_budget_ms: share x piece, capped, floored; negatives are 0
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 400.0, 0.0) == 400.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.35, 3000.0, 0.0) == 700.0);
    CHECK(piped_decode_budget_ms(2000.0, 0.0, 400.0, 150.0) == 150.0);
    CHECK(piped_decode_budget_ms(2000.0, 2.0, 0.0, 0.0) == 4000.0);   // no cap
    CHECK(piped_decode_budget_ms(-5.0, -1.0, 400.0, -3.0) == 0.0);
    // piped_read_lead_in: on, beside decoding slots, within 1000 ms + the budget of the previous piecewise read's end
    CHECK(piped_read_lead_in(true, true, 0.0, 400.0));
    CHECK(piped_read_lead_in(true, true, 1400.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, 1401.0, 400.0));
    CHECK(!piped_read_lead_in(true, true, -1.0, 400.0));
    CHECK(!piped_read_lead_in(false, true, 0.0, 400.0));
    CHECK(!piped_read_lead_in(true, false, 0.0, 400.0));
    if (fails == 0) std::printf("batch_read_test: ok\n");
    return fails == 0 ? 0 : 1;
}
