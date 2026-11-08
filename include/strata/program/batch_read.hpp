// include/strata/program/batch_read.hpp - how --batch reads a prompt in pieces (header-only: it is arithmetic).
//
// A prompt read with --batch stops between two pieces to (a) let the slots decode and (b) look for a BYIELD (a
// shorter request is waiting).  The pieces always end on a multiple of the prompt chunk, so they read exactly the
// chunks a single run would.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::program {

/// Tokens of the next piece of a --batch prompt read, or 0 for "the whole rest in one run".
///   beside_slots: some slot is decoding (its windows run between the pieces)
///   layer_split / piped: the read goes through a layer split's stages / the pipelined groups
///   chunks_per_piece: for a layer split with no slot decoding, the pieces' length in chunks (0: one run)
/// A layer split reads its stages as a pipeline over one run's chunks, so a piece costs a pipeline fill; with slots
/// decoding the pieces are single chunks anyway.  With no slot decoding a piece is `chunks_per_piece` chunks: long
/// enough to keep the pipeline full, short enough that a BYIELD is taken within a few chunks (it used to wait for the
/// whole prompt, and the long prompt then held its slot: "ERR BGEN: no such free slot" for the request that should
/// have taken over).
inline int64_t batch_read_piece(bool batch, bool piped, bool layer_split, bool beside_slots, int64_t chunk,
                                int64_t chunks_per_piece) {
    if (!batch || piped) return 0;
    chunk = std::max<int64_t>(chunk, 1);
    if (!layer_split || beside_slots) return chunk;
    return chunks_per_piece > 0 ? chunk * chunks_per_piece : 0;
}

/// What a configuration has that keeps the concurrent read (STRATA_PIPED_CONCURRENT=1) off.
struct ConcurrentReadSetup {
    bool prefill_help = false;      ///< STRATA_PREFILL_HELP: the idle stage's help allocates on the read's thread
    bool split_mtp_batch = false;   ///< legacy path excludes STRATA_SPLIT_MTP_BATCH; the resident extension checks it separately
    bool fs_slots = false;          ///< STRATA_FS_SLOTS (Foresight swap space): serving a doorbell makes a CUDA call
    bool pcie_dma = false;          ///< --pcie-mode dma: the same (its copies)
    bool helper_caches = false;     ///< --expert-cache-device1..3 (and --remote-expert-opt on them), a prompt source of
                                    ///< its own; --remote-expert-opt alone is inert on a plain layer split
    bool staged_source = false;     ///< the GGUF source stages its blobs (in place, unbuffered, STRATA_IO_PREFETCH):
                                    ///< it recycles the buffers it hands out, one thread at a time
    bool other_source = false;      ///< an expert source that is neither the GGUF source nor the arena
};

/// Why the concurrent read stays off with `s` (the first reason, for the log line), or nullptr when it can run.
inline const char* concurrent_read_off(const ConcurrentReadSetup& s) {
    if (s.prefill_help) return "STRATA_PREFILL_HELP";
    if (s.split_mtp_batch) return "STRATA_SPLIT_MTP_BATCH";
    if (s.fs_slots) return "STRATA_FS_SLOTS";
    if (s.pcie_dma) return "--pcie-mode dma";
    if (s.helper_caches) return "the helper caches (--expert-cache-device1..3)";
    if (s.staged_source)
        return "an expert source that stages its blobs (the GGUF in place, unbuffered, STRATA_IO_PREFETCH)";
    if (s.other_source) return "this expert source";
    return nullptr;
}

/// STRATA_PIPED_CONCURRENT_MTP: each read rechecks live ownership/residency, after the existing pipeline drain.
/// The old configuration guards remain except split MTP and a known file source whose consumer is only prefill:
/// resident windows skip expert dispatch, and the lookahead must be idle. CUDA allocation barriers still require PLE publication
/// before launching a resident graph, rather than by a later host service call.
struct ResidentConcurrentReadSetup {
    bool cuda_backend = false, piped_mtp = false, split_mtp_batch = false, active_slots = false;
    bool private_buffers = false, all_resident = false, no_loans = false, draft_resident = false;
    bool stable_tier = false, text_only = false, single_prompt = false, lookahead_idle = false;
};

inline const char* resident_concurrent_read_off(const ResidentConcurrentReadSetup& s) {
    if (!s.cuda_backend) return "backend";
    if (!s.piped_mtp) return "not_piped_mtp";
    if (!s.split_mtp_batch) return "split_mtp_disabled";
    if (!s.active_slots) return "no_active_slots";
    if (!s.private_buffers) return "shared_prompt_buffers";
    if (!s.all_resident) return "nonresident_stage";
    if (!s.no_loans) return "active_loan";
    if (!s.draft_resident) return "draft_weights_not_resident";
    if (!s.stable_tier) return "mutable_tier_or_kv";
    if (!s.text_only) return "image_input";
    if (!s.single_prompt) return "co_read";
    if (!s.lookahead_idle) return "decode_lookahead_busy";
    return nullptr;
}

/// Tokens of the next piece of a prompt read on the pipelined slot groups (--batch-groups > 1 on a layer split, #1906),
/// or 0 for "the whole rest in one run" (the opt-in is off).
///   piece_tokens: STRATA_PIPED_READ_PIECE, rounded down to whole chunks (at least one, at most kPipedMaxChunks)
/// A piece is one pipelined run over its chunks, so the stages stay overlapped inside it and only its end drains the
/// pipeline.  At a piece's end the decoding slots decode, and a waiting shorter request can take over (BYIELD), as
/// between the pieces of a read off the pipeline.
inline constexpr int64_t kPipedMaxChunks = 32;
inline int64_t piped_read_piece(int64_t piece_tokens, int64_t chunk) {
    if (piece_tokens <= 0) return 0;
    chunk = std::max<int64_t>(chunk, 1);
    return std::clamp<int64_t>(piece_tokens / chunk, 1, kPipedMaxChunks) * chunk;
}

/// STRATA_PIPED_READ_DECODE_ONLY: leave an idle read whole, also after the last decoding slot finishes.
/// Re-evaluate at a drained piece boundary; the prompt's planned chunk stays the same.
inline int64_t piped_read_piece(int64_t piece_tokens, int64_t chunk, bool decode_only, bool beside_slots) {
    return decode_only && !beside_slots ? 0 : piped_read_piece(piece_tokens, chunk);
}

/// The lead-in belongs to a completed read beside still-active slots. In decode-only mode an idle remainder,
/// a cancellation, a give-way or a failed read must not leave a budget for a later, unrelated admission.
struct PipedReadHistory {
    bool have_previous = false;
    double piece_ms = 0.0;

    void finish(bool decode_only, bool completed, bool beside_slots) {
        if (decode_only && (!completed || !beside_slots)) {
            have_previous = false;
            piece_ms = 0.0;
        }
    }
};

/// The slots' decode time between two pieces: share x the piece's time (STRATA_BATCH_DECODE_SHARE, the rule the
/// one-GPU path uses between chunks), at most cap_ms (STRATA_PIPED_DECODE_MAX_MS; <= 0: no cap), at least floor_ms
/// (STRATA_PIPED_DECODE_MIN_MS).
inline double piped_decode_budget_ms(double piece_ms, double share, double cap_ms, double floor_ms) {
    double b = std::max(0.0, share) * std::max(0.0, piece_ms);
    if (cap_ms > 0.0) b = std::min(b, cap_ms);
    return std::max(b, std::max(0.0, floor_ms));
}

/// Prompts arriving back to back: a piecewise read that starts within kPipedLeadInWindowMs + budget_ms of the
/// previous piecewise read's end first gives the slots one decode budget (otherwise they stop for both reads).
///   ms_since_prev_end: < 0 when there was no piecewise read yet
inline constexpr double kPipedLeadInWindowMs = 1000.0;
inline bool piped_read_lead_in(bool enabled, bool beside_slots, double ms_since_prev_end, double budget_ms) {
    return enabled && beside_slots && ms_since_prev_end >= 0.0 &&
           ms_since_prev_end <= kPipedLeadInWindowMs + std::max(0.0, budget_ms);
}

}  // namespace strata::program
