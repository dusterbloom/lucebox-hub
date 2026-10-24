// KvFlashSlotPager — Track C, C1.1: N KVFlash sessions on one GPU
// (C0 R1, trackc/C0.md). The KV rows stay ONE allocation of N x Q x 64 rows
// plus a scratch block; slot s owns rows [sQ*64, (s+1)Q*64) as a private free
// list. A slot pager is the existing KvFlashPager confined to its slice
// (KvFlashConfig::row_base), not a fork: victims, pins, truncate, serialize
// and reset are the single-session code, so they cannot cross a slice.
//
// Also here, pure and CPU-testable: slice geometry, host-budget admission
// (C0 section 4, R5) and the per-step plan (C0 R3: pins, out before in, one
// fence, scorer last). No device code.
//
// Not in C1.1 (later steps): the R3(a) pending-free list that delays reuse of
// a block until its D2H completes, the arena stream shared by all slots, the
// block-table column writer, resident_prefix_covers (R2).

#pragma once

#include "kvflash_pager.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

// ── Slice geometry ─────────────────────────────────────────────────────

struct KvFlashSlice {
    int     slot = 0;
    int     block_lo = 0, block_hi = 0;   // [lo, hi) in arena blocks (block = chunk = chunk_tokens)
    int64_t row_lo = 0, row_hi = 0;       // [lo, hi) in arena rows
};

inline KvFlashSlice kvflash_slice(int slot, int quota_chunks, int chunk_tokens = 64) {
    KvFlashSlice s;
    s.slot = slot;
    s.block_lo = slot * quota_chunks;
    s.block_hi = s.block_lo + quota_chunks;
    s.row_lo = (int64_t)s.block_lo * chunk_tokens;
    s.row_hi = (int64_t)s.block_hi * chunk_tokens;
    return s;
}

// Rows of the shared KV allocation: every slice plus one scratch block.
inline int64_t kvflash_arena_rows(int n_slots, int quota_chunks, int chunk_tokens = 64) {
    return ((int64_t)n_slots * quota_chunks + 1) * chunk_tokens;
}

// First row of the scratch block (owned by no slot).
inline int64_t kvflash_scratch_row(int n_slots, int quota_chunks, int chunk_tokens = 64) {
    return (int64_t)n_slots * quota_chunks * chunk_tokens;
}

// Owner of an arena row, -1 for the scratch block and out-of-range rows.
inline int kvflash_slot_of_row(int64_t row, int n_slots, int quota_chunks, int chunk_tokens = 64) {
    if (row < 0 || row >= kvflash_scratch_row(n_slots, quota_chunks, chunk_tokens)) return -1;
    return (int)(row / ((int64_t)quota_chunks * chunk_tokens));
}

// A quota must leave an evictable block (the pager's own floor).
inline bool kvflash_quota_ok(int quota_chunks, const KvFlashConfig & cfg = {}) {
    return quota_chunks * cfg.chunk_tokens >= KvFlashPager::min_pool_tokens(cfg);
}

// Quota in chunks for a pool of `pool_tokens`; 0 when it is not a whole number of chunks.
inline int kvflash_quota_chunks(int pool_tokens, int chunk_tokens = 64) {
    return pool_tokens > 0 && pool_tokens % chunk_tokens == 0 ? pool_tokens / chunk_tokens : 0;
}

// ── The slot pager ─────────────────────────────────────────────────────

class KvFlashSlotPager : public KvFlashPager {
public:
    // Attach slot `slot` of `n_slots` with a quota of `quota_chunks` chunks.
    // cfg.pool_tokens and cfg.row_base are derived; the K/V tensors are the
    // shared arena (>= kvflash_arena_rows rows, or none for the no-tensor path).
    bool attach_slice(int slot, int n_slots, int quota_chunks, KvFlashConfig cfg,
                      const std::vector<ggml_tensor *> & attn_k,
                      const std::vector<ggml_tensor *> & attn_v) {
        if (slot < 0 || slot >= n_slots || !kvflash_quota_ok(quota_chunks, cfg)) return false;
        slice_ = kvflash_slice(slot, quota_chunks, cfg.chunk_tokens);
        cfg.pool_tokens = quota_chunks * cfg.chunk_tokens;
        cfg.row_base = (int)slice_.row_lo;
        // The scratch block must exist too, or the arena is not the one the
        // table was sized for.
        if (!attn_k.empty() && (int64_t)attn_k[0]->ne[1] < kvflash_arena_rows(n_slots, quota_chunks, cfg.chunk_tokens))
            return false;
        return attach(cfg, attn_k, attn_v);
    }

    const KvFlashSlice & slice() const { return slice_; }

    // Arena block holding chunk c (the block-table entry), -1 if not resident.
    int global_block_of(int c) const {
        const int b = block_of(c);
        return b < 0 ? -1 : slice_.block_lo + b;
    }

private:
    KvFlashSlice slice_;
};

// ── Admission (C0 section 4, R1, R5) ───────────────────────────────────

enum class KvFlashAdmit { Ok, BusySlot, BusyHost, PromptTooLong };

struct KvFlashAdmitInput {
    bool    slot_free = false;               // slot, its slice and its state slab are all free
    int64_t prompt_tokens = 0, max_new_tokens = 0, max_ctx = 0;
    int     chunk_tokens = 64;
    int64_t chunk_bytes = 0;                 // host backing per chunk
    int64_t ladder_bytes = 0;                // rungs x live state this slot will hold
    int64_t transient_reserve_bytes = 0;     // global: max(scorer staging, serialize)
    int64_t host_used_bytes = 0, host_budget_bytes = 0;
};

// Host bytes this request reserves: backing for every chunk of prompt + new
// tokens plus its ladder. ponytail: backing is the upper bound (every chunk
// evicted at least once over a long run, R5), not chunks - Q; the ceiling is
// admitting fewer sessions than fit when most never evict.
inline int64_t kvflash_host_need_bytes(const KvFlashAdmitInput & in) {
    const int64_t tokens = in.prompt_tokens + in.max_new_tokens;
    const int64_t chunks = (tokens + in.chunk_tokens - 1) / in.chunk_tokens;
    return chunks * in.chunk_bytes + in.ladder_bytes;
}

// A permanent refusal outranks a transient busy.
inline KvFlashAdmit kvflash_admit(const KvFlashAdmitInput & in) {
    if (in.prompt_tokens > in.max_ctx) return KvFlashAdmit::PromptTooLong;
    if (!in.slot_free) return KvFlashAdmit::BusySlot;
    if (in.host_used_bytes + kvflash_host_need_bytes(in) + in.transient_reserve_bytes > in.host_budget_bytes)
        return KvFlashAdmit::BusyHost;
    return KvFlashAdmit::Ok;
}

// ── Step plan (C0 R3) ──────────────────────────────────────────────────

enum class KvFlashOpKind { Pin, PageOut, PageIn, AllocSpan, TableDelta, Fence, Score };

struct KvFlashOp {
    KvFlashOpKind kind;
    int slot;
    int arg;      // chunk for Pin/PageOut/PageIn, tokens for AllocSpan, unused otherwise
};

struct KvFlashStepSlot {
    int slot = 0;                       // unique within a step
    std::vector<int> pin_chunks;        // chunks holding this step's rows, pinned to step end
    std::vector<int> page_out, page_in;
    int  alloc_tokens = 0;              // rows this step appends (prefill ubatch or decode row)
    bool wants_score = false;
};

// Order for one engine step: pin every in-flight chunk, then all page-outs,
// then all page-ins, then alloc_span and the synchronous table delta per slot,
// then ONE fence (only if page work was queued), then at most one scorer call.
// Slots run in ascending id; idle slots emit nothing. Slots whose scorer call
// is pushed to a later step are listed in `deferred_score`.
inline std::vector<KvFlashOp> kvflash_plan_step(std::vector<KvFlashStepSlot> slots,
                                                std::vector<int> * deferred_score = nullptr) {
    std::sort(slots.begin(), slots.end(),
              [](const KvFlashStepSlot & a, const KvFlashStepSlot & b) { return a.slot < b.slot; });
    std::vector<KvFlashOp> ops;
    bool page_work = false;
    for (const auto & s : slots) for (int c : s.pin_chunks) ops.push_back({KvFlashOpKind::Pin, s.slot, c});
    for (const auto & s : slots) for (int c : s.page_out) ops.push_back({KvFlashOpKind::PageOut, s.slot, c});
    for (const auto & s : slots) for (int c : s.page_in)  ops.push_back({KvFlashOpKind::PageIn, s.slot, c});
    for (const auto & s : slots) page_work = page_work || !s.page_out.empty() || !s.page_in.empty();
    for (const auto & s : slots)
        if (s.alloc_tokens > 0) ops.push_back({KvFlashOpKind::AllocSpan, s.slot, s.alloc_tokens});
    for (const auto & s : slots)
        if (s.alloc_tokens > 0 || !s.page_out.empty() || !s.page_in.empty())
            ops.push_back({KvFlashOpKind::TableDelta, s.slot, 0});
    if (page_work) ops.push_back({KvFlashOpKind::Fence, -1, 0});
    bool scored = false;
    for (const auto & s : slots) {
        if (!s.wants_score) continue;
        if (!scored) { ops.push_back({KvFlashOpKind::Score, s.slot, 0}); scored = true; }
        else if (deferred_score) deferred_score->push_back(s.slot);
    }
    return ops;
}

} // namespace luce::common
