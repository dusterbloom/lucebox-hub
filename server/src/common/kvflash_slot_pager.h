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

// R3(a) headroom: page-outs per step are capped at `headroom_blocks`, so the
// slot's usable quota is Q - headroom and a page-in never waits on a block
// freed in the same step. The usable quota must still leave an evictable block.
inline int kvflash_usable_chunks(int quota_chunks, int headroom_blocks) {
    return quota_chunks - headroom_blocks;
}
inline bool kvflash_headroom_ok(int quota_chunks, int headroom_blocks, const KvFlashConfig & cfg = {}) {
    return headroom_blocks >= 0 &&
           kvflash_usable_chunks(quota_chunks, headroom_blocks) * cfg.chunk_tokens >=
               KvFlashPager::min_pool_tokens(cfg);
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
        const KvFlashSlice sl = kvflash_slice(slot, quota_chunks, cfg.chunk_tokens);
        cfg.pool_tokens = quota_chunks * cfg.chunk_tokens;
        cfg.row_base = (int)sl.row_lo;
        // The scratch block must exist too, or the arena is not the one the
        // table was sized for.
        if (!attn_k.empty() && (int64_t)attn_k[0]->ne[1] < kvflash_arena_rows(n_slots, quota_chunks, cfg.chunk_tokens))
            return false;
        if (!attach(cfg, attn_k, attn_v)) return false;
        slice_ = sl;   // only after the attach took: a refusal leaves the old slice and pager consistent
        return true;
    }

    const KvFlashSlice & slice() const { return slice_; }

    // One block-table row of `n_entries` logical chunks: the arena block of a
    // resident chunk, -1 (null) for an evicted or not yet materialized one.
    // Only this slot's pager writes its column; entries never leave the slice.
    void fill_block_table(int32_t * dst, int n_entries) const {
        for (int c = 0; c < n_entries; c++) dst[c] = block_of(c);
    }

private:
    // The slice descriptor and the pager must change together; attach_slice
    // is the only way to attach a slot pager.
    using KvFlashPager::attach;
    KvFlashSlice slice_;
};

// ── Admission (C0 section 4, R1, R5) ───────────────────────────────────

enum class KvFlashAdmit { Ok, BusySlot, BusyHost, PromptTooLong, BadQuota };

struct KvFlashAdmitInput {
    bool    slot_free = false;               // slot, its slice and its state slab are all free
    int     quota_chunks = 0, headroom_blocks = 0;   // usable = quota - headroom must stay above the pager floor
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

// A permanent refusal outranks a transient busy. The host bound does not
// depend on the headroom: it counts every chunk, resident or not.
inline KvFlashAdmit kvflash_admit(const KvFlashAdmitInput & in) {
    KvFlashConfig cfg; cfg.chunk_tokens = in.chunk_tokens;
    if (!kvflash_headroom_ok(in.quota_chunks, in.headroom_blocks, cfg)) return KvFlashAdmit::BadQuota;
    if (in.prompt_tokens > in.max_ctx) return KvFlashAdmit::PromptTooLong;
    if (!in.slot_free) return KvFlashAdmit::BusySlot;
    if (in.host_used_bytes + kvflash_host_need_bytes(in) + in.transient_reserve_bytes > in.host_budget_bytes)
        return KvFlashAdmit::BusyHost;
    return KvFlashAdmit::Ok;
}

// ── Step plan (C0 R3) ──────────────────────────────────────────────────

enum class KvFlashOpKind { Pin, PageOut, PageIn, AllocSpan, TableDelta, Fence, Score, EndStep };

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
    int  headroom_blocks = 1 << 30;     // page-outs allowed this step; the rest wait (default: no cap)
};

// What a step pushed to the next one.
struct KvFlashStepDeferred {
    std::vector<int> score;             // slots whose scorer call waits
    std::vector<KvFlashOp> page_out;    // PageOut ops over a slot's headroom
};

// Order for one engine step: pin every in-flight chunk, then all page-outs,
// then all page-ins, then alloc_span and the synchronous table delta per slot,
// then ONE fence (only if page work was queued), then at most one scorer call,
// then EndStep (always present, always last: step pins expire). Page-outs over
// a slot's headroom_blocks move to `deferred`, in order, for the next step.
// Slots run in ascending id; idle slots emit nothing; the scorer slots that
// lose the one call are listed in deferred->score.
inline std::vector<KvFlashOp> kvflash_plan_step(std::vector<KvFlashStepSlot> slots,
                                                KvFlashStepDeferred * deferred = nullptr) {
    std::sort(slots.begin(), slots.end(),
              [](const KvFlashStepSlot & a, const KvFlashStepSlot & b) { return a.slot < b.slot; });
    for (auto & s : slots) {
        const size_t cap = (size_t)std::max(0, s.headroom_blocks);
        if (s.page_out.size() <= cap) continue;
        if (deferred)
            for (size_t i = cap; i < s.page_out.size(); i++)
                deferred->page_out.push_back({KvFlashOpKind::PageOut, s.slot, s.page_out[i]});
        s.page_out.resize(cap);
    }
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
        else if (deferred) deferred->score.push_back(s.slot);
    }
    ops.push_back({KvFlashOpKind::EndStep, -1, 0});
    return ops;
}

} // namespace luce::common
