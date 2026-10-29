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
// block-table column writer.

#pragma once

#include "kvflash_pager.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
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

// Table entries per chunk when the kernel's physical block is `block_tokens`
// (the WMMA kernel needs 16-token blocks, so a 64-token chunk is 4 entries).
// 0 when the block does not divide the chunk.
inline int kvflash_table_entries(int chunk_tokens, int block_tokens) {
    return block_tokens > 0 && chunk_tokens % block_tokens == 0
               ? chunk_tokens / block_tokens : 0;
}

// Rows of the shared KV allocation: every slice plus one scratch block.
inline int64_t kvflash_arena_rows(int n_slots, int quota_chunks, int chunk_tokens = 64) {
    return ((int64_t)n_slots * quota_chunks + 1) * chunk_tokens;
}

// "auto" sizes ONE pool from the free memory; the arena holds N of them, so each slot
// gets its share of (free - reserve). The reserve (compute buffers) is held once.
inline KvFlashAutoBudget kvflash_budget_per_slot(KvFlashAutoBudget b, int n_slots) {
    if (n_slots > 1 && b.free_bytes > b.reserve_bytes)
        b.free_bytes = b.reserve_bytes + (b.free_bytes - b.reserve_bytes) / n_slots;
    return b;
}

// The arena a KVFlash slot engine allocates: `rows` is ctx_alloc, `pool_tokens`
// the N x Q x chunk the pool is derived as. --kv-pool-tokens is never a size
// here; given and different, it is refused with `error` set (rows 0).
struct KvFlashArenaSizing { int64_t rows = 0, pool_tokens = 0; std::string error; };
inline KvFlashArenaSizing kvflash_arena_sizing(int n_slots, int quota_chunks, int64_t kv_pool_tokens,
                                               int chunk_tokens = 64) {
    KvFlashArenaSizing r;
    if (n_slots < 1 || quota_chunks < 1) { r.error = "KVFlash arena needs at least one slot and one chunk of quota"; return r; }
    r.pool_tokens = (int64_t)n_slots * quota_chunks * chunk_tokens;
    if (kv_pool_tokens != 0 && kv_pool_tokens != r.pool_tokens) {
        r.error = "--kv-pool-tokens " + std::to_string(kv_pool_tokens) + " conflicts with KVFlash: the arena is " +
                  std::to_string(n_slots) + " slots x " + std::to_string(quota_chunks) + " chunks x " +
                  std::to_string(chunk_tokens) + " = " + std::to_string(r.pool_tokens) + " tokens; omit the flag";
        r.pool_tokens = 0;
        return r;
    }
    r.rows = kvflash_arena_rows(n_slots, quota_chunks, chunk_tokens);
    if (r.rows > INT32_MAX) { r.error = "KVFlash arena exceeds INT32_MAX rows"; r.rows = r.pool_tokens = 0; }
    return r;
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
        set_deferred_free(true);   // R3(a): a paged-out block is reusable only after completion
        slice_ = sl;   // only after the attach took: a refusal leaves the old slice and pager consistent
        return true;
    }

    const KvFlashSlice & slice() const { return slice_; }

    // True iff chunks [0, ceil(n_tok/chunk)) are all resident: the only prefix a
    // capture may take in C1.3 (sparse and pager-aware checkpoints are C2).
    bool resident_prefix_covers(int64_t n_tok) const {
        const int nc = (int)((n_tok + chunk_tokens() - 1) / chunk_tokens());
        for (int c = 0; c < nc; c++) if (block_of(c) < 0) return false;
        return true;
    }

    // One block-table row of `n_entries` logical chunks: the arena block of a
    // resident chunk, -1 (null) for an evicted or not yet materialized one.
    // Only this slot's pager writes its column; entries never leave the slice.
    void fill_block_table(int32_t * dst, int n_entries) const {
        for (int c = 0; c < n_entries; c++) dst[c] = block_of(c);
    }

    // The same column for a kernel whose physical block is `block_tokens`:
    // chunk c with arena block B becomes the entries B*k + 0..k-1 (k =
    // kvflash_table_entries), four -1 when the chunk is not resident. `dst`
    // holds n_chunks*k entries. A block size that does not divide the chunk
    // writes nothing. block_tokens == chunk_tokens is the 2-argument form.
    void fill_block_table(int32_t * dst, int n_chunks, int block_tokens) const {
        const int k = kvflash_table_entries(chunk_tokens(), block_tokens);
        for (int c = 0; c < n_chunks; c++) {
            const int b = block_of(c);
            for (int j = 0; j < k; j++) dst[(size_t)c * k + j] = b < 0 ? -1 : b * k + j;
        }
    }

private:
    // The slice descriptor and the pager must change together; attach_slice
    // is the only way to attach a slot pager.
    using KvFlashPager::attach;
    KvFlashSlice slice_;
};

// ── Admission (C0 section 4, R1, R5) ───────────────────────────────────

// HostTooSmall is permanent: the request alone (plus the global transient)
// exceeds the whole host budget, so queueing it as BusyHost would wait forever.
enum class KvFlashAdmit { Ok, BusySlot, BusyHost, PromptTooLong, BadQuota, HostTooSmall };

struct KvFlashAdmitInput {
    bool    slot_free = false;               // slot, its slice and its state slab are all free
    int     quota_chunks = 0, headroom_blocks = 0;   // usable = quota - headroom must stay above the pager floor
    int64_t prompt_tokens = 0, max_new_tokens = 0, max_ctx = 0;
    int     chunk_tokens = 64;
    int64_t chunk_bytes = 0;                 // host backing per chunk
    int64_t ladder_bytes = 0;                // rungs x live state this slot will hold
    int64_t reserve_floor_bytes = 0;         // a resume keeps at least what the slot held while its last park is still being written
    int64_t stage_chunk_bytes = 0;           // park staging per chunk of this request (0: ladder_bytes already holds it all)
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
    return std::max(chunks * (in.chunk_bytes + in.stage_chunk_bytes) + in.ladder_bytes, in.reserve_floor_bytes);
}

// The disk bundle's staging at park (C1.5 F3): the end slab snapshot and its packed file, the packed
// rung slabs, the records of the changed chunks (at most every chunk of the session),
// the ledger and the head. `fixed` does not depend on the session; `per_chunk` is one chunk's record,
// its ledger entry and its 4-byte history tokens. Reserving nc_max chunks of it (the max_ctx bound,
// kvflash_park_staging_max_bytes) charged a 2K session ~5 GiB; reserving the session's own chunks
// (KvFlashAdmitInput::stage_chunk_bytes) charges what the park can hold.
struct KvFlashParkStaging { int64_t fixed_bytes = 0, per_chunk_bytes = 0; };
constexpr int kKvFlashLadderRungs = 3;   // a sticky session's rungs: checkpoint, trunk and the third (rule v5)
inline KvFlashParkStaging kvflash_park_staging(int64_t slab_bytes, int64_t n_slab_tensors, int64_t header_bytes,
                                               int64_t chunk_tokens, int64_t chunk_bytes) {
    KvFlashParkStaging s;
    s.fixed_bytes = (2 + kKvFlashLadderRungs) * slab_bytes + 48 + 16 * n_slab_tensors + header_bytes + chunk_bytes;
    s.per_chunk_bytes = chunk_bytes + 16 + 4 * chunk_tokens;
    return s;
}
// The ladder charge of one request (KvFlashAdmitInput::ladder_bytes): the R state snapshot of a recall turn, the rungs of a
// sticky session, and the bundle staging when sessions are kept on disk.
inline int64_t kvflash_ladder_bytes(int64_t slab_bytes, bool recall, bool sticky, bool staged, const KvFlashParkStaging & staging) {
    return slab_bytes * ((recall ? 1 : 0) + (sticky ? kKvFlashLadderRungs : 0)) + (staged ? staging.fixed_bytes : 0);
}
inline int64_t kvflash_park_staging_max_bytes(const KvFlashParkStaging & s, int64_t max_ctx, int64_t chunk_tokens) {
    return s.fixed_bytes + ((max_ctx + chunk_tokens - 1) / chunk_tokens) * s.per_chunk_bytes;
}

// A permanent refusal outranks a transient busy. The host bound does not
// depend on the headroom: it counts every chunk, resident or not.
inline KvFlashAdmit kvflash_admit(const KvFlashAdmitInput & in) {
    KvFlashConfig cfg; cfg.chunk_tokens = in.chunk_tokens;
    if (!kvflash_headroom_ok(in.quota_chunks, in.headroom_blocks, cfg)) return KvFlashAdmit::BadQuota;
    if (in.prompt_tokens > in.max_ctx) return KvFlashAdmit::PromptTooLong;
    const int64_t alone = kvflash_host_need_bytes(in) + in.transient_reserve_bytes;
    if (alone > in.host_budget_bytes) return KvFlashAdmit::HostTooSmall;
    if (!in.slot_free) return KvFlashAdmit::BusySlot;
    if (in.host_used_bytes + alone > in.host_budget_bytes) return KvFlashAdmit::BusyHost;
    return KvFlashAdmit::Ok;
}

// Host RESERVATIONS, apart from allocated bytes: admit takes a request's
// worst-case backing (kvflash_host_need_bytes), release gives it back and is
// called on EVERY exit (finish, cancel, error, abort_step). The global
// transient is charged once, not per request. Not thread-safe: the engine
// thread owns it.
class KvFlashHostLedger {
public:
    KvFlashHostLedger(int64_t budget_bytes, int64_t transient_reserve_bytes)
        : budget_(budget_bytes), transient_(transient_reserve_bytes) {}

    // `in.host_*` and `in.transient_reserve_bytes` are overwritten from the ledger.
    // A request id already holding a reservation is refused (BusySlot), never charged twice.
    KvFlashAdmit admit(uint64_t request_id, KvFlashAdmitInput in) {
        if (reserved_.count(request_id)) return KvFlashAdmit::BusySlot;
        in.host_budget_bytes = budget_;
        in.transient_reserve_bytes = transient_;
        in.host_used_bytes = used_;
        const KvFlashAdmit v = kvflash_admit(in);
        last_refused_need_ = v == KvFlashAdmit::BusyHost ? kvflash_host_need_bytes(in) + transient_ : 0;
        if (v == KvFlashAdmit::Ok) {
            const int64_t need = kvflash_host_need_bytes(in);
            reserved_[request_id] = need;
            used_ += need;
        }
        return v;
    }
    // What admit() would answer now, charging nothing: the probe of a returning session before its bundle is read.
    KvFlashAdmit would_admit(uint64_t request_id, KvFlashAdmitInput in) const {
        if (reserved_.count(request_id)) return KvFlashAdmit::BusySlot;
        in.host_budget_bytes = budget_;
        in.transient_reserve_bytes = transient_;
        in.host_used_bytes = used_;
        return kvflash_admit(in);
    }
    void release(uint64_t request_id) {   // idempotent
        const auto it = reserved_.find(request_id);
        if (it == reserved_.end()) return;
        used_ -= it->second;
        reserved_.erase(it);
    }
    int64_t used_bytes() const { return used_; }
    int64_t budget_bytes() const { return budget_; }
    int64_t reserved_of(uint64_t request_id) const { const auto it = reserved_.find(request_id); return it == reserved_.end() ? 0 : it->second; }
    int64_t last_refused_need_bytes() const { return last_refused_need_; }   // the last admit's need when it was BusyHost, else 0
    int reservation_count() const { return (int)reserved_.size(); }

private:
    int64_t budget_, transient_, used_ = 0, last_refused_need_ = 0;
    std::unordered_map<uint64_t, int64_t> reserved_;
};

// ── Step plan (C0 R3) ──────────────────────────────────────────────────

enum class KvFlashOpKind { Pin, PageOut, CompleteOut, PageIn, AllocSpan, TableDelta, Fence, Score, EndStep };

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
    // Blocks the step needs and has: PageIn and AllocSpan take blocks, and when
    // they need more than `free_blocks` (allocatable now, NOT counting the
    // blocks this step's own page-outs free) the pager pages out inside the
    // call, so the step has page work even with empty page lists. Defaults
    // are the safe side: no free block, and `alloc_new_chunks` unknown (-1) is
    // the worst case for alloc_tokens, so a caller that does not say gets a fence.
    int  free_blocks = 0;
    int  alloc_new_chunks = -1;         // chunks AllocSpan materializes; -1 = unknown
    bool wants_score = false;
    int  headroom_blocks = 1 << 30;     // page-outs allowed this step; the rest wait (default: no cap)
};

// What a step pushed to the next one.
struct KvFlashStepDeferred {
    std::vector<int> score;             // slots whose scorer call waits
    std::vector<KvFlashOp> page_out;    // PageOut ops over a slot's headroom, or on a chunk pinned this step
};

// Chunks AllocSpan may materialize for `tokens` rows starting anywhere.
inline int kvflash_alloc_chunks_bound(int tokens, int chunk_tokens = 64) {
    return tokens <= 0 ? 0 : (tokens + chunk_tokens - 2) / chunk_tokens + 1;
}

// Order for one engine step: pin every in-flight chunk, then all page-outs,
// then (only if a page-in or allocation must reuse a block this step's
// page-outs free) CompleteOut, the completion phase that waits for the
// page-outs' D2H and settles their pending frees, then all page-ins, then
// alloc_span and the synchronous table delta per slot, then ONE Fence (when
// any page work was queued, including paging done inside PageIn/AllocSpan; it
// is the host-observed completion before compute and before the scorer), then
// at most one scorer call, then EndStep (always present, always last: step
// pins expire). Page-outs over a slot's headroom_blocks, or on a chunk this
// slot pins for the step, move to `deferred` for the next step. Slots run in
// ascending id; idle slots emit nothing; scorer slots that lose the one call
// are listed in deferred->score.
inline std::vector<KvFlashOp> kvflash_plan_step(std::vector<KvFlashStepSlot> slots,
                                                KvFlashStepDeferred * deferred = nullptr,
                                                int chunk_tokens = 64) {
    std::sort(slots.begin(), slots.end(),
              [](const KvFlashStepSlot & a, const KvFlashStepSlot & b) { return a.slot < b.slot; });
    for (auto & s : slots) {
        const size_t cap = (size_t)std::max(0, s.headroom_blocks);
        std::vector<int> keep;
        for (int c : s.page_out) {
            const bool pinned = std::find(s.pin_chunks.begin(), s.pin_chunks.end(), c) != s.pin_chunks.end();
            if (!pinned && keep.size() < cap) keep.push_back(c);
            else if (deferred) deferred->page_out.push_back({KvFlashOpKind::PageOut, s.slot, c});
        }
        s.page_out = std::move(keep);
    }
    std::vector<KvFlashOp> ops;
    bool page_work = false, reuse_after_out = false;
    for (const auto & s : slots) {
        const int alloc = s.alloc_tokens <= 0 ? 0
                        : s.alloc_new_chunks >= 0 ? s.alloc_new_chunks
                        : kvflash_alloc_chunks_bound(s.alloc_tokens, chunk_tokens);
        const bool takes_blocks_beyond_free = (int)s.page_in.size() + alloc > s.free_blocks;
        page_work = page_work || !s.page_out.empty() || !s.page_in.empty() || takes_blocks_beyond_free;
        reuse_after_out = reuse_after_out || (!s.page_out.empty() && takes_blocks_beyond_free);
    }
    for (const auto & s : slots) for (int c : s.pin_chunks) ops.push_back({KvFlashOpKind::Pin, s.slot, c});
    for (const auto & s : slots) for (int c : s.page_out) ops.push_back({KvFlashOpKind::PageOut, s.slot, c});
    if (reuse_after_out) ops.push_back({KvFlashOpKind::CompleteOut, -1, 0});
    for (const auto & s : slots) for (int c : s.page_in)  ops.push_back({KvFlashOpKind::PageIn, s.slot, c});
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
