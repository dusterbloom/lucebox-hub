// KvFlashStepDriver (Track C, C1.3-05): what the seq engine runs around the
// graph of one step. kvflash_preflight_step (D2) decides which slots start the
// step, kvflash_plan_step orders their pager work, kvflash_execute_step runs
// it through this class (the KvFlashStepBackend over the slot pagers and the
// arena). C1.3 is LRU only (D6): no op asks for a score, scorer_calls() stays 0.
//
// Engine use: run_step before the graph; abort_step() on any failure after it;
// settle() after the argmax readback sync queues the page-outs that keep every
// live slot at its usable quota (Q - headroom), so the next step finds the
// free blocks it needs, a decode row never idles, and a prefill cut is not paced
// at one eviction per step. The step pins protect the step's own chunks and
// outlive the graph: the engine calls finish_step() after the readback. A
// slot's table column is re-uploaded only when table_stale(slot).
// Engine thread only; pagers and arena must outlive it.

#pragma once

#include "kvflash_arena.h"
#include "kvflash_step_exec.h"

#include <algorithm>
#include <vector>

namespace luce::common {

// D2: a slot starts a step only if the blocks it needs are free AND completed.
struct KvFlashPreflightSlot { int slot; int free_completed_blocks; int need_blocks; };

inline std::vector<int> kvflash_preflight_step(const std::vector<KvFlashPreflightSlot> & slots) {
    std::vector<int> ran;
    for (const auto & s : slots) if (s.free_completed_blocks >= s.need_blocks) ran.push_back(s.slot);
    return ran;
}

struct KvFlashStepWork { int slot; int64_t kv_start; int tokens; };   // rows the slot appends this step

class KvFlashStepDriver final : public KvFlashStepBackend {
public:
    KvFlashStepDriver(std::vector<KvFlashSlotPager *> pagers, KvFlashArena & arena, int headroom_blocks = 8)
        : pagers_(std::move(pagers)), arena_(arena), headroom_(headroom_blocks), start_(pagers_.size(), 0),
          uploaded_(pagers_.size(), ~uint64_t(0)) {}

    struct Result { bool ok = true; std::vector<int> ran; };

    // Settle the previous step's page-outs, preflight, plan and execute. A slot
    // that does not fit idles this step; the page-outs for the blocks it lacks
    // are queued now and the next call settles them (the recovery step).
    // ok=false: an op failed and abort_step already ran.
    Result run_step(const std::vector<KvFlashStepWork> & work, void * compute_stream) {
        Result r;
        compute_stream_ = compute_stream;
        arena_.complete_all();
        std::vector<KvFlashPreflightSlot> pre;
        std::vector<KvFlashStepSlot> slots;
        std::vector<int64_t> starts;
        for (const auto & w : work) {
            if (w.tokens <= 0) continue;
            const KvFlashSlotPager & p = *pagers_[(size_t)w.slot];
            KvFlashStepSlot s;
            s.slot = w.slot; s.alloc_tokens = w.tokens; s.alloc_new_chunks = 0;
            const int ct = p.chunk_tokens();
            for (int c = (int)(w.kv_start / ct); c <= (int)((w.kv_start + w.tokens - 1) / ct); c++) {
                s.pin_chunks.push_back(c);
                s.alloc_new_chunks += p.block_of(c) < 0;
            }
            s.free_blocks = p.n_pool_blocks() - p.resident_blocks() - p.pending_free_blocks();
            // D2: a slot above its usable quota (after a truncate or restore) is recovering; it needs the
            // whole headroom free before any step, even one that allocates nothing.
            const bool over = p.resident_blocks() > p.n_pool_blocks() - headroom_;
            pre.push_back({w.slot, s.free_blocks, over ? std::max(s.alloc_new_chunks, headroom_) : s.alloc_new_chunks});
            slots.push_back(std::move(s));
            starts.push_back(w.kv_start);
        }
        r.ran = kvflash_preflight_step(pre);
        std::vector<KvFlashStepSlot> fit;
        for (size_t i = 0; i < slots.size(); i++) {
            const int slot = slots[i].slot;
            if (std::find(r.ran.begin(), r.ran.end(), slot) != r.ran.end()) {
                start_[(size_t)slot] = starts[i];
                fit.push_back(std::move(slots[i]));
            } else {
                KvFlashSlotPager & p = *pagers_[(size_t)slot];
                // D2: a recovering slot pages out until headroom blocks are free, not just this need
                if (queue_page_outs_(p, std::max(pre[i].need_blocks, headroom_) - pre[i].free_completed_blocks - p.pending_free_blocks()) < 0) {
                    abort_step();
                    r.ok = false;
                    return r;
                }
            }
        }
        r.ok = kvflash_execute_step(kvflash_plan_step(std::move(fit), nullptr, pagers_.front()->chunk_tokens()), *this).status == KvFlashExec::Ok;
        return r;
    }

    bool run(const KvFlashOp & op) override {
        KvFlashSlotPager & p = *pagers_[(size_t)std::max(0, op.slot)];
        const int ct = p.chunk_tokens();
        switch (op.kind) {
            case KvFlashOpKind::Pin:         p.pin_step_range((int64_t)op.arg * ct, (int64_t)(op.arg + 1) * ct); return true;
            case KvFlashOpKind::PageOut:     return p.page_out_checked(op.arg, p.generation());
            case KvFlashOpKind::CompleteOut: arena_.complete_all(); return true;
            case KvFlashOpKind::PageIn:      return p.page_in(op.arg);
            case KvFlashOpKind::AllocSpan:   return p.alloc_span((int)start_[(size_t)op.slot], op.arg);
            case KvFlashOpKind::TableDelta:  return true;   // table_stale() is the epoch compare
            case KvFlashOpKind::Fence:       arena_.join_compute(compute_stream_); return true;
            case KvFlashOpKind::Score:       scorer_calls_++; return true;
            case KvFlashOpKind::EndStep:     return true;
        }
        return false;
    }
    void end_step() override {}   // the executor's EndStep: pins stay until finish_step (after the readback)
    void finish_step() { for (KvFlashSlotPager * p : pagers_) p->end_step(); }
    void abort_step() override { for (KvFlashSlotPager * p : pagers_) p->abort_step(); }

    int scorer_calls() const { return scorer_calls_; }
    // The one host wait that makes every queued page-out's block allocatable (before a truncate).
    void complete_all() { arena_.complete_all(); }
    // A slot's recall scored this step (the plan's Score op runs on the engine thread).
    void note_score() { scorer_calls_++; }

    // After the readback: bring each listed slot down to Q - headroom resident
    // blocks (LRU victims, async). Returns the page-outs queued, -1 if one failed (abort_step ran).
    int settle(const std::vector<int> & live_slots) {
        int queued = 0;
        for (int slot : live_slots) {
            KvFlashSlotPager & p = *pagers_[(size_t)slot];
            const int n = queue_page_outs_(p, std::min(headroom_, p.resident_blocks() - (p.n_pool_blocks() - headroom_)));
            if (n < 0) { abort_step(); return -1; }
            queued += n;
        }
        return queued;
    }

    // The slot's column changed since it was last written.
    bool table_stale(int slot) const { return pagers_[(size_t)slot]->epoch() != uploaded_[(size_t)slot]; }
    void write_table(int slot, int32_t * dst, int n_chunks, int block_tokens) {
        pagers_[(size_t)slot]->fill_block_table(dst, n_chunks, block_tokens);
        uploaded_[(size_t)slot] = pagers_[(size_t)slot]->epoch();
    }

private:
    // Page-outs queued for n blocks; -1 when one fails (a host buffer could not be allocated).
    static int queue_page_outs_(KvFlashSlotPager & p, int n) {
        int queued = 0;
        if (n > 0) for (int c : p.lru_victims(n)) { if (!p.page_out_checked(c, p.generation())) return -1; queued++; }
        return queued;
    }

    std::vector<KvFlashSlotPager *> pagers_;
    KvFlashArena & arena_;
    int headroom_;
    std::vector<int64_t> start_;
    std::vector<uint64_t> uploaded_;
    void * compute_stream_ = nullptr;
    int scorer_calls_ = 0;
};

}  // namespace luce::common
