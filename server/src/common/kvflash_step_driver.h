// KvFlashStepDriver (Track C, C1.3-05): what the seq engine runs around the
// graph of one step. kvflash_preflight_step (D2) decides which slots start the
// step, kvflash_plan_step orders their pager work, kvflash_execute_step runs
// it through this class (the KvFlashStepBackend over the slot pagers and the
// arena). C1.3 is LRU only (D6): no op asks for a score, scorer_calls() stays 0.
//
// Engine use: complete_all and the plan run before the graph; the engine calls
// end_step() after the argmax readback sync (pins must outlive the compute) and
// abort_step() on any failure after run_step. A slot's table column is
// re-uploaded only when table_stale(slot).
// Engine thread only; pagers and arena must outlive it.

#pragma once

#include "kvflash_arena.h"
#include "kvflash_step_exec.h"

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
    KvFlashStepDriver(std::vector<KvFlashSlotPager *> pagers, KvFlashArena & arena)
        : pagers_(std::move(pagers)), arena_(arena), start_(pagers_.size(), 0),
          uploaded_(pagers_.size(), ~uint64_t(0)) {}

    struct Result { bool ok = true; std::vector<int> ran; };

    // Settle the previous step's page-outs, preflight, plan and execute. A slot
    // that does not fit idles this step; its allocation is attempted once so the
    // pager queues a page-out that the next call settles (the recovery step).
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
            pre.push_back({w.slot, s.free_blocks, s.alloc_new_chunks});
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
                pagers_[(size_t)slot]->alloc_span((int)starts[i], slots[i].alloc_tokens);
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
    void end_step() override { for (KvFlashSlotPager * p : pagers_) p->end_step(); }
    void abort_step() override { for (KvFlashSlotPager * p : pagers_) p->abort_step(); }

    int scorer_calls() const { return scorer_calls_; }

    // The slot's column changed since it was last written.
    bool table_stale(int slot) const { return pagers_[(size_t)slot]->epoch() != uploaded_[(size_t)slot]; }
    void write_table(int slot, int32_t * dst, int n_chunks, int block_tokens) {
        pagers_[(size_t)slot]->fill_block_table(dst, n_chunks, block_tokens);
        uploaded_[(size_t)slot] = pagers_[(size_t)slot]->epoch();
    }

private:
    std::vector<KvFlashSlotPager *> pagers_;
    KvFlashArena & arena_;
    std::vector<int64_t> start_;
    std::vector<uint64_t> uploaded_;
    void * compute_stream_ = nullptr;
    int scorer_calls_ = 0;
};

}  // namespace luce::common
