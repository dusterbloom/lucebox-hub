// Inkling2SeqEngine & Inkling2SlotKv — Concurrent multi-slot SeqEngine for Inkling 2 Mini
// (1D AR / DSpark / 2DSD + Last-Loop Shared-KV W=64 + 3-Tier Dual-GPU KVFlash 2.0)
// built on SlotKv v2, SeqSlotManager, and Inkling2SlotStateAdapter.

#pragma once

#include "common/concurrency/paged_kv_pool.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/seq_slot_manager.h"
#include "common/concurrency/slot_kv.h"
#include "inkling2_cache.h"
#include "inkling2_internal.h"
#include "inkling2_slot_state.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// SlotKv v2 adapter for Inkling 2 Mini multi-slot serving.
// Wraps per-slot bookkeeping over PagedKvPool AND forwards build_table_patch()
// straight to each slot's Inkling2KvFlash (emitting disjoint Tier-0 & Tier-1.5
// block tables with -1 holes when KVFlash is active).
class Inkling2SlotKv final : public SlotKv {
public:
    Inkling2SlotKv(int max_ctx,
                   const std::vector<Inkling2Cache *> & caches,
                   uint32_t block_size = 64);

    uint32_t   block_size() const override { return pool_.block_size(); }
    uint32_t   attn_block_size() const override { return pool_.block_size(); }
    uint32_t   physical_block_count() const override { return pool_.physical_block_count(); }
    uint32_t   max_sequences() const override { return pool_.max_sequences(); }
    uint32_t   free_block_count() const override { return pool_.free_block_count(); }
    SlotKvCaps caps() const override;

    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                   PagedKvSequenceHandle & h) override;
    PagedKvStatus acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                   const SlotAcquireOpts & opts,
                                   PagedKvSequenceHandle & h) override;
    PagedKvStatus reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) override;
    PagedKvAppendResult append(PagedKvSequenceHandle h, uint32_t n, bool first_last) override;
    PagedKvStatus rollback_append(PagedKvSequenceHandle h, uint32_t n) override;
    PagedKvStatus clear(PagedKvSequenceHandle h) override;
    PagedKvStatus release(PagedKvSequenceHandle h) override;
    PagedKvStatus owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const override;

    bool build_table_patch(PagedKvSequenceHandle h, uint32_t token_extent,
                           uint32_t max_attn_blocks, SlotTablePatch & out,
                           bool remote = false) const override;

private:
    static uint32_t compute_pool_blocks_(int max_ctx, size_t n_slots, uint32_t bs);

    PagedKvPool                  pool_;
    std::vector<Inkling2Cache *> caches_;
};

// Optional step-forward hook allowing deterministic unit tests to drive the full
// SeqEngine contract (check_seq_engine_contract) without loading a 17 GB GGUF.
struct Inkling2SeqStepResult {
    bool                 ok = false;
    int32_t              greedy_token = -1;
    std::vector<float>   logits; // populated when need_logits == true
    std::string          error;
};

using Inkling2SeqForwardFn = std::function<Inkling2SeqStepResult(
    int slot, const int32_t * tokens, int n_tokens, int pos0, bool need_logits, int capture_q_rows)>;

class Inkling2SeqEngine final : public SeqEngine {
public:
    Inkling2SeqEngine(ggml_backend_t backend,
                      const Inkling2Weights & weights,
                      std::vector<Inkling2Cache *> caches,
                      int max_ctx,
                      int prefill_chunk = 512,
                      int verify_width = 0,
                      Inkling2SeqForwardFn custom_forward = {});
    ~Inkling2SeqEngine() override;

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    KvPoolTokens kv_pool_tokens() const override { return slots_.kv_pool_tokens(); }

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    AdmitResult admit_bounded(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler,
                              int max_new_tokens,
                              const AdmitMeta & meta = {}) override;

    StepPlanLimits step_plan_limits(int decode_rows) const override;
    bool           reserve_decode(const StepPlan & plan) override;
    StepResult     step(const StepPlan & plan) override;

    size_t         kv_offload_capacity() const override;
    KvOffloadState kv_offload_state(int slot) const override;
    bool           offload_kv(int slot, size_t max_bytes, std::string & error) override;
    bool           restore_kv(int slot, std::string & error) override;
    bool           evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool           kv_restore_feasible(int slot) const override;

    void retire(int slot) override;
    bool token_is_eos(int32_t token) const override;
    int  max_decode_tokens() const override;

    // Expose SlotKv v2 & SlotStateAdapter for KvFlashSessionRuntime & unit tests.
    Inkling2SlotKv &           slot_kv() { return slot_kv_; }
    const Inkling2SlotKv &     slot_kv() const { return slot_kv_; }
    Inkling2SlotStateAdapter & state_adapter() { return state_adapter_; }
    const Inkling2SlotStateAdapter & state_adapter() const { return state_adapter_; }

private:
    struct SlotRuntime {
        std::vector<int32_t> prompt;
        int                  rewind_pos = -1;
        int                  q_cap_start = 0;
        bool                 rewind_done = false;
        // Host-parked state slab (offload_kv / restore_kv)
        SlotStateSlab        parked_slab;
    };

    std::string validate_plan_(const StepPlan & plan) const;
    Inkling2SeqStepResult run_forward_(int slot, const int32_t * tokens, int n_tokens,
                                       int pos0, bool need_logits, int capture_q_rows);
    int32_t sample_slot_(SeqSlot & seq, const Inkling2SeqStepResult & r, bool is_greedy);
    bool    try_solo_dspark_step_(const StepInput & in, DecodeOutput & out);

    ggml_backend_t               backend_ = nullptr;
    const Inkling2Weights &      weights_;
    std::vector<Inkling2Cache *> caches_;
    Inkling2SlotKv               slot_kv_;
    SeqSlotManager               slots_;
    Inkling2SlotStateAdapter     state_adapter_;
    int                          prefill_chunk_ = 512;
    int                          verify_width_  = 0;
    Inkling2SeqForwardFn         custom_forward_;
    std::vector<SlotRuntime>     rt_;
};

} // namespace luce::common
