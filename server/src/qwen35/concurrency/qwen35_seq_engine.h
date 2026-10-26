// Qwen35SeqEngine — SeqEngine implementation for the paged Qwen3.5/3.6
// backend (--max-concurrency N).
//
// Three layers, each with one job:
//   Qwen35SlotManager          host bookkeeping — pool-handle lifecycle,
//                           admission arithmetic, per-slot sampler/RNG/
//                           penalty history, the position counters
//   Qwen35SeqEngine         the device half — chunked slot prefill, the
//                           batched decode forward, sampling, and the
//                           block-table / kv-length uploads
//   Qwen35Backend           the model — weights, cache, step graph, the
//                           paged pool, park/unpark, generate()
//
// The engine borrows the backend's GPU state rather than copying accessors
// for it: it is a friend of Qwen35Backend so that concurrent serving can be
// its own subsystem without widening the backend's public surface.
//
// Single-threaded by the SeqEngine contract — the HTTP scheduler thread is
// the only caller of the engine, the pool, and the device uploads, so there
// is no locking anywhere below here.

#pragma once

#include "common/adaptive_spec_width.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/kvflash_slot_kv.h"
#include "common/concurrency/paged_kv_offload.h"
#include "common/kvflash_engine_step.h"
#include "common/kvflash_step_driver.h"
#include "common/kvflash_slot_bundle.h"
#include "common/kvflash_session_store.h"
#include "common/kvflash_slot_turn.h"
#include "common/kvflash_gpu_mass.h"
#include "common/dflash_draft_kv.h"
#include "common/dflash_feature_ring.h"
#include "qwen35_slot_manager.h"
#include "../qwen35_image_request.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

namespace luce::common {

class Qwen35Backend;

struct FixedChainConfig {
    bool enabled = false;
    int width = 0;
    int scratch_base = 0;
    int scratch_stride = 0;
};

class Qwen35SeqEngine final : public SeqEngine {
public:
    // `pool` and `backend` must outlive the engine. `scratch_row` is outside
    // the pool and any per-slot tree slabs; it is the K/V destination of
    // graph-bucket padding rows.
    // `max_prefills` bounds scheduler-selected prompt slices per traversal.
    Qwen35SeqEngine(Qwen35Backend & backend, PagedKvPool & pool,
                    int max_ctx, int64_t scratch_row,
                    FixedChainConfig fixed_chain,
                    int max_prefills = 8,
                    int mixed_prefill_tokens = 2048,
                    int long_mixed_prefill_tokens = 4096,
                    int long_prefill_threshold = 768,
                    int idle_prefill_tokens = 4096,
                    int prefill_quantum = 512);
    // KVFlash slots (Track C): the slots allocate through `kv` (one arena, one
    // slice per slot) and `driver` runs the pager work of every step (LRU only,
    // D1 cuts of kKvFlashCutU). No pool, no offload, no prefix store, no images.
    // `kv` and `driver` must outlive the engine.
    Qwen35SeqEngine(Qwen35Backend & backend, KvFlashSlotKv & kv, KvFlashStepDriver & driver,
                    int max_ctx, int64_t scratch_row,
                    int max_prefills = 8,
                    int mixed_prefill_tokens = 2048,
                    int long_mixed_prefill_tokens = 4096,
                    int long_prefill_threshold = 768,
                    int idle_prefill_tokens = 4096);
    ~Qwen35SeqEngine() override;

    // Destroy every graph that captures draft-weight tensors. Qwen35Backend
    // calls this before freeing draft weights during park; states rebuild
    // lazily after unpark.
    void release_draft_graphs();

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    // KVFlash slots take no prefix store yet: resident-only capture (D5) needs the arena table behind it.
    bool supports_prefix_store() const override { return !kvf_; }
    bool prefix_store_in_backend_snapshots() const override { return !kvf_; }
    size_t estimate_prefix_store_bytes(int tokens) const override;

    void discard_prefix_store(PrefixStoreRef checkpoint) override;

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;
    AdmitResult admit_bounded(uint64_t request_id,
                              const std::vector<int32_t> & prompt,
                              const SamplerCfg & sampler,
                              int max_new_tokens,
                              const AdmitMeta & meta) override;

    AdmitResult admit_with_prefix(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        const PrefixStorePlan & plan) override;

    StepResult step(const StepPlan & plan) override;
    // A step that carries a prompt chunk decodes without the drafter, one
    // token per request. Beside a prompt of more than 8192 new tokens (a
    // document, not an agent's next step), the decoding requests get half
    // the time back in steps of their own, with it.
    StepPlanLimits step_plan_limits(int decode_rows) const override {
        const bool mixed = decode_rows > 0;
        const int per_sequence = mixed ? 512 : 2048;
        int total_cap = idle_prefill_tokens_;
        if (mixed) {
            total_cap = mixed_prefill_tokens_;
            if (slots_.has_prefill_prompt_at_least(long_prefill_threshold_)) {
                total_cap = std::max(total_cap, long_mixed_prefill_tokens_);
            }
        }
        const StepPlanLimits limits{
            max_prefills_,
            per_sequence,
            std::min(max_prefills_ * per_sequence, total_cap),
            prefill_quantum_,
            /*prefill_time_share=*/mixed ? 0.5 : 1.0,
            /*time_share_prompt_tokens=*/8192,
        };
        return kvf_ ? kvflash_step_limits(limits) : limits;
    }

    bool reserve_decode(const StepPlan & plan) override;
    size_t kv_offload_capacity() const override { return offload_ ? offload_->capacity() : 0; }
    KvOffloadState kv_offload_state(int slot) const override { return offload_ ? offload_->state(slot) : KvOffloadState{}; }
    bool offload_kv(int slot, size_t bytes, std::string & error) override {
        if (!offload_) { error = "KVFlash slots cannot be parked"; return false; }
        return offload_->suspend(slot, bytes, error);
    }
    bool restore_kv(int slot, std::string & error) override;
    bool evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool kv_restore_feasible(int slot) const override {
        return slots_.kv_restore_feasible(slot);
    }
    void retire(int slot) override;
    bool supports_images() const override;
    AdmitResult admit_images(uint64_t request_id,
                             const std::vector<int32_t> & prompt,
                             const SamplerCfg & sampler,
                             const ImagePromptHandle & images) override;

    bool token_is_eos(int32_t token) const override;


private:
    struct PrefillStage {
        bool ready = false;
        int kv_pos = 0;
        int chunk = 0;
        bool commit = false;
        std::vector<int64_t> rows;
        std::vector<float> embeddings;
        // Axis-major [4 x chunk] rotary positions when the slot holds images;
        // empty means the plain kv_pos + i positions.
        std::vector<int32_t> positions;
    };

    // Per-slot image state: the payload (kept alive for re-prefill after
    // eviction), its encoded rows, and how far rotary positions run ahead of
    // KV positions after the images.
    struct SlotImages {
        ImagePromptHandle payload;
        Qwen35ImageRows rows;
        int rope_delta = 0;
    };
    std::vector<SlotImages> slot_images_;
    int rope_delta(int slot) const {
        return slot >= 0 && slot < static_cast<int>(slot_images_.size())
            ? slot_images_[static_cast<size_t>(slot)].rope_delta : 0;
    }
    void clear_slot_images(int slot) {
        if (slot >= 0 && slot < static_cast<int>(slot_images_.size())) {
            slot_images_[static_cast<size_t>(slot)] = SlotImages{};
        }
    }

    struct PreparedChainDraft {
        std::vector<int32_t> tokens;
    };
    struct PreparedChainRound {
        std::vector<PreparedChainDraft> drafts;
    };

    int max_prefills_;
    int mixed_prefill_tokens_;
    int long_mixed_prefill_tokens_;
    int long_prefill_threshold_;
    int idle_prefill_tokens_;
    int prefill_quantum_;

    bool upload_block_table_delta(int slot, int first_block,
                                  const int32_t * blocks, size_t count);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                              const char * log_message,
                              const char * client_message);
    PrefillStage stage_prefill_chunk(int slot, int max_tokens,
                                     std::vector<PrefillOutput> & outputs);
    int32_t sample_graph_row(int slot, int logits_row,
                             const int32_t * cached_argmax = nullptr,
                             std::vector<float> * logits_scratch = nullptr);
    std::vector<uint8_t> select_chain_lanes(
        const StepPlan & plan) const;
    // width: the block the round drafts (0 = the configured block).
    bool chain_spec_input_capable(const StepInput & input, int width = 0) const;
    DraftFeatureMirror * slot_feature_mirror(int slot);
    DraftKvState * ensure_slot_draft_kv(int slot, bool batched = false);
    void reset_slot_draft_kv(int slot);
    // The block a round with these selected lanes drafts.
    int chain_draft_width(const std::vector<uint8_t> & selected) const;
    std::optional<PreparedChainRound> prepare_chain_drafts(
        const std::vector<StepInput> & inputs,
        const std::vector<uint8_t> & selected);
    StepResult step_chain_spec(
        const StepPlan & plan, const std::vector<uint8_t> & selected,
        PreparedChainRound && prepared);
    // Verify width for one batched chain round. Each verify graph bucket
    // keeps its own AdaptiveSpecWidth: a round's cost depends on the graph's
    // rows (bucket lanes x width), while the width that maximizes committed
    // tokens per unit cost does not depend on how many lanes share it.
    // cap: the longest chain the round drafted (0 = the configured block).
    int choose_chain_width(int bucket, int cap = 0);
    void observe_chain_width(int bucket, int width,
                             const std::vector<size_t> & accepted,
                             double step_ms);
    PrefixStoreEvent capture_prefix(
        int slot, PrefixCaptureTicket ticket);
    bool arm_capture(
        int slot, PrefixCaptureTicket ticket, int restored_tokens);
    int checkpoint_index(PrefixStoreRef checkpoint) const;

    PagedKvPool * pool_ = nullptr;                    // null with KVFlash slots
    Qwen35Backend & b_;
    Qwen35SlotManager  slots_;
    std::unique_ptr<PagedKvOffload> offload_;         // null with KVFlash slots
    KvFlashSlotKv *     kvf_kv_ = nullptr;
    KvFlashStepDriver * kvf_ = nullptr;
    void init_slots_();
    bool upload_kvflash_column(int slot, int kv_len);
    bool settle_kvflash_();
    // C1.4: per-slot mode 4. The turn plan, the scores the pager's hook reads
    // (stable: recall_ is sized once), the state at R and the step bookkeeping.
    struct SlotRecall {
        KvFlashSlotTurn turn;
        std::vector<float> scores;
        double score_ms = 0.0;       // mode 5: the drafter's time for `scores`
        SlotStateSnapshot snap;
        bool snap_ok = false;
        bool tainted = false;        // in a failed step cohort: never parked
    };
    std::vector<SlotRecall> recall_;
    bool recall_on_ = false;                          // the recall profile is on for the slot engine
    // C1.4 item 2: sticky sessions. A retired slot keeps its KV and recurrent
    // state parked; a prompt that extends its history resumes it in place.
    // The prompt-end checkpoint: the recurrent state after the prompt, before the reply.
    // A client re-renders the reply from text, so the next prompt may diverge inside it;
    // the resume then falls back to this state and replays the reply.
    // C1.5 F1: two rungs, `trunk` below `ckpt`: the checkpoint before the current one, where a branch
    // forked (kvflash_slot_turn.h Rules A and B).
    struct SlotCkpt {
        int pos = -1;
        SlotStateSnapshot slab;
        uint32_t file = 0;               // its slab file id in the slot's bundle (slab.<file>), 0: not on disk
    };
    struct ParkedSession {
        std::vector<int32_t> history;    // the tokens its KV holds (== its cur_pos at park)
        uint64_t tick = 0;
        bool valid = false;
        SlotCkpt ckpt;
        SlotCkpt trunk;
    };
    std::vector<ParkedSession> parked_;
    std::vector<SlotCkpt> ckpt_;                      // the live slot's prompt-end checkpoint, taken at commit
    std::vector<SlotCkpt> trunk_;                     // the rung below it
    int end_rung_min_ = -1;                           // LUCE_KVFLASH_END_RUNG_MIN: a recall turn's header rung when end - R reaches it; -1 off
    void place_ckpt_(int slot, SlotCkpt nu);          // Rule A with a slab in hand (the R rung)
    void take_ckpt_(int slot, int pos);               // Rule A with the slot's state now (the header rung)
    // C1.4 item 3 / C1.5 F3: LUCE_KVFLASH_SLOT_BUNDLE_DIR keeps each parked session on disk (sessions/<key>/, an
    // append-only chunk log, slab files and a head), written at park, adopted at start, so a restart resumes
    // every session where it parked. KVFlash 2.0 sessions (kvflash_session_store.h): a slot is a cache over the
    // session store, a session leaving its slot is evicted to disk (its directory stays) and a request whose
    // sid names it restores it; only a session that never reached disk is dropped. A park appends only the chunks from
    // the one holding the turn's resume point L on; the rest keep their records.
    using SlotBundle = KvFlashSlotBundle;   // per slot: the session in it, its directory and what the directory holds
    KvFlashSessionBook book_;                          // per slot: the session in it and what its directory holds; the session store behind it
    std::vector<std::atomic<bool>> bundle_failed_;    // set by the writer thread: the next park rewrites the slot whole
    std::string bundle_dir_;
    uint64_t bundle_identity_ = 0;
    KvFlashBundleLimits bundle_lim_;
    std::unique_ptr<KvFlashBundleWriter> bundle_writer_;   // the park saves, off the step thread (after bundle_failed_: it uses it)
    int bundle_lock_ = -1;          // flock on bundle_dir_/.lock for the engine's lifetime
    void init_bundles_();
    std::string adopt_bundle_(int slot, const std::string & key, bool * refused);   // load sessions/<key>/ into the slot's pager; "" or why not (refused: the bundle itself is unusable here)
    std::vector<bool> active_slots_() const;   // which slots decode a request now
    KvFlashRestore restore_from_disk_(const AdmitMeta & who, const std::vector<int32_t> & prompt, int e_role);   // a disk session of this request into a slot
    void save_bundle_(int slot, ParkedSession & ps);   // ps is unchanged on return but for the rungs' file ids
    uint64_t park_tick_ = 0;
    bool sticky_on_ = false;
    void drop_session_(int slot, const char * why);   // forget the session, and its directory: only for one that never reached disk
    bool evict_session_(int slot, const char * why);  // the session leaves the slot, its bundle stays on disk; false: its latest park is not on disk, it stays
    int evict_oldest_(const char * why);              // the oldest parked session that can be evicted: its slot, -1 when none
    std::vector<KvFlashOwnedSession> parked_list_() const;   // every parked session with its owner
    bool begin_resume_(int slot);                     // residency to the recent state before the turn's first pass
    std::vector<std::pair<int, int>> pending_rewind_;  // (slot, cut tokens) whose first pass ended this step
    int capture_rows_ = 0;                            // query rows the graph captured this step
    void plan_slot_turn_(int slot, const std::vector<int32_t> & prompt, int kv_offset);
    // Mode 5: the drafter's picks of the slot's turn, applied when its prefill reaches R (no first pass, no rewind).
    void apply_recall5_(int slot);
    bool rewind_slot_turn_(int slot, int cut_tokens);
    std::vector<int32_t> column_buf_;
    int64_t         scratch_row_ = 0;
    FixedChainConfig fixed_chain_;
    bool            fixed_chain_ready_ = false;
    struct ChainWidthState {
        AdaptiveSpecWidth controller;
        std::vector<int>  offers;   // rounds offered at each width
        std::vector<int>  samples;  // clean cost samples at each width
        long long rounds = 0;
        double    accepted_sum = 0.0;
        size_t    lane_rounds = 0;
        // Per-depth trials and acceptances (decayed). A depth is a trial
        // only when it was offered and every shallower candidate was
        // accepted, so a narrow width does not bias the deeper estimates.
        std::vector<double> depth_trials;
        std::vector<double> depth_accepts;
        std::vector<float>  conditional;  // reused per round
        explicit ChainWidthState(int max_width)
            : controller(max_width, 2, true),
              offers(static_cast<size_t>(max_width) + 1, 0),
              samples(static_cast<size_t>(max_width) + 1, 0),
              depth_trials(static_cast<size_t>(max_width), 0.0),
              depth_accepts(static_cast<size_t>(max_width), 0.0),
              conditional(static_cast<size_t>(std::max(1, max_width - 1)), 0.7f) {}
    };
    bool adaptive_chain_width_ = false;
    std::vector<int> chain_width_choices_;  // ascending, includes the block
    std::map<int, ChainWidthState> chain_width_by_bucket_;
    std::chrono::steady_clock::time_point chain_round_t0_{};
    std::vector<size_t> chain_accepted_scratch_;
    ggml_context *  feature_view_ctx_ = nullptr;
    std::vector<DraftFeatureMirror> slot_feature_mirrors_;
    std::vector<std::unique_ptr<DraftKvState>> slot_draft_kv_;
    std::vector<std::unique_ptr<DraftKvState>> dummy_draft_kv_;
    DraftKvBatchGraph batch_draft_graph_;
    // Rounds with two or more lanes draft a shorter block than the configured
    // one: a long block pays for itself only when one request decodes alone
    // (verify cost grows with lanes x width, acceptance stays per lane).
    // batched_dw_ is b_.dw_ with that block size, copied again at every
    // batched round so it always holds the live tensors (a draft park and
    // unpark reloads b_.dw_); it is never freed. 0 = every round drafts the
    // configured block.
    int batched_draft_width_ = 0;
    DraftWeights batched_dw_;
    std::vector<std::unique_ptr<DraftKvState>> slot_draft_kv_batched_;
    std::vector<std::unique_ptr<DraftKvState>> dummy_draft_kv_batched_;
    DraftKvBatchGraph batch_draft_graph_batched_;

    // Hoisted per-step buffers (reused across step() calls).
    std::vector<int>         reserve_growth_;
    std::vector<int>         output_rows_;
    std::vector<int32_t>     live_tokens_;
    std::vector<int32_t>     live_positions_;
    std::vector<int64_t>     live_physical_rows_;
    std::vector<int32_t>     live_slot_ids_;
    std::vector<int32_t>     dec_tokens_;
    std::vector<int64_t>     dec_rows_;
    std::vector<int32_t>     active_slot_ids_;
    std::vector<int32_t>     state_slot_ids_;
    std::vector<int32_t>     seq_lens_;
    std::vector<int32_t>     query_slot_ids_;
    std::vector<int32_t>     query_positions_;
    std::vector<int32_t>     logits_rows_;
    std::vector<int32_t>     feature_rows_;
    std::vector<float>       embed_buf_;
    std::vector<int32_t>     pos_buf_;
    std::vector<int64_t>     rows_buf_;
    std::vector<int32_t>     argmax_buf_;
    std::vector<float>       logits_buf_;
};

}  // namespace luce::common
