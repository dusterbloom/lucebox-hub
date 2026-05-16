// Qwen35DFlashTarget — DFlashTarget implementation for qwen35 hybrid models.
//
// Wraps the existing qwen35 target infrastructure (TargetWeights, TargetCache,
// StepGraph, DraftFeatureMirror) behind the generic DFlashTarget interface.
// This adapter enables the generic spec-decode loop to drive qwen35 verification.

#pragma once

#include "common/dflash_target.h"
#include "internal.h"         // TargetWeights, TargetCache, DraftWeights
#include "step_graph.h"
#include "graph_builders.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash27b {

class Qwen35DFlashTarget : public DFlashTarget {
public:
    // Non-owning references — caller must ensure lifetime.
    Qwen35DFlashTarget(TargetWeights & w,
                       TargetCache & cache,
                       ggml_backend_t backend,
                       StepGraph & sg,
                       int kq_stride_pad,
                       int fa_window);

    ~Qwen35DFlashTarget() override;

    ggml_backend_t backend() const override { return backend_; }

    // Phase B+ fused-LM-head path: the Qwen3.6 MTP head's step graph can
    // append `mul_mat(w_.output, x_normed) -> argmax` directly so it
    // avoids a hidden -> host -> separate-cgraph round trip per call.
    ggml_tensor * lm_head_weight() const override { return w_.output; }

    // Mirror the causal window onto MTP head's flash-attn so it sees the
    // same active context as the target's full-attention blocks.
    int fa_window() const override { return fa_window_; }

    // Enable per-position post-norm hidden capture during verify_batch.
    // Off by default; MTP modules that depend on hidden_at_pos() flip it on
    // in attach().  Non-MTP paths (target_gen, DFlash drafter spec-decode)
    // leave it off and avoid pinning the full [n_embd, n_tokens] tensor.
    void enable_hidden_seq_capture(bool on) { capture_hidden_seq_ = on; }

    // Hidden-sequence capture granularity.  FULL_SEQ downloads the entire
    // [n_tokens, n_embd] post-norm + pre-norm hidden tensors device->host
    // on every verify_batch (needed by warm_head_kv during prefill, which
    // consumes per-position hiddens via last_hidden_seq()).  LAST_ROW_ONLY
    // downloads only row n_tokens-1 — sufficient for decode-side chain
    // verifies whose only consumer is hidden_at_pos(base_pos - 1), i.e. the
    // last token of the just-verified batch.  Switching to LAST_ROW_ONLY
    // after prefill collapses two ~80 KB device->host syncs (post-norm +
    // pre-norm, hidden_dim=5120, D+1=4 tokens) into two ~20 KB single-row
    // syncs and saves the 2x WSL2 scheduler latency hit per verify.
    enum class VerifyCaptureMode {
        FULL_SEQ,        // default — required during prefill / warm_head_kv
        LAST_ROW_ONLY,   // decode mode — only hidden_at_pos(base_pos-1) used
    };
    void set_hidden_capture_mode(VerifyCaptureMode mode) {
        capture_mode_ = mode;
    }
    VerifyCaptureMode hidden_capture_mode() const { return capture_mode_; }

    // ── DFlashTarget interface ──────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr) override;

    // Tree-verify override.  Stage 1 stub: only handles the degenerate
    // single-token case (tree.n_nodes == 0) by dispatching to verify_batch.
    // For real DDTree shapes (n_nodes > 0) it returns false so the harness
    // falls back to chain-verify.  Stage 2 will wire build_target_step_tree
    // + ancestor mask here.
    bool verify_tree(const std::vector<int32_t> & flat_tokens,
                     const DDTree & tree,
                     int base_pos,
                     std::vector<int32_t> & out_argmax,
                     std::vector<float> * out_logits = nullptr) override;

    bool snapshot_kv() override;
    bool restore_kv() override;

    // Stage 3 (oracle blocker 5.3): rollback DeltaNet SSM + conv + full-attn KV
    // to the slot of `dfs_idx` from the MOST RECENT verify_tree() call. The
    // verify_tree run captured per-DFS-slot SSM intermediate states (cache_.
    // ssm_intermediate[il]) and the full conv input window (cache_.
    // conv_input_cache[il]); this method dequantizes/copies those into the
    // live state buffers so the next verify_batch / verify_tree sees correct
    // DeltaNet history.  Full-attention KV that was written at slot
    // (root_base_pos + accepted_dfs[d]) is compacted to (root_base_pos + d)
    // for d=1..commit_n-1 when the accepted walk deviates from the DFS spine.
    //
    // `accepted_dfs` is the accepted-path DFS-index list returned by
    // follow_verified_tree (size >= 1, accepted_dfs[0] == 0 = root).  The
    // caller may pass only the committed prefix (i.e. truncate at n_gen).
    //
    // After this call cache_.cur_pos = root_base_pos + commit_n so the next
    // verify_batch writes at the right offset.  Returns false if verify_tree
    // was not called, or if dfs_idx is out of range.
    bool restore_kv_at_dfs(const std::vector<int> & accepted_dfs) override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    bool project_hidden_to_logits(const float * hidden,
                                  int n_tokens,
                                  std::vector<float> & logits_out,
                                  int & out_vocab) override;

    int hidden_size() const override { return w_.n_embd; }
    int mask_token_id() const override;
    const std::vector<int> & capture_layer_ids() const override;

    // Return the backbone's final post-norm hidden state for the last committed
    // token (n_embd floats, F32).  Populated by the most recent verify_batch.
    // Returns nullptr if verify_batch has not been called yet.
    const float * last_hidden() const override { return last_hidden_cpu_.empty() ? nullptr : last_hidden_cpu_.data(); }

    // Full post-norm hidden sequence from the last verify_batch, laid out as
    // [token_0_hidden, ..., token_{n_tokens-1}_hidden], n_tokens * n_embd floats.
    const float * last_hidden_seq(int * out_n_tokens) const override {
        if (out_n_tokens) *out_n_tokens = last_hidden_seq_n_;
        return last_hidden_seq_cpu_.empty() ? nullptr : last_hidden_seq_cpu_.data();
    }

    // Absolute-position accessor: returns &last_hidden_seq[(pos - chunk_start) * n_embd]
    // if pos is within [last_verify_chunk_start_, last_verify_chunk_start_ + n_seq).
    const float * hidden_at_pos(int abs_pos) const override {
        const int rel = abs_pos - last_verify_chunk_start_;
        if (rel < 0 || rel >= last_hidden_seq_n_ ||
            last_hidden_seq_cpu_.empty()) {
            return nullptr;
        }
        return last_hidden_seq_cpu_.data() + (size_t)rel * w_.n_embd;
    }

    // Pre-final-output-norm variant of hidden_at_pos.  Mirrors llama.cpp
    // PR #22673's `t_h_pre_norm`: the Qwen3.6 MTP head's hnorm normalises
    // h_prev internally, so feeding it the post-output-norm tensor double-
    // normalises and compounds per-depth rejection.  The chain GPU loop's
    // outer h_prev_0 seed must use this accessor; the intra-iter re-feed
    // already uses the MTP graph's pre-norm output (state_->last_hidden).
    // Returns nullptr if the most recent verify_batch did NOT capture the
    // pre-norm sequence (enable_hidden_seq_capture(true) flips this on).
    const float * hidden_at_pos_pre_norm(int abs_pos) const override {
        const int rel = abs_pos - last_verify_chunk_start_;
        if (rel < 0 || rel >= last_hidden_seq_n_ ||
            last_hidden_seq_pre_norm_cpu_.empty()) {
            return nullptr;
        }
        return last_hidden_seq_pre_norm_cpu_.data() + (size_t)rel * w_.n_embd;
    }

private:
    TargetWeights & w_;
    TargetCache & cache_;
    ggml_backend_t backend_;
    StepGraph & sg_;
    int kq_stride_pad_;
    int fa_window_;

    // Cached vector form of capture layer IDs (built once in constructor).
    std::vector<int> capture_ids_;

    // LM-head projection graph (lazily built).
    StepGraph proj_sg_;

    // CPU-side copy of the last token's post-norm hidden state.
    // Filled by verify_batch after each GPU graph_compute.
    mutable std::vector<float> last_hidden_cpu_;

    // CPU-side copy of the full [n_tokens, n_embd] post-norm hidden sequence
    // from the last verify_batch.  Used by Qwen3.6 MTP warm_head_kv to seed
    // the head's per-position K/V cache during prefill.
    mutable std::vector<float> last_hidden_seq_cpu_;
    // CPU-side copy of the full [n_tokens, n_embd] PRE-final-output-norm
    // hidden sequence — populated alongside last_hidden_seq_cpu_ when
    // capture_hidden_seq_ is on.  Consumed by hidden_at_pos_pre_norm() so
    // the Qwen3.6 MTP chain's outer h_prev_0 seed dodges double-normalisation.
    mutable std::vector<float> last_hidden_seq_pre_norm_cpu_;
    mutable int                last_hidden_seq_n_ = 0;
    // Absolute position of the FIRST token captured in last_hidden_seq_cpu_.
    // Used by hidden_at_pos to translate an absolute sequence position to an
    // index inside the captured chunk.  -1 means no captured chunk yet.
    mutable int                last_verify_chunk_start_ = -1;

    // Whether verify_batch should request the full post-norm hidden sequence
    // and copy it to the host.  Toggled by enable_hidden_seq_capture().
    bool                       capture_hidden_seq_ = false;

    // Granularity of the device->host download when capture_hidden_seq_ is on.
    // Toggled by set_hidden_capture_mode().  See VerifyCaptureMode docs.
    VerifyCaptureMode          capture_mode_ = VerifyCaptureMode::FULL_SEQ;

    // ── Per-instance accumulators for DFLASH_VERIFY_PROFILE=1 ──
    // Summed wall-clock (ms) across every verify_batch call on this target;
    // emitted with a per-process line from the destructor.  All zero and
    // untouched when profiling is off.
    mutable double             vprof_sum_set_         = 0.0;
    mutable double             vprof_sum_compute_     = 0.0;
    mutable double             vprof_sum_get_hidden_  = 0.0;
    mutable double             vprof_sum_get_hpre_    = 0.0;
    mutable double             vprof_sum_get_argmax_  = 0.0;
    mutable double             vprof_sum_total_       = 0.0;
    mutable long long          vprof_n_calls_         = 0;

    // ── Stage 3: state captured by verify_tree for restore_kv_at_dfs() ──
    // base_pos of the most recent verify_tree call (= root slot in KV).
    int                        last_tree_base_pos_ = -1;
    // n_nodes of the most recent verify_tree.
    int                        last_tree_n_nodes_  = 0;
    // Copies of tree.parents/depths from the most recent verify_tree so
    // rollback's conv-window walk (which traverses ancestry, not DFS) is
    // self-contained.
    std::vector<int>           last_tree_parents_;
    std::vector<int>           last_tree_depths_;
};

}  // namespace dflash27b
