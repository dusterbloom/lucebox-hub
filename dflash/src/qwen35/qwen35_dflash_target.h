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

    // Enable per-position post-norm hidden capture during verify_batch.
    // Off by default; MTP modules that depend on hidden_at_pos() flip it on
    // in attach().  Non-MTP paths (target_gen, DFlash drafter spec-decode)
    // leave it off and avoid pinning the full [n_embd, n_tokens] tensor.
    void enable_hidden_seq_capture(bool on) { capture_hidden_seq_ = on; }

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
    mutable int                last_hidden_seq_n_ = 0;
    // Absolute position of the FIRST token captured in last_hidden_seq_cpu_.
    // Used by hidden_at_pos to translate an absolute sequence position to an
    // index inside the captured chunk.  -1 means no captured chunk yet.
    mutable int                last_verify_chunk_start_ = -1;

    // Whether verify_batch should request the full post-norm hidden sequence
    // and copy it to the host.  Toggled by enable_hidden_seq_capture().
    bool                       capture_hidden_seq_ = false;
};

}  // namespace dflash27b
