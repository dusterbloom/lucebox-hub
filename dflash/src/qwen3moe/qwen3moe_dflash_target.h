// Qwen3MoeDFlashTarget — DFlashTarget implementation for Qwen3-MoE models.
//
// Wraps Qwen3MoeWeights, Qwen3MoeCache, and DraftFeatureMirror behind the
// generic DFlashTarget interface so the universal DFlash spec-decode loop
// (run_dflash_spec_decode / Qwen35Backend::do_spec_decode pattern) can drive
// Qwen3-MoE verification without knowing its MoE or cache internals.

#pragma once

#include "common/dflash_target.h"
#include "common/dflash_feature_ring.h"
#include "common/step_graph.h"
#include "qwen3moe_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cstdint>

namespace dflash::common {

class Qwen3MoeDFlashTarget : public DFlashTarget {
public:
    // Non-owning references — caller must ensure lifetime.
    Qwen3MoeDFlashTarget(Qwen3MoeWeights    & w,
                         Qwen3MoeCache      & cache,
                         ggml_backend_t       backend,
                         DraftFeatureMirror & feature_mirror,
                         int                  fa_window,
                         int                  kq_stride_pad,
                         int                  n_capture_layers,
                         int                  mask_token_id);

    ~Qwen3MoeDFlashTarget() override;

    // ── DFlashTarget interface ────────────────────────────────────────────

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax = nullptr) override;

    bool snapshot_kv() override;
    bool restore_kv()  override;

    bool is_eos(int token) const override;

    bool embed_tokens(const int32_t * tokens, int n,
                      float * out) const override;

    bool project_hidden_to_tokens(const float * hidden,
                                  int n_tokens,
                                  std::vector<int32_t> & tokens_out) override;

    int  hidden_size()    const override { return w_.n_embd; }
    int  mask_token_id()  const override { return mask_token_id_; }
    const std::vector<int> & capture_layer_ids() const override { return capture_ids_; }

private:
    Qwen3MoeWeights    & w_;
    Qwen3MoeCache      & cache_;
    ggml_backend_t       backend_;
    DraftFeatureMirror & feature_mirror_;
    int                  fa_window_;
    int                  kq_stride_pad_;
    int                  mask_token_id_;
    std::vector<int>     capture_ids_;

    // LM-head projection graph (persistent gallocr, rebuilt lazily when n_tokens changes).
    StepGraph            proj_sg_;
    int                  proj_sg_n_ = 0;   // n_tokens last allocated for

    // KV rollback snapshot (allocated once, reused across calls).
    Qwen3MoeSnapshot     verify_snap_;
};

} // namespace dflash::common
