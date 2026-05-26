// Qwen3MoeBackend — dflash ModelBackend for Qwen3-MoE inference.
//
// Phase A target: load Qwen3-MoE GGUF, run AR decode for /v1/chat/completions.
// Phase B (later): pair with a Qwen3-0.6B drafter for spec-decode.
//
// See plan: ~/.claude/plans/sparkling-hopping-porcupine.md

#pragma once

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "common/sampler.h"
#include "qwen3moe_internal.h"
#include "qwen3moe_dflash_target.h"        // Qwen3MoeDFlashTarget
#include "dflash_feature_ring.h"            // DraftFeatureMirror, draft_feature_mirror_init
#include "step_graph.h"                     // StepGraph, step_graph_destroy
#include "internal.h"                       // DraftWeights, load_draft_gguf, free_draft_weights

#include "ggml.h"
#include "ggml-backend.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen3MoeBackendConfig {
    const char *    model_path = nullptr;
    DevicePlacement device;
    int             stream_fd  = -1;
    int             chunk      = 512;
    // Reserved for Phase B (spec-decode with Qwen3-0.6B drafter).
    const char *    draft_path = nullptr;
    int             draft_gpu  = -1;
    int             draft_ctx_max = 4096;
    int             fa_window      = 2048;   // FA sliding window for verify_batch (Phase B.2)
    int             kq_stride_pad  = 32;     // mask alignment for FA; must equal KQ_MASK_PAD
};

class Qwen3MoeBackend : public ModelBackend {
public:
    explicit Qwen3MoeBackend(const Qwen3MoeBackendConfig & cfg);
    ~Qwen3MoeBackend() override;

    Qwen3MoeBackend(const Qwen3MoeBackend &)             = delete;
    Qwen3MoeBackend & operator=(const Qwen3MoeBackend &) = delete;

    bool init();

    // ── ModelBackend overrides ────────────────────────────────────────────
    void print_ready_banner() const override;

    bool park(const std::string & what)   override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO &        io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO &        io) override;

    bool handle_compress(const std::string & line,
                         const DaemonIO &    io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO &    io) override;

    bool supports_dflash_spec_decode() const override {
        return cfg_.draft_path != nullptr && feature_mirror_.target_feat != nullptr;
    }

    void shutdown() override;

    // Lazy accessor for the DFlash target adapter.
    DFlashTarget * dflash_target();

    bool do_spec_decode(int                           committed,
                        int                           n_gen,
                        std::vector<int32_t>        & out_tokens,
                        const DaemonIO              & io,
                        const std::vector<int32_t>  * hint_tokens = nullptr);

private:
    Qwen3MoeBackendConfig cfg_;
    ggml_backend_t        backend_ = nullptr;
    Qwen3MoeWeights       w_;
    Qwen3MoeCache         cache_;
    bool                  parked_ = false;

    SamplerCfg            sampler_;
    std::mt19937_64       sampler_rng_{std::random_device{}()};

    static constexpr int  PREFIX_SLOTS = 64;
    Qwen3MoeSnapshot      snapshots_[PREFIX_SLOTS];

    // ── DFlash drafter (Phase B) ──────────────────────────────────────────
    ggml_backend_t       draft_backend_  = nullptr;
    bool                 split_gpus_     = false;
    DraftWeights         dw_;
    DraftFeatureMirror   feature_mirror_;
    StepGraph            draft_sg_;

    // DFlash target adapter (Phase B.2/B.3) — created lazily.
    std::unique_ptr<Qwen3MoeDFlashTarget> dflash_target_;

    // Forward pass primitives (implemented in qwen3moe_graph.cpp /
    // qwen3moe_backend.cpp once Phase A wiring is in place).
    bool do_step(const float * embed,
                 int           n_tokens,
                 int           kv_start,
                 std::vector<float> & out_logits);

    int  do_prefill(const std::vector<int32_t> & tokens,
                    const DaemonIO &             io,
                    int                          kv_offset = 0);

    bool do_decode(int                            committed,
                   int                            n_gen,
                   std::vector<int32_t> &         out_tokens,
                   const DaemonIO &               io);

    std::vector<float> last_logits_;
    int32_t            last_prefill_tok_ = -1;   // argmax of prefill's final logit; seeds spec-decode

    // ── Cached decode graph (n_tokens==1 fast path) ──────────────────────
    // ggml-cuda's CUDA-graph cache keys off cgraph->nodes[0] pointer, and
    // requires identical node ne/nb/data across calls. We rebuild the graph
    // only when kv_len padded to DECODE_KV_PAD changes (every 256 tokens),
    // so the topology stays stable across most consecutive decode steps.
    static constexpr int DECODE_KV_PAD = 256;
    ggml_context * decode_ctx_           = nullptr;
    ggml_cgraph  * decode_gf_            = nullptr;
    ggml_gallocr_t decode_galloc_        = nullptr;
    ggml_tensor  * decode_inp_           = nullptr;  // unused (inline_embed path)
    ggml_tensor  * decode_token_ids_     = nullptr;  // [1]           I32 token id
    ggml_tensor  * decode_positions_     = nullptr;  // [1]           I32
    ggml_tensor  * decode_mask_          = nullptr;  // [kv_len_pad,1] F16
    ggml_tensor  * decode_k_idxs_        = nullptr;  // [1]           I64
    ggml_tensor  * decode_v_idxs_        = nullptr;  // [1]           I64
    ggml_tensor  * decode_logits_        = nullptr;  // [vocab, 1]   F32 output
    ggml_tensor  * decode_next_id_       = nullptr;  // [1]          I32 GPU argmax
    int            decode_kv_len_padded_ = -1;       // -1 = no cached graph

    // Cached-graph single-token decode step. Takes a token id directly and
    // runs ggml_get_rows for the embedding inside the cached graph — saves
    // one full graph_compute + D2H per token compared to embed_tokens + do_step.
    // For greedy sampling pass out_next_id (4-byte D2H from GPU argmax).
    // For temp>0 / penalties / top-p pass out_logits (full vocab D2H).
    bool do_decode_step(int32_t              token_id,
                        int                  kv_start,
                        std::vector<float> * out_logits,
                        int32_t            * out_next_id);
};

}  // namespace dflash::common
