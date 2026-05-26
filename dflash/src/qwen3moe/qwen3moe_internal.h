// Qwen3MoeBackend — internal types for Qwen3-MoE inference.
//
// Architecture: Qwen3-MoE (e.g. Qwen3-30B-A3B, SR2AM-v1.0-30B).
//   - 48 layers, all routed-MoE (no dense lead, no shared expert).
//   - hidden=2048, n_ff_exp=768, n_expert=128, n_expert_used=8.
//   - 32 attention heads, 4 KV heads (GQA), head_dim=128.
//   - rope_theta=1e7, vocab=151936, max ctx 262144.
//   - Standard Qwen3 attention block (q_norm, k_norm, RoPE NEOX,
//     flash_attn_ext, GQA via Hk groups).
//
// GGUF tensor layout (one packed 3D tensor per layer per expert weight):
//   blk.N.ffn_gate_inp.weight        [n_embd, n_expert]
//   blk.N.ffn_gate_exps.weight       [n_embd, n_ff_exp, n_expert]
//   blk.N.ffn_up_exps.weight         [n_embd, n_ff_exp, n_expert]
//   blk.N.ffn_down_exps.weight       [n_ff_exp, n_embd, n_expert]
//
// This header mirrors the shape of `qwen3_drafter_model.h` (the standalone
// Qwen3-0.6B drafter weights) but adds the MoE-specific tensors and config.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "common/sampler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <random>

namespace dflash::common {

// ── Per-layer weights ──────────────────────────────────────────────────────

struct Qwen3MoeLayer {
    // Attention block (same shape as dense Qwen3)
    ggml_tensor * attn_norm = nullptr;  // [hidden]
    ggml_tensor * wq        = nullptr;  // [hidden, n_head * head_dim]
    ggml_tensor * wk        = nullptr;  // [hidden, n_head_kv * head_dim]
    ggml_tensor * wv        = nullptr;  // [hidden, n_head_kv * head_dim]
    ggml_tensor * wo        = nullptr;  // [n_head * head_dim, hidden]
    ggml_tensor * q_norm    = nullptr;  // [head_dim]
    ggml_tensor * k_norm    = nullptr;  // [head_dim]

    // MoE FFN block — packed 3D tensors over experts.
    ggml_tensor * ffn_norm     = nullptr;  // [hidden]
    ggml_tensor * ffn_gate_inp = nullptr;  // [hidden, n_expert]   router weights
    ggml_tensor * ffn_gate_exps = nullptr; // [hidden, n_ff_exp, n_expert]
    ggml_tensor * ffn_up_exps   = nullptr; // [hidden, n_ff_exp, n_expert]
    ggml_tensor * ffn_down_exps = nullptr; // [n_ff_exp, hidden, n_expert]
};

// ── Top-level weights ──────────────────────────────────────────────────────

struct Qwen3MoeWeights {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    ggml_tensor * tok_embd = nullptr;  // [hidden, vocab]
    ggml_tensor * out_norm = nullptr;  // [hidden]
    ggml_tensor * output   = nullptr;  // [hidden, vocab]  (lm_head)

    std::vector<Qwen3MoeLayer> layers;  // size == n_layer

    // Architecture metadata.
    int   n_layer        = 0;
    int   n_head         = 0;
    int   n_head_kv      = 0;
    int   n_embd         = 0;
    int   n_ff_exp       = 0;   // expert intermediate size (moe_intermediate_size)
    int   n_expert       = 0;
    int   n_expert_used  = 0;
    int   head_dim       = 0;
    int   n_vocab        = 0;
    int   n_ctx_max      = 0;
    float rope_theta     = 0.0f;
    float norm_eps       = 1e-6f;  // RMS-norm epsilon (qwen3moe.attention.layer_norm_rms_epsilon)
    bool  norm_topk_prob = true;
};

// ── KV cache (BF16 ring) ──────────────────────────────────────────────────

struct Qwen3MoeCache {
    int cur_pos = 0;
    int max_ctx = 0;
    int n_layer = 0;

    // Per-layer K/V: [head_dim, n_head_kv, max_ctx] in BF16/F16.
    // Position is the OUTER dim — reshape to [D*Hk, max_ctx] for set_rows
    // writes; view as [D, Hk, kv_len] + permute(0,2,1,3) for flash_attn reads.
    std::vector<ggml_tensor *> k;
    std::vector<ggml_tensor *> v;

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

// ── Snapshot (for prefix-cache / restore_and_generate) ────────────────────

struct Qwen3MoeSnapshot {
    int  cur_pos = 0;
    std::vector<ggml_tensor *> k_snap;
    std::vector<ggml_tensor *> v_snap;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int32_t last_tok = -1;
};

// ── Public C-ish API used by qwen3moe_backend.cpp ─────────────────────────

bool  create_qwen3moe_cache(ggml_backend_t backend,
                            const Qwen3MoeWeights & w,
                            int max_ctx,
                            Qwen3MoeCache & out);
void  free_qwen3moe_cache(Qwen3MoeCache & c);

void  free_qwen3moe_snapshot(Qwen3MoeSnapshot & s);

// Load Qwen3-MoE GGUF into `out`. Implemented in qwen3moe_loader.cpp.
bool  load_qwen3moe_gguf(const std::string & gguf_path,
                         ggml_backend_t backend,
                         Qwen3MoeWeights & out);
void  free_qwen3moe_weights(Qwen3MoeWeights & w);

}  // namespace dflash::common
