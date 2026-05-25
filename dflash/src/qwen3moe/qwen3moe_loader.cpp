// Qwen3MoeBackend GGUF loader — STUB.
//
// Phase A TODO (next session): read GGUF metadata under the `qwen3moe.*`
// namespace (block_count, embedding_length, attention.head_count/_kv,
// attention.key_length, expert_count, expert_used_count,
// expert_feed_forward_length, rope.freq_base), then for each layer load:
//   - attn_norm, attn_q/k/v/output (.weight)
//   - attn_q_norm, attn_k_norm
//   - ffn_norm
//   - ffn_gate_inp                       [n_embd, n_expert]
//   - ffn_gate_exps, ffn_up_exps         [n_embd, n_ff_exp, n_expert]
//   - ffn_down_exps                      [n_ff_exp, n_embd, n_expert]
// Plus the top-level token_embd, output_norm, output (lm_head).
//
// Adapt from dflash/src/qwen3/qwen3_loader.cpp::load_qwen3_drafter_model.
// Critical: do NOT inherit the 0.6B-specific default fallbacks (n_embd=1024
// etc.) — read every shape from the GGUF metadata. The IQ2_XXS/Q2_K weight
// types are supported by the bundled ggml-cuda; no special handling needed.

#include "qwen3moe_internal.h"

#include <cstdio>

namespace dflash::common {

bool load_qwen3moe_gguf(const std::string & gguf_path,
                        ggml_backend_t      /*backend*/,
                        Qwen3MoeWeights &   /*out*/) {
    std::fprintf(stderr,
                 "[qwen3moe-loader] STUB: load_qwen3moe_gguf not implemented yet "
                 "(path=%s)\n",
                 gguf_path.c_str());
    return false;
}

void free_qwen3moe_weights(Qwen3MoeWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
}

bool create_qwen3moe_cache(ggml_backend_t      /*backend*/,
                           const Qwen3MoeWeights & /*w*/,
                           int                 /*max_ctx*/,
                           Qwen3MoeCache &     /*out*/) {
    std::fprintf(stderr, "[qwen3moe-cache] STUB: create_qwen3moe_cache not implemented\n");
    return false;
}

void free_qwen3moe_cache(Qwen3MoeCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx);                c.ctx = nullptr; }
    c.k.clear();
    c.v.clear();
    c.cur_pos = 0;
}

void free_qwen3moe_snapshot(Qwen3MoeSnapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx);                s.ctx = nullptr; }
    s.k_snap.clear();
    s.v_snap.clear();
    s.cur_pos = 0;
    s.last_tok = -1;
}

}  // namespace dflash::common
