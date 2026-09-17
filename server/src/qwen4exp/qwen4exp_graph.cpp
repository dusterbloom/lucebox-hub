// Qwen4Exp forward graph — see qwen4exp_graph.h.
//
// Structure (upstream llama.cpp src/models/qwen4exp.cpp):
//   embed -> hc_init (hc copies of the embedding)
//   per layer:
//     [PLE on PLE layers]
//     hc_mix(attn) -> linear (gated delta net) | full (dense GQA) -> hc_combine
//     hc_mix(ffn)  -> MoE (512 top-10 + gated shared expert)        -> hc_combine
//   hc_mix(output) -> lm_head
//
// Single sequence (n_seqs = 1). The learned indexer is not consulted: dense
// full attention is exact and measured break-even against sparse selection.
// The PLE table is served by Qwen4ExpPleReader (host pread pool).

#include "qwen4exp_graph.h"

#include "delta_net_chunked.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash::common {
namespace {

ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, float s = 1.0f) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    return s == 1.0f ? y : ggml_scale(c, y, s);
}

// ── Hyper-connections ───────────────────────────────────────────────────

// build_hc_mix: grouped RMSNorm over one stream, low-rank mixer, sigmoid gate,
// mean-collapse across the hc streams; also returns the scatter injection.
[[maybe_unused]] ggml_tensor * hc_mix(ggml_context * c, ggml_tensor * x, ggml_tensor * w_norm,
                     ggml_tensor * w_down, ggml_tensor * w_up, ggml_tensor * w_inject,
                     ggml_tensor ** inject, int64_t n_embd, int64_t hc, float eps) {
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * xn = ggml_rms_norm(c, x, eps);          // per (hc, token)
    xn = ggml_reshape_2d(c, xn, hc_dim, nt);
    xn = ggml_mul(c, xn, w_norm);                         // [hc_dim, nt] gamma

    ggml_tensor * lo = mm(c, w_down, xn);                 // [hc_lr, nt]
    lo = ggml_silu(c, ggml_scale(c, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(c, mm(c, w_up, lo));// [hc_dim, nt]

    ggml_tensor * gated = ggml_mul(c, xn, gate);
    gated = ggml_reshape_3d(c, gated, n_embd, hc, nt);

    const size_t row = ggml_row_size(gated->type, n_embd);
    ggml_tensor * mixed = ggml_cont(c, ggml_view_2d(c, gated, n_embd, nt, row * hc, 0));
    for (int64_t s = 1; s < hc; ++s) {
        ggml_tensor * v = ggml_view_2d(c, gated, n_embd, nt, row * hc, row * s);
        mixed = ggml_add(c, mixed, v);
    }
    mixed = ggml_scale(c, mixed, 1.0f / (float) hc);

    if (inject) {
        *inject = mm(c, w_inject, xn);                    // [hc, nt]
    }
    return mixed;
}

// build_hc_combine: scatter the block output back across the hc residual
// streams with 2*sigmoid(inject) weights.
[[maybe_unused]] ggml_tensor * hc_combine(ggml_context * c, ggml_tensor * residual,
                         ggml_tensor * block_out, ggml_tensor * inject,
                         int64_t n_embd, int64_t hc, int64_t nt) {
    ggml_tensor * w = ggml_sigmoid(c, ggml_scale(c, inject, 1.0f / (float) hc));
    w = ggml_scale(c, w, 2.0f);
    w = ggml_reshape_3d(c, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(c, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(c, b, n_embd, hc, nt, 1);

    return ggml_add(c, residual, ggml_mul(c, b, w));
}

// ── MoE FFN: 512 experts top-10 (softmax), gated shared expert ──────────

[[maybe_unused]] ggml_tensor * build_moe(ggml_context * c, ggml_tensor * cur,
                        const Qwen4ExpLayer & L, const Qwen4ExpWeights & w) {
    const int64_t n_embd   = w.n_embd;
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_expert = w.n_expert;
    const int64_t n_used   = w.n_expert_used;

    ggml_tensor * logits = mm(c, L.ffn_gate_inp, cur);      // [n_expert, T]
    ggml_tensor * probs  = ggml_soft_max(c, logits);
    ggml_tensor * sel    = ggml_argsort_top_k(c, probs, (int) n_used);  // [n_used, T]

    ggml_tensor * probs3 = ggml_reshape_3d(c, probs, 1, n_expert, n_tokens);
    ggml_tensor * wsel   = ggml_reshape_2d(c, ggml_get_rows(c, probs3, sel), n_used, n_tokens);
    wsel = ggml_div(c, wsel, ggml_clamp(c, ggml_sum_rows(c, wsel), 6.103515625e-5f, INFINITY));

    ggml_tensor * cur3 = ggml_reshape_3d(c, cur, n_embd, 1, n_tokens);

    ggml_tensor * gate = ggml_mul_mat_id(c, L.ffn_gate_exps, cur3, sel);
    ggml_tensor * up   = ggml_mul_mat_id(c, L.ffn_up_exps,   cur3, sel);
    ggml_tensor * gu   = ggml_swiglu_split(c, gate, up);

    ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, gu, sel);   // [n_embd, n_used, T]
    ggml_tensor * wv   = ggml_reshape_3d(c, wsel, 1, n_used, n_tokens);
    down = ggml_mul(c, down, wv);

    ggml_tensor * sum_shape = ggml_new_tensor_3d(c, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * routed = ggml_reshape_2d(c, ggml_repeat_back(c, down, sum_shape), n_embd, n_tokens);

    // Shared expert with its own sigmoid gate.
    ggml_tensor * sh_gate = mm(c, L.ffn_gate_shexp, cur);
    ggml_tensor * sh_up   = mm(c, L.ffn_up_shexp, cur);
    ggml_tensor * sh_gu   = ggml_swiglu_split(c, sh_gate, sh_up);
    ggml_tensor * shared  = mm(c, L.ffn_down_shexp, sh_gu);

    ggml_tensor * shared_gate = ggml_sigmoid(c, mm(c, L.ffn_gate_inp_shexp, cur));
    shared = ggml_mul(c, shared, ggml_repeat(c, shared_gate, shared));

    return ggml_add(c, routed, shared);
}

}  // namespace

// The attention (gated delta net + dense GQA), PLE wiring, and the layer loop
// are the next increment. Until they land this returns a clean failure so the
// backend never emits tokens from a half-built graph.
Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits) {
    (void) backend;
    (void) w;
    (void) cache;
    (void) tokens;
    (void) n_tokens;
    (void) pos0;
    (void) out_logits;

    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
            "[qwen4exp] forward graph not implemented yet "
            "(HC + MoE helpers are in place; attention and PLE are next)\n");
        warned = true;
    }
    return Qwen4ExpForwardResult{};
}

}  // namespace dflash::common
