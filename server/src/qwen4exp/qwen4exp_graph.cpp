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
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash::common {
namespace {

ggml_tensor * mm(ggml_context * c, ggml_tensor * w, ggml_tensor * x, float s = 1.0f) {
    ggml_tensor * y = ggml_mul_mat(c, w, x);
    return s == 1.0f ? y : ggml_scale(c, y, s);
}

// Broadcast a [.., 1, ..] tensor along dim 1 to `hc` copies by concatenation.
// Avoids GGML_OP_REPEAT, which segfaults on gfx1151 for the hyper-connection
// shapes (the first op of the first forward).
ggml_tensor * repeat_dim1(ggml_context * c, ggml_tensor * x, int64_t hc) {
    ggml_tensor * r = x;
    for (int64_t i = 1; i < hc; ++i) {
        r = ggml_concat(c, r, x, 1);
    }
    return r;
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

    ggml_tensor * b = repeat_dim1(c, ggml_reshape_3d(c, block_out, n_embd, 1, nt), hc);

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
    shared = ggml_mul(c, shared, shared_gate);   // [n_embd,T] * [1,T] broadcasts over dim 0

    return ggml_add(c, routed, shared);
}

// ── Linear attention: gated delta net (36 layers) ───────────────────────

ggml_tensor * build_linear_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                                const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                                ggml_tensor * ssm_state, ggml_tensor * conv_state) {
    const int64_t D      = w.ssm_d_state;              // 128
    const int64_t Hk     = w.ssm_n_group;              // 16
    const int64_t Hv     = w.linear_value_heads;       // 48
    const int64_t d_in   = w.ssm_d_inner;              // 6144
    const int64_t T      = cur->ne[1];
    const int64_t kernel = w.ssm_d_conv;
    const int64_t conv_channels = 2 * Hk * D + d_in;   // 10240
    const float   eps    = w.rms_eps;

    ggml_tensor * qkv = mm(c, L.attn_qkv, cur);        // [conv_channels, T]
    ggml_tensor * z   = mm(c, L.attn_gate, cur);       // [d_in, T]

    ggml_tensor * beta = ggml_sigmoid(c,
        ggml_reshape_4d(c, mm(c, L.ssm_beta, cur), 1, Hv, T, 1));
    ggml_tensor * alpha = ggml_reshape_3d(c, mm(c, L.ssm_alpha, cur), Hv, T, 1);
    alpha = ggml_softplus(c, ggml_add(c, alpha, L.ssm_dt_bias));
    ggml_tensor * gate = ggml_reshape_4d(c, ggml_mul(c, alpha, L.ssm_a), 1, Hv, T, 1);

    // depthwise causal conv over [history | qkv]
    ggml_tensor * hist = ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1);
    ggml_tensor * qkv_t = ggml_reshape_3d(c,
        ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, qkv, conv_channels, T))),
        T, conv_channels, 1);
    ggml_tensor * conv_input = ggml_concat(c, hist, qkv_t, 0);

    // Advance T tokens along dim 0 (time); nb[0] is the element size, so the
    // tail offset is T*nb[0], NOT T*nb[1] (which steps whole channel rows).
    ggml_tensor * new_hist = ggml_cont(c, ggml_view_3d(c, conv_input, kernel - 1, conv_channels, 1,
        conv_input->nb[1], conv_input->nb[2], (size_t) T * conv_input->nb[0]));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_hist,
        ggml_reshape_3d(c, conv_state, kernel - 1, conv_channels, 1)));

    ggml_tensor * conv = ggml_silu(c, ggml_ssm_conv(c, conv_input, L.ssm_conv1d));

    const size_t esz     = ggml_element_size(conv);
    const size_t tstride = (size_t) conv_channels * esz;
    ggml_tensor * q_c = ggml_l2_norm(c, ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, 0), eps);
    ggml_tensor * k_c = ggml_l2_norm(c, ggml_view_3d(c, conv, D, Hk, T, D * esz, tstride, D * Hk * esz), eps);
    ggml_tensor * v_c = ggml_view_3d(c, conv, D, Hv, T, D * esz, tstride, 2 * D * Hk * esz);

    ggml_tensor * state4 = ggml_reshape_4d(c, ssm_state, D, D, Hv, 1);
    ggml_tensor * gdn = ggml_gated_delta_net(c, q_c, k_c, v_c, gate, beta, state4);

    // packed: [ attn S_v*H_v*T | final_state S_v*S_v*H_v | intermediates ]
    ggml_tensor * attn = ggml_view_4d(c, gdn, D, Hv, T, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * Hv),
        ggml_row_size(gdn->type, D * Hv * T), 0);
    ggml_tensor * new_state = ggml_view_4d(c, gdn, D, D, Hv, 1,
        ggml_row_size(gdn->type, D),
        ggml_row_size(gdn->type, D * D),
        ggml_row_size(gdn->type, D * D * Hv),
        ggml_row_size(gdn->type, D * Hv * T));
    ggml_build_forward_expand(gf, ggml_cpy(c, new_state, state4));

    // gated RMSNorm: sigmoid(z) gate (the one numerical difference from Qwen3.5)
    ggml_tensor * normed = ggml_mul(c, ggml_rms_norm(c, attn, eps), L.ssm_norm);
    ggml_tensor * out = ggml_mul(c, normed, ggml_sigmoid(c, ggml_reshape_4d(c, z, D, Hv, T, 1)));

    ggml_tensor * final = ggml_reshape_3d(c, out, d_in, T, 1);
    return ggml_reshape_2d(c, mm(c, L.ssm_out, final), w.n_embd, T);
}

// ── Full attention: dense GQA (12 layers) ───────────────────────────────

ggml_tensor * build_full_attn(ggml_context * c, ggml_cgraph * gf, ggml_tensor * cur,
                              const Qwen4ExpLayer & L, const Qwen4ExpWeights & w,
                              ggml_tensor * k_cache, ggml_tensor * v_cache,
                              ggml_tensor * positions, ggml_tensor * mask,
                              int64_t kv_len, int64_t pos0) {
    const int64_t D      = w.n_embd_head_k;   // 256
    const int64_t Hq     = w.n_head;          // 24
    const int64_t Hk     = w.n_head_kv;       // 2
    const int64_t T      = cur->ne[1];
    const float   eps    = w.rms_eps;

    // wq holds [q | gate] interleaved per head
    ggml_tensor * qfull = mm(c, L.wq, cur);   // [2*D*Hq, T]
    const size_t  qe    = ggml_element_size(qfull);
    ggml_tensor * Q = ggml_rms_norm(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, 0), eps);
    Q = ggml_mul(c, Q, L.q_norm);
    ggml_tensor * gate = ggml_cont_2d(c,
        ggml_view_3d(c, qfull, D, Hq, T, 2 * D * qe, 2 * D * Hq * qe, D * qe), D * Hq, T);

    ggml_tensor * K = ggml_rms_norm(c,
        ggml_reshape_3d(c, mm(c, L.wk, cur), D, Hk, T), eps);
    K = ggml_mul(c, K, L.k_norm);
    ggml_tensor * V = ggml_reshape_3d(c, mm(c, L.wv, cur), D, Hk, T);

    int sections[4] = { w.rope_sections[0], w.rope_sections[1],
                        w.rope_sections[2], w.rope_sections[3] };
    Q = ggml_rope_multi(c, Q, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_multi(c, K, positions, nullptr, w.rope_dimension_count, sections,
                        GGML_ROPE_TYPE_MROPE, 0, w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write K/V into the persistent cache at [pos0, pos0+T)
    ggml_tensor * Kt = ggml_permute(c, ggml_cast(c, K, k_cache->type), 0, 2, 1, 3);
    ggml_tensor * Vt = ggml_permute(c, ggml_cast(c, V, v_cache->type), 0, 2, 1, 3);
    ggml_build_forward_expand(gf, ggml_cpy(c, Kt,
        ggml_view_3d(c, k_cache, D, T, Hk, k_cache->nb[1], k_cache->nb[2],
                     k_cache->nb[1] * (size_t) pos0)));
    ggml_build_forward_expand(gf, ggml_cpy(c, Vt,
        ggml_view_3d(c, v_cache, D, T, Hk, v_cache->nb[1], v_cache->nb[2],
                     v_cache->nb[1] * (size_t) pos0)));

    ggml_tensor * K_full = ggml_view_3d(c, k_cache, D, kv_len, Hk,
        k_cache->nb[1], k_cache->nb[2], 0);
    ggml_tensor * V_full = ggml_view_3d(c, v_cache, D, kv_len, Hk,
        v_cache->nb[1], v_cache->nb[2], 0);

    ggml_tensor * Qfa = ggml_cont(c, ggml_permute(c, Q, 0, 2, 1, 3));  // [D, T, Hq]
    ggml_tensor * attn = ggml_flash_attn_ext(c, Qfa, K_full, V_full, mask,
        1.0f / std::sqrt((float) D), 0.0f, 0.0f);                       // [D, Hq, T, 1]

    ggml_tensor * attn2 = ggml_reshape_2d(c, attn, D * Hq, T);
    attn2 = ggml_mul(c, attn2, ggml_sigmoid(c, gate));
    return mm(c, L.wo, attn2);
}

// ── Per-layer n-gram embedding (PLE) ────────────────────────────────────

ggml_tensor * build_ple(ggml_context * c, ggml_cgraph * gf, ggml_tensor * hidden,
                        ggml_tensor * ple_emb, const Qwen4ExpLayer & L,
                        const Qwen4ExpWeights & w, ggml_tensor * ple_conv_state) {
    const int64_t n_embd = w.n_embd;
    const int64_t hc     = w.n_hc;
    const int64_t hc_dim = hc * n_embd;
    const int64_t T      = hidden->ne[2];
    const float   eps    = w.rms_eps;

    ggml_tensor * key   = mm(c, L.ple_key,   ple_emb);   // [hc_dim, T]
    ggml_tensor * value = mm(c, L.ple_value, ple_emb);   // [n_embd, T]

    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * nw) {
        ggml_tensor * t = ggml_reshape_3d(c, x, n_embd, hc, T);
        t = ggml_rms_norm(c, t, eps);
        t = ggml_reshape_2d(c, t, hc_dim, T);
        t = ggml_mul(c, t, nw);
        return ggml_reshape_3d(c, t, n_embd, hc, T);
    };

    key = grouped_norm(key, L.ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, L.ple_norm_query);

    ggml_tensor * s = ggml_sum_rows(c, ggml_mul(c, key, query));   // [1, hc, T]
    s = ggml_scale(c, s, 1.0f / std::sqrt((float) n_embd));
    ggml_tensor * mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, s), 1e-6f, INFINITY));
    ggml_tensor * ple_gate = ggml_sigmoid(c, ggml_mul(c, ggml_sgn(c, s), mag));

    ggml_tensor * v3 = repeat_dim1(c,
        ggml_reshape_3d(c, value, n_embd, 1, T), hc);
    ggml_tensor * gated = ggml_mul(c, v3, ple_gate);

    ggml_tensor * normalized = ggml_reshape_2d(c,
        grouped_norm(ggml_reshape_2d(c, gated, hc_dim, T), L.ple_norm_conv), hc_dim, T);

    // depthwise causal conv, dilated by the n-gram size: sum of shifted copies
    const int64_t kern = w.ple_conv_kernel;
    const int64_t dil  = w.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    ggml_tensor * norm_t = ggml_reshape_3d(c,
        ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, normalized, hc_dim, T))),
        T, hc_dim, 1);
    ggml_tensor * padded = ggml_concat(c,
        ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1), norm_t, 0);

    ggml_build_forward_expand(gf, ggml_cpy(c,
        ggml_cont(c, ggml_view_3d(c, padded, hist, hc_dim, 1, padded->nb[1], padded->nb[2],
                                  (size_t) T * padded->nb[0])),
        ggml_reshape_3d(c, ple_conv_state, hist, hc_dim, 1)));

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(c, ggml_transpose(c,
            ggml_view_3d(c, padded, T, hc_dim, 1, padded->nb[1], padded->nb[2],
                         ggml_row_size(padded->type, start))));
        ggml_tensor * wk = ggml_cont(c,
            ggml_view_2d(c, L.ple_conv1d, 1, hc_dim, L.ple_conv1d->nb[1],
                         k * L.ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(c, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(c, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(c, shifted, wk);
        conv_out = conv_out ? ggml_add(c, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(c, ggml_cont(c, ggml_silu(c, conv_out)), n_embd, hc, T);

    return ggml_add(c, hidden, ggml_add(c, gated, conv_out));
}

}  // namespace

Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits) {
    Qwen4ExpForwardResult res;
    if (n_tokens <= 0 || pos0 < 0 || !tokens) return res;
    if (pos0 + n_tokens > cache.max_ctx) {
        std::fprintf(stderr, "[qwen4exp] context overflow: %d + %d > %d\n",
                     pos0, n_tokens, cache.max_ctx);
        return res;
    }

    // map layer -> cache slot for the hybrid cache
    std::vector<int> lin_idx(w.n_layer, -1);
    std::vector<int> full_idx(w.n_layer, -1);
    for (size_t i = 0; i < cache.linear_layer_ids.size(); ++i) lin_idx[cache.linear_layer_ids[i]] = (int) i;
    for (size_t i = 0; i < cache.full_layer_ids.size(); ++i)   full_idx[cache.full_layer_ids[i]] = (int) i;

    // CPU embedding
    std::vector<float> emb((size_t) w.n_embd * n_tokens);
    if (!w.embedder.embed(tokens, n_tokens, emb.data())) {
        std::fprintf(stderr, "[qwen4exp] cpu embedding failed\n");
        return res;
    }

    // PLE n-gram rows (host side): reuse the cache's rolling token window.
    const bool has_ple = !cache.ple_layer_ids.empty() && w.ple_reader.available();
    const int64_t ple_heads = w.ple_n_heads;
    std::vector<int32_t> ple_rows(has_ple ? (size_t) ple_heads * n_tokens : 0);
    std::vector<float>   ple_data(has_ple ? (size_t) w.ple_head_dim * ple_heads * n_tokens : 0);
    std::vector<int32_t> ple_prev = cache.ple_prev;   // oldest first, size <= ng-1
    if (has_ple) {
        const int64_t ng = w.ple_ngram_size;
        std::vector<int32_t> seq = ple_prev;
        seq.insert(seq.end(), tokens, tokens + n_tokens);
        const int64_t base = (int64_t) ple_prev.size();
        for (int64_t i = 0; i < n_tokens; ++i) {
            const int64_t pos = base + i;
            std::vector<uint64_t> ctx(ng);
            ctx[0] = (uint64_t) tokens[i];
            bool cut = false;
            for (int64_t s = 1; s < ng; ++s) {
                if (cut || pos - s < 0) { ctx[s] = (uint64_t) w.ple_eos_token_id; cut = true; }
                else {
                    const int32_t t = seq[(size_t) (pos - s)];
                    if (t < 0 || t == w.ple_eos_token_id) cut = true;
                    ctx[s] = cut ? (uint64_t) w.ple_eos_token_id : (uint64_t) t;
                }
            }
            for (int64_t n = 2; n <= ng; ++n) {
                uint64_t mixed = ctx[0] * w.ple_layer_multipliers[0];
                for (int64_t j = 1; j < n; ++j) {
                    mixed ^= ctx[j] * w.ple_layer_multipliers[(size_t) j];
                }
                const int64_t head_base = (n - 2) * w.ple_heads_per_ngram;
                for (int64_t q = 0; q < w.ple_heads_per_ngram; ++q) {
                    const int64_t h = head_base + q;
                    const int64_t row = (int64_t) (mixed % (uint64_t) w.ple_head_vocab_sizes[h]) +
                                        w.ple_head_offsets[h];
                    ple_rows[(size_t) (i * ple_heads + h)] = (int32_t) row;
                }
            }
        }
        if (!w.ple_reader.gather(ple_rows.data(), (int64_t) ple_rows.size(), ple_data.data())) {
            std::fprintf(stderr, "[qwen4exp] PLE gather failed\n");
            return res;
        }
        // roll the window forward
        const size_t keep = (size_t) std::min<int64_t>(ng - 1, n_tokens + (int64_t) ple_prev.size());
        std::vector<int32_t> next;
        next.reserve(keep);
        const size_t total = ple_prev.size() + (size_t) n_tokens;
        for (size_t k = total - keep; k < total; ++k) {
            next.push_back(k < ple_prev.size() ? ple_prev[k] : tokens[k - ple_prev.size()]);
        }
        cache.ple_prev = std::move(next);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 200000 +
                  ggml_graph_overhead_custom(200000, false) + (1u << 20);
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return res;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 200000, false);

    const int64_t T      = n_tokens;
    const int64_t kv_len = pos0 + n_tokens;

    ggml_tensor * inp_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, T);
    ggml_set_input(inp_emb);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_input(positions);
    ggml_tensor * mask = nullptr;
    // Opt-in until validated on gfx1151: QWEN4EXP_KQ_MASK_DEV=1 builds the causal
    // mask on device instead of filling it on the host and uploading it.
    static const bool dev_mask = getenv("QWEN4EXP_KQ_MASK_DEV") != nullptr;
    if (T > 1) {
        if (!dev_mask) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, T);
            ggml_set_input(mask);
        } else {
            // fill(0) -> diag_mask_inf(pos0) -> f16, all on device.
            ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_len, T);
            m = ggml_fill_inplace(ctx, m, 0.0f);
            m = ggml_diag_mask_inf_inplace(ctx, m, (int) pos0);
            mask = ggml_cast(ctx, m, GGML_TYPE_F16);
        }
    }
    ggml_tensor * ple_in = nullptr;
    if (has_ple) {
        ple_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.ple_head_dim * ple_heads, T);
        ggml_set_input(ple_in);
    }

    ggml_tensor * res_hc = repeat_dim1(ctx,
        ggml_reshape_3d(ctx, inp_emb, w.n_embd, 1, T), w.n_hc);

    for (int il = 0; il < w.n_layer; ++il) {
        const Qwen4ExpLayer & L = w.layers[il];

        if (L.is_ple && has_ple) {
            res_hc = build_ple(ctx, gf, res_hc, ple_in, L, w,
                               cache.ple_conv_state.empty() ? nullptr :
                                   cache.ple_conv_state[0]);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = hc_mix(ctx, res_hc, L.hc_attn_norm, L.hc_attn_down,
                                   L.hc_attn_up, L.hc_attn_inject, &inject,
                                   w.n_embd, w.n_hc, w.rms_eps);
        if (L.is_full_attention) {
            const int fi = full_idx[il];
            cur = build_full_attn(ctx, gf, cur, L, w,
                                  cache.attn_k[fi], cache.attn_v[fi],
                                  positions, mask, kv_len, pos0);
        } else {
            const int li = lin_idx[il];
            cur = build_linear_attn(ctx, gf, cur, L, w,
                                    cache.ssm_state[li], cache.conv_state[li]);
        }
        res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, T);

        cur = hc_mix(ctx, res_hc, L.hc_ffn_norm, L.hc_ffn_down,
                     L.hc_ffn_up, L.hc_ffn_inject, &inject, w.n_embd, w.n_hc, w.rms_eps);
        cur = build_moe(ctx, cur, L, w);
        res_hc = hc_combine(ctx, res_hc, cur, inject, w.n_embd, w.n_hc, T);
    }

    ggml_tensor * final = hc_mix(ctx, res_hc, w.output_hc_norm, w.output_hc_down,
                                 w.output_hc_up, nullptr, nullptr,
                                 w.n_embd, w.n_hc, w.rms_eps);
    ggml_tensor * last = T > 1
        ? ggml_view_2d(ctx, final, w.n_embd, 1, final->nb[1], (size_t) (T - 1) * final->nb[1])
        : final;
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, last);
    ggml_set_output(logits);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen4exp] graph alloc failed (T=%lld kv_len=%lld)\n",
                     (long long) T, (long long) kv_len);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return res;
    }

    ggml_backend_tensor_set(inp_emb, emb.data(), 0, sizeof(float) * emb.size());
    // M-RoPE wants 4 sections per token, section-major: [s*T + i]. Sections
    // 0..2 carry the position, section 3 is zero (llama.cpp text convention).
    std::vector<int32_t> pos((size_t) 4 * T, 0);
    for (int64_t i = 0; i < T; ++i) {
        const int32_t p = (int32_t) (pos0 + i);
        pos[(size_t) (0 * T + i)] = p;
        pos[(size_t) (1 * T + i)] = p;
        pos[(size_t) (2 * T + i)] = p;
        pos[(size_t) (3 * T + i)] = 0;
    }
    ggml_backend_tensor_set(positions, pos.data(), 0, sizeof(int32_t) * pos.size());
    if (!dev_mask && mask) {
        std::vector<ggml_fp16_t> m((size_t) kv_len * T);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t row = 0; row < T; ++row) {
            const int64_t vis = pos0 + row;
            for (int64_t col = 0; col < kv_len; ++col) {
                m[(size_t) (row * kv_len + col)] = (col <= vis) ? zero : ninf;
            }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, sizeof(ggml_fp16_t) * m.size());
    }
    if (ple_in) {
        ggml_backend_tensor_set(ple_in, ple_data.data(), 0, sizeof(float) * ple_data.size());
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen4exp] graph compute failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return res;
    }

    out_logits.resize((size_t) w.n_vocab);
    ggml_backend_tensor_get(logits, out_logits.data(), 0, sizeof(float) * w.n_vocab);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);

    res.ok = true;
    res.n_tokens = n_tokens;
    res.pos0 = pos0;
    return res;
}

}  // namespace dflash::common

