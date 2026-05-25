// Qwen3MoeBackend — step-graph builder (Phase A.3).
//
// Implements Qwen3MoeBackend::do_step: one ggml graph covering all 48 MoE
// layers, K/V written into persistent cache, logits returned for the last
// token.
//
// Attention block:  verbatim port of qwen3_backend.cpp::do_step
//                   (Q/K/V proj, q_norm, k_norm, RoPE NEOX, BF16 cache
//                    view+cpy, flash_attn_ext, output proj, residual).
// MoE FFN block:    adapted from gemma4_graph.cpp::build_gemma4_moe_block
//                   with the 5 Qwen3-MoE deltas documented below.
//
// Qwen3-MoE FFN deltas vs Gemma4 MoE:
//   1. No shared expert  — SR2AM-30B mlp_only_layers=[] → skip ffn_gate/up/down.
//   2. SwiGLU (silu)     — not GELU.
//   3. Separate gate/up  — ffn_gate_exps + ffn_up_exps, NOT fused gate_up_exps.
//   4. norm_topk_prob    — renormalize top-k probs so they sum to 1 per token.
//   5. No aux norms      — no ffn_pre_norm_2, ffn_post_norm, ffn_gate_inp_s.

#include "qwen3moe_backend.h"

#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dflash::common {

bool Qwen3MoeBackend::do_step(const float * embed,
                               int           n_tokens,
                               int           kv_start,
                               std::vector<float> & out_logits) {
    const int hidden    = w_.n_embd;
    const int H         = w_.n_head;
    const int Hk        = w_.n_head_kv;
    const int D         = w_.head_dim;
    const int n_expert  = w_.n_expert;
    const int n_used    = w_.n_expert_used;
    const int vocab     = w_.n_vocab;
    const float eps     = w_.norm_eps;
    const int kv_len    = kv_start + n_tokens;

    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    // ── Graph context ────────────────────────────────────────────────────────
    // 8192 nodes is sufficient for 48 MoE layers; bump to 16384 if needed.
    constexpr int GRAPH_NODES = 8192;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * GRAPH_NODES
                  + ggml_graph_overhead_custom(GRAPH_NODES, false)
                  + 4 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "[qwen3moe] do_step: ggml_init failed\n");
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_NODES, false);

    // ── Input tensors ────────────────────────────────────────────────────────
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(inp, "inp");
    ggml_set_input(inp);
    ggml_tensor * cur = inp;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Causal mask — only needed for prefill (n_tokens > 1)
    ggml_tensor * attn_mask = nullptr;
    if (n_tokens > 1) {
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, n_tokens);
        ggml_set_name(attn_mask, "attn_mask");
        ggml_set_input(attn_mask);
    }

    // ── Per-layer forward ────────────────────────────────────────────────────
    for (int il = 0; il < w_.n_layer; ++il) {
        const auto & L = w_.layers[il];

        // ── Attention ────────────────────────────────────────────────────────

        // Pre-attention norm
        ggml_tensor * normed = ggml_rms_norm(ctx, cur, eps);
        normed = ggml_mul(ctx, normed, L.attn_norm);

        // Q/K/V projections (no bias)
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, normed);  // [H*D, n_tokens]
        ggml_tensor * K = ggml_mul_mat(ctx, L.wk, normed);  // [Hk*D, n_tokens]
        ggml_tensor * V = ggml_mul_mat(ctx, L.wv, normed);  // [Hk*D, n_tokens]

        // Reshape to [D, heads, n_tokens]
        Q = ggml_reshape_3d(ctx, Q, D, H,  n_tokens);
        K = ggml_reshape_3d(ctx, K, D, Hk, n_tokens);
        V = ggml_reshape_3d(ctx, V, D, Hk, n_tokens);

        // Per-head norms
        Q = ggml_rms_norm(ctx, Q, eps);
        Q = ggml_mul(ctx, Q, L.q_norm);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul(ctx, K, L.k_norm);

        // RoPE NEOX
        Q = ggml_rope_ext(ctx, Q, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w_.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w_.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

        // Cast K/V to BF16/F16 for cache
        ggml_tensor * K_half = ggml_cast(ctx, K, half_type);
        ggml_tensor * V_half = ggml_cast(ctx, V, half_type);

        // Permute [D, Hk, n_tokens] → [D, n_tokens, Hk] for cache write
        ggml_tensor * Kt = ggml_permute(ctx, K_half, 0, 2, 1, 3);
        ggml_tensor * Vt = ggml_permute(ctx, V_half, 0, 2, 1, 3);

        // Write into cache at kv_start..kv_start+n_tokens
        ggml_tensor * k_dst = ggml_view_3d(ctx, cache_.k[il],
                                            D, n_tokens, Hk,
                                            cache_.k[il]->nb[1],
                                            cache_.k[il]->nb[2],
                                            cache_.k[il]->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kt, k_dst));

        ggml_tensor * v_dst = ggml_view_3d(ctx, cache_.v[il],
                                            D, n_tokens, Hk,
                                            cache_.v[il]->nb[1],
                                            cache_.v[il]->nb[2],
                                            cache_.v[il]->nb[1] * (size_t)kv_start);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vt, v_dst));

        // Full K/V view [D, kv_len, Hk] from cache
        ggml_tensor * K_full = ggml_view_3d(ctx, cache_.k[il],
                                             D, kv_len, Hk,
                                             cache_.k[il]->nb[1],
                                             cache_.k[il]->nb[2], 0);
        ggml_tensor * V_full = ggml_view_3d(ctx, cache_.v[il],
                                             D, kv_len, Hk,
                                             cache_.v[il]->nb[1],
                                             cache_.v[il]->nb[2], 0);

        // Permute Q [D, H, n_tokens] → [D, n_tokens, H] for flash_attn_ext
        ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
        Qfa = ggml_cont(ctx, Qfa);

        // Flash attention
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, K_full, V_full,
                                                  attn_mask,
                                                  1.0f / std::sqrt((float)D),
                                                  0.0f, 0.0f);
        ggml_tensor * attn_2d = ggml_reshape_2d(ctx, attn, H * D, n_tokens);

        // Output projection + residual (no bias)
        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn_2d);
        cur = ggml_add(ctx, cur, attn_out);

        // ── MoE FFN ──────────────────────────────────────────────────────────

        // Pre-FFN norm
        ggml_tensor * ffn_in = ggml_rms_norm(ctx, cur, eps);
        ffn_in = ggml_mul(ctx, ffn_in, L.ffn_norm);

        // Router: direct mul_mat (no rms_norm/scale/ffn_gate_inp_s — delta #5)
        ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_in);  // [n_expert, n_tokens]

        // Softmax over expert dimension
        ggml_tensor * probs = ggml_soft_max(ctx, logits);

        // Top-k on probs (post-softmax — not on raw logits)
        ggml_tensor * selected = ggml_top_k(ctx, probs, n_used);  // i32 [n_used, n_tokens]

        // Gather top-k probs: probs → [1, n_expert, n_tokens] → get_rows → [1, n_used, n_tokens]
        ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
        ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);

        // norm_topk_prob renorm (delta #4 — OMITTING BREAKS MATH500):
        //   weights [1, n_used, n_tokens] → [n_used, n_tokens]
        //   sum_rows → [1, n_tokens], clamp, div, reshape back to [1, n_used, n_tokens]
        weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);          // [1, n_tokens]
        weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, weights_sum);                    // broadcast
        weights = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);

        // Reshape input for mul_mat_id: [n_embd, 1, n_tokens]
        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, ffn_in, hidden, 1, n_tokens);

        // Two separate mul_mat_id calls (delta #3 — NOT fused like Gemma4)
        ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, selected);
        // gate_e: [n_ff_exp, n_used, n_tokens]
        ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur_3d, selected);
        // up_e:   [n_ff_exp, n_used, n_tokens]

        // SwiGLU (delta #2 — silu, not gelu)
        ggml_tensor * gu = ggml_mul(ctx, ggml_silu(ctx, gate_e), up_e);

        // Down projection
        ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);
        // experts: [n_embd, n_used, n_tokens]

        // Apply weights (broadcast over n_embd)
        experts = ggml_mul(ctx, experts, weights);

        // Weighted sum over expert dimension (Gemma4 pattern, gemma4_graph.cpp:127-133)
        ggml_build_forward_expand(gf, experts);
        ggml_tensor * routed = nullptr;
        for (int i = 0; i < n_used; ++i) {
            ggml_tensor * slice = ggml_view_2d(ctx, experts,
                hidden, n_tokens,
                experts->nb[2],
                (size_t)i * experts->nb[1]);
            ggml_build_forward_expand(gf, slice);
            routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
        }

        // Residual (onto cur, not ffn_in — delta #5 of spec)
        cur = ggml_add(ctx, cur, routed);
    }

    // ── Output norm + lm_head (last token only) ──────────────────────────────
    ggml_tensor * last_hidden;
    if (n_tokens > 1) {
        last_hidden = ggml_view_2d(ctx, cur, hidden, 1,
                                    cur->nb[1],
                                    (size_t)(n_tokens - 1) * cur->nb[1]);
    } else {
        last_hidden = cur;
    }
    ggml_tensor * normed_out = ggml_rms_norm(ctx, last_hidden, eps);
    normed_out = ggml_mul(ctx, normed_out, w_.out_norm);
    ggml_tensor * logits_out = ggml_mul_mat(ctx, w_.output, normed_out);
    ggml_set_output(logits_out);
    ggml_set_name(logits_out, "logits");
    ggml_build_forward_expand(gf, logits_out);

    // ── Allocate and compute ──────────────────────────────────────────────────
    // Static gallocr: allocated once, reused across all decode/prefill steps.
    // Avoids per-token cudaMalloc/cudaFree (~5-10ms/token). Pattern mirrors
    // gemma4_graph.cpp:513. Leaked at process exit, fine for long-running server.
    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr,
            "[qwen3moe] graph alloc failed (n_tokens=%d kv_start=%d kv_len=%d)\n",
            n_tokens, kv_start, kv_len);
        ggml_free(ctx);
        return false;
    }

    // Fill inputs
    ggml_backend_tensor_set(inp, embed, 0,
                            sizeof(float) * (size_t)hidden * n_tokens);

    {
        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
        ggml_backend_tensor_set(positions, pos.data(), 0,
                                sizeof(int32_t) * n_tokens);
    }

    if (attn_mask) {
        // Causal mask: token i attends to positions [0, kv_start+i]
        std::vector<ggml_fp16_t> mask_data((size_t)kv_len * n_tokens);
        const ggml_fp16_t zero_h    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int row = 0; row < n_tokens; ++row) {
            const int last_visible = kv_start + row;
            for (int col = 0; col < kv_len; ++col) {
                mask_data[(size_t)row * kv_len + col] =
                    (col <= last_visible) ? zero_h : neg_inf_h;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                                sizeof(ggml_fp16_t) * mask_data.size());
    }

    auto st = ggml_backend_graph_compute(backend_, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[qwen3moe] graph compute failed (status=%d n_tokens=%d kv_start=%d)\n",
            (int)st, n_tokens, kv_start);
        ggml_free(ctx);
        return false;
    }

    // Read logits
    out_logits.resize(vocab);
    ggml_backend_tensor_get(logits_out, out_logits.data(), 0,
                            sizeof(float) * vocab);

    ggml_free(ctx);
    return true;
}

}  // namespace dflash::common
