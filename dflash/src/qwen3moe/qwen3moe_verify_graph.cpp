// Qwen3MoeDFlashTarget — verify graph builder.
//
// build_qwen3moe_verify_step: like do_step but with:
//   (a) n_tokens >= 1; proper causal mask always built.
//   (b) All-token logits (not just last token) so spec-decode verify can
//       compute argmax per position.
//   (c) Capture-layer tensors marked as outputs so the caller can copy them
//       to the DraftFeatureMirror after compute.
//
// The return value packs the named tensors needed by verify_batch:
//   logits_out       — [vocab, n_tokens] logits (argmax computed on host)
//   argmax_out       — [n_tokens] i32 argmax (ggml_argmax, GPU side)
//   capture_out[k]   — [hidden, n_tokens] f32 for each capture layer
//
// The graph is built fresh per call and uses a static gallocr so the CUDA
// buffer survives across calls (avoids cudaMalloc/cudaFree per verify step).

#include "qwen3moe_verify_graph.h"
#include "qwen3moe_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dflash::common {

bool build_qwen3moe_verify_graph(
        Qwen3MoeVerifyGraphResult & out,
        const Qwen3MoeWeights     & w,
        Qwen3MoeCache             & cache,
        ggml_backend_t              backend,
        int                         n_tokens,
        int                         kv_start,
        const std::vector<int>    & capture_ids)
{
    const int hidden   = w.n_embd;
    const int H        = w.n_head;
    const int Hk       = w.n_head_kv;
    const int D        = w.head_dim;
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const float eps    = w.norm_eps;
    const int kv_len   = kv_start + n_tokens;

    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    // Graph context — enough nodes for 48 MoE layers + capture outputs.
    constexpr int GRAPH_NODES = 16384;
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * GRAPH_NODES
                  + ggml_graph_overhead_custom(GRAPH_NODES, false)
                  + 8 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_NODES, false);

    // Input tensors.
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
    ggml_set_name(inp, "verify_inp");
    ggml_set_input(inp);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "verify_positions");
    ggml_set_input(positions);

    // Causal mask: always built (n_tokens can be 1 for replay).
    ggml_tensor * attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, n_tokens);
    ggml_set_name(attn_mask, "verify_attn_mask");
    ggml_set_input(attn_mask);

    // K/V write indices into the position-major cache [D, Hk, max_ctx].
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(k_idxs, "verify_k_idxs");
    ggml_set_input(k_idxs);
    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(v_idxs, "verify_v_idxs");
    ggml_set_input(v_idxs);

    ggml_tensor * cur = inp;

    // Capture tensors for the feature mirror.
    const int n_cap = (int)capture_ids.size();
    std::vector<ggml_tensor *> cap_outputs(n_cap, nullptr);

    // Per-layer forward pass (mirrors do_step exactly except for capture).
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[il];

        // ── Attention block ──────────────────────────────────────────────────

        ggml_tensor * normed = ggml_rms_norm(ctx, cur, eps);
        normed = ggml_mul(ctx, normed, L.attn_norm);

        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, normed);
        ggml_tensor * K = ggml_mul_mat(ctx, L.wk, normed);
        ggml_tensor * V = ggml_mul_mat(ctx, L.wv, normed);

        Q = ggml_reshape_3d(ctx, Q, D, H,  n_tokens);
        K = ggml_reshape_3d(ctx, K, D, Hk, n_tokens);
        V = ggml_reshape_3d(ctx, V, D, Hk, n_tokens);

        Q = ggml_rms_norm(ctx, Q, eps);
        Q = ggml_mul(ctx, Q, L.q_norm);
        K = ggml_rms_norm(ctx, K, eps);
        K = ggml_mul(ctx, K, L.k_norm);

        Q = ggml_rope_ext(ctx, Q, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr,
                          D, GGML_ROPE_TYPE_NEOX, 0,
                          w.rope_theta, 1.0f,
                          0.0f, 1.0f, 0.0f, 0.0f);

        // K/V cache layout: [D, Hk, max_ctx]. Writes via ggml_set_rows
        // with dynamic I64 idx tensors so the destination offset isn't
        // baked into graph node data pointers (mirrors qwen3moe_graph.cpp
        // and avoids the broken kv_start-as-view-offset pattern).
        (void)half_type;  // cache.k[il]->type drives the write conversion
        ggml_tensor * K_2d = ggml_reshape_2d(ctx, K, D * Hk, n_tokens);
        ggml_tensor * V_2d = ggml_reshape_2d(ctx, V, D * Hk, n_tokens);
        ggml_tensor * cache_k_2d = ggml_reshape_2d(ctx, cache.k[il], D * Hk, cache.max_ctx);
        ggml_tensor * cache_v_2d = ggml_reshape_2d(ctx, cache.v[il], D * Hk, cache.max_ctx);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k_2d, K_2d, k_idxs));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v_2d, V_2d, v_idxs));

        // Reads: view [D, Hk, kv_len] from the position-major cache then
        // permute (0,2,1,3) to [D, kv_len, Hk] for flash_attn_ext.
        ggml_tensor * K_full = ggml_view_3d(ctx, cache.k[il],
                                             D, Hk, kv_len,
                                             cache.k[il]->nb[1],
                                             cache.k[il]->nb[2], 0);
        ggml_tensor * V_full = ggml_view_3d(ctx, cache.v[il],
                                             D, Hk, kv_len,
                                             cache.v[il]->nb[1],
                                             cache.v[il]->nb[2], 0);
        K_full = ggml_permute(ctx, K_full, 0, 2, 1, 3);
        V_full = ggml_permute(ctx, V_full, 0, 2, 1, 3);

        ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
        Qfa = ggml_cont(ctx, Qfa);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, K_full, V_full,
                                                  attn_mask,
                                                  1.0f / std::sqrt((float)D),
                                                  0.0f, 0.0f);
        ggml_tensor * attn_2d = ggml_reshape_2d(ctx, attn, H * D, n_tokens);

        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn_2d);
        cur = ggml_add(ctx, cur, attn_out);

        // ── MoE FFN block ────────────────────────────────────────────────────

        ggml_tensor * ffn_in = ggml_rms_norm(ctx, cur, eps);
        ffn_in = ggml_mul(ctx, ffn_in, L.ffn_norm);

        ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_in);
        ggml_tensor * probs  = ggml_soft_max(ctx, logits);
        ggml_tensor * selected = ggml_top_k(ctx, probs, n_used);

        ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
        ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);

        weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights);
        weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, weights_sum);
        weights = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);

        ggml_tensor * cur_3d = ggml_reshape_3d(ctx, ffn_in, hidden, 1, n_tokens);

        ggml_tensor * gate_e = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur_3d, selected);
        ggml_tensor * up_e   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur_3d, selected);
        ggml_tensor * gu     = ggml_mul(ctx, ggml_silu(ctx, gate_e), up_e);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);
        experts = ggml_mul(ctx, experts, weights);

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

        cur = ggml_add(ctx, cur, routed);

        // ── Feature capture ──────────────────────────────────────────────────
        // After the residual add at each capture layer, make the hidden state
        // an explicit output so verify_batch can copy it to the feature mirror.
        for (int k = 0; k < n_cap; ++k) {
            if (capture_ids[k] == il) {
                // Force a contiguous copy so the tensor has a standalone buffer
                // (cur is typically a view/op chain; we need data pointer stable).
                ggml_tensor * cap = ggml_cont(ctx, cur);
                ggml_set_output(cap);
                char cap_name[32];
                std::snprintf(cap_name, sizeof(cap_name), "capture_%d", k);
                ggml_set_name(cap, cap_name);
                ggml_build_forward_expand(gf, cap);
                cap_outputs[k] = cap;
                break;
            }
        }
    }

    // Output norm + lm_head over ALL tokens (not just last).
    ggml_tensor * normed_out = ggml_rms_norm(ctx, cur, eps);
    normed_out = ggml_mul(ctx, normed_out, w.out_norm);
    ggml_tensor * logits_out = ggml_mul_mat(ctx, w.output, normed_out);
    ggml_set_name(logits_out, "verify_logits");
    ggml_set_output(logits_out);
    ggml_build_forward_expand(gf, logits_out);

    ggml_tensor * argmax_out = ggml_argmax(ctx, logits_out);
    ggml_set_name(argmax_out, "verify_argmax");
    ggml_set_output(argmax_out);
    ggml_build_forward_expand(gf, argmax_out);

    // Allocate with static gallocr.
    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[qwen3moe] verify graph alloc failed (n=%d kv=%d)\n",
                     n_tokens, kv_start);
        ggml_free(ctx);
        return false;
    }

    out.ctx       = ctx;
    out.gf        = gf;
    out.inp       = inp;
    out.positions = positions;
    out.attn_mask = attn_mask;
    out.k_idxs    = k_idxs;
    out.v_idxs    = v_idxs;
    out.argmax    = argmax_out;
    out.captures  = std::move(cap_outputs);
    return true;
}

} // namespace dflash::common
