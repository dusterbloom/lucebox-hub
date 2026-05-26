// Qwen3MoeBackend — step-graph builder (Phase A.3 + Phase 2 CUDA-graph-replay).
//
// Implements Qwen3MoeBackend::do_step: one ggml graph covering all 48 MoE
// layers, K/V written into persistent cache, logits returned for the last
// token.
//
// Attention block:  verbatim port of qwen3_backend.cpp::do_step
//                   (Q/K/V proj, q_norm, k_norm, RoPE NEOX, BF16 cache
//                    write via ggml_set_rows, flash_attn_ext, output proj,
//                    residual).
// MoE FFN block:    adapted from gemma4_graph.cpp::build_gemma4_moe_block
//                   with the 5 Qwen3-MoE deltas documented below.
//
// Qwen3-MoE FFN deltas vs Gemma4 MoE:
//   1. No shared expert  — SR2AM-30B mlp_only_layers=[] → skip ffn_gate/up/down.
//   2. SwiGLU (silu)     — not GELU.
//   3. Separate gate/up  — ffn_gate_exps + ffn_up_exps, NOT fused gate_up_exps.
//   4. norm_topk_prob    — renormalize top-k probs so they sum to 1 per token.
//   5. No aux norms      — no ffn_pre_norm_2, ffn_post_norm, ffn_gate_inp_s.
//
// Phase 2 graph reuse (n_tokens=1 decode):
//   - K/V cache layout is [D, Hk, max_ctx] (positions are OUTER), so each
//     kv slot is D*Hk-element contiguous. Per-step K/V writes use
//     ggml_set_rows with an I64 input tensor of indices — the write position
//     is *contents* of the index tensor, not a graph-topology offset. This
//     keeps node->src->data pointers stable across calls so ggml-cuda's
//     CUDA-graph cache (gated on GGML_CUDA_GRAPHS) hits warmup and replays
//     the captured kernel stream instead of reissuing ~1500 launches per
//     token.
//   - kv_len for K/V *read* views is padded up to DECODE_KV_PAD (256) so
//     view ne shapes stay identical across many consecutive decode steps.
//     A mask zeros out positions past the real kv_len. Graph topology
//     only changes every 256 tokens.
//   - The ggml_context + cgraph + gallocr + input/output tensor handles
//     are held on the backend and reused. The graph is rebuilt only when
//     the padded kv_len changes.
//
// Pattern reference: llama.cpp/src/llama-kv-cache.cpp::cpy_k (set_rows on
// reshaped [n_embd_gqa, kv_size]) and ::get_n_kv (pad to 256).

#include "qwen3moe_backend.h"

#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dflash::common {

// Per-layer node construction. Builds Q/K/V → flash_attn → MoE FFN for every
// transformer layer plus final norm + lm_head, then ggml_build_forward_expand
// onto `gf`. Caller owns `ctx` and `gf`. Input/output tensor handles are
// returned via out-params so the caller can ggml_backend_tensor_set the
// inputs per step.
//
// `kv_len_for_views` is the K/V read-view size — for decode this is the
// 256-padded value, for prefill it is the raw kv_start + n_tokens. Mask
// covers [0, kv_len_for_views) with -inf for positions past the true
// kv_len. Cache writes are at indices supplied at run time via k_idxs/v_idxs.
static void build_qwen3moe_step_nodes(
    ggml_context *           ctx,
    ggml_cgraph *            gf,
    const Qwen3MoeWeights &  w,
    const Qwen3MoeCache &    cache,
    int                      n_tokens,
    int                      kv_len_for_views,
    ggml_type                half_type,
    bool                     inline_embed,
    ggml_tensor **           out_inp,      // null when inline_embed
    ggml_tensor **           out_token_ids,// null when !inline_embed
    ggml_tensor **           out_positions,
    ggml_tensor **           out_mask,
    ggml_tensor **           out_k_idxs,
    ggml_tensor **           out_v_idxs,
    ggml_tensor **           out_logits) {

    const int hidden    = w.n_embd;
    const int H         = w.n_head;
    const int Hk        = w.n_head_kv;
    const int D         = w.head_dim;
    const int n_expert  = w.n_expert;
    const int n_used    = w.n_expert_used;
    const float eps     = w.norm_eps;
    const int max_ctx   = cache.max_ctx;

    // ── Input tensors ────────────────────────────────────────────────────────
    // Either take F32 [hidden, n_tokens] embeddings directly (prefill path)
    // OR take I32 [n_tokens] token ids and run ggml_get_rows inline (decode
    // fast path — saves one full graph_compute + D2H per token).
    ggml_tensor * inp       = nullptr;
    ggml_tensor * token_ids = nullptr;
    ggml_tensor * cur       = nullptr;
    if (inline_embed) {
        token_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_name(token_ids, "token_ids");
        ggml_set_input(token_ids);
        cur = ggml_get_rows(ctx, w.tok_embd, token_ids);  // [hidden, n_tokens] F32
    } else {
        inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, n_tokens);
        ggml_set_name(inp, "inp");
        ggml_set_input(inp);
        cur = inp;
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * attn_mask =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len_for_views, n_tokens);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(k_idxs, "k_idxs");
    ggml_set_input(k_idxs);

    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(v_idxs, "v_idxs");
    ggml_set_input(v_idxs);

    // ── Per-layer forward ────────────────────────────────────────────────────
    for (int il = 0; il < w.n_layer; ++il) {
        const auto & L = w.layers[il];

        // ── Attention ────────────────────────────────────────────────────────
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

        // K/V are [D, Hk, n_tokens] F32 after rope. Write directly into the
        // cache via set_rows — its kernel converts F32 → cache half-type on
        // the fly (ggml_set_rows asserts src is F32). Cache shape:
        // [D, Hk, max_ctx] → reshape to [D*Hk, max_ctx]; source K/V →
        // reshape to [D*Hk, n_tokens]; indices → [n_tokens] I64.
        (void)half_type;  // cache type now driven by cache.k[il]->type
        ggml_tensor * K_2d  = ggml_reshape_2d(ctx, K, D * Hk, n_tokens);
        ggml_tensor * V_2d  = ggml_reshape_2d(ctx, V, D * Hk, n_tokens);

        ggml_tensor * cache_k_2d =
            ggml_reshape_2d(ctx, cache.k[il], D * Hk, max_ctx);
        ggml_tensor * cache_v_2d =
            ggml_reshape_2d(ctx, cache.v[il], D * Hk, max_ctx);

        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, cache_k_2d, K_2d, k_idxs));
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, cache_v_2d, V_2d, v_idxs));

        // Read views: take first kv_len_for_views slots from cache, then
        // permute (0,2,1,3) so flash_attn_ext sees [D, kv_len, Hk].
        ggml_tensor * K_full = ggml_view_3d(ctx, cache.k[il],
                                             D, Hk, kv_len_for_views,
                                             cache.k[il]->nb[1],
                                             cache.k[il]->nb[2], 0);
        ggml_tensor * V_full = ggml_view_3d(ctx, cache.v[il],
                                             D, Hk, kv_len_for_views,
                                             cache.v[il]->nb[1],
                                             cache.v[il]->nb[2], 0);
        K_full = ggml_permute(ctx, K_full, 0, 2, 1, 3);  // [D, kv_len, Hk]
        V_full = ggml_permute(ctx, V_full, 0, 2, 1, 3);

        // Permute Q [D, H, n_tokens] → [D, n_tokens, H] for flash_attn_ext
        ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
        Qfa = ggml_cont(ctx, Qfa);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, K_full, V_full,
                                                  attn_mask,
                                                  1.0f / std::sqrt((float)D),
                                                  0.0f, 0.0f);
        ggml_tensor * attn_2d = ggml_reshape_2d(ctx, attn, H * D, n_tokens);

        ggml_tensor * attn_out = ggml_mul_mat(ctx, L.wo, attn_2d);
        cur = ggml_add(ctx, cur, attn_out);

        // ── MoE FFN ──────────────────────────────────────────────────────────
        ggml_tensor * ffn_in = ggml_rms_norm(ctx, cur, eps);
        ffn_in = ggml_mul(ctx, ffn_in, L.ffn_norm);

        ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_in);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
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
        ggml_tensor * gu = ggml_mul(ctx, ggml_silu(ctx, gate_e), up_e);
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
    normed_out = ggml_mul(ctx, normed_out, w.out_norm);
    ggml_tensor * logits_out = ggml_mul_mat(ctx, w.output, normed_out);
    ggml_set_output(logits_out);
    ggml_set_name(logits_out, "logits");
    ggml_build_forward_expand(gf, logits_out);

    if (out_inp)       *out_inp       = inp;
    if (out_token_ids) *out_token_ids = token_ids;
    *out_positions = positions;
    *out_mask      = attn_mask;
    *out_k_idxs    = k_idxs;
    *out_v_idxs    = v_idxs;
    *out_logits    = logits_out;
}

// Same as build_qwen3moe_step_nodes but also adds a GPU argmax on the logits
// output, exposed as an I32 [1] tensor. Used by the inline-embed decode path
// so greedy sampling skips the 608 KB logits D2H.
static void build_qwen3moe_step_nodes_with_argmax(
    ggml_context *           ctx,
    ggml_cgraph *            gf,
    const Qwen3MoeWeights &  w,
    const Qwen3MoeCache &    cache,
    int                      n_tokens,
    int                      kv_len_for_views,
    ggml_type                half_type,
    bool                     inline_embed,
    ggml_tensor **           out_inp,
    ggml_tensor **           out_token_ids,
    ggml_tensor **           out_positions,
    ggml_tensor **           out_mask,
    ggml_tensor **           out_k_idxs,
    ggml_tensor **           out_v_idxs,
    ggml_tensor **           out_logits,
    ggml_tensor **           out_next_id) {

    build_qwen3moe_step_nodes(
        ctx, gf, w, cache, n_tokens, kv_len_for_views, half_type, inline_embed,
        out_inp, out_token_ids, out_positions, out_mask,
        out_k_idxs, out_v_idxs, out_logits);

    ggml_tensor * next_id = ggml_argmax(ctx, *out_logits);
    ggml_set_output(next_id);
    ggml_set_name(next_id, "next_id");
    ggml_build_forward_expand(gf, next_id);
    *out_next_id = next_id;
}

bool Qwen3MoeBackend::do_step(const float * embed,
                               int           n_tokens,
                               int           kv_start,
                               std::vector<float> & out_logits) {
    const int hidden    = w_.n_embd;
    const int vocab     = w_.n_vocab;
    const int kv_len    = kv_start + n_tokens;
    const int max_ctx   = cache_.max_ctx;

    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    // ── Fresh-build path (prefill / multi-token verify) ──────────────────────
    // For the n_tokens=1 hot path, callers should use do_decode_step() — it
    // takes a token id directly and runs the embed lookup inside a cached
    // CUDA-graph-replayable graph.
    (void)max_ctx;  // unused in this path
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

    ggml_tensor * inp        = nullptr;
    ggml_tensor * positions  = nullptr;
    ggml_tensor * attn_mask  = nullptr;
    ggml_tensor * k_idxs     = nullptr;
    ggml_tensor * v_idxs     = nullptr;
    ggml_tensor * logits_out = nullptr;

    build_qwen3moe_step_nodes(
        ctx, gf, w_, cache_,
        n_tokens,
        /*kv_len_for_views=*/kv_len,
        half_type,
        /*inline_embed=*/false,
        &inp, /*out_token_ids=*/nullptr,
        &positions, &attn_mask, &k_idxs, &v_idxs, &logits_out);

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

    ggml_backend_tensor_set(inp, embed, 0,
                            sizeof(float) * (size_t)hidden * n_tokens);
    {
        std::vector<int32_t> pos(n_tokens);
        for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
        ggml_backend_tensor_set(positions, pos.data(), 0,
                                sizeof(int32_t) * n_tokens);
    }
    {
        std::vector<int64_t> idx(n_tokens);
        for (int i = 0; i < n_tokens; ++i) idx[i] = kv_start + i;
        ggml_backend_tensor_set(k_idxs, idx.data(), 0,
                                sizeof(int64_t) * n_tokens);
        ggml_backend_tensor_set(v_idxs, idx.data(), 0,
                                sizeof(int64_t) * n_tokens);
    }
    {
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

    out_logits.resize(vocab);
    ggml_backend_tensor_get(logits_out, out_logits.data(), 0,
                            sizeof(float) * vocab);

    ggml_free(ctx);
    return true;
}

// ── Single-token decode step (cached graph, inline embed, GPU argmax) ────────
bool Qwen3MoeBackend::do_decode_step(int32_t              token_id,
                                      int                  kv_start,
                                      std::vector<float> * out_logits,
                                      int32_t            * out_next_id) {
    const int vocab   = w_.n_vocab;
    const int kv_len  = kv_start + 1;
    const int max_ctx = cache_.max_ctx;

    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    int kv_len_padded =
        ((kv_len + DECODE_KV_PAD - 1) / DECODE_KV_PAD) * DECODE_KV_PAD;
    if (kv_len_padded > max_ctx) kv_len_padded = max_ctx;

    if (!decode_gf_ || decode_kv_len_padded_ != kv_len_padded) {
        constexpr int GRAPH_NODES = 8192;
        if (!decode_ctx_) {
            ggml_init_params ip{};
            ip.mem_size = ggml_tensor_overhead() * GRAPH_NODES
                          + ggml_graph_overhead_custom(GRAPH_NODES, false)
                          + 4 * 1024 * 1024;
            ip.no_alloc = true;
            decode_ctx_ = ggml_init(ip);
            if (!decode_ctx_) {
                std::fprintf(stderr,
                    "[qwen3moe] do_decode_step: ggml_init failed\n");
                return false;
            }
            decode_galloc_ = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
        } else {
            ggml_reset(decode_ctx_);
        }
        decode_gf_ = ggml_new_graph_custom(decode_ctx_, GRAPH_NODES, false);

        ggml_tensor * mask = nullptr;
        build_qwen3moe_step_nodes_with_argmax(
            decode_ctx_, decode_gf_, w_, cache_,
            /*n_tokens=*/1,
            /*kv_len_for_views=*/kv_len_padded,
            half_type,
            /*inline_embed=*/true,
            /*out_inp=*/nullptr, &decode_token_ids_,
            &decode_positions_, &mask,
            &decode_k_idxs_, &decode_v_idxs_, &decode_logits_,
            &decode_next_id_);
        decode_mask_ = mask;

        if (!ggml_gallocr_alloc_graph(decode_galloc_, decode_gf_)) {
            std::fprintf(stderr,
                "[qwen3moe] decode gallocr_alloc_graph failed (kv_pad=%d)\n",
                kv_len_padded);
            return false;
        }
        decode_kv_len_padded_ = kv_len_padded;
    }

    // Per-step inputs.
    ggml_backend_tensor_set(decode_token_ids_, &token_id, 0, sizeof(int32_t));
    {
        int32_t pos = kv_start;
        ggml_backend_tensor_set(decode_positions_, &pos, 0, sizeof(int32_t));
    }
    {
        int64_t idx = kv_start;
        ggml_backend_tensor_set(decode_k_idxs_, &idx, 0, sizeof(int64_t));
        ggml_backend_tensor_set(decode_v_idxs_, &idx, 0, sizeof(int64_t));
    }
    {
        std::vector<ggml_fp16_t> mask_data((size_t)decode_kv_len_padded_);
        const ggml_fp16_t zero_h    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf_h = ggml_fp32_to_fp16(-INFINITY);
        for (int j = 0; j < decode_kv_len_padded_; ++j) {
            mask_data[j] = (j <= kv_start) ? zero_h : neg_inf_h;
        }
        ggml_backend_tensor_set(decode_mask_, mask_data.data(), 0,
                                sizeof(ggml_fp16_t) * mask_data.size());
    }

    auto st = ggml_backend_graph_compute(backend_, decode_gf_);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr,
            "[qwen3moe] decode graph compute failed (status=%d kv_start=%d)\n",
            (int)st, kv_start);
        return false;
    }

    if (out_next_id) {
        ggml_backend_tensor_get(decode_next_id_, out_next_id, 0, sizeof(int32_t));
    }
    if (out_logits) {
        out_logits->resize(vocab);
        ggml_backend_tensor_get(decode_logits_, out_logits->data(), 0,
                                sizeof(float) * vocab);
    }
    return true;
}

}  // namespace dflash::common
