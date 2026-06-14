// Gemma4 forward graph builder + step function.
//
// Architecture (from deps/llama.cpp/src/models/gemma4-iswa.cpp):
//   - Scale input embeddings by sqrt(n_embd)
//   - For each layer:
//     a. Pre-attn RMSNorm
//     b. Q/K/V projections + per-head Q/K RMSNorm + RoPE
//        (KV-sharing layers skip K/V proj, reuse source layer's KV cache)
//     c. Write K/V to cache, flash attention (full or SWA)
//     d. Post-attn RMSNorm + residual
//     e. Dense FFN (lead layer) or MoE (shared GELU-gated + routed experts)
//     f. FFN post-norm + residual
//     g. Per-layer embedding injection (gated)
//     h. Output scale
//   - Final RMSNorm + lm_head
//   - Logit softcapping: tanh(logits/cap)*cap

#include "gemma4_internal.h"
#include "common/ggml_graph_precision.h"
#include "common/gpu_runtime_compat.h"
#include "../common/kvflash_pager.h"
#include "dflash27b.h"
#include "flashprefill.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

#ifdef DFLASH27B_BACKEND_CUDA
#include "diffusion/diffusion_sampling.h"
#include <cuda_runtime.h>
#endif

namespace dflash::common {

static constexpr float GEMMA4_EPS = 1e-6f;

static ggml_tensor * gemma4_rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                          ggml_tensor * weight, float eps = GEMMA4_EPS) {
    x = rms_norm_input_f32(ctx, x);
    weight = graph_tensor_f32(ctx, weight);
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

// Dense GELU-gated FFN (layer 0 / lead dense layers).
// Gemma4 uses GELU not SiLU: cur = down( gelu(gate(x)) * up(x) )
static ggml_tensor * build_gemma4_dense_ffn(ggml_context * ctx, ggml_tensor * cur,
                                              const Gemma4Layer & L) {
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate, cur);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up,   cur);
    // Use fused geglu_split to match ref build_ffn(LLM_FFN_GELU, LLM_FFN_PAR)
    ggml_tensor * gu   = ggml_geglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.ffn_down, gu);
}

// MoE block: shared expert (GELU-gated) + routed experts (softmax gating).
// Gemma4-specific routing: attn_out → rms_norm → scale by 1/sqrt(n_embd) → mul ffn_gate_inp_s → ffn_gate_inp → softmax → top-k
static ggml_tensor * build_gemma4_moe_block(ggml_context * ctx, ggml_tensor * attn_out,
                                              ggml_tensor * cur_normed,
                                              const Gemma4Weights & w,
                                              const Gemma4Layer & L,
                                              int n_tokens) {
    const int n_expert = w.n_expert;
    const int n_used   = w.n_expert_used;
    const int n_embd   = w.n_embd;

    // ---- Shared expert (GELU-gated MLP) ----
    ggml_tensor * sh_gate = ggml_mul_mat(ctx, L.ffn_gate, cur_normed);
    ggml_tensor * sh_up   = ggml_mul_mat(ctx, L.ffn_up,   cur_normed);
    // Use fused geglu_split to match ref build_ffn(LLM_FFN_GELU, LLM_FFN_PAR)
    ggml_tensor * sh_gu   = ggml_geglu_split(ctx, sh_gate, sh_up);
    ggml_tensor * shared  = ggml_mul_mat(ctx, L.ffn_down, sh_gu);

    if (L.ffn_post_norm_1) {
        shared = gemma4_rms_norm_mul(ctx, shared, L.ffn_post_norm_1, w.norm_eps);
    }

    // ---- Routed experts ----
    if (!L.ffn_gate_inp || n_expert == 0) {
        // No MoE on this layer, shared-only
        return shared;
    }

    // Pre-norm for routed input
    ggml_tensor * cur_moe = cur_normed;
    if (L.ffn_pre_norm_2) {
        cur_moe = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_pre_norm_2, w.norm_eps);
    }

    // Router: rms_norm(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp_s → ffn_gate_inp → softmax
    ggml_tensor * router_in = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, attn_out), w.norm_eps);
    router_in = ggml_scale(ctx, router_in, 1.0f / std::sqrt((float)n_embd));
    if (L.ffn_gate_inp_s) {
        router_in = ggml_mul(ctx, router_in, L.ffn_gate_inp_s);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, router_in); // [n_expert, n_tokens]

    // Softmax over experts
    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    // Top-k selection — use argsort_top_k to match reference (argsort DESC + view)
    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, n_used);

    // Gather weights at selected indices
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs_3d, selected);
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);

    // Fix 1: renormalize top-k routed weights by their sum (ref llama-graph.cpp norm_w path)
    {
        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, weights_sum); // [n_used, n_tokens]
    }

    // Routed expert forward via mul_mat_id with fused gate+up
    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur_moe, n_embd, 1, n_tokens);
    ggml_tensor * gate_up_e = ggml_mul_mat_id(ctx, L.ffn_gate_up_exps, cur_3d, selected);
    // gate_up_e is [n_ff_exp*2, n_used, n_tokens] — split and GELU-gate
    const int n_ff_exp = w.n_ff_exp;
    ggml_tensor * gate_e = ggml_view_3d(ctx, gate_up_e,
        n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
        gate_up_e->nb[1], gate_up_e->nb[2], 0);
    ggml_tensor * up_e = ggml_view_3d(ctx, gate_up_e,
        n_ff_exp, gate_up_e->ne[1], gate_up_e->ne[2],
        gate_up_e->nb[1], gate_up_e->nb[2],
        (size_t)n_ff_exp * ggml_element_size(gate_up_e));
    gate_e = ggml_cont(ctx, gate_e);
    up_e = ggml_cont(ctx, up_e);
    // Use fused geglu_split (matches ref ggml_geglu_split → same table-GELU kernel)
    ggml_tensor * gu = ggml_geglu_split(ctx, gate_e, up_e);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);

    // Apply per-expert down scale (ref llama-model.cpp:1364-1365 generic pass loads
    // ffn_down_exps_s even though diffusion-gemma load_arch_tensors doesn't explicitly
    // request it; ref llama-graph.cpp:1758-1764 applies it before the weighted sum).
    if (L.ffn_down_exps_s) {
        // down_exps_s: [n_expert] → gather selected rows → [1, n_used, n_tokens]
        ggml_tensor * s = ggml_reshape_3d(ctx, L.ffn_down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx, s, selected);   // [1, n_used, n_tokens]
        experts = ggml_mul(ctx, experts, s);   // [n_embd, n_used, n_tokens]
    }

    // Weighted sum of expert outputs
    ggml_tensor * w_view = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    experts = ggml_mul(ctx, experts, w_view);

    ggml_tensor * routed = nullptr;
    for (int i = 0; i < n_used; ++i) {
        ggml_tensor * slice = ggml_view_2d(ctx, experts,
            n_embd, n_tokens,
            experts->nb[2],
            (size_t)i * experts->nb[1]);
        routed = (i == 0) ? slice : ggml_add(ctx, routed, slice);
    }

    if (L.ffn_post_norm_2) {
        routed = gemma4_rms_norm_mul(ctx, routed, L.ffn_post_norm_2, w.norm_eps);
    }

    return ggml_add(ctx, shared, routed);
}

// Attention block for a single layer (handles both full and SWA).
// When no_cache=true the K/V tensors are used directly in F32 without a
// cache round-trip (matches the reference unified forward which never writes
// to the F16 KV cache during denoise).
static ggml_tensor * build_gemma4_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const Gemma4Weights & w,
    const Gemma4Layer & L,
    Gemma4Cache & cache,
    int il,
    ggml_tensor * cur,
    ggml_tensor * positions,
    ggml_tensor * attn_mask_full,
    ggml_tensor * attn_mask_swa,
    int kv_start,
    int n_tokens,
    ggml_tensor * kv_idx_full = nullptr,   // [n_tokens] I32 absolute rows (graph input)
    ggml_tensor * kv_idx_swa  = nullptr,   // [n_tokens] I32 ring rows pos%swa_size (graph input)
    bool no_cache = false,                  // bypass F16 cache; use K/V in F32 directly
    ggml_tensor * attn_mask_full_f32 = nullptr,  // F32 mask for no-cache standard attention
    ggml_tensor * attn_mask_swa_f32  = nullptr)  // F32 SWA mask for no-cache standard attention
{
    const int head_dim   = gemma4_head_dim(w, il);
    const int n_head     = w.n_head;
    const int n_head_kv  = gemma4_n_head_kv(w, il);
    const int q_dim      = n_head * head_dim;
    const bool is_swa    = gemma4_is_swa_layer(w, il);
    const bool has_kv    = gemma4_has_kv(w, il);

    // Q projection (all layers have Q)
    ggml_tensor * Qcur = ggml_mul_mat(ctx, L.wq, cur);
    Qcur = ggml_reshape_3d(ctx, Qcur, head_dim, n_head, n_tokens);

    // Q RMSNorm per head
    if (L.q_norm) {
        Qcur = gemma4_rms_norm_mul(ctx, Qcur, L.q_norm, w.norm_eps);
    }

    // RoPE for Q
    const float rope_base = is_swa ? w.rope_freq_base_swa : w.rope_freq_base_full;
    ggml_tensor * freq_factors = is_swa ? nullptr : (L.rope_freqs ? L.rope_freqs : w.rope_freqs_global);
    Qcur = ggml_rope_ext(ctx, Qcur, positions, freq_factors,
                          head_dim, GGML_ROPE_TYPE_NEOX,
                          0, rope_base, 1.0f,
                          0.0f, 1.0f, 32.0f, 1.0f);

    // Determine which cache layer to use
    int cache_il = cache.kv_source[il];
    ggml_tensor * cache_k = cache.k[cache_il];
    ggml_tensor * cache_v = cache.v[cache_il];
    const int cache_len = (int)cache_k->ne[1];  // max_ctx for full, swa_size for SWA

    // K/V tensors for FA: populated by cache path or no_cache direct path.
    ggml_tensor * Kfa = nullptr;
    ggml_tensor * Vfa = nullptr;
    int kv_len = 0;

    if (no_cache && has_kv) {
        // ── No-cache path (denoise unified forward) ────────────────────────
        // Use standard (non-flash) attention to match the reference exactness.
        // The reference diffusion-gemma-eval runs with flash_attn DISABLED by
        // default (ref diffusion-gemma-eval.cpp:93-95), which is what the golden
        // logits were generated with. Standard attention gives cosine=1.000 vs
        // the golden; FA gives cosine~0.967 due to F16 intermediates.
        //
        // Standard attention path (mirrors reference llama-graph.cpp:2106-2162):
        //   Q: [head_dim, n_head, n_tokens] → permute → [head_dim, n_tokens, n_head]
        //   K: [head_dim, n_head_kv, n_tokens] → permute → [head_dim, n_tokens, n_head_kv]
        //   V: [head_dim, n_head_kv, n_tokens] → permute → [head_dim, n_tokens, n_head_kv]
        //   kq = mul_mat(K, Q)   [n_tokens, n_tokens, n_head, 1]  (GQA broadcast)
        //   kq = soft_max_ext(kq, mask, kq_scale)
        //   kqv = mul_mat(V, kq)  [head_dim, n_tokens, n_head, 1]
        //   out = permute(kqv, 0,2,1,3) → [head_dim, n_head, n_tokens, 1]
        //   out = reshape_2d(out, q_dim, n_tokens)
        //   return mul_mat(wo, out)

        ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
        ggml_tensor * Vcur = L.wv ? ggml_mul_mat(ctx, L.wv, cur) : Kcur;

        Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

        if (L.k_norm) {
            Kcur = gemma4_rms_norm_mul(ctx, Kcur, L.k_norm, w.norm_eps);
        }
        Vcur = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, Vcur), w.norm_eps);

        Kcur = ggml_rope_ext(ctx, Kcur, positions, freq_factors,
                              head_dim, GGML_ROPE_TYPE_NEOX,
                              0, rope_base, 1.0f,
                              0.0f, 1.0f, 32.0f, 1.0f);

        // Permute all to [head_dim, n_tokens, n_head/n_head_kv]
        ggml_tensor * Qp = ggml_permute(ctx, Qcur, 0, 2, 1, 3);   // [head_dim, n_tokens, n_head]
        ggml_tensor * Kp = ggml_permute(ctx, Kcur, 0, 2, 1, 3);   // [head_dim, n_tokens, n_head_kv]
        ggml_tensor * Vp = ggml_permute(ctx, Vcur, 0, 2, 1, 3);   // [head_dim, n_tokens, n_head_kv]

        // kq = K^T @ Q: [n_tokens_k, n_tokens_q, n_head, 1]
        // ggml_mul_mat(K, Q) = K^T @ Q  (ne[0] of K = head_dim matches ne[0] of Q = head_dim)
        // GQA: n_head_kv broadcast to n_head
        ggml_tensor * kq = ggml_mul_mat(ctx, Kp, Qp);              // [n_tokens, n_tokens, n_head]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);  // ref: default to F32 accumulation

        // Mask: for standard attention ggml_soft_max_ext requires mask->ne[0] == kq->ne[0].
        // kq->ne[0] = n_tokens (unpadded). The padded F16 masks have ne[0]=kv_len_padded.
        // Use the caller-supplied F32 masks (unpadded) if available; otherwise pass nullptr.
        // (nullptr mask = no masking, which is correct for full bidirectional diffusion attention.)
        const float kq_scale_nc = 1.0f;
        ggml_tensor * std_mask = nullptr;
        if (is_swa && attn_mask_swa_f32)  std_mask = attn_mask_swa_f32;
        else if (!is_swa && attn_mask_full_f32) std_mask = attn_mask_full_f32;
        ggml_tensor * kq_sm = ggml_soft_max_ext(ctx, kq, std_mask, kq_scale_nc, 0.0f);

        // kqv: matches reference non-FA path (llama-graph.cpp build_attn_mha lines 2144-2162).
        // After permute(0,2,1,3), Vp = [head_dim, n_tokens, n_head_kv, 1].
        // v_trans = (Vp->nb[1] > Vp->nb[2]) = false since permute makes nb[1] < nb[2].
        // Reference: "if (!v_trans) v = ggml_cont(transpose(v))"
        // After cont(transpose(Vp)): shape = [n_tokens, head_dim, n_head_kv, 1].
        ggml_tensor * Vt = ggml_cont(ctx, ggml_transpose(ctx, Vp));  // [n_tokens, head_dim, n_head_kv]

        // mul_mat(Vt, kq_sm) contracts Vt.ne[0]=n_tokens with kq_sm.ne[0]=n_tokens.
        // Result: [head_dim, n_tokens_q, n_head, 1]
        ggml_tensor * kqv = ggml_mul_mat(ctx, Vt, kq_sm);

        // Permute and reshape: reference lines 2159-2162
        // kqv: [head_dim, n_tokens_q, n_head, 1]
        // permute(0,2,1,3): [head_dim, n_head, n_tokens_q, 1]
        ggml_tensor * kqv_out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        ggml_tensor * out = ggml_cont_2d(ctx, kqv_out, q_dim, n_tokens); // [q_dim, n_tokens]
        return ggml_mul_mat(ctx, L.wo, out);                              // [n_embd, n_tokens]
    } else if (has_kv) {
        // ── Cache path (autoregressive decode) ────────────────────────────
        // K/V projection + norm + RoPE + write to F16 cache
        ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
        ggml_tensor * Vcur = L.wv ? ggml_mul_mat(ctx, L.wv, cur) : Kcur;

        Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

        if (L.k_norm) {
            Kcur = gemma4_rms_norm_mul(ctx, Kcur, L.k_norm, w.norm_eps);
        }
        // V also gets RMSNorm (gemma4 specific)
        Vcur = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, Vcur), w.norm_eps);

        Kcur = ggml_rope_ext(ctx, Kcur, positions, freq_factors,
                              head_dim, GGML_ROPE_TYPE_NEOX,
                              0, rope_base, 1.0f,
                              0.0f, 1.0f, 32.0f, 1.0f);

        // Write K/V to cache (ring-buffer position for SWA layers)
        ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
        ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

        ggml_tensor * kvi = is_swa ? kv_idx_swa : kv_idx_full;
        if (kvi) {
            // CUDA-graph-stable append: dst is the whole cache tensor (stable
            // pointer), the row index is a graph INPUT (data changes per step,
            // pointer doesn't). A write_pos-offset view changes node properties
            // every step, which resets the ggml-cuda CUDA-graph warmup and
            // forfeits replay. For SWA layers the caller fills the index with
            // (pos % swa_size), which also handles ring wrap-around mid-chunk
            // correctly (the offset-view path wrote a contiguous block).
            ggml_tensor * Krows = ggml_cont(ctx, Kcur_T);
            ggml_tensor * Vrows = ggml_cont(ctx, Vcur_T);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_k, Krows, kvi));
            ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache_v, Vrows, kvi));
        } else {
            const int write_pos = is_swa ? (kv_start % cache_len) : kv_start;
            ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
                head_dim, n_tokens, n_head_kv,
                cache_k->nb[1], cache_k->nb[2],
                cache_k->nb[1] * (size_t)write_pos);
            ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
                head_dim, n_tokens, n_head_kv,
                cache_v->nb[1], cache_v->nb[2],
                cache_v->nb[1] * (size_t)write_pos);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));
        }

        // Read back from F16 cache for flash attention
        const int fa_window_l = cache.fa_window;
        const int full_win_start_l = (!is_swa && fa_window_l > 0 && kv_start > fa_window_l)
                                         ? (kv_start - fa_window_l) : 0;
        const int kv_len_raw = is_swa ? std::min(kv_start + n_tokens, cache_len)
                                      : (kv_start + n_tokens - full_win_start_l);
        kv_len = std::min((kv_len_raw + 255) & ~255, cache_len);

        const size_t cache_offset = is_swa ? 0 : (cache_k->nb[1] * (size_t)full_win_start_l);
        Kfa = ggml_view_3d(ctx, cache_k,
            head_dim, kv_len, n_head_kv,
            cache_k->nb[1], cache_k->nb[2], cache_offset);
        Vfa = ggml_view_3d(ctx, cache_v,
            head_dim, kv_len, n_head_kv,
            cache_v->nb[1], cache_v->nb[2], cache_offset);
    } else {
        // ── KV-sharing: cache already written by source layer ─────────────
        const int fa_window_l = cache.fa_window;
        const int full_win_start_l = (!is_swa && fa_window_l > 0 && kv_start > fa_window_l)
                                         ? (kv_start - fa_window_l) : 0;
        const int kv_len_raw = is_swa ? std::min(kv_start + n_tokens, cache_len)
                                      : (kv_start + n_tokens - full_win_start_l);
        kv_len = std::min((kv_len_raw + 255) & ~255, cache_len);

        const size_t cache_offset = is_swa ? 0 : (cache_k->nb[1] * (size_t)full_win_start_l);
        Kfa = ggml_view_3d(ctx, cache_k,
            head_dim, kv_len, n_head_kv,
            cache_k->nb[1], cache_k->nb[2], cache_offset);
        Vfa = ggml_view_3d(ctx, cache_v,
            head_dim, kv_len, n_head_kv,
            cache_v->nb[1], cache_v->nb[2], cache_offset);
    }

    ggml_tensor * Qfa = ggml_cont(ctx, ggml_permute(ctx, Qcur, 0, 2, 1, 3));

    // Gemma4 uses self.scaling = 1.0 (no QK scaling) because Q/K are already
    // RMS-normed per-head. Standard 1/sqrt(head_dim) is NOT used here.
    const float kq_scale = 1.0f;

    // For the windowed cache path, the mask may need to be offset to match the
    // KV window start. Re-derive full_win_start here from cache.fa_window.
    const int full_win_start = (!no_cache && !is_swa && cache.fa_window > 0 && kv_start > cache.fa_window)
                                   ? (kv_start - cache.fa_window) : 0;
    ggml_tensor * use_mask;
    if (is_swa) {
        use_mask = attn_mask_swa;
    } else if (full_win_start > 0) {
        // View the mask starting at full_win_start column
        use_mask = ggml_view_4d(ctx, attn_mask_full,
            kv_len, n_tokens, 1, 1,
            attn_mask_full->nb[1], attn_mask_full->nb[2], attn_mask_full->nb[3],
            (size_t)full_win_start * ggml_element_size(attn_mask_full));
    } else {
        use_mask = attn_mask_full;
    }
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, use_mask,
                                              kq_scale, 0.0f, 0.0f);
    // Match reference: set F32 precision for the FA accumulator to avoid
    // compounding F16-accumulation errors across 30 layers.
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    // Reshape to [q_dim, n_tokens] and output projection
    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);
    return ggml_mul_mat(ctx, L.wo, attn);
}

// Build one layer of the gemma4 graph.
static ggml_tensor * build_gemma4_layer(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const Gemma4Weights & w,
    Gemma4Cache & cache,
    int il,
    ggml_tensor * inp,
    ggml_tensor * positions,
    ggml_tensor * attn_mask_full,
    ggml_tensor * attn_mask_swa,
    ggml_tensor * per_layer_input,  // [n_embd_per_layer, n_tokens] or nullptr
    int kv_start,
    int n_tokens,
    int capture_idx = -1,  // >=0: write to target_feat at this capture slot
    ggml_tensor * kv_idx_full = nullptr,
    ggml_tensor * kv_idx_swa  = nullptr,
    bool no_cache = false,  // bypass F16 cache in attention block
    ggml_tensor * attn_mask_full_f32 = nullptr,  // F32 [n_tokens,n_tokens] mask for no-cache path
    ggml_tensor * attn_mask_swa_f32  = nullptr)  // F32 [n_tokens,n_tokens] SWA mask for no-cache path
{
    const Gemma4Layer & L = w.layers[il];
    ggml_tensor * inp_f32 = graph_tensor_f32(ctx, inp);

    // Pre-attn norm
    ggml_tensor * cur = gemma4_rms_norm_mul(ctx, inp_f32, L.attn_norm, w.norm_eps);

    // Attention
    cur = build_gemma4_attn_block(ctx, gf, w, L, cache, il, cur,
                                    positions, attn_mask_full, attn_mask_swa,
                                    kv_start, n_tokens, kv_idx_full, kv_idx_swa,
                                    no_cache, attn_mask_full_f32, attn_mask_swa_f32);

    // Post-attn norm
    if (L.attn_post_norm) {
        cur = gemma4_rms_norm_mul(ctx, cur, L.attn_post_norm, w.norm_eps);
    }

    // Residual
    ggml_tensor * attn_out = ggml_add(ctx, cur, inp_f32);

    // FFN
    const bool is_moe = (L.ffn_gate_inp != nullptr && il >= w.n_layer_dense_lead);
    if (is_moe) {
        // MoE: shared expert + routed experts
        ggml_tensor * cur_normed = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_norm, w.norm_eps);
        cur = build_gemma4_moe_block(ctx, attn_out, cur_normed, w, L, n_tokens);
    } else {
        // Dense FFN
        cur = gemma4_rms_norm_mul(ctx, attn_out, L.ffn_norm, w.norm_eps);
        cur = build_gemma4_dense_ffn(ctx, cur, L);
    }

    // FFN post-norm (applies to both dense and MoE paths)
    if (L.ffn_post_norm) {
        cur = gemma4_rms_norm_mul(ctx, cur, L.ffn_post_norm, w.norm_eps);
    }

    // Residual
    cur = ggml_add(ctx, cur, attn_out);

    // Per-layer embedding injection
    if (per_layer_input && L.per_layer_inp_gate && L.per_layer_proj) {
        ggml_tensor * pe_in = cur;
        // Gate: cur -> [n_embd_per_layer, n_tokens]
        ggml_tensor * gate = ggml_mul_mat(ctx, L.per_layer_inp_gate, cur);
        gate = ggml_gelu(ctx, gate);
        // Element-wise mul with per-layer input
        gate = ggml_mul(ctx, gate, per_layer_input);
        // Project back: [n_embd_per_layer, n_tokens] -> [n_embd, n_tokens]
        ggml_tensor * proj = ggml_mul_mat(ctx, L.per_layer_proj, gate);
        if (L.per_layer_post_norm) {
            proj = gemma4_rms_norm_mul(ctx, proj, L.per_layer_post_norm, w.norm_eps);
        }
        cur = ggml_add(ctx, pe_in, proj);
    }

    // Output scale
    if (L.out_scale) {
        cur = ggml_mul(ctx, cur, L.out_scale);
    }

    // Feature capture for DFlash spec-decode
    if (capture_idx >= 0 && cache.target_feat) {
        const int hidden = w.n_embd;
        const size_t elt = ggml_element_size(cache.target_feat);
        const size_t col_stride = cache.target_feat->nb[1];
        const int cap = cache.target_feat_cap;
        const int slot_start = kv_start % cap;
        const int pre_n = std::min(n_tokens, cap - slot_start);
        const int post_n = n_tokens - pre_n;

        ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

        // First slice: [slot_start..slot_start+pre_n) in the ring.
        {
            const size_t offset =
                (size_t)slot_start * col_stride +
                (size_t)capture_idx * hidden * elt;
            ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                hidden, pre_n, col_stride, offset);
            ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                hidden, pre_n, cur_2d->nb[1], 0);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
        }

        // Second slice: wrap-around at [0..post_n) if needed.
        if (post_n > 0) {
            const size_t offset =
                (size_t)capture_idx * hidden * elt;
            ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                hidden, post_n, col_stride, offset);
            ggml_tensor * src = ggml_view_2d(ctx, cur_2d,
                hidden, post_n, cur_2d->nb[1],
                (size_t)pre_n * cur_2d->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
        }
    }

    return cur;
}

// Helper: get a 2D slice from a 3D tensor along ne[2] (same as llama.cpp ggml_view_2d_slice).
static ggml_tensor * gemma4_view_2d_slice(ggml_context * ctx, ggml_tensor * x, int idx) {
    return ggml_view_2d(ctx, x, x->ne[0], x->ne[1],
                        ggml_row_size(x->type, x->ne[0]),
                        (size_t)idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

static ggml_tensor * build_gemma4_per_layer_input(
    ggml_context * ctx,
    const Gemma4Weights & w,
    ggml_tensor * embed,
    ggml_tensor * token_ids,
    int n_tokens,
    int layer_idx) {
    if (!token_ids || !w.per_layer_tok_embd || !w.per_layer_model_proj ||
        !w.per_layer_proj_norm || w.n_embd_per_layer <= 0) {
        return nullptr;
    }
    const int D = w.n_embd_per_layer;
    const size_t elt_tok = ggml_element_size(w.per_layer_tok_embd);
    const size_t elt_norm = ggml_element_size(w.per_layer_proj_norm);

    ggml_tensor * tok_embd_layer = ggml_view_2d(
        ctx, w.per_layer_tok_embd, D, w.n_vocab,
        w.per_layer_tok_embd->nb[1], (size_t)layer_idx * D * elt_tok);
    ggml_tensor * inp_pl = ggml_get_rows(ctx, tok_embd_layer, token_ids);
    inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));

    ggml_tensor * proj_w_layer = ggml_view_2d(
        ctx, w.per_layer_model_proj, w.n_embd, D,
        w.per_layer_model_proj->nb[1],
        (size_t)layer_idx * D * w.per_layer_model_proj->nb[1]);
    ggml_tensor * proj = ggml_mul_mat(ctx, proj_w_layer, embed);
    proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
    proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
    ggml_tensor * norm_w = ggml_view_1d(
        ctx, w.per_layer_proj_norm, D, (size_t)layer_idx * D * elt_norm);
    proj = ggml_mul(ctx, proj, norm_w);

    ggml_tensor * per_layer = ggml_add(ctx, proj, inp_pl);
    return ggml_scale(ctx, per_layer, 1.0f / std::sqrt(2.0f));
}

void gemma4_layer_step_graph_free(Gemma4LayerStepGraph & sg) {
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    sg.gf = nullptr;
    sg.positions = nullptr;
    sg.token_ids = nullptr;
    sg.attn_mask_full = nullptr;
    sg.attn_mask_swa = nullptr;
}

void gemma4_layer_step_graph_destroy(Gemma4LayerStepGraph & sg) {
    if (sg.alloc) {
        ggml_gallocr_free(sg.alloc);
        sg.alloc = nullptr;
    }
    gemma4_layer_step_graph_free(sg);
}

bool build_gemma4_layer_step(
    Gemma4LayerStepGraph & sg,
    const Gemma4Weights &  w,
    Gemma4Cache &          cache,
    ggml_backend_t         backend,
    int                    layer_idx,
    ggml_tensor *          act_in,
    ggml_tensor *          orig_embed,
    ggml_tensor *          act_out,
    int                    chunk_start,
    int                    n_tokens,
    int                    kv_start) {
    gemma4_layer_step_graph_free(sg);
    if (layer_idx < 0 || layer_idx >= w.n_layer) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;
    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    ggml_tensor * inp = ggml_view_2d(
        sg.ctx, act_in, w.n_embd, n_tokens,
        act_in->nb[1], (size_t)chunk_start * act_in->nb[1]);
    ggml_set_input(inp);

    ggml_tensor * embed = ggml_view_2d(
        sg.ctx, orig_embed, w.n_embd, n_tokens,
        orig_embed->nb[1], (size_t)chunk_start * orig_embed->nb[1]);
    ggml_set_input(embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.positions);

    sg.token_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.token_ids);

    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len_padded = (kv_len_raw + 255) & ~255;
    sg.attn_mask_full = ggml_new_tensor_4d(
        sg.ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask_full);
    ggml_tensor * mask_full_f16 = ggml_cast(sg.ctx, sg.attn_mask_full, GGML_TYPE_F16);

    const int swa_size = cache.swa_size;
    if (swa_size <= 0) return false;
    const int swa_len_raw = std::min(kv_start + n_tokens, swa_size);
    const int swa_len_padded = (swa_len_raw + 255) & ~255;
    sg.attn_mask_swa = ggml_new_tensor_4d(
        sg.ctx, GGML_TYPE_F32, swa_len_padded, n_tokens, 1, 1);
    ggml_set_input(sg.attn_mask_swa);
    ggml_tensor * mask_swa_f16 = ggml_cast(sg.ctx, sg.attn_mask_swa, GGML_TYPE_F16);

    ggml_tensor * pl_input = build_gemma4_per_layer_input(
        sg.ctx, w, embed, sg.token_ids, n_tokens, layer_idx);
    ggml_tensor * layer_out = build_gemma4_layer(
        sg.ctx, sg.gf, w, cache, layer_idx, inp, sg.positions,
        mask_full_f16, mask_swa_f16, pl_input, kv_start, n_tokens);
    if (!layer_out) return false;

    ggml_tensor * out_view = ggml_view_2d(
        sg.ctx, act_out, w.n_embd, n_tokens,
        act_out->nb[1], (size_t)chunk_start * act_out->nb[1]);
    ggml_build_forward_expand(sg.gf, ggml_cpy(sg.ctx, layer_out, out_view));

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

bool compute_gemma4_split_projection(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    ggml_tensor *           act,
    int                     token_offset,
    int                     n_tokens,
    std::vector<int32_t> *  out_argmax,
    std::vector<float> *    out_logits) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * act_view = ggml_view_2d(
        ctx, act, w.n_embd, n_tokens, act->nb[1],
        (size_t)token_offset * act->nb[1]);
    ggml_tensor * cur = gemma4_rms_norm_mul(ctx, act_view, w.out_norm, w.norm_eps);
    cur = ggml_mul_mat(ctx, w.output, cur);
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }
    ggml_tensor * logits = cur;
    ggml_tensor * argmax = nullptr;
    if (out_logits) {
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }
    if (out_argmax) {
        argmax = ggml_argmax(ctx, logits);
        ggml_set_output(argmax);
        ggml_build_forward_expand(gf, argmax);
    }

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc, gf)) {
        if (alloc) ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }
    if (out_argmax) {
        out_argmax->resize((size_t)n_tokens);
        ggml_backend_tensor_get(argmax, out_argmax->data(), 0,
                                sizeof(int32_t) * (size_t)n_tokens);
    }
    if (out_logits) {
        out_logits->resize((size_t)w.n_vocab * (size_t)n_tokens);
        ggml_backend_tensor_get(logits, out_logits->data(), 0,
                                sizeof(float) * (size_t)w.n_vocab * (size_t)n_tokens);
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

bool compute_gemma4_split_argmax(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    ggml_tensor *           act,
    int                     token_offset,
    int                     n_tokens,
    std::vector<int32_t> &  out_argmax) {
    return compute_gemma4_split_projection(
        backend, w, act, token_offset, n_tokens, &out_argmax, nullptr);
}

bool gemma4_step(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<float> &    out_logits,
    const KvFlashPager *    kvflash)
{
    if (kvflash && cache.fa_window > 0) {
        std::fprintf(stderr, "gemma4_step: kvflash and fa_window are mutually "
                             "exclusive full-attention policies\n");
        return false;
    }
    // Allocate graph context. Persistent thread_local arena: rebuilt graphs
    // land at identical addresses every step, so the ggml-cuda CUDA-graph
    // cache (keyed on nodes[0], memcmps node properties) can replay the
    // captured graph instead of re-launching every kernel per token.
    const size_t arena_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    static thread_local std::vector<uint8_t> g_arena;
    if (g_arena.size() < arena_size) g_arena.resize(arena_size);
    ggml_init_params ip{};
    ip.mem_size = arena_size;
    ip.mem_buffer = g_arena.data();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Input tensors
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    // K/V append row indices (set_rows path; data-only per step -> stable
    // node properties -> CUDA-graph replay). DFLASH_GEMMA4_NO_KVPAD=1 restores
    // the legacy offset-view cpy append.
    static const bool g_no_kvpad = (std::getenv("DFLASH_GEMMA4_NO_KVPAD") != nullptr);
    ggml_tensor * kvi_full = nullptr, * kvi_swa = nullptr;
    if (!g_no_kvpad) {
        kvi_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(kvi_full);
        kvi_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(kvi_swa);
    }

    // Token IDs input (for per-layer embedding lookup)
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }

    // Attention masks (full + SWA)
    // Full-attention mask: covers all positions [0, kv_start+n_tokens),
    // clamped to the full-layer tensor capacity (pool-sized under kvflash) —
    // must agree with the FA span clamp in build_gemma4_attn_block.
    int full_cap = cache.max_ctx;
    for (int il = 0; il < (int)cache.k.size(); ++il) {
        if (cache.k[(size_t)il] && !gemma4_is_swa_layer(w, il)) {
            full_cap = (int)cache.k[(size_t)il]->ne[1];
            break;
        }
    }
    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len_padded = std::min((kv_len_raw + 255) & ~255, full_cap);
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);

    // SWA mask: covers the ring buffer [0, swa_size) with ring-buffer indexing
    const int swa_size = cache.swa_size;
    const int swa_len_raw = std::min(kv_start + n_tokens, swa_size);
    const int swa_len_padded = (swa_len_raw + 255) & ~255;
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, swa_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embedding computation (reference: gemma4-iswa.cpp build_inp_per_layer + project_per_layer_inputs)
    ggml_tensor * per_layer_all = nullptr;  // final shape: [n_embd_per_layer, n_tokens, n_layer]
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;

        // 1. Token per-layer embedding lookup + scale
        //    get_rows(per_layer_tok_embd[D*L, n_vocab], tok_ids) → [D*L, n_tokens]
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, n_tokens);  // [D, L, n_tokens]
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));

        // 2. Project main embedding through per_layer_model_proj
        //    mul_mat(per_layer_model_proj[n_embd, D*L], ie[n_embd, n_tokens]) → [D*L, n_tokens]
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, n_tokens);  // [D, L, n_tokens]

        // 3. RMS norm on projection (normalizes over ne[0]=D for each (layer, token))
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
        // Reshape norm weight from [D*L] to [D, L] for broadcast mul over n_tokens
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);

        // 4. Add token embedding + projection, scale by 1/sqrt(2)
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));

        // 5. Permute to [D, n_tokens, L] for easy per-layer slicing
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Build the graph
    ggml_tensor * cur = ie;  // [n_embd, n_tokens] already scaled by sqrt(n_embd) in caller

    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) {
            // Slice [n_embd_per_layer, n_tokens] for this layer
            pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);
        }
        // Determine capture index for this layer (-1 if not a capture layer)
        int cap_idx = -1;
        if (cache.target_feat) {
            for (int k = 0; k < cache.n_capture_layers; k++) {
                if (cache.capture_layer_ids[k] == il) { cap_idx = k; break; }
            }
        }
        cur = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                   mk_full_f16, mk_swa_f16, pl_input,
                                   kv_start, n_tokens, cap_idx,
                                   kvi_full, kvi_swa);
    }

    // Final norm
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);

    // Extract last token only for logits
    if (n_tokens > 1) {
        cur = ggml_view_2d(ctx, cur, w.n_embd, 1,
                            cur->nb[1],
                            (size_t)(n_tokens - 1) * cur->nb[1]);
    }

    // lm_head
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, 1]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "gemma4_step: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));
    if (!kvi_full && kvflash) {
        std::fprintf(stderr, "gemma4_step: kvflash requires the set_rows path "
                             "(DFLASH_GEMMA4_NO_KVPAD is incompatible)\n");
        ggml_free(ctx);
        return false;
    }
    std::vector<float> kvf_mfull;  // slot-space full mask (kvflash)
    if (kvi_full) {
        // Full layers append at the absolute position (or the kvflash pool
        // slot); SWA layers at the ring slot. Per-token modular indices also
        // land chunks that cross the ring wrap boundary correctly (the
        // offset-view path wrote one contiguous block).
        if (kvflash) {
            // Rows + slot-space full mask in one pass (shared helper; the
            // mask is uploaded below where the legacy path builds its own).
            std::vector<int32_t> rows;
            if (!kvflash_fill_rows_and_masks(*kvflash, kv_start, n_tokens,
                                             kv_len_padded, /*swa_window=*/0,
                                             rows, &kvf_mfull, nullptr)) {
                ggml_free(ctx);
                return false;
            }
            ggml_backend_tensor_set(kvi_full, rows.data(), 0, ggml_nbytes(kvi_full));
        } else {
            ggml_backend_tensor_set(kvi_full, pos.data(), 0, ggml_nbytes(kvi_full));
        }
        GGML_ASSERT(swa_size > 0);
        std::vector<int32_t> ring((size_t)n_tokens);
        for (int i = 0; i < n_tokens; ++i) ring[i] = (kv_start + i) % swa_size;
        ggml_backend_tensor_set(kvi_swa, ring.data(), 0, ggml_nbytes(kvi_swa));
    }

    // Set token IDs for per-layer embedding
    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    // Causal mask (full attention) — padded positions are masked with -inf.
    // kvflash: SLOT-space mask already built alongside the append rows.
    std::vector<float> mfull;
    if (kvflash) {
        mfull = std::move(kvf_mfull);
    } else {
        mfull.assign((size_t)kv_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len_raw; ++k) {
                mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
    }
    ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));

    // SWA ring-buffer mask — maps cache indices to absolute positions
    const int W = w.sliding_window;
    std::vector<float> mswa((size_t)swa_len_padded * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        const int win_lo = std::max(0, abs_q - W + 1);
        // The ring buffer stores the most recent min(abs_q+1, swa_size) entries.
        // Cache slot j holds absolute position: depends on how many tokens written.
        const int total_written = abs_q + 1;  // positions [0..abs_q] written so far
        GGML_ASSERT(swa_size > 0 && "SWA branch entered with uninitialised cache.swa_size");
        for (int abs_k = win_lo; abs_k <= abs_q; ++abs_k) {
            // Map absolute position to ring-buffer slot
            const int slot = abs_k % swa_size;
            if (slot < swa_len_raw) {
                mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
            }
        }
        (void)total_written;
    }
    ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_step: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read logits
    out_logits.resize((size_t)w.n_vocab);
    ggml_backend_tensor_get(cur, out_logits.data(), 0,
                             out_logits.size() * sizeof(float));

    cache.cur_pos = kv_len_raw;
    ggml_free(ctx);
    return true;
}

// ── gemma4_verify_batch ─────────────────────────────────────────────────
// Like gemma4_step but returns argmax for ALL token positions (not just last).

bool gemma4_verify_batch(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     kv_start,
    std::vector<int32_t> &  out_argmax,
    const KvFlashPager *    kvflash)
{
    if (kvflash && cache.fa_window > 0) {
        std::fprintf(stderr, "gemma4_verify_batch: kvflash and fa_window are "
                             "mutually exclusive\n");
        return false;
    }
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

    // Input tensors
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    // Token IDs for per-layer embedding
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }

    // kvflash: full-layer writes must go through set_rows to land in pool
    // slots; SWA ring rows ride the same mechanism (pos % swa_size).
    ggml_tensor * kvi_full = nullptr, * kvi_swa = nullptr;
    if (kvflash) {
        kvi_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(kvi_full);
        kvi_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(kvi_swa);
    }

    // Attention masks (padded; full width clamps to the full-layer tensor
    // capacity, which is pool-sized under kvflash — must agree with the FA
    // span clamp in build_gemma4_attn_block)
    int full_cap = cache.max_ctx;
    for (int il = 0; il < (int)cache.k.size(); ++il) {
        if (cache.k[(size_t)il] && !gemma4_is_swa_layer(w, il)) {
            full_cap = (int)cache.k[(size_t)il]->ne[1];
            break;
        }
    }
    const int kv_len_raw = kv_start + n_tokens;
    const int kv_len_padded = std::min((kv_len_raw + 255) & ~255, full_cap);
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);

    // SWA mask: ring-buffer sized
    const int swa_size = cache.swa_size;
    const int swa_len_raw = std::min(kv_start + n_tokens, swa_size);
    const int swa_len_padded = (swa_len_raw + 255) & ~255;
    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, swa_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embedding computation (same as gemma4_step)
    ggml_tensor * per_layer_all = nullptr;
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, n_tokens);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, n_tokens);
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Build graph (all layers)
    ggml_tensor * cur = ie;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) {
            pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);
        }
        int cap_idx = -1;
        if (cache.target_feat) {
            for (int k = 0; k < cache.n_capture_layers; k++) {
                if (cache.capture_layer_ids[k] == il) { cap_idx = k; break; }
            }
        }
        cur = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                   mk_full_f16, mk_swa_f16, pl_input,
                                   kv_start, n_tokens, cap_idx,
                                   kvi_full, kvi_swa);
    }

    // Final norm
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);

    // lm_head for ALL tokens (no slicing)
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, n_tokens]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    // Argmax per token
    cur = ggml_argmax(ctx, cur);  // [n_tokens]
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc_verify = nullptr;
    if (!galloc_verify) galloc_verify = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_verify, gf)) {
        std::fprintf(stderr, "gemma4_verify_batch: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set inputs
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    // Masks (kvflash: slot-space full mask + slot rows via the shared helper)
    std::vector<float> mfull;
    if (kvflash) {
        std::vector<int32_t> rows;
        if (!kvflash_fill_rows_and_masks(*kvflash, kv_start, n_tokens,
                                         kv_len_padded, /*swa_window=*/0,
                                         rows, &mfull, nullptr)) {
            ggml_free(ctx);
            return false;
        }
        ggml_backend_tensor_set(kvi_full, rows.data(), 0, ggml_nbytes(kvi_full));
        std::vector<int32_t> ring((size_t)n_tokens);
        for (int i = 0; i < n_tokens; ++i) ring[(size_t)i] = (kv_start + i) % swa_size;
        ggml_backend_tensor_set(kvi_swa, ring.data(), 0, ggml_nbytes(kvi_swa));
    } else {
        mfull.assign((size_t)kv_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q && k < kv_len_raw; ++k) {
                mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
    }
    ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));

    // SWA ring-buffer mask
    const int W = w.sliding_window;
    std::vector<float> mswa((size_t)swa_len_padded * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        const int abs_q = kv_start + q;
        const int win_lo = std::max(0, abs_q - W + 1);
        for (int abs_k = win_lo; abs_k <= abs_q; ++abs_k) {
            const int slot = abs_k % swa_size;
            if (slot < swa_len_raw) {
                mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
            }
        }
    }
    ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_verify_batch: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read argmax
    out_argmax.resize(n_tokens);
    ggml_backend_tensor_get(cur, out_argmax.data(), 0, sizeof(int32_t) * n_tokens);

    cache.cur_pos = kv_len_raw;
    ggml_free(ctx);
    return true;
}

// ── DenoiseGallocCache ─────────────────────────────────────────────────
// Phase-3 perf fix: cache only the ggml_gallocr_t across denoising steps.
//
// The graph topology (n_nodes, tensor shapes) is identical across all steps
// for a given (n_tokens, n_prompt, do_sc) shape. ggml_gallocr_alloc_graph
// fast-paths when the topology matches: no GPU reallocation, just re-assigns
// the same device-memory addresses to the new tensors.
//
// We rebuild ggml_context + ggml_cgraph every call — this is cheap (pure CPU
// host memory), avoids the CUDA graph capture/replay issue that arises when
// the same ggml_cgraph pointer is reused across multiple compute calls, and
// keeps ggml_scale(ctx, x, sc_temp_inv) working with per-step float constants.
//
// Net savings: ~10 ms/step (GPU memory allocation eliminated via fast-path;
// CUDA graph capture/replay trap avoided; graph rebuild is ~0.7 ms/step on CPU).
struct DenoiseGallocCache {
    int            n_tokens = -1;
    int            n_prompt = -1;
    bool           do_sc    = false;
    ggml_backend_t backend  = nullptr;
    ggml_gallocr_t galloc   = nullptr;

    bool matches(int nt, int np, bool sc, ggml_backend_t be) const {
        return galloc && n_tokens == nt && n_prompt == np &&
               do_sc == sc && backend == be;
    }
    void free_all() {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        n_tokens = -1; n_prompt = -1; backend = nullptr;
    }
};

static DenoiseGallocCache s_denoise_galloc;

// ── gemma4_denoise_batch ────────────────────────────────────────────────
// Region-aware bidirectional forward over [prompt | canvas] for DiffusionGemma.
//
// Three region-aware behaviours (matching diffusion-gemma.cpp from PR #24423):
//   1. Canvas embed: rms_norm_noscale + optional SC MLP injection (ref :347-360).
//   2. Attention mask: prompt-causal / canvas-bidirectional split per-layer SWA
//      pattern (ref :28-81).
//   3. Per-layer scalar: enc_out_scale for prompt rows, out_scale for canvas rows
//      (ref :474-487).
// Returns full canvas logits [n_vocab, C] F32. Phase-2 unified path (no KV cache).
bool gemma4_denoise_batch(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     n_prompt,
    const float *           sc_logits,
    float                   sc_use,
    float                   sc_temp_inv,
    ggml_tensor *           sc_embT,
    std::vector<float> &    out_logits
#ifdef DFLASH27B_BACKEND_CUDA
    , DenoiseBatchGpuMode * dev
#endif
    )
{
    // P = prompt, C = canvas
    const int P = n_prompt;
    const int C = n_tokens - P;

    if (n_tokens <= 0 || C <= 0 || P < 0) {
        std::fprintf(stderr, "gemma4_denoise_batch: bad split (n=%d P=%d C=%d)\n",
                     n_tokens, P, C);
        return false;
    }
    // max_ctx must be at least (n_tokens+255)&~255 because build_gemma4_attn_block
    // pads the full-attn kv_len to a 256 boundary and views the cache at that size.
    const int min_ctx = (n_tokens + 255) & ~255;
    if (cache.max_ctx < min_ctx) {
        std::fprintf(stderr,
            "gemma4_denoise_batch: max_ctx %d < min required %d for n_tokens=%d\n",
            cache.max_ctx, min_ctx, n_tokens);
        return false;
    }
    if (cache.swa_size > 0 && n_tokens > cache.swa_size) {
        std::fprintf(stderr,
            "gemma4_denoise_batch: n_tokens %d exceeds SWA ring %d "
            "(warm-prefix path not yet implemented)\n", n_tokens, cache.swa_size);
        return false;
    }

    // In GPU mode, SC is active when dev->sc_dev_in != nullptr (device-resident SC).
    // In CPU mode, SC is active when sc_logits != nullptr (host-resident SC).
    const bool sc_active =
#ifdef DFLASH27B_BACKEND_CUDA
        (dev && dev->sc_dev_in) ? true :
#endif
        (sc_logits != nullptr);
    const bool do_sc = (sc_active && sc_embT != nullptr && w.sc_pre_norm != nullptr &&
                        w.sc_gate != nullptr && w.sc_up != nullptr && w.sc_down != nullptr);

    // ── Phase-3: gallocr cache — avoid GPU realloc each step ──────────
    // Invalidate on shape/SC change (new generation or first call).
    if (!s_denoise_galloc.matches(n_tokens, n_prompt, do_sc, backend)) {
        s_denoise_galloc.free_all();
    }

    // Mask geometry (constant per generation).
    const int kv_len_raw     = n_tokens;
    const int kv_len_padded  = (kv_len_raw + 255) & ~255;
    const int swa_size       = cache.swa_size;
    const int swa_len_raw    = swa_size > 0 ? std::min(n_tokens, swa_size) : n_tokens;
    const int swa_len_padded = (swa_len_raw + 255) & ~255;

    // Build graph (fresh context every call — cheap, avoids CUDA-graph stale-data bug).
    // Graph context. The graph includes per-layer embeddings, SC MLP, 30 layers,
    // final norm, lm_head — budget amply for 8192 nodes.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, 32768, false);

    // ── Input tensors ─────────────────────────────────────────────────

    // Full embedded input [n_embd, P+C] (prompt already scaled by caller)
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);

    // RoPE positions: prompt = 0..P-1, canvas = P..P+C-1 (ref: canvas continues
    // past prompt, does NOT restart at 0). (plan.md §RoPE positions)
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);

    // Token IDs for per-layer embedding lookup
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }

    // ── SC logits input [n_vocab, C] ─────────────────────────────────
    // sc_use and sc_temp_inv are baked as ggml_scale constants each call since
    // we rebuild the graph — no [1]-tensor trick needed.
    ggml_tensor * sc_logits_t = nullptr;
    if (do_sc) {
        sc_logits_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_vocab, C);
        ggml_set_input(sc_logits_t);
    }

    // ── Attention masks ───────────────────────────────────────────────
    // Unified square [P+C, P+C] mask: separate full-attn and SWA variants.
    // Built on the host in set_input below (ref diffusion-gemma.cpp:28-81).
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);

    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, swa_len_padded, n_tokens, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Unpadded F32 masks for standard (non-FA) no-cache attention path.
    // ggml_soft_max_ext requires mask->ne[0] == kq->ne[0] = n_tokens exactly.
    // Pre-allocate the mask data in host vectors and back the tensors with CPU
    // buffers created from pointers. This avoids the gallocr buffer-not-set issue
    // that occurs when fresh input tensors are created but the CUDA gallocr doesn't
    // allocate them into a device buffer before ggml_backend_tensor_set is called.
    //
    // Unpadded F32 masks for standard (non-FA) no-cache attention path.
    // ggml_soft_max_ext requires mask->ne[0] == kq->ne[0] = n_tokens exactly.
    // Extract [n_tokens, n_tokens] from the padded F32 masks via cont(view).
    // The padded mask has nb[1] = kv_len_padded * sizeof(float), so a view of
    // ne[0]=n_tokens with the same nb[1] extracts the first n_tokens columns.
    ggml_tensor * mk_full_f32_sq = (kv_len_padded == n_tokens)
        ? mk_full
        : ggml_cont(ctx, ggml_view_4d(ctx, mk_full, n_tokens, n_tokens, 1, 1,
                                       mk_full->nb[1], mk_full->nb[2], mk_full->nb[3], 0));
    ggml_tensor * mk_swa_f32_sq  = (swa_len_padded == n_tokens)
        ? mk_swa
        : ggml_cont(ctx, ggml_view_4d(ctx, mk_swa, n_tokens, n_tokens, 1, 1,
                                       mk_swa->nb[1], mk_swa->nb[2], mk_swa->nb[3], 0));

    // ── Per-layer embeddings (same as gemma4_step) ────────────────────
    ggml_tensor * per_layer_all = nullptr;
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, n_tokens);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, n_tokens);
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // ── Canvas embedding: bare rms_norm + optional SC MLP ────────────
    // Prompt rows are already scaled (sqrt(n_embd) applied in caller). Canvas
    // rows get rms_norm_noscale — the SC MLP result is added to the canvas
    // embedding before that norm. (ref diffusion-gemma.cpp:361-384)
    //
    // self_cond MLP (ref :347-360):
    //   probs = softmax(sc_logits * sc_temp_inv)
    //   soft  = sc_embT @ probs ; soft *= sqrt(n_embd)
    //   normed = rms_norm(soft, sc_pre_norm)
    //   g = gelu(sc_gate @ normed) ; u = sc_up @ normed
    //   sc_sig = sc_down @ (g * u) ; sc_sig *= sc_use
    //   canvas = rms_norm(canvas + sc_sig)           // bare, no scale weight

    ggml_tensor * cur_embed = ie;

    if (P > 0 && C > 0) {
        // Split prompt and canvas embedding rows
        ggml_tensor * prompt_embed = ggml_view_2d(ctx, ie, w.n_embd, P,
                                                   ie->nb[1], 0);
        ggml_tensor * canvas_embed = ggml_view_2d(ctx, ie, w.n_embd, C,
                                                   ie->nb[1], (size_t)P * ie->nb[1]);
        canvas_embed = ggml_cont(ctx, canvas_embed);

        if (do_sc) {
            // SC MLP subgraph (ref diffusion-gemma.cpp:347-360)
            ggml_tensor * probs = ggml_soft_max(ctx,
                ggml_scale(ctx, sc_logits_t, sc_temp_inv));           // [n_vocab, C]
            // sc_embT {n_vocab, n_embd} F16; ggml_mul_mat(A,B)=A^T@B
            // A={n_vocab,n_embd}: A^T={n_embd,n_vocab}; B={n_vocab,C} → [n_embd,C]
            ggml_tensor * soft = ggml_mul_mat(ctx, sc_embT, probs);    // [n_embd, C]
            soft = ggml_scale(ctx, soft, std::sqrt((float)w.n_embd));  // ref :352
            // SC MLP pre-norm (with weight sc_pre_norm)
            ggml_tensor * normed = gemma4_rms_norm_mul(ctx, soft, w.sc_pre_norm, w.norm_eps); // ref :354
            // gate path: ggml_gelu = tanh-approx GELU (same as backbone; ref :355)
            ggml_tensor * g = ggml_gelu(ctx, ggml_mul_mat(ctx, w.sc_gate, normed)); // [n_ff, C]
            ggml_tensor * u = ggml_mul_mat(ctx, w.sc_up, normed);                   // [n_ff, C]
            ggml_tensor * sc_sig = ggml_mul_mat(ctx, w.sc_down,
                                                ggml_mul(ctx, g, u));               // [n_embd, C]
            sc_sig = ggml_scale(ctx, sc_sig, sc_use);                               // ref :358; 0.0 on step 0
            canvas_embed = ggml_add(ctx, canvas_embed, sc_sig);
        }
        // Bare rms_norm (no scale weight) for canvas (ref :360 / :383)
        canvas_embed = ggml_rms_norm(ctx, canvas_embed, w.norm_eps);

        // Reassemble [prompt | canvas]
        cur_embed = ggml_concat(ctx, ggml_cont(ctx, prompt_embed),
                                ggml_cont(ctx, canvas_embed), 1);
    } else if (P == 0) {
        // Pure-canvas (no prompt): SC + rms_norm
        if (do_sc) {
            ggml_tensor * canvas_all = ggml_cont(ctx, ie);
            ggml_tensor * probs = ggml_soft_max(ctx,
                ggml_scale(ctx, sc_logits_t, sc_temp_inv));
            ggml_tensor * soft  = ggml_mul_mat(ctx, sc_embT, probs);
            soft = ggml_scale(ctx, soft, std::sqrt((float)w.n_embd));
            ggml_tensor * normed = gemma4_rms_norm_mul(ctx, soft, w.sc_pre_norm, w.norm_eps);
            ggml_tensor * g = ggml_gelu(ctx, ggml_mul_mat(ctx, w.sc_gate, normed));
            ggml_tensor * u = ggml_mul_mat(ctx, w.sc_up, normed);
            ggml_tensor * sc_sig = ggml_mul_mat(ctx, w.sc_down, ggml_mul(ctx, g, u));
            sc_sig = ggml_scale(ctx, sc_sig, sc_use);
            canvas_all = ggml_add(ctx, canvas_all, sc_sig);
            cur_embed  = ggml_rms_norm(ctx, canvas_all, w.norm_eps);
        } else {
            cur_embed = ggml_rms_norm(ctx, ggml_cont(ctx, ie), w.norm_eps);
        }
    }
    // P == n_tokens (all prompt, no canvas) would be caught by C<=0 guard above.

    // ── Transformer layers ────────────────────────────────────────────
    ggml_tensor * cur = cur_embed;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);

        // build_gemma4_layer handles attn_norm, Q/K/V, RoPE, FA, post_norm,
        // residual, FFN, ffn_post_norm, per_layer_inject — but NOT out_scale
        // (we handle it here region-aware instead of the uniform path in the layer).
        // Pass the SWA-appropriate mask; the layer selects the right one via is_swa.
        ggml_tensor * layer_out = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                                      mk_full_f16, mk_swa_f16, pl_input,
                                                      /*kv_start=*/0, n_tokens,
                                                      /*capture_idx=*/-1,
                                                      /*kv_idx_full=*/nullptr,
                                                      /*kv_idx_swa=*/nullptr,
                                                      /*no_cache=*/true,
                                                      /*attn_mask_full_f32=*/mk_full_f32_sq,
                                                      /*attn_mask_swa_f32=*/mk_swa_f32_sq);

        // ── Region-aware per-layer scalar (ref diffusion-gemma.cpp:474-487) ──
        // enc_out_scale for prompt rows, out_scale for canvas rows.
        // build_gemma4_layer already applied out_scale (its own residual path);
        // we need to override: remove the uniform out_scale it applies and redo
        // region-split. BUT: build_gemma4_layer applies out_scale internally for
        // the branch's existing tensors. Looking at the implementation, out_scale
        // is applied inside build_gemma4_layer at the end. Since enc_out_scale
        // for the branch != out_scale for prompt rows, we must NOT call
        // build_gemma4_layer's internal out_scale application for the prompt rows.
        //
        // Solution: build_gemma4_layer applies L.out_scale (if non-null) after FFN.
        // We temporarily disable it by treating the returned value as already having
        // out_scale applied to ALL rows (which is wrong for prompt), then:
        //   prompt_corrected = prompt_rows * (enc_out_scale / out_scale)  — NOT clean.
        //
        // Cleaner: since build_gemma4_layer already multiplies by out_scale, and we
        // want enc_out_scale on prompt:
        //   prompt_corrected = prompt_rows * enc_out_scale / out_scale
        // But scalar division is tricky. Instead, don't use build_gemma4_layer's
        // out_scale application for this forward — we duplicate the layer logic here.
        //
        // Actually, inspect build_gemma4_layer: it applies L.out_scale at the end.
        // The scale is a 1-element tensor. We need to divide the prompt portion back
        // out and multiply by enc_out_scale. Given out_scale is a device scalar we
        // can't easily read it on CPU. The correct approach is to duplicate the layer
        // body without out_scale and apply region-aware scales ourselves.
        //
        // For Phase-2 correctness, we undo the uniform out_scale on prompt rows and
        // reapply enc_out_scale. Canvas rows already have the correct out_scale.
        // Undo on prompt: multiply by (1/out_scale) then by enc_out_scale.
        // ggml doesn't have div-by-tensor, but we can do mul(layer_out, recip).
        // Since we need 1/out_scale on the GPU and out_scale is a [1] tensor, we use
        // a workaround: apply enc_out_scale / out_scale via ggml_div.

        const Gemma4Layer & L = w.layers[il];
        if (P > 0 && C > 0 && L.out_scale && L.enc_out_scale) {
            // layer_out has out_scale applied to ALL rows by build_gemma4_layer.
            // Correct prompt rows: multiply by enc_out_scale * (1/out_scale).
            // ggml_div(enc_out_scale, out_scale) gives the correction factor [1].
            ggml_tensor * prompt_rows = ggml_cont(ctx,
                ggml_view_2d(ctx, layer_out, w.n_embd, P, layer_out->nb[1], 0));
            ggml_tensor * canvas_rows = ggml_cont(ctx,
                ggml_view_2d(ctx, layer_out, w.n_embd, C,
                             layer_out->nb[1], (size_t)P * layer_out->nb[1]));
            // correction = enc_out_scale / out_scale
            ggml_tensor * correction = ggml_div(ctx, L.enc_out_scale, L.out_scale);
            prompt_rows = ggml_mul(ctx, prompt_rows, correction);
            cur = ggml_concat(ctx, prompt_rows, canvas_rows, 1);
        } else if (P == 0 && L.out_scale) {
            // All canvas — build_gemma4_layer already applied out_scale, correct.
            cur = layer_out;
        } else if (C == 0 && L.out_scale && L.enc_out_scale) {
            // All prompt — need enc_out_scale, but build_gemma4_layer applied out_scale.
            ggml_tensor * correction = ggml_div(ctx, L.enc_out_scale, L.out_scale);
            cur = ggml_mul(ctx, layer_out, correction);
        } else {
            cur = layer_out;
        }
    }

    // ── Final norm + lm_head over canvas rows only ────────────────────
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);

    // Slice to canvas rows before lm_head (ref plan.md §CANVAS-LOGITS RETURN)
    if (P > 0) {
        cur = ggml_cont(ctx,
            ggml_view_2d(ctx, cur, w.n_embd, C,
                         cur->nb[1], (size_t)P * cur->nb[1]));
    }

    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, C]
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // ── Allocate ──────────────────────────────────────────────────────
    // Use cached gallocr when topology matches; allocates fresh GPU memory only
    // on the first call (or after shape change). Subsequent calls fast-path:
    // same buffer assignments, no GPU realloc.
    if (!s_denoise_galloc.galloc) {
        s_denoise_galloc.galloc  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        s_denoise_galloc.n_tokens = n_tokens;
        s_denoise_galloc.n_prompt = n_prompt;
        s_denoise_galloc.do_sc    = do_sc;
        s_denoise_galloc.backend  = backend;
    }
    if (!ggml_gallocr_alloc_graph(s_denoise_galloc.galloc, gf)) {
        std::fprintf(stderr, "gemma4_denoise_batch: gallocr_alloc_graph failed\n");
        s_denoise_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

    // ── Upload inputs ─────────────────────────────────────────────────
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));

    // RoPE: prompt = 0..P-1, canvas = P..P+C-1 (ref plan.md §RoPE positions)
    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = i;  // absolute position for all
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    if (do_sc && sc_logits_t) {
#ifdef DFLASH27B_BACKEND_CUDA
        if (dev && dev->sc_dev_in) {
            // GPU mode: SC input is already on device — D2D copy, no PCIe traffic.
            const size_t sc_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
            cudaError_t err = cudaMemcpy(sc_logits_t->data, dev->sc_dev_in,
                                         sc_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gemma4_denoise_batch: SC D2D failed: %s\n",
                             cudaGetErrorString(err));
                ggml_free(ctx);
                return false;
            }
        } else
#endif
        {
            // CPU mode: H2D upload of host sc_logits buffer.
            ggml_backend_tensor_set(sc_logits_t, sc_logits, 0,
                                    (size_t)w.n_vocab * (size_t)C * sizeof(float));
        }
    }

    // ── Region-aware attention mask (ref diffusion-gemma.cpp:28-81) ──
    // prompt q:  causal over prompt keys only (SWA-clipped if swa layer).
    // canvas q, global:  attend all P+C keys.
    // canvas q, SWA:     attend all C canvas keys + last (n_swa-1) prompt keys.
    //
    // We build two masks (full-attn and SWA); build_gemma4_layer selects the right
    // one per layer based on is_swa. The masks are [kv_len_padded, n_tokens].
    // The canvas_prompt_lo = P - (sliding_window - 1) for SWA canvas queries.
    const int n_swa = w.sliding_window;
    const int canvas_prompt_lo = P - (n_swa > 0 ? n_swa - 1 : 0);

    // Full-attention mask (global layers: canvas sees all, prompt is causal).
    {
        std::vector<float> mfull((size_t)kv_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const bool q_is_canvas = (q >= P);
            for (int k = 0; k < kv_len_raw; ++k) {
                const bool k_is_canvas = (k >= P);
                bool allow;
                if (q_is_canvas) {
                    allow = true;  // canvas global: attend all prompt+canvas
                } else {
                    // prompt causal: only earlier/equal prompt positions, never canvas
                    allow = (!k_is_canvas) && (k <= q);
                }
                if (allow) mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    }

    // SWA mask (sliding-window layers): canvas sees last (n_swa-1) prompt + all canvas;
    // prompt queries causal + SWA-clipped (no farther than n_swa positions).
    {
        std::vector<float> mswa((size_t)swa_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const bool q_is_canvas = (q >= P);
            for (int k = 0; k < kv_len_raw; ++k) {
                const bool k_is_canvas = (k >= P);
                bool allow;
                if (q_is_canvas) {
                    // SWA canvas: all canvas keys + last (n_swa-1) prompt positions
                    allow = k_is_canvas || (k >= canvas_prompt_lo);
                } else {
                    // SWA prompt: causal + sliding window
                    allow = (!k_is_canvas) && (k <= q) &&
                            (n_swa <= 0 || q - k < n_swa);
                }
                if (allow) {
                    const int slot = (swa_size > 0) ? (k % swa_size) : k;
                    if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    // (Square masks mk_full_f32_sq / mk_swa_f32_sq are derived from mk_full / mk_swa
    //  via ggml_cont(view) — no separate fill needed.)

    // ── Compute ───────────────────────────────────────────────────────
#ifdef DFLASH27B_BACKEND_CUDA
    cudaEvent_t ev_fwd0 = nullptr, ev_fwd1 = nullptr;
    cudaEvent_t ev_sc1  = nullptr, ev_d2h1 = nullptr;
    if (dev) {
        cudaEventCreate(&ev_fwd0);
        cudaEventCreate(&ev_fwd1);
        cudaEventCreate(&ev_sc1);
        cudaEventCreate(&ev_d2h1);
        cudaEventRecord(ev_fwd0, /*stream=*/0);
    }
#endif

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_denoise_batch: graph_compute failed\n");
        s_denoise_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

    // ── Output ────────────────────────────────────────────────────────
#ifdef DFLASH27B_BACKEND_CUDA
    if (dev) {
        cudaEventRecord(ev_fwd1, /*stream=*/0);

        // GPU mode: keep logits device-resident.

        // 1. D2D copy cur->data → sc_dev_out for next step's SC.
        if (dev->sc_dev_out) {
            const size_t logit_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
            cudaError_t err = cudaMemcpy(dev->sc_dev_out, cur->data,
                                         logit_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gemma4_denoise_batch: SC-out D2D failed: %s\n",
                             cudaGetErrorString(err));
                ggml_free(ctx);
                return false;
            }
        }
        cudaEventRecord(ev_sc1, /*stream=*/0);

        // 2. Allocate small per-step device result buffers and run sampling kernel.
        const int C_local = C;
        int32_t * d_samp = nullptr;
        float   * d_ent  = nullptr;
        int32_t * d_amax = nullptr;
        cudaMalloc(&d_samp, (size_t)C_local * sizeof(int32_t));
        cudaMalloc(&d_ent,  (size_t)C_local * sizeof(float));
        cudaMalloc(&d_amax, (size_t)C_local * sizeof(int32_t));

        dflash::diffusion::diffusion_sample_gpu(
            static_cast<const float *>(cur->data),
            dev->u_dev,
            dev->temp_inv,
            C_local,
            w.n_vocab,
            d_samp, d_ent, d_amax,
            /*stream=*/0);

        // 3. Sync + copy tiny results (~3 KB) to host.
        cudaDeviceSynchronize();

        dev->out_sampled->resize(C_local);
        dev->out_entropy->resize(C_local);
        dev->out_argmax->resize(C_local);
        cudaMemcpy(dev->out_sampled->data(), d_samp,
                   (size_t)C_local * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(dev->out_entropy->data(), d_ent,
                   (size_t)C_local * sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(dev->out_argmax->data(),  d_amax,
                   (size_t)C_local * sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(d_samp);
        cudaFree(d_ent);
        cudaFree(d_amax);

        // 4. Emit per-step split timing.
        cudaEventRecord(ev_d2h1, /*stream=*/0);
        cudaEventSynchronize(ev_d2h1);
        float ms_fwd = 0, ms_sc = 0, ms_d2h = 0;
        cudaEventElapsedTime(&ms_fwd, ev_fwd0, ev_fwd1);
        cudaEventElapsedTime(&ms_sc,  ev_fwd1, ev_sc1);
        cudaEventElapsedTime(&ms_d2h, ev_sc1,  ev_d2h1);
        std::fprintf(stderr,
            "[dg-split] fwd=%.1f ms  sc_d2d=%.1f ms  samp+d2h=%.1f ms  C=%d\n",
            ms_fwd, ms_sc, ms_d2h, C_local);
        cudaEventDestroy(ev_fwd0);
        cudaEventDestroy(ev_fwd1);
        cudaEventDestroy(ev_sc1);
        cudaEventDestroy(ev_d2h1);

        // out_logits intentionally left empty (logits stay device-resident).
    } else
#endif
    {
        // CPU mode: D2H of full logits [n_vocab, C].
        out_logits.resize((size_t)w.n_vocab * (size_t)C);
        ggml_backend_tensor_get(cur, out_logits.data(), 0, sizeof(float) * out_logits.size());
    }

    cache.cur_pos = n_tokens;
    ggml_free(ctx);
    // s_denoise_galloc.galloc owns the device buffers — do NOT free it here.
    return true;
}

// ── L0: gemma4_prefill_prompt_for_denoise ──────────────────────────────────
// Populates the KV cache with the P prompt tokens.  Must be called once per
// generation before the canvas loop.  After success, cache.cur_pos == P.
//
// Builds a graph over P tokens with:
//   - causal mask (same as AR step)
//   - no_cache = false → writes K/V to cache[0..P-1]
//   - per-layer embeddings computed for P tokens
// Does NOT compute logits (we only need the KV side-effect).
bool gemma4_prefill_prompt_for_denoise(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     P)
{
    if (P <= 0) { cache.cur_pos = 0; return true; }  // no-op for empty prompt

    const int min_ctx = (P + 255) & ~255;
    if (cache.max_ctx < min_ctx) {
        std::fprintf(stderr, "gemma4_prefill_prompt_for_denoise: max_ctx %d < %d for P=%d\n",
                     cache.max_ctx, min_ctx, P);
        return false;
    }

    // Build a small graph: just process P tokens through all layers (cache write side-effect)
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);

    // Input embedding: [n_embd, P] F32
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, P);
    ggml_set_input(ie);

    // RoPE positions [0..P-1]
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_set_input(pp);

    // kv_idx tensors (set_rows stable pointer path)
    ggml_tensor * kvi_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_set_input(kvi_full);
    ggml_tensor * kvi_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_set_input(kvi_swa);

    // Token IDs for per-layer embeddings
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
        ggml_set_input(tok_ids);
    }

    // Attention masks (causal for prompt)
    const int kv_len_padded  = (P + 255) & ~255;
    const int swa_size       = cache.swa_size;
    const int swa_len_raw    = swa_size > 0 ? std::min(P, swa_size) : P;
    const int swa_len_padded = (swa_len_raw + 255) & ~255;

    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, P, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);

    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, swa_len_padded, P, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embeddings (same pattern as gemma4_step / gemma4_denoise_batch)
    ggml_tensor * per_layer_all = nullptr;
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, P);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, P);
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Run all layers (cache write side-effect; no_cache=false)
    ggml_tensor * cur = ie;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);

        ggml_tensor * layer_out = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                                      mk_full_f16, mk_swa_f16, pl_input,
                                                      /*kv_start=*/0, P,
                                                      /*capture_idx=*/-1,
                                                      kvi_full, kvi_swa,
                                                      /*no_cache=*/false);
        // Apply enc_out_scale to prompt hidden states (matches denoise_batch region logic)
        const Gemma4Layer & L = w.layers[il];
        if (L.out_scale && L.enc_out_scale) {
            ggml_tensor * correction = ggml_div(ctx, L.enc_out_scale, L.out_scale);
            cur = ggml_mul(ctx, layer_out, correction);
        } else {
            cur = layer_out;
        }
    }
    // Expose as output so gallocr allocates the full graph
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    static ggml_gallocr_t s_prefill_galloc = nullptr;
    if (!s_prefill_galloc) {
        s_prefill_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (!ggml_gallocr_alloc_graph(s_prefill_galloc, gf)) {
        std::fprintf(stderr, "gemma4_prefill_prompt_for_denoise: gallocr failed\n");
        ggml_free(ctx);
        return false;
    }

    // Upload inputs
    ggml_backend_tensor_set(ie, embed, 0, (size_t)w.n_embd * P * sizeof(float));
    std::vector<int32_t> pos(P);
    for (int i = 0; i < P; ++i) pos[i] = i;
    ggml_backend_tensor_set(pp, pos.data(), 0, (size_t)P * sizeof(int32_t));
    ggml_backend_tensor_set(kvi_full, pos.data(), 0, (size_t)P * sizeof(int32_t));
    if (swa_size > 0) {
        std::vector<int32_t> ring(P);
        for (int i = 0; i < P; ++i) ring[i] = i % swa_size;
        ggml_backend_tensor_set(kvi_swa, ring.data(), 0, (size_t)P * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(kvi_swa, pos.data(), 0, (size_t)P * sizeof(int32_t));
    }
    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)P * sizeof(int32_t));
    }

    // Causal prompt mask (full attention layers)
    {
        std::vector<float> mfull((size_t)kv_len_padded * P, -INFINITY);
        for (int q = 0; q < P; ++q) {
            for (int k = 0; k <= q; ++k) mfull[(size_t)q * kv_len_padded + k] = 0.0f;
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    }
    // SWA causal mask (ring-buffer indexed)
    {
        const int W = w.sliding_window;
        std::vector<float> mswa((size_t)swa_len_padded * P, -INFINITY);
        for (int q = 0; q < P; ++q) {
            const int win_lo = (W > 0) ? std::max(0, q - W + 1) : 0;
            for (int k = win_lo; k <= q; ++k) {
                const int slot = (swa_size > 0) ? (k % swa_size) : k;
                if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_prefill_prompt_for_denoise: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    cache.cur_pos = P;
    ggml_free(ctx);
    return true;
}

// ── L0 gallocr cache for canvas-only steps ─────────────────────────────────
// Same fast-path as s_denoise_galloc but keyed on (C, P, do_sc).
struct CanvasGallocCache {
    int            C      = -1;
    int            P      = -1;
    bool           do_sc  = false;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc  = nullptr;

    bool matches(int c, int p, bool sc, ggml_backend_t be) const {
        return galloc && C == c && P == p && do_sc == sc && backend == be;
    }
    void free_all() {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        C = -1; P = -1; backend = nullptr;
    }
};
static CanvasGallocCache s_canvas_galloc;

// ── L0: gemma4_denoise_canvas ───────────────────────────────────────────────
// Canvas-only denoising step using cached prompt KV.
// embed[P..P+C-1] are the canvas token embeddings (unscaled; this function
// applies bare rms_norm + optional SC, matching gemma4_denoise_batch).
// cache.cur_pos must equal P (set by gemma4_prefill_prompt_for_denoise).
//
// Attention: canvas queries attend ALL P+C KV positions:
//   - Full-attn layers: keys [0..P+C-1] (prompt cached + canvas written now)
//   - SWA layers:       last (n_swa-1) prompt keys + all C canvas keys
bool gemma4_denoise_canvas(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     n_prompt,
    const float *           sc_logits,
    float                   sc_use,
    float                   sc_temp_inv,
    ggml_tensor *           sc_embT,
    std::vector<float> &    out_logits
#ifdef DFLASH27B_BACKEND_CUDA
    , DenoiseBatchGpuMode * dev
#endif
    )
{
    const int P = n_prompt;
    const int C = n_tokens - P;

    if (C <= 0 || P < 0) {
        std::fprintf(stderr, "gemma4_denoise_canvas: bad split (n=%d P=%d C=%d)\n",
                     n_tokens, P, C);
        return false;
    }
    if (cache.cur_pos != P) {
        std::fprintf(stderr, "gemma4_denoise_canvas: cache.cur_pos=%d != P=%d; "
                     "call gemma4_prefill_prompt_for_denoise first\n",
                     cache.cur_pos, P);
        return false;
    }

    const int total   = P + C;        // total KV positions after canvas write
    const int min_ctx = (total + 255) & ~255;
    if (cache.max_ctx < min_ctx) {
        std::fprintf(stderr, "gemma4_denoise_canvas: max_ctx %d < %d for P+C=%d\n",
                     cache.max_ctx, min_ctx, total);
        return false;
    }

    const bool sc_active =
#ifdef DFLASH27B_BACKEND_CUDA
        (dev && dev->sc_dev_in) ? true :
#endif
        (sc_logits != nullptr);
    const bool do_sc = (sc_active && sc_embT != nullptr && w.sc_pre_norm != nullptr &&
                        w.sc_gate != nullptr && w.sc_up != nullptr && w.sc_down != nullptr);

    if (!s_canvas_galloc.matches(C, P, do_sc, backend)) {
        s_canvas_galloc.free_all();
    }

    // Mask geometry: canvas forward kv covers [0..P+C-1]
    const int kv_len_raw     = total;
    const int kv_len_padded  = (kv_len_raw + 255) & ~255;
    const int swa_size       = cache.swa_size;
    // SWA ring covers positions up to min(P+C, swa_size)
    const int swa_len_raw    = swa_size > 0 ? std::min(total, swa_size) : total;
    const int swa_len_padded = (swa_len_raw + 255) & ~255;

    // Build graph — canvas tokens only (C tokens)
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf  = ggml_new_graph_custom(ctx, 32768, false);

    // Canvas embedding input: [n_embd, C] F32
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, C);
    ggml_set_input(ie);

    // RoPE positions for canvas: [P..P+C-1]
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
    ggml_set_input(pp);

    // kv_idx for the C canvas tokens (positions P..P+C-1)
    ggml_tensor * kvi_full = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
    ggml_set_input(kvi_full);
    ggml_tensor * kvi_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
    ggml_set_input(kvi_swa);

    // Token IDs for per-layer embeddings (canvas tokens only)
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, C);
        ggml_set_input(tok_ids);
    }

    // SC input [n_vocab, C]
    ggml_tensor * sc_logits_t = nullptr;
    if (do_sc) {
        sc_logits_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_vocab, C);
        ggml_set_input(sc_logits_t);
    }

    // Attention masks (canvas q attends ALL P+C keys)
    ggml_tensor * mk_full = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kv_len_padded, C, 1, 1);
    ggml_set_input(mk_full);
    ggml_tensor * mk_full_f16 = ggml_cast(ctx, mk_full, GGML_TYPE_F16);

    ggml_tensor * mk_swa = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, swa_len_padded, C, 1, 1);
    ggml_set_input(mk_swa);
    ggml_tensor * mk_swa_f16 = ggml_cast(ctx, mk_swa, GGML_TYPE_F16);

    // Per-layer embeddings for canvas tokens
    ggml_tensor * per_layer_all = nullptr;
    if (tok_ids) {
        const int D = w.n_embd_per_layer;
        const int L = w.n_layer;
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D, L, C);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D));
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, ie);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)w.n_embd));
        proj = ggml_reshape_3d(ctx, proj, D, L, C);
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), w.norm_eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D, L);
        proj = ggml_mul(ctx, proj, norm_w);
        per_layer_all = ggml_add(ctx, proj, inp_pl);
        per_layer_all = ggml_scale(ctx, per_layer_all, 1.0f / std::sqrt(2.0f));
        per_layer_all = ggml_cont(ctx, ggml_permute(ctx, per_layer_all, 0, 2, 1, 3));
    }

    // Canvas embedding: bare rms_norm + optional SC MLP (same as denoise_batch)
    ggml_tensor * cur_embed = ie;
    if (do_sc) {
        ggml_tensor * probs = ggml_soft_max(ctx,
            ggml_scale(ctx, sc_logits_t, sc_temp_inv));
        ggml_tensor * soft = ggml_mul_mat(ctx, sc_embT, probs);
        soft = ggml_scale(ctx, soft, std::sqrt((float)w.n_embd));
        ggml_tensor * normed = gemma4_rms_norm_mul(ctx, soft, w.sc_pre_norm, w.norm_eps);
        ggml_tensor * g = ggml_gelu(ctx, ggml_mul_mat(ctx, w.sc_gate, normed));
        ggml_tensor * u = ggml_mul_mat(ctx, w.sc_up, normed);
        ggml_tensor * sc_sig = ggml_mul_mat(ctx, w.sc_down, ggml_mul(ctx, g, u));
        sc_sig = ggml_scale(ctx, sc_sig, sc_use);
        cur_embed = ggml_add(ctx, ie, sc_sig);
    }
    // Bare rms_norm (no scale weight) — matches canvas path in denoise_batch
    cur_embed = ggml_rms_norm(ctx, ggml_cont(ctx, cur_embed), w.norm_eps);

    // Run all layers with kv_start=P, no_cache=false
    // build_gemma4_layer writes canvas K/V to cache[P..P+C-1] and reads
    // ALL P+C K/V (prompt cached + canvas written this step) for attention.
    ggml_tensor * cur = cur_embed;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * pl_input = nullptr;
        if (per_layer_all) pl_input = gemma4_view_2d_slice(ctx, per_layer_all, il);

        ggml_tensor * layer_out = build_gemma4_layer(ctx, gf, w, cache, il, cur, pp,
                                                      mk_full_f16, mk_swa_f16, pl_input,
                                                      /*kv_start=*/P, C,
                                                      /*capture_idx=*/-1,
                                                      kvi_full, kvi_swa,
                                                      /*no_cache=*/false);

        // build_gemma4_layer already applies out_scale internally (see ~line 497).
        // For canvas-only forward, out_scale is correct as-is (no enc_out_scale correction
        // needed, since all tokens are canvas). Use layer_out directly.
        cur = layer_out;
    }

    // Final norm + lm_head over C canvas tokens
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, C]
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate via cached gallocr
    if (!s_canvas_galloc.galloc) {
        s_canvas_galloc.galloc   = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        s_canvas_galloc.C        = C;
        s_canvas_galloc.P        = P;
        s_canvas_galloc.do_sc    = do_sc;
        s_canvas_galloc.backend  = backend;
    }
    if (!ggml_gallocr_alloc_graph(s_canvas_galloc.galloc, gf)) {
        std::fprintf(stderr, "gemma4_denoise_canvas: gallocr_alloc_graph failed\n");
        s_canvas_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

    // Upload inputs

    // Canvas embed = ie[P..P+C-1] from the full embed array
    ggml_backend_tensor_set(ie, embed + (size_t)P * w.n_embd, 0,
                            (size_t)C * w.n_embd * sizeof(float));

    // RoPE positions [P..P+C-1]
    std::vector<int32_t> pos(C);
    for (int i = 0; i < C; ++i) pos[i] = P + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, (size_t)C * sizeof(int32_t));

    // kv_idx for canvas (absolute positions P..P+C-1)
    ggml_backend_tensor_set(kvi_full, pos.data(), 0, (size_t)C * sizeof(int32_t));
    if (swa_size > 0) {
        std::vector<int32_t> ring(C);
        for (int i = 0; i < C; ++i) ring[i] = (P + i) % swa_size;
        ggml_backend_tensor_set(kvi_swa, ring.data(), 0, (size_t)C * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(kvi_swa, pos.data(), 0, (size_t)C * sizeof(int32_t));
    }

    // Token IDs (canvas slice)
    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids + P, 0, (size_t)C * sizeof(int32_t));
    }

    // SC input
    if (do_sc && sc_logits_t) {
#ifdef DFLASH27B_BACKEND_CUDA
        if (dev && dev->sc_dev_in) {
            const size_t sc_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
            cudaError_t err = cudaMemcpy(sc_logits_t->data, dev->sc_dev_in,
                                         sc_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gemma4_denoise_canvas: SC D2D failed: %s\n",
                             cudaGetErrorString(err));
                ggml_free(ctx);
                return false;
            }
        } else
#endif
        {
            ggml_backend_tensor_set(sc_logits_t, sc_logits, 0,
                                    (size_t)w.n_vocab * (size_t)C * sizeof(float));
        }
    }

    // Full-attention mask: canvas q attends ALL P+C keys (0..P+C-1)
    {
        std::vector<float> mfull((size_t)kv_len_padded * C, -INFINITY);
        for (int q = 0; q < C; ++q) {
            for (int k = 0; k < total; ++k) {
                mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    }

    // SWA mask: canvas q attends all C canvas keys + last (n_swa-1) prompt keys
    // Canvas keys land at ring slots (P+i) % swa_size.
    // Prompt keys in the last (n_swa-1) positions land at slots (P-j) % swa_size, j=1..n_swa-1.
    {
        const int n_swa = w.sliding_window;
        std::vector<float> mswa((size_t)swa_len_padded * C, -INFINITY);
        for (int q = 0; q < C; ++q) {
            // All C canvas keys
            for (int ci = 0; ci < C; ++ci) {
                const int slot = (swa_size > 0) ? ((P + ci) % swa_size) : (P + ci);
                if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
            }
            // Last (n_swa-1) prompt keys
            if (n_swa > 0) {
                const int prompt_lo = std::max(0, P - (n_swa - 1));
                for (int pk = prompt_lo; pk < P; ++pk) {
                    const int slot = (swa_size > 0) ? (pk % swa_size) : pk;
                    if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
                }
            } else {
                // No SWA limit: all prompt keys visible
                for (int pk = 0; pk < P; ++pk) {
                    const int slot = (swa_size > 0) ? (pk % swa_size) : pk;
                    if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    // Compute
#ifdef DFLASH27B_BACKEND_CUDA
    cudaEvent_t ev_fwd0 = nullptr, ev_fwd1 = nullptr;
    cudaEvent_t ev_sc1  = nullptr, ev_d2h1 = nullptr;
    if (dev) {
        cudaEventCreate(&ev_fwd0);
        cudaEventCreate(&ev_fwd1);
        cudaEventCreate(&ev_sc1);
        cudaEventCreate(&ev_d2h1);
        cudaEventRecord(ev_fwd0, /*stream=*/0);
    }
#endif

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_denoise_canvas: graph_compute failed\n");
        s_canvas_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

#ifdef DFLASH27B_BACKEND_CUDA
    if (dev) {
        cudaEventRecord(ev_fwd1, /*stream=*/0);

        if (dev->sc_dev_out) {
            const size_t logit_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
            cudaError_t err = cudaMemcpy(dev->sc_dev_out, cur->data,
                                         logit_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "gemma4_denoise_canvas: SC-out D2D failed: %s\n",
                             cudaGetErrorString(err));
                ggml_free(ctx);
                return false;
            }
        }
        cudaEventRecord(ev_sc1, /*stream=*/0);

        const int C_local = C;
        int32_t * d_samp = nullptr;
        float   * d_ent  = nullptr;
        int32_t * d_amax = nullptr;
        cudaMalloc(&d_samp, (size_t)C_local * sizeof(int32_t));
        cudaMalloc(&d_ent,  (size_t)C_local * sizeof(float));
        cudaMalloc(&d_amax, (size_t)C_local * sizeof(int32_t));

        dflash::diffusion::diffusion_sample_gpu(
            static_cast<const float *>(cur->data),
            dev->u_dev,
            dev->temp_inv,
            C_local,
            w.n_vocab,
            d_samp, d_ent, d_amax,
            /*stream=*/0);

        cudaDeviceSynchronize();

        dev->out_sampled->resize(C_local);
        dev->out_entropy->resize(C_local);
        dev->out_argmax->resize(C_local);
        cudaMemcpy(dev->out_sampled->data(), d_samp,
                   (size_t)C_local * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(dev->out_entropy->data(), d_ent,
                   (size_t)C_local * sizeof(float),   cudaMemcpyDeviceToHost);
        cudaMemcpy(dev->out_argmax->data(),  d_amax,
                   (size_t)C_local * sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(d_samp);
        cudaFree(d_ent);
        cudaFree(d_amax);

        cudaEventRecord(ev_d2h1, /*stream=*/0);
        cudaEventSynchronize(ev_d2h1);
        float ms_fwd = 0, ms_sc = 0, ms_d2h = 0;
        cudaEventElapsedTime(&ms_fwd, ev_fwd0, ev_fwd1);
        cudaEventElapsedTime(&ms_sc,  ev_fwd1, ev_sc1);
        cudaEventElapsedTime(&ms_d2h, ev_sc1,  ev_d2h1);
        std::fprintf(stderr,
            "[dg-canvas-split] fwd=%.1f ms  sc_d2d=%.1f ms  samp+d2h=%.1f ms  C=%d P=%d\n",
            ms_fwd, ms_sc, ms_d2h, C_local, P);
        cudaEventDestroy(ev_fwd0);
        cudaEventDestroy(ev_fwd1);
        cudaEventDestroy(ev_sc1);
        cudaEventDestroy(ev_d2h1);
    } else
#endif
    {
        out_logits.resize((size_t)w.n_vocab * (size_t)C);
        ggml_backend_tensor_get(cur, out_logits.data(), 0, sizeof(float) * out_logits.size());
    }

    // Reset cache.cur_pos to P so next step re-enters with correct position
    cache.cur_pos = P;
    ggml_free(ctx);
    return true;
}

// ── Dead code removed: gemma4_denoise_batch_dev ─────────────────────────
// Replaced by dev-mode flag in gemma4_denoise_batch (see header).
#ifdef DFLASH27B_BACKEND_CUDA_REMOVED_DEAD_CODE
bool gemma4_denoise_batch_dev_REMOVED(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     n_tokens,
    int                     n_prompt,
    const float *           sc_dev_in,
    float                   sc_use,
    float                   sc_temp_inv,
    ggml_tensor *           sc_embT,
    float *                 sc_dev_out,
    const float *           u_dev,
    float                   temp_inv,
    std::vector<int32_t> &  out_sampled,
    std::vector<float>   &  out_entropy,
    std::vector<int32_t> &  out_argmax)
{
    const int P = n_prompt;
    const int C = n_tokens - P;

    if (n_tokens <= 0 || C <= 0 || P < 0) {
        std::fprintf(stderr, "gemma4_denoise_batch_dev: bad split (n=%d P=%d C=%d)\n",
                     n_tokens, P, C);
        return false;
    }
    const int min_ctx = (n_tokens + 255) & ~255;
    if (cache.max_ctx < min_ctx) {
        std::fprintf(stderr,
            "gemma4_denoise_batch_dev: max_ctx %d < min required %d\n",
            cache.max_ctx, min_ctx);
        return false;
    }
    if (cache.swa_size > 0 && n_tokens > cache.swa_size) {
        std::fprintf(stderr,
            "gemma4_denoise_batch_dev: n_tokens %d exceeds SWA ring %d\n",
            n_tokens, cache.swa_size);
        return false;
    }

    // SC is active when sc_dev_in != nullptr AND embT/norm/gate/up/down present.
    const bool do_sc = (sc_dev_in != nullptr && sc_embT != nullptr &&
                        w.sc_pre_norm != nullptr &&
                        w.sc_gate != nullptr && w.sc_up != nullptr && w.sc_down != nullptr);

    // Reuse the cached gallocr from gemma4_denoise_batch — same graph topology.
    if (!s_denoise_galloc.matches(n_tokens, n_prompt, do_sc, backend)) {
        s_denoise_galloc.free_all();
    }

    const int kv_len_raw     = n_tokens;
    const int kv_len_padded  = (kv_len_raw + 255) & ~255;
    const int swa_size       = cache.swa_size;
    const int swa_len_raw    = swa_size > 0 ? std::min(n_tokens, swa_size) : n_tokens;
    const int swa_len_padded = (swa_len_raw + 255) & ~255;

    // Build graph (identical to gemma4_denoise_batch).
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32768 + ggml_graph_overhead() + 32 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, 32768, false);

    // ── Input tensors (same as gemma4_denoise_batch) ──────────────────
    ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(ie);
    ggml_tensor * pp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(pp);
    ggml_tensor * tok_ids = nullptr;
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        tok_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(tok_ids);
    }
    ggml_tensor * sc_logits_t = nullptr;
    if (do_sc) {
        sc_logits_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_vocab, C);
        ggml_set_input(sc_logits_t);
    }

    // ── Attention masks ───────────────────────────────────────────────
    ggml_tensor * mk_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_len_padded, n_tokens);
    ggml_set_input(mk_full);
    ggml_tensor * mk_swa = nullptr;
    if (swa_size > 0) {
        mk_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, swa_len_padded, n_tokens);
        ggml_set_input(mk_swa);
    }
    // Square variants for the global layers (n_tokens × n_tokens subset).
    ggml_tensor * mk_full_f32_sq = ggml_cont(ctx,
        ggml_view_2d(ctx, mk_full, kv_len_padded, n_tokens,
                     mk_full->nb[1], 0));
    ggml_tensor * mk_swa_f32_sq = mk_swa ? ggml_cont(ctx,
        ggml_view_2d(ctx, mk_swa, swa_len_padded, n_tokens,
                     mk_swa->nb[1], 0)) : nullptr;

    // ── Canvas embed (identical to gemma4_denoise_batch) ──────────────
    // Reuse gemma4_denoise_batch's graph building helpers (they're in the
    // same translation unit, no header needed).
    // We can't directly call the static helper functions from here since they're
    // defined in gemma4_graph.cpp. Since this function is in the same TU, it
    // can call them — but they're lambdas captured in the denoise_batch scope.
    //
    // Solution: duplicate the canvas embed + layer loop here, calling the same
    // module-level helpers (build_gemma4_layer, etc.). These are file-scope
    // functions in this TU, so they're accessible.

    // ── Rebuild canvas embed inline (same as gemma4_denoise_batch lines 1262-1383) ──
    // Canvas embed: rms_norm_noscale (no scale weight) for canvas positions.
    // Prompt positions keep the scaled embedding from the caller.
    // We view prompt and canvas rows from the input `ie` tensor.
    ggml_tensor * cur_embed;
    {
        ggml_tensor * prompt_rows = (P > 0)
            ? ggml_cont(ctx, ggml_view_2d(ctx, ie, w.n_embd, P, ie->nb[1], 0))
            : nullptr;
        ggml_tensor * canvas_pre = ggml_cont(ctx,
            ggml_view_2d(ctx, ie, w.n_embd, C,
                         ie->nb[1], (size_t)P * ie->nb[1]));

        // Canvas: rms_norm (no scale weight — replaces the sqrt(n_embd) mul).
        ggml_tensor * canvas_normed = ggml_rms_norm(ctx, canvas_pre, GEMMA4_EPS);

        // SC MLP injection (only when do_sc).
        if (do_sc && sc_logits_t != nullptr) {
            // softmax(sc_logits_t * sc_temp_inv) → probs [n_vocab, C]
            ggml_tensor * sc_scaled = ggml_scale(ctx, sc_logits_t, sc_temp_inv);
            ggml_tensor * sc_probs  = ggml_soft_max(ctx, sc_scaled);  // [n_vocab, C]

            // SC embed: sc_embT {n_vocab, n_embd} @ sc_probs {n_vocab, C} → {n_embd, C}
            // ggml_mul_mat(A,B) = A^T @ B; A={n_vocab,n_embd}, B={n_vocab,C}
            // → result = {n_embd, C}
            ggml_tensor * sc_embed = ggml_mul_mat(ctx, sc_embT, sc_probs);
            sc_embed = ggml_scale(ctx, sc_embed, sc_use);

            // sc_pre_norm on canvas hidden + SC signal (ref diffusion-gemma.cpp:348-357).
            ggml_tensor * sc_pre_normed = gemma4_rms_norm_mul(ctx, canvas_normed,
                                                               w.sc_pre_norm);
            ggml_tensor * sc_gate_v     = ggml_mul_mat(ctx, w.sc_gate, sc_pre_normed);
            ggml_tensor * sc_up_v       = ggml_mul_mat(ctx, w.sc_up, sc_pre_normed);
            sc_gate_v = ggml_gelu(ctx, sc_gate_v);
            ggml_tensor * sc_inner = ggml_mul(ctx, sc_gate_v, sc_up_v);
            // Down-project and add to sc_embed signal.
            ggml_tensor * sc_down_v = ggml_mul_mat(ctx, w.sc_down, sc_inner);
            sc_embed = ggml_add(ctx, sc_embed, sc_down_v);

            canvas_normed = ggml_add(ctx, canvas_normed, sc_embed);
        }

        if (prompt_rows) {
            cur_embed = ggml_concat(ctx, prompt_rows, canvas_normed, 1);
        } else {
            cur_embed = canvas_normed;
        }
    }

    // Per-layer token embedding injection (same as in gemma4_denoise_batch).
    if (tok_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        ggml_tensor * per_layer_embd = ggml_get_rows(ctx, w.per_layer_tok_embd, tok_ids);
        per_layer_embd = ggml_reshape_2d(ctx, per_layer_embd,
                                         (int64_t)w.n_embd_per_layer * w.n_layer, n_tokens);
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, per_layer_embd);
        // proj shape: {n_embd * n_layer, n_tokens}; stored as [n_embd, n_layer, n_tokens]
        // Injected per-layer inside build_gemma4_layer; pass as external input.
        // (The same approach as in denoise_batch: build_gemma4_layer receives per-layer embd.)
        // NOTE: per_layer injection is applied inside build_gemma4_layer below.
        // Store in a local for use in the layer loop.
        (void)proj;  // injected via build_gemma4_layer (same call signature).
    }

    // ── Layer loop ────────────────────────────────────────────────────
    ggml_tensor * cur = cur_embed;
    for (int il = 0; il < w.n_layer; ++il) {
        ggml_tensor * layer_out = build_gemma4_layer(
            ctx, gf, w, cache, il, cur, pp, tok_ids,
            mk_full, mk_swa, mk_full_f32_sq, mk_swa_f32_sq,
            n_tokens, n_prompt, kv_len_padded, swa_len_padded, swa_size);

        const Gemma4Layer & L = w.layers[il];
        if (P > 0 && C > 0 && L.out_scale && L.enc_out_scale) {
            ggml_tensor * prompt_rows = ggml_cont(ctx,
                ggml_view_2d(ctx, layer_out, w.n_embd, P, layer_out->nb[1], 0));
            ggml_tensor * canvas_rows = ggml_cont(ctx,
                ggml_view_2d(ctx, layer_out, w.n_embd, C,
                             layer_out->nb[1], (size_t)P * layer_out->nb[1]));
            ggml_tensor * correction = ggml_div(ctx, L.enc_out_scale, L.out_scale);
            prompt_rows = ggml_mul(ctx, prompt_rows, correction);
            cur = ggml_concat(ctx, prompt_rows, canvas_rows, 1);
        } else if (P == 0 && L.out_scale) {
            cur = layer_out;
        } else if (C == 0 && L.out_scale && L.enc_out_scale) {
            ggml_tensor * correction = ggml_div(ctx, L.enc_out_scale, L.out_scale);
            cur = ggml_mul(ctx, layer_out, correction);
        } else {
            cur = layer_out;
        }
    }

    // ── Final norm + lm_head + softcap ────────────────────────────────
    cur = gemma4_rms_norm_mul(ctx, cur, w.out_norm, w.norm_eps);
    if (P > 0) {
        cur = ggml_cont(ctx,
            ggml_view_2d(ctx, cur, w.n_embd, C,
                         cur->nb[1], (size_t)P * cur->nb[1]));
    }
    cur = ggml_mul_mat(ctx, w.output, cur);  // [n_vocab, C]
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // ── Allocate ──────────────────────────────────────────────────────
    if (!s_denoise_galloc.galloc) {
        s_denoise_galloc.galloc  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        s_denoise_galloc.n_tokens = n_tokens;
        s_denoise_galloc.n_prompt = n_prompt;
        s_denoise_galloc.do_sc    = do_sc;
        s_denoise_galloc.backend  = backend;
    }
    if (!ggml_gallocr_alloc_graph(s_denoise_galloc.galloc, gf)) {
        std::fprintf(stderr, "gemma4_denoise_batch_dev: gallocr_alloc_graph failed\n");
        s_denoise_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

    // ── Upload non-SC inputs ──────────────────────────────────────────
    ggml_backend_tensor_set(ie, embed, 0, ggml_nbytes(ie));

    std::vector<int32_t> pos((size_t)n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = i;
    ggml_backend_tensor_set(pp, pos.data(), 0, ggml_nbytes(pp));

    if (tok_ids && token_ids) {
        ggml_backend_tensor_set(tok_ids, token_ids, 0, (size_t)n_tokens * sizeof(int32_t));
    }

    // ── SC input: device-to-device copy (no H2D) ─────────────────────
    if (do_sc && sc_logits_t) {
        const size_t sc_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
        cudaError_t err = cudaMemcpy(sc_logits_t->data, sc_dev_in,
                                     sc_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "gemma4_denoise_batch_dev: cudaMemcpy SC D2D failed: %s\n",
                cudaGetErrorString(err));
            ggml_free(ctx);
            return false;
        }
    }

    // ── Attention masks (same as gemma4_denoise_batch) ────────────────
    const int n_swa = w.sliding_window;
    const int canvas_prompt_lo = P - (n_swa > 0 ? n_swa - 1 : 0);
    {
        std::vector<float> mfull((size_t)kv_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const bool q_is_canvas = (q >= P);
            for (int k = 0; k < kv_len_raw; ++k) {
                const bool k_is_canvas = (k >= P);
                bool allow;
                if (q_is_canvas) {
                    allow = true;
                } else {
                    allow = (!k_is_canvas) && (k <= q);
                }
                if (allow) mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    }
    if (mk_swa) {
        std::vector<float> mswa((size_t)swa_len_padded * n_tokens, -INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            const bool q_is_canvas = (q >= P);
            for (int k = 0; k < kv_len_raw; ++k) {
                const bool k_is_canvas = (k >= P);
                bool allow;
                if (q_is_canvas) {
                    allow = k_is_canvas || (k >= canvas_prompt_lo);
                } else {
                    allow = (!k_is_canvas) && (k <= q) &&
                            (n_swa <= 0 || q - k < n_swa);
                }
                if (allow) {
                    const int slot = (swa_size > 0) ? (k % swa_size) : k;
                    if (slot < swa_len_raw) mswa[(size_t)q * swa_len_padded + slot] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(mk_swa, mswa.data(), 0, ggml_nbytes(mk_swa));
    }

    // ── Compute ───────────────────────────────────────────────────────
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_denoise_batch_dev: graph_compute failed\n");
        s_denoise_galloc.free_all();
        ggml_free(ctx);
        return false;
    }

    // ── SC output: device-to-device copy (save cur for next step) ────
    if (sc_dev_out) {
        const size_t logit_bytes = (size_t)w.n_vocab * (size_t)C * sizeof(float);
        cudaError_t err = cudaMemcpy(sc_dev_out, cur->data,
                                     logit_bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr,
                "gemma4_denoise_batch_dev: cudaMemcpy SC out D2D failed: %s\n",
                cudaGetErrorString(err));
            ggml_free(ctx);
            return false;
        }
    }

    // ── GPU sampling kernel ────────────────────────────────────────────
    // Allocate small device output arrays (allocated by caller via diffusion_gemma).
    out_sampled.resize(C);
    out_entropy.resize(C);
    out_argmax.resize(C);

    // Run the kernel on cur->data (device pointer, valid after graph_compute).
    // Results copied to host below.
    int32_t * d_sampled = nullptr;
    float   * d_entropy = nullptr;
    int32_t * d_argmax  = nullptr;
    cudaMalloc(&d_sampled, (size_t)C * sizeof(int32_t));
    cudaMalloc(&d_entropy, (size_t)C * sizeof(float));
    cudaMalloc(&d_argmax,  (size_t)C * sizeof(int32_t));

    dflash::diffusion::diffusion_sample_gpu(
        static_cast<const float *>(cur->data),
        u_dev,
        temp_inv,
        C,
        w.n_vocab,
        d_sampled,
        d_entropy,
        d_argmax,
        /*stream=*/0);

    // Sync and copy tiny results to host (~3 KB total).
    cudaDeviceSynchronize();
    cudaMemcpy(out_sampled.data(), d_sampled, (size_t)C * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_entropy.data(), d_entropy, (size_t)C * sizeof(float),   cudaMemcpyDeviceToHost);
    cudaMemcpy(out_argmax.data(),  d_argmax,  (size_t)C * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_sampled);
    cudaFree(d_entropy);
    cudaFree(d_argmax);

    cache.cur_pos = n_tokens;
    ggml_free(ctx);
    return true;
}
#endif  // DFLASH27B_BACKEND_CUDA_REMOVED_DEAD_CODE

// ── gemma4_project_hidden ───────────────────────────────────────────────
// Runs out_norm + lm_head + softcap + argmax on external hidden states.

bool gemma4_project_hidden(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    const float *           hidden,
    int                     n_tokens,
    std::vector<int32_t> &  out_tokens)
{
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Input: hidden states [n_embd, n_tokens]
    // NOTE: The DFlash draft model already applies its own final RMSNorm,
    // so we skip the target's out_norm and go directly to lm_head.
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(inp);

    // lm_head (skip out_norm — draft already normalized)
    ggml_tensor * cur = ggml_mul_mat(ctx, w.output, inp);  // [n_vocab, n_tokens]

    // Logit softcapping
    if (w.final_logit_softcap > 0.0f) {
        cur = ggml_scale(ctx, cur, 1.0f / w.final_logit_softcap);
        cur = ggml_tanh(ctx, cur);
        cur = ggml_scale(ctx, cur, w.final_logit_softcap);
    }

    // Argmax
    cur = ggml_argmax(ctx, cur);  // [n_tokens]
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Allocate
    static ggml_gallocr_t galloc_proj = nullptr;
    if (!galloc_proj) galloc_proj = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc_proj, gf)) {
        std::fprintf(stderr, "gemma4_project_hidden: gallocr_alloc_graph failed\n");
        ggml_free(ctx);
        return false;
    }

    // Set input
    ggml_backend_tensor_set(inp, hidden, 0, sizeof(float) * (size_t)n_tokens * w.n_embd);

    // Compute
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemma4_project_hidden: graph_compute failed\n");
        ggml_free(ctx);
        return false;
    }

    // Read result
    out_tokens.resize(n_tokens);
    ggml_backend_tensor_get(cur, out_tokens.data(), 0, sizeof(int32_t) * n_tokens);

    ggml_free(ctx);
    return true;
}

// ── gemma4_prefill_bsa ──────────────────────────────────────────────────
// Full-prompt BSA prefill: processes all tokens at once, layer-by-layer.
// SWA layers use flash_prefill_forward_bf16 (block-sparse attention).
// Full-attention layers use ggml_flash_attn_ext (dense, exact).
// After all layers: fills KV cache for subsequent decode.

// Persistent buffer helper (same pattern as Qwen3).
struct G4PersBuf {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *         t   = nullptr;
};

static bool g4_make_pers(ggml_backend_t backend, ggml_type type, int n_dim,
                          const int64_t * dims, G4PersBuf & out) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4 + 1024;
    ip.no_alloc   = true;
    ip.mem_buffer = nullptr;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;
    if      (n_dim == 1) out.t = ggml_new_tensor_1d(out.ctx, type, dims[0]);
    else if (n_dim == 2) out.t = ggml_new_tensor_2d(out.ctx, type, dims[0], dims[1]);
    else if (n_dim == 3) out.t = ggml_new_tensor_3d(out.ctx, type, dims[0], dims[1], dims[2]);
    else return false;
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    return out.buf != nullptr;
}

static void g4_free_pers(G4PersBuf & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.t = nullptr;
}

static int g4_bsa_chunk_size() {
    if (const char * e = std::getenv("DFLASH_G4_BSA_CHUNK")) {
        int v = std::atoi(e);
        if (v >= 512) return v;
    }
    return 4096;
}

bool gemma4_prefill_bsa(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     S,
    std::vector<float> &    out_logits)
{
    const int hidden  = w.n_embd;
    const int n_layer = w.n_layer;
    const int n_head  = w.n_head;
    const float eps   = w.norm_eps;

    // Determine max dimensions across all layers for buffer allocation.
    int max_q_dim = 0, max_kv_dim = 0;
    for (int il = 0; il < n_layer; ++il) {
        const int D  = gemma4_head_dim(w, il);
        const int Hk = gemma4_n_head_kv(w, il);
        max_q_dim  = std::max(max_q_dim, D * n_head);
        max_kv_dim = std::max(max_kv_dim, D * Hk);
    }

    // Use BF16 only for sm_80+ (native BF16 tensor cores). Volta/Turing
    // use F16 with F16 WMMA kernels; other arches use F16 with ggml FA fallback.
    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_SM80_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    // Allocate persistent buffers.
    G4PersBuf hidden_buf{}, Q_buf{}, K_buf{}, V_buf{}, attn_out_buf{};
    int64_t d_h[]  = {(int64_t)hidden, (int64_t)S};
    int64_t d_q[]  = {(int64_t)max_q_dim, (int64_t)S};
    int64_t d_kv[] = {(int64_t)max_kv_dim, (int64_t)S};

    auto cleanup_all = [&]() {
        g4_free_pers(hidden_buf);
        g4_free_pers(Q_buf);
        g4_free_pers(K_buf);
        g4_free_pers(V_buf);
        g4_free_pers(attn_out_buf);
    };

    if (!g4_make_pers(backend, GGML_TYPE_F32, 2, d_h, hidden_buf) ||
        !g4_make_pers(backend, half_type, 2, d_q, Q_buf) ||
        !g4_make_pers(backend, half_type, 2, d_kv, K_buf) ||
        !g4_make_pers(backend, half_type, 2, d_kv, V_buf) ||
        !g4_make_pers(backend, half_type, 2, d_q, attn_out_buf)) {
        std::fprintf(stderr, "[gemma4-bsa] persistent buffer alloc failed\n");
        cleanup_all();
        return false;
    }

    // Upload embedded+scaled input to hidden_buf.
    ggml_backend_tensor_set(hidden_buf.t, embed, 0, (size_t)hidden * S * sizeof(float));

    // Precompute per-layer embeddings on GPU if the model has them.
    // per_layer_all: [n_embd_per_layer, S, n_layer] — computed once, sliced per layer.
    G4PersBuf per_layer_buf{};
    if (token_ids && w.per_layer_tok_embd && w.per_layer_model_proj && w.n_embd_per_layer > 0) {
        const int D_pl = w.n_embd_per_layer;
        const int L_pl = n_layer;
        int64_t d_pl[] = {(int64_t)D_pl, (int64_t)S, (int64_t)L_pl};
        if (!g4_make_pers(backend, GGML_TYPE_F32, 3, d_pl, per_layer_buf)) {
            std::fprintf(stderr, "[gemma4-bsa] per-layer buf alloc failed\n");
            cleanup_all();
            return false;
        }

        // Build a graph to compute per-layer embeddings.
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
        ggml_set_input(tok);
        ggml_tensor * h_in = ggml_view_2d(ctx, hidden_buf.t, hidden, S,
                                            hidden * sizeof(float), 0);

        // get_rows(per_layer_tok_embd, tok) → [D_pl*L_pl, S]
        ggml_tensor * inp_pl = ggml_get_rows(ctx, w.per_layer_tok_embd, tok);
        inp_pl = ggml_reshape_3d(ctx, inp_pl, D_pl, L_pl, S);
        inp_pl = ggml_scale(ctx, inp_pl, std::sqrt((float)D_pl));

        // Project main embedding: mul_mat(per_layer_model_proj, h_in)
        ggml_tensor * proj = ggml_mul_mat(ctx, w.per_layer_model_proj, h_in);
        proj = ggml_scale(ctx, proj, 1.0f / std::sqrt((float)hidden));
        proj = ggml_reshape_3d(ctx, proj, D_pl, L_pl, S);

        // RMS norm on projection
        proj = ggml_rms_norm(ctx, rms_norm_input_f32(ctx, proj), eps);
        ggml_tensor * norm_w = ggml_reshape_2d(ctx, w.per_layer_proj_norm, D_pl, L_pl);
        proj = ggml_mul(ctx, proj, norm_w);

        // Add + scale
        ggml_tensor * pl_all = ggml_add(ctx, proj, inp_pl);
        pl_all = ggml_scale(ctx, pl_all, 1.0f / std::sqrt(2.0f));

        // Permute to [D_pl, S, L_pl] and copy to persistent buffer
        pl_all = ggml_cont(ctx, ggml_permute(ctx, pl_all, 0, 2, 1, 3));
        ggml_tensor * cpy = ggml_cpy(ctx, pl_all, per_layer_buf.t);
        ggml_set_output(cpy);
        ggml_build_forward_expand(gf, cpy);

        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(ga, gf)) {
            std::fprintf(stderr, "[gemma4-bsa] per-layer graph alloc failed\n");
            ggml_gallocr_free(ga); ggml_free(ctx);
            g4_free_pers(per_layer_buf); cleanup_all();
            return false;
        }
        ggml_backend_tensor_set(tok, token_ids, 0, (size_t)S * sizeof(int32_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "gemma4_prefill_bsa: per-layer embed graph_compute failed\n");
            ggml_gallocr_free(ga); ggml_free(ctx);
            g4_free_pers(per_layer_buf); cleanup_all();
            return false;
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }

    // Gallocr for per-layer graphs (reused).
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const int CHUNK = g4_bsa_chunk_size();

    // FlashPrefill config for SWA layers.
    const int block_size = 128;
    const int swa_window_blocks = (w.sliding_window + block_size - 1) / block_size;
    flashprefill::FlashPrefillConfig swa_cfg;
    swa_cfg.block_size     = block_size;
    swa_cfg.attention_sink = 0;
    swa_cfg.window         = swa_window_blocks;
    swa_cfg.last_n_full    = 0;
    swa_cfg.alpha          = 2.0f;  // > 1.0 disables dynamic block selection

    // Scale for attention: Gemma4 uses 1.0 (Q/K already RMS-normed per head).
    const float kq_scale = 1.0f;

    // ── Per-layer loop ──
    for (int il = 0; il < n_layer; ++il) {
        const Gemma4Layer & L = w.layers[il];
        const bool is_swa    = gemma4_is_swa_layer(w, il);
        const bool has_kv    = gemma4_has_kv(w, il);
        const int D          = gemma4_head_dim(w, il);
        const int Hk         = gemma4_n_head_kv(w, il);
        const int q_dim      = D * n_head;
        const int kv_dim     = D * Hk;

        // ── Graph A (chunked): pre_norm + Q/K/V proj + norms + RoPE → persistent bufs ──
        const float rope_base = is_swa ? w.rope_freq_base_swa : w.rope_freq_base_full;
        ggml_tensor * freq_factors_ref = is_swa ? nullptr :
            (L.rope_freqs ? L.rope_freqs : w.rope_freqs_global);

        for (int cs = 0; cs < S; cs += CHUNK) {
            const int cl = std::min(CHUNK, S - cs);

            ggml_init_params ipA{};
            ipA.mem_size = ggml_tensor_overhead() * 64
                           + ggml_graph_overhead_custom(512, false)
                           + 128 * 1024;
            ipA.no_alloc = true;
            ggml_context * gA = ggml_init(ipA);
            if (!gA) { std::fprintf(stderr, "[gemma4-bsa] graph A init failed\n"); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf); return false; }
            ggml_cgraph * gfA = ggml_new_graph_custom(gA, 512, false);

            // View into hidden_buf for this chunk.
            const size_t h_esz = sizeof(float);
            ggml_tensor * h_view = ggml_view_2d(gA, hidden_buf.t,
                                                hidden, cl,
                                                hidden * h_esz,
                                                (size_t)cs * hidden * h_esz);

            // Positions for RoPE.
            ggml_tensor * pos_t = ggml_new_tensor_1d(gA, GGML_TYPE_I32, cl);
            ggml_set_input(pos_t);

            // Pre-attn norm.
            ggml_tensor * h_norm = ggml_rms_norm(gA, rms_norm_input_f32(gA, h_view), eps);
            h_norm = ggml_mul(gA, h_norm, L.attn_norm);

            // Q projection + norm + RoPE.
            ggml_tensor * Q = ggml_mul_mat(gA, L.wq, h_norm);
            Q = ggml_reshape_3d(gA, Q, D, n_head, cl);
            if (L.q_norm) {
                Q = gemma4_rms_norm_mul(gA, Q, L.q_norm, eps);
            }
            Q = ggml_rope_ext(gA, Q, pos_t, freq_factors_ref, D,
                              GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f,
                              0.0f, 1.0f, 32.0f, 1.0f);
            // Reshape Q to [q_dim, cl] and copy to Q_buf.
            Q = ggml_reshape_2d(gA, Q, q_dim, cl);

            const size_t q_esz = ggml_type_size(half_type);
            ggml_tensor * Q_dst = ggml_view_2d(gA, Q_buf.t, q_dim, cl,
                                               q_esz * max_q_dim,
                                               (size_t)cs * q_esz * max_q_dim);
            ggml_build_forward_expand(gfA, ggml_cpy(gA, Q, Q_dst));

            if (has_kv) {
                // K projection + norm + RoPE.
                ggml_tensor * K = ggml_mul_mat(gA, L.wk, h_norm);
                K = ggml_reshape_3d(gA, K, D, Hk, cl);
                if (L.k_norm) {
                    K = gemma4_rms_norm_mul(gA, K, L.k_norm, eps);
                }
                K = ggml_rope_ext(gA, K, pos_t, freq_factors_ref, D,
                                  GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f,
                                  0.0f, 1.0f, 32.0f, 1.0f);
                K = ggml_reshape_2d(gA, K, kv_dim, cl);

                // V projection + RMSNorm (Gemma4 specific).
                ggml_tensor * V = L.wv ? ggml_mul_mat(gA, L.wv, h_norm)
                                       : ggml_mul_mat(gA, L.wk, h_norm);
                V = ggml_reshape_3d(gA, V, D, Hk, cl);
                V = ggml_rms_norm(gA, rms_norm_input_f32(gA, V), eps);
                V = ggml_reshape_2d(gA, V, kv_dim, cl);

                const size_t kv_esz = ggml_type_size(half_type);
                ggml_tensor * K_dst = ggml_view_2d(gA, K_buf.t, kv_dim, cl,
                                                   kv_esz * max_kv_dim,
                                                   (size_t)cs * kv_esz * max_kv_dim);
                ggml_tensor * V_dst = ggml_view_2d(gA, V_buf.t, kv_dim, cl,
                                                   kv_esz * max_kv_dim,
                                                   (size_t)cs * kv_esz * max_kv_dim);
                ggml_build_forward_expand(gfA, ggml_cpy(gA, K, K_dst));
                ggml_build_forward_expand(gfA, ggml_cpy(gA, V, V_dst));

                // Write to KV cache for subsequent decode.
                // K is [kv_dim, cl] = [D*Hk, cl]. Cache is [D, cache_len, Hk] F16.
                // Reshape K to [D, Hk, cl] → permute to [D, cl, Hk] → copy into cache slot.
                ggml_tensor * cache_k_t = cache.k[il];
                ggml_tensor * cache_v_t = cache.v[il];
                if (cache_k_t) {
                    const int cache_len_il = (int)cache_k_t->ne[1];
                    const int ring_pos = is_swa ? (cs % cache_len_il) : cs;

                    // Lambda to copy a sub-range of K/V into cache.
                    auto write_kv_range = [&](int src_off, int dst_ring, int n) {
                        if (n <= 0) return;
                        // K[src_off:src_off+n] → cache_k[dst_ring:dst_ring+n]
                        ggml_tensor * Ks = (src_off == 0 && n == cl) ? K
                            : ggml_view_2d(gA, K, kv_dim, n,
                                           K->nb[1], (size_t)src_off * K->nb[1]);
                        ggml_tensor * K3 = ggml_reshape_3d(gA, Ks, D, Hk, n);
                        ggml_tensor * Kp = ggml_cont(gA, ggml_permute(gA, K3, 0, 2, 1, 3));
                        ggml_tensor * k_slot = ggml_view_3d(gA, cache_k_t,
                            D, n, Hk,
                            cache_k_t->nb[1], cache_k_t->nb[2],
                            cache_k_t->nb[1] * (size_t)dst_ring);
                        ggml_build_forward_expand(gfA, ggml_cpy(gA, Kp, k_slot));

                        ggml_tensor * Vs = (src_off == 0 && n == cl) ? V
                            : ggml_view_2d(gA, V, kv_dim, n,
                                           V->nb[1], (size_t)src_off * V->nb[1]);
                        ggml_tensor * V3 = ggml_reshape_3d(gA, Vs, D, Hk, n);
                        ggml_tensor * Vp = ggml_cont(gA, ggml_permute(gA, V3, 0, 2, 1, 3));
                        ggml_tensor * v_slot = ggml_view_3d(gA, cache_v_t,
                            D, n, Hk,
                            cache_v_t->nb[1], cache_v_t->nb[2],
                            cache_v_t->nb[1] * (size_t)dst_ring);
                        ggml_build_forward_expand(gfA, ggml_cpy(gA, Vp, v_slot));
                    };

                    if (!is_swa && ring_pos + cl > cache_len_il) {
                        // Full-attention layer: positions exceed cache — truncate.
                        const int n_fit = cache_len_il - ring_pos;
                        if (n_fit > 0) write_kv_range(0, ring_pos, n_fit);
                    } else if (is_swa && ring_pos + cl > cache_len_il) {
                        // SWA ring wrap — split into two writes.
                        const int first_n = cache_len_il - ring_pos;
                        write_kv_range(0, ring_pos, first_n);
                        write_kv_range(first_n, 0, cl - first_n);
                    } else {
                        write_kv_range(0, ring_pos, cl);
                    }
                }
            }

            if (!ggml_gallocr_alloc_graph(galloc, gfA)) {
                std::fprintf(stderr, "[gemma4-bsa] graph A alloc failed layer=%d cs=%d\n", il, cs);
                ggml_free(gA); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
                return false;
            }

            // Set positions.
            std::vector<int32_t> pos((size_t)cl);
            for (int i = 0; i < cl; ++i) pos[i] = cs + i;
            ggml_backend_tensor_set(pos_t, pos.data(), 0, (size_t)cl * sizeof(int32_t));

            ggml_backend_graph_compute(backend, gfA);
            ggml_backend_synchronize(backend);
            ggml_free(gA);
        }

        // ── Attention ──
        // KV-sharing: layers without has_kv don't overwrite K_buf/V_buf, so they
        // still hold the source layer's data (kv_source[il] < il for sharing layers).

        bool used_bsa = false;
        if (is_swa && D == 128) {
            // ── BSA sparse-FA for SWA layers (head_dim=128) ──
            const bool q_contiguous = (q_dim == max_q_dim);
            const bool kv_contiguous = (kv_dim == max_kv_dim);

            int rc;
            if (q_contiguous && kv_contiguous) {
                rc = flashprefill::flash_prefill_forward(
                    backend, Q_buf.t->data, K_buf.t->data,
                    V_buf.t->data, attn_out_buf.t->data,
                    1, S, n_head, Hk, D, kq_scale, half_type, swa_cfg);
            } else {
                // Non-contiguous: allocate temporary packed buffers.
                G4PersBuf Q_pack{}, K_pack{}, V_pack{}, O_pack{};
                int64_t dq[] = {(int64_t)q_dim, (int64_t)S};
                int64_t dk[] = {(int64_t)kv_dim, (int64_t)S};
                if (!g4_make_pers(backend, half_type, 2, dq, Q_pack) ||
                    !g4_make_pers(backend, half_type, 2, dk, K_pack) ||
                    !g4_make_pers(backend, half_type, 2, dk, V_pack) ||
                    !g4_make_pers(backend, half_type, 2, dq, O_pack)) {
                    std::fprintf(stderr, "[gemma4-bsa] pack buf alloc failed\n");
                    g4_free_pers(Q_pack); g4_free_pers(K_pack);
                    g4_free_pers(V_pack); g4_free_pers(O_pack);
                    ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
                    return false;
                }

                const size_t esz = ggml_type_size(half_type);
                cudaMemcpy2D(Q_pack.t->data, q_dim * esz,
                             Q_buf.t->data, max_q_dim * esz,
                             q_dim * esz, S, cudaMemcpyDeviceToDevice);
                cudaMemcpy2D(K_pack.t->data, kv_dim * esz,
                             K_buf.t->data, max_kv_dim * esz,
                             kv_dim * esz, S, cudaMemcpyDeviceToDevice);
                cudaMemcpy2D(V_pack.t->data, kv_dim * esz,
                             V_buf.t->data, max_kv_dim * esz,
                             kv_dim * esz, S, cudaMemcpyDeviceToDevice);

                rc = flashprefill::flash_prefill_forward(
                    backend, Q_pack.t->data, K_pack.t->data,
                    V_pack.t->data, O_pack.t->data,
                    1, S, n_head, Hk, D, kq_scale, half_type, swa_cfg);

                // Copy packed output back to strided attn_out_buf.
                cudaMemcpy2D(attn_out_buf.t->data, max_q_dim * esz,
                             O_pack.t->data, q_dim * esz,
                             q_dim * esz, S, cudaMemcpyDeviceToDevice);

                g4_free_pers(Q_pack); g4_free_pers(K_pack);
                g4_free_pers(V_pack); g4_free_pers(O_pack);
            }

            if (rc != 0) {
                std::fprintf(stderr, "[gemma4-bsa] flash_prefill failed layer=%d rc=%d\n", il, rc);
                ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
                return false;
            }
            cudaDeviceSynchronize();
            used_bsa = true;
        }

        if (!used_bsa) {
            // Build a ggml graph for dense causal attention for this layer.
            // Process the full sequence in one FA call (or chunked if too large).
            for (int cs = 0; cs < S; cs += CHUNK) {
                const int cl = std::min(CHUNK, S - cs);
                const int kv_len = cs + cl;  // attend to all positions up to current

                ggml_init_params ipFA{};
                ipFA.mem_size = ggml_tensor_overhead() * 32
                               + ggml_graph_overhead_custom(64, false)
                               + 128 * 1024;
                ipFA.no_alloc = true;
                ggml_context * gFA = ggml_init(ipFA);
                ggml_cgraph * gfFA = ggml_new_graph_custom(gFA, 64, false);

                const size_t esz = ggml_type_size(half_type);

                // Q view: [D, n_head, cl] from Q_buf
                ggml_tensor * Qfa = ggml_view_3d(gFA, Q_buf.t,
                    D, n_head, cl,
                    esz * D, esz * max_q_dim,
                    (size_t)cs * esz * max_q_dim);

                // K view: [D, Hk, kv_len] from K_buf
                ggml_tensor * Kfa = ggml_view_3d(gFA, K_buf.t,
                    D, Hk, kv_len,
                    esz * D, esz * max_kv_dim,
                    0);

                // V view: [D, Hk, kv_len] from V_buf
                ggml_tensor * Vfa = ggml_view_3d(gFA, V_buf.t,
                    D, Hk, kv_len,
                    esz * D, esz * max_kv_dim,
                    0);

                // Causal mask: [kv_len_padded, cl]
                const int kv_len_padded = (kv_len + 255) & ~255;
                ggml_tensor * mask = ggml_new_tensor_4d(gFA, GGML_TYPE_F32,
                    kv_len_padded, cl, 1, 1);
                ggml_set_input(mask);
                ggml_tensor * mask_f16 = ggml_cast(gFA, mask, GGML_TYPE_F16);

                ggml_tensor * attn = ggml_flash_attn_ext(gFA, Qfa, Kfa, Vfa, mask_f16,
                                                          kq_scale, 0.0f, 0.0f);

                // Write output to attn_out_buf: [q_dim, cl] at offset cs.
                attn = ggml_reshape_2d(gFA, attn, q_dim, cl);
                ggml_tensor * O_dst = ggml_view_2d(gFA, attn_out_buf.t, q_dim, cl,
                                                   esz * max_q_dim,
                                                   (size_t)cs * esz * max_q_dim);
                ggml_tensor * cpy_op = ggml_cpy(gFA, attn, O_dst);
                ggml_set_output(cpy_op);
                ggml_build_forward_expand(gfFA, cpy_op);

                if (!ggml_gallocr_alloc_graph(galloc, gfFA)) {
                    std::fprintf(stderr, "[gemma4-bsa] dense FA alloc failed layer=%d\n", il);
                    ggml_free(gFA); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
                    return false;
                }

                // Fill causal mask.
                std::vector<float> m((size_t)kv_len_padded * cl, -INFINITY);
                for (int q = 0; q < cl; ++q) {
                    const int abs_q = cs + q;
                    for (int k = 0; k <= abs_q && k < kv_len; ++k) {
                        m[(size_t)q * kv_len_padded + k] = 0.0f;
                    }
                }
                ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));

                ggml_backend_graph_compute(backend, gfFA);
                ggml_backend_synchronize(backend);
                ggml_free(gFA);
            }
        }

        // ── Graph B (chunked): o_proj + post_norm + residual + FFN + per_layer + scale ──
        for (int cs = 0; cs < S; cs += CHUNK) {
            const int cl = std::min(CHUNK, S - cs);

            ggml_init_params ipB{};
            ipB.mem_size = ggml_tensor_overhead() * 128
                          + ggml_graph_overhead_custom(1024, false)
                          + 2 * 1024 * 1024;
            ipB.no_alloc = true;
            ggml_context * gB = ggml_init(ipB);
            if (!gB) { std::fprintf(stderr, "[gemma4-bsa] graph B init failed\n"); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf); return false; }
            ggml_cgraph * gfB = ggml_new_graph_custom(gB, 1024, false);

            const size_t h_esz = sizeof(float);
            const size_t a_esz = ggml_type_size(half_type);

            // Hidden state for this chunk (residual input).
            ggml_tensor * h_in = ggml_view_2d(gB, hidden_buf.t, hidden, cl,
                                               hidden * h_esz,
                                               (size_t)cs * hidden * h_esz);

            // Attention output for this chunk.
            ggml_tensor * a_in = ggml_view_2d(gB, attn_out_buf.t, q_dim, cl,
                                              a_esz * max_q_dim,
                                              (size_t)cs * a_esz * max_q_dim);

            // o_proj: [q_dim, n_embd] × [q_dim, cl] → [n_embd, cl]
            ggml_tensor * cur = ggml_mul_mat(gB, L.wo, a_in);

            // Post-attn norm.
            if (L.attn_post_norm) {
                cur = gemma4_rms_norm_mul(gB, cur, L.attn_post_norm, eps);
            }

            // Residual after attention.
            ggml_tensor * attn_res = ggml_add(gB, cur, h_in);

            // FFN.
            const bool is_moe = (L.ffn_gate_inp != nullptr && il >= w.n_layer_dense_lead);
            ggml_tensor * ffn_out;
            if (is_moe) {
                ggml_tensor * normed = gemma4_rms_norm_mul(gB, attn_res, L.ffn_norm, eps);
                ffn_out = build_gemma4_moe_block(gB, attn_res, normed, w, L, cl);
            } else {
                cur = gemma4_rms_norm_mul(gB, attn_res, L.ffn_norm, eps);
                ffn_out = build_gemma4_dense_ffn(gB, cur, L);
            }

            // FFN post-norm.
            if (L.ffn_post_norm) {
                ffn_out = gemma4_rms_norm_mul(gB, ffn_out, L.ffn_post_norm, eps);
            }

            // Residual after FFN.
            cur = ggml_add(gB, ffn_out, attn_res);

            // Per-layer embedding injection.
            if (per_layer_buf.t && L.per_layer_inp_gate && L.per_layer_proj) {
                const int D_pl = w.n_embd_per_layer;
                // Slice per_layer_buf [D_pl, S, n_layer] → [D_pl, cl] for this layer+chunk
                ggml_tensor * pl_slice = ggml_view_2d(gB, per_layer_buf.t,
                    D_pl, cl,
                    D_pl * sizeof(float),
                    ((size_t)il * S + cs) * D_pl * sizeof(float));

                ggml_tensor * gate = ggml_mul_mat(gB, L.per_layer_inp_gate, cur);
                gate = ggml_gelu(gB, gate);
                gate = ggml_mul(gB, gate, pl_slice);
                ggml_tensor * proj = ggml_mul_mat(gB, L.per_layer_proj, gate);
                if (L.per_layer_post_norm) {
                    proj = gemma4_rms_norm_mul(gB, proj, L.per_layer_post_norm, eps);
                }
                cur = ggml_add(gB, cur, proj);
            }

            // Output scale.
            if (L.out_scale) {
                cur = ggml_mul(gB, cur, L.out_scale);
            }

            // Write back to hidden_buf.
            ggml_tensor * h_dst = ggml_view_2d(gB, hidden_buf.t, hidden, cl,
                                                hidden * h_esz,
                                                (size_t)cs * hidden * h_esz);
            ggml_tensor * cpy = ggml_cpy(gB, cur, h_dst);
            ggml_set_output(cpy);
            ggml_build_forward_expand(gfB, cpy);

            if (!ggml_gallocr_alloc_graph(galloc, gfB)) {
                std::fprintf(stderr, "[gemma4-bsa] graph B alloc failed layer=%d cs=%d\n", il, cs);
                ggml_free(gB); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
                return false;
            }

            ggml_backend_graph_compute(backend, gfB);
            ggml_backend_synchronize(backend);
            ggml_free(gB);
        }

        // Feature capture: write hidden states at capture layers to target_feat ring.
        if (cache.target_feat) {
            int cap_idx = -1;
            for (int k = 0; k < cache.n_capture_layers; k++) {
                if (cache.capture_layer_ids[k] == il) { cap_idx = k; break; }
            }
            if (cap_idx >= 0) {
                const int cap = cache.target_feat_cap;
                const size_t feat_col_stride = cache.target_feat->nb[1];
                const size_t feat_elt = ggml_element_size(cache.target_feat);
                // Write last min(S, cap) positions into the ring buffer.
                const int write_start = (S > cap) ? (S - cap) : 0;
                const int write_n = std::min(S, cap);
                for (int cs = write_start; cs < write_start + write_n; cs += CHUNK) {
                    const int cl = std::min(CHUNK, write_start + write_n - cs);
                    const int slot_start = cs % cap;

                    ggml_init_params ipC{};
                    ipC.mem_size = ggml_tensor_overhead() * 8
                                  + ggml_graph_overhead() + 64 * 1024;
                    ipC.no_alloc = true;
                    ggml_context * gC = ggml_init(ipC);
                    ggml_cgraph * gfC = ggml_new_graph(gC);

                    ggml_tensor * h_src = ggml_view_2d(gC, hidden_buf.t,
                        hidden, cl, hidden * sizeof(float),
                        (size_t)cs * hidden * sizeof(float));

                    const size_t offset = (size_t)slot_start * feat_col_stride
                                        + (size_t)cap_idx * hidden * feat_elt;
                    ggml_tensor * feat_dst = ggml_view_2d(gC, cache.target_feat,
                        hidden, cl, feat_col_stride, offset);

                    ggml_build_forward_expand(gfC, ggml_cpy(gC, h_src, feat_dst));

                    if (ggml_gallocr_alloc_graph(galloc, gfC)) {
                        ggml_backend_graph_compute(backend, gfC);
                    }
                    ggml_free(gC);
                }
            }
        }
    }  // end layer loop

    // ── Fill KV cache for decode ──
    // KV cache was not populated during the BSA layer loop because K_buf/V_buf
    // get overwritten each layer. We re-project K/V for each KV-owning layer
    // from the hidden states that were stored before each layer's attention.
    //
    // However, we don't have the pre-norm hidden states anymore (hidden_buf has
    // the final output). The correct approach is to write KV cache during the
    // layer loop. Since this is a v1 implementation, we use the fallback:
    // after BSA prefill returns, the caller (do_prefill) will run a single
    // gemma4_step with the last chunk to populate the cache for decode.
    //
    // TODO: Move KV cache writes into the layer loop for zero-redundancy.
    // For now, the caller handles cache population by running a trailing
    // gemma4_step over the last swa_size tokens.

    // ── Final norm + logits (last token only) ──
    {
        ggml_init_params ipF{};
        ipF.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 1024 * 1024;
        ipF.no_alloc = true;
        ggml_context * gF = ggml_init(ipF);
        ggml_cgraph * gfF = ggml_new_graph(gF);

        // View last token of hidden_buf.
        ggml_tensor * h_last = ggml_view_2d(gF, hidden_buf.t, hidden, 1,
                                             hidden * sizeof(float),
                                             (size_t)(S - 1) * hidden * sizeof(float));

        // Final RMSNorm.
        ggml_tensor * normed = gemma4_rms_norm_mul(gF, h_last, w.out_norm, eps);

        // lm_head.
        ggml_tensor * logits = ggml_mul_mat(gF, w.output, normed);

        // Softcapping.
        if (w.final_logit_softcap > 0.0f) {
            logits = ggml_scale(gF, logits, 1.0f / w.final_logit_softcap);
            logits = ggml_tanh(gF, logits);
            logits = ggml_scale(gF, logits, w.final_logit_softcap);
        }

        ggml_set_output(logits);
        ggml_build_forward_expand(gfF, logits);

        if (!ggml_gallocr_alloc_graph(galloc, gfF)) {
            std::fprintf(stderr, "[gemma4-bsa] final graph alloc failed\n");
            ggml_free(gF); ggml_gallocr_free(galloc); cleanup_all(); g4_free_pers(per_layer_buf);
            return false;
        }

        ggml_backend_graph_compute(backend, gfF);
        ggml_backend_synchronize(backend);

        out_logits.resize((size_t)w.n_vocab);
        ggml_backend_tensor_get(logits, out_logits.data(), 0,
                                 out_logits.size() * sizeof(float));
        ggml_free(gF);
    }

    // Update cache position.
    cache.cur_pos = S;

    ggml_gallocr_free(galloc);
    cleanup_all();
    g4_free_pers(per_layer_buf);
    return true;
}

}  // namespace dflash::common
