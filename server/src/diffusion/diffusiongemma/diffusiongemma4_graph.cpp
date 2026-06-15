// DiffusionGemma4 forward graph builders.
//
// Contains the diffusion-specific forward functions extracted from gemma4_graph.cpp:
//   - gemma4_denoise_batch: bidirectional forward over [prompt | canvas]
//   - gemma4_prefill_prompt_for_denoise: KV cache populate for prompt
//   - gemma4_denoise_canvas: canvas-only step using cached prompt KV
//
// Shared helpers (gemma4_rms_norm_mul, build_gemma4_layer, etc.) are defined
// in gemma4_graph.cpp and forward-declared in gemma4_internal.h.

#include "gemma4_internal.h"
#include "common/ggml_graph_precision.h"
#include "common/gpu_runtime_compat.h"
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
// Populates the KV cache with prompt tokens.  Cold prefill uses kv_start=0;
// restore prefill appends a prompt delta at kv_start=snapshot.cur_pos.  After
// success, cache.cur_pos == kv_start + P.
//
// Builds a graph over P tokens with:
//   - causal mask (same as AR step)
//   - no_cache = false → writes K/V to cache[kv_start..kv_start+P-1]
//   - per-layer embeddings computed for P tokens
// Does NOT compute logits (we only need the KV side-effect).
bool gemma4_prefill_prompt_for_denoise(
    ggml_backend_t          backend,
    const Gemma4Weights &   w,
    Gemma4Cache &           cache,
    const float *           embed,
    const int32_t *         token_ids,
    int                     P,
    int                     kv_start)
{
    if (kv_start < 0) return false;
    if (P <= 0) { cache.cur_pos = kv_start; return true; }
    if (cache.cur_pos != kv_start) {
        std::fprintf(stderr,
            "gemma4_prefill_prompt_for_denoise: cache.cur_pos=%d != kv_start=%d\n",
            cache.cur_pos, kv_start);
        return false;
    }

    const int total   = kv_start + P;
    const int min_ctx = (total + 255) & ~255;
    if (cache.max_ctx < min_ctx) {
        std::fprintf(stderr,
            "gemma4_prefill_prompt_for_denoise: max_ctx %d < %d for total=%d\n",
            cache.max_ctx, min_ctx, total);
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

    // RoPE positions [kv_start..kv_start+P-1]
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
    const int kv_len_padded  = (total + 255) & ~255;
    const int swa_size       = cache.swa_size;
    const int swa_len_raw    = swa_size > 0 ? std::min(total, swa_size) : total;
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
                                                      kv_start, P,
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
    for (int i = 0; i < P; ++i) pos[i] = kv_start + i;
    ggml_backend_tensor_set(pp, pos.data(), 0, (size_t)P * sizeof(int32_t));
    ggml_backend_tensor_set(kvi_full, pos.data(), 0, (size_t)P * sizeof(int32_t));
    if (swa_size > 0) {
        std::vector<int32_t> ring(P);
        for (int i = 0; i < P; ++i) ring[i] = (kv_start + i) % swa_size;
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
            const int abs_q = kv_start + q;
            for (int k = 0; k <= abs_q; ++k) {
                mfull[(size_t)q * kv_len_padded + k] = 0.0f;
            }
        }
        ggml_backend_tensor_set(mk_full, mfull.data(), 0, ggml_nbytes(mk_full));
    }
    // SWA causal mask (ring-buffer indexed)
    {
        const int W = w.sliding_window;
        std::vector<float> mswa((size_t)swa_len_padded * P, -INFINITY);
        for (int q = 0; q < P; ++q) {
            const int abs_q = kv_start + q;
            const int win_lo = (W > 0) ? std::max(0, abs_q - W + 1) : 0;
            for (int k = win_lo; k <= abs_q; ++k) {
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

    cache.cur_pos = total;
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

}  // namespace dflash::common
