// Qwen3MoeDFlashTarget — DFlashTarget adapter for Qwen3-MoE models.

#include "qwen3moe_dflash_target.h"
#include "qwen3moe_verify_graph.h"
#include "common/dflash_feature_ring.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace dflash::common {

// embed_tokens helper (mirrors the static function in qwen3moe_backend.cpp).
static bool embed_tokens_impl(ggml_backend_t backend,
                               ggml_tensor *  tok_embd,
                               const int32_t * ids,
                               int             n,
                               int             hidden,
                               float *         embed_buf) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
    ip.no_alloc = true;
    ggml_context * ectx = ggml_init(ip);
    if (!ectx) return false;

    ggml_tensor * id_t = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, n);
    ggml_set_input(id_t);
    ggml_tensor * emb  = ggml_get_rows(ectx, tok_embd, id_t);
    ggml_tensor * out_t = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, n);
    ggml_tensor * cpy  = ggml_cpy(ectx, emb, out_t);
    ggml_set_output(cpy);
    ggml_cgraph * gf = ggml_new_graph(ectx);
    ggml_build_forward_expand(gf, cpy);

    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(ectx);
        return false;
    }
    ggml_backend_tensor_set(id_t, ids, 0, sizeof(int32_t) * n);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(cpy, embed_buf, 0, sizeof(float) * (size_t)hidden * n);
    ggml_free(ectx);
    return true;
}

// ── Constructor ───────────────────────────────────────────────────────────────

Qwen3MoeDFlashTarget::Qwen3MoeDFlashTarget(
        Qwen3MoeWeights    & w,
        Qwen3MoeCache      & cache,
        ggml_backend_t       backend,
        DraftFeatureMirror & feature_mirror,
        int                  fa_window,
        int                  kq_stride_pad,
        int                  n_capture_layers,
        int                  mask_token_id)
    : w_(w), cache_(cache), backend_(backend),
      feature_mirror_(feature_mirror),
      fa_window_(fa_window), kq_stride_pad_(kq_stride_pad),
      mask_token_id_(mask_token_id)
{
    (void)fa_window_; (void)kq_stride_pad_;

    // Evenly-spaced capture layers.
    // For 48 layers, n=5: step=11 → [1, 12, 23, 34, 45]
    const int n  = std::max(1, n_capture_layers);
    const int nl = w_.n_layer;
    capture_ids_.resize(n);
    if (n == 1) {
        capture_ids_[0] = nl / 2;
    } else {
        const int step = std::max(1, (nl - 2) / (n - 1));
        for (int k = 0; k < n; ++k) {
            capture_ids_[k] = 1 + k * step;
        }
    }
}

// ── Destructor ────────────────────────────────────────────────────────────────

Qwen3MoeDFlashTarget::~Qwen3MoeDFlashTarget() {
    step_graph_destroy(proj_sg_);
    free_qwen3moe_snapshot(verify_snap_);
}

// ── verify_batch ─────────────────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax)
{
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;
    const int kv_len = base_pos + n_tokens;

    // 1. Embed input tokens.
    std::vector<float> embed_buf((size_t)n_tokens * hidden);
    if (!embed_tokens_impl(backend_, w_.tok_embd,
                           tokens.data(), n_tokens, hidden, embed_buf.data())) {
        std::fprintf(stderr, "[qwen3moe] verify_batch: embed failed (n=%d)\n", n_tokens);
        return false;
    }

    // 2. Build verify graph (all-token logits + capture outputs).
    Qwen3MoeVerifyGraphResult vg;
    if (!build_qwen3moe_verify_graph(vg, w_, cache_, backend_,
                                      n_tokens, base_pos, capture_ids_)) {
        std::fprintf(stderr, "[qwen3moe] verify_batch: graph build failed (base=%d n=%d)\n",
                     base_pos, n_tokens);
        return false;
    }

    // 3. Fill inputs.
    ggml_backend_tensor_set(vg.inp, embed_buf.data(), 0,
                            sizeof(float) * embed_buf.size());

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = base_pos + i;
    ggml_backend_tensor_set(vg.positions, pos.data(), 0,
                            sizeof(int32_t) * n_tokens);

    // Causal mask: row q attends to columns [0, base_pos+q].
    std::vector<ggml_fp16_t> mask_data((size_t)kv_len * n_tokens);
    const ggml_fp16_t zero_h    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf_h = ggml_fp32_to_fp16(-INFINITY);
    for (int row = 0; row < n_tokens; ++row) {
        const int last_visible = base_pos + row;
        for (int col = 0; col < kv_len; ++col) {
            mask_data[(size_t)row * kv_len + col] =
                (col <= last_visible) ? zero_h : neg_inf_h;
        }
    }
    ggml_backend_tensor_set(vg.attn_mask, mask_data.data(), 0,
                            sizeof(ggml_fp16_t) * mask_data.size());

    // 4. Compute.
    auto st = ggml_backend_graph_compute(backend_, vg.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[qwen3moe] verify_batch: compute failed (status=%d base=%d n=%d)\n",
                     (int)st, base_pos, n_tokens);
        ggml_free(vg.ctx);
        return false;
    }

    // 5. Read argmax.
    std::vector<int32_t> argmax_buf(n_tokens);
    ggml_backend_tensor_get(vg.argmax, argmax_buf.data(), 0,
                            sizeof(int32_t) * n_tokens);
    last_tok = argmax_buf[n_tokens - 1];
    if (all_argmax) {
        *all_argmax = std::move(argmax_buf);
    }

    // 6. Copy capture activations into the draft feature mirror.
    // vg.captures[k] are GPU-resident (gallocr-allocated on backend_'s device).
    // Pass them directly to copy_capture_slice_to_draft_ring which uses
    // cudaMemcpyPeerAsync internally — no host roundtrip.
    const int target_device = feature_mirror_.target_device;
    const int n_cap = (int)capture_ids_.size();
    if (feature_mirror_.target_feat) {
        for (int k = 0; k < n_cap; ++k) {
            if (!vg.captures[k]) continue;
            if (!copy_capture_slice_to_draft_ring(feature_mirror_, k,
                                                   vg.captures[k],
                                                   /*src_device=*/target_device,
                                                   /*chunk_start=*/0,
                                                   /*start_pos=*/base_pos,
                                                   n_tokens)) {
                std::fprintf(stderr, "[qwen3moe] verify_batch: capture copy failed (k=%d)\n", k);
                ggml_free(vg.ctx);
                return false;
            }
        }
    }

    cache_.cur_pos = base_pos + n_tokens;
    ggml_free(vg.ctx);
    return true;
}

// ── snapshot_kv ──────────────────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::snapshot_kv() {
    verify_snap_.cur_pos = cache_.cur_pos;

    // Allocate snapshot tensors lazily (first call).
    if (verify_snap_.k_snap.empty()) {
        const int n_layer = cache_.n_layer;
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
        ip.no_alloc = true;
        verify_snap_.ctx = ggml_init(ip);
        if (!verify_snap_.ctx) return false;

        verify_snap_.k_snap.resize(n_layer, nullptr);
        verify_snap_.v_snap.resize(n_layer, nullptr);
        for (int il = 0; il < n_layer; ++il) {
            verify_snap_.k_snap[il] = ggml_dup_tensor(verify_snap_.ctx, cache_.k[il]);
            verify_snap_.v_snap[il] = ggml_dup_tensor(verify_snap_.ctx, cache_.v[il]);
        }
        verify_snap_.buf = ggml_backend_alloc_ctx_tensors(verify_snap_.ctx, backend_);
        if (!verify_snap_.buf) {
            ggml_free(verify_snap_.ctx);
            verify_snap_.ctx = nullptr;
            verify_snap_.k_snap.clear();
            verify_snap_.v_snap.clear();
            return false;
        }
    }

    for (int il = 0; il < cache_.n_layer; ++il) {
        ggml_backend_tensor_copy(cache_.k[il], verify_snap_.k_snap[il]);
        ggml_backend_tensor_copy(cache_.v[il], verify_snap_.v_snap[il]);
    }
    return true;
}

// ── restore_kv ───────────────────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::restore_kv() {
    if (verify_snap_.k_snap.empty()) return false;

    for (int il = 0; il < cache_.n_layer; ++il) {
        ggml_backend_tensor_copy(verify_snap_.k_snap[il], cache_.k[il]);
        ggml_backend_tensor_copy(verify_snap_.v_snap[il], cache_.v[il]);
    }
    cache_.cur_pos = verify_snap_.cur_pos;
    return true;
}

// ── is_eos ────────────────────────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::is_eos(int token) const {
    return token == 151643 || token == 151645;
}

// ── embed_tokens ─────────────────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                         float * out) const {
    return embed_tokens_impl(backend_, w_.tok_embd, tokens, n, w_.n_embd, out);
}

// ── project_hidden_to_tokens ─────────────────────────────────────────────────

bool Qwen3MoeDFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (n_tokens <= 0) return false;

    // Rebuild projection graph if n_tokens changed.
    if (proj_sg_n_ != n_tokens) {
        step_graph_destroy(proj_sg_);
        proj_sg_n_ = 0;

        const int hdim = w_.n_embd;

        ggml_init_params ip{};
        ip.mem_size = 64 * 1024 * 1024;
        ip.no_alloc = true;
        proj_sg_.ctx = ggml_init(ip);
        if (!proj_sg_.ctx) return false;

        proj_sg_.hidden_input = ggml_new_tensor_2d(proj_sg_.ctx, GGML_TYPE_F32,
                                                    hdim, n_tokens);
        ggml_set_name(proj_sg_.hidden_input, "proj_hidden");
        ggml_set_input(proj_sg_.hidden_input);

        proj_sg_.gf = ggml_new_graph_custom(proj_sg_.ctx, 1024, false);

        // BUG1 FIX: do NOT apply rms_norm + out_norm here.
        // build_draft_step already applies the draft's final rms_norm (w.out_norm)
        // before emitting hidden_states, so the input is already-normalized.
        // Applying the target's out_norm a second time is double normalization.
        proj_sg_.logits = ggml_mul_mat(proj_sg_.ctx, w_.output, proj_sg_.hidden_input);
        ggml_set_name(proj_sg_.logits, "proj_logits");
        ggml_set_output(proj_sg_.logits);

        proj_sg_.argmax_tokens = ggml_argmax(proj_sg_.ctx, proj_sg_.logits);
        ggml_set_name(proj_sg_.argmax_tokens, "proj_argmax");
        ggml_set_output(proj_sg_.argmax_tokens);
        ggml_build_forward_expand(proj_sg_.gf, proj_sg_.argmax_tokens);

        if (!proj_sg_.alloc) {
            proj_sg_.alloc = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend_));
        }
        if (!ggml_gallocr_alloc_graph(proj_sg_.alloc, proj_sg_.gf)) {
            step_graph_destroy(proj_sg_);
            return false;
        }
        proj_sg_n_ = n_tokens;
    }

    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * w_.n_embd);

    auto st = ggml_backend_graph_compute(backend_, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    tokens_out.resize(n_tokens);
    ggml_backend_tensor_get(proj_sg_.argmax_tokens, tokens_out.data(), 0,
                            sizeof(int32_t) * n_tokens);
    return true;
}

} // namespace dflash::common
