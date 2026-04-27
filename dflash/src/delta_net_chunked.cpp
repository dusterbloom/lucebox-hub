// Chunked gated delta-net: sequential token processing for GDA (non-KDA).
//
// The llama.cpp reference chunked factorization (delta-net-base.cpp) has
// four interrelated bugs in the non-KDA path that we could not fix because
// ggml's element-wise mul on CUDA silently fails for mask tensors, making
// it impossible to zero the diagonal of the D matrix.
//
// Instead, we process tokens sequentially within the ggml graph (unrolled
// loop), matching the autoregressive algorithm exactly.  This avoids the D
// matrix and triangular solve entirely.  For the small token counts used
// in speculative decoding verify (4-16 tokens), the overhead is minimal.
//
// The KDA path is kept as-is from the reference (also untested — needs
// separate verification).

#include "delta_net_chunked.h"

#include <cmath>
#include <cstdint>

namespace dflash27b {

static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

DeltaNetChunkedResult build_delta_net_chunked(
        ggml_context * ctx0,
        ggml_cgraph  * gf,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s,
        ggml_tensor  * ssm_inter_cap) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && q->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf((float)S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    g = ggml_permute(ctx0, g, 0, 2, 1, 3);
    b = ggml_permute(ctx0, b, 0, 2, 1, 3);

    if (!kda) {
        // ── Sequential GDA path (unrolled per-token loop) ────────────
        //
        // For each token t (mirrors the validated autoregressive reference
        // in llama.cpp's build_delta_net_autoregressive):
        //   1. S = exp(g_t) * S               (apply decay)
        //   2. delta = (v_t - S^T @ k_t) * beta_t  (compute delta)
        //   3. S = S + k_t ⊗ delta             (rank-1 update)
        //   4. o_t = S^T @ q_t                 (output from updated state)
        //
        // To match the autoregressive reference exactly we use
        //   ggml_mul + ggml_sum_rows  (instead of ggml_mul_mat)
        //   ggml_repeat + ggml_mul    (instead of ggml_out_prod)
        // and we make per-token slices of g and b CONTIGUOUS before passing
        // them to ggml_exp / ggml_mul: the CUDA backends for these ops
        // require fully-contiguous src for their fast paths and may
        // silently fall back to a stride-ignoring kernel otherwise, which
        // produces head-dependent drift that compounds across tokens.

        ggml_tensor * o = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
            S_v, n_tokens, H_v, n_seqs);

        s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);

        for (int64_t t = 0; t < n_tokens; t++) {
            // Per-token slices. Make q/k/v contiguous before they enter the
            // broadcasted mul/repeat path: the CUDA bin_bcast kernels are
            // sensitive to the stride pattern of permuted views.
            ggml_tensor * q_t = ggml_cont(ctx0, ggml_view_4d(ctx0, q, S_k, 1, H_k, n_seqs,
                q->nb[1], q->nb[2], q->nb[3], t * q->nb[1]));
            ggml_tensor * k_t = ggml_cont(ctx0, ggml_view_4d(ctx0, k, S_k, 1, H_k, n_seqs,
                k->nb[1], k->nb[2], k->nb[3], t * k->nb[1]));
            ggml_tensor * v_t = ggml_cont(ctx0, ggml_view_4d(ctx0, v, S_v, 1, H_v, n_seqs,
                v->nb[1], v->nb[2], v->nb[3], t * v->nb[1]));

            // g and b slices: force contiguous so ggml_exp (CUDA unary
            // requires fully-contiguous src) gets a valid input and so
            // beta multiplication doesn't pick up strided reads.
            ggml_tensor * g_t = ggml_cont(ctx0, ggml_view_4d(ctx0, g,
                g->ne[0], 1, H_v, n_seqs,
                g->nb[1], g->nb[2], g->nb[3], t * g->nb[1]));
            ggml_tensor * b_t = ggml_cont(ctx0, ggml_view_4d(ctx0, b,
                1, 1, H_v, n_seqs,
                b->nb[1], b->nb[2], b->nb[3], t * b->nb[1]));

            // 1. S = exp(g_t) * S   (g_t broadcast: per-head scalar)
            s = ggml_mul(ctx0, s, ggml_exp(ctx0, g_t));

            // 2. delta = (v_t - S^T @ k_t) * beta_t
            //    Use mul_mat instead of elementwise mul + sum_rows. This
            //    directly computes the per-head dot product with the state's
            //    transposed storage layout and avoids the broadcast kernels
            //    that were drifting on CUDA.
            ggml_tensor * sk = ggml_mul_mat(ctx0, s, k_t);          // [S_v, 1, H, ns]
            ggml_tensor * d  = ggml_sub(ctx0, v_t, sk);
            d                = ggml_mul(ctx0, d, b_t);               // scale by beta

            // 3. S += k_t ⊗ delta   (rank-1 update)
            //    Use out_prod to build the outer product directly from the
            //    contiguous token slice and the per-token delta.
            ggml_tensor * kd = ggml_out_prod(ctx0, k_t, d);          // [S_v, S_v, H, ns]
            s                 = ggml_add(ctx0, s, kd);

            // Capture per-token state for fast-rollback spec-decode.
            // f32 → f16 cpy into a separate cache buffer; the in-loop `s`
            // is NOT read back from this buffer.
            if (ssm_inter_cap && gf) {
                ggml_tensor * slot_t = ggml_view_4d(ctx0, ssm_inter_cap,
                    ssm_inter_cap->ne[0],
                    ssm_inter_cap->ne[1],
                    ssm_inter_cap->ne[2],
                    1,
                    ssm_inter_cap->nb[1],
                    ssm_inter_cap->nb[2],
                    ssm_inter_cap->nb[3],
                    (size_t)t * ssm_inter_cap->nb[3]);
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, s, slot_t));
            }

            // 4. o_t = S^T @ q_t
            ggml_tensor * o_t = ggml_mul_mat(ctx0, s, q_t);         // [S_v, 1, H, ns]

            // Write into output tensor at position t (slice o[*, t, *, *]).
            ggml_tensor * o_view = ggml_view_4d(ctx0, o,
                S_v, 1, H_v, n_seqs,
                o->nb[1], o->nb[2], o->nb[3],
                (size_t)t * o->nb[1]);
            if (gf) {
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, o_t, o_view));
            } else {
                o = ggml_set_inplace(ctx0, o, o_t,
                    o->nb[1], o->nb[2], o->nb[3], t * o->nb[1]);
            }
        }

        // o: [S_v, n_tokens, H_v, n_seqs] → [S_v, H_v, n_tokens, n_seqs]
        o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));

        DeltaNetChunkedResult r;
        r.output    = o;
        r.new_state = s;
        r.sl_mask   = nullptr;
        return r;
    }

    // ── KDA chunked path (from delta-net-base.cpp, unchanged) ────────
    const int CS = 16;

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (int)((n_tokens + pad) / CS);

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));

    const int64_t CHB = n_chunks * H_k * n_seqs;

    ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);
    ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);

    g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);

    ggml_tensor * decay_mask;
    decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
    decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
    decay_mask = ggml_exp(ctx0, decay_mask);

    decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

    ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
    ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
    ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

    ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
    ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);

    ggml_tensor * kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
    ggml_tensor * kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);

    kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
    kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));

    ggml_tensor * attn;
    attn = kb;

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);

    attn = ggml_neg(ctx0, attn);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);

    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);

    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1], g_cs->nb[2], g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));

    g_last = ggml_cont(ctx0, g_last);

    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));

    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);

    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk);
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk);
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk);
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk);
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk);

        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s);

        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);

        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);

        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);

        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new);

        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
    }

    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            sizeof(float),
            sizeof(float) * S_v,
            sizeof(float) * S_v * CS * n_chunks,
            0);
    o = ggml_cont     (ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);

    DeltaNetChunkedResult r;
    r.output    = o;
    r.new_state = s;
    r.sl_mask   = nullptr;
    return r;
}

} // namespace dflash27b
