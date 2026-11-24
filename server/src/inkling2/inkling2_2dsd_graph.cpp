// Mixed-depth static-bucketed 2DSD target runner (Prelude / Fused Multi-Pass Core /
// Fused Multi-Depth Coda / Commit) and 60-plane multi-depth Prefill for Inkling 2 Mini.

#include "inkling2_2dsd_graph.h"
#include "inkling2_2dsd_draft.h"
#include "inkling2_kvflash.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace luce::common {
namespace {

static inline void set_op_i32(ggml_tensor * t, int idx, int32_t v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}
static inline void set_op_f32(ggml_tensor * t, int idx, float v) {
    std::memcpy(&t->op_params[idx], &v, sizeof(v));
}

static ggml_tensor * row_view_2d(ggml_context * c, ggml_tensor * src, int64_t row_off, int64_t n_rows) {
    return ggml_view_2d(c, src, n_rows, src->ne[1], src->nb[1], (size_t) row_off * src->nb[0]);
}

static ggml_tensor * rms_norm_w(ggml_context * c, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// Reused Inkling2 fused op builders (matching inkling2_graph.cpp)
static ggml_tensor * make_clip_qr_rel_table(
        ggml_context * c,
        ggml_tensor * q_raw, ggml_tensor * q_clip_w,
        ggml_tensor * r_raw, ggml_tensor * r_clip_w,
        ggml_tensor * pos_buf, ggml_tensor * rel_bias, ggml_tensor * sink,
        int head_dim, int d_rel, int n_head,
        float q_thresh, float r_thresh, float eps, int log_floor, float log_alpha,
        ggml_tensor ** out_q, ggml_tensor ** out_rt) {
    const int64_t rel_extent = rel_bias->ne[1];
    const int64_t T = q_raw->ne[1];
    const int64_t q_elems  = (int64_t) head_dim * n_head * T;
    const int64_t rt_elems = (rel_extent + 1) * n_head * T;
    ggml_tensor * buf = ggml_new_tensor_1d(c, GGML_TYPE_F32, q_elems + rt_elems);
    buf->op = GGML_OP_INKLING2_OP;
    buf->src[0] = q_raw;
    buf->src[1] = q_clip_w;
    buf->src[2] = r_raw;
    buf->src[3] = r_clip_w;
    buf->src[4] = pos_buf;
    buf->src[5] = rel_bias;
    buf->src[6] = sink;
    set_op_i32(buf, 0, 31);
    set_op_i32(buf, 1, head_dim);
    set_op_i32(buf, 2, d_rel);
    set_op_i32(buf, 3, n_head);
    set_op_f32(buf, 4, q_thresh);
    set_op_f32(buf, 5, r_thresh);
    set_op_f32(buf, 6, eps);
    set_op_i32(buf, 7, log_floor);
    set_op_f32(buf, 8, log_alpha);

    *out_q = ggml_view_3d(c, buf, head_dim, n_head, T,
                          (size_t) head_dim * sizeof(float),
                          (size_t) head_dim * n_head * sizeof(float), 0);
    *out_rt = ggml_view_3d(c, buf, rel_extent + 1, n_head, T,
                           (size_t) (rel_extent + 1) * sizeof(float),
                           (size_t) (rel_extent + 1) * n_head * sizeof(float),
                           (size_t) q_elems * sizeof(float));
    return buf;
}

static ggml_tensor * make_fused_router_q8_topk(
        ggml_context * c, ggml_tensor * w_q8, ggml_tensor * cur, ggml_tensor * bias,
        ggml_tensor * flags, int top_k, float route_scale, float eps, int target_pct) {
    const int64_t T = cur->ne[1];
    ggml_tensor * logits = ggml_mul_mat(c, w_q8, cur);
    ggml_tensor * res = ggml_new_tensor_3d(c, GGML_TYPE_I32, top_k, T, target_pct > 0 ? 3 : 2);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = logits;
    res->src[1] = nullptr;
    res->src[2] = bias;
    res->src[3] = flags;
    set_op_i32(res, 0, 24);
    set_op_i32(res, 1, top_k);
    set_op_f32(res, 2, route_scale);
    set_op_f32(res, 3, eps);
    set_op_i32(res, 4, target_pct);
    return res;
}

static ggml_tensor * make_loop_ctrl_post_flagged(
        ggml_context * c,
        ggml_tensor * cur, ggml_tensor * msg, ggml_tensor * gf_out,
        ggml_tensor * norm_w, ggml_tensor * flags, float eps) {
    ggml_tensor * res = ggml_inkling2_loop_ctrl_post(c, cur, msg, gf_out, norm_w, eps);
    res->src[4] = flags;
    return res;
}

static ggml_tensor * make_tree_sconv(
        ggml_context * c,
        ggml_tensor * x, ggml_tensor * w_static,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta, ggml_tensor * dyn_taps, ggml_tensor * addend,
        int chan_offset, int head_dim_dyn, int m_stride) {
    ggml_tensor * res = ggml_new_tensor_2d(c, GGML_TYPE_F32, x->ne[0], x->ne[1]);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x;
    res->src[1] = w_static;
    res->src[2] = prefix_conv;
    res->src[3] = tree_conv;
    res->src[4] = meta;
    res->src[5] = dyn_taps;
    res->src[6] = addend;
    set_op_i32(res, 0, 33);
    set_op_i32(res, 1, chan_offset);
    set_op_i32(res, 2, head_dim_dyn);
    set_op_i32(res, 3, m_stride);
    return res;
}

static ggml_tensor * make_tree_sconv_kv_write(
        ggml_context * c,
        ggml_tensor * x, ggml_tensor * w_static,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * meta, ggml_tensor * dyn_taps,
        ggml_tensor * tree_kv, ggml_tensor * prev_dep,
        int chan_offset, int head_dim, int n_head_kv, int is_v,
        int m_stride, float threshold, float eps) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = x;
    res->src[1] = w_static;
    res->src[2] = prefix_conv;
    res->src[3] = tree_conv;
    res->src[4] = meta;
    res->src[5] = dyn_taps;
    res->src[6] = tree_kv;
    res->src[7] = prev_dep;
    set_op_i32(res, 0, 34);
    set_op_i32(res, 1, chan_offset);
    set_op_i32(res, 2, head_dim);
    set_op_i32(res, 3, n_head_kv);
    set_op_i32(res, 4, is_v);
    set_op_i32(res, 5, m_stride);
    set_op_f32(res, 6, threshold);
    set_op_f32(res, 7, eps);
    return res;
}

static ggml_tensor * make_tree_attn_rel(
        ggml_context * c,
        ggml_tensor * q,
        ggml_tensor * prefix_k, ggml_tensor * prefix_v,
        ggml_tensor * tree_k,   ggml_tensor * tree_v,
        ggml_tensor * rel_table, ggml_tensor * meta, ggml_tensor * prefix_len_buf,
        ggml_tensor * gate, ggml_tensor * dep_kv,
        int swa_window, int rel_extent, int m_stride, int path_w,
        int block_size = 0, int partial_out = 0, int exclude_sink = 0,
        int shared_kv_window = 0, int shared_tail_cap = 0) {
    const int64_t Hq = q->ne[1];
    const int64_t T  = q->ne[2];
    ggml_tensor * res = partial_out
        ? ggml_new_tensor_2d(c, GGML_TYPE_F32, (q->ne[0] + 2) * Hq, T)
        : ggml_new_tensor_2d(c, GGML_TYPE_F32, q->ne[0] * Hq, T);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = q;
    res->src[1] = prefix_k;
    res->src[2] = prefix_v;
    res->src[3] = tree_k;
    res->src[4] = tree_v;
    res->src[5] = rel_table;
    res->src[6] = meta;
    res->src[7] = prefix_len_buf;
    res->src[8] = gate;
    res->src[9] = dep_kv;
    set_op_i32(res, 0, 35);
    set_op_i32(res, 1, swa_window);
    set_op_i32(res, 2, rel_extent);
    set_op_i32(res, 3, m_stride);
    set_op_i32(res, 4, path_w);
    set_op_i32(res, 5, block_size);
    set_op_i32(res, 6, partial_out);
    set_op_i32(res, 7, exclude_sink);
    set_op_i32(res, 8, shared_kv_window);
    set_op_i32(res, 9, shared_tail_cap);
    return res;
}

static ggml_tensor * make_tree_commit(
        ggml_context * c,
        ggml_tensor * prefix_k, ggml_tensor * prefix_v,
        ggml_tensor * tree_k,   ggml_tensor * tree_v,
        ggml_tensor * prefix_conv, ggml_tensor * tree_conv,
        ggml_tensor * commit_slots, ggml_tensor * commit_meta,
        int swa_window,
        ggml_tensor * block_table = nullptr,
        int block_size = 0,
        int shared_kv_window = 0,
        int shared_tail_cap = 0) {
    ggml_tensor * res = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    res->op = GGML_OP_INKLING2_OP;
    res->src[0] = prefix_k;
    res->src[1] = prefix_v;
    res->src[2] = tree_k;
    res->src[3] = tree_v;
    res->src[4] = prefix_conv;
    res->src[5] = tree_conv;
    res->src[6] = commit_slots;
    res->src[7] = commit_meta;
    res->src[8] = block_table;
    set_op_i32(res, 0, 36);
    set_op_i32(res, 1, swa_window);
    set_op_i32(res, 2, block_size);
    set_op_i32(res, 3, shared_kv_window);
    set_op_i32(res, 4, shared_tail_cap);
    return res;
}

static int bucket_rows(int n) {
    static constexpr int kBuckets[] = { 1, 2, 4, 8, 12, 16, 24, 32, 48, 64 };
    for (int b : kBuckets) if (n <= b) return b;
    return ((n + 31) / 32) * 32;
}

static int bucket_path(int w) {
    static constexpr int kBuckets[] = { 8, 16, 32, 64 };
    for (int b : kBuckets) if (w <= b) return b;
    return ((w + 31) / 32) * 32;
}

// Create a zero-copy 3D/2D view of a single plane inside a 4D/3D tree.prefix_* tensor (for prefill).
static ggml_tensor * plane_view_kv(ggml_context * c, ggml_tensor * kv4d, int plane) {
    return ggml_view_3d(c, kv4d, kv4d->ne[0], kv4d->ne[1], kv4d->ne[2],
                        kv4d->nb[1], kv4d->nb[2], (size_t) plane * kv4d->nb[3]);
}
static ggml_tensor * plane_view_conv(ggml_context * c, ggml_tensor * conv3d, int plane) {
    return ggml_view_2d(c, conv3d, conv3d->ne[0], conv3d->ne[1],
                        conv3d->nb[1], (size_t) plane * conv3d->nb[2]);
}

// Build one physical layer inside a 2DSD phase graph (Prelude / Core / Coda).
static ggml_tensor * build_tree_layer(
        ggml_context * c,
        ggml_cgraph * gf,
        const Inkling2Weights & w,
        Inkling2TreeCache & tc,
        Inkling2TwoDSDPhaseGraph & g,
        int ip,
        ggml_tensor * cur,
        std::vector<ggml_tensor *> & blocks,
        const std::vector<ggml_tensor *> * bank_blocks,
        ggml_tensor * bank_out,
        bool has_later,
        bool use_split,
        int row_b,
        int path_b,
        int kf_block_size,
        bool use_tier15) {
    const auto & L = w.layers[ip];
    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;
    const int m_stride = 11 + path_b;
    const int swa = L.is_swa ? w.swa_window : 0;

    ggml_tensor * inpSA = cur;

    // Cross-pass messages for core layers when any row has pass > 0
    ggml_tensor * messages = nullptr;
    const Inkling2LoopLayer * loop = nullptr;
    if (has_later && bank_out != nullptr) {
        loop = &w.loop_layers[ip - w.core_start];
        messages = ggml_inkling2_attn_res_mix(
            c, bank_out, loop->msg_query, nullptr,
            (bank_blocks && !bank_blocks->empty()) ? bank_blocks->data() : nullptr,
            bank_blocks ? (int) bank_blocks->size() : 0,
            w.rms_eps); // [H, 4, T]
    }
    auto get_msg = [&](int ctrl) -> ggml_tensor * {
        return ggml_view_2d(c, messages, w.n_embd, row_b,
                            messages->nb[2], (size_t) ctrl * messages->nb[1]);
    };

    // Attention input mix + optional branchless per-row loop controller (FLAG_LATER)
    ggml_tensor * inputs[3] = { cur, cur, cur };
    const bool same_qkv_input = L.is_swa && (messages == nullptr);

    if (messages == nullptr) {
        if (L.attn_res_query == nullptr) {
            inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else if (L.is_swa) {
            inputs[0] = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, L.attn_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            inputs[1] = inputs[2] = inputs[0];
        } else {
            ggml_tensor * mixed3 = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, L.attn_qkv_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_view_2d(c, mixed3, w.n_embd, row_b,
                                         mixed3->nb[2], (size_t) j * mixed3->nb[1]);
            }
        }
    } else {
        ggml_tensor * pre_qkv = cur;
        if (L.attn_res_query) {
            ggml_tensor * mixed = ggml_inkling2_attn_res_mix(
                c, cur, L.attn_res_query, nullptr,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            pre_qkv = L.is_swa ? mixed : ggml_permute(c, mixed, 0, 2, 1, 3);
        }
        ggml_tensor * msg_qkv_v = ggml_view_3d(c, messages, w.n_embd, 3, row_b, messages->nb[1], messages->nb[2], 0);
        ggml_tensor * msg_qkv   = ggml_permute(c, msg_qkv_v, 0, 2, 1, 3); // [H, T, 3]
        ggml_tensor * nw_qkv    = L.is_swa ? L.attn_norm : L.attn_qkv_norm;
        ggml_tensor * cpre      = ggml_inkling2_loop_ctrl_pre(c, pre_qkv, msg_qkv, w.rms_eps);
        ggml_tensor * gate      = ggml_mul_mat(c, loop->ctrl_down_qkv, cpre);
        ggml_tensor * gf_o      = ggml_mul_mat(c, loop->ctrl_gf_qkv, gate);
        ggml_tensor * qkv_in    = make_loop_ctrl_post_flagged(c, pre_qkv, msg_qkv, gf_o, nw_qkv, g.inp_flags, w.rms_eps);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_view_2d(c, qkv_in, w.n_embd, row_b, qkv_in->nb[1], (size_t) j * qkv_in->nb[2]);
        }
    }

    // Fused attention projections (Q|K|V|Gate|R|Sink)
    ggml_tensor * Q_raw = nullptr, * K_raw = nullptr, * V_raw = nullptr;
    ggml_tensor * Gate  = nullptr, * R_raw = nullptr, * Sink  = nullptr;
    ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;

    if (same_qkv_input && L.wqkvgrs) {
        ggml_tensor * proj = ggml_mul_mat(c, L.wqkvgrs, inputs[0]);
        Q_raw = row_view_2d(c, proj, 0, q_sz);
        Gate  = row_view_2d(c, proj, q_sz, q_sz);
        R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
        Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
        K_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq, kv_sz);
        V_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
    } else {
        if (L.wqgrs) {
            ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
            Q_raw = row_view_2d(c, proj, 0, q_sz);
            Gate  = row_view_2d(c, proj, q_sz, q_sz);
            R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
            Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
        } else {
            Q_raw = ggml_mul_mat(c, L.wq, inputs[0]);
            Gate  = ggml_mul_mat(c, L.wqkv_gate, inputs[0]);
            R_raw = ggml_mul_mat(c, L.attn_rel, inputs[0]);
            Sink  = ggml_mul_mat(c, L.attn_sink_proj, inputs[0]);
        }
        if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
            ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
            K_raw = row_view_2d(c, kd, 0, kv_sz);
            k_dyn = row_view_2d(c, kd, kv_sz, (int64_t) w.sconv_kernel * Hk);
            ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
            V_raw = row_view_2d(c, vd, 0, kv_sz);
            v_dyn = row_view_2d(c, vd, kv_sz, (int64_t) w.sconv_kernel * Hk);
        } else {
            K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
            V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
            if (!L.is_swa) {
                k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
            }
        }
    }

    ggml_tensor * Q = nullptr, * rt = nullptr;
    make_clip_qr_rel_table(
        c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
        L.is_swa ? nullptr : g.inp_pos, L.attn_rel_bias, Sink,
        D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
        L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
        &Q, &rt);

    // Tree K/V ShortConv + KV write (chain dep_k -> dep_v so tree_attn needs only 1 dep src!)
    ggml_tensor * dep_k = make_tree_sconv_kv_write(
        c, K_raw, L.attn_k_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, k_dyn,
        tc.tree_k[ip], nullptr, 0, D, Hk, 0, m_stride, w.qk_clip, w.rms_eps);
    ggml_tensor * dep_kv = make_tree_sconv_kv_write(
        c, V_raw, L.attn_v_conv, tc.prefix_conv[ip], tc.tree_conv[ip], g.inp_meta, v_dyn,
        tc.tree_v[ip], dep_k, kv_sz, D, Hk, 1, m_stride, w.qk_clip, w.rms_eps);

    const bool shared_global = (!L.is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
    ggml_tensor * attn_out = nullptr;
    if (!L.is_swa && kf_block_size > 0 && use_tier15 &&
        ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip] && g.inp_prefix_len_r) {
        // Dual-GPU 3-Tier Split-KV: Tier 1.5 Strix Halo attends warm prefix blocks ONLY (tree_k=nullptr!),
        // Tier 0 R9700 attends hot prefix blocks + W-tail rings + in-flight tree_k/v, merged via split_combine!
        ggml_tensor * part_r = make_tree_attn_rel(
            c, Q, tc.remote_prefix_k[ip], tc.remote_prefix_v[ip], nullptr, nullptr,
            rt, g.inp_meta, g.inp_prefix_len_r, nullptr, nullptr, 0, w.rel_extent, m_stride, path_b,
            /*block_size=*/kf_block_size, /*partial_out=*/1, /*exclude_sink=*/0,
            /*shared_kv_window=*/0, /*shared_tail_cap=*/0);
        g.expert_nodes.push_back(part_r);
        ggml_tensor * part_l = make_tree_attn_rel(
            c, Q, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
            rt, g.inp_meta, g.inp_prefix_len, nullptr, dep_kv, 0, w.rel_extent, m_stride, path_b,
            /*block_size=*/kf_block_size, /*partial_out=*/1, /*exclude_sink=*/1,
            shared_global ? tc.shared_kv_window : 0,
            shared_global ? tc.shared_tail_cap : 0);
        attn_out = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
    } else {
        attn_out = make_tree_attn_rel(
            c, Q, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
            rt, g.inp_meta, g.inp_prefix_len, Gate, dep_kv, swa, w.rel_extent, m_stride, path_b,
            /*block_size=*/(!L.is_swa && kf_block_size > 0) ? kf_block_size : 0,
            /*partial_out=*/0, /*exclude_sink=*/0,
            shared_global ? tc.shared_kv_window : 0,
            shared_global ? tc.shared_tail_cap : 0);
    }

    // Output proj + tree attn_post ShortConv
    cur = ggml_mul_mat(c, L.wo, attn_out);
    if (ip % w.attn_res_block == 0) {
        cur = make_tree_sconv(c, cur, L.attn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip],
                              g.inp_meta, nullptr, nullptr, 2 * kv_sz, 0, m_stride);
        blocks.push_back(inpSA);
    } else {
        cur = make_tree_sconv(c, cur, L.attn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip],
                              g.inp_meta, nullptr, inpSA, 2 * kv_sz, 0, m_stride);
    }
    ggml_tensor * ffn_inp = cur;

    // FFN AttnRes mix + optional flagged loop controller
    if (messages == nullptr) {
        cur = ggml_inkling2_attn_res_mix(
            c, cur, L.ffn_res_query, L.ffn_norm,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
    } else {
        cur = ggml_inkling2_attn_res_mix(
            c, cur, L.ffn_res_query, nullptr,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
        ggml_tensor * mf   = get_msg(Inkling2LoopLayer::CTRL_FFN);
        ggml_tensor * cd_w = loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
            ? loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
            : loop->ctrl_down[Inkling2LoopLayer::CTRL_FFN];
        ggml_tensor * gate = ggml_mul_mat(c, cd_w, ggml_inkling2_loop_ctrl_pre(c, cur, mf, w.rms_eps));
        ggml_tensor * gf_o = ggml_mul_mat(c, loop->ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gate);
        cur = make_loop_ctrl_post_flagged(c, cur, mf, gf_o, L.ffn_norm, g.inp_flags, w.rms_eps);
    }

    // Fused Q8_0 router (with FLAG_PADDING -> copies row 0 experts!) + MoE SITU (Single or Dual-GPU)
    const bool do_split = use_split && w.expert_backend && w.split_exp > 0 && w.split_exp < w.n_expert &&
                          L.exp_gate_up_tail && row_b <= 64 && w.n_expert == 128 && w.n_expert_used == 8;
    ggml_tensor * w_router = L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp;

    if (do_split) {
        ggml_tensor * packed = make_fused_router_q8_topk(
            c, w_router, cur, L.ffn_exp_probs_b, g.inp_flags, w.n_expert_used, w.route_scale, 1e-20f, w.split_pct);
        ggml_tensor * hot_sel  = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 0);
        ggml_tensor * cold_sel = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], packed->nb[2]);
        ggml_tensor * wsel     = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 2 * packed->nb[2]);
        ggml_build_forward_expand(gf, hot_sel);
        ggml_build_forward_expand(gf, cold_sel);
        ggml_build_forward_expand(gf, wsel);

        ggml_tensor * x3 = ggml_reshape_3d(c, cur, w.n_embd, 1, row_b);
        ggml_tensor * hot_gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, hot_sel);
        ggml_tensor * hot_act  = ggml_inkling2_situ(c, hot_gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * hot_down = ggml_mul_mat_id(c, L.ffn_down_exps, hot_act, hot_sel);

        const int cold_sum_f16 = (row_b > 16) ? 1 : 0;
        ggml_tensor * cold_in  = (row_b > 32) ? ggml_cast(c, cur, GGML_TYPE_F16) : cur;
        ggml_tensor * cold_f32 = (cold_in->type == GGML_TYPE_F16) ? ggml_cast(c, cold_in, GGML_TYPE_F32) : cold_in;
        if (cold_f32 != cold_in) g.expert_nodes.push_back(cold_f32);
        ggml_tensor * cold_x3   = ggml_reshape_3d(c, cold_f32, w.n_embd, 1, row_b);
        ggml_tensor * cold_gu   = ggml_mul_mat_id(c, L.exp_gate_up_tail, cold_x3, cold_sel);
        ggml_tensor * cold_act  = ggml_inkling2_situ(c, cold_gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * cold_down = ggml_mul_mat_id(c, L.exp_down_tail, cold_act, cold_sel);
        ggml_tensor * cold_sum  = ggml_inkling2_moe_masked_sum(c, cold_down, wsel, cold_sel, nullptr, cold_sum_f16);
        g.expert_nodes.insert(g.expert_nodes.end(), { cold_gu, cold_act, cold_down, cold_sum });
        ggml_build_forward_expand(gf, cold_sum);
        ggml_build_forward_expand(gf, hot_down);

        cur = ggml_inkling2_moe_masked_sum(c, hot_down, wsel, hot_sel, cold_sum, 0);
    } else {
        ggml_tensor * packed = make_fused_router_q8_topk(
            c, w_router, cur, L.ffn_exp_probs_b, g.inp_flags, w.n_expert_used, w.route_scale, 1e-20f, 0);
        ggml_tensor * sel  = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], 0);
        ggml_tensor * wsel = ggml_view_2d(c, packed, w.n_expert_used, row_b, packed->nb[1], packed->nb[2]);

        ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, row_b);
        ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
        ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * down = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
        cur = ggml_inkling2_moe_masked_sum(c, down, wsel, sel, nullptr, 0);
    }

    // Tree ffn_post ShortConv + residual add
    cur = make_tree_sconv(c, cur, L.ffn_post_conv, tc.prefix_conv[ip], tc.tree_conv[ip],
                          g.inp_meta, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, m_stride);
    return cur;
}

static bool alloc_phase_graph(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        Inkling2TwoDSDPhaseGraph & g,
        size_t max_nodes,
        bool use_split,
        bool use_tier15) {
    const bool need_sched = (use_split || use_tier15) && !g.expert_nodes.empty() && w.expert_backend && cache.split_cpu;
    if (need_sched) {
        ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
        g.sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
        if (!g.sched) return false;
        ggml_backend_sched_set_late_cross_input_split(g.sched, true);
        ggml_backend_cuda_set_copy_tail(target_backend, true);
        ggml_backend_cuda_set_copy_tail(w.expert_backend, true);
        ggml_backend_sched_reset(g.sched);
        const std::unordered_set<const ggml_tensor *> on_exp(g.expert_nodes.begin(), g.expert_nodes.end());
        for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
            ggml_tensor * node = ggml_graph_node(g.gf, i);
            for (ggml_tensor * src : node->src) {
                for (ggml_tensor * t = src; t; t = t->view_src) {
                    if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                        ggml_backend_t pin_be = (t == g.inp_prefix_len_r) ? w.expert_backend : target_backend;
                        ggml_backend_sched_set_tensor_backend(g.sched, t, pin_be);
                    }
                }
            }
            const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                 node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                 node->op == GGML_OP_TRANSPOSE;
            if (on_exp.count(node)) {
                ggml_backend_sched_set_tensor_backend(g.sched, node, w.expert_backend);
            } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                ggml_backend_sched_set_tensor_backend(g.sched, node, target_backend);
            }
        }
        return ggml_backend_sched_alloc_graph(g.sched, g.gf);
    }
    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
    return g.alloc && ggml_gallocr_alloc_graph(g.alloc, g.gf);
}

static bool build_phase_graph(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        Inkling2TwoDSDPhaseGraph & g,
        int phase_id, // 0=prelude, 1=core, 2=coda
        int row_b,
        int path_b,
        bool has_later,
        bool use_split) {
    auto & tc = cache.tree;
    const int m_stride = 11 + path_b;
    const size_t max_nodes = (phase_id == 1) ? 12288 : 6144;
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (12 << 20), nullptr, true };
    g.ctx = ggml_init(gp);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, max_nodes, false);
    ggml_context * c = g.ctx;
    g.expert_nodes.clear();

    const bool use_paged  = (cache.kvflash && cache.kvflash->active());
    const bool use_tier15 = (use_paged && cache.kvflash->tier15_active());
    const int  kf_bs      = use_paged ? cache.kvflash->block_size() : 0;
    const int  pref_elems = use_paged ? (2 + cache.kvflash->max_blocks()) : 1;

    g.inp_meta         = ggml_new_tensor_2d(c, GGML_TYPE_I32, m_stride, row_b);
    g.inp_prefix_len   = ggml_new_tensor_1d(c, GGML_TYPE_I32, pref_elems);
    g.inp_prefix_len_r = nullptr;
    if (use_tier15) {
        g.inp_prefix_len_r = ggml_new_tensor_1d(c, GGML_TYPE_I32, pref_elems);
        ggml_set_input(g.inp_prefix_len_r);
    }
    g.inp_pos        = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    g.inp_flags      = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    g.inp_slot       = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
    ggml_set_input(g.inp_meta);
    ggml_set_input(g.inp_prefix_len);
    ggml_set_input(g.inp_pos);
    ggml_set_input(g.inp_flags);
    ggml_set_input(g.inp_slot);

    if (phase_id == 0) {
        // ── Prelude (L0..core_start-1) ─────────────────────────────────
        g.inp_tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
        ggml_set_input(g.inp_tokens);

        ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, g.inp_tokens), w.tok_norm, w.rms_eps);
        std::vector<ggml_tensor *> blocks;
        ggml_tensor * tap_l2 = nullptr;

        for (int ip = 0; ip < w.core_start; ++ip) {
            if (ip == w.twodsd.feature_layers[0]) tap_l2 = cur;
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, cur, blocks, nullptr, nullptr, false, use_split, row_b, path_b, kf_bs, use_tier15);
        }
        for (size_t b = 0; b < tc.pre_blocks.size() && b < blocks.size(); ++b) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_blocks[b], blocks[b], g.inp_slot));
        }
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_partial, cur, g.inp_slot));
        if (tap_l2) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.pre_tap, tap_l2, g.inp_slot));
        }
    } else if (phase_id == 1) {
        // ── Fused Multi-Pass Core (L6..17 across passes 0/1/2 together!) ──
        g.inp_flat_cur = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
        ggml_set_input(g.inp_flat_cur);
        if (has_later) {
            g.inp_flat_prev = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b);
            ggml_set_input(g.inp_flat_prev);
        }

        std::vector<ggml_tensor *> blocks;
        blocks.reserve(tc.pre_blocks.size() + tc.core_blocks.size());
        for (ggml_tensor * pb : tc.pre_blocks) {
            blocks.push_back(ggml_get_rows(c, pb, g.inp_slot));
        }
        ggml_tensor * cur = ggml_get_rows(c, tc.pre_partial, g.inp_slot);

        std::vector<ggml_tensor *> bank_blocks;
        ggml_tensor * bank_out = nullptr;
        if (has_later) {
            // Bank skips core_blocks[0] (= pre_partial closed at L6), keeping core_blocks[1..] + core_partial[prev]
            for (size_t b = 1; b < tc.core_blocks.size(); ++b) {
                bank_blocks.push_back(ggml_get_rows(c, tc.core_blocks[b], g.inp_flat_prev));
            }
            bank_out = ggml_get_rows(c, tc.core_partial, g.inp_flat_prev);
        }

        ggml_tensor * core_tap_out[3] = { nullptr, nullptr, nullptr };
        for (int ip = w.core_start; ip < w.core_end; ++ip) {
            for (int ti = 0; ti < 3; ++ti) {
                if (ip == w.twodsd.feature_layers[1 + ti]) core_tap_out[ti] = cur;
            }
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, cur, blocks,
                                   has_later ? &bank_blocks : nullptr, bank_out,
                                   has_later, use_split, row_b, path_b, kf_bs, use_tier15);
        }

        const size_t pre_n = tc.pre_blocks.size();
        for (size_t b = 0; b < tc.core_blocks.size() && (pre_n + b) < blocks.size(); ++b) {
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_blocks[b], blocks[pre_n + b], g.inp_flat_cur));
        }
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_partial, cur, g.inp_flat_cur));
        for (int ti = 0; ti < 3; ++ti) {
            if (core_tap_out[ti]) {
                ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.core_taps[ti], core_tap_out[ti], g.inp_flat_cur));
            }
        }
    } else {
        // ── Fused Multi-Depth Coda (L18..23 across depths {1, 2, 3}!) ──
        g.inp_flat_cur = ggml_new_tensor_1d(c, GGML_TYPE_I32, row_b); // depth_idx * S + slot
        ggml_set_input(g.inp_flat_cur);

        std::vector<ggml_tensor *> blocks;
        blocks.reserve(tc.pre_blocks.size() + tc.core_blocks.size() + 2);
        for (ggml_tensor * pb : tc.pre_blocks) {
            blocks.push_back(ggml_get_rows(c, pb, g.inp_slot));
        }
        for (ggml_tensor * cb : tc.core_blocks) {
            blocks.push_back(ggml_get_rows(c, cb, g.inp_flat_cur));
        }
        ggml_tensor * cur = ggml_get_rows(c, tc.core_partial, g.inp_flat_cur);

        ggml_tensor * tap_l22 = nullptr;
        for (int ip = w.core_end; ip < w.n_layer_phys; ++ip) {
            if (ip == w.twodsd.feature_layers[4]) tap_l22 = cur;
            cur = build_tree_layer(c, g.gf, w, tc, g, ip, cur, blocks, nullptr, nullptr, false, use_split, row_b, path_b, kf_bs, use_tier15);
        }

        // Assemble 5-tap depth features [5*H, row_b] -> write into node_features[depth_idx*S + slot] (F16)
        ggml_tensor * t0 = ggml_get_rows(c, tc.pre_tap, g.inp_slot);
        ggml_tensor * t1 = ggml_get_rows(c, tc.core_taps[0], g.inp_flat_cur);
        ggml_tensor * t2 = ggml_get_rows(c, tc.core_taps[1], g.inp_flat_cur);
        ggml_tensor * t3 = ggml_get_rows(c, tc.core_taps[2], g.inp_flat_cur);
        ggml_tensor * f01  = ggml_concat(c, t0, t1, 0);
        ggml_tensor * f23  = ggml_concat(c, t2, t3, 0);
        ggml_tensor * f03  = ggml_concat(c, f01, f23, 0);
        ggml_tensor * f04  = ggml_concat(c, f03, tap_l22, 0); // F32 [5*H, row_b]
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, tc.node_features, f04, g.inp_flat_cur));

        // Final AttnRes mix + output_norm + LM head (+ optional 79/21 split) + softcap + argmax
        ggml_tensor * out_norm = ggml_inkling2_attn_res_mix(
            c, cur, w.output_res_query, w.output_norm,
            blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);

        ggml_tensor * raw_logits = nullptr;
        if (use_split && w.expert_backend && w.output_tail && row_b <= 32) {
            const int64_t head_rows = w.output->ne[1] - w.output_tail->ne[1];
            ggml_tensor * head_w = ggml_view_2d(c, w.output, w.output->ne[0], head_rows, w.output->nb[1], 0);
            ggml_tensor * head   = ggml_mul_mat(c, head_w, out_norm);
            ggml_tensor * tail   = ggml_mul_mat(c, w.output_tail, out_norm);
            ggml_backend_cuda_mul_mat_rdna4_rows(head);
            ggml_backend_cuda_mul_mat_rdna4_rows(tail);
            g.expert_nodes.push_back(tail);
            raw_logits = ggml_concat(c, head, tail, 0);
        } else {
            raw_logits = ggml_mul_mat(c, w.output, out_norm);
        }
        ggml_tensor * logits = ggml_inkling2_softcap_slice(c, raw_logits, w.n_vocab_unpadded, w.logit_softcap);
        g.out_greedy = ggml_argmax(c, logits);
        ggml_set_output(g.out_greedy);
        ggml_build_forward_expand(g.gf, g.out_greedy);
    }

    if (!alloc_phase_graph(target_backend, w, cache, g, max_nodes, use_split, use_tier15)) {
        if (g.sched) { ggml_backend_sched_free(g.sched); g.sched = nullptr; }
        if (g.alloc) { ggml_gallocr_free(g.alloc); g.alloc = nullptr; }
        ggml_free(g.ctx); g.ctx = nullptr;
        g.gf = nullptr;
        return false;
    }
    g.built = true;
    return true;
}

struct StagedPhaseInputs {
    int row_b  = 0;
    int path_b = 0;
    std::vector<int32_t> meta;
    std::vector<int32_t> pos;
    std::vector<int32_t> flags;
    std::vector<int32_t> tokens;
    std::vector<int32_t> slot;
    std::vector<int32_t> flat_cur;
    std::vector<int32_t> flat_prev;
};

static StagedPhaseInputs stage_phase_rows(
        const std::vector<const Inkling2TwoDSDNodeDesc *> & rows,
        const std::vector<int> & planes,
        const std::vector<int> & flags_in,
        const std::vector<int> & extra_in,
        int max_nodes,
        int prefix_len) {
    const int n = (int) rows.size();
    int max_path = 1;
    for (const auto * nd : rows) {
        max_path = std::max(max_path, (int) nd->ancestors.size() + 1);
    }
    const int row_b  = bucket_rows(n);
    const int path_b = bucket_path(max_path);
    const int m_stride = 11 + path_b;
    const int S = max_nodes + 1;
    const int scratch = max_nodes;

    StagedPhaseInputs st;
    st.row_b  = row_b;
    st.path_b = path_b;
    st.meta.assign((size_t) row_b * m_stride, -1);
    st.pos.assign(row_b, prefix_len);
    st.flags.assign(row_b, kInkling2TwoDSDFlagsPadding);
    st.tokens.assign(row_b, rows[0]->token);
    st.slot.assign(row_b, scratch);
    st.flat_cur.assign(row_b, scratch);
    st.flat_prev.assign(row_b, scratch);

    // Map (slot -> row index in this call) for same-call ShortConv hist_row resolution
    std::unordered_map<int, int> row_of;
    row_of.reserve((size_t) n * 2);
    for (int r = 0; r < n; ++r) row_of[rows[r]->slot] = r;

    // Initialize padding rows safely: scratch slot, plane 0, prefix history (-1,-2,-3), self in path[0]
    for (int r = n; r < row_b; ++r) {
        int32_t * mt = st.meta.data() + (size_t) r * m_stride;
        mt[0] = scratch;
        mt[1] = prefix_len;
        mt[2] = 0;
        mt[3] = kInkling2TwoDSDFlagsPadding;
        mt[4] = 0;
        mt[5] = -1; mt[6] = -1; mt[7] = -1;
        mt[8] = -1; mt[9] = -2; mt[10] = -3;
        mt[11] = scratch;
    }

    for (int r = 0; r < n; ++r) {
        const auto & nd = *rows[r];
        const int plane = planes[r];
        const int flg   = flags_in[r];
        const int ext   = extra_in[r];
        int32_t * mt = st.meta.data() + (size_t) r * m_stride;

        mt[0] = nd.slot;
        mt[1] = nd.pos;
        mt[2] = plane;
        mt[3] = flg;
        mt[4] = ext;

        const int n_anc = (int) nd.ancestors.size();
        for (int j = 0; j < 3; ++j) {
            const int behind = n_anc - 3 + j;
            if (behind >= 0) {
                const int anc_slot = nd.ancestors[behind];
                auto it = row_of.find(anc_slot);
                if (it != row_of.end() && planes[it->second] == plane) {
                    mt[5 + j] = it->second; // same-call input row
                    mt[8 + j] = anc_slot;
                } else {
                    mt[5 + j] = -1;
                    mt[8 + j] = anc_slot;   // tree_conv[plane, anc_slot]
                }
            } else {
                mt[5 + j] = -1;
                // prefix_conv index = 3 + behind in {0, 1, 2}, encoded as -(idx + 1) in {-1, -2, -3}
                const int pidx = 3 + behind;
                mt[8 + j] = -(pidx + 1);
            }
        }
        for (int e = 0; e < n_anc; ++e) mt[11 + e] = nd.ancestors[e];
        mt[11 + n_anc] = nd.slot;

        st.pos[r]       = nd.pos;
        st.flags[r]     = flg;
        st.tokens[r]    = nd.token;
        st.slot[r]      = nd.slot;
        st.flat_cur[r]  = plane * S + nd.slot;
        st.flat_prev[r] = std::max(0, plane - 1) * S + nd.slot;
    }
    return st;
}

static bool run_phase(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        int phase_id,
        const StagedPhaseInputs & st,
        bool has_later,
        bool allow_split,
        std::vector<int32_t> * out_greedy = nullptr) {
    auto & tc = cache.tree;
    const bool use_split = allow_split && w.expert_backend && w.split_exp > 0 && st.row_b >= 2 && st.row_b <= 64;
    const auto key = std::make_tuple(phase_id, st.row_b, st.path_b, has_later ? 1 : 0, use_split ? 1 : 0);
    auto & g = tc.phase_graphs[key];
    if (!g.built) {
        if (!build_phase_graph(target_backend, w, cache, g, phase_id, st.row_b, st.path_b, has_later, use_split)) {
            return false;
        }
    }

    const bool use_paged  = (cache.kvflash && cache.kvflash->active());
    const bool use_tier15 = (use_paged && cache.kvflash->tier15_active());
    ggml_backend_tensor_set(g.inp_meta,       st.meta.data(),  0, st.meta.size()  * sizeof(int32_t));
    if (use_paged) {
        std::vector<int32_t> pbuf;
        cache.kvflash->fill_tree_prefix_buf(tc.prefix_len, /*remote=*/false, pbuf);
        ggml_backend_tensor_set(g.inp_prefix_len, pbuf.data(), 0, pbuf.size() * sizeof(int32_t));
        if (use_tier15 && g.inp_prefix_len_r) {
            cache.kvflash->fill_tree_prefix_buf(tc.prefix_len, /*remote=*/true, pbuf);
            ggml_backend_tensor_set(g.inp_prefix_len_r, pbuf.data(), 0, pbuf.size() * sizeof(int32_t));
        }
    } else {
        const int32_t pref_len = tc.prefix_len;
        ggml_backend_tensor_set(g.inp_prefix_len, &pref_len, 0, sizeof(pref_len));
    }
    ggml_backend_tensor_set(g.inp_pos,        st.pos.data(),   0, st.pos.size()   * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_flags,      st.flags.data(), 0, st.flags.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.inp_slot,       st.slot.data(),  0, st.slot.size()  * sizeof(int32_t));
    if (g.inp_tokens)    ggml_backend_tensor_set(g.inp_tokens,    st.tokens.data(),    0, st.tokens.size()    * sizeof(int32_t));
    if (g.inp_flat_cur)  ggml_backend_tensor_set(g.inp_flat_cur,  st.flat_cur.data(),  0, st.flat_cur.size()  * sizeof(int32_t));
    if (g.inp_flat_prev) ggml_backend_tensor_set(g.inp_flat_prev, st.flat_prev.data(), 0, st.flat_prev.size() * sizeof(int32_t));

    ggml_status status = GGML_STATUS_SUCCESS;
    if (g.sched) {
        status = ggml_backend_sched_graph_compute_async(g.sched, g.gf);
        if (status == GGML_STATUS_SUCCESS && w.expert_backend) {
            ggml_backend_cuda_hold(w.expert_backend, 8000);
        }
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_sched_synchronize(g.sched);
        }
    } else {
        status = ggml_backend_graph_compute(target_backend, g.gf);
    }
    if (status != GGML_STATUS_SUCCESS) return false;

    if (out_greedy && g.out_greedy) {
        out_greedy->resize((size_t) st.row_b);
        ggml_backend_tensor_get(g.out_greedy, out_greedy->data(), 0, (size_t) st.row_b * sizeof(int32_t));
    }
    return true;
}

} // namespace

// ─── 60-Plane Multi-Depth Prefill Chunk ─────────────────────────────────
bool inkling2_2dsd_prefill_chunk(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int32_t * tokens,
        int n_tokens,
        int pos0,
        bool is_last_chunk,
        int32_t & out_greedy_tok,
        int capture_q_rows) {
    if (!w.twodsd.loaded || !cache.tree.active || n_tokens <= 0) return false;
    auto & tc = cache.tree;
    const int Hq = w.n_head, Hk = w.n_head_kv, D = w.head_dim;
    const int q_sz = Hq * D, kv_sz = Hk * D, r_sz = w.d_rel * Hq;

    const bool use_paged  = (cache.kvflash && cache.kvflash->active());
    const bool use_tier15 = (use_paged && cache.kvflash->tier15_active());
    const int  kf_bs      = use_paged ? cache.kvflash->block_size() : 0;
    const int  q_cap_n    = (use_paged && capture_q_rows > 0) ? std::min(capture_q_rows, n_tokens) : 0;

    if (use_paged) {
        if (!cache.kvflash->prepare_span(pos0, n_tokens)) return false;
    }

    // Map (phys layer ip, logical plane) -> global plane index [0 .. L_global-1] matching create_inkling2_cache!
    // Under Shared-KV (W>0), only the final pass (plane == P-1) is captured for L11/L17/L23.
    std::vector<std::array<int, 4>> global_plane_idx((size_t) w.n_layer_phys, {-1, -1, -1, -1});
    int n_global_planes = 0;
    if (use_paged) {
        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            if (w.layers[ip].is_swa) continue;
            const int p_st = (int) tc.prefix_k[ip]->ne[3];
            const int P    = tc.planes[ip];
            if (p_st == 1 && P > 1) {
                global_plane_idx[(size_t) ip][P - 1] = n_global_planes++;
            } else {
                for (int p = 0; p < P && p < 4; ++p) {
                    global_plane_idx[(size_t) ip][p] = n_global_planes++;
                }
            }
        }
    }

    const size_t max_nodes = 24576;
    struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + (24 << 20), nullptr, true };
    ggml_context * c = ggml_init(gp);
    if (!c) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(c, max_nodes, false);

    ggml_tensor * inp_tok       = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ggml_tensor * inp_pos       = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tokens);
    ggml_tensor * inp_meta      = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
    ggml_tensor * inp_meta_r    = nullptr;
    ggml_tensor * inp_bt_local  = nullptr;
    ggml_tensor * inp_bt_remote = nullptr;
    ggml_tensor * q_cap_out     = nullptr;
    std::vector<ggml_tensor *> expert_nodes;
    ggml_set_input(inp_tok);
    ggml_set_input(inp_pos);
    ggml_set_input(inp_meta);
    if (use_paged) {
        inp_bt_local = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
        ggml_set_input(inp_bt_local);
        if (use_tier15) {
            inp_meta_r    = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
            inp_bt_remote = ggml_new_tensor_1d(c, GGML_TYPE_I32, cache.kvflash->max_blocks());
            ggml_set_input(inp_meta_r);
            ggml_set_input(inp_bt_remote);
        }
        if (q_cap_n > 0 && n_global_planes > 0) {
            q_cap_out = ggml_new_tensor_4d(c, GGML_TYPE_F32, D, Hq, n_global_planes, q_cap_n);
            ggml_set_output(q_cap_out);
        }
    }

    auto run_prefill_layer = [&](int ip, int plane, ggml_tensor * cur,
                                 std::vector<ggml_tensor *> & blocks,
                                 const std::vector<ggml_tensor *> * bank,
                                 ggml_tensor * bank_out) -> ggml_tensor * {
        const auto & L = w.layers[ip];
        ggml_tensor * inpSA = cur;
        const bool shared_global = (!L.is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
        const int  kv_plane      = (tc.prefix_k[ip]->ne[3] == 1) ? 0 : plane;
        ggml_tensor * k_p = shared_global
            ? ggml_view_3d(c, tc.prefix_k[ip], tc.prefix_k[ip]->ne[0], tc.global_pool_cap, tc.prefix_k[ip]->ne[2],
                           tc.prefix_k[ip]->nb[1], tc.prefix_k[ip]->nb[2], 0)
            : plane_view_kv(c, tc.prefix_k[ip], kv_plane);
        ggml_tensor * v_p = shared_global
            ? ggml_view_3d(c, tc.prefix_v[ip], tc.prefix_v[ip]->ne[0], tc.global_pool_cap, tc.prefix_v[ip]->ne[2],
                           tc.prefix_v[ip]->nb[1], tc.prefix_v[ip]->nb[2], 0)
            : plane_view_kv(c, tc.prefix_v[ip], kv_plane);
        ggml_tensor * st  = plane_view_conv(c, tc.prefix_conv[ip], plane);

        ggml_tensor * messages = nullptr;
        const Inkling2LoopLayer * loop = nullptr;
        if (bank_out != nullptr) {
            loop = &w.loop_layers[ip - w.core_start];
            messages = ggml_inkling2_attn_res_mix(
                c, bank_out, loop->msg_query, nullptr,
                (bank && !bank->empty()) ? bank->data() : nullptr,
                bank ? (int) bank->size() : 0, w.rms_eps);
        }
        auto get_msg = [&](int ctrl) -> ggml_tensor * {
            return ggml_view_2d(c, messages, w.n_embd, n_tokens,
                                messages->nb[2], (size_t) ctrl * messages->nb[1]);
        };

        ggml_tensor * inputs[3] = { cur, cur, cur };
        const bool same_qkv = L.is_swa && (messages == nullptr);
        if (messages == nullptr) {
            if (L.attn_res_query == nullptr) {
                inputs[0] = rms_norm_w(c, cur, L.attn_norm, w.rms_eps);
                inputs[1] = inputs[2] = inputs[0];
            } else if (L.is_swa) {
                inputs[0] = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, L.attn_norm,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                inputs[1] = inputs[2] = inputs[0];
            } else {
                ggml_tensor * m3 = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, L.attn_qkv_norm,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                for (int j = 0; j < 3; ++j)
                    inputs[j] = ggml_view_2d(c, m3, w.n_embd, n_tokens, m3->nb[2], (size_t) j * m3->nb[1]);
            }
        } else {
            ggml_tensor * pre_qkv = cur;
            if (L.attn_res_query) {
                ggml_tensor * mx = ggml_inkling2_attn_res_mix(
                    c, cur, L.attn_res_query, nullptr,
                    blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
                pre_qkv = L.is_swa ? mx : ggml_permute(c, mx, 0, 2, 1, 3);
            }
            ggml_tensor * mqv = ggml_view_3d(c, messages, w.n_embd, 3, n_tokens, messages->nb[1], messages->nb[2], 0);
            ggml_tensor * mq  = ggml_permute(c, mqv, 0, 2, 1, 3);
            ggml_tensor * nw  = L.is_swa ? L.attn_norm : L.attn_qkv_norm;
            ggml_tensor * gt  = ggml_mul_mat(c, loop->ctrl_down_qkv, ggml_inkling2_loop_ctrl_pre(c, pre_qkv, mq, w.rms_eps));
            ggml_tensor * gfo = ggml_mul_mat(c, loop->ctrl_gf_qkv, gt);
            ggml_tensor * qi  = ggml_inkling2_loop_ctrl_post(c, pre_qkv, mq, gfo, nw, w.rms_eps);
            for (int j = 0; j < 3; ++j)
                inputs[j] = ggml_view_2d(c, qi, w.n_embd, n_tokens, qi->nb[1], (size_t) j * qi->nb[2]);
        }

        ggml_tensor * Q_raw, * K_raw, * V_raw, * Gate, * R_raw, * Sink;
        ggml_tensor * k_dyn = nullptr, * v_dyn = nullptr;
        if (same_qkv && L.wqkvgrs) {
            ggml_tensor * proj = ggml_mul_mat(c, L.wqkvgrs, inputs[0]);
            Q_raw = row_view_2d(c, proj, 0, q_sz);
            Gate  = row_view_2d(c, proj, q_sz, q_sz);
            R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
            Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
            K_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq, kv_sz);
            V_raw = row_view_2d(c, proj, 2 * q_sz + r_sz + Hq + kv_sz, kv_sz);
        } else {
            if (L.wqgrs) {
                ggml_tensor * proj = ggml_mul_mat(c, L.wqgrs, inputs[0]);
                Q_raw = row_view_2d(c, proj, 0, q_sz);
                Gate  = row_view_2d(c, proj, q_sz, q_sz);
                R_raw = row_view_2d(c, proj, 2 * q_sz, r_sz);
                Sink  = row_view_2d(c, proj, 2 * q_sz + r_sz, Hq);
            } else {
                Q_raw = ggml_mul_mat(c, L.wq, inputs[0]);
                Gate  = ggml_mul_mat(c, L.wqkv_gate, inputs[0]);
                R_raw = ggml_mul_mat(c, L.attn_rel, inputs[0]);
                Sink  = ggml_mul_mat(c, L.attn_sink_proj, inputs[0]);
            }
            if (!L.is_swa && L.wk_dyn && L.wv_dyn) {
                ggml_tensor * kd = ggml_mul_mat(c, L.wk_dyn, inputs[1]);
                K_raw = row_view_2d(c, kd, 0, kv_sz);
                k_dyn = row_view_2d(c, kd, kv_sz, (int64_t) w.sconv_kernel * Hk);
                ggml_tensor * vd = ggml_mul_mat(c, L.wv_dyn, inputs[2]);
                V_raw = row_view_2d(c, vd, 0, kv_sz);
                v_dyn = row_view_2d(c, vd, kv_sz, (int64_t) w.sconv_kernel * Hk);
            } else {
                K_raw = ggml_mul_mat(c, L.wk, inputs[1]);
                V_raw = ggml_mul_mat(c, L.wv, inputs[2]);
                if (!L.is_swa) {
                    k_dyn = ggml_mul_mat(c, L.attn_k_conv_proj, inputs[1]);
                    v_dyn = ggml_mul_mat(c, L.attn_v_conv_proj, inputs[2]);
                }
            }
        }

        const int swa = L.is_swa ? w.swa_window : 0;
        ggml_tensor * Q = nullptr, * rt = nullptr;
        make_clip_qr_rel_table(
            c, Q_raw, L.attn_q_clip, R_raw, L.attn_rel_clip,
            L.is_swa ? nullptr : inp_pos, L.attn_rel_bias, Sink,
            D, w.d_rel, Hq, w.qk_clip, w.rel_clip, w.rms_eps,
            L.is_swa ? 0 : w.log_scale_floor, L.is_swa ? 0.0f : w.log_scale_alpha,
            &Q, &rt);

        if (!L.is_swa && q_cap_out && plane >= 0 && plane < 4) {
            const int gidx = global_plane_idx[(size_t) ip][plane];
            if (gidx >= 0) {
                ggml_tensor * q_tail = ggml_view_3d(
                    c, Q, D, Hq, q_cap_n, Q->nb[1], Q->nb[2],
                    (size_t) (n_tokens - q_cap_n) * Q->nb[2]);
                ggml_tensor * q_dst = ggml_view_3d(
                    c, q_cap_out, D, Hq, q_cap_n, q_cap_out->nb[1], q_cap_out->nb[3],
                    (size_t) gidx * q_cap_out->nb[2]);
                ggml_build_forward_expand(gf, ggml_cpy(c, q_tail, q_dst));
            }
        }

        ggml_tensor * kd_dep = nullptr, * vd_dep = nullptr;
        if (shared_global && plane < tc.planes[ip] - 1) {
            // Last-Loop Shared-KV early pass: write to pool_view for in-chunk prefill AND
            // mirror the recent W-window into this plane's dedicated tail ring!
            const size_t tail_off = (size_t) (tc.global_pool_cap + plane * tc.shared_tail_cap) * tc.prefix_k[ip]->nb[1];
            ggml_tensor * k_tail = ggml_view_3d(
                c, tc.prefix_k[ip], tc.prefix_k[ip]->ne[0], tc.shared_tail_cap, tc.prefix_k[ip]->ne[2],
                tc.prefix_k[ip]->nb[1], tc.prefix_k[ip]->nb[2], tail_off);
            ggml_tensor * v_tail = ggml_view_3d(
                c, tc.prefix_v[ip], tc.prefix_v[ip]->ne[0], tc.shared_tail_cap, tc.prefix_v[ip]->ne[2],
                tc.prefix_v[ip]->nb[1], tc.prefix_v[ip]->nb[2], tail_off);
            ggml_tensor * kc = ggml_inkling2_sconv(c, K_raw, L.attn_k_conv, st, st, k_dyn, 0, D, 1);
            ggml_tensor * vc = ggml_inkling2_sconv(c, V_raw, L.attn_v_conv, st, st, v_dyn, kv_sz, D, 1);
            if (use_paged) {
                kd_dep = ggml_inkling2_kv_write_paged(c, kc, k_p, inp_pos, inp_bt_local, kf_bs, D, Hk, 0, w.qk_clip, w.rms_eps);
                vd_dep = ggml_inkling2_kv_write_paged(c, vc, v_p, inp_pos, inp_bt_local, kf_bs, D, Hk, 1, w.qk_clip, w.rms_eps);
            } else {
                kd_dep = ggml_inkling2_kv_write(c, kc, k_p, inp_pos, D, Hk, 0, 0, w.qk_clip, w.rms_eps);
                vd_dep = ggml_inkling2_kv_write(c, vc, v_p, inp_pos, D, Hk, 1, 0, w.qk_clip, w.rms_eps);
            }
            // Slice to the final min(n_tokens, shared_tail_cap) tokens before writing the W-ring so
            // parallel GPU threadblocks never race on colliding (pos % shared_tail_cap) slots when n_tokens > shared_tail_cap!
            const int m_tail    = std::min(n_tokens, tc.shared_tail_cap);
            const int tail_skip = n_tokens - m_tail;
            ggml_tensor * kc_tail  = (tail_skip > 0)
                ? ggml_view_2d(c, kc, kc->ne[0], m_tail, kc->nb[1], (size_t) tail_skip * kc->nb[1])
                : kc;
            ggml_tensor * vc_tail  = (tail_skip > 0)
                ? ggml_view_2d(c, vc, vc->ne[0], m_tail, vc->nb[1], (size_t) tail_skip * vc->nb[1])
                : vc;
            ggml_tensor * pos_tail = (tail_skip > 0)
                ? ggml_view_1d(c, inp_pos, m_tail, (size_t) tail_skip * inp_pos->nb[0])
                : inp_pos;
            ggml_tensor * kt_dep = ggml_inkling2_kv_write(c, kc_tail, k_tail, pos_tail, D, Hk, 0, tc.shared_tail_cap, w.qk_clip, w.rms_eps);
            ggml_tensor * vt_dep = ggml_inkling2_kv_write(c, vc_tail, v_tail, pos_tail, D, Hk, 1, tc.shared_tail_cap, w.qk_clip, w.rms_eps);
            ggml_build_forward_expand(gf, kt_dep);
            ggml_build_forward_expand(gf, vt_dep);
        } else if (!L.is_swa && use_paged) {
            kd_dep = ggml_inkling2_sconv_kv_write_paged(
                c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                inp_bt_local, kf_bs, 0, D, Hk, 0, w.qk_clip, w.rms_eps, 1);
            vd_dep = ggml_inkling2_sconv_kv_write_paged(
                c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                inp_bt_local, kf_bs, kv_sz, D, Hk, 1, w.qk_clip, w.rms_eps, 1);
        } else {
            kd_dep = ggml_inkling2_sconv_kv_write(
                c, K_raw, L.attn_k_conv, st, st, k_dyn, k_p, inp_pos,
                0, D, Hk, 0, swa, w.qk_clip, w.rms_eps, 1);
            vd_dep = ggml_inkling2_sconv_kv_write(
                c, V_raw, L.attn_v_conv, st, st, v_dyn, v_p, inp_pos,
                kv_sz, D, Hk, 1, swa, w.qk_clip, w.rms_eps, 1);
        }
        ggml_tensor * ao = nullptr;
        if (!L.is_swa && use_paged) {
            if (use_tier15 && ip < (int) tc.remote_prefix_k.size() && tc.remote_prefix_k[ip]) {
                ggml_tensor * rk_p = plane_view_kv(c, tc.remote_prefix_k[ip], kv_plane);
                ggml_tensor * rv_p = plane_view_kv(c, tc.remote_prefix_v[ip], kv_plane);
                ggml_tensor * part_r = ggml_inkling2_attn_rel_paged(
                    c, Q, rk_p, rv_p, rt, inp_meta_r, nullptr, nullptr, nullptr,
                    inp_bt_remote, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/1);
                expert_nodes.push_back(part_r);
                ggml_tensor * part_l = ggml_inkling2_attn_rel_paged(
                    c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, nullptr,
                    inp_bt_local, kf_bs, w.rel_extent, /*partial=*/1, /*include_sink=*/0);
                ao = ggml_inkling2_attn_split_combine(c, part_r, part_l, Gate, D, Hq);
            } else {
                ao = ggml_inkling2_attn_rel_paged(
                    c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate,
                    inp_bt_local, kf_bs, w.rel_extent, /*partial=*/0, /*include_sink=*/1);
            }
        } else {
            ao = ggml_inkling2_attn_rel(
                c, Q, k_p, v_p, rt, inp_meta, kd_dep, vd_dep, Gate, swa, w.rel_extent, 0, 1);
        }

        cur = ggml_mul_mat(c, L.wo, ao);
        if (ip % w.attn_res_block == 0) {
            cur = ggml_inkling2_sconv(c, cur, L.attn_post_conv, st, st, nullptr, 2 * kv_sz, 0, 1);
            blocks.push_back(inpSA);
        } else {
            cur = ggml_inkling2_sconv_add(c, cur, L.attn_post_conv, st, st, nullptr, inpSA, 2 * kv_sz, 0, 1);
        }
        ggml_tensor * ffn_inp = cur;
        if (messages == nullptr) {
            cur = ggml_inkling2_attn_res_mix(
                c, cur, L.ffn_res_query, L.ffn_norm,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
        } else {
            cur = ggml_inkling2_attn_res_mix(
                c, cur, L.ffn_res_query, nullptr,
                blocks.empty() ? nullptr : blocks.data(), (int) blocks.size(), w.rms_eps);
            ggml_tensor * mf   = get_msg(Inkling2LoopLayer::CTRL_FFN);
            ggml_tensor * cd_w = loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                ? loop->ctrl_down_f16[Inkling2LoopLayer::CTRL_FFN]
                : loop->ctrl_down[Inkling2LoopLayer::CTRL_FFN];
            ggml_tensor * gt  = ggml_mul_mat(c, cd_w, ggml_inkling2_loop_ctrl_pre(c, cur, mf, w.rms_eps));
            ggml_tensor * gfo = ggml_mul_mat(c, loop->ctrl_gf[Inkling2LoopLayer::CTRL_FFN], gt);
            cur = ggml_inkling2_loop_ctrl_post(c, cur, mf, gfo, L.ffn_norm, w.rms_eps);
        }

        ggml_tensor * rl   = ggml_mul_mat(c, L.ffn_gate_inp_q8 ? L.ffn_gate_inp_q8 : L.ffn_gate_inp, cur);
        ggml_tensor * sel  = ggml_inkling2_sigmoid_topk_ids(c, rl, L.ffn_exp_probs_b, w.n_expert_used);
        ggml_tensor * wsel = ggml_inkling2_sigmoid_topk_wts(c, rl, sel, w.route_scale, 1e-20f);
        ggml_tensor * x3   = ggml_reshape_3d(c, cur, w.n_embd, 1, n_tokens);
        ggml_tensor * gu   = ggml_mul_mat_id(c, L.ffn_gate_up_exps, x3, sel);
        ggml_tensor * act  = ggml_inkling2_situ(c, gu, w.situ_beta, w.situ_linear_beta);
        ggml_tensor * dn   = ggml_mul_mat_id(c, L.ffn_down_exps, act, sel);
        cur = ggml_ds4_moe_fused_combine_shared(c, dn, wsel, nullptr);
        cur = ggml_inkling2_sconv_add(c, cur, L.ffn_post_conv, st, st, nullptr, ffn_inp, 2 * kv_sz + w.n_embd, 0, 1);
        return cur;
    };

    // 1. Prelude (L0..5, plane 0)
    ggml_tensor * cur = rms_norm_w(c, ggml_get_rows(c, w.tok_embd, inp_tok), w.tok_norm, w.rms_eps);
    std::vector<ggml_tensor *> prelude_blocks;
    ggml_tensor * tap_l2 = nullptr;
    for (int ip = 0; ip < w.core_start; ++ip) {
        if (ip == w.twodsd.feature_layers[0]) tap_l2 = cur;
        cur = run_prefill_layer(ip, 0, cur, prelude_blocks, nullptr, nullptr);
    }
    ggml_tensor * prelude_cur = cur;

    // 2. For each pass p in 0..n_passes-1: run Core (L6..17, plane p) AND Coda (L18..23, plane p)
    std::vector<ggml_tensor *> bank;
    ggml_tensor * bank_out = nullptr;
    ggml_tensor * loop3_taps[5] = { tap_l2, nullptr, nullptr, nullptr, nullptr };
    ggml_tensor * final_out_norm = nullptr;
    ggml_tensor * prev_coda_out = nullptr;

    for (int pass = 0; pass < w.n_passes; ++pass) {
        std::vector<ggml_tensor *> blocks = prelude_blocks;
        ggml_tensor * p_cur = prelude_cur;

        for (int ip = w.core_start; ip < w.core_end; ++ip) {
            if (pass == w.n_passes - 1) {
                for (int ti = 0; ti < 3; ++ti) {
                    if (ip == w.twodsd.feature_layers[1 + ti]) loop3_taps[1 + ti] = p_cur;
                }
            }
            p_cur = run_prefill_layer(ip, pass, p_cur, blocks, (pass > 0) ? &bank : nullptr, (pass > 0) ? bank_out : nullptr);
        }

        // Update bank for next pass BEFORE coda modifies `blocks`
        std::vector<ggml_tensor *> next_bank(blocks.begin() + (ptrdiff_t) prelude_blocks.size() + 1, blocks.end());
        ggml_tensor * next_bank_out = p_cur;

        // Coda for depth = pass + 1 (populates Coda plane `pass` KV & conv states!)
        // Under Shared-KV + multi-GPU scheduler, chain a zero-valued scalar dependency from
        // the previous pass's coda so ggml_backend_sched executes coda passes strictly in
        // 0 -> 1 -> 2 order and never lets an earlier pass overwrite L23's shared plane 0.
        std::vector<ggml_tensor *> coda_blocks = blocks;
        ggml_tensor * c_cur = p_cur;
        if (prev_coda_out && tc.shared_kv_window > 0 && use_tier15) {
            ggml_tensor * z = ggml_scale(c, ggml_view_2d(c, prev_coda_out, 1, 1, prev_coda_out->nb[1], 0), 0.0f);
            c_cur = ggml_add(c, c_cur, z);
        }
        for (int ip = w.core_end; ip < w.n_layer_phys; ++ip) {
            if (pass == w.n_passes - 1 && ip == w.twodsd.feature_layers[4]) {
                loop3_taps[4] = c_cur;
            }
            c_cur = run_prefill_layer(ip, pass, c_cur, coda_blocks, nullptr, nullptr);
        }
        prev_coda_out = c_cur;
        if (pass == w.n_passes - 1) {
            final_out_norm = ggml_inkling2_attn_res_mix(
                c, c_cur, w.output_res_query, w.output_norm,
                coda_blocks.empty() ? nullptr : coda_blocks.data(), (int) coda_blocks.size(), w.rms_eps);
        } else {
            ggml_build_forward_expand(gf, c_cur);
        }

        bank = std::move(next_bank);
        bank_out = next_bank_out;
    }

    // Pack loop-3 5-tap features [5*H, n_tokens]
    ggml_tensor * f01 = ggml_concat(c, loop3_taps[0], loop3_taps[1], 0);
    ggml_tensor * f23 = ggml_concat(c, loop3_taps[2], loop3_taps[3], 0);
    ggml_tensor * feats_out = ggml_concat(c, ggml_concat(c, f01, f23, 0), loop3_taps[4], 0);
    ggml_set_output(feats_out);
    ggml_build_forward_expand(gf, feats_out);

    ggml_tensor * greedy_t = nullptr;
    if (is_last_chunk) {
        ggml_tensor * lm_in = ggml_view_2d(c, final_out_norm, w.n_embd, 1, final_out_norm->nb[1],
                                           (size_t) (n_tokens - 1) * final_out_norm->nb[1]);
        ggml_tensor * raw_l = ggml_mul_mat(c, w.output, lm_in);
        ggml_tensor * soft  = ggml_inkling2_softcap_slice(c, raw_l, w.n_vocab_unpadded, w.logit_softcap);
        greedy_t = ggml_argmax(c, soft);
        ggml_set_output(greedy_t);
        ggml_build_forward_expand(gf, greedy_t);
    } else {
        ggml_build_forward_expand(gf, final_out_norm);
    }

    const bool need_sched = use_tier15 && !expert_nodes.empty() && w.expert_backend && cache.split_cpu;
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t ga = nullptr;
    if (need_sched) {
        ggml_backend_t backends[3] = { target_backend, w.expert_backend, cache.split_cpu };
        sched = ggml_backend_sched_new(backends, nullptr, 3, max_nodes, true, false);
        if (!sched) { ggml_free(c); return false; }
        ggml_backend_sched_set_late_cross_input_split(sched, true);
        ggml_backend_sched_reset(sched);
        const std::unordered_set<const ggml_tensor *> on_exp(expert_nodes.begin(), expert_nodes.end());
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            for (ggml_tensor * src : node->src) {
                for (ggml_tensor * t = src; t; t = t->view_src) {
                    if ((t->flags & GGML_TENSOR_FLAG_INPUT) && !t->buffer) {
                        ggml_backend_t pin_be = (t == inp_meta_r || t == inp_bt_remote) ? w.expert_backend : target_backend;
                        ggml_backend_sched_set_tensor_backend(sched, t, pin_be);
                    }
                }
            }
            const bool is_view = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                                 node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
                                 node->op == GGML_OP_TRANSPOSE;
            if (on_exp.count(node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, w.expert_backend);
            } else if (!is_view && ggml_backend_supports_op(target_backend, node)) {
                ggml_backend_sched_set_tensor_backend(sched, node, target_backend);
            }
        }
        if (q_cap_out && !q_cap_out->buffer) {
            ggml_backend_sched_set_tensor_backend(sched, q_cap_out, target_backend);
        }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_backend_sched_free(sched);
            ggml_free(c);
            return false;
        }
    } else {
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!ga || !ggml_gallocr_alloc_graph(ga, gf)) {
            if (ga) ggml_gallocr_free(ga);
            ggml_free(c);
            return false;
        }
    }

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = pos0 + i;
    int32_t meta[4] = { pos0, pos0 + n_tokens, 0, pos0 + n_tokens };
    ggml_backend_tensor_set(inp_tok,  tokens,     0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_pos,  pos.data(), 0, (size_t) n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(inp_meta, meta,       0, sizeof(meta));
    if (use_paged && inp_bt_local) {
        const auto & btl = cache.kvflash->bt_local();
        ggml_backend_tensor_set(inp_bt_local, btl.data(), 0, btl.size() * sizeof(int32_t));
        if (use_tier15 && inp_meta_r && inp_bt_remote) {
            const auto & btr = cache.kvflash->bt_remote();
            ggml_backend_tensor_set(inp_meta_r,    meta,       0, sizeof(meta));
            ggml_backend_tensor_set(inp_bt_remote, btr.data(), 0, btr.size() * sizeof(int32_t));
        }
    }

    const bool ok = sched
        ? (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
        : (ggml_backend_graph_compute(target_backend, gf) == GGML_STATUS_SUCCESS);
    bool draft_ok = false;
    if (ok) {
        if (use_paged && q_cap_out && q_cap_n > 0) {
            std::vector<float> q_host((size_t) ggml_nelements(q_cap_out));
            ggml_backend_tensor_get(q_cap_out, q_host.data(), 0, ggml_nbytes(q_cap_out));
            cache.kvflash->append_queries(q_host.data(), q_cap_n);
        }
        draft_ok = inkling2_2dsd_commit_prefix_kv_from_feats(target_backend, w, cache, feats_out, pos0, n_tokens);
        if (draft_ok && is_last_chunk && greedy_t) {
            ggml_backend_tensor_get(greedy_t, &out_greedy_tok, 0, sizeof(int32_t));
        }
        tc.prefix_len = pos0 + n_tokens;
        cache.cur_pos = pos0 + n_tokens;
    }
    if (sched) ggml_backend_sched_free(sched);
    if (ga)    ggml_gallocr_free(ga);
    ggml_free(c);
    return ok && draft_ok;
}

// ─── Advance Active Tree Nodes by +1 Loop ───────────────────────────────
bool inkling2_2dsd_advance_nodes(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
        uint32_t readout_mask,
        bool allow_split,
        std::vector<int> & out_readout_rows,
        std::vector<int32_t> & out_readout_argmax) {
    out_readout_rows.clear();
    out_readout_argmax.clear();
    if (!cache.tree.active || nodes.empty()) return false;
    auto & tc = cache.tree;

    // 1. Prelude for fresh nodes (depth == 0)
    std::vector<const Inkling2TwoDSDNodeDesc *> fresh_rows;
    std::vector<int> fresh_planes, fresh_flags, fresh_extra;
    for (const auto & nd : nodes) {
        if (nd.depth == 0) {
            fresh_rows.push_back(&nd);
            fresh_planes.push_back(0);
            fresh_flags.push_back(0);
            fresh_extra.push_back(nd.token);
        }
    }
    if (!fresh_rows.empty()) {
        auto st = stage_phase_rows(fresh_rows, fresh_planes, fresh_flags, fresh_extra, tc.max_nodes, tc.prefix_len);
        if (!run_phase(target_backend, w, cache, /*phase_id=*/0, st, /*has_later=*/false, allow_split)) {
            return false;
        }
    }

    // 2. Fused Multi-Pass Core for ALL active nodes (passes 0, 1, 2 together!)
    std::vector<const Inkling2TwoDSDNodeDesc *> all_rows;
    std::vector<int> all_planes, all_flags, all_extra;
    all_rows.reserve(nodes.size());
    all_planes.reserve(nodes.size());
    all_flags.reserve(nodes.size());
    all_extra.reserve(nodes.size());
    bool any_later = false;
    for (const auto & nd : nodes) {
        all_rows.push_back(&nd);
        all_planes.push_back(nd.depth); // pass 0, 1, or 2
        const int flg = (nd.depth > 0) ? kInkling2TwoDSDFlagsLater : 0;
        if (nd.depth > 0) any_later = true;
        all_flags.push_back(flg);
        all_extra.push_back(std::max(0, nd.depth - 1));
    }
    {
        auto st = stage_phase_rows(all_rows, all_planes, all_flags, all_extra, tc.max_nodes, tc.prefix_len);
        if (!run_phase(target_backend, w, cache, /*phase_id=*/1, st, any_later, allow_split)) {
            return false;
        }
    }

    // 3. Fused Multi-Depth Coda for nodes reaching a readout depth (depth + 1 in readout_mask)
    std::vector<const Inkling2TwoDSDNodeDesc *> read_rows;
    std::vector<int> read_planes, read_flags, read_extra;
    for (int i = 0; i < (int) nodes.size(); ++i) {
        const int new_depth = nodes[i].depth + 1;
        if ((readout_mask & (1u << new_depth)) != 0u) {
            out_readout_rows.push_back(i);
            read_rows.push_back(&nodes[i]);
            read_planes.push_back(new_depth - 1); // 0, 1, or 2
            read_flags.push_back(0);
            read_extra.push_back(new_depth - 1);
        }
    }
    if (!read_rows.empty()) {
        auto st = stage_phase_rows(read_rows, read_planes, read_flags, read_extra, tc.max_nodes, tc.prefix_len);
        std::vector<int32_t> greedy_padded;
        if (!run_phase(target_backend, w, cache, /*phase_id=*/2, st, /*has_later=*/false, allow_split, &greedy_padded)) {
            return false;
        }
        out_readout_argmax.assign(greedy_padded.begin(), greedy_padded.begin() + (ptrdiff_t) read_rows.size());
    }
    return true;
}

// ─── Commit Verified Path from Tree -> 60-Plane Prefix & 5-Layer Drafter KV ──
bool inkling2_2dsd_commit_nodes(
        ggml_backend_t target_backend,
        const Inkling2Weights & w,
        Inkling2Cache & cache,
        const int * commit_slots,
        int n_commit) {
    if (!cache.tree.active || !commit_slots || n_commit <= 0) return false;
    if (n_commit > kInkling2TwoDSDMaxCommit) {
        for (int off = 0; off < n_commit; off += kInkling2TwoDSDMaxCommit) {
            const int m = std::min(kInkling2TwoDSDMaxCommit, n_commit - off);
            if (!inkling2_2dsd_commit_nodes(target_backend, w, cache, commit_slots + off, m)) return false;
        }
        return true;
    }

    auto & tc = cache.tree;
    const bool use_paged = (cache.kvflash && cache.kvflash->active());
    const int  kf_bs     = use_paged ? cache.kvflash->block_size() : 0;
    if (use_paged) {
        if (!cache.kvflash->prepare_span(tc.prefix_len, n_commit)) return false;
    }

    auto & cg = tc.commit_graph;
    if (!cg.built) {
        const size_t max_nodes = (size_t) w.n_layer_phys * 16 + 64;
        struct ggml_init_params gp = { max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false) + 65536, nullptr, true };
        cg.ctx = ggml_init(gp);
        cg.gf  = ggml_new_graph_custom(cg.ctx, max_nodes, false);
        cg.slots = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, kInkling2TwoDSDMaxCommit);
        cg.meta  = ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, 2);
        cg.block_table = use_paged ? ggml_new_tensor_1d(cg.ctx, GGML_TYPE_I32, cache.kvflash->max_blocks()) : nullptr;
        ggml_set_input(cg.slots);
        ggml_set_input(cg.meta);
        if (cg.block_table) ggml_set_input(cg.block_table);

        for (int ip = 0; ip < w.n_layer_phys; ++ip) {
            const int swa = w.layers[ip].is_swa ? w.swa_window : 0;
            const bool shared_global = (!w.layers[ip].is_swa && tc.shared_kv_window > 0 && tc.planes[ip] > 1);
            ggml_tensor * op = make_tree_commit(
                cg.ctx, tc.prefix_k[ip], tc.prefix_v[ip], tc.tree_k[ip], tc.tree_v[ip],
                tc.prefix_conv[ip], tc.tree_conv[ip], cg.slots, cg.meta, swa,
                (!w.layers[ip].is_swa && use_paged) ? cg.block_table : nullptr,
                (!w.layers[ip].is_swa && use_paged) ? kf_bs : 0,
                shared_global ? tc.shared_kv_window : 0,
                shared_global ? tc.shared_tail_cap : 0);
            ggml_build_forward_expand(cg.gf, op);
        }
        cg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(target_backend));
        if (!cg.alloc || !ggml_gallocr_alloc_graph(cg.alloc, cg.gf)) {
            if (cg.alloc) { ggml_gallocr_free(cg.alloc); cg.alloc = nullptr; }
            ggml_free(cg.ctx); cg.ctx = nullptr;
            cg.gf = nullptr;
            return false;
        }
        cg.built = true;
    }

    int32_t slots_pad[kInkling2TwoDSDMaxCommit] = {};
    for (int i = 0; i < n_commit; ++i) slots_pad[i] = commit_slots[i];
    int32_t meta[2] = { tc.prefix_len, n_commit };
    ggml_backend_tensor_set(cg.slots, slots_pad, 0, sizeof(slots_pad));
    ggml_backend_tensor_set(cg.meta,  meta,      0, sizeof(meta));
    if (use_paged && cg.block_table) {
        const auto & btl = cache.kvflash->bt_local();
        ggml_backend_tensor_set(cg.block_table, btl.data(), 0, btl.size() * sizeof(int32_t));
    }

    if (ggml_backend_graph_compute_async(target_backend, cg.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!inkling2_2dsd_commit_prefix_kv_from_slots(target_backend, w, cache, commit_slots, n_commit, tc.prefix_len)) {
        return false;
    }

    tc.prefix_len += n_commit;
    cache.cur_pos  = tc.prefix_len;
    return true;
}

} // namespace luce::common
