#include "kimi_k3_internal.h"
#include "kimi_k3_progressive_provider.h"

#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_stream.h"
#include "common/moe_router_graph.h"
#include "common/cuda_graph_overrides.h"
#include "internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace dflash::common {
namespace {

struct KimiDivergenceTraceFileHeader {
    char magic[8] = {'K', '3', 'D', 'V', 'T', '0', '0', '2'};
    uint32_t version = 2;
    uint32_t hidden_dimension = 0;
    uint32_t latent_dimension = 0;
    uint32_t expert_count = 0;
    uint32_t top_k = 0;
    uint32_t attn_res_block_size = 0;
    uint32_t reserved = 0;
};

struct KimiDivergenceTraceRecordHeader {
    int32_t model_layer = -1;
    int32_t base_position = 0;
    int32_t token_count = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(KimiDivergenceTraceFileHeader) == 36);
static_assert(sizeof(KimiDivergenceTraceRecordHeader) == 16);

constexpr uint32_t kDivergenceTraceAttnResBoundary = 1U << 0;

// H17-only trace writer. An unset environment variable leaves the production
// path untouched. Records contain exactly the five requested comparison
// boundaries plus the layer input needed to identify AttnRes amplification.
class KimiDivergenceTraceWriter {
public:
    explicit KimiDivergenceTraceWriter(const KimiK3Weights & w) {
        const char * path = std::getenv("DFLASH_KIMI_DIVERGENCE_TRACE_OUT");
        if (!path || !*path) return;
        requested_ = true;
        path_ = path;
        file_ = std::fopen(path, "wb");
        if (!file_) return;
        KimiDivergenceTraceFileHeader header;
        header.hidden_dimension = static_cast<uint32_t>(w.n_embd);
        header.latent_dimension = static_cast<uint32_t>(w.n_expert_latent);
        header.expert_count = static_cast<uint32_t>(w.n_expert);
        header.top_k = static_cast<uint32_t>(w.n_expert_used);
        header.attn_res_block_size =
            static_cast<uint32_t>(w.attn_res_block_size);
        if (!write(&header, sizeof(header))) close_failed();
    }

    ~KimiDivergenceTraceWriter() {
        if (file_) {
            (void) std::fflush(file_);
            (void) std::fclose(file_);
        }
    }

    bool requested() const { return requested_; }
    bool good() const { return file_ != nullptr && !failed_; }
    const std::string & path() const { return path_; }

    bool append(
            int model_layer,
            int base_position,
            int token_count,
            bool attn_res_boundary,
            const std::vector<float> & layer_input,
            const std::vector<float> & pre_moe_hidden,
            const std::vector<float> & router_logits,
            const std::vector<int32_t> & selected_ids,
            const std::vector<float> & pre_expert_latent,
            const std::vector<float> & routed_latent,
            const std::vector<float> & moe_output,
            const std::vector<float> & post_moe_hidden) {
        if (!good()) return false;
        KimiDivergenceTraceRecordHeader header;
        header.model_layer = model_layer;
        header.base_position = base_position;
        header.token_count = token_count;
        header.flags = attn_res_boundary
            ? kDivergenceTraceAttnResBoundary : 0;
        const bool ok =
            write(&header, sizeof(header)) &&
            write_vector(layer_input) &&
            write_vector(pre_moe_hidden) &&
            write_vector(router_logits) &&
            write_vector(selected_ids) &&
            write_vector(pre_expert_latent) &&
            write_vector(routed_latent) &&
            write_vector(moe_output) &&
            write_vector(post_moe_hidden) &&
            std::fflush(file_) == 0;
        if (!ok) close_failed();
        return ok;
    }

private:
    bool write(const void * data, size_t bytes) {
        return file_ && bytes > 0 &&
            std::fwrite(data, 1, bytes, file_) == bytes;
    }

    template <typename T>
    bool write_vector(const std::vector<T> & values) {
        return !values.empty() &&
            write(values.data(), values.size() * sizeof(T));
    }

    void close_failed() {
        failed_ = true;
        if (file_) {
            (void) std::fclose(file_);
            file_ = nullptr;
        }
    }

    FILE * file_ = nullptr;
    bool requested_ = false;
    bool failed_ = false;
    std::string path_;
};

KimiDivergenceTraceWriter & kimi_divergence_trace_writer(
        const KimiK3Weights & w) {
    static KimiDivergenceTraceWriter writer(w);
    return writer;
}

ggml_tensor * rms_norm(ggml_context * ctx,
                       ggml_tensor * x,
                       ggml_tensor * weight,
                       float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return weight ? ggml_mul(ctx, x, weight) : x;
}

ggml_tensor * situ(ggml_context * ctx,
                   ggml_tensor * gate,
                   ggml_tensor * up,
                   float beta,
                   float linear_beta) {
    ggml_tensor * a = ggml_scale(ctx,
        ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / beta)), beta);
    a = ggml_mul(ctx, a, ggml_sigmoid(ctx, gate));
    if (linear_beta > 0.0f) {
        up = ggml_scale(ctx,
            ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / linear_beta)),
            linear_beta);
    }
    return ggml_mul(ctx, a, up);
}

struct AttnResBank {
    ggml_context * ctx = nullptr;
    float eps = 1.0e-5f;
    int64_t n_embd = 0;
    int64_t n_tokens = 1;
    std::vector<ggml_tensor *> checkpoints;

    void push(ggml_tensor * cur) {
        checkpoints.push_back(
            ggml_reshape_3d(ctx, cur, n_embd, n_tokens, 1));
    }

    ggml_tensor * mix(ggml_tensor * cur, ggml_tensor * score_weight) {
        if (checkpoints.empty()) return cur;
        const int64_t n = static_cast<int64_t>(checkpoints.size());
        ggml_tensor * mixed = nullptr;

        // AttnRes chooses a different checkpoint mixture for every token.
        // Express the small contraction as independent token slices.  This is
        // the exact one-token algebra used by the original Kimi path, avoids
        // relying on fragile rank-3 broadcast rules, and is cheap at the
        // bounded speculative widths (currently <= 16).
        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_tensor * src = nullptr; // [hidden, checkpoint]
            for (ggml_tensor * checkpoint : checkpoints) {
                ggml_tensor * checkpoint_token = ggml_view_2d(
                    ctx, checkpoint, n_embd, 1, checkpoint->nb[1],
                    static_cast<size_t>(token) * checkpoint->nb[1]);
                src = src
                    ? ggml_concat(ctx, src, checkpoint_token, 1)
                    : checkpoint_token;
            }
            ggml_tensor * cur_token = ggml_view_2d(
                ctx, cur, n_embd, 1, cur->nb[1],
                static_cast<size_t>(token) * cur->nb[1]);

            ggml_tensor * score_src = rms_norm(ctx, src, score_weight, eps);
            score_src = ggml_reshape_2d(
                ctx, ggml_sum_rows(ctx, score_src), n, 1);
            ggml_tensor * score_cur = ggml_sum_rows(
                ctx, rms_norm(ctx, cur_token, score_weight, eps));
            ggml_tensor * probs = ggml_soft_max(
                ctx, ggml_concat(ctx, score_src, score_cur, 0));
            ggml_tensor * p_src = ggml_cont(ctx,
                ggml_view_2d(ctx, probs, n, 1, probs->nb[1], 0));
            ggml_tensor * p_cur = ggml_cont(ctx,
                ggml_view_2d(ctx, probs, 1, 1, probs->nb[1],
                             probs->nb[0] * static_cast<size_t>(n)));

            ggml_tensor * src_t = ggml_cont(
                ctx, ggml_permute(ctx, src, 1, 0, 2, 3));
            ggml_tensor * out_token = ggml_add(
                ctx, ggml_mul_mat(ctx, src_t, p_src),
                ggml_mul(ctx, cur_token, p_cur));
            mixed = mixed
                ? ggml_concat(ctx, mixed, out_token, 1)
                : out_token;
        }
        return mixed;
    }
};

ggml_tensor * kda_conv1d(ggml_context * ctx,
                         ggml_cgraph * graph,
                         ggml_tensor * all_state,
                         int qkv,
                         ggml_tensor * x,
                         ggml_tensor * projection,
                         ggml_tensor * conv_weight,
                         int d_conv,
                         int head_dim,
                         int n_head,
                         bool commit_state) {
    const int64_t d_inner = static_cast<int64_t>(head_dim) * n_head;
    const int64_t state_rows = d_conv - 1;
    const int64_t n_tokens = x->ne[1];
    const size_t block_offset = static_cast<size_t>(qkv) * d_inner * all_state->nb[1];
    ggml_tensor * state = ggml_view_3d(ctx, all_state,
        state_rows, d_inner, 1, all_state->nb[1], all_state->nb[2],
        block_offset);

    ggml_tensor * projected = ggml_mul_mat(ctx, projection, x);
    projected = ggml_reshape_3d(ctx, projected, d_inner, n_tokens, 1);
    ggml_tensor * conv_input = ggml_concat(ctx, state,
                                           ggml_transpose(ctx, projected), 0);

    // Drop the oldest row and persist the newest d_conv-1 values.
    if (commit_state) {
        ggml_tensor * newest = ggml_view_3d(ctx, conv_input,
            state_rows, d_inner, 1, conv_input->nb[1], conv_input->nb[2],
            static_cast<size_t>(n_tokens) * conv_input->nb[0]);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, newest, state));
    }

    ggml_tensor * cw = ggml_reshape_2d(ctx, conv_weight, d_conv, d_inner);
    ggml_tensor * out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, cw));
    out = ggml_reshape_4d(ctx, out, head_dim, n_head, n_tokens, 1);
    return out;
}

ggml_tensor * build_kda(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        bool commit_state,
                        bool capture_replay,
                        int replay_token_offset = 0) {
    const int head_dim = w.kda_head_dim;
    const int n_head = w.n_head;
    const int n_tokens = static_cast<int>(cur->ne[1]);
    const int64_t d_inner = static_cast<int64_t>(head_dim) * n_head;

    if (capture_replay) {
        GGML_ASSERT(cache.replay_input != nullptr);
        GGML_ASSERT(replay_token_offset >= 0);
        GGML_ASSERT(replay_token_offset + n_tokens <= cache.replay_input->ne[1]);
        ggml_tensor * replay_dst = ggml_view_2d(
            ctx, cache.replay_input, w.n_embd, n_tokens,
            cache.replay_input->nb[1],
            static_cast<size_t>(replay_token_offset) *
                cache.replay_input->nb[1]);
        ggml_build_forward_expand(graph, ggml_cpy(ctx, cur, replay_dst));
    }

    ggml_tensor * q = kda_conv1d(ctx, graph, cache.conv_state, 0, cur,
        layer.wq, layer.ssm_q_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);
    ggml_tensor * k = kda_conv1d(ctx, graph, cache.conv_state, 1, cur,
        layer.wk, layer.ssm_k_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);
    ggml_tensor * v = kda_conv1d(ctx, graph, cache.conv_state, 2, cur,
        layer.wv, layer.ssm_v_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state);

    ggml_tensor * decay = ggml_mul_mat(ctx, layer.ssm_f_a, cur);
    decay = ggml_mul_mat(ctx, layer.ssm_f_b, decay);
    decay = ggml_add(ctx, decay, layer.ssm_dt_b);
    ggml_tensor * A = ggml_reshape_3d(ctx, layer.ssm_a, 1, n_head, 1);
    if (std::isfinite(w.kda_gate_lower_bound)) {
        decay = ggml_reshape_3d(ctx, decay, head_dim, n_head, n_tokens);
        decay = ggml_mul(ctx, decay, A);
        decay = ggml_sigmoid(ctx, ggml_scale(ctx, decay, -1.0f));
        decay = ggml_scale(ctx, decay, w.kda_gate_lower_bound);
    } else {
        decay = ggml_softplus(ctx, decay);
        decay = ggml_reshape_3d(ctx, decay, head_dim, n_head, n_tokens);
        decay = ggml_mul(ctx, decay, A);
    }
    decay = ggml_reshape_4d(ctx, decay, head_dim, n_head, n_tokens, 1);

    ggml_tensor * beta = ggml_mul_mat(ctx, layer.ssm_beta, cur);
    beta = ggml_sigmoid(ctx,
        ggml_reshape_4d(ctx, beta, 1, n_head, n_tokens, 1));

    q = ggml_l2_norm(ctx, q, w.rms_eps);
    k = ggml_l2_norm(ctx, k, w.rms_eps);
    ggml_tensor * state = ggml_reshape_4d(ctx, cache.ssm_state,
        head_dim, head_dim, n_head, 1);
    ggml_tensor * packed = ggml_gated_delta_net(ctx, q, k, v, decay, beta, state);
    ggml_gated_delta_net_set_skip_intermediate(packed, true);

    const size_t elt = ggml_element_size(packed);
    ggml_tensor * output = ggml_view_4d(ctx, packed,
        head_dim, n_head, n_tokens, 1,
        static_cast<size_t>(head_dim) * elt,
        static_cast<size_t>(head_dim) * n_head * elt,
        static_cast<size_t>(head_dim) * n_head * n_tokens * elt, 0);
    ggml_tensor * new_state = ggml_view_4d(ctx, packed,
        head_dim, head_dim, n_head, 1,
        static_cast<size_t>(head_dim) * elt,
        static_cast<size_t>(head_dim) * head_dim * elt,
        static_cast<size_t>(head_dim) * head_dim * n_head * elt,
        static_cast<size_t>(head_dim) * n_head * n_tokens * elt);
    if (commit_state) {
        ggml_build_forward_expand(graph,
            ggml_cpy(ctx, new_state, cache.ssm_state));
    }

    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ssm_g, cur);
    gate = ggml_reshape_3d(ctx, gate, head_dim, n_head, n_tokens);
    output = ggml_reshape_3d(ctx, output, head_dim, n_head, n_tokens);
    output = rms_norm(ctx, output, layer.ssm_o_norm, w.rms_eps);
    output = ggml_mul(ctx, output, ggml_sigmoid(ctx, gate));
    output = ggml_cont_2d(ctx, output, d_inner, n_tokens);
    return ggml_mul_mat(ctx, layer.wo, output);
}

bool serial_offloaded_moe_rows_enabled();

ggml_tensor * build_mla(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        int position,
                        ggml_tensor * attn_mask) {
    const int n_head = w.n_head;
    const int kv_rank = w.kv_lora_rank;
    const int key_dim = w.mla_k_head_dim;
    const int value_dim = w.mla_v_head_dim;
    const int rope_dim = w.rope_dim;
    const int nope_dim = key_dim - rope_dim;
    const int compact_dim = kv_rank + rope_dim;
    const int n_tokens = static_cast<int>(cur->ne[1]);
    const int kv_len = position + n_tokens;

    ggml_tensor * gate_input = cur;
    ggml_tensor * q_cur = nullptr;
    if (layer.wq_a) {
        q_cur = ggml_mul_mat(ctx, layer.wq_a, cur);
        q_cur = rms_norm(ctx, q_cur, layer.wq_a_norm, w.rms_eps);
        q_cur = ggml_mul_mat(ctx, layer.wq_b, q_cur);
    } else {
        q_cur = ggml_mul_mat(ctx, layer.wq, cur);
    }

    ggml_tensor * compact_pe = ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
    ggml_tensor * compact = ggml_view_2d(ctx, compact_pe, kv_rank, n_tokens,
        ggml_row_size(compact_pe->type, compact_dim), 0);
    ggml_tensor * k_pe = ggml_view_3d(ctx, compact_pe, rope_dim, n_tokens, 1,
        ggml_row_size(compact_pe->type, compact_dim),
        ggml_row_size(compact_pe->type, compact_dim) * n_tokens,
        ggml_row_size(compact_pe->type, kv_rank));
    compact = rms_norm(ctx, compact, layer.wkv_a_norm, w.rms_eps);

    ggml_tensor * q_nope = ggml_view_3d(ctx, q_cur, nope_dim, n_head, n_tokens,
        ggml_row_size(q_cur->type, key_dim),
        ggml_row_size(q_cur->type, key_dim) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx, q_cur, rope_dim, n_head, n_tokens,
        ggml_row_size(q_cur->type, key_dim),
        ggml_row_size(q_cur->type, key_dim) * n_head,
        ggml_row_size(q_cur->type, nope_dim));
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    q_nope = ggml_mul_mat(ctx, layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx, q_nope, 0, 2, 1, 3);
    ggml_tensor * q = ggml_concat(ctx, q_nope, q_pe, 0);

    ggml_tensor * compact_3d =
        ggml_reshape_3d(ctx, compact, kv_rank, n_tokens, 1);
    ggml_tensor * current_k = ggml_concat(ctx, compact_3d, k_pe, 0);

    ggml_tensor * dst = ggml_view_2d(ctx, cache.mla_k,
        compact_dim, n_tokens, cache.mla_k->nb[2],
        static_cast<size_t>(position) * cache.mla_k->nb[2]);
    ggml_build_forward_expand(graph, ggml_cpy(ctx, current_k, dst));

    ggml_tensor * k = ggml_view_3d(ctx, cache.mla_k,
        compact_dim, 1, kv_len, cache.mla_k->nb[1], cache.mla_k->nb[2], 0);
    ggml_tensor * v = ggml_view_3d(ctx, k,
        kv_rank, 1, kv_len, k->nb[1], k->nb[2], 0);

    // Same non-flash absorbed-MLA algebra as llama.cpp. Avoiding flash here
    // is intentional: current upstream K3 support also disables it for this
    // graph, and this path is the numerical oracle for later fused kernels.
    const bool v_trans = v->nb[1] > v->nb[2];
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_soft_max_ext(ctx, scores, attn_mask,
                               1.0f / std::sqrt(static_cast<float>(key_dim)),
                               0.0f);
    if (!v_trans) v = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * out = nullptr;
    if (n_tokens > 1 && serial_offloaded_moe_rows_enabled()) {
        for (int token = 0; token < n_tokens; ++token) {
            const int64_t causal_length = position + token + 1;
            ggml_tensor * v_prefix = ggml_view_2d(
                ctx, v, causal_length, kv_rank, v->nb[1], 0);
            v_prefix = ggml_cont(ctx, v_prefix);
            ggml_tensor * probability_row = ggml_view_3d(
                ctx, scores, causal_length, 1, n_head,
                scores->nb[1], scores->nb[2],
                static_cast<size_t>(token) * scores->nb[1]);
            probability_row = ggml_cont(ctx, probability_row);
            ggml_tensor * row = ggml_mul_mat(
                ctx, v_prefix, probability_row);
            out = out ? ggml_concat(ctx, out, row, 1) : row;
        }
    } else {
        out = ggml_mul_mat(ctx, v, scores);
    }
    out = ggml_mul_mat(ctx, layer.wv_b, out);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out,
                       static_cast<int64_t>(value_dim) * n_head, n_tokens);

    if (layer.wqkv_gate) {
        ggml_tensor * output_gate = ggml_sigmoid(ctx,
            ggml_mul_mat(ctx, layer.wqkv_gate, gate_input));
        out = ggml_mul(ctx, out, output_gate);
    }
    return ggml_mul_mat(ctx, layer.wo, out);
}

TopKMoeRouterResult build_kimi_router(ggml_context * ctx,
                                      ggml_cgraph * graph,
                                      const KimiK3Weights & w,
                                      const KimiK3Layer & layer,
                                      ggml_tensor * cur,
                                      ggml_tensor ** raw_logits = nullptr) {
    const int n_tokens = static_cast<int>(cur->ne[1]);
    ggml_tensor * logits = ggml_mul_mat(ctx, layer.ffn_gate_inp, cur);
    if (raw_logits) *raw_logits = logits;
    TopKMoeRouterResult router;
    if (w.expert_gating_func == 2) {
        router = build_sigmoid_topk_moe_router(ctx, graph, logits,
            layer.ffn_exp_probs_b, w.n_expert, w.n_expert_used, n_tokens,
            w.expert_weights_norm, w.expert_weights_scale, false);
    } else {
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, w.n_expert_used);
        ggml_tensor * probs_3d =
            ggml_reshape_3d(ctx, probs, 1, w.n_expert, n_tokens);
        ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
        weights = ggml_reshape_2d(
            ctx, weights, w.n_expert_used, n_tokens);
        if (w.expert_weights_norm) {
            ggml_tensor * sum = ggml_clamp(ctx, ggml_sum_rows(ctx, weights),
                                           6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, sum);
        }
        if (w.expert_weights_scale != 1.0f) {
            weights = ggml_scale(ctx, weights, w.expert_weights_scale);
        }
        router.selected = selected;
        router.weights_2d = weights;
        router.weights_3d = ggml_reshape_3d(
            ctx, weights, 1, w.n_expert_used, n_tokens);
    }
    return router;
}

ggml_tensor * build_latent_moe(ggml_context * ctx,
                               ggml_cgraph * graph,
                               const KimiK3Weights & w,
                               const KimiK3Layer & layer,
                               ggml_tensor * cur) {
    ggml_tensor * identity = cur;
    ggml_tensor * routed_in = ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
    TopKMoeRouterResult router =
        build_kimi_router(ctx, graph, w, layer, identity);

    ggml_tensor * routed_3d = ggml_reshape_3d(ctx, routed_in,
                                               w.n_expert_latent,
                                               1, cur->ne[1]);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, layer.ffn_gate_exps,
                                         routed_3d, router.selected);
    ggml_tensor * up = ggml_mul_mat_id(ctx, layer.ffn_up_exps,
                                       routed_3d, router.selected);
    ggml_tensor * activated = situ(ctx, gate, up,
                                    w.situ_beta, w.situ_linear_beta);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, layer.ffn_down_exps,
                                            activated, router.selected);
    experts = ggml_mul(ctx, experts, router.weights_3d);
    ggml_tensor * sum_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                 w.n_expert_latent,
                                                 1, cur->ne[1]);
    ggml_tensor * moe = ggml_repeat_back(ctx, experts, sum_shape);
    moe = ggml_reshape_2d(ctx, moe, w.n_expert_latent, cur->ne[1]);
    if (layer.ffn_routed_norm) {
        moe = rms_norm(ctx, moe, layer.ffn_routed_norm, w.rms_eps);
    }
    moe = ggml_mul_mat(ctx, layer.ffn_routed_up, moe);

    ggml_tensor * shared_gate = ggml_mul_mat(ctx, layer.ffn_gate_shexp, identity);
    ggml_tensor * shared_up = ggml_mul_mat(ctx, layer.ffn_up_shexp, identity);
    ggml_tensor * shared = situ(ctx, shared_gate, shared_up,
                                w.situ_beta, w.situ_linear_beta);
    shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);
    return ggml_add(ctx, moe, shared);
}

struct GraphInput {
    ggml_tensor * tensor = nullptr;
    const void * data = nullptr;
    size_t bytes = 0;
    KimiK3RoutedOutputProvider * device_provider = nullptr;
    const ggml_tensor * device_tensor = nullptr;
};

class PendingDeviceOutputGuard {
public:
    explicit PendingDeviceOutputGuard(KimiK3RoutedOutputProvider * provider)
        : provider_(provider) {}
    ~PendingDeviceOutputGuard() {
        if (provider_) provider_->discard_device_output();
    }

private:
    KimiK3RoutedOutputProvider * provider_ = nullptr;
};

struct GraphOutput {
    ggml_tensor * tensor = nullptr;
    void * data = nullptr;
    size_t bytes = 0;
};

bool read_token_embeddings_on_host(
        const KimiK3Weights & w,
        const std::vector<int32_t> & tokens,
        std::vector<float> & hidden) {
    if (!w.tok_embd || w.tok_embd->ne[0] != w.n_embd ||
        w.tok_embd->ne[1] != w.n_vocab ||
        hidden.size() != tokens.size() * static_cast<size_t>(w.n_embd)) {
        set_last_error("Kimi-K3 embedding: invalid host fallback shape");
        return false;
    }

    const ggml_type_traits * traits =
        ggml_get_type_traits(w.tok_embd->type);
    const size_t row_bytes =
        ggml_row_size(w.tok_embd->type, w.n_embd);
    if (!traits || !traits->to_float || row_bytes == 0 ||
        w.tok_embd->nb[1] < row_bytes) {
        set_last_error(std::string("Kimi-K3 embedding: no host decoder for ") +
                       ggml_type_name(w.tok_embd->type));
        return false;
    }

    std::vector<uint8_t> row(row_bytes);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const size_t offset =
            static_cast<size_t>(tokens[i]) * w.tok_embd->nb[1];
        ggml_backend_tensor_get(
            w.tok_embd, row.data(), offset, row.size());
        traits->to_float(
            row.data(),
            hidden.data() + i * static_cast<size_t>(w.n_embd),
            w.n_embd);
    }
    return true;
}

bool run_host_boundary_graph(ggml_backend_t backend,
                             ggml_context * ctx,
                             ggml_cgraph * graph,
                             const std::vector<GraphInput> & inputs,
                             const std::vector<GraphOutput> & outputs,
                             const char * phase) {
    using BoundaryProfileClock = std::chrono::steady_clock;
    const char * boundary_profile_environment =
        std::getenv("DFLASH_KIMI_BOUNDARY_PROFILE");
    const bool profile_boundary = boundary_profile_environment &&
        *boundary_profile_environment &&
        std::strcmp(boundary_profile_environment, "0") != 0;
    const BoundaryProfileClock::time_point boundary_start = profile_boundary
        ? BoundaryProfileClock::now() : BoundaryProfileClock::time_point{};
    BoundaryProfileClock::time_point boundary_mark = boundary_start;
    uint64_t expand_ns = 0;
    uint64_t allocate_ns = 0;
    uint64_t input_ns = 0;
    uint64_t compute_ns = 0;
    uint64_t output_ns = 0;
    const auto boundary_elapsed_ns = [](BoundaryProfileClock::time_point start) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                BoundaryProfileClock::now() - start).count());
    };
    for (const GraphOutput & output : outputs) {
        if (!output.tensor || !output.data || output.bytes == 0) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph output");
            return false;
        }
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
    }
    if (profile_boundary) {
        expand_ns = boundary_elapsed_ns(boundary_mark);
        boundary_mark = BoundaryProfileClock::now();
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph allocation failed");
        if (allocator) ggml_gallocr_free(allocator);
        return false;
    }
    if (profile_boundary) {
        allocate_ns = boundary_elapsed_ns(boundary_mark);
        boundary_mark = BoundaryProfileClock::now();
    }
    for (const GraphInput & input : inputs) {
        const int sources = (input.data ? 1 : 0) +
            (input.device_provider ? 1 : 0) +
            (input.device_tensor ? 1 : 0);
        if (!input.tensor || input.bytes == 0 || sources != 1) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph input");
            ggml_gallocr_free(allocator);
            return false;
        }
        if (input.device_provider) {
            std::string copy_error;
            if (!input.device_provider->copy_device_output(
                    backend, input.tensor, &copy_error)) {
                set_last_error(std::string("Kimi-K3 ") + phase +
                    ": device input copy failed: " + copy_error);
                ggml_gallocr_free(allocator);
                return false;
            }
        } else if (input.device_tensor) {
            if (ggml_nbytes(input.device_tensor) != input.bytes) {
                set_last_error(std::string("Kimi-K3 ") + phase +
                    ": device tensor input shape mismatch");
                ggml_gallocr_free(allocator);
                return false;
            }
            ggml_backend_tensor_copy_async(
                backend, backend, input.device_tensor, input.tensor);
        } else {
            ggml_backend_tensor_set(
                input.tensor, input.data, 0, input.bytes);
        }
    }
    if (profile_boundary) {
        input_ns = boundary_elapsed_ns(boundary_mark);
        boundary_mark = BoundaryProfileClock::now();
    }
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph compute failed with status " +
                       std::to_string(static_cast<int>(status)));
        ggml_gallocr_free(allocator);
        return false;
    }
    if (profile_boundary) {
        compute_ns = boundary_elapsed_ns(boundary_mark);
        boundary_mark = BoundaryProfileClock::now();
    }
    for (const GraphOutput & output : outputs) {
        ggml_backend_tensor_get(
            output.tensor, output.data, 0, output.bytes);
    }
    if (profile_boundary) {
        output_ns = boundary_elapsed_ns(boundary_mark);
        const uint64_t total_ns = boundary_elapsed_ns(boundary_start);
        std::fprintf(stderr,
            "[kimi-k3-boundary] phase=\"%s\" expand_ms=%.3f "
            "allocate_ms=%.3f input_ms=%.3f compute_ms=%.3f "
            "output_ms=%.3f total_ms=%.3f\n",
            phase, expand_ns / 1.0e6, allocate_ns / 1.0e6,
            input_ns / 1.0e6, compute_ns / 1.0e6,
            output_ns / 1.0e6, total_ns / 1.0e6);
    }
    ggml_gallocr_free(allocator);
    return true;
}

ggml_context * new_kimi_step_context();

void populate_attn_res_bank(
    ggml_context * ctx,
    const KimiK3Weights & w,
    int n_tokens,
    const std::vector<std::vector<float>> & host_checkpoints,
    AttnResBank & bank,
    std::vector<GraphInput> & inputs);

bool serial_offloaded_moe_rows_enabled() {
    const char * raw =
        std::getenv("DFLASH_KIMI_S0_SERIAL_CORE_ROWS");
    return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool serial_streamed_expert_rows_enabled() {
    const char * raw =
        std::getenv("DFLASH_KIMI_S0_SERIAL_EXPERT_ROWS");
    return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool run_offloaded_moe_preparation(
        KimiK3MoeCoreOffload & offload,
        const KimiK3Weights & w,
        int model_layer,
        int n_tokens,
        const std::vector<float> & normalized_hidden,
        std::vector<float> & routed_input,
        std::vector<int32_t> & selected,
        std::vector<float> & route_weights,
        std::vector<float> * shared_output,
        std::vector<float> * router_logits_output) {
    if (!offload.enabled() || model_layer < w.n_dense_lead ||
        model_layer >= static_cast<int>(offload.layers.size()) ||
        normalized_hidden.size() !=
            static_cast<size_t>(w.n_embd) * n_tokens) {
        set_last_error("Kimi-K3 accelerator MoE preparation: invalid input");
        return false;
    }
    // S0 diagnostic: replay each row through the exact single-row accelerator
    // graph.  This isolates batch-width arithmetic in the shared latent/router/
    // shared-expert preparation without changing expert selection, weights, or
    // accumulation order.  It is intentionally opt-in until the parity and
    // performance consequences are measured on the real K3 verifier.
    if (n_tokens > 1 && serial_offloaded_moe_rows_enabled()) {
        const size_t hidden_width = static_cast<size_t>(w.n_embd);
        const size_t latent_width = static_cast<size_t>(w.n_expert_latent);
        const size_t route_width = static_cast<size_t>(w.n_expert_used);
        const size_t router_width = static_cast<size_t>(w.n_expert);
        for (int token = 0; token < n_tokens; ++token) {
            const size_t token_index = static_cast<size_t>(token);
            std::vector<float> hidden_row(
                normalized_hidden.begin() + token_index * hidden_width,
                normalized_hidden.begin() + (token_index + 1) * hidden_width);
            std::vector<float> routed_row(latent_width);
            std::vector<int32_t> selected_row(route_width);
            std::vector<float> weights_row(route_width);
            std::vector<float> shared_row;
            std::vector<float> router_logits_row;
            if (shared_output) shared_row.resize(hidden_width);
            if (router_logits_output) router_logits_row.resize(router_width);
            if (!run_offloaded_moe_preparation(
                    offload, w, model_layer, 1, hidden_row, routed_row,
                    selected_row, weights_row,
                    shared_output ? &shared_row : nullptr,
                    offload.router && router_logits_output
                        ? &router_logits_row : nullptr)) {
                return false;
            }
            if (offload.latent) {
                std::copy(
                    routed_row.begin(), routed_row.end(),
                    routed_input.begin() + token_index * latent_width);
            }
            if (offload.router) {
                std::copy(
                    selected_row.begin(), selected_row.end(),
                    selected.begin() + token_index * route_width);
                std::copy(
                    weights_row.begin(), weights_row.end(),
                    route_weights.begin() + token_index * route_width);
            }
            if (shared_output) {
                std::copy(
                    shared_row.begin(), shared_row.end(),
                    shared_output->begin() + token_index * hidden_width);
            }
            if (offload.router && router_logits_output) {
                std::copy(
                    router_logits_row.begin(), router_logits_row.end(),
                    router_logits_output->begin() +
                        token_index * router_width);
            }
        }
        return true;
    }
    const KimiK3MoeCoreOffloadLayer & source =
        offload.layers[static_cast<size_t>(model_layer)];
    KimiK3Layer layer;
    layer.ffn_gate_inp = source.ffn_gate_inp;
    layer.ffn_exp_probs_b = source.ffn_exp_probs_b;
    layer.ffn_routed_down = source.ffn_routed_down;
    layer.ffn_gate_shexp = source.ffn_gate_shexp;
    layer.ffn_up_shexp = source.ffn_up_shexp;
    layer.ffn_down_shexp = source.ffn_down_shexp;

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 accelerator MoE preparation: context failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(hidden_in);
    std::vector<GraphOutput> outputs;
    if (offload.latent) {
        ggml_tensor * routed =
            ggml_mul_mat(ctx, layer.ffn_routed_down, hidden_in);
        outputs.push_back({
            routed, routed_input.data(),
            routed_input.size() * sizeof(float)});
    }
    if (offload.router) {
        ggml_tensor * raw_logits = nullptr;
        TopKMoeRouterResult router = build_kimi_router(
            ctx, graph, w, layer, hidden_in,
            router_logits_output ? &raw_logits : nullptr);
        ggml_tensor * selected_out = ggml_cont(ctx, router.selected);
        ggml_tensor * weights_out = ggml_cont(ctx, router.weights_2d);
        outputs.push_back({
            selected_out, selected.data(),
            selected.size() * sizeof(int32_t)});
        outputs.push_back({
            weights_out, route_weights.data(),
            route_weights.size() * sizeof(float)});
        if (router_logits_output) {
            outputs.push_back({
                raw_logits, router_logits_output->data(),
                router_logits_output->size() * sizeof(float)});
        }
    }
    if (offload.shared && shared_output) {
        ggml_tensor * gate =
            ggml_mul_mat(ctx, layer.ffn_gate_shexp, hidden_in);
        ggml_tensor * up =
            ggml_mul_mat(ctx, layer.ffn_up_shexp, hidden_in);
        ggml_tensor * shared =
            situ(ctx, gate, up, w.situ_beta, w.situ_linear_beta);
        shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);
        outputs.push_back({
            shared, shared_output->data(),
            shared_output->size() * sizeof(float)});
    }
    if (outputs.empty()) {
        ggml_free(ctx);
        set_last_error(
            "Kimi-K3 accelerator MoE preparation: no selected family");
        return false;
    }
    const bool ok = run_host_boundary_graph(
        offload.backend, ctx, graph,
        {{hidden_in, normalized_hidden.data(),
          normalized_hidden.size() * sizeof(float)}},
        outputs, "accelerator MoE preparation");
    ggml_free(ctx);
    return ok;
}

bool run_offloaded_moe_join(
        KimiK3MoeCoreOffload & offload,
        const KimiK3Weights & w,
        int model_layer,
        int n_tokens,
        const std::vector<float> & prefix,
        const std::vector<float> & routed_output,
        const std::vector<float> & shared_output,
        std::vector<float> & hidden_output,
        std::vector<float> * moe_output) {
    if (!offload.join_enabled() || model_layer < w.n_dense_lead ||
        model_layer >= static_cast<int>(offload.layers.size())) {
        set_last_error("Kimi-K3 accelerator MoE join: invalid layer");
        return false;
    }
    if (n_tokens > 1 && serial_offloaded_moe_rows_enabled()) {
        const size_t hidden_width = static_cast<size_t>(w.n_embd);
        const size_t latent_width = static_cast<size_t>(w.n_expert_latent);
        for (int token = 0; token < n_tokens; ++token) {
            const size_t token_index = static_cast<size_t>(token);
            std::vector<float> prefix_row(
                prefix.begin() + token_index * hidden_width,
                prefix.begin() + (token_index + 1) * hidden_width);
            std::vector<float> routed_row(
                routed_output.begin() + token_index * latent_width,
                routed_output.begin() + (token_index + 1) * latent_width);
            std::vector<float> shared_row(
                shared_output.begin() + token_index * hidden_width,
                shared_output.begin() + (token_index + 1) * hidden_width);
            std::vector<float> hidden_row(hidden_width);
            std::vector<float> moe_row;
            if (moe_output) moe_row.resize(hidden_width);
            if (!run_offloaded_moe_join(
                    offload, w, model_layer, 1, prefix_row, routed_row,
                    shared_row, hidden_row,
                    moe_output ? &moe_row : nullptr)) {
                return false;
            }
            std::copy(
                hidden_row.begin(), hidden_row.end(),
                hidden_output.begin() + token_index * hidden_width);
            if (moe_output) {
                std::copy(
                    moe_row.begin(), moe_row.end(),
                    moe_output->begin() + token_index * hidden_width);
            }
        }
        return true;
    }
    const KimiK3MoeCoreOffloadLayer & layer =
        offload.layers[static_cast<size_t>(model_layer)];
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 accelerator MoE join: context failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * prefix_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_tensor * routed_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_expert_latent, n_tokens);
    ggml_tensor * shared_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(prefix_in);
    ggml_set_input(routed_in);
    ggml_set_input(shared_in);
    ggml_tensor * routed = routed_in;
    if (layer.ffn_routed_norm) {
        routed = rms_norm(
            ctx, routed, layer.ffn_routed_norm, w.rms_eps);
    }
    routed = ggml_mul_mat(ctx, layer.ffn_routed_up, routed);
    ggml_tensor * combined = ggml_add(ctx, routed, shared_in);
    ggml_tensor * hidden = ggml_add(ctx, prefix_in, combined);
    std::vector<GraphOutput> outputs = {{
        hidden, hidden_output.data(), hidden_output.size() * sizeof(float)}};
    if (moe_output) {
        outputs.push_back({
            combined, moe_output->data(), moe_output->size() * sizeof(float)});
    }
    const bool ok = run_host_boundary_graph(
        offload.backend, ctx, graph,
        {
            {prefix_in, prefix.data(), prefix.size() * sizeof(float)},
            {routed_in, routed_output.data(),
             routed_output.size() * sizeof(float)},
            {shared_in, shared_output.data(),
             shared_output.size() * sizeof(float)},
        },
        outputs, "accelerator MoE join");
    ggml_free(ctx);
    return ok;
}

bool run_offloaded_complete_preparation(
        KimiK3MoeCoreOffload & offload,
        const KimiK3Weights & w,
        int model_layer,
        const std::vector<float> & hidden,
        const std::vector<std::vector<float>> & checkpoints,
        bool banked,
        std::vector<float> & prefix_output,
        std::vector<float> & normalized_hidden_output,
        std::vector<float> & routed_input,
        std::vector<int32_t> & selected,
        std::vector<float> & route_weights,
        std::vector<float> & shared_output) {
    if (!offload.complete_preparation_enabled(model_layer) ||
        hidden.size() != static_cast<size_t>(w.n_embd)) {
        set_last_error("Kimi-K3 complete accelerator preparation: invalid input");
        return false;
    }
    const KimiK3MoeCoreOffloadLayer & source =
        offload.layers[static_cast<size_t>(model_layer)];
    KimiK3Layer layer;
    layer.recurrent = true;
    layer.attn_norm = source.attn_norm;
    layer.ffn_norm = source.ffn_norm;
    layer.attn_res_score = source.attn_res_score;
    layer.ffn_res_score = source.ffn_res_score;
    layer.wq = source.wq;
    layer.wk = source.wk;
    layer.wv = source.wv;
    layer.wo = source.wo;
    layer.ssm_q_conv = source.ssm_q_conv;
    layer.ssm_k_conv = source.ssm_k_conv;
    layer.ssm_v_conv = source.ssm_v_conv;
    layer.ssm_f_a = source.ssm_f_a;
    layer.ssm_f_b = source.ssm_f_b;
    layer.ssm_beta = source.ssm_beta;
    layer.ssm_a = source.ssm_a;
    layer.ssm_dt_b = source.ssm_dt_b;
    layer.ssm_g = source.ssm_g;
    layer.ssm_o_norm = source.ssm_o_norm;
    layer.ffn_gate_inp = source.ffn_gate_inp;
    layer.ffn_exp_probs_b = source.ffn_exp_probs_b;
    layer.ffn_routed_down = source.ffn_routed_down;
    layer.ffn_gate_shexp = source.ffn_gate_shexp;
    layer.ffn_up_shexp = source.ffn_up_shexp;
    layer.ffn_down_shexp = source.ffn_down_shexp;
    KimiK3LayerCache cache;
    cache.conv_state = source.conv_state;
    cache.ssm_state = source.ssm_state;

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 complete accelerator preparation: context failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(hidden_in);
    inputs.push_back({
        hidden_in, hidden.data(), hidden.size() * sizeof(float)});
    AttnResBank residuals;
    populate_attn_res_bank(ctx, w, 1, checkpoints, residuals, inputs);
    ggml_tensor * prefix = hidden_in;
    ggml_tensor * cur = residuals.mix(prefix, layer.attn_res_score);
    if (banked) residuals.push(prefix);
    cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
    cur = build_kda(
        ctx, graph, w, layer, cache, cur,
        /*commit_state=*/true, /*capture_replay=*/false);
    prefix = banked ? cur : ggml_add(ctx, prefix, cur);
    cur = residuals.mix(prefix, layer.ffn_res_score);
    cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);
    ggml_tensor * routed =
        ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
    TopKMoeRouterResult router =
        build_kimi_router(ctx, graph, w, layer, cur);
    ggml_tensor * selected_out = ggml_cont(ctx, router.selected);
    ggml_tensor * weights_out = ggml_cont(ctx, router.weights_2d);
    ggml_tensor * shared_gate =
        ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
    ggml_tensor * shared_up =
        ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
    ggml_tensor * shared = situ(
        ctx, shared_gate, shared_up,
        w.situ_beta, w.situ_linear_beta);
    shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);
    const bool ok = run_host_boundary_graph(
        offload.backend, ctx, graph, inputs,
        {
            {prefix, prefix_output.data(),
             prefix_output.size() * sizeof(float)},
            {cur, normalized_hidden_output.data(),
             normalized_hidden_output.size() * sizeof(float)},
            {routed, routed_input.data(),
             routed_input.size() * sizeof(float)},
            {selected_out, selected.data(),
             selected.size() * sizeof(int32_t)},
            {weights_out, route_weights.data(),
             route_weights.size() * sizeof(float)},
            {shared, shared_output.data(),
             shared_output.size() * sizeof(float)},
        },
        "complete accelerator routed preparation");
    ggml_free(ctx);
    return ok;
}

void populate_attn_res_bank(
        ggml_context * ctx,
        const KimiK3Weights & w,
        int n_tokens,
        const std::vector<std::vector<float>> & host_checkpoints,
        AttnResBank & bank,
        std::vector<GraphInput> & inputs) {
    bank.ctx = ctx;
    bank.eps = w.rms_eps;
    bank.n_embd = w.n_embd;
    bank.n_tokens = n_tokens;
    for (const std::vector<float> & checkpoint : host_checkpoints) {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
        ggml_set_input(tensor);
        inputs.push_back({
            tensor, checkpoint.data(),
            checkpoint.size() * sizeof(float)});
        bank.push(tensor);
    }
}

ggml_context * new_kimi_step_context() {
    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.no_alloc = true;
    return ggml_init(params);
}

ggml_context * new_kimi_persistent_context() {
    ggml_init_params params{};
    params.mem_size = 2ull * 1024ull * 1024ull;
    params.no_alloc = true;
    return ggml_init(params);
}

struct PersistentRoutedGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden = nullptr;
    std::vector<ggml_tensor *> checkpoints;
    ggml_tensor * prefix = nullptr;
    ggml_tensor * routed = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * shared = nullptr;
    int checkpoint_count = 0;
    uint64_t executions = 0;
};

struct PersistentRoutedDeviceOutputs {
    const ggml_tensor * prefix = nullptr;
    const ggml_tensor * routed = nullptr;
    const ggml_tensor * shared = nullptr;
};

struct PersistentRoutedJoinGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * prefix = nullptr;
    ggml_tensor * routed = nullptr;
    ggml_tensor * shared = nullptr;
    ggml_tensor * hidden = nullptr;
    uint64_t executions = 0;
};

bool build_persistent_routed_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        KimiK3LayerCache & layer_cache,
        const std::vector<ggml_tensor *> & shared_checkpoints,
        int checkpoint_count,
        bool banked,
        PersistentRoutedGraph & out) {
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 32768, false);
    out.hidden = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, 1);
    if (!out.graph || !out.hidden) return false;
    ggml_set_input(out.hidden);
    out.checkpoint_count = checkpoint_count;

    AttnResBank residuals;
    residuals.ctx = out.ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = 1;
    if (checkpoint_count < 0 ||
        shared_checkpoints.size() < static_cast<size_t>(checkpoint_count)) {
        return false;
    }
    out.checkpoints.reserve(static_cast<size_t>(checkpoint_count));
    for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
        ggml_tensor * tensor =
            shared_checkpoints[static_cast<size_t>(checkpoint)];
        out.checkpoints.push_back(tensor);
        residuals.push(tensor);
    }

    out.prefix = out.hidden;
    ggml_tensor * cur = residuals.mix(
        out.prefix, layer.attn_res_score);
    if (banked) residuals.push(out.prefix);
    cur = rms_norm(out.ctx, cur, layer.attn_norm, w.rms_eps);
    cur = build_kda(
        out.ctx, out.graph, w, layer, layer_cache, cur,
        /*commit_state=*/true, /*capture_replay=*/false);
    out.prefix = banked ? cur : ggml_add(out.ctx, out.prefix, cur);
    cur = residuals.mix(out.prefix, layer.ffn_res_score);
    cur = rms_norm(out.ctx, cur, layer.ffn_norm, w.rms_eps);

    out.routed = ggml_mul_mat(out.ctx, layer.ffn_routed_down, cur);
    TopKMoeRouterResult router = build_kimi_router(
        out.ctx, out.graph, w, layer, cur);
    out.selected = ggml_cont(out.ctx, router.selected);
    out.route_weights = ggml_cont(out.ctx, router.weights_2d);
    ggml_tensor * shared_gate =
        ggml_mul_mat(out.ctx, layer.ffn_gate_shexp, cur);
    ggml_tensor * shared_up =
        ggml_mul_mat(out.ctx, layer.ffn_up_shexp, cur);
    out.shared = situ(
        out.ctx, shared_gate, shared_up,
        w.situ_beta, w.situ_linear_beta);
    out.shared = ggml_mul_mat(
        out.ctx, layer.ffn_down_shexp, out.shared);

    for (ggml_tensor * output : {
             out.prefix, out.routed, out.selected,
             out.route_weights, out.shared}) {
        if (!output) return false;
        ggml_set_output(output);
        ggml_build_forward_expand(out.graph, output);
    }
    return true;
}

bool build_persistent_routed_join_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        PersistentRoutedJoinGraph & out) {
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 4096, false);
    out.prefix = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, 1);
    out.routed = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_expert_latent, 1);
    out.shared = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, 1);
    if (!out.graph || !out.prefix || !out.routed || !out.shared) {
        return false;
    }
    ggml_set_input(out.prefix);
    ggml_set_input(out.routed);
    ggml_set_input(out.shared);
    ggml_tensor * routed = out.routed;
    if (layer.ffn_routed_norm) {
        routed = rms_norm(
            out.ctx, routed, layer.ffn_routed_norm, w.rms_eps);
    }
    routed = ggml_mul_mat(out.ctx, layer.ffn_routed_up, routed);
    ggml_tensor * moe_shared = ggml_add(out.ctx, routed, out.shared);
    out.hidden = ggml_add(out.ctx, out.prefix, moe_shared);
    if (!out.hidden) return false;
    ggml_set_output(out.hidden);
    ggml_build_forward_expand(out.graph, out.hidden);
    return true;
}

class PersistentRoutedPreparation {
public:
    ~PersistentRoutedPreparation() {
        if (backend_) ggml_backend_synchronize(backend_);
        if (backend_) {
            std::fprintf(stderr,
                "[kimi-k3-p46] finalized graphs=%zu executions=%llu "
                "workspace-bytes=%zu metadata-bytes=%zu "
                "join-graphs=%zu join-executions=%llu "
                "join-workspace-bytes=%zu\n",
                graph_count_,
                static_cast<unsigned long long>(executions_),
                workspace_bytes_, metadata_bytes_, join_graph_count_,
                static_cast<unsigned long long>(join_executions_),
                join_workspace_bytes_);
        }
        if (join_allocator_) ggml_gallocr_free(join_allocator_);
        for (PersistentRoutedJoinGraph & entry : join_entries_) {
            if (entry.ctx) ggml_free(entry.ctx);
        }
        if (allocator_) ggml_gallocr_free(allocator_);
        for (PersistentRoutedGraph & entry : entries_) {
            if (entry.ctx) ggml_free(entry.ctx);
        }
        if (checkpoint_buffer_) {
            ggml_backend_buffer_free(checkpoint_buffer_);
        }
        if (checkpoint_ctx_) ggml_free(checkpoint_ctx_);
    }

    bool initialize(
            ggml_backend_t backend,
            const KimiK3Weights & w,
            KimiK3Cache & cache,
            bool persistent_join,
            std::string * error) {
        if (backend_ || !backend || w.n_layer <= 0 ||
            w.attn_res_block_size <= 0 ||
            static_cast<int>(cache.layers.size()) != w.n_layer) {
            return fail(error, "invalid persistent routed-preparation state");
        }
        ggml_backend_dev_t device = ggml_backend_get_device(backend);
        if (!device || (ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_GPU &&
                ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_IGPU)) {
            return fail(error,
                "P46 persistent routed preparation requires CUDA or HIP");
        }
        backend_ = backend;
        weights_ = &w;
        entries_.resize(static_cast<size_t>(w.n_layer));
        persistent_join_ = persistent_join;
        if (persistent_join_) {
            join_entries_.resize(static_cast<size_t>(w.n_layer));
        }

        checkpoint_ctx_ = new_kimi_persistent_context();
        const int checkpoint_capacity =
            (w.n_layer + w.attn_res_block_size - 1) /
            w.attn_res_block_size;
        if (!checkpoint_ctx_ || checkpoint_capacity <= 0) {
            return fail(error,
                "cannot create persistent routed checkpoint metadata");
        }
        checkpoints_.reserve(static_cast<size_t>(checkpoint_capacity));
        for (int checkpoint = 0; checkpoint < checkpoint_capacity;
                ++checkpoint) {
            ggml_tensor * tensor = ggml_new_tensor_2d(
                checkpoint_ctx_, GGML_TYPE_F32, w.n_embd, 1);
            if (!tensor) {
                return fail(error,
                    "cannot create persistent routed checkpoint tensor");
            }
            ggml_set_input(tensor);
            checkpoints_.push_back(tensor);
        }
        checkpoint_buffer_ = ggml_backend_alloc_ctx_tensors(
            checkpoint_ctx_, backend_);
        if (!checkpoint_buffer_) {
            return fail(error,
                "cannot allocate persistent routed checkpoint buffer");
        }
        metadata_bytes_ += ggml_used_mem(checkpoint_ctx_);

        ggml_gallocr_t measure = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!measure) {
            return fail(error,
                "cannot create persistent routed-preparation measure allocator");
        }
        PersistentRoutedGraph * largest = nullptr;
        size_t largest_bytes = 0;
        for (int il = w.n_dense_lead; il < w.n_layer; ++il) {
            const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
            if (!layer.recurrent) continue;
            PersistentRoutedGraph & entry = entries_[static_cast<size_t>(il)];
            const int checkpoint_count =
                (il + w.attn_res_block_size - 1) /
                w.attn_res_block_size;
            if (!build_persistent_routed_graph(
                    w, layer, cache.layers[static_cast<size_t>(il)],
                    checkpoints_,
                    checkpoint_count,
                    il % w.attn_res_block_size == 0, entry)) {
                ggml_gallocr_free(measure);
                return fail(error,
                    "cannot build persistent routed-preparation graph for layer " +
                    std::to_string(il));
            }
            size_t required = 0;
            ggml_gallocr_reserve_n_size(
                measure, entry.graph, nullptr, nullptr, &required);
            if (required > largest_bytes) {
                largest_bytes = required;
                largest = &entry;
            }
            metadata_bytes_ += ggml_used_mem(entry.ctx);
            ++graph_count_;
        }
        ggml_gallocr_free(measure);
        if (!largest || graph_count_ == 0 || largest_bytes == 0) {
            return fail(error,
                "persistent routed preparation found no recurrent routed graph");
        }

        allocator_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!allocator_ ||
            !ggml_gallocr_reserve(allocator_, largest->graph)) {
            return fail(error,
                "cannot reserve persistent routed-preparation workspace");
        }
        const size_t reserved =
            ggml_gallocr_get_buffer_size(allocator_, 0);
        for (PersistentRoutedGraph & entry : entries_) {
            if (!entry.graph) continue;
            if (!ggml_gallocr_alloc_graph(allocator_, entry.graph) ||
                ggml_gallocr_get_buffer_size(allocator_, 0) != reserved) {
                return fail(error,
                    "persistent routed-preparation workspace changed while "
                    "allocating immutable graphs");
            }
        }
        workspace_bytes_ = reserved;

        if (persistent_join_) {
            ggml_gallocr_t join_measure = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!join_measure) {
                return fail(error,
                    "cannot create persistent routed-join measure allocator");
            }
            PersistentRoutedJoinGraph * largest_join = nullptr;
            size_t largest_join_bytes = 0;
            for (int il = w.n_dense_lead; il < w.n_layer; ++il) {
                PersistentRoutedJoinGraph & entry =
                    join_entries_[static_cast<size_t>(il)];
                if (!build_persistent_routed_join_graph(
                        w, w.layers[static_cast<size_t>(il)], entry)) {
                    ggml_gallocr_free(join_measure);
                    return fail(error,
                        "cannot build persistent routed-join graph for layer " +
                        std::to_string(il));
                }
                size_t required = 0;
                ggml_gallocr_reserve_n_size(
                    join_measure, entry.graph, nullptr, nullptr, &required);
                if (required > largest_join_bytes) {
                    largest_join_bytes = required;
                    largest_join = &entry;
                }
                metadata_bytes_ += ggml_used_mem(entry.ctx);
                ++join_graph_count_;
            }
            ggml_gallocr_free(join_measure);
            if (!largest_join || join_graph_count_ == 0 ||
                largest_join_bytes == 0) {
                return fail(error,
                    "persistent routed join found no routed graph");
            }
            join_allocator_ = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!join_allocator_ || !ggml_gallocr_reserve(
                    join_allocator_, largest_join->graph)) {
                return fail(error,
                    "cannot reserve persistent routed-join workspace");
            }
            const size_t join_reserved =
                ggml_gallocr_get_buffer_size(join_allocator_, 0);
            for (PersistentRoutedJoinGraph & entry : join_entries_) {
                if (!entry.graph) continue;
                if (!ggml_gallocr_alloc_graph(join_allocator_, entry.graph) ||
                    ggml_gallocr_get_buffer_size(
                        join_allocator_, 0) != join_reserved) {
                    return fail(error,
                        "persistent routed-join workspace changed while "
                        "allocating immutable graphs");
                }
            }
            join_workspace_bytes_ = join_reserved;
        }
        std::fprintf(stderr,
            "[kimi-k3-p46] initialized graphs=%zu workspace-bytes=%zu "
            "metadata-bytes=%zu join-graphs=%zu "
            "join-workspace-bytes=%zu backend=%s\n",
            graph_count_, workspace_bytes_, metadata_bytes_, join_graph_count_,
            join_workspace_bytes_,
            ggml_backend_name(backend_));
        return true;
    }

    bool matches(
            ggml_backend_t backend,
            const KimiK3Weights & w,
            bool persistent_join) const {
        return backend_ == backend && weights_ == &w &&
            persistent_join_ == persistent_join;
    }

    void begin_forward() {
        checkpoint_updates_ = 0;
    }

    bool update_checkpoint(
            const std::vector<float> & values,
            std::string * error) {
        if (!backend_ || !weights_ ||
            checkpoint_updates_ >= checkpoints_.size() ||
            values.size() != static_cast<size_t>(weights_->n_embd)) {
            return fail(error,
                "persistent routed checkpoint update is invalid");
        }
        ggml_backend_tensor_set_async(
            backend_, checkpoints_[checkpoint_updates_], values.data(), 0,
            values.size() * sizeof(float));
        ++checkpoint_updates_;
        return true;
    }

    bool evaluate(
            int model_layer,
            const std::vector<float> & hidden,
            const ggml_tensor * hidden_device,
            const std::vector<std::vector<float>> & checkpoints,
            std::vector<float> & prefix,
            std::vector<float> & routed,
            std::vector<int32_t> & selected,
            std::vector<float> & route_weights,
            std::vector<float> & shared,
            PersistentRoutedDeviceOutputs * device_outputs,
            std::string * error) {
        if (device_outputs) *device_outputs = {};
        if (!backend_ || !weights_ || model_layer < 0 ||
            model_layer >= static_cast<int>(entries_.size())) {
            return fail(error, "invalid persistent routed-preparation request");
        }
        PersistentRoutedGraph & entry =
            entries_[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        if (!entry.graph || hidden.size() != static_cast<size_t>(w.n_embd) ||
            checkpoints.size() != entry.checkpoints.size() ||
            checkpoint_updates_ < entry.checkpoints.size() ||
            prefix.size() != static_cast<size_t>(w.n_embd) ||
            routed.size() != static_cast<size_t>(w.n_expert_latent) ||
            selected.size() != static_cast<size_t>(w.n_expert_used) ||
            route_weights.size() != static_cast<size_t>(w.n_expert_used) ||
            shared.size() != static_cast<size_t>(w.n_embd)) {
            return fail(error,
                "persistent routed-preparation shape mismatch at layer " +
                std::to_string(model_layer));
        }
        const size_t hidden_bytes = hidden.size() * sizeof(float);
        if (hidden_device) {
            if (ggml_nbytes(hidden_device) != hidden_bytes) {
                return fail(error,
                    "persistent routed-preparation device hidden shape "
                    "mismatch");
            }
            ggml_backend_tensor_copy_async(
                backend_, backend_, hidden_device, entry.hidden);
        } else {
            ggml_backend_tensor_set_async(
                backend_, entry.hidden, hidden.data(), 0, hidden_bytes);
        }
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            if (checkpoints[i].size() != static_cast<size_t>(w.n_embd)) {
                return fail(error,
                    "persistent routed-preparation checkpoint shape mismatch");
            }
        }
        ScopedCudaGraphOverrides replay_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute_async(backend_, entry.graph) !=
                GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(backend_);
            return fail(error,
                "persistent routed-preparation graph compute failed at layer " +
                std::to_string(model_layer));
        }
        if (!device_outputs) {
            ggml_backend_tensor_get_async(
                backend_, entry.prefix, prefix.data(), 0,
                prefix.size() * sizeof(float));
        }
        if (!device_outputs) {
            ggml_backend_tensor_get_async(
                backend_, entry.routed, routed.data(), 0,
                routed.size() * sizeof(float));
        }
        ggml_backend_tensor_get_async(
            backend_, entry.selected, selected.data(), 0,
            selected.size() * sizeof(int32_t));
        ggml_backend_tensor_get_async(
            backend_, entry.route_weights, route_weights.data(), 0,
            route_weights.size() * sizeof(float));
        if (!device_outputs) {
            ggml_backend_tensor_get_async(
                backend_, entry.shared, shared.data(), 0,
                shared.size() * sizeof(float));
        }
        ggml_backend_synchronize(backend_);
        if (device_outputs) {
            device_outputs->prefix = entry.prefix;
            device_outputs->routed = entry.routed;
            device_outputs->shared = entry.shared;
        }
        ++entry.executions;
        ++executions_;
        return true;
    }

    bool evaluate_join(
            int model_layer,
            const std::vector<float> & prefix_host,
            const ggml_tensor * prefix_device,
            KimiK3RoutedOutputProvider & routed_provider,
            const std::vector<float> & shared_host,
            const ggml_tensor * shared_device,
            std::vector<float> & hidden,
            const ggml_tensor ** hidden_device,
            std::string * error) {
        if (hidden_device) *hidden_device = nullptr;
        if (!persistent_join_ || !backend_ || !weights_ ||
            model_layer < 0 ||
            model_layer >= static_cast<int>(join_entries_.size())) {
            return fail(error, "invalid persistent routed-join request");
        }
        PersistentRoutedJoinGraph & entry =
            join_entries_[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        const size_t hidden_bytes =
            static_cast<size_t>(w.n_embd) * sizeof(float);
        const size_t routed_bytes =
            static_cast<size_t>(w.n_expert_latent) * sizeof(float);
        if (!entry.graph || !entry.prefix || !entry.routed ||
            !entry.shared || !entry.hidden ||
            prefix_host.size() != static_cast<size_t>(w.n_embd) ||
            shared_host.size() != static_cast<size_t>(w.n_embd) ||
            hidden.size() != static_cast<size_t>(w.n_embd) ||
            (prefix_device && ggml_nbytes(prefix_device) != hidden_bytes) ||
            (shared_device && ggml_nbytes(shared_device) != hidden_bytes) ||
            ggml_nbytes(entry.routed) != routed_bytes) {
            return fail(error,
                "persistent routed-join shape mismatch at layer " +
                std::to_string(model_layer));
        }
        if (prefix_device) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, prefix_device, entry.prefix);
        } else {
            ggml_backend_tensor_set_async(
                backend_, entry.prefix, prefix_host.data(), 0, hidden_bytes);
        }
        std::string copy_error;
        if (!routed_provider.copy_device_output(
                backend_, entry.routed, &copy_error)) {
            ggml_backend_synchronize(backend_);
            return fail(error,
                "persistent routed-join expert copy failed: " + copy_error);
        }
        if (shared_device) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, shared_device, entry.shared);
        } else {
            ggml_backend_tensor_set_async(
                backend_, entry.shared, shared_host.data(), 0, hidden_bytes);
        }
        ScopedCudaGraphOverrides replay_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute_async(
                backend_, entry.graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(backend_);
            return fail(error,
                "persistent routed-join graph compute failed at layer " +
                std::to_string(model_layer));
        }
        if (!hidden_device) {
            ggml_backend_tensor_get_async(
                backend_, entry.hidden, hidden.data(), 0, hidden_bytes);
        }
        ggml_backend_synchronize(backend_);
        if (hidden_device) *hidden_device = entry.hidden;
        ++entry.executions;
        ++join_executions_;
        return true;
    }

private:
    static bool fail(std::string * error, const std::string & message) {
        if (error) *error = message;
        return false;
    }

    ggml_backend_t backend_ = nullptr;
    const KimiK3Weights * weights_ = nullptr;
    bool persistent_join_ = false;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_gallocr_t join_allocator_ = nullptr;
    ggml_context * checkpoint_ctx_ = nullptr;
    ggml_backend_buffer_t checkpoint_buffer_ = nullptr;
    std::vector<ggml_tensor *> checkpoints_;
    size_t checkpoint_updates_ = 0;
    std::vector<PersistentRoutedGraph> entries_;
    std::vector<PersistentRoutedJoinGraph> join_entries_;
    size_t graph_count_ = 0;
    size_t join_graph_count_ = 0;
    size_t workspace_bytes_ = 0;
    size_t join_workspace_bytes_ = 0;
    size_t metadata_bytes_ = 0;
    uint64_t executions_ = 0;
    uint64_t join_executions_ = 0;
};

bool parse_strict_binary_environment(
        const char * name,
        bool & enabled,
        std::string * error) {
    const char * raw = std::getenv(name);
    if (!raw || !*raw || std::strcmp(raw, "0") == 0) {
        enabled = false;
        return true;
    }
    if (std::strcmp(raw, "1") == 0) {
        enabled = true;
        return true;
    }
    if (error) *error = std::string(name) + " must be 0 or 1";
    return false;
}

void free_persistent_routed_preparation(void *& opaque) {
    delete static_cast<PersistentRoutedPreparation *>(opaque);
    opaque = nullptr;
}

class ExactMultirowSnapshotGuard {
public:
    ExactMultirowSnapshotGuard(ggml_backend_t backend, KimiK3Cache & cache)
        : backend_(backend), cache_(cache) {
        cache_.recurrent_state_pristine = false;
    }

    ~ExactMultirowSnapshotGuard() { restore(); }

    void restore() {
        if (!active_) return;
        for (KimiK3LayerCache & layer : cache_.layers) {
            if (!layer.ssm_state) continue;
            GGML_ASSERT(layer.ssm_state_snap && layer.conv_state_snap);
            ggml_backend_tensor_copy_async(
                backend_, backend_, layer.ssm_state_snap, layer.ssm_state);
            ggml_backend_tensor_copy_async(
                backend_, backend_, layer.conv_state_snap, layer.conv_state);
        }
        ggml_backend_synchronize(backend_);
        cache_.recurrent_state_pristine = true;
        active_ = false;
    }

private:
    ggml_backend_t backend_ = nullptr;
    KimiK3Cache & cache_;
    bool active_ = true;
};

struct ExactMultirowLayerRow {
    std::vector<float> hidden;
    std::vector<float> prefix;
    std::vector<float> routed;
    std::vector<int32_t> selected;
    std::vector<float> route_weights;
    std::vector<float> shared;
};

bool exact_multirow_embedding_row(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        int32_t token,
        float * output) {
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 P58 embedding: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(id);
    ggml_tensor * embedding = ggml_get_rows(ctx, w.tok_embd, id);
    bool ok = false;
    if (ggml_backend_supports_op(backend, embedding)) {
        ok = run_host_boundary_graph(
            backend, ctx, graph,
            {{id, &token, sizeof(token)}},
            {{embedding, output,
              static_cast<size_t>(w.n_embd) * sizeof(float)}},
            "P58 one-row embedding");
    } else {
        std::vector<float> row(static_cast<size_t>(w.n_embd));
        ok = read_token_embeddings_on_host(w, {token}, row) &&
            row.size() == static_cast<size_t>(w.n_embd);
        if (ok) std::copy(row.begin(), row.end(), output);
    }
    ggml_free(ctx);
    return ok;
}

bool exact_multirow_layer_row(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        int model_layer,
        int base_pos,
        int token_index,
        const float * hidden_row,
        const std::vector<std::vector<float>> & checkpoints,
        ExactMultirowLayerRow & output) {
    const KimiK3Layer & layer = w.layers[static_cast<size_t>(model_layer)];
    KimiK3LayerCache & layer_cache =
        cache.layers[static_cast<size_t>(model_layer)];
    const bool dense = model_layer < w.n_dense_lead;
    const bool banked = model_layer % w.attn_res_block_size == 0;
    const size_t hidden_width = static_cast<size_t>(w.n_embd);

    std::vector<std::vector<float>> checkpoint_rows;
    checkpoint_rows.reserve(checkpoints.size());
    for (const std::vector<float> & checkpoint : checkpoints) {
        const size_t begin = static_cast<size_t>(token_index) * hidden_width;
        if (checkpoint.size() < begin + hidden_width) {
            set_last_error("Kimi-K3 P58 checkpoint shape is invalid");
            return false;
        }
        checkpoint_rows.emplace_back(
            checkpoint.begin() + static_cast<std::ptrdiff_t>(begin),
            checkpoint.begin() +
                static_cast<std::ptrdiff_t>(begin + hidden_width));
    }

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 P58 layer: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(hidden_in);
    inputs.push_back({hidden_in, hidden_row, hidden_width * sizeof(float)});

    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, 1, checkpoint_rows, residuals, inputs);
    ggml_tensor * prefix = hidden_in;
    ggml_tensor * cur = residuals.mix(prefix, layer.attn_res_score);
    if (banked) residuals.push(prefix);
    cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
    std::vector<float> mla_mask_values;
    if (layer.recurrent) {
        cur = build_kda(
            ctx, graph, w, layer, layer_cache, cur,
            /*commit_state=*/true,
            /*capture_replay=*/true, token_index);
    } else {
        const int position = base_pos + token_index;
        mla_mask_values.assign(
            static_cast<size_t>(position + 1), 0.0f);
        ggml_tensor * mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, position + 1, 1);
        ggml_set_input(mask);
        inputs.push_back({
            mask, mla_mask_values.data(),
            mla_mask_values.size() * sizeof(float)});
        cur = build_mla(
            ctx, graph, w, layer, layer_cache, cur, position, mask);
    }
    prefix = banked ? cur : ggml_add(ctx, prefix, cur);
    cur = residuals.mix(prefix, layer.ffn_res_score);
    cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);

    std::vector<GraphOutput> outputs;
    if (dense) {
        ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
        ggml_tensor * dense_output = situ(
            ctx, gate, up, w.situ_beta, w.situ_linear_beta);
        dense_output = ggml_mul_mat(ctx, layer.ffn_down, dense_output);
        ggml_tensor * hidden_out = ggml_add(ctx, prefix, dense_output);
        output.hidden.resize(hidden_width);
        outputs.push_back({
            hidden_out, output.hidden.data(), hidden_width * sizeof(float)});
    } else {
        ggml_tensor * routed =
            ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
        TopKMoeRouterResult router =
            build_kimi_router(ctx, graph, w, layer, cur);
        ggml_tensor * selected = ggml_cont(ctx, router.selected);
        ggml_tensor * route_weights = ggml_cont(ctx, router.weights_2d);
        ggml_tensor * shared_gate =
            ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
        ggml_tensor * shared_up =
            ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
        ggml_tensor * shared = situ(
            ctx, shared_gate, shared_up,
            w.situ_beta, w.situ_linear_beta);
        shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);

        output.prefix.resize(hidden_width);
        output.routed.resize(static_cast<size_t>(w.n_expert_latent));
        output.selected.resize(static_cast<size_t>(w.n_expert_used));
        output.route_weights.resize(static_cast<size_t>(w.n_expert_used));
        output.shared.resize(hidden_width);
        outputs = {
            {prefix, output.prefix.data(), hidden_width * sizeof(float)},
            {routed, output.routed.data(),
             output.routed.size() * sizeof(float)},
            {selected, output.selected.data(),
             output.selected.size() * sizeof(int32_t)},
            {route_weights, output.route_weights.data(),
             output.route_weights.size() * sizeof(float)},
            {shared, output.shared.data(), hidden_width * sizeof(float)},
        };
    }

    if (!layer.recurrent &&
        (mla_mask_values.empty() || inputs.empty() ||
         inputs.back().data != mla_mask_values.data() ||
         inputs.back().bytes !=
             mla_mask_values.size() * sizeof(float))) {
        ggml_free(ctx);
        set_last_error("Kimi-K3 P58 MLA mask lifetime invariant failed");
        return false;
    }
    const bool ok = run_host_boundary_graph(
        backend, ctx, graph, inputs, outputs,
        dense ? "P58 one-row dense layer" :
                "P58 one-row routed preparation");
    ggml_free(ctx);
    return ok;
}

bool exact_multirow_join_row(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        const float * prefix,
        const float * routed_output,
        const float * shared,
        float * hidden_output) {
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 P58 join: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_tensor * prefix_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_tensor * routed_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_expert_latent, 1);
    ggml_tensor * shared_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(prefix_in);
    ggml_set_input(routed_in);
    ggml_set_input(shared_in);
    ggml_tensor * routed = routed_in;
    if (layer.ffn_routed_norm) {
        routed = rms_norm(ctx, routed, layer.ffn_routed_norm, w.rms_eps);
    }
    routed = ggml_mul_mat(ctx, layer.ffn_routed_up, routed);
    ggml_tensor * hidden = ggml_add(
        ctx, prefix_in, ggml_add(ctx, routed, shared_in));
    const bool ok = run_host_boundary_graph(
        backend, ctx, graph,
        {
            {prefix_in, prefix,
             static_cast<size_t>(w.n_embd) * sizeof(float)},
            {routed_in, routed_output,
             static_cast<size_t>(w.n_expert_latent) * sizeof(float)},
            {shared_in, shared,
             static_cast<size_t>(w.n_embd) * sizeof(float)},
        },
        {{hidden, hidden_output,
          static_cast<size_t>(w.n_embd) * sizeof(float)}},
        "P58 one-row routed join");
    ggml_free(ctx);
    return ok;
}

bool exact_multirow_output_row(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        const float * hidden_row,
        int token_index,
        const std::vector<std::vector<float>> & checkpoints,
        const KimiK3ForwardOptions & options,
        KimiK3ForwardResult & result) {
    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    std::vector<std::vector<float>> checkpoint_rows;
    checkpoint_rows.reserve(checkpoints.size());
    for (const std::vector<float> & checkpoint : checkpoints) {
        const size_t begin = static_cast<size_t>(token_index) * hidden_width;
        checkpoint_rows.emplace_back(
            checkpoint.begin() + static_cast<std::ptrdiff_t>(begin),
            checkpoint.begin() +
                static_cast<std::ptrdiff_t>(begin + hidden_width));
    }
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 P58 output: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(hidden_in);
    inputs.push_back({hidden_in, hidden_row, hidden_width * sizeof(float)});
    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, 1, checkpoint_rows, residuals, inputs);
    ggml_tensor * output_hidden =
        residuals.mix(hidden_in, w.output_res_score);
    output_hidden = rms_norm(
        ctx, output_hidden, w.output_norm, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, output_hidden);
    ggml_tensor * argmax = ggml_argmax(ctx, logits);
    std::vector<GraphOutput> outputs;
    if (options.read_logits) {
        outputs.push_back({
            logits,
            result.logits.data() +
                static_cast<size_t>(token_index) * w.n_vocab,
            static_cast<size_t>(w.n_vocab) * sizeof(float)});
    }
    if (options.read_argmax) {
        outputs.push_back({
            argmax, result.argmax.data() + token_index, sizeof(int32_t)});
    }
    const bool ok = run_host_boundary_graph(
        backend, ctx, graph, inputs, outputs, "P58 one-row output");
    ggml_free(ctx);
    return ok;
}

bool streamed_kimi_k3_forward_exact_multirow(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        const std::vector<int32_t> & tokens,
        int base_pos,
        const KimiK3ForwardOptions & options,
        KimiK3ForwardResult & result,
        MoeHybridStreamEngine * stream_engine) {
    using Clock = std::chrono::steady_clock;
    const int macro_width = static_cast<int>(tokens.size());
    for (const char * name : {
             "DFLASH_KIMI_DIVERGENCE_TRACE_OUT",
             "DFLASH_KIMI_LAYER1_TRACE_OUT",
             "DFLASH_KIMI_P20_IO_TRACE",
             "DFLASH_KIMI_P28_ORACLE_TRACE",
             "DFLASH_KIMI_P40_CACHE_TRACE",
             "DFLASH_MOE_ROUTE_STATS_OUT"}) {
        const char * value = std::getenv(name);
        if (value && *value) {
            set_last_error(
                std::string("Kimi-K3 P58 exact multirow is incompatible with ") +
                name);
            return false;
        }
    }
    for (const char * name : {
             "DFLASH_MOE_DUAL_STREAM_TRACE",
             "DFLASH_KIMI_S0_SERIAL_CORE_ROWS",
             "DFLASH_KIMI_S0_SERIAL_EXPERT_ROWS"}) {
        const char * value = std::getenv(name);
        if (value && *value && std::strcmp(value, "0") != 0) {
            set_last_error(
                std::string("Kimi-K3 P58 exact multirow is incompatible with ") +
                name);
            return false;
        }
    }
    if (!kimi_k3_exact_multirow_width(tokens.size()) ||
        !options.capture_replay ||
        !options.routed_output_provider ||
        !options.routed_output_provider->prefill_service() ||
        !options.routed_output_provider->prefill_service()->supports_width(
            tokens.size()) ||
        options.moe_core_offload || options.capture_layer_ids ||
        options.panel_capture || options.panel_capture_layer_ids ||
        options.panel_captures || options.expert_observer ||
        options.stop_before_moe_layer >= 0 || !stream_engine ||
        !stream_engine->is_bound()) {
        set_last_error("Kimi-K3 P58 exact multirow envelope is invalid");
        return false;
    }

    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    const size_t hidden_values = hidden_width * tokens.size();
    const auto elapsed_ns = [](Clock::time_point begin) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - begin).count());
    };
    const Clock::time_point total_begin = Clock::now();
    uint64_t core_ns = 0;
    uint64_t expert_ns = 0;
    uint64_t join_ns = 0;
    uint64_t output_ns = 0;
    ExactMultirowSnapshotGuard recurrent_guard(backend, cache);

    std::vector<float> hidden(hidden_values);
    Clock::time_point phase_begin = Clock::now();
    for (int token = 0; token < macro_width; ++token) {
        if (!exact_multirow_embedding_row(
                backend, w, tokens[static_cast<size_t>(token)],
                hidden.data() + static_cast<size_t>(token) * hidden_width)) {
            return false;
        }
    }
    core_ns += elapsed_ns(phase_begin);

    std::vector<std::vector<float>> checkpoints;
    checkpoints.reserve(static_cast<size_t>(
        (w.n_layer + w.attn_res_block_size - 1) /
        w.attn_res_block_size));
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        const bool banked = il % w.attn_res_block_size == 0;
        const std::vector<float> checkpoint_value = hidden;

        phase_begin = Clock::now();
        if (il < w.n_dense_lead) {
            std::vector<float> next_hidden(hidden_values);
            for (int token = 0; token < macro_width; ++token) {
                ExactMultirowLayerRow row;
                if (!exact_multirow_layer_row(
                        backend, w, cache, il, base_pos, token,
                        hidden.data() +
                            static_cast<size_t>(token) * hidden_width,
                        checkpoints, row) ||
                    row.hidden.size() != hidden_width) {
                    return false;
                }
                std::copy(
                    row.hidden.begin(), row.hidden.end(),
                    next_hidden.begin() +
                        static_cast<std::ptrdiff_t>(token * hidden_width));
            }
            core_ns += elapsed_ns(phase_begin);
            if (banked) checkpoints.push_back(checkpoint_value);
            hidden.swap(next_hidden);
            continue;
        }

        std::vector<float> prefix(hidden_values);
        std::vector<float> routed(
            static_cast<size_t>(w.n_expert_latent) * tokens.size());
        std::vector<int32_t> selected(
            static_cast<size_t>(w.n_expert_used) * tokens.size());
        std::vector<float> route_weights(
            static_cast<size_t>(w.n_expert_used) * tokens.size());
        std::vector<float> shared(hidden_values);
        for (int token = 0; token < macro_width; ++token) {
            ExactMultirowLayerRow row;
            if (!exact_multirow_layer_row(
                    backend, w, cache, il, base_pos, token,
                    hidden.data() +
                        static_cast<size_t>(token) * hidden_width,
                    checkpoints, row)) {
                return false;
            }
            std::copy(row.prefix.begin(), row.prefix.end(),
                prefix.begin() +
                    static_cast<std::ptrdiff_t>(token * hidden_width));
            std::copy(row.routed.begin(), row.routed.end(),
                routed.begin() + static_cast<std::ptrdiff_t>(
                    token * w.n_expert_latent));
            std::copy(row.selected.begin(), row.selected.end(),
                selected.begin() + static_cast<std::ptrdiff_t>(
                    token * w.n_expert_used));
            std::copy(row.route_weights.begin(), row.route_weights.end(),
                route_weights.begin() + static_cast<std::ptrdiff_t>(
                    token * w.n_expert_used));
            std::copy(row.shared.begin(), row.shared.end(),
                shared.begin() +
                    static_cast<std::ptrdiff_t>(token * hidden_width));
        }
        core_ns += elapsed_ns(phase_begin);
        if (banked) checkpoints.push_back(checkpoint_value);

        for (size_t route = 0; route < selected.size(); ++route) {
            if (selected[route] < 0 || selected[route] >= w.n_expert ||
                !std::isfinite(route_weights[route])) {
                set_last_error("Kimi-K3 P58 native router output is invalid");
                return false;
            }
        }
        if (!options.routed_output_provider->handles_layer(il)) {
            set_last_error(
                "Kimi-K3 P58 calibrated provider does not handle layer " +
                std::to_string(il));
            return false;
        }
        const MoeStreamExpertSpec spec = make_kimi_k3_stream_spec(w, layer);
        MoeStreamRouteBatch routes;
        routes.layer = il - w.n_dense_lead;
        routes.n_expert = w.n_expert;
        routes.top_k = w.n_expert_used;
        routes.n_tokens = macro_width;
        routes.inputs = routed.data();
        routes.selected_ids = selected.data();
        routes.selected_weights = route_weights.data();
        std::vector<float> routed_output;
        std::string provider_error;
        phase_begin = Clock::now();
        if (!options.routed_output_provider->prefill_service()->evaluate_layer(
                il, base_pos, spec, routes, *stream_engine,
                routed_output, &provider_error)) {
            set_last_error(
                "Kimi-K3 P58 routed layer " + std::to_string(il) +
                " failed: " + provider_error);
            return false;
        }
        expert_ns += elapsed_ns(phase_begin);
        if (routed_output.size() !=
            static_cast<size_t>(w.n_expert_latent) * tokens.size()) {
            set_last_error("Kimi-K3 P58 provider returned an invalid shape");
            return false;
        }

        phase_begin = Clock::now();
        std::vector<float> next_hidden(hidden_values);
        for (int token = 0; token < macro_width; ++token) {
            if (!exact_multirow_join_row(
                    backend, w, layer,
                    prefix.data() +
                        static_cast<size_t>(token) * hidden_width,
                    routed_output.data() + static_cast<size_t>(token) *
                        w.n_expert_latent,
                    shared.data() +
                        static_cast<size_t>(token) * hidden_width,
                    next_hidden.data() +
                        static_cast<size_t>(token) * hidden_width)) {
                return false;
            }
        }
        join_ns += elapsed_ns(phase_begin);
        hidden.swap(next_hidden);
    }

    if (options.read_logits) {
        result.logits.resize(
            static_cast<size_t>(w.n_vocab) * tokens.size());
    }
    if (options.read_argmax) result.argmax.resize(tokens.size());
    phase_begin = Clock::now();
    for (int token = 0; token < macro_width; ++token) {
        if (!exact_multirow_output_row(
                backend, w,
                hidden.data() + static_cast<size_t>(token) * hidden_width,
                token, checkpoints, options, result)) {
            return false;
        }
    }
    output_ns += elapsed_ns(phase_begin);
    cache.cur_pos = base_pos + macro_width;
    recurrent_guard.restore();

    const char * profile = std::getenv("DFLASH_KIMI_STAGE_PROFILE");
    if (profile && *profile && std::strcmp(profile, "0") != 0) {
        const uint64_t total_ns = elapsed_ns(total_begin);
        const uint64_t classified = core_ns + expert_ns + join_ns + output_ns;
        std::fprintf(stderr,
            "[kimi-k3-p58-stage] position=%d tokens=%d total_ms=%.3f "
            "one_row_core_ms=%.3f experts_ms=%.3f one_row_join_ms=%.3f "
            "one_row_output_ms=%.3f other_ms=%.3f\n",
            base_pos, macro_width, total_ns / 1.0e6, core_ns / 1.0e6,
            expert_ns / 1.0e6, join_ns / 1.0e6, output_ns / 1.0e6,
            (total_ns > classified ? total_ns - classified : 0) / 1.0e6);
    }
    return true;
}

bool streamed_kimi_k3_forward(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        const std::vector<int32_t> & tokens,
        int base_pos,
        const KimiK3ForwardOptions & options,
        KimiK3ForwardResult & result,
        MoeHybridStreamEngine * stream_engine,
        MoeStreamDualOwnerExecutor * dual_stream_executor,
        const MoeStreamDualOwnerPolicy * stream_owner_policy,
        MoeHybridRoutingStats * routing_stats) {
    using ProfileClock = std::chrono::steady_clock;
    const char * profile_environment =
        std::getenv("DFLASH_KIMI_STAGE_PROFILE");
    const bool profile_stages = profile_environment &&
        *profile_environment && std::strcmp(profile_environment, "0") != 0;
    const ProfileClock::time_point profile_forward_start =
        profile_stages ? ProfileClock::now() : ProfileClock::time_point{};
    uint64_t profile_embedding_ns = 0;
    uint64_t profile_dense_ns = 0;
    uint64_t profile_routed_preparation_ns = 0;
    uint64_t profile_offloaded_preparation_ns = 0;
    uint64_t profile_expert_ns = 0;
    uint64_t profile_join_ns = 0;
    uint64_t profile_output_ns = 0;
    const auto profile_elapsed_ns = [](ProfileClock::time_point start) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                ProfileClock::now() - start).count());
    };
    const int n_tokens = static_cast<int>(tokens.size());
    const size_t hidden_values =
        static_cast<size_t>(w.n_embd) * static_cast<size_t>(n_tokens);
    std::vector<float> hidden(hidden_values);
    KimiDivergenceTraceWriter & divergence_trace =
        kimi_divergence_trace_writer(w);
    if (divergence_trace.requested() && !divergence_trace.good()) {
        set_last_error(
            "Kimi-K3 cannot open H17 divergence trace " +
            divergence_trace.path());
        return false;
    }
    const bool trace_divergence = divergence_trace.good();
    bool p46_requested = false;
    std::string p46_error;
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP",
            p46_requested, &p46_error)) {
        set_last_error(p46_error);
        return false;
    }
    bool p52_requested = false;
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P52_PERSISTENT_ROUTED_JOIN",
            p52_requested, &p46_error)) {
        set_last_error(p46_error);
        return false;
    }
    if (p52_requested && !p46_requested) {
        set_last_error(
            "P52 persistent routed join requires P46 persistent routed "
            "preparation");
        return false;
    }
    bool p53_requested = false;
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P53_DEVICE_HIDDEN_CHAIN",
            p53_requested, &p46_error)) {
        set_last_error(p46_error);
        return false;
    }
    if (p53_requested && !p52_requested) {
        set_last_error(
            "P53 device hidden chain requires P52 persistent routed join");
        return false;
    }
    PersistentRoutedPreparation * persistent_routed_preparation = nullptr;
    if (p46_requested) {
        bool p45_requested = false;
        if (!parse_strict_binary_environment(
                "DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE",
                p45_requested, &p46_error)) {
            set_last_error(p46_error);
            return false;
        }
        if (!p45_requested) {
            set_last_error(
                "P46 persistent routed preparation requires P45 async "
                "compact queue");
            return false;
        }
        if (p52_requested &&
            (!options.routed_output_provider ||
             !options.routed_output_provider->requires_device_output())) {
            set_last_error(
                "P52 persistent routed join requires a device-output "
                "routed provider");
            return false;
        }
        const bool core_offloaded = options.moe_core_offload &&
            options.moe_core_offload->enabled();
        if (n_tokens != 1 || options.capture_replay || trace_divergence ||
            options.stop_before_moe_layer >= 0 || options.panel_capture ||
            options.panel_capture_layer_ids || options.panel_captures ||
            core_offloaded) {
            set_last_error(
                "P46 persistent routed preparation supports only ordinary "
                "single-token GPU-core execution without replay, tracing, "
                "panel capture, or core offload");
            return false;
        }
        if (!cache.persistent_routed_preparation) {
            auto * created = new (std::nothrow) PersistentRoutedPreparation;
            if (!created || !created->initialize(
                    backend, w, cache, p52_requested, &p46_error)) {
                delete created;
                set_last_error(
                    "Kimi-K3 P46 initialization failed: " + p46_error);
                return false;
            }
            cache.persistent_routed_preparation = created;
        }
        persistent_routed_preparation =
            static_cast<PersistentRoutedPreparation *>(
                cache.persistent_routed_preparation);
        if (!persistent_routed_preparation->matches(
                backend, w, p52_requested)) {
            set_last_error(
                "P46 persistent routed preparation backend/model changed");
            return false;
        }
        persistent_routed_preparation->begin_forward();
    }
    if (options.panel_capture) {
        *options.panel_capture = KimiK3MoePanelCapture{};
    }
    std::vector<int> panel_capture_at_layer(
        static_cast<size_t>(w.n_layer), -1);
    if (options.panel_captures) {
        options.panel_captures->assign(
            options.panel_capture_layer_ids->size(),
            KimiK3MoePanelCapture{});
        for (size_t index = 0;
             index < options.panel_capture_layer_ids->size(); ++index) {
            panel_capture_at_layer[static_cast<size_t>(
                (*options.panel_capture_layer_ids)[index])] =
                    static_cast<int>(index);
        }
    }

    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    for (int i = 0; i < n_capture; ++i) {
        capture_at_layer[static_cast<size_t>((*options.capture_layer_ids)[i])] = i;
    }
    result.captured_hidden.assign(
        static_cast<size_t>(n_capture) * hidden_values, 0.0f);

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= base_pos + q; ++k) {
            mla_mask[static_cast<size_t>(q) * kv_len + k] = 0.0f;
        }
    }

    {
        const ProfileClock::time_point profile_start =
            profile_stages ? ProfileClock::now() : ProfileClock::time_point{};
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 embedding: context allocation failed");
            return false;
        }
        ggml_cgraph * graph =
            ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * ids =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(ids);
        ggml_tensor * embedding =
            ggml_get_rows(ctx, w.tok_embd, ids);
        const bool ok = ggml_backend_supports_op(backend, embedding)
            ? run_host_boundary_graph(
                backend, ctx, graph,
                {{ids, tokens.data(), sizeof(int32_t) * tokens.size()}},
                {{embedding, hidden.data(),
                  hidden.size() * sizeof(float)}},
                "embedding")
            : read_token_embeddings_on_host(w, tokens, hidden);
        ggml_free(ctx);
        if (!ok) return false;
        if (profile_stages) {
            profile_embedding_ns += profile_elapsed_ns(profile_start);
        }
    }

    std::vector<std::vector<float>> checkpoints;
    checkpoints.reserve(
        static_cast<size_t>(
            (w.n_layer + w.attn_res_block_size - 1) /
            w.attn_res_block_size));
    const auto append_checkpoint = [&](const std::vector<float> & value) {
        checkpoints.push_back(value);
        if (persistent_routed_preparation &&
            !persistent_routed_preparation->update_checkpoint(
                checkpoints.back(), &p46_error)) {
            set_last_error("Kimi-K3 P46 checkpoint update failed: " +
                p46_error);
            return false;
        }
        return true;
    };
    const ggml_tensor * current_hidden_device = nullptr;

    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer =
            w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache =
            cache.layers[static_cast<size_t>(il)];
        const bool banked =
            il % w.attn_res_block_size == 0;
        if (p53_requested && banked && current_hidden_device) {
            ggml_backend_tensor_get(
                current_hidden_device, hidden.data(), 0,
                hidden.size() * sizeof(float));
        }
        const std::vector<float> checkpoint_value = hidden;
        KimiK3MoeCoreOffload * core_offload =
            options.moe_core_offload &&
            options.moe_core_offload->enabled() &&
            il < static_cast<int>(options.moe_core_offload->layers.size())
                ? options.moe_core_offload : nullptr;
        const bool complete_preparation = core_offload &&
            core_offload->complete_preparation_enabled(il);
        if (complete_preparation &&
            (n_tokens != 1 || options.capture_replay || trace_divergence ||
             options.stop_before_moe_layer == il)) {
            set_last_error(
                "Kimi-K3 complete accelerator preparation currently "
                "supports only ordinary single-token execution without "
                "replay or divergence tracing");
            return false;
        }

        const bool persistent_preparation =
            persistent_routed_preparation &&
            il >= w.n_dense_lead && layer.recurrent &&
            !complete_preparation;
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        std::vector<GraphInput> inputs;
        ggml_tensor * prefix = nullptr;
        ggml_tensor * cur = nullptr;
        if (!persistent_preparation) {
            ctx = new_kimi_step_context();
            if (!ctx) {
                set_last_error("Kimi-K3 layer: context allocation failed");
                return false;
            }
            graph = ggml_new_graph_custom(ctx, 32768, false);
            ggml_tensor * hidden_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
            ggml_set_input(hidden_in);
            inputs.push_back({
                hidden_in,
                p53_requested && current_hidden_device
                    ? nullptr : hidden.data(),
                hidden.size() * sizeof(float), nullptr,
                p53_requested ? current_hidden_device : nullptr});
            prefix = hidden_in;
            cur = hidden_in;
        }
        if (!persistent_preparation && !complete_preparation) {
            AttnResBank residuals;
            populate_attn_res_bank(
                ctx, w, n_tokens, checkpoints, residuals, inputs);
            cur = residuals.mix(prefix, layer.attn_res_score);
            if (banked) residuals.push(prefix);

            cur = rms_norm(
                ctx, cur, layer.attn_norm, w.rms_eps);
            if (layer.recurrent) {
                cur = build_kda(
                    ctx, graph, w, layer, layer_cache, cur,
                    /*commit_state=*/!options.capture_replay,
                    options.capture_replay);
            } else {
                ggml_tensor * mask = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F32, kv_len, n_tokens);
                ggml_set_input(mask);
                inputs.push_back({
                    mask, mla_mask.data(), mla_mask.size() * sizeof(float)});
                cur = build_mla(
                    ctx, graph, w, layer, layer_cache,
                    cur, base_pos, mask);
            }
            prefix = banked
                ? cur : ggml_add(ctx, prefix, cur);
            cur = residuals.mix(prefix, layer.ffn_res_score);
            cur = rms_norm(
                ctx, cur, layer.ffn_norm, w.rms_eps);
        }

        if (il < w.n_dense_lead) {
            const ProfileClock::time_point profile_start =
                profile_stages ? ProfileClock::now() :
                    ProfileClock::time_point{};
            ggml_tensor * gate =
                ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up =
                ggml_mul_mat(ctx, layer.ffn_up, cur);
            ggml_tensor * dense = situ(
                ctx, gate, up,
                w.situ_beta, w.situ_linear_beta);
            dense = ggml_mul_mat(
                ctx, layer.ffn_down, dense);
            ggml_tensor * hidden_out =
                ggml_add(ctx, prefix, dense);
            std::vector<float> next_hidden(hidden_values);
            const bool ok = run_host_boundary_graph(
                backend, ctx, graph, inputs,
                {{hidden_out, next_hidden.data(),
                  next_hidden.size() * sizeof(float)}},
                "dense layer");
            ggml_free(ctx);
            if (!ok) return false;
            if (profile_stages) {
                profile_dense_ns += profile_elapsed_ns(profile_start);
            }
            if (banked && !append_checkpoint(checkpoint_value)) return false;
            hidden.swap(next_hidden);
            const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
            if (capture_idx >= 0) {
                std::memcpy(
                    result.captured_hidden.data() +
                        static_cast<size_t>(capture_idx) * hidden_values,
                    hidden.data(), hidden_values * sizeof(float));
            }
            continue;
        }

        const bool router_offloaded =
            core_offload && (core_offload->router || complete_preparation);
        const bool latent_offloaded =
            core_offload && (core_offload->latent || complete_preparation);
        const bool shared_offloaded =
            core_offload && (core_offload->shared || complete_preparation);
        const bool stop_at_capture_boundary =
            options.stop_before_moe_layer == il;
        const bool preparation_offloaded =
            core_offload &&
            (router_offloaded || latent_offloaded ||
             (shared_offloaded && !stop_at_capture_boundary));
        const bool join_offloaded =
            core_offload && core_offload->join_enabled();
        const bool alternate_provider = options.routed_output_provider &&
            options.routed_output_provider->handles_layer(il);
        const bool device_routed_output = alternate_provider &&
            options.routed_output_provider->requires_device_output();
        const bool retain_persistent_join_inputs =
            persistent_preparation && device_routed_output &&
            n_tokens == 1 && !join_offloaded && !trace_divergence &&
            !stop_at_capture_boundary;
        PersistentRoutedDeviceOutputs persistent_device_outputs;
        ggml_tensor * routed_in = nullptr;
        ggml_tensor * selected_out = nullptr;
        ggml_tensor * route_weights_out = nullptr;
        ggml_tensor * router_logits_out = nullptr;
        ggml_tensor * shared = nullptr;
        if (!persistent_preparation && !latent_offloaded) {
            routed_in = ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
        }
        if (!persistent_preparation && !router_offloaded) {
            ggml_tensor * router_logits = nullptr;
            TopKMoeRouterResult router = build_kimi_router(
                ctx, graph, w, layer, cur, &router_logits);
            // argsort_top_k returns a strided view of the full argsort result.
            // Materialize the tiny host-boundary tensors so the graph allocator
            // cannot recycle their backing storage before the readback.
            selected_out = ggml_cont(ctx, router.selected);
            route_weights_out = ggml_cont(ctx, router.weights_2d);
            router_logits_out = trace_divergence
                ? ggml_cont(ctx, router_logits) : nullptr;
        }
        if (!persistent_preparation &&
            !stop_at_capture_boundary && !shared_offloaded) {
            ggml_tensor * shared_gate =
                ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
            ggml_tensor * shared_up =
                ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
            shared = situ(
                ctx, shared_gate, shared_up,
                w.situ_beta, w.situ_linear_beta);
            shared = ggml_mul_mat(
                ctx, layer.ffn_down_shexp, shared);
        }

        std::vector<float> prefix_host;
        std::vector<float> normalized_hidden_host;
        std::vector<float> routed_input_host(
            static_cast<size_t>(w.n_expert_latent) * n_tokens);
        std::vector<int32_t> selected(
            static_cast<size_t>(w.n_expert_used) * n_tokens);
        std::vector<float> route_weights(
            static_cast<size_t>(w.n_expert_used) * n_tokens);
        std::vector<float> shared_host;
        std::vector<float> pre_moe_hidden_host;
        std::vector<float> router_logits_host;
        std::vector<GraphOutput> preparation_outputs;
        if (!persistent_preparation && preparation_offloaded) {
            normalized_hidden_host.resize(hidden_values);
            if (!complete_preparation) {
                preparation_outputs.push_back({
                    cur, normalized_hidden_host.data(),
                    normalized_hidden_host.size() * sizeof(float)});
            }
        }
        if (!stop_at_capture_boundary) {
            prefix_host.resize(hidden_values);
            shared_host.resize(hidden_values);
            if (!persistent_preparation && !complete_preparation) {
                preparation_outputs.push_back({
                    prefix, prefix_host.data(),
                    prefix_host.size() * sizeof(float)});
            }
        }
        if (!persistent_preparation && !latent_offloaded) {
            preparation_outputs.push_back({
                routed_in, routed_input_host.data(),
                routed_input_host.size() * sizeof(float)});
        }
        if (!persistent_preparation && !router_offloaded) {
            preparation_outputs.push_back({
                selected_out, selected.data(),
                selected.size() * sizeof(int32_t)});
            preparation_outputs.push_back({
                route_weights_out, route_weights.data(),
                route_weights.size() * sizeof(float)});
        }
        if (!persistent_preparation &&
            !stop_at_capture_boundary && !shared_offloaded) {
            preparation_outputs.push_back({
                shared, shared_host.data(),
                shared_host.size() * sizeof(float)});
        }
        if (trace_divergence) {
            router_logits_host.resize(
                static_cast<size_t>(w.n_expert) * n_tokens);
            if (!preparation_offloaded) {
                pre_moe_hidden_host.resize(hidden_values);
                preparation_outputs.push_back({
                    cur, pre_moe_hidden_host.data(),
                    pre_moe_hidden_host.size() * sizeof(float)});
            }
            if (!router_offloaded) {
                preparation_outputs.push_back({
                    router_logits_out, router_logits_host.data(),
                    router_logits_host.size() * sizeof(float)});
            }
        }
        const ProfileClock::time_point profile_preparation_start =
            profile_stages ? ProfileClock::now() :
                ProfileClock::time_point{};
        bool prep_ok = complete_preparation;
        if (persistent_preparation) {
            prep_ok = persistent_routed_preparation->evaluate(
                il, hidden,
                p53_requested ? current_hidden_device : nullptr,
                checkpoints, prefix_host,
                routed_input_host, selected, route_weights,
                shared_host, retain_persistent_join_inputs
                    ? &persistent_device_outputs : nullptr,
                &p46_error);
            if (!prep_ok) {
                set_last_error(
                    "Kimi-K3 P46 routed layer " + std::to_string(il) +
                    " failed: " + p46_error);
            }
        } else if (!complete_preparation) {
            prep_ok = run_host_boundary_graph(
                backend, ctx, graph, inputs, preparation_outputs,
                "routed layer preparation");
        }
        if (ctx) ggml_free(ctx);
        if (!prep_ok) return false;
        if (profile_stages) {
            profile_routed_preparation_ns +=
                profile_elapsed_ns(profile_preparation_start);
        }
        if (complete_preparation) {
            const ProfileClock::time_point profile_offload_start =
                profile_stages ? ProfileClock::now() :
                    ProfileClock::time_point{};
            if (!run_offloaded_complete_preparation(
                    *core_offload, w, il, hidden, checkpoints, banked,
                    prefix_host, normalized_hidden_host,
                    routed_input_host, selected, route_weights,
                    shared_host)) {
                return false;
            }
            if (profile_stages) {
                profile_offloaded_preparation_ns +=
                    profile_elapsed_ns(profile_offload_start);
            }
        } else if (preparation_offloaded) {
            if (trace_divergence) {
                pre_moe_hidden_host = normalized_hidden_host;
            }
            const ProfileClock::time_point profile_offload_start =
                profile_stages ? ProfileClock::now() :
                    ProfileClock::time_point{};
            if (!run_offloaded_moe_preparation(
                    *core_offload, w, il, n_tokens,
                    normalized_hidden_host, routed_input_host, selected,
                    route_weights,
                    stop_at_capture_boundary ? nullptr : &shared_host,
                    trace_divergence ? &router_logits_host : nullptr)) {
                return false;
            }
            if (profile_stages) {
                profile_offloaded_preparation_ns +=
                    profile_elapsed_ns(profile_offload_start);
            }
        }
        if (banked && !append_checkpoint(checkpoint_value)) return false;
        for (size_t route = 0; route < selected.size(); ++route) {
            if (selected[route] < 0 || selected[route] >= w.n_expert) {
                set_last_error(
                    "Kimi-K3 routed layer " + std::to_string(il) +
                    ": native router returned invalid expert " +
                    std::to_string(selected[route]) + " at route " +
                    std::to_string(route));
                return false;
            }
            if (!std::isfinite(route_weights[route])) {
                set_last_error(
                    "Kimi-K3 routed layer " + std::to_string(il) +
                    ": native router returned a non-finite weight");
                return false;
            }
        }

        const int panel_capture_index =
            panel_capture_at_layer[static_cast<size_t>(il)];
        if (panel_capture_index >= 0) {
            KimiK3MoePanelCapture & capture =
                (*options.panel_captures)[
                    static_cast<size_t>(panel_capture_index)];
            capture.layer = il;
            capture.base_pos = base_pos;
            capture.n_tokens = n_tokens;
            capture.latent_dimension = w.n_expert_latent;
            capture.top_k = w.n_expert_used;
            capture.latent = routed_input_host;
            capture.expert_ids = selected;
            capture.router_weights = route_weights;
        }

        if (stop_at_capture_boundary) {
            KimiK3MoePanelCapture & capture = *options.panel_capture;
            capture.layer = il;
            capture.base_pos = base_pos;
            capture.n_tokens = n_tokens;
            capture.latent_dimension = w.n_expert_latent;
            capture.top_k = w.n_expert_used;
            capture.latent = std::move(routed_input_host);
            capture.expert_ids = std::move(selected);
            capture.router_weights = std::move(route_weights);
            cache.cur_pos = base_pos + n_tokens;
            return true;
        }

        const MoeStreamExpertSpec spec = make_kimi_k3_stream_spec(w, layer);

        MoeStreamRouteBatch route_batch;
        route_batch.layer = il - w.n_dense_lead;
        route_batch.n_expert = w.n_expert;
        route_batch.top_k = w.n_expert_used;
        route_batch.n_tokens = n_tokens;
        route_batch.inputs = retain_persistent_join_inputs
            ? nullptr : routed_input_host.data();
        route_batch.device_inputs = retain_persistent_join_inputs
            ? persistent_device_outputs.routed : nullptr;
        route_batch.selected_ids = selected.data();
        route_batch.selected_weights = route_weights.data();
        route_batch.expert_observer = options.expert_observer;
        if (routing_stats && !routing_stats->observe(
                route_batch.layer, selected.data(),
                static_cast<int>(selected.size()))) {
            set_last_error(
                "Kimi-K3 routed layer " + std::to_string(il) +
                ": failed to record native route statistics");
            return false;
        }
        std::vector<float> routed_output;
        std::string stream_error;
        MoeStreamDualOwnerStats owner_stats;
        PendingDeviceOutputGuard device_output_guard(
            device_routed_output ? options.routed_output_provider : nullptr);
        const bool dual_owner = dual_stream_executor != nullptr &&
            options.expert_observer == nullptr && !alternate_provider;
        if (!stream_engine) {
            set_last_error(
                "Kimi-K3 routed layer: no streamed expert engine");
            return false;
        }
        const ProfileClock::time_point profile_expert_start =
            profile_stages ? ProfileClock::now() :
                ProfileClock::time_point{};
        bool route_ok = false;
        if (device_routed_output &&
            (n_tokens != 1 || join_offloaded || trace_divergence)) {
            set_last_error(
                "Kimi-K3 P42 device output requires one token, local join, "
                "and divergence tracing disabled");
            return false;
        }
        if (device_routed_output) {
            route_ok = options.routed_output_provider->evaluate_device(
                il, base_pos, spec, route_batch, *stream_engine, backend,
                &stream_error);
        } else if (n_tokens > 1 && serial_streamed_expert_rows_enabled() &&
            !dual_owner) {
            routed_output.assign(
                static_cast<size_t>(spec.output_dim) * n_tokens, 0.0f);
            route_ok = true;
            for (int token = 0; token < n_tokens; ++token) {
                MoeStreamRouteBatch row = route_batch;
                row.n_tokens = 1;
                row.inputs = route_batch.inputs +
                    static_cast<size_t>(token) * spec.input_dim;
                row.selected_ids = route_batch.selected_ids +
                    static_cast<size_t>(token) * route_batch.top_k;
                row.selected_weights = route_batch.selected_weights +
                    static_cast<size_t>(token) * route_batch.top_k;
                std::vector<float> row_output;
                const bool row_ok = alternate_provider
                    ? options.routed_output_provider->evaluate(
                        il, base_pos + token, spec, row, *stream_engine,
                        row_output, &stream_error)
                    : eval_moe_streamed_experts(
                        *stream_engine, spec, row,
                        row_output, &stream_error);
                if (!row_ok) {
                    route_ok = false;
                    break;
                }
                std::copy(
                    row_output.begin(), row_output.end(),
                    routed_output.begin() +
                        static_cast<size_t>(token) * spec.output_dim);
            }
        } else {
            route_ok = alternate_provider
                ? options.routed_output_provider->evaluate(
                    il, base_pos, spec, route_batch, *stream_engine,
                    routed_output, &stream_error)
                : dual_owner ? dual_stream_executor->eval(
                    spec, route_batch, *stream_owner_policy,
                    routed_output, &owner_stats, &stream_error)
                : eval_moe_streamed_experts(
                    *stream_engine, spec, route_batch,
                    routed_output, &stream_error);
        }
        if (!route_ok) {
            set_last_error(
                "Kimi-K3 routed layer " +
                std::to_string(il) +
                ": streamed expert evaluation failed: " +
                stream_error);
            return false;
        }
        if (profile_stages) {
            profile_expert_ns += profile_elapsed_ns(profile_expert_start);
        }
        const char * trace = std::getenv("DFLASH_MOE_DUAL_STREAM_TRACE");
        if (dual_owner && trace && *trace && std::strcmp(trace, "0") != 0) {
            std::fprintf(stderr,
                "[kimi-k3] dual-owner layer=%d routes=%d/%d experts=%d/%d "
                "primary=%.3fms secondary=%.3fms wall=%.3fms\n",
                route_batch.layer,
                owner_stats.primary_routes, owner_stats.secondary_routes,
                owner_stats.primary_experts, owner_stats.secondary_experts,
                owner_stats.primary_us / 1000.0,
                owner_stats.secondary_us / 1000.0,
                owner_stats.wall_us / 1000.0);
        }

        std::vector<float> next_hidden(hidden_values);
        std::vector<float> moe_output_host;
        if (trace_divergence) {
            moe_output_host.resize(hidden_values);
        }
        bool join_ok = false;
        const ProfileClock::time_point profile_join_start =
            profile_stages ? ProfileClock::now() :
                ProfileClock::time_point{};
        if (p52_requested) {
            if (!persistent_routed_preparation || !device_routed_output ||
                n_tokens != 1 || join_offloaded || trace_divergence) {
                set_last_error(
                    "P52 persistent routed join requires one-token local "
                    "device-output execution");
                return false;
            }
            join_ok = persistent_routed_preparation->evaluate_join(
                il, prefix_host,
                retain_persistent_join_inputs
                    ? persistent_device_outputs.prefix : nullptr,
                *options.routed_output_provider,
                shared_host,
                retain_persistent_join_inputs
                    ? persistent_device_outputs.shared : nullptr,
                next_hidden,
                p53_requested ? &current_hidden_device : nullptr,
                &stream_error);
            if (!join_ok) {
                set_last_error(
                    "Kimi-K3 P52 routed join failed at layer " +
                    std::to_string(il) + ": " + stream_error);
            }
        } else if (join_offloaded) {
            join_ok = run_offloaded_moe_join(
                *core_offload, w, il, n_tokens,
                prefix_host, routed_output, shared_host, next_hidden,
                trace_divergence ? &moe_output_host : nullptr);
        } else {
            ctx = new_kimi_step_context();
            if (!ctx) {
                set_last_error(
                    "Kimi-K3 routed layer join: context allocation failed");
                return false;
            }
            graph = ggml_new_graph_custom(ctx, 4096, false);
            ggml_tensor * prefix_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
            ggml_tensor * routed_out_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_expert_latent, n_tokens);
            ggml_tensor * shared_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
            ggml_set_input(prefix_in);
            ggml_set_input(routed_out_in);
            ggml_set_input(shared_in);
            ggml_tensor * routed = routed_out_in;
            if (layer.ffn_routed_norm) {
                routed = rms_norm(
                    ctx, routed, layer.ffn_routed_norm, w.rms_eps);
            }
            routed = ggml_mul_mat(
                ctx, layer.ffn_routed_up, routed);
            ggml_tensor * moe_shared =
                ggml_add(ctx, routed, shared_in);
            ggml_tensor * hidden_out =
                ggml_add(ctx, prefix_in, moe_shared);
            std::vector<GraphOutput> join_outputs = {{
                hidden_out, next_hidden.data(),
                next_hidden.size() * sizeof(float)}};
            if (trace_divergence) {
                join_outputs.push_back({
                    moe_shared, moe_output_host.data(),
                    moe_output_host.size() * sizeof(float)});
            }
            join_ok = run_host_boundary_graph(
                backend, ctx, graph,
                {
                    {prefix_in,
                     retain_persistent_join_inputs
                         ? nullptr : prefix_host.data(),
                     prefix_host.size() * sizeof(float), nullptr,
                     retain_persistent_join_inputs
                         ? persistent_device_outputs.prefix : nullptr},
                    {routed_out_in,
                     device_routed_output ? nullptr : routed_output.data(),
                     static_cast<size_t>(w.n_expert_latent) * n_tokens *
                         sizeof(float),
                     device_routed_output
                         ? options.routed_output_provider : nullptr},
                    {shared_in,
                     retain_persistent_join_inputs
                         ? nullptr : shared_host.data(),
                     shared_host.size() * sizeof(float), nullptr,
                     retain_persistent_join_inputs
                         ? persistent_device_outputs.shared : nullptr},
                },
                join_outputs,
                "routed layer join");
            ggml_free(ctx);
        }
        if (!join_ok) return false;
        if (profile_stages) {
            profile_join_ns += profile_elapsed_ns(profile_join_start);
        }
        if (trace_divergence && !divergence_trace.append(
                il, base_pos, n_tokens, banked,
                checkpoint_value, pre_moe_hidden_host,
                router_logits_host, selected, routed_input_host, routed_output,
                moe_output_host, next_hidden)) {
            set_last_error(
                "Kimi-K3 cannot append H17 divergence trace " +
                divergence_trace.path());
            return false;
        }
        if (!p53_requested) {
            hidden.swap(next_hidden);
            current_hidden_device = nullptr;
        }
        const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
        if (capture_idx >= 0) {
            if (p53_requested && current_hidden_device) {
                ggml_backend_tensor_get(
                    current_hidden_device, hidden.data(), 0,
                    hidden.size() * sizeof(float));
            }
            std::memcpy(
                result.captured_hidden.data() +
                    static_cast<size_t>(capture_idx) * hidden_values,
                hidden.data(), hidden_values * sizeof(float));
        }
    }

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 output: context allocation failed");
        return false;
    }
    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx, 8192, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(hidden_in);
    inputs.push_back({
        hidden_in,
        p53_requested && current_hidden_device ? nullptr : hidden.data(),
        hidden.size() * sizeof(float), nullptr,
        p53_requested ? current_hidden_device : nullptr});
    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, n_tokens, checkpoints, residuals, inputs);
    ggml_tensor * output_hidden =
        residuals.mix(hidden_in, w.output_res_score);
    output_hidden = rms_norm(
        ctx, output_hidden, w.output_norm, w.rms_eps);
    ggml_tensor * output =
        ggml_mul_mat(ctx, w.output, output_hidden);
    ggml_tensor * argmax = ggml_argmax(ctx, output);
    std::vector<GraphOutput> outputs;
    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        outputs.push_back({
            output, result.logits.data(), result.logits.size() * sizeof(float)});
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        outputs.push_back({
            argmax, result.argmax.data(), result.argmax.size() * sizeof(int32_t)});
    }
    const ProfileClock::time_point profile_output_start =
        profile_stages ? ProfileClock::now() : ProfileClock::time_point{};
    const bool output_ok = run_host_boundary_graph(
        backend, ctx, graph, inputs,
        outputs,
        "output");
    ggml_free(ctx);
    if (!output_ok) return false;
    if (profile_stages) {
        profile_output_ns += profile_elapsed_ns(profile_output_start);
        const uint64_t total_ns =
            profile_elapsed_ns(profile_forward_start);
        const uint64_t classified_ns =
            profile_embedding_ns + profile_dense_ns +
            profile_routed_preparation_ns +
            profile_offloaded_preparation_ns + profile_expert_ns +
            profile_join_ns + profile_output_ns;
        const uint64_t other_ns = total_ns > classified_ns
            ? total_ns - classified_ns : 0;
        std::fprintf(stderr,
            "[kimi-k3-stage] position=%d tokens=%d total_ms=%.3f "
            "embedding_ms=%.3f dense_ms=%.3f routed_prep_ms=%.3f "
            "offload_prep_ms=%.3f experts_ms=%.3f join_ms=%.3f "
            "output_ms=%.3f other_ms=%.3f\n",
            base_pos, n_tokens, total_ns / 1.0e6,
            profile_embedding_ns / 1.0e6, profile_dense_ns / 1.0e6,
            profile_routed_preparation_ns / 1.0e6,
            profile_offloaded_preparation_ns / 1.0e6,
            profile_expert_ns / 1.0e6, profile_join_ns / 1.0e6,
            profile_output_ns / 1.0e6, other_ns / 1.0e6);
    }

    cache.cur_pos = base_pos + n_tokens;
    return true;
}

} // namespace

bool kimi_k3_read_token_embeddings_on_host(
        const KimiK3Weights & w,
        const std::vector<int32_t> & tokens,
        std::vector<float> & hidden) {
    return read_token_embeddings_on_host(w, tokens, hidden);
}

bool benchmark_kimi_k3_kda_layer(
        ggml_backend_t cpu_backend,
        ggml_backend_t accelerator_backend,
        const KimiK3Weights & w,
        int model_layer,
        int iterations,
        KimiK3KdaLayerBenchmarkResult & result,
        std::string * error) {
    result = KimiK3KdaLayerBenchmarkResult{};
    auto fail = [&](const std::string & message) {
        if (error) *error = message;
        return false;
    };
    if (!cpu_backend || !accelerator_backend || model_layer < 0 ||
        model_layer >= w.n_layer || iterations <= 0 || w.n_embd <= 0 ||
        w.kda_head_dim <= 0 || w.n_head <= 0 || w.ssm_d_conv <= 1) {
        return fail("invalid KDA layer benchmark configuration");
    }
    const KimiK3Layer & cpu_layer =
        w.layers[static_cast<size_t>(model_layer)];
    if (!cpu_layer.recurrent) {
        return fail("selected model layer is not recurrent KDA");
    }

    const ggml_tensor * sources[] = {
        cpu_layer.wq, cpu_layer.wk, cpu_layer.wv, cpu_layer.wo,
        cpu_layer.ssm_q_conv, cpu_layer.ssm_k_conv,
        cpu_layer.ssm_v_conv, cpu_layer.ssm_f_a, cpu_layer.ssm_f_b,
        cpu_layer.ssm_beta, cpu_layer.ssm_a, cpu_layer.ssm_dt_b,
        cpu_layer.ssm_g, cpu_layer.ssm_o_norm,
    };
    for (const ggml_tensor * source : sources) {
        if (!source) return fail("selected KDA layer has a missing tensor");
    }

    ggml_context * accelerator_ctx = nullptr;
    ggml_backend_buffer_t accelerator_buffer = nullptr;
    ggml_context * cpu_state_ctx = nullptr;
    ggml_backend_buffer_t cpu_state_buffer = nullptr;
    auto cleanup = [&]() {
        if (accelerator_buffer) ggml_backend_buffer_free(accelerator_buffer);
        if (accelerator_ctx) ggml_free(accelerator_ctx);
        if (cpu_state_buffer) ggml_backend_buffer_free(cpu_state_buffer);
        if (cpu_state_ctx) ggml_free(cpu_state_ctx);
    };

    constexpr size_t tensor_count = sizeof(sources) / sizeof(sources[0]);
    ggml_init_params accelerator_params{};
    accelerator_params.mem_size =
        ggml_tensor_overhead() * (tensor_count + 8) + 16384;
    accelerator_params.no_alloc = true;
    accelerator_ctx = ggml_init(accelerator_params);
    if (!accelerator_ctx) return fail("cannot allocate accelerator metadata");

    KimiK3Layer accelerator_layer;
    ggml_tensor ** destinations[] = {
        &accelerator_layer.wq, &accelerator_layer.wk,
        &accelerator_layer.wv, &accelerator_layer.wo,
        &accelerator_layer.ssm_q_conv, &accelerator_layer.ssm_k_conv,
        &accelerator_layer.ssm_v_conv, &accelerator_layer.ssm_f_a,
        &accelerator_layer.ssm_f_b, &accelerator_layer.ssm_beta,
        &accelerator_layer.ssm_a, &accelerator_layer.ssm_dt_b,
        &accelerator_layer.ssm_g, &accelerator_layer.ssm_o_norm,
    };
    size_t weight_bytes = 0;
    for (size_t i = 0; i < tensor_count; ++i) {
        *destinations[i] = ggml_dup_tensor(accelerator_ctx, sources[i]);
        if (!*destinations[i]) {
            cleanup();
            return fail("cannot duplicate accelerator KDA tensor metadata");
        }
        weight_bytes += ggml_nbytes(sources[i]);
    }
    accelerator_layer.recurrent = true;

    const int64_t d_inner =
        static_cast<int64_t>(w.kda_head_dim) * w.n_head;
    KimiK3LayerCache accelerator_cache;
    accelerator_cache.conv_state = ggml_new_tensor_2d(
        accelerator_ctx, GGML_TYPE_F32,
        w.ssm_d_conv - 1, 3 * d_inner);
    accelerator_cache.ssm_state = ggml_new_tensor_3d(
        accelerator_ctx, GGML_TYPE_F32,
        w.kda_head_dim, w.kda_head_dim, w.n_head);
    if (!accelerator_cache.conv_state || !accelerator_cache.ssm_state) {
        cleanup();
        return fail("cannot allocate accelerator KDA state metadata");
    }
    accelerator_buffer = ggml_backend_alloc_ctx_tensors(
        accelerator_ctx, accelerator_backend);
    if (!accelerator_buffer) {
        cleanup();
        return fail("cannot allocate accelerator KDA tensor buffer");
    }
    for (size_t i = 0; i < tensor_count; ++i) {
        ggml_backend_tensor_copy(sources[i], *destinations[i]);
    }
    ggml_backend_tensor_memset(
        accelerator_cache.conv_state, 0, 0,
        ggml_nbytes(accelerator_cache.conv_state));
    ggml_backend_tensor_memset(
        accelerator_cache.ssm_state, 0, 0,
        ggml_nbytes(accelerator_cache.ssm_state));
    ggml_backend_synchronize(accelerator_backend);

    ggml_init_params cpu_state_params{};
    cpu_state_params.mem_size = ggml_tensor_overhead() * 4 + 4096;
    cpu_state_params.no_alloc = true;
    cpu_state_ctx = ggml_init(cpu_state_params);
    if (!cpu_state_ctx) {
        cleanup();
        return fail("cannot allocate CPU KDA state metadata");
    }
    KimiK3LayerCache cpu_cache;
    cpu_cache.conv_state = ggml_new_tensor_2d(
        cpu_state_ctx, GGML_TYPE_F32,
        w.ssm_d_conv - 1, 3 * d_inner);
    cpu_cache.ssm_state = ggml_new_tensor_3d(
        cpu_state_ctx, GGML_TYPE_F32,
        w.kda_head_dim, w.kda_head_dim, w.n_head);
    cpu_state_buffer = ggml_backend_alloc_ctx_tensors(
        cpu_state_ctx, cpu_backend);
    if (!cpu_state_buffer) {
        cleanup();
        return fail("cannot allocate CPU KDA state buffer");
    }
    ggml_backend_buffer_clear(cpu_state_buffer, 0);

    std::vector<float> input(static_cast<size_t>(w.n_embd));
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(
            0.05 * std::sin((static_cast<double>(i) + 1.0) * pi / 180.0));
    }
    std::vector<float> cpu_output(input.size());
    std::vector<float> accelerator_output(input.size());

    auto run_once = [&](ggml_backend_t backend,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        std::vector<float> & output,
                        double & elapsed_ms) {
        const auto start = std::chrono::steady_clock::now();
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) return false;
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_tensor * hidden = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, 1);
        ggml_set_input(hidden);
        ggml_tensor * kda = build_kda(
            ctx, graph, w, layer, cache, hidden,
            /*commit_state=*/false, /*capture_replay=*/false);
        const bool ok = run_host_boundary_graph(
            backend, ctx, graph,
            {{hidden, input.data(), input.size() * sizeof(float)}},
            {{kda, output.data(), output.size() * sizeof(float)}},
            "isolated KDA benchmark");
        ggml_free(ctx);
        elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return ok;
    };

    double warmup_ms = 0.0;
    if (!run_once(cpu_backend, cpu_layer, cpu_cache, cpu_output, warmup_ms) ||
        !run_once(accelerator_backend, accelerator_layer,
                  accelerator_cache, accelerator_output, warmup_ms)) {
        cleanup();
        return fail("isolated KDA warmup graph failed");
    }
    std::vector<double> cpu_times;
    std::vector<double> accelerator_times;
    cpu_times.reserve(static_cast<size_t>(iterations));
    accelerator_times.reserve(static_cast<size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        double cpu_ms = 0.0;
        double accelerator_ms = 0.0;
        if (!run_once(cpu_backend, cpu_layer, cpu_cache, cpu_output, cpu_ms) ||
            !run_once(accelerator_backend, accelerator_layer,
                      accelerator_cache, accelerator_output,
                      accelerator_ms)) {
            cleanup();
            return fail("isolated KDA measured graph failed");
        }
        cpu_times.push_back(cpu_ms);
        accelerator_times.push_back(accelerator_ms);
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        return values.size() % 2 != 0
            ? values[middle]
            : 0.5 * (values[middle - 1] + values[middle]);
    };
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double squared_candidate = 0.0;
    double dot = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < cpu_output.size(); ++i) {
        const double reference = cpu_output[i];
        const double candidate = accelerator_output[i];
        const double difference = candidate - reference;
        squared_error += difference * difference;
        squared_reference += reference * reference;
        squared_candidate += candidate * candidate;
        dot += reference * candidate;
        max_abs = std::max(max_abs, std::abs(difference));
    }
    result.model_layer = model_layer;
    result.iterations = iterations;
    result.weight_bytes = weight_bytes;
    result.cpu_median_ms = median(cpu_times);
    result.accelerator_median_ms = median(accelerator_times);
    result.speedup = result.accelerator_median_ms > 0.0
        ? result.cpu_median_ms / result.accelerator_median_ms : 0.0;
    result.relative_l2 = squared_reference > 0.0
        ? std::sqrt(squared_error / squared_reference) : 0.0;
    result.cosine = squared_reference > 0.0 && squared_candidate > 0.0
        ? dot / std::sqrt(squared_reference * squared_candidate) : 0.0;
    result.max_abs = max_abs;
    cleanup();
    return true;
}

bool benchmark_kimi_k3_routed_preparation(
        ggml_backend_t cpu_backend,
        ggml_backend_t accelerator_backend,
        const KimiK3Weights & w,
        int model_layer,
        int iterations,
        KimiK3RoutedPreparationBenchmarkResult & result,
        std::string * error) {
    result = KimiK3RoutedPreparationBenchmarkResult{};
    auto fail = [&](const std::string & message) {
        if (error) *error = message;
        return false;
    };
    if (!cpu_backend || !accelerator_backend || model_layer < w.n_dense_lead ||
        model_layer >= w.n_layer || iterations <= 0 || w.n_embd <= 0 ||
        w.n_expert_latent <= 0 || w.n_expert_used <= 0 ||
        w.attn_res_block_size <= 0) {
        return fail("invalid routed-preparation benchmark configuration");
    }
    const KimiK3Layer & cpu_layer =
        w.layers[static_cast<size_t>(model_layer)];
    if (!cpu_layer.recurrent) {
        return fail("selected routed-preparation layer is not recurrent KDA");
    }

    const ggml_tensor * sources[] = {
        cpu_layer.attn_norm, cpu_layer.ffn_norm,
        cpu_layer.attn_res_score, cpu_layer.ffn_res_score,
        cpu_layer.wq, cpu_layer.wk, cpu_layer.wv, cpu_layer.wo,
        cpu_layer.ssm_q_conv, cpu_layer.ssm_k_conv,
        cpu_layer.ssm_v_conv, cpu_layer.ssm_f_a, cpu_layer.ssm_f_b,
        cpu_layer.ssm_beta, cpu_layer.ssm_a, cpu_layer.ssm_dt_b,
        cpu_layer.ssm_g, cpu_layer.ssm_o_norm,
        cpu_layer.ffn_gate_inp, cpu_layer.ffn_exp_probs_b,
        cpu_layer.ffn_routed_down, cpu_layer.ffn_gate_shexp,
        cpu_layer.ffn_up_shexp, cpu_layer.ffn_down_shexp,
    };
    for (const ggml_tensor * source : sources) {
        if (!source) {
            return fail("selected routed-preparation layer has a missing tensor");
        }
    }

    ggml_context * accelerator_ctx = nullptr;
    ggml_backend_buffer_t accelerator_buffer = nullptr;
    ggml_context * cpu_state_ctx = nullptr;
    ggml_backend_buffer_t cpu_state_buffer = nullptr;
    ggml_context * persistent_ctx = nullptr;
    ggml_gallocr_t persistent_allocator = nullptr;
    auto cleanup = [&]() {
        if (persistent_allocator) ggml_gallocr_free(persistent_allocator);
        if (persistent_ctx) ggml_free(persistent_ctx);
        if (accelerator_buffer) ggml_backend_buffer_free(accelerator_buffer);
        if (accelerator_ctx) ggml_free(accelerator_ctx);
        if (cpu_state_buffer) ggml_backend_buffer_free(cpu_state_buffer);
        if (cpu_state_ctx) ggml_free(cpu_state_ctx);
    };

    constexpr size_t tensor_count = sizeof(sources) / sizeof(sources[0]);
    ggml_init_params accelerator_params{};
    accelerator_params.mem_size =
        ggml_tensor_overhead() * (tensor_count + 8) + 32768;
    accelerator_params.no_alloc = true;
    accelerator_ctx = ggml_init(accelerator_params);
    if (!accelerator_ctx) {
        return fail("cannot allocate routed-preparation accelerator metadata");
    }
    KimiK3Layer accelerator_layer;
    ggml_tensor ** destinations[] = {
        &accelerator_layer.attn_norm, &accelerator_layer.ffn_norm,
        &accelerator_layer.attn_res_score,
        &accelerator_layer.ffn_res_score,
        &accelerator_layer.wq, &accelerator_layer.wk,
        &accelerator_layer.wv, &accelerator_layer.wo,
        &accelerator_layer.ssm_q_conv,
        &accelerator_layer.ssm_k_conv,
        &accelerator_layer.ssm_v_conv,
        &accelerator_layer.ssm_f_a, &accelerator_layer.ssm_f_b,
        &accelerator_layer.ssm_beta, &accelerator_layer.ssm_a,
        &accelerator_layer.ssm_dt_b, &accelerator_layer.ssm_g,
        &accelerator_layer.ssm_o_norm,
        &accelerator_layer.ffn_gate_inp,
        &accelerator_layer.ffn_exp_probs_b,
        &accelerator_layer.ffn_routed_down,
        &accelerator_layer.ffn_gate_shexp,
        &accelerator_layer.ffn_up_shexp,
        &accelerator_layer.ffn_down_shexp,
    };
    size_t weight_bytes = 0;
    for (size_t i = 0; i < tensor_count; ++i) {
        *destinations[i] = ggml_dup_tensor(accelerator_ctx, sources[i]);
        if (!*destinations[i]) {
            cleanup();
            return fail("cannot duplicate routed-preparation tensor metadata");
        }
        weight_bytes += ggml_nbytes(sources[i]);
    }
    accelerator_layer.recurrent = true;

    const int64_t d_inner =
        static_cast<int64_t>(w.kda_head_dim) * w.n_head;
    KimiK3LayerCache accelerator_cache;
    accelerator_cache.conv_state = ggml_new_tensor_2d(
        accelerator_ctx, GGML_TYPE_F32,
        w.ssm_d_conv - 1, 3 * d_inner);
    accelerator_cache.ssm_state = ggml_new_tensor_3d(
        accelerator_ctx, GGML_TYPE_F32,
        w.kda_head_dim, w.kda_head_dim, w.n_head);
    if (!accelerator_cache.conv_state || !accelerator_cache.ssm_state) {
        cleanup();
        return fail("cannot allocate routed-preparation accelerator state");
    }
    accelerator_buffer = ggml_backend_alloc_ctx_tensors(
        accelerator_ctx, accelerator_backend);
    if (!accelerator_buffer) {
        cleanup();
        return fail("cannot allocate routed-preparation accelerator buffer");
    }
    for (size_t i = 0; i < tensor_count; ++i) {
        ggml_backend_tensor_copy(sources[i], *destinations[i]);
    }
    ggml_backend_tensor_memset(
        accelerator_cache.conv_state, 0, 0,
        ggml_nbytes(accelerator_cache.conv_state));
    ggml_backend_tensor_memset(
        accelerator_cache.ssm_state, 0, 0,
        ggml_nbytes(accelerator_cache.ssm_state));
    ggml_backend_synchronize(accelerator_backend);

    ggml_init_params cpu_state_params{};
    cpu_state_params.mem_size = ggml_tensor_overhead() * 4 + 4096;
    cpu_state_params.no_alloc = true;
    cpu_state_ctx = ggml_init(cpu_state_params);
    if (!cpu_state_ctx) {
        cleanup();
        return fail("cannot allocate routed-preparation CPU state metadata");
    }
    KimiK3LayerCache cpu_cache;
    cpu_cache.conv_state = ggml_new_tensor_2d(
        cpu_state_ctx, GGML_TYPE_F32,
        w.ssm_d_conv - 1, 3 * d_inner);
    cpu_cache.ssm_state = ggml_new_tensor_3d(
        cpu_state_ctx, GGML_TYPE_F32,
        w.kda_head_dim, w.kda_head_dim, w.n_head);
    cpu_state_buffer = ggml_backend_alloc_ctx_tensors(
        cpu_state_ctx, cpu_backend);
    if (!cpu_state_buffer) {
        cleanup();
        return fail("cannot allocate routed-preparation CPU state buffer");
    }
    ggml_backend_buffer_clear(cpu_state_buffer, 0);

    const int checkpoint_count =
        (model_layer + w.attn_res_block_size - 1) /
        w.attn_res_block_size;
    std::vector<float> hidden(static_cast<size_t>(w.n_embd));
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(
            0.05 * std::sin((static_cast<double>(i) + 1.0) * pi / 180.0));
    }
    std::vector<std::vector<float>> checkpoints(
        static_cast<size_t>(checkpoint_count),
        std::vector<float>(static_cast<size_t>(w.n_embd)));
    for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
        for (size_t i = 0; i < hidden.size(); ++i) {
            checkpoints[static_cast<size_t>(checkpoint)][i] =
                static_cast<float>(0.04 * std::cos(
                    (static_cast<double>(i) + 1.0) *
                    (checkpoint + 1.0) * pi / 256.0));
        }
    }

    struct Outputs {
        std::vector<float> prefix;
        std::vector<float> routed;
        std::vector<int32_t> selected;
        std::vector<float> route_weights;
        std::vector<float> shared;
    };
    auto make_outputs = [&]() {
        Outputs out;
        out.prefix.resize(static_cast<size_t>(w.n_embd));
        out.routed.resize(static_cast<size_t>(w.n_expert_latent));
        out.selected.resize(static_cast<size_t>(w.n_expert_used));
        out.route_weights.resize(static_cast<size_t>(w.n_expert_used));
        out.shared.resize(static_cast<size_t>(w.n_embd));
        return out;
    };
    Outputs cpu_output = make_outputs();
    Outputs accelerator_output = make_outputs();
    Outputs persistent_output = make_outputs();
    const bool banked = model_layer % w.attn_res_block_size == 0;

    struct RoutedPreparationGraph {
        ggml_cgraph * graph = nullptr;
        std::vector<ggml_tensor *> inputs;
        ggml_tensor * prefix = nullptr;
        ggml_tensor * routed = nullptr;
        ggml_tensor * selected = nullptr;
        ggml_tensor * route_weights = nullptr;
        ggml_tensor * shared = nullptr;
    };
    auto build_graph = [&](ggml_context * ctx,
                           const KimiK3Layer & layer,
                           KimiK3LayerCache & cache,
                           RoutedPreparationGraph & built) {
        built.graph = ggml_new_graph_custom(ctx, 32768, false);
        ggml_tensor * hidden_in =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
        if (!built.graph || !hidden_in) return false;
        ggml_set_input(hidden_in);
        built.inputs.push_back(hidden_in);
        AttnResBank residuals;
        residuals.ctx = ctx;
        residuals.eps = w.rms_eps;
        residuals.n_embd = w.n_embd;
        residuals.n_tokens = 1;
        for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
            ggml_tensor * tensor =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.n_embd, 1);
            if (!tensor) return false;
            ggml_set_input(tensor);
            built.inputs.push_back(tensor);
            residuals.push(tensor);
        }
        built.prefix = hidden_in;
        ggml_tensor * cur =
            residuals.mix(built.prefix, layer.attn_res_score);
        if (banked) residuals.push(built.prefix);
        cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
        cur = build_kda(
            ctx, built.graph, w, layer, cache, cur,
            /*commit_state=*/false, /*capture_replay=*/false);
        built.prefix = banked ? cur : ggml_add(ctx, built.prefix, cur);
        cur = residuals.mix(built.prefix, layer.ffn_res_score);
        cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);

        built.routed = ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
        TopKMoeRouterResult router =
            build_kimi_router(ctx, built.graph, w, layer, cur);
        built.selected = ggml_cont(ctx, router.selected);
        built.route_weights = ggml_cont(ctx, router.weights_2d);
        ggml_tensor * shared_gate =
            ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
        ggml_tensor * shared_up =
            ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
        built.shared = situ(
            ctx, shared_gate, shared_up,
            w.situ_beta, w.situ_linear_beta);
        built.shared =
            ggml_mul_mat(ctx, layer.ffn_down_shexp, built.shared);
        return built.prefix && built.routed && built.selected &&
            built.route_weights && built.shared;
    };
    auto graph_inputs = [&](const RoutedPreparationGraph & built) {
        std::vector<GraphInput> inputs;
        inputs.reserve(built.inputs.size());
        inputs.push_back({
            built.inputs[0], hidden.data(), hidden.size() * sizeof(float)});
        for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
            const std::vector<float> & values =
                checkpoints[static_cast<size_t>(checkpoint)];
            inputs.push_back({
                built.inputs[static_cast<size_t>(checkpoint) + 1],
                values.data(), values.size() * sizeof(float)});
        }
        return inputs;
    };
    auto graph_outputs = [&](const RoutedPreparationGraph & built,
                             Outputs & output) {
        return std::vector<GraphOutput>{
            {built.prefix, output.prefix.data(),
             output.prefix.size() * sizeof(float)},
            {built.routed, output.routed.data(),
             output.routed.size() * sizeof(float)},
            {built.selected, output.selected.data(),
             output.selected.size() * sizeof(int32_t)},
            {built.route_weights, output.route_weights.data(),
             output.route_weights.size() * sizeof(float)},
            {built.shared, output.shared.data(),
             output.shared.size() * sizeof(float)},
        };
    };
    auto run_once = [&](ggml_backend_t backend,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        Outputs & output,
                        double & elapsed_ms) {
        const auto start = std::chrono::steady_clock::now();
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) return false;
        RoutedPreparationGraph built;
        if (!build_graph(ctx, layer, cache, built)) {
            ggml_free(ctx);
            return false;
        }

        const bool ok = run_host_boundary_graph(
            backend, ctx, built.graph, graph_inputs(built),
            graph_outputs(built, output),
            "complete routed-preparation benchmark");
        ggml_free(ctx);
        elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return ok;
    };

    persistent_ctx = new_kimi_step_context();
    RoutedPreparationGraph persistent_graph;
    if (!persistent_ctx || !build_graph(
            persistent_ctx, accelerator_layer,
            accelerator_cache, persistent_graph)) {
        cleanup();
        return fail("cannot build persistent routed-preparation graph");
    }
    for (const GraphOutput & output :
         graph_outputs(persistent_graph, persistent_output)) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(persistent_graph.graph, output.tensor);
    }
    persistent_allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(accelerator_backend));
    if (!persistent_allocator || !ggml_gallocr_alloc_graph(
            persistent_allocator, persistent_graph.graph)) {
        cleanup();
        return fail("cannot allocate persistent routed-preparation graph");
    }
    result.persistent_compute_buffer_bytes =
        ggml_gallocr_get_buffer_size(persistent_allocator, 0);
    result.persistent_metadata_bytes = ggml_used_mem(persistent_ctx);
    result.persistent_graph_nodes =
        ggml_graph_n_nodes(persistent_graph.graph);
    auto run_persistent_once = [&](double & elapsed_ms) {
        const auto start = std::chrono::steady_clock::now();
        for (const GraphInput & input : graph_inputs(persistent_graph)) {
            ggml_backend_tensor_set(
                input.tensor, input.data, 0, input.bytes);
        }
        ScopedCudaGraphOverrides replay_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute(
                accelerator_backend,
                persistent_graph.graph) != GGML_STATUS_SUCCESS) {
            return false;
        }
        for (const GraphOutput & output :
             graph_outputs(persistent_graph, persistent_output)) {
            ggml_backend_tensor_get(
                output.tensor, output.data, 0, output.bytes);
        }
        elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return true;
    };

    double warmup_ms = 0.0;
    if (!run_once(
            cpu_backend, cpu_layer, cpu_cache, cpu_output, warmup_ms) ||
        !run_once(
            accelerator_backend, accelerator_layer,
            accelerator_cache, accelerator_output, warmup_ms)) {
        cleanup();
        return fail("complete routed-preparation warmup graph failed");
    }
    for (int warmup = 0; warmup < 3; ++warmup) {
        if (!run_persistent_once(warmup_ms)) {
            cleanup();
            return fail("persistent routed-preparation warmup graph failed");
        }
    }
    std::vector<double> cpu_times;
    std::vector<double> accelerator_times;
    std::vector<double> persistent_times;
    cpu_times.reserve(static_cast<size_t>(iterations));
    accelerator_times.reserve(static_cast<size_t>(iterations));
    persistent_times.reserve(static_cast<size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        double cpu_ms = 0.0;
        double accelerator_ms = 0.0;
        double persistent_ms = 0.0;
        if (!run_once(
                cpu_backend, cpu_layer, cpu_cache,
                cpu_output, cpu_ms) ||
            !run_once(
                accelerator_backend, accelerator_layer,
                accelerator_cache, accelerator_output, accelerator_ms) ||
            !run_persistent_once(persistent_ms)) {
            cleanup();
            return fail("complete routed-preparation measured graph failed");
        }
        cpu_times.push_back(cpu_ms);
        accelerator_times.push_back(accelerator_ms);
        persistent_times.push_back(persistent_ms);
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        return values.size() % 2 != 0
            ? values[middle]
            : 0.5 * (values[middle - 1] + values[middle]);
    };
    double max_abs = 0.0;
    const auto relative_l2 = [&](const std::vector<float> & reference,
                                 const std::vector<float> & candidate) {
        double squared_error = 0.0;
        double squared_reference = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double difference =
                static_cast<double>(candidate[i]) - reference[i];
            squared_error += difference * difference;
            squared_reference +=
                static_cast<double>(reference[i]) * reference[i];
            max_abs = std::max(max_abs, std::abs(difference));
        }
        return squared_reference > 0.0
            ? std::sqrt(squared_error / squared_reference) : 0.0;
    };

    result.model_layer = model_layer;
    result.checkpoint_count = checkpoint_count;
    result.iterations = iterations;
    result.weight_bytes = weight_bytes;
    result.cpu_median_ms = median(cpu_times);
    result.accelerator_median_ms = median(accelerator_times);
    result.speedup = result.accelerator_median_ms > 0.0
        ? result.cpu_median_ms / result.accelerator_median_ms : 0.0;
    result.persistent_accelerator_median_ms = median(persistent_times);
    result.persistent_speedup_vs_transient =
        result.persistent_accelerator_median_ms > 0.0
            ? result.accelerator_median_ms /
                result.persistent_accelerator_median_ms
            : 0.0;
    const auto byte_equal = [](const auto & lhs, const auto & rhs) {
        return lhs.size() == rhs.size() &&
            (lhs.empty() || std::memcmp(
                lhs.data(), rhs.data(),
                lhs.size() * sizeof(lhs[0])) == 0);
    };
    result.persistent_prefix_byte_equal = byte_equal(
        accelerator_output.prefix, persistent_output.prefix);
    result.persistent_routed_byte_equal = byte_equal(
        accelerator_output.routed, persistent_output.routed);
    result.persistent_shared_byte_equal = byte_equal(
        accelerator_output.shared, persistent_output.shared);
    result.persistent_route_weight_byte_equal = byte_equal(
        accelerator_output.route_weights, persistent_output.route_weights);
    result.persistent_selected_id_equal = byte_equal(
        accelerator_output.selected, persistent_output.selected);
    const auto accumulate_persistent_max_abs = [&](const auto & reference,
                                                   const auto & candidate) {
        for (size_t i = 0; i < reference.size(); ++i) {
            result.persistent_max_abs = std::max(
                result.persistent_max_abs,
                std::abs(static_cast<double>(candidate[i]) -
                         static_cast<double>(reference[i])));
        }
    };
    accumulate_persistent_max_abs(
        accelerator_output.prefix, persistent_output.prefix);
    accumulate_persistent_max_abs(
        accelerator_output.routed, persistent_output.routed);
    accumulate_persistent_max_abs(
        accelerator_output.shared, persistent_output.shared);
    accumulate_persistent_max_abs(
        accelerator_output.route_weights, persistent_output.route_weights);
    result.prefix_relative_l2 = relative_l2(
        cpu_output.prefix, accelerator_output.prefix);
    result.routed_relative_l2 = relative_l2(
        cpu_output.routed, accelerator_output.routed);
    result.shared_relative_l2 = relative_l2(
        cpu_output.shared, accelerator_output.shared);
    result.route_weight_relative_l2 = relative_l2(
        cpu_output.route_weights, accelerator_output.route_weights);
    result.max_abs = max_abs;
    result.selected_id_count = static_cast<int>(cpu_output.selected.size());
    for (size_t i = 0; i < cpu_output.selected.size(); ++i) {
        if (cpu_output.selected[i] == accelerator_output.selected[i]) {
            ++result.selected_id_agreement;
        }
    }
    cleanup();
    return true;
}

bool create_kimi_k3_cache(ggml_backend_t backend,
                          const KimiK3Weights & w,
                          int max_ctx,
                          KimiK3Cache & out,
                          int max_verify_tokens) {
    free_kimi_k3_cache(out);
    if (!backend || max_ctx <= 0) return false;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() *
        static_cast<size_t>(w.n_layer * 6 + 16) + 16384;
    params.no_alloc = true;
    out.ctx = ggml_init(params);
    if (!out.ctx) return false;

    out.layers.resize(static_cast<size_t>(w.n_layer));
    const int64_t d_inner = static_cast<int64_t>(w.kda_head_dim) * w.n_head;
    const int compact_dim = w.kv_lora_rank + w.rope_dim;
    for (int il = 0; il < w.n_layer; ++il) {
        KimiK3LayerCache & layer_cache = out.layers[static_cast<size_t>(il)];
        char name[80];
        if (w.layers[static_cast<size_t>(il)].recurrent) {
            layer_cache.conv_state = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                w.ssm_d_conv - 1, 3 * d_inner);
            layer_cache.ssm_state = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                w.kda_head_dim, w.kda_head_dim, w.n_head);
            std::snprintf(name, sizeof(name), "kimi_k3_conv_state_%d", il);
            ggml_set_name(layer_cache.conv_state, name);
            std::snprintf(name, sizeof(name), "kimi_k3_ssm_state_%d", il);
            ggml_set_name(layer_cache.ssm_state, name);
            if (max_verify_tokens > 0) {
                layer_cache.conv_state_snap = ggml_dup_tensor(
                    out.ctx, layer_cache.conv_state);
                layer_cache.ssm_state_snap = ggml_dup_tensor(
                    out.ctx, layer_cache.ssm_state);
                layer_cache.replay_input = ggml_new_tensor_2d(
                    out.ctx, GGML_TYPE_F32, w.n_embd, max_verify_tokens);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_conv_state_snap_%d", il);
                ggml_set_name(layer_cache.conv_state_snap, name);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_ssm_state_snap_%d", il);
                ggml_set_name(layer_cache.ssm_state_snap, name);
                std::snprintf(
                    name, sizeof(name), "kimi_k3_replay_input_%d", il);
                ggml_set_name(layer_cache.replay_input, name);
            }
        } else {
            layer_cache.mla_k = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F16,
                compact_dim, 1, max_ctx);
            std::snprintf(name, sizeof(name), "kimi_k3_mla_k_%d", il);
            ggml_set_name(layer_cache.mla_k, name);
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        free_kimi_k3_cache(out);
        return false;
    }
    out.max_ctx = max_ctx;
    out.max_verify_tokens = std::max(0, max_verify_tokens);
    reset_kimi_k3_cache(out);
    return true;
}

void reset_kimi_k3_cache(KimiK3Cache & cache) {
    if (cache.buf) ggml_backend_buffer_clear(cache.buf, 0);
    cache.cur_pos = 0;
    cache.snapshot_pos = -1;
    cache.replay_base_pos = -1;
    cache.replay_n_tokens = 0;
    cache.snapshot_valid = false;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = false;
    cache.replay_exact_rows = false;
}

void free_kimi_k3_cache(KimiK3Cache & cache) {
    free_persistent_routed_preparation(
        cache.persistent_routed_preparation);
    if (cache.buf) ggml_backend_buffer_free(cache.buf);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = KimiK3Cache{};
}

bool kimi_k3_forward(ggml_backend_t backend,
                     const KimiK3Weights & w,
                     KimiK3Cache & cache,
                     const std::vector<int32_t> & tokens,
                     int base_pos,
                     const KimiK3ForwardOptions & options,
                     KimiK3ForwardResult & result,
                     MoeHybridStreamEngine * stream_engine,
                     MoeStreamDualOwnerExecutor * dual_stream_executor,
                     const MoeStreamDualOwnerPolicy * stream_owner_policy,
                     MoeHybridRoutingStats * routing_stats) {
    result = KimiK3ForwardResult{};
    const int n_tokens = static_cast<int>(tokens.size());
    const bool panel_stop = options.stop_before_moe_layer >= 0;
    const bool panel_multi_requested =
        options.panel_capture_layer_ids || options.panel_captures;
    const bool panel_multi =
        options.panel_capture_layer_ids && options.panel_captures;
    if (!backend || !w.ctx || !cache.ctx || n_tokens <= 0 || base_pos < 0 ||
        base_pos != cache.cur_pos || base_pos + n_tokens > cache.max_ctx ||
        (!options.read_logits && !options.read_argmax && !panel_stop)) {
        set_last_error("Kimi-K3 forward: invalid backend, output, or cache span");
        return false;
    }
    if (panel_stop &&
        (!options.panel_capture || options.read_logits || options.read_argmax ||
         options.stop_before_moe_layer < w.n_dense_lead ||
         options.stop_before_moe_layer >= w.n_layer)) {
        set_last_error("Kimi-K3 forward: invalid pre-expert panel capture request");
        return false;
    }
    if (!panel_stop && options.panel_capture) {
        set_last_error("Kimi-K3 forward: panel capture requires a stop layer");
        return false;
    }
    if (panel_multi_requested &&
        (!panel_multi || panel_stop || options.panel_capture ||
         options.panel_capture_layer_ids->empty())) {
        set_last_error("Kimi-K3 forward: invalid multi-layer panel capture request");
        return false;
    }
    if (panel_multi) {
        std::vector<bool> seen(static_cast<size_t>(w.n_layer), false);
        for (int layer : *options.panel_capture_layer_ids) {
            if (layer < w.n_dense_lead || layer >= w.n_layer ||
                seen[static_cast<size_t>(layer)]) {
                set_last_error(
                    "Kimi-K3 forward: invalid or duplicate multi-layer "
                    "panel capture layer");
                return false;
            }
            seen[static_cast<size_t>(layer)] = true;
        }
    }
    if (!w.routed_experts_streamed &&
        (panel_stop || panel_multi || options.expert_observer)) {
        set_last_error(
            "Kimi-K3 forward: panel capture/observation requires streamed experts");
        return false;
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= w.n_vocab) {
            set_last_error("Kimi-K3 forward: token is outside the vocabulary");
            return false;
        }
    }

    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    for (int i = 0; i < n_capture; ++i) {
        const int layer = (*options.capture_layer_ids)[static_cast<size_t>(i)];
        if (layer < 0 || layer >= w.n_layer ||
            capture_at_layer[static_cast<size_t>(layer)] >= 0) {
            set_last_error("Kimi-K3 forward: invalid or duplicate capture layer");
            return false;
        }
        capture_at_layer[static_cast<size_t>(layer)] = i;
    }
    if (options.capture_replay &&
        (n_tokens > cache.max_verify_tokens || !cache.snapshot_valid ||
         cache.snapshot_pos != base_pos)) {
        set_last_error("Kimi-K3 forward: ReplaySSM capture has no matching snapshot");
        return false;
    }
    if (options.exact_multirow_core &&
        (!options.capture_replay ||
         !kimi_k3_exact_multirow_width(static_cast<size_t>(n_tokens)) ||
         panel_stop || panel_multi ||
         options.capture_layer_ids || options.expert_observer ||
         options.moe_core_offload || dual_stream_executor || routing_stats ||
         !options.routed_output_provider ||
         !options.routed_output_provider->prefill_service() ||
         !options.routed_output_provider->prefill_service()->supports_width(
             static_cast<size_t>(n_tokens)))) {
        set_last_error(
            "Kimi-K3 forward: P58 exact multirow request is outside its "
            "qualified envelope");
        return false;
    }
    if (options.exact_multirow_core && !w.routed_experts_streamed) {
        set_last_error(
            "Kimi-K3 forward: P58 exact multirow requires streamed experts");
        return false;
    }

    if (w.routed_experts_streamed) {
        if (!panel_stop && (!stream_engine || !stream_engine->is_bound())) {
            set_last_error(
                "Kimi-K3 forward: file-backed experts require a bound stream engine");
            return false;
        }
        if (dual_stream_executor &&
            (!dual_stream_executor->is_ready() || !stream_owner_policy)) {
            set_last_error(
                "Kimi-K3 forward: dual-owner streaming requires a ready "
                "executor and an ownership policy");
            return false;
        }
        const bool forward_ok = options.exact_multirow_core
            ? streamed_kimi_k3_forward_exact_multirow(
                backend, w, cache, tokens, base_pos, options, result,
                stream_engine)
            : streamed_kimi_k3_forward(
                backend, w, cache, tokens, base_pos, options, result,
                stream_engine, dual_stream_executor,
                stream_owner_policy, routing_stats);
        if (!forward_ok) {
            return false;
        }
        if (options.capture_replay) {
            cache.replay_base_pos = base_pos;
            cache.replay_n_tokens = n_tokens;
            cache.replay_valid = true;
            cache.recurrent_state_pristine = true;
            cache.replay_exact_rows = options.exact_multirow_core;
        } else {
            cache.replay_valid = false;
            cache.recurrent_state_pristine = false;
            cache.replay_exact_rows = false;
        }
        return true;
    }

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int q = 0; q < n_tokens; ++q) {
        for (int k = 0; k <= base_pos + q; ++k) {
            mla_mask[static_cast<size_t>(q) * kv_len + k] = 0.0f;
        }
    }

    ggml_init_params params{};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        set_last_error("Kimi-K3 forward: graph context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);

    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(ids, "token_ids");
    ggml_set_input(ids);
    ggml_tensor * hidden = ggml_get_rows(ctx, w.tok_embd, ids);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kv_len, n_tokens);
    ggml_set_name(mask, "kimi_k3_mla_causal_mask");
    ggml_set_input(mask);

    std::vector<ggml_tensor *> capture_tensors(
        static_cast<size_t>(n_capture), nullptr);
    AttnResBank residuals;
    residuals.ctx = ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = n_tokens;
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache = cache.layers[static_cast<size_t>(il)];
        ggml_tensor * prefix = hidden;
        ggml_tensor * cur = residuals.mix(prefix, layer.attn_res_score);
        const bool banked = il % w.attn_res_block_size == 0;
        if (banked) residuals.push(prefix);

        cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
        cur = layer.recurrent
            ? build_kda(ctx, graph, w, layer, layer_cache, cur,
                        /*commit_state=*/!options.capture_replay,
                        options.capture_replay)
            : build_mla(ctx, graph, w, layer, layer_cache, cur, base_pos, mask);
        prefix = banked ? cur : ggml_add(ctx, prefix, cur);

        cur = residuals.mix(prefix, layer.ffn_res_score);
        cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);
        if (il < w.n_dense_lead) {
            ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
            cur = situ(ctx, gate, up, w.situ_beta, w.situ_linear_beta);
            cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
        } else {
            cur = build_latent_moe(ctx, graph, w, layer, cur);
        }
        hidden = ggml_add(ctx, prefix, cur);
        const int capture_idx = capture_at_layer[static_cast<size_t>(il)];
        if (capture_idx >= 0) {
            capture_tensors[static_cast<size_t>(capture_idx)] = hidden;
            ggml_set_output(hidden);
            ggml_build_forward_expand(graph, hidden);
        }
    }

    hidden = residuals.mix(hidden, w.output_res_score);
    hidden = rms_norm(ctx, hidden, w.output_norm, w.rms_eps);
    ggml_tensor * output = ggml_mul_mat(ctx, w.output, hidden);
    ggml_set_name(output, "logits");
    ggml_tensor * argmax = ggml_argmax(ctx, output);
    if (options.read_logits) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }
    if (options.read_argmax) {
        ggml_set_output(argmax);
        ggml_build_forward_expand(graph, argmax);
    }

    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        set_last_error("Kimi-K3 forward: graph allocation failed");
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(
        ids, tokens.data(), 0, sizeof(int32_t) * tokens.size());
    ggml_backend_tensor_set(
        mask, mla_mask.data(), 0, sizeof(float) * mla_mask.size());
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        set_last_error("Kimi-K3 forward: graph compute failed with status " +
                       std::to_string(static_cast<int>(status)));
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }

    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        ggml_backend_tensor_get(output, result.logits.data(), 0,
                                result.logits.size() * sizeof(float));
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        ggml_backend_tensor_get(argmax, result.argmax.data(), 0,
                                result.argmax.size() * sizeof(int32_t));
    }
    const size_t hidden_values =
        static_cast<size_t>(w.n_embd) * static_cast<size_t>(n_tokens);
    result.captured_hidden.resize(
        static_cast<size_t>(n_capture) * hidden_values);
    for (int i = 0; i < n_capture; ++i) {
        ggml_backend_tensor_get(
            capture_tensors[static_cast<size_t>(i)],
            result.captured_hidden.data() + static_cast<size_t>(i) * hidden_values,
            0, hidden_values * sizeof(float));
    }

    cache.cur_pos = base_pos + n_tokens;
    if (options.capture_replay) {
        cache.replay_base_pos = base_pos;
        cache.replay_n_tokens = n_tokens;
        cache.replay_valid = true;
        cache.recurrent_state_pristine = true;
    } else {
        cache.replay_valid = false;
        cache.recurrent_state_pristine = false;
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return true;
}

bool kimi_k3_replay_snapshot(ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend || cache.max_verify_tokens <= 0) return false;
    for (KimiK3LayerCache & layer : cache.layers) {
        if (!layer.ssm_state) continue;
        if (!layer.ssm_state_snap || !layer.conv_state_snap ||
            !layer.replay_input) {
            return false;
        }
        ggml_backend_tensor_copy_async(
            backend, backend, layer.ssm_state, layer.ssm_state_snap);
        ggml_backend_tensor_copy_async(
            backend, backend, layer.conv_state, layer.conv_state_snap);
    }
    ggml_backend_synchronize(backend);
    cache.snapshot_pos = cache.cur_pos;
    cache.snapshot_valid = true;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = true;
    cache.replay_exact_rows = false;
    return true;
}

bool kimi_k3_replay_restore(ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend || !cache.snapshot_valid || cache.snapshot_pos < 0) return false;
    if (!cache.recurrent_state_pristine) {
        for (KimiK3LayerCache & layer : cache.layers) {
            if (!layer.ssm_state) continue;
            if (!layer.ssm_state_snap || !layer.conv_state_snap) return false;
            ggml_backend_tensor_copy_async(
                backend, backend, layer.ssm_state_snap, layer.ssm_state);
            ggml_backend_tensor_copy_async(
                backend, backend, layer.conv_state_snap, layer.conv_state);
        }
        ggml_backend_synchronize(backend);
    }
    cache.cur_pos = cache.snapshot_pos;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = true;
    cache.replay_exact_rows = false;
    return true;
}

bool kimi_k3_replay_commit(ggml_backend_t backend,
                           const KimiK3Weights & w,
                           KimiK3Cache & cache,
                           int base_pos,
                           int commit_n) {
    if (!backend || !cache.snapshot_valid || !cache.replay_valid ||
        !cache.recurrent_state_pristine || cache.snapshot_pos != base_pos ||
        cache.replay_base_pos != base_pos || commit_n <= 0 ||
        commit_n > cache.replay_n_tokens) {
        return false;
    }
    const bool exact_rows = cache.replay_exact_rows;
    cache.recurrent_state_pristine = false;
    const auto commit_span = [&](int token_offset, int token_count) {
        ggml_init_params params{};
        params.mem_size = 64ull * 1024ull * 1024ull;
        params.no_alloc = true;
        ggml_context * ctx = ggml_init(params);
        if (!ctx) return false;
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
        for (int il = 0; il < w.n_layer; ++il) {
            const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
            if (!layer.recurrent) continue;
            KimiK3LayerCache & layer_cache =
                cache.layers[static_cast<size_t>(il)];
            if (!layer_cache.replay_input) {
                ggml_free(ctx);
                return false;
            }
            ggml_tensor * replay = ggml_view_2d(
                ctx, layer_cache.replay_input, w.n_embd, token_count,
                layer_cache.replay_input->nb[1],
                static_cast<size_t>(token_offset) *
                    layer_cache.replay_input->nb[1]);
            (void)build_kda(
                ctx, graph, w, layer, layer_cache, replay,
                /*commit_state=*/true,
                /*capture_replay=*/false);
        }
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator) ggml_gallocr_free(allocator);
            ggml_free(ctx);
            return false;
        }
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return status == GGML_STATUS_SUCCESS;
    };

    bool commit_ok = true;
    if (exact_rows) {
        for (int token = 0; token < commit_n && commit_ok; ++token) {
            commit_ok = commit_span(token, 1);
        }
    } else {
        commit_ok = commit_span(0, commit_n);
    }
    if (!commit_ok) {
        (void)kimi_k3_replay_restore(backend, cache);
        return false;
    }
    cache.cur_pos = base_pos + commit_n;
    cache.snapshot_valid = false;
    cache.replay_valid = false;
    cache.replay_exact_rows = false;
    return true;
}

bool kimi_k3_step(ggml_backend_t backend,
                  const KimiK3Weights & w,
                  KimiK3Cache & cache,
                  int32_t token,
                  int position,
                  std::vector<float> & logits,
                  MoeHybridStreamEngine * stream_engine,
                  MoeStreamDualOwnerExecutor * dual_stream_executor,
                  const MoeStreamDualOwnerPolicy * stream_owner_policy,
                  MoeHybridRoutingStats * routing_stats,
                  KimiK3RoutedOutputProvider * routed_output_provider,
                  KimiK3MoeCoreOffload * moe_core_offload) {
    KimiK3ForwardOptions options;
    options.read_logits = true;
    options.read_argmax = false;
    options.routed_output_provider = routed_output_provider;
    options.moe_core_offload = moe_core_offload;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend, w, cache, std::vector<int32_t>{token}, position,
            options, result, stream_engine, dual_stream_executor,
            stream_owner_policy, routing_stats)) {
        return false;
    }
    logits = std::move(result.logits);
    return true;
}

} // namespace dflash::common
