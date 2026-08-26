#include "kimi_k3_internal.h"
#include "kimi_k3_routed_provider.h"

#include "common/cuda_graph_overrides.h"
#include "common/moe_router_graph.h"
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

// Kimi K3's official DSpark capture stream is not the raw post-layer hidden
// value.  It is the AttnRes mixture that the following layer's attention
// consumes (or the output-side mixture after the final layer).  Keep this
// experiment opt-in until the matched draft-acceptance A/B is qualified.
bool use_dspark_attn_res_capture() {
    static const char * value =
        std::getenv("DFLASH_KIMI_DSPARK_ATTN_RES_CAPTURE");
    static const bool enabled =
        value && std::strcmp(value, "1") == 0;
    return enabled;
}

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
                         bool commit_state,
                         ggml_tensor ** terminal_state = nullptr) {
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
    if (commit_state || terminal_state) {
        ggml_tensor * newest = ggml_view_3d(ctx, conv_input,
            state_rows, d_inner, 1, conv_input->nb[1], conv_input->nb[2],
            static_cast<size_t>(n_tokens) * conv_input->nb[0]);
        if (commit_state) {
            ggml_build_forward_expand(graph, ggml_cpy(ctx, newest, state));
        }
        if (terminal_state) *terminal_state = newest;
    }

    ggml_tensor * cw = ggml_reshape_2d(ctx, conv_weight, d_conv, d_inner);
    ggml_tensor * out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, cw));
    out = ggml_reshape_4d(ctx, out, head_dim, n_head, n_tokens, 1);
    return out;
}

struct KdaTerminalState {
    ggml_tensor * conv = nullptr;
    ggml_tensor * ssm = nullptr;
};

ggml_tensor * build_kda(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        bool commit_state,
                        bool capture_replay,
                        int replay_token_offset = 0,
                        KdaTerminalState * terminal_state = nullptr) {
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

    ggml_tensor * q_terminal = nullptr;
    ggml_tensor * k_terminal = nullptr;
    ggml_tensor * v_terminal = nullptr;
    ggml_tensor * q = kda_conv1d(ctx, graph, cache.conv_state, 0, cur,
        layer.wq, layer.ssm_q_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state, terminal_state ? &q_terminal : nullptr);
    ggml_tensor * k = kda_conv1d(ctx, graph, cache.conv_state, 1, cur,
        layer.wk, layer.ssm_k_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state, terminal_state ? &k_terminal : nullptr);
    ggml_tensor * v = kda_conv1d(ctx, graph, cache.conv_state, 2, cur,
        layer.wv, layer.ssm_v_conv, w.ssm_d_conv, head_dim, n_head,
        commit_state, terminal_state ? &v_terminal : nullptr);

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
    if (terminal_state) {
        GGML_ASSERT(!commit_state && q_terminal && k_terminal && v_terminal);
        terminal_state->conv = ggml_concat(
            ctx, ggml_concat(ctx, q_terminal, k_terminal, 1), v_terminal, 1);
        terminal_state->ssm = new_state;
    }

    ggml_tensor * gate = ggml_mul_mat(ctx, layer.ssm_g, cur);
    gate = ggml_reshape_3d(ctx, gate, head_dim, n_head, n_tokens);
    output = ggml_reshape_3d(ctx, output, head_dim, n_head, n_tokens);
    output = rms_norm(ctx, output, layer.ssm_o_norm, w.rms_eps);
    output = ggml_mul(ctx, output, ggml_sigmoid(ctx, gate));
    output = ggml_cont_2d(ctx, output, d_inner, n_tokens);
    return ggml_mul_mat(ctx, layer.wo, output);
}

ggml_tensor * build_mla_absorbed_attention(
        ggml_context * ctx,
        ggml_cgraph * graph,
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        KimiK3LayerCache & cache,
        ggml_tensor * q_cur,
        ggml_tensor * compact_pe,
        ggml_tensor * gate_input,
        ggml_tensor * projected_gate,
        int position,
        ggml_tensor * attn_mask) {
    const int n_head = w.n_head;
    const int kv_rank = w.kv_lora_rank;
    const int key_dim = w.mla_k_head_dim;
    const int value_dim = w.mla_v_head_dim;
    const int rope_dim = w.rope_dim;
    const int nope_dim = key_dim - rope_dim;
    const int compact_dim = kv_rank + rope_dim;
    const int n_tokens = static_cast<int>(q_cur->ne[1]);
    const int kv_len = position + n_tokens;
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
    ggml_tensor * out = ggml_mul_mat(ctx, v, scores);
    out = ggml_mul_mat(ctx, layer.wv_b, out);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out,
                       static_cast<int64_t>(value_dim) * n_head, n_tokens);

    if (projected_gate || layer.wqkv_gate) {
        ggml_tensor * output_gate = projected_gate
            ? ggml_sigmoid(ctx, projected_gate)
            : ggml_sigmoid(ctx,
                ggml_mul_mat(ctx, layer.wqkv_gate, gate_input));
        out = ggml_mul(ctx, out, output_gate);
    }
    return out;
}

ggml_tensor * build_mla(ggml_context * ctx,
                        ggml_cgraph * graph,
                        const KimiK3Weights & w,
                        const KimiK3Layer & layer,
                        KimiK3LayerCache & cache,
                        ggml_tensor * cur,
                        int position,
                        ggml_tensor * attn_mask) {
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
    ggml_tensor * out = build_mla_absorbed_attention(
        ctx, graph, w, layer, cache, q_cur, compact_pe, gate_input,
        /*projected_gate=*/nullptr, position, attn_mask);
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

struct GraphDevicePublish {
    ggml_tensor * source = nullptr;
    ggml_tensor * destination = nullptr;
};

struct GraphExecutionTiming {
    uint64_t compute_ns = 0;
    uint64_t publish_ns = 0;
};

bool same_tensor_layout(
        const ggml_tensor * left, const ggml_tensor * right) {
    if (!left || !right || left->type != right->type) return false;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        if (left->ne[dimension] != right->ne[dimension] ||
            left->nb[dimension] != right->nb[dimension]) {
            return false;
        }
    }
    return true;
}

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
                             const char * phase,
                             const std::vector<GraphDevicePublish> & publishes = {},
                             GraphExecutionTiming * timing = nullptr) {
    (void)ctx;
    using Clock = std::chrono::steady_clock;
    if (timing) *timing = GraphExecutionTiming{};
    for (const GraphOutput & output : outputs) {
        if (!output.tensor || !output.data || output.bytes == 0 ||
            output.bytes != ggml_nbytes(output.tensor)) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph output");
            return false;
        }
    }
    for (const GraphInput & input : inputs) {
        const int sources = (input.data ? 1 : 0) +
            (input.device_provider ? 1 : 0) +
            (input.device_tensor ? 1 : 0);
        if (!input.tensor || input.bytes == 0 || sources != 1 ||
            input.bytes != ggml_nbytes(input.tensor) ||
            (input.device_tensor &&
             ggml_nbytes(input.device_tensor) != input.bytes)) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid graph input");
            return false;
        }
    }
    for (const GraphDevicePublish & publish : publishes) {
        if (!same_tensor_layout(publish.source, publish.destination)) {
            set_last_error(std::string("Kimi-K3 ") + phase +
                           ": invalid device publication");
            return false;
        }
    }
    for (const GraphOutput & output : outputs) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
    }
    for (const GraphDevicePublish & publish : publishes) {
        ggml_set_output(publish.source);
        ggml_build_forward_expand(graph, publish.source);
    }
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph allocation failed");
        if (allocator) ggml_gallocr_free(allocator);
        return false;
    }
    for (const GraphInput & input : inputs) {
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
            ggml_backend_tensor_copy_async(
                backend, backend, input.device_tensor, input.tensor);
        } else {
            ggml_backend_tensor_set(
                input.tensor, input.data, 0, input.bytes);
        }
    }
    Clock::time_point compute_begin;
    if (timing) compute_begin = Clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (timing) {
        timing->compute_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - compute_begin).count());
    }
    if (status != GGML_STATUS_SUCCESS) {
        set_last_error(std::string("Kimi-K3 ") + phase +
                       ": graph compute failed with status " +
                       std::to_string(static_cast<int>(status)));
        ggml_gallocr_free(allocator);
        return false;
    }
    if (!publishes.empty()) {
        Clock::time_point publish_begin;
        if (timing) publish_begin = Clock::now();
        for (const GraphDevicePublish & publish : publishes) {
            ggml_backend_tensor_copy_async(
                backend, backend, publish.source, publish.destination);
        }
        // Publication sources are gallocr scratch. Keep them live until the
        // complete recurrent state is visible to the following causal group.
        ggml_backend_synchronize(backend);
        if (timing) {
            timing->publish_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - publish_begin).count());
        }
    }
    for (const GraphOutput & output : outputs) {
        ggml_backend_tensor_get(
            output.tensor, output.data, 0, output.bytes);
    }
    ggml_gallocr_free(allocator);
    return true;
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
    ggml_tensor * attn_input = nullptr;
    std::vector<ggml_tensor *> checkpoints;
    ggml_tensor * prefix = nullptr;
    ggml_tensor * routed = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * shared = nullptr;
    ggml_tensor * router_input = nullptr;
    ggml_tensor * replay_staging = nullptr;
    std::vector<ggml_tensor *> replay_destinations;
    int checkpoint_count = 0;
    bool deferred_router = false;
    uint64_t executions = 0;
};

struct RoutedTailGraphOutputs {
    ggml_tensor * routed = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * shared = nullptr;
    ggml_tensor * router_input = nullptr;
};

struct PersistentPreparedTailGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * prepared = nullptr;
    RoutedTailGraphOutputs outputs;
    bool deferred_router = false;
    uint64_t executions = 0;
};

constexpr int kDeferredRouterWidth = 8;
constexpr int kExactCoreGroupWidth = 4;
constexpr int kExactCoreGroupWidth8 = 8;

struct PersistentRouter8Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * selected = nullptr;
    ggml_tensor * route_weights = nullptr;
};

struct PersistentExactCore8Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * hidden = nullptr;
    std::vector<ggml_tensor *> checkpoints;
    ggml_tensor * attn_input = nullptr;
    ggml_tensor * prefix = nullptr;
    ggml_tensor * prepared = nullptr;
    KdaTerminalState terminal;
    ggml_tensor * conv_destination = nullptr;
    ggml_tensor * ssm_destination = nullptr;
    bool prepared_to_host = false;
};

int exact_qk_core_launches(
        const KimiK3Layer & layer,
        std::string * unsupported);

bool build_persistent_exact_core8_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        KimiK3LayerCache & layer_cache,
        int checkpoint_count,
        bool banked,
        bool prepared_to_host,
        PersistentExactCore8Graph & out) {
    if (!layer_cache.replay_input ||
        layer_cache.replay_input->ne[0] != w.n_embd ||
        layer_cache.replay_input->ne[1] < kExactCoreGroupWidth8) {
        return false;
    }
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 32768, false);
    out.hidden = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, kExactCoreGroupWidth8);
    if (!out.graph || !out.hidden) return false;
    ggml_set_input(out.hidden);

    AttnResBank residuals;
    residuals.ctx = out.ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = kExactCoreGroupWidth8;
    out.checkpoints.reserve(static_cast<size_t>(checkpoint_count));
    for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            out.ctx, GGML_TYPE_F32, w.n_embd, kExactCoreGroupWidth8);
        if (!tensor) return false;
        ggml_set_input(tensor);
        out.checkpoints.push_back(tensor);
        residuals.push(tensor);
    }

    ggml_tensor * cur = out.hidden;
    out.attn_input = residuals.mix(cur, layer.attn_res_score);
    cur = out.attn_input;
    if (banked) residuals.push(out.hidden);
    cur = rms_norm(out.ctx, cur, layer.attn_norm, w.rms_eps);
    cur = build_kda(
        out.ctx, out.graph, w, layer, layer_cache, cur,
        /*commit_state=*/false, /*capture_replay=*/true,
        /*replay_token_offset=*/0, &out.terminal);
    out.prefix = banked ? cur : ggml_add(out.ctx, out.hidden, cur);
    out.prepared = residuals.mix(out.prefix, layer.ffn_res_score);
    out.prepared = rms_norm(
        out.ctx, out.prepared, layer.ffn_norm, w.rms_eps);
    out.prepared_to_host = prepared_to_host;
    out.conv_destination = layer_cache.conv_state;
    out.ssm_destination = layer_cache.ssm_state;
    if (!out.attn_input || !out.prefix || !out.prepared ||
        !out.terminal.conv || !out.terminal.ssm ||
        !same_tensor_layout(out.terminal.conv, out.conv_destination) ||
        !same_tensor_layout(out.terminal.ssm, out.ssm_destination)) {
        return false;
    }

    // Match exact_multirow_core_group's output/publication traversal. The
    // attention input is retained for the optional DSpark capture readback;
    // it is already a dependency of prefix and adds no arithmetic.
    const std::vector<ggml_tensor *> outputs = prepared_to_host
        ? std::vector<ggml_tensor *>({
              out.prefix, out.prepared, out.attn_input,
              out.terminal.conv, out.terminal.ssm})
        : std::vector<ggml_tensor *>({
              out.prefix, out.attn_input, out.terminal.conv,
              out.terminal.ssm, out.prepared});
    for (ggml_tensor * output : outputs) {
        ggml_set_output(output);
        ggml_build_forward_expand(out.graph, output);
    }
    return true;
}

bool build_routed_tail_graph(
        ggml_context * ctx,
        ggml_cgraph * graph,
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        ggml_tensor * prepared,
        bool deferred_router,
        RoutedTailGraphOutputs & out) {
    if (!ctx || !graph || !prepared) return false;
    out.routed = ggml_mul_mat(ctx, layer.ffn_routed_down, prepared);
    if (deferred_router) {
        out.router_input = prepared;
    } else {
        TopKMoeRouterResult router = build_kimi_router(
            ctx, graph, w, layer, prepared);
        out.selected = ggml_cont(ctx, router.selected);
        out.route_weights = ggml_cont(ctx, router.weights_2d);
    }
    ggml_tensor * shared_gate =
        ggml_mul_mat(ctx, layer.ffn_gate_shexp, prepared);
    ggml_tensor * shared_up =
        ggml_mul_mat(ctx, layer.ffn_up_shexp, prepared);
    out.shared = situ(
        ctx, shared_gate, shared_up, w.situ_beta, w.situ_linear_beta);
    out.shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, out.shared);
    const bool router_valid = deferred_router
        ? out.router_input == prepared
        : out.router_input == nullptr && out.selected && out.route_weights;
    return out.routed && out.shared && router_valid;
}

bool build_persistent_prepared_tail_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        bool deferred_router,
        int width,
        PersistentPreparedTailGraph & out) {
    if (width <= 0) return false;
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 16384, false);
    out.prepared = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, width);
    if (!out.graph || !out.prepared) return false;
    ggml_set_input(out.prepared);
    out.deferred_router = deferred_router;
    if (!build_routed_tail_graph(
            out.ctx, out.graph, w, layer, out.prepared,
            deferred_router, out.outputs)) {
        return false;
    }
    if (deferred_router) {
        for (ggml_tensor * output : {
                 out.outputs.routed, out.outputs.router_input,
                 out.outputs.shared}) {
            ggml_set_output(output);
            ggml_build_forward_expand(out.graph, output);
        }
    } else {
        for (ggml_tensor * output : {
                 out.outputs.routed, out.outputs.selected,
                 out.outputs.route_weights, out.outputs.shared}) {
            ggml_set_output(output);
            ggml_build_forward_expand(out.graph, output);
        }
    }
    return true;
}

bool build_persistent_router8_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        ggml_tensor * staging,
        PersistentRouter8Graph & out) {
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx || !staging || staging->ne[0] != w.n_embd ||
        staging->ne[1] != kDeferredRouterWidth ||
        staging->type != GGML_TYPE_F32) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 4096, false);
    if (!out.graph) return false;
    TopKMoeRouterResult router = build_kimi_router(
        out.ctx, out.graph, w, layer, staging);
    out.selected = ggml_cont(out.ctx, router.selected);
    out.route_weights = ggml_cont(out.ctx, router.weights_2d);
    for (ggml_tensor * output : {out.selected, out.route_weights}) {
        if (!output) return false;
        ggml_set_output(output);
        ggml_build_forward_expand(out.graph, output);
    }
    return true;
}

bool build_persistent_routed_graph(
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        KimiK3LayerCache & layer_cache,
        int checkpoint_count,
        bool banked,
        bool replay_variant,
        bool deferred_router,
        PersistentRoutedGraph & out) {
    out.ctx = new_kimi_persistent_context();
    if (!out.ctx) return false;
    out.graph = ggml_new_graph_custom(out.ctx, 32768, false);
    out.hidden = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, w.n_embd, 1);
    if (!out.graph || !out.hidden) return false;
    ggml_set_input(out.hidden);
    out.checkpoint_count = checkpoint_count;
    out.deferred_router = deferred_router;

    AttnResBank residuals;
    residuals.ctx = out.ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = 1;
    out.checkpoints.reserve(static_cast<size_t>(checkpoint_count));
    for (int checkpoint = 0; checkpoint < checkpoint_count; ++checkpoint) {
        ggml_tensor * tensor = ggml_new_tensor_2d(
            out.ctx, GGML_TYPE_F32, w.n_embd, 1);
        if (!tensor) return false;
        ggml_set_input(tensor);
        out.checkpoints.push_back(tensor);
        residuals.push(tensor);
    }

    out.prefix = out.hidden;
    out.attn_input = residuals.mix(out.prefix, layer.attn_res_score);
    ggml_tensor * cur = out.attn_input;
    if (banked) residuals.push(out.prefix);
    cur = rms_norm(out.ctx, cur, layer.attn_norm, w.rms_eps);
    // Only the macro variant records this normalized KDA input. The ordinary
    // graph stays byte-for-byte on the established decode path even when the
    // cache is macro-capable.
    if (replay_variant) {
        if (!layer_cache.replay_input) return false;
        out.replay_staging = ggml_dup_tensor(out.ctx, cur);
        if (!out.replay_staging) return false;
        ggml_set_output(out.replay_staging);
        ggml_build_forward_expand(
            out.graph, ggml_cpy(out.ctx, cur, out.replay_staging));
        out.replay_destinations.reserve(
            static_cast<size_t>(layer_cache.replay_input->ne[1]));
        for (int64_t token = 0;
             token < layer_cache.replay_input->ne[1]; ++token) {
            ggml_tensor * destination = ggml_view_2d(
                out.ctx, layer_cache.replay_input, w.n_embd, 1,
                layer_cache.replay_input->nb[1],
                static_cast<size_t>(token) *
                    layer_cache.replay_input->nb[1]);
            if (!destination ||
                ggml_backend_view_init(destination) != GGML_STATUS_SUCCESS) {
                return false;
            }
            out.replay_destinations.push_back(destination);
        }
    }
    cur = build_kda(
        out.ctx, out.graph, w, layer, layer_cache, cur,
        /*commit_state=*/true, /*capture_replay=*/false);
    out.prefix = banked ? cur : ggml_add(out.ctx, out.prefix, cur);
    cur = residuals.mix(out.prefix, layer.ffn_res_score);
    cur = rms_norm(out.ctx, cur, layer.ffn_norm, w.rms_eps);

    RoutedTailGraphOutputs tail;
    if (!build_routed_tail_graph(
            out.ctx, out.graph, w, layer, cur, deferred_router, tail)) {
        return false;
    }
    out.routed = tail.routed;
    out.selected = tail.selected;
    out.route_weights = tail.route_weights;
    out.shared = tail.shared;
    out.router_input = tail.router_input;

    if (deferred_router) {
        for (ggml_tensor * output : {
                 out.prefix, out.routed, out.router_input, out.shared}) {
            if (!output) return false;
            ggml_set_output(output);
            ggml_build_forward_expand(out.graph, output);
        }
    } else {
        // Preserve the established P46 graph traversal and allocation order.
        for (ggml_tensor * output : {
                 out.prefix, out.routed, out.selected,
                 out.route_weights, out.shared}) {
            if (!output) return false;
            ggml_set_output(output);
            ggml_build_forward_expand(out.graph, output);
        }
    }
    if (use_dspark_attn_res_capture()) {
        ggml_set_output(out.attn_input);
        ggml_build_forward_expand(out.graph, out.attn_input);
    }
    return true;
}

class PersistentRoutedPreparation {
public:
    ~PersistentRoutedPreparation() {
        if (backend_) ggml_backend_synchronize(backend_);
        size_t invalidated_graphs = 0;
        if (backend_ && ggml_backend_is_cuda(backend_)) {
            for (auto * entries : {&entries_, &replay_entries_}) {
                for (PersistentRoutedGraph & entry : *entries) {
                    if (!entry.ctx) continue;
                    invalidated_graphs +=
                        ggml_backend_cuda_graph_invalidate_range(
                            backend_, ggml_get_mem_buffer(entry.ctx),
                            ggml_get_mem_size(entry.ctx));
                }
            }
            for (PersistentPreparedTailGraph & entry :
                 prepared_tail_entries_) {
                if (!entry.ctx) continue;
                invalidated_graphs +=
                    ggml_backend_cuda_graph_invalidate_range(
                        backend_, ggml_get_mem_buffer(entry.ctx),
                        ggml_get_mem_size(entry.ctx));
            }
            for (PersistentRouter8Graph & entry : router8_entries_) {
                if (!entry.ctx) continue;
                invalidated_graphs +=
                    ggml_backend_cuda_graph_invalidate_range(
                        backend_, ggml_get_mem_buffer(entry.ctx),
                        ggml_get_mem_size(entry.ctx));
            }
            for (PersistentExactCore8Graph & entry : exact_core8_entries_) {
                if (!entry.ctx) continue;
                invalidated_graphs +=
                    ggml_backend_cuda_graph_invalidate_range(
                        backend_, ggml_get_mem_buffer(entry.ctx),
                        ggml_get_mem_size(entry.ctx));
            }
        }
        if (backend_) {
            std::fprintf(stderr,
                "[kimi-k3-p46] finalized graphs=%zu executions=%llu "
                "replay-executions=%llu prepared-tail-executions=%llu "
                "router8-executions=%llu "
                "exact-core8-executions=%llu "
                "workspace-bytes=%zu exact-core8-workspace-bytes=%zu "
                "metadata-bytes=%zu invalidated-native-graphs=%zu\n",
                graph_count_,
                static_cast<unsigned long long>(executions_),
                static_cast<unsigned long long>(replay_executions_),
                static_cast<unsigned long long>(prepared_tail_executions_),
                static_cast<unsigned long long>(router8_executions_),
                static_cast<unsigned long long>(exact_core8_executions_),
                workspace_bytes_, exact_core8_workspace_bytes_,
                metadata_bytes_, invalidated_graphs);
        }
        if (allocator_) ggml_gallocr_free(allocator_);
        if (exact_core8_allocator_) {
            ggml_gallocr_free(exact_core8_allocator_);
        }
        for (auto * entries : {&entries_, &replay_entries_}) {
            for (PersistentRoutedGraph & entry : *entries) {
                if (entry.ctx) ggml_free(entry.ctx);
            }
        }
        for (PersistentPreparedTailGraph & entry : prepared_tail_entries_) {
            if (entry.ctx) ggml_free(entry.ctx);
        }
        for (PersistentRouter8Graph & entry : router8_entries_) {
            if (entry.ctx) ggml_free(entry.ctx);
        }
        for (PersistentExactCore8Graph & entry : exact_core8_entries_) {
            if (entry.ctx) ggml_free(entry.ctx);
        }
        if (router8_staging_buffer_) {
            ggml_backend_buffer_free(router8_staging_buffer_);
        }
        if (router8_staging_ctx_) ggml_free(router8_staging_ctx_);
    }

    bool initialize(
            ggml_backend_t backend,
            const KimiK3Weights & w,
            KimiK3Cache & cache,
            bool deferred_router,
            int prepared_tail_width,
            bool include_mla_prepared_tail,
            bool include_exact_core8,
            std::string * error) {
        if (backend_ || !backend || w.n_layer <= 0 ||
            static_cast<int>(cache.layers.size()) != w.n_layer ||
            (prepared_tail_width != 0 && prepared_tail_width != 1 &&
             prepared_tail_width != kExactCoreGroupWidth &&
             prepared_tail_width != kExactCoreGroupWidth8) ||
            (prepared_tail_width > 1 &&
             !deferred_router) ||
            (include_mla_prepared_tail &&
             prepared_tail_width != kExactCoreGroupWidth &&
             prepared_tail_width != kExactCoreGroupWidth8)) {
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
        replay_entries_.resize(static_cast<size_t>(w.n_layer));
        if (include_exact_core8) {
            exact_core8_entries_.resize(static_cast<size_t>(w.n_layer));
        }
        if (prepared_tail_width > 0) {
            prepared_tail_entries_.resize(static_cast<size_t>(w.n_layer));
        }
        if (deferred_router) {
            router8_staging_ctx_ = new_kimi_persistent_context();
            if (!router8_staging_ctx_) {
                return fail(error,
                    "cannot create persistent width8 router staging context");
            }
            router8_staging_ = ggml_new_tensor_2d(
                router8_staging_ctx_, GGML_TYPE_F32, w.n_embd,
                kDeferredRouterWidth);
            if (!router8_staging_) {
                return fail(error,
                    "cannot create persistent width8 router staging tensor");
            }
            router8_staging_rows_.reserve(kDeferredRouterWidth);
            for (int row = 0; row < kDeferredRouterWidth; ++row) {
                ggml_tensor * view = ggml_view_2d(
                    router8_staging_ctx_, router8_staging_, w.n_embd, 1,
                    router8_staging_->nb[1],
                    static_cast<size_t>(row) * router8_staging_->nb[1]);
                if (!view) {
                    return fail(error,
                        "cannot create persistent width8 router staging view");
                }
                router8_staging_rows_.push_back(view);
            }
            if (prepared_tail_width == kExactCoreGroupWidth) {
                router8_staging_halves_.reserve(
                    kDeferredRouterWidth / kExactCoreGroupWidth);
                for (int row = 0; row < kDeferredRouterWidth;
                     row += kExactCoreGroupWidth) {
                    ggml_tensor * view = ggml_view_2d(
                        router8_staging_ctx_, router8_staging_, w.n_embd,
                        kExactCoreGroupWidth, router8_staging_->nb[1],
                        static_cast<size_t>(row) * router8_staging_->nb[1]);
                    if (!view) {
                        return fail(error,
                            "cannot create persistent width8 router half view");
                    }
                    router8_staging_halves_.push_back(view);
                }
            }
            router8_staging_buffer_ = ggml_backend_alloc_ctx_tensors(
                router8_staging_ctx_, backend_);
            if (!router8_staging_buffer_) {
                return fail(error,
                    "cannot allocate persistent width8 router staging tensor");
            }
            router8_staging_bytes_ = ggml_nbytes(router8_staging_);
            metadata_bytes_ += ggml_used_mem(router8_staging_ctx_);
        }

        ggml_gallocr_t measure = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!measure) {
            return fail(error,
                "cannot create persistent routed-preparation measure allocator");
        }
        ggml_cgraph * largest = nullptr;
        size_t largest_bytes = 0;
        ggml_cgraph * largest_exact_core8 = nullptr;
        size_t largest_exact_core8_bytes = 0;
        const auto measure_graph = [&](ggml_context * ctx,
                                       ggml_cgraph * graph) {
            size_t required = 0;
            ggml_gallocr_reserve_n_size(
                measure, graph, nullptr, nullptr, &required);
            if (required > largest_bytes) {
                largest_bytes = required;
                largest = graph;
            }
            metadata_bytes_ += ggml_used_mem(ctx);
            ++graph_count_;
        };
        const auto measure_exact_core8_graph = [&] (
                ggml_context * ctx, ggml_cgraph * graph) {
            size_t required = 0;
            ggml_gallocr_reserve_n_size(
                measure, graph, nullptr, nullptr, &required);
            if (required > largest_exact_core8_bytes) {
                largest_exact_core8_bytes = required;
                largest_exact_core8 = graph;
            }
            metadata_bytes_ += ggml_used_mem(ctx);
            ++graph_count_;
        };
        for (int il = w.n_dense_lead; il < w.n_layer; ++il) {
            const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
            if (layer.recurrent) {
                PersistentRoutedGraph & entry =
                    entries_[static_cast<size_t>(il)];
                const int checkpoint_count =
                    (il + w.attn_res_block_size - 1) /
                    w.attn_res_block_size;
                if (!build_persistent_routed_graph(
                        w, layer, cache.layers[static_cast<size_t>(il)],
                        checkpoint_count,
                        il % w.attn_res_block_size == 0,
                        /*replay_variant=*/false,
                        /*deferred_router=*/false, entry)) {
                    ggml_gallocr_free(measure);
                    return fail(error,
                        "cannot build persistent routed-preparation graph for layer " +
                        std::to_string(il));
                }
                measure_graph(entry.ctx, entry.graph);
                if (cache.layers[static_cast<size_t>(il)].replay_input) {
                    PersistentRoutedGraph & replay =
                        replay_entries_[static_cast<size_t>(il)];
                    if (!build_persistent_routed_graph(
                            w, layer, cache.layers[static_cast<size_t>(il)],
                            checkpoint_count,
                            il % w.attn_res_block_size == 0,
                            /*replay_variant=*/true,
                            deferred_router, replay)) {
                        ggml_gallocr_free(measure);
                        return fail(error,
                            "cannot build persistent macro replay graph for layer " +
                            std::to_string(il));
                    }
                    measure_graph(replay.ctx, replay.graph);
                    ++replay_graph_count_;
                }
            }
            if (prepared_tail_width > 0 &&
                (layer.recurrent || include_mla_prepared_tail)) {
                PersistentPreparedTailGraph & prepared =
                    prepared_tail_entries_[static_cast<size_t>(il)];
                if (prepared_tail_width == kExactCoreGroupWidth ||
                    prepared_tail_width == kExactCoreGroupWidth8) {
                    for (const ggml_tensor * weight : {
                             layer.ffn_routed_down, layer.ffn_gate_shexp,
                             layer.ffn_up_shexp, layer.ffn_down_shexp}) {
                        const bool exact_type = weight &&
                            (prepared_tail_width == kExactCoreGroupWidth8
                                ? weight->type == GGML_TYPE_Q6_K
                                : (weight->type == GGML_TYPE_Q4_K ||
                                   weight->type == GGML_TYPE_Q6_K));
                        if (!exact_type) {
                            ggml_gallocr_free(measure);
                            return fail(error,
                                "exact prepared tail width " +
                                std::to_string(prepared_tail_width) +
                                " requires four " +
                                (prepared_tail_width == kExactCoreGroupWidth8
                                    ? std::string("Q6_K")
                                    : std::string("Q4_K/Q6_K")) +
                                " matrices at layer " +
                                std::to_string(il));
                        }
                    }
                }
                if (!build_persistent_prepared_tail_graph(
                        w, layer, deferred_router, prepared_tail_width,
                        prepared)) {
                    ggml_gallocr_free(measure);
                    return fail(error,
                        "cannot build persistent prepared-tail graph for layer " +
                        std::to_string(il));
                }
                measure_graph(prepared.ctx, prepared.graph);
                ++prepared_tail_graph_count_;
            }
        }
        if (deferred_router) {
            router8_entries_.resize(static_cast<size_t>(w.n_layer));
            for (int il = w.n_dense_lead; il < w.n_layer; ++il) {
                PersistentRouter8Graph & router8 =
                    router8_entries_[static_cast<size_t>(il)];
                if (!build_persistent_router8_graph(
                        w, w.layers[static_cast<size_t>(il)],
                        router8_staging_, router8)) {
                    ggml_gallocr_free(measure);
                    return fail(error,
                        "cannot build persistent width8 router graph for layer " +
                        std::to_string(il));
                }
                measure_graph(router8.ctx, router8.graph);
                ++router8_graph_count_;
            }
        }
        if (include_exact_core8) {
            for (int il = 0; il < w.n_layer; ++il) {
                const KimiK3Layer & layer =
                    w.layers[static_cast<size_t>(il)];
                if (!layer.recurrent) continue;
                std::string unsupported;
                if (exact_qk_core_launches(layer, &unsupported) <= 0) {
                    ggml_gallocr_free(measure);
                    return fail(error,
                        "persistent exact Core8 found unsupported KDA "
                        "projection at layer " + std::to_string(il) +
                        ": " + unsupported);
                }
                PersistentExactCore8Graph & core8 =
                    exact_core8_entries_[static_cast<size_t>(il)];
                const int checkpoint_count =
                    (il + w.attn_res_block_size - 1) /
                    w.attn_res_block_size;
                if (!build_persistent_exact_core8_graph(
                        w, layer, cache.layers[static_cast<size_t>(il)],
                        checkpoint_count,
                        il % w.attn_res_block_size == 0,
                        il < w.n_dense_lead, core8)) {
                    ggml_gallocr_free(measure);
                    return fail(error,
                        "cannot build persistent exact Core8 graph for layer " +
                        std::to_string(il));
                }
                measure_exact_core8_graph(core8.ctx, core8.graph);
                ++exact_core8_graph_count_;
            }
        }
        ggml_gallocr_free(measure);
        if (!largest || graph_count_ == 0 || largest_bytes == 0) {
            return fail(error,
                "persistent routed preparation found no recurrent routed graph");
        }

        allocator_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!allocator_ ||
            !ggml_gallocr_reserve(allocator_, largest)) {
            return fail(error,
                "cannot reserve persistent routed-preparation workspace");
        }
        const size_t reserved =
            ggml_gallocr_get_buffer_size(allocator_, 0);
        for (auto * entries : {&entries_, &replay_entries_}) {
            for (PersistentRoutedGraph & entry : *entries) {
                if (!entry.graph) continue;
                if (!ggml_gallocr_alloc_graph(allocator_, entry.graph) ||
                    ggml_gallocr_get_buffer_size(allocator_, 0) != reserved) {
                    return fail(error,
                        "persistent routed-preparation workspace changed while "
                        "allocating immutable graphs");
                }
            }
        }
        for (PersistentPreparedTailGraph & entry : prepared_tail_entries_) {
            if (!entry.graph) continue;
            if (!ggml_gallocr_alloc_graph(allocator_, entry.graph) ||
                ggml_gallocr_get_buffer_size(allocator_, 0) != reserved) {
                return fail(error,
                    "persistent prepared-tail workspace changed while "
                    "allocating immutable graphs");
            }
        }
        for (PersistentRouter8Graph & entry : router8_entries_) {
            if (!entry.graph) continue;
            if (!ggml_gallocr_alloc_graph(allocator_, entry.graph) ||
                ggml_gallocr_get_buffer_size(allocator_, 0) != reserved) {
                return fail(error,
                    "persistent width8 router workspace changed while "
                    "allocating immutable graphs");
            }
        }
        workspace_bytes_ = reserved;
        if (include_exact_core8) {
            if (!largest_exact_core8 || largest_exact_core8_bytes == 0) {
                return fail(error,
                    "persistent exact Core8 found no recurrent graph");
            }
            exact_core8_allocator_ = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!exact_core8_allocator_ ||
                !ggml_gallocr_reserve(
                    exact_core8_allocator_, largest_exact_core8)) {
                return fail(error,
                    "cannot reserve persistent exact Core8 workspace");
            }
            const size_t exact_reserved =
                ggml_gallocr_get_buffer_size(exact_core8_allocator_, 0);
            for (PersistentExactCore8Graph & entry : exact_core8_entries_) {
                if (!entry.graph) continue;
                if (!ggml_gallocr_alloc_graph(
                        exact_core8_allocator_, entry.graph) ||
                    ggml_gallocr_get_buffer_size(
                        exact_core8_allocator_, 0) != exact_reserved) {
                    return fail(error,
                        "persistent exact Core8 workspace changed while "
                        "allocating immutable graphs");
                }
            }
            exact_core8_workspace_bytes_ = exact_reserved;
        }
        deferred_router_ = deferred_router;
        prepared_tail_width_ = prepared_tail_width;
        includes_mla_prepared_tail_ = include_mla_prepared_tail;
        includes_exact_core8_ = include_exact_core8;
        std::fprintf(stderr,
            "[kimi-k3-p46] initialized graphs=%zu replay-graphs=%zu "
            "prepared-tail-graphs=%zu prepared-tail-width=%d "
            "prepared-tail-mla=%s router8-graphs=%zu "
            "exact-core8-graphs=%zu "
            "router8-staging-bytes=%zu "
            "workspace-bytes=%zu exact-core8-workspace-bytes=%zu "
            "metadata-bytes=%zu backend=%s\n",
            graph_count_, replay_graph_count_, prepared_tail_graph_count_,
            prepared_tail_width_,
            includes_mla_prepared_tail_ ? "enabled" : "disabled",
            router8_graph_count_,
            exact_core8_graph_count_,
            router8_staging_bytes_,
            workspace_bytes_, exact_core8_workspace_bytes_, metadata_bytes_,
            ggml_backend_name(backend_));
        return true;
    }

    bool matches(
            ggml_backend_t backend,
            const KimiK3Weights & w,
            bool deferred_router,
            int prepared_tail_width,
            bool include_mla_prepared_tail,
            bool include_exact_core8) const {
        return backend_ == backend && weights_ == &w &&
            deferred_router_ == deferred_router &&
            prepared_tail_width_ == prepared_tail_width &&
            includes_mla_prepared_tail_ == include_mla_prepared_tail &&
            includes_exact_core8_ == include_exact_core8;
    }

    bool matches_identity(
            ggml_backend_t backend,
            const KimiK3Weights & w) const {
        return backend_ == backend && weights_ == &w;
    }

    bool supports(
            ggml_backend_t backend,
            const KimiK3Weights & w,
            bool deferred_router,
            int prepared_tail_width,
            bool include_mla_prepared_tail,
            bool include_exact_core8) const {
        return matches_identity(backend, w) &&
            (!deferred_router || deferred_router_) &&
            (prepared_tail_width == 0 ||
             prepared_tail_width_ == prepared_tail_width) &&
            (!include_mla_prepared_tail || includes_mla_prepared_tail_) &&
            (!include_exact_core8 || includes_exact_core8_);
    }

    bool evaluate(
            int model_layer,
            const std::vector<float> & hidden,
            const std::vector<std::vector<float>> & checkpoints,
            std::vector<float> & prefix,
            std::vector<float> & routed,
            std::vector<int32_t> & selected,
            std::vector<float> & route_weights,
            std::vector<float> & shared,
            int router_stage_row,
            int replay_token_offset,
            float * attn_capture,
            std::string * error) {
        if (!backend_ || !weights_ || model_layer < 0 ||
            model_layer >= static_cast<int>(entries_.size())) {
            return fail(error, "invalid persistent routed-preparation request");
        }
        std::vector<PersistentRoutedGraph> & selected_entries =
            replay_token_offset >= 0 ? replay_entries_ : entries_;
        PersistentRoutedGraph & entry =
            selected_entries[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        if (!entry.graph || hidden.size() != static_cast<size_t>(w.n_embd) ||
            checkpoints.size() != entry.checkpoints.size() ||
            prefix.size() != static_cast<size_t>(w.n_embd) ||
            routed.size() != static_cast<size_t>(w.n_expert_latent) ||
            selected.size() != static_cast<size_t>(w.n_expert_used) ||
            route_weights.size() != static_cast<size_t>(w.n_expert_used) ||
            shared.size() != static_cast<size_t>(w.n_embd)) {
            return fail(error,
                "persistent routed-preparation shape mismatch at layer " +
                std::to_string(model_layer));
        }
        if (entry.deferred_router != (router_stage_row >= 0) ||
            router_stage_row >= static_cast<int>(router8_staging_rows_.size())) {
            return fail(error,
                "persistent routed-preparation router shape mismatch at layer " +
                std::to_string(model_layer));
        }
        if (replay_token_offset < -1 ||
            (replay_token_offset >= 0 &&
             (!entry.replay_staging ||
              static_cast<size_t>(replay_token_offset) >=
                  entry.replay_destinations.size()))) {
            return fail(error,
                "persistent routed-preparation replay offset mismatch at layer " +
                std::to_string(model_layer));
        }
        ggml_backend_tensor_set(
            entry.hidden, hidden.data(), 0,
            hidden.size() * sizeof(float));
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            if (checkpoints[i].size() != static_cast<size_t>(w.n_embd)) {
                return fail(error,
                    "persistent routed-preparation checkpoint shape mismatch");
            }
            ggml_backend_tensor_set(
                entry.checkpoints[i], checkpoints[i].data(), 0,
                checkpoints[i].size() * sizeof(float));
        }
        ScopedCudaGraphOverrides replay_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute(backend_, entry.graph) !=
                GGML_STATUS_SUCCESS) {
            return fail(error,
                "persistent routed-preparation graph compute failed at layer " +
                std::to_string(model_layer));
        }
        if (attn_capture) {
            ggml_backend_tensor_get(
                entry.attn_input, attn_capture, 0,
                static_cast<size_t>(w.n_embd) * sizeof(float));
        }
        ggml_backend_tensor_get(
            entry.prefix, prefix.data(), 0,
            prefix.size() * sizeof(float));
        ggml_backend_tensor_get(
            entry.routed, routed.data(), 0,
            routed.size() * sizeof(float));
        if (router_stage_row >= 0) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, entry.router_input,
                router8_staging_rows_[static_cast<size_t>(router_stage_row)]);
        } else {
            ggml_backend_tensor_get(
                entry.selected, selected.data(), 0,
                selected.size() * sizeof(int32_t));
            ggml_backend_tensor_get(
                entry.route_weights, route_weights.data(), 0,
                route_weights.size() * sizeof(float));
        }
        ggml_backend_tensor_get(
            entry.shared, shared.data(), 0,
            shared.size() * sizeof(float));
        if (replay_token_offset >= 0) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, entry.replay_staging,
                entry.replay_destinations[
                    static_cast<size_t>(replay_token_offset)]);
            // The staging workspace, subsequent P46 graphs, replay commit,
            // and rollback all use this same backend stream. Their queue order
            // publishes the row without adding one synchronization per layer.
            ++replay_executions_;
        }
        ++entry.executions;
        ++executions_;
        return true;
    }

    bool evaluate_prepared(
            int model_layer,
            const float * prepared,
            std::vector<float> & routed,
            std::vector<int32_t> & selected,
            std::vector<float> & route_weights,
            std::vector<float> & shared,
            int router_stage_row,
            std::string * error) {
        if (!backend_ || !weights_ || prepared_tail_width_ != 1 || !prepared ||
            model_layer < weights_->n_dense_lead ||
            model_layer >= static_cast<int>(prepared_tail_entries_.size())) {
            return fail(error, "invalid persistent prepared-tail request");
        }
        PersistentPreparedTailGraph & entry =
            prepared_tail_entries_[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        if (!entry.graph || !entry.prepared ||
            routed.size() != static_cast<size_t>(w.n_expert_latent) ||
            selected.size() != static_cast<size_t>(w.n_expert_used) ||
            route_weights.size() != static_cast<size_t>(w.n_expert_used) ||
            shared.size() != static_cast<size_t>(w.n_embd)) {
            return fail(error,
                "persistent prepared-tail shape mismatch at layer " +
                std::to_string(model_layer));
        }
        if (router_stage_row < -1 ||
            entry.deferred_router != (router_stage_row >= 0) ||
            router_stage_row >= static_cast<int>(router8_staging_rows_.size())) {
            return fail(error,
                "persistent prepared-tail router shape mismatch at layer " +
                std::to_string(model_layer));
        }
        ggml_backend_tensor_set(
            entry.prepared, prepared, 0,
            static_cast<size_t>(w.n_embd) * sizeof(float));
        // The grouped causal graph already published the normalized pre-KDA
        // row to replay_input. This post-F4 tail is stateless and must never
        // overwrite that replay contract.
        ScopedCudaGraphOverrides graph_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute(backend_, entry.graph) !=
                GGML_STATUS_SUCCESS) {
            return fail(error,
                "persistent prepared-tail graph compute failed at layer " +
                std::to_string(model_layer));
        }
        ggml_backend_tensor_get(
            entry.outputs.routed, routed.data(), 0,
            routed.size() * sizeof(float));
        if (router_stage_row >= 0) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, entry.outputs.router_input,
                router8_staging_rows_[static_cast<size_t>(router_stage_row)]);
        } else {
            ggml_backend_tensor_get(
                entry.outputs.selected, selected.data(), 0,
                selected.size() * sizeof(int32_t));
            ggml_backend_tensor_get(
                entry.outputs.route_weights, route_weights.data(), 0,
                route_weights.size() * sizeof(float));
        }
        ggml_backend_tensor_get(
            entry.outputs.shared, shared.data(), 0,
            shared.size() * sizeof(float));
        ++entry.executions;
        ++prepared_tail_executions_;
        return true;
    }

    ggml_tensor * prepared_tail_input(
            int model_layer, int width) const {
        if (!backend_ || !weights_ ||
            prepared_tail_width_ != width ||
            (width != kExactCoreGroupWidth &&
             width != kExactCoreGroupWidth8) ||
            !deferred_router_ ||
            model_layer < weights_->n_dense_lead ||
            model_layer >=
                static_cast<int>(prepared_tail_entries_.size())) {
            return nullptr;
        }
        const PersistentPreparedTailGraph & entry =
            prepared_tail_entries_[static_cast<size_t>(model_layer)];
        return entry.graph && entry.prepared &&
                entry.prepared->ne[1] == width
            ? entry.prepared : nullptr;
    }

    ggml_tensor * exact_group_prepared_destination(
            int model_layer, int group_width) const {
        ggml_tensor * tail_input = prepared_tail_input(
            model_layer, prepared_tail_width_);
        if (!tail_input || group_width < prepared_tail_width_ ||
            group_width % prepared_tail_width_ != 0) {
            return nullptr;
        }
        if (group_width == prepared_tail_width_) return tail_input;
        if (group_width != kExactCoreGroupWidth8 ||
            prepared_tail_width_ != kExactCoreGroupWidth ||
            !router8_staging_ ||
            router8_staging_->ne[0] != weights_->n_embd ||
            router8_staging_->ne[1] != kExactCoreGroupWidth8) {
            return nullptr;
        }
        // This buffer is allocated independently of the transient core
        // gallocr. Tail4 needs both width8 halves to remain alive while its
        // one persistent input is reused; Tail8 publishes directly to that
        // input and takes the early return above.
        return router8_staging_;
    }

    ggml_tensor * prepared_tail_width4_source_half(int half) const {
        if (prepared_tail_width_ != kExactCoreGroupWidth ||
            half < 0 ||
            half >= static_cast<int>(router8_staging_halves_.size())) {
            return nullptr;
        }
        return router8_staging_halves_[static_cast<size_t>(half)];
    }

    bool evaluate_prepared_exact(
            int model_layer,
            int token_begin,
            float * routed,
            size_t routed_count,
            float * shared,
            size_t shared_count,
            ggml_tensor * prepared_source,
            std::string * error) {
        const int width = prepared_tail_width_;
        if (!backend_ || !weights_ ||
            (width != kExactCoreGroupWidth &&
             width != kExactCoreGroupWidth8) ||
            !deferred_router_ ||
            model_layer < weights_->n_dense_lead ||
            model_layer >=
                static_cast<int>(prepared_tail_entries_.size()) ||
            token_begin < 0 || token_begin % width != 0 ||
            !routed || !shared) {
            return fail(error,
                "invalid persistent exact prepared-tail request");
        }
        const int router_part =
            (token_begin % kDeferredRouterWidth) / width;
        if (router_part < 0 ||
            (width == kExactCoreGroupWidth &&
             router_part >=
                static_cast<int>(router8_staging_halves_.size())) ||
            (width == kExactCoreGroupWidth8 && router_part != 0)) {
            return fail(error,
                "invalid persistent exact router staging partition");
        }
        PersistentPreparedTailGraph & entry =
            prepared_tail_entries_[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        const size_t expected_routed =
            static_cast<size_t>(w.n_expert_latent) * width;
        const size_t expected_shared =
            static_cast<size_t>(w.n_embd) * width;
        ggml_tensor * router_destination = width == kExactCoreGroupWidth8
            ? router8_staging_
            : router8_staging_halves_[static_cast<size_t>(router_part)];
        if (!entry.graph || !entry.prepared || !entry.deferred_router ||
            entry.prepared->ne[1] != width ||
            !same_tensor_layout(
                entry.outputs.router_input, router_destination) ||
            (prepared_source &&
             !same_tensor_layout(prepared_source, entry.prepared)) ||
            routed_count != expected_routed ||
            shared_count != expected_shared) {
            return fail(error,
                "persistent exact prepared-tail width " +
                std::to_string(width) + " shape mismatch at layer " +
                std::to_string(model_layer));
        }

        const size_t exact_qk_launches_before = width == kExactCoreGroupWidth8
            ? ggml_backend_cuda_get_exact_qk_width8_launch_count()
            : ggml_backend_cuda_get_exact_qk_width4_launch_count();
        const bool first_execution = entry.executions == 0;
        if (prepared_source) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, prepared_source, entry.prepared);
        }
        {
            // Keep every tail column on the same one-column vec-dot and
            // reduction order as the exact causal graph. Subsequent calls
            // may replay the already-qualified native graph.
            ScopedCudaGraphOverrides graph_scope(
                /*disable_graphs=*/false,
                /*mmvq_max_ncols=*/width,
                /*skip_property_check=*/true,
                /*exact_qk_width4=*/width == kExactCoreGroupWidth,
                /*exact_qk_width8=*/width == kExactCoreGroupWidth8);
            if (ggml_backend_graph_compute(backend_, entry.graph) !=
                    GGML_STATUS_SUCCESS) {
                return fail(error,
                    "persistent exact prepared-tail width " +
                    std::to_string(width) + " graph compute failed at layer " +
                    std::to_string(model_layer));
            }
        }
        const size_t exact_qk_launches = (width == kExactCoreGroupWidth8
            ? ggml_backend_cuda_get_exact_qk_width8_launch_count()
            : ggml_backend_cuda_get_exact_qk_width4_launch_count()) -
                exact_qk_launches_before;
        if (first_execution && exact_qk_launches != 4) {
            return fail(error,
                "persistent exact prepared tail width " +
                std::to_string(width) + " dispatched " +
                std::to_string(exact_qk_launches) +
                " exact Q4_K/Q6_K kernels instead of four at layer " +
                std::to_string(model_layer));
        }

        ggml_backend_tensor_copy_async(
            backend_, backend_, entry.outputs.router_input,
            router_destination);
        ggml_backend_tensor_get(
            entry.outputs.routed, routed, 0,
            routed_count * sizeof(float));
        ggml_backend_tensor_get(
            entry.outputs.shared, shared, 0,
            shared_count * sizeof(float));
        ++entry.executions;
        ++prepared_tail_executions_;
        return true;
    }

    bool evaluate_exact_core8(
            int model_layer,
            int token_begin,
            const std::vector<float> & hidden,
            const std::vector<std::vector<float>> & checkpoints,
            std::vector<float> & prefix,
            std::vector<float> & prepared,
            ggml_tensor * prepared_destination,
            float * attn_capture,
            uint64_t & graph_ns,
            uint64_t & publish_ns,
            std::string * error) {
        using Clock = std::chrono::steady_clock;
        if (!backend_ || !weights_ || !includes_exact_core8_ ||
            token_begin != 0 ||
            model_layer < 0 ||
            model_layer >= static_cast<int>(exact_core8_entries_.size())) {
            return fail(error,
                "persistent exact Core8 request is outside its envelope");
        }
        PersistentExactCore8Graph & entry =
            exact_core8_entries_[static_cast<size_t>(model_layer)];
        const KimiK3Weights & w = *weights_;
        const KimiK3Layer & layer =
            w.layers[static_cast<size_t>(model_layer)];
        const size_t values =
            static_cast<size_t>(w.n_embd) * kExactCoreGroupWidth8;
        if (!layer.recurrent || !entry.graph || !entry.hidden ||
            hidden.size() != values ||
            checkpoints.size() != entry.checkpoints.size() ||
            entry.prepared_to_host == (prepared_destination != nullptr) ||
            (prepared_destination &&
             !same_tensor_layout(entry.prepared, prepared_destination))) {
            return fail(error,
                "persistent exact Core8 shape mismatch at layer " +
                std::to_string(model_layer));
        }
        for (const std::vector<float> & checkpoint : checkpoints) {
            if (checkpoint.size() != values) {
                return fail(error,
                    "persistent exact Core8 checkpoint shape mismatch");
            }
        }
        std::string unsupported;
        const int expected_exact_launches =
            exact_qk_core_launches(layer, &unsupported);
        if (expected_exact_launches <= 0) {
            return fail(error,
                "persistent exact Core8 unsupported projection at layer " +
                std::to_string(model_layer) + ": " + unsupported);
        }

        ggml_backend_tensor_set(
            entry.hidden, hidden.data(), 0, values * sizeof(float));
        for (size_t checkpoint = 0; checkpoint < checkpoints.size();
             ++checkpoint) {
            ggml_backend_tensor_set(
                entry.checkpoints[checkpoint], checkpoints[checkpoint].data(),
                0, values * sizeof(float));
        }

        const size_t launches_before =
            ggml_backend_cuda_get_exact_qk_width8_launch_count();
        const Clock::time_point graph_begin = Clock::now();
        {
            // This probe persists GGML metadata/allocation only. Native HIP
            // graph replay stays disabled so every eager exact-QK launch is
            // counted on every execution.
            ScopedCudaGraphOverrides graph_scope(
                /*disable_graphs=*/true,
                /*mmvq_max_ncols=*/kExactCoreGroupWidth8,
                /*skip_property_check=*/false,
                /*exact_qk_width4=*/false,
                /*exact_qk_width8=*/true);
            if (ggml_backend_graph_compute(backend_, entry.graph) !=
                    GGML_STATUS_SUCCESS) {
                return fail(error,
                    "persistent exact Core8 graph compute failed at layer " +
                    std::to_string(model_layer));
            }
        }
        graph_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - graph_begin).count());
        const size_t observed_exact_launches =
            ggml_backend_cuda_get_exact_qk_width8_launch_count() -
            launches_before;
        if (observed_exact_launches !=
                static_cast<size_t>(expected_exact_launches)) {
            return fail(error,
                "persistent exact Core8 Q4_K/Q6_K dispatch mismatch at layer " +
                std::to_string(model_layer) + ": expected " +
                std::to_string(expected_exact_launches) + ", observed " +
                std::to_string(observed_exact_launches));
        }

        const Clock::time_point publish_begin = Clock::now();
        ggml_backend_tensor_copy_async(
            backend_, backend_, entry.terminal.conv, entry.conv_destination);
        ggml_backend_tensor_copy_async(
            backend_, backend_, entry.terminal.ssm, entry.ssm_destination);
        if (prepared_destination) {
            ggml_backend_tensor_copy_async(
                backend_, backend_, entry.prepared, prepared_destination);
        }
        ggml_backend_synchronize(backend_);
        publish_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - publish_begin).count());

        prefix.resize(values);
        prepared.clear();
        ggml_backend_tensor_get(
            entry.prefix, prefix.data(), 0, values * sizeof(float));
        if (entry.prepared_to_host) {
            prepared.resize(values);
            ggml_backend_tensor_get(
                entry.prepared, prepared.data(), 0, values * sizeof(float));
        }
        if (attn_capture) {
            ggml_backend_tensor_get(
                entry.attn_input, attn_capture, 0, values * sizeof(float));
        }
        ++exact_core8_executions_;
        return true;
    }

    bool evaluate_router8(
            int model_layer,
            int32_t * selected,
            size_t selected_count,
            float * route_weights,
            size_t weight_count,
            std::string * error) {
        if (!backend_ || !weights_ || !deferred_router_ ||
            model_layer < weights_->n_dense_lead ||
            model_layer >= static_cast<int>(router8_entries_.size()) ||
            !selected || !route_weights ||
            selected_count !=
                static_cast<size_t>(weights_->n_expert_used) *
                    kDeferredRouterWidth ||
            weight_count !=
                static_cast<size_t>(weights_->n_expert_used) *
                    kDeferredRouterWidth) {
            return fail(error, "invalid persistent width8 router request");
        }
        PersistentRouter8Graph & entry =
            router8_entries_[static_cast<size_t>(model_layer)];
        if (!entry.graph) {
            return fail(error,
                "persistent width8 router graph is missing at layer " +
                std::to_string(model_layer));
        }
        const size_t selected_bytes =
            static_cast<size_t>(weights_->n_expert_used) *
            kDeferredRouterWidth *
            sizeof(int32_t);
        const size_t weight_bytes =
            static_cast<size_t>(weights_->n_expert_used) *
            kDeferredRouterWidth *
            sizeof(float);
        ScopedCudaGraphOverrides graph_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/0,
            /*skip_property_check=*/true);
        if (ggml_backend_graph_compute(backend_, entry.graph) !=
                GGML_STATUS_SUCCESS) {
            return fail(error,
                "persistent width8 router graph compute failed at layer " +
                std::to_string(model_layer));
        }
        ggml_backend_tensor_get_async(backend_,
            entry.selected, selected, 0, selected_bytes);
        ggml_backend_tensor_get_async(backend_,
            entry.route_weights, route_weights, 0, weight_bytes);
        ggml_backend_synchronize(backend_);
        ++router8_executions_;
        return true;
    }

    ggml_tensor * router8_staging_row(int row) const {
        if (!deferred_router_ || row < 0 ||
            row >= static_cast<int>(router8_staging_rows_.size())) {
            return nullptr;
        }
        return router8_staging_rows_[static_cast<size_t>(row)];
    }

private:
    static bool fail(std::string * error, const std::string & message) {
        if (error) *error = message;
        return false;
    }

    ggml_backend_t backend_ = nullptr;
    const KimiK3Weights * weights_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_gallocr_t exact_core8_allocator_ = nullptr;
    std::vector<PersistentRoutedGraph> entries_;
    std::vector<PersistentRoutedGraph> replay_entries_;
    std::vector<PersistentPreparedTailGraph> prepared_tail_entries_;
    std::vector<PersistentRouter8Graph> router8_entries_;
    std::vector<PersistentExactCore8Graph> exact_core8_entries_;
    ggml_context * router8_staging_ctx_ = nullptr;
    ggml_backend_buffer_t router8_staging_buffer_ = nullptr;
    ggml_tensor * router8_staging_ = nullptr;
    std::vector<ggml_tensor *> router8_staging_rows_;
    std::vector<ggml_tensor *> router8_staging_halves_;
    size_t graph_count_ = 0;
    size_t replay_graph_count_ = 0;
    size_t prepared_tail_graph_count_ = 0;
    size_t router8_graph_count_ = 0;
    size_t exact_core8_graph_count_ = 0;
    size_t router8_staging_bytes_ = 0;
    size_t workspace_bytes_ = 0;
    size_t exact_core8_workspace_bytes_ = 0;
    size_t metadata_bytes_ = 0;
    uint64_t executions_ = 0;
    uint64_t replay_executions_ = 0;
    uint64_t prepared_tail_executions_ = 0;
    uint64_t router8_executions_ = 0;
    uint64_t exact_core8_executions_ = 0;
    bool deferred_router_ = false;
    bool includes_mla_prepared_tail_ = false;
    bool includes_exact_core8_ = false;
    int prepared_tail_width_ = 0;
};

PersistentRoutedPreparation * ensure_persistent_routed_preparation(
        ggml_backend_t backend,
        const KimiK3Weights & weights,
        KimiK3Cache & cache,
        bool deferred_router,
        int prepared_tail_width,
        bool include_mla_prepared_tail,
        bool include_exact_core8,
        bool require_exact_router_mode,
        std::string * error) {
    if (cache.persistent_routed_preparation) {
        auto * existing = static_cast<PersistentRoutedPreparation *>(
            cache.persistent_routed_preparation);
        const bool same_identity =
            existing->matches_identity(backend, weights);
        const bool mode_mismatch = require_exact_router_mode
            ? !existing->matches(
                backend, weights, deferred_router, prepared_tail_width,
                include_mla_prepared_tail, include_exact_core8)
            : !existing->supports(
                backend, weights, deferred_router, prepared_tail_width,
                include_mla_prepared_tail, include_exact_core8);
        if (same_identity && mode_mismatch) {
            // Exact macro replay graphs have one fixed router mode. Ordinary
            // width-one calls may reuse the router8 superset because its
            // established entries are unchanged; macro mode changes rebuild.
            delete existing;
            cache.persistent_routed_preparation = nullptr;
        }
    }
    if (!cache.persistent_routed_preparation) {
        auto * created = new (std::nothrow) PersistentRoutedPreparation;
        if (!created || !created->initialize(
                backend, weights, cache, deferred_router,
                prepared_tail_width, include_mla_prepared_tail,
                include_exact_core8,
                error)) {
            delete created;
            return nullptr;
        }
        cache.persistent_routed_preparation = created;
    }
    auto * persistent = static_cast<PersistentRoutedPreparation *>(
        cache.persistent_routed_preparation);
    const bool supported = require_exact_router_mode
        ? persistent->matches(
            backend, weights, deferred_router, prepared_tail_width,
            include_mla_prepared_tail, include_exact_core8)
        : persistent->supports(
            backend, weights, deferred_router, prepared_tail_width,
            include_mla_prepared_tail, include_exact_core8);
    if (!supported) {
        if (error) *error = "P46 backend/model changed";
        return nullptr;
    }
    return persistent;
}

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


bool restore_recurrent_snapshot(
        ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend) return false;
    for (const KimiK3LayerCache & layer : cache.layers) {
        if (layer.ssm_state &&
            (!layer.conv_state || !layer.ssm_state_snap ||
             !layer.conv_state_snap)) {
            return false;
        }
    }
    for (KimiK3LayerCache & layer : cache.layers) {
        if (!layer.ssm_state) continue;
        ggml_backend_tensor_copy_async(
            backend, backend, layer.ssm_state_snap, layer.ssm_state);
        ggml_backend_tensor_copy_async(
            backend, backend, layer.conv_state_snap, layer.conv_state);
    }
    ggml_backend_synchronize(backend);
    cache.recurrent_state_pristine = true;
    return true;
}

bool exact_terminal_pending(const KimiK3Cache & cache) {
    return cache.snapshot_valid && cache.replay_valid &&
        cache.replay_exact_rows && !cache.recurrent_state_pristine;
}

class ExactMultirowSnapshotGuard {
public:
    ExactMultirowSnapshotGuard(ggml_backend_t backend, KimiK3Cache & cache)
        : backend_(backend), cache_(cache) {
        cache_.recurrent_state_pristine = false;
    }

    ~ExactMultirowSnapshotGuard() { restore(); }

    void retain_terminal() { active_ = false; }

    void restore() {
        if (!active_) return;
        const bool restored = restore_recurrent_snapshot(backend_, cache_);
        GGML_ASSERT(restored);
        active_ = false;
    }

private:
    ggml_backend_t backend_ = nullptr;
    KimiK3Cache & cache_;
    bool active_ = true;
};

class LayerRouteObservationGuard {
public:
    explicit LayerRouteObservationGuard(KimiK3RoutedPrefillService * service)
        : service_(service) {}
    ~LayerRouteObservationGuard() {
        if (active_ && service_) service_->abort_layer_route_observation(); }
    void complete() { active_ = false; }
private:
    KimiK3RoutedPrefillService * service_ = nullptr;
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
        ExactMultirowLayerRow & output,
        PersistentRoutedPreparation * persistent = nullptr,
        bool defer_router = false,
        float * attn_capture = nullptr) {
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

    if (persistent && !dense && layer.recurrent) {
        std::vector<float> hidden(
            hidden_row, hidden_row + hidden_width);
        output.prefix.resize(hidden_width);
        output.routed.resize(static_cast<size_t>(w.n_expert_latent));
        output.selected.resize(static_cast<size_t>(w.n_expert_used));
        output.route_weights.resize(static_cast<size_t>(w.n_expert_used));
        output.shared.resize(hidden_width);
        std::string persistent_error;
        if (!persistent->evaluate(
                model_layer, hidden, checkpoint_rows, output.prefix,
                output.routed, output.selected, output.route_weights,
                output.shared,
                defer_router ? token_index % kDeferredRouterWidth : -1,
                token_index, attn_capture, &persistent_error)) {
            set_last_error(
                "Kimi-K3 P46 macro layer " +
                std::to_string(model_layer) + " failed: " +
                persistent_error);
            return false;
        }
        return true;
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
    ggml_tensor * attn_input = residuals.mix(prefix, layer.attn_res_score);
    ggml_tensor * cur = attn_input;
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
        ggml_tensor * selected = nullptr;
        ggml_tensor * route_weights = nullptr;
        if (!defer_router) {
            TopKMoeRouterResult router =
                build_kimi_router(ctx, graph, w, layer, cur);
            selected = ggml_cont(ctx, router.selected);
            route_weights = ggml_cont(ctx, router.weights_2d);
        }
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
            {shared, output.shared.data(), hidden_width * sizeof(float)},
        };
        if (defer_router) {
            ggml_tensor * staging = persistent
                ? persistent->router8_staging_row(
                    token_index % kDeferredRouterWidth) : nullptr;
            if (!staging) {
                ggml_free(ctx);
                set_last_error(
                    "Kimi-K3 width8 router staging row is unavailable");
                return false;
            }
            ggml_build_forward_expand(
                graph, ggml_cpy(ctx, cur, staging));
        } else {
            outputs.push_back({selected, output.selected.data(),
                               output.selected.size() * sizeof(int32_t)});
            outputs.push_back({route_weights, output.route_weights.data(),
                               output.route_weights.size() * sizeof(float)});
        }
    }
    if (attn_capture) {
        outputs.push_back({
            attn_input, attn_capture, hidden_width * sizeof(float)});
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

struct ExactMultirowCoreGroup {
    std::vector<float> prefix;
    std::vector<float> prepared;
    uint64_t graph_ns = 0;
    uint64_t publish_ns = 0;
};

bool run_exact_qk_width8_graph(
        ggml_backend_t backend,
        ggml_context * ctx,
        ggml_cgraph * graph,
        const std::vector<GraphInput> & inputs,
        const std::vector<GraphOutput> & outputs,
        const char * phase,
        size_t expected_launches,
        const std::vector<GraphDevicePublish> & publishes = {},
        GraphExecutionTiming * timing = nullptr) {
    const size_t launches_before =
        ggml_backend_cuda_get_exact_qk_width8_launch_count();
    bool ok = false;
    {
        ScopedCudaGraphOverrides graph_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/kExactCoreGroupWidth8,
            /*skip_property_check=*/false,
            /*exact_qk_width4=*/false,
            /*exact_qk_width8=*/true);
        ok = run_host_boundary_graph(
            backend, ctx, graph, inputs, outputs, phase, publishes, timing);
    }
    const size_t observed =
        ggml_backend_cuda_get_exact_qk_width8_launch_count() -
        launches_before;
    if (ok && observed != expected_launches) {
        set_last_error(
            std::string("Kimi-K3 ") + phase +
            ": exact width8 Q4_K/Q6_K dispatch mismatch: expected " +
            std::to_string(expected_launches) + ", observed " +
            std::to_string(observed));
        return false;
    }
    return ok;
}

bool exact_mla_projection_types(const KimiK3Layer & layer,
                                std::string * unsupported) {
    for (const auto & projection : {
             std::pair<const char *, const ggml_tensor *>("q_a", layer.wq_a),
             {"q_b", layer.wq_b},
             {"kv_a", layer.wkv_a_mqa},
             {"gate", layer.wqkv_gate},
             {"wo", layer.wo}}) {
        if (!projection.second ||
            projection.second->type != GGML_TYPE_Q6_K) {
            if (unsupported) {
                *unsupported = std::string(projection.first) + "=" +
                    (projection.second
                        ? ggml_type_name(projection.second->type) : "missing");
            }
            return false;
        }
    }
    return true;
}

bool exact_mla_scalar_attention_group(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        const KimiK3Layer & layer,
        KimiK3LayerCache & layer_cache,
        int base_position,
        const float * q_cur_values,
        const float * compact_pe_values,
        const float * gate_values,
        float * output) {
    constexpr int width8 = kExactCoreGroupWidth8;
    const size_t q_width = static_cast<size_t>(layer.wq_b->ne[1]);
    const size_t compact_width =
        static_cast<size_t>(layer.wkv_a_mqa->ne[1]);
    const size_t gate_width = static_cast<size_t>(layer.wqkv_gate->ne[1]);
    const size_t output_width = static_cast<size_t>(layer.wo->ne[0]);
    if (!q_cur_values || !compact_pe_values || !gate_values || !output ||
        q_width == 0 || compact_width == 0 || gate_width != output_width ||
        base_position < 0) {
        set_last_error("Kimi-K3 exact MLA scalar attention group is invalid");
        return false;
    }
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 exact MLA scalar attention context failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_tensor * q_cur = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(q_width), width8);
    ggml_tensor * compact_pe = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(compact_width), width8);
    ggml_tensor * projected_gate = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, static_cast<int64_t>(gate_width), width8);
    for (ggml_tensor * input : {q_cur, compact_pe, projected_gate}) {
        ggml_set_input(input);
    }
    std::vector<std::vector<float>> masks;
    masks.reserve(width8);
    std::vector<GraphInput> inputs = {
        {q_cur, q_cur_values, q_width * width8 * sizeof(float)},
        {compact_pe, compact_pe_values,
         compact_width * width8 * sizeof(float)},
        {projected_gate, gate_values,
         gate_width * width8 * sizeof(float)},
    };
    inputs.reserve(3 + width8);
    ggml_tensor * pre_wo = nullptr;
    for (int row = 0; row < width8; ++row) {
        ggml_tensor * q_row = ggml_view_2d(
            ctx, q_cur, static_cast<int64_t>(q_width), 1, q_cur->nb[1],
            static_cast<size_t>(row) * q_cur->nb[1]);
        ggml_tensor * compact_row = ggml_view_2d(
            ctx, compact_pe, static_cast<int64_t>(compact_width), 1,
            compact_pe->nb[1],
            static_cast<size_t>(row) * compact_pe->nb[1]);
        ggml_tensor * gate_row = ggml_view_2d(
            ctx, projected_gate, static_cast<int64_t>(gate_width), 1,
            projected_gate->nb[1],
            static_cast<size_t>(row) * projected_gate->nb[1]);
        const int position = base_position + row;
        masks.emplace_back(static_cast<size_t>(position + 1), 0.0f);
        ggml_tensor * mask = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, position + 1, 1);
        ggml_set_input(mask);
        inputs.push_back({
            mask, masks.back().data(), masks.back().size() * sizeof(float)});
        ggml_tensor * row_output = build_mla_absorbed_attention(
            ctx, graph, w, layer, layer_cache, q_row, compact_row,
            /*gate_input=*/nullptr, gate_row, position, mask);
        // Expanding each scalar row before constructing the next one keeps
        // its KV publication ahead of every later row on the backend stream.
        ggml_build_forward_expand(graph, row_output);
        pre_wo = pre_wo
            ? ggml_concat(ctx, pre_wo, row_output, 1) : row_output;
    }
    const bool ok = run_host_boundary_graph(
        backend, ctx, graph, inputs,
        {{pre_wo, output, output_width * width8 * sizeof(float)}},
        "P58 exact MLA8 scalar absorbed attention");
    ggml_free(ctx);
    return ok;
}

bool exact_multirow_mla_group(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        int model_layer,
        int base_pos,
        int token_begin,
        const std::vector<float> & hidden,
        const std::vector<std::vector<float>> & checkpoints,
        ExactMultirowCoreGroup & output,
        ggml_tensor * prepared_destination,
        float * attn_capture) {
    constexpr int width8 = kExactCoreGroupWidth8;
    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    const size_t group_values = hidden_width * width8;
    if (model_layer < w.n_dense_lead || model_layer >= w.n_layer ||
        token_begin < 0 || token_begin % width8 != 0 ||
        static_cast<size_t>(token_begin) * hidden_width + group_values >
            hidden.size()) {
        set_last_error("Kimi-K3 exact MLA8 request is outside its envelope");
        return false;
    }
    const KimiK3Layer & layer = w.layers[static_cast<size_t>(model_layer)];
    KimiK3LayerCache & layer_cache =
        cache.layers[static_cast<size_t>(model_layer)];
    std::string unsupported;
    if (layer.recurrent ||
        !exact_mla_projection_types(layer, &unsupported)) {
        set_last_error(
            "Kimi-K3 exact MLA8 requires five resident Q6_K projections at "
            "layer " + std::to_string(model_layer) + ": " + unsupported);
        return false;
    }

    std::vector<std::vector<float>> checkpoint_group;
    checkpoint_group.reserve(checkpoints.size());
    for (const std::vector<float> & checkpoint : checkpoints) {
        const size_t begin = static_cast<size_t>(token_begin) * hidden_width;
        if (checkpoint.size() < begin + group_values) {
            set_last_error("Kimi-K3 exact MLA8 checkpoint shape is invalid");
            return false;
        }
        checkpoint_group.emplace_back(
            checkpoint.begin() + static_cast<std::ptrdiff_t>(begin),
            checkpoint.begin() +
                static_cast<std::ptrdiff_t>(begin + group_values));
    }

    const size_t q_a_width = static_cast<size_t>(w.q_lora_rank);
    const size_t q_width = static_cast<size_t>(layer.wq_b->ne[1]);
    const size_t compact_width =
        static_cast<size_t>(layer.wkv_a_mqa->ne[1]);
    const size_t gate_width = static_cast<size_t>(layer.wqkv_gate->ne[1]);
    const size_t pre_wo_width = static_cast<size_t>(layer.wo->ne[0]);
    if (q_a_width == 0 || q_width == 0 || compact_width == 0 ||
        gate_width == 0 || gate_width != pre_wo_width) {
        set_last_error("Kimi-K3 exact MLA8 projection shape is invalid");
        return false;
    }

    std::vector<float> base_prefix(group_values);
    std::copy_n(
        hidden.data() + static_cast<size_t>(token_begin) * hidden_width,
        group_values, base_prefix.data());
    std::vector<float> q_a(q_a_width * width8);
    std::vector<float> compact_pe(compact_width * width8);
    std::vector<float> projected_gate(gate_width * width8);
    GraphExecutionTiming pre_timing;
    {
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 exact MLA8 preprojection context failed");
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
        std::vector<GraphInput> inputs;
        ggml_tensor * hidden_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, width8);
        ggml_set_input(hidden_in);
        inputs.push_back({
            hidden_in,
            hidden.data() + static_cast<size_t>(token_begin) * hidden_width,
            group_values * sizeof(float)});
        AttnResBank residuals;
        populate_attn_res_bank(
            ctx, w, width8, checkpoint_group, residuals, inputs);
        ggml_tensor * attn_input =
            residuals.mix(hidden_in, layer.attn_res_score);
        ggml_tensor * cur = attn_input;
        cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
        ggml_tensor * q_a_tensor = ggml_mul_mat(ctx, layer.wq_a, cur);
        ggml_tensor * compact_tensor =
            ggml_mul_mat(ctx, layer.wkv_a_mqa, cur);
        ggml_tensor * gate_tensor =
            ggml_mul_mat(ctx, layer.wqkv_gate, cur);
        std::vector<GraphOutput> outputs = {
                {q_a_tensor, q_a.data(), q_a.size() * sizeof(float)},
                {compact_tensor, compact_pe.data(),
                 compact_pe.size() * sizeof(float)},
                {gate_tensor, projected_gate.data(),
                 projected_gate.size() * sizeof(float)},
            };
        if (attn_capture) {
            outputs.push_back({
                attn_input, attn_capture, group_values * sizeof(float)});
        }
        const bool ok = run_exact_qk_width8_graph(
            backend, ctx, graph, inputs, outputs,
            "P58 exact MLA8 preprojections", 3, {}, &pre_timing);
        ggml_free(ctx);
        if (!ok) return false;
    }

    std::vector<float> q_cur(q_width * width8);
    GraphExecutionTiming q_b_timing;
    {
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 exact MLA8 q_b context failed");
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_tensor * q_a_input = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.q_lora_rank, width8);
        ggml_set_input(q_a_input);
        ggml_tensor * normalized_q_a = nullptr;
        for (int row = 0; row < width8; ++row) {
            ggml_tensor * q_a_row = ggml_view_2d(
                ctx, q_a_input, w.q_lora_rank, 1, q_a_input->nb[1],
                static_cast<size_t>(row) * q_a_input->nb[1]);
            ggml_tensor * normalized_row = rms_norm(
                ctx, q_a_row, layer.wq_a_norm, w.rms_eps);
            normalized_q_a = normalized_q_a
                ? ggml_concat(ctx, normalized_q_a, normalized_row, 1)
                : normalized_row;
        }
        ggml_tensor * q_cur_tensor =
            ggml_mul_mat(ctx, layer.wq_b, normalized_q_a);
        const bool ok = run_exact_qk_width8_graph(
            backend, ctx, graph,
            {{q_a_input, q_a.data(), q_a.size() * sizeof(float)}},
            {{q_cur_tensor, q_cur.data(), q_cur.size() * sizeof(float)}},
            "P58 exact MLA8 q_b projection", 1, {}, &q_b_timing);
        ggml_free(ctx);
        if (!ok) return false;
    }

    std::vector<float> pre_wo(pre_wo_width * width8);
    const auto scalar_begin = std::chrono::steady_clock::now();
    if (!exact_mla_scalar_attention_group(
            backend, w, layer, layer_cache, base_pos + token_begin,
            q_cur.data(), compact_pe.data(), projected_gate.data(),
            pre_wo.data())) {
        return false;
    }
    const uint64_t scalar_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - scalar_begin).count());

    GraphExecutionTiming finish_timing;
    {
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 exact MLA8 finish context failed");
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
        std::vector<GraphInput> inputs;
        ggml_tensor * prefix_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, width8);
        ggml_tensor * pre_wo_in = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, static_cast<int64_t>(pre_wo_width), width8);
        ggml_set_input(prefix_in);
        ggml_set_input(pre_wo_in);
        inputs.push_back({
            prefix_in, base_prefix.data(), group_values * sizeof(float)});
        inputs.push_back({
            pre_wo_in, pre_wo.data(), pre_wo.size() * sizeof(float)});
        AttnResBank residuals;
        populate_attn_res_bank(
            ctx, w, width8, checkpoint_group, residuals, inputs);
        const bool banked = model_layer % w.attn_res_block_size == 0;
        if (banked) residuals.push(prefix_in);
        ggml_tensor * mla_out = ggml_mul_mat(ctx, layer.wo, pre_wo_in);
        ggml_tensor * prefix = banked
            ? mla_out : ggml_add(ctx, prefix_in, mla_out);
        ggml_tensor * prepared = residuals.mix(prefix, layer.ffn_res_score);
        prepared = rms_norm(ctx, prepared, layer.ffn_norm, w.rms_eps);
        output.prefix.resize(group_values);
        output.prepared.clear();
        std::vector<GraphOutput> outputs = {
            {prefix, output.prefix.data(), group_values * sizeof(float)},
        };
        std::vector<GraphDevicePublish> publishes;
        if (prepared_destination) {
            publishes.push_back({prepared, prepared_destination});
        } else {
            output.prepared.resize(group_values);
            outputs.push_back({
                prepared, output.prepared.data(), group_values * sizeof(float)});
        }
        const bool ok = run_exact_qk_width8_graph(
            backend, ctx, graph, inputs, outputs,
            "P58 exact MLA8 output projection", 1, publishes,
            &finish_timing);
        ggml_free(ctx);
        if (!ok) return false;
    }
    output.graph_ns = pre_timing.compute_ns + q_b_timing.compute_ns +
        scalar_ns + finish_timing.compute_ns;
    output.publish_ns = finish_timing.publish_ns;
    return true;
}

int exact_qk_core_launches(
        const KimiK3Layer & layer,
        std::string * unsupported) {
    int launches = 0;
    for (const auto & named_weight : {
             std::pair<const char *, const ggml_tensor *>("wq", layer.wq),
             std::pair<const char *, const ggml_tensor *>("wk", layer.wk),
             std::pair<const char *, const ggml_tensor *>("wv", layer.wv),
             std::pair<const char *, const ggml_tensor *>(
                 "ssm_f_a", layer.ssm_f_a),
             std::pair<const char *, const ggml_tensor *>(
                 "ssm_f_b", layer.ssm_f_b),
             std::pair<const char *, const ggml_tensor *>(
                 "ssm_beta", layer.ssm_beta),
             std::pair<const char *, const ggml_tensor *>(
                 "ssm_g", layer.ssm_g),
             std::pair<const char *, const ggml_tensor *>("wo", layer.wo)}) {
        const ggml_tensor * weight = named_weight.second;
        if (!weight) {
            if (unsupported) *unsupported = named_weight.first;
            return -1;
        }
        if (weight->type == GGML_TYPE_Q4_K ||
            weight->type == GGML_TYPE_Q6_K) {
            ++launches;
        } else if (weight->type != GGML_TYPE_Q2_K &&
                   weight->type != GGML_TYPE_Q8_0 &&
                   weight->type != GGML_TYPE_F32) {
            if (unsupported) {
                *unsupported = std::string(named_weight.first) + "=" +
                    ggml_type_name(weight->type);
            }
            return -1;
        }
    }
    return launches;
}

bool exact_multirow_core_group(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        KimiK3Cache & cache,
        int model_layer,
        int token_begin,
        int group_width,
        const std::vector<float> & hidden,
        const std::vector<std::vector<float>> & checkpoints,
        ExactMultirowCoreGroup & output,
        ggml_tensor * prepared_destination,
        float * attn_capture) {
    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    const size_t group_values = hidden_width * group_width;
    if (model_layer < 0 || model_layer >= w.n_layer || token_begin < 0 ||
        (group_width != kExactCoreGroupWidth &&
         group_width != kExactCoreGroupWidth8) ||
        static_cast<size_t>(token_begin) * hidden_width + group_values >
            hidden.size()) {
        set_last_error(
            "Kimi-K3 exact core group is outside its width4/width8 envelope");
        return false;
    }

    const KimiK3Layer & layer = w.layers[static_cast<size_t>(model_layer)];
    if (!layer.recurrent) {
        set_last_error("Kimi-K3 exact core grouping requires a KDA layer");
        return false;
    }
    KimiK3LayerCache & layer_cache =
        cache.layers[static_cast<size_t>(model_layer)];
    const bool banked = model_layer % w.attn_res_block_size == 0;
    std::vector<std::vector<float>> checkpoint_group;
    checkpoint_group.reserve(checkpoints.size());
    for (const std::vector<float> & checkpoint : checkpoints) {
        const size_t begin = static_cast<size_t>(token_begin) * hidden_width;
        if (checkpoint.size() < begin + group_values) {
            set_last_error("Kimi-K3 exact core checkpoint shape is invalid");
            return false;
        }
        checkpoint_group.emplace_back(
            checkpoint.begin() + static_cast<std::ptrdiff_t>(begin),
            checkpoint.begin() +
                static_cast<std::ptrdiff_t>(begin + group_values));
    }

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 exact core group: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32768, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, group_width);
    ggml_set_input(hidden_in);
    inputs.push_back({
        hidden_in,
        hidden.data() + static_cast<size_t>(token_begin) * hidden_width,
        group_values * sizeof(float)});

    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, group_width, checkpoint_group, residuals, inputs);
    ggml_tensor * prefix = hidden_in;
    ggml_tensor * attn_input = residuals.mix(prefix, layer.attn_res_score);
    ggml_tensor * cur = attn_input;
    if (banked) residuals.push(prefix);
    cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
    KdaTerminalState terminal;
    cur = build_kda(
        ctx, graph, w, layer, layer_cache, cur,
        /*commit_state=*/false,
        /*capture_replay=*/true, token_begin, &terminal);
    prefix = banked ? cur : ggml_add(ctx, prefix, cur);
    cur = residuals.mix(prefix, layer.ffn_res_score);
    cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);

    output.prefix.resize(group_values);
    output.prepared.clear();
    std::vector<GraphOutput> graph_outputs = {
        {prefix, output.prefix.data(), group_values * sizeof(float)},
    };
    if (!prepared_destination) {
        output.prepared.resize(group_values);
        graph_outputs.push_back({
            cur, output.prepared.data(), group_values * sizeof(float)});
    }
    if (attn_capture) {
        graph_outputs.push_back({
            attn_input, attn_capture, group_values * sizeof(float)});
    }
    std::vector<GraphDevicePublish> graph_publishes = {
        {terminal.conv, layer_cache.conv_state},
        {terminal.ssm, layer_cache.ssm_state},
    };
    if (prepared_destination) {
        graph_publishes.push_back({cur, prepared_destination});
    }
    GraphExecutionTiming timing;
    const bool width8 = group_width == kExactCoreGroupWidth8;
    std::string unsupported_weight;
    const int expected_exact_launches =
        exact_qk_core_launches(layer, &unsupported_weight);
    if (width8 && expected_exact_launches <= 0) {
        ggml_free(ctx);
        if (expected_exact_launches < 0) {
            set_last_error(
                "Kimi-K3 exact core width" + std::to_string(group_width) +
                " found unsupported KDA projection at layer " +
                std::to_string(model_layer) + ": " + unsupported_weight);
        } else {
            set_last_error(
                "Kimi-K3 exact core width" + std::to_string(group_width) +
                " found no exact-capable Q4_K/Q6_K KDA projection at "
                "layer " + std::to_string(model_layer));
        }
        return false;
    }
    const size_t exact_qk_launches_before = width8
        ? ggml_backend_cuda_get_exact_qk_width8_launch_count()
        : ggml_backend_cuda_get_exact_qk_width4_launch_count();
    bool ok = false;
    {
        // Plain multirow matmul normally crosses to MMQ. Scope the MMVQ
        // ceiling to this exact K3 graph so the Q4_K/Q6_K kernel can select
        // itself without changing process-wide dispatch policy.
        ScopedCudaGraphOverrides exact_core_scope(
            /*disable_graphs=*/false,
            /*mmvq_max_ncols=*/group_width,
            /*skip_property_check=*/false,
            /*exact_qk_width4=*/!width8,
            /*exact_qk_width8=*/width8);
        ok = run_host_boundary_graph(
            backend, ctx, graph, inputs, graph_outputs,
            width8 ? "P58 exact width-eight KDA core" :
                     "P58 exact width-four KDA core",
            graph_publishes,
            &timing);
    }
    const size_t exact_qk_launches_after = width8
        ? ggml_backend_cuda_get_exact_qk_width8_launch_count()
        : ggml_backend_cuda_get_exact_qk_width4_launch_count();
    const size_t observed_exact_launches =
        exact_qk_launches_after - exact_qk_launches_before;
    if (ok && ((width8 && observed_exact_launches !=
                            static_cast<size_t>(expected_exact_launches)) ||
               (!width8 && observed_exact_launches == 0))) {
        set_last_error(
            "Kimi-K3 exact width" + std::to_string(group_width) +
            " Q4_K/Q6_K dispatch count mismatch at layer " +
            std::to_string(model_layer) + ": expected " +
            (width8 ? std::to_string(expected_exact_launches) :
                      std::string("at least one")) + ", observed " +
            std::to_string(observed_exact_launches));
        ok = false;
    }
    output.graph_ns = timing.compute_ns;
    output.publish_ns = timing.publish_ns;
    ggml_free(ctx);
    return ok;
}

bool exact_multirow_prepared_row(
        ggml_backend_t backend,
        const KimiK3Weights & w,
        int model_layer,
        int token_index,
        const float * prefix_row,
        const float * prepared_row,
        ExactMultirowLayerRow & output,
        PersistentRoutedPreparation * persistent,
        bool defer_router) {
    const KimiK3Layer & layer = w.layers[static_cast<size_t>(model_layer)];
    const bool dense = model_layer < w.n_dense_lead;
    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    if (!dense && persistent) {
        output.prefix.assign(prefix_row, prefix_row + hidden_width);
        output.routed.resize(static_cast<size_t>(w.n_expert_latent));
        output.selected.resize(static_cast<size_t>(w.n_expert_used));
        output.route_weights.resize(static_cast<size_t>(w.n_expert_used));
        output.shared.resize(hidden_width);
        std::string persistent_error;
        if (!persistent->evaluate_prepared(
                model_layer, prepared_row, output.routed, output.selected,
                output.route_weights, output.shared,
                defer_router ? token_index % kDeferredRouterWidth : -1,
                &persistent_error)) {
            set_last_error(
                "Kimi-K3 persistent prepared tail failed at layer " +
                std::to_string(model_layer) + ": " + persistent_error);
            return false;
        }
        return true;
    }
    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error(
            "Kimi-K3 exact row preparation: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_tensor * cur = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, 1);
    ggml_set_input(cur);
    std::vector<GraphInput> inputs = {
        {cur, prepared_row, hidden_width * sizeof(float)},
    };
    std::vector<GraphOutput> outputs;
    if (dense) {
        ggml_tensor * prefix = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, w.n_embd, 1);
        ggml_set_input(prefix);
        inputs.push_back({
            prefix, prefix_row, hidden_width * sizeof(float)});
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
        ggml_tensor * selected = nullptr;
        ggml_tensor * route_weights = nullptr;
        if (!defer_router) {
            TopKMoeRouterResult router =
                build_kimi_router(ctx, graph, w, layer, cur);
            selected = ggml_cont(ctx, router.selected);
            route_weights = ggml_cont(ctx, router.weights_2d);
        }
        ggml_tensor * shared_gate =
            ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
        ggml_tensor * shared_up =
            ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
        ggml_tensor * shared = situ(
            ctx, shared_gate, shared_up,
            w.situ_beta, w.situ_linear_beta);
        shared = ggml_mul_mat(ctx, layer.ffn_down_shexp, shared);

        output.prefix.assign(prefix_row, prefix_row + hidden_width);
        output.routed.resize(static_cast<size_t>(w.n_expert_latent));
        output.selected.resize(static_cast<size_t>(w.n_expert_used));
        output.route_weights.resize(static_cast<size_t>(w.n_expert_used));
        output.shared.resize(hidden_width);
        outputs = {
            {routed, output.routed.data(),
             output.routed.size() * sizeof(float)},
            {shared, output.shared.data(), hidden_width * sizeof(float)},
        };
        if (defer_router) {
            ggml_tensor * staging = persistent
                ? persistent->router8_staging_row(
                    token_index % kDeferredRouterWidth) : nullptr;
            if (!staging) {
                ggml_free(ctx);
                set_last_error(
                    "Kimi-K3 width8 router staging row is unavailable");
                return false;
            }
            ggml_build_forward_expand(
                graph, ggml_cpy(ctx, cur, staging));
        } else {
            outputs.push_back({selected, output.selected.data(),
                               output.selected.size() * sizeof(int32_t)});
            outputs.push_back({route_weights, output.route_weights.data(),
                               output.route_weights.size() * sizeof(float)});
        }
    }
    const bool ok = run_host_boundary_graph(
        backend, ctx, graph, inputs, outputs,
        dense ? "P58 exact row dense tail" :
                "P58 exact row routed preparation");
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
        KimiK3ForwardResult & result,
        float * attn_capture) {
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
    ggml_tensor * output_attn_input =
        residuals.mix(hidden_in, w.output_res_score);
    ggml_tensor * output_hidden = rms_norm(
        ctx, output_attn_input, w.output_norm, w.rms_eps);
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
    if (attn_capture) {
        outputs.push_back({
            output_attn_input, attn_capture, hidden_width * sizeof(float)});
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
    const int active_rows = options.active_rows > 0
        ? options.active_rows : macro_width;
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
    const bool capture_hidden = options.capture_layer_ids &&
        !options.capture_layer_ids->empty();
    if (!kimi_k3_exact_multirow_width(tokens.size()) ||
        active_rows <= 0 || active_rows > macro_width ||
        (active_rows != macro_width && macro_width != 8) ||
        !options.capture_replay ||
        !options.routed_output_provider ||
        !options.routed_output_provider->prefill_service() ||
        !options.routed_output_provider->prefill_service()->supports_width(
            tokens.size()) ||
        (capture_hidden && macro_width != kExactCoreGroupWidth8) ||
        !stream_engine ||
        !stream_engine->is_bound()) {
        set_last_error("Kimi-K3 P58 exact multirow envelope is invalid");
        return false;
    }

    int exact_core_group_width = 1;
    bool exact_qk_width4 = false;
    bool exact_qk_width8 = false;
    bool tokenwise_mmvq = false;
    std::string exact_core_error;
    if (!parse_strict_binary_environment(
            "DFLASH_CUDA_MMVQ_QK_EXACT_WIDTH4",
            exact_qk_width4, &exact_core_error)) {
        set_last_error(exact_core_error);
        return false;
    }
    if (!parse_strict_binary_environment(
            "DFLASH_CUDA_MMVQ_QK_EXACT_WIDTH8",
            exact_qk_width8, &exact_core_error)) {
        set_last_error(exact_core_error);
        return false;
    }
    if (!parse_strict_binary_environment(
            "DFLASH_CUDA_MMVQ_TOKENWISE",
            tokenwise_mmvq, &exact_core_error)) {
        set_last_error(exact_core_error);
        return false;
    }
    int exact_mla_group_width = 1;
    const char * exact_mla_group =
        std::getenv("DFLASH_KIMI_P58_EXACT_MLA_GROUP_WIDTH");
    if (exact_mla_group && *exact_mla_group &&
        std::strcmp(exact_mla_group, "0") != 0) {
        if (std::strcmp(exact_mla_group, "8") != 0) {
            set_last_error(
                "DFLASH_KIMI_P58_EXACT_MLA_GROUP_WIDTH must be 0 or 8");
            return false;
        }
        exact_mla_group_width = kExactCoreGroupWidth8;
        if (macro_width % exact_mla_group_width != 0) {
            set_last_error(
                "Kimi-K3 exact MLA grouping requires a macro width "
                "divisible by eight");
            return false;
        }
    }
    const char * exact_core_group =
        std::getenv("DFLASH_KIMI_P58_EXACT_CORE_GROUP_WIDTH");
    if (exact_core_group && *exact_core_group &&
        std::strcmp(exact_core_group, "0") != 0) {
        if (std::strcmp(exact_core_group, "4") != 0 &&
            std::strcmp(exact_core_group, "8") != 0) {
            set_last_error(
                "DFLASH_KIMI_P58_EXACT_CORE_GROUP_WIDTH must be 0, 4, or 8");
            return false;
        }
        exact_core_group_width =
            std::strcmp(exact_core_group, "8") == 0
            ? kExactCoreGroupWidth8 : kExactCoreGroupWidth;
        if (macro_width % exact_core_group_width != 0) {
            set_last_error(
                "Kimi-K3 exact core grouping requires a macro width "
                "divisible by its group width");
            return false;
        }

        const bool matching_exact_qk =
            exact_core_group_width == kExactCoreGroupWidth
            ? exact_qk_width4 : exact_qk_width8;
        if (!matching_exact_qk || !tokenwise_mmvq) {
            set_last_error(
                "Kimi-K3 exact core grouping requires its matching exact "
                "Q4_K/Q6_K kernel and tokenwise MMVQ fallback");
            return false;
        }
    }
    const bool needs_exact_qk_width4 =
        exact_core_group_width == kExactCoreGroupWidth;
    const bool needs_exact_qk_width8 =
        exact_core_group_width == kExactCoreGroupWidth8 ||
        exact_mla_group_width == kExactCoreGroupWidth8;
    if (exact_qk_width4 != needs_exact_qk_width4 ||
        exact_qk_width8 != needs_exact_qk_width8) {
        set_last_error(
            "Kimi-K3 exact Q4_K/Q6_K and matching core-group flags must "
            "be enabled together");
        return false;
    }

    bool persistent_requested = false;
    bool persistent_core8 = false;
    bool deferred_router = false;
    std::string persistent_error;
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP",
            persistent_requested, &persistent_error)) {
        set_last_error(persistent_error);
        return false;
    }
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P58_PERSISTENT_CORE8",
            persistent_core8, &persistent_error)) {
        set_last_error(persistent_error);
        return false;
    }
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_ROUTER_WIDTH8",
            deferred_router, &persistent_error)) {
        set_last_error(persistent_error);
        return false;
    }
    if (deferred_router &&
        (!persistent_requested ||
         macro_width % kDeferredRouterWidth != 0)) {
        set_last_error(
            "Kimi-K3 deferred router width8 requires P46 and a macro width "
            "divisible by 8");
        return false;
    }
    int exact_tail_group_width = 0;
    const char * exact_tail_group =
        std::getenv("DFLASH_KIMI_P58_EXACT_TAIL_GROUP_WIDTH");
    if (exact_tail_group && *exact_tail_group &&
        std::strcmp(exact_tail_group, "0") != 0) {
        if (std::strcmp(exact_tail_group, "4") != 0 &&
            std::strcmp(exact_tail_group, "8") != 0) {
            set_last_error(
                "DFLASH_KIMI_P58_EXACT_TAIL_GROUP_WIDTH must be 0, 4, or 8");
            return false;
        }
        exact_tail_group_width = std::strcmp(exact_tail_group, "8") == 0
            ? kExactCoreGroupWidth8 : kExactCoreGroupWidth;
        if ((exact_core_group_width == 1 && exact_mla_group_width == 1) ||
            (!exact_qk_width4 && !exact_qk_width8) || !tokenwise_mmvq ||
            !persistent_requested || !deferred_router) {
            set_last_error(
                "Kimi-K3 exact prepared tail requires exact KDA/MLA, "
                "P46, Router8, exact Q4_K/Q6_K, and tokenwise MMVQ");
            return false;
        }
        if ((exact_core_group_width > 1 &&
             exact_core_group_width % exact_tail_group_width != 0) ||
            (exact_mla_group_width > 1 &&
             exact_mla_group_width % exact_tail_group_width != 0)) {
            set_last_error(
                "Kimi-K3 exact prepared-tail width must divide each enabled "
                "KDA/MLA group width");
            return false;
        }
    }
    if (exact_core_group_width == kExactCoreGroupWidth8 &&
        exact_tail_group_width == 0) {
        set_last_error(
            "Kimi-K3 exact KDA8 requires an exact prepared tail");
        return false;
    }
    if (exact_mla_group_width == kExactCoreGroupWidth8 &&
        (exact_tail_group_width == 0 || !persistent_requested ||
         !deferred_router || !exact_qk_width8 || !tokenwise_mmvq)) {
        set_last_error(
            "Kimi-K3 exact MLA8 requires P46, an exact tail, Router8, exact "
            "Q6_K width8, and tokenwise MMVQ");
        return false;
    }
    if (exact_mla_group_width == kExactCoreGroupWidth8) {
        int qualified_mla_layers = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
            if (layer.recurrent) continue;
            std::string unsupported;
            if (il < w.n_dense_lead ||
                !exact_mla_projection_types(layer, &unsupported) ||
                !layer.wq_a_norm || !layer.wk_b || !layer.wv_b ||
                !cache.layers[static_cast<size_t>(il)].mla_k) {
                set_last_error(
                    "Kimi-K3 exact MLA8 layer closure failed at layer " +
                    std::to_string(il) + ": " + unsupported);
                return false;
            }
            ++qualified_mla_layers;
        }
        if (qualified_mla_layers == 0) {
            set_last_error("Kimi-K3 exact MLA8 found no MLA layers");
            return false;
        }
    }
    if (persistent_core8 &&
        (!persistent_requested || macro_width != kExactCoreGroupWidth8 ||
         active_rows != kExactCoreGroupWidth8 ||
         exact_core_group_width != kExactCoreGroupWidth8 ||
         exact_tail_group_width != kExactCoreGroupWidth8 ||
         !deferred_router)) {
        set_last_error(
            "DFLASH_KIMI_P58_PERSISTENT_CORE8 requires one full V8, "
            "P46, exact Core8/Tail8, and Router8");
        return false;
    }
    if (active_rows != macro_width &&
        (exact_core_group_width != 1 || exact_mla_group_width != 1 ||
         exact_tail_group_width != 0 || deferred_router)) {
        set_last_error(
            "Kimi-K3 active-row replay requires scalar exact core/router");
        return false;
    }
    if (active_rows != macro_width) {
        for (const char * name : {
                 "DFLASH_KIMI_P40_WIDE_ASYNC_JOIN",
                 "DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH"}) {
            const char * value = std::getenv(name);
            if (value && *value && std::strcmp(value, "0") != 0) {
                set_last_error(
                    std::string("Kimi-K3 active-row replay requires ") +
                    name + "=0");
                return false;
            }
        }
    }
    PersistentRoutedPreparation * persistent = nullptr;
    if (persistent_requested) {
        persistent = ensure_persistent_routed_preparation(
            backend, w, cache, deferred_router,
            (exact_core_group_width > 1 || exact_mla_group_width > 1)
                ? (exact_tail_group_width > 0 ? exact_tail_group_width : 1)
                : 0,
            /*include_mla_prepared_tail=*/
                exact_mla_group_width == kExactCoreGroupWidth8,
            /*include_exact_core8=*/persistent_core8,
            /*require_exact_router_mode=*/true,
            &persistent_error);
        if (!persistent) {
            set_last_error(
                "Kimi-K3 P46 macro initialization failed: " +
                persistent_error);
            return false;
        }
    }

    const size_t hidden_width = static_cast<size_t>(w.n_embd);
    const size_t hidden_values = hidden_width * active_rows;
    const bool canonical_capture =
        capture_hidden && use_dspark_attn_res_capture();
    std::vector<int> capture_at_layer;
    if (capture_hidden) {
        capture_at_layer.assign(static_cast<size_t>(w.n_layer), -1);
        for (size_t index = 0; index < options.capture_layer_ids->size();
             ++index) {
            capture_at_layer[static_cast<size_t>(
                (*options.capture_layer_ids)[index])] =
                    static_cast<int>(index);
        }
        result.captured_hidden.assign(
            options.capture_layer_ids->size() * hidden_values, 0.0f);
    }
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
    uint64_t grouped_graph_ns = 0;
    uint64_t grouped_publish_ns = 0;
    ExactMultirowSnapshotGuard recurrent_guard(backend, cache);

    std::vector<float> hidden(hidden_values);
    const auto capture_for_consumer = [&] (
            int layer, int token_begin) -> float * {
        if (!canonical_capture || layer <= 0) return nullptr;
        const int capture = capture_at_layer[static_cast<size_t>(layer - 1)];
        if (capture < 0) return nullptr;
        return result.captured_hidden.data() +
            static_cast<size_t>(capture) * hidden_values +
            static_cast<size_t>(token_begin) * hidden_width;
    };
    const auto publish_raw_hidden_capture = [&] (int layer) {
        if (!capture_hidden || canonical_capture) return;
        const int capture = capture_at_layer[static_cast<size_t>(layer)];
        if (capture < 0) return;
        std::memcpy(
            result.captured_hidden.data() +
                static_cast<size_t>(capture) * hidden_values,
            hidden.data(), hidden.size() * sizeof(float));
    };
    Clock::time_point phase_begin = Clock::now();
    for (int token = 0; token < active_rows; ++token) {
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
            if (exact_core_group_width > 1 &&
                layer.recurrent) {
                for (int group = 0; group < macro_width;
                     group += exact_core_group_width) {
                    ExactMultirowCoreGroup core;
                    const bool core_ok = persistent_core8
                        ? persistent->evaluate_exact_core8(
                            il, group, hidden, checkpoints,
                            core.prefix, core.prepared,
                            /*prepared_destination=*/nullptr,
                            capture_for_consumer(il, group),
                            core.graph_ns, core.publish_ns,
                            &persistent_error)
                        : exact_multirow_core_group(
                            backend, w, cache, il, group,
                            exact_core_group_width, hidden,
                            checkpoints, core,
                            /*prepared_destination=*/nullptr,
                            capture_for_consumer(il, group));
                    if (!core_ok) {
                        if (persistent_core8) {
                            set_last_error(
                                "Kimi-K3 persistent exact Core8 failed at "
                                "layer " + std::to_string(il) + ": " +
                                persistent_error);
                        }
                        return false;
                    }
                    grouped_graph_ns += core.graph_ns;
                    grouped_publish_ns += core.publish_ns;
                    for (int row_index = 0;
                         row_index < exact_core_group_width; ++row_index) {
                        ExactMultirowLayerRow row;
                        const int token = group + row_index;
                        if (!exact_multirow_prepared_row(
                                backend, w, il, token,
                                core.prefix.data() +
                                    static_cast<size_t>(row_index) *
                                        hidden_width,
                                core.prepared.data() +
                                    static_cast<size_t>(row_index) *
                                        hidden_width,
                                row, persistent,
                                /*defer_router=*/false) ||
                            row.hidden.size() != hidden_width) {
                            return false;
                        }
                        std::copy(
                            row.hidden.begin(), row.hidden.end(),
                            next_hidden.begin() +
                                static_cast<std::ptrdiff_t>(
                                    token * hidden_width));
                    }
                }
            } else {
                for (int token = 0; token < active_rows; ++token) {
                    ExactMultirowLayerRow row;
                    if (!exact_multirow_layer_row(
                            backend, w, cache, il, base_pos, token,
                            hidden.data() +
                                static_cast<size_t>(token) * hidden_width,
                            checkpoints, row, persistent,
                            /*defer_router=*/false,
                            capture_for_consumer(il, token)) ||
                        row.hidden.size() != hidden_width) {
                        return false;
                    }
                    std::copy(
                        row.hidden.begin(), row.hidden.end(),
                        next_hidden.begin() +
                            static_cast<std::ptrdiff_t>(
                                token * hidden_width));
                }
            }
            core_ns += elapsed_ns(phase_begin);
            if (banked) checkpoints.push_back(checkpoint_value);
            hidden.swap(next_hidden);
            publish_raw_hidden_capture(il);
            continue;
        }

        std::vector<float> prefix(hidden_values);
        std::vector<float> routed(
            static_cast<size_t>(w.n_expert_latent) * active_rows);
        std::vector<int32_t> selected(
            static_cast<size_t>(w.n_expert_used) * active_rows);
        std::vector<float> route_weights(
            static_cast<size_t>(w.n_expert_used) * active_rows);
        std::vector<float> shared(hidden_values);
        const MoeStreamExpertSpec spec = make_kimi_k3_stream_spec(w, layer);
        KimiK3RoutedPrefillService * prefill_service =
            options.routed_output_provider->prefill_service();
        std::string provider_error;
        bool observation_active = false;
        if (!prefill_service->begin_layer_route_observation(
                il, base_pos, spec, static_cast<size_t>(active_rows),
                &observation_active,
                &provider_error)) {
            set_last_error(
                "Kimi-K3 P58 routed observation failed at layer " +
                std::to_string(il) + ": " + provider_error);
            return false;
        }
        LayerRouteObservationGuard observation_guard(
            observation_active ? prefill_service : nullptr);
        const auto finish_deferred_router_group = [&] (int token) {
            if ((token + 1) % kDeferredRouterWidth != 0) return true;
            const int group_begin = token + 1 - kDeferredRouterWidth;
            const size_t route_begin =
                static_cast<size_t>(group_begin) * w.n_expert_used;
            if (!persistent->evaluate_router8(
                    il,
                    selected.data() + route_begin,
                    static_cast<size_t>(w.n_expert_used) *
                        kDeferredRouterWidth,
                    route_weights.data() + route_begin,
                    static_cast<size_t>(w.n_expert_used) *
                        kDeferredRouterWidth,
                    &persistent_error)) {
                set_last_error(
                    "Kimi-K3 persistent width8 router failed at layer " +
                    std::to_string(il) + " group " +
                    std::to_string(group_begin / kDeferredRouterWidth) +
                    ": " + persistent_error);
                return false;
            }
            if (observation_active) {
                for (int row_index = group_begin;
                     row_index <= token; ++row_index) {
                    const size_t offset =
                        static_cast<size_t>(row_index) * w.n_expert_used;
                    if (!prefill_service->observe_completed_route_row(
                            row_index, selected.data() + offset,
                            route_weights.data() + offset,
                            w.n_expert_used, &provider_error)) {
                        set_last_error(
                            "Kimi-K3 P58 routed observation failed at layer " +
                            std::to_string(il) + ": " + provider_error);
                        return false;
                    }
                }
            }
            return true;
        };
        const auto copy_prepared_row = [&] (
                int token, const ExactMultirowLayerRow & row) {
            std::copy(row.prefix.begin(), row.prefix.end(),
                prefix.begin() +
                    static_cast<std::ptrdiff_t>(token * hidden_width));
            std::copy(row.routed.begin(), row.routed.end(),
                routed.begin() + static_cast<std::ptrdiff_t>(
                    token * w.n_expert_latent));
            std::copy(row.shared.begin(), row.shared.end(),
                shared.begin() +
                    static_cast<std::ptrdiff_t>(token * hidden_width));
            if (deferred_router) {
                if (!finish_deferred_router_group(token)) return false;
            } else {
                std::copy(row.selected.begin(), row.selected.end(),
                    selected.begin() + static_cast<std::ptrdiff_t>(
                        token * w.n_expert_used));
                std::copy(row.route_weights.begin(), row.route_weights.end(),
                    route_weights.begin() + static_cast<std::ptrdiff_t>(
                        token * w.n_expert_used));
                if (observation_active &&
                    !prefill_service->observe_completed_route_row(
                        token, row.selected.data(), row.route_weights.data(),
                        w.n_expert_used, &provider_error)) {
                    set_last_error(
                        "Kimi-K3 P58 routed observation failed at layer " +
                        std::to_string(il) + ": " + provider_error);
                    return false;
                }
            }
            return true;
        };
        const int layer_group_width = layer.recurrent
            ? exact_core_group_width : exact_mla_group_width;
        if (layer_group_width > 1) {
            for (int group = 0; group < macro_width;
                 group += layer_group_width) {
                ExactMultirowCoreGroup core;
                const bool core_width8 =
                    layer_group_width == kExactCoreGroupWidth8;
                // Tail8 publishes directly into its persistent input. Tail4
                // over a width8 core first retains both halves in Router8
                // staging. In both cases the in-order stream restores the
                // router input before evaluate_router8 can consume it.
                ggml_tensor * prepared_destination = exact_tail_group_width > 0
                    ? persistent->exact_group_prepared_destination(
                        il, layer_group_width)
                    : nullptr;
                if (exact_tail_group_width > 0 && !prepared_destination) {
                    set_last_error(
                        "Kimi-K3 persistent prepared-tail input is "
                        "unavailable at layer " + std::to_string(il));
                    return false;
                }
                const bool grouped_ok = layer.recurrent && persistent_core8
                    ? persistent->evaluate_exact_core8(
                        il, group, hidden, checkpoints,
                        core.prefix, core.prepared, prepared_destination,
                        capture_for_consumer(il, group),
                        core.graph_ns, core.publish_ns,
                        &persistent_error)
                    : layer.recurrent
                    ? exact_multirow_core_group(
                        backend, w, cache, il, group, layer_group_width,
                        hidden, checkpoints, core, prepared_destination,
                        capture_for_consumer(il, group))
                    : exact_multirow_mla_group(
                        backend, w, cache, il, base_pos, group, hidden,
                        checkpoints, core, prepared_destination,
                        capture_for_consumer(il, group));
                if (!grouped_ok) {
                    if (layer.recurrent && persistent_core8) {
                        set_last_error(
                            "Kimi-K3 persistent exact Core8 failed at layer " +
                            std::to_string(il) + ": " + persistent_error);
                    }
                    return false;
                }
                grouped_graph_ns += core.graph_ns;
                grouped_publish_ns += core.publish_ns;
                if (exact_tail_group_width > 0) {
                    std::copy(
                        core.prefix.begin(), core.prefix.end(),
                        prefix.begin() + static_cast<std::ptrdiff_t>(
                            group * hidden_width));
                    const size_t routed_offset =
                        static_cast<size_t>(group) * w.n_expert_latent;
                    const size_t shared_offset =
                        static_cast<size_t>(group) * hidden_width;
                    const int tail_calls = layer_group_width /
                        exact_tail_group_width;
                    for (int tail = 0; tail < tail_calls; ++tail) {
                        const int tail_token =
                            group + tail * exact_tail_group_width;
                        ggml_tensor * prepared_source =
                            core_width8 && exact_tail_group_width ==
                                kExactCoreGroupWidth
                            ? persistent->prepared_tail_width4_source_half(tail)
                            : nullptr;
                        if (core_width8 && exact_tail_group_width ==
                                kExactCoreGroupWidth && !prepared_source) {
                            set_last_error(
                                "Kimi-K3 width8 prepared-tail source is "
                                "unavailable at layer " + std::to_string(il));
                            return false;
                        }
                        if (!persistent->evaluate_prepared_exact(
                                il, tail_token,
                                routed.data() + routed_offset +
                                    static_cast<size_t>(tail) *
                                        w.n_expert_latent *
                                        exact_tail_group_width,
                                static_cast<size_t>(w.n_expert_latent) *
                                    exact_tail_group_width,
                                shared.data() + shared_offset +
                                    static_cast<size_t>(tail) * hidden_width *
                                        exact_tail_group_width,
                                hidden_width * exact_tail_group_width,
                                prepared_source,
                                &persistent_error)) {
                            set_last_error(
                                "Kimi-K3 persistent exact prepared tail width " +
                                std::to_string(exact_tail_group_width) +
                                " failed at layer " + std::to_string(il) +
                                " group " + std::to_string(
                                    tail_token / exact_tail_group_width) +
                                ": " + persistent_error);
                            return false;
                        }
                    }
                    if (!finish_deferred_router_group(
                            group + layer_group_width - 1)) {
                        return false;
                    }
                    continue;
                }
                for (int row_index = 0;
                     row_index < layer_group_width; ++row_index) {
                    ExactMultirowLayerRow row;
                    const int token = group + row_index;
                    if (!exact_multirow_prepared_row(
                            backend, w, il, token,
                            core.prefix.data() +
                                static_cast<size_t>(row_index) * hidden_width,
                            core.prepared.data() +
                                static_cast<size_t>(row_index) * hidden_width,
                            row, persistent, deferred_router) ||
                        !copy_prepared_row(token, row)) {
                        return false;
                    }
                }
            }
        } else {
            for (int token = 0; token < active_rows; ++token) {
                ExactMultirowLayerRow row;
                if (!exact_multirow_layer_row(
                        backend, w, cache, il, base_pos, token,
                        hidden.data() +
                            static_cast<size_t>(token) * hidden_width,
                        checkpoints, row, persistent, deferred_router,
                        capture_for_consumer(il, token)) ||
                    !copy_prepared_row(token, row)) {
                    return false;
                }
            }
        }
        core_ns += elapsed_ns(phase_begin);
        if (banked) checkpoints.push_back(checkpoint_value);

        const size_t active_route_count =
            static_cast<size_t>(active_rows) * w.n_expert_used;
        for (size_t route = 0; route < active_route_count; ++route) {
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
        MoeStreamRouteBatch routes;
        routes.layer = il - w.n_dense_lead;
        routes.n_expert = w.n_expert;
        routes.top_k = w.n_expert_used;
        routes.n_tokens = active_rows;
        routes.inputs = routed.data();
        routes.selected_ids = selected.data();
        routes.selected_weights = route_weights.data();
        std::vector<float> routed_output;
        phase_begin = Clock::now();
        if (!prefill_service->evaluate_layer(
                il, base_pos, spec, routes, *stream_engine,
                routed_output, &provider_error)) {
            set_last_error(
                "Kimi-K3 P58 routed layer " + std::to_string(il) +
                " failed: " + provider_error);
            return false;
        }
        observation_guard.complete();
        expert_ns += elapsed_ns(phase_begin);
        if (routed_output.size() !=
            static_cast<size_t>(w.n_expert_latent) * active_rows) {
            set_last_error("Kimi-K3 P58 provider returned an invalid shape");
            return false;
        }

        phase_begin = Clock::now();
        std::vector<float> next_hidden(hidden_values);
        for (int token = 0; token < active_rows; ++token) {
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
        publish_raw_hidden_capture(il);
    }

    if (options.read_logits) {
        result.logits.resize(
            static_cast<size_t>(w.n_vocab) * active_rows);
    }
    if (options.read_argmax) result.argmax.resize(active_rows);
    phase_begin = Clock::now();
    for (int token = 0; token < active_rows; ++token) {
        float * final_capture = nullptr;
        if (canonical_capture && !capture_at_layer.empty()) {
            const int capture =
                capture_at_layer[static_cast<size_t>(w.n_layer - 1)];
            if (capture >= 0) {
                final_capture = result.captured_hidden.data() +
                    static_cast<size_t>(capture) * hidden_values +
                    static_cast<size_t>(token) * hidden_width;
            }
        }
        if (!exact_multirow_output_row(
                backend, w,
                hidden.data() + static_cast<size_t>(token) * hidden_width,
                token, checkpoints, options, result, final_capture)) {
            return false;
        }
    }
    output_ns += elapsed_ns(phase_begin);
    cache.cur_pos = base_pos + active_rows;
    recurrent_guard.retain_terminal();

    const char * profile = std::getenv("DFLASH_KIMI_STAGE_PROFILE");
    if (profile && *profile && std::strcmp(profile, "0") != 0) {
        const uint64_t total_ns = elapsed_ns(total_begin);
        const uint64_t classified = core_ns + expert_ns + join_ns + output_ns;
        std::fprintf(stderr,
            "[kimi-k3-p58-stage] position=%d tokens=%d active_rows=%d "
            "total_ms=%.3f "
            "one_row_core_ms=%.3f experts_ms=%.3f one_row_join_ms=%.3f "
            "one_row_output_ms=%.3f grouped_graph_ms=%.3f "
            "grouped_publish_ms=%.3f exact_core_group_width=%d "
            "exact_mla_group_width=%d "
            "exact_tail_group_width=%d exact_qk_width4=%d "
            "exact_qk_width8=%d "
            "scoped_mmvq_max=%d other_ms=%.3f\n",
            base_pos, macro_width, active_rows, total_ns / 1.0e6,
            core_ns / 1.0e6,
            expert_ns / 1.0e6, join_ns / 1.0e6, output_ns / 1.0e6,
            grouped_graph_ns / 1.0e6, grouped_publish_ns / 1.0e6,
            exact_core_group_width,
            exact_mla_group_width,
            exact_tail_group_width,
            exact_qk_width4 ? 1 : 0,
            exact_qk_width8 ? 1 : 0,
            std::max(exact_core_group_width, exact_mla_group_width) > 1
                ? std::max(exact_core_group_width, exact_mla_group_width) : 0,
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
        MoeHybridStreamEngine & stream_engine) {
    const int n_tokens = static_cast<int>(tokens.size());
    const size_t hidden_values =
        static_cast<size_t>(w.n_embd) * tokens.size();
    std::vector<float> hidden(hidden_values);

    bool persistent_requested = false;
    std::string persistent_error;
    if (!parse_strict_binary_environment(
            "DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP",
            persistent_requested, &persistent_error)) {
        set_last_error(persistent_error);
        return false;
    }
    PersistentRoutedPreparation * persistent = nullptr;
    if (persistent_requested) {
        bool async_queue = false;
        if (!parse_strict_binary_environment(
                "DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", async_queue,
                &persistent_error)) {
            set_last_error(persistent_error);
            return false;
        }
        if (!async_queue || n_tokens != 1 || options.capture_replay) {
            set_last_error(
                "Kimi-K3 P46 requires P45 and ordinary one-token execution");
            return false;
        }
        persistent = ensure_persistent_routed_preparation(
            backend, w, cache, /*deferred_router=*/false,
            /*prepared_tail_width=*/0,
            /*include_mla_prepared_tail=*/false,
            /*include_exact_core8=*/false,
            /*require_exact_router_mode=*/false,
            &persistent_error);
        if (!persistent) {
            set_last_error(
                "Kimi-K3 P46 initialization failed: " + persistent_error);
            return false;
        }
    }

    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    for (int index = 0; index < n_capture; ++index) {
        capture_at_layer[static_cast<size_t>(
            (*options.capture_layer_ids)[static_cast<size_t>(index)])] = index;
    }
    result.captured_hidden.assign(
        static_cast<size_t>(n_capture) * hidden_values, 0.0f);
    const bool canonical_capture =
        n_capture > 0 && use_dspark_attn_res_capture();
    const auto capture_for_consumer = [&] (int layer) -> float * {
        if (!canonical_capture || layer <= 0) return nullptr;
        const int capture = capture_at_layer[static_cast<size_t>(layer - 1)];
        return capture < 0 ? nullptr :
            result.captured_hidden.data() +
                static_cast<size_t>(capture) * hidden_values;
    };

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int query = 0; query < n_tokens; ++query) {
        for (int key = 0; key <= base_pos + query; ++key) {
            mla_mask[static_cast<size_t>(query) * kv_len + key] = 0.0f;
        }
    }

    {
        ggml_context * ctx = new_kimi_step_context();
        if (!ctx) {
            set_last_error("Kimi-K3 embedding: context allocation failed");
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * ids =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_set_input(ids);
        ggml_tensor * embedding = ggml_get_rows(ctx, w.tok_embd, ids);
        const bool ok = ggml_backend_supports_op(backend, embedding)
            ? run_host_boundary_graph(
                backend, ctx, graph,
                {{ids, tokens.data(), sizeof(int32_t) * tokens.size()}},
                {{embedding, hidden.data(), hidden.size() * sizeof(float)}},
                "embedding")
            : read_token_embeddings_on_host(w, tokens, hidden);
        ggml_free(ctx);
        if (!ok) return false;
    }

    std::vector<std::vector<float>> checkpoints;
    checkpoints.reserve(static_cast<size_t>(
        (w.n_layer + w.attn_res_block_size - 1) /
        w.attn_res_block_size));
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache =
            cache.layers[static_cast<size_t>(il)];
        const bool banked = il % w.attn_res_block_size == 0;
        const std::vector<float> checkpoint = hidden;
        const bool persistent_layer =
            persistent && il >= w.n_dense_lead && layer.recurrent;

        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        std::vector<GraphInput> inputs;
        ggml_tensor * prefix = nullptr;
        ggml_tensor * cur = nullptr;
        ggml_tensor * attn_input = nullptr;
        float * attn_capture = capture_for_consumer(il);
        if (!persistent_layer) {
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
                hidden_in, hidden.data(), hidden.size() * sizeof(float)});
            prefix = hidden_in;
            AttnResBank residuals;
            populate_attn_res_bank(
                ctx, w, n_tokens, checkpoints, residuals, inputs);
            attn_input = residuals.mix(prefix, layer.attn_res_score);
            cur = attn_input;
            if (banked) residuals.push(prefix);
            cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
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
                    ctx, graph, w, layer, layer_cache, cur, base_pos, mask);
            }
            prefix = banked ? cur : ggml_add(ctx, prefix, cur);
            cur = residuals.mix(prefix, layer.ffn_res_score);
            cur = rms_norm(ctx, cur, layer.ffn_norm, w.rms_eps);
        }

        if (il < w.n_dense_lead) {
            ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
            ggml_tensor * dense = situ(
                ctx, gate, up, w.situ_beta, w.situ_linear_beta);
            dense = ggml_mul_mat(ctx, layer.ffn_down, dense);
            ggml_tensor * hidden_out = ggml_add(ctx, prefix, dense);
            std::vector<float> next_hidden(hidden_values);
            std::vector<GraphOutput> outputs = {{
                hidden_out, next_hidden.data(),
                next_hidden.size() * sizeof(float)}};
            if (attn_capture) {
                outputs.push_back({
                    attn_input, attn_capture, hidden_values * sizeof(float)});
            }
            const bool ok = run_host_boundary_graph(
                backend, ctx, graph, inputs, outputs,
                "dense layer");
            ggml_free(ctx);
            if (!ok) return false;
            if (banked) checkpoints.push_back(checkpoint);
            hidden.swap(next_hidden);
        } else {
            std::vector<float> prefix_host(hidden_values);
            std::vector<float> routed_input(
                static_cast<size_t>(w.n_expert_latent) * n_tokens);
            std::vector<int32_t> selected(
                static_cast<size_t>(w.n_expert_used) * n_tokens);
            std::vector<float> route_weights(
                static_cast<size_t>(w.n_expert_used) * n_tokens);
            std::vector<float> shared(hidden_values);
            bool prepared = false;
            if (persistent_layer) {
                prepared = persistent->evaluate(
                    il, hidden, checkpoints, prefix_host, routed_input,
                    selected, route_weights, shared,
                    /*router_stage_row=*/-1,
                    /*replay_token_offset=*/-1, attn_capture,
                    &persistent_error);
                if (!prepared) {
                    set_last_error(
                        "Kimi-K3 P46 layer " + std::to_string(il) +
                        " failed: " + persistent_error);
                }
            } else {
                ggml_tensor * routed =
                    ggml_mul_mat(ctx, layer.ffn_routed_down, cur);
                TopKMoeRouterResult router =
                    build_kimi_router(ctx, graph, w, layer, cur);
                // Materialize argsort views before host readback.
                ggml_tensor * selected_out = ggml_cont(ctx, router.selected);
                ggml_tensor * weights_out =
                    ggml_cont(ctx, router.weights_2d);
                ggml_tensor * shared_gate =
                    ggml_mul_mat(ctx, layer.ffn_gate_shexp, cur);
                ggml_tensor * shared_up =
                    ggml_mul_mat(ctx, layer.ffn_up_shexp, cur);
                ggml_tensor * shared_out = situ(
                    ctx, shared_gate, shared_up,
                    w.situ_beta, w.situ_linear_beta);
                shared_out =
                    ggml_mul_mat(ctx, layer.ffn_down_shexp, shared_out);
                std::vector<GraphOutput> outputs = {
                        {prefix, prefix_host.data(),
                         prefix_host.size() * sizeof(float)},
                        {routed, routed_input.data(),
                         routed_input.size() * sizeof(float)},
                        {selected_out, selected.data(),
                         selected.size() * sizeof(int32_t)},
                        {weights_out, route_weights.data(),
                         route_weights.size() * sizeof(float)},
                        {shared_out, shared.data(),
                         shared.size() * sizeof(float)},
                    };
                if (attn_capture) {
                    outputs.push_back({
                        attn_input, attn_capture,
                        hidden_values * sizeof(float)});
                }
                prepared = run_host_boundary_graph(
                    backend, ctx, graph, inputs, outputs,
                    "routed layer preparation");
                ggml_free(ctx);
            }
            if (!prepared) return false;
            if (banked) checkpoints.push_back(checkpoint);
            for (size_t route = 0; route < selected.size(); ++route) {
                if (selected[route] < 0 || selected[route] >= w.n_expert ||
                    !std::isfinite(route_weights[route])) {
                    set_last_error(
                        "Kimi-K3 native router returned invalid output");
                    return false;
                }
            }

            const MoeStreamExpertSpec spec =
                make_kimi_k3_stream_spec(w, layer);
            MoeStreamRouteBatch routes;
            routes.layer = il - w.n_dense_lead;
            routes.n_expert = w.n_expert;
            routes.top_k = w.n_expert_used;
            routes.n_tokens = n_tokens;
            routes.inputs = routed_input.data();
            routes.selected_ids = selected.data();
            routes.selected_weights = route_weights.data();
            const bool alternate = options.routed_output_provider &&
                options.routed_output_provider->handles_layer(il);
            const bool device_output = alternate &&
                options.routed_output_provider->requires_device_output();
            PendingDeviceOutputGuard output_guard(
                device_output ? options.routed_output_provider : nullptr);
            if (device_output && n_tokens != 1) {
                set_last_error(
                    "Kimi-K3 device output requires one-token execution");
                return false;
            }
            std::vector<float> routed_output;
            std::string stream_error;
            const bool evaluated = device_output
                ? options.routed_output_provider->evaluate_device(
                    il, base_pos, spec, routes, stream_engine, backend,
                    &stream_error)
                : alternate
                    ? options.routed_output_provider->evaluate(
                        il, base_pos, spec, routes, stream_engine,
                        routed_output, &stream_error)
                    : eval_moe_streamed_experts(
                        stream_engine, spec, routes, routed_output,
                        &stream_error);
            if (!evaluated || (!device_output && routed_output.size() !=
                    static_cast<size_t>(w.n_expert_latent) * n_tokens)) {
                set_last_error(
                    "Kimi-K3 streamed expert layer " +
                    std::to_string(il) + " failed: " + stream_error);
                return false;
            }

            ctx = new_kimi_step_context();
            if (!ctx) {
                set_last_error("Kimi-K3 join: context allocation failed");
                return false;
            }
            graph = ggml_new_graph_custom(ctx, 4096, false);
            ggml_tensor * prefix_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
            ggml_tensor * routed_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_expert_latent, n_tokens);
            ggml_tensor * shared_in = ggml_new_tensor_2d(
                ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
            ggml_set_input(prefix_in);
            ggml_set_input(routed_in);
            ggml_set_input(shared_in);
            ggml_tensor * routed_join = routed_in;
            if (layer.ffn_routed_norm) {
                routed_join = rms_norm(
                    ctx, routed_join, layer.ffn_routed_norm, w.rms_eps);
            }
            routed_join =
                ggml_mul_mat(ctx, layer.ffn_routed_up, routed_join);
            ggml_tensor * hidden_out = ggml_add(
                ctx, prefix_in, ggml_add(ctx, routed_join, shared_in));
            std::vector<float> next_hidden(hidden_values);
            const bool joined = run_host_boundary_graph(
                backend, ctx, graph,
                {
                    {prefix_in, prefix_host.data(),
                     prefix_host.size() * sizeof(float)},
                    {routed_in,
                     device_output ? nullptr : routed_output.data(),
                     static_cast<size_t>(w.n_expert_latent) * n_tokens *
                         sizeof(float),
                     device_output ? options.routed_output_provider : nullptr},
                    {shared_in, shared.data(),
                     shared.size() * sizeof(float)},
                },
                {{hidden_out, next_hidden.data(),
                  next_hidden.size() * sizeof(float)}},
                "routed layer join");
            ggml_free(ctx);
            if (!joined) return false;
            hidden.swap(next_hidden);
        }

        const int capture = capture_at_layer[static_cast<size_t>(il)];
        if (!canonical_capture && capture >= 0) {
            std::memcpy(
                result.captured_hidden.data() +
                    static_cast<size_t>(capture) * hidden_values,
                hidden.data(), hidden.size() * sizeof(float));
        }
    }

    ggml_context * ctx = new_kimi_step_context();
    if (!ctx) {
        set_last_error("Kimi-K3 output: context allocation failed");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
    std::vector<GraphInput> inputs;
    ggml_tensor * hidden_in = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, w.n_embd, n_tokens);
    ggml_set_input(hidden_in);
    inputs.push_back({
        hidden_in, hidden.data(), hidden.size() * sizeof(float)});
    AttnResBank residuals;
    populate_attn_res_bank(
        ctx, w, n_tokens, checkpoints, residuals, inputs);
    ggml_tensor * output_attn_input =
        residuals.mix(hidden_in, w.output_res_score);
    ggml_tensor * output_hidden =
        rms_norm(ctx, output_attn_input, w.output_norm, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, output_hidden);
    ggml_tensor * argmax = ggml_argmax(ctx, logits);
    std::vector<GraphOutput> outputs;
    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        outputs.push_back({
            logits, result.logits.data(), result.logits.size() * sizeof(float)});
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        outputs.push_back({
            argmax, result.argmax.data(), result.argmax.size() * sizeof(int32_t)});
    }
    if (canonical_capture && n_capture > 0) {
        const int capture =
            capture_at_layer[static_cast<size_t>(w.n_layer - 1)];
        if (capture >= 0) {
            outputs.push_back({
                output_attn_input,
                result.captured_hidden.data() +
                    static_cast<size_t>(capture) * hidden_values,
                hidden_values * sizeof(float)});
        }
    }
    const bool output_ok = run_host_boundary_graph(
        backend, ctx, graph, inputs, outputs, "output");
    ggml_free(ctx);
    if (!output_ok) return false;
    cache.cur_pos = base_pos + n_tokens;
    return true;
}

} // namespace

bool kimi_k3_prepare_persistent_core8(
        ggml_backend_t backend,
        const KimiK3Weights & weights,
        KimiK3Cache & cache,
        std::string & error) {
    return ensure_persistent_routed_preparation(
        backend, weights, cache,
        /*deferred_router=*/true,
        /*prepared_tail_width=*/kExactCoreGroupWidth8,
        /*include_mla_prepared_tail=*/true,
        /*include_exact_core8=*/true,
        /*require_exact_router_mode=*/true,
        &error) != nullptr;
}

bool kimi_k3_read_token_embeddings_on_host(
        const KimiK3Weights & weights,
        const std::vector<int32_t> & tokens,
        std::vector<float> & hidden) {
    return read_token_embeddings_on_host(weights, tokens, hidden);
}

void kimi_k3_destroy_graph_state(void *& state) {
    free_persistent_routed_preparation(state);
}

bool kimi_k3_forward(ggml_backend_t backend,
                     const KimiK3Weights & w,
                     KimiK3Cache & cache,
                     const std::vector<int32_t> & tokens,
                     int base_pos,
                     const KimiK3ForwardOptions & options,
                     KimiK3ForwardResult & result,
                     MoeHybridStreamEngine * stream_engine) {
    result = KimiK3ForwardResult{};
    if (exact_terminal_pending(cache)) {
        set_last_error("Kimi-K3 forward: exact terminal state is unresolved");
        return false;
    }
    const int n_tokens = static_cast<int>(tokens.size());
    if (!backend || !w.ctx || !cache.ctx || n_tokens <= 0 || base_pos < 0 ||
        base_pos != cache.cur_pos || base_pos + n_tokens > cache.max_ctx ||
        (!options.read_logits && !options.read_argmax) ||
        options.active_rows < 0 || options.active_rows > n_tokens ||
        (options.active_rows != 0 && !options.exact_multirow_core)) {
        set_last_error("Kimi-K3 forward: invalid backend, output, or cache span");
        return false;
    }
    for (int32_t token : tokens) {
        if (token < 0 || token >= w.n_vocab) {
            set_last_error("Kimi-K3 forward: token is outside the vocabulary");
            return false;
        }
    }

    std::vector<bool> captured(static_cast<size_t>(w.n_layer), false);
    if (options.capture_layer_ids) {
        for (int layer : *options.capture_layer_ids) {
            if (layer < 0 || layer >= w.n_layer ||
                captured[static_cast<size_t>(layer)]) {
                set_last_error(
                    "Kimi-K3 forward: invalid or duplicate capture layer");
                return false;
            }
            captured[static_cast<size_t>(layer)] = true;
        }
    }
    if (options.capture_replay &&
        (n_tokens > cache.max_verify_tokens || !cache.snapshot_valid ||
         cache.snapshot_pos != base_pos)) {
        set_last_error(
            "Kimi-K3 forward: replay capture has no matching snapshot");
        return false;
    }
    if (options.exact_multirow_core &&
        (!w.routed_experts_streamed || !options.capture_replay ||
         !kimi_k3_exact_multirow_width(tokens.size()) ||
         (options.capture_layer_ids && !options.capture_layer_ids->empty() &&
          n_tokens != kExactCoreGroupWidth8) ||
         !options.routed_output_provider ||
         !options.routed_output_provider->prefill_service() ||
         !options.routed_output_provider->prefill_service()->supports_width(
             tokens.size()))) {
        set_last_error(
            "Kimi-K3 forward: P58 exact multirow request is outside its "
            "qualified envelope");
        return false;
    }

    if (w.routed_experts_streamed) {
        if (!stream_engine || !stream_engine->is_bound()) {
            set_last_error(
                "Kimi-K3 forward: streamed experts require a bound engine");
            return false;
        }
        const bool ok = options.exact_multirow_core
            ? streamed_kimi_k3_forward_exact_multirow(
                backend, w, cache, tokens, base_pos, options, result,
                stream_engine)
            : streamed_kimi_k3_forward(
                backend, w, cache, tokens, base_pos, options, result,
                *stream_engine);
        if (!ok) {
            if (options.capture_replay) {
                cache.replay_valid = false;
                cache.replay_exact_rows = false;
            }
            return false;
        }
        if (options.capture_replay) {
            cache.replay_base_pos = base_pos;
            cache.replay_n_tokens = options.active_rows > 0
                ? options.active_rows : n_tokens;
            cache.replay_valid = true;
            cache.recurrent_state_pristine = !options.exact_multirow_core;
            cache.replay_exact_rows = options.exact_multirow_core;
        } else {
            cache.replay_valid = false;
            cache.recurrent_state_pristine = false;
            cache.replay_exact_rows = false;
        }
        return true;
    }
    if (options.exact_multirow_core || options.routed_output_provider) {
        set_last_error(
            "Kimi-K3 forward: routed provider requires streamed experts");
        return false;
    }

    const int kv_len = base_pos + n_tokens;
    std::vector<float> mla_mask(
        static_cast<size_t>(kv_len) * n_tokens, -INFINITY);
    for (int query = 0; query < n_tokens; ++query) {
        for (int key = 0; key <= base_pos + query; ++key) {
            mla_mask[static_cast<size_t>(query) * kv_len + key] = 0.0f;
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
    ggml_set_input(ids);
    ggml_tensor * hidden = ggml_get_rows(ctx, w.tok_embd, ids);
    ggml_tensor * mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, kv_len, n_tokens);
    ggml_set_input(mask);

    const int n_capture = options.capture_layer_ids
        ? static_cast<int>(options.capture_layer_ids->size()) : 0;
    std::vector<int> capture_at_layer(static_cast<size_t>(w.n_layer), -1);
    for (int index = 0; index < n_capture; ++index) {
        capture_at_layer[static_cast<size_t>(
            (*options.capture_layer_ids)[static_cast<size_t>(index)])] = index;
    }
    std::vector<ggml_tensor *> capture_tensors(
        static_cast<size_t>(n_capture), nullptr);
    const bool canonical_capture =
        n_capture > 0 && use_dspark_attn_res_capture();
    AttnResBank residuals;
    residuals.ctx = ctx;
    residuals.eps = w.rms_eps;
    residuals.n_embd = w.n_embd;
    residuals.n_tokens = n_tokens;
    for (int il = 0; il < w.n_layer; ++il) {
        const KimiK3Layer & layer = w.layers[static_cast<size_t>(il)];
        KimiK3LayerCache & layer_cache =
            cache.layers[static_cast<size_t>(il)];
        ggml_tensor * prefix = hidden;
        ggml_tensor * attn_input =
            residuals.mix(prefix, layer.attn_res_score);
        if (canonical_capture && il > 0) {
            const int capture =
                capture_at_layer[static_cast<size_t>(il - 1)];
            if (capture >= 0) {
                capture_tensors[static_cast<size_t>(capture)] = attn_input;
                ggml_set_output(attn_input);
                ggml_build_forward_expand(graph, attn_input);
            }
        }
        ggml_tensor * cur = attn_input;
        const bool banked = il % w.attn_res_block_size == 0;
        if (banked) residuals.push(prefix);
        cur = rms_norm(ctx, cur, layer.attn_norm, w.rms_eps);
        cur = layer.recurrent
            ? build_kda(
                ctx, graph, w, layer, layer_cache, cur,
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
        const int capture = capture_at_layer[static_cast<size_t>(il)];
        if (!canonical_capture && capture >= 0) {
            capture_tensors[static_cast<size_t>(capture)] = hidden;
            ggml_set_output(hidden);
            ggml_build_forward_expand(graph, hidden);
        }
    }

    ggml_tensor * output_attn_input =
        residuals.mix(hidden, w.output_res_score);
    if (canonical_capture && n_capture > 0) {
        const int capture =
            capture_at_layer[static_cast<size_t>(w.n_layer - 1)];
        if (capture >= 0) {
            capture_tensors[static_cast<size_t>(capture)] = output_attn_input;
            ggml_set_output(output_attn_input);
            ggml_build_forward_expand(graph, output_attn_input);
        }
    }
    hidden = rms_norm(ctx, output_attn_input, w.output_norm, w.rms_eps);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, hidden);
    ggml_tensor * argmax = ggml_argmax(ctx, logits);
    if (options.read_logits) {
        ggml_set_output(logits);
        ggml_build_forward_expand(graph, logits);
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
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        set_last_error("Kimi-K3 forward: graph compute failed");
        ggml_gallocr_free(allocator);
        ggml_free(ctx);
        return false;
    }
    if (options.read_logits) {
        result.logits.resize(static_cast<size_t>(w.n_vocab) * n_tokens);
        ggml_backend_tensor_get(
            logits, result.logits.data(), 0,
            result.logits.size() * sizeof(float));
    }
    if (options.read_argmax) {
        result.argmax.resize(static_cast<size_t>(n_tokens));
        ggml_backend_tensor_get(
            argmax, result.argmax.data(), 0,
            result.argmax.size() * sizeof(int32_t));
    }
    const size_t capture_values =
        static_cast<size_t>(w.n_embd) * n_tokens;
    result.captured_hidden.resize(
        static_cast<size_t>(n_capture) * capture_values);
    for (int index = 0; index < n_capture; ++index) {
        ggml_backend_tensor_get(
            capture_tensors[static_cast<size_t>(index)],
            result.captured_hidden.data() +
                static_cast<size_t>(index) * capture_values,
            0, capture_values * sizeof(float));
    }
    cache.cur_pos = base_pos + n_tokens;
    if (options.capture_replay) {
        cache.replay_base_pos = base_pos;
        cache.replay_n_tokens = n_tokens;
        cache.replay_valid = true;
        cache.recurrent_state_pristine = true;
        cache.replay_exact_rows = false;
    } else {
        cache.replay_valid = false;
        cache.recurrent_state_pristine = false;
        cache.replay_exact_rows = false;
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    return true;
}

bool kimi_k3_replay_snapshot(ggml_backend_t backend, KimiK3Cache & cache) {
    if (!backend || cache.max_verify_tokens <= 0 ||
        exact_terminal_pending(cache)) return false;
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
    if (!cache.recurrent_state_pristine &&
        !restore_recurrent_snapshot(backend, cache)) return false;
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
    const bool retained_terminal = exact_terminal_pending(cache);
    if (!backend || !cache.snapshot_valid || !cache.replay_valid ||
        (!cache.recurrent_state_pristine && !retained_terminal) ||
        cache.snapshot_pos != base_pos ||
        cache.replay_base_pos != base_pos || commit_n <= 0 ||
        commit_n > cache.replay_n_tokens ||
        (retained_terminal &&
         cache.cur_pos != base_pos + cache.replay_n_tokens)) {
        return false;
    }
    const bool exact_rows = cache.replay_exact_rows;
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
    if (!retained_terminal || commit_n != cache.replay_n_tokens) {
        if (retained_terminal &&
            !restore_recurrent_snapshot(backend, cache)) return false;
        cache.recurrent_state_pristine = false;
        if (exact_rows) {
            for (int token = 0; token < commit_n && commit_ok; ++token) {
                commit_ok = commit_span(token, 1);
            }
        } else {
            commit_ok = commit_span(0, commit_n);
        }
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
                  const KimiK3Weights & weights,
                  KimiK3Cache & cache,
                  int32_t token,
                  int position,
                  std::vector<float> & logits,
                  MoeHybridStreamEngine * stream_engine,
                  KimiK3RoutedOutputProvider * routed_output_provider) {
    KimiK3ForwardOptions options;
    options.read_logits = true;
    options.read_argmax = false;
    options.routed_output_provider = routed_output_provider;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend, weights, cache, std::vector<int32_t>{token}, position,
            options, result, stream_engine)) {
        return false;
    }
    logits = std::move(result.logits);
    return true;
}


} // namespace dflash::common
