#include "kimi_k3_internal.h"

#include <algorithm>
#include <cstdio>

namespace dflash::common {

bool create_kimi_k3_cache(ggml_backend_t backend,
                          const KimiK3Weights & weights,
                          int max_ctx,
                          KimiK3Cache & out,
                          int max_verify_tokens) {
    free_kimi_k3_cache(out);
    if (!backend || max_ctx <= 0) return false;

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() *
        static_cast<size_t>(weights.n_layer * 6 + 16) + 16384;
    params.no_alloc = true;
    out.ctx = ggml_init(params);
    if (!out.ctx) return false;

    out.layers.resize(static_cast<size_t>(weights.n_layer));
    const int64_t d_inner =
        static_cast<int64_t>(weights.kda_head_dim) * weights.n_head;
    const int compact_dim = weights.kv_lora_rank + weights.rope_dim;
    for (int il = 0; il < weights.n_layer; ++il) {
        KimiK3LayerCache & layer_cache =
            out.layers[static_cast<size_t>(il)];
        char name[80];
        if (weights.layers[static_cast<size_t>(il)].recurrent) {
            layer_cache.conv_state = ggml_new_tensor_2d(
                out.ctx, GGML_TYPE_F32, weights.ssm_d_conv - 1, 3 * d_inner);
            layer_cache.ssm_state = ggml_new_tensor_3d(
                out.ctx, GGML_TYPE_F32, weights.kda_head_dim,
                weights.kda_head_dim, weights.n_head);
            std::snprintf(name, sizeof(name), "kimi_k3_conv_state_%d", il);
            ggml_set_name(layer_cache.conv_state, name);
            std::snprintf(name, sizeof(name), "kimi_k3_ssm_state_%d", il);
            ggml_set_name(layer_cache.ssm_state, name);

            if (max_verify_tokens > 0) {
                layer_cache.conv_state_snap =
                    ggml_dup_tensor(out.ctx, layer_cache.conv_state);
                layer_cache.ssm_state_snap =
                    ggml_dup_tensor(out.ctx, layer_cache.ssm_state);
                layer_cache.replay_input = ggml_new_tensor_2d(
                    out.ctx, GGML_TYPE_F32, weights.n_embd,
                    max_verify_tokens);
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
            layer_cache.mla_k = ggml_new_tensor_3d(
                out.ctx, GGML_TYPE_F16, compact_dim, 1, max_ctx);
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
    kimi_k3_destroy_graph_state(cache.persistent_routed_preparation);
    if (cache.buf) ggml_backend_buffer_free(cache.buf);
    if (cache.ctx) ggml_free(cache.ctx);
    cache = KimiK3Cache{};
}

} // namespace dflash::common
