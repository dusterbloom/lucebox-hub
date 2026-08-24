#include "kimi_k3_internal.h"

#include "internal.h"

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

void free_kimi_k3_prefix_snapshot(KimiK3PrefixSnapshot & snapshot) {
    if (snapshot.buf) ggml_backend_buffer_free(snapshot.buf);
    if (snapshot.ctx) ggml_free(snapshot.ctx);
    snapshot = KimiK3PrefixSnapshot{};
}

bool save_kimi_k3_prefix_snapshot(
        const KimiK3Weights & weights,
        const KimiK3Cache & cache,
        ggml_backend_t snapshot_backend,
        const std::vector<float> & final_logits,
        KimiK3PrefixSnapshot & snapshot) {
    if (!snapshot_backend || cache.cur_pos <= 0 ||
        cache.cur_pos > cache.max_ctx ||
        cache.layers.size() != static_cast<size_t>(weights.n_layer) ||
        final_logits.size() != static_cast<size_t>(weights.n_vocab)) {
        set_last_error("Kimi-K3 prefix snapshot has invalid source state");
        return false;
    }

    const bool needs_alloc = !snapshot.ctx ||
        snapshot.cur_pos != cache.cur_pos ||
        snapshot.max_ctx != cache.max_ctx ||
        snapshot.conv_state.size() != cache.layers.size();
    if (needs_alloc) {
        free_kimi_k3_prefix_snapshot(snapshot);

        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() *
            (cache.layers.size() * 3 + 16) + 16384;
        params.no_alloc = true;
        snapshot.ctx = ggml_init(params);
        if (!snapshot.ctx) {
            set_last_error("Kimi-K3 prefix snapshot context allocation failed");
            return false;
        }

        snapshot.conv_state.assign(cache.layers.size(), nullptr);
        snapshot.ssm_state.assign(cache.layers.size(), nullptr);
        snapshot.mla_k.assign(cache.layers.size(), nullptr);
        for (size_t il = 0; il < cache.layers.size(); ++il) {
            const KimiK3LayerCache & source = cache.layers[il];
            if (source.conv_state || source.ssm_state) {
                if (!source.conv_state || !source.ssm_state) {
                    set_last_error(
                        "Kimi-K3 prefix snapshot found incomplete recurrent state");
                    free_kimi_k3_prefix_snapshot(snapshot);
                    return false;
                }
                snapshot.conv_state[il] =
                    ggml_dup_tensor(snapshot.ctx, source.conv_state);
                snapshot.ssm_state[il] =
                    ggml_dup_tensor(snapshot.ctx, source.ssm_state);
            } else if (source.mla_k) {
                snapshot.mla_k[il] = ggml_new_tensor_3d(
                    snapshot.ctx, source.mla_k->type,
                    source.mla_k->ne[0], source.mla_k->ne[1],
                    cache.cur_pos);
            } else {
                set_last_error(
                    "Kimi-K3 prefix snapshot found a layer without state");
                free_kimi_k3_prefix_snapshot(snapshot);
                return false;
            }
        }

        snapshot.buf =
            ggml_backend_alloc_ctx_tensors(snapshot.ctx, snapshot_backend);
        if (!snapshot.buf) {
            set_last_error("Kimi-K3 prefix snapshot buffer allocation failed");
            free_kimi_k3_prefix_snapshot(snapshot);
            return false;
        }
        snapshot.cur_pos = cache.cur_pos;
        snapshot.max_ctx = cache.max_ctx;
        std::fprintf(stderr,
            "[kimi-k3-snap] allocated cur_pos=%d bytes=%zu backend=%s\n",
            cache.cur_pos, ggml_backend_buffer_get_size(snapshot.buf),
            ggml_backend_name(snapshot_backend));
    }

    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const KimiK3LayerCache & source = cache.layers[il];
        if (source.conv_state) {
            if (!snapshot.conv_state[il] || !snapshot.ssm_state[il]) {
                set_last_error("Kimi-K3 prefix snapshot recurrent layout mismatch");
                return false;
            }
            ggml_backend_tensor_copy(
                source.conv_state, snapshot.conv_state[il]);
            ggml_backend_tensor_copy(
                source.ssm_state, snapshot.ssm_state[il]);
            continue;
        }
        ggml_tensor * destination = snapshot.mla_k[il];
        if (!source.mla_k || !destination) {
            set_last_error("Kimi-K3 prefix snapshot MLA layout mismatch");
            return false;
        }
        ggml_backend_tensor_get(
            source.mla_k, destination->data, 0, ggml_nbytes(destination));
    }
    snapshot.final_logits = final_logits;
    snapshot.cur_pos = cache.cur_pos;
    snapshot.max_ctx = cache.max_ctx;
    return true;
}

bool restore_kimi_k3_prefix_snapshot(
        const KimiK3PrefixSnapshot & snapshot,
        KimiK3Cache & cache) {
    if (!cache.buf || !snapshot.ctx || !snapshot.buf ||
        snapshot.cur_pos <= 0 ||
        snapshot.cur_pos > cache.max_ctx ||
        snapshot.max_ctx != cache.max_ctx ||
        snapshot.conv_state.size() != cache.layers.size() ||
        snapshot.ssm_state.size() != cache.layers.size() ||
        snapshot.mla_k.size() != cache.layers.size()) {
        set_last_error("Kimi-K3 prefix snapshot is stale or incompatible");
        return false;
    }

    // Validate the complete topology before touching live state. A rejected
    // slot must leave the currently serving cache intact.
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        const KimiK3LayerCache & destination = cache.layers[il];
        if (destination.conv_state || destination.ssm_state) {
            if (!destination.conv_state || !destination.ssm_state ||
                !snapshot.conv_state[il] || !snapshot.ssm_state[il] ||
                snapshot.mla_k[il] ||
                destination.conv_state->type != snapshot.conv_state[il]->type ||
                destination.ssm_state->type != snapshot.ssm_state[il]->type ||
                !ggml_are_same_shape(
                    destination.conv_state, snapshot.conv_state[il]) ||
                !ggml_are_same_shape(
                    destination.ssm_state, snapshot.ssm_state[il])) {
                set_last_error(
                    "Kimi-K3 prefix restore recurrent layout mismatch");
                return false;
            }
            continue;
        }

        ggml_tensor * source = snapshot.mla_k[il];
        if (!destination.mla_k || !source || snapshot.conv_state[il] ||
            snapshot.ssm_state[il] ||
            destination.mla_k->type != source->type ||
            destination.mla_k->ne[0] != source->ne[0] ||
            destination.mla_k->ne[1] != source->ne[1] ||
            source->ne[2] != snapshot.cur_pos ||
            destination.mla_k->ne[2] < source->ne[2] ||
            destination.mla_k->ne[3] != source->ne[3] ||
            !ggml_is_contiguous(source) ||
            ggml_nbytes(source) > ggml_nbytes(destination.mla_k)) {
            set_last_error("Kimi-K3 prefix restore MLA layout mismatch");
            return false;
        }
    }

    // Do not clear the whole cache allocation here. MLA is bounded by cur_pos,
    // recurrent state is fully overwritten below, and replay metadata is
    // invalidated after the copy. Clearing M1024 replay scratch would turn an
    // otherwise small restore into a multi-GiB memory-fabric operation.
    for (size_t il = 0; il < cache.layers.size(); ++il) {
        KimiK3LayerCache & destination = cache.layers[il];
        if (destination.conv_state) {
            ggml_backend_tensor_copy(
                snapshot.conv_state[il], destination.conv_state);
            ggml_backend_tensor_copy(
                snapshot.ssm_state[il], destination.ssm_state);
            continue;
        }
        ggml_tensor * source = snapshot.mla_k[il];
        ggml_backend_tensor_set(
            destination.mla_k, source->data, 0, ggml_nbytes(source));
    }

    cache.cur_pos = snapshot.cur_pos;
    cache.snapshot_pos = -1;
    cache.replay_base_pos = -1;
    cache.replay_n_tokens = 0;
    cache.snapshot_valid = false;
    cache.replay_valid = false;
    cache.recurrent_state_pristine = false;
    cache.replay_exact_rows = false;
    return true;
}

} // namespace dflash::common
