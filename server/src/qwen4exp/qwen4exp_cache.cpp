#include "qwen4exp_cache.h"

#include <cstdio>

namespace dflash::common {

bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, ggml_type kv_type, Qwen4ExpCache & out) {
    if (max_ctx <= 0) return false;

    out.full_layer_ids.clear();
    out.linear_layer_ids.clear();
    for (int il = 0; il < w.n_layer; ++il) {
        if (w.layers[il].is_full_attention) out.full_layer_ids.push_back(il);
        else                                 out.linear_layer_ids.push_back(il);
    }
    const size_t n_full   = out.full_layer_ids.size();
    const size_t n_linear = out.linear_layer_ids.size();
    if (n_full + n_linear != static_cast<size_t>(w.n_layer)) return false;

    // conv_channels = 2 * n_group * d_state + d_inner, the width of the fused
    // q|k|v projection the depthwise conv runs over.
    const int64_t conv_channels =
        2 * static_cast<int64_t>(w.ssm_n_group) * w.ssm_d_state + w.ssm_d_inner;
    const int64_t S_v = w.ssm_d_state;
    const int64_t H_v = w.linear_value_heads;
    const int64_t kernel = w.ssm_d_conv;

    if (w.n_embd_head_k != 256 || w.n_head_kv != 2 || S_v != 128 ||
        kernel < 2 || conv_channels <= 0) {
        std::fprintf(stderr,
            "[qwen4exp] cache: unsupported state shape (head=%d kv=%d d_state=%lld conv=%lld)\n",
            w.n_embd_head_k, w.n_head_kv, static_cast<long long>(S_v),
            static_cast<long long>(conv_channels));
        return false;
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (static_cast<size_t>(w.n_layer) * 3 + 8) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) return false;

    out.attn_k.assign(n_full, nullptr);
    out.attn_v.assign(n_full, nullptr);
    out.ssm_state.assign(n_linear, nullptr);
    out.conv_state.assign(n_linear, nullptr);

    for (size_t i = 0; i < n_full; ++i) {
        out.attn_k[i] = ggml_new_tensor_3d(out.ctx, kv_type,
            w.n_embd_head_k, max_ctx, w.n_head_kv);
        out.attn_v[i] = ggml_new_tensor_3d(out.ctx, kv_type,
            w.n_embd_head_v, max_ctx, w.n_head_kv);
    }
    for (size_t i = 0; i < n_linear; ++i) {
        // Recurrent state is independent of context length.
        out.ssm_state[i] = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32, S_v, S_v, H_v);
        out.conv_state[i] = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, kernel - 1, conv_channels);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    out.kv_type = kv_type;

    // A fresh cache must start from zero recurrent state, not whatever the
    // backend buffer happened to contain.
    reset_qwen4exp_state(backend, out);

    const size_t kv_bytes_per_token =
        n_full * (static_cast<size_t>(w.n_embd_head_k) + w.n_embd_head_v) *
        w.n_head_kv * ggml_type_size(kv_type) / ggml_blck_size(kv_type);
    std::fprintf(stderr,
        "[qwen4exp] cache: %zu full + %zu linear layers, kv=%s %.2f MiB @ ctx=%d, "
        "ssm_state %.1f MiB, conv_state %.1f MiB\n",
        n_full, n_linear, ggml_type_name(kv_type),
        kv_bytes_per_token * static_cast<size_t>(max_ctx) / (1024.0 * 1024.0),
        max_ctx,
        n_linear * static_cast<double>(S_v * S_v * H_v * 4) / (1024.0 * 1024.0),
        n_linear * static_cast<double>((kernel - 1) * conv_channels * 4) / (1024.0 * 1024.0));
    return true;
}

void free_qwen4exp_cache(Qwen4ExpCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.full_layer_ids.clear();
    c.linear_layer_ids.clear();
    c.cur_pos = 0;
    c.max_ctx = 0;
}

void reset_qwen4exp_state(ggml_backend_t backend, Qwen4ExpCache & c) {
    (void) backend;
    for (ggml_tensor * t : c.ssm_state) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    for (ggml_tensor * t : c.conv_state) {
        if (t) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    c.cur_pos = 0;
}

}  // namespace dflash::common
