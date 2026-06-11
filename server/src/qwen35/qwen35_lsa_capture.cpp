#include "qwen35_lsa_capture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>

namespace dflash::common {
namespace {

bool is_full_attention_layer(const Qwen35LsaModelShape & shape, int layer) {
    return layer >= 0 && layer < shape.n_layer &&
           shape.full_attention_interval > 0 &&
           ((layer + 1) % shape.full_attention_interval) == 0;
}

bool validate_f32_tensor(ggml_tensor * tensor,
                         int64_t ne0,
                         int64_t ne1,
                         int64_t ne2,
                         const char * name,
                         std::string & error) {
    if (!tensor) {
        error = std::string("missing ") + name + " capture";
        return false;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
        tensor->ne[0] != ne0 || tensor->ne[1] != ne1 ||
        tensor->ne[2] != ne2) {
        error = std::string(name) + " capture geometry is invalid";
        return false;
    }
    return true;
}

}  // namespace

Qwen35LsaCaptureConfig qwen35_lsa_default_capture_config() {
    Qwen35LsaCaptureConfig config;
    config.qk_layers.reserve(16);
    for (int layer = 3; layer < 64; layer += 4) {
        config.qk_layers.push_back(layer);
    }
    return config;
}

Qwen35LsaCaptureConfig qwen35_lsa_capture_config(
    const Qwen35LsaModelShape & shape,
    int hidden_layer) {
    Qwen35LsaCaptureConfig config;
    config.hidden_layer = hidden_layer;
    for (int layer = shape.full_attention_interval - 1;
         layer < shape.n_layer;
         layer += shape.full_attention_interval) {
        config.qk_layers.push_back(layer);
    }
    return config;
}

Qwen35LsaCaptureConfig qwen35_lsa_capture_config(
    const TargetWeights & weights) {
    int hidden_layer = 0;
    if (weights.n_capture_layers >= 4) {
        hidden_layer = weights.capture_layer_ids[3];
    } else {
        hidden_layer = std::max(0, (3 * weights.n_layer) / 4 - 1);
    }
    return qwen35_lsa_capture_config(
        {weights.n_layer, weights.full_attention_interval},
        hidden_layer);
}

bool configure_qwen35_lsa_capture(const Qwen35LsaModelShape & shape,
                                  const Qwen35LsaCaptureConfig & config,
                                  QwenGraphInputs & inputs,
                                  std::string & error) {
    error.clear();
    if (shape.n_layer <= 0 || shape.n_layer > 64 ||
        shape.full_attention_interval <= 0) {
        error = "Qwen LSA capture requires 1-64 model layers";
        return false;
    }
    if (config.hidden_layer < 0 || config.hidden_layer >= shape.n_layer) {
        error = "Qwen LSA hidden capture layer is invalid";
        return false;
    }
    if (config.qk_layers.empty()) {
        error = "Qwen LSA requires at least one Q/K capture layer";
        return false;
    }

    uint64_t mask = 0;
    std::unordered_set<int> seen;
    for (int layer : config.qk_layers) {
        if (!is_full_attention_layer(shape, layer)) {
            error = "Qwen LSA Q/K capture layer is not a full-attention layer";
            return false;
        }
        if (!seen.insert(layer).second) {
            error = "Qwen LSA Q/K capture layer is duplicated";
            return false;
        }
        mask |= uint64_t{1} << layer;
    }
    inputs.lsa_hidden_capture_layer = config.hidden_layer;
    inputs.lsa_qk_capture_mask = mask;
    return true;
}

bool configure_qwen35_lsa_capture(const TargetWeights & weights,
                                  const Qwen35LsaCaptureConfig & config,
                                  QwenGraphInputs & inputs,
                                  std::string & error) {
    return configure_qwen35_lsa_capture(
        {weights.n_layer, weights.full_attention_interval},
        config, inputs, error);
}

bool read_qwen35_lsa_capture(const TargetWeights & weights,
                             const Qwen35LsaCaptureConfig & config,
                             const QwenGraphOutputs & outputs,
                             int n_tokens,
                             Qwen35LsaCaptureBatch & batch,
                             std::string & error) {
    batch = {};
    error.clear();
    if (n_tokens <= 0 || weights.n_embd <= 0 || weights.n_head <= 0 ||
        weights.n_head_kv <= 0 || weights.n_embd_head_k <= 0) {
        error = "Qwen LSA capture dimensions are invalid";
        return false;
    }
    if (!validate_f32_tensor(outputs.lsa_hidden, weights.n_embd, n_tokens, 1,
                             "hidden", error)) {
        return false;
    }
    if (outputs.lsa_k_pre_rope.size() !=
            static_cast<size_t>(weights.n_layer) ||
        outputs.lsa_q_post_rope.size() !=
            static_cast<size_t>(weights.n_layer) ||
        outputs.lsa_k_post_rope.size() !=
            static_cast<size_t>(weights.n_layer)) {
        error = "Qwen LSA Q/K output vectors do not match model layers";
        return false;
    }

    batch.n_tokens = n_tokens;
    batch.hidden.resize(
        static_cast<size_t>(weights.n_embd) * n_tokens);
    ggml_backend_tensor_get(outputs.lsa_hidden, batch.hidden.data(), 0,
                            batch.hidden.size() * sizeof(float));

    for (int layer : config.qk_layers) {
        ggml_tensor * key_pre = outputs.lsa_k_pre_rope[(size_t)layer];
        ggml_tensor * query_post = outputs.lsa_q_post_rope[(size_t)layer];
        ggml_tensor * key_post = outputs.lsa_k_post_rope[(size_t)layer];
        if (!validate_f32_tensor(key_pre, weights.n_embd_head_k,
                                 weights.n_head_kv, n_tokens,
                                 "pre-RoPE key", error) ||
            !validate_f32_tensor(query_post, weights.n_embd_head_k,
                                 weights.n_head, n_tokens,
                                 "post-RoPE query", error) ||
            !validate_f32_tensor(key_post, weights.n_embd_head_k,
                                 weights.n_head_kv, n_tokens,
                                 "post-RoPE key", error)) {
            batch = {};
            return false;
        }
        Qwen35LsaLayerCapture capture;
        capture.layer = layer;
        capture.k_pre_rope.resize(
            static_cast<size_t>(weights.n_embd_head_k) *
            weights.n_head_kv * n_tokens);
        capture.q_post_rope.resize(
            static_cast<size_t>(weights.n_embd_head_k) *
            weights.n_head * n_tokens);
        capture.k_post_rope.resize(
            static_cast<size_t>(weights.n_embd_head_k) *
            weights.n_head_kv * n_tokens);
        ggml_backend_tensor_get(key_pre, capture.k_pre_rope.data(), 0,
                                capture.k_pre_rope.size() * sizeof(float));
        ggml_backend_tensor_get(query_post, capture.q_post_rope.data(), 0,
                                capture.q_post_rope.size() * sizeof(float));
        ggml_backend_tensor_get(key_post, capture.k_post_rope.data(), 0,
                                capture.k_post_rope.size() * sizeof(float));
        batch.layers.push_back(std::move(capture));
    }
    return true;
}

bool pool_qwen35_lsa_block_keys(const std::vector<float> & key_pre_rope,
                                int n_tokens,
                                int kv_heads,
                                int head_dim,
                                std::vector<float> & pooled,
                                std::string & error) {
    pooled.clear();
    error.clear();
    if (n_tokens <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        key_pre_rope.size() !=
            static_cast<size_t>(n_tokens) * kv_heads * head_dim) {
        error = "Qwen LSA block-key geometry is invalid";
        return false;
    }
    if (!std::all_of(key_pre_rope.begin(), key_pre_rope.end(),
                     [](float value) { return std::isfinite(value); })) {
        error = "Qwen LSA block keys contain a non-finite value";
        return false;
    }

    pooled.assign(static_cast<size_t>(kv_heads) * head_dim, 0.0f);
    for (int token = 0; token < n_tokens; ++token) {
        for (int head = 0; head < kv_heads; ++head) {
            const size_t input =
                (static_cast<size_t>(token) * kv_heads + head) * head_dim;
            const size_t output = static_cast<size_t>(head) * head_dim;
            for (int dim = 0; dim < head_dim; ++dim) {
                pooled[output + dim] += key_pre_rope[input + dim];
            }
        }
    }
    for (int head = 0; head < kv_heads; ++head) {
        const size_t offset = static_cast<size_t>(head) * head_dim;
        float norm = 0.0f;
        for (int dim = 0; dim < head_dim; ++dim) {
            pooled[offset + dim] /= n_tokens;
            norm += pooled[offset + dim] * pooled[offset + dim];
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int dim = 0; dim < head_dim; ++dim) {
            pooled[offset + dim] /= norm;
        }
    }
    return true;
}

}  // namespace dflash::common
