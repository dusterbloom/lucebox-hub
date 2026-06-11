#pragma once

#include "internal.h"

#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35LsaCaptureConfig {
    int hidden_layer = 46;
    std::vector<int> qk_layers;
};

struct Qwen35LsaModelShape {
    int n_layer = 64;
    int full_attention_interval = 4;
};

struct Qwen35LsaLayerCapture {
    int layer = -1;
    // Tensor order is [token, head, head_dim] with head_dim contiguous.
    std::vector<float> k_pre_rope;
    std::vector<float> q_post_rope;
    std::vector<float> k_post_rope;
};

struct Qwen35LsaCaptureBatch {
    int n_tokens = 0;
    std::vector<float> hidden;
    std::vector<Qwen35LsaLayerCapture> layers;
};

Qwen35LsaCaptureConfig qwen35_lsa_default_capture_config();
Qwen35LsaCaptureConfig qwen35_lsa_capture_config(
    const Qwen35LsaModelShape & shape,
    int hidden_layer);
Qwen35LsaCaptureConfig qwen35_lsa_capture_config(
    const TargetWeights & weights);

bool configure_qwen35_lsa_capture(const Qwen35LsaModelShape & shape,
                                  const Qwen35LsaCaptureConfig & config,
                                  QwenGraphInputs & inputs,
                                  std::string & error);

bool configure_qwen35_lsa_capture(const TargetWeights & weights,
                                  const Qwen35LsaCaptureConfig & config,
                                  QwenGraphInputs & inputs,
                                  std::string & error);

bool read_qwen35_lsa_capture(const TargetWeights & weights,
                             const Qwen35LsaCaptureConfig & config,
                             const QwenGraphOutputs & outputs,
                             int n_tokens,
                             Qwen35LsaCaptureBatch & batch,
                             std::string & error);

// Mean pool token keys and L2-normalize each KV head. Input tensor order is
// [token, head, head_dim], matching contiguous ggml [head_dim, head, token].
bool pool_qwen35_lsa_block_keys(const std::vector<float> & key_pre_rope,
                                int n_tokens,
                                int kv_heads,
                                int head_dim,
                                std::vector<float> & pooled,
                                std::string & error);

}  // namespace dflash::common
