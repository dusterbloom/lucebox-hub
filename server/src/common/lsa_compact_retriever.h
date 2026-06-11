#pragma once

#include "lsa_runtime.h"

#include <string>
#include <vector>

namespace dflash::common {

struct LsaCompactConfig {
    int hidden_size = 5120;
    int rank = 256;
    int kv_heads = 4;
    int head_dim = 256;
    float score_temperature = 0.1f;
    float decision_threshold = 0.02f;
    float logit_scale = 12.0f;
};

class LsaCompactRetriever final : public LsaRetriever {
public:
    LsaCompactRetriever() = default;
    explicit LsaCompactRetriever(LsaCompactConfig config);

    int hidden_size() const override { return config_.hidden_size; }
    int key_size() const override { return config_.kv_heads * config_.head_dim; }
    const LsaCompactConfig & config() const { return config_; }

    bool load_artifact(const std::string & path, std::string & error);
    bool load_f16_weights(const std::string & path, std::string & error);
    bool set_weights(std::vector<float> down, std::vector<float> up,
                     std::string & error);

    bool score(const std::vector<float> & hidden,
               const std::vector<LsaChunk> & chunks,
               std::vector<float> & scores,
               std::string & error) override;

private:
    LsaCompactConfig config_;
    std::vector<float> down_;
    std::vector<float> up_;
};

}  // namespace dflash::common
