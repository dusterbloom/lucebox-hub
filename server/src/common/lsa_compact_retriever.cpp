#include "lsa_compact_retriever.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace dflash::common {
namespace {

float half_to_float(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | mantissa << 13;
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

bool valid_config(const LsaCompactConfig & config) {
    return config.hidden_size > 0 && config.rank > 0 && config.kv_heads > 0 &&
           config.head_dim > 0 && std::isfinite(config.score_temperature) &&
           config.score_temperature > 0 &&
           std::isfinite(config.decision_threshold) &&
           std::isfinite(config.logit_scale) && config.logit_scale > 0;
}

}  // namespace

LsaCompactRetriever::LsaCompactRetriever(LsaCompactConfig config)
    : config_(config) {}

bool LsaCompactRetriever::set_weights(std::vector<float> down,
                                      std::vector<float> up,
                                      std::string & error) {
    error.clear();
    if (!valid_config(config_)) {
        error = "compact retriever configuration is invalid";
        return false;
    }
    const size_t down_count =
        static_cast<size_t>(config_.rank) * config_.hidden_size;
    const size_t up_count =
        static_cast<size_t>(key_size()) * config_.rank;
    if (down.size() != down_count || up.size() != up_count) {
        error = "compact retriever weight shape is invalid";
        return false;
    }
    const auto finite = [](const std::vector<float> & values) {
        return std::all_of(values.begin(), values.end(),
                           [](float value) { return std::isfinite(value); });
    };
    if (!finite(down) || !finite(up)) {
        error = "compact retriever weights contain a non-finite value";
        return false;
    }
    down_ = std::move(down);
    up_ = std::move(up);
    return true;
}

bool LsaCompactRetriever::load_f16_weights(const std::string & path,
                                           std::string & error) {
    error.clear();
    if (!valid_config(config_)) {
        error = "compact retriever configuration is invalid";
        return false;
    }
    const size_t down_count =
        static_cast<size_t>(config_.rank) * config_.hidden_size;
    const size_t up_count =
        static_cast<size_t>(key_size()) * config_.rank;
    const size_t count = down_count + up_count;
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
        error = "compact retriever weight size overflows";
        return false;
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        error = "failed to open compact retriever weights";
        return false;
    }
    const auto size = input.tellg();
    if (size < 0 || static_cast<uint64_t>(size) != count * sizeof(uint16_t)) {
        error = "compact retriever weight file size is invalid";
        return false;
    }
    input.seekg(0);
    std::vector<uint16_t> packed(count);
    input.read(reinterpret_cast<char *>(packed.data()),
               static_cast<std::streamsize>(count * sizeof(uint16_t)));
    if (!input) {
        error = "failed to read compact retriever weights";
        return false;
    }

    std::vector<float> down(down_count);
    std::vector<float> up(up_count);
    for (size_t i = 0; i < down_count; ++i) down[i] = half_to_float(packed[i]);
    for (size_t i = 0; i < up_count; ++i) {
        up[i] = half_to_float(packed[down_count + i]);
    }
    return set_weights(std::move(down), std::move(up), error);
}

bool LsaCompactRetriever::score(const std::vector<float> & hidden,
                                const std::vector<LsaChunk> & chunks,
                                std::vector<float> & scores,
                                std::string & error) {
    error.clear();
    if (hidden.size() != static_cast<size_t>(config_.hidden_size)) {
        error = "hidden size does not match compact retriever";
        return false;
    }
    if (down_.empty() || up_.empty()) {
        error = "compact retriever weights are not loaded";
        return false;
    }

    std::vector<float> rank(config_.rank, 0.0f);
    for (int row = 0; row < config_.rank; ++row) {
        const float * weight =
            down_.data() + static_cast<size_t>(row) * config_.hidden_size;
        float sum = 0.0f;
        for (int column = 0; column < config_.hidden_size; ++column) {
            sum += weight[column] * hidden[column];
        }
        rank[row] = sum / (1.0f + std::exp(-sum));
    }

    std::vector<float> query(key_size(), 0.0f);
    for (int row = 0; row < key_size(); ++row) {
        const float * weight =
            up_.data() + static_cast<size_t>(row) * config_.rank;
        for (int column = 0; column < config_.rank; ++column) {
            query[row] += weight[column] * rank[column];
        }
    }
    for (int head = 0; head < config_.kv_heads; ++head) {
        float norm = 0.0f;
        float * begin = query.data() + static_cast<size_t>(head) * config_.head_dim;
        for (int dim = 0; dim < config_.head_dim; ++dim) {
            norm += begin[dim] * begin[dim];
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int dim = 0; dim < config_.head_dim; ++dim) begin[dim] /= norm;
    }

    scores.clear();
    scores.reserve(chunks.size());
    for (const LsaChunk & chunk : chunks) {
        if (chunk.index_key.size() != static_cast<size_t>(key_size())) {
            error = "chunk key size does not match compact retriever";
            return false;
        }
        float maximum = -std::numeric_limits<float>::infinity();
        std::vector<float> head_scores(config_.kv_heads);
        for (int head = 0; head < config_.kv_heads; ++head) {
            const size_t offset = static_cast<size_t>(head) * config_.head_dim;
            float dot = 0.0f;
            float key_norm = 0.0f;
            for (int dim = 0; dim < config_.head_dim; ++dim) {
                const float key = chunk.index_key[offset + dim];
                dot += query[offset + dim] * key;
                key_norm += key * key;
            }
            head_scores[head] = dot / std::sqrt(std::max(key_norm, 1e-12f));
            maximum = std::max(maximum,
                               head_scores[head] / config_.score_temperature);
        }
        float exponential_sum = 0.0f;
        for (float value : head_scores) {
            exponential_sum +=
                std::exp(value / config_.score_temperature - maximum);
        }
        const float pooled =
            config_.score_temperature *
            (maximum + std::log(exponential_sum / config_.kv_heads));
        const float logit =
            (pooled - config_.decision_threshold) * config_.logit_scale;
        scores.push_back(1.0f / (1.0f + std::exp(-logit)));
    }
    return true;
}

}  // namespace dflash::common
