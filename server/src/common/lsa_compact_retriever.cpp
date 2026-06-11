#include "lsa_compact_retriever.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
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

uint64_t fnv1a64(const std::vector<uint8_t> & bytes) {
    uint64_t value = 14695981039346656037ULL;
    for (uint8_t byte : bytes) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    return value;
}

std::string hex64(uint64_t value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (int index = 15; index >= 0; --index) {
        result[static_cast<size_t>(index)] = digits[value & 0xfU];
        value >>= 4;
    }
    return result;
}

}  // namespace

LsaCompactRetriever::LsaCompactRetriever(LsaCompactConfig config)
    : config_(config) {}

bool LsaCompactRetriever::load_artifact(const std::string & path,
                                        std::string & error) {
    namespace fs = std::filesystem;
    error.clear();
    fs::path manifest_path(path);
    std::error_code fs_error;
    if (fs::is_directory(manifest_path, fs_error)) {
        manifest_path /= "encoder.json";
    }
    std::ifstream input(manifest_path);
    if (!input) {
        error = "failed to open compact retriever manifest";
        return false;
    }

    nlohmann::json manifest;
    try {
        input >> manifest;
    } catch (const std::exception &) {
        error = "failed to parse compact retriever manifest";
        return false;
    }
    try {
        if (manifest.at("schema").get<std::string>() !=
            "luce.lsa.qwen35.encoder.v1") {
            error = "unsupported compact retriever schema";
            return false;
        }
        const auto & dataset = manifest.at("dataset");
        LsaCompactConfig parsed;
        parsed.hidden_size = dataset.at("hidden_size").get<int>();
        parsed.rank = manifest.at("rank").get<int>();
        parsed.kv_heads = dataset.at("kv_heads").get<int>();
        parsed.head_dim = dataset.at("head_dim").get<int>();
        parsed.score_temperature =
            manifest.at("score_temperature").get<float>();
        parsed.decision_threshold =
            manifest.at("decision_threshold").get<float>();
        parsed.logit_scale = manifest.at("logit_scale").get<float>();
        if (!valid_config(parsed)) {
            error = "compact retriever manifest configuration is invalid";
            return false;
        }
        if (parsed.kv_heads >
            std::numeric_limits<int>::max() / parsed.head_dim) {
            error = "compact retriever manifest key size overflows";
            return false;
        }

        const auto & weight = manifest.at("weight_file");
        if (weight.at("dtype").get<std::string>() != "float16-le") {
            error = "compact retriever weight dtype is unsupported";
            return false;
        }
        const std::string name = weight.at("name").get<std::string>();
        const fs::path relative(name);
        if (relative.empty() || relative.is_absolute() ||
            relative.filename() != relative) {
            error = "compact retriever weight path is unsafe";
            return false;
        }

        const size_t hidden_size = static_cast<size_t>(parsed.hidden_size);
        const size_t rank = static_cast<size_t>(parsed.rank);
        const size_t kv_heads = static_cast<size_t>(parsed.kv_heads);
        const size_t head_dim = static_cast<size_t>(parsed.head_dim);
        if (rank > std::numeric_limits<size_t>::max() / hidden_size ||
            kv_heads > std::numeric_limits<size_t>::max() / head_dim) {
            error = "compact retriever manifest weight size overflows";
            return false;
        }
        const size_t down_count = rank * hidden_size;
        const size_t key_size = kv_heads * head_dim;
        if (key_size > std::numeric_limits<size_t>::max() / rank) {
            error = "compact retriever manifest weight size overflows";
            return false;
        }
        const size_t up_count = key_size * rank;
        if (down_count > std::numeric_limits<size_t>::max() - up_count ||
            down_count + up_count >
                std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
            error = "compact retriever manifest weight size overflows";
            return false;
        }
        const size_t expected_size =
            (down_count + up_count) * sizeof(uint16_t);
        if (weight.at("size_bytes").get<size_t>() != expected_size) {
            error = "compact retriever manifest weight size is invalid";
            return false;
        }
        const auto & layout = weight.at("layout");
        if (!layout.is_array() || layout.size() != 2 ||
            layout[0].at("name").get<std::string>() != "down.weight" ||
            layout[1].at("name").get<std::string>() != "up.weight" ||
            layout[0].at("shape") !=
                nlohmann::json::array({parsed.rank, parsed.hidden_size}) ||
            layout[1].at("shape") !=
                nlohmann::json::array(
                    {static_cast<int>(key_size), parsed.rank}) ||
            layout[0].at("offset_bytes").get<size_t>() != 0 ||
            layout[1].at("offset_bytes").get<size_t>() !=
                down_count * sizeof(uint16_t)) {
            error = "compact retriever manifest layout is invalid";
            return false;
        }

        const fs::path weight_path = manifest_path.parent_path() / relative;
        std::ifstream weights(weight_path, std::ios::binary | std::ios::ate);
        if (!weights || weights.tellg() < 0 ||
            static_cast<size_t>(weights.tellg()) != expected_size) {
            error = "compact retriever artifact weight size is invalid";
            return false;
        }
        weights.seekg(0);
        std::vector<uint8_t> bytes(expected_size);
        weights.read(reinterpret_cast<char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!weights) {
            error = "failed to read compact retriever artifact weights";
            return false;
        }
        if (hex64(fnv1a64(bytes)) !=
            weight.at("fnv1a64").get<std::string>()) {
            error = "compact retriever artifact checksum mismatch";
            return false;
        }

        LsaCompactRetriever candidate(parsed);
        if (!candidate.load_f16_weights(weight_path.string(), error)) {
            return false;
        }
        *this = std::move(candidate);
        return true;
    } catch (const std::exception &) {
        error = "compact retriever manifest is incomplete";
        return false;
    }
}

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
