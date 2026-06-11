#include "lsa_attention_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace dflash::common {
namespace {

bool finite_vector(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

}  // namespace

bool lsa_reference_packed_attention(const std::vector<float> & query,
                                    const std::vector<float> & packed_key,
                                    const std::vector<float> & packed_value,
                                    int q_heads,
                                    int kv_heads,
                                    int head_dim,
                                    const LsaPackedPlan & plan,
                                    int query_position,
                                    std::vector<float> & output,
                                    std::string & error) {
    output.clear();
    error.clear();
    if (q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_heads % kv_heads != 0 || query_position < 0 ||
        plan.token_capacity <= 0 ||
        plan.active_tokens() > plan.token_capacity) {
        error = "LSA reference attention geometry is invalid";
        return false;
    }
    const size_t query_size = static_cast<size_t>(q_heads) * head_dim;
    const size_t cache_size =
        static_cast<size_t>(kv_heads) * plan.token_capacity * head_dim;
    if (query.size() != query_size || packed_key.size() != cache_size ||
        packed_value.size() != cache_size) {
        error = "LSA reference attention tensor size is invalid";
        return false;
    }
    if (!finite_vector(query) || !finite_vector(packed_key) ||
        !finite_vector(packed_value)) {
        error = "LSA reference attention contains a non-finite value";
        return false;
    }
    if (!std::is_sorted(plan.source_positions.begin(),
                        plan.source_positions.end())) {
        error = "LSA reference attention positions are not sorted";
        return false;
    }

    output.assign(query_size, 0.0f);
    const int query_heads_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(plan.source_positions.size());

    for (int query_head = 0; query_head < q_heads; ++query_head) {
        const int kv_head = query_head / query_heads_per_kv;
        const size_t query_offset =
            static_cast<size_t>(query_head) * head_dim;
        const size_t cache_head_offset =
            static_cast<size_t>(kv_head) * plan.token_capacity * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        int visible = 0;
        for (size_t token = 0; token < plan.source_positions.size(); ++token) {
            if (plan.source_positions[token] > query_position) break;
            const size_t key_offset =
                cache_head_offset + token * head_dim;
            float score = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim) {
                score += query[query_offset + dim] *
                         packed_key[key_offset + dim];
            }
            scores[token] = score * scale;
            maximum = std::max(maximum, scores[token]);
            ++visible;
        }
        if (visible == 0) {
            error = "LSA reference attention has no visible key";
            output.clear();
            return false;
        }

        float denominator = 0.0f;
        for (int token = 0; token < visible; ++token) {
            scores[static_cast<size_t>(token)] =
                std::exp(scores[static_cast<size_t>(token)] - maximum);
            denominator += scores[static_cast<size_t>(token)];
        }
        for (int token = 0; token < visible; ++token) {
            const float probability =
                scores[static_cast<size_t>(token)] / denominator;
            const size_t value_offset =
                cache_head_offset + static_cast<size_t>(token) * head_dim;
            for (int dim = 0; dim < head_dim; ++dim) {
                output[query_offset + dim] +=
                    probability * packed_value[value_offset + dim];
            }
        }
    }
    return true;
}

bool lsa_reference_dense_attention(const std::vector<float> & query,
                                   const std::vector<float> & key,
                                   const std::vector<float> & value,
                                   int q_heads,
                                   int kv_heads,
                                   int head_dim,
                                   int source_tokens,
                                   int query_position,
                                   std::vector<float> & output,
                                   std::string & error) {
    if (source_tokens <= 0) {
        error = "dense LSA reference requires source tokens";
        output.clear();
        return false;
    }
    LsaPackedPlan dense;
    dense.committed_tokens = source_tokens;
    dense.token_capacity = source_tokens;
    dense.source_positions.resize(static_cast<size_t>(source_tokens));
    std::iota(dense.source_positions.begin(), dense.source_positions.end(), 0);
    return lsa_reference_packed_attention(
        query, key, value, q_heads, kv_heads, head_dim, dense,
        query_position, output, error);
}

}  // namespace dflash::common
