#include "common/lsa_packed_kv.h"

#include "common/lsa_attention_oracle.h"
#include "common/attn_masks.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

std::vector<LsaChunk> make_catalog(int chunks, int chunk_tokens) {
    std::vector<LsaChunk> out;
    for (int id = 0; id < chunks; ++id) {
        LsaChunk chunk;
        chunk.id = id;
        chunk.token_begin = id * chunk_tokens;
        chunk.token_end = chunk.token_begin + chunk_tokens;
        out.push_back(std::move(chunk));
    }
    return out;
}

void test_plan_combines_retrieval_sink_and_recent() {
    LsaPackedConfig config;
    config.token_capacity = 160;
    config.sink_tokens = 32;
    config.recent_tokens = 64;

    LsaPackedPlan plan;
    std::string error;
    CHECK(build_lsa_packed_plan(
        make_catalog(4, 64), {1}, 256, config, plan, error));
    CHECK(plan.active_tokens() == 160);
    CHECK(plan.source_positions.front() == 0);
    CHECK(plan.source_positions[31] == 31);
    CHECK(plan.source_positions[32] == 64);
    CHECK(plan.source_positions[95] == 127);
    CHECK(plan.source_positions[96] == 192);
    CHECK(plan.source_positions.back() == 255);

    config.token_capacity = 159;
    CHECK(!build_lsa_packed_plan(
        make_catalog(4, 64), {1}, 256, config, plan, error));
    CHECK(error.find("exceeds") != std::string::npos);
}

void test_all_chunks_matches_dense_order() {
    LsaPackedConfig config;
    config.token_capacity = 256;
    config.sink_tokens = 0;
    config.recent_tokens = 0;

    LsaPackedPlan plan;
    std::string error;
    CHECK(build_lsa_packed_plan(
        make_catalog(4, 64), {3, 1, 0, 2, 1}, 256, config, plan, error));
    CHECK(plan.active_tokens() == 256);
    for (int position = 0; position < 256; ++position) {
        CHECK(plan.source_positions[static_cast<size_t>(position)] == position);
    }
}

void test_original_position_causal_mask() {
    LsaPackedPlan plan;
    plan.committed_tokens = 12;
    plan.token_capacity = 8;
    plan.source_positions = {0, 2, 5, 9};

    std::vector<uint16_t> mask;
    std::string error;
    CHECK(build_lsa_packed_causal_mask(
        plan, {4, 9}, 4, mask, error, 8));
    CHECK(mask.size() == static_cast<size_t>(8 * KQ_MASK_PAD));
    CHECK(mask[0] == F16_ZERO);
    CHECK(mask[1] == F16_ZERO);
    CHECK(mask[2] == F16_NEG_INF);
    CHECK(mask[3] == F16_NEG_INF);
    CHECK(mask[8 + 0] == F16_ZERO);
    CHECK(mask[8 + 1] == F16_ZERO);
    CHECK(mask[8 + 2] == F16_ZERO);
    CHECK(mask[8 + 3] == F16_ZERO);
    CHECK(mask[8 + 4] == F16_NEG_INF);
}

void test_step_plan_appends_current_tokens_and_write_rows() {
    LsaPackedPlan history;
    history.committed_tokens = 12;
    history.token_capacity = 8;
    history.source_positions = {0, 2, 5, 9};

    LsaPackedStepPlan step;
    std::vector<uint16_t> mask;
    std::string error;
    CHECK(build_lsa_packed_step_plan(
        history, {12, 13}, 2, 4, step, mask, error, 8));
    CHECK(step.historical_tokens == 4);
    CHECK(step.key_positions ==
          std::vector<int>({0, 2, 5, 9, 12, 13}));
    CHECK(step.write_rows ==
          std::vector<int64_t>({4, 5, 4, 5}));
    CHECK(mask[4] == F16_ZERO);
    CHECK(mask[5] == F16_NEG_INF);
    CHECK(mask[8 + 4] == F16_ZERO);
    CHECK(mask[8 + 5] == F16_ZERO);
    CHECK(mask[8 + 6] == F16_NEG_INF);

    history.token_capacity = 5;
    CHECK(!build_lsa_packed_step_plan(
        history, {12, 13}, 2, 4, step, mask, error, 8));
    CHECK(error.find("exceeds") != std::string::npos);
}

void test_token_axis_gather_preserves_head_layout() {
    constexpr int head_dim = 2;
    constexpr int source_tokens = 5;
    constexpr int heads = 2;
    std::vector<uint16_t> source(head_dim * source_tokens * heads);
    for (int head = 0; head < heads; ++head) {
        for (int token = 0; token < source_tokens; ++token) {
            for (int dim = 0; dim < head_dim; ++dim) {
                const size_t index =
                    static_cast<size_t>(head) * source_tokens * head_dim +
                    static_cast<size_t>(token) * head_dim + dim;
                source[index] =
                    static_cast<uint16_t>(head * 100 + token * 10 + dim);
            }
        }
    }

    LsaPackedPlan plan;
    plan.committed_tokens = source_tokens;
    plan.token_capacity = 4;
    plan.source_positions = {1, 4};

    std::vector<uint8_t> packed_bytes;
    std::string error;
    CHECK(gather_lsa_token_axis(
        source.data(), source.size() * sizeof(uint16_t),
        head_dim, source_tokens, heads, sizeof(uint16_t),
        plan, packed_bytes, error));
    std::vector<uint16_t> packed(packed_bytes.size() / sizeof(uint16_t));
    std::memcpy(packed.data(), packed_bytes.data(), packed_bytes.size());

    CHECK(packed[0] == 10);
    CHECK(packed[1] == 11);
    CHECK(packed[2] == 40);
    CHECK(packed[3] == 41);
    CHECK(packed[4] == 0);
    CHECK(packed[5] == 0);
    const size_t second_head = head_dim * plan.token_capacity;
    CHECK(packed[second_head + 0] == 110);
    CHECK(packed[second_head + 1] == 111);
    CHECK(packed[second_head + 2] == 140);
    CHECK(packed[second_head + 3] == 141);
}

void test_stride_aware_gather_supports_quantized_rows() {
    LsaTokenAxisLayout layout;
    layout.source_tokens = 3;
    layout.heads = 2;
    layout.row_bytes = 3;
    layout.token_stride_bytes = 5;
    layout.head_stride_bytes = 17;
    std::vector<uint8_t> source(37, 0xee);
    for (int head = 0; head < layout.heads; ++head) {
        for (int token = 0; token < layout.source_tokens; ++token) {
            const size_t offset =
                static_cast<size_t>(head) * layout.head_stride_bytes +
                static_cast<size_t>(token) * layout.token_stride_bytes;
            for (size_t byte = 0; byte < layout.row_bytes; ++byte) {
                source[offset + byte] = static_cast<uint8_t>(
                    head * 100 + token * 10 + static_cast<int>(byte));
            }
        }
    }

    LsaPackedPlan plan;
    plan.committed_tokens = 3;
    plan.token_capacity = 3;
    plan.source_positions = {2, 0};
    std::vector<uint8_t> packed;
    std::string error;
    CHECK(gather_lsa_token_rows(
        source.data(), source.size(), layout, plan, packed, error));
    CHECK(packed.size() == 18);
    CHECK(packed[0] == 20);
    CHECK(packed[1] == 21);
    CHECK(packed[2] == 22);
    CHECK(packed[3] == 0);
    CHECK(packed[4] == 1);
    CHECK(packed[5] == 2);
    CHECK(packed[6] == 0);
    const size_t second_head = 9;
    CHECK(packed[second_head + 0] == 120);
    CHECK(packed[second_head + 3] == 100);
    CHECK(packed[second_head + 6] == 0);
}

void test_all_chunks_attention_matches_dense() {
    constexpr int head_dim = 2;
    constexpr int source_tokens = 4;
    constexpr int kv_heads = 2;
    constexpr int q_heads = 4;
    std::vector<float> key(head_dim * source_tokens * kv_heads);
    std::vector<float> value(head_dim * source_tokens * kv_heads);
    for (int head = 0; head < kv_heads; ++head) {
        for (int token = 0; token < source_tokens; ++token) {
            const size_t offset =
                static_cast<size_t>(head) * source_tokens * head_dim +
                static_cast<size_t>(token) * head_dim;
            key[offset] = 0.1f * (head + 1) * (token + 1);
            key[offset + 1] = 0.2f * (token + 1);
            value[offset] = static_cast<float>(head * 10 + token);
            value[offset + 1] = static_cast<float>(head * 10 - token);
        }
    }
    const std::vector<float> query = {
        1.0f, 0.0f,
        0.5f, 0.5f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

    LsaPackedConfig config;
    config.token_capacity = source_tokens;
    config.sink_tokens = 0;
    config.recent_tokens = 0;
    LsaPackedPlan plan;
    std::string error;
    CHECK(build_lsa_packed_plan(
        make_catalog(2, 2), {0, 1}, source_tokens, config, plan, error));

    std::vector<uint8_t> packed_key_bytes;
    std::vector<uint8_t> packed_value_bytes;
    CHECK(gather_lsa_token_axis(
        key.data(), key.size() * sizeof(float), head_dim, source_tokens,
        kv_heads, sizeof(float), plan, packed_key_bytes, error));
    CHECK(gather_lsa_token_axis(
        value.data(), value.size() * sizeof(float), head_dim, source_tokens,
        kv_heads, sizeof(float), plan, packed_value_bytes, error));
    std::vector<float> packed_key(
        packed_key_bytes.size() / sizeof(float));
    std::vector<float> packed_value(
        packed_value_bytes.size() / sizeof(float));
    std::memcpy(packed_key.data(), packed_key_bytes.data(),
                packed_key_bytes.size());
    std::memcpy(packed_value.data(), packed_value_bytes.data(),
                packed_value_bytes.size());

    std::vector<float> dense_output;
    std::vector<float> packed_output;
    CHECK(lsa_reference_dense_attention(
        query, key, value, q_heads, kv_heads, head_dim, source_tokens,
        3, dense_output, error));
    CHECK(lsa_reference_packed_attention(
        query, packed_key, packed_value, q_heads, kv_heads, head_dim,
        plan, 3, packed_output, error));
    CHECK(dense_output.size() == packed_output.size());
    for (size_t i = 0; i < dense_output.size(); ++i) {
        CHECK(std::abs(dense_output[i] - packed_output[i]) < 1e-6f);
    }
}

}  // namespace

int main() {
    test_plan_combines_retrieval_sink_and_recent();
    test_all_chunks_matches_dense_order();
    test_original_position_causal_mask();
    test_step_plan_appends_current_tokens_and_write_rows();
    test_token_axis_gather_preserves_head_layout();
    test_stride_aware_gather_supports_quantized_rows();
    test_all_chunks_attention_matches_dense();
    if (failures != 0) {
        std::fprintf(stderr, "%d packed KV test(s) failed\n", failures);
        return 1;
    }
    std::printf("LSA packed KV tests passed\n");
    return 0;
}
