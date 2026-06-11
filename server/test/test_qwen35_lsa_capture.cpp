#include "qwen35/qwen35_lsa_capture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <limits>
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

void test_default_capture_contract() {
    Qwen35LsaModelShape shape;
    Qwen35LsaCaptureConfig config = qwen35_lsa_default_capture_config();
    QwenGraphInputs inputs{};
    std::string error;
    CHECK(config.hidden_layer == 46);
    CHECK(config.qk_layers.size() == 16);
    CHECK(config.qk_layers.front() == 3);
    CHECK(config.qk_layers.back() == 63);
    CHECK(configure_qwen35_lsa_capture(shape, config, inputs, error));
    CHECK(inputs.lsa_hidden_capture_layer == 46);
    for (int layer = 0; layer < 64; ++layer) {
        const bool selected =
            (inputs.lsa_qk_capture_mask & (uint64_t{1} << layer)) != 0;
        CHECK(selected == (((layer + 1) % 4) == 0));
    }

    config.qk_layers = {47, 47};
    CHECK(!configure_qwen35_lsa_capture(shape, config, inputs, error));
    CHECK(error.find("duplicated") != std::string::npos);
    config.qk_layers = {46};
    CHECK(!configure_qwen35_lsa_capture(shape, config, inputs, error));
    CHECK(error.find("full-attention") != std::string::npos);

    config = qwen35_lsa_capture_config({32, 4}, 22);
    CHECK(config.hidden_layer == 22);
    CHECK(config.qk_layers.size() == 8);
    CHECK(config.qk_layers.front() == 3);
    CHECK(config.qk_layers.back() == 31);
}

void test_block_key_pooling() {
    const std::vector<float> keys = {
        1.0f, 0.0f, 0.0f, 2.0f,
        3.0f, 0.0f, 0.0f, 4.0f,
    };
    std::vector<float> pooled;
    std::string error;
    CHECK(pool_qwen35_lsa_block_keys(keys, 2, 2, 2, pooled, error));
    CHECK(pooled.size() == 4);
    CHECK(std::abs(pooled[0] - 1.0f) < 1e-6f);
    CHECK(std::abs(pooled[1]) < 1e-6f);
    CHECK(std::abs(pooled[2]) < 1e-6f);
    CHECK(std::abs(pooled[3] - 1.0f) < 1e-6f);

    std::vector<float> bad = keys;
    bad[0] = std::numeric_limits<float>::infinity();
    CHECK(!pool_qwen35_lsa_block_keys(bad, 2, 2, 2, pooled, error));
    CHECK(error.find("non-finite") != std::string::npos);
}

}  // namespace

int main() {
    test_default_capture_contract();
    test_block_key_pooling();
    if (failures != 0) {
        std::fprintf(stderr, "%d Qwen LSA capture test(s) failed\n", failures);
        return 1;
    }
    std::printf("Qwen LSA capture tests passed\n");
    return 0;
}
