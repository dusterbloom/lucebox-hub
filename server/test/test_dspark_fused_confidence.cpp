#include "common/dspark_head.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using dflash::common::dspark_markov_correct_greedy_chain_fused;
using dflash::common::DraftWeights;

constexpr int kHidden = 2;
constexpr int kRank = 2;
constexpr int kVocab = 4;
constexpr int kQLen = 4;

const std::vector<float> kLmHead = {
     1.0f,  0.0f,
     0.0f,  1.0f,
    -1.0f,  0.0f,
     0.0f, -1.0f,
};

const std::vector<float> kMarkovW1 = {
     0.25f, -0.50f,
     1.00f,  0.50f,
    -0.50f,  0.75f,
     0.10f,  0.20f,
};

const std::vector<float> kMarkovW2(kRank * kVocab, 0.0f);

const std::vector<float> kHiddenRows = {
     8.0f,  8.0f,  // seed row: intentionally ignored by the fused head
     2.0f,  0.1f,  // token 0
     0.1f,  3.0f,  // token 1
    -4.0f,  0.1f,  // token 2
};

int check(bool condition, const char * expression, int line) {
    if (condition) return 0;
    std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
    return 1;
}

#define CHECK(expr) do { if (check((expr), #expr, __LINE__)) return 1; } while (false)

struct Fixture {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    DraftWeights weights{};
    ggml_tensor * lm_head = nullptr;
    std::vector<float> confidence_w;
    float confidence_b = -0.15f;

    Fixture(int confidence_dim, bool install_confidence) {
        backend = ggml_backend_cpu_init();

        ggml_init_params params{};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);

        lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, kVocab);
        weights.dspark.markov_w1 =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kRank, kVocab);
        weights.dspark.markov_w2 =
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kRank, kVocab);
        if (install_confidence) {
            weights.dspark.confidence_w =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, confidence_dim, 1);
            weights.dspark.confidence_b =
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        }

        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(lm_head, kLmHead.data(), 0,
                                kLmHead.size() * sizeof(float));
        ggml_backend_tensor_set(weights.dspark.markov_w1, kMarkovW1.data(), 0,
                                kMarkovW1.size() * sizeof(float));
        ggml_backend_tensor_set(weights.dspark.markov_w2, kMarkovW2.data(), 0,
                                kMarkovW2.size() * sizeof(float));

        weights.n_embd = kHidden;
        weights.dspark.enabled = true;
        weights.dspark.markov_rank = kRank;
        weights.dspark.vocab_size = kVocab;
        weights.dspark.confidence_dim = install_confidence ? confidence_dim : 0;

        if (install_confidence) {
            confidence_w.resize((size_t)confidence_dim);
            for (int i = 0; i < confidence_dim; ++i) {
                confidence_w[(size_t)i] = 0.2f - 0.1f * (float)i;
            }
            ggml_backend_tensor_set(weights.dspark.confidence_w,
                                    confidence_w.data(), 0,
                                    confidence_w.size() * sizeof(float));
            ggml_backend_tensor_set(weights.dspark.confidence_b,
                                    &confidence_b, 0, sizeof(confidence_b));
        }
    }

    ~Fixture() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }

    Fixture(const Fixture &) = delete;
    Fixture & operator=(const Fixture &) = delete;
};

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<float> reference_confidence(const Fixture & fixture,
                                        const std::vector<int32_t> & tokens) {
    std::vector<float> result;
    for (int i = 0; i < kQLen - 1; ++i) {
        const float * hidden = kHiddenRows.data() + (size_t)(i + 1) * kHidden;
        float logit = fixture.confidence_b;
        for (int j = 0; j < kHidden; ++j) {
            logit += hidden[j] * fixture.confidence_w[(size_t)j];
        }
        if (fixture.weights.dspark.confidence_dim == kHidden + kRank) {
            const int32_t previous_token = tokens[(size_t)i];
            const float * previous_embedding =
                kMarkovW1.data() + (size_t)previous_token * kRank;
            for (int j = 0; j < kRank; ++j) {
                logit += previous_embedding[j] *
                         fixture.confidence_w[(size_t)kHidden + (size_t)j];
            }
        }
        result.push_back(sigmoid(logit));
    }
    return result;
}

int test_confidence_shape(int confidence_dim) {
    Fixture fixture(confidence_dim, true);
    CHECK(fixture.backend != nullptr);
    CHECK(fixture.ctx != nullptr);
    CHECK(fixture.buffer != nullptr);

    std::vector<int32_t> legacy_tokens;
    CHECK(dspark_markov_correct_greedy_chain_fused(
        fixture.weights, fixture.backend, fixture.lm_head, kHiddenRows.data(),
        kQLen, /*last_tok=*/3, legacy_tokens));

    std::vector<int32_t> confidence_tokens;
    std::vector<float> confidence;
    CHECK(dspark_markov_correct_greedy_chain_fused(
        fixture.weights, fixture.backend, fixture.lm_head, kHiddenRows.data(),
        kQLen, /*last_tok=*/3, confidence_tokens, &confidence));

    CHECK(legacy_tokens == std::vector<int32_t>({3, 0, 1, 2}));
    CHECK(confidence_tokens == legacy_tokens);
    CHECK(confidence.size() == (size_t)kQLen - 1);

    const std::vector<float> expected =
        reference_confidence(fixture, confidence_tokens);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(std::abs(confidence[i] - expected[i]) < 1e-5f);
    }
    return 0;
}

int test_missing_or_invalid_head() {
    Fixture no_head(/*confidence_dim=*/0, /*install_confidence=*/false);
    std::vector<int32_t> baseline;
    CHECK(dspark_markov_correct_greedy_chain_fused(
        no_head.weights, no_head.backend, no_head.lm_head, kHiddenRows.data(),
        kQLen, /*last_tok=*/3, baseline));

    std::vector<int32_t> tokens;
    std::vector<float> confidence = {42.0f};
    CHECK(dspark_markov_correct_greedy_chain_fused(
        no_head.weights, no_head.backend, no_head.lm_head, kHiddenRows.data(),
        kQLen, /*last_tok=*/3, tokens, &confidence));
    CHECK(tokens == baseline);
    CHECK(confidence.empty());

    Fixture invalid_head(/*confidence_dim=*/1, /*install_confidence=*/true);
    confidence = {42.0f};
    CHECK(dspark_markov_correct_greedy_chain_fused(
        invalid_head.weights, invalid_head.backend, invalid_head.lm_head,
        kHiddenRows.data(), kQLen, /*last_tok=*/3, tokens, &confidence));
    CHECK(tokens == baseline);
    CHECK(confidence.empty());
    return 0;
}

}  // namespace

int main() {
    if (test_confidence_shape(kHidden) != 0) return 1;
    if (test_confidence_shape(kHidden + kRank) != 0) return 1;
    if (test_missing_or_invalid_head() != 0) return 1;
    return 0;
}
