// Exact attention-mass scorer and R-LRU block picker (pure CPU).

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_exact_mass.h"
#include "kvflash_prefill_reselect.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <random>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashExactMassFixture {};

// Naive double reference over float keys K[c][l][hk][t][j].
std::vector<double> naive(const std::vector<float> & K, int C, int ct, const KvFlashQkDims & d,
                          const std::vector<float> & q) {
    const int group = d.n_q_heads / d.n_kv_heads;
    std::vector<double> out((size_t)C, 0.0);
    for (int l = 0; l < d.n_layers; l++)
        for (int hq = 0; hq < d.n_q_heads; hq++) {
            std::vector<double> s;
            for (int c = 0; c < C; c++)
                for (int t = 0; t < ct; t++) {
                    const float * k = &K[((((size_t)c * d.n_layers + l) * d.n_kv_heads + hq / group) * ct + t) * d.head_dim];
                    const float * qq = &q[((size_t)l * d.n_q_heads + hq) * d.head_dim];
                    double dot = 0;
                    for (int j = 0; j < d.head_dim; j++) dot += (double)qq[j] * k[j];
                    s.push_back(dot / std::sqrt((double)d.head_dim));
                }
            double mx = *std::max_element(s.begin(), s.end()), Z = 0;
            for (double v : s) Z += std::exp(v - mx);
            for (int c = 0; c < C; c++) {
                double m = 0;
                for (int t = 0; t < ct; t++) m += std::exp(s[(size_t)c * ct + t] - mx);
                out[(size_t)c] += m / Z / (d.n_layers * d.n_q_heads);
            }
        }
    return out;
}

// Returns false on a setup failure (REQUIRE cannot be used in a helper).
bool run_case(ggml_type type, double tol, std::vector<float> & out, std::vector<double> & ref) {
    KvFlashQkDims d; d.n_layers = 2; d.n_q_heads = 4; d.n_kv_heads = 2; d.head_dim = 32;
    const int C = 5, ct = 8;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> K((size_t)C * d.n_layers * d.n_kv_heads * ct * d.head_dim);
    for (auto & v : K) v = nd(rng);
    std::vector<float> q((size_t)d.n_layers * d.n_q_heads * d.head_dim);
    for (auto & v : q) v = nd(rng);
    std::vector<uint8_t> buf;
    const size_t nel = K.size();
    if (type == GGML_TYPE_F32) {
        buf.resize(nel * 4);
        std::memcpy(buf.data(), K.data(), nel * 4);
    } else {
        buf.resize(nel * 2);
        ggml_fp32_to_fp16_row(K.data(), (ggml_fp16_t *)buf.data(), (int64_t)nel);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)buf.data(), K.data(), (int64_t)nel);   // reference sees the rounded keys
    }
    const size_t esz = type == GGML_TYPE_F32 ? 4 : 2;
    KvFlashExactKeys keys;
    keys.n_chunks = C; keys.chunk_tokens = ct; keys.type = type;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * {
        return buf.data() + ((((size_t)c * d.n_layers + l) * d.n_kv_heads + h) * ct * d.head_dim) * esz;
    };
    (void)tol;
    if (!kvflash_exact_mass_scores(keys, d, q.data(), out, 3)) return false;
    ref = naive(K, C, ct, d, q);
    return true;
}

}

TEST_CASE(KvflashExactMassFixture, matches_naive_reference_f32_gqa) {
    std::vector<float> out; std::vector<double> ref;
    REQUIRE(run_case(GGML_TYPE_F32, 1e-5, out, ref));
    double sum = 0;
    for (size_t c = 0; c < ref.size(); c++) { CHECK(std::fabs(out[c] - ref[c]) < 1e-5); sum += out[c]; }
    CHECK(std::fabs(sum - 1.0) < 1e-4);   // each head's masses sum to 1
}
TEST_CASE(KvflashExactMassFixture, matches_naive_reference_f16_gqa) {
    std::vector<float> out; std::vector<double> ref;
    REQUIRE(run_case(GGML_TYPE_F16, 1e-5, out, ref));
    double sum = 0;
    for (size_t c = 0; c < ref.size(); c++) { CHECK(std::fabs(out[c] - ref[c]) < 1e-5); sum += out[c]; }
    CHECK(std::fabs(sum - 1.0) < 1e-4);
}

TEST_CASE(KvflashExactMassFixture, absent_chunk_scores_zero_and_rest_renormalize) {
    KvFlashQkDims d; d.n_layers = 1; d.n_q_heads = 2; d.n_kv_heads = 1; d.head_dim = 4;
    std::vector<float> k(2 * 4, 0.5f), q(2 * 4, 1.0f);
    KvFlashExactKeys keys;
    keys.n_chunks = 3; keys.chunk_tokens = 2; keys.type = GGML_TYPE_F32;
    keys.seg = [&](int c, int, int) -> const uint8_t * { return c == 1 ? nullptr : (const uint8_t *)k.data(); };
    std::vector<float> out;
    REQUIRE(kvflash_exact_mass_scores(keys, d, q.data(), out, 1));
    CHECK(out[1] == 0.0f);
    CHECK(std::fabs(out[0] - 0.5f) < 1e-6);
    CHECK(std::fabs(out[2] - 0.5f) < 1e-6);
}

TEST_CASE(KvflashExactMassFixture, multi_query_equals_single) {
    KvFlashQkDims d; d.n_layers = 1; d.n_q_heads = 2; d.n_kv_heads = 1; d.head_dim = 8;
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> k(4 * 3 * 8), q1(16), q2(16);
    for (auto & v : k) v = nd(rng);
    for (auto & v : q1) v = nd(rng);
    for (auto & v : q2) v = nd(rng);
    KvFlashExactKeys keys;
    keys.n_chunks = 4; keys.chunk_tokens = 3; keys.type = GGML_TYPE_F32;
    keys.seg = [&](int c, int, int) -> const uint8_t * { return (const uint8_t *)(k.data() + (size_t)c * 24); };
    std::vector<std::vector<float>> o;
    REQUIRE(kvflash_exact_mass_scores_multi(keys, d, {q1.data(), q2.data()}, o, 2));
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(keys, d, q1.data(), a, 1));
    REQUIRE(kvflash_exact_mass_scores(keys, d, q2.data(), b, 1));
    for (int c = 0; c < 4; c++) {
        CHECK(std::fabs(o[0][(size_t)c] - a[(size_t)c]) < 1e-6);
        CHECK(std::fabs(o[1][(size_t)c] - b[(size_t)c]) < 1e-6);
    }
}

// Microbenchmark, off by default: LUCE_EXACT_BENCH=1. 36K tokens, 16 layers,
// 24 q heads, 4 kv heads, 256 dim, f16 (one chunk's data reused for all).
TEST_CASE(KvflashExactMassFixture, bench_36k_tokens_f16) {
    if (!std::getenv("LUCE_EXACT_BENCH")) return;
    KvFlashQkDims d; d.n_layers = 16; d.n_q_heads = 24; d.n_kv_heads = 4; d.head_dim = 256;
    const int ct = 64, C = 576;
    std::mt19937 rng(1); std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nel = (size_t)d.n_layers * d.n_kv_heads * ct * d.head_dim;
    std::vector<float> f(nel), q((size_t)d.n_layers * d.n_q_heads * d.head_dim);
    for (auto & v : f) v = nd(rng);
    for (auto & v : q) v = nd(rng);
    std::vector<ggml_fp16_t> buf(nel);
    ggml_fp32_to_fp16_row(f.data(), buf.data(), (int64_t)nel);
    KvFlashExactKeys keys;
    keys.n_chunks = C; keys.chunk_tokens = ct; keys.type = GGML_TYPE_F16;
    const size_t seg_elems = (size_t)ct * d.head_dim;
    keys.seg = [&](int, int l, int h) -> const uint8_t * {
        return (const uint8_t *)(buf.data() + ((size_t)l * d.n_kv_heads + h) * seg_elems);
    };
    std::vector<float> out;
    const auto t0 = std::chrono::steady_clock::now();
    REQUIRE(kvflash_exact_mass_scores(keys, d, q.data(), out));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[bench] exact mass 36864 tokens x16L x24H x256D f16: %.1f ms (%u hw threads)\n",
                ms, std::thread::hardware_concurrency());
}

TEST_CASE(KvflashExactMassFixture, pick_k_zero_and_no_eligible) {
    std::vector<float> s(30, 1.0f);
    CHECK(kvflash_recall_pick(s, 30, 0).empty());
    CHECK(kvflash_recall_pick(s, 10, 3).empty());      // only block 0 below the window: sink block excluded
    CHECK(kvflash_recall_pick({}, 30, 3).empty());
}
TEST_CASE(KvflashExactMassFixture, pick_fewer_eligible_than_k) {
    std::vector<float> s(40, 1.0f);
    CHECK((kvflash_recall_pick(s, 40, 5) == std::vector<int>{3, 2, 1}));   // ties: newer first
}
TEST_CASE(KvflashExactMassFixture, pick_partial_block_at_window_edge_is_ineligible) {
    std::vector<float> s(40, 1.0f);
    s[35] = 100.0f;   // block 3 straddles the window edge (first_window=35)
    CHECK((kvflash_recall_pick(s, 35, 5) == std::vector<int>{2, 1}));
}
TEST_CASE(KvflashExactMassFixture, pick_ranks_by_block_sum_and_excludes_sink) {
    std::vector<float> s(50, 0.0f);
    s[0] = 1000.0f;                 // sink chunk: block 0 never eligible
    s[15] = 5.0f; s[42] = 3.0f; s[27] = 3.0f;
    CHECK((kvflash_recall_pick(s, 50, 2) == std::vector<int>{1, 4}));      // 5 > 3 == 3 (newer 4 beats 2)
    CHECK((kvflash_recall_pick(s, 50, 1) == std::vector<int>{1}));
}

TEST_CASE(KvflashExactMassFixture, dump_line_has_fields_and_4_significant_digits) {
    const std::string s = kvflash_exact_dump_line(1000, 16, 995, {0.123456f, 0.5f}, {0.25f, 0.000123456f});
    CHECK(s.find("\"prompt_len\":1000") != std::string::npos);
    CHECK(s.find("\"kv_offset\":16") != std::string::npos);
    CHECK(s.find("\"r\":995") != std::string::npos);
    CHECK(s.find("\"n_chunks\":2") != std::string::npos);
    CHECK(s.find("\"mass_r\":[0.1235,0.5]") != std::string::npos);
    CHECK(s.find("\"mass_end\":[0.25,0.0001235]") != std::string::npos);
    CHECK(s.front() == '{' && s.back() == '}');
}
