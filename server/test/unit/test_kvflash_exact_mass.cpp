// Exact attention-mass scorer and R-LRU block picker (pure CPU).

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_exact_mass.h"
#include "kvflash_exact_fast.h"
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

// ---- fast scorer vs the double-precision reference -------------------------
namespace {
struct FastCase {
    KvFlashQkDims d; int C = 0, ct = 0; ggml_type type = GGML_TYPE_Q4_0;
    std::vector<uint8_t> buf; std::vector<float> q; size_t seg_bytes = 0;
    KvFlashExactKeys keys() const {
        KvFlashExactKeys k; k.n_chunks = C; k.chunk_tokens = ct; k.type = type;
        k.seg = [this](int c, int l, int h) -> const uint8_t * {
            return buf.data() + (((size_t)c * d.n_layers + l) * d.n_kv_heads + h) * seg_bytes;
        };
        return k;
    }
};
FastCase make_case(ggml_type type, int n_layers, int hq, int hkv, int D, int C, int ct, unsigned seed) {
    FastCase f; f.type = type; f.C = C; f.ct = ct;
    f.d.n_layers = n_layers; f.d.n_q_heads = hq; f.d.n_kv_heads = hkv; f.d.head_dim = D;
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nseg = (size_t)C * n_layers * hkv;
    f.seg_bytes = ggml_row_size(type, D) * (size_t)ct;
    std::vector<float> src((size_t)ct * D);
    f.buf.resize(nseg * f.seg_bytes);
    const auto * tt = ggml_get_type_traits(type);
    for (size_t s = 0; s < nseg; s++) {
        for (auto & v : src) v = nd(rng);
        uint8_t * dst = f.buf.data() + s * f.seg_bytes;
        if (type == GGML_TYPE_F32) std::memcpy(dst, src.data(), f.seg_bytes);
        else for (int t = 0; t < ct; t++) tt->from_float_ref(&src[(size_t)t * D], dst + (size_t)t * ggml_row_size(type, D), D);
    }
    f.q.resize((size_t)n_layers * hq * D);
    for (auto & v : f.q) v = nd(rng);
    return f;
}
// Max relative per-chunk and per-block difference of fast vs reference; false on failure.
bool compare_fast(const FastCase & f, int cpb, double & chunk_rel, double & block_rel,
                  std::vector<int> & pick_ref, std::vector<int> & pick_fast, int k) {
    std::vector<float> a, b;
    const auto keys = f.keys();
    if (!kvflash_exact_mass_scores(keys, f.d, f.q.data(), a, 3)) return false;
    if (!kvflash_exact_mass_scores_fast(keys, f.d, f.q.data(), b, 3)) return false;
    chunk_rel = 0;
    for (size_t c = 0; c < a.size(); c++) chunk_rel = std::max(chunk_rel, std::fabs((double)a[c] - b[c]) / std::fabs((double)a[c]));
    const auto ba = kvflash_block_scores(a, cpb), bb = kvflash_block_scores(b, cpb);
    block_rel = 0;
    for (size_t i = 0; i < ba.size(); i++) block_rel = std::max(block_rel, std::fabs(ba[i] - bb[i]) / std::fabs(ba[i]));
    pick_ref = kvflash_recall_pick(a, f.C, k, cpb);
    pick_fast = kvflash_recall_pick(b, f.C, k, cpb);
    return true;
}
}

TEST_CASE(KvflashExactMassFixture, fast_matches_reference_q4_0_gqa_6) {
    for (bool scalar : {false, true}) {
        fastmass::force_scalar() = scalar;
        // qwen35-27B shape scaled down: group 6, D multiple of 32, 5 blocks of 4 chunks
        const auto f = make_case(GGML_TYPE_Q4_0, 3, 12, 2, 64, 20, 16, 11);
        double cr, br; std::vector<int> pr, pf;
        REQUIRE(compare_fast(f, 4, cr, br, pr, pf, 2));
        CHECK(cr < 1e-4);
        CHECK(br < 1e-4);
        CHECK((pr == pf));
    }
    fastmass::force_scalar() = false;
}
TEST_CASE(KvflashExactMassFixture, fast_matches_reference_other_types_and_groups) {
    struct Cfg { ggml_type t; int hq, hkv; };
    for (const Cfg c : {Cfg{GGML_TYPE_F16, 8, 2}, Cfg{GGML_TYPE_F32, 3, 1}, Cfg{GGML_TYPE_Q8_0, 14, 2}, Cfg{GGML_TYPE_Q4_0, 7, 1}, Cfg{GGML_TYPE_Q4_0, 2, 2}}) {
        const auto f = make_case(c.t, 2, c.hq, c.hkv, 64, 12, 8, 5);   // group 4, 3, 7, 7, 1 (7 > the 6-wide tile)
        double cr, br; std::vector<int> pr, pf;
        REQUIRE(compare_fast(f, 3, cr, br, pr, pf, 2));
        CHECK(cr < 1e-4);
        CHECK(br < 1e-4);
    }
}
TEST_CASE(KvflashExactMassFixture, fast_absent_chunk_scores_zero) {
    auto f = make_case(GGML_TYPE_Q4_0, 1, 6, 1, 32, 6, 8, 2);
    auto keys = f.keys();
    const auto seg = keys.seg;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return c == 2 ? nullptr : seg(c, l, h); };
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(keys, f.d, f.q.data(), a, 2));
    REQUIRE(kvflash_exact_mass_scores_fast(keys, f.d, f.q.data(), b, 2));
    CHECK(b[2] == 0.0f);
    for (int c = 0; c < 6; c++) CHECK(std::fabs(a[(size_t)c] - b[(size_t)c]) <= 1e-4 * std::max(a[(size_t)c], 1e-6f));
}
// Picks must agree whenever the k-th/(k+1)-th block gap exceeds the numeric noise (1e-4 relative).
TEST_CASE(KvflashExactMassFixture, fast_picks_identical_without_near_ties) {
    int checked = 0;
    for (unsigned seed = 1; seed <= 12; seed++) {
        const auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 40, 16, seed);
        std::vector<float> a, b;
        REQUIRE(kvflash_exact_mass_scores(f.keys(), f.d, f.q.data(), a, 3));
        REQUIRE(kvflash_exact_mass_scores_fast(f.keys(), f.d, f.q.data(), b, 3));
        auto bs = kvflash_block_scores(a, 4);
        std::vector<double> sorted(bs.begin() + 1, bs.end());   // block 0 is the sink, never eligible
        std::sort(sorted.rbegin(), sorted.rend());
        for (int k : {1, 2, 5}) {
            if ((sorted[(size_t)k - 1] - sorted[(size_t)k]) / sorted[(size_t)k - 1] < 1e-4) continue;   // near-tie: see next test
            CHECK((kvflash_recall_pick(a, 40, k, 4) == kvflash_recall_pick(b, 40, k, 4)));
            checked++;
        }
    }
    CHECK(checked > 20);
}
// Documented near-tie behaviour. (1) Bit-identical blocks tie exactly in BOTH
// scorers (same ops on the same bytes), so the newer-first tie-break is kept.
// (2) Blocks whose scores differ by less than the float noise (~1e-6 relative)
// may legitimately rank either way between the two scorers; that is the only
// way picks can differ, and the GPU shadow run logs every such gap.
TEST_CASE(KvflashExactMassFixture, fast_exact_ties_keep_newer_first_and_near_ties_are_bounded) {
    auto f = make_case(GGML_TYPE_Q4_0, 2, 12, 2, 64, 40, 16, 3);
    const size_t per_chunk = (size_t)f.d.n_layers * f.d.n_kv_heads * f.seg_bytes;
    // blocks 2 and 6 (chunks 8-11 and 24-27) get identical key bytes
    for (int i = 0; i < 4; i++) std::memcpy(&f.buf[(size_t)(24 + i) * per_chunk], &f.buf[(size_t)(8 + i) * per_chunk], per_chunk);
    std::vector<float> a, b;
    REQUIRE(kvflash_exact_mass_scores(f.keys(), f.d, f.q.data(), a, 3));
    REQUIRE(kvflash_exact_mass_scores_fast(f.keys(), f.d, f.q.data(), b, 3));
    const auto ba = kvflash_block_scores(a, 4), bb = kvflash_block_scores(b, 4);
    // identical K gives identical logsumexp per chunk, but the softmax is over all chunks, so block 2 == block 6 in both
    CHECK(ba[2] == ba[6]);
    CHECK(bb[2] == bb[6]);
    // gap between the two scorers on any block stays far below any realistic rank gap
    for (size_t i = 0; i < ba.size(); i++) CHECK(std::fabs(ba[i] - bb[i]) / ba[i] < 1e-4);
}
TEST_CASE(KvflashExactMassFixture, fast_bench_36k_q4_0) {
    if (!std::getenv("LUCE_EXACT_BENCH")) return;
    const auto f = make_case(GGML_TYPE_Q4_0, 16, 24, 4, 256, 64, 64, 1);   // 64 chunks, replicated
    KvFlashExactKeys keys = f.keys();
    const auto seg = keys.seg;
    keys.n_chunks = 576;
    keys.seg = [&](int c, int l, int h) { return seg(c % 64, l, h); };
    std::vector<float> a, b;
    auto t0 = std::chrono::steady_clock::now();
    REQUIRE(kvflash_exact_mass_scores(keys, f.d, f.q.data(), a));
    auto t1 = std::chrono::steady_clock::now();
    REQUIRE(kvflash_exact_mass_scores_fast(keys, f.d, f.q.data(), b));
    auto t2 = std::chrono::steady_clock::now();
    std::printf("[bench] q4_0 36864 tokens: reference %.1f ms, fast %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
}
