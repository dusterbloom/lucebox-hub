// KVFlash 2.0b M7: automatic AM compaction at the seal of a 640-token block.
// Model-free. Covers
//   1. the seal-trigger policy (which blocks compact, with P and W; pins and the exact tail exempt);
//   2. the option parser and "off unless --kvflash-am-ratio is set";
//   3. the capture-ring budget;
//   4. the row helpers (ascending order, gather);
//   5. the whole per-block CPU path (M4 selection, then the M5 fit) on the reference fixtures
//      (LUCE_AM_FIXTURES, e.g. /mnt/kimi-k3/build/am_fixtures_v2: traj1_12k, article_r1e).
//
// The fixture gate holds the end-to-end path to the bars of the milestones it composes:
//   M4: per KV head at most kRefMaxSwaps of 64 kept keys differ from the reference selection
//       and overall overlap >= kRefOverlap (98.5%, measured 98.8-99.6%);
//   M5: the output error of the engine's (beta, C2) on the engine's own selection is within
//       kObjSlack of the reference's output error on the reference selection (the M5 gate binds on
//       objectives, not elementwise, because cond(A) makes C2 ill-posed elementwise).
#include "CppUnitTestFramework.hpp"
#include "common/am_compact.h"
#include "am_fixture.h"
#include "common/kvflash_pager.h"
#include "server/http_server.h"
#include "ggml-alloc.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace luce::am;

namespace {

struct AmCompactFixture {};

AutoConfig on(int p = 0, int w = 0) {
    AutoConfig c;
    c.ratio = 0.10;
    c.prefix_blocks = p;
    c.recent_blocks = w;
    return c;
}

SealInfo seal(int block, int prompt_len, int tail = 0, bool pinned = false) {
    SealInfo s;
    s.block = block;
    s.prompt_len = prompt_len;
    s.exact_tail_tokens = tail;
    s.pinned = pinned;
    return s;
}

// Which of the sealed blocks of a prompt compact.
std::vector<int> compacting(const AutoConfig & c, int prompt_len, int tail = 0, std::set<int> pins = {}) {
    std::vector<int> out;
    for (int b = 0; b < prompt_len / kAutoBlockTokens; b++)
        if (should_compact(c, seal(b, prompt_len, tail, pins.count(b) > 0))) out.push_back(b);
    return out;
}

std::vector<int> range(int lo, int hi) {
    std::vector<int> v;
    for (int i = lo; i < hi; i++) v.push_back(i);
    return v;
}

}  // namespace

// ---- 1. seal-trigger policy -------------------------------------------------------------

TEST_CASE(AmCompactFixture, nothing_compacts_unless_a_ratio_is_set) {
    AutoConfig off;
    CHECK(!off.enabled());
    CHECK(compacting(off, 20000).empty());
}

TEST_CASE(AmCompactFixture, with_0_and_0_every_sealed_block_compacts) {
    // 6651 tokens = 10 sealed blocks and a 251-token open tail (article_r1e's shape)
    CHECK(compacting(on(), 6651) == range(0, 10));
}

TEST_CASE(AmCompactFixture, the_exact_prefix_keeps_the_first_P_blocks) {
    CHECK(compacting(on(3, 0), 6651) == range(3, 10));
    CHECK(compacting(on(10, 0), 6651).empty());
    CHECK(compacting(on(50, 0), 6651).empty());
}

TEST_CASE(AmCompactFixture, the_recent_window_keeps_the_last_W_sealed_blocks_of_the_prompt) {
    CHECK(compacting(on(0, 2), 6651) == range(0, 8));
    CHECK(compacting(on(0, 10), 6651).empty());
    CHECK(compacting(on(0, 99), 6651).empty());
}

TEST_CASE(AmCompactFixture, prefix_and_window_combine) {
    CHECK(compacting(on(2, 3), 6651) == range(2, 7));
    CHECK(compacting(on(5, 6), 6651).empty());   // they overlap: nothing left
}

TEST_CASE(AmCompactFixture, pinned_blocks_never_compact) {
    CHECK(compacting(on(), 6651, 0, {0, 4}) == std::vector<int>({1, 2, 3, 5, 6, 7, 8, 9}));
}

TEST_CASE(AmCompactFixture, a_block_touching_the_exact_tail_stays_exact) {
    // 7100 tokens: 11 sealed blocks, block 10 ends at 7040, 60 tokens before the end of the prompt
    CHECK(compacting(on(), 7100, 0) == range(0, 11));
    CHECK(compacting(on(), 7100, 60) == range(0, 11));    // the tail starts at 7040: block 10 ends exactly there
    CHECK(compacting(on(), 7100, 61) == range(0, 10));    // one more tail token reaches into block 10
    CHECK(compacting(on(), 7100, 700) == range(0, 10));   // the tail starts at 6400: block 9 ends exactly there
    CHECK(compacting(on(), 7100, 701) == range(0, 9));
}

TEST_CASE(AmCompactFixture, the_tail_longer_than_the_prompt_exempts_every_block) {
    CHECK(compacting(on(), 7100, 7100).empty());
    CHECK(compacting(on(), 7100, 99999).empty());
}

TEST_CASE(AmCompactFixture, a_block_that_is_not_complete_in_the_prompt_never_compacts) {
    CHECK(!should_compact(on(), seal(10, 6651)));   // block 10 would end at 7040 > 6651
    CHECK(!should_compact(on(), seal(-1, 6651)));
}

// ---- 2. options -------------------------------------------------------------------------------

TEST_CASE(AmCompactFixture, no_options_means_off_and_no_error) {
    std::string err;
    AutoConfig c = auto_config_from_strings(nullptr, nullptr, nullptr, &err);
    CHECK(!c.enabled());
    CHECK(err.empty());
    c = auto_config_from_strings("", "", "", &err);
    CHECK(!c.enabled());
    CHECK(err.empty());
    c = auto_config_from_strings("0", nullptr, nullptr, &err);
    CHECK(!c.enabled());
    CHECK(err.empty());
}

TEST_CASE(AmCompactFixture, ratio_010_with_P_and_W_parses) {
    std::string err;
    const AutoConfig c = auto_config_from_strings("0.10", "2", "3", &err);
    CHECK(err.empty());
    CHECK(c.enabled());
    CHECK(c.prefix_blocks == 2);
    CHECK(c.recent_blocks == 3);
    const AutoConfig d = auto_config_from_strings("0.1", nullptr, nullptr, &err);
    CHECK(d.enabled());
    CHECK(d.prefix_blocks == 0);
    CHECK(d.recent_blocks == 0);
}

TEST_CASE(AmCompactFixture, unsupported_or_malformed_options_are_rejected_and_leave_it_off) {
    for (const char * bad : { "0.05", "0.25", "1", "-0.1", "abc", "0.10x", "nan" }) {
        std::string err;
        const AutoConfig c = auto_config_from_strings(bad, nullptr, nullptr, &err);
        CHECK(!c.enabled());
        CHECK(!err.empty());
    }
    std::string err;
    CHECK(!auto_config_from_strings("0.10", "-1", nullptr, &err).enabled());
    CHECK(!err.empty());
    err.clear();
    CHECK(!auto_config_from_strings("0.10", nullptr, "x", &err).enabled());
    CHECK(!err.empty());
}

TEST_CASE(AmCompactFixture, the_automatic_path_needs_the_pager_and_a_supported_architecture) {
    CHECK(auto_active(on(), true, true));
    CHECK(!auto_active(on(), false, true));
    CHECK(!auto_active(on(), true, false));
    CHECK(!auto_active(AutoConfig{}, true, true));
}

// ---- 3. budget --------------------------------------------------------------------------------

TEST_CASE(AmCompactFixture, the_reserve_covers_the_ring_and_one_layer_of_f32_block_scratch) {
    const int64_t ring = 2 * 7864320LL * 16;   // two planes of 640 x 24 x 256 x 2 B, 16 layers = 240 MiB
    const int64_t r = auto_reserve_bytes(ring, 256, 4);
    const int64_t block_f32 = 640LL * 256 * 4 * 4;   // K or V of one layer
    CHECK(r >= ring + 2 * block_f32);
    CHECK(r <= ring + 64LL * 1024 * 1024);
    CHECK(auto_reserve_bytes(0, 256, 4) > 0);
    CHECK(auto_reserve_bytes(ring, 256, 4) > auto_reserve_bytes(ring / 2, 256, 4));
}

// ---- 4. row helpers -----------------------------------------------------------------------------

TEST_CASE(AmCompactFixture, sort_sel_ascending_orders_every_head) {
    std::vector<int32_t> sel = { 5, 1, 3, 9, 2, 7, 0, 4 };
    sort_sel_ascending(sel, 2, 4);
    CHECK(sel == std::vector<int32_t>({ 1, 3, 5, 9, 0, 2, 4, 7 }));
}

TEST_CASE(AmCompactFixture, gather_rows_copies_the_kept_rows_in_order) {
    const size_t rb = 3;
    std::vector<uint8_t> rows(6 * rb);
    for (size_t i = 0; i < rows.size(); i++) rows[i] = (uint8_t) i;
    const int32_t sel[3] = { 1, 4, 5 };
    std::vector<uint8_t> out(3 * rb, 0xEE);
    gather_rows(rows.data(), rb, sel, 3, out.data());
    CHECK(out == std::vector<uint8_t>({ 3, 4, 5, 12, 13, 14, 15, 16, 17 }));
}

// ---- 5. the per-block CPU path on a synthetic block ------------------------------------------------

namespace {

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float) ((s >> 40) & 0xFFFFFF) / (float) (1 << 23) - 1.0f;
    }
};

}  // namespace

TEST_CASE(AmCompactFixture, keeping_every_key_reproduces_the_block_exactly) {
    // slots == n_keys: the kept set is every key, so the fit has the trivial solution beta = 0 (w = 1)
    // and C2 = V. Checks the plumbing (head grouping, layouts, selection order) with a known answer.
    const int d = 16, n = 24, T = 16, nh = 4, nkv = 2, slots = T;
    Rng r;
    std::vector<float> k((size_t) d * T * nkv), v((size_t) d * T * nkv);
    for (auto & x : k) x = 0.5f * r.next();
    for (auto & x : v) x = r.next();
    std::vector<ggml_fp16_t> q((size_t) d * n * nh);
    for (auto & x : q) x = ggml_fp32_to_fp16(0.5f * r.next());
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    luce_am::GramAccumulator acc;
    BlockFit out;
    std::string err;
    luce_am::NnlsParams np;
    const bool ok = compact_block_host(be, acc, d, n, nh, nkv, T, slots, q.data(), k.data(), v.data(), np, out, &err);
    ggml_backend_free(be);
    if (!ok) std::printf("  compact_block_host: %s\n", err.c_str());
    REQUIRE(ok);
    REQUIRE((int) out.sel.size() == slots * nkv);
    REQUIRE((int) out.fit.beta.size() == slots * nkv);
    REQUIRE((int) out.fit.c2.size() == d * slots * nkv);
    for (int h = 0; h < nkv; h++)
        for (int s = 0; s < slots; s++) CHECK(out.sel[(size_t) h * slots + s] == s);
    double worst_b = 0, worst_c = 0;
    for (size_t i = 0; i < out.fit.beta.size(); i++) worst_b = std::max(worst_b, (double) std::fabs(out.fit.beta[i]));
    for (int h = 0; h < nkv; h++)
        for (size_t i = 0; i < (size_t) slots * d; i++)
            worst_c = std::max(worst_c, (double) std::fabs(out.fit.c2[(size_t) h * slots * d + i] - v[(size_t) h * T * d + i]));
    std::printf("  keep-all: max |beta| %.2e, max |C2 - V| %.2e\n", worst_b, worst_c);
    CHECK(worst_b < 0.1);
    CHECK(worst_c < 0.1);
}

// ---- 6. the whole path on the reference fixtures -----------------------------------------------------------

namespace {

constexpr double kRefOverlap = 0.985;
constexpr int    kRefMaxSwaps = 2;
constexpr double kObjSlack   = 0.05;

// Output error of (beta, C2) on selection `sel` for one KV head, in double: queries q [n][d], keys K and
// values V [T][d]; error = || softmax_sel(s_sel + beta) C2 - softmax_all(s) V ||_F / || softmax_all(s) V ||_F.
struct HeadScorer {
    int n = 0, T = 0, d = 0;
    std::vector<double> s;       // [n][T] scaled logits
    std::vector<double> otrue;   // [n][d]
    double onorm2 = 0;
    HeadScorer(const float * q, int n_, const float * K, const float * V, int T_, int d_) : n(n_), T(T_), d(d_) {
        s.assign((size_t) n * T, 0.0);
        otrue.assign((size_t) n * d, 0.0);
        const double scale = 1.0 / std::sqrt((double) d);
        std::vector<double> e(T);
        for (int r = 0; r < n; r++) {
            double mx = -1e300;
            for (int j = 0; j < T; j++) {
                double a = 0;
                for (int c = 0; c < d; c++) a += (double) q[(size_t) r * d + c] * (double) K[(size_t) j * d + c];
                s[(size_t) r * T + j] = a * scale;
                mx = std::max(mx, s[(size_t) r * T + j]);
            }
            double z = 0;
            for (int j = 0; j < T; j++) z += (e[j] = std::exp(s[(size_t) r * T + j] - mx));
            for (int j = 0; j < T; j++) {
                const double p = e[j] / z;
                for (int c = 0; c < d; c++) otrue[(size_t) r * d + c] += p * (double) V[(size_t) j * d + c];
            }
        }
        for (double x : otrue) onorm2 += x * x;
    }
    double err(const int32_t * sel, const float * beta, const float * c2, int t) const {
        double tot = 0;
        std::vector<double> a(t);
        for (int r = 0; r < n; r++) {
            double mx = -1e300, z = 0;
            for (int i = 0; i < t; i++) mx = std::max(mx, s[(size_t) r * T + sel[i]] + (double) beta[i]);
            for (int i = 0; i < t; i++) z += (a[i] = std::exp(s[(size_t) r * T + sel[i]] + (double) beta[i] - mx));
            for (int c = 0; c < d; c++) {
                double o = 0;
                for (int i = 0; i < t; i++) o += a[i] / z * (double) c2[(size_t) i * d + c];
                const double df = o - otrue[(size_t) r * d + c];
                tot += df * df;
            }
        }
        return std::sqrt(tot / std::max(onorm2, 1e-300));
    }
};

}  // namespace

TEST_CASE(AmCompactFixture, fixture_blocks_select_and_fit_within_the_m4_and_m5_bars) {
    const char * root = std::getenv("LUCE_AM_FIXTURES");
    if (!root || !*root) {
        std::printf("  SKIP: LUCE_AM_FIXTURES is not set (the fixture gate needs the dumped reference blocks)\n");
        return;
    }
    const std::vector<std::string> dirs = am_fixture::case_dirs(root);
    REQUIRE(!dirs.empty());
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    luce_am::GramAccumulator acc;
    int cases = 0;
    for (const std::string & dir : dirs) {
        am_fixture::Case c;
        std::string err;
        REQUIRE(am_fixture::load_case(dir, c, &err));
        if (c.slots == 0) continue;   // queries-only case
        const int d = c.head_dim, nh = c.n_head, nkv = c.n_kv, T = c.n_tokens, t = c.slots;
        REQUIRE(d > 0 && nh > 0 && nkv > 0 && nh % nkv == 0 && t > 0 && t <= T);
        // two layers of each case keep the CPU run short: the first and the last
        std::vector<int> layers = { c.layers.front() };
        if (c.layers.size() > 1) layers.push_back(c.layers.back());
        for (int layer : layers) {
            std::vector<float> q32, fitq, k, v, beta_ref, c2_ref;
            std::vector<int32_t> idx_ref;
            REQUIRE(am_fixture::read_f32(c, layer, "q", (size_t) d * T * nh, q32, &err));
            const std::vector<long> fs = am_fixture::entry_shape(c, layer, "fitq", "f32");
            REQUIRE(fs.size() == 3 && fs[0] == d && fs[2] == nkv);
            const int nfit = (int) fs[1];
            REQUIRE(am_fixture::read_f32(c, layer, "fitq", (size_t) d * nfit * nkv, fitq, &err));
            REQUIRE(am_fixture::read_f32(c, layer, "k", (size_t) d * T * nkv, k, &err));
            REQUIRE(am_fixture::read_f32(c, layer, "v", (size_t) d * T * nkv, v, &err));
            REQUIRE(am_fixture::read_i32(c, layer, "idx", (size_t) t * nkv, idx_ref, &err));
            REQUIRE(am_fixture::read_f32(c, layer, "beta", (size_t) t * nkv, beta_ref, &err));
            REQUIRE(am_fixture::read_f32(c, layer, "c2", (size_t) d * t * nkv, c2_ref, &err));
            std::vector<ggml_fp16_t> q16(q32.size());
            for (size_t i = 0; i < q32.size(); i++) q16[i] = ggml_fp32_to_fp16(q32[i]);

            BlockFit out;
            luce_am::NnlsParams np;
            const bool ok = compact_block_host(be, acc, d, T, nh, nkv, T, t, q16.data(), k.data(), v.data(), np, out, &err);
            if (!ok) std::printf("  compact_block_host: %s\n", err.c_str());
            REQUIRE(ok);
            REQUIRE((int) out.sel.size() == t * nkv);

            // M4: selection against the reference, per head and overall
            int same = 0;
            int worst_swaps = 0;
            for (int h = 0; h < nkv; h++) {
                std::set<int32_t> ref(idx_ref.begin() + (size_t) h * t, idx_ref.begin() + (size_t) (h + 1) * t);
                int sw = 0;
                for (int s = 0; s < t; s++) {
                    const int32_t j = out.sel[(size_t) h * t + s];
                    if (s > 0) CHECK(out.sel[(size_t) h * t + s - 1] < j);
                    sw += ref.count(j) ? 0 : 1;
                }
                same += t - sw;
                worst_swaps = std::max(worst_swaps, sw);
            }
            const double overlap = (double) same / (double) (t * nkv);
            // M5: output error on the objective the reference is held to, in parallel over KV heads
            std::vector<double> e_eng(nkv), e_ref(nkv);
            {
                std::vector<std::thread> th;
                for (int h = 0; h < nkv; h++)
                    th.emplace_back([&, h] {
                        const HeadScorer hs(&fitq[(size_t) h * nfit * d], nfit, &k[(size_t) h * T * d], &v[(size_t) h * T * d], T, d);
                        e_eng[h] = hs.err(&out.sel[(size_t) h * t], &out.fit.beta[(size_t) h * t], &out.fit.c2[(size_t) h * t * d], t);
                        e_ref[h] = hs.err(&idx_ref[(size_t) h * t], &beta_ref[(size_t) h * t], &c2_ref[(size_t) h * t * d], t);
                    });
                for (auto & x : th) x.join();
            }
            std::printf("  case %-12s layer %2d  overlap %.4f worst swaps %d  output error engine/reference per head:", c.name.c_str(), layer, overlap, worst_swaps);
            for (int h = 0; h < nkv; h++) std::printf(" %.4f/%.4f", e_eng[h], e_ref[h]);
            std::printf("\n");
            CHECK(overlap >= kRefOverlap);
            CHECK(worst_swaps <= kRefMaxSwaps);
            for (int h = 0; h < nkv; h++) {
                CHECK(std::isfinite(e_eng[h]));
                CHECK(e_eng[h] <= e_ref[h] * (1.0 + kObjSlack) + 1e-6);
            }
            for (float b : out.fit.beta) CHECK(std::isfinite(b));
        }
        cases++;
    }
    ggml_backend_free(be);
    REQUIRE(cases > 0);
}

// ---- 6. request field and pager layout the engine relies on -----------------------------------

TEST_CASE(AmCompactFixture, the_exact_tail_request_field_parses_to_a_non_negative_int_or_minus_one) {
    using luce::common::parse_am_exact_tail;
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({})")) == -1);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({"kvflash_am_exact_tail": 123})")) == 123);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({"kvflash_am_exact_tail": 0})")) == 0);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({"kvflash_am_exact_tail": -5})")) == -1);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({"kvflash_am_exact_tail": "9"})")) == -1);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"({"kvflash_am_exact_tail": 1.5})")) == -1);
    CHECK(parse_am_exact_tail(nlohmann::json::parse(R"([1,2])")) == -1);
}

// A compacted 640-token block is one pager chunk: its 64 rows sit in the head chunk's block, the
// nine covered chunks own no block, and every row's mask position lies inside the block (fill_slot_pos
// is mask metadata; the keys carry their own RoPE, and the Qwen3.5 masks do not read it).
TEST_CASE(AmCompactFixture, a_compacted_block_is_its_head_chunk_with_the_covered_chunks_unmapped) {
    constexpr int kChunk = 64, kPool = 1024, kHeads = 2, kD = 4, kLayers = 2;
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip{};
    ip.mem_size = 16 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> k, v;
    for (int l = 0; l < kLayers; l++) {
        k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kD, kPool, kHeads));
        v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kD, kPool, kHeads));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    luce::common::KvFlashConfig cfg;
    cfg.pool_tokens = kPool;
    luce::common::KvFlashPager p;
    REQUIRE(p.attach(cfg, k, v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));
    const int c0 = 0, span = kAutoBlockTokens / kChunk;
    REQUIRE(p.mark_compacted(c0, span, std::vector<float>((size_t)kLayers * kHeads * kChunk, 0.0f)));

    CHECK(p.is_compacted(c0));
    CHECK(p.block_of(c0) >= 0);
    for (int c = c0 + 1; c < c0 + span; c++) {
        CHECK(p.is_covered(c));
        CHECK(p.block_of(c) == -1);
    }
    CHECK(p.block_of(span) >= 0);   // the chunk after the block is exact and resident
    std::vector<int32_t> pos(kPool);
    p.fill_slot_pos(pos.data());
    const int blk = p.block_of(c0);
    int prev = -1;
    for (int i = 0; i < kChunk; i++) {
        const int32_t x = pos[(size_t)blk * kChunk + i];
        CHECK(x >= c0 * kChunk);
        CHECK(x < (c0 + span) * kChunk);
        CHECK(x > prev);
        prev = x;
    }
    p.synchronize_paging();
    if (buf) ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
}
