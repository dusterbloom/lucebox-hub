// Selector-v2 GPU scorer (kvflash_gpu_mass.h) against the AVX2 CPU scorer, on the ggml CPU backend so ctest needs no
// GPU. Same graph, same pool / staging / bias logic; only the backend differs. Pass: identical recall picks and
// select-score relative error < 1e-2 against kvflash_exact_mass_scores_fast_multi.

#include "CppUnitTestFramework.hpp"

#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"
#include "kvflash_prefill_reselect.h"

#include "ggml-cpu.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashGpuMassFixture {};

struct Outcome {
    bool ok = false;
    std::string why;
    double sel_rel = 0, mean_rel = 0, top_rel = 0, max_rel = 0;
    bool picks_same = false;
    size_t n_picks = 0;
};

// Small Qwen-like geometry: 2 layers, 2 kv heads x 3 q heads, D 64, 16-token chunks, a 24-block pool.
SynthCfg small_cfg(ggml_type kt, int sealed, int nq, int keep_newest = -1) {
    SynthCfg c;
    c.L = 2; c.HK = 2; c.G = 3; c.D = 64; c.CT = 16; c.NB = 24;
    c.KT = kt; c.sealed = sealed; c.N = nq; c.keep_newest = keep_newest; c.tail = 16;
    return c;
}

// `scored` chunks are scored out of a synth built with cfg.sealed >= scored: the chunks beyond stay in their pool
// blocks (the server's chunks >= sealed) and must be masked. host_ok=false makes host_seg return null.
Outcome run(const SynthCfg & cfg, int scored) {
    Outcome o;
    Synth s;
    s.build(cfg);
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { o.why = "no cpu backend"; return o; }
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(cfg.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < cfg.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, cfg.KT, cfg.D, (int64_t)cfg.NB * cfg.CT, cfg.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < cfg.L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));

    // CPU oracle over the first `scored` chunks.
    KvFlashExactKeys keys;
    keys.n_chunks = scored; keys.chunk_tokens = cfg.CT; keys.type = cfg.KT;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return s.blk[(size_t)c] >= 0 ? s.pool_seg(s.blk[(size_t)c], l, h) : s.host_seg(c, l, h); };
    const auto qp = s.qptrs();
    std::vector<std::vector<float>> outs_cpu, outs_gpu;
    std::vector<KvFlashMassPlanes> planes_cpu, planes_gpu;
    kvflash_exact_mass_scores_fast_multi(keys, s.dims(), qp, outs_cpu, 0, nullptr, &planes_cpu);

    KvFlashGpuMass gm(be);
    std::vector<int> block_of(s.blk.begin(), s.blk.begin() + scored);
    KvFlashGpuMassTiming tm;
    o.ok = gm.score(pool_k, block_of, [&](int c, int l, int h) { return s.host_seg(c, l, h); }, cfg.CT, s.dims(), qp, outs_gpu, &planes_gpu, &tm, &o.why);
    if (o.ok) {
        const auto sel_cpu = kvflash_select_scores(planes_cpu), sel_gpu = kvflash_select_scores(planes_gpu);
        auto rel = [](const std::vector<float> & a, const std::vector<float> & b) {
            double m = 0;
            for (size_t c = 0; c < a.size(); c++) if (a[c] > 1e-9) m = std::max(m, (double)std::fabs(a[c] - b[c]) / a[c]);
            return m;
        };
        o.sel_rel = rel(sel_cpu, sel_gpu);
        for (int n = 0; n < cfg.N; n++) {
            o.mean_rel = std::max(o.mean_rel, rel(outs_cpu[(size_t)n], outs_gpu[(size_t)n]));
            o.top_rel = std::max(o.top_rel, rel(planes_cpu[(size_t)n].top, planes_gpu[(size_t)n].top));
            o.max_rel = std::max(o.max_rel, rel(planes_cpu[(size_t)n].max, planes_gpu[(size_t)n].max));
        }
        // Pick as many units as there are planted chunks the queries cover below the window: the rest is noise.
        const int fwc = scored - cfg.tail, unit = 2;
        int k_units = 0;
        for (int j = 0; j < s.n_hot && j < cfg.N; j++) k_units += 10 + 7 * j + 1 < fwc;
        k_units = std::max(1, k_units);
        const auto pc = kvflash_recall_pick(sel_cpu, fwc, k_units, unit), pg = kvflash_recall_pick(sel_gpu, fwc, k_units, unit);
        o.picks_same = pc == pg;
        o.n_picks = pc.size();
    }
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
    return o;
}

}  // namespace

// The framework's asserts need the test object, so this is a macro, not a helper.
#define expect_parity(expr) do { \
    const Outcome o_ = (expr); \
    if (!o_.ok) std::fprintf(stderr, "gpu mass: %s\n", o_.why.c_str()); \
    REQUIRE(o_.ok); \
    CHECK(o_.picks_same); \
    CHECK(o_.n_picks > 0); \
    CHECK(o_.sel_rel < 1e-2); \
    CHECK(o_.mean_rel < 1e-2); \
    CHECK(o_.top_rel < 1e-2); \
} while (0)

TEST_CASE(KvflashGpuMassFixture, q4_0_mixed_residency_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8), 60)); }
TEST_CASE(KvflashGpuMassFixture, q8_0_mixed_residency_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 8), 60)); }
TEST_CASE(KvflashGpuMassFixture, all_resident_matches_cpu) {
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 36, 8);   // 36 chunks in a 40-block pool: nothing evicted, 4 blocks free
    c.NB = 40; c.tail = 8;
    expect_parity(run(c, 36));
}
TEST_CASE(KvflashGpuMassFixture, all_evicted_but_sinks_matches_cpu) { expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8, 0), 60)); }
TEST_CASE(KvflashGpuMassFixture, fewer_queries_than_64_matches_cpu) {
    expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 3), 60));
    expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 1), 60));
}

// Free blocks hold loud junk keys; they and chunks >= the scored count must get -inf.
TEST_CASE(KvflashGpuMassFixture, free_pool_blocks_are_masked) {
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 30, 8);   // 30 chunks, 24 blocks: sinks + 22 newest resident, 0 free
    c.NB = 40; c.junk_chunks = 1; c.tail = 8;        // 40 blocks: 10 free ones with junk
    expect_parity(run(c, 30));
}
TEST_CASE(KvflashGpuMassFixture, pool_blocks_holding_chunks_beyond_sealed_are_masked) {
    // 60 chunks laid out, only the first 52 sealed: the 8 newest are resident in the pool and must not score.
    expect_parity(run(small_cfg(GGML_TYPE_Q4_0, 60, 8), 52));
    expect_parity(run(small_cfg(GGML_TYPE_Q8_0, 60, 8), 44));
}

TEST_CASE(KvflashGpuMassFixture, reuse_across_calls_and_growth) {
    // One module, a small call then a larger one then the small again: buffers grow, results stay right.
    Synth s;
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 60, 8);
    s.build(c);
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(c.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < c.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, c.KT, c.D, (int64_t)c.NB * c.CT, c.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < c.L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));
    KvFlashGpuMass gm(be);
    std::vector<std::vector<float>> first, again, mid;
    std::vector<KvFlashMassPlanes> pl;
    auto host = [&](int ch, int l, int h) { return s.host_seg(ch, l, h); };
    const auto qp = s.qptrs();
    std::string why;
    std::vector<int> b40(s.blk.begin(), s.blk.begin() + 40), b60 = s.blk;
    REQUIRE(gm.score(pool_k, b40, host, c.CT, s.dims(), qp, first, &pl, nullptr, &why));
    REQUIRE(gm.score(pool_k, b60, host, c.CT, s.dims(), qp, mid, &pl, nullptr, &why));
    REQUIRE(gm.score(pool_k, b40, host, c.CT, s.dims(), qp, again, &pl, nullptr, &why));
    CHECK(first == again);
    CHECK(mid[0].size() == 60u);
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
}

TEST_CASE(KvflashGpuMassFixture, bad_inputs_fail_without_crashing) {
    Synth s;
    SynthCfg c = small_cfg(GGML_TYPE_Q4_0, 30, 2);
    s.build(c);
    ggml_backend_t be = ggml_backend_cpu_init();
    REQUIRE(be != nullptr);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(c.L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(ip);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < c.L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, c.KT, c.D, (int64_t)c.NB * c.CT, c.HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    KvFlashGpuMass gm(be);
    std::vector<std::vector<float>> o;
    std::string why;
    auto host = [&](int ch, int l, int h) { return s.host_seg(ch, l, h); };
    auto missing = [&](int, int, int) -> const uint8_t * { return nullptr; };
    std::vector<int> bo = s.blk;
    CHECK(!gm.score(pool_k, {}, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));            // nothing sealed
    bo[3] = c.NB;                                                                                          // outside the pool
    CHECK(!gm.score(pool_k, bo, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    bo = s.blk; bo[12] = bo[13];                                                                           // two chunks, one block
    CHECK(!gm.score(pool_k, bo, host, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    std::vector<int> ev(30, -1);                                                                           // all evicted, host copy absent
    CHECK(!gm.score(pool_k, ev, missing, c.CT, s.dims(), s.qptrs(), o, nullptr, nullptr, &why));
    CHECK(why.find("host K") != std::string::npos);
    CHECK(gm.score(pool_k, s.blk, host, c.CT, s.dims(), {}, o, nullptr, nullptr, &why));                  // no queries
    CHECK(o.empty());
    // A working set far over free memory is refused before anything is allocated.
    std::vector<const float *> many((size_t)4000000, s.qs.data());
    CHECK(!gm.score(pool_k, s.blk, host, c.CT, s.dims(), many, o, nullptr, nullptr, &why));
    CHECK(why.find("free") != std::string::npos);
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
}

TEST_CASE(KvflashGpuMassFixture, chunk_of_maps_blocks_and_marks_the_rest_masked) {
    std::vector<int> ev;
    // pool of 5 blocks; chunks 0..5: 0->3, 1 evicted, 2->0, 3 evicted, 4->4, 5 evicted. Blocks 1, 2 hold nothing sealed.
    const auto m = kvflash_gpu_mass_chunk_of({3, -1, 0, -1, 4, -1}, 5, ev);
    REQUIRE(m.size() == 8u);
    CHECK(m[0] == 2); CHECK(m[1] == -1); CHECK(m[2] == -1); CHECK(m[3] == 0); CHECK(m[4] == 4);
    CHECK(m[5] == 1); CHECK(m[6] == 3); CHECK(m[7] == 5);
    CHECK(ev == std::vector<int>({1, 3, 5}));
    CHECK(kvflash_gpu_mass_chunk_of({5}, 5, ev).empty());
}
