// KVFlash 2.0b M3 deferred gate: the queries luce_server captures during
// prefill (Q4_K_M weights) against the bf16 reference dump of the same block
// (optimizations/kvflash/lucebox/a1/am_fixture_dump.py, case traj1_12k_exactq).
//
// The reader and the cosine arithmetic are tested on synthetic fixtures the
// tests write themselves. The gate runs only when LUCE_AM_FIXTURES (directory
// holding the cases) and LUCE_AM_GGUF (the Q4_K_M target) are set.
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "am_fixture.h"

#include "internal.h"
#include "qwen35/qwen35_backend.h"
#include "../../src/qwen35/prefill_helpers.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace luce::common;
namespace fs = std::filesystem;
using namespace luce::amfix;

namespace {
struct AmGateFixture {};

// A fixture directory next to the test binary's cwd (never /tmp), removed on scope exit.
struct TmpDir {
    fs::path p;
    TmpDir() : p(fs::current_path() / ("am_fixture_test_" + std::to_string(::getpid()))) {
        fs::remove_all(p); fs::create_directories(p);
    }
    ~TmpDir() { std::error_code ec; fs::remove_all(p, ec); }
};

template <class T> void put(const fs::path & f, const std::vector<T> & v) {
    std::ofstream o(f, std::ios::binary);
    o.write((const char *)v.data(), (std::streamsize)(v.size() * sizeof(T)));
}

// Two heads, 3 tokens, head_dim 4: a manifest in the exact shape write_case emits.
void write_synthetic(const fs::path & dir, const std::vector<float> & q, int32_t pos0) {
    put(dir / "L03_q.f32", q);
    put(dir / "positions.i32", std::vector<int32_t>{pos0, pos0 + 1, pos0 + 2});
    put(dir / "prefix_tokens.i32", std::vector<int32_t>{5, 6, 7, 8, 9, 10});
    std::ofstream m(dir / "manifest.json");
    m << R"({
 "case": "synthetic", "block_index": 1, "start_token": 3, "end_token": 6, "n_tokens": 3, "head_dim": 4, "n_head": 2,
 "model_identity": {"model_path": "/m/Qwen3.8-27B", "_name_or_path": "/m/Qwen3.8-27B", "revision": null},
 "layers": [3],
 "files": [
  {"layer": null, "name": "positions", "file": "positions.i32", "dtype": "int32", "shape_ggml": [3], "shape_torch": [3], "bytes": 12},
  {"layer": null, "name": "prefix_tokens", "file": "prefix_tokens.i32", "dtype": "int32", "shape_ggml": [6], "shape_torch": [6], "bytes": 24},
  {"layer": 3, "name": "q", "file": "L03_q.f32", "dtype": "float32", "shape_ggml": [4, 3, 2], "shape_torch": [2, 3, 4], "bytes": 96}
 ]})";
}

std::vector<float> ramp(int n, float s = 1.0f) {
    std::vector<float> v(n);
    for (int i = 0; i < n; i++) v[i] = s * (float)std::sin(0.7 * i + 0.3);
    return v;
}
uint16_t f16(float x) { return ggml_fp32_to_fp16(x); }
std::vector<uint16_t> to_f16(const std::vector<float> & v) {
    std::vector<uint16_t> o(v.size());
    for (size_t i = 0; i < v.size(); i++) o[i] = f16(v[i]);
    return o;
}
}  // namespace

TEST_CASE(AmGateFixture, reader_loads_manifest_and_tensors) {
    TmpDir t; const auto q = ramp(24);
    write_synthetic(t.p, q, 3);
    Fixture fx; std::string err;
    REQUIRE(Fixture::open(t.p.string(), fx, err));
    CHECK((fx.layers() == std::vector<int>{3}));
    const File * f = fx.find("q", 3);
    REQUIRE(f != nullptr);
    CHECK((f->shape_ggml == std::vector<int64_t>{4, 3, 2}));
    std::vector<float> got; std::vector<int32_t> pos, toks;
    REQUIRE(fx.read_f32("q", 3, got, err));
    CHECK(got == q);
    REQUIRE(fx.read_i32("positions", pos, err));
    CHECK((pos == std::vector<int32_t>{3, 4, 5}));
    REQUIRE(fx.read_i32("prefix_tokens", toks, err));
    CHECK(toks.size() == 6);
    CHECK(fx.manifest["block_index"].get<int>() == 1);
}

TEST_CASE(AmGateFixture, reader_rejects_a_truncated_file_and_a_missing_entry) {
    TmpDir t;
    write_synthetic(t.p, ramp(24), 3);
    put(t.p / "L03_q.f32", ramp(23));  // 4 bytes short of the manifest
    Fixture fx; std::string err;
    CHECK(!Fixture::open(t.p.string(), fx, err));
    CHECK(err.find("L03_q.f32") != std::string::npos);

    TmpDir t2;
    write_synthetic(t2.p, ramp(24), 3);
    REQUIRE(Fixture::open(t2.p.string(), fx, err));
    std::vector<float> v;
    CHECK(!fx.read_f32("q", 7, v, err));        // no such layer
    CHECK(!fx.read_f32("positions", -1, v, err));   // wrong dtype
    CHECK(!Fixture::open((t2.p / "nope").string(), fx, err));
}

TEST_CASE(AmGateFixture, cosine_is_one_for_equal_minus_one_for_negated_and_isolates_the_perturbed_head) {
    const int H = 2, n = 3, d = 4;
    const auto q = ramp(H * n * d);
    auto r = cosine_per_head(q.data(), to_f16(q).data(), H, n, d);
    CHECK(std::fabs(r.mean[0] - 1.0) < 1e-3);
    CHECK(std::fabs(r.mean[1] - 1.0) < 1e-3);
    std::vector<float> neg = q; for (auto & x : neg) x = -x;
    r = cosine_per_head(q.data(), to_f16(neg).data(), H, n, d);
    CHECK(std::fabs(r.mean[0] + 1.0) < 1e-3);
    // Head 1 replaced by an orthogonal-ish pattern, head 0 untouched.
    std::vector<float> mix = q;
    for (int i = 0; i < n * d; i++) mix[(size_t)n * d + i] = (i % 2 ? 1.0f : -1.0f) * (float)(i + 1);
    r = cosine_per_head(q.data(), to_f16(mix).data(), H, n, d);
    CHECK(r.mean[0] > 0.999);
    CHECK(r.mean[1] < 0.9);
    CHECK(r.min[1] <= r.mean[1]);
}

TEST_CASE(AmGateFixture, cosine_of_a_zero_vector_is_zero_not_nan) {
    const std::vector<float> z(12, 0.0f);
    const auto r = cosine_per_head(z.data(), to_f16(z).data(), 1, 3, 4);
    CHECK(r.mean[0] == 0.0);
}

TEST_CASE(AmGateFixture, summarize_reports_min_median_mean) {
    const Summary s = summarize({0.5, 1.0, 0.9, 0.7});
    CHECK(s.min == 0.5);
    CHECK(std::fabs(s.median - 0.8) < 1e-12);
    CHECK(std::fabs(s.mean - 0.775) < 1e-12);
    CHECK(summarize({0.3, 0.1, 0.2}).median == 0.2);
}

TEST_CASE(AmGateFixture, identity_check_accepts_the_same_model_and_rejects_each_difference) {
    TmpDir t; write_synthetic(t.p, ramp(24), 3);
    Fixture fx; std::string err;
    REQUIRE(Fixture::open(t.p.string(), fx, err));
    GgufFacts g;
    g.n_layer = 4; g.full_attention_interval = 4; g.n_head = 2; g.head_dim = 4;
    g.names = {"Qwen3.8 27B", "Qwen3.8-27B-Instruct"};
    CHECK(identity_mismatch(fx.manifest, g).empty());
    GgufFacts a = g; a.n_head = 3;
    CHECK(!identity_mismatch(fx.manifest, a).empty());
    GgufFacts b = g; b.head_dim = 8;
    CHECK(!identity_mismatch(fx.manifest, b).empty());
    GgufFacts c = g; c.n_layer = 8;  // layers 3 and 7 would be full attention, the dump lists only 3
    CHECK(!identity_mismatch(fx.manifest, c).empty());
    GgufFacts d = g; d.names = {"Qwen3.6 27B"};
    CHECK(!identity_mismatch(fx.manifest, d).empty());
    GgufFacts e = g; e.names = {};  // nothing to compare by name: structure alone decides
    CHECK(identity_mismatch(fx.manifest, e).empty());
}

// ---- the gate ----
namespace {
struct GateBackend : Qwen35Backend {
    using Qwen35Backend::Qwen35Backend;
    int want_block = -1;
    std::vector<int> sealed;
    std::vector<uint16_t> plane;  // [layer][head][token][dim] F16, fa layers in order
    TargetWeights & weights() { return target_weights(); }
protected:
    void kvflash_am_seal_block(int block, int slot) override {
        sealed.push_back(block);
        if (block != want_block) return;
        TargetCache & c = target_cache();
        ggml_tensor * t = c.q_blk_ring[(size_t)slot];
        const size_t per_layer = (size_t)t->ne[0] * t->ne[1] * t->ne[2];
        plane.assign(per_layer * (size_t)t->ne[3], 0);
        ggml_backend_tensor_get(t, plane.data(), 0, plane.size() * sizeof(uint16_t));
    }
};

GgufFacts gguf_names(const std::string & path) {
    GgufFacts g;
    gguf_init_params ip{}; ip.no_alloc = true; ip.ctx = nullptr;
    gguf_context * c = gguf_init_from_file(path.c_str(), ip);
    if (!c) return g;
    for (const char * k : {"general.name", "general.basename", "general.base_model.0.name", "general.size_label"}) {
        const int64_t id = gguf_find_key(c, k);
        if (id >= 0 && gguf_get_kv_type(c, id) == GGUF_TYPE_STRING) g.names.push_back(gguf_get_val_str(c, id));
    }
    gguf_free(c);
    return g;
}
}  // namespace

TEST_CASE(AmGateFixture, q4km_queries_match_the_bf16_dump_of_trajectory_1_block) {
    const char * fdir = std::getenv("LUCE_AM_FIXTURES");
    const char * gguf = std::getenv("LUCE_AM_GGUF");
    if (!fdir || !gguf) { std::printf("[am-gate] SKIP: set LUCE_AM_FIXTURES and LUCE_AM_GGUF\n"); return; }

    Fixture fx; std::string err;
    REQUIRE(Fixture::open((fs::path(fdir) / "traj1_12k_exactq").string(), fx, err));
    const int block = fx.manifest["block_index"].get<int>();
    const int T = fx.manifest["n_tokens"].get<int>();
    const int H = fx.manifest["n_head"].get<int>();
    const int D = fx.manifest["head_dim"].get<int>();
    std::vector<int32_t> prefix, positions;
    REQUIRE(fx.read_i32("prefix_tokens", prefix, err));
    REQUIRE(fx.read_i32("positions", positions, err));
    REQUIRE((int)prefix.size() == (block + 1) * T);
    REQUIRE((int)positions.size() == T);

    const luce_test::ScopedEnvVar cap("LUCE_KVFLASH_AM_CAPTURE", "1");
    const luce_test::ScopedEnvVar nokvf("LUCE_KVFLASH", nullptr);
    Qwen35Config cfg;
    cfg.target_path = gguf;
    cfg.device.gpu = 0;
    cfg.device.max_ctx = ((int)prefix.size() + 255) / 256 * 256 + 256;
    GateBackend be(cfg);
    be.want_block = block;
    REQUIRE(be.init());

    // Same base model? Stop before comparing anything if not.
    TargetWeights & w = be.weights();
    GgufFacts facts = gguf_names(gguf);
    facts.n_layer = w.n_layer; facts.full_attention_interval = w.full_attention_interval;
    facts.n_head = w.n_head; facts.head_dim = w.n_embd_head_k;
    const std::string mism = identity_mismatch(fx.manifest, facts);
    std::printf("[am-gate] identity: manifest model=%s ; gguf names=", fx.manifest["model_identity"].dump().c_str());
    for (const auto & n : facts.names) std::printf("'%s' ", n.c_str());
    std::printf("; %s\n", mism.empty() ? "SAME BASE MODEL" : mism.c_str());
    REQUIRE(mism.empty());

    // The positions the engine feeds RoPE for this block, against the dump's.
    const int block_start = block * T;
    std::vector<int32_t> eng((size_t)4 * T);
    fill_qwen35_mrope_positions(eng.data(), block_start, T);
    for (int i = 0; i < T; i++) CHECK(eng[(size_t)i] == positions[(size_t)i]);

    GenerateRequest req; req.prompt = prefix; req.n_gen = 1;
    DaemonIO io;
    (void)be.generate_impl(req, io);
    REQUIRE(!be.sealed.empty());
    CHECK(be.sealed.back() == block);
    REQUIRE(!be.plane.empty());

    const int n_fa = (int)(be.plane.size() / ((size_t)D * T * H));
    const auto & layers = fx.manifest["layers"];
    REQUIRE((int)layers.size() == n_fa);
    std::vector<double> layer_means, head_means;
    std::vector<std::pair<int, double>> per_layer;
    double worst_token = 1.0;
    for (int i = 0; i < n_fa; i++) {
        const int layer = layers[(size_t)i].get<int>();
        CHECK(layer / w.full_attention_interval == i);
        std::vector<float> ref;
        REQUIRE(fx.read_f32("q", layer, ref, err));
        REQUIRE((int)ref.size() == H * T * D);
        const auto r = cosine_per_head(ref.data(), be.plane.data() + (size_t)i * H * T * D, H, T, D);
        const Summary hs = summarize(r.mean);
        for (double m : r.min) worst_token = std::min(worst_token, m);
        std::printf("[am-gate] layer %2d  head cos min %.4f median %.4f mean %.4f\n", layer, hs.min, hs.median, hs.mean);
        layer_means.push_back(hs.mean);
        head_means.insert(head_means.end(), r.mean.begin(), r.mean.end());
    }
    const Summary ls = summarize(layer_means), hs = summarize(head_means);
    std::printf("[am-gate] per-layer mean cosine: min %.4f median %.4f mean %.4f\n", ls.min, ls.median, ls.mean);
    std::printf("[am-gate] per-head  mean cosine: min %.4f median %.4f mean %.4f (%zu heads); worst single (head,token) %.4f\n",
                hs.min, hs.median, hs.mean, head_means.size(), worst_token);
    CHECK(ls.min > 0.9);  // tripwire only: wrong RoPE, head order, layer map or pre-norm Q lands far below
}
