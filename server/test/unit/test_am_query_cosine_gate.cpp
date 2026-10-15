// KVFlash 2.0b M3 deferred gate: the queries luce_server captures during
// prefill (Q4_K_M weights) against the bf16 reference dump of the same block
// (optimizations/kvflash/lucebox/a1/am_fixture_dump.py, case traj1_12k_exactq).
//
// The reader and the cosine arithmetic are tested on synthetic fixtures the
// tests write themselves. The gate runs only when LUCE_AM_FIXTURES (directory
// holding the cases) and LUCE_AM_GGUF (the Q4_K_M target) are set.
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "am_fixture_gate.h"

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

TEST_CASE(AmGateFixture, head_gate_fails_one_negated_head_that_the_layer_mean_hides) {
    const int H = 24, n = 3, d = 4;
    const auto q = ramp(H * n * d);
    auto e = q;
    for (int i = 0; i < n * d; i++) e[(size_t)7 * n * d + i] = -e[(size_t)7 * n * d + i];
    const auto r = cosine_per_head(q.data(), to_f16(e).data(), H, n, d);
    CHECK(summarize(r.mean).mean > 0.9);  // the layer average passes, which is the hole
    const std::string v = head_violation(r);
    CHECK(v.find("head 7") != std::string::npos);
    const auto ok = cosine_per_head(q.data(), to_f16(q).data(), H, n, d);
    CHECK(head_violation(ok).empty());
}

TEST_CASE(AmGateFixture, head_gate_fails_a_positive_scale_error_that_cosine_cannot_see) {
    const int H = 4, n = 3, d = 4;
    const auto q = ramp(H * n * d);
    auto e = q;
    for (int i = 0; i < n * d; i++) e[(size_t)2 * n * d + i] *= 1.2f;
    const auto r = cosine_per_head(q.data(), to_f16(e).data(), H, n, d);
    CHECK(r.mean[2] > 0.999);
    CHECK(std::fabs(r.ratio[2] - 1.2) < 1e-2);
    CHECK(head_violation(r).find("head 2") != std::string::npos);
}

TEST_CASE(AmGateFixture, reader_rejects_dimensions_that_overflow_or_are_not_positive) {
    for (const char * shape : {"[4611686018427387910, 4]", "[-4, -3]", "[0, 4]", "[3037000500, 3037000500]"}) {
        TmpDir t; write_synthetic(t.p, ramp(24), 3);
        std::string m;
        { std::ifstream in(t.p / "manifest.json"); m.assign(std::istreambuf_iterator<char>(in), {}); }
        const std::string from = "\"shape_ggml\": [4, 3, 2]";
        const auto at = m.find(from);
        REQUIRE(at != std::string::npos);
        m.replace(at, from.size(), std::string("\"shape_ggml\": ") + shape);
        { std::ofstream o(t.p / "manifest.json"); o << m; }
        Fixture fx; std::string err;
        CHECK(!Fixture::open(t.p.string(), fx, err));
        CHECK(err.find("L03_q.f32") != std::string::npos);
    }
}

TEST_CASE(AmGateFixture, identity_check_survives_a_null_name_or_path) {
    TmpDir t; write_synthetic(t.p, ramp(24), 3);
    Fixture fx; std::string err;
    REQUIRE(Fixture::open(t.p.string(), fx, err));
    fx.manifest["model_identity"]["_name_or_path"] = nullptr;
    GgufFacts g;
    g.n_layer = 4; g.full_attention_interval = 4; g.n_head = 2; g.head_dim = 4;
    g.names = {"Qwen3.8 27B"};
    CHECK(identity_mismatch(fx.manifest, g).empty());   // falls back to model_path
    g.names = {"Qwen3.6 27B"};
    CHECK(!identity_mismatch(fx.manifest, g).empty());
}

// ---- the gate ----
namespace {
struct GateBackend : Qwen35Backend {
    using Qwen35Backend::Qwen35Backend;
    int want_block = -1;
    std::vector<int> sealed;
    std::vector<uint16_t> plane;  // [layer][head][token][dim] F16, fa layers in order
    TargetWeights & weights() { return target_weights(); }
    TargetCache & cache() { return target_cache(); }
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
    const char * kv_dump = std::getenv("LUCE_AM_DUMP_KV");  // diagnosis aid: K and V rows of the block at the first full-attention layer
    // The gate compares against an unrotated reference, so the KV cache must be F16: under the default Q4_0
    // (and TQ3) the graph FWHT-rotates K and Q, and the captured Q would be in the rotated basis.
    const luce_test::ScopedEnvVar kvf16("LUCE_KV_F16", "1");
    const luce_test::ScopedEnvVar kvq4("LUCE_KV_Q4", nullptr), kvtq3("LUCE_KV_TQ3", nullptr), kvk("LUCE_KV_K", nullptr),
                                  kvv("LUCE_KV_V", nullptr), kvrot("LUCE_KV_ROTATE", nullptr);
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
    if (kv_dump) {
        TargetCache & c = be.cache();
        for (int which = 0; which < 2; which++) {
            ggml_tensor * t = which ? c.attn_v[0] : c.attn_k[0];
            REQUIRE(t->type == GGML_TYPE_F16);
            std::vector<uint16_t> out;  // [kv head][token][dim]
            for (int h = 0; h < (int)t->ne[2]; h++) {
                std::vector<uint16_t> rows((size_t)D * T);
                ggml_backend_tensor_get(t, rows.data(), (size_t)h * t->nb[2] + (size_t)block * T * t->nb[1], rows.size() * 2);
                out.insert(out.end(), rows.begin(), rows.end());
            }
            std::ofstream o(std::string(kv_dump) + (which ? "_v.f16" : "_k.f16"), std::ios::binary);
            o.write((const char *)out.data(), (std::streamsize)(out.size() * 2));
        }
    }
    REQUIRE(be.cache().attn_k[0]->type == GGML_TYPE_F16);
    REQUIRE(!be.sealed.empty());
    CHECK(be.sealed.back() == block);
    REQUIRE(!be.plane.empty());

    if (const char * dump = std::getenv("LUCE_AM_DUMP_PLANE")) {  // diagnosis aid: the raw F16 plane, [layer][head][token][dim]
        std::ofstream o(dump, std::ios::binary);
        o.write((const char *)be.plane.data(), (std::streamsize)(be.plane.size() * sizeof(uint16_t)));
    }
    const int n_fa = (int)(be.plane.size() / ((size_t)D * T * H));
    const auto & layers = fx.manifest["layers"];
    REQUIRE((int)layers.size() == n_fa);
    std::vector<double> layer_means, head_means, head_ratios;
    std::string violations;
    std::vector<std::pair<int, double>> per_layer;
    double worst_token = 1.0, worst_cos = 2.0, worst_ratio_dev = -1.0;
    int wc_layer = -1, wc_head = -1, wr_layer = -1, wr_head = -1;
    double wr_val = 0;
    for (int i = 0; i < n_fa; i++) {
        const int layer = layers[(size_t)i].get<int>();
        CHECK(layer / w.full_attention_interval == i);
        std::vector<float> ref;
        REQUIRE(fx.read_f32("q", layer, ref, err));
        REQUIRE((int)ref.size() == H * T * D);
        const auto r = cosine_per_head(ref.data(), be.plane.data() + (size_t)i * H * T * D, H, T, D);
        const Summary hs = summarize(r.mean);
        for (double m : r.min) worst_token = std::min(worst_token, m);
        for (int h = 0; h < H; h++) {
            if (r.mean[(size_t)h] < worst_cos) { worst_cos = r.mean[(size_t)h]; wc_layer = layer; wc_head = h; }
            const double dev = std::fabs(r.ratio[(size_t)h] - 1.0);
            if (dev > worst_ratio_dev) { worst_ratio_dev = dev; wr_layer = layer; wr_head = h; wr_val = r.ratio[(size_t)h]; }
        }
        std::printf("[am-gate] layer %2d  head cos min %.4f median %.4f mean %.4f\n", layer, hs.min, hs.median, hs.mean);
        layer_means.push_back(hs.mean);
        head_ratios.insert(head_ratios.end(), r.ratio.begin(), r.ratio.end());
        const std::string v = head_violation(r);
        if (!v.empty()) violations += "layer " + std::to_string(layer) + " " + v + "\n";
        head_means.insert(head_means.end(), r.mean.begin(), r.mean.end());
    }
    const Summary ls = summarize(layer_means), hs = summarize(head_means);
    std::printf("[am-gate] per-layer mean cosine: min %.4f median %.4f mean %.4f\n", ls.min, ls.median, ls.mean);
    std::printf("[am-gate] per-head  mean cosine: min %.4f median %.4f mean %.4f (%zu heads); worst single (head,token) %.4f\n",
                hs.min, hs.median, hs.mean, head_means.size(), worst_token);
    const Summary rs = summarize(head_ratios);
    auto p10 = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[(size_t)((v.size() - 1) * 0.10)]; };
    std::printf("[am-gate] SUMMARY case=%s kv=%s/%s per-head cos min %.4f median %.4f p10 %.4f | norm ratio min %.4f median %.4f max %.4f | "
                "worst cos (layer %d, head %d) %.4f | worst ratio (layer %d, head %d) %.4f\n",
                fx.manifest["case"].get<std::string>().c_str(), ggml_type_name(be.cache().attn_k[0]->type),
                ggml_type_name(be.cache().attn_v[0]->type), hs.min, hs.median, p10(head_means), rs.min, rs.median,
                *std::max_element(head_ratios.begin(), head_ratios.end()), wc_layer, wc_head, worst_cos, wr_layer, wr_head, wr_val);
    std::printf("[am-gate] per-head norm ratio eng/ref: min %.4f median %.4f mean %.4f\n", rs.min, rs.median, rs.mean);
    if (!violations.empty()) std::printf("[am-gate] head violations:\n%s", violations.c_str());
    CHECK(violations.empty());  // every head: mean cosine > 0.85 and norm ratio in [0.95, 1.05]; layer means above are reported only
}
