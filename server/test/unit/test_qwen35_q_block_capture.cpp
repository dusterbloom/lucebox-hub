// KVFlash 2.0b M3: query-block capture. A synthetic 2-layer model (every
// layer full attention, F32 weights, CPU backend) is run through
// build_qwen35_graph, and the captured F16 queries are compared with a scalar
// double-precision reference written in this file: rms norm, Q/gate split,
// per-head Q norm, interleaved M-RoPE, then (for layer 1) the whole of layer 0
// (causal GQA attention, sigmoid gate, wo, residual, post norm, SwiGLU).
// Nothing in the reference reads the capture tensors.
#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"
#include "internal.h"
#include "../../src/qwen35/prefill_helpers.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(LUCE_BACKEND_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct QBlockCaptureFixture {};

// Real Qwen3.8 attention shape (head_dim 256, n_rot 64, sections 11/11/10/0, GQA ratio 6), small width.
constexpr int kEmbd = 64, kHd = 256, kHeads = 12, kKvHeads = 2, kFf = 96;
constexpr int kRot = 64, kMaxCtx = 32, kLayers = 2;
constexpr int kSections[4] = {11, 11, 10, 0};
constexpr float kTheta = 10000.0f, kEps = 1e-6f;

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    float next() {  // uniform in [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float)((s >> 40) & 0xFFFFFF) / (float)(1 << 23) - 1.0f;
    }
};

struct HostLayer {
    std::vector<float> attn_norm, post_norm, wq, wk, wv, wo, q_norm, k_norm, wg, wu, wd;
};

// Synthetic weights: tensors live in a CPU buffer, a host copy feeds the reference.
struct Model {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    TargetWeights w;
    std::vector<HostLayer> host;
    std::vector<float> out_norm_h;

    ggml_tensor * mk(std::vector<float> & dst, float scale, bool norm, int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        dst.resize((size_t)(ne0 * ne1));
        Rng r; static uint64_t salt = 1; r.s ^= (salt++) * 0xD1B54A32D192ED03ull;
        for (auto & v : dst) v = norm ? 1.0f + 0.1f * r.next() : scale * r.next();
        return t;
    }

    bool init() {
#if defined(LUCE_BACKEND_CUDA) || defined(GGML_USE_HIP)
        // LUCE_TEST_QBLK_GPU=1 runs the same cases on GPU 0 (the strided F32
        // to F16 copy and flash attention are then the CUDA/HIP kernels).
        // Default is the CPU backend so the test needs no GPU.
        if (std::getenv("LUCE_TEST_QBLK_GPU")) backend = ggml_backend_cuda_init(0);
        else
#endif
        backend = ggml_backend_cpu_init();
        if (!backend) return false;
        ggml_init_params ip{};
        ip.mem_size = 1 << 22;
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        w.n_layer = kLayers;
        w.full_attention_interval = 1;
        w.n_embd_head_k = kHd; w.n_embd_head_v = kHd;
        w.n_head = kHeads; w.n_head_kv = kKvHeads;
        w.n_embd = kEmbd; w.n_ff = kFf;
        w.rope_dimension_count = kRot; w.rope_theta = kTheta; w.rms_eps = kEps;
        w.n_capture_layers = 0;
        for (int i = 0; i < 4; i++) w.rope_sections[i] = kSections[i];
        const int q_dim = kHd * kHeads, kv_dim = kHd * kKvHeads;
        w.layers.resize(kLayers);
        host.resize(kLayers);
        std::vector<ggml_tensor *> ts;
        std::vector<std::vector<float> *> hs;
        auto add = [&](ggml_tensor *& slot, std::vector<float> & h, float sc, bool norm, int64_t a, int64_t b = 1) {
            slot = mk(h, sc, norm, a, b); ts.push_back(slot); hs.push_back(&h);
        };
        for (int l = 0; l < kLayers; l++) {
            TargetLayer & L = w.layers[l]; HostLayer & H = host[l];
            add(L.attn_norm, H.attn_norm, 0, true, kEmbd);
            add(L.attn_post_norm, H.post_norm, 0, true, kEmbd);
            add(L.wq, H.wq, 0.3f, false, kEmbd, 2 * q_dim);
            add(L.wk, H.wk, 0.3f, false, kEmbd, kv_dim);
            add(L.wv, H.wv, 0.3f, false, kEmbd, kv_dim);
            add(L.wo, H.wo, 0.3f, false, q_dim, kEmbd);
            add(L.q_norm, H.q_norm, 0, true, kHd);
            add(L.k_norm, H.k_norm, 0, true, kHd);
            add(L.w_gate, H.wg, 0.3f, false, kEmbd, kFf);
            add(L.w_up, H.wu, 0.3f, false, kEmbd, kFf);
            add(L.w_down, H.wd, 0.3f, false, kFf, kEmbd);
        }
        add(w.out_norm, out_norm_h, 0, true, kEmbd);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) return false;
        for (size_t i = 0; i < ts.size(); i++)
            ggml_backend_tensor_set(ts[i], hs[i]->data(), 0, hs[i]->size() * sizeof(float));
        return true;
    }
    void destroy() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};

// ---- scalar reference (double) ----
using Vec = std::vector<double>;

Vec rmsnorm(const Vec & x, const std::vector<float> & g) {
    double ss = 0; for (double v : x) ss += v * v;
    const double inv = 1.0 / std::sqrt(ss / (double)x.size() + kEps);
    Vec o(x.size());
    for (size_t i = 0; i < x.size(); i++) o[i] = x[i] * inv * g[i];
    return o;
}
// ggml mul_mat(W[n_in, n_out], x[n_in]) -> [n_out]
Vec matvec(const std::vector<float> & W, const Vec & x, int n_out) {
    const int n_in = (int)x.size();
    Vec o(n_out, 0.0);
    for (int j = 0; j < n_out; j++) {
        double a = 0; for (int i = 0; i < n_in; i++) a += (double)W[(size_t)j * n_in + i] * x[i];
        o[j] = a;
    }
    return o;
}
// Interleaved M-RoPE on a text position (axes 0..2 carry pos, axis 3 is 0),
// NEOX pairing (i, i + n_rot/2) as ggml does for the M-RoPE types.
void rope(double * v, int pos) {
    const int sect_dims = kSections[0] + kSections[1] + kSections[2] + kSections[3];
    const double scale = std::pow((double)kTheta, -2.0 / kRot);
    Vec out(v, v + kRot);
    for (int ic = 0; ic < kRot / 2; ic++) {
        const int sector = ic % sect_dims;
        double p = 0;  // axis 3 = 0
        if (sector % 3 == 1 && sector < 3 * kSections[1]) p = pos;
        else if (sector % 3 == 2 && sector < 3 * kSections[2]) p = pos;
        else if (sector % 3 == 0 && sector < 3 * kSections[0]) p = pos;
        const double th = p * std::pow(scale, ic);
        const double c = std::cos(th), s = std::sin(th);
        const double x0 = v[ic], x1 = v[ic + kRot / 2];
        out[ic] = x0 * c - x1 * s;
        out[ic + kRot / 2] = x0 * s + x1 * c;
    }
    std::memcpy(v, out.data(), sizeof(double) * kRot);
}
struct QK { std::vector<Vec> q; std::vector<Vec> k, v; std::vector<Vec> gate; };  // per token, flattened heads

// Q, gate, K (normed+roped), V for every token of a layer input.
QK project(const HostLayer & H, const std::vector<Vec> & h, int pos0) {
    QK r;
    for (size_t t = 0; t < h.size(); t++) {
        const Vec x = rmsnorm(h[t], H.attn_norm);
        const Vec qg = matvec(H.wq, x, 2 * kHd * kHeads);
        Vec q(kHd * kHeads), g(kHd * kHeads);
        for (int hh = 0; hh < kHeads; hh++) {
            Vec qh(qg.begin() + hh * 2 * kHd, qg.begin() + hh * 2 * kHd + kHd);
            qh = rmsnorm(qh, H.q_norm);
            rope(qh.data(), pos0 + (int)t);
            std::copy(qh.begin(), qh.end(), q.begin() + hh * kHd);
            std::copy(qg.begin() + hh * 2 * kHd + kHd, qg.begin() + (hh + 1) * 2 * kHd, g.begin() + hh * kHd);
        }
        Vec k = matvec(H.wk, x, kHd * kKvHeads), v = matvec(H.wv, x, kHd * kKvHeads);
        for (int hh = 0; hh < kKvHeads; hh++) {
            Vec kh(k.begin() + hh * kHd, k.begin() + (hh + 1) * kHd);
            kh = rmsnorm(kh, H.k_norm);
            rope(kh.data(), pos0 + (int)t);
            std::copy(kh.begin(), kh.end(), k.begin() + hh * kHd);
        }
        r.q.push_back(q); r.gate.push_back(g); r.k.push_back(k); r.v.push_back(v);
    }
    return r;
}
// Output hidden states of one layer.
std::vector<Vec> layer_forward(const HostLayer & H, const std::vector<Vec> & h, const QK & p) {
    std::vector<Vec> out;
    const int group = kHeads / kKvHeads;
    for (size_t t = 0; t < h.size(); t++) {
        Vec att(kHd * kHeads, 0.0);
        for (int hh = 0; hh < kHeads; hh++) {
            const int kvh = hh / group;
            Vec sc(t + 1);
            double mx = -1e300;
            for (size_t j = 0; j <= t; j++) {
                double a = 0;
                for (int d = 0; d < kHd; d++) a += p.q[t][hh * kHd + d] * p.k[j][kvh * kHd + d];
                sc[j] = a / std::sqrt((double)kHd); mx = std::max(mx, sc[j]);
            }
            double den = 0; for (auto & s : sc) { s = std::exp(s - mx); den += s; }
            for (size_t j = 0; j <= t; j++)
                for (int d = 0; d < kHd; d++) att[hh * kHd + d] += sc[j] / den * p.v[j][kvh * kHd + d];
        }
        for (int i = 0; i < kHd * kHeads; i++) att[i] *= 1.0 / (1.0 + std::exp(-p.gate[t][i]));
        Vec o = matvec(H.wo, att, kEmbd);
        for (int i = 0; i < kEmbd; i++) o[i] += h[t][i];
        const Vec post = rmsnorm(o, H.post_norm);
        const Vec g = matvec(H.wg, post, kFf), u = matvec(H.wu, post, kFf);
        Vec gu(kFf);
        for (int i = 0; i < kFf; i++) gu[i] = g[i] / (1.0 + std::exp(-g[i])) * u[i];
        Vec d = matvec(H.wd, gu, kEmbd);
        for (int i = 0; i < kEmbd; i++) d[i] += o[i];
        out.push_back(d);
    }
    return out;
}
// Reference queries per layer: ref[layer][token] = flattened [n_head * head_dim].
std::vector<std::vector<Vec>> reference_queries(const Model & m, const std::vector<std::vector<float>> & emb) {
    std::vector<Vec> h;
    for (auto & e : emb) h.emplace_back(e.begin(), e.end());
    std::vector<std::vector<Vec>> ref;
    for (int l = 0; l < kLayers; l++) {
        const QK p = project(m.host[l], h, 0);
        ref.push_back(p.q);
        h = layer_forward(m.host[l], h, p);
    }
    return ref;
}

float f16_to_f32(uint16_t h) { return ggml_fp16_to_fp32((ggml_fp16_t)h); }

// One ubatch through build_qwen35_graph on the CPU backend.
struct UbatchRun {
    int nodes = 0;
    std::vector<float> hidden;       // result_norm of the last ubatch
    std::vector<float> q_cap_last;   // cache.q_cap after the ubatch
};
UbatchRun run_ubatch(Model & m, TargetCache & cache, const std::vector<std::vector<float>> & emb,
               int kv_start, int n, bool q_capture, int first, int count, int dst) {
    UbatchRun r;
    std::vector<uint8_t> arena(64u << 20);
    ggml_init_params ip{}; ip.mem_size = arena.size(); ip.mem_buffer = arena.data(); ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kEmbd, n, 1);
    ggml_set_input(inp);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
    ggml_set_input(pos);
    const int kv_pad = align_up(kMaxCtx + n, 32), q_pad = align_up(n, KQ_MASK_PAD);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_input(mask);

    QwenGraphInputs gi{};
    gi.inp_embed = inp; gi.positions = pos; gi.attn_mask = mask;
    gi.n_tokens = n; gi.kv_start = kv_start;
    gi.q_capture = q_capture;
    gi.q_blk_first = first; gi.q_blk_count = count; gi.q_blk_dst = dst;
    QwenGraphOutputs go = build_qwen35_graph(ctx, gf, m.w, cache, gi);
    (void)go;
    ggml_tensor * out = ggml_graph_get_tensor(gf, "result_norm");
    r.nodes = ggml_graph_n_nodes(gf);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    bool ok = ga && out && ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        std::vector<float> e((size_t)kEmbd * n);
        for (int t = 0; t < n; t++) std::memcpy(&e[(size_t)t * kEmbd], emb[kv_start + t].data(), sizeof(float) * kEmbd);
        ggml_backend_tensor_set(inp, e.data(), 0, e.size() * sizeof(float));
        std::vector<int32_t> pb((size_t)4 * n, 0);
        fill_qwen35_mrope_positions(pb.data(), kv_start, n);
        ggml_backend_tensor_set(pos, pb.data(), 0, pb.size() * sizeof(int32_t));
        upload_qwen35_causal_mask(mask, kv_start, n, 32);
        ok = ggml_backend_graph_compute(m.backend, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            r.hidden.resize((size_t)kEmbd * n);
            ggml_backend_tensor_get(out, r.hidden.data(), 0, r.hidden.size() * sizeof(float));
            if (cache.q_cap) {
                r.q_cap_last.resize((size_t)ggml_nelements(cache.q_cap));
                ggml_backend_tensor_get(cache.q_cap, r.q_cap_last.data(), 0, r.q_cap_last.size() * sizeof(float));
            }
        }
    }
    if (!ok) r.hidden.clear();
    if (ga) ggml_gallocr_free(ga);
    ggml_free(ctx);
    return r;
}

std::vector<std::vector<float>> make_embeddings(int n) {
    Rng r; r.s ^= 0x1234567ull;
    std::vector<std::vector<float>> e(n, std::vector<float>(kEmbd));
    for (auto & t : e) for (auto & v : t) v = r.next();
    return e;
}

bool make_cache(Model & m, TargetCache & c) {
    return create_target_cache_partial(m.w, kMaxCtx, /*max_verify_tokens=*/1, m.backend, c,
                                       /*prefill_only=*/true, 0, kLayers,
                                       /*allocate_target_feat=*/false, /*ctx_alloc=*/0,
                                       false, false, 1, false, GGML_TYPE_F16, GGML_TYPE_F16);
}

// Row `row` of layer `l`, head `h` of the F16 plane, as floats.
std::vector<uint16_t> read_q_blk_raw(const TargetCache & c, int l, int row) {
    const ggml_tensor * t = c.q_blk;
    std::vector<uint16_t> raw((size_t)t->ne[0] * t->ne[2]);
    for (int h = 0; h < (int)t->ne[2]; h++) {
        ggml_backend_tensor_get(t, raw.data() + (size_t)h * t->ne[0],
                                (size_t)l * t->nb[3] + (size_t)h * t->nb[2] + (size_t)row * t->nb[1],
                                sizeof(uint16_t) * t->ne[0]);
    }
    return raw;
}
std::vector<float> read_q_blk(const TargetCache & c, int l, int row) {
    const std::vector<uint16_t> raw = read_q_blk_raw(c, l, row);
    std::vector<float> o(raw.size());
    for (size_t i = 0; i < raw.size(); i++) o[i] = f16_to_f32(raw[i]);
    return o;
}
// max |a - b| / max |b|
double rel_err(const std::vector<float> & a, const Vec & b) {
    double mx = 0, ref = 0;
    for (size_t i = 0; i < a.size(); i++) { mx = std::max(mx, std::fabs(a[i] - b[i])); ref = std::max(ref, std::fabs(b[i])); }
    return mx / ref;
}
}  // namespace

TEST_CASE(QBlockCaptureFixture, bytes_match_the_plan_for_qwen38_27b_dims) {
    TargetWeights w;  // defaults are the 27B: head_dim 256, 24 heads, 64 layers / 4
    w.n_layer = 64; w.full_attention_interval = 4;
    CHECK(q_block_capture_bytes(w, 640) == (size_t)640 * 24 * 256 * 2 * 16);
    CHECK(q_block_capture_bytes(w, 640) / 16 == (size_t)7864320);  // 7.9 MB per layer
}

TEST_CASE(QBlockCaptureFixture, captured_queries_match_the_scalar_reference_across_ubatches) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    Model m; REQUIRE(m.init());
    const int kBlock = 8, kPrompt = 14;
    const auto emb = make_embeddings(kPrompt);
    const auto ref = reference_queries(m, emb);

    TargetCache cache; REQUIRE(make_cache(m, cache));
    REQUIRE(cache.q_blk == nullptr);   // capture off: nothing allocated
    REQUIRE(alloc_q_block_capture(cache, m.w, kBlock));
    REQUIRE(cache.q_blk != nullptr);
    CHECK(cache.q_blk->type == GGML_TYPE_F16);
    CHECK(cache.q_blk->ne[0] == kHd);
    CHECK(cache.q_blk->ne[1] == kBlock);
    CHECK(cache.q_blk->ne[2] == kHeads);
    CHECK(cache.q_blk->ne[3] == kLayers);

    // Sentinel in every row so untouched rows are detectable.
    {
        std::vector<uint16_t> s((size_t)ggml_nelements(cache.q_blk), 0x7E7E);
        ggml_backend_tensor_set(cache.q_blk, s.data(), 0, s.size() * 2);
    }
    // ubatch A: tokens 0..5 (kv_start 0) fill block rows 0..5.
    // ubatch B: tokens 6..13 (kv_start 6, not a block start) straddle the block
    // end: only its first 2 tokens belong to the block, rows 6..7.
    REQUIRE(!run_ubatch(m, cache, emb, 0, 6, false, 0, 6, 0).hidden.empty());
    REQUIRE(!run_ubatch(m, cache, emb, 6, 8, false, 0, 2, 6).hidden.empty());

    for (int l = 0; l < kLayers; l++) {
        double worst = 0;
        for (int row = 0; row < kBlock; row++) {
            const double e = rel_err(read_q_blk(cache, l, row), ref[l][row]);
            worst = std::max(worst, e);
            // F16 rounding of Q is up to 4.9e-4 relative (measured 3.7e-4); layer 1 also sees the F16 KV cache of layer 0 (measured 6.7e-4).
            CHECK(e < (l == 0 ? 1e-3 : 2e-3));
        }
        std::fprintf(stderr, "[q_blk] layer %d worst max|dq|/max|q| vs scalar reference: %.2e\n", l, worst);
    }
    free_q_block_capture(cache);
    CHECK(cache.q_blk == nullptr);
    free_target_cache(cache);
    m.destroy();
}

TEST_CASE(QBlockCaptureFixture, window_inside_a_ubatch_and_bitwise_match_with_q_cap) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    Model m; REQUIRE(m.init());
    const int kBlock = 8, kPrompt = 14;
    const auto emb = make_embeddings(kPrompt);
    const auto ref = reference_queries(m, emb);

    TargetCache cache; REQUIRE(make_cache(m, cache));
    REQUIRE(alloc_q_block_capture(cache, m.w, kBlock));
    {
        std::vector<uint16_t> s((size_t)ggml_nelements(cache.q_blk), 0x7E7E);
        ggml_backend_tensor_set(cache.q_blk, s.data(), 0, s.size() * 2);
    }
    // One ubatch of all 14 tokens; the second block is tokens 8..13 -> rows 0..5
    // (q_blk_first 8). Also ask for the existing last-token q_cap.
    const UbatchRun r = run_ubatch(m, cache, emb, 0, kPrompt, /*q_capture=*/true, 8, 6, 0);
    REQUIRE(!r.hidden.empty());
    for (int l = 0; l < kLayers; l++) {
        for (int row = 0; row < 6; row++) {
            CHECK(rel_err(read_q_blk(cache, l, row), ref[l][8 + row]) < (l == 0 ? 1e-3 : 2e-3));
        }
        // rows 6, 7 were not captured: still the sentinel
        for (int row = 6; row < kBlock; row++) {
            for (uint16_t v : read_q_blk_raw(cache, l, row)) CHECK(v == 0x7E7E);
        }
        // Token 13 is block row 5. The last-token q_cap is the same graph tensor in
        // F32, so the F16 capture must be its exact rounding.
        const std::vector<float> got = read_q_blk(cache, l, 5);
        bool same = true;
        for (int h = 0; h < kHeads; h++)
            for (int d = 0; d < kHd; d++) {
                const float qc = r.q_cap_last[((size_t)l * kHeads + h) * kHd + d];
                const uint16_t want = ggml_fp32_to_fp16(qc);
                const float wantf = f16_to_f32(want);
                same &= std::memcmp(&wantf, &got[(size_t)h * kHd + d], 4) == 0;
            }
        CHECK(same);
    }
    free_target_cache(cache);   // must release the capture buffer too
    CHECK(cache.q_blk == nullptr);
    m.destroy();
}

TEST_CASE(QBlockCaptureFixture, capture_off_is_a_no_op) {
    const luce_test::ScopedEnvVar kv_f16("LUCE_KV_F16", "1");
    Model m; REQUIRE(m.init());
    const auto emb = make_embeddings(10);

    TargetCache base; REQUIRE(make_cache(m, base));
    const UbatchRun off = run_ubatch(m, base, emb, 0, 10, false, 0, 0, 0);
    REQUIRE(!off.hidden.empty());
    CHECK(base.q_blk == nullptr);
    CHECK(base.q_blk_buf == nullptr);

    // A request for a window without a buffer adds nothing.
    TargetCache nobuf; REQUIRE(make_cache(m, nobuf));
    const UbatchRun req = run_ubatch(m, nobuf, emb, 0, 10, false, 0, 10, 0);
    CHECK(req.nodes == off.nodes);
    CHECK(req.hidden == off.hidden);

    // A buffer with count 0 adds nothing either.
    TargetCache idle; REQUIRE(make_cache(m, idle));
    REQUIRE(alloc_q_block_capture(idle, m.w, 16));
    const UbatchRun zero = run_ubatch(m, idle, emb, 0, 10, false, 0, 0, 0);
    CHECK(zero.nodes == off.nodes);
    CHECK(zero.hidden == off.hidden);

    // Capture on leaves every output bit-identical and only adds copy nodes.
    TargetCache on; REQUIRE(make_cache(m, on));
    REQUIRE(alloc_q_block_capture(on, m.w, 16));
    const UbatchRun cap = run_ubatch(m, on, emb, 0, 10, false, 0, 10, 0);
    CHECK(cap.hidden == off.hidden);
    CHECK(cap.nodes > off.nodes);

    free_target_cache(base); free_target_cache(nobuf); free_target_cache(idle); free_target_cache(on);
    m.destroy();
}

TEST_CASE(QBlockCaptureFixture, chunks_are_cut_at_block_boundaries) {
    CHECK(q_block_chunk_tokens(0, 512, 640) == 512);
    CHECK(q_block_chunk_tokens(512, 512, 640) == 128);
    CHECK(q_block_chunk_tokens(640, 512, 640) == 512);
    CHECK(q_block_chunk_tokens(1152, 512, 640) == 128);
    CHECK(q_block_chunk_tokens(1280, 7, 640) == 7);
}

TEST_CASE(QBlockCaptureFixture, plan_marks_the_sealing_chunk_and_skips_partial_blocks) {
    // Cold prefill: chunk 512..640 is the tail of block 0 and seals it.
    QBlockPlan p = plan_q_block_window(512, 128, 640, 0);
    CHECK(p.count == 128); CHECK(p.first == 0); CHECK(p.dst == 512);
    CHECK(p.block == 0);   CHECK(p.seals);
    // Next chunk starts block 1 at row 0 and does not seal.
    p = plan_q_block_window(640, 384, 640, 0);
    CHECK(p.count == 384); CHECK(p.dst == 0); CHECK(p.block == 1); CHECK(!p.seals);
    p = plan_q_block_window(1024, 256, 640, 0);
    CHECK(p.dst == 384); CHECK(p.block == 1); CHECK(p.seals);
    // A prefill that resumes inside block 0 never captures or seals block 0 ...
    p = plan_q_block_window(100, 540, 640, 100);
    CHECK(p.count == 0); CHECK(!p.seals);
    // ... but captures block 1 in full.
    p = plan_q_block_window(640, 640, 640, 100);
    CHECK(p.count == 640); CHECK(p.dst == 0); CHECK(p.seals);
    // The last, incomplete block is captured but never sealed.
    p = plan_q_block_window(1280, 300, 640, 0);
    CHECK(p.count == 300); CHECK(!p.seals);
}
