// ggml_flash_attn_ext_add_kv_bias differential test (CPU and CUDA/HIP).
//
// KVFlash AM compaction fits a per-KV-head key bias beta[ik2][ic] that is
// added to the scaled (and softcapped) logit of key ic for every query head
// that maps to KV head ik2. This test pins that contract for every backend
// that claims FLASH_ATTN_EXT with a kv_bias:
//   * output matches an independent double-precision reference built from the
//     dequantized K/V, within the FLASH_ATTN_EXT NMSE tolerance of
//     test-backend-ops (5e-4);
//   * a zero kv_bias is bit-identical to no kv_bias on the same backend;
//   * a -inf kv_bias column behaves like a masked key.
// A backend whose supports_op declines kv_bias is reported as UNSUPP (the op
// must never be silently ignored); set FATTN_KV_BIAS_STRICT=1 to make UNSUPP
// fail the run.
//
// usage: test_fattn_kv_bias [--cpu-only] [name-substring]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <vector>

#include <csignal>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr int GGML_KQ_MASK_PAD = 64;  // llama.cpp mask row padding
constexpr double NMSE_TOL = 5e-4;  // test-backend-ops FLASH_ATTN_EXT tolerance

uint32_t g_rng = 0x9e3779b9u;
float next_float() {  // [-1, 1)
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t) (g_rng >> 8) - 8388608) / 8388608.0f;
}

struct Shape { const char * name; int D, n_head, n_head_kv, n_kv; };
const Shape LONG_KV = {"D256 h24/4 kv4608", 256, 24, 4, 4608};  // > 4096: a chunked path must carry its chunk offset
const Shape SHAPES[] = {
    {"D256 h24/4 kv1024", 256, 24, 4, 1024},  // Qwen3.8-27B attention
    {"D64 h4/2 kv576",     64,  4, 2,  576},
    {"D72 h4/2 kv576",     72,  4, 2,  576},  // tile kernel on sm_86 (F16 only)
};
const ggml_type KV_TYPES[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_TQ3_0 };
const int N_QS[] = { 1, 16, 512 };

struct Case {
    Shape shape; ggml_type type; int n_q; bool mask; bool sinks; float softcap; float max_bias = 0.0f;
    std::string name() const {
        char b[160];
        std::snprintf(b, sizeof b, "%s %s q%d%s%s", shape.name, ggml_type_name(type), n_q,
                      mask ? " mask" : "", sinks ? " sinks" : "");
        std::string n = b;
        if (softcap > 0) n += " softcap" + std::to_string(softcap).substr(0, 4);
        if (max_bias > 0) n += " alibi";
        return n;
    }
};

struct Data {
    std::vector<float>   q;        // [D, n_q, n_head]
    std::vector<uint8_t> k, v;     // quantized rows [head_kv][kv]
    std::vector<float>   k_ref, v_ref;  // dequantized copies of k, v
    std::vector<ggml_fp16_t> mask; // [n_kv, pad(n_q)]
    std::vector<float>   sinks;    // [n_head]
    std::vector<float>   beta;     // [n_kv, n_head_kv]
    std::vector<float>   zeros;
};

Data make_data(const Case & c) {
    const Shape & s = c.shape;
    Data d;
    d.q.resize((size_t) s.D * c.n_q * s.n_head);
    for (float & x : d.q) x = next_float();
    const ggml_type_traits * tt = ggml_get_type_traits(c.type);
    const size_t rows = (size_t) s.n_kv * s.n_head_kv;
    const size_t row_bytes = ggml_row_size(c.type, s.D);
    for (int kv = 0; kv < 2; ++kv) {
        std::vector<float> f(rows * s.D);
        for (float & x : f) x = next_float();
        std::vector<uint8_t> & bytes = kv ? d.v : d.k;
        std::vector<float>   & deq   = kv ? d.v_ref : d.k_ref;
        bytes.resize(rows * row_bytes);
        deq.resize(rows * s.D);
        for (size_t r = 0; r < rows; ++r) {
            if (c.type == GGML_TYPE_F16) {
                ggml_fp16_t * dst = (ggml_fp16_t *) (bytes.data() + r * row_bytes);
                for (int i = 0; i < s.D; ++i) {
                    dst[i] = ggml_fp32_to_fp16(f[r * s.D + i]);
                    deq[r * s.D + i] = ggml_fp16_to_fp32(dst[i]);
                }
            } else {
                tt->from_float_ref(f.data() + r * s.D, bytes.data() + r * row_bytes, s.D);
                tt->to_float(bytes.data() + r * row_bytes, deq.data() + r * s.D, s.D);
            }
        }
    }
    const int n_pad = GGML_PAD(c.n_q, GGML_KQ_MASK_PAD);
    d.mask.assign((size_t) s.n_kv * n_pad, ggml_fp32_to_fp16(0.0f));
    for (int i = 0; i < n_pad; ++i) {  // causal: query i sees keys <= n_kv - n_q + i
        for (int j = 0; j < s.n_kv; ++j) {
            if (j > s.n_kv - c.n_q + i) d.mask[(size_t) i * s.n_kv + j] = ggml_fp32_to_fp16(-INFINITY);
        }
    }
    d.sinks.resize(s.n_head);
    for (float & x : d.sinks) x = next_float() * 2.0f;
    d.beta.resize((size_t) s.n_kv * s.n_head_kv);
    for (int h = 0; h < s.n_head_kv; ++h) {
        for (int j = 0; j < s.n_kv; ++j) {
            float & b = d.beta[(size_t) h * s.n_kv + j];
            b = next_float() * 3.0f;  // [-3, 3]
            // some -inf columns, never the first key (always visible)
            if (j > 0 && (next_float() > 0.8f)) b = -INFINITY;
        }
    }
    d.zeros.assign(d.beta.size(), 0.0f);
    return d;
}

// Double-precision reference. `beta` and `mask` may be null.
std::vector<float> reference(const Case & c, const Data & d, const float * beta,
                             const ggml_fp16_t * mask, bool bias_before_cap = false) {
    const Shape & s = c.shape;
    const float scale = 1.0f / std::sqrt((float) s.D);
    std::vector<float> out((size_t) s.D * s.n_head * c.n_q);
    // ALiBi slope scales the mask only, never the kv_bias.
    const int n_log2 = 1 << (int) std::floor(std::log2((double) s.n_head));
    const double m0 = std::pow(2.0, -c.max_bias / n_log2), m1 = std::pow(2.0, -(c.max_bias / 2.0) / n_log2);
    auto head = [&](int h) {
        const double slope = c.max_bias <= 0 ? 1.0 : h < n_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_log2) + 1);
        const int hk = h / (s.n_head / s.n_head_kv);
        std::vector<double> p(s.n_kv), acc(s.D);
        for (int i = 0; i < c.n_q; ++i) {
            const float * q = d.q.data() + ((size_t) h * c.n_q + i) * s.D;
            double mx = -INFINITY;
            for (int j = 0; j < s.n_kv; ++j) {
                const float * k = d.k_ref.data() + ((size_t) hk * s.n_kv + j) * s.D;
                double dot = 0.0;
                for (int e = 0; e < s.D; ++e) dot += (double) q[e] * k[e];
                double l = dot * scale;
                const double b = beta ? beta[(size_t) hk * s.n_kv + j] : 0.0;
                if (bias_before_cap) l += b;  // the wrong order, only to prove a case can tell
                if (c.softcap > 0) l = c.softcap * std::tanh(l / c.softcap);
                if (mask) l += slope * ggml_fp16_to_fp32(mask[(size_t) i * s.n_kv + j]);
                if (!bias_before_cap) l += b;
                p[j] = l;
                mx = std::max(mx, l);
            }
            if (c.sinks) mx = std::max(mx, (double) d.sinks[h]);
            double den = c.sinks ? std::exp(d.sinks[h] - mx) : 0.0;
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < s.n_kv; ++j) {
                const double w = std::exp(p[j] - mx);  // exp(-inf) == 0
                den += w;
                const float * v = d.v_ref.data() + ((size_t) hk * s.n_kv + j) * s.D;
                for (int e = 0; e < s.D; ++e) acc[e] += w * v[e];
            }
            // ggml output layout [D, n_head, n_q]
            for (int e = 0; e < s.D; ++e) out[((size_t) i * s.n_head + h) * s.D + e] = (float) (acc[e] / den);
        }
    };
    std::vector<std::future<void>> fs;
    for (int h = 0; h < s.n_head; ++h) fs.push_back(std::async(std::launch::async, head, h));
    for (auto & f : fs) f.get();
    return out;
}

enum class Run { OK, UNSUPPORTED, FAILED };

// One FLASH_ATTN_EXT on `backend`; null `mask` / `beta` = no mask / no kv_bias.
Run run_fa(ggml_backend_t backend, const Case & c, const Data & d, const float * beta,
           const ggml_fp16_t * mask, std::vector<float> & out) {
    const Shape & s = c.shape;
    ggml_init_params ip = { 16u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.D, c.n_q, s.n_head, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, c.type, s.D, s.n_kv, s.n_head_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, c.type, s.D, s.n_kv, s.n_head_kv, 1);
    ggml_tensor * m = mask ? ggml_new_tensor_2d(ctx, GGML_TYPE_F16, s.n_kv, GGML_PAD(c.n_q, GGML_KQ_MASK_PAD)) : nullptr;
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / std::sqrt((float) s.D), c.max_bias, c.softcap);
    ggml_tensor * sk = nullptr, * kb = nullptr;
    if (c.sinks) {
        sk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.n_head);
        ggml_flash_attn_ext_add_sinks(fa, sk);
    }
    if (beta) {
        kb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.n_kv, s.n_head_kv);
        ggml_flash_attn_ext_add_kv_bias(fa, kb);
    }
    Run rc = Run::OK;
    if (!ggml_backend_supports_op(backend, fa)) {
        rc = Run::UNSUPPORTED;
    } else if (ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend)) {
        ggml_backend_tensor_set(q, d.q.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, d.k.data(), 0, d.k.size());
        ggml_backend_tensor_set(v, d.v.data(), 0, d.v.size());
        if (m)  ggml_backend_tensor_set(m, mask, 0, ggml_nbytes(m));
        if (sk) ggml_backend_tensor_set(sk, d.sinks.data(), 0, ggml_nbytes(sk));
        if (kb) ggml_backend_tensor_set(kb, beta, 0, ggml_nbytes(kb));
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, fa);
        if (ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS) {
            ggml_backend_synchronize(backend);
            out.resize(ggml_nelements(fa));
            ggml_backend_tensor_get(fa, out.data(), 0, out.size() * sizeof(float));
        } else {
            rc = Run::FAILED;
        }
        ggml_backend_buffer_free(buf);
    } else {
        rc = Run::FAILED;
    }
    ggml_free(ctx);
    return rc;
}

struct Err { double nmse, max_abs; };
Err compare(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double e = (double) a[i] - ref[i];
        num += e * e; den += (double) ref[i] * ref[i];
        mx = std::max(mx, std::fabs(e));
        if (std::isnan(a[i])) return { INFINITY, INFINITY };
    }
    return { num / std::max(den, 1e-30), mx };
}

struct Tally { int pass = 0, fail = 0, unsupp = 0, skip = 0; double worst_abs = 0, worst_nmse = 0; };

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void run_case(const char * bname, ggml_backend_t be, const Case & c, const Data & d,
              const std::vector<float> & ref_bias, const std::vector<float> & ref_inf,
              const std::vector<float> & beta_inf, Tally & t) {
    const ggml_fp16_t * mask = c.mask ? d.mask.data() : nullptr;
    std::vector<float> base, biased, zero, inf;
    const Run r0 = run_fa(be, c, d, nullptr, mask, base);
    if (r0 == Run::FAILED) {  // a crash/alloc failure is never a skip
        ++t.fail;
        std::printf("FAIL    %-5s %s (no-bias compute failed)\n", bname, c.name().c_str());
        return;
    }
    if (r0 == Run::UNSUPPORTED) {
        ++t.skip;
        std::printf("SKIP    %-5s %s (no-bias op unsupported)\n", bname, c.name().c_str());
        return;
    }
    const Run rb = run_fa(be, c, d, d.beta.data(), mask, biased);
    if (rb == Run::UNSUPPORTED) {
        ++t.unsupp;
        std::printf("UNSUPP  %-5s %s\n", bname, c.name().c_str());
        return;
    }
    const Run rz = run_fa(be, c, d, d.zeros.data(), mask, zero);
    const Run ri = run_fa(be, c, d, beta_inf.data(), mask, inf);
    if (rb != Run::OK || rz != Run::OK || ri != Run::OK) {
        ++t.fail;
        std::printf("FAIL    %-5s %s (compute failed)\n", bname, c.name().c_str());
        return;
    }
    const Err e = compare(biased, ref_bias);
    const Err ei = compare(inf, ref_inf);
    const bool identity = bit_equal(zero, base);
    const bool ok = e.nmse <= NMSE_TOL && ei.nmse <= NMSE_TOL && identity;
    t.worst_abs = std::max({ t.worst_abs, e.max_abs, ei.max_abs });
    t.worst_nmse = std::max({ t.worst_nmse, e.nmse, ei.nmse });
    ++(ok ? t.pass : t.fail);
    std::printf("%s %-5s %-44s nmse %.2e abs %.2e | -inf nmse %.2e abs %.2e | zero-bias %s\n",
                ok ? "PASS   " : "FAIL   ", bname, c.name().c_str(), e.nmse, e.max_abs, ei.nmse,
                ei.max_abs, identity ? "bit-identical" : "DIFFERS");
}

// src[9] is shared with the DS4 lane KV: add_kv_bias(op, nullptr) on a DS4 op
// must abort instead of silently clearing the lane KV, and the accessor must
// not mistake the lane KV for a bias.
bool test_lane_kv_guard() {
    ggml_context * ctx = ggml_init({ 16u << 20, nullptr, true });
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 2, 4);
    ggml_tensor * kv = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 128, 1);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, kv, kv, nullptr, 0.125f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_ds4_lane_rows(fa, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2),
                                          ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 2));
    ggml_tensor * k2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 128, 2);
    ggml_tensor * v2 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 128, 2);
    ggml_tensor * fb = ggml_flash_attn_ext(ctx, q, k2, v2, nullptr, 0.125f, 0.0f, 0.0f);
    ggml_tensor * kb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 2);
    ggml_flash_attn_ext_add_kv_bias(fb, kb);
    const bool acc_ok = ggml_flash_attn_ext_get_kv_bias(fa) == nullptr &&
                        ggml_flash_attn_ext_get_kv_bias(fb) == kb;
    const bool lane_kept = fa->src[9] != nullptr;
#ifdef _WIN32
    // The abort check needs fork/waitpid; skip it on Windows.
    ggml_free(ctx);
    std::printf("%s lane-kv guard: accessor %s, lane kv kept %s, abort case SKIPPED on Windows\n",
                acc_ok && lane_kept ? "PASS   " : "FAIL   ", acc_ok ? "ok" : "WRONG", lane_kept ? "yes" : "NO");
    return acc_ok && lane_kept;
#else
    fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        std::freopen("/dev/null", "w", stderr);  // the GGML_ASSERT message is expected
        ggml_flash_attn_ext_add_kv_bias(fa, nullptr);
        _exit(0);  // returned: the guard is missing
    }
    int st = 0;
    waitpid(pid, &st, 0);
    const bool aborted = WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
    ggml_free(ctx);
    const bool ok = acc_ok && lane_kept && aborted;
    std::printf("%s lane-kv guard: accessor %s, lane kv kept %s, add_kv_bias(nullptr) on DS4 op %s\n",
                ok ? "PASS   " : "FAIL   ", acc_ok ? "ok" : "WRONG", lane_kept ? "yes" : "NO",
                aborted ? "aborts" : "RETURNED");
    return ok;
#endif
}

// A maskless op with QSA selected ids (src[5]) and a kv_bias: no kernel honours that pair (the QSA predicates
// reject kv_bias, the generic kernels abort on maskless selected ids), so supports_op must decline it and the
// scheduler never hands it to the backend.
bool test_qsa_bias_not_admitted(ggml_backend_t gpu) {
    bool ok = true;
    for (int nq : { 1, 128 }) {
        ggml_context * ctx = ggml_init({ 16u << 20, nullptr, true });
        ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, nq, 24, 1);
        ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 1024, 2, 1);
        ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 1024, 2, 1);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 0.0625f, 0.0f, 0.0f);
        fa->src[5] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 64, nq);  // QSA selected ids
        ggml_flash_attn_ext_add_kv_bias(fa, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 2));
        const bool admitted = ggml_backend_supports_op(gpu, fa);
        std::printf("%s qsa ids + kv_bias, %d queries: gpu supports_op %s\n",
                    admitted ? "FAIL   " : "PASS   ", nq, admitted ? "ADMITS (dispatch would abort)" : "declines");
        ok = ok && !admitted;
        ggml_free(ctx);
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    bool cpu_only = false;
    std::string filter;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu-only")) cpu_only = true; else filter = argv[i];
    }
    const bool guard_ok = test_lane_kv_guard();
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = nullptr;
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    if (!cpu_only) {
        gpu = ggml_backend_cuda_init(0);
        if (!gpu) { std::fprintf(stderr, "no GPU backend\n"); return 1; }
        ggml_backend_cuda_set_graphs_disabled_override(true);
    }
#endif
    const bool qsa_ok = !gpu || test_qsa_bias_not_admitted(gpu);
    std::vector<Case> cases;
    for (const Shape & s : SHAPES)
        for (ggml_type ty : KV_TYPES) {
            if (s.D % ggml_blck_size(ty) != 0) continue;  // e.g. D72 only fits F16
            for (int nq : N_QS)
                for (int mask = 0; mask < 2; ++mask)
                    for (int sinks = 0; sinks < 2; ++sinks)
                        cases.push_back({ s, ty, nq, (bool) mask, (bool) sinks, 0.0f });
        }
    // softcap: the bias is added after the cap (D64 softcap has no mma kernel)
    for (ggml_type ty : { GGML_TYPE_F16, GGML_TYPE_Q8_0 })
        for (int nq : N_QS) cases.push_back({ SHAPES[0], ty, nq, true, false, 20.0f });

    // ALiBi: slope scales the mask, not the bias (the chunked path refuses ALiBi: UNSUPP there)
    for (ggml_type ty : { GGML_TYPE_F16, GGML_TYPE_Q8_0 })
        for (int nq : N_QS) cases.push_back({ SHAPES[0], ty, nq, true, false, 0.0f, 8.0f });
    // saturating softcap: bias-before-cap is clamped to +-cap, so only the right order matches
    for (ggml_type ty : { GGML_TYPE_F16, GGML_TYPE_Q8_0 })
        for (int nq : N_QS) cases.push_back({ SHAPES[0], ty, nq, true, false, 0.1f });
    // kv beyond one 4096 chunk
    for (ggml_type ty : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0 })
        for (int nq : N_QS) cases.push_back({ LONG_KV, ty, nq, true, false, 0.0f });

    int fail_nodisc = 0;
    Tally tc, tg;
    for (const Case & c : cases) {
        if (!filter.empty() && c.name().find(filter) == std::string::npos) continue;
        const Data d = make_data(c);
        std::vector<float> beta_inf(d.beta.size(), 0.0f);  // every 5th key (j > 0) is -inf
        for (size_t i = 0; i < beta_inf.size(); ++i) if (i % c.shape.n_kv > 0 && i % 5 == 0) beta_inf[i] = -INFINITY;
        const ggml_fp16_t * mask = c.mask ? d.mask.data() : nullptr;
        const std::vector<float> ref_bias = reference(c, d, d.beta.data(), mask);
        const std::vector<float> ref_inf  = reference(c, d, beta_inf.data(), mask);
        if (c.softcap > 0 && c.softcap < 1.0f) {  // the case must fail the wrong order, or it proves nothing
            const Err w = compare(reference(c, d, d.beta.data(), mask, true), ref_bias);
            std::printf("        %s: bias-before-cap reference differs by nmse %.2e (tolerance %.0e)\n", c.name().c_str(), w.nmse, NMSE_TOL);
            if (w.nmse <= 10 * NMSE_TOL) {
                ++fail_nodisc;
                std::printf("FAIL    %s: wrong cap order is not distinguishable (nmse %.2e)\n", c.name().c_str(), w.nmse);
            }
        }
        // ggml-cpu's FA calls K's vec_dot unchecked; TQ3_0 has none on CPU.
        if (ggml_get_type_traits_cpu(c.type)->vec_dot == nullptr) {
            ++tc.skip;
            std::printf("SKIP    cpu   %s (no CPU vec_dot)\n", c.name().c_str());
        } else {
            run_case("cpu", cpu, c, d, ref_bias, ref_inf, beta_inf, tc);
        }
        if (gpu) run_case("gpu", gpu, c, d, ref_bias, ref_inf, beta_inf, tg);
    }
    std::printf("\nSUMMARY cpu: pass %d fail %d unsupp %d skip %d, max abs err %.3e, max nmse %.3e\n",
                tc.pass, tc.fail, tc.unsupp, tc.skip, tc.worst_abs, tc.worst_nmse);
    if (gpu) std::printf("SUMMARY gpu: pass %d fail %d unsupp %d skip %d, max abs err %.3e, max nmse %.3e\n",
                tg.pass, tg.fail, tg.unsupp, tg.skip, tg.worst_abs, tg.worst_nmse);
    const bool strict = std::getenv("FATTN_KV_BIAS_STRICT") != nullptr;
    const int bad = !guard_ok + !qsa_ok + fail_nodisc + tc.fail + tg.fail + tc.unsupp + (strict ? tg.unsupp : 0);
    return bad ? 1 : 0;
}
