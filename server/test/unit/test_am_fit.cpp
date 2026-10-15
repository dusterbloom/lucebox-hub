// KVFlash 2.0b M4: HighestAttnKeys scoring (common/am_fit) against an
// independent double-precision C++ reference of the compaction repo's
// _select_keys_highest_attention (score_method 'rms', GQA-pooled queries).
//
// Pins, on the CPU backend and (when present) the CUDA/HIP backend:
//   * the selected key indices are identical to the reference (as sets) on
//     synthetic K/Q: MHA T=700 n=4200 t=64, and the real-model shape
//     640 queries x 640 keys, 24 query heads over 4 KV heads, head_dim 256;
//   * chunking over queries does not change the selection;
//   * exact score ties resolve by the documented rule (higher score first,
//     lower key index among equals), including a tie group that straddles the
//     top-k cutoff and the all-keys-equal case;
//   * the GQA grouping is contiguous (query head h*G+g belongs to KV head h).
//
// usage: test_am_fit [--cpu-only] [--bench]
//   --bench: time scoring of one 640-token block at the real shape, 16 layers.
#include "common/am_fit.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using luce::am::ScoreDims;

namespace {

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
float next_uniform() {  // (0, 1)
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return ((g_rng >> 40) + 0.5f) / 16777216.0f;
}
float next_gauss() {  // Box-Muller
    const float u = next_uniform(), v = next_uniform();
    return std::sqrt(-2.0f * std::log(u)) * std::cos(6.2831853f * v);
}

int g_fail = 0;
void check(bool ok, const std::string & what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

struct Problem {
    ScoreDims dims;
    std::vector<ggml_fp16_t> q;  // [d, n, n_head]
    std::vector<ggml_fp16_t> k;  // [d, T, n_kv]
};

Problem make_problem(int d, int n_head, int n_kv, int T, int n) {
    Problem p;
    p.dims = { d, n_head, n_kv, T, n };
    p.q.resize((size_t) d * n * n_head);
    p.k.resize((size_t) d * T * n_kv);
    for (auto & x : p.q) x = ggml_fp32_to_fp16(next_gauss());
    // Keys with a spread of norms so the attention mass is not uniform.
    for (int h = 0; h < n_kv; ++h)
        for (int j = 0; j < T; ++j) {
            const float scale = std::exp(0.5f * next_gauss());
            for (int c = 0; c < d; ++c)
                p.k[((size_t) h * T + j) * d + c] = ggml_fp32_to_fp16(scale * next_gauss());
        }
    return p;
}

// Reference scores in double: sum over the head's pooled queries of w^2.
// scale <= 0 selects 1/sqrt(d); skip_last drops the last query row of every head
// (used only to prove a fixture notices an omitted row).
std::vector<double> ref_scores(const Problem & p, double scale = 0.0, int skip_last = 0) {
    const ScoreDims & D = p.dims;
    const int G = D.n_head / D.n_kv_head;
    const double inv = scale > 0 ? scale : 1.0 / std::sqrt((double) D.head_dim);
    std::vector<double> out((size_t) D.n_kv_head * D.n_keys, 0.0);
    std::vector<double> kd((size_t) D.n_keys * D.head_dim), s(D.n_keys);
    for (int h = 0; h < D.n_kv_head; ++h) {
        for (size_t i = 0; i < kd.size(); ++i)
            kd[i] = ggml_fp16_to_fp32(p.k[(size_t) h * D.n_keys * D.head_dim + i]);
        for (int g = 0; g < G; ++g) {
            const int head = h * G + g;
            for (int i = 0; i < D.n_queries - skip_last; ++i) {
                const ggml_fp16_t * qr = &p.q[((size_t) head * D.n_queries + i) * D.head_dim];
                double mx = -1e300;
                for (int j = 0; j < D.n_keys; ++j) {
                    double a = 0;
                    for (int c = 0; c < D.head_dim; ++c)
                        a += (double) ggml_fp16_to_fp32(qr[c]) * kd[(size_t) j * D.head_dim + c];
                    s[j] = a * inv;
                    mx = std::max(mx, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < D.n_keys; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
                for (int j = 0; j < D.n_keys; ++j) {
                    const double w = s[j] / sum;
                    out[(size_t) h * D.n_keys + j] += w * w;
                }
            }
        }
    }
    return out;
}

// Reference selection, written independently of select_top_k: full stable sort.
std::vector<int32_t> ref_topk(const double * sc, int n, int k) {
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[a] > sc[b]; });
    idx.resize(k);
    return idx;
}

std::vector<int32_t> sorted_copy(std::vector<int32_t> v) { std::sort(v.begin(), v.end()); return v; }

struct Backend { const char * name; ggml_backend_t be; };

// Runs one problem on one backend and compares with the reference.
// strict_order: the rank sequence (not only the set) must match.
void run_case(const Backend & b, const std::string & name, const Problem & p, int top_k, int chunk_rows,
              bool strict_order, const std::vector<double> & ref) {
    const ScoreDims & D = p.dims;
    std::vector<int32_t> got;
    std::string err;
    const bool ok = luce::am::select_keys(b.be, D, p.q.data(), GGML_TYPE_F16, p.k.data(), top_k, chunk_rows, got, &err);
    const std::string tag = std::string(b.name) + " " + name;
    if (!ok) { check(false, tag + " (" + err + ")"); return; }
    int bad_heads = 0, bad_order = 0;
    for (int h = 0; h < D.n_kv_head; ++h) {
        const std::vector<int32_t> want = ref_topk(&ref[(size_t) h * D.n_keys], D.n_keys, top_k);
        const std::vector<int32_t> have(got.begin() + (size_t) h * top_k, got.begin() + (size_t) (h + 1) * top_k);
        if (sorted_copy(want) != sorted_copy(have)) ++bad_heads;
        if (want != have) ++bad_order;
    }
    check(bad_heads == 0 && (!strict_order || bad_order == 0),
          tag + " heads=" + std::to_string(D.n_kv_head) + " set_mismatch=" + std::to_string(bad_heads) +
          " rank_order_mismatch=" + std::to_string(bad_order));
}

void test_select_top_k_pure() {
    using luce::am::select_top_k;
    {
        const float s[] = { 1, 5, 3, 5, 2 };
        int32_t o[3];
        select_top_k(s, 5, 3, o);
        check(o[0] == 1 && o[1] == 3 && o[2] == 2, "select_top_k tie on 5.0 -> lower index first, then 3.0");
    }
    {
        const float s[] = { 2, 2, 2, 2, 2, 2 };
        int32_t o[4];
        select_top_k(s, 6, 4, o);
        check(o[0] == 0 && o[1] == 1 && o[2] == 2 && o[3] == 3, "select_top_k all equal -> indices 0..k-1");
    }
    {
        const float s[] = { 3, 1, 3, 3, 1, 3, 0 };  // tie group {0,2,3,5} straddles k=2
        int32_t o[2];
        select_top_k(s, 7, 2, o);
        check(o[0] == 0 && o[1] == 2, "select_top_k tie group straddling the cutoff keeps the lowest indices");
    }
    {
        const float s[] = { -1, -3, -2 };
        int32_t o[3];
        select_top_k(s, 3, 3, o);
        check(o[0] == 0 && o[1] == 2 && o[2] == 1, "select_top_k k == n sorts negatives descending");
    }
    {
        const float s[] = { 0.5f };
        int32_t o[1];
        select_top_k(s, 1, 1, o);
        check(o[0] == 0, "select_top_k n == k == 1");
    }
}

void run_backend_suite(const Backend & b) {
    g_rng = 0x9e3779b97f4a7c15ull;  // same fixtures on every backend
    // MHA, T = 700 keys, n = 4200 queries, t = 64.
    {
        Problem p = make_problem(128, 1, 1, 700, 4200);
        const auto ref = ref_scores(p);
        run_case(b, "MHA T=700 n=4200 t=64", p, 64, 8192, false, ref);
        run_case(b, "MHA T=700 n=4200 t=64 chunk_rows=1000", p, 64, 1000, false, ref);
        run_case(b, "MHA T=700 n=4200 t=64 chunk_rows=1", p, 64, 1, false, ref);
    }
    // Real shape: 640 x 640, 24 query heads over 4 KV heads, head_dim 256.
    Problem real = make_problem(256, 24, 4, 640, 640);
    const auto real_ref = ref_scores(real);
    run_case(b, "GQA real shape 640x640 24/4 d=256 t=64", real, 64, 8192, false, real_ref);
    run_case(b, "GQA real shape chunk_rows=1000 (41 queries/chunk, ragged tail)", real, 64, 1000, false, real_ref);
    // The GQA grouping must be contiguous: an interleaved mapping (head % n_kv)
    // gives different selections on the same data, so the reference match above
    // is only possible with h*G+g -> h.
    {
        Problem wrong = real;  // same K, queries permuted into the interleaved mapping
        const int G = 6, nkv = 4, nq = 640, d = 256;
        for (int head = 0; head < 24; ++head) {
            const int h = head / G, g = head % G;
            const int src = g * nkv + h;  // interleaved layout holding the same heads
            std::memcpy(&wrong.q[(size_t) src * nq * d], &real.q[(size_t) head * nq * d], (size_t) nq * d * sizeof(ggml_fp16_t));
        }
        std::vector<int32_t> a, c;
        std::string err;
        const bool ok = luce::am::select_keys(b.be, real.dims, real.q.data(), GGML_TYPE_F16, real.k.data(), 64, 8192, a, &err) &&
                        luce::am::select_keys(b.be, wrong.dims, wrong.q.data(), GGML_TYPE_F16, wrong.k.data(), 64, 8192, c, &err);
        check(ok && a != c, std::string(b.name) + " GQA grouping is contiguous (permuted heads change the selection)");
    }
    // Exact ties: 106 identical boosted keys (every 6th key from 5) dominate;
    // k = 64 cuts inside the tie group, so the lowest 64 group indices win.
    {
        Problem p = make_problem(256, 24, 4, 640, 640);
        std::vector<int> group;
        for (int j = 5; j < 640; j += 6) group.push_back(j);
        // Give every query a shared component so the group dominates the attention.
        std::vector<float> dir(256, 0.0f);
        for (float & x : dir) x = 2.0f * next_gauss();
        for (int head = 0; head < 24; ++head)
            for (int i = 0; i < 640; ++i)
                for (int c = 0; c < 256; ++c) {
                    ggml_fp16_t & x = p.q[((size_t) head * 640 + i) * 256 + c];
                    x = ggml_fp32_to_fp16(ggml_fp16_to_fp32(x) + dir[c]);
                }
        double nrm = 0;
        for (float x : dir) nrm += (double) x * x;
        nrm = std::sqrt(nrm);
        for (int h = 0; h < 4; ++h)
            for (int j : group)
                for (int c = 0; c < 256; ++c)
                    p.k[((size_t) h * 640 + j) * 256 + c] = ggml_fp32_to_fp16((float) (dir[c] / nrm * 24.0));
        const auto ref = ref_scores(p);
        run_case(b, "ties: 106-key identical group straddling t=64", p, 64, 8192, true, ref);
        run_case(b, "ties: same, chunk_rows=1000", p, 64, 1000, true, ref);
        std::vector<int32_t> got;
        std::string err;
        const bool ok = luce::am::select_keys(b.be, p.dims, p.q.data(), GGML_TYPE_F16, p.k.data(), 64, 8192, got, &err);
        bool want_group = ok;
        for (int h = 0; ok && h < 4; ++h)
            for (int r = 0; r < 64; ++r)
                want_group = want_group && got[(size_t) h * 64 + r] == group[r];
        check(want_group, std::string(b.name) + " ties: selection is exactly the 64 lowest group indices, in index order");
    }
    // All keys identical: every score ties, expect indices 0..t-1.
    {
        Problem p = make_problem(128, 8, 2, 300, 200);
        for (int h = 0; h < 2; ++h)
            for (int j = 1; j < 300; ++j)
                std::memcpy(&p.k[((size_t) h * 300 + j) * 128], &p.k[(size_t) h * 300 * 128], 128 * sizeof(ggml_fp16_t));
        const auto ref = ref_scores(p);
        run_case(b, "ties: all 300 keys identical, t=64", p, 64, 8192, true, ref);
    }
}


// Raw-score agreement metric: worst |got-ref| / (1e-4*ref + 1e-6*max(ref)) over
// all keys; <= 1 means every score is within 1e-4 relative (plus a 1e-6 floor
// relative to the largest score of the problem).
double score_err(const std::vector<float> & got, const std::vector<double> & ref) {
    if (got.size() != ref.size()) return 1e30;
    double mx = 0;
    for (double r : ref) mx = std::max(mx, r);
    double worst = 0;
    for (size_t i = 0; i < ref.size(); ++i)
        worst = std::max(worst, std::fabs(got[i] - ref[i]) / (1e-4 * ref[i] + 1e-6 * mx));
    return worst;
}

void run_scores(const Backend & b, const std::string & name, const Problem & p, int chunk_rows,
                const std::vector<double> & ref) {
    std::vector<float> got;
    std::string err;
    const bool ok = luce::am::score_keys(b.be, p.dims, p.q.data(), GGML_TYPE_F16, p.k.data(), chunk_rows, got, &err);
    const double e = ok ? score_err(got, ref) : 1e30;
    char buf[64];
    std::snprintf(buf, sizeof buf, " err/tol=%.3g", e);
    check(ok && e <= 1.0, std::string(b.name) + " scores " + name + buf + (ok ? "" : " (" + err + ")"));
}

std::vector<float> to_float(const std::vector<double> & v) { return std::vector<float>(v.begin(), v.end()); }

void run_extra_suite(const Backend & b) {
    g_rng = 0x1234abcd5678ef01ull;  // same fixtures on every backend
    // Raw score values against the double reference, whole and chunked.
    {
        Problem p = make_problem(128, 1, 1, 700, 4200);
        const auto ref = ref_scores(p);
        run_scores(b, "MHA T=700 n=4200", p, 8192, ref);
        run_scores(b, "MHA T=700 n=4200 chunk_rows=1000", p, 1000, ref);
    }
    Problem real = make_problem(256, 24, 4, 640, 640);
    const auto real_ref = ref_scores(real);
    run_scores(b, "real shape", real, 8192, real_ref);
    run_scores(b, "real shape chunk_rows=1000", real, 1000, real_ref);
    run_scores(b, "real shape chunk_rows=24 (one query per head per chunk)", real, 24, real_ref);
    run_scores(b, "real shape chunk_rows=192 (8 queries per chunk, MMF width)", real, 192, real_ref);

    // chunk_rows below n_head cannot be honoured and must be rejected.
    {
        std::vector<float> sc;
        std::string err;
        const bool lo = luce::am::score_keys(b.be, real.dims, real.q.data(), GGML_TYPE_F16, real.k.data(), 23, sc, &err);
        check(!lo && err.find("chunk_rows") != std::string::npos, std::string(b.name) + " chunk_rows < n_head is rejected: '" + err + "'");
        std::vector<int32_t> sel;
        check(!luce::am::select_keys(b.be, real.dims, real.q.data(), GGML_TYPE_F16, real.k.data(), 64, 1, sel, &err),
              std::string(b.name) + " select_keys chunk_rows=1 with 24 heads is rejected");
    }

    // The last query row alone decides the top-k: every other row is zero (uniform
    // attention, equal contribution to all keys). A chunked loop that drops the
    // final row leaves all scores tied and selects keys 0..t-1.
    {
        const int d = 64, nh = 8, nkv = 2, T = 200, n = 100, t = 8, G = nh / nkv;
        Problem p = make_problem(d, nh, nkv, T, n);
        std::fill(p.q.begin(), p.q.end(), ggml_fp32_to_fp16(0.0f));
        // Eight planted identical keys per KV head; the last query row of every
        // head in the group points at them. Everything else is uniform attention.
        for (int h = 0; h < nkv; ++h) {
            std::vector<ggml_fp16_t> v(d);
            for (auto & x : v) x = ggml_fp32_to_fp16(2.0f * next_gauss());
            for (int i = 0; i < t; ++i)
                std::memcpy(&p.k[((size_t) h * T + 40 + 17 * i + h) * d], v.data(), d * sizeof(ggml_fp16_t));
            for (int g = 0; g < G; ++g)
                std::memcpy(&p.q[((size_t) (h * G + g) * n + (n - 1)) * d], v.data(), d * sizeof(ggml_fp16_t));
        }
        const auto ref = ref_scores(p);
        const auto ref_dropped = ref_scores(p, 0.0, 1);
        bool sensitive = true;
        for (int h = 0; h < nkv; ++h) {
            auto order = ref_topk(&ref[(size_t) h * T], T, t + 1);
            auto c = sorted_copy(ref_topk(&ref_dropped[(size_t) h * T], T, t));
            const double s8 = ref[(size_t) h * T + order[t - 1]], s9 = ref[(size_t) h * T + order[t]];
            // the last row picks a top-8 that differs from the all-tied one, with a
            // rank-8/rank-9 gap far above the fp32-vs-double noise
            sensitive = sensitive && sorted_copy(std::vector<int32_t>(order.begin(), order.begin() + t)) != c &&
                        (s8 - s9) > 1e-3 * s8;
        }
        check(sensitive, "last-row fixture: dropping the last row changes the top-8, rank-8/9 gap > 1e-3 relative");
        for (int rows : { 8192, 8, 24 }) {
            run_case(b, "last-row fixture chunk_rows=" + std::to_string(rows), p, t, rows, false, ref);
            run_scores(b, "last-row fixture chunk_rows=" + std::to_string(rows), p, rows, ref);
        }
    }
    // Scale sensitivity: sharp logits so that 1/sqrt(d) vs 1/d vs no scale give
    // scores that differ by far more than the tolerance.
    {
        Problem p = make_problem(64, 4, 2, 128, 64);
        for (auto & x : p.q) x = ggml_fp32_to_fp16(2.0f * ggml_fp16_to_fp32(x));
        const auto ref = ref_scores(p);
        const double e_invd = score_err(to_float(ref_scores(p, 1.0 / 64)), ref);
        const double e_one = score_err(to_float(ref_scores(p, 1.0)), ref);
        check(e_invd > 100 && e_one > 100, "scale fixture: 1/d and unscaled scores are far outside the tolerance");
        run_scores(b, "scale fixture (1/sqrt(d))", p, 8192, ref);
        run_case(b, "scale fixture", p, 16, 8192, false, ref);
    }
    // Tensor entry point on device-resident planes of 4D buffers shaped like
    // cache.q_blk [d, 640, n_head, n_layers]: plane 1 holds the real data, the
    // neighbours hold other data, so a wrong plane offset changes the result.
    {
        const int d = 256, nh = 24, nkv = 4, T = 640, n = 640, planes = 3, plane = 1;
        Problem other = make_problem(d, nh, nkv, T, n);
        ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n, nh, planes);
        ggml_tensor * k4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, T, nkv, planes);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, b.be);
        for (int pl = 0; pl < planes; ++pl) {
            const Problem & src = pl == plane ? real : other;
            ggml_backend_tensor_set(q4, src.q.data(), pl * q4->nb[3], q4->nb[3]);
            ggml_backend_tensor_set(k4, src.k.data(), pl * k4->nb[3], k4->nb[3]);
        }
        ggml_context * vctx = ggml_init(ip);
        ggml_tensor * qv = ggml_view_3d(vctx, q4, d, n, nh, q4->nb[1], q4->nb[2], plane * q4->nb[3]);
        ggml_tensor * kv = ggml_view_3d(vctx, k4, d, T, nkv, k4->nb[1], k4->nb[2], plane * k4->nb[3]);
        std::vector<float> sc;
        std::string err;
        const bool ok = luce::am::score_keys(b.be, qv, kv, 8192, sc, &err);
        const double e = ok ? score_err(sc, real_ref) : 1e30;
        check(ok && e <= 1.0, std::string(b.name) + " tensor form on plane views of 4D device buffers matches the reference " + err);
        std::vector<float> host_sc;
        luce::am::score_keys(b.be, real.dims, real.q.data(), GGML_TYPE_F16, real.k.data(), 8192, host_sc, &err);
        check(ok && sc == host_sc, std::string(b.name) + " tensor form is bit-identical to the host-pointer form");
        std::vector<int32_t> sel;
        const bool sok = luce::am::select_keys(b.be, qv, kv, 64, 1000, sel, &err);
        bool same = sok;
        for (int h = 0; sok && h < nkv; ++h) {
            auto want = sorted_copy(ref_topk(&real_ref[(size_t) h * T], T, 64));
            same = same && want == sorted_copy(std::vector<int32_t>(sel.begin() + h * 64, sel.begin() + (h + 1) * 64));
        }
        check(same, std::string(b.name) + " tensor-form select_keys matches the reference");
        ggml_backend_buffer_free(buf);
        ggml_free(vctx);
        ggml_free(ctx);
    }
}

// General (not F16-representable) F32 inputs through the tensor form, against a double
// reference that sees the same float values. Catches matmul paths that round the logits
// below fp32 (F16 output, TF32 tensor cores) on narrow chunks, which F16-exact test data hides.
void run_f32_general(const Backend & b) {
    g_rng = 0x51ed270b8a1c3d29ull;
    const int d = 256, nh = 24, nkv = 4, T = 640, n = 640;
    std::vector<float> q((size_t) d * n * nh), kf((size_t) d * T * nkv);
    for (float & x : q) x = next_gauss();
    for (int h = 0; h < nkv; ++h)
        for (int j = 0; j < T; ++j) {
            const float scale = std::exp(0.5f * next_gauss());
            for (int c = 0; c < d; ++c) kf[((size_t) h * T + j) * d + c] = scale * next_gauss();
        }
    for (ggml_type kt : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        std::vector<ggml_fp16_t> k16;
        std::vector<float> kref = kf;  // what the backend will see
        if (kt == GGML_TYPE_F16) {
            k16.resize(kf.size());
            for (size_t i = 0; i < kf.size(); ++i) { k16[i] = ggml_fp32_to_fp16(kf[i]); kref[i] = ggml_fp16_to_fp32(k16[i]); }
        }
        // double reference on the exact float values
        const int G = nh / nkv;
        std::vector<double> ref((size_t) nkv * T, 0.0), s(T);
        for (int h = 0; h < nkv; ++h)
            for (int g = 0; g < G; ++g)
                for (int i = 0; i < n; ++i) {
                    double mx = -1e300;
                    for (int j = 0; j < T; ++j) {
                        double a = 0;
                        for (int c = 0; c < d; ++c) a += (double) q[((size_t) (h * G + g) * n + i) * d + c] * kref[((size_t) h * T + j) * d + c];
                        s[j] = a / std::sqrt((double) d);
                        mx = std::max(mx, s[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < T; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
                    for (int j = 0; j < T; ++j) ref[(size_t) h * T + j] += (s[j] / sum) * (s[j] / sum);
                }
        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n, nh);
        ggml_tensor * ktn = ggml_new_tensor_3d(ctx, kt, d, T, nkv);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, b.be);
        ggml_backend_tensor_set(qt, q.data(), 0, ggml_nbytes(qt));
        ggml_backend_tensor_set(ktn, kt == GGML_TYPE_F32 ? (const void *) kf.data() : (const void *) k16.data(), 0, ggml_nbytes(ktn));
        for (int rows : { 8192, 1000, 192, 24 }) {
            std::vector<float> got;
            std::string err;
            const bool ok = luce::am::score_keys(b.be, qt, ktn, rows, got, &err);
            const double e = ok ? score_err(got, ref) : 1e30;
            char nb[96];
            std::snprintf(nb, sizeof nb, " K=%s chunk_rows=%d err/tol=%.3g", kt == GGML_TYPE_F32 ? "F32" : "F16", rows, e);
            check(ok && e <= 1.0, std::string(b.name) + " general F32 Q inputs vs double reference" + nb + (ok ? "" : " " + err));
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

void run_bench(const Backend & b) {
    Problem p = make_problem(256, 24, 4, 640, 640);
    std::vector<int32_t> out;
    std::string err;
    for (int i = 0; i < 3; ++i) luce::am::select_keys(b.be, p.dims, p.q.data(), GGML_TYPE_F16, p.k.data(), 64, 8192, out, &err);
    for (int rep = 0; rep < 5; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int layer = 0; layer < 16; ++layer)
            if (!luce::am::select_keys(b.be, p.dims, p.q.data(), GGML_TYPE_F16, p.k.data(), 64, 8192, out, &err)) {
                std::printf("bench failed: %s\n", err.c_str());
                return;
            }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("BENCH %s one 640-token block, 16 layers, 24/4 heads d=256: %.2f ms (%.3f ms/layer)\n", b.name, ms, ms / 16);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    bool cpu_only = false, bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu-only")) cpu_only = true;
        else if (!std::strcmp(argv[i], "--bench")) bench = true;
    }
    test_select_top_k_pure();
    std::vector<Backend> backends;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    backends.push_back({ "cpu ", cpu });
    ggml_backend_t gpu = nullptr;
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    if (!cpu_only) {
        gpu = ggml_backend_cuda_init(0);
        if (gpu) backends.push_back({ "cuda", gpu });
        else std::printf("SKIP  no GPU backend available\n");
    }
#else
    (void) cpu_only;
#endif
    for (const Backend & b : backends) {
        if (bench) run_bench(b); else { run_backend_suite(b); run_extra_suite(b); run_f32_general(b); }
    }
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
