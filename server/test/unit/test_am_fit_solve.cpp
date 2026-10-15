// AM fit solver tests (KVFlash 2.0b M5): CPU and CUDA/HIP against a standalone fp64 reference.
// The reference in this file (CGS2 least squares, projected gradient on the explicit design matrix,
// scores and softmax in double) shares no code with the production solvers.
//
//   1. host solvers (both routes): known w and C2 recovered to 1e-6;
//   2. parity with the reference: 2 projected-gradient steps at step 1/L after a clamp that leaves a
//      nonzero gradient (a wrong step count or step size is detected);
//   3. near-dependent but full-rank M does not trigger the ridge; exactly dependent M does; NaN fails;
//   4. backend design matrices and Gram sums match the reference, incl. softmax tails down to 1e-30;
//   5. end to end on duplicated keys (known w, known C2); fit_block vs the reference pipeline;
//   6. a NaN head fails the whole block and publishes nothing;
//   7. the device-tensor entry point gives the host path's result with less traffic;
//   8. real-data gate, only when LUCE_AM_FIXTURES points at the dumped fixtures.
//
// usage: test_am_fit_solve [--cpu-only] [--time]
#include "common/am_fit_solve.h"
#include "am_fixture.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace luce_am;

namespace {

int g_fail = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            ++g_fail;                                                      \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);             \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
        }                                                                  \
    } while (0)

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
double rnd01() {  // [0, 1)
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return (double) (g_rng >> 11) / 9007199254740992.0;
}
double rnd_norm() {
    const double u1 = rnd01() + 1e-300, u2 = rnd01();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

float round_bf16(float x) {  // round to nearest even
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&x, &u, 4);
    return x;
}

// ==== standalone fp64 reference ====================================================================

// Row-major n x t matrix helpers.
struct Mat {
    int n = 0, m = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int n_, int m_) : n(n_), m(m_), a((size_t) n_ * m_, 0.0) {}
    double & at(int i, int j) { return a[(size_t) i * m + j]; }
    double at(int i, int j) const { return a[(size_t) i * m + j]; }
};

// Least squares min |A X - B| by classical Gram-Schmidt with reorthogonalisation (CGS2).
// False when A is numerically rank deficient (a column is orthogonal to the span of the others).
bool ref_lstsq(const Mat & A, const Mat & B, Mat & X) {
    const int n = A.n, t = A.m, m = B.m;
    std::vector<std::vector<double>> Q(t, std::vector<double>(n));
    Mat R(t, t);
    for (int j = 0; j < t; ++j) {
        std::vector<double> v(n);
        for (int r = 0; r < n; ++r) v[r] = A.at(r, j);
        double n0 = 0;
        for (double x : v) n0 += x * x;
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < j; ++i) {
                double c = 0;
                for (int r = 0; r < n; ++r) c += Q[i][r] * v[r];
                for (int r = 0; r < n; ++r) v[r] -= c * Q[i][r];
                R.at(i, j) += c;
            }
        }
        double nv = 0;
        for (double x : v) nv += x * x;
        if (!(std::sqrt(nv) > 1e-13 * std::sqrt(n0))) return false;
        R.at(j, j) = std::sqrt(nv);
        for (int r = 0; r < n; ++r) Q[j][r] = v[r] / R.at(j, j);
    }
    X = Mat(t, m);
    for (int c = 0; c < m; ++c) {
        std::vector<double> qb(t);
        for (int i = 0; i < t; ++i) {
            double s = 0;
            for (int r = 0; r < n; ++r) s += Q[i][r] * B.at(r, c);
            qb[i] = s;
        }
        for (int i = t - 1; i >= 0; --i) {
            double s = qb[i];
            for (int j = i + 1; j < t; ++j) s -= R.at(i, j) * X.at(j, c);
            X.at(i, c) = s / R.at(i, i);
        }
    }
    return true;
}

// The reference's NNLS step (`_nnls_pg`): lstsq init, clamp, power iteration on M, pg_iters steps of 1/L.
// step_scale multiplies the step (1 = the reference).
std::vector<double> ref_beta(const Mat & M, const std::vector<double> & y, double lo, double hi, int pg_iters,
                             int power_iters, double step_scale = 1.0) {
    const int n = M.n, t = M.m;
    Mat Y(n, 1);
    for (int r = 0; r < n; ++r) Y.at(r, 0) = y[r];
    Mat W;
    if (!ref_lstsq(M, Y, W)) return {};
    std::vector<double> w(t);
    for (int i = 0; i < t; ++i) w[i] = std::min(std::max(W.at(i, 0), lo), hi);
    std::vector<double> u(t, 1.0 / std::sqrt((double) t)), v(n);
    auto mu = [&](const std::vector<double> & x, std::vector<double> & out) {
        for (int r = 0; r < n; ++r) { double s = 0; for (int i = 0; i < t; ++i) s += M.at(r, i) * x[i]; out[r] = s; }
    };
    auto mtv = [&](const std::vector<double> & x, std::vector<double> & out) {
        for (int i = 0; i < t; ++i) { double s = 0; for (int r = 0; r < n; ++r) s += M.at(r, i) * x[r]; out[i] = s; }
    };
    for (int it = 0; it < power_iters; ++it) {
        mu(u, v);
        double nv = 0;
        for (double x : v) nv += x * x;
        nv = std::sqrt(nv);
        for (double & x : v) x /= nv;
        mtv(v, u);
        double nu = 0;
        for (double x : u) nu += x * x;
        nu = std::sqrt(nu);
        for (double & x : u) x /= nu;
    }
    mu(u, v);
    double sig2 = 0;
    for (double x : v) sig2 += x * x;
    const double L = std::max(sig2, 1e-6);
    std::vector<double> res(n), grad(t);
    for (int it = 0; it < pg_iters; ++it) {
        mu(w, res);
        for (int r = 0; r < n; ++r) res[r] -= y[r];
        mtv(res, grad);
        for (int i = 0; i < t; ++i) w[i] = std::min(std::max(w[i] - step_scale * grad[i] / L, lo), hi);
    }
    for (double & x : w) x = std::log(x);
    return w;
}

// Column-major copy of a row-major matrix (the solvers' input layout).
std::vector<double> to_cm(const Mat & A) {
    std::vector<double> o((size_t) A.n * A.m);
    for (int i = 0; i < A.n; ++i)
        for (int j = 0; j < A.m; ++j) o[(size_t) j * A.n + i] = A.at(i, j);
    return o;
}

void gram_of(const Mat & A, const Mat & B, std::vector<double> & G, std::vector<double> & AtB) {
    const int n = A.n, t = A.m, d = B.m;
    G.assign((size_t) t * t, 0.0);
    AtB.assign((size_t) t * d, 0.0);
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < t; ++i) {
            const double ai = A.at(r, i);
            for (int j = 0; j < t; ++j) G[(size_t) i * t + j] += ai * A.at(r, j);
            for (int k = 0; k < d; ++k) AtB[(size_t) i * d + k] += ai * B.at(r, k);
        }
}

// ==== 1. host recovery ============================================================================

void test_host_recovery(int n, int t, int d, SolveRoute route) {
    std::printf("[host recovery %s n=%d t=%d d=%d]\n", route == SolveRoute::Qr ? "qr" : "gram", n, t, d);
    NnlsParams p;
    p.route = route;
    Mat M(n, t), Y1(n, 1);
    std::vector<double> w(t), y(n);
    for (auto & v : M.a) v = 0.05 + rnd01();
    for (auto & v : w) v = std::exp(-2.5 + 5.0 * rnd01());
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += M.at(r, i) * w[i];
        y[r] = s;
        Y1.at(r, 0) = s;
    }
    std::vector<double> beta(t);
    SolveStatus st;
    if (route == SolveRoute::Qr) {
        st = solve_beta_qr(to_cm(M).data(), y.data(), n, t, p, beta.data());
    } else {
        std::vector<double> G, b;
        gram_of(M, Y1, G, b);
        st = solve_beta(G.data(), b.data(), t, p, beta.data());
    }
    CHECK(st == SolveStatus::Ok, "status %d", (int) st);
    double worst = 0;
    for (int i = 0; i < t; ++i) worst = std::max(worst, std::fabs(beta[i] - std::log(w[i])));
    CHECK(worst < 1e-6, "beta error %.3e", worst);

    Mat X(n, t), C2(t, d), Yv(n, d);
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += (X.at(r, i) = std::exp(rnd_norm()));
        for (int i = 0; i < t; ++i) X.at(r, i) /= s;
    }
    for (auto & v : C2.a) v = rnd_norm();
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < t; ++i)
            for (int k = 0; k < d; ++k) Yv.at(r, k) += X.at(r, i) * C2.at(i, k);
    std::vector<double> C2fit((size_t) t * d);
    SolveStatus sv;
    if (route == SolveRoute::Qr) {
        sv = solve_values_qr(to_cm(X).data(), to_cm(Yv).data(), n, t, d, p, C2fit.data());
    } else {
        std::vector<double> AtA, AtY;
        gram_of(X, Yv, AtA, AtY);
        sv = solve_values(AtA.data(), AtY.data(), t, d, p, C2fit.data());
    }
    CHECK(sv == SolveStatus::Ok, "status %d", (int) sv);
    worst = 0;
    for (size_t i = 0; i < C2.a.size(); ++i) worst = std::max(worst, std::fabs(C2fit[i] - C2.a[i]));
    CHECK(worst < 1e-6, "C2 error %.3e", worst);
}

// ==== 2. parity with the reference NNLS (clamp with a nonzero gradient) ============================

void test_pg_parity() {
    std::printf("[parity: clamp + projected gradient]\n");
    const int n = 400, t = 10;
    Mat M(n, t);
    std::vector<double> w(t), y(n);
    for (auto & v : M.a) v = 0.05 + rnd01();
    for (int i = 0; i < t; ++i) w[i] = (i % 3 == 0) ? 30.0 : (i % 3 == 1 ? 0.02 : 1.5);  // two thirds outside the box
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += M.at(r, i) * w[i];
        y[r] = s + 0.02 * rnd_norm();  // inconsistent: the least squares residual is not zero
    }
    NnlsParams p;
    const std::vector<double> ref = ref_beta(M, y, p.lo, p.hi, 2, 3);
    CHECK(!ref.empty(), "reference lstsq failed");
    // The test is only meaningful if the iteration count and the step change the answer.
    auto dist = [&](const std::vector<double> & a) {
        double d = 0;
        for (int i = 0; i < t; ++i) d = std::max(d, std::fabs(a[i] - ref[i]));
        return d;
    };
    const double d1 = dist(ref_beta(M, y, p.lo, p.hi, 1, 3));
    const double d3 = dist(ref_beta(M, y, p.lo, p.hi, 3, 3));
    const double dh = dist(ref_beta(M, y, p.lo, p.hi, 2, 3, 0.5));
    std::printf("  sensitivity: 1 step %.2e, 3 steps %.2e, half step %.2e\n", d1, d3, dh);
    CHECK(d1 > 1e-4 && d3 > 1e-4 && dh > 1e-4, "problem does not pin the iteration count and step size");

    std::vector<double> beta(t), G, b;
    Mat Y1(n, 1);
    for (int r = 0; r < n; ++r) Y1.at(r, 0) = y[r];
    p.route = SolveRoute::Qr;
    CHECK(solve_beta_qr(to_cm(M).data(), y.data(), n, t, p, beta.data()) == SolveStatus::Ok, "qr status");
    CHECK(dist(beta) < 1e-9, "qr route off the reference by %.2e", dist(beta));
    gram_of(M, Y1, G, b);
    CHECK(solve_beta(G.data(), b.data(), t, p, beta.data()) == SolveStatus::Ok, "gram status");
    CHECK(dist(beta) < 1e-8, "gram route off the reference by %.2e", dist(beta));
    // bf16 rounding of beta is applied when asked.
    p.round_beta_bf16 = true;
    solve_beta_qr(to_cm(M).data(), y.data(), n, t, p, beta.data());
    for (int i = 0; i < t; ++i) CHECK(beta[i] == (double) round_bf16((float) beta[i]), "beta[%d] not bf16", i);
}

// ==== 3. conditioning: ridge only when the solve fails ==============================================

void test_conditioning() {
    std::printf("[conditioning]\n");
    const int n = 300, t = 12, d = 5;
    // Near-dependent but full rank: column 5 = column 2 + 1e-8 * noise (cond ~ 1e8).
    Mat M(n, t), Y1(n, 1);
    std::vector<double> w(t), y(n);
    for (auto & v : M.a) v = 0.05 + rnd01();
    for (int r = 0; r < n; ++r) M.at(r, 5) = M.at(r, 2) + 1e-8 * rnd_norm();
    for (auto & v : w) v = std::exp(-2.0 + 4.0 * rnd01());
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += M.at(r, i) * w[i];
        y[r] = s;
        Y1.at(r, 0) = s;
    }
    NnlsParams p;
    std::vector<double> beta(t);
    SolveStatus st = solve_beta_qr(to_cm(M).data(), y.data(), n, t, p, beta.data());
    CHECK(st == SolveStatus::Ok, "near-dependent full-rank M engaged the ridge (status %d)", (int) st);
    double worst = 0;
    for (int i = 0; i < t; ++i) worst = std::max(worst, std::fabs(beta[i] - std::log(w[i])));
    std::printf("  near-dependent (cond 1e8): qr beta error %.2e\n", worst);
    CHECK(worst < 1e-4, "near-dependent beta error %.2e", worst);
    // The Gram route squares the condition number; informational only.
    std::vector<double> G, b;
    gram_of(M, Y1, G, b);
    NnlsParams pg;
    pg.route = SolveRoute::Gram;
    st = solve_beta(G.data(), b.data(), t, pg, beta.data());
    worst = 0;
    for (int i = 0; i < t; ++i) worst = std::max(worst, std::fabs(beta[i] - std::log(w[i])));
    std::printf("  near-dependent (cond 1e8): gram route status %d, beta error %.2e\n", (int) st, worst);

    // Exactly dependent: the ridge fallback engages and everything stays finite and in the box.
    Mat Md(n, t), Xd(n, t), Yd(n, d);
    std::vector<double> yd(n);
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < t; ++i) Md.at(r, i) = 0.05 + rnd01();
        Md.at(r, 5) = Md.at(r, 2);
        Md.at(r, 9) = 2.0 * Md.at(r, 2);
        yd[r] = 1.0 + rnd01();
        for (int i = 0; i < t; ++i) Xd.at(r, i) = Md.at(r, i) / 10.0;
        for (int k = 0; k < d; ++k) Yd.at(r, k) = rnd_norm();
    }
    st = solve_beta_qr(to_cm(Md).data(), yd.data(), n, t, p, beta.data());
    CHECK(st == SolveStatus::RidgeUsed, "dependent beta status %d", (int) st);
    for (int i = 0; i < t; ++i) CHECK(std::isfinite(beta[i]) && std::fabs(beta[i]) <= 3.0 + 1e-12, "beta[%d]=%g", i, beta[i]);
    std::vector<double> C2((size_t) t * d);
    st = solve_values_qr(to_cm(Xd).data(), to_cm(Yd).data(), n, t, d, p, C2.data());
    CHECK(st == SolveStatus::RidgeUsed, "dependent values status %d", (int) st);
    for (double v : C2) CHECK(std::isfinite(v), "C2 not finite");
    // NaN in the design: Failed, neutral output.
    Md.at(3, 3) = std::nan("");
    st = solve_beta_qr(to_cm(Md).data(), yd.data(), n, t, p, beta.data());
    CHECK(st == SolveStatus::Failed, "nan status %d", (int) st);
    for (int i = 0; i < t; ++i) CHECK(beta[i] == 0.0, "neutral beta[%d]=%g", i, beta[i]);
    std::vector<double> Gd, bd;
    Mat Yd1(n, 1);
    for (int r = 0; r < n; ++r) Yd1.at(r, 0) = yd[r];
    gram_of(Md, Yd1, Gd, bd);
    st = solve_beta(Gd.data(), bd.data(), t, pg, beta.data());
    CHECK(st == SolveStatus::Failed, "gram nan status %d", (int) st);
}

// ==== block problems and the reference pipeline ======================================================

struct Problem {
    BlockShape shape;
    std::vector<ggml_fp16_t> q;
    std::vector<float> k, v;
    std::vector<int32_t> sel;
    BlockInputs in() const { return BlockInputs{q.data(), k.data(), v.data(), sel.data()}; }
};

Problem make_problem(int d, int n_tokens, int n_head, int n_kv, int T, int t, int chunk) {
    Problem p;
    p.shape.head_dim = d; p.shape.n_tokens = n_tokens; p.shape.n_head = n_head; p.shape.n_head_kv = n_kv;
    p.shape.n_keys = T; p.shape.t = t; p.shape.scale = 1.0f / std::sqrt((float) d); p.shape.chunk_tokens = chunk;
    p.q.resize((size_t) d * n_tokens * n_head);
    p.k.resize((size_t) d * T * n_kv);
    p.v.resize((size_t) d * T * n_kv);
    p.sel.resize((size_t) t * n_kv);
    for (auto & x : p.q) x = ggml_fp32_to_fp16((float) rnd_norm());
    for (auto & x : p.k) x = (float) rnd_norm();
    for (auto & x : p.v) x = (float) rnd_norm();
    for (int g = 0; g < n_kv; ++g) {  // t distinct indices, unordered
        std::vector<int32_t> idx(T);
        for (int i = 0; i < T; ++i) idx[i] = i;
        for (int i = T - 1; i > 0; --i) std::swap(idx[i], idx[(int) (rnd01() * (i + 1))]);
        for (int i = 0; i < t; ++i) p.sel[(size_t) g * t + i] = idx[i];
    }
    return p;
}

// Explicit design matrices of KV head g, everything in double (rows: query head of the group, then token).
struct Design { Mat M, A, Yv; std::vector<double> y; };

Design ref_design(const Problem & p, int g, const float * beta /* [t, n_kv] or null */) {
    const BlockShape & s = p.shape;
    const int d = s.head_dim, T = s.n_keys, t = s.t, n_rep = s.n_head / s.n_head_kv;
    const int n = n_rep * s.n_tokens;
    Design D;
    D.M = Mat(n, t); D.A = Mat(n, t); D.Yv = Mat(n, d); D.y.assign(n, 0.0);
    const float * K = p.k.data() + (size_t) g * T * d;
    const float * V = p.v.data() + (size_t) g * T * d;
    std::vector<double> sc(T), e(T), qv(d), x(t);
    int row = 0;
    for (int h = g * n_rep; h < (g + 1) * n_rep; ++h) {
        for (int tok = 0; tok < s.n_tokens; ++tok, ++row) {
            const ggml_fp16_t * qp = p.q.data() + ((size_t) h * s.n_tokens + tok) * d;
            for (int c = 0; c < d; ++c) qv[c] = ggml_fp16_to_fp32(qp[c]);
            double mx = -1e300;
            for (int j = 0; j < T; ++j) {
                double a = 0;
                for (int c = 0; c < d; ++c) a += qv[c] * K[(size_t) j * d + c];
                if (s.bf16_scores) a = round_bf16((float) a);
                sc[j] = a * (double) s.scale;
                mx = std::max(mx, sc[j]);
            }
            double ysum = 0;
            for (int j = 0; j < T; ++j) ysum += (e[j] = std::exp(sc[j] - mx));
            D.y[row] = ysum;
            for (int i = 0; i < t; ++i) D.M.at(row, i) = e[p.sel[(size_t) g * t + i]];
            if (!beta) continue;
            double lm = -1e300;
            for (int i = 0; i < t; ++i) { x[i] = sc[p.sel[(size_t) g * t + i]] + beta[(size_t) g * t + i]; lm = std::max(lm, x[i]); }
            double ls = 0;
            for (int i = 0; i < t; ++i) ls += (x[i] = std::exp(x[i] - lm));
            for (int i = 0; i < t; ++i) D.A.at(row, i) = x[i] / ls;
            for (int j = 0; j < T; ++j)
                for (int c = 0; c < d; ++c) D.Yv.at(row, c) += e[j] / ysum * V[(size_t) j * d + c];
        }
    }
    return D;
}

// The whole reference fit of one head: beta then C2, nothing from the production solvers.
bool ref_fit_head(const Problem & p, int g, bool bf16_beta, std::vector<float> & beta_g, Mat & C2) {
    const int t = p.shape.t;
    Design D = ref_design(p, g, nullptr);
    const std::vector<double> b = ref_beta(D.M, D.y, std::exp(-3.0), std::exp(3.0), 2, 3);
    if (b.empty()) return false;
    beta_g.resize(t);
    for (int i = 0; i < t; ++i) beta_g[i] = bf16_beta ? round_bf16((float) b[i]) : (float) b[i];
    Problem single = p;  // value stage needs beta of this head at index g
    std::vector<float> full((size_t) t * p.shape.n_head_kv, 0.0f);
    std::copy(beta_g.begin(), beta_g.end(), full.begin() + (size_t) g * t);
    D = ref_design(p, g, full.data());
    return ref_lstsq(D.A, D.Yv, C2);
}

double max_rel(const std::vector<double> & a, const std::vector<double> & ref) {
    double scale = 0, err = 0;
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::fabs(a[i] - ref[i]));
    return err / std::max(scale, 1e-30);
}

// ==== 4. backend accumulation vs the reference =======================================================

void test_accumulate(ggml_backend_t be, const char * name, const char * label, int d, int n_tokens, int n_head, int n_kv,
                     int T, int t, int chunk, int compensated = -1, bool info_only = false, bool bf16_scores = false) {
    std::printf("[accumulate %s %s d%d tok%d h%d/%d T%d t%d chunk%d]\n", name, label, d, n_tokens, n_head, n_kv, T, t, chunk);
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, chunk);
    p.shape.compensated = compensated;
    p.shape.bf16_scores = bf16_scores;
    std::vector<float> beta((size_t) t * n_kv);
    for (auto & b : beta) b = (float) (-1.0 + 2.0 * rnd01());

    GramAccumulator acc;
    std::string err;
    if (!acc.init(be, p.shape, p.in(), &err)) { CHECK(false, "init: %s", err.c_str()); return; }
    BetaGrams bg;
    ValueGrams vg;
    BetaDesign bd;
    ValueDesign vd;
    CHECK(acc.beta_grams(bg, &err), "beta_grams: %s", err.c_str());
    CHECK(acc.value_grams(beta.data(), vg, &err), "value_grams: %s", err.c_str());
    CHECK(acc.beta_design(bd, &err), "beta_design: %s", err.c_str());
    CHECK(acc.value_design(beta.data(), vd, &err), "value_design: %s", err.c_str());
    const int rows = n_tokens * (n_head / n_kv);
    if (bg.MtM.size() != (size_t) n_kv * t * t || bd.rows != rows || vd.Y.size() != (size_t) n_kv * d * rows) { CHECK(false, "output size"); return; }

    double e[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < n_kv; ++g) {
        const Design D = ref_design(p, g, beta.data());
        std::vector<double> G, b, Ga, Ya;
        Mat Y1(rows, 1);
        for (int r = 0; r < rows; ++r) Y1.at(r, 0) = D.y[r];
        gram_of(D.M, Y1, G, b);
        gram_of(D.A, D.Yv, Ga, Ya);
        auto sub = [&](const std::vector<double> & all, size_t len) {
            return std::vector<double>(all.begin() + (size_t) g * len, all.begin() + (size_t) (g + 1) * len);
        };
        e[0] = std::max(e[0], max_rel(sub(bg.MtM, (size_t) t * t), G));
        e[1] = std::max(e[1], max_rel(sub(bg.Mty, t), b));
        e[2] = std::max(e[2], max_rel(sub(vg.AtA, (size_t) t * t), Ga));
        e[3] = std::max(e[3], max_rel(sub(vg.AtY, (size_t) t * d), Ya));
        // The design matrices carry the same information as the Grams, in another row order: compare their Grams.
        Mat Md(rows, t), Ad(rows, t), Yd(rows, d), yd1(rows, 1);
        for (int r = 0; r < rows; ++r) {
            yd1.at(r, 0) = bd.y[(size_t) g * rows + r];
            for (int i = 0; i < t; ++i) {
                Md.at(r, i) = bd.M[((size_t) g * t + i) * rows + r];
                Ad.at(r, i) = vd.A[((size_t) g * t + i) * rows + r];
            }
            for (int c = 0; c < d; ++c) Yd.at(r, c) = vd.Y[((size_t) g * d + c) * rows + r];
        }
        std::vector<double> G2, b2, Ga2, Ya2;
        gram_of(Md, yd1, G2, b2);
        gram_of(Ad, Yd, Ga2, Ya2);
        e[4] = std::max(e[4], max_rel(G2, G));
        e[5] = std::max(e[5], max_rel(b2, b));
        e[6] = std::max(e[6], max_rel(Ga2, Ga));
        e[7] = std::max(e[7], max_rel(Ya2, Ya));
    }
    std::printf("  max rel err: Grams MtM %.2e Mty %.2e AtA %.2e AtY %.2e | design->Gram %.2e %.2e %.2e %.2e\n",
                e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);
    const double tol = bf16_scores ? 1e-4 : 2e-5;
    if (!info_only) for (double x : e) CHECK(x < tol, "accumulation mismatch %.2e", x);
}

// Softmax tails down to 1e-30 must survive the compensated matmuls: one selected key is anti-aligned
// with every query, so its whole column of M is ~1e-30.
void test_tails(ggml_backend_t be, const char * name) {
    std::printf("[tails %s]\n", name);
    const int d = 64, n_tokens = 64, n_head = 2, n_kv = 1, T = 100, t = 8;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 32);
    p.shape.scale = 1.0f / 8.0f;
    p.shape.compensated = 1;
    for (size_t h = 0; h < (size_t) n_head * n_tokens; ++h) {
        for (int c = 0; c < d; ++c)
            p.q[h * d + c] = ggml_fp32_to_fp16((float) (0.3 * rnd_norm() + (c == 0 ? 4.0 : 0.0)));
    }
    for (int j = 0; j < T; ++j) {
        p.k[(size_t) j * d] = (float) (0.5 * rnd_norm());
        p.sel[0] = 0;
    }
    for (int c = 1; c < d; ++c) p.k[c] = 0.0f;
    for (int i = 1; i < t; ++i) p.sel[i] = i;  // key 0 plus distinct others
    // Anti-align key 0 until its column of M is in the 1e-30 regime.
    double k0 = 100.0;
    for (int it = 0; it < 60; ++it, k0 *= 1.05) {
        p.k[0] = (float) -k0;
        const Design Dt = ref_design(p, 0, nullptr);
        double col = 0;
        for (int r = 0; r < Dt.M.n; ++r) col = std::max(col, Dt.M.at(r, 0));
        if (col < 1e-30) break;
    }
    GramAccumulator acc;
    std::string err;
    if (!acc.init(be, p.shape, p.in(), &err)) { CHECK(false, "init: %s", err.c_str()); return; }
    BetaGrams bg;
    CHECK(acc.beta_grams(bg, &err), "beta_grams: %s", err.c_str());
    const Design D = ref_design(p, 0, nullptr);
    Mat Y1(D.M.n, 1);
    for (int r = 0; r < D.M.n; ++r) Y1.at(r, 0) = D.y[r];
    std::vector<double> G, b;
    gram_of(D.M, Y1, G, b);
    std::printf("  reference Mty[0] = %.3e, MtM[0][1] = %.3e\n", b[0], G[1]);
    CHECK(b[0] > 1e-35 && b[0] < 1e-25, "test is not in the tail regime: %.3e", b[0]);
    const double r1 = std::fabs(bg.Mty[0] - b[0]) / b[0];
    double r2 = 0;
    for (int j = 1; j < t; ++j) r2 = std::max(r2, std::fabs(bg.MtM[j] - G[j]) / G[j]);
    std::printf("  elementwise rel err: Mty[0] %.2e, MtM[0][j] %.2e\n", r1, r2);
    CHECK(r1 < 1e-3 && r2 < 1e-3, "tail operands lost");
    // The design route carries the tail as well.
    BetaDesign bd;
    CHECK(acc.beta_design(bd, &err), "beta_design: %s", err.c_str());
    double colsum = 0;
    for (int r = 0; r < bd.rows; ++r) colsum += bd.M[r];
    CHECK(colsum > 0 && colsum < 1e-20, "design column 0 sum %.3e", colsum);
}

// ==== 5. end to end ===================================================================================

// Selected key i appears c_i = 1 + i % 7 times in the block (same K and same V), so the full softmax
// over T keys equals the softmax over the selected keys with beta = log c_i and C2 = V_sel.
void test_dup(ggml_backend_t be, const char * name, SolveRoute route) {
    std::printf("[dup recovery %s %s]\n", name, route == SolveRoute::Qr ? "qr" : "gram");
    const int d = 64, n_tokens = 150, n_head = 6, n_kv = 2, t = 16;
    int T = 0;
    for (int i = 0; i < t; ++i) T += 1 + i % 7;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 64);
    for (int g = 0; g < n_kv; ++g) {
        std::vector<int> perm(T);
        for (int i = 0; i < T; ++i) perm[i] = i;
        for (int i = T - 1; i > 0; --i) std::swap(perm[i], perm[(int) (rnd01() * (i + 1))]);
        int pos = 0;
        for (int i = 0; i < t; ++i) {
            const int first = perm[pos];
            p.sel[(size_t) g * t + i] = first;
            for (int c = 0; c < 1 + i % 7; ++c, ++pos) {
                std::memcpy(&p.k[((size_t) g * T + perm[pos]) * d], &p.k[((size_t) g * T + first) * d], d * sizeof(float));
                std::memcpy(&p.v[((size_t) g * T + perm[pos]) * d], &p.v[((size_t) g * T + first) * d], d * sizeof(float));
            }
        }
    }
    NnlsParams np;
    np.route = route;
    FitResult res;
    std::string err;
    if (!fit_block(be, p.shape, p.in(), np, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
    double eb = 0, ec = 0;
    for (int g = 0; g < n_kv; ++g) {
        CHECK(res.status[g] == SolveStatus::Ok, "status[%d]=%d", g, (int) res.status[g]);
        for (int i = 0; i < t; ++i) {
            eb = std::max(eb, (double) std::fabs(res.beta[(size_t) g * t + i] - (float) std::log(1.0 + i % 7)));
            const float * vs = &p.v[((size_t) g * T + p.sel[(size_t) g * t + i]) * d];
            for (int c = 0; c < d; ++c) ec = std::max(ec, (double) std::fabs(res.c2[((size_t) g * t + i) * d + c] - vs[c]));
        }
    }
    std::printf("  beta err %.2e  C2 err %.2e\n", eb, ec);
    CHECK(eb < 1e-4 && ec < 1e-4, "recovery off");
}

// fit_block against the standalone fp64 reference pipeline.
void test_fit_vs_ref(ggml_backend_t be, const char * name, SolveRoute route, bool bf16) {
    std::printf("[fit vs fp64 reference %s %s%s]\n", name, route == SolveRoute::Qr ? "qr" : "gram", bf16 ? " bf16" : "");
    const int d = 32, n_tokens = 90, n_head = 4, n_kv = 2, T = 80, t = 12;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 40);
    p.shape.bf16_scores = bf16;
    NnlsParams np;
    np.route = route;
    np.round_beta_bf16 = bf16;
    FitResult res;
    std::string err;
    if (!fit_block(be, p.shape, p.in(), np, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
    double eb = 0, ec = 0;
    for (int g = 0; g < n_kv; ++g) {
        std::vector<float> bref;
        Mat C2;
        if (!ref_fit_head(p, g, bf16, bref, C2)) { CHECK(false, "reference failed"); return; }
        for (int i = 0; i < t; ++i) {
            eb = std::max(eb, (double) std::fabs(res.beta[(size_t) g * t + i] - bref[i]));
            for (int c = 0; c < d; ++c) ec = std::max(ec, std::fabs(res.c2[((size_t) g * t + i) * d + c] - C2.at(i, c)));
        }
    }
    std::printf("  beta err %.2e  C2 err %.2e\n", eb, ec);
    const double tol = bf16 ? 2e-2 : 1e-4;  // bf16 rounding flips (beta 2^-8, scores) move single entries
    CHECK(eb < tol && ec < tol, "pipeline differs from the reference");
}

// ==== 6. whole-block failure ===========================================================================

void test_block_failure(ggml_backend_t be, const char * name, SolveRoute route) {
    std::printf("[block failure %s %s]\n", name, route == SolveRoute::Qr ? "qr" : "gram");
    Problem p = make_problem(32, 70, 4, 2, 80, 8, 32);
    NnlsParams np;
    np.route = route;
    FitResult res;
    std::string err;
    CHECK(fit_block(be, p.shape, p.in(), np, res, &err), "clean block must fit: %s", err.c_str());
    CHECK(!res.beta.empty() && !res.c2.empty(), "clean block published nothing");
    // One NaN in the keys of KV head 1 only.
    p.k[(size_t) 32 * 80 + 5 * 32 + 3] = std::nanf("");
    FitResult bad;
    bad.beta.assign(3, 1.0f);  // stale content must not survive a failure
    const bool ok = fit_block(be, p.shape, p.in(), np, bad, &err);
    CHECK(!ok, "a NaN head must fail the whole block");
    CHECK(bad.beta.empty() && bad.c2.empty(), "failed fit published beta/C2");
    CHECK(err.find("head 1") != std::string::npos, "error does not name the head: %s", err.c_str());
}

// ==== 7. device-tensor entry point ==========================================================================

void test_device_path(ggml_backend_t be, const char * name) {
    std::printf("[device tensors %s]\n", name);
    const int d = 64, n_tokens = 70, n_head = 6, n_kv = 2, T = 80, t = 8, n_layers = 3, pad = 16;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 32);
    // K -> F16 values (what the cache holds), so both paths see the same numbers.
    for (auto & x : p.k) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * qblk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n_tokens, n_head, n_layers);  // like M3's q_blk
    ggml_tensor * kc = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, T + pad, n_kv);                  // padded: non-contiguous view
    ggml_tensor * vc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, T + pad, n_kv);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    CHECK(buf != nullptr, "alloc");
    if (!buf) { ggml_free(ctx); return; }
    const int layer = 2;
    std::vector<ggml_fp16_t> qall((size_t) d * n_tokens * n_head * n_layers, ggml_fp32_to_fp16(0.0f));
    std::copy(p.q.begin(), p.q.end(), qall.begin() + (size_t) layer * d * n_tokens * n_head);
    ggml_backend_tensor_set(qblk, qall.data(), 0, ggml_nbytes(qblk));
    std::vector<ggml_fp16_t> kh((size_t) d * (T + pad) * n_kv, ggml_fp32_to_fp16(7.0f));
    std::vector<float> vh((size_t) d * (T + pad) * n_kv, 9.0f);
    for (int g = 0; g < n_kv; ++g)
        for (int j = 0; j < T; ++j)
            for (int c = 0; c < d; ++c) {
                kh[((size_t) g * (T + pad) + j) * d + c] = ggml_fp32_to_fp16(p.k[((size_t) g * T + j) * d + c]);
                vh[((size_t) g * (T + pad) + j) * d + c] = p.v[((size_t) g * T + j) * d + c];
            }
    ggml_backend_tensor_set(kc, kh.data(), 0, ggml_nbytes(kc));
    ggml_backend_tensor_set(vc, vh.data(), 0, ggml_nbytes(vc));
    ggml_tensor * qv = ggml_view_3d(ctx, qblk, d, n_tokens, n_head, qblk->nb[1], qblk->nb[2], (size_t) layer * qblk->nb[3]);
    ggml_tensor * kv = ggml_view_3d(ctx, kc, d, T, n_kv, kc->nb[1], kc->nb[2], 0);
    ggml_tensor * vv = ggml_view_3d(ctx, vc, d, T, n_kv, vc->nb[1], vc->nb[2], 0);

    NnlsParams np;
    FitResult host, dev;
    std::string err;
    GramAccumulator a_host, a_dev;
    CHECK(fit_block(a_host, be, p.shape, p.in(), np, host, &err), "host path: %s", err.c_str());
    BlockTensors bt{qv, kv, vv, p.sel.data()};
    CHECK(fit_block(a_dev, be, p.shape, bt, np, dev, &err), "device path: %s", err.c_str());
    double eb = 0, ec = 0;
    if (host.beta.size() == dev.beta.size() && host.c2.size() == dev.c2.size()) {
        for (size_t i = 0; i < host.beta.size(); ++i) eb = std::max(eb, (double) std::fabs(host.beta[i] - dev.beta[i]));
        for (size_t i = 0; i < host.c2.size(); ++i) ec = std::max(ec, (double) std::fabs(host.c2[i] - dev.c2[i]));
    } else {
        CHECK(false, "result sizes differ");
    }
    std::printf("  host vs device path: beta diff %.2e, C2 diff %.2e\n", eb, ec);
    CHECK(eb < 1e-6 && ec < 1e-6, "device path differs from the host path");
    std::printf("  transfers per layer, host path:   H2D %d calls %.2f MB, D2H %d calls %.2f MB\n", host.transfers.h2d_calls,
                host.transfers.h2d_bytes / 1e6, host.transfers.d2h_calls, host.transfers.d2h_bytes / 1e6);
    std::printf("  transfers per layer, device path: H2D %d calls %.2f MB, D2H %d calls %.2f MB\n", dev.transfers.h2d_calls,
                dev.transfers.h2d_bytes / 1e6, dev.transfers.d2h_calls, dev.transfers.d2h_bytes / 1e6);
    CHECK(dev.transfers.h2d_bytes < host.transfers.h2d_bytes / 4, "device path still uploads q/k/v");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// ==== 8. real-data gate ==========================================================================================

struct Err { double abs = 0, rel = 0; };  // rel: max |a - b| / max |b| over the tensor
Err compare(const float * a, const std::vector<float> & ref, size_t off, size_t n) {
    Err e;
    double scale = 0;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, (double) std::fabs(ref[off + i]));
    for (size_t i = 0; i < n; ++i) e.abs = std::max(e.abs, (double) std::fabs(a[i] - ref[off + i]));
    e.rel = e.abs / std::max(scale, 1e-30);
    return e;
}

void run_fixture_gate(ggml_backend_t be, const char * name) {
    const std::vector<std::string> dirs = am_fixture::case_dirs(std::getenv("LUCE_AM_FIXTURES"));
    if (dirs.empty()) { std::printf("[fixture gate %s] SKIP: LUCE_AM_FIXTURES unset or no cases\n", name); return; }
    const int d = 256, n_tok = 640, n_head = 24, n_kv = 4, T = 640, t = 64;
    struct Cfg { const char * label; SolveRoute route; bool ref_sem; };
    const Cfg cfgs[] = {{"qr+bf16-semantics", SolveRoute::Qr, true}, {"qr+fp32", SolveRoute::Qr, false},
                        {"gram+bf16-semantics", SolveRoute::Gram, true}, {"gram+fp32", SolveRoute::Gram, false}};
    GramAccumulator acc;
    for (const std::string & dir : dirs) {
        am_fixture::Case c;
        std::string err;
        if (!am_fixture::load_case(dir, c, &err)) { CHECK(false, "%s", err.c_str()); continue; }
        if (c.name.find("exactq") != std::string::npos) continue;  // queries only, no fit outputs
        std::printf("[fixture gate %s] case %s, %zu layers\n", name, c.name.c_str(), c.layers.size());
        for (const Cfg & cfg : cfgs) {
            double worst_b = 0, worst_c = 0;
            for (int layer : c.layers) {
                std::vector<float> q, k, v, beta_ref, c2_ref;
                std::vector<int32_t> idx;
                if (!am_fixture::read_f32(c, layer, "q", (size_t) d * n_tok * n_head, q, &err) ||
                    !am_fixture::read_f32(c, layer, "k", (size_t) d * T * n_kv, k, &err) ||
                    !am_fixture::read_f32(c, layer, "v", (size_t) d * T * n_kv, v, &err) ||
                    !am_fixture::read_i32(c, layer, "idx", (size_t) t * n_kv, idx, &err) ||
                    !am_fixture::read_f32(c, layer, "beta", (size_t) t * n_kv, beta_ref, &err) ||
                    !am_fixture::read_f32(c, layer, "c2", (size_t) d * t * n_kv, c2_ref, &err)) {
                    CHECK(false, "%s", err.c_str());
                    break;
                }
                std::vector<ggml_fp16_t> q16(q.size());
                for (size_t i = 0; i < q.size(); ++i) q16[i] = ggml_fp32_to_fp16(q[i]);
                BlockShape s;
                s.head_dim = d; s.n_tokens = n_tok; s.n_head = n_head; s.n_head_kv = n_kv; s.n_keys = T; s.t = t;
                s.scale = 1.0f / std::sqrt((float) d);
                s.bf16_scores = cfg.ref_sem;
                NnlsParams np;
                np.route = cfg.route;
                np.round_beta_bf16 = cfg.ref_sem;
                FitResult res;
                BlockInputs in{q16.data(), k.data(), v.data(), idx.data()};
                if (!fit_block(acc, be, s, in, np, res, &err)) { CHECK(false, "layer %d: %s", layer, err.c_str()); continue; }
                for (int g = 0; g < n_kv; ++g) {
                    const Err eb = compare(&res.beta[(size_t) g * t], beta_ref, (size_t) g * t, t);
                    const Err ec = compare(&res.c2[(size_t) g * t * d], c2_ref, (size_t) g * t * d, (size_t) t * d);
                    std::printf("  %-20s L%02d h%d  beta abs %.2e rel %.2e   C2 abs %.2e rel %.2e   status %d\n", cfg.label, layer, g,
                                eb.abs, eb.rel, ec.abs, ec.rel, (int) res.status[g]);
                    worst_b = std::max(worst_b, eb.rel);
                    worst_c = std::max(worst_c, ec.rel);
                }
            }
            std::printf("  %-20s WORST rel: beta %.2e  C2 %.2e\n", cfg.label, worst_b, worst_c);
            // The binding gate: the faithful route with the reference's semantics, within 1e-3 relative.
            if (cfg.route == SolveRoute::Qr && cfg.ref_sem) CHECK(worst_b <= 1e-3 && worst_c <= 1e-3, "R3 gate missed on %s", c.name.c_str());
        }
    }
}

// ==== timing at the real shape (synthetic data) ====================================================================

void time_real_shape(ggml_backend_t be, const char * name, SolveRoute route) {
    // One 640-token block, 16 full-attention layers, Qwen3.8-27B: d256, 24 heads, 4 KV heads, t = 64.
    Problem p = make_problem(256, 640, 24, 4, 640, 64, BlockShape{}.chunk_tokens);
    NnlsParams np;
    np.route = route;
    FitResult res;
    std::string err;
    GramAccumulator acc;  // one per device in the engine; graphs and buffers persist across layers
    double best = 1e30, total = 0, ph[5] = {0, 0, 0, 0, 0};
    const int reps = 5;
    for (int rep = 0; rep < reps + 1; ++rep) {  // first run is warm-up
        const auto t0 = std::chrono::steady_clock::now();
        for (int layer = 0; layer < 16; ++layer) {
            if (!fit_block(acc, be, p.shape, p.in(), np, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
            if (rep > 0) {
                ph[0] += res.ms_init; ph[1] += res.ms_beta_grams; ph[2] += res.ms_beta_solve;
                ph[3] += res.ms_value_grams; ph[4] += res.ms_value_solve;
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rep > 0) { best = std::min(best, ms); total += ms; }
    }
    std::printf("TIMING %s %s: one 640-token block, 16 layers x 4 KV heads, t=64: best %.1f ms, mean %.1f ms (%d reps)\n",
                name, route == SolveRoute::Qr ? "qr" : "gram", best, total / reps, reps);
    std::printf("  phases per block (ms, mean of %d): init %.1f, beta %s %.1f, beta solve %.1f, value %s %.1f, value solve %.1f\n",
                reps, ph[0] / reps, route == SolveRoute::Qr ? "design" : "Grams", ph[1] / reps, ph[2] / reps,
                route == SolveRoute::Qr ? "design" : "Grams", ph[3] / reps, ph[4] / reps);
    std::printf("  per layer: H2D %d calls %.2f MB, D2H %d calls %.2f MB (host-pointer inputs)\n", res.transfers.h2d_calls,
                res.transfers.h2d_bytes / 1e6, res.transfers.d2h_calls, res.transfers.d2h_bytes / 1e6);
}

}  // namespace

int main(int argc, char ** argv) {
    bool cpu_only = false, time_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu-only")) cpu_only = true;
        if (!std::strcmp(argv[i], "--time")) time_only = true;
    }

    if (!time_only) {
        for (SolveRoute r : {SolveRoute::Qr, SolveRoute::Gram}) {
            test_host_recovery(300, 24, 8, r);
            test_host_recovery(3840, 64, 16, r);
        }
        test_pg_parity();
        test_conditioning();
    }

    struct Be { const char * name; ggml_backend_t be; };
    std::vector<Be> backends;
    backends.push_back({"CPU", ggml_backend_cpu_init()});
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    if (!cpu_only) {
        if (ggml_backend_t gpu = ggml_backend_cuda_init(0)) backends.push_back({"GPU", gpu});
        else { std::printf("GPU backend init failed\n"); ++g_fail; }
    }
#else
    (void) cpu_only;
#endif
    for (const Be & b : backends) {
        if (time_only) {
            time_real_shape(b.be, b.name, SolveRoute::Qr);
            time_real_shape(b.be, b.name, SolveRoute::Gram);
            continue;
        }
        test_accumulate(b.be, b.name, "small", 32, 70, 6, 2, 80, 8, 32);          // remainder chunk
        test_accumulate(b.be, b.name, "small-comp", 32, 70, 6, 2, 80, 8, 32, 1);  // split matmuls forced on
        test_accumulate(b.be, b.name, "mha", 16, 33, 3, 3, 40, 6, 33);            // n_rep = 1, one chunk
        test_accumulate(b.be, b.name, "bf16-scores", 32, 70, 6, 2, 80, 8, 32, -1, false, true);
        test_accumulate(b.be, b.name, "real", 256, 640, 24, 4, 640, 64, 64);      // Qwen3.8-27B block, default chunk
        test_accumulate(b.be, b.name, "real-plain", 256, 640, 24, 4, 640, 64, 64, 0, true);  // informational: no compensation
        test_tails(b.be, b.name);
        for (SolveRoute r : {SolveRoute::Qr, SolveRoute::Gram}) {
            test_dup(b.be, b.name, r);
            test_fit_vs_ref(b.be, b.name, r, false);
            test_block_failure(b.be, b.name, r);
        }
        test_fit_vs_ref(b.be, b.name, SolveRoute::Qr, true);
        test_device_path(b.be, b.name);
        run_fixture_gate(b.be, b.name);
    }
    for (const Be & b : backends) ggml_backend_free(b.be);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
