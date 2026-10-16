#include "common/am_fit_solve.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <utility>

// Deviations from the reference (algorithms/base.py), both documented in the header:
//  * the power iteration for L starts from the normalised ones vector, not a random one (it is
//    deterministic; L only sets the step size of two projected-gradient steps, and M >= 0 so the ones
//    vector always overlaps the top singular vector);
//  * beta is rounded to bf16 only when NnlsParams::round_beta_bf16 is set.

namespace luce_am {

// =====================================================================================
// Host solvers (fp64)
// =====================================================================================

namespace {

// In-place lower Cholesky factor of the row-major SPD matrix `a` (t x t). False when a pivot
// is not above rel_tol times its original diagonal (this also rejects NaN).
bool cholesky(std::vector<double> & a, int t, double rel_tol) {
    std::vector<double> diag0(t);
    for (int i = 0; i < t; ++i) diag0[i] = a[(size_t) i * t + i];
    for (int j = 0; j < t; ++j) {
        double s = a[(size_t) j * t + j];
        for (int k = 0; k < j; ++k) s -= a[(size_t) j * t + k] * a[(size_t) j * t + k];
        if (!(diag0[j] > 0.0) || !(s > rel_tol * diag0[j])) return false;
        const double piv = std::sqrt(s);
        a[(size_t) j * t + j] = piv;
        for (int i = j + 1; i < t; ++i) {
            double v = a[(size_t) i * t + j];
            for (int k = 0; k < j; ++k) v -= a[(size_t) i * t + k] * a[(size_t) j * t + k];
            a[(size_t) i * t + j] = v / piv;
        }
    }
    return true;
}

// Solve L L^T X = B in place; B is row-major [t][nrhs].
void cholesky_solve(const std::vector<double> & l, int t, int nrhs, double * b) {
    for (int c = 0; c < nrhs; ++c) {
        for (int i = 0; i < t; ++i) {
            double v = b[(size_t) i * nrhs + c];
            for (int k = 0; k < i; ++k) v -= l[(size_t) i * t + k] * b[(size_t) k * nrhs + c];
            b[(size_t) i * nrhs + c] = v / l[(size_t) i * t + i];
        }
        for (int i = t - 1; i >= 0; --i) {
            double v = b[(size_t) i * nrhs + c];
            for (int k = i + 1; k < t; ++k) v -= l[(size_t) k * t + i] * b[(size_t) k * nrhs + c];
            b[(size_t) i * nrhs + c] = v / l[(size_t) i * t + i];
        }
    }
}

bool all_finite(const double * x, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!std::isfinite(x[i])) return false;
    return true;
}

float round_bf16(float x) {  // nearest even
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    if ((u & 0x7F800000u) != 0x7F800000u) u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
    std::memcpy(&x, &u, sizeof u);
    return x;
}

// Normal-equations solve X = (sym(G))^-1 B, no ridge: false unless every pivot is > rel_tol * diag.
bool spd_solve(const double * G, const double * B, int t, int nrhs, double rel_tol, double ridge, double * X) {
    std::vector<double> l((size_t) t * t);
    for (int i = 0; i < t; ++i)
        for (int j = 0; j < t; ++j) l[(size_t) i * t + j] = 0.5 * (G[(size_t) i * t + j] + G[(size_t) j * t + i]);
    for (int i = 0; i < t; ++i) l[(size_t) i * t + i] += ridge;
    if (!cholesky(l, t, rel_tol)) return false;
    std::memcpy(X, B, sizeof(double) * (size_t) t * nrhs);
    cholesky_solve(l, t, nrhs, X);
    return all_finite(X, (size_t) t * nrhs);
}

// Gram route: plain solve first; the ridge only when that fails (a non-positive pivot or non-finite result).
SolveStatus solve_spd(const double * G, const double * B, int t, int nrhs, const NnlsParams & p, double * X) {
    if (spd_solve(G, B, t, nrhs, p.chol_rel_tol, 0.0, X)) return SolveStatus::Ok;
    if (p.gram_ridge && spd_solve(G, B, t, nrhs, p.chol_rel_tol, p.ridge, X)) return SolveStatus::RidgeUsed;
    return SolveStatus::Failed;
}

template <class F>
void parallel_for(int n, F && f) {
    const int nt = std::max(1, std::min<int>(n, (int) std::thread::hardware_concurrency()));
    if (nt == 1) { f(0, n); return; }
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i) th.emplace_back([&f, i, nt, n] { f((int) ((int64_t) n * i / nt), (int) ((int64_t) n * (i + 1) / nt)); });
    for (auto & x : th) x.join();
}

// Least squares min |A X - B| by Householder QR (what LAPACK gels does). A_cm column-major n x t,
// B_cm column-major n x m, X row-major [t][m]. False when R is numerically singular
// (min|R_kk| <= tol * max|R_kk|, this also catches NaN) or the result is not finite.
bool qr_lstsq(const double * A_cm, int n, int t, const double * B_cm, int m, double tol, double * X) {
    if (n < t) return false;
    std::vector<double> a(A_cm, A_cm + (size_t) n * t), rd(t), vv(t);
    for (int k = 0; k < t; ++k) {
        double * ck = &a[(size_t) k * n];
        double nrm = 0;
        for (int r = k; r < n; ++r) nrm += ck[r] * ck[r];
        nrm = std::sqrt(nrm);
        const double alpha = ck[k] > 0 ? -nrm : nrm;
        rd[k] = alpha;
        ck[k] -= alpha;  // v = x - alpha e1
        double v2 = 0;
        for (int r = k; r < n; ++r) v2 += ck[r] * ck[r];
        vv[k] = v2;
        if (!(v2 > 0.0)) continue;
        for (int j = k + 1; j < t; ++j) {
            double * cj = &a[(size_t) j * n];
            double s = 0;
            for (int r = k; r < n; ++r) s += ck[r] * cj[r];
            const double f = 2.0 * s / v2;
            for (int r = k; r < n; ++r) cj[r] -= f * ck[r];
        }
    }
    double mn = 1e300, mx = 0;
    for (int k = 0; k < t; ++k) { mn = std::min(mn, std::fabs(rd[k])); mx = std::max(mx, std::fabs(rd[k])); }
    if (!(mn > tol * mx) || !std::isfinite(mx)) return false;

    std::atomic<int> bad{0};
    parallel_for(m, [&](int c0, int c1) {
        std::vector<double> b(n);
        for (int c = c0; c < c1; ++c) {
            std::memcpy(b.data(), B_cm + (size_t) c * n, sizeof(double) * n);
            for (int k = 0; k < t; ++k) {
                if (!(vv[k] > 0.0)) continue;
                const double * ck = &a[(size_t) k * n];
                double s = 0;
                for (int r = k; r < n; ++r) s += ck[r] * b[r];
                const double f = 2.0 * s / vv[k];
                for (int r = k; r < n; ++r) b[r] -= f * ck[r];
            }
            for (int i = t - 1; i >= 0; --i) {  // R x = (Q^T b)[0:t]
                double s = b[i];
                for (int j = i + 1; j < t; ++j) s -= a[(size_t) j * n + i] * X[(size_t) j * m + c];
                X[(size_t) i * m + c] = s / rd[i];
            }
            for (int i = 0; i < t; ++i) if (!std::isfinite(X[(size_t) i * m + c])) { bad = 1; break; }
        }
    });
    return !bad;
}

// The reference's fallback: normal equations with an absolute ridge on the diagonal.
bool ridge_lstsq(const double * A_cm, int n, int t, const double * B_cm, int m, double ridge, double * X) {
    std::vector<double> G((size_t) t * t), R((size_t) t * m);
    parallel_for(t, [&](int i0, int i1) {
        for (int i = i0; i < i1; ++i) {
            const double * ci = A_cm + (size_t) i * n;
            for (int j = 0; j < t; ++j) {
                const double * cj = A_cm + (size_t) j * n;
                double s = 0;
                for (int r = 0; r < n; ++r) s += ci[r] * cj[r];
                G[(size_t) i * t + j] = s;
            }
            for (int c = 0; c < m; ++c) {
                const double * bc = B_cm + (size_t) c * n;
                double s = 0;
                for (int r = 0; r < n; ++r) s += ci[r] * bc[r];
                R[(size_t) i * m + c] = s;
            }
        }
    });
    return spd_solve(G.data(), R.data(), t, m, 0.0, ridge, X);
}

SolveStatus lstsq_with_fallback(const double * A_cm, int n, int t, const double * B_cm, int m, const NnlsParams & p, double * X) {
    if (qr_lstsq(A_cm, n, t, B_cm, m, p.qr_rank_tol, X)) return SolveStatus::Ok;
    if (ridge_lstsq(A_cm, n, t, B_cm, m, p.ridge, X)) return SolveStatus::RidgeUsed;
    return SolveStatus::Failed;
}

// Clamp, the 1/L step size from `mul` (u -> G u, symmetric PSD), and the projected-gradient steps
// with gradient `grad(w, out)`. Common to both routes.
template <class GramMul, class Grad>
SolveStatus finish_beta(std::vector<double> & w, int t, SolveStatus st, const NnlsParams & p, GramMul && gmul, Grad && grad,
                        double * beta_out) {
    for (double & v : w) v = std::min(std::max(v, p.lo), p.hi);
    std::vector<double> u(t, 1.0 / std::sqrt((double) t)), v(t);
    for (int it = 0; it < p.power_iters; ++it) {
        gmul(u, v);
        double nrm = 0;
        for (double x : v) nrm += x * x;
        nrm = std::sqrt(nrm);
        if (!(nrm > 0.0)) break;
        for (int i = 0; i < t; ++i) u[i] = v[i] / nrm;
    }
    gmul(u, v);
    double rq = 0;  // u^T G u = |M u|^2
    for (int i = 0; i < t; ++i) rq += u[i] * v[i];
    const double L = std::max(rq, 1e-6);
    std::vector<double> g(t);
    for (int it = 0; it < p.pg_iters; ++it) {
        grad(w, g);
        for (int i = 0; i < t; ++i) w[i] = std::min(std::max(w[i] - g[i] / L, p.lo), p.hi);
    }
    for (int i = 0; i < t; ++i) {
        beta_out[i] = std::log(w[i]);
        if (p.round_beta_bf16) beta_out[i] = round_bf16((float) beta_out[i]);
    }
    if (!all_finite(beta_out, t)) {
        std::fill(beta_out, beta_out + t, 0.0);
        return SolveStatus::Failed;
    }
    return st;
}

}  // namespace

SolveStatus solve_beta(const double * G, const double * b, int t, const NnlsParams & p, double * beta_out) {
    std::fill(beta_out, beta_out + t, 0.0);
    std::vector<double> w(t);
    const SolveStatus st = solve_spd(G, b, t, 1, p, w.data());  // least-squares init
    if (st == SolveStatus::Failed) return st;
    std::vector<double> sym((size_t) t * t);
    for (int i = 0; i < t; ++i)
        for (int j = 0; j < t; ++j) sym[(size_t) i * t + j] = 0.5 * (G[(size_t) i * t + j] + G[(size_t) j * t + i]);
    auto gmul = [&](const std::vector<double> & x, std::vector<double> & out) {
        for (int i = 0; i < t; ++i) {
            double s = 0;
            for (int j = 0; j < t; ++j) s += sym[(size_t) i * t + j] * x[j];
            out[i] = s;
        }
    };
    // gradient of 0.5 |M w - y|^2: G w - b
    auto grad = [&](const std::vector<double> & x, std::vector<double> & out) {
        gmul(x, out);
        for (int i = 0; i < t; ++i) out[i] -= b[i];
    };
    return finish_beta(w, t, st, p, gmul, grad, beta_out);
}

SolveStatus solve_values(const double * AtA, const double * AtY, int t, int d, const NnlsParams & p, double * C2_out) {
    const SolveStatus st = solve_spd(AtA, AtY, t, d, p, C2_out);
    if (st == SolveStatus::Failed) std::fill(C2_out, C2_out + (size_t) t * d, 0.0);
    return st;
}

SolveStatus solve_beta_qr(const double * M_cm, const double * y, int n, int t, const NnlsParams & p, double * beta_out) {
    std::fill(beta_out, beta_out + t, 0.0);
    std::vector<double> w(t);
    const SolveStatus st = lstsq_with_fallback(M_cm, n, t, y, 1, p, w.data());
    if (st == SolveStatus::Failed) return st;
    std::vector<double> tmp(n);
    auto mu = [&](const std::vector<double> & x, std::vector<double> & out) {  // out = M x
        std::fill(out.begin(), out.end(), 0.0);
        for (int i = 0; i < t; ++i) {
            const double * c = M_cm + (size_t) i * n;
            const double xi = x[i];
            for (int r = 0; r < n; ++r) out[r] += c[r] * xi;
        }
    };
    auto mtv = [&](const std::vector<double> & x, std::vector<double> & out) {  // out = M^T x
        for (int i = 0; i < t; ++i) {
            const double * c = M_cm + (size_t) i * n;
            double s = 0;
            for (int r = 0; r < n; ++r) s += c[r] * x[r];
            out[i] = s;
        }
    };
    auto gmul = [&](const std::vector<double> & x, std::vector<double> & out) { mu(x, tmp); mtv(tmp, out); };
    auto grad = [&](const std::vector<double> & x, std::vector<double> & out) {
        mu(x, tmp);
        for (int r = 0; r < n; ++r) tmp[r] -= y[r];
        mtv(tmp, out);
    };
    return finish_beta(w, t, st, p, gmul, grad, beta_out);
}

SolveStatus solve_values_qr(const double * A_cm, const double * Y_cm, int n, int t, int d, const NnlsParams & p, double * C2_out) {
    const SolveStatus st = lstsq_with_fallback(A_cm, n, t, Y_cm, d, p, C2_out);
    if (st == SolveStatus::Failed) std::fill(C2_out, C2_out + (size_t) t * d, 0.0);
    return st;
}

bool narrow_f32(const double * src, size_t n, float * dst) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(src[i]) || std::fabs(src[i]) > (double) FLT_MAX) return false;
        dst[i] = (float) src[i];
        if (!std::isfinite(dst[i])) return false;
    }
    return true;
}

// =====================================================================================
// Backend accumulation
// =====================================================================================

namespace {

// One cached graph: shapes depend on the chunk's token count and the stage.
struct ChunkGraph {
    ggml_context *   ctx = nullptr;
    ggml_cgraph *    gf = nullptr;
    ggml_gallocr_t   galloc = nullptr;
    ggml_tensor *    qc = nullptr;     // input [d, nq, n_head] F16
    ggml_tensor *    out_a = nullptr;
    ggml_tensor *    out_b = nullptr;
    ~ChunkGraph() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
};

enum Stage { BetaGram = 1, ValueGram = 2, BetaDesignS = 3, ValueDesignS = 4 };

void set_err(std::string * err, const std::string & m) { if (err) *err = m; }

// An F32 operand as hi + lo (lo null: hi is the whole operand).
struct Split { ggml_tensor * hi; ggml_tensor * lo; };

// Veltkamp split with the exponent preserved: hi = c - (c - x) with c = 8193 x keeps 11 significant bits
// (an exact TF32 operand), lo = the same split of x - hi. Unlike an F16 cast nothing underflows: operands
// down to the fp32 normal range (1e-38) keep their relative precision. 2^-22 is dropped (lo of lo).
Split vsplit(ggml_context * c, ggml_tensor * x, bool comp) {
    if (!comp) return {x, nullptr};
    auto top = [&](ggml_tensor * v) {
        ggml_tensor * k = ggml_scale(c, v, 8193.0f);
        return ggml_sub(c, k, ggml_sub(c, k, v));
    };
    ggml_tensor * hi = top(x);
    return {hi, top(ggml_sub(c, x, hi))};
}

// sum_k a[k,:] * b[k,:] with the lo*lo term dropped.
ggml_tensor * mm(ggml_context * c, const Split & a, const Split & b) {
    ggml_tensor * r = ggml_mul_mat(c, a.hi, b.hi);
    if (b.lo) r = ggml_add(c, r, ggml_mul_mat(c, a.hi, b.lo));
    if (a.lo) r = ggml_add(c, r, ggml_mul_mat(c, a.lo, b.hi));
    return r;
}

bool same_shape(const BlockShape & a, const BlockShape & b) {
    return a.head_dim == b.head_dim && a.n_tokens == b.n_tokens && a.n_head == b.n_head && a.n_head_kv == b.n_head_kv &&
           a.n_keys == b.n_keys && a.t == b.t && a.scale == b.scale && a.chunk_tokens == b.chunk_tokens &&
           a.compensated == b.compensated && a.bf16_scores == b.bf16_scores;
}

}  // namespace

struct GramAccumulator::Impl {
    ggml_backend_t        be = nullptr;
    BlockShape            s;
    int                   n_rep = 1;
    bool                  comp = false;
    ggml_context *        ctx = nullptr;     // persistent block tensors
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *k_hi = nullptr, *k_lo = nullptr, *ks_hi = nullptr, *ks_lo = nullptr, *vt_hi = nullptr, *vt_lo = nullptr;
    ggml_tensor *beta = nullptr, *sel = nullptr;
    std::vector<ggml_tensor *> qch;          // contiguous F16 [d, nq, n_head] per query chunk
    ggml_context *        sctx = nullptr;    // staging for the host form
    ggml_backend_buffer_t sbuf = nullptr;
    ggml_tensor *sq = nullptr, *sk = nullptr, *sv = nullptr;
    ggml_gallocr_t        prep_galloc = nullptr;
    std::map<std::pair<int, int>, ChunkGraph *> graphs;  // (stage, nq)
    TransferStats         xf;

    ~Impl() {
        for (auto & kv : graphs) delete kv.second;
        if (prep_galloc) ggml_gallocr_free(prep_galloc);
        if (sbuf) ggml_backend_buffer_free(sbuf);
        if (sctx) ggml_free(sctx);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }

    void put(ggml_tensor * t, const void * data) {
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        ++xf.h2d_calls;
        xf.h2d_bytes += ggml_nbytes(t);
    }
    void get(ggml_tensor * t, void * data) {
        ggml_backend_tensor_get(t, data, 0, ggml_nbytes(t));
        ++xf.d2h_calls;
        xf.d2h_bytes += ggml_nbytes(t);
    }
    ChunkGraph * graph(int stage, int nq, std::string * err);
    bool prep(const BlockTensors & in, std::string * err);
    template <class Acc> bool run(int stage, std::string * err, Acc && acc);
};

ChunkGraph * GramAccumulator::Impl::graph(int stage, int nq, std::string * err) {
    auto it = graphs.find({stage, nq});
    if (it != graphs.end()) return it->second;

    std::unique_ptr<ChunkGraph> owner(new ChunkGraph());   // freed on every exit, including an exception
    ChunkGraph * g = owner.get();
    const int d = s.head_dim, T = s.n_keys, n_kv = s.n_head_kv, R = nq * n_rep;
    ggml_init_params ip = { ggml_tensor_overhead() * 96 + ggml_graph_overhead_custom(384, false), nullptr, true };
    g->ctx = ggml_init(ip);
    ggml_context * c = g->ctx;
    g->qc = ggml_new_tensor_3d(c, GGML_TYPE_F16, d, nq, s.n_head);
    ggml_set_input(g->qc);
    // Rows of one KV head's problem are the query heads of its group, group-major: heads
    // [g*n_rep, (g+1)*n_rep) are contiguous, so this reshape stacks them as rows (row = head_local*nq + token).
    // Queries are F16, so already exact as a TF32 operand: no lo half.
    const Split qg = {ggml_reshape_3d(c, ggml_cast(c, g->qc, GGML_TYPE_F32), d, R, n_kv), nullptr};
    auto scores = [&](const Split & kk) {
        ggml_tensor * raw = mm(c, kk, qg);
        // The reference runs q.k in bf16: the raw score is rounded to bf16 before the scale.
        if (s.bf16_scores) raw = ggml_cast(c, ggml_cast(c, raw, GGML_TYPE_BF16), GGML_TYPE_F32);
        return ggml_scale(c, raw, s.scale);
    };
    ggml_tensor * sel_s = scores({ks_hi, ks_lo});   // [t, R, n_kv]
    ggml_tensor * full_s = scores({k_hi, k_lo});    // [T, R, n_kv]

    if (stage == BetaGram || stage == BetaDesignS) {
        ggml_tensor * mx = ggml_pool_2d(c, full_s, GGML_OP_POOL_MAX, T, 1, T, 1, 0, 0);  // [1, R, n_kv]
        ggml_tensor * y = ggml_sum_rows(c, ggml_exp(c, ggml_sub(c, full_s, mx)));        // [1, R, n_kv]
        ggml_tensor * es = ggml_exp(c, ggml_sub(c, sel_s, mx));                          // [t, R, n_kv]
        if (stage == BetaDesignS) {
            g->out_a = es;
            g->out_b = y;
        } else {
            ggml_tensor * et = ggml_cont(c, ggml_transpose(c, es));                      // [R, t, n_kv]
            const Split ets = vsplit(c, et, comp);
            g->out_a = mm(c, ets, ets);                                                  // M^T M
            g->out_b = mm(c, ets, vsplit(c, ggml_reshape_3d(c, y, R, 1, n_kv), comp));   // M^T y
        }
    } else {
        ggml_tensor * x = ggml_soft_max(c, ggml_add(c, sel_s, beta));                    // [t, R, n_kv]
        ggml_tensor * yv = mm(c, {vt_hi, vt_lo}, vsplit(c, ggml_soft_max(c, full_s), comp));  // [d, R, n_kv]
        if (stage == ValueDesignS) {
            g->out_a = x;
            g->out_b = yv;
        } else {
            ggml_tensor * xt = ggml_cont(c, ggml_transpose(c, x));                       // [R, t, n_kv]
            ggml_tensor * yt = ggml_cont(c, ggml_transpose(c, yv));                      // [R, d, n_kv]
            const Split xts = vsplit(c, xt, comp);
            g->out_a = mm(c, xts, xts);                                                  // A^T A
            g->out_b = mm(c, xts, vsplit(c, yt, comp));                                  // A^T Y
        }
    }
    ggml_set_output(g->out_a);
    ggml_set_output(g->out_b);
    g->gf = ggml_new_graph_custom(c, 384, false);
    ggml_build_forward_expand(g->gf, g->out_a);
    ggml_build_forward_expand(g->gf, g->out_b);
    for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i) {
        ggml_tensor * n = ggml_graph_node(g->gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            set_err(err, std::string("backend does not support ") + ggml_op_name(n->op));
            return nullptr;
        }
    }
    g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!g->galloc || !ggml_gallocr_alloc_graph(g->galloc, g->gf)) {
        set_err(err, "compute buffer allocation failed");
        return nullptr;
    }
    graphs[{stage, nq}] = g;
    owner.release();
    return g;
}

// Device-side preparation: K, V cast to F32, split into TF32-exact halves, the selected keys gathered,
// V transposed, and the query chunks copied into contiguous per-chunk tensors. No host traffic.
bool GramAccumulator::Impl::prep(const BlockTensors & in, std::string * err) {
    const int d = s.head_dim, T = s.n_keys, n_kv = s.n_head_kv;
    (void) T; (void) n_kv; (void) d;
    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cown(ggml_init(ip), ggml_free);   // freed on an exception too
    ggml_context * c = cown.get();
    if (!c) { set_err(err, "prep context allocation failed"); return false; }
    ggml_cgraph * gf = ggml_new_graph_custom(c, 512, false);
    auto f32 = [&](ggml_tensor * x) { return x->type == GGML_TYPE_F32 ? ggml_cont(c, x) : ggml_cast(c, x, GGML_TYPE_F32); };
    ggml_tensor * k32 = f32(in.k);
    ggml_tensor * v32 = f32(in.v);
    ggml_tensor * ks = ggml_get_rows(c, in.k, sel);                         // [d, t, n_kv] F32
    ggml_tensor * vt = ggml_cont(c, ggml_transpose(c, v32));                // [T, d, n_kv]
    auto store = [&](ggml_tensor * x, ggml_tensor * hi, ggml_tensor * lo) {
        const Split sp = vsplit(c, x, comp);
        ggml_build_forward_expand(gf, ggml_cpy(c, sp.hi, hi));
        if (lo) ggml_build_forward_expand(gf, ggml_cpy(c, sp.lo, lo));
    };
    store(k32, k_hi, k_lo);
    store(ks, ks_hi, ks_lo);
    store(vt, vt_hi, vt_lo);
    for (size_t ci = 0; ci < qch.size(); ++ci) {
        const int t0 = (int) ci * s.chunk_tokens;
        ggml_tensor * src = ggml_view_3d(c, in.q, qch[ci]->ne[0], qch[ci]->ne[1], qch[ci]->ne[2], in.q->nb[1], in.q->nb[2],
                                         (size_t) t0 * in.q->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(c, src, qch[ci]));
    }
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf) && ok; ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            set_err(err, std::string("backend does not support ") + ggml_op_name(n->op));
            ok = false;
        }
    }
    if (ok && !prep_galloc) prep_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (ok && (!prep_galloc || !ggml_gallocr_alloc_graph(prep_galloc, gf))) {
        set_err(err, "prep buffer allocation failed");
        ok = false;
    }
    if (ok && ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        set_err(err, "prep graph compute failed");
        ok = false;
    }
    return ok;
}

// Runs every query chunk of a stage and hands the two outputs (host F32) to acc(a, b, token0, nq).
template <class Acc>
bool GramAccumulator::Impl::run(int stage, std::string * err, Acc && acc) {
    std::vector<float> a, b;
    for (size_t ci = 0; ci < qch.size(); ++ci) {
        const int t0 = (int) ci * s.chunk_tokens;
        const int nq = (int) qch[ci]->ne[1];
        ChunkGraph * g = graph(stage, nq, err);
        if (!g) return false;
        ggml_backend_tensor_copy(qch[ci], g->qc);  // device to device
        if (ggml_backend_graph_compute(be, g->gf) != GGML_STATUS_SUCCESS) {
            set_err(err, "graph compute failed");
            return false;
        }
        a.resize(ggml_nelements(g->out_a));
        b.resize(ggml_nelements(g->out_b));
        get(g->out_a, a.data());
        get(g->out_b, b.data());
        acc(a, b, t0, nq);
    }
    return true;
}

GramAccumulator::~GramAccumulator() { delete impl_; }

namespace {
bool valid_shape(const BlockShape & s) {
    return s.head_dim > 0 && s.n_tokens > 0 && s.n_head > 0 && s.n_head_kv > 0 && s.n_keys > 0 && s.t > 0 &&
           s.chunk_tokens > 0 && s.n_head % s.n_head_kv == 0 && s.t <= s.n_keys;
}
}  // namespace

bool GramAccumulator::init(ggml_backend_t backend, const BlockShape & s, const BlockTensors & in, std::string * err) {
    if (!backend || !in.q || !in.k || !in.v || !in.sel) { set_err(err, "null input"); return false; }
    if (!valid_shape(s)) { set_err(err, "bad block shape"); return false; }
    const int d = s.head_dim, T = s.n_keys, t = s.t, n_kv = s.n_head_kv;
    auto bad = [&](const ggml_tensor * x, int64_t n1, int64_t n2, bool is_q) {
        const bool type_ok = is_q ? x->type == GGML_TYPE_F16 : (x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32);
        return !type_ok || x->ne[0] != d || x->ne[1] != n1 || x->ne[2] != n2 || x->nb[0] != ggml_type_size(x->type);
    };
    if (bad(in.q, s.n_tokens, s.n_head, true) || bad(in.k, T, n_kv, false) || bad(in.v, T, n_kv, false)) {
        set_err(err, "q/k/v tensor shape or type does not match the block shape");
        return false;
    }
    for (size_t i = 0; i < (size_t) t * n_kv; ++i) {
        if (in.sel[i] < 0 || in.sel[i] >= T) { set_err(err, "selected index out of range"); return false; }
    }

    // Same backend and shape as the previous block: keep the device buffers and the compiled graphs,
    // only the data changes.
    if (impl_ && (impl_->be != backend || !same_shape(impl_->s, s))) { delete impl_; impl_ = nullptr; }
    if (!impl_) {
        std::unique_ptr<Impl> impl(new Impl());
        impl->be = backend;
        impl->s = s;
        impl->n_rep = s.n_head / s.n_head_kv;
        const int n_chunks = (s.n_tokens + s.chunk_tokens - 1) / s.chunk_tokens;
        ggml_init_params ip = { ggml_tensor_overhead() * (16 + n_chunks), nullptr, true };
        impl->ctx = ggml_init(ip);
        impl->comp = s.compensated < 0
            ? ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU
            : s.compensated != 0;
        auto f32_3d = [&](int64_t a, int64_t b, int64_t c) { return ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, a, b, c); };
        impl->k_hi  = f32_3d(d, T, n_kv);
        impl->ks_hi = f32_3d(d, t, n_kv);
        impl->vt_hi = f32_3d(T, d, n_kv);
        if (impl->comp) {
            impl->k_lo  = f32_3d(d, T, n_kv);
            impl->ks_lo = f32_3d(d, t, n_kv);
            impl->vt_lo = f32_3d(T, d, n_kv);
        }
        impl->beta = f32_3d(t, 1, n_kv);
        impl->sel = ggml_new_tensor_2d(impl->ctx, GGML_TYPE_I32, t, n_kv);
        for (int t0 = 0; t0 < s.n_tokens; t0 += s.chunk_tokens)
            impl->qch.push_back(ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F16, d, std::min(s.chunk_tokens, s.n_tokens - t0), s.n_head));
        impl->buf = ggml_backend_alloc_ctx_tensors(impl->ctx, backend);
        if (!impl->buf) { set_err(err, "block buffer allocation failed"); return false; }
        impl_ = impl.release();
    }
    impl_->put(impl_->sel, in.sel);
    return impl_->prep(in, err);
}

bool GramAccumulator::init(ggml_backend_t backend, const BlockShape & s, const BlockInputs & in, std::string * err) {
    if (!backend || !in.q || !in.k || !in.v || !in.sel) { set_err(err, "null input"); return false; }
    if (!valid_shape(s)) { set_err(err, "bad block shape"); return false; }
    const int d = s.head_dim;
    // Staging tensors of the host form live next to the block state; they follow its lifetime.
    if (impl_ && (impl_->be != backend || !same_shape(impl_->s, s))) { delete impl_; impl_ = nullptr; }
    if (!impl_ || !impl_->sbuf) {
        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
        std::unique_ptr<ggml_context, decltype(&ggml_free)> sown(ggml_init(ip), ggml_free);
        ggml_context * sctx = sown.get();
        if (!sctx) { set_err(err, "staging context allocation failed"); return false; }
        ggml_tensor * sq = ggml_new_tensor_3d(sctx, GGML_TYPE_F16, d, s.n_tokens, s.n_head);
        ggml_tensor * sk = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, d, s.n_keys, s.n_head_kv);
        ggml_tensor * sv = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, d, s.n_keys, s.n_head_kv);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> bown(
            ggml_backend_alloc_ctx_tensors(sctx, backend), ggml_backend_buffer_free);
        ggml_backend_buffer_t sbuf = bown.get();
        if (!sbuf) { set_err(err, "staging buffer allocation failed"); return false; }
        BlockTensors bt{sq, sk, sv, in.sel};
        // The device form creates impl_ when it is missing; attach the staging afterwards.
        ggml_backend_tensor_set(sq, in.q, 0, ggml_nbytes(sq));
        ggml_backend_tensor_set(sk, in.k, 0, ggml_nbytes(sk));
        ggml_backend_tensor_set(sv, in.v, 0, ggml_nbytes(sv));
        if (!init(backend, s, bt, err)) return false;
        impl_->sctx = sown.release(); impl_->sbuf = bown.release(); impl_->sq = sq; impl_->sk = sk; impl_->sv = sv;
        impl_->xf.h2d_calls += 3;
        impl_->xf.h2d_bytes += ggml_nbytes(sq) + ggml_nbytes(sk) + ggml_nbytes(sv);
        return true;
    }
    impl_->put(impl_->sq, in.q);
    impl_->put(impl_->sk, in.k);
    impl_->put(impl_->sv, in.v);
    BlockTensors bt{impl_->sq, impl_->sk, impl_->sv, in.sel};
    return init(backend, s, bt, err);
}

TransferStats GramAccumulator::transfers() const { return impl_ ? impl_->xf : TransferStats{}; }

bool GramAccumulator::beta_grams(BetaGrams & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, n_kv = impl_->s.n_head_kv;
    out.MtM.assign((size_t) n_kv * t * t, 0.0);
    out.Mty.assign((size_t) n_kv * t, 0.0);
    return impl_->run(BetaGram, err, [&](const std::vector<float> & a, const std::vector<float> & b, int, int) {
        // ggml [t, t, n_kv] and [t, 1, n_kv]; G is symmetric so the index order is immaterial.
        for (size_t i = 0; i < a.size(); ++i) out.MtM[i] += a[i];
        for (size_t i = 0; i < b.size(); ++i) out.Mty[i] += b[i];
    });
}

bool GramAccumulator::value_grams(const float * beta, ValueGrams & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, d = impl_->s.head_dim, n_kv = impl_->s.n_head_kv;
    impl_->put(impl_->beta, beta);
    out.AtA.assign((size_t) n_kv * t * t, 0.0);
    out.AtY.assign((size_t) n_kv * t * d, 0.0);
    return impl_->run(ValueGram, err, [&](const std::vector<float> & a, const std::vector<float> & b, int, int) {
        for (size_t i = 0; i < a.size(); ++i) out.AtA[i] += a[i];
        // ggml [t, d, n_kv]: (slot i, dim c) at i + c*t; stored row-major [slot][dim].
        for (int g = 0; g < n_kv; ++g)
            for (int c = 0; c < d; ++c)
                for (int i = 0; i < t; ++i) out.AtY[((size_t) g * t + i) * d + c] += b[((size_t) g * d + c) * t + i];
    });
}

bool GramAccumulator::beta_design(BetaDesign & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, n_kv = impl_->s.n_head_kv, n_rep = impl_->n_rep;
    const size_t rows = (size_t) impl_->s.n_tokens * n_rep;
    out.rows = (int) rows;
    out.M.assign((size_t) n_kv * t * rows, 0.0);
    out.y.assign((size_t) n_kv * rows, 0.0);
    return impl_->run(BetaDesignS, err, [&](const std::vector<float> & es, const std::vector<float> & y, int t0, int nq) {
        const size_t R = (size_t) nq * n_rep, row0 = (size_t) t0 * n_rep;  // chunk rows are contiguous
        for (int g = 0; g < n_kv; ++g)
            for (size_t r = 0; r < R; ++r) {
                out.y[g * rows + row0 + r] = y[r + R * g];
                for (int i = 0; i < t; ++i) out.M[((size_t) g * t + i) * rows + row0 + r] = es[i + t * (r + R * g)];
            }
    });
}

bool GramAccumulator::value_design(const float * beta, ValueDesign & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, d = impl_->s.head_dim, n_kv = impl_->s.n_head_kv, n_rep = impl_->n_rep;
    const size_t rows = (size_t) impl_->s.n_tokens * n_rep;
    impl_->put(impl_->beta, beta);
    out.rows = (int) rows;
    out.A.assign((size_t) n_kv * t * rows, 0.0);
    out.Y.assign((size_t) n_kv * d * rows, 0.0);
    return impl_->run(ValueDesignS, err, [&](const std::vector<float> & x, const std::vector<float> & yv, int t0, int nq) {
        const size_t R = (size_t) nq * n_rep, row0 = (size_t) t0 * n_rep;
        for (int g = 0; g < n_kv; ++g)
            for (size_t r = 0; r < R; ++r) {
                for (int i = 0; i < t; ++i) out.A[((size_t) g * t + i) * rows + row0 + r] = x[i + t * (r + R * g)];
                for (int c = 0; c < d; ++c) out.Y[((size_t) g * d + c) * rows + row0 + r] = yv[c + (size_t) d * (r + R * g)];
            }
    });
}

// =====================================================================================
// Whole-block fit
// =====================================================================================

namespace {

bool fit_after_init(GramAccumulator & acc, const BlockShape & shape, const NnlsParams & p_in, FitResult & out, std::string * err,
                    std::chrono::steady_clock::time_point t0, const TransferStats & x0) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const int t = shape.t, d = shape.head_dim, n_kv = shape.n_head_kv;
    NnlsParams p = p_in;
    p.gram_ridge = false;
    const bool qr = p.route == SolveRoute::Qr;
    const auto t1 = clk::now();

    std::vector<float> beta_f((size_t) t * n_kv, 0.0f), c2_f((size_t) d * t * n_kv, 0.0f);
    std::vector<SolveStatus> status(n_kv, SolveStatus::Ok);
    std::vector<double> beta(t), c2((size_t) t * d);
    BetaGrams bg; BetaDesign bd; ValueGrams vg; ValueDesign vd;

    if (!(qr ? acc.beta_design(bd, err) : acc.beta_grams(bg, err))) return false;
    const auto t2 = clk::now();
    int fallbacks = 0;
    bool have_bd = qr;
    for (int g = 0; g < n_kv; ++g) {
        status[g] = qr ? solve_beta_qr(&bd.M[(size_t) g * t * bd.rows], &bd.y[(size_t) g * bd.rows], bd.rows, t, p, beta.data())
                       : solve_beta(&bg.MtM[(size_t) g * t * t], &bg.Mty[(size_t) g * t], t, p, beta.data());
        if (!qr && status[g] != SolveStatus::Ok) {  // Cholesky failed or non-finite: redo this head on the QR route
            if (!have_bd) { if (!acc.beta_design(bd, err)) return false; have_bd = true; }
            status[g] = solve_beta_qr(&bd.M[(size_t) g * t * bd.rows], &bd.y[(size_t) g * bd.rows], bd.rows, t, p, beta.data());
            ++fallbacks;
        }
        if (status[g] != SolveStatus::Failed && !narrow_f32(beta.data(), t, &beta_f[(size_t) g * t])) status[g] = SolveStatus::Failed;
    }
    const auto t3 = clk::now();
    // A head whose beta solve failed has no valid beta: stop before the value stage.
    for (int g = 0; g < n_kv; ++g)
        if (status[g] == SolveStatus::Failed) { set_err(err, "head " + std::to_string(g) + ": beta solve failed"); return false; }

    if (!(qr ? acc.value_design(beta_f.data(), vd, err) : acc.value_grams(beta_f.data(), vg, err))) return false;
    const auto t4 = clk::now();
    bool have_vd = qr;
    for (int g = 0; g < n_kv; ++g) {
        SolveStatus sv = qr ? solve_values_qr(&vd.A[(size_t) g * t * vd.rows], &vd.Y[(size_t) g * d * vd.rows], vd.rows, t, d, p, c2.data())
                            : solve_values(&vg.AtA[(size_t) g * t * t], &vg.AtY[(size_t) g * t * d], t, d, p, c2.data());
        if (!qr && sv != SolveStatus::Ok) {
            if (!have_vd) { if (!acc.value_design(beta_f.data(), vd, err)) return false; have_vd = true; }
            sv = solve_values_qr(&vd.A[(size_t) g * t * vd.rows], &vd.Y[(size_t) g * d * vd.rows], vd.rows, t, d, p, c2.data());
            ++fallbacks;
        }
        if (sv > status[g]) status[g] = sv;  // Ok < RidgeUsed < Failed
        // A finite double outside the F32 range must not be published: fail the block.
        if (sv != SolveStatus::Failed && !narrow_f32(c2.data(), c2.size(), &c2_f[(size_t) g * t * d])) status[g] = SolveStatus::Failed;
    }
    const auto t5 = clk::now();
    for (int g = 0; g < n_kv; ++g)
        if (status[g] == SolveStatus::Failed) { set_err(err, "head " + std::to_string(g) + ": value solve failed"); return false; }

    out.beta = std::move(beta_f);
    out.c2 = std::move(c2_f);
    out.status = std::move(status);
    out.ms_init = ms(t0, t1);
    out.ms_beta_grams = ms(t1, t2);
    out.ms_beta_solve = ms(t2, t3);
    out.ms_value_grams = ms(t3, t4);
    out.ms_value_solve = ms(t4, t5);
    const TransferStats x1 = acc.transfers();
    out.qr_fallbacks = fallbacks;
    out.transfers = {x1.h2d_calls - x0.h2d_calls, x1.d2h_calls - x0.d2h_calls, x1.h2d_bytes - x0.h2d_bytes, x1.d2h_bytes - x0.d2h_bytes};
    return true;
}

void clear(FitResult & out) { out = FitResult{}; }

}  // namespace

bool fit_block(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err) {
    GramAccumulator acc;
    return fit_block(acc, backend, shape, in, p, out, err);
}

bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err) {
    clear(out);
    const auto t0 = std::chrono::steady_clock::now();
    const TransferStats x0 = acc.transfers();
    if (!acc.init(backend, shape, in, err)) return false;
    const bool ok = fit_after_init(acc, shape, p, out, err, t0, x0);
    if (!ok) clear(out);
    return ok;
}

bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockTensors & in,
               const NnlsParams & p, FitResult & out, std::string * err) {
    clear(out);
    const auto t0 = std::chrono::steady_clock::now();
    const TransferStats x0 = acc.transfers();
    if (!acc.init(backend, shape, in, err)) return false;
    const bool ok = fit_after_init(acc, shape, p, out, err, t0, x0);
    if (!ok) clear(out);
    return ok;
}

}  // namespace luce_am
