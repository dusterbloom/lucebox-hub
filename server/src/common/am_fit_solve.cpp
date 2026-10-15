#include "common/am_fit_solve.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>

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

// X = (sym(G))^-1 B, with the reference's fallback: on failure add `ridge` to the diagonal and retry.
SolveStatus solve_spd(const double * G, const double * B, int t, int nrhs, const NnlsParams & p, double * X) {
    std::vector<double> sym((size_t) t * t);
    for (int i = 0; i < t; ++i)
        for (int j = 0; j < t; ++j) sym[(size_t) i * t + j] = 0.5 * (G[(size_t) i * t + j] + G[(size_t) j * t + i]);
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::vector<double> l(sym);
        if (attempt == 1) for (int i = 0; i < t; ++i) l[(size_t) i * t + i] += p.ridge;
        if (!cholesky(l, t, p.chol_rel_tol)) continue;
        std::memcpy(X, B, sizeof(double) * (size_t) t * nrhs);
        cholesky_solve(l, t, nrhs, X);
        if (all_finite(X, (size_t) t * nrhs)) return attempt == 0 ? SolveStatus::Ok : SolveStatus::RidgeUsed;
    }
    return SolveStatus::Failed;
}

}  // namespace

SolveStatus solve_beta(const double * G, const double * b, int t, const NnlsParams & p, double * beta_out) {
    std::fill(beta_out, beta_out + t, 0.0);
    std::vector<double> w(t);
    const SolveStatus st = solve_spd(G, b, t, 1, p, w.data());  // least-squares init
    if (st == SolveStatus::Failed) return st;
    for (double & v : w) v = std::min(std::max(v, p.lo), p.hi);

    // L = ||M||_2^2 by power iteration on M^T M (non-negative, so the ones vector has overlap with
    // the top eigenvector; the reference starts from a random vector).
    std::vector<double> u(t, 1.0 / std::sqrt((double) t)), v(t);
    auto gmul = [&](const std::vector<double> & x, std::vector<double> & out) {
        for (int i = 0; i < t; ++i) {
            double s = 0;
            for (int j = 0; j < t; ++j) s += 0.5 * (G[(size_t) i * t + j] + G[(size_t) j * t + i]) * x[j];
            out[i] = s;
        }
    };
    for (int it = 0; it < p.power_iters; ++it) {
        gmul(u, v);
        double nrm = 0;
        for (double x : v) nrm += x * x;
        nrm = std::sqrt(nrm);
        if (!(nrm > 0.0)) break;
        for (int i = 0; i < t; ++i) u[i] = v[i] / nrm;
    }
    gmul(u, v);
    double rq = 0;
    for (int i = 0; i < t; ++i) rq += u[i] * v[i];
    const double sigma = std::max(std::sqrt(std::max(rq, 0.0)), 1e-6);
    const double L = std::max(sigma * sigma, 1e-6);

    // Projected gradient on 0.5 |M w - y|^2: grad = G w - b.
    std::vector<double> gw(t);
    for (int it = 0; it < p.pg_iters; ++it) {
        gmul(w, gw);
        for (int i = 0; i < t; ++i) w[i] = std::min(std::max(w[i] - (gw[i] - b[i]) / L, p.lo), p.hi);
    }
    for (int i = 0; i < t; ++i) beta_out[i] = std::log(w[i]);
    if (!all_finite(beta_out, t)) {
        std::fill(beta_out, beta_out + t, 0.0);
        return SolveStatus::Failed;
    }
    return st;
}

SolveStatus solve_values(const double * AtA, const double * AtY, int t, int d, const NnlsParams & p, double * C2_out) {
    const SolveStatus st = solve_spd(AtA, AtY, t, d, p, C2_out);
    if (st == SolveStatus::Failed) std::fill(C2_out, C2_out + (size_t) t * d, 0.0);
    return st;
}

// =====================================================================================
// Backend Gram accumulation
// =====================================================================================

namespace {

// One cached graph: shapes depend on the chunk's token count and the stage.
struct ChunkGraph {
    ggml_context *   ctx = nullptr;
    ggml_cgraph *    gf = nullptr;
    ggml_gallocr_t   galloc = nullptr;
    ggml_tensor *    qc = nullptr;     // input [d, nq, n_head] F16
    ggml_tensor *    out_a = nullptr;  // G   (stage 1: M^T M, stage 2: A^T A)  [t, t, n_kv]
    ggml_tensor *    out_b = nullptr;  // rhs (stage 1: M^T y [t,1,n_kv],       stage 2: A^T Y [t,d,n_kv])
    ~ChunkGraph() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
    }
};

}  // namespace

struct GramAccumulator::Impl {
    ggml_backend_t        be = nullptr;
    BlockShape            s;
    int                   n_rep = 1;
    const ggml_fp16_t *   q = nullptr;
    ggml_context *        ctx = nullptr;     // persistent block tensors
    ggml_backend_buffer_t buf = nullptr;
    bool                  comp = false;      // compensated (split) matmuls
    // Block operands as {hi, lo}: lo is null unless comp. hi == the plain operand otherwise.
    ggml_tensor *k_hi = nullptr, *k_lo = nullptr, *ks_hi = nullptr, *ks_lo = nullptr, *vt_hi = nullptr, *vt_lo = nullptr;
    ggml_tensor *beta = nullptr;
    std::map<std::pair<int, int>, ChunkGraph *> graphs;  // (stage, nq)

    ~Impl() {
        for (auto & kv : graphs) delete kv.second;
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }

    ChunkGraph * graph(int stage, int nq, std::string * err);
    template <class Acc> bool run(int stage, std::string * err, Acc && acc);
};

namespace {
void set_err(std::string * err, const std::string & m) { if (err) *err = m; }

// x ~ hi + lo with hi and lo each carrying 11 significant bits, so both are exact TF32 operands
// (the same precision as the F16 casts of graph_split, without the F16 exponent range).
inline float round11(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    u = (u + 0x1000u) & ~0x1FFFu;  // round to nearest, keep sign, 8 exponent and 10 mantissa bits
    std::memcpy(&x, &u, sizeof u);
    return x;
}
void host_split(const float * x, size_t n, std::vector<float> & hi, std::vector<float> & lo) {
    hi.resize(n);
    lo.resize(n);
    for (size_t i = 0; i < n; ++i) {
        hi[i] = round11(x[i]);
        lo[i] = round11(x[i] - hi[i]);
    }
}

// An F32 operand as hi + lo (lo null: hi is the whole operand).
struct Split { ggml_tensor * hi; ggml_tensor * lo; };

Split graph_split(ggml_context * c, ggml_tensor * x, bool comp) {
    if (!comp) return {x, nullptr};
    ggml_tensor * hi = ggml_cast(c, ggml_cast(c, x, GGML_TYPE_F16), GGML_TYPE_F32);
    ggml_tensor * lo = ggml_cast(c, ggml_cast(c, ggml_sub(c, x, hi), GGML_TYPE_F16), GGML_TYPE_F32);
    return {hi, lo};
}

// sum_k a[k,:] * b[k,:] with the lo*lo term dropped (~2^-22 relative).
ggml_tensor * mm(ggml_context * c, const Split & a, const Split & b) {
    ggml_tensor * r = ggml_mul_mat(c, a.hi, b.hi);
    if (b.lo) r = ggml_add(c, r, ggml_mul_mat(c, a.hi, b.lo));
    if (a.lo) r = ggml_add(c, r, ggml_mul_mat(c, a.lo, b.hi));
    return r;
}
}

ChunkGraph * GramAccumulator::Impl::graph(int stage, int nq, std::string * err) {
    auto it = graphs.find({stage, nq});
    if (it != graphs.end()) return it->second;

    auto * g = new ChunkGraph();
    const int d = s.head_dim, T = s.n_keys, n_kv = s.n_head_kv, R = nq * n_rep;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(256, false), nullptr, true };
    g->ctx = ggml_init(ip);
    ggml_context * c = g->ctx;
    g->qc = ggml_new_tensor_3d(c, GGML_TYPE_F16, d, nq, s.n_head);
    ggml_set_input(g->qc);
    // Rows of one KV head's problem are the query heads of its group, group-major: heads
    // [g*n_rep, (g+1)*n_rep) are contiguous, so this reshape stacks them as rows.
    // Queries are F16, so already exact as a TF32 operand: no lo half.
    const Split qg = {ggml_reshape_3d(c, ggml_cast(c, g->qc, GGML_TYPE_F32), d, R, n_kv), nullptr};
    ggml_tensor * sel_s = ggml_scale(c, mm(c, {ks_hi, ks_lo}, qg), s.scale);       // [t, R, n_kv]
    ggml_tensor * full_s = ggml_scale(c, mm(c, {k_hi, k_lo}, qg), s.scale);        // [T, R, n_kv]

    if (stage == 1) {
        ggml_tensor * mx = ggml_pool_2d(c, full_s, GGML_OP_POOL_MAX, T, 1, T, 1, 0, 0);  // [1, R, n_kv]
        ggml_tensor * y = ggml_sum_rows(c, ggml_exp(c, ggml_sub(c, full_s, mx)));        // [1, R, n_kv]
        ggml_tensor * es = ggml_exp(c, ggml_sub(c, sel_s, mx));                          // [t, R, n_kv]
        ggml_tensor * et = ggml_cont(c, ggml_transpose(c, es));                          // [R, t, n_kv]
        const Split ets = graph_split(c, et, comp);
        g->out_a = mm(c, ets, ets);                                                      // M^T M
        g->out_b = mm(c, ets, graph_split(c, ggml_reshape_3d(c, y, R, 1, n_kv), comp));  // M^T y
    } else {
        ggml_tensor * x = ggml_soft_max(c, ggml_add(c, sel_s, beta));                    // [t, R, n_kv]
        ggml_tensor * yv = mm(c, {vt_hi, vt_lo}, graph_split(c, ggml_soft_max(c, full_s), comp));  // [d, R, n_kv]
        ggml_tensor * xt = ggml_cont(c, ggml_transpose(c, x));                           // [R, t, n_kv]
        ggml_tensor * yt = ggml_cont(c, ggml_transpose(c, yv));                          // [R, d, n_kv]
        const Split xts = graph_split(c, xt, comp);
        g->out_a = mm(c, xts, xts);                                                      // A^T A
        g->out_b = mm(c, xts, graph_split(c, yt, comp));                                 // A^T Y
    }
    ggml_set_output(g->out_a);
    ggml_set_output(g->out_b);
    g->gf = ggml_new_graph_custom(c, 256, false);
    ggml_build_forward_expand(g->gf, g->out_a);
    ggml_build_forward_expand(g->gf, g->out_b);
    for (int i = 0; i < ggml_graph_n_nodes(g->gf); ++i) {
        ggml_tensor * n = ggml_graph_node(g->gf, i);
        if (!ggml_backend_supports_op(be, n)) {
            set_err(err, std::string("backend does not support ") + ggml_op_name(n->op));
            delete g;
            return nullptr;
        }
    }
    g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!g->galloc || !ggml_gallocr_alloc_graph(g->galloc, g->gf)) {
        set_err(err, "compute buffer allocation failed");
        delete g;
        return nullptr;
    }
    graphs[{stage, nq}] = g;
    return g;
}

// Runs every query chunk of a stage and hands the two outputs (host F32) to acc(out_a, out_b).
template <class Acc>
bool GramAccumulator::Impl::run(int stage, std::string * err, Acc && acc) {
    const int d = s.head_dim;
    std::vector<float> a, b;
    std::vector<ggml_fp16_t> qstage;
    for (int t0 = 0; t0 < s.n_tokens; t0 += s.chunk_tokens) {
        const int nq = std::min(s.chunk_tokens, s.n_tokens - t0);
        ChunkGraph * g = graph(stage, nq, err);
        if (!g) return false;
        // One contiguous upload per chunk: the chunk of every head, head-major.
        qstage.resize((size_t) d * nq * s.n_head);
        for (int h = 0; h < s.n_head; ++h)
            std::memcpy(&qstage[(size_t) h * d * nq], q + ((size_t) h * s.n_tokens + t0) * d, sizeof(ggml_fp16_t) * d * nq);
        ggml_backend_tensor_set(g->qc, qstage.data(), 0, ggml_nbytes(g->qc));
        if (ggml_backend_graph_compute(be, g->gf) != GGML_STATUS_SUCCESS) {
            set_err(err, "graph compute failed");
            return false;
        }
        a.resize(ggml_nelements(g->out_a));
        b.resize(ggml_nelements(g->out_b));
        ggml_backend_tensor_get(g->out_a, a.data(), 0, ggml_nbytes(g->out_a));
        ggml_backend_tensor_get(g->out_b, b.data(), 0, ggml_nbytes(g->out_b));
        acc(a, b);
    }
    return true;
}

GramAccumulator::~GramAccumulator() { delete impl_; }

namespace {
bool same_shape(const BlockShape & a, const BlockShape & b) {
    return a.head_dim == b.head_dim && a.n_tokens == b.n_tokens && a.n_head == b.n_head && a.n_head_kv == b.n_head_kv &&
           a.n_keys == b.n_keys && a.t == b.t && a.scale == b.scale && a.chunk_tokens == b.chunk_tokens &&
           a.compensated == b.compensated;
}
}  // namespace

bool GramAccumulator::init(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in, std::string * err) {
    const BlockShape & s = shape;
    if (!backend || !in.q || !in.k || !in.v || !in.sel) { set_err(err, "null input"); return false; }
    if (s.head_dim <= 0 || s.n_tokens <= 0 || s.n_head <= 0 || s.n_head_kv <= 0 || s.n_keys <= 0 || s.t <= 0 ||
        s.chunk_tokens <= 0 || s.n_head % s.n_head_kv != 0 || s.t > s.n_keys) {
        set_err(err, "bad block shape");
        return false;
    }
    for (size_t i = 0; i < (size_t) s.t * s.n_head_kv; ++i) {
        if (in.sel[i] < 0 || in.sel[i] >= s.n_keys) { set_err(err, "selected index out of range"); return false; }
    }
    const int d = s.head_dim, T = s.n_keys, t = s.t, n_kv = s.n_head_kv;

    // Same backend and shape as the previous block: keep the device buffers and the compiled graphs,
    // only the data changes.
    if (impl_ && (impl_->be != backend || !same_shape(impl_->s, s))) { delete impl_; impl_ = nullptr; }
    if (!impl_) {
        auto impl = new Impl();
        impl->be = backend;
        impl->s = s;
        impl->n_rep = s.n_head / s.n_head_kv;
        ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
        impl->ctx = ggml_init(ip);
        impl->comp = s.compensated < 0
            ? ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU
            : s.compensated != 0;
        impl->k_hi  = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, d, T, n_kv);
        impl->ks_hi = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, d, t, n_kv);
        impl->vt_hi = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, T, d, n_kv);
        if (impl->comp) {
            impl->k_lo  = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, d, T, n_kv);
            impl->ks_lo = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, d, t, n_kv);
            impl->vt_lo = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, T, d, n_kv);
        }
        impl->beta = ggml_new_tensor_3d(impl->ctx, GGML_TYPE_F32, t, 1, n_kv);
        impl->buf = ggml_backend_alloc_ctx_tensors(impl->ctx, backend);
        if (!impl->buf) { set_err(err, "block buffer allocation failed"); delete impl; return false; }
        impl_ = impl;
    }
    Impl * impl = impl_;
    impl->q = in.q;

    std::vector<float> ksel((size_t) d * t * n_kv), vt((size_t) T * d * n_kv);
    for (int g = 0; g < n_kv; ++g) {
        for (int i = 0; i < t; ++i)
            std::memcpy(&ksel[((size_t) g * t + i) * d], in.k + ((size_t) g * T + in.sel[(size_t) g * t + i]) * d, sizeof(float) * d);
        for (int j = 0; j < T; ++j)
            for (int c = 0; c < d; ++c) vt[((size_t) g * d + c) * T + j] = in.v[((size_t) g * T + j) * d + c];
    }
    auto upload = [&](ggml_tensor * hi, ggml_tensor * lo, const float * x, size_t n) {
        if (!lo) { ggml_backend_tensor_set(hi, x, 0, ggml_nbytes(hi)); return; }
        std::vector<float> h, l;
        host_split(x, n, h, l);
        ggml_backend_tensor_set(hi, h.data(), 0, ggml_nbytes(hi));
        ggml_backend_tensor_set(lo, l.data(), 0, ggml_nbytes(lo));
    };
    upload(impl->k_hi, impl->k_lo, in.k, (size_t) d * T * n_kv);
    upload(impl->ks_hi, impl->ks_lo, ksel.data(), ksel.size());
    upload(impl->vt_hi, impl->vt_lo, vt.data(), vt.size());
    return true;
}

bool GramAccumulator::beta_grams(BetaGrams & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, n_kv = impl_->s.n_head_kv;
    out.MtM.assign((size_t) n_kv * t * t, 0.0);
    out.Mty.assign((size_t) n_kv * t, 0.0);
    return impl_->run(1, err, [&](const std::vector<float> & a, const std::vector<float> & b) {
        // ggml [t, t, n_kv] and [t, 1, n_kv]; G is symmetric so the index order is immaterial.
        for (size_t i = 0; i < a.size(); ++i) out.MtM[i] += a[i];
        for (size_t i = 0; i < b.size(); ++i) out.Mty[i] += b[i];
    });
}

bool GramAccumulator::value_grams(const float * beta, ValueGrams & out, std::string * err) {
    if (!impl_) { set_err(err, "not initialised"); return false; }
    const int t = impl_->s.t, d = impl_->s.head_dim, n_kv = impl_->s.n_head_kv;
    ggml_backend_tensor_set(impl_->beta, beta, 0, ggml_nbytes(impl_->beta));
    out.AtA.assign((size_t) n_kv * t * t, 0.0);
    out.AtY.assign((size_t) n_kv * t * d, 0.0);
    return impl_->run(2, err, [&](const std::vector<float> & a, const std::vector<float> & b) {
        for (size_t i = 0; i < a.size(); ++i) out.AtA[i] += a[i];
        // ggml [t, d, n_kv]: (slot i, dim c) at i + c*t; stored row-major [slot][dim].
        for (int g = 0; g < n_kv; ++g)
            for (int c = 0; c < d; ++c)
                for (int i = 0; i < t; ++i) out.AtY[((size_t) g * t + i) * d + c] += b[((size_t) g * d + c) * t + i];
    });
}

bool fit_block(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err) {
    GramAccumulator acc;
    return fit_block(acc, backend, shape, in, p, out, err);
}

bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t0 = clk::now();
    if (!acc.init(backend, shape, in, err)) return false;
    const int t = shape.t, d = shape.head_dim, n_kv = shape.n_head_kv;

    const auto t1 = clk::now();
    BetaGrams bg;
    if (!acc.beta_grams(bg, err)) return false;
    const auto t2 = clk::now();
    out.beta.assign((size_t) t * n_kv, 0.0f);
    out.status.assign(n_kv, SolveStatus::Ok);
    std::vector<double> beta(t);
    for (int g = 0; g < n_kv; ++g) {
        out.status[g] = solve_beta(&bg.MtM[(size_t) g * t * t], &bg.Mty[(size_t) g * t], t, p, beta.data());
        for (int i = 0; i < t; ++i) out.beta[(size_t) g * t + i] = (float) beta[i];
    }

    const auto t3 = clk::now();
    ValueGrams vg;
    if (!acc.value_grams(out.beta.data(), vg, err)) return false;
    const auto t4 = clk::now();
    out.c2.assign((size_t) d * t * n_kv, 0.0f);
    std::vector<double> c2((size_t) t * d);
    for (int g = 0; g < n_kv; ++g) {
        const SolveStatus sv = solve_values(&vg.AtA[(size_t) g * t * t], &vg.AtY[(size_t) g * t * d], t, d, p, c2.data());
        if (sv > out.status[g]) out.status[g] = sv;  // Ok < RidgeUsed < Failed
        for (size_t i = 0; i < c2.size(); ++i) out.c2[(size_t) g * t * d + i] = (float) c2[i];
    }
    const auto t5 = clk::now();
    out.ms_init = ms(t0, t1);
    out.ms_beta_grams = ms(t1, t2);
    out.ms_beta_solve = ms(t2, t3);
    out.ms_value_grams = ms(t3, t4);
    out.ms_value_solve = ms(t4, t5);
    return true;
}

}  // namespace luce_am
