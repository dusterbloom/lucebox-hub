// AM fit solver tests (KVFlash 2.0b M5): CPU and CUDA/HIP against plain C++ fp64.
//   1. noise-free problems with known w and known C2 are recovered to 1e-6;
//   2. an ill-conditioned M engages the ridge fallback and stays finite;
//   3. the backend Gram accumulation matches a plain double reference;
//   4. end to end (duplicated keys): known w = multiplicity, known C2 = V_sel.
//
// usage: test_am_fit_solve [--cpu-only]
#include "common/am_fit_solve.h"

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

// G = M^T M, b = M^T y for row-major M [n][t].
void gram(const std::vector<double> & M, const std::vector<double> & Y, int n, int t, int d,
          std::vector<double> & G, std::vector<double> & B) {
    G.assign((size_t) t * t, 0.0);
    B.assign((size_t) t * d, 0.0);
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < t; ++i) {
            const double mi = M[(size_t) r * t + i];
            for (int j = 0; j < t; ++j) G[(size_t) i * t + j] += mi * M[(size_t) r * t + j];
            for (int k = 0; k < d; ++k) B[(size_t) i * d + k] += mi * Y[(size_t) r * d + k];
        }
    }
}

// ---- 1. host recovery ---------------------------------------------------------

void test_host_recovery(int n, int t, int d) {
    std::printf("[host recovery n=%d t=%d d=%d]\n", n, t, d);
    // beta: M = exp-scores, y = M w with w inside the box
    std::vector<double> M((size_t) n * t), w(t), y((size_t) n);
    for (auto & v : M) v = 0.05 + rnd01();
    for (auto & v : w) v = std::exp(-2.5 + 5.0 * rnd01());
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += M[(size_t) r * t + i] * w[i];
        y[r] = s;
    }
    std::vector<double> G, b, unused;
    gram(M, y, n, t, 1, G, b);
    std::vector<double> beta(t);
    const SolveStatus st = solve_beta(G.data(), b.data(), t, NnlsParams{}, beta.data());
    CHECK(st == SolveStatus::Ok, "status %d", (int) st);
    double worst = 0;
    for (int i = 0; i < t; ++i) worst = std::max(worst, std::fabs(beta[i] - std::log(w[i])));
    CHECK(worst < 1e-6, "beta error %.3e", worst);

    // C2: X = softmax rows, Y = X C2
    std::vector<double> X((size_t) n * t), C2((size_t) t * d), Y((size_t) n * d, 0.0);
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += (X[(size_t) r * t + i] = std::exp(rnd_norm()));
        for (int i = 0; i < t; ++i) X[(size_t) r * t + i] /= s;
    }
    for (auto & v : C2) v = rnd_norm();
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < t; ++i)
            for (int k = 0; k < d; ++k) Y[(size_t) r * d + k] += X[(size_t) r * t + i] * C2[(size_t) i * d + k];
    std::vector<double> AtA, AtY, C2fit((size_t) t * d);
    gram(X, Y, n, t, d, AtA, AtY);
    const SolveStatus sv = solve_values(AtA.data(), AtY.data(), t, d, NnlsParams{}, C2fit.data());
    CHECK(sv == SolveStatus::Ok, "status %d", (int) sv);
    worst = 0;
    for (size_t i = 0; i < C2.size(); ++i) worst = std::max(worst, std::fabs(C2fit[i] - C2[i]));
    CHECK(worst < 1e-6, "C2 error %.3e", worst);
}

// A solution outside the box is clamped, never extrapolated.
void test_host_box() {
    std::printf("[host box clamp]\n");
    const int n = 100, t = 8;
    std::vector<double> M((size_t) n * t), w(t), y((size_t) n);
    for (auto & v : M) v = 0.05 + rnd01();
    for (int i = 0; i < t; ++i) w[i] = (i % 2) ? 100.0 : 1e-4;  // far outside [e^-3, e^3]
    for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int i = 0; i < t; ++i) s += M[(size_t) r * t + i] * w[i];
        y[r] = s;
    }
    std::vector<double> G, b, beta(t);
    gram(M, y, n, t, 1, G, b);
    solve_beta(G.data(), b.data(), t, NnlsParams{}, beta.data());
    for (int i = 0; i < t; ++i) CHECK(beta[i] >= -3.0 - 1e-12 && beta[i] <= 3.0 + 1e-12, "beta[%d]=%g", i, beta[i]);
}

// ---- 2. ill-conditioned -> ridge, finite -----------------------------------------

void test_host_ridge() {
    std::printf("[host ridge fallback]\n");
    const int n = 200, t = 12, d = 5;
    std::vector<double> M((size_t) n * t), y((size_t) n), X((size_t) n * t), Y((size_t) n * d);
    for (int r = 0; r < n; ++r) {
        for (int i = 0; i < t; ++i) M[(size_t) r * t + i] = 0.05 + rnd01();
        M[(size_t) r * t + 5] = M[(size_t) r * t + 2];                // exactly collinear
        M[(size_t) r * t + 9] = 2.0 * M[(size_t) r * t + 2];          // and a multiple
        y[r] = 1.0 + rnd01();
        for (int i = 0; i < t; ++i) X[(size_t) r * t + i] = M[(size_t) r * t + i] / 10.0;
        for (int k = 0; k < d; ++k) Y[(size_t) r * d + k] = rnd_norm();
    }
    std::vector<double> G, b, beta(t);
    gram(M, y, n, t, 1, G, b);
    const SolveStatus st = solve_beta(G.data(), b.data(), t, NnlsParams{}, beta.data());
    CHECK(st == SolveStatus::RidgeUsed, "beta status %d", (int) st);
    for (int i = 0; i < t; ++i) CHECK(std::isfinite(beta[i]) && std::fabs(beta[i]) <= 3.0 + 1e-12, "beta[%d]=%g", i, beta[i]);

    std::vector<double> AtA, AtY, C2((size_t) t * d);
    gram(X, Y, n, t, d, AtA, AtY);
    const SolveStatus sv = solve_values(AtA.data(), AtY.data(), t, d, NnlsParams{}, C2.data());
    CHECK(sv == SolveStatus::RidgeUsed, "values status %d", (int) sv);
    for (double v : C2) CHECK(std::isfinite(v), "C2 not finite");

    // Garbage in -> Failed, neutral out (beta 0), still finite.
    std::vector<double> Gbad(G);
    Gbad[0] = std::nan("");
    const SolveStatus sf = solve_beta(Gbad.data(), b.data(), t, NnlsParams{}, beta.data());
    CHECK(sf == SolveStatus::Failed, "nan status %d", (int) sf);
    for (int i = 0; i < t; ++i) CHECK(beta[i] == 0.0, "neutral beta[%d]=%g", i, beta[i]);
}

// ---- 3. backend Gram accumulation vs a plain double reference ----------------------

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

struct RefGrams { std::vector<double> MtM, Mty, AtA, AtY; };

// Plain double reference of everything the backend accumulates.
RefGrams ref_grams(const Problem & p, const float * beta /* [t, n_kv] or null for beta stage only */) {
    const BlockShape & s = p.shape;
    const int d = s.head_dim, T = s.n_keys, t = s.t, n_rep = s.n_head / s.n_head_kv;
    RefGrams r;
    r.MtM.assign((size_t) s.n_head_kv * t * t, 0.0);
    r.Mty.assign((size_t) s.n_head_kv * t, 0.0);
    r.AtA.assign((size_t) s.n_head_kv * t * t, 0.0);
    r.AtY.assign((size_t) s.n_head_kv * t * d, 0.0);
    std::vector<double> sc(T), e(T), qv(d), x(t), yv(d);
    for (int g = 0; g < s.n_head_kv; ++g) {
        const float * K = p.k.data() + (size_t) g * T * d;
        const float * V = p.v.data() + (size_t) g * T * d;
        for (int h = g * n_rep; h < (g + 1) * n_rep; ++h) {
            for (int tok = 0; tok < s.n_tokens; ++tok) {
                const ggml_fp16_t * qp = p.q.data() + ((size_t) h * s.n_tokens + tok) * d;
                for (int c = 0; c < d; ++c) qv[c] = ggml_fp16_to_fp32(qp[c]);
                double mx = -1e300;
                for (int j = 0; j < T; ++j) {
                    double a = 0;
                    for (int c = 0; c < d; ++c) a += qv[c] * K[(size_t) j * d + c];
                    sc[j] = a * (double) s.scale;
                    mx = std::max(mx, sc[j]);
                }
                double ysum = 0;
                for (int j = 0; j < T; ++j) ysum += (e[j] = std::exp(sc[j] - mx));
                for (int i = 0; i < t; ++i) x[i] = e[p.sel[(size_t) g * t + i]];
                for (int i = 0; i < t; ++i) {
                    r.Mty[(size_t) g * t + i] += x[i] * ysum;
                    for (int j = 0; j < t; ++j) r.MtM[((size_t) g * t + i) * t + j] += x[i] * x[j];
                }
                if (!beta) continue;
                // value stage: A = softmax over sel with beta, Y = softmax over all keys @ V
                double lm = -1e300;
                for (int i = 0; i < t; ++i) { x[i] = sc[p.sel[(size_t) g * t + i]] + beta[(size_t) g * t + i]; lm = std::max(lm, x[i]); }
                double ls = 0;
                for (int i = 0; i < t; ++i) ls += (x[i] = std::exp(x[i] - lm));
                for (int i = 0; i < t; ++i) x[i] /= ls;
                std::fill(yv.begin(), yv.end(), 0.0);
                for (int j = 0; j < T; ++j)
                    for (int c = 0; c < d; ++c) yv[c] += e[j] / ysum * V[(size_t) j * d + c];
                for (int i = 0; i < t; ++i) {
                    for (int j = 0; j < t; ++j) r.AtA[((size_t) g * t + i) * t + j] += x[i] * x[j];
                    for (int c = 0; c < d; ++c) r.AtY[((size_t) g * t + i) * d + c] += x[i] * yv[c];
                }
            }
        }
    }
    return r;
}

double max_rel(const std::vector<double> & a, const std::vector<double> & ref) {
    double scale = 0, err = 0;
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::fabs(a[i] - ref[i]));
    return err / std::max(scale, 1e-30);
}

void test_gram(ggml_backend_t be, const char * name, const char * label, int d, int n_tokens, int n_head, int n_kv,
               int T, int t, int chunk, int compensated = -1, bool info_only = false) {
    std::printf("[gram %s %s d%d tok%d h%d/%d T%d t%d chunk%d]\n", name, label, d, n_tokens, n_head, n_kv, T, t, chunk);
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, chunk);
    p.shape.compensated = compensated;
    std::vector<float> beta((size_t) t * n_kv);
    for (auto & b : beta) b = (float) (-1.0 + 2.0 * rnd01());
    const RefGrams ref = ref_grams(p, beta.data());

    GramAccumulator acc;
    std::string err;
    if (!acc.init(be, p.shape, p.in(), &err)) { CHECK(false, "init: %s", err.c_str()); return; }
    BetaGrams bg;
    ValueGrams vg;
    CHECK(acc.beta_grams(bg, &err), "beta_grams: %s", err.c_str());
    CHECK(acc.value_grams(beta.data(), vg, &err), "value_grams: %s", err.c_str());
    if (bg.MtM.size() != ref.MtM.size() || vg.AtY.size() != ref.AtY.size()) { CHECK(false, "output size"); return; }
    const double e1 = max_rel(bg.MtM, ref.MtM), e2 = max_rel(bg.Mty, ref.Mty);
    const double e3 = max_rel(vg.AtA, ref.AtA), e4 = max_rel(vg.AtY, ref.AtY);
    std::printf("  max rel err: MtM %.2e Mty %.2e AtA %.2e AtY %.2e\n", e1, e2, e3, e4);
    if (!info_only) CHECK(e1 < 2e-5 && e2 < 2e-5 && e3 < 2e-5 && e4 < 2e-5, "gram mismatch");
}

// ---- 4. end to end on duplicated keys ---------------------------------------------

// Selected key i appears c_i = 1 + i % 7 times in the block (same K and same V), so the full
// softmax over T keys equals the softmax over the selected keys with beta = log c_i and
// C2 = V_sel: the fit must recover exactly that.
void test_dup(ggml_backend_t be, const char * name) {
    std::printf("[dup recovery %s]\n", name);
    const int d = 64, n_tokens = 150, n_head = 6, n_kv = 2, t = 16;
    int T = 0;
    for (int i = 0; i < t; ++i) T += 1 + i % 7;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 64);
    std::vector<int> cnt(t);
    for (int g = 0; g < n_kv; ++g) {
        std::vector<int> perm(T);  // slot i owns c_i positions
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
    FitResult res;
    std::string err;
    if (!fit_block(be, p.shape, p.in(), NnlsParams{}, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
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

// fit_block == reference Grams through the same host solvers.
void test_fit_vs_ref(ggml_backend_t be, const char * name) {
    std::printf("[fit vs fp64 reference %s]\n", name);
    const int d = 32, n_tokens = 90, n_head = 4, n_kv = 2, T = 80, t = 12;
    Problem p = make_problem(d, n_tokens, n_head, n_kv, T, t, 40);
    FitResult res;
    std::string err;
    if (!fit_block(be, p.shape, p.in(), NnlsParams{}, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
    const RefGrams r0 = ref_grams(p, nullptr);
    double eb = 0, ec = 0;
    std::vector<float> beta_ref((size_t) t * n_kv);
    for (int g = 0; g < n_kv; ++g) {
        std::vector<double> b(t);
        solve_beta(&r0.MtM[(size_t) g * t * t], &r0.Mty[(size_t) g * t], t, NnlsParams{}, b.data());
        for (int i = 0; i < t; ++i) beta_ref[(size_t) g * t + i] = (float) b[i];
    }
    const RefGrams r1 = ref_grams(p, beta_ref.data());
    for (int g = 0; g < n_kv; ++g) {
        std::vector<double> c2((size_t) t * d);
        solve_values(&r1.AtA[(size_t) g * t * t], &r1.AtY[(size_t) g * t * d], t, d, NnlsParams{}, c2.data());
        for (int i = 0; i < t; ++i) {
            eb = std::max(eb, (double) std::fabs(res.beta[(size_t) g * t + i] - beta_ref[(size_t) g * t + i]));
            for (int c = 0; c < d; ++c) ec = std::max(ec, std::fabs(res.c2[((size_t) g * t + i) * d + c] - c2[(size_t) i * d + c]));
        }
    }
    std::printf("  beta err %.2e  C2 err %.2e\n", eb, ec);
    CHECK(eb < 1e-4 && ec < 1e-4, "pipeline differs from reference");
}

// ---- 7. timing at the real shape (synthetic data) ----------------------------------

void time_real_shape(ggml_backend_t be, const char * name) {
    // One 640-token block, 16 full-attention layers, Qwen3.8-27B: d256, 24 heads, 4 KV heads, t = 64.
    Problem p = make_problem(256, 640, 24, 4, 640, 64, BlockShape{}.chunk_tokens);
    FitResult res;
    std::string err;
    GramAccumulator acc;  // one per device in the engine; graphs and buffers persist across layers
    double best = 1e30, total = 0, ph[5] = {0, 0, 0, 0, 0};
    const int reps = 5;
    for (int rep = 0; rep < reps + 1; ++rep) {  // first run is warm-up
        const auto t0 = std::chrono::steady_clock::now();
        for (int layer = 0; layer < 16; ++layer) {
            if (!fit_block(acc, be, p.shape, p.in(), NnlsParams{}, res, &err)) { CHECK(false, "fit_block: %s", err.c_str()); return; }
            if (rep > 0) {
                ph[0] += res.ms_init; ph[1] += res.ms_beta_grams; ph[2] += res.ms_beta_solve;
                ph[3] += res.ms_value_grams; ph[4] += res.ms_value_solve;
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rep > 0) { best = std::min(best, ms); total += ms; }
    }
    std::printf("TIMING %s: one 640-token block, 16 layers x 4 KV heads, t=64: best %.1f ms, mean %.1f ms (%d reps)\n",
                name, best, total / reps, reps);
    std::printf("  phases per block (ms, mean of %d): init %.1f, beta Grams %.1f, beta solve %.1f, value Grams %.1f, value solve %.1f\n",
                reps, ph[0] / reps, ph[1] / reps, ph[2] / reps, ph[3] / reps, ph[4] / reps);
}

}  // namespace

int main(int argc, char ** argv) {
    bool cpu_only = false, time_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu-only")) cpu_only = true;
        if (!std::strcmp(argv[i], "--time")) time_only = true;
    }

    if (!time_only) {
        test_host_recovery(300, 24, 8);
        test_host_recovery(3840, 64, 16);
        test_host_box();
        test_host_ridge();
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
        if (time_only) { time_real_shape(b.be, b.name); continue; }
        test_gram(b.be, b.name, "small", 32, 70, 6, 2, 80, 8, 32);       // remainder chunk
        test_gram(b.be, b.name, "small-comp", 32, 70, 6, 2, 80, 8, 32, 1);  // split matmuls forced on
        test_gram(b.be, b.name, "mha", 16, 33, 3, 3, 40, 6, 33);         // n_rep = 1, one chunk
        test_gram(b.be, b.name, "real", 256, 640, 24, 4, 640, 64, 64);  // Qwen3.8-27B block, default chunk
        // Informational: what the GPU gives without compensation, and with a deep (256-token) chunk.
        test_gram(b.be, b.name, "real-plain", 256, 640, 24, 4, 640, 64, 64, 0, true);
        test_gram(b.be, b.name, "real-chunk256", 256, 640, 24, 4, 640, 64, 256, -1, true);
        test_dup(b.be, b.name);
        test_fit_vs_ref(b.be, b.name);
    }
    for (const Be & b : backends) ggml_backend_free(b.be);
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
