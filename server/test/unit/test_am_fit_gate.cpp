// KVFlash 2.0b M4 R3 gate: am_fit scoring and selection against the reference fixtures dumped
// from the compaction repo's own fit (generator: optimizations/kvflash/lucebox/a1/am_fixture_dump.py).
//
// Always runs the self tests (fixture reader, comparison rules, the verdict, and the whole gate
// pipeline on a synthetic fixture directory it writes itself). The real gate runs only when
// LUCE_AM_FIXTURES names a directory of case directories (traj1_first, traj1_12k, article_r1e).
//
// What the fixtures are. The reference fit does not see the block's captured queries `q`; it
// receives the grouped queries `fitq` [n_kv, n_fit, head_dim] (n_fit = 3839 at the real shape: one
// query is dropped at random), and it computes its logits in bf16. Its dumped scores are therefore
// ~4e-3 (normalized) away from exact math on the same inputs: that is the reference's own floor,
// not an engine error. The gate reads `fitq` (a case without it fails; there is no fallback to `q`)
// and scores it with am_fit as n_kv "heads" of n_fit queries each (G = 1), converting our sum w^2
// to the reference's rms: sqrt(sum / n_fit).
//
// Three bars, fixed constants (no override), per case, on the q16 arm (Q rounded to F16 as the
// engine feeds it; the q32 arm is reported as a diagnostic):
//   1. Math: engine rms scores vs a float64 host computation on the same fitq and k: per head,
//      max |diff| / head max <= kMathTol (1e-3).
//   2. Selection vs exact math: the engine's top-k equals the float64 top-k except for swaps whose
//      float64 score is within kSelTol (1e-3) x head max of the float64 k-th score.
//   3. Agreement with the reference selection: overlap across all heads >= kRefOverlap (99%), at
//      most kRefMaxSwaps (2 of 64, i.e. >= 62/64) swapped keys in any head, and every swapped key
//      (either direction) has a float64 score within kRefGapTol (2e-2) x head max of the float64
//      score of the reference's last-ranked (k-th) key. 2e-2 is about twice the measured 1.06e-2
//      floor of the reference's bf16 logits.
// The reference-score error (engine vs dumped scores, normalized by head max) is reported, not
// bound. At least one case with scores must run; dimensions, indices and values are validated.
//
// usage: test_am_fit_gate [--cpu-only]
#include "am_fixture.h"
#include "common/am_fit.h"

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
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

int g_fail = 0;
void check(bool ok, const std::string & what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

constexpr double kMathTol     = 1e-3;
constexpr double kSelTol      = 1e-3;
constexpr double kRefOverlap  = 0.99;
constexpr int    kRefMaxSwaps = 2;
constexpr double kRefGapTol   = 2e-2;

// ---------------------------------------------------------------- set comparison (pure)

struct SetCmp {
    bool   valid   = true;   // false: bad indices / non-finite scores / bad tolerance
    int    swapped = 0;      // keys in `ours` that are not in `ref`
    double worst   = 0.0;    // max |score - kth| / norm over keys in the symmetric difference
    bool   within  = true;   // every key in the symmetric difference has gap <= tol
};

// `scores`: per-key scores used for the gap (float64 host scores). kth: the score the gap is measured
// from. norm: the scale (head max). tol: the bound on gap/norm. NaN anywhere never passes.
SetCmp compare_sets(const int32_t * ours, const int32_t * ref, int k, int n, const double * scores, double kth, double norm, double tol) {
    SetCmp r;
    if (!(tol >= 0.0) || !std::isfinite(tol) || !(norm > 0.0) || !std::isfinite(norm) || !std::isfinite(kth) || k < 1 || n < 1) {
        r.valid = r.within = false;
        return r;
    }
    for (int i = 0; i < k; ++i)
        if (ref[i] < 0 || ref[i] >= n || ours[i] < 0 || ours[i] >= n || !std::isfinite(scores[ref[i]]) || !std::isfinite(scores[ours[i]])) {
            r.valid = r.within = false;
            return r;
        }
    auto in = [k](const int32_t * set, int32_t key) { return std::find(set, set + k, key) != set + k; };
    auto note = [&](double gap) {
        if (std::isfinite(gap)) r.worst = std::max(r.worst, gap);
        if (!(gap <= tol)) r.within = false;
    };
    for (int i = 0; i < k; ++i) {
        if (!in(ref, ours[i])) { ++r.swapped; note(std::fabs(scores[ours[i]] - kth) / norm); }
        if (!in(ours, ref[i])) note(std::fabs(scores[ref[i]] - kth) / norm);
    }
    return r;
}

void test_compare_sets() {
    const double sc[] = { 10, 9, 8, 7, 6.99, 6.0, 1 };  // top-4: 0 1 2 3; 4th score 7; head max 10
    const int32_t ref[] = { 0, 1, 2, 3 };
    {
        const int32_t o[] = { 3, 2, 1, 0 };
        const SetCmp c = compare_sets(o, ref, 4, 7, sc, 7.0, 10.0, 1e-3);
        check(c.valid && c.within && c.swapped == 0 && c.worst == 0.0, "compare_sets: same set in another order, 0 swaps");
    }
    {
        const int32_t o[] = { 0, 1, 2, 4 };  // |6.99 - 7| / 10 = 1e-3
        const SetCmp c = compare_sets(o, ref, 4, 7, sc, 7.0, 10.0, 1.0001e-3);
        check(c.valid && c.within && c.swapped == 1 && std::fabs(c.worst - 1e-3) < 1e-9, "compare_sets: a swap at gap 1e-3 passes a 1.0001e-3 bound");
        check(!compare_sets(o, ref, 4, 7, sc, 7.0, 10.0, 0.9e-3).within, "compare_sets: the same swap fails a 0.9e-3 bound");
    }
    {
        const int32_t o[] = { 0, 1, 2, 5 };  // 6.0: gap 0.1
        const SetCmp c = compare_sets(o, ref, 4, 7, sc, 7.0, 10.0, 2e-2);
        check(c.valid && !c.within && c.swapped == 1 && c.worst > 0.09, "compare_sets: a swap far from the cutoff fails");
    }
    {
        const double flat[] = { 5, 5, 5, 5, 5 };
        const int32_t r2[] = { 0, 1 }, o2[] = { 3, 4 };
        const SetCmp c = compare_sets(o2, r2, 2, 5, flat, 5.0, 5.0, 1e-3);
        check(c.valid && c.within && c.swapped == 2 && c.worst == 0.0, "compare_sets: exact ties swap freely");
    }
    {
        const int32_t o[] = { 0, 1, 2, 4 };
        check(!compare_sets(o, ref, 4, 7, sc, 7.0, 10.0, std::nan("")).within, "compare_sets: a NaN tolerance never accepts a swap");
        check(!compare_sets(o, ref, 4, 7, sc, std::nan(""), 10.0, 1e-3).within, "compare_sets: a NaN cutoff never accepts a swap");
        check(!compare_sets(o, ref, 4, 7, sc, 7.0, std::nan(""), 1e-3).within, "compare_sets: a NaN norm never accepts a swap");
        const double nan_sc[] = { 10, 9, 8, 7, std::nan(""), 6, 1 };
        check(!compare_sets(o, ref, 4, 7, nan_sc, 7.0, 10.0, 1e-3).within, "compare_sets: a NaN score never accepts a swap");
        const int32_t oor[] = { 0, 1, 2, 9 }, neg[] = { 0, 1, 2, -1 };
        check(!compare_sets(oor, ref, 4, 7, sc, 7.0, 10.0, 1e-3).within && !compare_sets(neg, ref, 4, 7, sc, 7.0, 10.0, 1e-3).within,
              "compare_sets: out-of-range and negative keys are not accepted");
    }
}

// ---------------------------------------------------------------- statistics and verdict (pure)

struct Dist { double med = 0, p90 = 0, max = 0; };
Dist dist(std::vector<double> v) {
    Dist d;
    if (v.empty()) return d;
    std::sort(v.begin(), v.end());
    d.med = v[v.size() / 2];
    d.p90 = v[std::min(v.size() - 1, (size_t) std::ceil(0.9 * v.size()) - 1)];
    d.max = v.back();
    return d;
}

struct ArmStats {
    std::vector<double> math_err;      // check 1: per head, max |engine - f64| / head max
    std::vector<double> sel_gap;       // check 2: per head, worst swap gap vs the f64 top-k (norm: head max)
    std::vector<char>   sel_valid;     //          per head, comparison was valid
    std::vector<double> ref_gap;       // check 3: per head, worst swap gap vs the reference selection
    std::vector<char>   ref_valid;
    std::vector<int>    ref_swaps;     // check 3: per head, engine keys not in the reference's top-k
    std::vector<double> ref_score_err; // reported only: per head, max |engine - dumped| / head max
    int                 slots = 0;
};

// Empty when the arm passes the gate, else the reason.
std::string gate_verdict(const ArmStats & s) {
    const size_t n = s.math_err.size();
    if (n == 0 || s.slots <= 0 || s.sel_gap.size() != n || s.ref_gap.size() != n || s.ref_swaps.size() != n || s.sel_valid.size() != n || s.ref_valid.size() != n)
        return "no (or inconsistent) per-head results";
    for (size_t h = 0; h < n; ++h) {
        if (!(s.math_err[h] <= kMathTol)) return "head " + std::to_string(h) + ": math error " + std::to_string(s.math_err[h]) + " above " + std::to_string(kMathTol);
        if (!s.sel_valid[h] || !(s.sel_gap[h] <= kSelTol)) return "head " + std::to_string(h) + ": selection differs from exact math beyond tie tolerance";
        if (!s.ref_valid[h]) return "head " + std::to_string(h) + ": invalid comparison with the reference selection";
        if (s.ref_swaps[h] > kRefMaxSwaps) return "head " + std::to_string(h) + ": " + std::to_string(s.ref_swaps[h]) + " swaps vs the reference, bound " + std::to_string(kRefMaxSwaps);
        if (!(s.ref_gap[h] <= kRefGapTol)) return "head " + std::to_string(h) + ": a swapped key is " + std::to_string(s.ref_gap[h]) + " (x head max) from the reference cutoff, bound " + std::to_string(kRefGapTol);
    }
    long swapped = 0;
    for (int x : s.ref_swaps) swapped += x;
    const double overlap = 1.0 - (double) swapped / ((double) n * s.slots);
    if (overlap < kRefOverlap) return "overlap with the reference selection " + std::to_string(overlap) + " below " + std::to_string(kRefOverlap);
    return "";
}

// ---------------------------------------------------------------- scoring through am_fit and float64 on the host

struct Backend { const char * name; ggml_backend_t be; };

ggml_fp16_t to_f16(float x) { return ggml_fp32_to_fp16(x); }

// Scores fitq [d, nq, n_kv] (as n_kv heads, G = 1) against k [d, T, n_kv] (F32) through the tensor
// entry point. Returns our per-key rms scores [n_kv][T] = sqrt(sum w^2 / nq).
bool our_rms_scores(const Backend & b, const std::vector<float> & q, const std::vector<float> & k, int d, int nq, int T, int n_kv,
                    bool q_f16, std::vector<float> & rms, std::string * err) {
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(ip), ggml_free);
    ggml_tensor * qt = ggml_new_tensor_3d(ctx.get(), q_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, d, nq, n_kv);
    ggml_tensor * kt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, T, n_kv);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf(ggml_backend_alloc_ctx_tensors(ctx.get(), b.be), ggml_backend_buffer_free);
    if (!buf) { if (err) *err = "buffer allocation failed"; return false; }
    if (q_f16) {
        std::vector<ggml_fp16_t> h(q.size());
        for (size_t i = 0; i < q.size(); ++i) h[i] = to_f16(q[i]);
        ggml_backend_tensor_set(qt, h.data(), 0, ggml_nbytes(qt));
    } else {
        ggml_backend_tensor_set(qt, q.data(), 0, ggml_nbytes(qt));
    }
    ggml_backend_tensor_set(kt, k.data(), 0, ggml_nbytes(kt));
    if (!luce::am::score_keys(b.be, qt, kt, 8192, rms, err)) return false;
    const double inv_rows = 1.0 / (double) nq;
    for (float & s : rms) s = (float) std::sqrt(s * inv_rows);
    return true;
}

// Float64 reference of the repo's rms score for one KV head: q [nq, d], k [T, d] (float inputs,
// double math), multithreaded over queries. Returns sqrt(mean_i w_i^2) per key.
std::vector<double> f64_rms_head(const float * q, const float * k, int d, int nq, int T) {
    std::vector<double> kd((size_t) T * d);
    for (size_t i = 0; i < kd.size(); ++i) kd[i] = k[i];
    const int nt = (int) std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::vector<double>> acc(nt, std::vector<double>(T, 0.0));
    const double inv = 1.0 / std::sqrt((double) d);
    auto work = [&](int t) {
        std::vector<double> s(T), qd(d);
        for (int i = t; i < nq; i += nt) {
            for (int c = 0; c < d; ++c) qd[c] = q[(size_t) i * d + c];
            double mx = -1e300;
            for (int j = 0; j < T; ++j) {
                double a = 0;
                const double * kr = &kd[(size_t) j * d];
                for (int c = 0; c < d; ++c) a += qd[c] * kr[c];
                s[j] = a * inv;
                mx = std::max(mx, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < T; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            for (int j = 0; j < T; ++j) { const double w = s[j] / sum; acc[t][j] += w * w; }
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) th.emplace_back(work, t);
    for (auto & x : th) x.join();
    std::vector<double> out(T, 0.0);
    for (int j = 0; j < T; ++j) {
        double a = 0;
        for (int t = 0; t < nt; ++t) a += acc[t][j];
        out[j] = std::sqrt(a / nq);
    }
    return out;
}

// top-k of double scores, higher first, ties by lower index.
std::vector<int32_t> f64_topk(const std::vector<double> & sc, int k) {
    std::vector<int32_t> idx(sc.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[a] > sc[b]; });
    idx.resize(k);
    return idx;
}

long file_size(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 ? (long) st.st_size : -1;
}

// ---------------------------------------------------------------- the gate over a fixture root

// Returns false only on a fixture / run error. Fills per-arm stats (index 0: q16, 1: q32).
bool gate_case(const Backend & b, const am_fixture::Case & c, ArmStats st[2], std::string * err) {
    auto fail = [&](const std::string & m) { if (err) *err = c.name + ": " + m; return false; };
    std::string manifest;
    am_fixture::read_file(c.dir + "/manifest.json", manifest);
    int64_t d = 0, n = 0, nh = 0, nkv = 0, slots = 0;
    if (!am_fixture::manifest_int(manifest, "head_dim", d) || !am_fixture::manifest_int(manifest, "n_tokens", n) ||
        !am_fixture::manifest_int(manifest, "n_head", nh) || !am_fixture::manifest_int(manifest, "n_kv", nkv) ||
        !am_fixture::manifest_int(manifest, "slots", slots))
        return fail("manifest lacks head_dim/n_tokens/n_head/n_kv/slots");
    // Bound every dimension before it sizes an allocation or a subscript.
    constexpr int64_t kMaxDim = 1 << 16;
    if (d < 1 || d > kMaxDim || n < 1 || n > kMaxDim || nh < 1 || nh > 1024 || nkv < 1 || nkv > nh || nh % nkv != 0 || slots < 1 || slots > n)
        return fail("implausible dimensions head_dim=" + std::to_string(d) + " n_tokens=" + std::to_string(n) + " n_head=" + std::to_string(nh) +
                    " n_kv=" + std::to_string(nkv) + " slots=" + std::to_string(slots));
    const size_t sm = manifest.find("\"score_method\"");
    if (sm != std::string::npos && manifest.find("rms", sm) > sm + 48)
        return fail("the fixture's score_method is not rms, scores are not comparable");
    for (ArmStats * s = st; s < st + 2; ++s) *s = ArmStats{};
    st[0].slots = st[1].slots = (int) slots;
    for (int layer : c.layers) {
        // fitq is mandatory: the queries the reference fit received (n_fit per KV head).
        const long fq_bytes = file_size(am_fixture::layer_path(c, layer, "fitq", "f32"));
        if (fq_bytes < 0) return fail("layer " + std::to_string(layer) + " has no fitq (the queries the reference fit saw); not falling back to q");
        const long row = (long) (nkv * d * sizeof(float));
        if (fq_bytes == 0 || fq_bytes % row != 0) return fail("fitq size " + std::to_string(fq_bytes) + " is not a multiple of n_kv*head_dim*4");
        const int64_t nq = fq_bytes / row;
        if (nq < 1 || nq > (int64_t) 1 << 17) return fail("implausible fitq query count " + std::to_string(nq));
        std::vector<float> fq, k, ref;
        std::vector<int32_t> idx;
        if (!am_fixture::read_f32(c, layer, "fitq", nkv * nq * d, fq, err) || !am_fixture::read_f32(c, layer, "k", d * n * nkv, k, err) ||
            !am_fixture::read_f32(c, layer, "scores", n * nkv, ref, err) || !am_fixture::read_i32(c, layer, "idx", slots * nkv, idx, err))
            return false;
        for (float x : fq) if (!std::isfinite(x)) return fail("non-finite value in fitq");
        for (float x : k) if (!std::isfinite(x)) return fail("non-finite value in k");
        for (float x : ref) if (!std::isfinite(x) || x < 0.0f) return fail("non-finite or negative reference score");
        for (int h = 0; h < nkv; ++h) {
            std::vector<char> seen(n, 0);
            for (int r = 0; r < slots; ++r) {
                const int32_t key = idx[(size_t) h * slots + r];
                if (key < 0 || key >= n) return fail("reference index " + std::to_string(key) + " out of range [0, " + std::to_string(n) + ")");
                if (seen[key]) return fail("reference indices are not unique (head " + std::to_string(h) + ")");
                seen[key] = 1;
            }
        }
        // float64 scores, once per head (shared by both arms)
        std::vector<std::vector<double>> f64(nkv);
        for (int h = 0; h < nkv; ++h) f64[h] = f64_rms_head(&fq[(size_t) h * nq * d], &k[(size_t) h * n * d], (int) d, (int) nq, (int) n);
        for (int arm = 0; arm < 2; ++arm) {
            std::vector<float> rms;
            if (!our_rms_scores(b, fq, k, (int) d, (int) nq, (int) n, (int) nkv, arm == 0, rms, err)) return false;
            for (float x : rms) if (!std::isfinite(x)) return fail("non-finite score from am_fit");
            ArmStats & s = st[arm];
            for (int h = 0; h < nkv; ++h) {
                const std::vector<double> & ex = f64[h];
                const double hmax = *std::max_element(ex.begin(), ex.end());
                if (!(hmax > 0.0) || !std::isfinite(hmax)) return fail("float64 scores are degenerate");
                double me = 0, re = 0;
                for (int j = 0; j < n; ++j) {
                    me = std::max(me, std::fabs((double) rms[(size_t) h * n + j] - ex[j]) / hmax);
                    re = std::max(re, std::fabs((double) rms[(size_t) h * n + j] - ref[(size_t) h * n + j]) / hmax);
                }
                std::vector<int32_t> ours(slots);
                luce::am::select_top_k(&rms[(size_t) h * n], (int) n, (int) slots, ours.data());
                const std::vector<int32_t> exact = f64_topk(ex, (int) slots);
                const SetCmp c2 = compare_sets(ours.data(), exact.data(), (int) slots, (int) n, ex.data(), ex[exact[slots - 1]], hmax, kSelTol);
                const int32_t * rk = &idx[(size_t) h * slots];
                const SetCmp c3 = compare_sets(ours.data(), rk, (int) slots, (int) n, ex.data(), ex[rk[slots - 1]], hmax, kRefGapTol);
                s.math_err.push_back(me);
                s.sel_gap.push_back(c2.worst);
                s.sel_valid.push_back(c2.valid && c2.within);
                s.ref_gap.push_back(c3.worst);
                s.ref_valid.push_back(c3.valid);
                s.ref_swaps.push_back(c3.swapped);
                s.ref_score_err.push_back(re);
            }
        }
    }
    return true;
}

std::string fmt_dist(const char * name, const std::vector<double> & v) {
    const Dist d = dist(v);
    char buf[128];
    std::snprintf(buf, sizeof buf, " %s med/p90/max=%.3g/%.3g/%.3g", name, d.med, d.p90, d.max);
    return buf;
}

// Runs every case under `root`; true when the gate passes. Prints GATE lines per case and arm.
bool gate_root(const Backend & b, const std::string & root, int * ran_out = nullptr) {
    const std::vector<std::string> dirs = am_fixture::case_dirs(root.c_str());
    bool pass = true;
    int ran = 0;
    for (const std::string & dir : dirs) {
        am_fixture::Case c;
        std::string err;
        if (!am_fixture::load_case(dir, c, &err)) { std::printf("GATEFAIL %s %s: %s\n", b.name, dir.c_str(), err.c_str()); pass = false; continue; }
        if (!std::ifstream(am_fixture::layer_path(c, c.layers[0], "scores", "f32"))) {
            std::printf("SKIP  %s %s: no scores/idx (queries-only case)\n", b.name, c.name.c_str());
            continue;
        }
        ArmStats st[2];
        if (!gate_case(b, c, st, &err)) { std::printf("GATEFAIL %s %s: %s\n", b.name, c.name.c_str(), err.c_str()); pass = false; continue; }
        ++ran;
        static const char * arm_name[2] = { "q16", "q32" };
        for (int a = 0; a < 2; ++a) {
            const ArmStats & s = st[a];
            long swapped = 0;
            std::vector<double> swaps;
            for (int x : s.ref_swaps) { swapped += x; swaps.push_back(x); }
            const double overlap = 1.0 - (double) swapped / ((double) s.ref_swaps.size() * s.slots);
            std::printf("GATE  %s %-14s %s heads=%zu |%s |%s |%s |%s | overlap=%.4f min_overlap=%d/%d |%s  [reported, unbound]\n", b.name, c.name.c_str(),
                        arm_name[a], s.math_err.size(), fmt_dist("1:math_err", s.math_err).c_str(), fmt_dist("2:sel_gap", s.sel_gap).c_str(),
                        fmt_dist("3:ref_swaps", swaps).c_str(), fmt_dist("3:ref_gap", s.ref_gap).c_str(), overlap,
                        s.slots - (int) *std::max_element(s.ref_swaps.begin(), s.ref_swaps.end()), s.slots, fmt_dist("ref_score_err", s.ref_score_err).c_str());
        }
        const std::string why = gate_verdict(st[0]);
        if (!why.empty()) { std::printf("GATEFAIL %s %s q16 arm: %s\n", b.name, c.name.c_str(), why.c_str()); pass = false; }
    }
    if (ran_out) *ran_out = ran;
    if (ran == 0) { std::printf("GATEFAIL %s: no case with scores ran under %s\n", b.name, root.c_str()); pass = false; }
    return pass;
}

// ---------------------------------------------------------------- self tests: verdict, reader, pipeline

ArmStats good_stats(int heads = 4, int slots = 64) {
    ArmStats s;
    s.slots = slots;
    for (int h = 0; h < heads; ++h) {
        s.math_err.push_back(1e-5); s.sel_gap.push_back(0); s.sel_valid.push_back(1);
        s.ref_gap.push_back(5e-3); s.ref_valid.push_back(1); s.ref_swaps.push_back(0); s.ref_score_err.push_back(4e-3);
    }
    return s;
}

void test_verdict() {
    check(gate_verdict(good_stats()).empty(), "verdict: clean stats pass");
    { ArmStats s = good_stats(); s.math_err[2] = 2e-3; check(!gate_verdict(s).empty(), "verdict: math error 2e-3 in one head fails (bar 1e-3)"); }
    { ArmStats s = good_stats(); s.math_err[2] = std::nan(""); check(!gate_verdict(s).empty(), "verdict: a NaN math error fails"); }
    { ArmStats s = good_stats(); s.sel_gap[1] = 5e-3; s.sel_valid[1] = 0; check(!gate_verdict(s).empty(), "verdict: selection differing from exact math beyond tie tolerance fails"); }
    { ArmStats s = good_stats(); s.ref_swaps[0] = 3; check(!gate_verdict(s).empty(), "verdict: 3 swaps in one head vs the reference fails (needs >= 62/64)"); }
    { ArmStats s = good_stats(); s.ref_swaps[0] = 2; check(gate_verdict(s).empty(), "verdict: 2 swaps in one head passes"); }
    { ArmStats s = good_stats(100); for (int i = 0; i < 50; ++i) s.ref_swaps[i] = 1; check(gate_verdict(s).empty(), "verdict: 50 of 6400 keys swapped (99.2% overlap) passes the overlap bar"); }
    { ArmStats s = good_stats(100); for (int i = 0; i < 100; ++i) s.ref_swaps[i] = 1; check(!gate_verdict(s).empty(), "verdict: 100 of 6400 keys swapped (98.4%) fails the overlap bar"); }
    { ArmStats s = good_stats(); s.ref_gap[3] = 3e-2; check(!gate_verdict(s).empty(), "verdict: a swapped key 3e-2 from the reference cutoff fails (bar 2e-2)"); }
    { ArmStats s = good_stats(); s.ref_gap[3] = 1.9e-2; check(gate_verdict(s).empty(), "verdict: a swapped key 1.9e-2 from the cutoff passes"); }
    { ArmStats s = good_stats(); s.ref_score_err[0] = 1.0; check(gate_verdict(s).empty(), "verdict: the reference-score error is reported, not bound"); }
    check(!gate_verdict(ArmStats{}).empty(), "verdict: no heads fails");
}

uint64_t g_rng = 0x2545f4914f6cdd1dull;
float next_gauss() {
    auto u = [] { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return ((g_rng >> 40) + 0.5f) / 16777216.0f; };
    return std::sqrt(-2.0f * std::log(u())) * std::cos(6.2831853f * u());
}

template <class T> void write_bin(const std::string & path, const std::vector<T> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v.data(), v.size() * sizeof(T));
}

void write_text(const std::string & path, const std::string & s) { std::ofstream(path) << s; }

void test_reader_and_pipeline(const std::vector<Backend> & backends) {
    char root[128];
    std::snprintf(root, sizeof root, "am_gate_selftest_%d", (int) getpid());
    mkdir(root, 0755);
    const std::string r = root, dir = r + "/synth";
    mkdir(dir.c_str(), 0755);

    // Synthetic fixture in the real format: fitq [n_kv, n_fit, d] with n_fit = G*n - 1 (a dropped
    // query), no q file at all (the gate must not need it), reference = exact rms of fitq.
    const int d = 64, n = 200, nh = 8, nkv = 2, slots = 16, G = nh / nkv, nfit = G * n - 1;
    const int layers[2] = { 3, 7 };
    std::vector<std::vector<float>> rms_ref(2);
    std::vector<std::vector<int32_t>> idx_ref(2);
    for (int li = 0; li < 2; ++li) {
        std::vector<float> fq((size_t) nkv * nfit * d), k((size_t) d * n * nkv);
        for (float & x : fq) x = ggml_fp16_to_fp32(to_f16(next_gauss()));  // exactly representable in F16
        for (int h = 0; h < nkv; ++h)
            for (int j = 0; j < n; ++j) {
                const float sc = std::exp(0.5f * next_gauss());
                for (int c = 0; c < d; ++c) k[((size_t) h * n + j) * d + c] = ggml_fp16_to_fp32(to_f16(sc * next_gauss()));
            }
        std::vector<float> rms((size_t) nkv * n);
        std::vector<int32_t> idx((size_t) nkv * slots);
        for (int h = 0; h < nkv; ++h) {
            const std::vector<double> ex = f64_rms_head(&fq[(size_t) h * nfit * d], &k[(size_t) h * n * d], d, nfit, n);
            const std::vector<int32_t> top = f64_topk(ex, slots);
            for (int j = 0; j < n; ++j) rms[(size_t) h * n + j] = (float) ex[j];
            for (int r2 = 0; r2 < slots; ++r2) idx[(size_t) h * slots + r2] = top[r2];
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "/L%02d_", layers[li]);
        write_bin(dir + nm + "fitq.f32", fq);
        write_bin(dir + nm + "k.f32", k);
        write_bin(dir + nm + "scores.f32", rms);
        write_bin(dir + nm + "idx.i32", idx);
        rms_ref[li] = rms; idx_ref[li] = idx;
    }
    write_bin(dir + "/tokens.i32", std::vector<int32_t>(n, 1));
    const std::string good_manifest =
        "{\n \"case\": \"synth\", \"n_tokens\": 200, \"slots\": 16, \"head_dim\": 64, \"n_head\": 8, \"n_kv\": 2,\n"
        " \"fit\": {\"algorithm_kwargs\": {\"score_method\": \"'rms'\"}},\n \"layers\": [\n  3,\n  7\n ],\n \"files\": []\n}\n";
    write_text(dir + "/manifest.json", good_manifest);

    // reader
    am_fixture::Case c;
    std::string err;
    check(am_fixture::load_case(dir, c, &err) && c.name == "synth" && c.layers == std::vector<int>({ 3, 7 }), "reader: load_case parses the layers array and the case name");
    check(!am_fixture::load_case(r + "/nope", c, &err) && !err.empty(), "reader: a missing manifest is an error");
    std::vector<float> f;
    check(am_fixture::load_case(dir, c) && am_fixture::read_f32(c, 7, "scores", (size_t) nkv * n, f) && f == rms_ref[1], "reader: read_f32 returns the dumped floats bit-exactly");
    std::vector<int32_t> iv;
    check(am_fixture::read_i32(c, 3, "idx", (size_t) nkv * slots, iv) && iv == idx_ref[0], "reader: read_i32 returns the dumped indices");
    check(!am_fixture::read_f32(c, 3, "scores", (size_t) nkv * n + 1, f, &err) && err.find("bytes") != std::string::npos, "reader: a size mismatch is an error naming the bytes");
    check(!am_fixture::read_f32(c, 9, "k", 1, f, &err), "reader: a missing layer file is an error");
    std::vector<int32_t> tk;
    check(am_fixture::read_case_i32(c, "tokens", n, tk) && tk.size() == (size_t) n && tk[0] == 1, "reader: read_case_i32 reads a per-case file");
    std::string mtext;
    int64_t v = 0;
    am_fixture::read_file(dir + "/manifest.json", mtext);
    check(am_fixture::manifest_int(mtext, "head_dim", v) && v == 64 && am_fixture::manifest_int(mtext, "n_kv", v) && v == 2 && !am_fixture::manifest_int(mtext, "absent", v), "reader: manifest_int finds integer fields");
    const std::vector<std::string> dirs = am_fixture::case_dirs(root);
    check(dirs.size() == 1 && dirs[0] == dir, "reader: case_dirs lists directories holding a manifest");
    check(am_fixture::case_dirs(nullptr).empty() && am_fixture::case_dirs("").empty(), "reader: an unset root lists nothing");

    for (const Backend & b : backends) {
        ArmStats st[2];
        const bool ok = gate_case(b, c, st, &err);
        char buf[200];
        std::snprintf(buf, sizeof buf, " q16: math_err max %.3g, ref swaps max %d; q32: math_err max %.3g", dist(st[0].math_err).max,
                      st[0].ref_swaps.empty() ? -1 : *std::max_element(st[0].ref_swaps.begin(), st[0].ref_swaps.end()), dist(st[1].math_err).max);
        check(ok && gate_verdict(st[0]).empty() && gate_verdict(st[1]).empty() && st[0].math_err.size() == 4,
              std::string(b.name) + " gate pipeline on the synthetic fixture (2 layers x 2 heads, n_fit = " + std::to_string(nfit) + ", no q file)" + buf + (ok ? "" : " " + err));
    }

    // ---- negative cases against the CPU backend
    {
        const Backend & b = backends[0];
        auto rewrite_scores = [&](const std::vector<float> & x) { write_bin(dir + "/L03_scores.f32", x); };
        auto rewrite_idx = [&](const std::vector<int32_t> & x) { write_bin(dir + "/L03_idx.i32", x); };
        ArmStats st[2];

        // a reference selection that is not ours must fail check 3: take ranks 8..23 of a 24-key top
        // list (half the keys are swapped, far from the cutoff)
        {
            std::vector<int32_t> wrong = idx_ref[0];
            for (int h = 0; h < nkv; ++h)
                for (int r2 = 0; r2 < slots / 2; ++r2) wrong[(size_t) h * slots + r2] = (int32_t) ((idx_ref[0][(size_t) h * slots] + 1 + r2 * 7 + 100) % n);
            // keep the indices unique per head
            for (int h = 0; h < nkv; ++h) {
                std::vector<char> seen(n, 0);
                for (int r2 = 0; r2 < slots; ++r2) {
                    int32_t key = wrong[(size_t) h * slots + r2];
                    while (seen[key]) key = (key + 1) % n;
                    seen[key] = 1;
                    wrong[(size_t) h * slots + r2] = key;
                }
            }
            rewrite_idx(wrong);
            const bool ok = gate_case(b, c, st, &err);
            check(ok && !gate_verdict(st[0]).empty(), "a reference selection that differs from the engine's fails the verdict: " + gate_verdict(st[0]));
            rewrite_idx(idx_ref[0]);
        }
        // doubled reference scores: reported, not bound
        {
            std::vector<float> dbl = rms_ref[0];
            for (float & x : dbl) x *= 2.0f;
            rewrite_scores(dbl);
            const bool ok = gate_case(b, c, st, &err);
            check(ok && gate_verdict(st[0]).empty() && dist(st[0].ref_score_err).max > 0.4, "doubling the reference scores is reported (error > 0.4) and does not bind");
            rewrite_scores(rms_ref[0]);
        }
        // fitq is mandatory
        {
            std::rename((dir + "/L03_fitq.f32").c_str(), (dir + "/L03_fitq.hold").c_str());
            write_bin(dir + "/L03_q.f32", std::vector<float>((size_t) d * n * nh, 0.5f));
            check(!gate_case(b, c, st, &err) && err.find("fitq") != std::string::npos, "a layer without fitq fails (no fallback to q): " + err);
            std::remove((dir + "/L03_q.f32").c_str());
            std::rename((dir + "/L03_fitq.hold").c_str(), (dir + "/L03_fitq.f32").c_str());
        }
        {
            std::vector<float> half(static_cast<size_t>(nkv) * nfit * d / 2);
            write_bin(dir + "/L03_fitq.hold", half);
            std::rename((dir + "/L03_fitq.f32").c_str(), (dir + "/L03_fitq.orig").c_str());
            std::rename((dir + "/L03_fitq.hold").c_str(), (dir + "/L03_fitq.f32").c_str());
            check(!gate_case(b, c, st, &err), "a truncated fitq (size not a multiple of n_kv*head_dim*4) fails");
            std::rename((dir + "/L03_fitq.orig").c_str(), (dir + "/L03_fitq.f32").c_str());
        }
        std::vector<int32_t> bad_idx = idx_ref[0];
        bad_idx[5] = -1;
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, st, &err) && err.find("index") != std::string::npos, "reference index -1 is rejected before use as a subscript: " + err);
        bad_idx = idx_ref[0];
        bad_idx[5] = n;
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, st, &err), "reference index == n is rejected");
        bad_idx = idx_ref[0];
        bad_idx[5] = bad_idx[4];
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, st, &err) && err.find("unique") != std::string::npos, "duplicate reference indices are rejected: " + err);
        rewrite_idx(idx_ref[0]);
        std::vector<float> nan_sc = rms_ref[0];
        nan_sc[10] = std::nanf("");
        rewrite_scores(nan_sc);
        check(!gate_case(b, c, st, &err), "a NaN reference score fails the gate");
        rewrite_scores(rms_ref[0]);

        auto set_manifest = [&](const std::string & from, const std::string & to) {
            std::string t = good_manifest;
            t.replace(t.find(from), from.size(), to);
            write_text(dir + "/manifest.json", t);
        };
        set_manifest("\"n_head\": 8", "\"n_head\": 7");
        check(!gate_case(b, c, st, &err), "n_head not a multiple of n_kv is rejected");
        set_manifest("\"n_tokens\": 200", "\"n_tokens\": 2000000000");
        check(!gate_case(b, c, st, &err), "an absurd n_tokens is rejected before any allocation");
        set_manifest("\"slots\": 16", "\"slots\": 201");
        check(!gate_case(b, c, st, &err), "slots > n_tokens is rejected");
        set_manifest("\"n_kv\": 2", "\"n_kv\": 0");
        check(!gate_case(b, c, st, &err), "n_kv == 0 is rejected");
        write_text(dir + "/manifest.json", good_manifest);

        int ran = 0;
        check(gate_root(b, r, &ran) && ran == 1, "gate_root runs the synthetic case");
        const std::string empty_root = r + "/qonly_root";
        mkdir(empty_root.c_str(), 0755);
        mkdir((empty_root + "/qonly").c_str(), 0755);
        write_text(empty_root + "/qonly/manifest.json", "{\"layers\": [3]}");
        check(!gate_root(b, empty_root, &ran) && ran == 0, "a root where no case ran (queries-only) fails the gate");
        std::remove((empty_root + "/qonly/manifest.json").c_str());
        rmdir((empty_root + "/qonly").c_str());
        check(!gate_root(b, empty_root, &ran) && ran == 0, "an empty root fails the gate");
        rmdir(empty_root.c_str());
    }

    for (int li = 0; li < 2; ++li)
        for (const char * nm : { "fitq.f32", "k.f32", "scores.f32", "idx.i32" }) {
            char p[160];
            std::snprintf(p, sizeof p, "%s/L%02d_%s", dir.c_str(), layers[li], nm);
            std::remove(p);
        }
    std::remove((dir + "/tokens.i32").c_str());
    std::remove((dir + "/manifest.json").c_str());
    rmdir(dir.c_str());
    rmdir(root);
}

}  // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    bool cpu_only = false;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--cpu-only")) cpu_only = true;
    test_compare_sets();
    test_verdict();
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
    test_reader_and_pipeline(backends);
    if (const char * root = std::getenv("LUCE_AM_FIXTURES")) {
        for (const Backend & b : backends) check(gate_root(b, root), std::string(b.name) + " real fixtures: gate passes");
    } else {
        std::printf("SKIP  real fixture gate (LUCE_AM_FIXTURES not set)\n");
    }
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
