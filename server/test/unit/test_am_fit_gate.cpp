// KVFlash 2.0b M4 R3 gate: am_fit selection against the reference fixtures dumped from the
// compaction repo's own fit (generator: optimizations/kvflash/lucebox/a1/am_fixture_dump.py).
//
// Always runs the self tests (fixture reader, tie-aware comparison, and the whole gate
// pipeline on a synthetic fixture directory it writes itself). The real gate runs only when
// LUCE_AM_FIXTURES names a directory of case directories (traj1_first, traj1_12k, article_r1e).
//
// For every case, full-attention layer and KV head the gate scores the dumped Q and K and
// compares with the dumped reference scores and rank-ordered indices, in two arms:
//   q16: Q rounded to F16, as the engine feeds it   (this arm gates)
//   q32: Q as dumped (F32)                           (diagnostic: separates the F16 rounding)
// Our score is sum w^2; the reference's is its 'rms' score sqrt(mean w^2), mean over the
// G * n pooled query rows, so ours is converted: rms = sqrt(sum / (G * n)).
//
// Tie-aware equality of one head: every key in our top-k but not in the reference's has a
// reference score within `tol` (relative) of the reference's k-th score, and every key in the
// reference's top-k but not in ours is likewise within `tol` of that k-th score.
// Bars (fixed constants, no override): scores within 1e-3 relative per head, selected-key
// overlap >= 99%, at most 2 boundary swaps per head, every head tie-aware equal, and at least one
// case with scores must run.
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
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {

int g_fail = 0;
void check(bool ok, const std::string & what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------- tie-aware comparison (pure)

struct TieCmp {
    int    swapped      = 0;     // keys in ours but not in the reference's top-k
    double worst_gap    = 0.0;   // max relative distance of a swapped key (either direction) to the k-th reference score
    bool   tie_aware_eq = true;
};

TieCmp compare_topk(const int32_t * ours, const int32_t * ref, int k, int n, const float * ref_scores, double tol) {
    TieCmp r;
    // Anything that is not a finite, usable comparison is "not equal"; NaN must never pass a `gap > tol` test.
    if (!(tol >= 0.0) || !std::isfinite(tol) || k < 1) { r.tie_aware_eq = false; return r; }
    double kth = 1e300;
    for (int i = 0; i < k; ++i) {
        if (ref[i] < 0 || ref[i] >= n || ours[i] < 0 || ours[i] >= n) { r.tie_aware_eq = false; return r; }
        kth = std::min<double>(kth, ref_scores[ref[i]]);
    }
    if (!std::isfinite(kth)) { r.tie_aware_eq = false; return r; }
    const double den = std::max(kth, 1e-30);
    auto in = [k](const int32_t * set, int32_t key) { return std::find(set, set + k, key) != set + k; };
    auto note = [&](double gap) {
        r.worst_gap = std::isfinite(gap) ? std::max(r.worst_gap, gap) : r.worst_gap;
        if (!(gap <= tol)) r.tie_aware_eq = false;  // NaN fails here
    };
    for (int i = 0; i < k; ++i) {
        if (!in(ref, ours[i])) {
            ++r.swapped;
            note(std::fabs(kth - ref_scores[ours[i]]) / den);
        }
        if (!in(ours, ref[i])) note(std::fabs(ref_scores[ref[i]] - kth) / den);
    }
    return r;
}

void test_compare_topk() {
    const float sc[] = { 10, 9, 8, 7, 6.9995f, 6.0f, 1 };  // reference top-4: 0 1 2 3, 4th score 7
    const int32_t ref[] = { 0, 1, 2, 3 };
    {
        const int32_t o[] = { 3, 2, 1, 0 };
        const TieCmp c = compare_topk(o, ref, 4, 7, sc, 1e-3);
        check(c.tie_aware_eq && c.swapped == 0 && c.worst_gap == 0.0, "compare_topk: same set in another order is equal, 0 swaps");
    }
    {
        const int32_t o[] = { 0, 1, 2, 4 };  // 6.9995 is within 1e-3 of 7
        const TieCmp c = compare_topk(o, ref, 4, 7, sc, 1e-3);
        check(c.tie_aware_eq && c.swapped == 1 && c.worst_gap > 6e-5 && c.worst_gap < 8e-5, "compare_topk: swap inside tolerance is equal, 1 swap, gap 7.1e-5");
    }
    {
        const int32_t o[] = { 0, 1, 2, 5 };  // 6.0 is 14% below 7
        const TieCmp c = compare_topk(o, ref, 4, 7, sc, 1e-3);
        check(!c.tie_aware_eq && c.swapped == 1 && c.worst_gap > 0.14, "compare_topk: swap outside tolerance fails");
    }
    {
        const float flat[] = { 5, 5, 5, 5, 5 };  // exact ties: any members are equal
        const int32_t r2[] = { 0, 1 }, o2[] = { 3, 4 };
        const TieCmp c = compare_topk(o2, r2, 2, 5, flat, 1e-3);
        check(c.tie_aware_eq && c.swapped == 2 && c.worst_gap == 0.0, "compare_topk: exact ties swap freely");
    }
}

// ---------------------------------------------------------------- scoring one head set through am_fit

struct Backend { const char * name; ggml_backend_t be; };

ggml_fp16_t to_f16(float x) { return ggml_fp32_to_fp16(x); }

// Scores q [d, n, n_head] (F16 or F32 upload) against k [d, n, n_kv] (F32) through the tensor
// entry point. Returns our per-key rms scores [n_kv][n].
bool our_rms_scores(const Backend & b, const std::vector<float> & q, const std::vector<float> & k, int d, int n, int n_head,
                    int n_kv, bool q_f16, std::vector<float> & rms, std::string * err) {
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(ip), ggml_free);
    ggml_tensor * qt = ggml_new_tensor_3d(ctx.get(), q_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, d, n, n_head);
    ggml_tensor * kt = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, n, n_kv);
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
    const double inv_rows = 1.0 / ((double) (n_head / n_kv) * n);
    for (float & s : rms) s = (float) std::sqrt(s * inv_rows);
    return true;
}

// ---------------------------------------------------------------- the gate over a fixture root

struct ArmStats {
    double max_rel_err = 0, max_err_over_tol = 0, worst_gap = 0;
    int    heads = 0, bad_heads = 0, swapped_total = 0, swapped_max = 0;
    int    score_bad_heads = 0, selected = 0;
};

// Gate bars (plan R3: same selected keys >= 99%, outputs within 1e-3 relative). Fixed constants:
// there is no knob to loosen them.
constexpr double kScoreTol       = 1e-3;
constexpr double kMinOverlap     = 0.99;
constexpr int    kMaxSwapsPerHead = 2;    // boundary swaps, each already required to sit within kScoreTol of the cutoff

// Empty when the arm passes the gate, else the reason.
std::string gate_verdict(const ArmStats & s) {
    if (s.heads <= 0 || s.selected <= 0) return "no heads were compared";
    if (s.bad_heads) return std::to_string(s.bad_heads) + " heads not tie-aware equal";
    if (s.score_bad_heads) return std::to_string(s.score_bad_heads) + " heads with a score error above tolerance";
    const double overlap = 1.0 - (double) s.swapped_total / s.selected;
    if (overlap < kMinOverlap) return "selected-key overlap " + std::to_string(overlap) + " below " + std::to_string(kMinOverlap);
    if (s.swapped_max > kMaxSwapsPerHead) return "a head has " + std::to_string(s.swapped_max) + " boundary swaps, bound is " + std::to_string(kMaxSwapsPerHead);
    return "";
}

// Returns false only on a fixture / run error. Fills per-arm stats (index 0: q16, 1: q32).
bool gate_case(const Backend & b, const am_fixture::Case & c, double tol, ArmStats st[2], std::string * err) {
    auto fail = [&](const std::string & m) { if (err) *err = c.name + ": " + m; return false; };
    if (!(tol > 0.0) || !std::isfinite(tol)) return fail("tolerance must be finite and positive");
    std::string manifest;
    am_fixture::read_file(c.dir + "/manifest.json", manifest);
    int64_t d = 0, n = 0, nh = 0, nkv = 0, slots = 0;
    if (!am_fixture::manifest_int(manifest, "head_dim", d) || !am_fixture::manifest_int(manifest, "n_tokens", n) ||
        !am_fixture::manifest_int(manifest, "n_head", nh) || !am_fixture::manifest_int(manifest, "n_kv", nkv) ||
        !am_fixture::manifest_int(manifest, "slots", slots))
        return fail("manifest lacks head_dim/n_tokens/n_head/n_kv/slots");
    // Bound every dimension before it sizes an allocation or a subscript (the files are then checked
    // against these counts by size).
    constexpr int64_t kMaxDim = 1 << 16;
    if (d < 1 || d > kMaxDim || n < 1 || n > kMaxDim || nh < 1 || nh > 1024 || nkv < 1 || nkv > nh || nh % nkv != 0 ||
        slots < 1 || slots > n)
        return fail("implausible dimensions head_dim=" + std::to_string(d) + " n_tokens=" + std::to_string(n) + " n_head=" +
                    std::to_string(nh) + " n_kv=" + std::to_string(nkv) + " slots=" + std::to_string(slots));
    const size_t sm = manifest.find("\"score_method\"");
    if (sm != std::string::npos && manifest.find("rms", sm) > sm + 48)
        return fail("the fixture's score_method is not rms, scores are not comparable");
    for (int layer : c.layers) {
        std::vector<float> q, k, ref;
        std::vector<int32_t> idx;
        if (!am_fixture::read_f32(c, layer, "q", d * n * nh, q, err) || !am_fixture::read_f32(c, layer, "k", d * n * nkv, k, err) ||
            !am_fixture::read_f32(c, layer, "scores", n * nkv, ref, err) || !am_fixture::read_i32(c, layer, "idx", slots * nkv, idx, err))
            return false;
        for (float x : q) if (!std::isfinite(x)) return fail("non-finite value in q");
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
        for (int arm = 0; arm < 2; ++arm) {
            std::vector<float> rms;
            if (!our_rms_scores(b, q, k, (int) d, (int) n, (int) nh, (int) nkv, arm == 0, rms, err)) return false;
            for (float x : rms) if (!std::isfinite(x)) return fail("non-finite score from am_fit");
            ArmStats & s = st[arm];
            for (int h = 0; h < nkv; ++h) {
                const float * rs = &ref[(size_t) h * n];
                const float * os = &rms[(size_t) h * n];
                double mx = 0;
                for (int j = 0; j < n; ++j) mx = std::max<double>(mx, rs[j]);
                double head_err = 0;
                for (int j = 0; j < n; ++j) {
                    const double e = std::fabs((double) os[j] - rs[j]);
                    s.max_rel_err = std::max(s.max_rel_err, e / std::max<double>(rs[j], 1e-30));
                    head_err = std::max(head_err, e / (tol * rs[j] + 1e-6 * mx));
                }
                s.max_err_over_tol = std::max(s.max_err_over_tol, head_err);
                if (!(head_err <= 1.0)) ++s.score_bad_heads;
                std::vector<int32_t> ours(slots);
                luce::am::select_top_k(os, (int) n, (int) slots, ours.data());
                const TieCmp t = compare_topk(ours.data(), &idx[(size_t) h * slots], (int) slots, (int) n, rs, tol);
                ++s.heads;
                s.selected += (int) slots;
                s.bad_heads += t.tie_aware_eq ? 0 : 1;
                s.swapped_total += t.swapped;
                s.swapped_max = std::max(s.swapped_max, t.swapped);
                s.worst_gap = std::max(s.worst_gap, t.worst_gap);
            }
        }
    }
    return true;
}

// Runs every case under `root`; true when the gate passes. Prints one GATE line per case and arm.
bool gate_root(const Backend & b, const std::string & root, double tol, int * ran_out = nullptr) {
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
        if (!gate_case(b, c, tol, st, &err)) { std::printf("GATEFAIL %s %s: %s\n", b.name, c.name.c_str(), err.c_str()); pass = false; continue; }
        ++ran;
        static const char * arm_name[2] = { "q16", "q32" };
        for (int a = 0; a < 2; ++a)
            std::printf("GATE  %s %-14s %s layers=%zu heads=%d  max_rel_err=%.3g max_err/tol=%.3g  swapped_total=%d swapped_max/head=%d worst_boundary_gap=%.3g  score_bad_heads=%d tie_aware_unequal_heads=%d  (tol %.0e)\n",
                        b.name, c.name.c_str(), arm_name[a], c.layers.size(), st[a].heads, st[a].max_rel_err, st[a].max_err_over_tol,
                        st[a].swapped_total, st[a].swapped_max, st[a].worst_gap, st[a].score_bad_heads, st[a].bad_heads, tol);
        const std::string why = gate_verdict(st[0]);
        if (!why.empty()) { std::printf("GATEFAIL %s %s q16 arm: %s\n", b.name, c.name.c_str(), why.c_str()); pass = false; }
    }
    if (ran_out) *ran_out = ran;
    if (ran == 0) { std::printf("GATEFAIL %s: no case with scores ran under %s\n", b.name, root.c_str()); pass = false; }
    return pass;
}

void report_gate(const Backend & b, const std::string & root, double tol) {
    check(gate_root(b, root, tol), std::string(b.name) + " real fixtures: gate passes");
}

// ---------------------------------------------------------------- self tests: reader + pipeline on a written fixture

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

    const int d = 64, n = 200, nh = 8, nkv = 2, slots = 16, G = nh / nkv;
    const int layers[2] = { 3, 7 };
    std::vector<std::vector<float>> qs, ks;
    std::vector<std::vector<float>> rms_ref(2);
    std::vector<std::vector<int32_t>> idx_ref(2);
    for (int li = 0; li < 2; ++li) {
        std::vector<float> q((size_t) d * n * nh), k((size_t) d * n * nkv);
        for (float & x : q) x = ggml_fp16_to_fp32(to_f16(next_gauss()));  // exactly representable in F16
        for (int h = 0; h < nkv; ++h)
            for (int j = 0; j < n; ++j) {
                const float sc = std::exp(0.5f * next_gauss());
                for (int c = 0; c < d; ++c) k[((size_t) h * n + j) * d + c] = ggml_fp16_to_fp32(to_f16(sc * next_gauss()));
            }
        // double reference of the repo's rms score
        std::vector<float> rms((size_t) nkv * n);
        std::vector<int32_t> idx((size_t) nkv * slots);
        for (int h = 0; h < nkv; ++h) {
            std::vector<double> acc(n, 0.0), s(n);
            for (int g = 0; g < G; ++g)
                for (int i = 0; i < n; ++i) {
                    double mx = -1e300;
                    for (int j = 0; j < n; ++j) {
                        double a = 0;
                        for (int c = 0; c < d; ++c) a += (double) q[((size_t) (h * G + g) * n + i) * d + c] * k[((size_t) h * n + j) * d + c];
                        s[j] = a / std::sqrt((double) d);
                        mx = std::max(mx, s[j]);
                    }
                    double sum = 0;
                    for (int j = 0; j < n; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
                    for (int j = 0; j < n; ++j) acc[j] += (s[j] / sum) * (s[j] / sum);
                }
            std::vector<int32_t> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return acc[a] > acc[c]; });
            for (int j = 0; j < n; ++j) rms[(size_t) h * n + j] = (float) std::sqrt(acc[j] / ((double) G * n));
            for (int r2 = 0; r2 < slots; ++r2) idx[(size_t) h * slots + r2] = order[r2];
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "/L%02d_", layers[li]);
        write_bin(dir + nm + "q.f32", q);
        write_bin(dir + nm + "k.f32", k);
        write_bin(dir + nm + "scores.f32", rms);
        write_bin(dir + nm + "idx.i32", idx);
        qs.push_back(q); ks.push_back(k); rms_ref[li] = rms; idx_ref[li] = idx;
    }
    write_bin(dir + "/tokens.i32", std::vector<int32_t>(n, 1));
    write_text(dir + "/manifest.json",
               "{\n \"case\": \"synth\", \"n_tokens\": 200, \"slots\": 16, \"head_dim\": 64, \"n_head\": 8, \"n_kv\": 2,\n"
               " \"fit\": {\"algorithm_kwargs\": {\"score_method\": \"'rms'\"}},\n \"layers\": [\n  3,\n  7\n ],\n \"files\": []\n}\n");

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
    check(!am_fixture::read_f32(c, 9, "q", 1, f, &err), "reader: a missing layer file is an error");
    std::vector<int32_t> tk;
    check(am_fixture::read_case_i32(c, "tokens", n, tk) && tk.size() == (size_t) n && tk[0] == 1, "reader: read_case_i32 reads a per-case file");
    std::string mtext;
    int64_t v = 0;
    am_fixture::read_file(dir + "/manifest.json", mtext);
    check(am_fixture::manifest_int(mtext, "head_dim", v) && v == 64 && am_fixture::manifest_int(mtext, "n_kv", v) && v == 2 && !am_fixture::manifest_int(mtext, "absent", v), "reader: manifest_int finds integer fields");
    const std::vector<std::string> dirs = am_fixture::case_dirs(root);
    check(dirs.size() == 1 && dirs[0] == dir, "reader: case_dirs lists directories holding a manifest");
    check(am_fixture::case_dirs(nullptr).empty() && am_fixture::case_dirs("").empty(), "reader: an unset root lists nothing");

    // pipeline: the gate on the synthetic fixture passes in both arms (Q is F16-exact)
    for (const Backend & b : backends) {
        ArmStats st[2];
        const bool ok = gate_case(b, c, 1e-3, st, &err);
        char buf[160];
        std::snprintf(buf, sizeof buf, " q16: err/tol=%.3g swapped=%d, q32: err/tol=%.3g swapped=%d", st[0].max_err_over_tol, st[0].swapped_total, st[1].max_err_over_tol, st[1].swapped_total);
        check(ok && st[0].bad_heads == 0 && st[1].bad_heads == 0 && st[0].max_err_over_tol <= 1.0 && st[1].max_err_over_tol <= 1.0,
              std::string(b.name) + " gate pipeline on the synthetic fixture (2 layers x 2 heads)" + buf + (ok ? "" : " " + err));
        // a deliberately wrong reference must be caught: shuffle one head's reference scores
        std::vector<float> bad = rms_ref[0];
        std::reverse(bad.begin(), bad.begin() + n);
        write_bin(dir + "/L03_scores.f32", bad);
        ArmStats bs[2];
        const bool ok2 = gate_case(b, c, 1e-3, bs, &err);
        check(ok2 && bs[0].max_err_over_tol > 1.0, std::string(b.name) + " gate pipeline reports a wrong reference score vector");
        write_bin(dir + "/L03_scores.f32", rms_ref[0]);
    }

    // ---- hardening: the gate must bind (Codex follow-up review)
    {
        const Backend & b = backends[0];
        auto rewrite_scores = [&](const std::vector<float> & v) { write_bin(dir + "/L03_scores.f32", v); };
        auto rewrite_idx = [&](const std::vector<int32_t> & v) { write_bin(dir + "/L03_idx.i32", v); };
        std::string good_manifest;
        am_fixture::read_file(dir + "/manifest.json", good_manifest);
        ArmStats st[2];

        std::vector<float> dbl = rms_ref[0];
        for (float & x : dbl) x *= 2.0f;
        rewrite_scores(dbl);
        const bool ok_dbl = gate_case(b, c, 1e-3, st, &err);
        check(ok_dbl && !gate_verdict(st[0]).empty() && st[0].score_bad_heads > 0,
              "gate binds on score error: doubling every reference score fails the verdict (" + gate_verdict(st[0]) + ")");
        rewrite_scores(rms_ref[0]);

        std::vector<int32_t> bad_idx = idx_ref[0];
        bad_idx[5] = -1;
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, 1e-3, st, &err) && err.find("index") != std::string::npos,
              "reference index -1 is rejected before use as a subscript: " + err);
        bad_idx = idx_ref[0];
        bad_idx[5] = n;
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, 1e-3, st, &err), "reference index == n is rejected");
        bad_idx = idx_ref[0];
        bad_idx[5] = bad_idx[4];
        rewrite_idx(bad_idx);
        check(!gate_case(b, c, 1e-3, st, &err) && err.find("unique") != std::string::npos,
              "duplicate reference indices are rejected: " + err);
        rewrite_idx(idx_ref[0]);

        std::vector<float> nan_sc = rms_ref[0];
        nan_sc[10] = std::nanf("");
        rewrite_scores(nan_sc);
        check(!gate_case(b, c, 1e-3, st, &err), "a NaN reference score fails the gate instead of leaving the max error at 0");
        rewrite_scores(rms_ref[0]);

        check(!gate_case(b, c, std::nan(""), st, &err), "a NaN tolerance is rejected");
        check(!gate_case(b, c, 0.0, st, &err) && !gate_case(b, c, -1.0, st, &err), "a non-positive tolerance is rejected");
        {
            const float sc2[] = { 10, 9, 8, 7, 1 };
            const int32_t rr[] = { 0, 1, 2, 3 }, oo[] = { 0, 1, 2, 4 };
            check(!compare_topk(oo, rr, 4, 5, sc2, std::nan("")).tie_aware_eq, "compare_topk: a NaN tolerance never accepts a swap");
            const float nan_sc2[] = { 10, 9, 8, 7, std::nanf("") };
            check(!compare_topk(oo, rr, 4, 5, nan_sc2, 1e-3).tie_aware_eq, "compare_topk: a NaN score never accepts a swap");
            const int32_t oor[] = { 0, 1, 2, 9 };
            check(!compare_topk(oor, rr, 4, 5, sc2, 1e-3).tie_aware_eq, "compare_topk: an out-of-range key is not equal");
        }

        auto set_manifest = [&](const std::string & from, const std::string & to) {
            std::string t = good_manifest;
            t.replace(t.find(from), from.size(), to);
            write_text(dir + "/manifest.json", t);
        };
        set_manifest("\"n_head\": 8", "\"n_head\": 7");
        check(!gate_case(b, c, 1e-3, st, &err), "n_head not a multiple of n_kv is rejected");
        set_manifest("\"n_tokens\": 200", "\"n_tokens\": 2000000000");
        check(!gate_case(b, c, 1e-3, st, &err), "an absurd n_tokens is rejected before any allocation");
        set_manifest("\"slots\": 16", "\"slots\": 201");
        check(!gate_case(b, c, 1e-3, st, &err), "slots > n_tokens is rejected");
        set_manifest("\"n_kv\": 2", "\"n_kv\": 0");
        check(!gate_case(b, c, 1e-3, st, &err), "n_kv == 0 is rejected");
        write_text(dir + "/manifest.json", good_manifest);

        ArmStats v;
        v.heads = 4; v.selected = 4 * 64; v.swapped_total = 3; v.swapped_max = 1;
        check(!gate_verdict(v).empty(), "verdict: 3 of 256 selected keys swapped is below the 99% overlap bar");
        v.swapped_total = 2; v.swapped_max = 3;
        check(!gate_verdict(v).empty(), "verdict: more than 2 boundary swaps in one head fails");
        v.swapped_total = 2; v.swapped_max = 2;
        check(gate_verdict(v).empty(), "verdict: 2 of 256 swapped (99.2% overlap, 2 per head) passes");

        int ran = 0;
        check(gate_root(b, r, 1e-3, &ran) && ran == 1, "gate_root runs the synthetic case");
        const std::string empty_root = r + "/qonly_root";
        mkdir(empty_root.c_str(), 0755);
        mkdir((empty_root + "/qonly").c_str(), 0755);
        write_text(empty_root + "/qonly/manifest.json", "{\"layers\": [3]}");
        check(!gate_root(b, empty_root, 1e-3, &ran) && ran == 0, "a root where no case ran (queries-only) fails the gate");
        std::remove((empty_root + "/qonly/manifest.json").c_str());
        rmdir((empty_root + "/qonly").c_str());
        check(!gate_root(b, empty_root, 1e-3, &ran) && ran == 0, "an empty root fails the gate");
        rmdir(empty_root.c_str());
    }

    // cleanup
    for (int li = 0; li < 2; ++li)
        for (const char * nm : { "q.f32", "k.f32", "scores.f32", "idx.i32" }) {
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
    test_compare_topk();
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
        for (const Backend & b : backends) report_gate(b, root, kScoreTol);
    } else {
        std::printf("SKIP  real fixture gate (LUCE_AM_FIXTURES not set)\n");
    }
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    std::printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
