// Parity test for the single-launch Q4_K routed gate/up kernel (mmb-routed.cuh, mmb_routed_glu_q4k_kernel).
//
// Claim under test: the balanced wave layout + one tile list + atomic work counter only change which wave computes
// which output and in what order tiles are visited; the BF16 weights and the per-output ascending-K WMMA order are
// those of the legacy two-launch kernels, so the F32 and bf16 outputs are byte-identical to them on skewed routing
// (0-row, 1-row, 65..127-row, exactly 128, >256-row experts), with fewer resident blocks than work items, with an
// M that is not a multiple of 64, and in the in-situ bf16-only mode. Both are spot-checked against a double
// reference of the dequantized weights.
//
// The launcher is declared with hipStream_t, same as test_mmb_q8f16_raster_parity.

#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int ggml_cuda_mmb_glu_scratch_words(int E, int n_rows);
void ggml_cuda_mmb_glu_launch(int mode, const uint8_t * Wg, const uint8_t * Wu, size_t expert_bytes, const uint16_t * Xh, float * D, uint16_t * Dh,
        const int32_t * ids_src, const int32_t * ids_dst, const int32_t * bounds, uint32_t * scratch, int E, int n_rows, int M, int K, int grid, hipStream_t stream);

namespace {

constexpr int E = 512, NU = 10, Q4K_BLOCK = 144;
constexpr double TOL = 2.0e-3;   // F32 accumulation of exact bf16 products vs double
// F16 kernel (mode 2) vs the exact double reference, error rms over the sample relative to the reference rms:
// 3.5e-4 measured on real routing (bench/exact/prefill/GUFO-PARITY.md 'Routed gate/up F16', bf16 path 2.0e-3), and
// the kernel must be no worse than the bf16 path (rms and maxabs) against the same exact reference.
constexpr double TOL_F16 = 1.0e-3;

uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return (float) h; }
uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
float bf2f(uint16_t h) { const uint32_t u = (uint32_t) h << 16; float f; std::memcpy(&f, &u, 4); return f; }

// value `within` (0..255) of one Q4_K block, as the bf16 the kernels stage (RNE of fma(d*sc, q, -dmin*mn))
float q4k_value_bf16(const uint8_t * b, int within) {
    const float d = h2f((uint16_t) (b[0] | (b[1] << 8))), dmin = h2f((uint16_t) (b[2] | (b[3] << 8)));
    const uint8_t * sc = b + 4; const uint8_t * qs = b + 16;
    const int j = within / 32;
    uint8_t s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const float ds = d * s, ms = dmin * m;
    const int slice = within / 64, w = within % 64;
    const uint8_t q = qs[slice * 32 + (w % 32)];
    const int v = w < 32 ? (q & 15) : (q >> 4);
    return bf2f(f2bf(std::fmaf(ds, (float) v, -ms)));
}

// exact value `within` of one Q4_K block (the F16 kernel's reference: dequant and dot in double)
double q4k_value_exact(const uint8_t * b, int within) {
    const double d = h2f((uint16_t) (b[0] | (b[1] << 8))), dmin = h2f((uint16_t) (b[2] | (b[3] << 8)));
    const uint8_t * sc = b + 4; const uint8_t * qs = b + 16;
    const int j = within / 32;
    uint8_t s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const int slice = within / 64, w = within % 64;
    const uint8_t q = qs[slice * 32 + (w % 32)];
    const int v = w < 32 ? (q & 15) : (q >> 4);
    return d * s * v - dmin * m;
}

bool check(bool cond, const char * what) { if (!cond) std::fprintf(stderr, "  FAIL: %s\n", what); return cond; }

struct Routing { std::vector<int32_t> ids_src, ids_dst, bounds; std::vector<int> counts; int n_rows = 0; };

// Skewed top-10 routing: weight exp(-3 rank/E) over a shuffled rank order, expert 0 always empty, expert 1 forced
// to >256 rows, expert 2 to exactly 128, expert 3 to exactly 1 row and expert 4 to 100 rows (needs T >= 385).
Routing make_routing(int T, std::mt19937 & rng) {
    std::vector<int> perm(E); for (int e = 0; e < E; ++e) perm[e] = e;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<double> w(E);
    for (int r = 0; r < E; ++r) w[perm[r]] = std::exp(-3.0 * r / E);
    w[0] = w[1] = w[2] = w[3] = w[4] = 0.0;
    std::vector<std::vector<int>> per_token(T);
    for (int t = 0; t < T; ++t) {
        std::vector<double> ww = w;
        for (int u = 0; u < NU; ++u) {
            std::discrete_distribution<int> dist(ww.begin(), ww.end());
            const int e = dist(rng); per_token[t].push_back(e); ww[e] = 0.0;
        }
    }
    int forced = 0;
    for (int t = 0; t < T && forced < 257; ++t) { per_token[t][0] = 1; ++forced; }
    for (int t = 0; t < T && forced < 257 + 128; ++t) { per_token[t][1] = 2; ++forced; }
    per_token[0][2] = 3;
    for (int t = 0; t < T && t < 100; ++t) per_token[t][3] = 4;
    Routing r; r.counts.assign(E, 0);
    for (int t = 0; t < T; ++t) for (int e : per_token[t]) ++r.counts[e];
    r.bounds.assign(E + 1, 0);
    for (int e = 0; e < E; ++e) r.bounds[e + 1] = r.bounds[e] + r.counts[e];
    r.n_rows = r.bounds[E];
    r.ids_src.assign(r.n_rows, -1); r.ids_dst.assign(r.n_rows, -1);
    std::vector<int> fill(r.bounds.begin(), r.bounds.end() - 1);
    for (int t = 0; t < T; ++t) for (int u = 0; u < NU; ++u) { const int e = per_token[t][u], pos = fill[e]++; r.ids_src[pos] = t; r.ids_dst[pos] = t * NU + u; }
    return r;
}

bool run_case(int T, int M, int K, int grid, std::mt19937 & rng) {
    const int nblk = K / 256, n_rows_x = T;
    const size_t row_bytes = (size_t) nblk * Q4K_BLOCK, expert_bytes = row_bytes * M, w_bytes = expert_bytes * E;
    Routing R = make_routing(T, rng);
    int band65 = 0, zero = 0, over256 = 0, exact128 = 0, one = 0;
    for (int c : R.counts) { zero += c == 0; band65 += c >= 65 && c <= 127; over256 += c > 256; exact128 += c == 128; one += c == 1; }
    std::printf("[mmb-routed-glu-parity] T=%d M=%d K=%d grid=%d rows=%d experts: zero=%d one=%d 65..127=%d ==128=%d >256=%d\n", T, M, K, grid, R.n_rows, zero, one, band65, exact128, over256);
    bool ok = check(zero > 0 && one > 0 && band65 > 0 && over256 > 0, "routing covers the expert-size bands");

    std::vector<uint8_t> wg(w_bytes), wu(w_bytes);
    std::uniform_real_distribution<float> dd(0.0005f, 0.0025f), md(0.0001f, 0.001f), xd(-1.f, 1.f);
    auto fill_w = [&](std::vector<uint8_t> & w) {
        for (size_t b = 0; b < w.size() / Q4K_BLOCK; ++b) {
            uint8_t * p = w.data() + b * Q4K_BLOCK;
            const uint16_t d16 = f2h(dd(rng)), m16 = f2h(md(rng));
            p[0] = d16 & 255; p[1] = d16 >> 8; p[2] = m16 & 255; p[3] = m16 >> 8;
            for (int i = 4; i < Q4K_BLOCK; ++i) p[i] = (uint8_t) (rng() & 255);
        }
    };
    fill_w(wg); fill_w(wu);
    std::vector<uint16_t> xb((size_t) n_rows_x * K), xh(xb.size());   // the F16 kernel reads the same values as F16 (exact)
    for (size_t i = 0; i < xb.size(); ++i) { xb[i] = f2bf(xd(rng)); xh[i] = f2h(bf2f(xb[i])); }

    const size_t out_n = (size_t) R.n_rows * M;
    uint8_t * d_wg = nullptr, * d_wu = nullptr; uint16_t * d_x = nullptr, * d_xh = nullptr, * d_dh = nullptr; float * d_d = nullptr;
    int32_t * d_src = nullptr, * d_dst = nullptr, * d_bounds = nullptr; uint32_t * d_scratch = nullptr;
    const int words = ggml_cuda_mmb_glu_scratch_words(E, R.n_rows);
    ok = check(hipMalloc(&d_wg, w_bytes) == hipSuccess && hipMalloc(&d_wu, w_bytes) == hipSuccess && hipMalloc(&d_x, xb.size() * 2) == hipSuccess && hipMalloc(&d_xh, xh.size() * 2) == hipSuccess
            && hipMalloc(&d_d, out_n * 4) == hipSuccess && hipMalloc(&d_dh, out_n * 2) == hipSuccess
            && hipMalloc(&d_src, R.n_rows * 4) == hipSuccess && hipMalloc(&d_dst, R.n_rows * 4) == hipSuccess && hipMalloc(&d_bounds, (E + 1) * 4) == hipSuccess
            && hipMalloc(&d_scratch, (size_t) words * 4) == hipSuccess, "hipMalloc") && ok;
    if (!ok) return false;
    hipMemcpy(d_wg, wg.data(), w_bytes, hipMemcpyHostToDevice); hipMemcpy(d_wu, wu.data(), w_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_x, xb.data(), xb.size() * 2, hipMemcpyHostToDevice); hipMemcpy(d_xh, xh.data(), xh.size() * 2, hipMemcpyHostToDevice);
    hipMemcpy(d_src, R.ids_src.data(), R.n_rows * 4, hipMemcpyHostToDevice); hipMemcpy(d_dst, R.ids_dst.data(), R.n_rows * 4, hipMemcpyHostToDevice);
    hipMemcpy(d_bounds, R.bounds.data(), (E + 1) * 4, hipMemcpyHostToDevice);

    std::vector<float> f_legacy(out_n), f_new(out_n), f_f16(out_n);
    std::vector<uint16_t> h_legacy(out_n), h_new(out_n), h_insitu(out_n), h_f16(out_n), h_f16_insitu(out_n);
    auto run = [&](int mode, bool f32, std::vector<float> * f, std::vector<uint16_t> * h) {
        hipMemset(d_d, 0, out_n * 4); hipMemset(d_dh, 0, out_n * 2);
        ggml_cuda_mmb_glu_launch(mode, d_wg, d_wu, expert_bytes, mode == 2 ? d_xh : d_x, f32 ? d_d : nullptr, d_dh, d_src, d_dst, d_bounds, d_scratch, E, R.n_rows, M, K, grid, 0);
        const hipError_t err = hipDeviceSynchronize();
        if (!check(err == hipSuccess, "kernel")) { std::fprintf(stderr, "  %s\n", hipGetErrorString(err)); return false; }
        if (f) hipMemcpy(f->data(), d_d, out_n * 4, hipMemcpyDeviceToHost);
        if (h) hipMemcpy(h->data(), d_dh, out_n * 2, hipMemcpyDeviceToHost);
        return true;
    };
    ok = run(0, true, &f_legacy, &h_legacy) && ok;
    ok = run(1, true, &f_new, &h_new) && ok;
    ok = run(1, false, nullptr, &h_insitu) && ok;
    ok = run(2, true, &f_f16, &h_f16) && ok;
    ok = run(2, false, nullptr, &h_f16_insitu) && ok;
    ok = check(std::memcmp(h_f16.data(), h_f16_insitu.data(), out_n * 2) == 0, "F16 kernel: bf16-only (in-situ) output byte-identical to the F32+bf16 run") && ok;
    ok = check(std::memcmp(f_legacy.data(), f_new.data(), out_n * 4) == 0, "F32 output byte-identical to legacy") && ok;
    ok = check(std::memcmp(h_legacy.data(), h_new.data(), out_n * 2) == 0, "bf16 output byte-identical to legacy") && ok;
    ok = check(std::memcmp(h_legacy.data(), h_insitu.data(), out_n * 2) == 0, "bf16-only (in-situ) output byte-identical to legacy") && ok;

    // double reference spot check (expert 1 >256 rows, expert 3 one row, plus random positions); the F16 kernel is
    // measured against the exact reference and must be no worse there than the bf16 kernel.
    double worst = 0, se_f16 = 0, se_bf16 = 0, s_ref = 0, max_f16 = 0, max_bf16 = 0;
    for (int n = 0; n < 512; ++n) {
        const int pos = n == 0 ? R.bounds[3] : n == 1 ? R.bounds[1] + 200 : (int) (rng() % R.n_rows), col = (int) (rng() % M);
        int e = 0; while (R.bounds[e + 1] <= pos) ++e;
        const int tok = R.ids_src[pos], dst = R.ids_dst[pos];
        const uint8_t * rg = wg.data() + (size_t) e * expert_bytes + (size_t) col * row_bytes, * ru = wu.data() + (size_t) e * expert_bytes + (size_t) col * row_bytes;
        double g = 0, u = 0, ge = 0, ue = 0;
        for (int k = 0; k < K; ++k) {
            const double x = bf2f(xb[(size_t) tok * K + k]);
            const uint8_t * bg = rg + (k / 256) * Q4K_BLOCK, * bu = ru + (k / 256) * Q4K_BLOCK;
            g += (double) q4k_value_bf16(bg, k % 256) * x; u += (double) q4k_value_bf16(bu, k % 256) * x;
            ge += q4k_value_exact(bg, k % 256) * x;         ue += q4k_value_exact(bu, k % 256) * x;
        }
        auto rel = [](double want, double got) { return std::fabs(want - got) / std::max(std::fabs(want), 1e-2); };
        const double want = g / (1.0 + std::exp(-g)) * u, got = f_new[(size_t) dst * M + col];
        const double want_e = ge / (1.0 + std::exp(-ge)) * ue, got_f16 = f_f16[(size_t) dst * M + col];
        worst = std::max(worst, rel(want, got));
        const double e16 = std::fabs(want_e - got_f16), e_bf = std::fabs(want_e - got);
        se_f16 += e16 * e16; se_bf16 += e_bf * e_bf; s_ref += want_e * want_e; max_f16 = std::max(max_f16, e16); max_bf16 = std::max(max_bf16, e_bf);
        const double got_h = bf2f(h_new[(size_t) dst * M + col]), got_h16 = bf2f(h_f16[(size_t) dst * M + col]);
        ok = check(std::fabs(got_h - got) <= std::fabs(got) * 4e-3 + 1e-6, "bf16 output is the rounded F32 output") && ok;
        ok = check(std::fabs(got_h16 - got_f16) <= std::fabs(got_f16) * 4e-3 + 1e-6, "F16 kernel: bf16 output is the rounded F32 output") && ok;
    }
    ok = check(worst <= TOL, "F32 output within tolerance of the double reference") && ok;
    const double rel_f16 = std::sqrt(se_f16 / s_ref), rel_bf16 = std::sqrt(se_bf16 / s_ref);
    ok = check(rel_f16 <= TOL_F16, "F16 kernel: F32 output within tolerance of the exact double reference") && ok;
    ok = check(rel_f16 <= rel_bf16 && max_f16 <= max_bf16, "F16 kernel: no worse than the bf16 kernel against the exact reference") && ok;
    std::printf("[mmb-routed-glu-parity]   byte-identical F32/bf16/in-situ %s, ref maxrel=%.2e; vs exact reference rms-rel / maxabs: F16 kernel %.2e / %.2e, bf16 kernel %.2e / %.2e\n", ok ? "ok" : "FAIL", worst, rel_f16, max_f16, rel_bf16, max_bf16);

    hipFree(d_wg); hipFree(d_wu); hipFree(d_x); hipFree(d_xh); hipFree(d_d); hipFree(d_dh); hipFree(d_src); hipFree(d_dst); hipFree(d_bounds); hipFree(d_scratch);
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[mmb-routed-glu-parity] SKIP: no HIP device\n");
        return 77;
    }
    std::mt19937 rng(20261014);
    bool ok = run_case(/*T=*/1024, /*M=*/128, /*K=*/512, /*grid=*/80, rng);   // full m slices, 2 Q4_K blocks per row
    ok = run_case(/*T=*/800, /*M=*/136, /*K=*/256, /*grid=*/7, rng) && ok;     // ragged last m slice, blocks loop over many items
    std::printf("[mmb-routed-glu-parity] %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[mmb-routed-glu-parity] SKIP: not a HIP build\n");
    return 77;
#endif
}
