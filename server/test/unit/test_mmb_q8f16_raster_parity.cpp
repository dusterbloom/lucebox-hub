// Parity test for the Q8_0 -> F16 WMMA prefill GEMM raster rule (mmb-q8f16.cuh).
//
// Claim under test: the raster group only changes the order in which output
// tiles are visited, so G=1 (default for K=6144: ssm_out / attn_output, whose
// F16 row pitch is a multiple of 4 KiB) and G=16 (every other K) are bit-exact
// to each other, and both match a double-precision reference of the F16
// products to F32 accumulation tolerance, including a ragged trailing token
// tile. The pure rule mmb_q8f16_raster_group is checked separately.
//
// The launcher is declared with hipStream_t, same as test_gdn_rowsplit_parity.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int ggml_cuda_mmb_q8f16_raster_group(int K, int forced);
void ggml_cuda_mmb_q8f16_launch_raster(const uint8_t * w, const uint16_t * x, float * y, int T, int M, int K, int G, hipStream_t stream);

namespace {

constexpr double TOL = 2.0e-3;   // |gpu - ref| / max(|ref|, 1e-2): F32 accumulation over K F16 products

uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; }
float h2f(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return (float) h; }

bool run_shape(int T, int M, int K) {
    const int nkb = K / 32;
    std::mt19937 rng(7 + K);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.005f, 0.02f), xd(-1.f, 1.f);
    std::vector<uint8_t> w((size_t) M * nkb * 34);
    std::vector<float> wf((size_t) M * K);
    for (int r = 0; r < M; ++r) for (int b = 0; b < nkb; ++b) {
        uint8_t * blk = w.data() + ((size_t) r * nkb + b) * 34;
        const uint16_t d = f2h(dd(rng)); std::memcpy(blk, &d, 2);
        for (int i = 0; i < 32; ++i) { const int8_t q = (int8_t) qd(rng); blk[2 + i] = (uint8_t) q; wf[(size_t) r * K + b * 32 + i] = (float) q * h2f(d); }
    }
    std::vector<uint16_t> x((size_t) T * K);
    std::vector<float> xf(x.size());
    for (size_t i = 0; i < x.size(); ++i) { x[i] = f2h(xd(rng)); xf[i] = h2f(x[i]); }

    uint8_t * dw = nullptr; uint16_t * dx = nullptr; float * dy = nullptr;
    const size_t ybytes = (size_t) T * M * sizeof(float);
    if (hipMalloc(&dw, w.size()) != hipSuccess || hipMalloc(&dx, x.size() * 2) != hipSuccess || hipMalloc(&dy, ybytes) != hipSuccess) {
        std::printf("  alloc failed\n");
        return false;
    }
    hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice);

    std::vector<float> out[2];
    const int groups[2] = {1, 16};
    for (int arm = 0; arm < 2; ++arm) {
        hipMemset(dy, 0, ybytes);
        ggml_cuda_mmb_q8f16_launch_raster(dw, dx, dy, T, M, K, groups[arm], (hipStream_t) 0);
        if (hipDeviceSynchronize() != hipSuccess) { std::printf("  G=%d launch failed\n", groups[arm]); return false; }
        out[arm].resize((size_t) T * M);
        hipMemcpy(out[arm].data(), dy, ybytes, hipMemcpyDeviceToHost);
    }
    hipFree(dw); hipFree(dx); hipFree(dy);

    const bool exact = std::memcmp(out[0].data(), out[1].data(), ybytes) == 0;
    // Reference on a random sample of outputs (the full T*M*K product is too slow on the host).
    double worst = 0.0;
    for (int n = 0; n < 256; ++n) {
        const int t = (int) (rng() % T), r = (int) (rng() % M);
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += (double) wf[(size_t) r * K + k] * xf[(size_t) t * K + k];
        const float g = out[0][(size_t) t * M + r];
        if (!std::isfinite(g)) { worst = NAN; break; }
        worst = std::fmax(worst, std::fabs(acc - g) / std::fmax(std::fabs(acc), 1e-2));
    }
    const bool ok = exact && std::isfinite(worst) && worst <= TOL;
    std::printf("[mmb-q8f16-raster-parity] T=%d M=%d K=%d G=1 vs G=16 %s, ref max_rel %.3g (tol %.1g) %s\n",
                T, M, K, exact ? "bit-exact" : "DIFFER", worst, TOL, ok ? "ok" : "FAIL");
    return ok;
}

bool run_rule() {
    bool ok = true;
    auto expect = [&](int K, int forced, int want) {
        const int got = ggml_cuda_mmb_q8f16_raster_group(K, forced);
        if (got != want) { std::printf("[mmb-q8f16-raster-parity] rule K=%d forced=%d -> %d, want %d FAIL\n", K, forced, got, want); ok = false; }
    };
    expect(6144, 0, 1);    // ssm_out / attn_output: 12288-byte pitch aliases 4 KiB channels
    expect(2048, 0, 1);
    expect(2560, 0, 16);   // qkv / gate / attn_q / k / v / shared-expert gate-up
    expect(640, 0, 16);
    expect(6144, 16, 16);  // LUCE_MMB_Q8F16_RASTER=16 restores the previous raster everywhere
    expect(2560, 4, 4);
    std::printf("[mmb-q8f16-raster-parity] rule %s\n", ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[mmb-q8f16-raster-parity] SKIP: no HIP device\n");
        return 77;
    }
    bool ok = run_rule();
    ok = run_shape(/*T=*/333, /*M=*/2560, /*K=*/6144) && ok;   // ragged last token tile, 3 token tiles
    ok = run_shape(/*T=*/2176, /*M=*/512, /*K=*/6144) && ok;   // 17 token tiles: a full 16-group plus a partial one
    ok = run_shape(/*T=*/300, /*M=*/768, /*K=*/2560) && ok;    // G=16 default shape, 3 row tiles
    return ok ? 0 : 1;
#else
    std::printf("[mmb-q8f16-raster-parity] SKIP: not a HIP build\n");
    return 77;
#endif
}
