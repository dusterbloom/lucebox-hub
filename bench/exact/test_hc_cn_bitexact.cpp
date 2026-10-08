// Standalone bit-identity test for hc_combine_norm_f32_b256 (baseline) vs
// hc_combine_norm_f32_b256_fast (LUCE_QWEN_HC_CN_FAST gamma-prefetch variant).
// Runs both kernels on identical random inputs through the exported test hook
// ggml_cuda_test_hc_combine_norm_bitexact (hc-cn.cu) and memcmp's the outputs
// byte-for-byte. No ggml graph involved -- pure kernel-vs-kernel comparison.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ggml_cuda_test_hc_combine_norm_bitexact(
    const float * inject, const float * residual, const float * block_out, const float * gamma,
    float * out_res_a, float * out_xn_a, float * out_res_b, float * out_xn_b,
    int n_embd, int hc, int n_tokens,
    float s1, float b1, float s2, float b2, float eps, void * raw_stream);

#define HIP_CHECK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e__), __FILE__, __LINE__); exit(1); } } while (0)

static bool run_case(const char * name, int n_embd, int hc, std::mt19937 & rng, bool wide_range) {
    const int rows = hc; // n_tokens = 1, matches production decode shape
    std::uniform_real_distribution<float> small(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wide(-50.0f, 50.0f);
    auto & dist = wide_range ? wide : small;

    std::vector<float> h_inject(hc), h_residual((size_t) rows * n_embd), h_block((size_t) n_embd),
        h_gamma((size_t) rows * n_embd);
    for (auto & v : h_inject)   v = dist(rng);
    for (auto & v : h_residual) v = dist(rng);
    for (auto & v : h_block)    v = dist(rng);
    for (auto & v : h_gamma)    v = dist(rng) * 0.1f + 1.0f; // gamma near 1.0, realistic norm scale

    float *d_inject, *d_residual, *d_block, *d_gamma;
    float *d_res_a, *d_xn_a, *d_res_b, *d_xn_b;
    const size_t row_bytes = (size_t) rows * n_embd * sizeof(float);
    HIP_CHECK(hipMalloc(&d_inject, hc * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_residual, row_bytes));
    HIP_CHECK(hipMalloc(&d_block, (size_t) n_embd * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gamma, row_bytes));
    HIP_CHECK(hipMalloc(&d_res_a, row_bytes));
    HIP_CHECK(hipMalloc(&d_xn_a, row_bytes));
    HIP_CHECK(hipMalloc(&d_res_b, row_bytes));
    HIP_CHECK(hipMalloc(&d_xn_b, row_bytes));

    HIP_CHECK(hipMemcpy(d_inject, h_inject.data(), hc * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_residual, h_residual.data(), row_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_block, h_block.data(), (size_t) n_embd * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gamma, h_gamma.data(), row_bytes, hipMemcpyHostToDevice));
    // poison outputs differently so a no-op kernel can't accidentally "pass"
    HIP_CHECK(hipMemset(d_res_a, 0xAA, row_bytes));
    HIP_CHECK(hipMemset(d_xn_a, 0xAA, row_bytes));
    HIP_CHECK(hipMemset(d_res_b, 0x55, row_bytes));
    HIP_CHECK(hipMemset(d_xn_b, 0x55, row_bytes));

    const float s1 = 1.3f, b1 = -0.2f, s2 = 0.7f, b2 = 0.05f, eps = 1e-5f;
    int ok = ggml_cuda_test_hc_combine_norm_bitexact(
        d_inject, d_residual, d_block, d_gamma, d_res_a, d_xn_a, d_res_b, d_xn_b,
        n_embd, hc, 1, s1, b1, s2, b2, eps, nullptr);
    if (!ok) { fprintf(stderr, "[%s] kernel launch/sync failed\n", name); return false; }

    std::vector<float> h_res_a(rows * (size_t) n_embd), h_xn_a(rows * (size_t) n_embd),
        h_res_b(rows * (size_t) n_embd), h_xn_b(rows * (size_t) n_embd);
    HIP_CHECK(hipMemcpy(h_res_a.data(), d_res_a, row_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_xn_a.data(), d_xn_a, row_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_res_b.data(), d_res_b, row_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_xn_b.data(), d_xn_b, row_bytes, hipMemcpyDeviceToHost));

    const bool res_eq = memcmp(h_res_a.data(), h_res_b.data(), row_bytes) == 0;
    const bool xn_eq  = memcmp(h_xn_a.data(),  h_xn_b.data(),  row_bytes) == 0;
    size_t first_mismatch = (size_t) -1;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < h_xn_a.size(); ++i) {
        float d = h_xn_a[i] - h_xn_b[i];
        if (d != 0.0f) {
            if (first_mismatch == (size_t) -1) first_mismatch = i;
            max_abs_diff = std::max(max_abs_diff, std::fabs(d));
        }
    }
    printf("[%s] n_embd=%d hc=%d out_res %s out_xn %s", name, n_embd, hc,
           res_eq ? "BIT-IDENTICAL" : "MISMATCH", xn_eq ? "BIT-IDENTICAL" : "MISMATCH");
    if (!xn_eq) printf(" (first_mismatch_idx=%zu max_abs_diff=%g)", first_mismatch, (double) max_abs_diff);
    printf("\n");

    hipFree(d_inject); hipFree(d_residual); hipFree(d_block); hipFree(d_gamma);
    hipFree(d_res_a); hipFree(d_xn_a); hipFree(d_res_b); hipFree(d_xn_b);
    return res_eq && xn_eq;
}

int main() {
    std::mt19937 rng(0xC0FFEE);
    bool all_ok = true;
    // production shape: n_embd=2560, hc=4
    for (int rep = 0; rep < 5; ++rep)
        all_ok &= run_case("random-small", 2560, 4, rng, false);
    for (int rep = 0; rep < 5; ++rep)
        all_ok &= run_case("random-wide", 2560, 4, rng, true);
    // odd n_embd (tail path: col+1>=n_embd boundary, not a clean multiple of 512)
    all_ok &= run_case("odd-n_embd", 2561, 4, rng, false);
    all_ok &= run_case("odd-n_embd-wide", 2561, 4, rng, true);
    // smaller/larger hc counts than production, within [1,16] supported range
    all_ok &= run_case("hc1", 2560, 1, rng, false);
    all_ok &= run_case("hc8", 1536, 8, rng, true);
    // n_embd at HC_CN_MAX_EMB boundary (3072), fully exercises all KP=6 iterations
    all_ok &= run_case("max-embd", 3072, 4, rng, true);

    printf(all_ok ? "ALL PASS\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
