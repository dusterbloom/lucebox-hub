// Standalone bit-identity test for the LUCE_QWEN_LAUNCH_CUT candidate:
// rms_norm_mul_q8_1_f32 (launch-cut.cu, fused rms_norm(x)*gamma + Q8_1
// quantize in one launch) vs a from-scratch reproduction of today's unfused
// two-kernel path (rms_norm_f32<block,true> then quantize_q8_1). Runs both
// through the exported test hook ggml_cuda_test_launch_cut_rms_norm_mul_q8_1
// (launch-cut.cu) and memcmp's the F32 norm output and the raw Q8_1 bytes
// byte-for-byte. No ggml graph involved -- pure kernel-vs-kernel comparison,
// same pattern as test_hc_cn_bitexact.cpp (commit 8ffa5afb).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ggml_cuda_test_launch_cut_rms_norm_mul_q8_1(
    const float * x, const float * gamma,
    float * dst_a, void * q8_a, float * dst_b, void * q8_b,
    int ncols, int nrows, float eps, void * raw_stream);

#define HIP_CHECK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e__), __FILE__, __LINE__); exit(1); } } while (0)

static bool run_case(const char * name, int ncols, int nrows, std::mt19937 & rng, bool wide_range) {
    std::uniform_real_distribution<float> small(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wide(-50.0f, 50.0f);
    auto & dist = wide_range ? wide : small;

    const size_t n = (size_t) ncols * nrows;
    const size_t q8_blocks = n / 32; // QK8_1 = 32
    const size_t q8_bytes = q8_blocks * 36; // sizeof(block_q8_1) = 2*half + 32 = 36

    std::vector<float> h_x(n), h_gamma(n);
    for (auto & v : h_x)     v = dist(rng);
    for (auto & v : h_gamma) v = dist(rng) * 0.1f + 1.0f; // gamma near 1.0, realistic norm scale

    float *d_x, *d_gamma, *d_dst_a, *d_dst_b;
    void *d_q8_a, *d_q8_b;
    HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gamma, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst_a, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst_b, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q8_a, q8_bytes));
    HIP_CHECK(hipMalloc(&d_q8_b, q8_bytes));

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gamma, h_gamma.data(), n * sizeof(float), hipMemcpyHostToDevice));
    // poison outputs differently so a no-op kernel can't accidentally "pass"
    HIP_CHECK(hipMemset(d_dst_a, 0xAA, n * sizeof(float)));
    HIP_CHECK(hipMemset(d_q8_a, 0xAA, q8_bytes));
    HIP_CHECK(hipMemset(d_dst_b, 0x55, n * sizeof(float)));
    HIP_CHECK(hipMemset(d_q8_b, 0x55, q8_bytes));

    const float eps = 1e-5f;
    int ok = ggml_cuda_test_launch_cut_rms_norm_mul_q8_1(
        d_x, d_gamma, d_dst_a, d_q8_a, d_dst_b, d_q8_b, ncols, nrows, eps, nullptr);
    if (!ok) { fprintf(stderr, "[%s] kernel launch/sync failed\n", name); return false; }

    std::vector<float> h_dst_a(n), h_dst_b(n);
    std::vector<uint8_t> h_q8_a(q8_bytes), h_q8_b(q8_bytes);
    HIP_CHECK(hipMemcpy(h_dst_a.data(), d_dst_a, n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_dst_b.data(), d_dst_b, n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_q8_a.data(), d_q8_a, q8_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_q8_b.data(), d_q8_b, q8_bytes, hipMemcpyDeviceToHost));

    const bool norm_eq = memcmp(h_dst_a.data(), h_dst_b.data(), n * sizeof(float)) == 0;
    const bool q8_eq   = memcmp(h_q8_a.data(), h_q8_b.data(), q8_bytes) == 0;

    printf("[%s] ncols=%d nrows=%d norm %s q8_1 %s\n", name, ncols, nrows,
           norm_eq ? "BIT-IDENTICAL" : "MISMATCH", q8_eq ? "BIT-IDENTICAL" : "MISMATCH");

    hipFree(d_x); hipFree(d_gamma); hipFree(d_dst_a); hipFree(d_dst_b);
    hipFree(d_q8_a); hipFree(d_q8_b);
    return norm_eq && q8_eq;
}

int main() {
    std::mt19937 rng(0xC0FFEE);
    bool all_ok = true;

    // production decode shape: n_embd = 2560, one row at a time (T=1 decode).
    for (int rep = 0; rep < 5; ++rep)
        all_ok &= run_case("decode-row-small", 2560, 1, rng, false);
    for (int rep = 0; rep < 5; ++rep)
        all_ok &= run_case("decode-row-wide", 2560, 1, rng, true);

    // multi-row (e.g. per-head norms sharing one launch): 12, 48 rows.
    all_ok &= run_case("multi-row-12", 2560, 12, rng, false);
    all_ok &= run_case("multi-row-48", 2560, 48, rng, true);

    // >=1024 columns takes the 1024-thread block path.
    all_ok &= run_case("wide-1024", 10240, 1, rng, false);
    all_ok &= run_case("wide-1024-wide", 10240, 4, rng, true);

    // small ncols (<256, still a multiple of QK8_1=32).
    all_ok &= run_case("small-128", 128, 1, rng, false);
    all_ok &= run_case("small-32", 32, 1, rng, true);

    // amax==0 row (all-zero input): exercises the q==0 branch, no div-by-zero.
    {
        const int ncols = 2560, nrows = 1;
        const size_t n = (size_t) ncols * nrows, q8_bytes = (n / 32) * 36;
        float *d_x, *d_gamma, *d_dst_a, *d_dst_b;
        void *d_q8_a, *d_q8_b;
        HIP_CHECK(hipMalloc(&d_x, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_gamma, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dst_a, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_dst_b, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_q8_a, q8_bytes));
        HIP_CHECK(hipMalloc(&d_q8_b, q8_bytes));
        HIP_CHECK(hipMemset(d_x, 0, n * sizeof(float)));
        std::vector<float> h_gamma(n, 1.0f);
        HIP_CHECK(hipMemcpy(d_gamma, h_gamma.data(), n * sizeof(float), hipMemcpyHostToDevice));
        int ok = ggml_cuda_test_launch_cut_rms_norm_mul_q8_1(
            d_x, d_gamma, d_dst_a, d_q8_a, d_dst_b, d_q8_b, ncols, nrows, 1e-5f, nullptr);
        std::vector<uint8_t> h_q8_a(q8_bytes), h_q8_b(q8_bytes);
        HIP_CHECK(hipMemcpy(h_q8_a.data(), d_q8_a, q8_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_q8_b.data(), d_q8_b, q8_bytes, hipMemcpyDeviceToHost));
        const bool eq = ok && memcmp(h_q8_a.data(), h_q8_b.data(), q8_bytes) == 0;
        printf("[all-zero] ncols=%d nrows=%d q8_1 %s\n", ncols, nrows, eq ? "BIT-IDENTICAL" : "MISMATCH");
        all_ok &= eq;
        hipFree(d_x); hipFree(d_gamma); hipFree(d_dst_a); hipFree(d_dst_b);
        hipFree(d_q8_a); hipFree(d_q8_b);
    }

    printf(all_ok ? "ALL PASS\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
