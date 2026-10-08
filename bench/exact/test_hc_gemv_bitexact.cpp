// Standalone bit-identity test + microbenchmark for the two hc_* GEMV kernels:
//   hc_upmix_row8_exact        (baseline)  vs  hc_upmix_row8_exact_fast        (LUCE_QWEN_HC_GEMV_FAST)
//   hc_down_inject_mixed       (baseline)  vs  hc_down_inject_mixed_fast       (LUCE_QWEN_HC_GEMV_FAST)
//
// Follows the pattern of test_hc_cn_bitexact.cpp (hc_combine_norm_f32_b256 vs _fast):
// runs both kernels on identical random device inputs through exported test hooks
// (mmvq.cu: ggml_cuda_test_hc_upmix_bitexact, ggml_cuda_test_hc_down_inject_bitexact) and
// memcmp's the outputs byte-for-byte. No ggml graph involved -- pure kernel-vs-kernel
// comparison, gfx1151-shaped (production n_embd=2560, hc=4, hc_dim=10240).
//
// Build (DO NOT run until the coordinator confirms "GPU free" -- another agent is timing
// on lucebox4 and a concurrent build/run perturbs it): on the box, in a FRESH build dir
// (never cp -a'd; TMPDIR must stay out of /tmp, e.g. /home/duster/qwen4exp-f16/scratch):
//
//   cd ~/qwen4exp-hc-gemv && cmake -S server/deps/llama.cpp -B build1151 \
//       <same cache options as ~/qwen4exp-exact-stack/build1151/CMakeCache.txt>
//   cmake --build build1151 --target ggml-hip -j32
//   hipcc -std=c++17 -O2 --offload-arch=gfx1151 \
//       -I server/deps/llama.cpp/ggml/include -I server/deps/llama.cpp/ggml/src \
//       bench/exact/test_hc_gemv_bitexact.cpp \
//       build1151/ggml/src/CMakeFiles/ggml-hip.dir/ggml-cuda/mmvq.cu.o \
//       -L build1151/ggml/src -lggml-hip -lggml-base -lamdhip64 \
//       -o bench/exact/test_hc_gemv_bitexact
//   ./bench/exact/test_hc_gemv_bitexact
//
// (Exact object/library paths depend on build1151's CMake layout -- adjust to match
// whatever ggml-hip target actually produces; the point is linking mmvq.cu's object code,
// which exports the two bitexact test hooks via extern "C" GGML_BACKEND_API.)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

// Mirrors block_q8_0 / block_q8_1 from ggml-common.h (sizeof 34 / 36 bytes respectively).
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
struct block_q8_1 { uint16_t d; uint16_t s; int8_t qs[32]; };
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout mismatch");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 layout mismatch");

extern "C" int ggml_cuda_test_hc_upmix_bitexact(
    const void * weights, const block_q8_1 * lo_q8, const float * xn,
    float * mixed_a, float * mixed_b, float scale, float bias, void * raw_stream);

extern "C" int ggml_cuda_test_hc_down_inject_bitexact(
    const void * down, const block_q8_1 * xq, float * down_dst_a, float * down_dst_b,
    const float * inject, const float * x, float * inject_dst_a, float * inject_dst_b,
    void * raw_stream);

extern "C" int ggml_cuda_bench_hc_upmix(
    int use_fast, const void * weights, const block_q8_1 * lo_q8, const float * xn,
    float * mixed, float scale, float bias, int n_reps, void * raw_stream);

extern "C" int ggml_cuda_bench_hc_down_inject(
    int use_fast, const void * down, const block_q8_1 * xq, float * down_dst,
    const float * inject, const float * x, float * inject_dst, int n_reps, void * raw_stream);

#define HIP_CHECK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e__), __FILE__, __LINE__); exit(1); } } while (0)

static uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

// Fills a buffer of block_q8_0 with random-but-finite data (no reference quantization
// relationship required -- this test only checks baseline-vs-fast agreement, not numeric
// correctness against an unquantized GEMM).
static void fill_q8_0(std::vector<block_q8_0> & blocks, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dscale(0.001f, 0.05f);
    std::uniform_int_distribution<int> dq(-120, 120);
    for (auto & b : blocks) {
        b.d = f2h(dscale(rng));
        for (auto & q : b.qs) q = (int8_t) dq(rng);
    }
}

static void fill_q8_1(std::vector<block_q8_1> & blocks, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dscale(0.001f, 0.05f);
    std::uniform_int_distribution<int> dq(-120, 120);
    for (auto & b : blocks) {
        b.d = f2h(dscale(rng));
        int32_t sum = 0;
        for (auto & q : b.qs) { q = (int8_t) dq(rng); sum += q; }
        b.s = f2h(__half2float(__ushort_as_half(b.d)) * (float) sum);
    }
}

// --- hc_upmix_row8_exact vs hc_upmix_row8_exact_fast ----------------------------------
static bool run_upmix_case(const char * name, std::mt19937 & rng) {
    constexpr int n_rows = 10240;       // hc_dim = n_embd(2560) * hc(4)
    constexpr int blocks_per_row = 10;  // 320 / QK8_0(32)
    constexpr int n_lo_blocks = 16;     // 512 (MATRIX_ROW_PADDING) / QK8_1(32), 576 B total
    constexpr int n_mixed = 2560;       // n_embd

    std::vector<block_q8_0> h_weights((size_t) n_rows * blocks_per_row);
    std::vector<block_q8_1> h_lo_q8(n_lo_blocks);
    std::vector<float> h_xn(n_rows);
    fill_q8_0(h_weights, rng);
    fill_q8_1(h_lo_q8, rng);
    std::uniform_real_distribution<float> dxn(-2.0f, 2.0f);
    for (auto & v : h_xn) v = dxn(rng);

    void * d_weights; block_q8_1 * d_lo_q8; float * d_xn, * d_mixed_a, * d_mixed_b;
    HIP_CHECK(hipMalloc(&d_weights, h_weights.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_lo_q8, h_lo_q8.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&d_xn, h_xn.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_mixed_a, n_mixed * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_mixed_b, n_mixed * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_lo_q8, h_lo_q8.data(), h_lo_q8.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xn, h_xn.data(), h_xn.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_mixed_a, 0xAA, n_mixed * sizeof(float)));
    HIP_CHECK(hipMemset(d_mixed_b, 0x55, n_mixed * sizeof(float)));

    int ok = ggml_cuda_test_hc_upmix_bitexact(d_weights, d_lo_q8, d_xn, d_mixed_a, d_mixed_b,
                                               0.25f, 0.0f, nullptr);
    bool pass = false;
    if (!ok) {
        fprintf(stderr, "[%s] kernel launch/sync failed\n", name);
    } else {
        std::vector<float> h_mixed_a(n_mixed), h_mixed_b(n_mixed);
        HIP_CHECK(hipMemcpy(h_mixed_a.data(), d_mixed_a, n_mixed * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_mixed_b.data(), d_mixed_b, n_mixed * sizeof(float), hipMemcpyDeviceToHost));
        pass = memcmp(h_mixed_a.data(), h_mixed_b.data(), n_mixed * sizeof(float)) == 0;
        size_t first = (size_t) -1; float max_diff = 0.0f;
        for (size_t i = 0; i < h_mixed_a.size(); ++i) {
            float d = h_mixed_a[i] - h_mixed_b[i];
            if (d != 0.0f) { if (first == (size_t) -1) first = i; max_diff = std::max(max_diff, std::fabs(d)); }
        }
        printf("[%s] mixed[2560] %s", name, pass ? "BIT-IDENTICAL" : "MISMATCH");
        if (!pass) printf(" (first_mismatch_idx=%zu max_abs_diff=%g)", first, (double) max_diff);
        printf("\n");
    }
    hipFree(d_weights); hipFree(d_lo_q8); hipFree(d_xn); hipFree(d_mixed_a); hipFree(d_mixed_b);
    return pass;
}

// --- hc_down_inject_mixed vs hc_down_inject_mixed_fast ---------------------------------
static bool run_down_inject_case(const char * name, std::mt19937 & rng) {
    constexpr int ncols = 10240;
    constexpr int down_rows = 320;
    constexpr int down_blocks_per_row = ncols / 32; // 320
    constexpr int xq_blocks = ncols / 32;            // 320, 11.52 KB
    constexpr int inject_rows = 4;

    std::vector<block_q8_0> h_down((size_t) down_rows * down_blocks_per_row);
    std::vector<block_q8_1> h_xq(xq_blocks);
    std::vector<float> h_inject((size_t) inject_rows * ncols);
    std::vector<float> h_x(ncols);
    fill_q8_0(h_down, rng);
    fill_q8_1(h_xq, rng);
    std::uniform_real_distribution<float> dw(-0.05f, 0.05f), dx(-2.0f, 2.0f);
    for (auto & v : h_inject) v = dw(rng);
    for (auto & v : h_x) v = dx(rng);

    void * d_down; block_q8_1 * d_xq; float * d_down_dst_a, * d_down_dst_b;
    float * d_inject, * d_x, * d_inject_dst_a, * d_inject_dst_b;
    HIP_CHECK(hipMalloc(&d_down, h_down.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_xq, h_xq.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&d_down_dst_a, down_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_down_dst_b, down_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_inject, h_inject.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_inject_dst_a, inject_rows * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_inject_dst_b, inject_rows * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_down, h_down.data(), h_down.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xq, h_xq.data(), h_xq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_inject, h_inject.data(), h_inject.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_down_dst_a, 0xAA, down_rows * sizeof(float)));
    HIP_CHECK(hipMemset(d_down_dst_b, 0x55, down_rows * sizeof(float)));
    HIP_CHECK(hipMemset(d_inject_dst_a, 0xAA, inject_rows * sizeof(float)));
    HIP_CHECK(hipMemset(d_inject_dst_b, 0x55, inject_rows * sizeof(float)));

    int ok = ggml_cuda_test_hc_down_inject_bitexact(d_down, d_xq, d_down_dst_a, d_down_dst_b,
                                                     d_inject, d_x, d_inject_dst_a, d_inject_dst_b,
                                                     nullptr);
    bool pass = false;
    if (!ok) {
        fprintf(stderr, "[%s] kernel launch/sync failed\n", name);
    } else {
        std::vector<float> h_down_a(down_rows), h_down_b(down_rows);
        std::vector<float> h_inj_a(inject_rows), h_inj_b(inject_rows);
        HIP_CHECK(hipMemcpy(h_down_a.data(), d_down_dst_a, down_rows * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_down_b.data(), d_down_dst_b, down_rows * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_inj_a.data(), d_inject_dst_a, inject_rows * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_inj_b.data(), d_inject_dst_b, inject_rows * sizeof(float), hipMemcpyDeviceToHost));
        const bool down_eq = memcmp(h_down_a.data(), h_down_b.data(), down_rows * sizeof(float)) == 0;
        const bool inj_eq = memcmp(h_inj_a.data(), h_inj_b.data(), inject_rows * sizeof(float)) == 0;
        pass = down_eq && inj_eq;
        printf("[%s] down_dst[320] %s  inject_dst[4] %s\n", name,
               down_eq ? "BIT-IDENTICAL" : "MISMATCH", inj_eq ? "BIT-IDENTICAL" : "MISMATCH");
    }
    hipFree(d_down); hipFree(d_xq); hipFree(d_down_dst_a); hipFree(d_down_dst_b);
    hipFree(d_inject); hipFree(d_x); hipFree(d_inject_dst_a); hipFree(d_inject_dst_b);
    return pass;
}

// --- Microbenchmark: µs/launch, old vs new, both kernels (reported only -- correctness
// gate above must pass first; this loop is informational and is NOT part of the pass/fail
// gate since its numbers are environment/GPU-state dependent). ------------------------
static double time_reps(int use_fast, bool is_upmix,
                         void * weights_or_down, block_q8_1 * q8, const float * act,
                         float * dst_a, float scale_or_inject_unused, float bias_unused,
                         const float * x_for_down, float * inject_dst, int reps) {
    // warm-up
    const int warmup = 20;
    for (int i = 0; i < warmup; ++i) {
        if (is_upmix) ggml_cuda_bench_hc_upmix(use_fast, weights_or_down, q8, act, dst_a, scale_or_inject_unused, bias_unused, 1, nullptr);
        else          ggml_cuda_bench_hc_down_inject(use_fast, weights_or_down, q8, dst_a, x_for_down, act, inject_dst, 1, nullptr);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    if (is_upmix) ggml_cuda_bench_hc_upmix(use_fast, weights_or_down, q8, act, dst_a, scale_or_inject_unused, bias_unused, reps, nullptr);
    else          ggml_cuda_bench_hc_down_inject(use_fast, weights_or_down, q8, dst_a, x_for_down, act, inject_dst, reps, nullptr);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

static void microbench() {
    printf("\n-- microbench (informational; run only once GPU is confirmed free) --\n");
    std::mt19937 rng(0xBEEF);
    constexpr int reps = 2000;

    // upmix: old vs new
    {
        constexpr int n_rows = 10240, blocks_per_row = 10, n_lo_blocks = 16, n_mixed = 2560;
        std::vector<block_q8_0> h_weights((size_t) n_rows * blocks_per_row);
        std::vector<block_q8_1> h_lo_q8(n_lo_blocks);
        std::vector<float> h_xn(n_rows);
        fill_q8_0(h_weights, rng); fill_q8_1(h_lo_q8, rng);
        std::uniform_real_distribution<float> dxn(-2.0f, 2.0f);
        for (auto & v : h_xn) v = dxn(rng);
        void * d_weights; block_q8_1 * d_lo_q8; float * d_xn, * d_mixed;
        HIP_CHECK(hipMalloc(&d_weights, h_weights.size() * sizeof(block_q8_0)));
        HIP_CHECK(hipMalloc(&d_lo_q8, h_lo_q8.size() * sizeof(block_q8_1)));
        HIP_CHECK(hipMalloc(&d_xn, h_xn.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_mixed, n_mixed * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_lo_q8, h_lo_q8.data(), h_lo_q8.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_xn, h_xn.data(), h_xn.size() * sizeof(float), hipMemcpyHostToDevice));

        double us_old = time_reps(0, true, d_weights, d_lo_q8, d_xn, d_mixed, 0.25f, 0.0f, nullptr, nullptr, reps);
        double us_new = time_reps(1, true, d_weights, d_lo_q8, d_xn, d_mixed, 0.25f, 0.0f, nullptr, nullptr, reps);
        printf("hc_upmix_row8_exact:      %.2f us/launch (old, %d reps)\n", us_old, reps);
        printf("hc_upmix_row8_exact_fast: %.2f us/launch (new, %d reps)\n", us_new, reps);
        hipFree(d_weights); hipFree(d_lo_q8); hipFree(d_xn); hipFree(d_mixed);
    }

    // down_inject: old vs new
    {
        constexpr int ncols = 10240, down_rows = 320, down_blocks_per_row = ncols / 32;
        constexpr int xq_blocks = ncols / 32, inject_rows = 4;
        std::vector<block_q8_0> h_down((size_t) down_rows * down_blocks_per_row);
        std::vector<block_q8_1> h_xq(xq_blocks);
        std::vector<float> h_inject((size_t) inject_rows * ncols);
        std::vector<float> h_x(ncols);
        fill_q8_0(h_down, rng); fill_q8_1(h_xq, rng);
        std::uniform_real_distribution<float> dw(-0.05f, 0.05f), dx(-2.0f, 2.0f);
        for (auto & v : h_inject) v = dw(rng);
        for (auto & v : h_x) v = dx(rng);
        void * d_down; block_q8_1 * d_xq; float * d_down_dst, * d_inject, * d_x, * d_inject_dst;
        HIP_CHECK(hipMalloc(&d_down, h_down.size() * sizeof(block_q8_0)));
        HIP_CHECK(hipMalloc(&d_xq, h_xq.size() * sizeof(block_q8_1)));
        HIP_CHECK(hipMalloc(&d_down_dst, down_rows * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inject, h_inject.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inject_dst, inject_rows * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_down, h_down.data(), h_down.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_xq, h_xq.data(), h_xq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_inject, h_inject.data(), h_inject.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice));

        double us_old = time_reps(0, false, d_down, d_xq, d_x, d_down_dst, 0.0f, 0.0f, d_inject, d_inject_dst, reps);
        double us_new = time_reps(1, false, d_down, d_xq, d_x, d_down_dst, 0.0f, 0.0f, d_inject, d_inject_dst, reps);
        printf("hc_down_inject_mixed:      %.2f us/launch (old, %d reps)\n", us_old, reps);
        printf("hc_down_inject_mixed_fast: %.2f us/launch (new, %d reps)\n", us_new, reps);
        hipFree(d_down); hipFree(d_xq); hipFree(d_down_dst); hipFree(d_inject); hipFree(d_x); hipFree(d_inject_dst);
    }
}

int main() {
    std::mt19937 rng(0xC0FFEE);
    bool all_ok = true;
    for (int rep = 0; rep < 5; ++rep) all_ok &= run_upmix_case("upmix-random", rng);
    for (int rep = 0; rep < 5; ++rep) all_ok &= run_down_inject_case("down-inject-random", rng);

    printf(all_ok ? "ALL PASS\n" : "FAIL\n");
    if (all_ok) microbench();
    return all_ok ? 0 : 1;
}
