// Standalone bit-identity test for the LUCE_QWEN_HC_LO_FUSED fusion:
// hc_down_inject_mixed_lo_fused (mmvq.cu) must produce byte-for-byte the same
// down_dst/inject_dst/lo_q8/lo_dst output as running the two unfused kernels
// back to back -- hc_down_inject_mixed followed by quantize_hc_lo_q8_1_cuda
// (quantize.cu). Runs both paths through the exported test hook
// ggml_cuda_test_hc_down_inject_lo_fused_bitexact (mmvq.cu) and memcmp's the
// outputs. No ggml graph involved -- pure kernel-vs-kernel comparison.
//
// Weight/activation buffers only need to be *structurally* valid block_q8_0 /
// block_q8_1 data (right sizeof/layout) -- the comparison is between two
// kernels fed byte-identical inputs, not a numerical-accuracy check, so the
// quantized values do not need to come from a real reference quantizer.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int ggml_cuda_test_hc_down_inject_lo_fused_bitexact(
    const void * down_w, const void * xq, const float * inject_w, const float * x,
    float * down_dst_a, float * inject_dst_a, void * lo_q8_a, float * lo_dst_a,
    float * down_dst_b, float * inject_dst_b, void * lo_q8_b, float * lo_dst_b,
    int in_place, void * raw_stream);

#define HIP_CHECK(x) do { hipError_t e__ = (x); if (e__ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e__), __FILE__, __LINE__); exit(1); } } while (0)

// Byte-identical to ggml-common.h's block_q8_0/block_q8_1 (not included to
// keep this test ggml-independent, matching test_hc_cn_bitexact.cpp's style).
#pragma pack(push, 1)
struct block_q8_0 { uint16_t d; int8_t qs[32]; };
struct block_q8_1 { uint16_t d; uint16_t s; int8_t qs[32]; };
#pragma pack(pop)
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout drift");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 layout drift");

static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;          // flush to zero (inputs kept well clear of this)
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u); // inf (inputs kept well clear of this)
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

constexpr int NCOLS = 10240;
constexpr int NROWS_DOWN = 320;
constexpr int BLOCKS_PER_ROW = NCOLS / 32;              // 320
constexpr int DOWN_BLOCKS = NROWS_DOWN * BLOCKS_PER_ROW; // 102400
constexpr int XQ_BLOCKS = NCOLS / 32;                    // 320
constexpr int NROWS_INJECT = 4;
constexpr int LO_Q8_PAD_BLOCKS = 16; // MATRIX_ROW_PADDING(512)/QK8_1(32)
constexpr int LO_Q8_REAL_BLOCKS = NROWS_DOWN / 32; // 10

static bool run_case(const char * name, std::mt19937 & rng, bool in_place) {
    std::uniform_real_distribution<float> small(-1.0f, 1.0f);
    std::uniform_int_distribution<int> qsdist(-120, 120);
    std::uniform_real_distribution<float> scaledist(0.01f, 0.5f);

    std::vector<block_q8_0> h_down_w(DOWN_BLOCKS);
    for (auto & b : h_down_w) {
        b.d = float_to_half(scaledist(rng));
        for (auto & q : b.qs) q = (int8_t) qsdist(rng);
    }
    std::vector<block_q8_1> h_xq(XQ_BLOCKS);
    for (auto & b : h_xq) {
        b.d = float_to_half(scaledist(rng));
        b.s = float_to_half(small(rng));
        for (auto & q : b.qs) q = (int8_t) qsdist(rng);
    }
    std::vector<float> h_inject_w((size_t) NROWS_INJECT * NCOLS), h_x(NCOLS);
    for (auto & v : h_inject_w) v = small(rng) * 0.05f;
    for (auto & v : h_x)        v = small(rng) * 0.05f;

    void *d_down_w, *d_xq;
    float *d_inject_w, *d_x;
    float *d_down_dst_a, *d_inject_dst_a, *d_lo_dst_a;
    float *d_down_dst_b, *d_inject_dst_b, *d_lo_dst_b;
    void *d_lo_q8_a, *d_lo_q8_b;
    HIP_CHECK(hipMalloc(&d_down_w, h_down_w.size() * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_xq, h_xq.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&d_inject_w, h_inject_w.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_x, h_x.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_down_dst_a, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_down_dst_b, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_inject_dst_a, NROWS_INJECT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_inject_dst_b, NROWS_INJECT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_lo_dst_a, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_lo_dst_b, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_lo_q8_a, LO_Q8_PAD_BLOCKS * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&d_lo_q8_b, LO_Q8_PAD_BLOCKS * sizeof(block_q8_1)));

    HIP_CHECK(hipMemcpy(d_down_w, h_down_w.data(), h_down_w.size() * sizeof(block_q8_0), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_xq, h_xq.data(), h_xq.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_inject_w, h_inject_w.data(), h_inject_w.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), hipMemcpyHostToDevice));
    // poison outputs differently so a no-op kernel can't accidentally "pass"
    HIP_CHECK(hipMemset(d_down_dst_a, 0xAA, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMemset(d_down_dst_b, 0x55, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMemset(d_inject_dst_a, 0xAA, NROWS_INJECT * sizeof(float)));
    HIP_CHECK(hipMemset(d_inject_dst_b, 0x55, NROWS_INJECT * sizeof(float)));
    HIP_CHECK(hipMemset(d_lo_dst_a, 0xAA, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMemset(d_lo_dst_b, 0x55, NROWS_DOWN * sizeof(float)));
    HIP_CHECK(hipMemset(d_lo_q8_a, 0xAA, LO_Q8_PAD_BLOCKS * sizeof(block_q8_1)));
    HIP_CHECK(hipMemset(d_lo_q8_b, 0x55, LO_Q8_PAD_BLOCKS * sizeof(block_q8_1)));

    int ok = ggml_cuda_test_hc_down_inject_lo_fused_bitexact(
        d_down_w, d_xq, d_inject_w, d_x,
        d_down_dst_a, d_inject_dst_a, d_lo_q8_a, d_lo_dst_a,
        d_down_dst_b, d_inject_dst_b, d_lo_q8_b, d_lo_dst_b,
        in_place ? 1 : 0, nullptr);
    if (!ok) { fprintf(stderr, "[%s] kernel launch/sync failed\n", name); return false; }

    std::vector<float> h_down_dst_a(NROWS_DOWN), h_down_dst_b(NROWS_DOWN);
    std::vector<float> h_inject_dst_a(NROWS_INJECT), h_inject_dst_b(NROWS_INJECT);
    std::vector<float> h_lo_dst_a(NROWS_DOWN), h_lo_dst_b(NROWS_DOWN);
    std::vector<block_q8_1> h_lo_q8_a(LO_Q8_REAL_BLOCKS), h_lo_q8_b(LO_Q8_REAL_BLOCKS);
    HIP_CHECK(hipMemcpy(h_down_dst_a.data(), d_down_dst_a, NROWS_DOWN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_down_dst_b.data(), d_down_dst_b, NROWS_DOWN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_inject_dst_a.data(), d_inject_dst_a, NROWS_INJECT * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_inject_dst_b.data(), d_inject_dst_b, NROWS_INJECT * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_lo_dst_a.data(), d_lo_dst_a, NROWS_DOWN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_lo_dst_b.data(), d_lo_dst_b, NROWS_DOWN * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_lo_q8_a.data(), d_lo_q8_a, LO_Q8_REAL_BLOCKS * sizeof(block_q8_1), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_lo_q8_b.data(), d_lo_q8_b, LO_Q8_REAL_BLOCKS * sizeof(block_q8_1), hipMemcpyDeviceToHost));

    const bool down_eq   = memcmp(h_down_dst_a.data(), h_down_dst_b.data(), NROWS_DOWN * sizeof(float)) == 0;
    const bool inject_eq = memcmp(h_inject_dst_a.data(), h_inject_dst_b.data(), NROWS_INJECT * sizeof(float)) == 0;
    const bool q8_eq      = memcmp(h_lo_q8_a.data(), h_lo_q8_b.data(), LO_Q8_REAL_BLOCKS * sizeof(block_q8_1)) == 0;
    // lo_dst only holds meaningful (non-poison) data when !in_place; when
    // in_place both paths leave it untouched, in which case comparing the
    // poison patterns would spuriously fail -- skip in that case.
    const bool lodst_eq = in_place || memcmp(h_lo_dst_a.data(), h_lo_dst_b.data(), NROWS_DOWN * sizeof(float)) == 0;

    printf("[%s] in_place=%d down_dst %s inject_dst %s lo_q8 %s lo_dst %s\n", name, in_place ? 1 : 0,
        down_eq ? "BIT-IDENTICAL" : "MISMATCH", inject_eq ? "BIT-IDENTICAL" : "MISMATCH",
        q8_eq ? "BIT-IDENTICAL" : "MISMATCH", lodst_eq ? "BIT-IDENTICAL" : "MISMATCH");

    hipFree(d_down_w); hipFree(d_xq); hipFree(d_inject_w); hipFree(d_x);
    hipFree(d_down_dst_a); hipFree(d_down_dst_b); hipFree(d_inject_dst_a); hipFree(d_inject_dst_b);
    hipFree(d_lo_dst_a); hipFree(d_lo_dst_b); hipFree(d_lo_q8_a); hipFree(d_lo_q8_b);
    return down_eq && inject_eq && q8_eq && lodst_eq;
}

int main() {
    std::mt19937 rng(0xC0FFEE);
    bool all_ok = true;
    for (int rep = 0; rep < 5; ++rep) all_ok &= run_case("random-in-place", rng, true);
    for (int rep = 0; rep < 5; ++rep) all_ok &= run_case("random-separate", rng, false);
    printf(all_ok ? "ALL PASS\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
