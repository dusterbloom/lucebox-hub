// Parity test for the two-wave routed gate/up GEMV (mmvq-routed-gu.cuh) that
// replaces the legacy 8-warp mul_mat_vec_q launch on the Qwen4exp T=1 decode
// path (LUCE_MMVQ_MOE_GU2W, default on for gfx1151).
//
// Claim under test: for the production MUL_MAT_ID shape (one token, ncols_dst
// == 1, SwiGLU fusion, Q4_K or Q5_K gate/up, K multiple of 512, M multiple of
// 8) the new kernel's dst is byte-identical to the legacy kernel's on random
// weights, random activations and routings with masked (negative) entries.
// The kernel keeps the legacy lane -> K-block mapping, accumulation order,
// warp reduction and epilogue; any reordering would show up here.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" void ggml_backend_cuda_launch_mmvq_routed_gate_up(
        int type, const void * up, const void * gate, const void * y_q8_1, const int32_t * ids, float * dst,
        int ncols_x, int nrows_x, int nchannels_dst, int variant, hipStream_t stream);

namespace {

uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; std::memcpy(&u, &h, 2); return u; }

// Random K-quant blocks: finite d/dmin halves, random scale and quant bytes.
std::vector<uint8_t> random_blocks(std::mt19937 & rng, size_t n_blocks, size_t block_bytes) {
    std::uniform_real_distribution<float> dd(0.0002f, 0.002f), dmd(0.00005f, 0.0008f);
    std::uniform_int_distribution<int> bd(0, 255);
    std::vector<uint8_t> v(n_blocks * block_bytes);
    uint8_t * p = v.data();
    for (size_t b = 0; b < n_blocks; ++b, p += block_bytes) {
        const uint16_t d = f2h(dd(rng)), dm = f2h(dmd(rng));
        std::memcpy(p, &d, 2);
        std::memcpy(p + 2, &dm, 2);
        for (size_t i = 4; i < block_bytes; ++i) {
            p[i] = (uint8_t) bd(rng);
        }
    }
    return v;
}

// One q8_1 activation row: ds = {d, d * sum(q)}, 32 int8 per block.
std::vector<uint8_t> random_q8_1_row(std::mt19937 & rng, int ncols) {
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.005f, 0.05f);
    std::vector<uint8_t> y((size_t) ncols / 32 * 36);
    for (int b = 0; b < ncols / 32; ++b) {
        int8_t * qs = (int8_t *) (y.data() + (size_t) b * 36 + 4);
        int sum = 0;
        for (int i = 0; i < 32; ++i) { qs[i] = (int8_t) qd(rng); sum += qs[i]; }
        const float d = dd(rng);
        const uint16_t dh = f2h(d), sh = f2h(d * (float) sum);
        std::memcpy(y.data() + (size_t) b * 36, &dh, 2);
        std::memcpy(y.data() + (size_t) b * 36 + 2, &sh, 2);
    }
    return y;
}

bool run_case(ggml_type type, int ncols, int nrows, int n_experts, int n_used, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t block_bytes = type == GGML_TYPE_Q4_K ? 144 : 176;
    const size_t n_blocks = (size_t) n_experts * nrows * (ncols / 256);
    const std::vector<uint8_t> h_gate = random_blocks(rng, n_blocks, block_bytes);
    const std::vector<uint8_t> h_up   = random_blocks(rng, n_blocks, block_bytes);
    const std::vector<uint8_t> h_y    = random_q8_1_row(rng, ncols);

    // Routing: distinct experts, with masked routes on some slots.
    std::vector<int32_t> h_ids(n_used);
    {
        std::vector<int> perm(n_experts);
        for (int e = 0; e < n_experts; ++e) perm[e] = e;
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int u = 0; u < n_used; ++u) h_ids[u] = perm[u];
        if (seed % 2 == 1) { h_ids[1] = -1; h_ids[n_used - 1] = -1; }
    }

    uint8_t * d_gate = nullptr, * d_up = nullptr, * d_y = nullptr;
    int32_t * d_ids = nullptr;
    float * d_legacy = nullptr, * d_new = nullptr;
    const size_t out_bytes = (size_t) n_used * nrows * sizeof(float);
    bool ok = true;
    ok = hipMalloc(&d_gate, h_gate.size()) == hipSuccess && ok;
    ok = hipMalloc(&d_up, h_up.size()) == hipSuccess && ok;
    ok = hipMalloc(&d_y, h_y.size()) == hipSuccess && ok;
    ok = hipMalloc(&d_ids, h_ids.size() * sizeof(int32_t)) == hipSuccess && ok;
    ok = hipMalloc(&d_legacy, out_bytes) == hipSuccess && ok;
    ok = hipMalloc(&d_new, out_bytes) == hipSuccess && ok;
    if (!ok) {
        std::fprintf(stderr, "  FAIL: hipMalloc\n");
        return false;
    }
    hipMemcpy(d_gate, h_gate.data(), h_gate.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_up, h_up.data(), h_up.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_y, h_y.data(), h_y.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemset(d_legacy, 0xFF, out_bytes);
    hipMemset(d_new, 0xAA, out_bytes);

    ggml_backend_cuda_launch_mmvq_routed_gate_up(type, d_up, d_gate, d_y, d_ids, d_legacy, ncols, nrows, n_used, 0, (hipStream_t) 0);
    ggml_backend_cuda_launch_mmvq_routed_gate_up(type, d_up, d_gate, d_y, d_ids, d_new,    ncols, nrows, n_used, 1, (hipStream_t) 0);
    ok = hipDeviceSynchronize() == hipSuccess;

    std::vector<float> h_legacy((size_t) n_used * nrows), h_new((size_t) n_used * nrows);
    hipMemcpy(h_legacy.data(), d_legacy, out_bytes, hipMemcpyDeviceToHost);
    hipMemcpy(h_new.data(), d_new, out_bytes, hipMemcpyDeviceToHost);
    hipFree(d_gate); hipFree(d_up); hipFree(d_y); hipFree(d_ids); hipFree(d_legacy); hipFree(d_new);

    size_t mismatches = 0, masked_nonzero = 0, finite = 0;
    for (int u = 0; u < n_used; ++u) {
        for (int r = 0; r < nrows; ++r) {
            const size_t i = (size_t) u * nrows + r;
            if (std::memcmp(&h_legacy[i], &h_new[i], sizeof(float)) != 0) ++mismatches;
            if (h_ids[u] < 0 && h_new[i] != 0.0f) ++masked_nonzero;
            if (std::isfinite(h_new[i])) ++finite;
        }
    }
    if (mismatches) std::fprintf(stderr, "  FAIL: %zu / %zu outputs differ from the legacy kernel\n", mismatches, h_new.size());
    if (masked_nonzero) std::fprintf(stderr, "  FAIL: %zu masked-route outputs are non-zero\n", masked_nonzero);
    if (finite != h_new.size()) std::fprintf(stderr, "  FAIL: %zu non-finite outputs\n", h_new.size() - finite);
    return ok && mismatches == 0 && masked_nonzero == 0 && finite == h_new.size();
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[mmvq-routed-gate-up-parity] SKIP: no HIP device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("[mmvq-routed-gate-up-parity] FAIL: backend init\n");
        return 1;
    }

    struct Shape { int ncols, nrows, experts, used; };
    bool ok = true;
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K }) {
        for (const Shape & s : { Shape{2560, 640, 32, 10}, Shape{512, 8, 4, 3}, Shape{1024, 72, 16, 10} }) {
            for (uint32_t seed = 1; seed <= 4; ++seed) {
                const bool c = run_case(type, s.ncols, s.nrows, s.experts, s.used, seed);
                std::printf("[mmvq-routed-gate-up-parity] %s K=%d M=%d experts=%d used=%d seed=%u %s\n",
                    ggml_type_name(type), s.ncols, s.nrows, s.experts, s.used, seed, c ? "ok" : "FAIL");
                ok = c && ok;
            }
        }
    }
    ggml_backend_free(backend);
    std::printf("[mmvq-routed-gate-up-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[mmvq-routed-gate-up-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
