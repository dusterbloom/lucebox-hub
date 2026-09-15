// Prefill GEMM micro-bench (RDNA4, gfx1201).
//
// Times mul_mat at the Qwen3.8-27B FFN shape (6144 x 15360 weights, 512
// activation columns) across the quant types the UD-IQ4_XS file actually
// ships, to price the sub-4-bit tensor tax the blog concedes. Throughput
// only; no correctness checks.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t ne0 = 6144;  // K
constexpr int64_t ne1 = 15360; // FFN rows (M)
constexpr int64_t ncols = 512; // prefill batch (N)

uint32_t g_rng = 0x9e3779b9u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f;
}

double bench_type(ggml_backend_t gpu, ggml_type wtype) {
    ggml_init_params params = { 4u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return -1.0;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, ne0, ne1);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0, ncols);
    ggml_set_input(w);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(ctx);
        return -1.0;
    }

    // Synthetic weights: quantize random f32 data. IQ types accept a
    // uniform (all-ones) importance matrix.
    const int64_t n_per_row = ne0;
    const int64_t nrows = ne1;
    std::vector<float> wf(nrows * n_per_row);
    for (float & v : wf) v = next_float();

    const bool needs_imat = ggml_quantize_requires_imatrix(wtype);
    std::vector<float> imat(needs_imat ? n_per_row : 0, 1.0f);

    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, nrows, n_per_row,
                        needs_imat ? imat.data() : nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    std::vector<ggml_fp16_t> xh(ne0 * ncols);
    for (auto & v : xh) v = ggml_fp32_to_fp16(next_float());
    ggml_backend_tensor_set(x, xh.data(), 0, xh.size() * sizeof(ggml_fp16_t));

    for (int i = 0; i < 3; ++i) {
        ggml_backend_graph_compute(gpu, gf);
    }
    hipDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    const int iters = 5;
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(gpu, gf);
    }
    hipDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double gbs = (double) ggml_nbytes(w) / ms / 1e6; // GB / ms -> TB/s-ish
    const double tflops = 2.0 * (double) ne0 * ne1 * ncols / ms / 1e9;
    std::printf("[bench-mmq] %-9s weight=%7.1f MiB %8.2f ms/iter %6.2f TB/s %6.1f TFLOP/s\n",
                ggml_type_name(wtype), ggml_nbytes(w) / 1048576.0, ms, gbs, tflops);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ms;
}

} // namespace

int main() {
#if defined(GGML_USE_HIP)
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess ||
        std::strncmp(props.gcnArchName, "gfx12", 5) != 0) {
        std::printf("[bench-mmq] SKIP: requires gfx12 (RDNA4)\n");
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 1;

    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS,
                         GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q3_K,
                         GGML_TYPE_Q5_K }) {
        bench_type(gpu, t);
    }
    ggml_backend_free(gpu);
    return 0;
#else
    std::printf("[bench-mmq] SKIP: HIP-only\n");
    return 77;
#endif
}
