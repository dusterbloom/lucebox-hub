// Standalone mmb-vs-library-GEMM differential for a single quantized MUL_MAT.
//
// Runs the same ggml_mul_mat(w[K,M] IQ4_NL, x[K,T] f32) graph on the CPU
// backend and on the CUDA/HIP backend and compares. The CPU result is the
// reference, so calibration is inherent: with GGML_CUDA_MMB=1 the CUDA path is
// mmb; with DFLASH_MMB_SHADOW=1 + QWEN4EXP_MMB_CUBLAS=1 the big shapes take the
// library GEMM. Run per shape to see which path is wrong.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t g_state = 0x12345678u;
static float rnd() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float) ((g_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

static bool run(ggml_backend_t backend, ggml_type wtype,
                const std::vector<uint8_t> & wq, const std::vector<float> & xf,
                int64_t M, int64_t K, int64_t T, std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 16 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(x);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, x);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed (%d)\n", (int) st);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t) M * T);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <M> <K> <T>\n", argv[0]); return 2; }
    const int64_t M = atoll(argv[1]), K = atoll(argv[2]), T = atoll(argv[3]);
    const ggml_type wtype = (argc > 4 && atoi(argv[4]) == 1) ? GGML_TYPE_Q6_K : GGML_TYPE_IQ4_NL;
    const size_t row = ggml_row_size(wtype, K);

    std::vector<float>   wf((size_t) K * M), xf((size_t) K * T);
    std::vector<uint8_t> wq(row * M);
    for (auto & v : wf) v = rnd();
    for (auto & v : xf) v = rnd();
    const ggml_type_traits * tr = ggml_get_type_traits(wtype);
    for (int64_t m = 0; m < M; ++m) tr->from_float_ref(wf.data() + m * K, wq.data() + m * row, K);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    std::vector<float> ref, out;
    if (!run(cpu, wtype, wq, xf, M, K, T, ref)) return 1;
    if (!run(gpu, wtype, wq, xf, M, K, T, out)) return 1;
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    double md = 0, norm = 0; int64_t ndiff = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs((double) out[i] - ref[i]);
        if (d > md) md = d;
        if (d > 0.2) ndiff++;
        norm += (double) ref[i] * ref[i];
    }
    std::fprintf(stderr, "[diff] M=%lld K=%lld T=%lld maxdiff=%.6g rel=%.4g ndiff=%lld refnorm=%.4g\n",
        (long long) M, (long long) K, (long long) T, md,
        norm > 0 ? md / std::sqrt(norm / (double) out.size()) : 0.0, (long long) ndiff, std::sqrt(norm));
    return 0;
}
